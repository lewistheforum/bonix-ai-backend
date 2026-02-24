import os
import time
import requests
import io
import cv2
import base64
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.dto.fracture_detection.fracture_detection_dto import BoundingBox, FractureDetectionResponse, AIResultAnalyze
from app.utils.logger import logger

# Try to import ultralytics and patch it with custom modules before loading YOLO
try:
    from app.services.fracture_detection.yolo_patch import patch_ultralytics
    patch_dict = patch_ultralytics()
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(f"ultralytics package or patch failed to load ({e}). Fracture detection will not work until installed.")

class FractureDetectionService:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "huggingface", "wrist_fracture_model.pt")
        self.repo_id = "lewisnguyn/wrist-fracture-detection"
        self.filename = "fracture_model.pt"
        self.is_loaded = False
        
        # Determine model path
        models_dir = os.path.dirname(self.model_path)
        os.makedirs(models_dir, exist_ok=True)
        
    def _download_model_if_needed(self):
        """Download the model from Hugging Face if it doesn't exist locally"""
        if os.path.exists(self.model_path):
            return True
            
        logger.info(f"Downloading model {self.repo_id}/{self.filename}...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=os.path.dirname(self.model_path),
                local_dir_use_symlinks=False
            )
            # Rename if the downloaded file has a different name
            if downloaded_path != self.model_path:
                os.rename(downloaded_path, self.model_path)
            logger.info(f"Model downloaded successfully to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model via huggingface_hub: {e}")
            
            # Fallback to direct download
            try:
                url = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"
                logger.info(f"Fallback direct download from {url}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Fallback download successful to {self.model_path}")
                return True
            except Exception as inner_e:
                logger.error(f"Fallback download failed: {inner_e}")
                return False

    def load_model(self):
        """Load the YOLO model for inference"""
        if self.is_loaded:
            return True
            
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Cannot load model: ultralytics is not installed")
            return False
            
        try:
            if not self._download_model_if_needed():
                logger.error("Failed to ensure model file exists")
                return False
                
            logger.info("Loading YOLO fracture detection model...")
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info("Fracture detection model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading fracture detection model: {e}")
            return False

    def _standardize_image(self, image_bytes: bytes) -> Image.Image:
        """
        Open image bytes and handle 16-bit grayscale conversion.
        Returns a PIL Image in RGB mode.
        """
        img = Image.open(io.BytesIO(image_bytes))
        
        # Handle 16-bit grayscale images properly
        if img.mode == "I;16" or img.mode == "I":
            img = img.point(lambda i: i * (1. / 256)).convert("L")
            
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        return img

    # ─── Step 1: Classify wrist X-ray ───────────────────────────────────
    async def verify_is_wrist_xray(self, image_bytes: bytes) -> bool:
        """
        Use OpenAI Vision to verify if the uploaded image is a wrist X-ray.
        Returns True if it is, False otherwise.
        Retries once if the first attempt returns NO (LLM can be inconsistent).
        """
        from app.config import settings
        
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. Skipping image classification step.")
            return True
            
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Standardize image to JPEG
            img = self._standardize_image(image_bytes)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            classification_prompt = (
                "You are a medical imaging assistant. "
                "Look at the attached image carefully. "
                "Is this image an X-ray of a wrist, hand, or forearm area? "
                "Medical X-rays may appear as grayscale images showing bone structures. "
                "Reply with ONLY the word 'YES' or 'NO'."
            )
            
            # Try up to 2 times (retry once if NO, since LLM can be inconsistent)
            for attempt in range(2):
                response = await client.chat.completions.create(
                    model=settings.OPENAI_CHAT_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": classification_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=10
                )
                
                result_text = response.choices[0].message.content.strip().upper()
                logger.info(f"OpenAI Vision classification attempt {attempt+1}: {result_text}")
                
                if "YES" in result_text:
                    return True
                    
                if attempt == 0:
                    logger.info("First classification attempt returned NO, retrying...")
            
            # Both attempts returned NO
            return False
            
        except Exception as e:
            logger.error(f"Error during OpenAI image classification: {e}")
            # Fail open - if API fails, still try to run the detection model
            return True

    # ─── Step 2: YOLO fracture detection ────────────────────────────────
    async def detect_fracture(self, image_bytes: bytes, db=None, notes: str = None) -> FractureDetectionResponse:
        """
        Detect fractures in the provided image bytes using YOLO model,
        then send annotated results to OpenAI for medical analysis (Step 3).
        
        Args:
            image_bytes: The raw image bytes from the user upload
            db: Optional database session for querying medicine categories
            notes: Optional patient notes (allergies, medical history, etc.)
            
        Returns:
            FractureDetectionResponse: The detection results with AI analysis
        """
        start_time = time.time()
        
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load fracture detection model")

        try:
            # Standardize image
            image = self._standardize_image(image_bytes)
            
            # Run YOLO inference
            results = self.model(image)
            
            # Parse results
            detections = []
            has_fracture = False
            
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id] if hasattr(result, 'names') and cls_id in result.names else f"class_{cls_id}"
                    
                    bbox = BoundingBox(
                        x_min=coords[0],
                        y_min=coords[1],
                        x_max=coords[2],
                        y_max=coords[3],
                        confidence=conf,
                        class_name=class_name
                    )
                    detections.append(bbox)
                    has_fracture = True
            
            # Generate annotated image with bounding boxes
            annotated_bgr = result.plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            
            buffered = io.BytesIO()
            annotated_pil.save(buffered, format="JPEG")
            annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # ─── Step 3: Analyze with OpenAI ────────────────────────
            logger.info(f"Step 3: Sending {len(detections)} detections to OpenAI for analysis...")
            ai_result = await self.analyze_with_openai(annotated_image_base64, detections, db=db, notes=notes)
            logger.info(f"Step 3 complete. AI result: {ai_result}")
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return FractureDetectionResponse(
                has_fracture=has_fracture,
                detections=detections,
                annotated_image_base64=annotated_image_base64,
                ai_result_analyze=ai_result,
                processing_time_ms=processing_time_ms,
                analyzed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error during fracture detection inference: {e}")
            raise Exception(f"Inference error: {str(e)}")

    # ─── Step 3: OpenAI medical analysis ────────────────────────────────
    async def analyze_with_openai(
        self,
        annotated_image_base64: str,
        detections: List[BoundingBox],
        db=None,
        notes: str = None
    ) -> Optional[AIResultAnalyze]:
        """
        Send the annotated detection image and detection results to OpenAI
        for medical analysis, treatment plan, and medication recommendations.
        
        Medicine recommendations are constrained to therapeutic classes
        available in the knowledge_base_medicines table.
        
        Args:
            annotated_image_base64: Base64 encoded annotated X-ray image
            detections: List of YOLO detection results
            db: Optional database session for querying medicine categories
            notes: Optional patient notes (allergies, medical history, other conditions)
        """
        from app.config import settings
        
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. Skipping AI analysis step.")
            return None
            
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Fetch available medicine categories from knowledge_base_medicines
            available_categories = []
            if db is not None:
                try:
                    from sqlalchemy import text
                    result = await db.execute(text("""
                        SELECT DISTINCT 
                            metadata->>'therapeutic_class' as therapeutic_class
                        FROM knowledge_base_medicines
                        WHERE deleted_at IS NULL
                            AND metadata->>'therapeutic_class' IS NOT NULL
                        ORDER BY therapeutic_class
                    """))
                    rows = result.fetchall()
                    available_categories = [row.therapeutic_class for row in rows if row.therapeutic_class]
                    logger.info(f"Loaded {len(available_categories)} medicine categories from knowledge base")
                except Exception as e:
                    logger.warning(f"Failed to load medicine categories: {e}")
            
            # Build a summary of detected classes
            detection_summary = ", ".join(
                [f"{d.class_name} ({d.confidence*100:.1f}%)" for d in detections]
            )
            
            # Build medicine category constraint for the prompt
            if available_categories:
                categories_list = "\n".join([f"- {cat}" for cat in available_categories])
                medicine_constraint = f"""

IMPORTANT: You MUST only recommend medicines from the following therapeutic categories available in our database. Pick the most relevant categories for the detected condition and suggest specific medicines from those categories:

{categories_list}

In the "medicine_categories" field, list the therapeutic categories you recommend.
In the "medicines" field, list specific medicine names that belong to those categories."""
            else:
                medicine_constraint = ""
            
            # Build patient notes context for the prompt
            if notes and notes.strip():
                patient_notes_section = f"""\n\nPATIENT NOTES: The patient has provided the following additional information (e.g. medical history, allergies, existing conditions). You MUST take this into account when recommending medicines — avoid medicines that conflict with allergies or existing conditions, and consider drug interactions:
{notes.strip()}"""
            else:
                patient_notes_section = ""
            
            prompt = f"""You are a doctor specializing in treating wrist bone diseases in children and adults. I have an image that detects signs of bone disease as shown above. Based on the image you have detected and the types of diseases listed below:

0: boneanomaly
1: bonelesion
2: foreignbody
3: fracture
4: metal
5: periosteal reaction
6: pronator sign
7: softtissue
8: text

The model detected the following: {detection_summary}{patient_notes_section}

Please provide an analysis, treatment plan, and appropriate medications for the condition. With medicines list, each medicine name should include the reason why this medicine is recommended for the detected condition.{medicine_constraint}

Note: Please only provide feedback in JSON format as follows:

{{
  "analyze": "string",
  "treatment_plan": ["string"],
  "medicine_categories": ["string"],
  "medicines": ["string"]
}}"""

            response = await client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{annotated_image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            raw_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI analysis raw response: {raw_text}")
            
            # Extract JSON from the response (handle markdown code fences)
            json_text = raw_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(json_text)
            
            return AIResultAnalyze(
                analyze=parsed.get("analyze", ""),
                treatment_plan=parsed.get("treatment_plan", []),
                medicine_categories=parsed.get("medicine_categories", []),
                medicines=parsed.get("medicines", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI analysis JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during OpenAI analysis: {e}")
            return None

# Singleton instance
fracture_detector = FractureDetectionService()
