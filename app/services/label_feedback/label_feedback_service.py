"""
Service for Label Feedback API
"""
import torch
import numpy as np
import os
import zipfile
import py_vncorenlp
import time

from PIL import Image
import requests
import io
import base64
from huggingface_hub import snapshot_download
from app.config import settings
from app.utils.logger import logger

class LabelFeedbackService:
    def __init__(self, token=None):
        # Determine base directory of this service file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Token handling
        self.token = token if token else settings.HF_TOKEN
        
        # Label description model config
        self.MODEL_PATH = "lewisnguyn/bonix-feedback-model-description-v2"
        self.VNCORENLP_REPO = "lewisnguyn/bonix-model-vncorenlp"
        self.MAX_LENGTH = 256
        
        self.rdrsegmenter = None
        
        # Label image model config
        self.IMAGE_MODEL_PATH = "lewisnguyn/bonix-feedback-model-image"
        
        self.desc_pipe = None
        self.img_pipe = None
        
        # Lazy load resources or load on init
        # For this implementation, we'll try to load on init but handle failures gracefully
        try:
            self.rdrsegmenter = self._setup_vncorenlp()
        except Exception as e:
            logger.warning(f"Failed to load LabelFeedbackService resources: {e}")

    # PREDICT DESCRIPTION LABEL SERVICE
    def _setup_vncorenlp(self):
        """Ensure VnCoreNLP is available from Hugging Face."""
        try:
            logger.info(f"Downloading/Loading VnCoreNLP from {self.VNCORENLP_REPO}...")
            
            # Use a local directory to avoid symlink issues with Java/VnCoreNLP
            # and use system cache to avoid cluttering project directory
            from pathlib import Path
            from huggingface_hub import hf_hub_download
            import zipfile
            
            local_model_dir = Path.home() / ".cache" / "bonix-ai" / "vncorenlp"
            local_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if jar file exists
            jar_files = list(local_model_dir.rglob("VnCoreNLP-*.jar"))
            if not jar_files:
                logger.info("VnCoreNLP not found locally. Downloading from Hugging Face...")
                zip_path = hf_hub_download(repo_id=self.VNCORENLP_REPO, filename="VnCoreNLP.zip", token=self.token)
                
                logger.info(f"Extracting {zip_path} to {local_model_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(local_model_dir)
                    
                jar_files = list(local_model_dir.rglob("VnCoreNLP-*.jar"))
                
            if not jar_files:
                raise Exception("VnCoreNLP jar file not found after extraction")
                
            save_dir = str(jar_files[0].parent)
            
            return py_vncorenlp.VnCoreNLP(save_dir=save_dir, annotators=["wseg"])
        except Exception as e:
            logger.error(f"Error setting up VnCoreNLP: {e}")
            return None



    def predict(self, text):
        if self.rdrsegmenter is None:
             self.rdrsegmenter = self._setup_vncorenlp()
             if self.rdrsegmenter is None:
                 raise Exception("VnCoreNLP not available")

        # Segmentation
        sentences = self.rdrsegmenter.word_segment(text)
        text_segmented = " ".join(sentences)

        # Local Inference API approach
        try:
            if not self.desc_pipe:
                from transformers import pipeline
                logger.info(f"Loading local description model {self.MODEL_PATH}...")
                self.desc_pipe = pipeline("text-classification", model=self.MODEL_PATH, token=self.token, top_k=None)
            response = self.desc_pipe(text_segmented)
        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            return []
        
        if not response:
            return []
            
        # API returns list of lists: [[{"label": "...", "score": 0.9}, ...]]
        scores_list = response[0] if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list) else response
        
        if not isinstance(scores_list, list):
            logger.error(f"Unexpected API response format: {response}")
            return []
            
        # Collect all predictions > 25%
        predictions = []
        for item in scores_list:
            if not isinstance(item, dict) or "label" not in item or "score" not in item:
                continue
            prob = item["score"]
            if prob > 0.25:
                # The API typically returns the label name directly from id2label
                predictions.append({"label": item["label"], "score": float(prob)})
        
        # Sort by probability descending
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter duplicates based on prefix before ":" (keep higher score)
        unique_predictions = []
        seen_prefixes = set()
        
        for pred in predictions:
            label = pred["label"]
            prefix = label.split(":", 1)[0]
            
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                unique_predictions.append(pred)
        
        return unique_predictions

    def check_model_status(self) -> dict:
        """Check status of all models"""
        return {
            "description_model": self.desc_pipe is not None,
            "vncorenlp": self.rdrsegmenter is not None,
            "image_model": self.img_pipe is not None
        }

    def describe_image(self, img_path):
        if img_path.startswith("http"):
            image_data = requests.get(img_path).content
        else:
            with open(img_path, "rb") as f:
                image_data = f.read()

        try:
            if not self.img_pipe:
                from transformers import pipeline
                logger.info(f"Loading local image model {self.IMAGE_MODEL_PATH}...")
                self.img_pipe = pipeline("image-to-text", model=self.IMAGE_MODEL_PATH, token=self.token)
            
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # The prompt can be passed if supported, but typically image-to-text pipelines take just the image
            # Try with prompt first, fallback to without prompt if model doesn't support it
            prompt = "<MORE_DETAILED_CAPTION>"
            try:
                response = self.img_pipe(image, prompt=prompt)
            except Exception:
                response = self.img_pipe(image)
            
            if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and "generated_text" in response[0]:
                return response[0]["generated_text"]
            elif isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"]
            return str(response)
        except Exception as e:
            logger.error(f"Image inference failed: {e}")
            return f"Failed to describe image via local model: {str(e)}"

# Create a singleton instance
label_feedback_service = LabelFeedbackService()
