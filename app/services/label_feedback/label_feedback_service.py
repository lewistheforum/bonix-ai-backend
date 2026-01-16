"""
Service for Label Feedback API
"""
import torch
import numpy as np
import os
import zipfile
import py_vncorenlp

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from huggingface_hub import snapshot_download
from app.config import settings

class LabelFeedbackService:
    def __init__(self, token=None):
        # Determine base directory of this service file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Token handling
        self.token = token if token else settings.HF_TOKEN
        
        # Label description model config
        # self.MODEL_PATH = "lewisnguyn/bonix-feedback-model-description"
        self.MODEL_PATH = "lewisnguyn/bonix-feedback-model-description-v2"
        self.VNCORENLP_REPO = "lewisnguyn/bonix-model-vncorenlp"
        self.MAX_LENGTH = 256
        
        self.tokenizer = None
        self.model = None
        self.rdrsegmenter = None
        
        # Label image model config
        self.IMAGE_MODEL_PATH = "lewisnguyn/bonix-feedback-model-image"
        self.image_model = None
        self.image_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lazy load resources or load on init
        # For this implementation, we'll try to load on init but handle failures gracefully
        try:
            self._load_resources()
        except Exception as e:
            print(f"Warning: Failed to load LabelFeedbackService resources: {e}")

    # PREDICT DESCRIPTION LABEL SERVICE
    def _setup_vncorenlp(self):
        """Ensure VnCoreNLP is available from Hugging Face."""
        try:
            print(f"Downloading/Loading VnCoreNLP from {self.VNCORENLP_REPO}...")
            
            # Use a local directory to avoid symlink issues with Java/VnCoreNLP
            # and use system cache to avoid cluttering project directory
            from pathlib import Path
            local_model_dir = Path.home() / ".cache" / "medicare-ai" / "vncorenlp"
            
            # Download/Get cache path for the VnCoreNLP model
            # Using local_dir to force actual files instead of symlinks
            model_dir = snapshot_download(repo_id=self.VNCORENLP_REPO, local_dir=local_model_dir, token=self.token)
            
            # VnCoreNLP expects the directory to contain VnCoreNLP-1.2.jar and models
            # We assume the repo structure matches what py_vncorenlp expects or contains the jar
            return py_vncorenlp.VnCoreNLP(save_dir=model_dir, annotators=["wseg"])
        except Exception as e:
            print(f"Error setting up VnCoreNLP: {e}")
            return None

    def _load_resources(self):
        print(f"Loading model from {self.MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, token=self.token)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_PATH, token=self.token)
        self.model.eval() # Set to eval mode
        
        print("Initializing VnCoreNLP...")
        self.rdrsegmenter = self._setup_vncorenlp()

    def predict(self, text):
        if not self.tokenizer or not self.model:
            self._load_resources()
            
        if not self.tokenizer or not self.model:
             raise Exception("Model not loaded")

        # Use labels from model config
        id2label = self.model.config.id2label
        if not id2label:
            raise Exception("Model config does not contain 'id2label'. Cannot map predictions.")

        if self.rdrsegmenter is None:
             self.rdrsegmenter = self._setup_vncorenlp()
             if self.rdrsegmenter is None:
                 raise Exception("VnCoreNLP not available")

        # Segmentation
        sentences = self.rdrsegmenter.word_segment(text)
        text_segmented = " ".join(sentences)
        # print(f"Segmented Text: {text_segmented}")

        # Tokenization
        inputs = self.tokenizer(
            text_segmented, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=self.MAX_LENGTH
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.detach().numpy()[0]
        probs = 1 / (1 + np.exp(-logits))
        
        # Collect all predictions > 10%
        predictions = []
        for idx, prob in enumerate(probs):
            if prob > 0.25:
                label_name = id2label[str(idx)] if str(idx) in id2label else id2label[idx]
                predictions.append({"label": label_name, "score": float(prob)})
        
        # Sort by probability descending
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter duplicates based on prefix before ":" (keep higher score)
        unique_predictions = []
        seen_prefixes = set()
        
        for pred in predictions:
            label = pred["label"]
            # Get prefix (part before the first colon, or the whole label if no colon)
            prefix = label.split(":", 1)[0]
            
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                unique_predictions.append(pred)
        
        return unique_predictions

    # PREDICT IMAGE LABEL SERVICE
    def _load_image_resources(self):
        print(f"Loading image model from {self.IMAGE_MODEL_PATH}...")
        self.image_model = AutoModelForCausalLM.from_pretrained(self.IMAGE_MODEL_PATH, trust_remote_code=True, token=self.token).to(self.device).eval()
        self.image_processor = AutoProcessor.from_pretrained(self.IMAGE_MODEL_PATH, trust_remote_code=True, token=self.token)
    
    def check_model_status(self) -> dict:
        """Check status of all models"""
        status = {
            "description_model": self.model is not None and self.tokenizer is not None,
            "vncorenlp": self.rdrsegmenter is not None,
            "image_model": False # Lazy loaded usually, but let's check if loaded or try to load?
        }
        
        # Check image model if it was attempted to load
        if self.image_model is not None and self.image_processor is not None:
            status["image_model"] = True
            
        return status

    def describe_image(self, img_path):
        if not self.image_model or not self.image_processor:
             self._load_image_resources()
             
        if img_path.startswith("http"):
            image = Image.open(requests.get(img_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')

        # Prompt đặc biệt của Microsoft để lấy mô tả chi tiết
        prompt = "<MORE_DETAILED_CAPTION>"

        inputs = self.image_processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.image_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.image_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.image_processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

        return parsed_answer[prompt]

# Create a singleton instance
label_feedback_service = LabelFeedbackService()
