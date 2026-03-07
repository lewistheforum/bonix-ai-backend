"""
Bad Word Detection Service - Vietnamese bad word detection using ViHateT5 model
"""
import os
import sys
import unicodedata
import requests
import time
from app.config import settings

from app.utils.logger import logger

class VietnameseBadWordDetector:
    """Core detector class using HuggingFace Inference API"""
    
    def __init__(self, model_path="lewisnguyn/bonix-bad-words-detection", token=None):
        """
        Initialize with local transformers pipeline
        Args:
            model_path: HuggingFace model ID
            token: HuggingFace token for private models
        """
        from transformers import pipeline
        
        # If token is not provided, try to get it from environment
        if token is None:
            token = settings.HF_TOKEN

        logger.info(f"Loading local transformers pipeline for {model_path}...")
        self.pipe = pipeline("text2text-generation", model=model_path, token=token)
        self.is_ready = True
        
    def check_model_status(self) -> bool:
        """Check if local model is loaded"""
        return getattr(self, "is_ready", False)

    def _query_api(self, payload):
        inputs = payload.get("inputs", "")
        parameters = payload.get("parameters", {})
        try:
            result = self.pipe(inputs, **parameters)
            
            # Text2Text Generation models returning list with 'generated_text' dict
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
                return result[0]
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Local model inference failed: {e}")
            return ""
    
    def detect_hate_speech(self, text: str) -> dict:
        """
        Detect if text contains hate speech
        Returns: {"text": str, "label": str, "is_toxic": bool}
        """
        prefixed_input = "hate-speech-detection: " + text
        try:
            output = self._query_api({"inputs": prefixed_input, "parameters": {"max_new_tokens": 256}})
        except Exception:
            output = ""

        is_toxic = any(tag in output.upper() for tag in ["[HATE]", "[OFFENSIVE]", "[TOXIC]"])
        
        return {
            "text": text,
            "label": output,  # CLEAN, OFFENSIVE, or HATE
            "is_toxic": is_toxic
        }
    
    def detect_toxic_speech(self, text: str) -> dict:
        """
        Detect toxic content
        Returns: {"text": str, "label": str, "is_toxic": bool}
        """
        prefixed_input = "toxic-speech-detection: " + text
        try:
            output = self._query_api({"inputs": prefixed_input, "parameters": {"max_new_tokens": 256}})
        except Exception:
            output = ""

        is_toxic = any(tag in output.upper() for tag in ["TOXIC"])

        return {
            "text": text,
            "label": output,  # TOXIC or NONE
            "is_toxic": is_toxic
        }
    
    def detect_hate_spans(self, text: str) -> dict:
        """
        Detect exact positions and words that are hateful
        Returns: {"text": str, "bad_words": list, "indices": list}
        """
        prefixed_input = "hate-spans-detection: " + text
        try:
            output = self._query_api({"inputs": prefixed_input, "parameters": {"max_new_tokens": 256}})
        except Exception:
            output = ""
            
        # Extract hate spans from output
        bad_words, indices = self._extract_hate_spans(text, output)
        
        return {
            "text": text,
            "bad_words": bad_words,
            "indices": indices,
            "output_raw": output
        }
    
    def _extract_hate_spans(self, original_text: str, model_output: str):
        """Extract bad words from model output with [hate] tags"""
        start_tag = '[hate]'
        end_tag = '[hate]'
        
        output_lower = unicodedata.normalize('NFC', model_output.lower())
        original_lower = unicodedata.normalize('NFC', original_text.lower())
        
        bad_words = []
        start_index = output_lower.find(start_tag)
        
        while start_index != -1:
            end_index = output_lower.find(end_tag, start_index + len(start_tag))
            if end_index != -1:
                word = output_lower[start_index + len(start_tag):end_index]
                bad_words.append(word)
                start_index = output_lower.find(start_tag, end_index + len(end_tag))
            else:
                break
        
        # Find indices of bad words in original text
        indices = []
        for word in bad_words:
            idx = original_lower.find(word)
            if idx != -1:
                indices.append(list(range(idx, idx + len(word))))
        
        return bad_words, indices


class BadWordDetectionService:
    """Singleton service for bad word detection"""
    
    _instance = None
    _detector = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._detector is None:
            try:
                logger.info("Initializing VietnameseBadWordDetector...")
                self._detector = VietnameseBadWordDetector()
                logger.info("VietnameseBadWordDetector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize VietnameseBadWordDetector: {e}")
                self._detector = None
                # Don't raise here, allow service to exist but report failure later
                
    def check_model_status(self) -> bool:
        """Check if the underlying detector is initialized"""
        if self._detector is None:
            return False
        return self._detector.check_model_status()
    
    def detect_hate_speech(self, text: str) -> dict:
        """
        Detect if text contains hate speech
        Returns: {"text": str, "label": str, "is_toxic": bool}
        """
        return self._detector.detect_hate_speech(text)
    
    def detect_toxic_speech(self, text: str) -> dict:
        """
        Detect toxic content
        Returns: {"text": str, "label": str, "is_toxic": bool}
        """
        return self._detector.detect_toxic_speech(text)
    
    def detect_hate_spans(self, text: str) -> dict:
        """
        Detect exact positions and words that are hateful
        Returns: {"text": str, "bad_words": list, "indices": list}
        """
        return self._detector.detect_hate_spans(text)


# Create singleton instance
bad_word_detector = BadWordDetectionService()
