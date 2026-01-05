"""
DTOs for Bad Word Detection API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class BadWordDetectionRequest(BaseModel):
    """Request DTO for bad word detection"""
    text: str = Field(..., description="Text to analyze for bad words")
    detection_type: Optional[str] = Field(
        "all",
        description="Detection type: 'hate_speech', 'toxic', 'hate_spans', or 'all'"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Đây là một đoạn văn bản cần kiểm tra",
                "detection_type": "all"
            }
        }


class HateSpeechResult(BaseModel):
    """Result for hate speech detection"""
    text: str = Field(..., description="Original text")
    label: str = Field(..., description="Classification label (CLEAN, OFFENSIVE, or HATE)")
    is_toxic: bool = Field(..., description="Whether the text is toxic")


class ToxicSpeechResult(BaseModel):
    """Result for toxic speech detection"""
    text: str = Field(..., description="Original text")
    label: str = Field(..., description="Classification label (TOXIC or NONE)")
    is_toxic: bool = Field(..., description="Whether the text is toxic")


class HateSpansResult(BaseModel):
    """Result for hate spans detection"""
    text: str = Field(..., description="Original text")
    bad_words: List[str] = Field(default_factory=list, description="List of detected bad words")
    indices: List[List[int]] = Field(default_factory=list, description="Character indices of bad words")
    output_raw: Optional[str] = Field(None, description="Raw model output")


class BadWordDetectionResponse(BaseModel):
    """Response DTO for bad word detection"""
    text: str = Field(..., description="Original text analyzed")
    is_toxic: bool = Field(..., description="Overall toxicity flag")
    hate_speech: Optional[HateSpeechResult] = Field(None, description="Hate speech detection result")
    # toxic_speech: Optional[ToxicSpeechResult] = Field(None, description="Toxic speech detection result")
    hate_spans: Optional[HateSpansResult] = Field(None, description="Hate spans detection result")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Đây là một đoạn văn bản cần kiểm tra",
                "is_toxic": False,
                "hate_speech": {
                    "text": "Đây là một đoạn văn bản cần kiểm tra",
                    "label": "CLEAN",
                    "is_toxic": False
                },
                # "toxic_speech": {
                #     "text": "Đây là một đoạn văn bản cần kiểm tra",
                #     "label": "NONE",
                #     "is_toxic": False
                # },
                "hate_spans": {
                    "text": "Đây là một đoạn văn bản cần kiểm tra",
                    "bad_words": [],
                    "indices": [],
                    "output_raw": ""
                },
                "analyzed_at": "2024-01-01T00:00:00"
            }
        }


class BatchBadWordDetectionRequest(BaseModel):
    """Request DTO for batch bad word detection"""
    texts: List[str] = Field(..., description="List of texts to analyze")
    detection_type: Optional[str] = Field(
        "all",
        description="Detection type: 'hate_speech', 'toxic', 'hate_spans', or 'all'"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Đây là văn bản thứ nhất",
                    "Đây là văn bản thứ hai"
                ],
                "detection_type": "all"
            }
        }


class BatchBadWordDetectionResponse(BaseModel):
    """Response DTO for batch bad word detection"""
    results: List[BadWordDetectionResponse] = Field(..., description="List of detection results")
    total_analyzed: int = Field(..., description="Total number of texts analyzed")
    total_toxic: int = Field(..., description="Total number of toxic texts found")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
