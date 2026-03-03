"""
DTOs for Label Feedback API
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class LabelFeedbackRequest(BaseModel):
    """Request DTO for label feedback"""
    text: str = Field(..., description="Text to analyze and classify")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Bác sĩ khám rất kỹ và 'mát tay', mình mới nắn chỉnh 2 buổi mà lưng đã đỡ đau hẳn. Tuy nhiên, khâu làm thủ tục hành chính còn rườm rà quá, mình đặt lịch trước rồi mà vẫn phải chờ hơn 45 phút mới được vào."
            }
        }

class LabelFeedbackResult(BaseModel):
    """Single prediction result"""
    label: str = Field(..., description="Predicted label")
    score: float = Field(..., description="Confidence score")

class LabelFeedbackData(BaseModel):
    """Response DTO for label feedback"""
    results: List[LabelFeedbackResult] = Field(..., description="List of top predictions")

class LabelFeedbackResponse(BaseModel):
    """Response wrapper for label feedback"""
    statusCode: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    data: Optional[LabelFeedbackData] = Field(None, description="Response data")

class LabelImageRequest(BaseModel):
    """Request DTO for label image"""
    image_url: str = Field(..., description="URL of the image to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://ykhoacantho.com/wp-content/uploads/2022/09/kham-benh-xuong-khop-can-tho-2022-13.jpg"
            }
        }

class LabelImageData(BaseModel):
    """Response DTO for label image"""
    description: str = Field(..., description="Description of the image")

class LabelImageResponse(BaseModel):
    """Response wrapper for label image"""
    statusCode: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    data: Optional[LabelImageData] = Field(None, description="Response data")
