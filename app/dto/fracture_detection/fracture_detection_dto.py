from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Minimum x coordinate")
    y_min: float = Field(..., description="Minimum y coordinate")
    x_max: float = Field(..., description="Maximum x coordinate")
    y_max: float = Field(..., description="Maximum y coordinate")
    confidence: float = Field(..., description="Confidence score [0, 1]")
    class_name: str = Field(..., description="Predicted class name")

class AIResultAnalyze(BaseModel):
    analyze: str = Field(..., description="Medical analysis of the detected conditions")
    treatment_plan: List[str] = Field(default_factory=list, description="Recommended treatment steps")
    medicine_categories: List[str] = Field(default_factory=list, description="Matched therapeutic categories from knowledge base")
    medicines: List[str] = Field(default_factory=list, description="Recommended medications")

class FractureDetectionResponse(BaseModel):
    has_fracture: bool = Field(..., description="True if at least one fracture is detected")
    detections: List[BoundingBox] = Field(default_factory=list, description="List of detected fractures")
    annotated_image_base64: Optional[str] = Field(default=None, description="Base64 encoded JPEG of the image with bounding boxes drawn")
    ai_result_analyze: Optional[AIResultAnalyze] = Field(default=None, description="AI-powered medical analysis from OpenAI")
    processing_time_ms: float = Field(..., description="Time taken to process the image in milliseconds")
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Timestamp of analysis")
