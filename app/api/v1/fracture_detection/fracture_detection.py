from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.common.api_response import ApiResponse
from app.common.message.status_code import StatusCode
from app.common.message.success_message import SuccessMessage
from app.common.message.error_message import ErrorMessage
from app.database import get_db
from app.utils.logger import logger

from app.dto.fracture_detection.fracture_detection_dto import FractureDetectionResponse
from app.services.fracture_detection.fracture_detection_service import fracture_detector

router = APIRouter()

@router.post("/detect", response_model=FractureDetectionResponse)
async def detect_fracture(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(default=None, description="Patient notes: The patient has liver disease, diabetes, and an allergy to chicken., etc."),
    db: AsyncSession = Depends(get_db)
):
    """
    Detect wrist fractures in an uploaded X-ray image.
    
    3-step pipeline:
      1. Classify — OpenAI Vision verifies this is a wrist X-ray
      2. Detect  — YOLO model detects fractures and generates annotated image
      3. Analyze — OpenAI provides medical analysis, treatment plan, and medications
                   (constrained to medicine categories in knowledge_base_medicines)
    
    Args:
        file: X-ray image (JPEG/PNG)
        notes: Optional patient info (allergies, medical history, etc.) to guide medicine recommendations
    """
    content_type = file.content_type.split(';')[0].strip()
    if content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise HTTPException(
            status_code=StatusCode.BAD_REQUEST, 
            detail="File provided must be an image (PNG or JPG)"
        )
        
    try:
        # Read the file contents as bytes
        image_bytes = await file.read()
        
        # Step 1: Classify — Verify it is a wrist X-ray
        is_wrist_xray = await fracture_detector.verify_is_wrist_xray(image_bytes)
        
        if not is_wrist_xray:
            logger.warning("Image failed OpenAI Vision wrist X-ray validation")
            raise HTTPException(
                status_code=StatusCode.BAD_REQUEST, 
                detail="The uploaded image is not a wrist X-ray. Please upload a valid wrist X-ray image."
            )
            
        # Step 2 & 3: Detect fractures with YOLO, then analyze with OpenAI
        result = await fracture_detector.detect_fracture(image_bytes, db=db, notes=notes)
        
        return FractureDetectionResponse(
            statusCode=StatusCode.SUCCESS, 
            message=SuccessMessage.INDEX, 
            data=result
        )
    except HTTPException:
        # Re-raise HTTP exceptions so FastAPI can handle them correctly
        raise
    except Exception as e:
        # Wrap the exception in the standard expected format
        raise HTTPException(
            status_code=StatusCode.INTERNAL_ERROR, 
            detail=f"Error processing the image for fracture detection: {str(e)}"
        )

@router.get("/health", response_model=ApiResponse[dict])
async def health_check():
    """
    Health check endpoint for fracture detection service
    """
    data = {
        "status": "healthy", 
        "service": "fracture-detection",
        "model_loaded": fracture_detector.is_loaded
    }
    return ApiResponse(
        statusCode=StatusCode.SUCCESS, 
        message=SuccessMessage.INDEX, 
        data=data
    )
