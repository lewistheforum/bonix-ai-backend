from fastapi import APIRouter, HTTPException
import os
import importlib.util

# Dynamic import for DTO
dto_path = os.path.join(os.path.dirname(__file__), "../../../dto/label_feedback/label_feedback_dto.py")
dto_path = os.path.abspath(dto_path)
spec = importlib.util.spec_from_file_location("label_feedback_dto", dto_path)
label_feedback_dto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_feedback_dto)

LabelFeedbackRequest = label_feedback_dto.LabelFeedbackRequest
LabelFeedbackResponse = label_feedback_dto.LabelFeedbackResponse
LabelImageRequest = label_feedback_dto.LabelImageRequest
LabelImageResponse = label_feedback_dto.LabelImageResponse

# Dynamic import for Service
service_path = os.path.join(os.path.dirname(__file__), "../../../services/label_feedback/label_feedback_service.py")
service_path = os.path.abspath(service_path)
spec = importlib.util.spec_from_file_location("label_feedback_service", service_path)
label_feedback_service_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_feedback_service_module)

label_feedback_service = label_feedback_service_module.label_feedback_service

router = APIRouter()

@router.post("/label-description", response_model=LabelFeedbackResponse)
async def label_description(request: LabelFeedbackRequest):
    """
    Analyze text feedback and return label predictions.
    """
    try:
        results = label_feedback_service.predict(request.text)
        return LabelFeedbackResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/label-image", response_model=LabelImageResponse)
async def label_image(request: LabelImageRequest):
    """
    Analyze image and return detailed description.
    """
    try:
        description = label_feedback_service.describe_image(request.image_url)
        return LabelImageResponse(description=description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
