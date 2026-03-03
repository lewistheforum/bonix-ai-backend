"""
Controller/Router for Recommendation Clinic API
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from app.dto.recommendation.recommendation_clinic_dto import (
    ClinicRecommendationRequest,
    PatientAppointmentRecommendationRequest,
    RecommendationClinicData,
    RecommendationClinicResponse,
    ClinicInfo,
    ClinicInfoResponse
)
from app.services.recommendation.recommendation_clinic_service import recommendation_clinic_service
from app.common.api_response import ApiResponse
from app.common.message.status_code import StatusCode
from app.common.message.success_message import SuccessMessage
from app.common.message.error_message import ErrorMessage

router = APIRouter()


@router.get("/clinics/{clinic_id}", response_model=ClinicInfoResponse)
async def get_clinic_by_id(clinic_id: str):
    """
    Get a specific clinic by its ID
    
    - **clinic_id**: The unique identifier of the clinic
    """
    try:
        clinic = await recommendation_clinic_service.get_clinic_by_id(clinic_id)
        if not clinic:
            raise HTTPException(status_code=StatusCode.NOT_FOUND, detail=ErrorMessage.CLINIC_NOT_FOUND)
        
        data = ClinicInfo(
            id=clinic["id"],
            email=clinic["email"],
            phone=clinic["phone"],
            clinic_name=clinic["clinic_name"],
            description=clinic["description"],
            specialized_in=clinic["specialized_in"],
            pros=clinic["pros"],
            paraclinical=clinic["paraclinical"],
            dob=clinic["dob"],
            profile_picture=clinic["profile_picture"],
            created_at=clinic["created_at"],
            updated_at=clinic["updated_at"]
        )
        return ClinicInfoResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/clinics/{clinic_id}/similar", response_model=RecommendationClinicResponse)
async def get_similar_clinics(clinic_id: str):
    """
    Get similar clinics based on a specific clinic's data for recommendation.
    Returns only clinics that are genuinely similar (above a similarity threshold).
    
    - **clinic_id**: The unique identifier of the clinic to find similar clinics for
    """
    try:
        # First check if the clinic exists
        clinic = await recommendation_clinic_service.get_clinic_by_id(clinic_id)
        if not clinic:
            raise HTTPException(status_code=StatusCode.NOT_FOUND, detail=ErrorMessage.CLINIC_NOT_FOUND)
        
        result = await recommendation_clinic_service.get_similar_clinics(clinic_id)
        return RecommendationClinicResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/clinics/recommend/patient-appointment", response_model=RecommendationClinicResponse)
async def get_recommendations_from_patient_appointments(
    request: PatientAppointmentRecommendationRequest
):
    """
    Get clinic recommendations based on patient's appointment history.
    
    This endpoint analyzes the given clinic IDs (from patient's past appointments)
    and recommends clinics that are similar/familiar to those clinics.
    
    - **clinicIds**: List of clinic IDs (maximum 5) from patient's appointment history
    - **limit**: Maximum number of recommendations to return (default: 5)
    
    The recommendation algorithm:
    1. Aggregates characteristics (specializations, pros, paraclinical services) from all input clinics
    2. Calculates similarity scores for all other clinics
    3. Adds frequency bonus for clinics matching characteristics that appear multiple times
    4. Returns top recommended clinics sorted by score
    """
    try:
        result = await recommendation_clinic_service.get_recommendations_from_patient_appointments(
            clinic_ids=request.clinic_ids,
            limit=request.limit or 5
        )
        return RecommendationClinicResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=result)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/db/tables", response_model=ApiResponse[Dict[str, Any]])
async def list_database_tables():
    """
    List all tables in the database (for debugging purposes)
    """
    from sqlalchemy import text
    from app.database import AsyncSessionLocal
    
    try:
        from app.config import settings
        
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(query)
            rows = result.fetchall()
            tables = [row[0] for row in rows]
        
        # Mask password for security
        db_url = settings.DATABASE_URL
        masked_url = db_url.split("@")[1] if "@" in db_url else "unknown"
            
        data = {
            "tables": tables, 
            "count": len(tables),
            "database_info": {
                "host_and_db": masked_url,
                "db_name_setting": settings.DB_NAME
            }
        }
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))

