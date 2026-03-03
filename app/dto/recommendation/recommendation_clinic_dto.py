"""
DTOs for Recommendation Clinic API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PatientAppointmentRecommendationRequest(BaseModel):
    """Request DTO for patient appointment-based clinic recommendation"""
    clinic_ids: List[str] = Field(
        ..., 
        alias="clinicIds",
        description="List of clinic IDs (maximum 5) from patient's appointment history",
        min_length=1,
        max_length=5
    )
    limit: Optional[int] = Field(5, description="Maximum number of recommendations to return (default: 5)")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "clinicIds": ["clinic-id-1", "clinic-id-2", "clinic-id-3"],
                "limit": 5
            }
        }


class ClinicRecommendationRequest(BaseModel):
    """Request DTO for clinic recommendation"""
    # id: str = Field(..., description="Clinic ID")
    description: Optional[str] = Field(None, description="Clinic description")
    specialized_in: Optional[List[str]] = Field(None, alias="specializedIn", description="List of specializations")
    pros: Optional[List[str]] = Field(None, description="Clinic advantages/pros")
    paraclinical: Optional[List[str]] = Field(None, description="Paraclinical services")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "CLI001",
                "description": "Clinic description",
                "specialized_in": ["Specialization 1", "Specialization 2"],
                "pros": ["Pros 1", "Pros 2"],
                "paraclinical": ["Paraclinical 1", "Paraclinical 2"]
            }
        }


class ClinicInfo(BaseModel):
    """Clinic information"""
    id: str = Field(..., description="Clinic ID")
    email: str = Field(..., description="Clinic email")
    phone: str = Field(..., description="Clinic phone number")
    clinic_name: str = Field(..., alias="clinicName", description="Clinic name")
    description: Optional[str] = Field(None, description="Clinic description")
    specialized_in: Optional[List[str]] = Field(None, alias="specializedIn", description="List of specializations")
    pros: Optional[List[str]] = Field(None, description="Clinic advantages/pros")
    paraclinical: Optional[List[str]] = Field(None, description="Paraclinical services")
    dob: Optional[datetime] = Field(None, description="Date of birth")
    profile_picture: Optional[str] = Field(None, alias="profilePicture", description="Profile picture URL")
    created_at: datetime = Field(..., alias="createdAt", description="Creation timestamp")
    updated_at: datetime = Field(..., alias="updatedAt", description="Last update timestamp")

    class Config:
        populate_by_name = True

class ClinicInfoResponse(BaseModel):
    """Response wrapper for a single clinic info"""
    statusCode: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    data: Optional[ClinicInfo] = Field(None, description="Response data")

class RecommendationClinicData(BaseModel):
    """Response DTO for clinic recommendation"""
    recommendationsClinicAdmins: List[ClinicInfo] = Field(..., description="List of recommended clinics")
    recommendationsClinicManagers: List[ClinicInfo] = Field(..., description="List of recommended clinics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommendationsClinicAdmins": [
                    {
                        "id": "CLI001",
                        "email": "citymedicalcenter@example.com",
                        "phone": "+84123456789",
                        "clinic_name": "City Medical Center",
                        "description": "Clinic description",
                        "specialized_in": ["Specialization 1", "Specialization 2"],
                        "pros": ["Pros 1", "Pros 2"],
                        "paraclinical": ["Paraclinical 1", "Paraclinical 2"],
                        "dob": "2024-01-01T00:00:00",
                        "profile_picture": "https://example.com/profile.jpg",
                        "created_at": "2024-01-01T00:00:00",
                        "updated_at": "2024-01-01T00:00:00"
                    }
                ],
                "recommendationsClinicManagers": [
                    {
                        "id": "CLI001",
                        "email": "citymedicalcenter@example.com",
                        "phone": "+84123456789",
                        "clinic_name": "City Medical Center",
                        "description": "Clinic description",
                        "specialized_in": ["Specialization 1", "Specialization 2"],
                        "pros": ["Pros 1", "Pros 2"],
                        "paraclinical": ["Paraclinical 1", "Paraclinical 2"],
                        "dob": "2024-01-01T00:00:00",
                        "profile_picture": "https://example.com/profile.jpg",
                        "created_at": "2024-01-01T00:00:00",
                        "updated_at": "2024-01-01T00:00:00"
                    }
                ]
            }
        }


class RecommendationClinicResponse(BaseModel):
    """Response wrapper for clinic recommendation"""
    statusCode: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    data: Optional[RecommendationClinicData] = Field(None, description="Response data")

