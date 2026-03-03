"""
DTOs for Chatbot API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")


class ChatbotRequest(BaseModel):
    """Request DTO for chatbot"""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    patient_id: Optional[str] = Field(None, description="Patient ID if applicable")
    context: Optional[dict] = Field(None, description="Additional context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "I have a headache and fever. What should I do?",
                "conversation_id": "CONV123",
                "patient_id": "PAT001",
                "context": {"language": "en"}
            }
        }


class ChatbotData(BaseModel):
    """Response DTO for chatbot"""
    response: str = Field(..., description="Chatbot response")
    conversation_id: str = Field(..., description="Conversation ID")
    suggested_actions: Optional[List[str]] = Field(None, description="Suggested actions")
    confidence_score: Optional[float] = Field(None, description="Response confidence score")
    generated_at: datetime = Field(..., description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on your symptoms, I recommend consulting with a doctor. You may have a common cold or flu. Rest, stay hydrated, and monitor your temperature.",
                "conversation_id": "CONV123",
                "suggested_actions": [
                    "Schedule appointment",
                    "Find nearby clinic",
                    "Get medication advice"
                ],
                "confidence_score": 0.82,
            }
        }


class ChatbotResponse(BaseModel):
    """Response wrapper for chatbot"""
    statusCode: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    data: Optional[ChatbotData] = Field(None, description="Response data")


class ConversationHistoryData(BaseModel):
    """Response DTO for conversation history"""
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    total_messages: int = Field(..., description="Total number of messages")

class ConversationHistoryResponse(BaseModel):
    """Response wrapper for conversation history"""
    statusCode: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    data: Optional[ConversationHistoryData] = Field(None, description="Response data")

