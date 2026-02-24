"""
RAG DTOs (Data Transfer Objects)

Request and response models for RAG chatbot API endpoints.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    """Chat history item for request."""
    role: str
    content: str

class RAGChatRequest(BaseModel):
    """Request model for RAG chat endpoint."""
    
    message: str = Field(
        ...,
        description="User's chat message",
        min_length=1,
        max_length=4000
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID to continue, or None for new conversation"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for tracking and personalization"
    )
    history: List[ChatHistoryItem] = Field(
        default_factory=list,
        description="Chat history for context"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What dental clinics are available near me?",
                "conversation_id": None,
                "user_id": "user-123",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there! How can I help you?"}
                ]
            }
        }


class RAGChatResponse(BaseModel):
    """Response model for RAG chat endpoint."""
    
    response: str = Field(
        description="AI-generated response"
    )
    conversation_id: str = Field(
        description="Conversation ID (new or existing)"
    )
    context_used: bool = Field(
        default=False,
        description="Whether knowledge base context was used"
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used for the response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on our records, here are some dental clinics...",
                "conversation_id": "conv-uuid-123",
                "context_used": True,
                "sources": [{"doc_id": "doc-1", "score": 0.85}],
                "timestamp": "2024-01-19T12:00:00Z"
            }
        }


class KnowledgeBaseIngestRequest(BaseModel):
    """Request model for knowledge base document ingestion."""
    
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="List of documents to ingest. Each document should have 'content' and optional 'metadata'",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "Our dental clinic offers teeth whitening services...",
                        "metadata": {"clinic_id": "clinic-1", "service_type": "dental"}
                    }
                ]
            }
        }


class KnowledgeBaseIngestResponse(BaseModel):
    """Response model for knowledge base ingestion."""
    
    success: bool
    documents_ingested: int
    message: str


class KnowledgeBaseSearchRequest(BaseModel):
    """Request model for knowledge base search."""
    
    query: str = Field(
        ...,
        description="Search query",
        min_length=1
    )
    k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=20
    )
    search_type: str = Field(
        default="hybrid",
        description="Search type: 'vector', 'keyword', or 'hybrid'"
    )


class KnowledgeBaseSearchResult(BaseModel):
    """Single search result."""
    
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseSearchResponse(BaseModel):
    """Response model for knowledge base search."""
    
    query: str
    results: List[KnowledgeBaseSearchResult]
    total: int
    search_type: str


class SyncKnowledgeBaseRequest(BaseModel):
    """Request model for syncing knowledge base from database."""
    
    sync_clinic_services: bool = Field(
        default=True,
        description="Sync clinic services data"
    )
    sync_doctor_profiles: bool = Field(
        default=True,
        description="Sync doctor profile data"
    )
    sync_clinic_info: bool = Field(
        default=True,
        description="Sync clinic information"
    )
    sync_staff_info: bool = Field(
        default=True,
        description="Sync staff information"
    )
    sync_blogs: bool = Field(
        default=True,
        description="Sync blog information"
    )
    sync_feedbacks: bool = Field(
        default=True,
        description="Sync doctor and clinic feedback"
    )
    sync_user_info: bool = Field(
        default=True,
        description="Sync patient user information"
    )
    sync_doctor_schedules: bool = Field(
        default=True,
        description="Sync doctor schedules"
    )
    sync_clinic_working_hours: bool = Field(
        default=True,
        description="Sync clinic working hours"
    )
    clear_existing: bool = Field(
        default=False,
        description="Clear existing data before sync"
    )


class SyncKnowledgeBaseResponse(BaseModel):
    """Response model for knowledge base sync."""
    
    success: bool
    clinic_services_synced: int
    doctor_profiles_synced: int
    clinic_info_synced: int
    staff_info_synced: int
    blogs_synced: int
    feedbacks_synced: int
    user_info_synced: int
    doctor_schedules_synced: int
    clinic_working_hours_synced: int
    total_synced: int
    message: str


class SyncMedicineKnowledgeBaseRequest(BaseModel):
    """Request model for syncing medicine knowledge base."""
    
    clear_existing: bool = Field(
        default=False,
        description="Clear existing medicine knowledge base data before sync"
    )


class SyncMedicineKnowledgeBaseResponse(BaseModel):
    """Response model for medicine knowledge base sync."""
    
    success: bool
    therapeutic_classes_synced: int
    total_medicines_processed: int
    message: str


class ConversationHistoryRequest(BaseModel):
    """Request model for getting conversation history."""
    
    conversation_id: str = Field(
        ...,
        description="Conversation ID"
    )
    limit: int = Field(
        default=50,
        description="Maximum number of messages to return",
        ge=1,
        le=100
    )


class MessageItem(BaseModel):
    """Single message in conversation history."""
    
    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    
    conversation_id: str
    messages: List[MessageItem]
    total: int
