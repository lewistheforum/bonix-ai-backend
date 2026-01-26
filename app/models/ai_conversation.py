"""
AI Conversation Model

SQLAlchemy model for storing AI chat conversations.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class AIConversation(Base):
    """
    AI Conversation Entity
    
    Stores conversation sessions for the RAG chatbot.
    
    Attributes:
        _id: Unique identifier (UUID)
        title: Conversation title (auto-generated or user-defined)
        description: Optional conversation description
        participants: JSONB array of participant IDs
        deleted_by: UUID of user who deleted the conversation
        metadata: Additional metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
        deleted_at: Soft delete timestamp
    """
    __tablename__ = "ai_conversations"
    
    _id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)
    description = Column(String(255), nullable=True)
    participants = Column(JSONB, nullable=True, default=list)
    deleted_by = Column(UUID(as_uuid=True), nullable=True)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship to messages
    messages = relationship("AIMessage", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<AIConversation(id={self._id}, title={self.title})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "_id": str(self._id),
            "title": self.title,
            "description": self.description,
            "participants": self.participants,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
