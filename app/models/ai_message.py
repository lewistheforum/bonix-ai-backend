"""
AI Message Model

SQLAlchemy model for storing AI chat messages with optional embeddings.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class AIMessage(Base):
    """
    AI Message Entity
    
    Stores individual messages within a conversation.
    Optionally includes embeddings for semantic search within conversation history.
    
    Attributes:
        _id: Unique identifier (UUID)
        conversation_id: Foreign key to ai_conversations
        sender_id: UUID of the message sender (user or system)
        role: Message role ('user', 'assistant', 'system')
        content: Message text content
        embedding: Optional vector embedding (1536 dimensions)
        metadata: Additional metadata (tool_calls, sources, etc.)
        deleted_by: UUID of user who deleted the message
        created_at: Creation timestamp
        updated_at: Last update timestamp
        deleted_at: Soft delete timestamp
    """
    __tablename__ = "ai_messages"
    
    _id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("ai_conversations._id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(UUID(as_uuid=True), nullable=True)
    role = Column(Text, nullable=False, default="user")
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    deleted_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship to conversation
    conversation = relationship("AIConversation", back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<AIMessage(id={self._id}, role={self.role}, content={self.content[:30]}...)>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "_id": str(self._id),
            "conversation_id": str(self.conversation_id),
            "sender_id": str(self.sender_id) if self.sender_id else None,
            "role": self.role,
            "content": self.content,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
