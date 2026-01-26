"""
Knowledge Base Model

SQLAlchemy model for storing knowledge base documents with vector embeddings
and full-text search support using PostgreSQL pgvector extension.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Text, DateTime, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from pgvector.sqlalchemy import Vector

from app.database import Base


class KnowledgeBase(Base):
    """
    Knowledge Base Entity
    
    Stores documents with vector embeddings for RAG retrieval.
    Supports both vector similarity search and full-text keyword search.
    
    Attributes:
        _id: Unique identifier (UUID)
        content: Document text content
        embedding: Vector embedding (1536 dimensions for OpenAI text-embedding-3-small)
        metadata: Additional metadata (source, doc_type, clinic_id, etc.)
        search_vector: PostgreSQL tsvector for full-text search
        created_at: Creation timestamp
        updated_at: Last update timestamp
        deleted_at: Soft delete timestamp
    """
    __tablename__ = "knowledge_base"
    
    _id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    search_vector = Column(TSVECTOR, nullable=True)  # Will be tsvector in PostgreSQL
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Indexes for efficient searching
    __table_args__ = (
        # Index for vector similarity search (using IVFFlat or HNSW)
        Index(
            'idx_knowledge_base_embedding',
            embedding,
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        # GIN index for full-text search
        Index(
            'idx_knowledge_base_search_vector',
            text("to_tsvector('english', content)"),
            postgresql_using='gin'
        ),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeBase(id={self._id}, content={self.content[:50]}...)>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "_id": str(self._id),
            "content": self.content,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
