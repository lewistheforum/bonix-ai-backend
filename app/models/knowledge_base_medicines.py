"""
Knowledge Base Medicines Model

SQLAlchemy model for storing medicine knowledge base documents grouped by
therapeutic class, with vector embeddings and full-text search support.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from pgvector.sqlalchemy import Vector

from app.database import Base


class KnowledgeBaseMedicines(Base):
    """
    Knowledge Base Medicines Entity

    Stores medicine data grouped by therapeutic class with vector embeddings
    for RAG retrieval. Each row represents one therapeutic class containing
    aggregated medicine information.

    Attributes:
        _id: Unique identifier (UUID)
        medicine_category: Aggregated medicine content for this therapeutic class
        embedding: Vector embedding (1536 dimensions for OpenAI text-embedding-3-small)
        metadata: Additional metadata (therapeutic_class, medicine_count, etc.)
        search_vector: PostgreSQL tsvector for full-text search
        created_at: Creation timestamp
        updated_at: Last update timestamp
        deleted_at: Soft delete timestamp
    """
    __tablename__ = "knowledge_base_medicines"

    _id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    medicine_category = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    search_vector = Column(TSVECTOR, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Table is managed by the primary backend server
    __table_args__ = {
        'extend_existing': True
    }

    def __repr__(self) -> str:
        return f"<KnowledgeBaseMedicines(id={self._id}, category={self.medicine_category[:50]}...)>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "_id": str(self._id),
            "medicine_category": self.medicine_category,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
