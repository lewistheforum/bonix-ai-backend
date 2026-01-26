"""
Vector Store Service

PostgreSQL + pgvector integration for vector similarity search.
"""
from typing import List, Tuple, Optional, Dict, Any
import uuid

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.services.rag.embeddings_service import embeddings_service
from app.utils.logger import logger


class VectorStoreService:
    """
    PostgreSQL Vector Store Service
    
    Provides vector similarity search using pgvector extension.
    Uses cosine similarity for finding semantically similar documents.
    
    Usage:
        vector_store = VectorStoreService()
        docs = await vector_store.similarity_search(db, "query text", k=5)
    """
    
    def __init__(self):
        """Initialize the vector store service."""
        self.embeddings_service = embeddings_service
    
    async def add_document(
        self,
        db: AsyncSession,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeBase:
        """
        Add a single document to the knowledge base.
        
        Args:
            db: Database session
            content: Document text content
            metadata: Optional metadata dictionary
            
        Returns:
            Created KnowledgeBase record
        """
        try:
            # Generate embedding
            embedding = await self.embeddings_service.embed_text(content)
            
            # Create knowledge base entry
            kb_entry = KnowledgeBase(
                _id=uuid.uuid4(),
                content=content,
                embedding=embedding,
                meta_data=metadata or {},
            )
            
            db.add(kb_entry)
            await db.flush()
            
            logger.info(f"Added document to knowledge base: {kb_entry._id}")
            return kb_entry
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def add_documents(
        self,
        db: AsyncSession,
        documents: List[Dict[str, Any]]
    ) -> List[KnowledgeBase]:
        """
        Add multiple documents to the knowledge base.
        
        Args:
            db: Database session
            documents: List of dicts with 'content' and optional 'metadata' keys
            
        Returns:
            List of created KnowledgeBase records
        """
        try:
            contents = [doc["content"] for doc in documents]
            embeddings = await self.embeddings_service.embed_documents(contents)
            
            kb_entries = []
            for doc, embedding in zip(documents, embeddings):
                kb_entry = KnowledgeBase(
                    _id=uuid.uuid4(),
                    content=doc["content"],
                    embedding=embedding,
                    meta_data=doc.get("metadata", {}),
                )
                db.add(kb_entry)
                kb_entries.append(kb_entry)
            
            await db.flush()
            logger.info(f"Added {len(kb_entries)} documents to knowledge base")
            return kb_entries
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def similarity_search(
        self,
        db: AsyncSession,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeBase]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            db: Database session
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar KnowledgeBase records
        """
        results = await self.similarity_search_with_score(db, query, k, filter_metadata)
        return [doc for doc, _ in results]
    
    async def similarity_search_with_score(
        self,
        db: AsyncSession,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[KnowledgeBase, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            db: Database session
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples (KnowledgeBase, similarity_score)
        """
        try:
            # Generate query embedding
            query_embedding = await self.embeddings_service.embed_text(query)
            
            # Use SQLAlchemy ORM with pgvector operators
            # 1 - cosine_distance gives cosine similarity
            # .cosine_distance() uses the <=> operator
            similarity_expr = (1 - KnowledgeBase.embedding.cosine_distance(query_embedding)).label("similarity")
            
            # Base query
            query_obj = select(
                KnowledgeBase,
                similarity_expr
            ).where(
                KnowledgeBase.deleted_at.is_(None)
            )
            
            # Prepare params dict
            params = {}
            
            # Apply metadata filters
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key == 'type' or key == 'doc_type':
                        # Handle type filtering (list or string)
                        # Value can be a single string or a list of strings
                        if isinstance(value, list):
                            # Construct OR clause dynamically for list
                            conditions = []
                            for i, v in enumerate(value):
                                param_name = f"type_filter_{i}"
                                conditions.append(f"metadata->>'type' = :{param_name}")
                                params[param_name] = v
                            
                            query_obj = query_obj.where(text(f"({' OR '.join(conditions)})"))
                            
                        else:
                            # Single value
                            query_obj = query_obj.where(
                                text("metadata->>'type' = :type_filter")
                            )
                            params['type_filter'] = value
                    else:
                        # Generic metadata filter
                        query_obj = query_obj.where(
                            KnowledgeBase.meta_data[key].astext == str(value)
                        )
            
            query_obj = query_obj.order_by(
                KnowledgeBase.embedding.cosine_distance(query_embedding)
            ).limit(k)
            
            result = await db.execute(query_obj, params)
            
            # Result contains (KnowledgeBase, similarity) tuples because of the selection
            documents_with_scores = []
            for row in result:
                # row is (KnowledgeBase, similarity)
                kb = row[0]
                similarity = float(row[1]) if row[1] is not None else 0.0
                documents_with_scores.append((kb, similarity))
            
            logger.info(f"Found {len(documents_with_scores)} similar documents for query")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise


# Singleton instance
vector_store_service = VectorStoreService()
