"""
Keyword Search Service

PostgreSQL full-text search integration using tsvector and tsquery.
"""
from typing import List, Tuple, Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.utils.logger import logger


class KeywordSearchService:
    """
    PostgreSQL Full-Text Search Service
    
    Provides keyword-based search using PostgreSQL's built-in
    full-text search capabilities (tsvector, tsquery, ts_rank).
    
    Usage:
        keyword_search = KeywordSearchService()
        docs = await keyword_search.search(db, "dental clinic", k=5)
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize the keyword search service.
        
        Args:
            language: PostgreSQL text search configuration (default: english)
        """
        self.language = language
    
    async def search(
        self,
        db: AsyncSession,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeBase]:
        """
        Search documents using full-text search.
        
        Args:
            db: Database session
            query: Search query string
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching KnowledgeBase records
        """
        results = await self.search_with_score(db, query, k, filter_metadata)
        return [doc for doc, _ in results]
    
    async def search_with_score(
        self,
        db: AsyncSession,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[KnowledgeBase, float]]:
        """
        Search documents with relevance scores.
        
        Args:
            db: Database session
            query: Search query string
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples (KnowledgeBase, relevance_score)
        """
        try:
            logger.info(f"============Keyword search received query: {query}")
            # Use websearch_to_tsquery for natural language queries (handles quotes, +/-)
            # But plainto_tsquery enforces AND. For Hybrid Search recall, we often want specific keywords.
            # However, "give me 5 doctors" failing because of "give" is bad.
            # We'll use a relaxed approach: convert words to OR query for recall, let ts_rank sort them.
            
            # Simple sanitization and OR-joining
            # Filter out empty strings
            terms = [t for t in query.split() if t.strip()]
            if not terms:
                 # fallback for empty query
                 or_query = ""
            else:
                 or_query = " | ".join(terms)
            
            # We use to_tsquery which accepts operators like |
            # Use stored search_vector column for performance and better ranking
            sql = text(f"""
                SELECT 
                    _id,
                    content,
                    embedding,
                    metadata,
                    created_at,
                    updated_at,
                    deleted_at,
                    ts_rank_cd(
                        search_vector,
                        to_tsquery('{self.language}', :or_query)
                    ) as rank
                FROM knowledge_base
                WHERE deleted_at IS NULL
                    AND search_vector @@ to_tsquery('{self.language}', :or_query)
            """)
            
            params = {"or_query": or_query, "k": k}
            
            # Apply metadata filters
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key == 'type' or key == 'doc_type':
                        # Handle type filtering (list or string)
                        if isinstance(value, list):
                            # Handle list of types
                            conditions = []
                            for i, v in enumerate(value):
                                param_name = f"type_filter_{i}"
                                conditions.append(f"metadata->>'type' = :{param_name}")
                                params[param_name] = v
                            
                            sql = text(str(sql) + f" AND ({' OR '.join(conditions)})")
                        else:
                            # Single value
                            sql = text(str(sql) + " AND metadata->>'type' = :type_filter")
                            params['type_filter'] = value
                    else:
                        # Generic metadata filter
                        param_name = f"filter_{key}"
                        sql = text(str(sql) + f" AND metadata->>'{key}' = :{param_name}")
                        params[param_name] = str(value)
            
            sql = text(str(sql) + " ORDER BY rank DESC LIMIT :k")
            
            result = await db.execute(sql, params)
            rows = result.fetchall()
            
            documents_with_scores = []
            for row in rows:
                kb = KnowledgeBase(
                    _id=row._id,
                    content=row.content,
                    embedding=row.embedding,
                    metadata=row.metadata,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    deleted_at=row.deleted_at,
                )
                rank = float(row.rank) if row.rank else 0.0
                documents_with_scores.append((kb, rank))
            
            logger.info(f"Keyword search found {len(documents_with_scores)} documents")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            raise
    
    async def search_phrase(
        self,
        db: AsyncSession,
        phrase: str,
        k: int = 5
    ) -> List[KnowledgeBase]:
        """
        Search for an exact phrase in documents.
        
        Args:
            db: Database session
            phrase: Exact phrase to search for
            k: Number of results to return
            
        Returns:
            List of matching KnowledgeBase records
        """
        try:
            # Use phraseto_tsquery for phrase matching
            sql = text(f"""
                SELECT 
                    _id,
                    content,
                    embedding,
                    metadata,
                    created_at,
                    updated_at,
                    deleted_at,
                    ts_rank_cd(
                        to_tsvector('{self.language}', content),
                        phraseto_tsquery('{self.language}', :phrase)
                    ) as rank
                FROM knowledge_base
                WHERE deleted_at IS NULL
                    AND to_tsvector('{self.language}', content) @@ phraseto_tsquery('{self.language}', :phrase)
                ORDER BY rank DESC
                LIMIT :k
            """)
            
            result = await db.execute(sql, {"phrase": phrase, "k": k})
            rows = result.fetchall()
            
            documents = []
            for row in rows:
                kb = KnowledgeBase(
                    _id=row._id,
                    content=row.content,
                    embedding=row.embedding,
                    metadata=row.metadata,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    deleted_at=row.deleted_at,
                )
                documents.append(kb)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in phrase search: {e}")
            raise


# Singleton instance
keyword_search_service = KeywordSearchService()
