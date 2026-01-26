"""
Hybrid Retriever

Combines vector similarity search and keyword search using
Reciprocal Rank Fusion (RRF) for improved retrieval accuracy.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.services.rag.vector_store_service import vector_store_service
from app.services.rag.keyword_search_service import keyword_search_service
from app.utils.logger import logger


@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata"""
    document: KnowledgeBase
    score: float
    source: str  # 'vector', 'keyword', or 'hybrid'


class HybridRetriever:
    """
    Hybrid Retriever
    
    Combines vector similarity search (semantic) and keyword search (lexical)
    using Reciprocal Rank Fusion (RRF) to get the best of both approaches.
    
    Vector search excels at semantic understanding while keyword search
    handles exact term matching and technical terminology better.
    
    Usage:
        retriever = HybridRetriever()
        docs = await retriever.retrieve(db, "dental implant procedure", k=5)
    """
    
    def __init__(
        self,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        rrf_k: int = 60
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            rrf_k: RRF constant (default 60, higher = smoother ranking)
        """
        self.vector_weight = vector_weight or settings.VECTOR_SEARCH_WEIGHT
        self.keyword_weight = keyword_weight or settings.KEYWORD_SEARCH_WEIGHT
        self.rrf_k = rrf_k
        
        # Normalize weights
        total = self.vector_weight + self.keyword_weight
        self.vector_weight /= total
        self.keyword_weight /= total
    
    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            db: Database session
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of RetrievalResult objects sorted by hybrid score
        """
        k = k or settings.RETRIEVAL_TOP_K
        
        # Fetch more results than needed for better fusion
        fetch_k = k * 2
        
        try:
            # Run both searches concurrently
            vector_results = await vector_store_service.similarity_search_with_score(
                db, query, fetch_k, filter_metadata
            )
            keyword_results = await keyword_search_service.search_with_score(
                db, query, fetch_k, filter_metadata
            )
            
            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                vector_results,
                keyword_results
            )
            
            # Return top k results
            return fused_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            raise
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[KnowledgeBase, float]],
        keyword_results: List[Tuple[KnowledgeBase, float]]
    ) -> List[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion to combine results.
        
        RRF Score = sum(1 / (k + rank)) for each result list
        
        Args:
            vector_results: Results from vector search with scores
            keyword_results: Results from keyword search with scores
            
        Returns:
            Fused and re-ranked results
        """
        # Dictionary to accumulate RRF scores by document ID
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process vector search results
        for rank, (doc, score) in enumerate(vector_results):
            doc_id = str(doc._id)
            rrf_score = self.vector_weight * (1.0 / (self.rrf_k + rank + 1))
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": doc,
                    "rrf_score": 0.0,
                    "vector_score": score,
                    "keyword_score": 0.0,
                    "sources": []
                }
            
            doc_scores[doc_id]["rrf_score"] += rrf_score
            doc_scores[doc_id]["sources"].append("vector")
        
        # Process keyword search results
        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = str(doc._id)
            rrf_score = self.keyword_weight * (1.0 / (self.rrf_k + rank + 1))
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": doc,
                    "rrf_score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": score,
                    "sources": []
                }
            
            doc_scores[doc_id]["rrf_score"] += rrf_score
            doc_scores[doc_id]["keyword_score"] = score
            if "keyword" not in doc_scores[doc_id]["sources"]:
                doc_scores[doc_id]["sources"].append("keyword")
        
        # Sort by RRF score and create results
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        results = []
        for item in sorted_docs:
            source = "hybrid" if len(item["sources"]) > 1 else item["sources"][0]
            results.append(RetrievalResult(
                document=item["document"],
                score=item["rrf_score"],
                source=source
            ))
        
        logger.info(f"RRF fusion produced {len(results)} results")
        return results
    
    async def retrieve_documents(
        self,
        db: AsyncSession,
        query: str,
        k: int = None
    ) -> List[KnowledgeBase]:
        """
        Retrieve documents only (without scores).
        
        Args:
            db: Database session
            query: Search query
            k: Number of results
            
        Returns:
            List of KnowledgeBase documents
        """
        results = await self.retrieve(db, query, k)
        return [r.document for r in results]


# Singleton instance with default weights
hybrid_retriever = HybridRetriever()
