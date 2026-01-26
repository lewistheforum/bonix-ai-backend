"""
RAG Services Package

Provides RAG (Retrieval-Augmented Generation) functionality including:
- Embeddings service (OpenAI text-embedding-3-small)
- Vector store service (PostgreSQL + pgvector)
- Keyword search service (PostgreSQL full-text search)
- Hybrid retriever (combining vector and keyword search)
- Conversation memory service
- Booking tool (StructuredTool for appointments)
- RAG chain (main chatbot logic)
- Knowledge base service (document ingestion)
"""
from app.services.rag.embeddings_service import EmbeddingsService
from app.services.rag.vector_store_service import VectorStoreService
from app.services.rag.keyword_search_service import KeywordSearchService
from app.services.rag.hybrid_retriever import HybridRetriever
from app.services.rag.conversation_memory_service import ConversationMemoryService
from app.services.rag.booking_tool import booking_tool, create_booking
from app.services.rag.rag_chain import RAGChatbot
from app.services.rag.knowledge_base_service import KnowledgeBaseService

__all__ = [
    "EmbeddingsService",
    "VectorStoreService",
    "KeywordSearchService",
    "HybridRetriever",
    "ConversationMemoryService",
    "booking_tool",
    "create_booking",
    "RAGChatbot",
    "KnowledgeBaseService",
]
