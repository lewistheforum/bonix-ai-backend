"""
RAG API Router

FastAPI router for RAG chatbot endpoints including:
- Chat endpoint with hybrid retrieval
- Knowledge base management
- Conversation history
"""
from datetime import datetime
from typing import Optional, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dto.rag import (
    RAGChatRequest,
    RAGChatResponse,
    KnowledgeBaseIngestRequest,
    KnowledgeBaseIngestResponse,
    KnowledgeBaseSearchRequest,
    KnowledgeBaseSearchResponse,
    KnowledgeBaseSearchResult,
    SyncKnowledgeBaseRequest,
    SyncKnowledgeBaseResponse,
    SyncMedicineKnowledgeBaseRequest,
    SyncMedicineKnowledgeBaseResponse,
    ConversationHistoryRequest,
    ConversationHistoryResponse,
    MessageItem,
    ConversationChatRequest,
    ConversationChatResponse,
)
from app.services.rag.rag_chain import rag_chatbot
from app.services.rag.hybrid_retriever import hybrid_retriever
from app.services.rag.vector_store_service import vector_store_service
from app.services.rag.keyword_search_service import keyword_search_service
from app.services.rag.knowledge_base_service import knowledge_base_service
from app.services.rag.medicine_knowledge_base_service import medicine_knowledge_base_service
from app.services.rag.conversation_memory_service import conversation_memory_service
from app.services.rag.conversation_chat_service import conversation_chat_service
from app.utils.logger import logger
from app.common.api_response import ApiResponse
from app.common.message.status_code import StatusCode
from app.common.message.success_message import SuccessMessage
from app.common.message.error_message import ErrorMessage

router = APIRouter(prefix="/rag", tags=["RAG Chatbot"])


@router.post("/chat", response_model=ApiResponse[RAGChatResponse])
async def chat_rag(
    request: RAGChatRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[RAGChatResponse]:
    """
    Chat with RAG (Retrieval Augmented Generation).
    
    Args:
        request: Chat request containing query and history
        db: Database session
        
    Returns:
        Chat response with answer and source documents
    """
    try:
        result = await rag_chatbot.chat(
            db=db,
            query=request.message,
            conversation_id=request.conversation_id,
            user_id=request.user_id
        )
        
        data = RAGChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            context_used=result["context_used"],
            sources=result["sources"],
            timestamp=datetime.utcnow()
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"RAG Chat error: {e}")
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/conversation/{conversation_id}/{user_id}", response_model=ApiResponse[ConversationChatResponse])
async def conversation_chat(
    conversation_id: str,
    user_id: str,
    request: ConversationChatRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[ConversationChatResponse]:
    """
    Chat within a specific conversation with conversation-scoped RAG.
    
    This endpoint combines:
    - Semantic search over past conversation messages (embedded on save)
    - Knowledge base hybrid retrieval (clinic, doctor, service data)
    - Windowed chat history for recent context
    
    Each user message and assistant response is automatically embedded,
    making previous discussion topics searchable in future turns.
    
    Args:
        conversation_id: UUID of the conversation
        user_id: UUID of the user
        request: Chat request containing the message
        db: Database session
        
    Returns:
        Chat response with answer and context metadata
    """
    try:
        result = await conversation_chat_service.chat(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            message=request.message
        )
        
        data = ConversationChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            context_used=result["context_used"],
            conversation_context_used=result["conversation_context_used"],
            sources=result["sources"],
            timestamp=result["timestamp"]
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"Conversation chat error: {e}")
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/knowledge-base/ingest", response_model=ApiResponse[KnowledgeBaseIngestResponse])
async def ingest_documents(
    request: KnowledgeBaseIngestRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[KnowledgeBaseIngestResponse]:
    """
    Ingest documents into the knowledge base.
    
    Documents are vectorized using OpenAI embeddings and stored in PostgreSQL
    with pgvector for efficient similarity search.
    
    Args:
        request: List of documents with content and optional metadata
        db: Database session
        
    Returns:
        Ingestion result with count of documents added
    """
    try:
        entries = await knowledge_base_service.ingest_documents(
            db=db,
            documents=request.documents
        )
        await db.commit()
        
        data = KnowledgeBaseIngestResponse(
            success=True,
            documents_ingested=len(entries),
            message=f"Successfully ingested {len(entries)} documents"
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        await db.rollback()
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/knowledge-base/search", response_model=ApiResponse[KnowledgeBaseSearchResponse])
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    k: int = Query(default=5, ge=1, le=20),
    search_type: str = Query(default="hybrid", description="Search type: vector, keyword, or hybrid"),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[KnowledgeBaseSearchResponse]:
    """
    Search the knowledge base.
    
    Supports three search modes:
    - vector: Semantic similarity search using embeddings
    - keyword: PostgreSQL full-text search
    - hybrid: Combined vector + keyword with RRF fusion
    
    Args:
        query: Search query string
        k: Number of results to return
        search_type: Type of search to perform
        db: Database session
        
    Returns:
        Search results with scores and metadata
    """
    try:
        results = []
        
        if search_type == "vector":
            docs = await vector_store_service.similarity_search_with_score(db, query, k)
            results = [
                KnowledgeBaseSearchResult(
                    id=str(doc._id),
                    content=doc.content[:500],
                    score=score,
                    source="vector",
                    metadata=doc.meta_data or {}
                )
                for doc, score in docs
            ]
            
        elif search_type == "keyword":
            docs = await keyword_search_service.search_with_score(db, query, k)
            results = [
                KnowledgeBaseSearchResult(
                    id=str(doc._id),
                    content=doc.content[:500],
                    score=score,
                    source="keyword",
                    metadata=doc.meta_data or {}
                )
                for doc, score in docs
            ]
            
        else:  # hybrid
            retrieval_results = await hybrid_retriever.retrieve(db, query, k)
            results = [
                KnowledgeBaseSearchResult(
                    id=str(r.document._id),
                    content=r.document.content[:500],
                    score=r.score,
                    source=r.source,
                    metadata=r.document.meta_data or {}
                )
                for r in retrieval_results
            ]
        
        data = KnowledgeBaseSearchResponse(
            query=query,
            results=results,
            total=len(results),
            search_type=search_type
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/knowledge-base/sync", response_model=ApiResponse[SyncKnowledgeBaseResponse])
async def sync_knowledge_base(
    request: SyncKnowledgeBaseRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[SyncKnowledgeBaseResponse]:
    """
    Sync knowledge base with data from the main database.
    
    This endpoint pulls data from clinic_services, doctor_information,
    and clinic tables, vectorizes the content, and stores it in the
    knowledge base for RAG retrieval.
    
    Args:
        request: Sync options (which data types to sync)
        db: Database session
        
    Returns:
        Sync results with counts per data type
    """
    try:
        if request.clear_existing:
            await knowledge_base_service.clear_knowledge_base(db)
        
        clinic_services = 0
        doctor_profiles = 0
        clinic_info = 0
        staff_info = 0
        blogs = 0
        feedbacks = 0
        user_info = 0
        doctor_schedules = 0
        clinic_working_hours = 0
        
        if request.sync_clinic_services:
            clinic_services = await knowledge_base_service.ingest_clinic_services(db)
        
        if request.sync_doctor_profiles:
            doctor_profiles = await knowledge_base_service.ingest_doctor_profiles(db)
        
        if request.sync_clinic_info:
            clinic_info = await knowledge_base_service.ingest_clinic_info(db)
            
        if request.sync_staff_info:
            staff_info = await knowledge_base_service.ingest_staff_info(db)
            
        if request.sync_blogs:
            blogs = await knowledge_base_service.ingest_blogs(db)
            
        if request.sync_feedbacks:
            feedbacks = await knowledge_base_service.ingest_feedbacks(db)
            
        if request.sync_user_info:
            user_info = await knowledge_base_service.ingest_user_info(db)
            
        if request.sync_doctor_schedules:
            doctor_schedules = await knowledge_base_service.ingest_doctor_schedules(db)
            
        if request.sync_clinic_working_hours:
            clinic_working_hours = await knowledge_base_service.ingest_clinic_working_hours(db)
        
        await db.commit()
        
        total = (clinic_services + doctor_profiles + clinic_info + staff_info + 
                 blogs + feedbacks + user_info + doctor_schedules + clinic_working_hours)
        
        data = SyncKnowledgeBaseResponse(
            success=True,
            clinic_services_synced=clinic_services,
            doctor_profiles_synced=doctor_profiles,
            clinic_info_synced=clinic_info,
            staff_info_synced=staff_info,
            blogs_synced=blogs,
            feedbacks_synced=feedbacks,
            user_info_synced=user_info,
            doctor_schedules_synced=doctor_schedules,
            clinic_working_hours_synced=clinic_working_hours,
            total_synced=total,
            message=f"Successfully synced {total} documents to knowledge base"
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"Sync error: {e}")
        await db.rollback()
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/knowledge-base/sync-medicines", response_model=ApiResponse[SyncMedicineKnowledgeBaseResponse])
async def sync_medicine_knowledge_base(
    request: SyncMedicineKnowledgeBaseRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[SyncMedicineKnowledgeBaseResponse]:
    """
    Sync medicine knowledge base from the medicines table.
    
    Groups medicines by therapeutic_class, generates embeddings for each
    group, and stores them in the knowledge_base_medicines table.
    
    Args:
        request: Sync options (clear_existing)
        db: Database session
        
    Returns:
        Sync results with counts
    """
    try:
        result = await medicine_knowledge_base_service.sync_medicines(
            db=db,
            clear_existing=request.clear_existing
        )
        
        await db.commit()
        
        data = SyncMedicineKnowledgeBaseResponse(
            success=True,
            therapeutic_classes_synced=result["therapeutic_classes_synced"],
            total_medicines_processed=result["total_medicines_processed"],
            message=(
                f"Successfully synced {result['therapeutic_classes_synced']} therapeutic classes "
                f"({result['total_medicines_processed']} medicines) to knowledge base"
            )
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"Medicine sync error: {e}")
        await db.rollback()
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/conversations/{conversation_id}/history", response_model=ApiResponse[ConversationHistoryResponse])
async def get_conversation_history(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[ConversationHistoryResponse]:
    """
    Get conversation history.
    
    Args:
        conversation_id: ID of the conversation
        limit: Maximum number of messages to return
        db: Database session
        
    Returns:
        List of messages in the conversation
    """
    try:
        from sqlalchemy import select
        from app.models.ai_message import AIMessage
        import uuid
        
        conv_uuid = uuid.UUID(conversation_id)
        result = await db.execute(
            select(AIMessage)
            .where(AIMessage.conversation_id == conv_uuid)
            .where(AIMessage.deleted_at.is_(None))
            .order_by(AIMessage.created_at)
            .limit(limit)
        )
        messages = result.scalars().all()
        
        message_items = [
            MessageItem(
                id=str(msg._id),
                role=msg.role,
                content=msg.content,
                timestamp=msg.created_at,
                metadata=msg.meta_data or {}
            )
            for msg in messages
        ]
        
        data = ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=message_items,
            total=len(message_items)
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.delete("/conversations/{conversation_id}", response_model=ApiResponse[Dict[str, str]])
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, str]]:
    """
    Delete a conversation and its messages.
    
    Args:
        conversation_id: ID of the conversation to delete
        db: Database session
        
    Returns:
        Deletion confirmation
    """
    try:
        success = await conversation_memory_service.clear_conversation(
            db, conversation_id
        )
        await db.commit()
        
        if success:
            data = {"message": f"Conversation {conversation_id} deleted successfully"}
            return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
        else:
            raise HTTPException(status_code=StatusCode.NOT_FOUND, detail=ErrorMessage.CONVERSATION_NOT_FOUND)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        await db.rollback()
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))
