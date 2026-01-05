"""
Controller/Router for Chatbot API
"""
from fastapi import APIRouter, HTTPException
from app.dto.chat_bot.chatbot_dto import (
    ChatbotRequest,
    ChatbotResponse,
    ConversationHistoryResponse
)
from app.services.chat_bot.chatbot_service import chatbot_service

router = APIRouter()


@router.post("/chat", response_model=ChatbotResponse)
async def chat(request: ChatbotRequest):
    """
    Send a message to the chatbot and get a response
    
    - **message**: User message
    - **conversation_id**: Optional conversation ID for context
    - **patient_id**: Optional patient ID
    - **context**: Optional additional context
    """
    try:
        return await chatbot_service.chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history by conversation ID
    
    - **conversation_id**: Conversation ID to retrieve
    """
    try:
        return await chatbot_service.get_conversation_history(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    
    - **conversation_id**: Conversation ID to delete
    """
    try:
        return await chatbot_service.delete_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for chatbot service
    """
    return {"status": "healthy", "service": "chatbot"}

