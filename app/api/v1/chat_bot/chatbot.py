"""
Controller/Router for Chatbot API
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.dto.chat_bot.chatbot_dto import (
    ChatbotRequest,
    ChatbotResponse,
    ConversationHistoryResponse
)
from app.services.chat_bot.chatbot_service import chatbot_service
from app.common.api_response import ApiResponse
from app.common.message.status_code import StatusCode
from app.common.message.success_message import SuccessMessage
from app.common.message.error_message import ErrorMessage

router = APIRouter()


@router.post("/chat", response_model=ApiResponse[ChatbotResponse])
async def chat(request: ChatbotRequest):
    """
    Send a message to the chatbot and get a response
    
    - **message**: User message
    - **conversation_id**: Optional conversation ID for context
    - **patient_id**: Optional patient ID
    - **context**: Optional additional context
    """
    try:
        result = await chatbot_service.chat(request)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=result)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/conversation/{conversation_id}", response_model=ApiResponse[ConversationHistoryResponse])
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history by conversation ID
    
    - **conversation_id**: Conversation ID to retrieve
    """
    try:
        result = await chatbot_service.get_conversation_history(conversation_id)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=result)
    except ValueError as e:
        raise HTTPException(status_code=StatusCode.NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.delete("/conversation/{conversation_id}", response_model=ApiResponse[Dict[str, Any]])
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    
    - **conversation_id**: Conversation ID to delete
    """
    try:
        result = await chatbot_service.delete_conversation(conversation_id)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=result)
    except ValueError as e:
        raise HTTPException(status_code=StatusCode.NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/health", response_model=ApiResponse[Dict[str, str]])
async def health_check():
    """
    Health check endpoint for chatbot service
    """
    data = {"status": "healthy", "service": "chatbot"}
    return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)

