"""Main FastAPI application"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.recommendation import recommendation_clinic
from app.api.v1.bad_word import bad_word_detection
from app.api.v1.label_feedback import label_feedback
from app.api.v1.fracture_detection import fracture_detection
from app.api.v1.rag.rag_router import router as rag_router
from app.config import settings
from app.database import check_db_connection
from app.utils.logger import logger


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(
        recommendation_clinic.router,
        prefix=f"{settings.API_V1_PREFIX}/recommendation-clinic",
        tags=["Recommendation Clinic"]
    )
    
    app.include_router(
        bad_word_detection.router,
        prefix=f"{settings.API_V1_PREFIX}/bad-word-detection",
        tags=["Bad Word Detection"]
    )
    
    app.include_router(
        label_feedback.router,
        prefix=f"{settings.API_V1_PREFIX}/feedback",
        tags=["Label Feedback"]
    )
    
    app.include_router(
        fracture_detection.router,
        prefix=f"{settings.API_V1_PREFIX}/fracture-detection",
        tags=["Fracture Detection"]
    )
    
    # RAG Chatbot router
    app.include_router(
        rag_router,
        prefix=f"{settings.API_V1_PREFIX}",
        tags=["RAG Chatbot"]
    )
    
    from app.common.api_response import ApiResponse
    from typing import Dict, Any

    @app.get("/", response_model=ApiResponse[Dict[str, Any]])
    async def root():
        data = {
            "message": "Welcome to Medicare AI Backend",
            "version": settings.APP_VERSION,
            "docs": "/docs"
        }
        return ApiResponse(statusCode=200, message="Success", data=data)
    
    @app.get("/health", response_model=ApiResponse[Dict[str, str]])
    async def health_check():
        data = {"status": "healthy"}
        return ApiResponse(statusCode=200, message="Success", data=data)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "statusCode": exc.status_code,
                "message": str(exc.detail),
                "data": None
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "statusCode": 422,
                "message": "Validation Error",
                "data": exc.errors()
            }
        )

    @app.on_event("startup")
    async def on_startup():
        """
        Verify database connectivity when the application starts.
        Note: Tables (ai_conversations, ai_messages, knowledge_base) are created
        and managed by the primary backend server, not this AI backend.
        """
        # Ensure models are loaded so they are registered in Base.metadata
        from app.models.ai_conversation import AIConversation
        from app.models.ai_message import AIMessage
        from app.models.knowledge_base import KnowledgeBase
        from app.models.knowledge_base_medicines import KnowledgeBaseMedicines

        # Note: We do NOT call init_db() here because the tables are already
        # created and managed by the primary backend server

        is_connected = await check_db_connection()
        if is_connected:
            logger.info("Database connected successfully at startup")
        else:
            logger.error("Database connection failed at startup")
    
    return app


app = create_application()

