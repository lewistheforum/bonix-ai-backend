"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.recommendation import recommendation_clinic
from app.api.v1.chat_bot import chatbot
from app.api.v1.bad_word import bad_word_detection
from app.api.v1.label_feedback import label_feedback
from app.api.rag_router import router as rag_router
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
        chatbot.router,
        prefix=f"{settings.API_V1_PREFIX}/chatbot",
        tags=["Chatbot"]
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
    
    # RAG Chatbot router
    app.include_router(
        rag_router,
        prefix=f"{settings.API_V1_PREFIX}",
        tags=["RAG Chatbot"]
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Medicare AI Backend",
            "version": settings.APP_VERSION,
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.on_event("startup")
    async def on_startup():
        """
        Verify database connectivity and initialize tables when the application starts.
        """
        # Ensure models are loaded so they are registered in Base.metadata
        from app.models.ai_conversation import AIConversation
        from app.models.ai_message import AIMessage
        from app.models.knowledge_base import KnowledgeBase
        from app.database import init_db

        # Initialize database tables
        await init_db()
        logger.info("Database tables initialized")

        is_connected = await check_db_connection()
        if is_connected:
            logger.info("Database connected successfully at startup")
        else:
            logger.error("Database connection failed at startup")
    
    return app


app = create_application()

