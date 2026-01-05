"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.recommendation import recommendation_clinic
from app.api.v1.chat_bot import chatbot
from app.api.v1.bad_word import bad_word_detection
from app.api.v1.label_feedback import label_feedback
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
        Verify database connectivity when the application starts.
        Logs a clear message on success or failure.
        """
        # Log whether DB credentials were loaded; mask password for safety
        # masked_pwd = (
        #     f"{settings.DB_PASSWORD[:2]}***" if settings.DB_PASSWORD else "<empty>"
        # )
       
        is_connected = await check_db_connection()
        if is_connected:
            logger.info("Database connected successfully at startup")
        else:
            logger.error("Database connection failed at startup")
    
    return app


app = create_application()

