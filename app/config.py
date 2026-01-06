"""
Configuration settings for the application
"""
from typing import Optional

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings"""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), case_sensitive=True, extra="ignore"
    )
    
    # App settings
    APP_NAME: str = "Medicare AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    
    # PostgreSQL Database settings
    DB_HOST: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    DB_PORT: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    DB_USER: str = Field(default="postgres", validation_alias="POSTGRES_USERNAME")
    DB_PASSWORD: str = Field(default="", validation_alias="POSTGRES_PASSWORD")
    DB_NAME: str = Field(default="medicare", validation_alias="POSTGRES_DATABASE")
    DB_POOL_SIZE: int = Field(default=5, validation_alias="POSTGRES_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=10, validation_alias="POSTGRES_MAX_OVERFLOW")
    DB_ECHO: bool = Field(default=False, validation_alias="POSTGRES_ECHO")
    # DB_URI: Optional[str] = Field(default=None, validation_alias="POSTGRES_URI")
    
    @staticmethod
    def _normalize_uri(uri: str, *, async_mode: bool) -> str:
        """
        Normalize postgres URI to include SQLAlchemy driver.
        - Supports URIs like postgres://... or postgresql://...
        - Appends +psycopg_async (async) or +psycopg (sync) when missing.
        """
        if not uri:
            return uri

        driver = "psycopg_async" if async_mode else "psycopg"

        if uri.startswith("postgres://"):
            return uri.replace("postgres://", f"postgresql+{driver}://", 1)

        if uri.startswith("postgresql://") and "+psycopg" not in uri:
            return uri.replace("postgresql://", f"postgresql+{driver}://", 1)

        return uri

    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL database URL"""
        return (
            f"postgresql+psycopg_async://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Construct synchronous PostgreSQL database URL"""
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    # AI/ML settings
    AI_MODEL_PATH: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = Field(default="", validation_alias="HF_TOKEN")
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]


settings = Settings()

