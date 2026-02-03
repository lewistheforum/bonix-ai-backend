"""
Embeddings Service

Provides OpenAI embeddings for text vectorization using text-embedding-3-small model.
"""
from typing import List, Optional, Union
import asyncio


from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.utils.logger import logger


class EmbeddingsService:
    """
    Embeddings Service
    
    Wraps LangChain's OpenAIEmbeddings or HuggingFaceEmbeddings for generating vector embeddings.
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the embeddings service.
        
        Args:
            provider: Optional provider name override. Defaults to settings.EMBEDDING_PROVIDER
            model: Optional model name override. Defaults to provider-specific setting path
        """
        self.provider = provider or settings.EMBEDDING_PROVIDER
        
        if self.provider == "openai":
            self.model = model or settings.OPENAI_EMBEDDING_MODEL
        elif self.provider == "gemini":
            self.model = model or settings.GEMINI_EMBEDDING_MODEL
            
        self._embeddings = None
        
    @property
    def embeddings(self) -> Union[object, GoogleGenerativeAIEmbeddings]:
        """Lazy initialization of embeddings client"""
        if self._embeddings is None:
            if self.provider == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is not set in environment variables")
                # Use the OpenAI SDK directly to avoid compatibility issues
                # where upstream wrappers may pass unsupported kwargs (e.g. `proxies`).
                from openai import OpenAI
                
                # Use the OpenAI SDK directly to avoid compatibility issues
                # where upstream wrappers may pass unsupported kwargs (e.g. `proxies`).
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

                class SimpleOpenAIEmbeddings:
                    def __init__(self, client, model: str):
                        self.client = client
                        self.model = model

                    def embed_query(self, text: str):
                        # API v1.x usage
                        resp = self.client.embeddings.create(model=self.model, input=text)
                        # Response is an object, not a dict
                        return resp.data[0].embedding

                    def embed_documents(self, texts: list):
                        # API v1.x usage
                        resp = self.client.embeddings.create(model=self.model, input=texts)
                        # Response is an object, not a dict
                        return [d.embedding for d in resp.data]

                self._embeddings = SimpleOpenAIEmbeddings(self.client, self.model)
                logger.info(f"Initialized OpenAI embeddings (direct SDK v1) with model: {self.model}")
                
            elif self.provider == "gemini":
                if not settings.GOOGLE_API_KEY:
                    raise ValueError("GOOGLE_API_KEY is not set in environment variables")
                
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.model,
                    google_api_key=settings.GOOGLE_API_KEY,
                )
                logger.info(f"Initialized Gemini embeddings with model: {self.model}")

        return self._embeddings
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # LangChain's embed_query is synchronous, run in executor
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.embeddings.embed_query,
                text
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # LangChain's embed_documents is synchronous, run in executor
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.embeddings.embed_documents,
                texts
            )
            logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.provider == "openai":
            if self.model == "text-embedding-3-small":
                return 1536
            elif self.model == "text-embedding-3-large":
                return 3072
            elif self.model == "text-embedding-ada-002":
                return 1536
            # Default fallback for OpenAI
            return 1536
            
        elif self.provider == "gemini":
            # models/text-embedding-004 is 768 dimensions
            if "text-embedding-004" in self.model:
                return 768
            return 768 # Default for Gemini embeddings currently

        return 1536


# Singleton instance
embeddings_service = EmbeddingsService()
