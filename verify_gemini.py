
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from app.config import settings
from app.services.rag.embeddings_service import embeddings_service
from app.services.rag.rag_chain import rag_chatbot

async def verify_embeddings():
    print("\n--- Verifying Embeddings ---")
    print(f"Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"Model: {embeddings_service.model}")
    
    try:
        text = "Hello, world!"
        embedding = await embeddings_service.embed_text(text)
        print(f"Successfully generated embedding. Dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return None

async def verify_chat():
    print("\n--- Verifying Chat ---")
    print(f"Model: {rag_chatbot.model}")
    
    try:
        # Check LLM initialization
        llm = rag_chatbot.llm
        print(f"LLM Initialized: {type(llm)}")
        
        # Simple invocation
        response = await llm.ainvoke("Say hello!")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"Failed to chat: {e}")

async def verify_retrieval():
    print("\n--- Verifying Retrieval (RAG) ---")
    from app.services.rag.vector_store_service import vector_store_service
    from app.database import AsyncSessionLocal
    
    try:
        async with AsyncSessionLocal() as db:
            results = await vector_store_service.similarity_search(db, "test query", k=1)
            print(f"Similarity search successful. Found {len(results)} results.")
    except Exception as e:
        print(f"Failed to retrieve: {e}")

async def main():
    if not settings.GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Tests will likely fail if using Gemini.")
    
    await verify_embeddings()
    await verify_chat()
    await verify_retrieval()

if __name__ == "__main__":
    asyncio.run(main())
