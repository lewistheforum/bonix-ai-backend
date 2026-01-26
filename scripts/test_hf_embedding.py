import asyncio
import os
import sys

sys.path.append(os.getcwd())

# Force HuggingFace provider
os.environ["EMBEDDING_PROVIDER"] = "huggingface"

from app.config import settings
settings.EMBEDDING_PROVIDER = "huggingface"

from app.services.rag.embeddings_service import embeddings_service

async def test_embedding():
    print(f"Provider: {embeddings_service.provider}")
    print(f"Model: {embeddings_service.model}")
    print(f"Dimension: {embeddings_service.get_embedding_dimension()}")
    
    try:
        text = "Hello world"
        vector = await embeddings_service.embed_text(text)
        print(f"Generated vector length: {len(vector)}")
        if len(vector) == 384:
            print("SUCCESS: Vector dimension is correct.")
        else:
            print(f"FAILURE: Expected 384, got {len(vector)}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_embedding())
