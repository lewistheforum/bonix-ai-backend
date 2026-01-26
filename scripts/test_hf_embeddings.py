
import asyncio
import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.rag.embeddings_service import EmbeddingsService
from app.config import settings

async def main():
    print("Testing Hugging Face Embeddings...")
    
    # Override settings for testing
    settings.EMBEDDING_PROVIDER = "huggingface"
    settings.HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialize service
    service = EmbeddingsService()
    
    # Test text embedding
    text = "This is a test sentence."
    print(f"Embedding text: '{text}'")
    vector = await service.embed_text(text)
    
    print(f"Vector length: {len(vector)}")
    print(f"First 5 dimensions: {vector[:5]}")
    
    # Verify dimension
    expected_dim = 384
    if len(vector) == expected_dim:
        print("SUCCESS: Vector dimension matches expected 384 for all-MiniLM-L6-v2")
    else:
        print(f"FAILURE: Vector dimension {len(vector)} does not match expected {expected_dim}")

if __name__ == "__main__":
    asyncio.run(main())
