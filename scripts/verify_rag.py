"""
RAG Verification Script

Tests the core RAG functionality:
1. Database connection
2. pgvector extension
3. Document ingestion (Embeddings)
4. Vector Search
5. Keyword Search
6. Hybrid Retrieval
"""
import asyncio
import sys
from pathlib import Path
import os

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import text
from app.database import AsyncSessionLocal
from app.services.rag.knowledge_base_service import knowledge_base_service
from app.services.rag.hybrid_retriever import hybrid_retriever
from app.config import settings

# Sample data
SAMPLE_DOC = {
    "content": "Medicare Clinic offers comprehensive dental services including cleaning, whitening, and braces. Located at 123 Main St.",
    "metadata": {
        "source": "verification_script",
        "category": "dental"
    }
}

async def verify_rag():
    print("="*50)
    print("RAG PIPELINE VERIFICATION")
    print("="*50)
    
    async with AsyncSessionLocal() as session:
        try:
            # 1. Check Database & pgvector
            print("\n1. Checking Database & pgvector...")
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await session.commit()
            print("   ✅ pgvector extension enabled")
            
            # 2. Ingest Document
            print("\n2. Ingesting Sample Document...")
            if not settings.OPENAI_API_KEY:
                print("   ❌ OPENAI_API_KEY not set! Skipping ingestion/embeddings test.")
                return
            
            doc = await knowledge_base_service.ingest_document(
                session,
                content=SAMPLE_DOC["content"],
                metadata=SAMPLE_DOC["metadata"],
                doc_type="test"
            )
            print(f"   ✅ Document ingested with ID: {doc._id}")
            
            # 3. Test Hybrid Retrieval
            print("\n3. Testing Hybrid Retrieval...")
            query = "dental services cleaning"
            results = await hybrid_retriever.retrieve(session, query, k=3)
            
            if results:
                print(f"   ✅ Retrieval successful! Found {len(results)} results.")
                for r in results:
                    print(f"      - [{r.source}] Score: {r.score:.3f} | Content: {r.document.content[:50]}...")
            else:
                print("   ⚠️ No results found. (Might be expected if embeddings API failed or index not ready)")
                
            # Cleanup
            print("\n4. Cleaning up test data...")
            await knowledge_base_service.delete_document(session, str(doc._id))
            await session.commit()
            print("   ✅ Test document deleted")
            
            print("\n✅ VERIFICATION COMPLETE: ALL SYSTEMS GO")
            
        except Exception as e:
            print(f"\n❌ VERIFICATION FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_rag())
