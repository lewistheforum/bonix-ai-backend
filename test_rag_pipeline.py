import asyncio
import sys
import os

# Add current directory to path so we can import app
sys.path.append(os.getcwd())



from app.database import AsyncSessionLocal
from app.services.rag.rag_chain import rag_chatbot
from app.services.rag.hybrid_retriever import hybrid_retriever

async def main():
    print("=== RAG Pipeline Inspector ===")
    print("This script checks what the RAG system retrieves for a given query.")
    
    # Allow passing query as argument, or prompt for it
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("\nEnter your question: ")
        
    print(f"\nAnalyzing query: '{query}'\n")

    async with AsyncSessionLocal() as db:
        print("-" * 50)
        print("1. RETRIEVED CONTEXT (What the LLM sees)")
        print("-" * 50)
        
        # We access the internal _get_context to see exactly what is passed to the LLM
        # This includes the "Direct DB fetch" logic for clinics if applicable
        try:
            context = await rag_chatbot._get_context(db, query)
            print(context)
        except Exception as e:
            print(f"Error getting context: {e}")
        
        print("\n" + "-" * 50)
        print("2. RAW HYBRID RETRIEVAL RESULTS (Deep dive)")
        print("-" * 50)
        # useful to see if the hybrid retriever finds it, even if _get_context logic changes
        try:
            results = await hybrid_retriever.retrieve(db, query, k=5)
            if not results:
                print("No results from hybrid retriever.")
            for i, res in enumerate(results, 1):
                print(f"[{i}] Score: {res.score:.4f} | Source: {res.source}")
                doc_type = res.document.meta_data.get('type', 'unknown') if res.document.meta_data else 'unknown'
                print(f"    Type: {doc_type}")
                print(f"    Content: {res.document.content[:200]}...")
                print()
        except Exception as e:
            print(f"Error in raw retrieval: {e}")

        print("-" * 50)
        print("3. GENERATING RESPONSE...")
        print("-" * 50)
        try:
            # We use a dummy user_id and conversation_id (None lets the backend generate/handle them)
            response = await rag_chatbot.chat(db, query, conversation_id=None, user_id=None)
            print("\nASSISTANT RESPONSE:")
            print(response["response"])
        except Exception as e:
            print(f"Error generating response: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
