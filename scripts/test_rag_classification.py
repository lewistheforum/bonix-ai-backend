import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.services.rag.rag_chain import rag_chatbot

async def test_classification():
    test_queries = [
        ("How much does a dental implant cost?", "clinic_service"),
        ("I want to book an appointment with Dr. John.", "doctor_profile"),
        ("Where is the clinic located?", "clinic_info"),
        ("Can I get the number of the reception?", "staff_info"),
        ("Any articles on oral hygiene?", "blog"),
        ("Is Dr. Sarah available on Monday?", "doctor_schedule"),
        ("What do people say about this clinic?", "feedback"),
        ("Show me my profile details.", "user_info"),
        ("Hello", "general (no filter)")
    ]
    
    print("Testing Query Classification...\n")
    
    for query, expected in test_queries:
        print(f"Query: {query}")
        try:
            # We can access the private method for testing purposes
            categories = await rag_chatbot._classify_query(query)
            print(f"Detected: {categories}")
            print(f"Expected context: {expected}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_classification())
