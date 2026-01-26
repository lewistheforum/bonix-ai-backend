import asyncio
import os
import sys
from sqlalchemy import text

# Add project root to path
sys.path.append(os.getcwd())

from app.database import AsyncSessionLocal

async def check_types():
    async with AsyncSessionLocal() as db:
        print("Checking distinct metadata types in KnowledgeBase...")
        
        # Query to get distinct values for 'type' and 'doc_type' from metadata
        # Note: syntax depends on JSONB usage, assuming metadata is JSONB
        try:
            # Check 'type'
            result_type = await db.execute(text("SELECT DISTINCT metadata->>'type' FROM knowledge_base"))
            types = [row[0] for row in result_type.fetchall() if row[0] is not None]
            print(f"\nFound 'type' values: {types}")
            
            # Check 'doc_type'
            result_doc_type = await db.execute(text("SELECT DISTINCT metadata->>'doc_type' FROM knowledge_base"))
            doc_types = [row[0] for row in result_doc_type.fetchall() if row[0] is not None]
            print(f"Found 'doc_type' values: {doc_types}")
            
        except Exception as e:
            print(f"Error querying: {e}")

if __name__ == "__main__":
    asyncio.run(check_types())
