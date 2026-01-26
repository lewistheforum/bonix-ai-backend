import asyncio
import os
import sys
from sqlalchemy import text

sys.path.append(os.getcwd())

from app.database import AsyncSessionLocal

async def check_search_vector():
    async with AsyncSessionLocal() as db:
        print("Checking search_vector column...")
        try:
            # Check if any row has non-null search_vector
            result = await db.execute(text("SELECT count(*) FROM knowledge_base WHERE search_vector IS NOT NULL"))
            count_populated = result.scalar()
            
            result_total = await db.execute(text("SELECT count(*) FROM knowledge_base"))
            count_total = result_total.scalar()
            
            print(f"Total documents: {count_total}")
            print(f"Documents with populated search_vector: {count_populated}")
            
            if count_populated > 0:
                # Show sample
                sample = await db.execute(text("SELECT content, search_vector FROM knowledge_base WHERE search_vector IS NOT NULL LIMIT 1"))
                row = sample.fetchone()
                print(f"Sample search_vector: {row[1]}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_search_vector())
