import asyncio
import os
import sys
from sqlalchemy import text

sys.path.append(os.getcwd())

from app.database import AsyncSessionLocal

async def drop_kb_table():
    async with AsyncSessionLocal() as db:
        print("Dropping knowledge_base table to handle schema change...")
        try:
            await db.execute(text("DROP TABLE IF EXISTS knowledge_base CASCADE"))
            await db.commit()
            print("Table dropped successfully.")
        except Exception as e:
            print(f"Error dropping table: {e}")
            await db.rollback()

if __name__ == "__main__":
    asyncio.run(drop_kb_table())
