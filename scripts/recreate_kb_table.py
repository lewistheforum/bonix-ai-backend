import asyncio
import os
import sys

sys.path.append(os.getcwd())

from app.database import engine, Base
from app.models.knowledge_base import KnowledgeBase

async def recreate_table():
    print("Recreating knowledge_base table...")
    async with engine.begin() as conn:
        # Drop table if exists
        await conn.run_sync(lambda sync_conn: KnowledgeBase.__table__.drop(sync_conn, checkfirst=True))
        print("Table dropped.")
        
        # Create table
        await conn.run_sync(lambda sync_conn: KnowledgeBase.__table__.create(sync_conn))
        print("Table created with new schema.")

if __name__ == "__main__":
    asyncio.run(recreate_table())
