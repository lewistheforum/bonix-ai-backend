
import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

# Add app directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import engine, Base, init_db
from app.utils.logger import logger

# Import models to ensure they are registered
from app.models.ai_conversation import AIConversation
from app.models.ai_message import AIMessage
from app.models.knowledge_base import KnowledgeBase

async def reset_rag_tables():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force reset without confirmation")
    args = parser.parse_args()

    print("WARNING: This will delete all data in knowledge_base, ai_messages, and ai_conversations tables.")
    print("These tables will be dropped and recreated with the new schema (1536 dimensions).")
    
    if not args.force:
        confirm = input("Are you sure you want to continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return

    print("Dropping RAG tables...")
    async with engine.begin() as conn:
        # Drop in order of dependencies (messages depends on conversations)
        await conn.execute(text("DROP TABLE IF EXISTS ai_messages CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS ai_conversations CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS knowledge_base CASCADE"))
        print("Tables dropped.")

    print("Recreating tables...")
    await init_db()
    print("Tables initialized successfully.")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(reset_rag_tables())
