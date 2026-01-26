
import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import init_db, engine
from app.utils.logger import logger

# Import models to ensure they are registered
from app.models.ai_conversation import AIConversation
from app.models.ai_message import AIMessage
from app.models.knowledge_base import KnowledgeBase

async def main():
    print("Initializing database...")
    try:
        await init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
