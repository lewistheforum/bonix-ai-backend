
import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

# Add app directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database import engine

async def check_tables():
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT exists (SELECT FROM information_schema.tables WHERE table_name = 'ai_conversations')"
        ))
        exists = result.scalar()
        if exists:
            print("SUCCESS: Table 'ai_conversations' exists.")
        else:
            print("FAILURE: Table 'ai_conversations' does not exist.")

if __name__ == "__main__":
    asyncio.run(check_tables())
