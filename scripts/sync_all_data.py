import asyncio
import logging
from app.database import AsyncSessionLocal
from app.services.rag.knowledge_base_service import knowledge_base_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.config import settings
settings.EMBEDDING_PROVIDER = "huggingface"

async def sync_all_data():
    """
    Sync all data from the main database to the knowledge base.
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.info("Starting full knowledge base sync...")
            
            # 1. Clinic Services
            logger.info("Syncing clinic services...")
            count_services = await knowledge_base_service.ingest_clinic_services(session)
            print(f"Synced {count_services} clinic services.")
            
            # 2. Doctor Profiles
            logger.info("Syncing doctor profiles...")
            count_doctors = await knowledge_base_service.ingest_doctor_profiles(session)
            print(f"Synced {count_doctors} doctor profiles.")
            
            # 3. Clinic Info
            logger.info("Syncing clinic info...")
            count_clinics = await knowledge_base_service.ingest_clinic_info(session)
            print(f"Synced {count_clinics} clinic info documents.")
            
            # 4. Staff Info
            logger.info("Syncing staff info...")
            count_staff = await knowledge_base_service.ingest_staff_info(session)
            print(f"Synced {count_staff} staff profiles.")
            
            # 5. Blogs
            logger.info("Syncing blogs...")
            count_blogs = await knowledge_base_service.ingest_blogs(session)
            print(f"Synced {count_blogs} blogs.")
            
            # 6. Feedbacks
            logger.info("Syncing feedbacks...")
            count_feedbacks = await knowledge_base_service.ingest_feedbacks(session)
            print(f"Synced {count_feedbacks} feedbacks.")
            
            # 7. User Info
            logger.info("Syncing user info...")
            count_users = await knowledge_base_service.ingest_user_info(session)
            print(f"Synced {count_users} user profiles.")
            
            # 8. Doctor Schedules
            logger.info("Syncing doctor schedules...")
            count_schedules = await knowledge_base_service.ingest_doctor_schedules(session)
            print(f"Synced {count_schedules} doctor schedule documents.")
            
            # 9. Clinic Working Hours
            logger.info("Syncing clinic working hours...")
            count_hours = await knowledge_base_service.ingest_clinic_working_hours(session)
            print(f"Synced {count_hours} working hours documents.")
            
            # Commit all changes
            await session.commit()
            
            total_docs = (
                count_services + count_doctors + count_clinics + 
                count_staff + count_blogs + count_feedbacks + 
                count_users + count_schedules + count_hours
            )
            
            logger.info(f"Full sync completed successfully. Total documents: {total_docs}")
            
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            await session.rollback()
            raise

import sys
from sqlalchemy import text

async def check_data_status():
    """
    Check and print the status of data in knowledge base.
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.info("Checking knowledge base data status...")
            
            # Query to count documents by doc_type
            # knowledge_base.metadata is a separate column from metadata in sqlalchemy
            # but in our model it is defined as meta_data = Column("metadata", JSONB...)
            # We access it via SQL
            
            # DEBUG: Check raw metadata keys
            debug_stmt = text("SELECT metadata FROM knowledge_base LIMIT 1")
            debug_res = await session.execute(debug_stmt)
            debug_row = debug_res.fetchone()
            if debug_row:
                logger.info(f"\nSAMPLE METADATA: {debug_row[0]}")

            stmt = text("""
                SELECT 
                    COALESCE(metadata->>'type', metadata->>'type') as type, 
                    count(*) as count 
                FROM knowledge_base 
                WHERE deleted_at IS NULL 
                GROUP BY COALESCE(metadata->>'type', metadata->>'type')
                ORDER BY count DESC
            """)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            print("\n" + "="*50)
            print("KNOWLEDGE BASE DATA STATUS")
            print("="*50)
            print(f"{'DOCUMENT TYPE':<30} | {'COUNT':<10}")
            print("-" * 43)
            
            total_docs = 0
            for row in rows:
                doc_type = row[0] or "unknown"
                count = row[1]
                print(f"{doc_type:<30} | {count:<10}")
                total_docs += count
                
            print("-" * 43)
            print(f"{'TOTAL DOCUMENTS':<30} | {total_docs:<10}")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Error checking status: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        asyncio.run(check_data_status())
    else:
        asyncio.run(sync_all_data())
        asyncio.run(check_data_status())
