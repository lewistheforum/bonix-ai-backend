
import asyncio
import sys
import os
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.database import get_db, init_db
from app.services.rag.knowledge_base_service import knowledge_base_service
from app.utils.logger import logger

async def run_ingestion():
    """Run full knowledge base ingestion."""
    print("Initializing database...")
    await init_db()
    
    print("Starting ingestion...")
    async for db in get_db():
        try:
            # Clear existing data first to ensure clean state
            print("Clearing existing knowledge base...")
            await knowledge_base_service.clear_knowledge_base(db)
            
            print("1. Ingesting Clinic Services...")
            services = await knowledge_base_service.ingest_clinic_services(db)
            print(f"   -> {services} services ingested")
            
            print("2. Ingesting Doctor Profiles...")
            doctors = await knowledge_base_service.ingest_doctor_profiles(db)
            print(f"   -> {doctors} doctors ingested")
            
            print("3. Ingesting Clinic Info...")
            clinics = await knowledge_base_service.ingest_clinic_info(db)
            print(f"   -> {clinics} clinics ingested")
            
            print("4. Ingesting Staff Info...")
            staff = await knowledge_base_service.ingest_staff_info(db)
            print(f"   -> {staff} staff ingested")
            
            print("5. Ingesting Blogs...")
            blogs = await knowledge_base_service.ingest_blogs(db)
            print(f"   -> {blogs} blogs ingested")
            
            print("6. Ingesting Feedbacks...")
            feedbacks = await knowledge_base_service.ingest_feedbacks(db)
            print(f"   -> {feedbacks} feedbacks ingested")
            
            print("7. Ingesting User Info...")
            users = await knowledge_base_service.ingest_user_info(db)
            print(f"   -> {users} users ingested")
            
            print("8. Ingesting Doctor Schedules...")
            schedules = await knowledge_base_service.ingest_doctor_schedules(db)
            print(f"   -> {schedules} schedules ingested")
            
            print("9. Ingesting Clinic Working Hours...")
            hours = await knowledge_base_service.ingest_clinic_working_hours(db)
            print(f"   -> {hours} working hour records ingested")
            
            await db.commit()
            print("\n✅ Ingestion complete!")
            
        except Exception as e:
            print(f"\n❌ Error during ingestion: {e}")
            await db.rollback()
        finally:
            break

if __name__ == "__main__":
    asyncio.run(run_ingestion())
