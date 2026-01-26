import asyncio
import os
import sys

sys.path.append(os.getcwd())

from app.database import AsyncSessionLocal
from app.services.rag.schedule_tool import find_clinic_schedule, find_doctor_schedule, set_schedule_db_session

async def test_tools():
    async with AsyncSessionLocal() as db:
        set_schedule_db_session(db)
        
        print("--- Testing Find Clinic Schedule ---")
        # Use a likely existing clinic/date from valid data or just a test one (might return 'not found' but no error)
        # From previous logs, we saw 'Bonix' clinics. Date 2026-02-04 was in the vector search result.
        res_clinic = await find_clinic_schedule("Bonix", "2026-02-04")
        print(res_clinic)
        
        print("\n--- Testing Find Doctor Schedule ---")
        # From previous logs: 'Dang Van H'
        res_doctor = await find_doctor_schedule("Dang", "2026-02-04")
        print(res_doctor)

if __name__ == "__main__":
    asyncio.run(test_tools())
