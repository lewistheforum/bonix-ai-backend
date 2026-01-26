"""
Schedule Tool

LangChain StructuredTools for querying clinic and doctor schedules.
Executes raw SQL queries to fetch real-time schedule information.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.logger import logger

# Global database session holder
_db_session: Optional[AsyncSession] = None

def set_schedule_db_session(db: AsyncSession):
    """Set the database session for the schedule tools."""
    global _db_session
    _db_session = db

# --- Clinic Schedule Tool ---

class ClinicScheduleInput(BaseModel):
    """Input schema for finding clinic schedules."""
    clinic_name: str = Field(..., description="Name of the clinic to find schedule for (e.g. 'Bonix', 'Trauma Orthopedics')")
    work_date: str = Field(..., description="Date to check schedule for in YYYY-MM-DD format")

async def find_clinic_schedule(clinic_name: str, work_date: str) -> str:
    """
    Find the working schedule for a specific clinic on a given date.
    Returns a formatted string of the schedule properly.
    """
    global _db_session
    if not _db_session:
        return "Database session not available."

    try:
        # Validate date
        try:
            datetime.strptime(work_date, "%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

        query = text("""
            WITH RankedSchedule AS (
                SELECT 
                    es.clinic_id,
                    cai.clinic_name as main_clinic_name,
                    cmi.clinic_branch_name,
                    cmi.full_name,
                    adr.address,
                    adr.ward_name,
                    adr.district_name,
                    adr.province_name,
                    di.full_name as doctor_name,
                    a0.role,
                    di.gender,
                    csh.start_hour,
                    csh.end_hour,
                    cs.shift AS shift_name,
                    es.work_date,
                    cr.room_name,
                    ROW_NUMBER() OVER(
                        PARTITION BY es.work_date, cs._id 
                        ORDER BY csh.start_hour ASC
                    ) as is_earliest,
                    ROW_NUMBER() OVER(
                        PARTITION BY es.work_date, cs._id 
                        ORDER BY csh.end_hour DESC
                    ) as is_latest
                FROM clinic_shift_hour csh
                JOIN clinic_shift cs ON csh.shift_id = cs._id
                JOIN employee_schedule es ON es.clinic_shift_id = cs._id
                JOIN clinic_room_employee_schedule cres ON cres.employee_schedule_id = es._id
                JOIN clinic_room cr ON cr._id = cres.clinic_room_id
                JOIN accounts a0 ON a0._id = es.employee_id
                JOIN doctor_information di ON di.account_id = a0._id
                JOIN accounts a1 ON a1._id = es.clinic_id
                JOIN clinic_manager_information cmi ON cmi.account_id = a1._id
                JOIN accounts a2 ON a2._id = cmi.account_id
                JOIN clinic_admin_information cai ON cai.account_id = a2.parent_id
                JOIN addresses adr ON adr.account_id = a2._id
                WHERE es.deleted_at IS NULL
            )
            SELECT 
                clinic_id,
                main_clinic_name,
                clinic_branch_name,
                full_name,
                address,
                ward_name,
                district_name,
                province_name,
                doctor_name,
                role,
                gender,
                work_date,
                room_name,
                shift_name,
                start_hour,
                end_hour
            FROM RankedSchedule
            where work_date = :work_date 
            and LOWER(main_clinic_name) LIKE LOWER(:clinic_name) 
            and (is_earliest = 1 OR is_latest = 1)
            ORDER BY clinic_id ASC
        """)
        
        # Add wildcards to clinic name for LIKE search
        clinic_name_wildcard = f"%{clinic_name}%"
        
        result = await _db_session.execute(query, {"work_date": work_date, "clinic_name": clinic_name_wildcard})
        rows = result.fetchall()
        
        if not rows:
            return f"No schedule found for clinic '{clinic_name}' on {work_date}."
            
        # Format output
        output = f"Schedule for {clinic_name} on {work_date}:\n"
        for row in rows:
            output += (
                f"- Branch: {row.clinic_branch_name}\n"
                f"  Address: {row.address}, {row.ward_name}, {row.district_name}, {row.province_name}\n"
                f"  Doctor: {row.doctor_name} ({row.role})\n"
                f"  Shift: {row.shift_name} ({row.start_hour} - {row.end_hour})\n"
                f"  Room: {row.room_name}\n"
                f"  ----------------\n"
            )
            
        return output

    except Exception as e:
        logger.error(f"Error finding clinic schedule: {e}")
        return f"Error retrieving schedule: {str(e)}"

# --- Doctor Schedule Tool ---

class DoctorScheduleInput(BaseModel):
    """Input schema for finding doctor schedules."""
    doctor_name: str = Field(..., description="Name of the doctor to find schedule for")
    work_date: str = Field(..., description="Date to check schedule for in YYYY-MM-DD format")

async def find_doctor_schedule(doctor_name: str, work_date: str) -> str:
    """
    Find the working schedule and appointment status for a specific doctor on a given date.
    """
    global _db_session
    if not _db_session:
        return "Database session not available."

    try:
        # Validate date
        try:
            datetime.strptime(work_date, "%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

        query = text("""
            WITH appointment_counts AS (
                -- First, count appointments per shift hour
                SELECT 
                    doctor_shift_hour_id, 
                    COUNT(*) AS appointment_count
                FROM appointments
                WHERE deleted_at IS NULL -- Assuming you only want active appointments
                GROUP BY doctor_shift_hour_id
            )
            SELECT 
                es.clinic_id,
                di.account_id,
                di.full_name as doctor_name_di,
                cai.clinic_name as main_clinic_name,
                cmi.clinic_branch_name,
                cmi.full_name,
                adr.address,
                adr.ward_name,
                adr.district_name,
                adr.province_name,
                di.full_name as doctor_name,
                a0."role",
                di.gender,
                cs.shift, -- Accessing enum from clinic_shift
                es.employee_id,
                es.work_date,
                csh.start_hour,
                csh.end_hour,
                cr.room_name,
                COALESCE(ac.appointment_count, 0) AS total_appointments
            FROM clinic_shift_hour csh
            JOIN clinic_shift cs ON csh.shift_id = cs._id
            JOIN employee_schedule es ON es.clinic_shift_id = cs._id
            JOIN clinic_room_employee_schedule cres ON cres.employee_schedule_id = es._id
            JOIN clinic_room cr ON cr._id = cres.clinic_room_id
            LEFT JOIN appointment_counts ac ON ac.doctor_shift_hour_id = csh._id
            JOIN accounts a0 ON a0._id  = es.employee_id   
            JOIN doctor_information di ON di.account_id = a0._id 
            JOIN accounts a1 ON a1._id  = es.clinic_id  
            JOIN clinic_manager_information cmi ON cmi.account_id = a1._id
            JOIN accounts a2 ON a2._id  = cmi.account_id 
            JOIN clinic_admin_information cai on cai.account_id = a2.parent_id 
            JOIN addresses adr ON adr.account_id = a2._id
            WHERE es.deleted_at IS NULL 
            and LOWER(di.full_name) like LOWER(:doctor_name) 
            and es.work_date = :work_date
        """)
        
        doctor_name_wildcard = f"%{doctor_name}%"
        
        result = await _db_session.execute(query, {"work_date": work_date, "doctor_name": doctor_name_wildcard})
        rows = result.fetchall()
        
        if not rows:
            return f"No schedule found for Dr. '{doctor_name}' on {work_date}."
            
        output = f"Schedule for Dr. {rows[0].doctor_name} on {work_date}:\n"
        for row in rows:
            output += (
                f"- Clinic: {row.main_clinic_name} ({row.clinic_branch_name})\n"
                f"  Address: {row.address}\n"
                f"  Time: {row.start_hour} - {row.end_hour} ({row.shift})\n"
                f"  Room: {row.room_name}\n"
                f"  Appointments booked: {row.total_appointments}\n"
                f"  ----------------\n"
            )
            
        return output

    except Exception as e:
        logger.error(f"Error finding doctor schedule: {e}")
        return f"Error retrieving schedule: {str(e)}"

# --- Synchronous Wrappers for LangChain ---

def _find_clinic_schedule_sync(**kwargs) -> str:
    """Synchronous wrapper for clinic schedule tool."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(find_clinic_schedule(**kwargs))

def _find_doctor_schedule_sync(**kwargs) -> str:
    """Synchronous wrapper for doctor schedule tool."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(find_doctor_schedule(**kwargs))

# --- Tool Definitions ---

clinic_schedule_tool = StructuredTool.from_function(
    func=_find_clinic_schedule_sync,
    name="findClinicSchedule",
    description="""Find the working schedule/hours for a specific clinic (or branch) on a specific date. 
    Useful for answering "Is the clinic open?", "What are the hours?", "When can I visit?".
    Returns shift times, addresses, and doctor assignments.""",
    args_schema=ClinicScheduleInput
)

doctor_schedule_tool = StructuredTool.from_function(
    func=_find_doctor_schedule_sync,
    name="findDoctorSchedule",
    description="""Find the working schedule and appointments for a specific doctor on a specific date.
    Useful for answering "Is Dr. X free?", "When is Dr. X working?", "Does Dr. X have appointments?".
    Returns availability, clinic location, and current booking counts.""",
    args_schema=DoctorScheduleInput
)
