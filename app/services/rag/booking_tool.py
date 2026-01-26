"""
Booking Tool

LangChain StructuredTool for scheduling medical appointments.
Simulates inserting booking data into the appointments table.
"""
from typing import Optional
from datetime import datetime, date
from pydantic import BaseModel, Field
import uuid

from langchain.tools import StructuredTool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.logger import logger


class BookingInput(BaseModel):
    """Input schema for the appointment booking tool."""
    
    patient_name: str = Field(
        description="Name of the patient booking the appointment"
    )
    patient_phone: str = Field(
        description="Phone number of the patient"
    )
    clinic_name: str = Field(
        description="Name of the clinic for the appointment"
    )
    doctor_name: Optional[str] = Field(
        default=None,
        description="Name of the preferred doctor (optional)"
    )
    appointment_date: str = Field(
        description="Desired appointment date in YYYY-MM-DD format"
    )
    appointment_time: str = Field(
        description="Desired appointment time in HH:MM format (24-hour)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for the appointment or symptoms"
    )


class BookingOutput(BaseModel):
    """Output schema for the booking tool."""
    
    success: bool
    booking_id: Optional[str] = None
    message: str
    appointment_details: Optional[dict] = None


# Global database session holder (set during request)
_db_session: Optional[AsyncSession] = None


def set_db_session(db: AsyncSession):
    """Set the database session for the booking tool."""
    global _db_session
    _db_session = db


async def create_booking(
    patient_name: str,
    patient_phone: str,
    clinic_name: str,
    appointment_date: str,
    appointment_time: str,
    doctor_name: Optional[str] = None,
    reason: Optional[str] = None
) -> dict:
    """
    Create a medical appointment booking.
    
    This function simulates inserting a booking into the appointments table.
    In production, this would connect to the actual booking system.
    
    Args:
        patient_name: Name of the patient
        patient_phone: Patient's phone number
        clinic_name: Name of the clinic
        appointment_date: Date in YYYY-MM-DD format
        appointment_time: Time in HH:MM format
        doctor_name: Optional preferred doctor
        reason: Optional reason for visit
        
    Returns:
        Dictionary with booking confirmation details
    """
    try:
        # Validate date format
        try:
            appt_date = datetime.strptime(appointment_date, "%Y-%m-%d").date()
        except ValueError:
            return {
                "success": False,
                "message": f"Invalid date format: {appointment_date}. Please use YYYY-MM-DD format."
            }
        
        # Validate time format
        try:
            appt_time = datetime.strptime(appointment_time, "%H:%M").time()
        except ValueError:
            return {
                "success": False,
                "message": f"Invalid time format: {appointment_time}. Please use HH:MM format."
            }
        
        # Check if date is not in the past
        if appt_date < date.today():
            return {
                "success": False,
                "message": "Cannot book appointments in the past."
            }
        
        # Generate booking ID
        booking_id = f"BK{uuid.uuid4().hex[:8].upper()}"
        
        # In a real implementation, we would:
        # 1. Look up the clinic by name
        # 2. Check doctor availability
        # 3. Insert into the appointments table
        # 4. Send confirmation notification
        
        # For now, we simulate a successful booking
        appointment_details = {
            "booking_id": booking_id,
            "patient_name": patient_name,
            "patient_phone": patient_phone,
            "clinic_name": clinic_name,
            "doctor_name": doctor_name or "To be assigned",
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "reason": reason or "General consultation",
            "status": "PENDING",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Created booking: {booking_id} for {patient_name} at {clinic_name}")
        
        # If database session is available, insert the record
        if _db_session:
            try:
                # Insert into a bookings tracking table (simulated)
                await _db_session.execute(
                    text("""
                        INSERT INTO ai_messages (
                            _id, conversation_id, role, content, metadata, created_at, updated_at
                        ) VALUES (
                            gen_random_uuid(),
                            gen_random_uuid(),
                            'system',
                            :content,
                            :metadata,
                            NOW(),
                            NOW()
                        )
                    """),
                    {
                        "content": f"Booking created: {booking_id}",
                        "metadata": str(appointment_details)
                    }
                )
            except Exception as db_error:
                logger.warning(f"Could not persist booking to DB: {db_error}")
        
        return {
            "success": True,
            "booking_id": booking_id,
            "message": f"Appointment successfully scheduled! Your booking ID is {booking_id}. "
                      f"Please arrive 15 minutes early for your appointment on {appointment_date} at {appointment_time}.",
            "appointment_details": appointment_details
        }
        
    except Exception as e:
        logger.error(f"Error creating booking: {e}")
        return {
            "success": False,
            "message": f"Failed to create booking: {str(e)}"
        }


def _create_booking_sync(**kwargs) -> str:
    """Synchronous wrapper for the async booking function."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(create_booking(**kwargs))
    
    if result["success"]:
        return result["message"]
    else:
        return f"Booking failed: {result['message']}"


# Create the LangChain StructuredTool
booking_tool = StructuredTool.from_function(
    func=_create_booking_sync,
    name="scheduleAppointment",
    description="""Schedule a medical appointment for a patient.
    Use this tool when the user wants to book an appointment at a clinic.
    Required information: patient name, phone number, clinic name, date, and time.
    Optional: preferred doctor and reason for visit.""",
    args_schema=BookingInput,
    return_direct=False
)
