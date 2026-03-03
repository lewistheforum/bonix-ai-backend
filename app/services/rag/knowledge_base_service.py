"""
Knowledge Base Service

Service for ingesting and managing knowledge base data.
Pulls data from various sources and vectorizes for RAG retrieval.
"""
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.services.rag.embeddings_service import embeddings_service
from app.utils.logger import logger


class KnowledgeBaseService:
    """
    Knowledge Base Management Service
    
    Handles:
    - Document ingestion with automatic vectorization
    - Batch processing for large datasets
    - Metadata management
    - Data sync from external sources (clinics, doctors, services)
    
    Usage:
        kb_service = KnowledgeBaseService()
        await kb_service.ingest_documents(db, documents)
    """
    
    def __init__(self):
        """Initialize the knowledge base service."""
        self.embeddings_service = embeddings_service
    
    async def ingest_document(
        self,
        db: AsyncSession,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "general"
    ) -> KnowledgeBase:
        """
        Ingest a single document into the knowledge base.
        
        Args:
            db: Database session
            content: Document content
            metadata: Optional metadata dictionary
            doc_type: Document type for categorization
            
        Returns:
            Created KnowledgeBase record
        """
        try:
            # Generate embedding
            embedding = await self.embeddings_service.embed_text(content)
            
            # Prepare metadata
            full_metadata = {
                "type": doc_type,
                "ingested_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Create record
            kb_entry = KnowledgeBase(
                _id=uuid.uuid4(),
                content=content,
                embedding=embedding,
                meta_data=metadata,
                search_vector=func.to_tsvector('english', content)
            )
            
            db.add(kb_entry)
            await db.flush()
            
            logger.info(f"Ingested document: {kb_entry._id} (type: {doc_type})")
            return kb_entry
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    async def ingest_documents(
        self,
        db: AsyncSession,
        documents: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> List[KnowledgeBase]:
        """
        Batch ingest multiple documents.
        
        Args:
            db: Database session
            documents: List of dicts with 'content', 'metadata', and optional 'doc_type'
            batch_size: Number of documents to process per batch
            
        Returns:
            List of created KnowledgeBase records
        """
        all_entries = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract contents for batch embedding
            contents = [doc["content"] for doc in batch]
            
            # Generate embeddings in batch
            embeddings = await self.embeddings_service.embed_documents(contents)
            
            # Create entries
            for doc, embedding in zip(batch, embeddings):
                metadata = {
                    "type": doc.get("type", "general"),
                    "ingested_at": datetime.utcnow().isoformat(),
                    **(doc.get("metadata", {}))
                }
                
                kb_entry = KnowledgeBase(
                    _id=uuid.uuid4(),
                    content=doc["content"],
                    embedding=embedding,
                    meta_data=metadata,
                    search_vector=func.to_tsvector('english', doc["content"])
                )
                db.add(kb_entry)
                all_entries.append(kb_entry)
            
            await db.flush()
            logger.info(f"Ingested batch {i // batch_size + 1}: {len(batch)} documents")
        
        return all_entries
    
    async def ingest_clinic_services(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest clinic services data from the main database.
        
        This queries the clinic_services with clinic admin/manager linking,
        pricing, and address information for complete service data.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            # Query clinic services with full clinic hierarchy and addresses
            result = await db.execute(text("""
                SELECT 
                    csca.category_name,
                    csca.type as category_type,
                    csca.is_active as category_is_active,
                    cs.service_name,
                    cs.service_code,
                    cs.description as service_description,
                    cs.service_functions,
                    cs.is_active as service_is_active,
                    csc.price,
                    (csc.price * (csc.discount / 100)) as discount_amount,
                    csc.duration_min,
                    csc.note_for_patient,
                    csc.is_active as config_is_active,
                    a.role,
                    a.username,
                    a.email,
                    a.phone,
                    a.status,
                    cmi.clinic_branch_name,
                    cmi.full_name as full_name_clinic_branch,
                    cmi.dob as clinic_branch_dob,
                    cai.clinic_name as clinic_main_name,
                    cai.description as clinic_description,
                    cai.specialized_in,
                    cai.pros,
                    cai.paraclinical,
                    cai.dob as clinic_main_dob,
                    adr.address,
                    adr.ward_name,
                    adr.district_name,
                    adr.province_name
                FROM clinic_service_config csc
                JOIN clinic_services cs ON csc.service_id = cs._id
                JOIN clinic_service_category csca ON cs.category_id = csca._id
                JOIN accounts a ON csc.clinic_id = a._id
                JOIN clinic_manager_information cmi ON cmi.account_id = a._id
                JOIN clinic_admin_information cai ON cai.account_id = a.parent_id
                JOIN addresses adr ON adr.account_id = a._id
                WHERE cs.deleted_at IS NULL AND cs.is_active = true
            """))
            
            services = result.fetchall()
            
            documents = []
            for service in services:
                # Build address string
                address_parts = filter(None, [
                    service.address,
                    service.ward_name,
                    service.district_name,
                    service.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                # Format specialized_in if it's a list/array
                specialized = service.specialized_in
                if isinstance(specialized, list):
                    specialized = ", ".join(specialized)
                
                content = f"""
Service: {service.service_name}
Code: {service.service_code}
Category: {service.category_name or 'General'}
Description: {service.service_description or 'No description available'}
Functions: {', '.join(service.service_functions) if service.service_functions else 'N/A'}
Price: {service.price or 'N/A'}
Discount: {service.discount_amount or 0}
Duration: {service.duration_min or 'N/A'} minutes
Note for patient: {service.note_for_patient or 'N/A'}
Clinic Branch: {service.clinic_branch_name or 'N/A'}
Main Clinic: {service.clinic_main_name or 'N/A'}
Clinic Description: {service.clinic_description or 'N/A'}
Specializations: {specialized or 'N/A'}
Address: {full_address}
Contact Email: {service.email or 'N/A'}
Contact Phone: {service.phone or 'N/A'}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "clinic_services",
                    "metadata": {
                        "type": "clinic_services",
                        "service_name": service.service_name,
                        "service_code": service.service_code,
                        "category": service.category_name,
                        "price": float(service.price) if service.price else None,
                        "duration_min": service.duration_min,
                        "clinic_branch": service.clinic_branch_name,
                        "clinic_main": service.clinic_main_name,
                        "province": service.province_name
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} clinic services")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting clinic services: {e}")
            return 0
    
    async def ingest_doctor_profiles(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest doctor information with clinic hierarchy and addresses.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
                SELECT 
                    di.full_name,
                    di.gender,
                    di.academic_degree,
                    di.experience,
                    di.position,
                    di.introduction_1,
                    di.work_process_2,
                    di.study_process_3,
                    di.members_4,
                    di.scientific_work_5,
                    di.papers_6,
                    di.dob as doctor_dob,
                    a2.username,
                    a2.email,
                    a2.phone,
                    a2.status,
                    cmi.clinic_branch_name,
                    cmi.full_name as full_name_clinic_branch,
                    cmi.dob as clinic_branch_dob,
                    cai.clinic_name as clinic_main_name,
                    cai.description as clinic_description,
                    cai.specialized_in,
                    cai.pros,
                    cai.paraclinical,
                    cai.dob as clinic_main_dob,
                    adr.address,
                    adr.ward_name,
                    adr.district_name,
                    adr.province_name
                FROM doctor_information di
                JOIN accounts a1 ON a1._id = di.account_id
                JOIN clinic_manager_information cmi ON cmi.account_id = a1.parent_id
                JOIN accounts a2 ON a2._id = cmi.account_id
                JOIN clinic_admin_information cai ON cai.account_id = a2.parent_id
                JOIN addresses adr ON adr.account_id = a2._id
                WHERE di.deleted_at IS NULL
            """))
            
            doctors = result.fetchall()
            
            documents = []
            for doctor in doctors:
                # Build address string
                address_parts = filter(None, [
                    doctor.address,
                    doctor.ward_name,
                    doctor.district_name,
                    doctor.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                # Format JSONB fields
                intro = doctor.introduction_1 if doctor.introduction_1 else "No introduction"
                work_process = doctor.work_process_2 if doctor.work_process_2 else "N/A"
                study_process = doctor.study_process_3 if doctor.study_process_3 else "N/A"
                members = doctor.members_4 if doctor.members_4 else "N/A"
                scientific_work = doctor.scientific_work_5 if doctor.scientific_work_5 else "N/A"
                papers = doctor.papers_6 if doctor.papers_6 else "N/A"
                
                content = f"""
Doctor: {doctor.full_name}
Gender: {doctor.gender or 'N/A'}
Academic Degree: {doctor.academic_degree or 'N/A'}
Position: {doctor.position or 'N/A'}
Experience: {doctor.experience or 'N/A'}
Date of Birth: {doctor.doctor_dob or 'N/A'}
Introduction: {intro}
Work Process: {work_process}
Study Process: {study_process}
Professional Memberships: {members}
Scientific Work: {scientific_work}
Papers/Awards: {papers}
Contact Email: {doctor.email or 'N/A'}
Contact Phone: {doctor.phone or 'N/A'}
Clinic Branch: {doctor.clinic_branch_name or 'N/A'}
Main Clinic: {doctor.clinic_main_name or 'N/A'}
Clinic Description: {doctor.clinic_description or 'N/A'}
Clinic Address: {full_address}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "doctor_profile",
                    "metadata": {
                        "type": "doctor_profile",
                        "doctor_name": doctor.full_name,
                        "gender": doctor.gender,
                        "degree": doctor.academic_degree,
                        "position": doctor.position,
                        "clinic_branch": doctor.clinic_branch_name,
                        "clinic_main": doctor.clinic_main_name,
                        "province": doctor.province_name
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} doctor profiles")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting doctor profiles: {e}")
            return 0
    
    async def ingest_clinic_info(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest clinic information from clinic manager joined with admin.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
                SELECT 
                    a._id as clinic_id,
                    a.username,
                    a.role,
                    a.email,
                    a.phone,
                    a.status,
                    cai.clinic_name,
                    cmi.clinic_branch_name,
                    cmi.full_name,
                    cmi.dob as clinic_branch_dob,
                    cai.description,
                    cai.specialized_in,
                    cai.pros,
                    cai.paraclinical,
                    cai.dob as clinic_main_dob,
                    adr.address,
                    adr.ward_name,
                    adr.district_name,
                    adr.province_name
                FROM clinic_manager_information cmi
                JOIN accounts a ON a._id = cmi.account_id
                JOIN clinic_admin_information cai ON cai.account_id = a.parent_id
                LEFT JOIN addresses adr ON adr.account_id = a._id
                WHERE a.deleted_at IS NULL
            """))
            
            clinics = result.fetchall()
            
            documents = []
            for clinic in clinics:
                # Build address string
                address_parts = filter(None, [
                    clinic.address,
                    clinic.ward_name,
                    clinic.district_name,
                    clinic.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                # Format JSONB fields
                specialized = clinic.specialized_in
                if isinstance(specialized, list):
                    specialized = ", ".join(specialized)
                
                pros = clinic.pros
                if isinstance(pros, list):
                    pros = ", ".join(pros)
                
                paraclinical = clinic.paraclinical
                if isinstance(paraclinical, list):
                    paraclinical = ", ".join(paraclinical)
                
                content = f"""
Clinic Information
Clinic Name: {clinic.clinic_name or 'N/A'}
Branch: {clinic.clinic_branch_name or 'N/A'}
Manager: {clinic.full_name or 'N/A'}
Description: {clinic.description or 'No description'}
Specializations: {specialized or 'N/A'}
Key Advantages: {pros or 'N/A'}
Paraclinical Services: {paraclinical or 'N/A'}
Address: {full_address}
Contact Email: {clinic.email or 'N/A'}
Contact Phone: {clinic.phone or 'N/A'}
Status: {clinic.status or 'N/A'}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "clinic_info",
                    "metadata": {
                        "type": "clinic_info",
                        "clinic_id": str(clinic.clinic_id),
                        "clinic_branch": clinic.clinic_branch_name,
                        "clinic_main": clinic.clinic_name,
                        "province": clinic.province_name,
                        "email": clinic.email
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} clinic profiles")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting clinic info: {e}")
            return 0
    
    async def ingest_staff_info(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest clinic staff information for contact purposes.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
                SELECT 
                    csi.full_name,
                    csi.gender,
                    csi.clinic_role,
                    csi.dob as staff_dob,
                    a2.username,
                    a1.email,
                    a1.phone,
                    a2.status,
                    cmi.clinic_branch_name,
                    cmi.full_name as full_name_clinic_branch,
                    cmi.dob as clinic_branch_dob,
                    cai.clinic_name as clinic_main_name,
                    cai.description,
                    cai.specialized_in,
                    cai.pros,
                    cai.paraclinical,
                    cai.dob as clinic_main_dob,
                    adr.address,
                    adr.ward_name,
                    adr.district_name,
                    adr.province_name
                FROM clinic_staff_information csi
                JOIN accounts a1 ON a1._id = csi.account_id
                JOIN clinic_manager_information cmi ON cmi.account_id = a1.parent_id
                JOIN accounts a2 ON a2._id = cmi.account_id
                JOIN clinic_admin_information cai ON cai.account_id = a2.parent_id
                JOIN addresses adr ON adr.account_id = a2._id
                WHERE csi.deleted_at IS NULL
            """))
            
            staff_members = result.fetchall()
            
            documents = []
            for staff in staff_members:
                # Build address string
                address_parts = filter(None, [
                    staff.address,
                    staff.ward_name,
                    staff.district_name,
                    staff.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                content = f"""
Staff: {staff.full_name}
Gender: {staff.gender or 'N/A'}
Role: {staff.clinic_role or 'N/A'}
Date of Birth: {staff.staff_dob or 'N/A'}
Contact Email: {staff.email or 'N/A'}
Contact Phone: {staff.phone or 'N/A'}
Clinic Branch: {staff.clinic_branch_name or 'N/A'}
Main Clinic: {staff.clinic_main_name or 'N/A'}
Clinic Address: {full_address}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "staff_info",
                    "metadata": {
                        "type": "staff_info",
                        "staff_name": staff.full_name,
                        "role": staff.clinic_role,
                        "clinic_branch": staff.clinic_branch_name,
                        "clinic_main": staff.clinic_main_name,
                        "province": staff.province_name
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} staff profiles")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting staff info: {e}")
            return 0
    
    async def ingest_blogs(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest blog information for informational purposes.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
                SELECT 
                    _id,
                    clinic_id,
                    title,
                    content,
                    thumbnail,
                    type
                FROM blogs
                WHERE deleted_at IS NULL
            """))
            
            blogs = result.fetchall()
            
            documents = []
            for blog in blogs:
                content = f"""
Blog Title: {blog.title or 'Untitled'}
Type: {blog.type or 'General'}
Content: {blog.content or 'No content'}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "blog_info",
                    "metadata": {
                        "type": "blog_info",
                        "blog_id": str(blog._id),
                        "clinic_id": str(blog.clinic_id) if blog.clinic_id else None,
                        "title": blog.title,
                        "blog_type": blog.type
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} blogs")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting blogs: {e}")
            return 0
    
    async def ingest_feedbacks(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest doctor and clinic feedback for recommendations.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
                SELECT 
                    f.appointment_id,
                    f.clinic_id,
                    f.doctor_id,
                    f.rating,
                    f.description,
                    f.type,
                    di.full_name,
                    di.gender,
                    di.dob,
                    a2.username,
                    a2.email,
                    a2.phone,
                    a2.role,
                    cmi.clinic_branch_name,
                    cmi.full_name as full_name_clinic_branch,
                    cmi.dob as clinic_branch_dob,
                    cai.clinic_name as clinic_main_name,
                    cai.description as clinic_description,
                    cai.specialized_in,
                    cai.pros,
                    cai.paraclinical,
                    cai.dob as clinic_main_dob,
                    adr.address,
                    adr.ward_name,
                    adr.district_name,
                    adr.province_name
                FROM feedbacks f
                LEFT JOIN doctor_information di ON di.account_id = f.doctor_id
                LEFT JOIN accounts a2 ON a2._id = di.account_id
                LEFT JOIN clinic_manager_information cmi ON cmi.account_id = f.clinic_id
                LEFT JOIN accounts a3 ON a3._id = f.clinic_id
                LEFT JOIN clinic_admin_information cai ON cai.account_id = a3.parent_id
                LEFT JOIN addresses adr ON adr.account_id = a3._id
                WHERE f.deleted_at IS NULL
            """))
            
            feedbacks = result.fetchall()
            
            documents = []
            for feedback in feedbacks:
                # Build address string
                address_parts = filter(None, [
                    feedback.address,
                    feedback.ward_name,
                    feedback.district_name,
                    feedback.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                content = f"""
Feedback Type: {feedback.type or 'General'}
Rating: {feedback.rating or 'N/A'}/5
Description: {feedback.description or 'No description'}
Doctor: {feedback.full_name or 'N/A'}
Doctor Gender: {feedback.gender or 'N/A'}
Clinic Branch: {feedback.clinic_branch_name or 'N/A'}
Main Clinic: {feedback.clinic_main_name or 'N/A'}
Clinic Address: {full_address}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "feedback",
                    "metadata": {
                        "type": "feedback",
                        "feedback_type": feedback.type,
                        "rating": feedback.rating,
                        "doctor_name": feedback.full_name,
                        "clinic_branch": feedback.clinic_branch_name,
                        "clinic_main": feedback.clinic_main_name
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} feedbacks")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting feedbacks: {e}")
            return 0
    
    async def ingest_user_info(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest patient user information.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
                SELECT 
                    ga.full_name,
                    ga.gender,
                    ga.dob,
                    a.username,
                    a.email,
                    a.phone,
                    a.role,
                    a.ban_counts,
                    a.ban_description
                FROM general_accounts ga
                JOIN accounts a ON ga.account_id = a._id AND a.role = 'PATIENT'
                WHERE ga.deleted_at IS NULL
            """))
            
            users = result.fetchall()
            
            documents = []
            for user in users:
                content = f"""
Patient: {user.full_name or user.username}
Gender: {user.gender or 'N/A'}
Date of Birth: {user.dob or 'N/A'}
Email: {user.email or 'N/A'}
Phone: {user.phone or 'N/A'}
Ban Count: {user.ban_counts or 0}
Ban Description: {user.ban_description or 'None'}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "user_info",
                    "metadata": {
                         "type": "user_info",
                        "user_name": user.full_name or user.username,
                        "email": user.email,
                        "phone": user.phone
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} user profiles")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting user info: {e}")
            return 0
    
    async def ingest_doctor_schedules(
        self,
        db: AsyncSession,
        clinic_id: Optional[str] = None,
        work_date: Optional[str] = None,
        employee_id: Optional[str] = None
    ) -> int:
        """
        Ingest doctor schedules with appointment counts.
        
        Args:
            db: Database session
            clinic_id: Optional filter by clinic
            work_date: Optional filter by date
            employee_id: Optional filter by doctor/employee
            
        Returns:
            Number of documents ingested
        """
        try:
            # Build query with optional filters
            query = """
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
            """
            
            params = {}
            if clinic_id:
                query += " AND es.clinic_id = :clinic_id"
                params["clinic_id"] = clinic_id
            if work_date:
                query += " AND es.work_date = :work_date"
                params["work_date"] = work_date
            if employee_id:
                query += " AND es.employee_id = :employee_id"
                params["employee_id"] = employee_id
            
            result = await db.execute(text(query), params)
            schedules = result.fetchall()
            
            documents = []
            for schedule in schedules:
                # Build address string
                address_parts = filter(None, [
                    schedule.address,
                    schedule.ward_name,
                    schedule.district_name,
                    schedule.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                content = f"""
Doctor Schedule
Doctor: {schedule.doctor_name}
Gender: {schedule.gender or 'N/A'}
Work Date: {schedule.work_date}
Shift: {schedule.shift or 'N/A'}
Room: {schedule.room_name or 'N/A'}
Total Appointments: {schedule.total_appointments}
Clinic Branch: {schedule.clinic_branch_name or 'N/A'}
Main Clinic: {schedule.main_clinic_name or 'N/A'}
Clinic Address: {full_address}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "schedule_info",
                    "metadata": {
                        "type": "schedule_info",
                        "doctor_name": schedule.doctor_name,
                        "work_date": str(schedule.work_date),
                        "shift": schedule.shift,
                        "room": schedule.room_name,
                        "appointments": schedule.total_appointments,
                        "clinic_branch": schedule.clinic_branch_name,
                        "clinic_main": schedule.main_clinic_name,
                        "clinic_id": str(schedule.clinic_id),
                        "employee_id": str(schedule.employee_id)
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} doctor schedules")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting doctor schedules: {e}")
            return 0
    
    async def ingest_clinic_working_hours(
        self,
        db: AsyncSession
    ) -> int:
        """
        Ingest general clinic working hours.
        
        Args:
            db: Database session
            
        Returns:
            Number of documents ingested
        """
        try:
            result = await db.execute(text("""
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
                WHERE is_earliest = 1 OR is_latest = 1
                ORDER BY clinic_id ASC
            """))
            
            working_hours = result.fetchall()
            
            documents = []
            for hours in working_hours:
                # Build address string
                address_parts = filter(None, [
                    hours.address,
                    hours.ward_name,
                    hours.district_name,
                    hours.province_name
                ])
                full_address = ", ".join(address_parts) or "N/A"
                
                content = f"""
Clinic Working Hours
Main Clinic: {hours.main_clinic_name or 'N/A'}
Branch: {hours.clinic_branch_name or 'N/A'}
Work Date: {hours.work_date}
Shift: {hours.shift_name or 'N/A'}
Start Hour: {hours.start_hour or 'N/A'}
End Hour: {hours.end_hour or 'N/A'}
Room: {hours.room_name or 'N/A'}
Doctor: {hours.doctor_name or 'N/A'}
Address: {full_address}
                """.strip()
                
                documents.append({
                    "content": content,
                    "type": "clinic_working_hours",
                    "metadata": {
                        "type": "clinic_working_hours",
                        "clinic_main": hours.main_clinic_name,
                        "clinic_branch": hours.clinic_branch_name,
                        "work_date": str(hours.work_date),
                        "shift": hours.shift_name,
                        "start_hour": str(hours.start_hour) if hours.start_hour else None,
                        "end_hour": str(hours.end_hour) if hours.end_hour else None
                    }
                })
            
            if documents:
                await self.ingest_documents(db, documents)
            
            logger.info(f"Ingested {len(documents)} clinic working hours")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error ingesting clinic working hours: {e}")
            return 0
    
    async def get_all_documents(
        self,
        db: AsyncSession,
        doc_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[KnowledgeBase]:
        """
        Get all documents from the knowledge base.
        
        Args:
            db: Database session
            doc_type: Optional filter by document type
            limit: Maximum number of documents to return
            
        Returns:
            List of KnowledgeBase records
        """
        query = select(KnowledgeBase).where(
            KnowledgeBase.deleted_at.is_(None)
        ).limit(limit)
        
        if doc_type:
            query = query.where(
                KnowledgeBase.meta_data["type"].astext == doc_type
            )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def delete_document(
        self,
        db: AsyncSession,
        document_id: str
    ) -> bool:
        """
        Soft delete a document from the knowledge base.
        
        Args:
            db: Database session
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            doc_uuid = uuid.UUID(document_id)
            result = await db.execute(
                select(KnowledgeBase).where(KnowledgeBase._id == doc_uuid)
            )
            document = result.scalar_one_or_none()
            
            if document:
                document.deleted_at = datetime.utcnow()
                await db.flush()
                logger.info(f"Deleted document: {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def clear_knowledge_base(
        self,
        db: AsyncSession,
        doc_type: Optional[str] = None
    ) -> int:
        """
        Clear all or specific type of documents from knowledge base.
        
        Args:
            db: Database session
            doc_type: Optional document type to clear
            
        Returns:
            Number of documents cleared
        """
        try:
            query = select(KnowledgeBase).where(
                KnowledgeBase.deleted_at.is_(None)
            )
            
            if doc_type:
                query = query.where(
                    KnowledgeBase.meta_data["type"].astext == doc_type
                )
            
            result = await db.execute(query)
            documents = result.scalars().all()
            
            for doc in documents:
                doc.deleted_at = datetime.utcnow()
            
            await db.flush()
            logger.info(f"Cleared {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return 0


# Singleton instance
knowledge_base_service = KnowledgeBaseService()
