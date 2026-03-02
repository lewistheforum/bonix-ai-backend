"""
Service for Recommendation Clinic business logic
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.recommendation.recommendation_clinic_dto import (
    ClinicRecommendationRequest,
    RecommendationClinicResponse,
    ClinicInfo
)
from app.database import AsyncSessionLocal


class RecommendationClinicService:
    """Service class for clinic recommendation operations"""
    
    # Minimum similarity score for a clinic to be considered "similar"
    MIN_SCORE_THRESHOLD = 0.5
    # Soft maximum number of results to avoid excessively large responses
    MAX_RESULTS = 10


    def _calculate_match_score(self, clinic: dict, request: ClinicRecommendationRequest) -> float:
        """
        Calculate match score between clinic and request criteria
        Focuses on: description, specialized_in, pros, paraclinical
        
        Args:
            clinic: Clinic data dictionary
            request: Clinic recommendation request
            
        Returns:
            Match score (higher is better)
        """
        score = 0.0
        
        # Match by specialized_in — strongest similarity signal
        if request.specialized_in:
            specialized_score = self._list_overlap_score(
                clinic.get("specialized_in") or [],
                request.specialized_in
            )
            score += specialized_score * 4.0
        
        # Match by paraclinical services
        if request.paraclinical:
            paraclinical_score = self._list_overlap_score(
                clinic.get("paraclinical") or [],
                request.paraclinical
            )
            score += paraclinical_score * 3.0
        
        # Match by description (text similarity)
        if request.description:
            description_score = self._text_similarity(
                clinic.get("description", "") or "",
                request.description
            )
            score += description_score * 2.0
        
        # Match by pros
        if request.pros:
            pros_score = self._list_overlap_score(
                clinic.get("pros") or [],
                request.pros
            )
            score += pros_score * 1.5
        
        return score
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _list_overlap_score(self, list1: List[str], list2: List[str]) -> float:
        """
        Calculate overlap score between two lists (case-insensitive)
        
        Args:
            list1: First list
            list2: Second list
            
        Returns:
            Overlap score (0.0 to 1.0)
        """
        if not list1 or not list2:
            return 0.0
        
        set1 = set(item.lower().strip() for item in list1)
        set2 = set(item.lower().strip() for item in list2)
        
        # Calculate exact matches first
        intersection = set1.intersection(set2)
        match_score = len(intersection)
        
        # For items not exactly matched, check partial matches
        unmatched_target = set2 - intersection
        unmatched_candidate = set1 - intersection
        
        for target_item in unmatched_target:
            best_partial = 0.0
            target_words = set(target_item.split())
            for candidate_item in unmatched_candidate:
                # Substring match (e.g., "x-ray" matches "digital x-ray")
                if target_item in candidate_item or candidate_item in target_item:
                    best_partial = max(best_partial, 0.5)
                    continue
                # Word-level overlap (e.g., "Bone Surgery" vs "Orthopedic Surgery")
                candidate_words = set(candidate_item.split())
                common_words = target_words & candidate_words
                if common_words and len(common_words) >= 1:
                    word_overlap = len(common_words) / max(len(target_words), len(candidate_words))
                    best_partial = max(best_partial, word_overlap * 0.4)
            match_score += best_partial
        
        return min(match_score / len(set2), 1.0) if set2 else 0.0
    
    def _calculate_frequency_bonus(
        self,
        clinic: dict,
        all_specialized_in: List[str],
        all_pros: List[str],
        all_paraclinical: List[str]
    ) -> float:
        """
        Calculate bonus score based on how frequently characteristics appear
        in the patient's appointment history.
        
        Args:
            clinic: Clinic data dictionary
            all_specialized_in: All specializations from input clinics (with duplicates)
            all_pros: All pros from input clinics (with duplicates)
            all_paraclinical: All paraclinical services from input clinics (with duplicates)
            
        Returns:
            Frequency bonus score
        """
        bonus = 0.0
        
        # Count frequency of each characteristic in the input
        from collections import Counter
        
        specialized_freq = Counter(item.lower() for item in all_specialized_in)
        pros_freq = Counter(item.lower() for item in all_pros)
        paraclinical_freq = Counter(item.lower() for item in all_paraclinical)
        
        # Add bonus for matching frequent characteristics
        clinic_specialized = [s.lower() for s in (clinic.get("specialized_in") or [])]
        clinic_pros = [p.lower() for p in (clinic.get("pros") or [])]
        clinic_paraclinical = [p.lower() for p in (clinic.get("paraclinical") or [])]
        
        for spec in clinic_specialized:
            if spec in specialized_freq:
                bonus += specialized_freq[spec] * 0.5  # Weight for specialization frequency
        
        for pro in clinic_pros:
            if pro in pros_freq:
                bonus += pros_freq[pro] * 0.3  # Weight for pros frequency
        
        for para in clinic_paraclinical:
            if para in paraclinical_freq:
                bonus += paraclinical_freq[para] * 0.4  # Weight for paraclinical frequency
        
        return bonus

    async def get_clinic_managers_by_admin_ids(self, admin_ids: List[str]) -> List[dict]:
        """
        Fetch clinic managers for specific clinic admins from database
        """
        if not admin_ids:
            return []
            
        query = text("""
            SELECT 
                accounts._id,
                accounts.email,
                accounts.phone,
                cmi.clinic_branch_name as clinic_name,
                cmi.dob,
                cmi.profile_picture,
                cmi.created_at,
                cmi.updated_at
            FROM accounts
            JOIN clinic_manager_information cmi ON accounts._id = cmi.account_id
            WHERE accounts.parent_id = ANY(:admin_ids) AND accounts.role = 'CLINIC_MANAGER'
        """)
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(query, {"admin_ids": admin_ids})
            rows = result.fetchall()
            
            managers = []
            for row in rows:
                manager = {
                    "id": str(row._id) if row._id else None,
                    "email": row.email,
                    "phone": row.phone,
                    "clinic_name": row.clinic_name,
                    "description": None,
                    "specialized_in": [],
                    "pros": [],
                    "paraclinical": [],
                    "dob": row.dob,
                    "profile_picture": row.profile_picture,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at
                }
                managers.append(manager)
            
            return managers

    async def get_all_clinics(self) -> List[dict]:
        """
        Fetch all clinics from database
        
        Returns:
            List of clinic data dictionaries
        """
        query = text("""
            SELECT 
                accounts._id,
                accounts.email,
                accounts.phone,
                cai.clinic_name,
                cai.description,
                cai.specialized_in,
                cai.pros,
                cai.paraclinical,
                cai.dob,
                cai.profile_picture,
                cai.created_at,
                cai.updated_at
            FROM accounts
            join clinic_admin_information cai 
            on accounts._id = cai.account_id
            join clinic_subcriptions cs on cs.clinic_id = accounts._id
            where cs.subscription_status = 'ACTIVE'
        """)
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(query)
            rows = result.fetchall()
            
            clinics = []
            for row in rows:
                clinic = {
                    "id": str(row._id) if row._id else None,
                    "email": row.email,
                    "phone": row.phone,
                    "clinic_name": row.clinic_name,
                    "description": row.description,
                    "specialized_in": row.specialized_in if row.specialized_in else [],
                    "pros": row.pros if row.pros else [],
                    "paraclinical": row.paraclinical if row.paraclinical else [],
                    "dob": row.dob,
                    "profile_picture": row.profile_picture,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at
                }
                clinics.append(clinic)
            
            return clinics
    
    async def get_clinic_by_id(self, clinic_id: str) -> Optional[dict]:
        """
        Fetch a specific clinic by its ID from database
        
        Args:
            clinic_id: The clinic ID to search for
            
        Returns:
            Clinic data dictionary or None if not found
        """
        query = text("""
            SELECT 
                accounts._id,
                accounts.email,
                accounts.phone,
                cai.clinic_name,
                cai.description,
                cai.specialized_in,
                cai.pros,
                cai.paraclinical,
                cai.dob,
                cai.profile_picture,
                cai.created_at,
                cai.updated_at
            FROM accounts, clinic_admin_information cai 
            WHERE accounts._id = cai.account_id  AND accounts._id = :clinic_id
        """)


        async with AsyncSessionLocal() as session:
            result = await session.execute(query, {"clinic_id": clinic_id})

            row = result.fetchone()
            
            if not row:
                return None
            
            return {
                "id": str(row._id) if row._id else None,
                "email": row.email,
                "phone": row.phone,
                "clinic_name": row.clinic_name,
                "description": row.description,
                "specialized_in": row.specialized_in if row.specialized_in else [],
                "pros": row.pros if row.pros else [],
                "paraclinical": row.paraclinical if row.paraclinical else [],
                "dob": row.dob,
                "profile_picture": row.profile_picture,
                "created_at": row.created_at,
                "updated_at": row.updated_at
            }

    async def get_similar_clinics(self, clinic_id: str) -> RecommendationClinicResponse:
        """
        Get similar clinics based on a specific clinic's data.
        Returns only clinics with a meaningful similarity score (above threshold).
        
        Args:
            clinic_id: The clinic ID to find similar clinics for
            
        Returns:
            RecommendationClinicResponse with genuinely similar clinics
        """
        # Fetch the target clinic
        target_clinic = await self.get_clinic_by_id(clinic_id)

        if not target_clinic:
            return RecommendationClinicResponse(recommendationsClinicAdmins=[], recommendationsClinicManagers=[])
        
        # Fetch all clinics
        all_clinics = await self.get_all_clinics()

        # Create a request object from the target clinic's data
        target_request = ClinicRecommendationRequest(
            description=target_clinic.get("description"),
            specialized_in=target_clinic.get("specialized_in"),
            pros=target_clinic.get("pros"),
            paraclinical=target_clinic.get("paraclinical")
        )

        # Score each clinic based on similarity to target clinic
        scored_clinics = []
        
        for clinic in all_clinics:
            # Skip the target clinic itself
            if clinic["id"] == clinic_id:
                continue
            
            score = self._calculate_match_score(clinic, target_request)
            
            # Only include clinics above the minimum similarity threshold
            if score >= self.MIN_SCORE_THRESHOLD:
                scored_clinics.append((clinic, score))
        
        # Sort by score (descending) — most similar first
        scored_clinics.sort(key=lambda x: x[1], reverse=True)
        
        # Apply soft max to avoid excessively large responses
        scored_clinics = scored_clinics[:self.MAX_RESULTS]
        
        # Format clinic information
        admin_infos = []
        admin_ids = []
        for clinic, score in scored_clinics:
            clinic_info = ClinicInfo(
                id=clinic["id"],
                email=clinic["email"],
                phone=clinic["phone"],
                clinic_name=clinic["clinic_name"],
                description=clinic["description"],
                specialized_in=clinic["specialized_in"],
                pros=clinic["pros"],
                paraclinical=clinic["paraclinical"],
                dob=clinic["dob"],
                profile_picture=clinic["profile_picture"],
                created_at=clinic["created_at"],
                updated_at=clinic["updated_at"]
            )
            admin_infos.append(clinic_info)
            if clinic["id"]:
                admin_ids.append(clinic["id"])

        manager_infos = []
        if admin_ids:
            managers = await self.get_clinic_managers_by_admin_ids(admin_ids)
            for m in managers:
                manager_info = ClinicInfo(
                    id=m["id"],
                    email=m["email"],
                    phone=m["phone"],
                    clinic_name=m["clinic_name"],
                    description=m["description"],
                    specialized_in=m["specialized_in"],
                    pros=m["pros"],
                    paraclinical=m["paraclinical"],
                    dob=m["dob"],
                    profile_picture=m["profile_picture"],
                    created_at=m["created_at"],
                    updated_at=m["updated_at"]
                )
                manager_infos.append(manager_info)

        return RecommendationClinicResponse(
            recommendationsClinicAdmins=admin_infos,
            recommendationsClinicManagers=manager_infos
        )

    async def get_recommendations_from_patient_appointments(
        self, 
        clinic_ids: List[str], 
        limit: int = 5
    ) -> RecommendationClinicResponse:
        """
        Get clinic recommendations based on patient's appointment history.
        Analyzes the given clinics and recommends similar ones.
        
        Args:
            clinic_ids: List of clinic IDs from patient's appointment history (max 5)
            limit: Maximum number of recommendations to return (default: 5)
            
        Returns:
            RecommendationClinicResponse with recommended clinics
        """

        # Fetch all input clinics
        input_clinics = []
        for clinic_id in clinic_ids:
            clinic = await self.get_clinic_by_id(clinic_id)
            if clinic:
                input_clinics.append(clinic)
        
        if not input_clinics:
            return RecommendationClinicResponse(recommendationsClinicAdmins=[], recommendationsClinicManagers=[])
        
        # Aggregate characteristics from all input clinics
        aggregated_specialized_in = []
        aggregated_pros = []
        aggregated_paraclinical = []
        aggregated_description_words = []
        
        for clinic in input_clinics:
            if clinic.get("specialized_in"):
                aggregated_specialized_in.extend(clinic["specialized_in"])
            if clinic.get("pros"):
                aggregated_pros.extend(clinic["pros"])
            if clinic.get("paraclinical"):
                aggregated_paraclinical.extend(clinic["paraclinical"])
            if clinic.get("description"):
                aggregated_description_words.extend(clinic["description"].split())
        
        # Create aggregated request for matching
        aggregated_request = ClinicRecommendationRequest(
            description=" ".join(aggregated_description_words) if aggregated_description_words else None,
            specialized_in=list(set(aggregated_specialized_in)) if aggregated_specialized_in else None,
            pros=list(set(aggregated_pros)) if aggregated_pros else None,
            paraclinical=list(set(aggregated_paraclinical)) if aggregated_paraclinical else None
        )
        
        # Fetch all clinics
        all_clinics = await self.get_all_clinics()
        
        # Create set of input clinic IDs for exclusion
        input_clinic_ids = set(clinic_ids)
        
        # Score each clinic based on familiarity with input clinics
        scored_clinics = []
        
        for clinic in all_clinics:
            # Skip input clinics
            if clinic["id"] in input_clinic_ids:
                continue
            
            # Calculate match score based on aggregated characteristics
            score = self._calculate_match_score(clinic, aggregated_request)
            
            # Add frequency bonus: if a characteristic appears multiple times in input clinics,
            # give extra score for clinics that match those frequent characteristics
            frequency_bonus = self._calculate_frequency_bonus(
                clinic,
                aggregated_specialized_in,
                aggregated_pros,
                aggregated_paraclinical
            )
            
            total_score = score + frequency_bonus
            scored_clinics.append((clinic, total_score))
        
        # Sort by score (descending)
        scored_clinics.sort(key=lambda x: x[1], reverse=True)
        
        # Format clinic information
        admin_infos = []
        admin_ids = []
        for clinic, score in scored_clinics:
            if score > 0 or len(admin_infos) < 3:  # Include at least 3 clinics if available
                clinic_info = ClinicInfo(
                    id=clinic["id"],
                    email=clinic["email"],
                    phone=clinic["phone"],
                    clinic_name=clinic["clinic_name"],
                    description=clinic["description"],
                    specialized_in=clinic["specialized_in"],
                    pros=clinic["pros"],
                    paraclinical=clinic["paraclinical"],
                    dob=clinic["dob"],
                    profile_picture=clinic["profile_picture"],
                    created_at=clinic["created_at"],
                    updated_at=clinic["updated_at"]
                )
                admin_infos.append(clinic_info)
                if clinic["id"]:
                    admin_ids.append(clinic["id"])
            
            if len(admin_infos) >= limit:
                break

        manager_infos = []
        if admin_ids:
            managers = await self.get_clinic_managers_by_admin_ids(admin_ids)
            for m in managers:
                manager_info = ClinicInfo(
                    id=m["id"],
                    email=m["email"],
                    phone=m["phone"],
                    clinic_name=m["clinic_name"],
                    description=m["description"],
                    specialized_in=m["specialized_in"],
                    pros=m["pros"],
                    paraclinical=m["paraclinical"],
                    dob=m["dob"],
                    profile_picture=m["profile_picture"],
                    created_at=m["created_at"],
                    updated_at=m["updated_at"]
                )
                manager_infos.append(manager_info)

        return RecommendationClinicResponse(
            recommendationsClinicAdmins=admin_infos,
            recommendationsClinicManagers=manager_infos
        )

    

# Create service instance
recommendation_clinic_service = RecommendationClinicService()
