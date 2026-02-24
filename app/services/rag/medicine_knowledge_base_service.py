"""
Medicine Knowledge Base Service

Service for syncing medicine data grouped by therapeutic class into the
knowledge_base_medicines table with vector embeddings for RAG retrieval.
"""
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from collections import defaultdict

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base_medicines import KnowledgeBaseMedicines
from app.services.rag.embeddings_service import embeddings_service
from app.utils.logger import logger


class MedicineKnowledgeBaseService:
    """
    Medicine Knowledge Base Service

    Handles:
    - Grouping medicines by therapeutic_class
    - Building rich text content per therapeutic class
    - Generating embeddings and tsvectors
    - Batch upserting into knowledge_base_medicines table
    """

    def __init__(self):
        """Initialize the service."""
        self.embeddings_service = embeddings_service

    async def sync_medicines(
        self,
        db: AsyncSession,
        clear_existing: bool = False
    ) -> Dict[str, int]:
        """
        Sync medicines from the medicines table into knowledge_base_medicines,
        grouped by therapeutic_class.

        Args:
            db: Database session
            clear_existing: If True, soft-delete all existing entries before sync

        Returns:
            Dict with therapeutic_classes_synced and total_medicines_processed
        """
        try:
            if clear_existing:
                await self.clear_medicine_knowledge_base(db)

            # Query all active medicines grouped by therapeutic_class
            result = await db.execute(text("""
                SELECT 
                    COALESCE(NULLIF(TRIM(therapeutic_class), ''), 'Other') as therapeutic_class,
                    COUNT(*) as medicine_count,
                    ARRAY_AGG(DISTINCT name) as medicine_names,
                    ARRAY_AGG(DISTINCT chemical_class) FILTER (WHERE chemical_class IS NOT NULL) as chemical_classes,
                    ARRAY_AGG(DISTINCT action_class) FILTER (WHERE action_class IS NOT NULL) as action_classes,
                    ARRAY_AGG(DISTINCT subtitle_0) FILTER (WHERE subtitle_0 IS NOT NULL) as subtitles,
                    ARRAY_AGG(DISTINCT side_effect) FILTER (WHERE side_effect IS NOT NULL) as side_effects,
                    BOOL_OR(habit_forming) as has_habit_forming
                FROM medicines
                WHERE deleted_at IS NULL
                GROUP BY COALESCE(NULLIF(TRIM(therapeutic_class), ''), 'Other')
                ORDER BY therapeutic_class
            """))

            groups = result.fetchall()

            if not groups:
                logger.info("No medicines with therapeutic_class found")
                return {"therapeutic_classes_synced": 0, "total_medicines_processed": 0}

            total_medicines = 0
            documents = []

            for group in groups:
                therapeutic_class = group.therapeutic_class
                medicine_count = group.medicine_count
                total_medicines += medicine_count

                # Limit medicine names list for content (avoid extremely long texts)
                medicine_names = group.medicine_names or []
                displayed_names = medicine_names[:50]
                remaining = len(medicine_names) - len(displayed_names)

                chemical_classes = group.chemical_classes or []
                action_classes = group.action_classes or []
                subtitles = group.subtitles or []
                side_effects = group.side_effects or []

                # Limit side effects to avoid token overflow
                displayed_side_effects = side_effects[:20]

                # Build content text for embedding
                content = f"""Therapeutic Class: {therapeutic_class}
Total Medicines: {medicine_count}
Medicines: {', '.join(displayed_names)}{'... and ' + str(remaining) + ' more' if remaining > 0 else ''}
Chemical Classes: {', '.join(chemical_classes) if chemical_classes else 'N/A'}
Action Classes: {', '.join(action_classes) if action_classes else 'N/A'}
Forms/Subtitles: {', '.join(subtitles[:20]) if subtitles else 'N/A'}
Known Side Effects: {', '.join(displayed_side_effects) if displayed_side_effects else 'N/A'}
Contains Habit-Forming Medicines: {'Yes' if group.has_habit_forming else 'No'}""".strip()

                metadata = {
                    "type": "medicine_therapeutic_class",
                    "therapeutic_class": therapeutic_class,
                    "medicine_count": medicine_count,
                    "chemical_classes": chemical_classes[:10],
                    "action_classes": action_classes[:10],
                    "has_habit_forming": group.has_habit_forming or False,
                }

                documents.append({
                    "content": content,
                    "metadata": metadata,
                })

            # Batch process embeddings and insert
            batch_size = 50
            classes_synced = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                contents = [doc["content"] for doc in batch]

                # Generate embeddings in batch
                embeddings = await self.embeddings_service.embed_documents(contents)

                for doc, embedding in zip(batch, embeddings):
                    entry = KnowledgeBaseMedicines(
                        _id=uuid.uuid4(),
                        medicine_category=doc["content"],
                        embedding=embedding,
                        meta_data=doc["metadata"],
                        search_vector=func.to_tsvector('english', doc["content"]),
                    )
                    db.add(entry)
                    classes_synced += 1

                await db.flush()
                logger.info(
                    f"Synced medicine KB batch {i // batch_size + 1}: "
                    f"{len(batch)} therapeutic classes"
                )

            logger.info(
                f"Synced {classes_synced} therapeutic classes "
                f"({total_medicines} total medicines)"
            )

            return {
                "therapeutic_classes_synced": classes_synced,
                "total_medicines_processed": total_medicines,
            }

        except Exception as e:
            logger.error(f"Error syncing medicines to knowledge base: {e}")
            raise

    async def clear_medicine_knowledge_base(
        self,
        db: AsyncSession
    ) -> int:
        """
        Soft-delete all existing entries in knowledge_base_medicines.

        Args:
            db: Database session

        Returns:
            Number of entries cleared
        """
        try:
            result = await db.execute(
                select(KnowledgeBaseMedicines).where(
                    KnowledgeBaseMedicines.deleted_at.is_(None)
                )
            )
            entries = result.scalars().all()

            for entry in entries:
                entry.deleted_at = datetime.utcnow()

            await db.flush()
            logger.info(f"Cleared {len(entries)} medicine knowledge base entries")
            return len(entries)

        except Exception as e:
            logger.error(f"Error clearing medicine knowledge base: {e}")
            return 0


# Singleton instance
medicine_knowledge_base_service = MedicineKnowledgeBaseService()
