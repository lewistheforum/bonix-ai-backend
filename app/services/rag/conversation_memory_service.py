"""
Conversation Memory Service

Database-backed conversation memory using LangChain's ConversationBufferWindowMemory.
Stores conversation history in ai_conversations and ai_messages tables.
"""
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.config import settings
from app.models.ai_conversation import AIConversation
from app.models.ai_message import AIMessage as AIMessageModel
from app.utils.logger import logger


class ConversationMemoryService:
    """
    Database-backed Conversation Memory Service
    
    Provides conversation memory functionality backed by PostgreSQL.
    Uses LangChain's ConversationBufferWindowMemory pattern but stores
    messages in the database for persistence.
    
    Usage:
        memory_service = ConversationMemoryService()
        memory = await memory_service.get_memory(db, conversation_id)
        await memory_service.save_message(db, conversation_id, "user", "Hello")
    """
    
    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize the conversation memory service.
        
        Args:
            window_size: Number of recent messages to keep in context
        """
        self.window_size = window_size or settings.CONVERSATION_MEMORY_WINDOW
    
    async def get_or_create_conversation(
        self,
        db: AsyncSession,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> AIConversation:
        """
        Get existing conversation or create a new one.
        
        Args:
            db: Database session
            conversation_id: Optional existing conversation ID
            user_id: User ID to set as participant
            title: Optional conversation title
            
        Returns:
            AIConversation instance
        """
        if conversation_id:
            try:
                conv_uuid = uuid.UUID(conversation_id)
                result = await db.execute(
                    select(AIConversation).where(AIConversation._id == conv_uuid)
                )
                conversation = result.scalar_one_or_none()
                if conversation:
                    return conversation
            except (ValueError, Exception) as e:
                logger.warning(f"Invalid conversation ID: {conversation_id}, creating new")
        
        # Create new conversation
        conversation = AIConversation(
            _id=uuid.uuid4(),
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            participants=[str(user_id)] if user_id else [],
            metadata={}
        )
        db.add(conversation)
        await db.flush()
        
        logger.info(f"Created new conversation: {conversation._id}")
        return conversation
    
    async def get_history(
        self,
        db: AsyncSession,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        Get conversation history as LangChain messages.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            limit: Optional limit on messages (defaults to window_size)
            
        Returns:
            List of LangChain BaseMessage objects
        """
        limit = limit or self.window_size
        
        try:
            conv_uuid = uuid.UUID(conversation_id)
            result = await db.execute(
                select(AIMessageModel)
                .where(AIMessageModel.conversation_id == conv_uuid)
                .where(AIMessageModel.deleted_at.is_(None))
                .order_by(desc(AIMessageModel.created_at))
                .limit(limit)
            )
            db_messages = result.scalars().all()
            
            # Convert to LangChain messages (reverse to get chronological order)
            messages = []
            for msg in reversed(db_messages):
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def save_message(
        self,
        db: AsyncSession,
        conversation_id: str,
        role: str,
        content: str,
        sender_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AIMessageModel:
        """
        Save a message to the conversation.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            sender_id: Optional sender ID
            metadata: Optional message metadata
            
        Returns:
            Created AIMessage instance
        """
        try:
            conv_uuid = uuid.UUID(conversation_id)
            sender_uuid = uuid.UUID(sender_id) if sender_id else None
            
            message = AIMessageModel(
                _id=uuid.uuid4(),
                conversation_id=conv_uuid,
                sender_id=sender_uuid,
                role=role,
                content=content,
                metadata=metadata or {}
            )
            db.add(message)
            await db.flush()
            
            logger.info(f"Saved {role} message to conversation {conversation_id}")
            return message
            
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise
    
    async def get_memory(
        self,
        db: AsyncSession,
        conversation_id: str
    ) -> ConversationBufferWindowMemory:
        """
        Get a ConversationBufferWindowMemory populated with history.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            Populated ConversationBufferWindowMemory instance
        """
        memory = ConversationBufferWindowMemory(
            k=self.window_size,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Load history
        messages = await self.get_history(db, conversation_id)
        
        # Populate memory with message pairs
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                if isinstance(messages[i], HumanMessage) and isinstance(messages[i + 1], AIMessage):
                    memory.save_context(
                        {"input": messages[i].content},
                        {"output": messages[i + 1].content}
                    )
        
        return memory
    
    async def clear_conversation(
        self,
        db: AsyncSession,
        conversation_id: str
    ) -> bool:
        """
        Soft delete all messages in a conversation.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            True if successful
        """
        try:
            conv_uuid = uuid.UUID(conversation_id)
            
            # Soft delete messages
            result = await db.execute(
                select(AIMessageModel)
                .where(AIMessageModel.conversation_id == conv_uuid)
                .where(AIMessageModel.deleted_at.is_(None))
            )
            messages = result.scalars().all()
            
            for msg in messages:
                msg.deleted_at = datetime.utcnow()
            
            await db.flush()
            logger.info(f"Cleared {len(messages)} messages from conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False

    async def save_message_with_embedding(
        self,
        db: AsyncSession,
        conversation_id: str,
        role: str,
        content: str,
        sender_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AIMessageModel:
        """
        Save a message and generate its vector embedding.
        
        The embedding is stored in the ai_messages.embedding column so that
        future queries can perform cosine-similarity search within the
        conversation to find semantically relevant past messages.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            sender_id: Optional sender ID
            metadata: Optional message metadata
            
        Returns:
            Created AIMessage instance (with embedding populated)
        """
        from app.services.rag.embeddings_service import embeddings_service
        
        # Save the message first
        message = await self.save_message(
            db, conversation_id, role, content, sender_id, metadata
        )
        
        # Generate and store the embedding inside a savepoint
        # so that if embedding fails, it doesn't abort the outer transaction
        try:
            async with db.begin_nested():
                embedding = await embeddings_service.embed_text(content)
                message.embedding = embedding
                await db.flush()
            logger.info(f"Embedded {role} message {message._id} in conversation {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to embed message {message._id}: {e}. Message saved without embedding.")
        
        return message

    async def search_conversation_by_similarity(
        self,
        db: AsyncSession,
        conversation_id: str,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search past messages in a conversation by semantic similarity.
        
        Uses pgvector cosine distance (<=>) on the ai_messages.embedding
        column, scoped to the given conversation_id.
        Uses a savepoint so that pgvector errors don't abort the outer transaction.
        
        Args:
            db: Database session
            conversation_id: Conversation ID to search within
            query: Query text to find similar messages
            k: Number of similar messages to return
            
        Returns:
            List of dicts with keys: role, content, score, created_at
        """
        from app.services.rag.embeddings_service import embeddings_service
        from sqlalchemy import text
        
        try:
            # Generate embedding for the query
            query_embedding = await embeddings_service.embed_text(query)
            
            conv_uuid = uuid.UUID(conversation_id)
            
            # Wrap in a savepoint so that if the pgvector query fails
            # (e.g. no embeddings exist yet), it doesn't poison the transaction
            async with db.begin_nested():
                # Use raw SQL for pgvector cosine distance operator
                sql = text("""
                    SELECT _id, role, content, created_at,
                           1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                    FROM ai_messages
                    WHERE conversation_id = :conv_id
                      AND deleted_at IS NULL
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:query_vec AS vector)
                    LIMIT :k
                """)
                
                result = await db.execute(sql, {
                    "query_vec": str(query_embedding),
                    "conv_id": str(conv_uuid),
                    "k": k
                })
                rows = result.fetchall()
            
            similar_messages = []
            for row in rows:
                similar_messages.append({
                    "id": str(row._mapping["_id"]),
                    "role": row._mapping["role"],
                    "content": row._mapping["content"],
                    "score": float(row._mapping["similarity"]),
                    "created_at": row._mapping["created_at"],
                })
            
            logger.info(
                f"Found {len(similar_messages)} similar messages in "
                f"conversation {conversation_id} for query '{query[:50]}...'"
            )
            return similar_messages
            
        except Exception as e:
            logger.error(f"Error searching conversation by similarity: {e}")
            return []


# Singleton instance
conversation_memory_service = ConversationMemoryService()
