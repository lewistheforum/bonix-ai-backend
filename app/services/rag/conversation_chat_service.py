"""
Conversation Chat Service

Orchestrates the conversation chat flow with:
- Conversation-scoped semantic retrieval (past messages)
- Knowledge base hybrid retrieval
- Auto-embedding of user and assistant messages
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.conversation_memory_service import conversation_memory_service
from app.services.rag.rag_chain import rag_chatbot
from app.utils.logger import logger


class ConversationChatService:
    """
    Conversation Chat Service
    
    Orchestrates a chat flow that combines:
    1. Conversation-scoped semantic search (past embedded messages)
    2. Knowledge base context (hybrid retriever)
    3. Windowed chat history (recent messages)
    
    Both user messages and assistant responses are embedded on save,
    so they become searchable for future turns.
    """

    async def chat(
        self,
        db: AsyncSession,
        conversation_id: str,
        user_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a chatbot response.
        
        Flow:
        1. Get or create conversation
        2. Save user message with embedding
        3. Search conversation for semantically similar past messages
        4. Retrieve knowledge base context
        5. Build combined context (conversation + KB)
        6. Invoke RAG agent with combined context
        7. Save assistant response with embedding
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: User UUID
            message: User's chat message
            
        Returns:
            Dict with response, conversation_id, context flags, sources, timestamp
        """
        try:
            # 1. Get or create conversation
            conversation = await conversation_memory_service.get_or_create_conversation(
                db, conversation_id, user_id
            )
            conv_id = str(conversation._id)

            # 2. Save user message with embedding
            await conversation_memory_service.save_message_with_embedding(
                db, conv_id, "user", message, sender_id=user_id
            )

            # 3. Search conversation history by semantic similarity
            conversation_context = ""
            conversation_context_used = False
            try:
                similar_messages = await conversation_memory_service.search_conversation_by_similarity(
                    db, conv_id, message, k=5
                )
                if similar_messages:
                    # Format conversation context for the LLM
                    conv_parts = []
                    for msg in similar_messages:
                        role_label = "User" if msg["role"] == "user" else "Assistant"
                        conv_parts.append(
                            f"[{role_label}] (relevance: {msg['score']:.3f}): {msg['content'][:500]}"
                        )
                    conversation_context = "\n".join(conv_parts)
                    conversation_context_used = True
            except Exception as e:
                logger.warning(f"Conversation similarity search failed: {e}")

            # 4 & 5. Build combined context and invoke RAG agent
            # We prepend conversation context to the knowledge base context
            # so the LLM sees both past conversation topics and KB data.
            combined_prefix = ""
            if conversation_context:
                combined_prefix = (
                    "=== Relevant Past Conversation Messages ===\n"
                    f"{conversation_context}\n\n"
                    "=== Knowledge Base Context ===\n"
                )

            # 6. Use the existing rag_chatbot.chat() which handles:
            #    - KB retrieval, chat history, agent invocation
            #    We pass combined_prefix as manual_context prefix
            #    But to keep things clean, we'll call the internal methods directly
            from app.services.rag.hybrid_retriever import hybrid_retriever
            from app.services.rag.booking_tool import set_db_session
            from app.services.rag.schedule_tool import set_schedule_db_session

            # Set DB sessions for tools
            set_db_session(db)
            set_schedule_db_session(db)

            # Classify query and get KB context
            classification = await rag_chatbot._classify_query(message)
            categories = classification.get("categories", [])
            kb_context = await rag_chatbot._get_context(db, message, classification=classification)

            # Combine contexts
            if combined_prefix:
                full_context = combined_prefix + kb_context
            else:
                full_context = kb_context

            # Get recent chat history
            chat_history = await conversation_memory_service.get_history(db, conv_id)

            # 7. Run the agent
            result = await rag_chatbot._run_agent(message, chat_history, full_context, categories)
            response_text = result.get("output", "I'm sorry, I couldn't generate a response.")

            # 8. Save assistant response with embedding
            await conversation_memory_service.save_message_with_embedding(
                db, conv_id, "assistant", response_text
            )

            # Commit the transaction
            await db.commit()

            return {
                "response": response_text,
                "conversation_id": conv_id,
                "context_used": len(kb_context) > 50,
                "conversation_context_used": conversation_context_used,
                "sources": [],
                "timestamp": datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error in conversation chat: {e}")
            await db.rollback()
            raise


# Singleton instance
conversation_chat_service = ConversationChatService()
