"""
RAG Chain

Main RAG chatbot implementation combining hybrid retrieval,
conversation memory, and OpenAI LLM with tool support.
"""
from typing import List, Optional, Dict, Any, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.services.rag.hybrid_retriever import hybrid_retriever, RetrievalResult
from app.services.rag.conversation_memory_service import conversation_memory_service
from app.services.rag.booking_tool import booking_tool, set_db_session
from app.services.rag.schedule_tool import (
    clinic_schedule_tool, 
    doctor_schedule_tool, 
    set_schedule_db_session
)
from app.services.rag.schema_context import SCHEMA_DESCRIPTION
from app.utils.logger import logger


# System prompt for the RAG chatbot
SYSTEM_PROMPT = """You are a helpful medical assistant chatbot for the Bonix clinic platform.

Your specific knowledge base consists strictly of the following ingested data types:
- Clinic Services (details, prices, clinics)
- Doctor Profiles (background, specialties, clinics)
- Clinic Information (branches, addresses, contacts)
- Staff Information (roles, contacts)
- Blogs/Health Articles
- Patient Feedbacks
- User Information

Your Primary Roles:
1. Information Provider: Answer questions ONLY based on the provided context from the data above.
2. Appointment Coordinator: Help patients schedule appointments.
3. Strict Guardrail: Do NOT answer any questions that are not relevant to the provided medical/clinic data.

Strict Relevance Rule:
- **Priority**: Your highest priority is to ANSWER the user's question using the provided context.
- **Extraction**: You must extract information from ANY part of the context (Clinic Info, Feedback, Bio, etc.).
- **Missing Information**: Only say "I could not find specific information" if the context is completely empty or completely unrelated.
- **Irrelevant Topics**: Refuse only if the question is widely off-topic (e.g. cooking, sports).

Appointment Booking Instructions:
- Gather these mandatory details before using `booking_tool`:
  1. Clinic Details (Name/Branch)
  2. Doctor Name
  3. Service Name
  4. Date & Time
  5. Patient Name & Contact Info
- If details are missing, ask for them one by one.

Schedule Lookup Instructions:
- For `findClinicSchedule`, you MUST have:
  1. Clinic Name (e.g., "Bonix", "District 1 Branch")
  2. Date (YYYY-MM-DD or relative like "today", "tomorrow")
- For `findDoctorSchedule`, you MUST have:
  1. Doctor Name
  2. Date
- If any of these are missing, ask the user specifically for the missing item before calling the tool.

System Database Schema:
{schema}

Context from knowledge base:
{context}

NOTE: Response answer only html tags format.
.
"""

# Router Prompt
ROUTER_PROMPT = """You are a query classifier for a medical chatbot.
Classify the user's query into one or more of the following categories based on what information they are looking for:

1. 'clinic_service': Services, treatments, prices, packages, procedures (e.g., "braces price", "implant cost").
2. 'doctor_profile': Doctor information, bios, specialties, qualifications, experience (e.g., "who is Dr. Smith", "best dentist").
3. 'clinic_info': Clinic locations, addresses, branches, general descriptions, facilities (e.g., "where is the clinic", "clinic in Hanoi").
4. 'staff_info': Nursing staff, receptionists, support staff (e.g., "nurse contact", "staff list").
5. 'blog': Health articles, tips, medical news, general knowledge (e.g., "how to brush teeth", "root canal definition").
6. 'find_doctor_schedule': Doctor availability, working hours, shifts, appointments (e.g., "is Dr. Smith free today", "schedule for next week").
7. 'find_clinic_schedule': Clinic working hours, schedules, shifts, appointments (e.g., "open hours for Bonix", "schedule of Hai Phong clinic").
8. 'feedback': Reviews, ratings, patient feedback (e.g., "is this doctor good", "clinic reviews").
9. 'user_info': User's own information, profile, history (e.g., "my profile", "my details").

Return ONLY a JSON list of the matching category codes. If unsure or if it's a general greeting, return ["general"].
Example: ["clinic_service", "find_clinic_schedule"]
"""

class RAGChatbot:
    """
    RAG Chatbot with Hybrid Retrieval and Tool Support
    
    Combines:
    - Hybrid retrieval (vector + keyword search)
    - Conversation memory (database-backed)
    - OpenAI LLM (gpt-4o or gpt-3.5-turbo)
    - Tool support (appointment booking)
    
    Usage:
        chatbot = RAGChatbot()
        response = await chatbot.chat(db, "Find me a dental clinic", conversation_id)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            model: Model name (default from settings)
            temperature: LLM temperature (0-1)
        """
        self.model = model or settings.OPENAI_CHAT_MODEL
        self.temperature = temperature
        self._llm = None
        self._agent_executor = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Lazy initialization of LLM"""
        if self._llm is None:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
            
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            logger.info(f"Initialized ChatOpenAI with model: {self.model}")
        return self._llm
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools."""
        # Define the prompt template
        from langchain_core.prompts import SystemMessagePromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Inject schema description safely
        prompt = prompt.partial(schema=SCHEMA_DESCRIPTION)
        
        # Create the agent with tools
        tools = [booking_tool, clinic_schedule_tool, doctor_schedule_tool]
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        return agent_executor
    
    @property
    def agent_executor(self) -> AgentExecutor:
        """Lazy initialization of agent executor"""
        if self._agent_executor is None:
            self._agent_executor = self._create_agent()
        return self._agent_executor
    
    
    def _detect_intents(self, query: str) -> Optional[Dict[str, Any]]:
        """Deprecated: Use _classify_query instead."""
        return None

    async def _classify_query(self, query: str) -> List[str]:
        """
        Classify the user's query into knowledge base categories using LLM.
        
        Args:
            query: User's query string
            
        Returns:
            List of detected doc_types
        """
        try:
            messages = [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=query)
            ]
            
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()
            
            # Clean up potential markdown formatting (```json ... ```)
            if "```" in content:
                content = content.replace("```json", "").replace("```", "")
            
            import json
            categories = json.loads(content)
            
            # Map categories to actual doc_types in database
            # Database values: clinic_services, doctor_infor, clinic_infor, staff_infor, 
            # blog_infor, schedule_infor, feedback, user_infor
            
            mapped_categories = []
            for cat in categories:
                if cat == "clinic_service":
                    mapped_categories.append("clinic_services")
                elif cat == "doctor_profile":
                    mapped_categories.append("doctor_profile")
                elif cat == "clinic_info":
                    mapped_categories.append("clinic_info")
                elif cat == "staff_info":
                    mapped_categories.append("staff_info")
                elif cat == "blog":
                    mapped_categories.append("blog_info")
                elif cat == "find_doctor_schedule":
                    mapped_categories.append("schedule_info")
                elif cat == "find_clinic_schedule":
                    mapped_categories.append("clinic_working_hours")
                elif cat == "feedback":
                    mapped_categories.append("feedback")
                elif cat == "user_info":
                    mapped_categories.append("user_info")
                elif cat == "general":
                    pass # No filter
            
            logger.info(f"Classified query '{query}' -> {mapped_categories}")
            return mapped_categories
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return []

    async def _get_context(
        self,
        db: AsyncSession,
        query: str,
        k: int = 5,
        categories: Optional[List[str]] = None
    ) -> str:
        """
        Retrieve relevant context from knowledge base.
        
        Args:
            db: Database session
            query: User query
            k: Number of documents to retrieve
            categories: Optional pre-classified categories
            
        Returns:
            Formatted context string
        """
        try:
            # Use a nested transaction (savepoint) so that if retrieval fails (e.g. pgvector error),
            # it doesn't abort the main transaction.
            async with db.begin_nested():
                # CLASSIFY QUERY AND APPLY FILTERS
                doc_types = categories if categories is not None else await self._classify_query(query)
                
                filter_metadata = None
                if doc_types:
                    filter_metadata = {"type": doc_types}
                
                results = await hybrid_retriever.retrieve(db, query, k, filter_metadata)
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            # Format context from retrieved documents
            context_parts = []
            for i, result in enumerate(results, 1):
                doc = result.document
                source = result.source
                score = result.score
                
                metadata_str = ""
                if doc.meta_data:
                    metadata_items = [f"{k}: {v}" for k, v in doc.meta_data.items() if v]
                    metadata_str = f" [{', '.join(metadata_items)}]" if metadata_items else ""
                
                context_parts.append(
                    f"{i}. {doc.content[:500]}...{metadata_str} (relevance: {score:.3f}, source: {source})"
                )
            
            
            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved context for query '{query}':\n{context}...")
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Unable to retrieve context from knowledge base."
    
    async def chat(
        self,
        db: AsyncSession,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        manual_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response.
        
        Args:
            db: Database session
            query: User's message
            conversation_id: Optional existing conversation ID
            user_id: Optional user ID
            manual_context: Optional pre-retrieved context to skip internal retrieval
            
        Returns:
            Dictionary with response, conversation_id, and metadata
        """
        try:
            # Set database session for tools
            set_db_session(db)
            set_schedule_db_session(db)
            
            # Get or create conversation
            conversation = await conversation_memory_service.get_or_create_conversation(
                db, conversation_id, user_id
            )
            conv_id = str(conversation._id)
            
            # Save user message
            await conversation_memory_service.save_message(
                db, conv_id, "user", query, user_id
            )
            
            # Get conversation history
            chat_history = await conversation_memory_service.get_history(
                db, conv_id
            )
            
            # Classify query if context not manually provided
            categories = None
            if not manual_context:
                categories = await self._classify_query(query)
            
            # Retrieve relevant context (or use provided)
            context = manual_context if manual_context else await self._get_context(db, query, categories=categories)
            
            # Run the agent with intent info
            result = await self._run_agent(query, chat_history, context, categories)
            
            # Extract response
            response_text = result.get("output", "I'm sorry, I couldn't generate a response.")
            
            # Save assistant response
            await conversation_memory_service.save_message(
                db, conv_id, "assistant", response_text
            )
            
            # Commit the transaction
            await db.commit()
            
            return {
                "response": response_text,
                "conversation_id": conv_id,
                "context_used": len(context) > 50,
                "sources": [],  # Could extract from context
                "timestamp": None  # Will be set by caller
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await db.rollback()
            raise
    
    async def _run_agent(
        self,
        query: str,
        chat_history: List,
        context: str,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the agent with the given inputs.
        
        Args:
            query: User query
            chat_history: Previous messages
            context: Retrieved context
            categories: Detected intent categories
            
        Returns:
            Agent execution result
        """
        import asyncio
        
        # Modify input to prompt tool usage if specific intents are detected
        agent_input = f"{query}"
        
        if categories:
            if "clinic_working_hours" in categories:
                agent_input += "\n\n(IMPORTANT: User is asking for clinic schedule. Use 'findClinicSchedule' tool. CRITICAL: Do NOT infer the clinic name from the retrieved context. If the user did not explicitly state the clinic name, you MUST ask for it. Do not call the tool with a guessed name.)"
            elif "schedule_info" in categories:
                 agent_input += "\n\n(IMPORTANT: User is asking for doctor schedule. Use 'findDoctorSchedule' tool. CRITICAL: Do NOT infer the doctor name from the retrieved context. If the user did not explicitly state the doctor name, you MUST ask for it. Do not call the tool with a guessed name.)"
        
        try:
            # Run agent in executor (it's synchronous internally)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.agent_executor.invoke({
                    "input": agent_input,
                    "chat_history": chat_history,
                    "context": context
                })
            )
            return result
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            # Fallback to direct LLM call without tools
            return await self._fallback_response(query, chat_history, context)
    
    async def _fallback_response(
        self,
        query: str,
        chat_history: List,
        context: str
    ) -> Dict[str, Any]:
        """
        Generate a fallback response without tools.
        
        Args:
            query: User query
            chat_history: Previous messages
            context: Retrieved context
            
        Returns:
            Response dictionary
        """
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(context=context, schema=SCHEMA_DESCRIPTION)),
                *chat_history,
                HumanMessage(content=query)
            ]
            
            response = await self.llm.ainvoke(messages)
            return {"output": response.content}
            
        except Exception as e:
            logger.error(f"Fallback response error: {e}")
            return {"output": "I apologize, but I'm having trouble processing your request. Please try again."}
    
    async def simple_chat(
        self,
        db: AsyncSession,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        Simple chat without conversation tracking (for testing).
        
        Args:
            db: Database session
            query: User query
            context: Optional pre-fetched context
            
        Returns:
            Response string
        """
        if context is None:
            context = await self._get_context(db, query)
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=query)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content


# Singleton instance
rag_chatbot = RAGChatbot()
