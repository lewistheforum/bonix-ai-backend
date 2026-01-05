"""
Service for Chatbot business logic
"""
from datetime import datetime
from typing import List, Optional
import uuid
from app.dto.chat_bot.chatbot_dto import (
    ChatbotRequest,
    ChatbotResponse,
    ChatMessage,
    ConversationHistoryResponse
)


class ChatbotService:
    """Service class for chatbot operations"""
    
    def __init__(self):
        # In a real application, this would connect to an AI model (OpenAI, etc.)
        self.conversations = {}  # Store conversation history
    
    async def chat(self, request: ChatbotRequest) -> ChatbotResponse:
        """
        Process chatbot message and generate response
        
        Args:
            request: Chatbot request DTO
            
        Returns:
            Chatbot response DTO
        """
        # Get or create conversation ID
        conversation_id = request.conversation_id or f"CONV{uuid.uuid4().hex[:8].upper()}"
        
        # Initialize conversation if new
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now()
        )
        self.conversations[conversation_id].append(user_message)
        
        # Generate response (simplified - in real app, call AI model)
        response_text = self._generate_response(request.message, request.context)
        
        # Add assistant response to history
        assistant_message = ChatMessage(
            role="assistant",
            content=response_text,
            timestamp=datetime.now()
        )
        self.conversations[conversation_id].append(assistant_message)
        
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(request.message)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(request.message)
        
        return ChatbotResponse(
            response=response_text,
            conversation_id=conversation_id,
            suggested_actions=suggested_actions,
            confidence_score=confidence_score,
            generated_at=datetime.now()
        )
    
    def _generate_response(self, message: str, context: Optional[dict] = None) -> str:
        """
        Generate chatbot response (simplified - replace with actual AI model)
        
        Args:
            message: User message
            context: Additional context
            
        Returns:
            Generated response
        """
        message_lower = message.lower()
        
        # Simple rule-based responses (replace with AI model)
        if any(word in message_lower for word in ["headache", "head", "pain"]):
            return "I understand you're experiencing a headache. This could be due to various reasons such as stress, dehydration, or tension. I recommend resting, staying hydrated, and if the pain persists or is severe, please consult with a healthcare professional."
        
        if any(word in message_lower for word in ["fever", "temperature", "hot"]):
            return "A fever can indicate your body is fighting an infection. Please monitor your temperature regularly. If it's above 38.5°C (101.3°F) or persists for more than 3 days, I recommend seeing a doctor. Stay hydrated and rest."
        
        if any(word in message_lower for word in ["cough", "coughing"]):
            return "A cough can be caused by various factors including cold, flu, or allergies. If it's persistent or accompanied by other symptoms like fever or difficulty breathing, please consult a healthcare provider. In the meantime, stay hydrated and consider using a humidifier."
        
        if any(word in message_lower for word in ["appointment", "schedule", "book"]):
            return "I can help you find a suitable clinic and schedule an appointment. Would you like me to recommend clinics based on your location or specific medical needs?"
        
        # Default response
        return "Thank you for your message. I'm here to help with your health-related questions. Could you provide more details about your symptoms or concerns? If you need immediate medical attention, please contact emergency services."
    
    def _generate_suggested_actions(self, message: str) -> List[str]:
        """
        Generate suggested actions based on message
        
        Args:
            message: User message
            
        Returns:
            List of suggested actions
        """
        message_lower = message.lower()
        actions = []
        
        if any(word in message_lower for word in ["symptom", "pain", "fever", "cough", "headache"]):
            actions.append("Schedule appointment")
            actions.append("Find nearby clinic")
        
        if any(word in message_lower for word in ["medication", "medicine", "drug"]):
            actions.append("Get medication advice")
            actions.append("Check drug interactions")
        
        if any(word in message_lower for word in ["emergency", "urgent", "severe"]):
            actions.append("Contact emergency services")
            actions.append("Find emergency clinic")
        
        if not actions:
            actions = ["Schedule appointment", "Find nearby clinic", "Get health advice"]
        
        return actions[:3]  # Return max 3 actions
    
    def _calculate_confidence(self, message: str) -> float:
        """
        Calculate confidence score for response
        
        Args:
            message: User message
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Simplified confidence calculation
        # In real app, this would be based on AI model confidence
        medical_keywords = [
            "symptom", "pain", "fever", "cough", "headache", "medicine",
            "doctor", "clinic", "appointment", "health"
        ]
        
        message_lower = message.lower()
        keyword_count = sum(1 for keyword in medical_keywords if keyword in message_lower)
        
        # Base confidence increases with relevant keywords
        confidence = 0.5 + (keyword_count * 0.1)
        return min(confidence, 0.95)
    
    async def get_conversation_history(self, conversation_id: str) -> ConversationHistoryResponse:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation history response DTO
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} not found")
        
        messages = self.conversations[conversation_id]
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            total_messages=len(messages)
        )
    
    async def delete_conversation(self, conversation_id: str) -> dict:
        """
        Delete conversation history
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Deletion confirmation
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} not found")
        
        del self.conversations[conversation_id]
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}


# Create service instance
chatbot_service = ChatbotService()

