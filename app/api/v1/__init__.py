# API v1 package initialization
from app.api.v1.chat_bot import chatbot as chat_bot_module
from app.api.v1.recommendation import recommendation_clinic as recommendation_clinic_module
from app.api.v1.bad_word import bad_word_detection as bad_word_detection_module
from app.api.v1.label_feedback import label_feedback as label_feedback_module

# Re-export modules for easy access
chat_bot = chat_bot_module
recommendation_clinic = recommendation_clinic_module
bad_word_detection = bad_word_detection_module
label_feedback = label_feedback_module


