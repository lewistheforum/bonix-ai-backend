"""
Controller/Router for Bad Word Detection API
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List, Dict
from deep_translator import GoogleTranslator

import sys
import importlib.util
import os
from app.common.api_response import ApiResponse
from app.common.message.status_code import StatusCode
from app.common.message.success_message import SuccessMessage
from app.common.message.error_message import ErrorMessage

# Import DTO from hyphenated directory
dto_path = os.path.join(os.path.dirname(__file__), "../../../dto/bad_word/bad_word_dto.py")
dto_path = os.path.abspath(dto_path)
spec = importlib.util.spec_from_file_location("bad_word_dto", dto_path)
bad_word_dto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bad_word_dto)

BadWordDetectionRequest = bad_word_dto.BadWordDetectionRequest
BadWordDetectionResponse = bad_word_dto.BadWordDetectionResponse
BatchBadWordDetectionRequest = bad_word_dto.BatchBadWordDetectionRequest
BatchBadWordDetectionResponse = bad_word_dto.BatchBadWordDetectionResponse
HateSpeechResult = bad_word_dto.HateSpeechResult
ToxicSpeechResult = bad_word_dto.ToxicSpeechResult
HateSpansResult = bad_word_dto.HateSpansResult

# Import service from hyphenated directory
service_path = os.path.join(os.path.dirname(__file__), "../../../services/bad_word/bad_word_detection_service.py")
service_path = os.path.abspath(service_path)
spec = importlib.util.spec_from_file_location("bad_word_detection_service", service_path)
bad_word_detection_service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bad_word_detection_service)
bad_word_detector = bad_word_detection_service.bad_word_detector

router = APIRouter()


def _preprocess_text_for_analysis(text: str) -> str:
    """
    Pre-filter for input language.
    Translate any non-Vietnamese text to Vietnamese for analysis.
    Uses GoogleTranslator with source='auto' to auto-detect input language.
    
    Returns:
        Translated text in Vietnamese (or original if already Vietnamese/translation fails)
    """
    try:
        # Skip translation for very short text or empty text
        if not text or len(text.strip()) < 2:
            return text
        
        # Use GoogleTranslator with auto-detection to translate to Vietnamese
        # If text is already Vietnamese, the translation will return similar text
        translator = GoogleTranslator(source='auto', target='vi')
        translated_text = translator.translate(text)
        
        return translated_text if translated_text else text
    except Exception as e:
        print(f"Translation error: {e}")
        # If translation fails, use original text
        return text


@router.post("/detect", response_model=ApiResponse[BadWordDetectionResponse])
async def detect_bad_words(request: BadWordDetectionRequest):
    """
    Detect bad words in a single text
    
    - **text**: Text to analyze for bad words
    - **detection_type**: Type of detection ('hate_speech', 'toxic', 'hate_spans', or 'all')
    """
    try:
        # Pre-filter: detect language and translate if needed
        text_for_analysis = _preprocess_text_for_analysis(request.text)
        
        result = await _analyze_text(text_for_analysis, request.detection_type, original_text=request.text)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=result)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/detect/batch", response_model=ApiResponse[BatchBadWordDetectionResponse])
async def detect_bad_words_batch(request: BatchBadWordDetectionRequest):
    """
    Detect bad words in multiple texts
    
    - **texts**: List of texts to analyze
    - **detection_type**: Type of detection ('hate_speech', 'toxic', 'hate_spans', or 'all')
    """
    try:
        results = []
        total_toxic = 0
        
        for text in request.texts:
            # Pre-filter: detect language and translate if needed
            text_for_analysis = _preprocess_text_for_analysis(text)
            result = await _analyze_text(text_for_analysis, request.detection_type, original_text=text)
            results.append(result)
            if result.is_toxic:
                total_toxic += 1
        
        data = BatchBadWordDetectionResponse(
            results=results,
            total_analyzed=len(request.texts),
            total_toxic=total_toxic,
            analyzed_at=datetime.now()
        )
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/detect/hate-speech", response_model=ApiResponse[HateSpeechResult])
async def detect_hate_speech(request: BadWordDetectionRequest):
    """
    Detect hate speech only
    
    - **text**: Text to analyze for hate speech
    """
    try:
        result = bad_word_detector.detect_hate_speech(request.text)
        data = HateSpeechResult(**result)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/detect/toxic", response_model=ApiResponse[ToxicSpeechResult])
async def detect_toxic_speech(request: BadWordDetectionRequest):
    """
    Detect toxic speech only
    
    - **text**: Text to analyze for toxic content
    """
    try:
        result = bad_word_detector.detect_toxic_speech(request.text)
        data = ToxicSpeechResult(**result)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.post("/detect/hate-spans", response_model=ApiResponse[HateSpansResult])
async def detect_hate_spans(request: BadWordDetectionRequest):
    """
    Detect hate spans (find exact bad words and their positions)
    
    - **text**: Text to analyze for hate spans
    """
    try:
        result = bad_word_detector.detect_hate_spans(request.text)
        data = HateSpansResult(**result)
        return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)
    except Exception as e:
        raise HTTPException(status_code=StatusCode.INTERNAL_ERROR, detail=str(e))


@router.get("/health", response_model=ApiResponse[Dict[str, str]])
async def health_check():
    """
    Health check endpoint for bad word detection service
    """
    data = {"status": "healthy", "service": "bad-word-detection"}
    return ApiResponse(statusCode=StatusCode.SUCCESS, message=SuccessMessage.INDEX, data=data)


async def _analyze_text(text: str, detection_type: str, original_text: str = None) -> BadWordDetectionResponse:
    """
    Internal helper to analyze text based on detection type
    
    Args:
        text: Text to analyze (may be translated to Vietnamese)
        detection_type: Type of detection
        original_text: Original text before translation (for response)
    """
    hate_speech_result = None
    # toxic_speech_result = None
    hate_spans_result = None
    is_toxic = False
    
    if detection_type in ["all", "hate_speech"]:
        result = bad_word_detector.detect_hate_speech(text)
        hate_speech_result = HateSpeechResult(**result)
        if result["is_toxic"]:
            is_toxic = True
    
    # if detection_type in ["all", "toxic"]:
    #     result = bad_word_detector.detect_toxic_speech(text)
    #     toxic_speech_result = ToxicSpeechResult(**result)
    #     if result["is_toxic"]:
    #         is_toxic = True
    
    if detection_type in ["all", "hate_spans"]:
        result = bad_word_detector.detect_hate_spans(text)
        hate_spans_result = HateSpansResult(**result)
        if result["bad_words"]:
            is_toxic = True
    
    # Use original text in response if provided, otherwise use the analyzed text
    response_text = original_text if original_text is not None else text
    
    return BadWordDetectionResponse(
        text=response_text,
        is_toxic=is_toxic,
        hate_speech=hate_speech_result,
        # toxic_speech=toxic_speech_result,
        hate_spans=hate_spans_result,
        analyzed_at=datetime.now()
    )
