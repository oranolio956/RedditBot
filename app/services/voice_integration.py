"""
Voice Processing Integration Service

Orchestrates the complete voice message processing pipeline:
- Voice message download and conversion
- Speech-to-text transcription
- AI response generation
- Text-to-speech for voice replies
- Performance monitoring and error handling
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import structlog
from contextlib import asynccontextmanager

from .voice_processor import get_voice_processor, VoiceProcessingError
from .whisper_client import get_whisper_client, WhisperError, WhisperResponse
from .tts_service import get_tts_service, TTSError, TTSResponse
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class VoiceIntegrationMetrics:
    """Track voice processing metrics and performance."""
    
    def __init__(self):
        self.stats = {
            'total_voice_messages': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'voice_responses_generated': 0,
            'avg_processing_time': 0.0,
            'avg_transcription_time': 0.0,
            'avg_tts_time': 0.0,
            'languages_processed': {},
            'error_types': {}
        }
    
    def record_voice_processing(
        self,
        success: bool,
        processing_time: float,
        transcription_time: Optional[float] = None,
        tts_time: Optional[float] = None,
        language: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """Record voice processing metrics."""
        self.stats['total_voice_messages'] += 1
        
        if success:
            self.stats['successful_transcriptions'] += 1
        else:
            self.stats['failed_transcriptions'] += 1
            if error_type:
                self.stats['error_types'][error_type] = (
                    self.stats['error_types'].get(error_type, 0) + 1
                )
        
        # Update average processing time
        total_successful = self.stats['successful_transcriptions']
        if total_successful > 0:
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (total_successful - 1) + processing_time)
                / total_successful
            )
        
        if transcription_time and total_successful > 0:
            self.stats['avg_transcription_time'] = (
                (self.stats['avg_transcription_time'] * (total_successful - 1) + transcription_time)
                / total_successful
            )
        
        if tts_time:
            self.stats['voice_responses_generated'] += 1
            responses_generated = self.stats['voice_responses_generated']
            self.stats['avg_tts_time'] = (
                (self.stats['avg_tts_time'] * (responses_generated - 1) + tts_time)
                / responses_generated
            )
        
        if language:
            self.stats['languages_processed'][language] = (
                self.stats['languages_processed'].get(language, 0) + 1
            )


class VoiceIntegrationService:
    """
    Complete voice processing integration service.
    
    Orchestrates the entire pipeline from voice message to intelligent response,
    with comprehensive error handling, performance monitoring, and fallback options.
    """
    
    def __init__(
        self,
        redis_client=None,
        enable_caching: bool = True,
        target_processing_time: float = 2.0
    ):
        """
        Initialize voice integration service.
        
        Args:
            redis_client: Redis client for caching
            enable_caching: Whether to enable caching
            target_processing_time: Target processing time in seconds
        """
        self.redis_client = redis_client
        self.enable_caching = enable_caching
        self.target_processing_time = target_processing_time
        self.settings = get_settings()
        
        # Initialize service components
        self.voice_processor = get_voice_processor()
        
        # Metrics tracking
        self.metrics = VoiceIntegrationMetrics()
        
        # Active processing sessions (for cancellation/monitoring)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "Voice integration service initialized",
            caching_enabled=enable_caching,
            target_processing_time=target_processing_time
        )
    
    @asynccontextmanager
    async def processing_session(self, session_id: str, user_id: int):
        """Context manager for tracking processing sessions."""
        session_info = {
            'user_id': user_id,
            'start_time': time.time(),
            'status': 'initializing'
        }
        
        self.active_sessions[session_id] = session_info
        
        try:
            yield session_info
        finally:
            # Clean up session
            self.active_sessions.pop(session_id, None)
    
    async def process_voice_message_complete(
        self,
        bot,
        voice_message,
        user_id: int,
        chat_id: int,
        user_preferences: Optional[Dict[str, Any]] = None,
        generate_voice_response: bool = True
    ) -> Dict[str, Any]:
        """
        Complete voice message processing pipeline.
        
        Args:
            bot: Telegram bot instance
            voice_message: Telegram voice message object
            user_id: User ID
            chat_id: Chat ID
            user_preferences: User preferences for processing
            generate_voice_response: Whether to generate voice response
            
        Returns:
            Dict with processing results and metadata
        """
        session_id = f"voice_{user_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        async with self.processing_session(session_id, user_id) as session:
            try:
                session['status'] = 'downloading'
                
                # Step 1: Process voice message (download + convert)
                logger.info("Starting voice message processing", session_id=session_id, user_id=user_id)
                
                mp3_path, voice_metadata = await self.voice_processor.process_voice_message(
                    bot=bot,
                    file_id=voice_message.file_id,
                    user_id=user_id,
                    optimize_for_speech=True
                )
                
                session['status'] = 'transcribing'
                session['voice_file'] = str(mp3_path)
                
                # Step 2: Transcribe speech to text
                whisper_client = await get_whisper_client(
                    redis_client=self.redis_client
                )
                
                # Determine language preference
                language_preference = None
                if user_preferences:
                    language_preference = user_preferences.get('language')
                
                transcription_start = time.time()
                transcription = await whisper_client.transcribe_audio(
                    audio_file_path=mp3_path,
                    language=language_preference,
                    use_cache=self.enable_caching
                )
                transcription_time = time.time() - transcription_start
                
                session['status'] = 'transcribed'
                session['transcribed_text'] = transcription.text
                session['detected_language'] = transcription.language
                
                logger.info(
                    "Voice transcription completed",
                    session_id=session_id,
                    text_length=len(transcription.text),
                    language=transcription.language,
                    confidence=transcription.confidence,
                    transcription_time=transcription_time
                )\n                \n                result = {\n                    'success': True,\n                    'transcription': {\n                        'text': transcription.text,\n                        'language': transcription.language,\n                        'confidence': transcription.confidence,\n                        'duration': transcription.duration,\n                        'processing_time': transcription.processing_time,\n                        'cached': transcription.cached\n                    },\n                    'voice_metadata': voice_metadata,\n                    'session_id': session_id,\n                    'mp3_file_path': str(mp3_path),\n                    'total_processing_time': time.time() - start_time\n                }\n                \n                # Step 3: Generate voice response if requested\n                if generate_voice_response and self._should_generate_voice_response(\n                    transcription, user_preferences, chat_id\n                ):\n                    session['status'] = 'generating_voice_response'\n                    \n                    try:\n                        voice_response_result = await self.generate_voice_response(\n                            text=transcription.text,\n                            response_language=transcription.language,\n                            user_preferences=user_preferences\n                        )\n                        \n                        if voice_response_result['success']:\n                            result['voice_response'] = voice_response_result\n                            session['status'] = 'completed_with_voice'\n                        else:\n                            result['voice_response_error'] = voice_response_result.get('error')\n                            session['status'] = 'completed_text_only'\n                            \n                    except Exception as e:\n                        logger.warning(\n                            \"Voice response generation failed\",\n                            session_id=session_id,\n                            error=str(e)\n                        )\n                        result['voice_response_error'] = str(e)\n                        session['status'] = 'completed_text_only'\n                else:\n                    session['status'] = 'completed_text_only'\n                \n                # Record successful processing metrics\n                self.metrics.record_voice_processing(\n                    success=True,\n                    processing_time=time.time() - start_time,\n                    transcription_time=transcription_time,\n                    tts_time=result.get('voice_response', {}).get('processing_time'),\n                    language=transcription.language\n                )\n                \n                logger.info(\n                    \"Voice message processing completed successfully\",\n                    session_id=session_id,\n                    total_time=time.time() - start_time,\n                    has_voice_response=bool(result.get('voice_response'))\n                )\n                \n                return result\n                \n            except (VoiceProcessingError, WhisperError) as e:\n                error_type = type(e).__name__\n                logger.error(\n                    \"Voice processing failed\",\n                    session_id=session_id,\n                    error=str(e),\n                    error_type=error_type\n                )\n                \n                # Record failed processing metrics\n                self.metrics.record_voice_processing(\n                    success=False,\n                    processing_time=time.time() - start_time,\n                    error_type=error_type\n                )\n                \n                session['status'] = 'failed'\n                session['error'] = str(e)\n                \n                return {\n                    'success': False,\n                    'error': str(e),\n                    'error_type': error_type,\n                    'session_id': session_id,\n                    'processing_time': time.time() - start_time\n                }\n                \n            except Exception as e:\n                logger.error(\n                    \"Unexpected error in voice processing\",\n                    session_id=session_id,\n                    error=str(e),\n                    exc_info=True\n                )\n                \n                self.metrics.record_voice_processing(\n                    success=False,\n                    processing_time=time.time() - start_time,\n                    error_type=\"unexpected_error\"\n                )\n                \n                session['status'] = 'failed'\n                session['error'] = str(e)\n                \n                return {\n                    'success': False,\n                    'error': f\"Unexpected error: {str(e)}\",\n                    'error_type': 'unexpected_error',\n                    'session_id': session_id,\n                    'processing_time': time.time() - start_time\n                }\n    \n    async def generate_voice_response(\n        self,\n        text: str,\n        response_language: Optional[str] = None,\n        user_preferences: Optional[Dict[str, Any]] = None\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Generate AI response and convert to voice.\n        \n        Args:\n            text: Transcribed text to respond to\n            response_language: Language for the response\n            user_preferences: User preferences\n            \n        Returns:\n            Dict with voice response results\n        \"\"\"\n        try:\n            start_time = time.time()\n            \n            # Step 1: Generate text response (this would integrate with your LLM service)\n            # For now, we'll create a simple response\n            response_text = await self._generate_text_response(text, user_preferences)\n            \n            # Step 2: Convert response to speech\n            tts_service = await get_tts_service(redis_client=self.redis_client)\n            \n            voice_response = await tts_service.generate_speech(\n                text=response_text,\n                language=response_language or 'en',\n                optimize_for_telegram=True,\n                use_cache=self.enable_caching\n            )\n            \n            return {\n                'success': True,\n                'response_text': response_text,\n                'voice_file_path': voice_response.audio_file_path,\n                'duration': voice_response.duration,\n                'file_size': voice_response.file_size,\n                'processing_time': time.time() - start_time,\n                'cached': voice_response.cached,\n                'language': response_language or 'en'\n            }\n            \n        except TTSError as e:\n            logger.error(\"TTS generation failed\", error=str(e))\n            return {\n                'success': False,\n                'error': str(e),\n                'error_type': 'tts_error'\n            }\n        except Exception as e:\n            logger.error(\"Voice response generation failed\", error=str(e))\n            return {\n                'success': False,\n                'error': str(e),\n                'error_type': 'unexpected_error'\n            }\n    \n    async def _generate_text_response(\n        self,\n        text: str,\n        user_preferences: Optional[Dict[str, Any]] = None\n    ) -> str:\n        \"\"\"\n        Generate intelligent text response to transcribed speech.\n        \n        This should integrate with your existing LLM service.\n        For now, providing a simple implementation.\n        \"\"\"\n        try:\n            # Try to use existing LLM integration if available\n            try:\n                from ..services.llm_integration import get_llm_integration_service\n                from ..database.connection import get_async_session\n                \n                async with get_async_session() as db_session:\n                    llm_service = await get_llm_integration_service(db_session)\n                    \n                    # Generate response using LLM service\n                    response_content, metadata = await llm_service.generate_response(\n                        message_text=text,\n                        context={'source': 'voice_message'}\n                    )\n                    \n                    return response_content\n                    \n            except Exception as llm_error:\n                logger.warning(f\"LLM service unavailable for voice response: {llm_error}\")\n                # Fall back to simple response\n                return await self._generate_fallback_response(text)\n                \n        except Exception as e:\n            logger.error(\"Text response generation failed\", error=str(e))\n            return \"I heard your message but I'm having trouble generating a response right now.\"\n    \n    async def _generate_fallback_response(self, text: str) -> str:\n        \"\"\"\n        Generate simple fallback response when LLM is unavailable.\n        \"\"\"\n        text_lower = text.lower()\n        \n        # Simple pattern matching for common interactions\n        if any(word in text_lower for word in ['hello', 'hi', 'hey']):\n            return \"Hello! I heard your greeting. How can I help you today?\"\n        elif '?' in text:\n            return \"That's an interesting question! Let me think about it.\"\n        elif any(word in text_lower for word in ['thank', 'thanks']):\n            return \"You're welcome! Is there anything else I can help you with?\"\n        elif any(word in text_lower for word in ['help', 'assist']):\n            return \"I'd be happy to help you! Could you tell me more about what you need?\"\n        else:\n            return \"I understood your message. How can I assist you further?\"\n    \n    def _should_generate_voice_response(\n        self,\n        transcription: WhisperResponse,\n        user_preferences: Optional[Dict[str, Any]],\n        chat_id: int\n    ) -> bool:\n        \"\"\"\n        Determine whether to generate a voice response based on various factors.\n        \"\"\"\n        try:\n            # Check user preferences\n            if user_preferences and not user_preferences.get('voice_responses_enabled', True):\n                return False\n            \n            # Check if transcription quality is good enough\n            if transcription.confidence and transcription.confidence < 0.7:\n                logger.info(\"Skipping voice response due to low transcription confidence\")\n                return False\n            \n            # Don't generate voice for very short transcriptions (likely noise)\n            if len(transcription.text.strip()) < 10:\n                return False\n            \n            # Check settings for voice response limits\n            settings = get_settings()\n            voice_settings = settings.voice_processing\n            \n            # Check quiet hours\n            current_hour = time.localtime().tm_hour\n            if (voice_settings.voice_quiet_hours_start <= current_hour or \n                current_hour <= voice_settings.voice_quiet_hours_end):\n                return False\n            \n            # Don't generate voice in group chats by default\n            if not voice_settings.enable_voice_in_groups and chat_id < 0:\n                return False\n            \n            return True\n            \n        except Exception as e:\n            logger.warning(\"Error determining voice response preference\", error=str(e))\n            return False\n    \n    def get_metrics(self) -> Dict[str, Any]:\n        \"\"\"Get current processing metrics.\"\"\"\n        return {\n            **self.metrics.stats,\n            'active_sessions': len(self.active_sessions),\n            'target_processing_time': self.target_processing_time,\n            'caching_enabled': self.enable_caching\n        }\n    \n    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:\n        \"\"\"Get information about currently active processing sessions.\"\"\"\n        current_time = time.time()\n        sessions_info = {}\n        \n        for session_id, session_info in self.active_sessions.items():\n            sessions_info[session_id] = {\n                **session_info,\n                'elapsed_time': current_time - session_info['start_time']\n            }\n        \n        return sessions_info\n    \n    async def cleanup_old_files(self, max_age_hours: int = 2) -> Dict[str, int]:\n        \"\"\"\n        Clean up old temporary files from voice processing.\n        \n        Args:\n            max_age_hours: Maximum age before cleanup\n            \n        Returns:\n            Dict with cleanup statistics\n        \"\"\"\n        try:\n            # Clean up voice processor files\n            voice_cleanup = await self.voice_processor.cleanup_old_files(max_age_hours)\n            \n            # Clean up TTS files\n            tts_service = await get_tts_service(redis_client=self.redis_client)\n            tts_cleanup = await tts_service.cleanup_old_files(max_age_hours)\n            \n            logger.info(\n                \"Voice integration cleanup completed\",\n                voice_files_removed=voice_cleanup,\n                tts_files_removed=tts_cleanup\n            )\n            \n            return {\n                'voice_files_removed': voice_cleanup,\n                'tts_files_removed': tts_cleanup,\n                'total_files_removed': voice_cleanup + tts_cleanup\n            }\n            \n        except Exception as e:\n            logger.error(\"Voice integration cleanup failed\", error=str(e))\n            return {\n                'voice_files_removed': 0,\n                'tts_files_removed': 0,\n                'total_files_removed': 0,\n                'error': str(e)\n            }\n\n\n# Global voice integration service instance\n_voice_integration_service: Optional[VoiceIntegrationService] = None\n\n\nasync def get_voice_integration_service(\n    redis_client=None,\n    enable_caching: bool = True\n) -> VoiceIntegrationService:\n    \"\"\"Get or create global voice integration service instance.\"\"\"\n    global _voice_integration_service\n    if _voice_integration_service is None:\n        _voice_integration_service = VoiceIntegrationService(\n            redis_client=redis_client,\n            enable_caching=enable_caching\n        )\n    return _voice_integration_service\n\n\nasync def process_voice_message(\n    bot,\n    voice_message,\n    user_id: int,\n    chat_id: int,\n    user_preferences: Optional[Dict[str, Any]] = None,\n    generate_voice_response: bool = True,\n    redis_client=None\n) -> Dict[str, Any]:\n    \"\"\"\n    Convenience function for processing voice messages.\n    \n    Args:\n        bot: Telegram bot instance\n        voice_message: Telegram voice message\n        user_id: User ID\n        chat_id: Chat ID\n        user_preferences: User preferences\n        generate_voice_response: Whether to generate voice response\n        redis_client: Redis client for caching\n        \n    Returns:\n        Processing results dictionary\n    \"\"\"\n    service = await get_voice_integration_service(\n        redis_client=redis_client\n    )\n    \n    return await service.process_voice_message_complete(\n        bot=bot,\n        voice_message=voice_message,\n        user_id=user_id,\n        chat_id=chat_id,\n        user_preferences=user_preferences,\n        generate_voice_response=generate_voice_response\n    )