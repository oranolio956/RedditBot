"""
OpenAI Whisper API Client

High-performance speech-to-text service with:
- OpenAI Whisper API integration
- Intelligent retry logic with exponential backoff
- Language detection and support
- Response caching with Redis
- Performance optimization for <2s response time
- Comprehensive error handling and monitoring
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import aiofiles
import structlog
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import aioredis
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class WhisperResponse(BaseModel):
    """Structured response from Whisper API."""
    text: str = Field(description="Transcribed text")
    language: Optional[str] = Field(None, description="Detected language code")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    processing_time: float = Field(description="API processing time in seconds")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed segments")
    cached: bool = Field(False, description="Whether result was cached")


class WhisperError(Exception):
    """Custom exception for Whisper API errors."""
    pass


class WhisperClient:
    """
    Production-ready OpenAI Whisper API client with caching and optimization.
    
    Features:
    - Async API calls with proper error handling
    - Redis caching for performance
    - Language detection and multilingual support
    - Retry logic with exponential backoff
    - Performance monitoring and metrics
    - File size and duration optimization
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        redis_client: Optional[aioredis.Redis] = None,
        cache_ttl: int = 3600 * 24,  # 24 hours cache
        max_file_size: int = 25 * 1024 * 1024,  # 25MB max
        enable_caching: bool = True
    ):
        """
        Initialize Whisper client.
        
        Args:
            api_key: OpenAI API key (will use env var if not provided)
            redis_client: Redis client for caching
            cache_ttl: Cache time-to-live in seconds
            max_file_size: Maximum file size in bytes
            enable_caching: Whether to enable response caching
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.max_file_size = max_file_size
        self.enable_caching = enable_caching
        
        # Supported languages with their codes
        self.supported_languages = {
            'auto': 'Auto-detect',
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish'
        }
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'total_audio_duration': 0.0,
            'languages_detected': {}
        }
        
        logger.info(
            "Whisper client initialized",
            caching_enabled=enable_caching,
            max_file_size=max_file_size,
            supported_languages=len(self.supported_languages)
        )
    
    async def _get_cache_key(self, audio_file_path: Path, params: Dict[str, Any]) -> str:
        """Generate cache key for audio file and parameters."""
        try:
            # Read file content for hash
            async with aiofiles.open(audio_file_path, 'rb') as f:
                content = await f.read()
            
            # Create hash from file content and parameters
            content_hash = hashlib.md5(content).hexdigest()
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            return f"whisper:{content_hash}:{params_hash}"
            
        except Exception as e:
            logger.warning("Failed to generate cache key", error=str(e))
            return f"whisper:fallback:{int(time.time())}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[WhisperResponse]:
        """Get cached transcription result."""
        if not self.enable_caching or not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                result_dict = json.loads(cached_data)
                result_dict['cached'] = True
                self.stats['cache_hits'] += 1
                logger.debug("Cache hit for transcription", cache_key=cache_key)
                return WhisperResponse(**result_dict)
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _cache_result(self, cache_key: str, result: WhisperResponse) -> None:
        """Cache transcription result."""
        if not self.enable_caching or not self.redis_client:
            return
        
        try:
            # Don't cache the 'cached' flag itself
            result_dict = result.dict()
            result_dict.pop('cached', None)
            
            cached_data = json.dumps(result_dict)
            await self.redis_client.setex(cache_key, self.cache_ttl, cached_data)
            logger.debug("Result cached", cache_key=cache_key, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))
    
    def _validate_audio_file(self, file_path: Path) -> None:
        """Validate audio file before processing."""
        if not file_path.exists():
            raise WhisperError(f"Audio file not found: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise WhisperError("Audio file is empty")
        
        if file_size > self.max_file_size:
            raise WhisperError(f"Audio file too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Check file extension
        allowed_extensions = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
        if file_path.suffix.lower() not in allowed_extensions:
            raise WhisperError(f"Unsupported file format: {file_path.suffix}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, structlog.INFO)
    )
    async def _make_whisper_request(
        self,
        audio_file_path: Path,
        model: str = "whisper-1",
        language: Optional[str] = None,
        response_format: str = "verbose_json",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Make request to Whisper API with retry logic."""
        try:
            # Open file for API request
            async with aiofiles.open(audio_file_path, 'rb') as audio_file:
                content = await audio_file.read()
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'response_format': response_format,
                'temperature': temperature
            }
            
            if language and language != 'auto':
                request_params['language'] = language
            
            # Make API request
            response = await self.client.audio.transcriptions.create(
                file=(audio_file_path.name, content, 'audio/mpeg'),
                **request_params
            )
            
            # Handle different response formats
            if response_format == "verbose_json":
                return response.dict()
            else:
                return {'text': response}
                
        except openai.RateLimitError as e:
            logger.error("Whisper API rate limit exceeded", error=str(e))
            raise WhisperError(f"Rate limit exceeded: {e}")
        except openai.APIError as e:
            logger.error("Whisper API error", error=str(e))
            raise WhisperError(f"API error: {e}")
        except Exception as e:
            logger.error("Unexpected error in Whisper request", error=str(e))
            raise WhisperError(f"Request failed: {e}")
    
    async def transcribe_audio(
        self,
        audio_file_path: Union[str, Path],
        language: Optional[str] = None,
        model: str = "whisper-1",
        temperature: float = 0.0,
        use_cache: bool = True
    ) -> WhisperResponse:
        """
        Transcribe audio file to text using OpenAI Whisper.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            model: Whisper model to use
            temperature: Sampling temperature (0.0 = deterministic)
            use_cache: Whether to use cached results
            
        Returns:
            WhisperResponse with transcription and metadata
        """
        start_time = time.time()
        file_path = Path(audio_file_path)
        
        try:
            self.stats['total_requests'] += 1
            
            # Validate input file
            self._validate_audio_file(file_path)
            
            # Prepare parameters for caching
            cache_params = {
                'model': model,
                'language': language,
                'temperature': temperature
            }
            
            # Check cache if enabled
            cached_result = None
            if use_cache:
                cache_key = await self._get_cache_key(file_path, cache_params)
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result:
                    logger.info(
                        "Transcription retrieved from cache",
                        file_path=str(file_path),
                        cache_key=cache_key,
                        text_length=len(cached_result.text)
                    )
                    return cached_result
            
            # Make API request
            logger.info("Starting Whisper transcription", file_path=str(file_path), language=language)
            
            response_data = await self._make_whisper_request(
                file_path,
                model=model,
                language=language,
                response_format="verbose_json",
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            # Extract response data
            text = response_data.get('text', '').strip()
            detected_language = response_data.get('language')
            duration = response_data.get('duration')
            segments = response_data.get('segments')
            
            # Calculate confidence (average of segment confidences if available)
            confidence = None
            if segments:
                confidences = [seg.get('confidence', 0) for seg in segments if 'confidence' in seg]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            # Create response object
            result = WhisperResponse(
                text=text,
                language=detected_language,
                confidence=confidence,
                duration=duration,
                processing_time=processing_time,
                segments=segments,
                cached=False
            )
            
            # Cache the result
            if use_cache and cached_result is None:
                await self._cache_result(cache_key, result)
            
            # Update statistics
            self.stats['successful_requests'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['successful_requests'] - 1) + 
                 processing_time) / self.stats['successful_requests']
            )
            
            if duration:
                self.stats['total_audio_duration'] += duration
            
            if detected_language:
                lang_count = self.stats['languages_detected'].get(detected_language, 0)
                self.stats['languages_detected'][detected_language] = lang_count + 1
            
            logger.info(
                "Transcription completed successfully",
                file_path=str(file_path),
                text_length=len(text),
                detected_language=detected_language,
                processing_time=processing_time,
                confidence=confidence,
                duration=duration
            )
            
            return result
            
        except WhisperError:
            self.stats['failed_requests'] += 1
            raise
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error("Transcription failed", error=str(e), file_path=str(file_path))
            raise WhisperError(f"Transcription failed: {e}")
    
    async def detect_language(
        self,
        audio_file_path: Union[str, Path],
        model: str = "whisper-1"
    ) -> Tuple[str, float]:
        """
        Detect the language of an audio file.
        
        Args:
            audio_file_path: Path to audio file
            model: Whisper model to use
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Transcribe with auto-detection
            result = await self.transcribe_audio(
                audio_file_path=audio_file_path,
                language=None,  # Auto-detect
                model=model,
                use_cache=False  # Don't cache language detection calls
            )
            
            language = result.language or 'unknown'
            confidence = result.confidence or 0.0
            
            logger.info(
                "Language detected",
                file_path=str(audio_file_path),
                language=language,
                confidence=confidence
            )
            
            return language, confidence
            
        except Exception as e:
            logger.error("Language detection failed", error=str(e))
            return 'unknown', 0.0
    
    async def transcribe_with_timestamps(
        self,
        audio_file_path: Union[str, Path],
        language: Optional[str] = None,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio with detailed timestamp information.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code or None for auto-detection
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Detailed transcription with timestamps
        """
        try:
            result = await self.transcribe_audio(
                audio_file_path=audio_file_path,
                language=language,
                temperature=0.0  # Deterministic for consistent timestamps
            )
            
            # Process segments for timestamps
            timestamped_segments = []
            if result.segments:
                for segment in result.segments:
                    segment_info = {
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', ''),
                        'confidence': segment.get('confidence', 0)
                    }
                    
                    # Add word-level timestamps if available and requested
                    if word_timestamps and 'words' in segment:
                        segment_info['words'] = segment['words']
                    
                    timestamped_segments.append(segment_info)
            
            return {
                'text': result.text,
                'language': result.language,
                'duration': result.duration,
                'segments': timestamped_segments,
                'processing_time': result.processing_time
            }
            
        except Exception as e:
            logger.error("Timestamped transcription failed", error=str(e))
            raise WhisperError(f"Timestamped transcription failed: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return self.supported_languages.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current client statistics."""
        return {
            **self.stats,
            'cache_enabled': self.enable_caching,
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1)
            ) * 100
        }
    
    async def cleanup_cache(self, pattern: str = "whisper:*") -> int:
        """
        Clean up cached transcription results.
        
        Args:
            pattern: Redis key pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info("Cache cleanup completed", keys_deleted=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.error("Cache cleanup failed", error=str(e))
            return 0


# Global Whisper client instance
_whisper_client: Optional[WhisperClient] = None

async def get_whisper_client(
    api_key: Optional[str] = None,
    redis_client: Optional[aioredis.Redis] = None
) -> WhisperClient:
    """Get or create global Whisper client instance."""
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = WhisperClient(
            api_key=api_key,
            redis_client=redis_client
        )
    return _whisper_client


async def transcribe_voice_message(
    audio_file_path: Union[str, Path],
    language: Optional[str] = None,
    redis_client: Optional[aioredis.Redis] = None,
    api_key: Optional[str] = None
) -> WhisperResponse:
    """
    Convenience function to transcribe a voice message.
    
    Args:
        audio_file_path: Path to audio file
        language: Language code or None for auto-detection
        redis_client: Redis client for caching
        api_key: OpenAI API key
        
    Returns:
        WhisperResponse with transcription
    """
    client = await get_whisper_client(api_key=api_key, redis_client=redis_client)
    return await client.transcribe_audio(
        audio_file_path=audio_file_path,
        language=language
    )