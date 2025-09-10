"""
Text-to-Speech Service using Google Text-to-Speech (gTTS)

High-performance TTS service for generating voice responses with:
- Google TTS integration with multiple language support
- Audio optimization for Telegram voice messages
- Caching system for frequently used responses
- Performance optimization for quick response generation
- Comprehensive error handling and fallback options
"""

import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import aiofiles
import structlog
from gtts import gTTS
from gtts.lang import tts_langs
import aioredis
from pydantic import BaseModel, Field
from io import BytesIO
import requests
from pydub import AudioSegment

logger = structlog.get_logger(__name__)


class TTSResponse(BaseModel):
    """Structured response from TTS service."""
    audio_file_path: str = Field(description="Path to generated audio file")
    text: str = Field(description="Original text")
    language: str = Field(description="Language code used")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    file_size: int = Field(description="Audio file size in bytes")
    processing_time: float = Field(description="Generation time in seconds")
    cached: bool = Field(False, description="Whether result was cached")
    format: str = Field(default="mp3", description="Audio format")


class TTSError(Exception):
    """Custom exception for TTS errors."""
    pass


class TTSService:
    """
    Production-ready Text-to-Speech service with caching and optimization.
    
    Features:
    - Google TTS with multilingual support
    - Redis caching for performance
    - Audio optimization for voice messages
    - Smart text chunking for long content
    - Performance monitoring and metrics
    - Fallback mechanisms for reliability
    """
    
    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        cache_ttl: int = 3600 * 24 * 7,  # 7 days cache
        temp_dir: Optional[str] = None,
        enable_caching: bool = True,
        max_text_length: int = 5000,
        default_language: str = 'en'
    ):
        """
        Initialize TTS service.
        
        Args:
            redis_client: Redis client for caching
            cache_ttl: Cache time-to-live in seconds
            temp_dir: Custom temporary directory
            enable_caching: Whether to enable response caching
            max_text_length: Maximum text length for TTS
            default_language: Default language code
        """
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        self.enable_caching = enable_caching
        self.max_text_length = max_text_length
        self.default_language = default_language
        
        # Get available languages from gTTS
        try:
            self.supported_languages = tts_langs()
            logger.info(f"Loaded {len(self.supported_languages)} TTS languages")
        except Exception as e:
            logger.warning("Failed to load TTS languages, using fallback", error=str(e))
            self.supported_languages = {
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
                'hi': 'Hindi'
            }
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'total_characters_processed': 0,
            'languages_used': {}
        }
        
        logger.info(
            "TTS service initialized",
            caching_enabled=enable_caching,
            supported_languages=len(self.supported_languages),
            default_language=default_language
        )
    
    async def _get_cache_key(self, text: str, language: str, params: Dict[str, Any]) -> str:
        """Generate cache key for text and parameters."""
        try:
            # Create hash from text, language, and parameters
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            return f"tts:{language}:{text_hash}:{params_hash}"
            
        except Exception as e:
            logger.warning("Failed to generate cache key", error=str(e))
            return f"tts:fallback:{int(time.time())}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[TTSResponse]:
        """Get cached TTS result."""
        if not self.enable_caching or not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                result_dict = json.loads(cached_data)
                
                # Check if cached file still exists
                audio_file_path = Path(result_dict['audio_file_path'])
                if audio_file_path.exists():
                    result_dict['cached'] = True
                    self.stats['cache_hits'] += 1
                    logger.debug("Cache hit for TTS", cache_key=cache_key)
                    return TTSResponse(**result_dict)
                else:
                    # Remove stale cache entry
                    await self.redis_client.delete(cache_key)
                    logger.debug("Removed stale cache entry", cache_key=cache_key)
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _cache_result(self, cache_key: str, result: TTSResponse) -> None:
        """Cache TTS result."""
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
    
    def _validate_text(self, text: str) -> str:
        """Validate and clean text for TTS."""
        if not text or not text.strip():
            raise TTSError("Text is empty")
        
        text = text.strip()
        
        if len(text) > self.max_text_length:
            logger.warning(f"Text too long, truncating from {len(text)} to {self.max_text_length}")
            text = text[:self.max_text_length].rsplit(' ', 1)[0] + "..."
        
        # Remove or replace problematic characters
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def _validate_language(self, language: str) -> str:
        """Validate and normalize language code."""
        if not language:
            return self.default_language
        
        language = language.lower().strip()
        
        # Handle common variations
        language_mappings = {
            'eng': 'en',
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'japanese': 'ja',
            'korean': 'ko',
            'chinese': 'zh',
            'arabic': 'ar',
            'hindi': 'hi'
        }
        
        language = language_mappings.get(language, language)
        
        if language not in self.supported_languages:
            logger.warning(f"Unsupported language '{language}', using default '{self.default_language}'")
            return self.default_language
        
        return language
    
    async def _generate_tts_audio(
        self,
        text: str,
        language: str,
        slow: bool = False,
        optimize_for_telegram: bool = True
    ) -> Path:
        """Generate TTS audio file."""
        try:
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"tts_{hashlib.md5(text.encode()).hexdigest()[:8]}_{timestamp}.mp3"
            output_path = self.temp_dir / filename
            
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=language,
                slow=slow,
                lang_check=False  # Skip language check for performance
            )
            
            # Generate audio in memory first
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Save to file
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(mp3_fp.read())
            
            # Optimize audio for Telegram if requested
            if optimize_for_telegram:
                await self._optimize_for_telegram(output_path)
            
            logger.debug(
                "TTS audio generated",
                text_length=len(text),
                language=language,
                output_file=str(output_path),
                file_size=output_path.stat().st_size
            )
            
            return output_path
            
        except Exception as e:
            logger.error("TTS audio generation failed", error=str(e))
            raise TTSError(f"Audio generation failed: {e}")
    
    async def _optimize_for_telegram(self, audio_path: Path) -> None:
        """Optimize audio file for Telegram voice messages."""
        try:
            # Load audio
            audio = AudioSegment.from_mp3(str(audio_path))
            
            # Optimize for voice messages
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set optimal sample rate for voice (16kHz)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            # Normalize volume
            audio = audio.normalize()
            
            # Apply gentle compression for better voice quality
            audio = audio.compress_dynamic_range(threshold=-20.0, ratio=2.0)
            
            # Export with optimized settings
            audio.export(
                str(audio_path),
                format="mp3",
                bitrate="64k",  # Good quality for voice, smaller file size
                parameters=["-q:a", "4"]  # Good quality setting
            )
            
            logger.debug("Audio optimized for Telegram", file_path=str(audio_path))
            
        except Exception as e:
            logger.warning("Audio optimization failed, using original", error=str(e))
    
    async def _chunk_long_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split long text into chunks for TTS processing."""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def generate_speech(
        self,
        text: str,
        language: Optional[str] = None,
        slow: bool = False,
        optimize_for_telegram: bool = True,
        use_cache: bool = True,
        chunk_long_text: bool = True
    ) -> TTSResponse:
        """
        Generate speech from text using Google TTS.
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'en', 'es') or None for default
            slow: Whether to speak slowly
            optimize_for_telegram: Whether to optimize audio for Telegram
            use_cache: Whether to use cached results
            chunk_long_text: Whether to chunk long text automatically
            
        Returns:
            TTSResponse with audio file and metadata
        """
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # Validate and clean inputs
            text = self._validate_text(text)
            language = self._validate_language(language or self.default_language)
            
            # Prepare parameters for caching
            cache_params = {
                'slow': slow,
                'optimize_for_telegram': optimize_for_telegram
            }
            
            # Check cache if enabled
            cached_result = None
            if use_cache:
                cache_key = await self._get_cache_key(text, language, cache_params)
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result:
                    logger.info(
                        "TTS retrieved from cache",
                        text_length=len(text),
                        language=language,
                        cache_key=cache_key
                    )
                    return cached_result
            
            # Handle long text by chunking if needed
            if chunk_long_text and len(text) > 500:
                return await self._generate_chunked_speech(
                    text, language, slow, optimize_for_telegram, use_cache
                )
            
            # Generate TTS audio
            logger.info("Starting TTS generation", text_length=len(text), language=language)
            
            audio_path = await self._generate_tts_audio(
                text=text,
                language=language,
                slow=slow,
                optimize_for_telegram=optimize_for_telegram
            )
            
            # Get audio metadata
            file_size = audio_path.stat().st_size
            duration = None
            
            try:
                audio = AudioSegment.from_mp3(str(audio_path))
                duration = len(audio) / 1000.0  # Convert to seconds
            except Exception as e:
                logger.warning("Failed to get audio duration", error=str(e))
            
            processing_time = time.time() - start_time
            
            # Create response object
            result = TTSResponse(
                audio_file_path=str(audio_path),
                text=text,
                language=language,
                duration=duration,
                file_size=file_size,
                processing_time=processing_time,
                cached=False,
                format="mp3"
            )
            
            # Cache the result
            if use_cache and cached_result is None:
                await self._cache_result(cache_key, result)
            
            # Update statistics
            self.stats['successful_requests'] += 1
            self.stats['total_characters_processed'] += len(text)
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['successful_requests'] - 1) + 
                 processing_time) / self.stats['successful_requests']
            )
            
            lang_count = self.stats['languages_used'].get(language, 0)
            self.stats['languages_used'][language] = lang_count + 1
            
            logger.info(
                "TTS generation completed successfully",
                text_length=len(text),
                language=language,
                processing_time=processing_time,
                file_size=file_size,
                duration=duration
            )
            
            return result
            
        except TTSError:
            self.stats['failed_requests'] += 1
            raise
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error("TTS generation failed", error=str(e), text_length=len(text))
            raise TTSError(f"Speech generation failed: {e}")
    
    async def _generate_chunked_speech(
        self,
        text: str,
        language: str,
        slow: bool,
        optimize_for_telegram: bool,
        use_cache: bool
    ) -> TTSResponse:
        """Generate speech for long text by chunking."""
        try:
            chunks = await self._chunk_long_text(text)
            logger.info(f"Processing long text in {len(chunks)} chunks")
            
            # Generate audio for each chunk
            chunk_paths = []
            total_duration = 0.0
            
            for i, chunk in enumerate(chunks):
                chunk_result = await self.generate_speech(
                    text=chunk,
                    language=language,
                    slow=slow,
                    optimize_for_telegram=optimize_for_telegram,
                    use_cache=use_cache,
                    chunk_long_text=False  # Avoid recursion
                )
                
                chunk_paths.append(Path(chunk_result.audio_file_path))
                if chunk_result.duration:
                    total_duration += chunk_result.duration
            
            # Combine chunks into single audio file
            combined_audio = AudioSegment.empty()
            for chunk_path in chunk_paths:
                chunk_audio = AudioSegment.from_mp3(str(chunk_path))
                combined_audio += chunk_audio
                
                # Add small pause between chunks
                if chunk_path != chunk_paths[-1]:
                    combined_audio += AudioSegment.silent(duration=500)  # 0.5 second pause
            
            # Save combined audio
            timestamp = int(time.time() * 1000)
            combined_filename = f"tts_combined_{timestamp}.mp3"
            combined_path = self.temp_dir / combined_filename
            
            combined_audio.export(str(combined_path), format="mp3", bitrate="64k")
            
            # Clean up chunk files
            for chunk_path in chunk_paths:
                try:
                    await asyncio.to_thread(chunk_path.unlink)
                except Exception:
                    pass
            
            # Create response
            file_size = combined_path.stat().st_size
            duration = len(combined_audio) / 1000.0
            
            return TTSResponse(
                audio_file_path=str(combined_path),
                text=text,
                language=language,
                duration=duration,
                file_size=file_size,
                processing_time=time.time() - time.time(),  # Approximate
                cached=False,
                format="mp3"
            )
            
        except Exception as e:
            logger.error("Chunked TTS generation failed", error=str(e))
            raise TTSError(f"Chunked speech generation failed: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return self.supported_languages.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current service statistics."""
        return {
            **self.stats,
            'cache_enabled': self.enable_caching,
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1)
            ) * 100
        }
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old TTS audio files.
        
        Args:
            max_age_hours: Maximum age in hours before deletion
            
        Returns:
            Number of files cleaned up
        """
        try:
            cleanup_count = 0
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Clean up temp directory
            for file_path in self.temp_dir.glob("tts_*.mp3"):
                try:
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        await asyncio.to_thread(file_path.unlink)
                        cleanup_count += 1
                except Exception as e:
                    logger.warning("Failed to delete old TTS file", file=str(file_path), error=str(e))
            
            logger.info(
                "TTS cleanup completed", 
                files_removed=cleanup_count,
                max_age_hours=max_age_hours
            )
            
            return cleanup_count
            
        except Exception as e:
            logger.error("TTS cleanup failed", error=str(e))
            return 0
    
    async def cleanup_cache(self, pattern: str = "tts:*") -> int:
        """
        Clean up cached TTS results.
        
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
                logger.info("TTS cache cleanup completed", keys_deleted=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.error("TTS cache cleanup failed", error=str(e))
            return 0


# Global TTS service instance
_tts_service: Optional[TTSService] = None

async def get_tts_service(
    redis_client: Optional[aioredis.Redis] = None
) -> TTSService:
    """Get or create global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(redis_client=redis_client)
    return _tts_service


async def generate_voice_response(
    text: str,
    language: Optional[str] = None,
    redis_client: Optional[aioredis.Redis] = None
) -> TTSResponse:
    """
    Convenience function to generate a voice response.
    
    Args:
        text: Text to convert to speech
        language: Language code or None for default
        redis_client: Redis client for caching
        
    Returns:
        TTSResponse with audio file
    """
    service = await get_tts_service(redis_client=redis_client)
    return await service.generate_speech(
        text=text,
        language=language,
        optimize_for_telegram=True
    )