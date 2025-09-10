"""
Voice Processing Service

High-performance audio processing for Telegram voice messages with:
- OGG to MP3 conversion using pydub
- Audio file validation and size limits
- Temporary file management with automatic cleanup
- Comprehensive error handling and logging
- Performance optimization for <2s processing time
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import aiofiles
import structlog
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import hashlib

logger = structlog.get_logger(__name__)


class VoiceProcessingError(Exception):
    """Custom exception for voice processing errors."""
    pass


class AudioFileValidator:
    """Validates audio files for security and format compliance."""
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    MAX_DURATION_SECONDS = 300  # 5 minutes
    
    # Supported formats for input
    SUPPORTED_INPUT_FORMATS = {'.ogg', '.oga', '.wav', '.mp3', '.m4a', '.aac', '.flac'}
    
    # Audio constraints
    MIN_SAMPLE_RATE = 8000  # 8kHz minimum
    MAX_SAMPLE_RATE = 48000  # 48kHz maximum
    MAX_CHANNELS = 2  # Stereo maximum
    
    @staticmethod
    async def validate_audio_file(file_path: Path) -> Dict[str, Any]:
        """
        Validate audio file format, size, and content.
        
        Returns:
            Dict with validation results and metadata
        """
        try:
            # Check file exists
            if not file_path.exists():
                raise VoiceProcessingError(f"Audio file not found: {file_path}")
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > AudioFileValidator.MAX_FILE_SIZE:
                raise VoiceProcessingError(
                    f"File too large: {file_size} bytes (max: {AudioFileValidator.MAX_FILE_SIZE})"
                )
            
            if file_size == 0:
                raise VoiceProcessingError("Empty audio file")
            
            # Check file extension
            suffix = file_path.suffix.lower()
            if suffix not in AudioFileValidator.SUPPORTED_INPUT_FORMATS:
                raise VoiceProcessingError(
                    f"Unsupported format: {suffix}. Supported: {AudioFileValidator.SUPPORTED_INPUT_FORMATS}"
                )
            
            # Load and analyze audio file
            try:
                audio = AudioSegment.from_file(str(file_path))
            except CouldntDecodeError as e:
                raise VoiceProcessingError(f"Could not decode audio file: {e}")
            except Exception as e:
                raise VoiceProcessingError(f"Error loading audio: {e}")
            
            # Validate audio properties
            duration_seconds = len(audio) / 1000.0
            if duration_seconds > AudioFileValidator.MAX_DURATION_SECONDS:
                raise VoiceProcessingError(
                    f"Audio too long: {duration_seconds:.1f}s (max: {AudioFileValidator.MAX_DURATION_SECONDS}s)"
                )
            
            if duration_seconds < 0.1:
                raise VoiceProcessingError("Audio too short (minimum 0.1 seconds)")
            
            sample_rate = audio.frame_rate
            if sample_rate < AudioFileValidator.MIN_SAMPLE_RATE:
                raise VoiceProcessingError(f"Sample rate too low: {sample_rate}Hz")
            
            if sample_rate > AudioFileValidator.MAX_SAMPLE_RATE:
                raise VoiceProcessingError(f"Sample rate too high: {sample_rate}Hz")
            
            channels = audio.channels
            if channels > AudioFileValidator.MAX_CHANNELS:
                raise VoiceProcessingError(f"Too many channels: {channels}")
            
            # Calculate audio hash for deduplication
            audio_hash = hashlib.md5(audio.raw_data).hexdigest()
            
            return {
                'valid': True,
                'file_size': file_size,
                'duration_seconds': duration_seconds,
                'sample_rate': sample_rate,
                'channels': channels,
                'format': suffix,
                'audio_hash': audio_hash,
                'bitrate': getattr(audio, 'bitrate', None)
            }
            
        except VoiceProcessingError:
            raise
        except Exception as e:
            logger.error("Unexpected error validating audio", error=str(e), file_path=str(file_path))
            raise VoiceProcessingError(f"Validation failed: {e}")


class VoiceProcessor:
    """
    High-performance voice message processor with format conversion and optimization.
    
    Features:
    - Async OGG to MP3 conversion
    - Automatic audio optimization for speech recognition
    - Temporary file management with cleanup
    - Performance monitoring and optimization
    - Error handling with detailed logging
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize voice processor.
        
        Args:
            temp_dir: Custom temporary directory (optional)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'cache_hits': 0
        }
        
        # Simple in-memory cache for recently processed files
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 100
        
        logger.info("Voice processor initialized", temp_dir=str(self.temp_dir))
    
    async def download_voice_message(
        self, 
        bot, 
        file_id: str, 
        user_id: int
    ) -> Path:
        """
        Download voice message from Telegram servers.
        
        Args:
            bot: Telegram bot instance
            file_id: Telegram file ID
            user_id: User ID for file naming
            
        Returns:
            Path to downloaded file
        """
        try:
            start_time = time.time()
            
            # Get file info
            file = await bot.get_file(file_id)
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"voice_{user_id}_{timestamp}.ogg"
            file_path = self.temp_dir / filename
            
            # Download file
            await bot.download_file(file.file_path, file_path)
            
            download_time = time.time() - start_time
            logger.info(
                "Voice message downloaded", 
                file_id=file_id,
                user_id=user_id,
                file_size=file_path.stat().st_size,
                download_time=download_time
            )
            
            return file_path
            
        except Exception as e:
            logger.error("Failed to download voice message", error=str(e), file_id=file_id)
            raise VoiceProcessingError(f"Download failed: {e}")
    
    async def convert_ogg_to_mp3(
        self, 
        input_path: Path, 
        optimize_for_speech: bool = True
    ) -> Path:
        """
        Convert OGG audio to MP3 format with speech optimization.
        
        Args:
            input_path: Path to input OGG file
            optimize_for_speech: Whether to optimize for speech recognition
            
        Returns:
            Path to converted MP3 file
        """
        try:
            start_time = time.time()
            
            # Generate output path
            output_path = input_path.with_suffix('.mp3')
            
            # Check cache first
            cache_key = f"{input_path.name}_{optimize_for_speech}"
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                if Path(cached_result['output_path']).exists():
                    self.processing_stats['cache_hits'] += 1
                    logger.debug("Using cached conversion", cache_key=cache_key)
                    return Path(cached_result['output_path'])
            
            # Load audio file
            try:
                audio = AudioSegment.from_ogg(str(input_path))
            except Exception as e:
                # Fallback to generic loader
                audio = AudioSegment.from_file(str(input_path))
            
            # Optimize for speech recognition
            if optimize_for_speech:
                audio = await self._optimize_for_speech(audio)
            
            # Convert to MP3 with high quality settings
            audio.export(
                str(output_path),
                format="mp3",
                bitrate="128k",  # Good quality for speech
                parameters=[
                    "-ac", "1",  # Force mono for speech
                    "-ar", "16000",  # 16kHz sample rate (optimal for speech recognition)
                    "-q:a", "2"  # High quality encoding
                ]
            )
            
            processing_time = time.time() - start_time
            
            # Update cache
            await self._update_cache(cache_key, {
                'output_path': str(output_path),
                'processing_time': processing_time,
                'timestamp': time.time()
            })
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['total_processed'] - 1) + 
                 processing_time) / self.processing_stats['total_processed']
            )
            
            logger.info(
                "Audio conversion completed",
                input_file=str(input_path),
                output_file=str(output_path),
                input_size=input_path.stat().st_size,
                output_size=output_path.stat().st_size,
                processing_time=processing_time,
                optimized=optimize_for_speech
            )
            
            return output_path
            
        except Exception as e:
            self.processing_stats['errors'] += 1
            logger.error("Audio conversion failed", error=str(e), input_path=str(input_path))
            raise VoiceProcessingError(f"Conversion failed: {e}")
    
    async def _optimize_for_speech(self, audio: AudioSegment) -> AudioSegment:
        """
        Optimize audio segment for speech recognition accuracy.
        
        Args:
            audio: Input audio segment
            
        Returns:
            Optimized audio segment
        """
        try:
            # Convert to mono (speech recognition works better with mono)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Normalize audio levels (speech recognition is sensitive to volume)
            # Target -20dB to -12dB for optimal recognition
            target_dbfs = -16.0
            current_dbfs = audio.dBFS
            
            if current_dbfs < target_dbfs - 5:
                # Boost quiet audio
                audio = audio + (target_dbfs - current_dbfs)
            elif current_dbfs > target_dbfs + 5:
                # Reduce loud audio
                audio = audio + (target_dbfs - current_dbfs)
            
            # Apply noise reduction (simple high-pass filter to remove low-frequency noise)
            # This removes rumble and background hum
            audio = audio.high_pass_filter(80)  # Remove frequencies below 80Hz
            
            # Apply gentle compression to even out volume levels
            # This helps with varying speech volumes
            audio = audio.compress_dynamic_range(
                threshold=-20.0,  # Start compressing at -20dB
                ratio=2.0,        # 2:1 compression ratio
                attack=5.0,       # 5ms attack
                release=50.0      # 50ms release
            )
            
            # Ensure sample rate is optimal for speech recognition (16kHz)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            logger.debug("Audio optimized for speech recognition")
            return audio
            
        except Exception as e:
            logger.warning("Speech optimization failed, using original audio", error=str(e))
            return audio
    
    async def _update_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Update conversion cache with size management."""
        try:
            # Remove oldest entries if cache is full
            if len(self._cache) >= self._cache_max_size:
                # Remove 20% of oldest entries
                sorted_items = sorted(
                    self._cache.items(), 
                    key=lambda x: x[1].get('timestamp', 0)
                )
                items_to_remove = len(sorted_items) // 5
                for old_key, _ in sorted_items[:items_to_remove]:
                    del self._cache[old_key]
            
            self._cache[key] = value
            
        except Exception as e:
            logger.warning("Cache update failed", error=str(e))
    
    async def process_voice_message(
        self,
        bot,
        file_id: str,
        user_id: int,
        optimize_for_speech: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Complete voice message processing pipeline.
        
        Args:
            bot: Telegram bot instance
            file_id: Telegram file ID
            user_id: User ID
            optimize_for_speech: Whether to optimize for speech recognition
            
        Returns:
            Tuple of (converted_file_path, processing_metadata)
        """
        ogg_path = None
        mp3_path = None
        
        try:
            start_time = time.time()
            
            # Step 1: Download voice message
            ogg_path = await self.download_voice_message(bot, file_id, user_id)
            
            # Step 2: Validate audio file
            validation_result = await AudioFileValidator.validate_audio_file(ogg_path)
            if not validation_result['valid']:
                raise VoiceProcessingError("Audio validation failed")
            
            # Step 3: Convert to MP3
            mp3_path = await self.convert_ogg_to_mp3(ogg_path, optimize_for_speech)
            
            # Step 4: Validate converted file
            mp3_validation = await AudioFileValidator.validate_audio_file(mp3_path)
            
            total_processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'processing_time': total_processing_time,
                'original_format': validation_result['format'],
                'original_size': validation_result['file_size'],
                'original_duration': validation_result['duration_seconds'],
                'converted_size': mp3_path.stat().st_size,
                'sample_rate': mp3_validation['sample_rate'],
                'channels': mp3_validation['channels'],
                'optimized_for_speech': optimize_for_speech,
                'audio_hash': validation_result['audio_hash'],
                'user_id': user_id,
                'file_id': file_id
            }
            
            logger.info(
                "Voice message processing completed",
                **metadata
            )
            
            # Clean up original OGG file
            if ogg_path and ogg_path.exists():
                await asyncio.to_thread(ogg_path.unlink)
            
            return mp3_path, metadata
            
        except Exception as e:
            # Clean up any temporary files on error
            if ogg_path and ogg_path.exists():
                try:
                    await asyncio.to_thread(ogg_path.unlink)
                except Exception:
                    pass
            
            if mp3_path and mp3_path.exists():
                try:
                    await asyncio.to_thread(mp3_path.unlink)
                except Exception:
                    pass
            
            logger.error("Voice message processing failed", error=str(e), user_id=user_id)
            raise VoiceProcessingError(f"Processing failed: {e}")
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary audio files.
        
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
            for file_path in self.temp_dir.glob("voice_*.mp3"):
                try:
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        await asyncio.to_thread(file_path.unlink)
                        cleanup_count += 1
                except Exception as e:
                    logger.warning("Failed to delete old file", file=str(file_path), error=str(e))
            
            # Clean up cache of old entries
            cache_cleaned = 0
            keys_to_remove = []
            for key, value in self._cache.items():
                if current_time - value.get('timestamp', 0) > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
                cache_cleaned += 1
            
            logger.info(
                "Cleanup completed", 
                files_removed=cleanup_count,
                cache_entries_removed=cache_cleaned,
                max_age_hours=max_age_hours
            )
            
            return cleanup_count
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            return 0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.processing_stats,
            'cache_size': len(self._cache),
            'temp_dir': str(self.temp_dir)
        }


# Global voice processor instance
_voice_processor: Optional[VoiceProcessor] = None

def get_voice_processor() -> VoiceProcessor:
    """Get or create global voice processor instance."""
    global _voice_processor
    if _voice_processor is None:
        _voice_processor = VoiceProcessor()
    return _voice_processor


async def process_telegram_voice(
    bot,
    voice_message,
    user_id: int,
    optimize_for_speech: bool = True
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convenience function to process Telegram voice message.
    
    Args:
        bot: Telegram bot instance
        voice_message: Telegram voice message object
        user_id: User ID
        optimize_for_speech: Whether to optimize for speech recognition
        
    Returns:
        Tuple of (mp3_file_path, metadata)
    """
    processor = get_voice_processor()
    return await processor.process_voice_message(
        bot=bot,
        file_id=voice_message.file_id,
        user_id=user_id,
        optimize_for_speech=optimize_for_speech
    )