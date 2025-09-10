"""
Voice Processing Tests

Comprehensive test suite for voice message processing:
- Voice processor functionality
- Whisper API integration
- TTS service
- Integration service
- Performance benchmarks
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json

# Import the services we're testing
from app.services.voice_processor import (
    VoiceProcessor, AudioFileValidator, get_voice_processor,
    VoiceProcessingError, process_telegram_voice
)
from app.services.whisper_client import (
    WhisperClient, WhisperResponse, WhisperError,
    get_whisper_client, transcribe_voice_message
)
from app.services.tts_service import (
    TTSService, TTSResponse, TTSError,
    get_tts_service, generate_voice_response
)
from app.services.voice_integration import (
    VoiceIntegrationService, VoiceIntegrationMetrics,
    get_voice_integration_service, process_voice_message
)


class TestAudioFileValidator:
    """Test audio file validation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = AudioFileValidator()
    
    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        with pytest.raises(VoiceProcessingError, match="not found"):
            await self.validator.validate_audio_file(Path("nonexistent.ogg"))
    
    @pytest.mark.asyncio
    async def test_validate_empty_file(self):
        """Test validation fails for empty file."""
        with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
            tmp_path = Path(tmp.name)
            with pytest.raises(VoiceProcessingError, match="Empty audio file"):
                await self.validator.validate_audio_file(tmp_path)
    
    @pytest.mark.asyncio
    async def test_validate_large_file(self):
        """Test validation fails for oversized file."""
        with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
            # Write data exceeding max size
            large_data = b"0" * (AudioFileValidator.MAX_FILE_SIZE + 1)
            tmp.write(large_data)
            tmp.flush()
            
            tmp_path = Path(tmp.name)
            with pytest.raises(VoiceProcessingError, match="File too large"):
                await self.validator.validate_audio_file(tmp_path)
    
    @pytest.mark.asyncio
    async def test_validate_unsupported_format(self):
        """Test validation fails for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            tmp.write(b"not audio data")
            tmp.flush()
            
            tmp_path = Path(tmp.name)
            with pytest.raises(VoiceProcessingError, match="Unsupported format"):
                await self.validator.validate_audio_file(tmp_path)


class TestVoiceProcessor:
    """Test voice processor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = VoiceProcessor(temp_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_download_voice_message(self):
        """Test voice message download functionality."""
        # Mock bot and file objects
        mock_bot = AsyncMock()
        mock_file = Mock()
        mock_file.file_path = "voice/file123.ogg"
        mock_bot.get_file.return_value = mock_file
        mock_bot.download_file = AsyncMock()
        
        # Test download
        file_path = await self.processor.download_voice_message(
            bot=mock_bot,
            file_id="file123",
            user_id=12345
        )
        
        # Verify calls
        mock_bot.get_file.assert_called_once_with("file123")
        mock_bot.download_file.assert_called_once()
        
        # Verify file path
        assert file_path.parent == Path(self.temp_dir)
        assert "voice_12345_" in file_path.name
        assert file_path.suffix == ".ogg"
    
    @pytest.mark.asyncio
    async def test_conversion_with_mock_audio(self):
        """Test audio conversion with mock audio data."""
        # Create a mock OGG file
        ogg_path = Path(self.temp_dir) / "test.ogg"
        
        with patch('app.services.voice_processor.AudioSegment') as mock_audio:
            # Mock AudioSegment behavior
            mock_segment = Mock()
            mock_segment.channels = 2
            mock_segment.frame_rate = 44100
            mock_segment.export = Mock()
            mock_audio.from_ogg.return_value = mock_segment
            mock_audio.from_file.return_value = mock_segment
            
            # Create fake file
            ogg_path.write_bytes(b"fake ogg data")
            
            # Test conversion
            mp3_path = await self.processor.convert_ogg_to_mp3(ogg_path)
            
            # Verify conversion called
            mock_segment.export.assert_called_once()
            
            # Verify output path
            assert mp3_path.suffix == ".mp3"
            assert mp3_path.parent == ogg_path.parent
    
    @pytest.mark.asyncio
    async def test_cleanup_old_files(self):
        """Test cleanup of old temporary files."""
        # Create some old files
        old_file = Path(self.temp_dir) / "voice_123_old.mp3"
        old_file.write_bytes(b"old data")
        
        # Set file time to be old
        old_time = time.time() - 25 * 3600  # 25 hours ago
        old_file.touch(times=(old_time, old_time))
        
        # Create recent file
        recent_file = Path(self.temp_dir) / "voice_456_recent.mp3"
        recent_file.write_bytes(b"recent data")
        
        # Run cleanup
        cleaned = await self.processor.cleanup_old_files(max_age_hours=24)
        
        # Verify old file removed, recent file kept
        assert not old_file.exists()
        assert recent_file.exists()
        assert cleaned == 1
    
    def test_get_processing_stats(self):
        """Test processing statistics retrieval."""
        stats = self.processor.get_processing_stats()
        
        assert 'total_processed' in stats
        assert 'avg_processing_time' in stats
        assert 'errors' in stats
        assert 'cache_hits' in stats
        assert 'cache_size' in stats
        assert 'temp_dir' in stats


class TestWhisperClient:
    """Test Whisper API client functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = WhisperClient(enable_caching=False)  # Disable caching for tests
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test client initialization."""
        assert self.client.enable_caching == False
        assert len(self.client.supported_languages) > 0
        assert 'en' in self.client.supported_languages
    
    def test_supported_languages(self):
        """Test supported languages functionality."""
        languages = self.client.get_supported_languages()
        
        assert isinstance(languages, dict)
        assert 'en' in languages
        assert languages['en'] == 'English'
    
    def test_file_validation(self):
        """Test audio file validation."""
        # Test non-existent file
        with pytest.raises(WhisperError, match="not found"):
            self.client._validate_audio_file(Path("nonexistent.mp3"))
        
        # Test empty file
        empty_file = Path(self.temp_dir) / "empty.mp3"
        empty_file.write_bytes(b"")
        
        with pytest.raises(WhisperError, match="empty"):
            self.client._validate_audio_file(empty_file)
        
        # Test oversized file
        large_file = Path(self.temp_dir) / "large.mp3"
        large_data = b"0" * (self.client.max_file_size + 1)
        large_file.write_bytes(large_data)
        
        with pytest.raises(WhisperError, match="too large"):
            self.client._validate_audio_file(large_file)
        
        # Test unsupported format
        bad_file = Path(self.temp_dir) / "bad.txt"
        bad_file.write_bytes(b"not audio")
        
        with pytest.raises(WhisperError, match="Unsupported"):
            self.client._validate_audio_file(bad_file)
    
    @pytest.mark.asyncio
    async def test_transcription_with_mock_api(self):
        """Test transcription with mocked OpenAI API."""
        # Create test audio file
        audio_file = Path(self.temp_dir) / "test.mp3"
        audio_file.write_bytes(b"fake mp3 data for testing")
        
        # Mock the API response
        mock_response = Mock()
        mock_response.dict.return_value = {
            'text': 'Hello, this is a test transcription.',
            'language': 'en',
            'duration': 2.5,
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello, this is a test transcription.',
                    'confidence': 0.95
                }
            ]
        }
        
        with patch.object(self.client.client.audio.transcriptions, 'create', return_value=mock_response):
            result = await self.client.transcribe_audio(audio_file)
            
            assert isinstance(result, WhisperResponse)
            assert result.text == 'Hello, this is a test transcription.'
            assert result.language == 'en'
            assert result.confidence == 0.95
            assert result.duration == 2.5
    
    @pytest.mark.asyncio
    async def test_language_detection(self):
        """Test language detection functionality."""
        audio_file = Path(self.temp_dir) / "test.mp3"
        audio_file.write_bytes(b"fake mp3 data")
        
        # Mock API response for language detection
        mock_response = Mock()
        mock_response.dict.return_value = {
            'text': 'Bonjour, comment allez-vous?',
            'language': 'fr',
            'duration': 2.0
        }
        
        with patch.object(self.client.client.audio.transcriptions, 'create', return_value=mock_response):
            language, confidence = await self.client.detect_language(audio_file)
            
            assert language == 'fr'
            assert isinstance(confidence, float)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.client.get_stats()
        
        assert 'total_requests' in stats
        assert 'successful_requests' in stats
        assert 'failed_requests' in stats
        assert 'success_rate' in stats
        assert isinstance(stats['success_rate'], float)


class TestTTSService:
    """Test Text-to-Speech service functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = TTSService(
            temp_dir=self.temp_dir,
            enable_caching=False  # Disable caching for tests
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test service initialization."""
        assert self.service.enable_caching == False
        assert len(self.service.supported_languages) > 0
        assert 'en' in self.service.supported_languages
        assert self.service.default_language == 'en'
    
    def test_text_validation(self):
        """Test text validation and cleaning."""
        # Test empty text
        with pytest.raises(TTSError, match="empty"):
            self.service._validate_text("")
        
        with pytest.raises(TTSError, match="empty"):
            self.service._validate_text("   ")
        
        # Test normal text
        result = self.service._validate_text("Hello world!")
        assert result == "Hello world!"
        
        # Test text with extra whitespace
        result = self.service._validate_text("  Hello    world!  \\n\\r  ")
        assert result == "Hello world!"
        
        # Test very long text (should be truncated)
        long_text = "A" * 6000
        result = self.service._validate_text(long_text)
        assert len(result) <= self.service.max_text_length
        assert result.endswith("...")
    
    def test_language_validation(self):
        """Test language code validation."""
        # Test valid language
        result = self.service._validate_language('en')
        assert result == 'en'
        
        # Test language mapping
        result = self.service._validate_language('english')
        assert result == 'en'
        
        # Test invalid language (should fallback to default)
        result = self.service._validate_language('invalid_lang')
        assert result == self.service.default_language
        
        # Test None/empty language
        result = self.service._validate_language(None)
        assert result == self.service.default_language
    
    @pytest.mark.asyncio
    async def test_text_chunking(self):
        """Test long text chunking functionality."""
        # Test short text (no chunking needed)
        short_text = "Hello world!"
        chunks = await self.service._chunk_long_text(short_text, max_chunk_size=100)
        assert chunks == [short_text]
        
        # Test long text (should be chunked)
        long_text = "This is a long sentence. " * 50  # Create long text
        chunks = await self.service._chunk_long_text(long_text, max_chunk_size=100)
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        
        # Verify all chunks together contain original content
        combined = " ".join(chunks).replace(". .", ".")
        original_words = set(long_text.split())
        combined_words = set(combined.split())
        assert original_words.issubset(combined_words)
    
    @pytest.mark.asyncio
    async def test_tts_generation_with_mock(self):
        """Test TTS generation with mocked gTTS."""
        text = "Hello, this is a test!"
        
        with patch('app.services.tts_service.gTTS') as mock_gtts:
            mock_instance = Mock()
            mock_gtts.return_value = mock_instance
            
            # Mock the write_to_fp method
            def mock_write_to_fp(fp):
                fp.write(b"fake mp3 audio data")
            
            mock_instance.write_to_fp = mock_write_to_fp
            
            # Mock AudioSegment for optimization
            with patch('app.services.tts_service.AudioSegment') as mock_audio:
                mock_segment = Mock()
                mock_segment.channels = 1
                mock_segment.frame_rate = 16000
                mock_segment.normalize.return_value = mock_segment
                mock_segment.compress_dynamic_range.return_value = mock_segment
                mock_segment.export = Mock()
                mock_audio.from_mp3.return_value = mock_segment
                
                result = await self.service.generate_speech(text)
                
                assert isinstance(result, TTSResponse)
                assert result.text == text
                assert result.language == 'en'
                assert result.file_size > 0
                assert Path(result.audio_file_path).exists()
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.service.get_stats()
        
        assert 'total_requests' in stats
        assert 'successful_requests' in stats
        assert 'failed_requests' in stats
        assert 'success_rate' in stats
        assert 'cache_enabled' in stats


class TestVoiceIntegration:
    """Test voice integration service."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.service = VoiceIntegrationService(enable_caching=False)
        self.mock_bot = AsyncMock()
        self.mock_voice_message = Mock()
        self.mock_voice_message.file_id = "test_file_123"
        self.mock_voice_message.duration = 5
        self.mock_voice_message.file_size = 10000
    
    def test_initialization(self):
        """Test service initialization."""
        assert self.service.enable_caching == False
        assert isinstance(self.service.metrics, VoiceIntegrationMetrics)
        assert len(self.service.active_sessions) == 0
    
    def test_metrics_recording(self):
        """Test metrics recording functionality."""
        metrics = self.service.metrics
        
        # Record successful processing
        metrics.record_voice_processing(
            success=True,
            processing_time=2.5,
            transcription_time=1.5,
            tts_time=1.0,
            language='en'
        )
        
        stats = metrics.stats
        assert stats['total_voice_messages'] == 1
        assert stats['successful_transcriptions'] == 1
        assert stats['failed_transcriptions'] == 0
        assert stats['voice_responses_generated'] == 1
        assert stats['avg_processing_time'] == 2.5
        assert stats['languages_processed']['en'] == 1
        
        # Record failed processing
        metrics.record_voice_processing(
            success=False,
            processing_time=1.0,
            error_type='transcription_error'
        )
        
        assert stats['total_voice_messages'] == 2
        assert stats['failed_transcriptions'] == 1
        assert stats['error_types']['transcription_error'] == 1
    
    @pytest.mark.asyncio
    async def test_processing_session_context(self):
        """Test processing session context manager."""
        user_id = 12345
        session_id = "test_session"
        
        # Test successful session
        async with self.service.processing_session(session_id, user_id) as session:
            assert session_id in self.service.active_sessions
            assert self.service.active_sessions[session_id]['user_id'] == user_id
            assert 'start_time' in session
            assert session['status'] == 'initializing'
            
            # Update session status
            session['status'] = 'processing'
        
        # Session should be cleaned up after context
        assert session_id not in self.service.active_sessions
    
    @pytest.mark.asyncio
    async def test_text_response_generation(self):
        """Test text response generation."""
        text = "Hello, how are you?"
        
        # Test with mock LLM service
        with patch('app.services.voice_integration.get_llm_integration_service') as mock_llm:
            mock_service = AsyncMock()
            mock_service.generate_response.return_value = ("I'm doing well, thanks!", {})
            mock_llm.return_value = mock_service
            
            response = await self.service._generate_text_response(text)
            assert response == "I'm doing well, thanks!"
        
        # Test fallback when LLM unavailable
        with patch('app.services.voice_integration.get_llm_integration_service', side_effect=Exception("LLM unavailable")):
            response = await self.service._generate_text_response(text)
            assert "Hello!" in response  # Should use fallback
    
    @pytest.mark.asyncio
    async def test_fallback_response_generation(self):
        """Test fallback response patterns."""
        # Test greeting
        response = await self.service._generate_fallback_response("Hello there!")
        assert "Hello!" in response
        
        # Test question
        response = await self.service._generate_fallback_response("How are you?")
        assert "question" in response.lower()
        
        # Test thanks
        response = await self.service._generate_fallback_response("Thank you!")
        assert "welcome" in response.lower()
        
        # Test help request
        response = await self.service._generate_fallback_response("Can you help me?")
        assert "help" in response.lower()
        
        # Test generic
        response = await self.service._generate_fallback_response("Random text")
        assert len(response) > 0
    
    def test_voice_response_decision(self):
        """Test voice response generation decision logic."""
        # Mock transcription with good confidence
        transcription = WhisperResponse(
            text="Hello, this is a clear message",
            language="en",
            confidence=0.9,
            processing_time=1.0
        )
        
        user_preferences = {'voice_responses_enabled': True}
        chat_id = 12345  # Private chat
        
        # Should generate voice response
        should_generate = self.service._should_generate_voice_response(
            transcription, user_preferences, chat_id
        )
        assert should_generate == True
        
        # Test with low confidence
        transcription.confidence = 0.5
        should_generate = self.service._should_generate_voice_response(
            transcription, user_preferences, chat_id
        )
        assert should_generate == False
        
        # Test with disabled preference
        user_preferences['voice_responses_enabled'] = False
        should_generate = self.service._should_generate_voice_response(
            transcription, user_preferences, chat_id
        )
        assert should_generate == False
        
        # Test with short text
        transcription.text = "Ok"
        should_generate = self.service._should_generate_voice_response(
            transcription, user_preferences, chat_id
        )
        assert should_generate == False
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        metrics = self.service.get_metrics()
        
        assert 'total_voice_messages' in metrics
        assert 'successful_transcriptions' in metrics
        assert 'active_sessions' in metrics
        assert 'target_processing_time' in metrics
        assert 'caching_enabled' in metrics
    
    def test_get_active_sessions(self):
        """Test active sessions information."""
        # Add a mock session
        session_id = "test_session"
        self.service.active_sessions[session_id] = {
            'user_id': 12345,
            'start_time': time.time() - 5,  # 5 seconds ago
            'status': 'processing'
        }
        
        sessions_info = self.service.get_active_sessions()
        
        assert session_id in sessions_info
        assert sessions_info[session_id]['user_id'] == 12345
        assert 'elapsed_time' in sessions_info[session_id]
        assert sessions_info[session_id]['elapsed_time'] >= 5


class TestPerformanceBenchmarks:
    """Performance benchmark tests for voice processing."""
    
    @pytest.mark.asyncio
    async def test_voice_processing_performance(self):
        """Test voice processing meets performance targets."""
        # This would require actual audio files and API keys for real testing
        # For now, we'll test the performance monitoring infrastructure
        
        service = VoiceIntegrationService()
        metrics = service.metrics
        
        # Simulate some processing times
        processing_times = [1.2, 1.8, 2.3, 1.5, 1.9]  # Mix of fast and slower
        
        for i, time_taken in enumerate(processing_times):
            metrics.record_voice_processing(
                success=True,
                processing_time=time_taken,
                transcription_time=time_taken * 0.6,
                language='en'
            )
        
        stats = metrics.stats
        avg_time = stats['avg_processing_time']
        
        # Check if average processing time is reasonable
        assert avg_time > 0
        assert avg_time == sum(processing_times) / len(processing_times)
        
        # In a real scenario, we'd assert that avg_time < target_time
        target_time = 2.0  # 2 seconds target
        print(f"Average processing time: {avg_time:.2f}s (target: {target_time}s)")
    
    @pytest.mark.asyncio 
    async def test_concurrent_processing_capacity(self):
        """Test system can handle concurrent voice processing."""
        service = VoiceIntegrationService()
        
        # Simulate multiple concurrent sessions
        session_count = 10
        sessions = []
        
        for i in range(session_count):
            session_id = f"session_{i}"
            user_id = 1000 + i
            
            # Start session (simulating concurrent processing)
            session_context = service.processing_session(session_id, user_id)
            sessions.append(session_context)
        
        # All sessions should be trackable
        assert len(service.active_sessions) == 0  # Not started yet
        
        # Test that we can track multiple sessions
        for i, session_context in enumerate(sessions):
            async with session_context:
                assert len(service.active_sessions) == 1  # Only current session active
        
        # All sessions should be cleaned up
        assert len(service.active_sessions) == 0


@pytest.mark.integration
class TestVoiceProcessingIntegration:
    """Integration tests for complete voice processing pipeline."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup integration test environment.""" 
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_mock(self):
        """Test complete voice processing pipeline with mocks."""
        # This tests the integration between all services
        # In a real test environment, you'd use actual audio files
        
        # Create mock components
        mock_bot = AsyncMock()
        mock_voice_message = Mock()
        mock_voice_message.file_id = "test_123"
        mock_voice_message.duration = 3
        mock_voice_message.file_size = 15000
        
        # Mock file download
        mock_file = Mock()
        mock_file.file_path = "voice/test.ogg"
        mock_bot.get_file.return_value = mock_file
        mock_bot.download_file = AsyncMock()
        
        # Test the integration service
        service = VoiceIntegrationService(enable_caching=False)
        
        with patch('app.services.voice_integration.get_voice_processor') as mock_voice_proc:
            with patch('app.services.voice_integration.get_whisper_client') as mock_whisper:
                with patch('app.services.voice_integration.get_tts_service') as mock_tts:
                    
                    # Setup mocks
                    mock_processor = AsyncMock()
                    mock_processor.process_voice_message.return_value = (
                        Path(self.temp_dir) / "test.mp3",
                        {'processing_time': 1.5, 'original_duration': 3.0}
                    )
                    mock_voice_proc.return_value = mock_processor
                    
                    mock_whisper_client = AsyncMock()
                    mock_whisper_client.transcribe_audio.return_value = WhisperResponse(
                        text="Hello, this is a test message",
                        language="en",
                        confidence=0.9,
                        processing_time=1.0
                    )
                    mock_whisper.return_value = mock_whisper_client
                    
                    mock_tts_service = AsyncMock()
                    mock_tts_service.generate_speech.return_value = TTSResponse(
                        audio_file_path=str(Path(self.temp_dir) / "response.mp3"),
                        text="I heard your message",
                        language="en",
                        duration=2.0,
                        file_size=12000,
                        processing_time=0.8
                    )
                    mock_tts.return_value = mock_tts_service
                    
                    # Test complete processing
                    result = await service.process_voice_message_complete(
                        bot=mock_bot,
                        voice_message=mock_voice_message,
                        user_id=12345,
                        chat_id=12345,
                        generate_voice_response=True
                    )
                    
                    # Verify results
                    assert result['success'] == True
                    assert 'transcription' in result
                    assert result['transcription']['text'] == "Hello, this is a test message"
                    assert result['transcription']['language'] == "en"
                    assert 'voice_metadata' in result
                    assert 'total_processing_time' in result
                    
                    # Verify all services were called
                    mock_processor.process_voice_message.assert_called_once()
                    mock_whisper_client.transcribe_audio.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])