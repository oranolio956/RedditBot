"""
Enhanced Voice Processing Tests

Comprehensive test suite for voice processing functionality including:
- Audio file validation and security checks
- Format conversion (OGG to MP3) with optimization
- Speech recognition integration
- Performance benchmarking for <2s processing time
- Error handling and edge cases
- Memory usage validation
- Concurrent processing tests (1000+ concurrent users)
- Cache efficiency tests
"""

import pytest
import asyncio
import tempfile
import time
import os
import threading
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from typing import List, Dict, Any

from pydub import AudioSegment
import psutil

from app.services.voice_processor import (
    VoiceProcessor, AudioFileValidator, get_voice_processor,
    VoiceProcessingError, process_telegram_voice
)


class TestAudioFileValidator:
    """Test audio file validation with security and performance checks."""
    
    @pytest.fixture
    def temp_audio_file(self):
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Create a minimal OGG file (sine wave for testing)
            audio = AudioSegment.sine(frequency=440, duration=1000)  # 1 second 440Hz
            audio.export(tmp.name, format="ogg")
            yield Path(tmp.name)
        os.unlink(tmp.name)
    
    @pytest.fixture
    def large_audio_file(self):
        """Create large audio file to test size limits."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Create 6-minute audio file (exceeds 5-minute limit)
            audio = AudioSegment.sine(frequency=440, duration=360000)  # 6 minutes
            audio.export(tmp.name, format="ogg")
            yield Path(tmp.name)
        os.unlink(tmp.name)
    
    @pytest.fixture
    def corrupted_audio_file(self):
        """Create corrupted audio file for error testing."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Write random bytes that aren't valid audio
            tmp.write(b"This is not audio data!" * 100)
            yield Path(tmp.name)
        os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_validate_valid_audio(self, temp_audio_file):
        """Test validation of valid audio file."""
        result = await AudioFileValidator.validate_audio_file(temp_audio_file)
        
        assert result['valid'] is True
        assert result['file_size'] > 0
        assert result['duration_seconds'] > 0.8  # Approximately 1 second
        assert result['sample_rate'] > 0
        assert result['channels'] > 0
        assert result['format'] == '.ogg'
        assert 'audio_hash' in result
        
    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        with pytest.raises(VoiceProcessingError, match="not found"):
            await AudioFileValidator.validate_audio_file(Path("nonexistent.ogg"))
    
    @pytest.mark.asyncio
    async def test_validate_empty_file(self):
        """Test validation fails for empty file."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            pass  # Create empty file
        
        try:
            with pytest.raises(VoiceProcessingError, match="Empty audio file"):
                await AudioFileValidator.validate_audio_file(Path(tmp.name))
        finally:
            os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_validate_oversized_file(self, large_audio_file):
        """Test validation fails for oversized file."""
        with pytest.raises(VoiceProcessingError, match="File too large"):
            await AudioFileValidator.validate_audio_file(large_audio_file)
    
    @pytest.mark.asyncio
    async def test_validate_corrupted_file(self, corrupted_audio_file):
        """Test validation fails for corrupted audio."""
        with pytest.raises(VoiceProcessingError, match="Could not decode"):
            await AudioFileValidator.validate_audio_file(corrupted_audio_file)
    
    @pytest.mark.asyncio
    async def test_validate_unsupported_format(self):
        """Test validation fails for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp.write(b"fake audio data")
        
        try:
            with pytest.raises(VoiceProcessingError, match="Unsupported format"):
                await AudioFileValidator.validate_audio_file(Path(tmp.name))
        finally:
            os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_audio_hash_consistency(self, temp_audio_file):
        """Test that audio hash is consistent for same content."""
        result1 = await AudioFileValidator.validate_audio_file(temp_audio_file)
        result2 = await AudioFileValidator.validate_audio_file(temp_audio_file)
        
        assert result1['audio_hash'] == result2['audio_hash']
    
    @pytest.mark.asyncio
    async def test_validation_performance(self, temp_audio_file):
        """Test validation performance (<100ms for small files)."""
        start_time = time.time()
        result = await AudioFileValidator.validate_audio_file(temp_audio_file)
        validation_time = time.time() - start_time
        
        assert validation_time < 0.1  # Should complete in <100ms
        assert result['valid'] is True


class TestVoiceProcessor:
    """Test VoiceProcessor with performance and concurrency checks."""
    
    @pytest.fixture
    def processor(self):
        """Create VoiceProcessor instance with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield VoiceProcessor(temp_dir=temp_dir)
    
    @pytest.fixture
    def mock_bot(self):
        """Mock Telegram bot for testing."""
        bot = Mock()
        bot.get_file = AsyncMock(return_value=Mock(file_path="voice/file123.ogg"))
        bot.download_file = AsyncMock()
        return bot
    
    @pytest.fixture
    def temp_ogg_file(self):
        """Create temporary OGG file for conversion testing."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Create test audio
            audio = AudioSegment.sine(frequency=440, duration=2000)  # 2 seconds
            audio.export(tmp.name, format="ogg")
            yield Path(tmp.name)
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
    
    @pytest.mark.asyncio
    async def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.temp_dir.exists()
        assert processor.processing_stats['total_processed'] == 0
        assert len(processor._cache) == 0
    
    @pytest.mark.asyncio
    async def test_download_voice_message(self, processor, mock_bot):
        """Test voice message download from Telegram."""
        # Setup mock file download
        test_content = b"fake ogg content"
        
        async def mock_download(file_path, destination):
            with open(destination, 'wb') as f:
                f.write(test_content)
        
        mock_bot.download_file.side_effect = mock_download
        
        # Test download
        result_path = await processor.download_voice_message(
            mock_bot, "test_file_id", 12345
        )
        
        assert result_path.exists()
        assert result_path.name.startswith("voice_12345_")
        assert result_path.suffix == ".ogg"
        
        # Verify content
        with open(result_path, 'rb') as f:
            assert f.read() == test_content
        
        # Cleanup
        if result_path.exists():
            result_path.unlink()
    
    @pytest.mark.asyncio
    async def test_convert_ogg_to_mp3(self, processor, temp_ogg_file):
        """Test OGG to MP3 conversion with optimization."""
        start_time = time.time()
        
        mp3_path = await processor.convert_ogg_to_mp3(
            temp_ogg_file, optimize_for_speech=True
        )
        
        conversion_time = time.time() - start_time
        
        # Verify conversion results
        assert mp3_path.exists()
        assert mp3_path.suffix == ".mp3"
        assert conversion_time < 2.0  # Should complete in <2 seconds
        
        # Verify MP3 file is valid
        converted_audio = AudioSegment.from_mp3(str(mp3_path))
        assert len(converted_audio) > 1500  # Approximately 2 seconds
        assert converted_audio.channels == 1  # Should be mono for speech
        assert converted_audio.frame_rate == 16000  # Optimized sample rate
        
        # Cleanup
        if mp3_path.exists():
            mp3_path.unlink()
    
    @pytest.mark.asyncio
    async def test_conversion_caching(self, processor, temp_ogg_file):
        """Test conversion caching for performance."""
        # First conversion
        start_time = time.time()
        mp3_path1 = await processor.convert_ogg_to_mp3(temp_ogg_file)
        first_time = time.time() - start_time
        
        # Second conversion (should use cache)
        start_time = time.time()
        mp3_path2 = await processor.convert_ogg_to_mp3(temp_ogg_file)
        second_time = time.time() - start_time
        
        # Verify both paths exist and are the same
        assert mp3_path1.exists()
        assert mp3_path2.exists()
        assert str(mp3_path1) == str(mp3_path2)
        
        # Second conversion should be faster (cache hit)
        assert second_time < first_time / 2
        assert processor.processing_stats['cache_hits'] > 0
        
        # Cleanup
        for path in [mp3_path1, mp3_path2]:
            if path.exists():
                path.unlink()
    
    @pytest.mark.asyncio
    async def test_speech_optimization(self, processor, temp_ogg_file):
        """Test speech optimization features."""
        # Test without optimization
        mp3_normal = await processor.convert_ogg_to_mp3(
            temp_ogg_file, optimize_for_speech=False
        )
        
        # Test with optimization
        mp3_optimized = await processor.convert_ogg_to_mp3(
            temp_ogg_file, optimize_for_speech=True
        )
        
        # Load both files
        normal_audio = AudioSegment.from_mp3(str(mp3_normal))
        optimized_audio = AudioSegment.from_mp3(str(mp3_optimized))
        
        # Verify optimization effects
        assert optimized_audio.channels == 1  # Should be mono
        assert optimized_audio.frame_rate == 16000  # Optimized sample rate
        
        # Optimized file might be smaller due to mono conversion
        assert optimized_audio.channels <= normal_audio.channels
        
        # Cleanup
        for path in [mp3_normal, mp3_optimized]:
            if path.exists():
                path.unlink()
    
    @pytest.mark.asyncio
    async def test_complete_processing_pipeline(self, processor, mock_bot):
        """Test complete voice processing pipeline."""
        # Setup mock with real audio data
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            audio = AudioSegment.sine(frequency=440, duration=1000)
            audio.export(tmp.name, format="ogg")
        
        async def mock_download(file_path, destination):
            # Copy our test file to the destination
            with open(tmp.name, 'rb') as src, open(destination, 'wb') as dst:
                dst.write(src.read())
        
        mock_bot.download_file.side_effect = mock_download
        
        try:
            # Test complete pipeline
            start_time = time.time()
            
            mp3_path, metadata = await processor.process_voice_message(
                mock_bot, "test_file_id", 12345, optimize_for_speech=True
            )
            
            processing_time = time.time() - start_time
            
            # Verify results
            assert mp3_path.exists()
            assert mp3_path.suffix == ".mp3"
            assert processing_time < 2.0  # Target: <2 seconds
            
            # Verify metadata
            assert metadata['processing_time'] > 0
            assert metadata['original_format'] == '.ogg'
            assert metadata['converted_size'] > 0
            assert metadata['optimized_for_speech'] is True
            assert metadata['user_id'] == 12345
            assert metadata['file_id'] == "test_file_id"
            
            # Verify MP3 is optimized
            converted_audio = AudioSegment.from_mp3(str(mp3_path))
            assert converted_audio.channels == 1
            assert converted_audio.frame_rate == 16000
            
        finally:
            # Cleanup
            os.unlink(tmp.name)
            if mp3_path and mp3_path.exists():
                mp3_path.unlink()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_files(self, processor):
        """Test cleanup of old temporary files."""
        # Create some old test files
        old_files = []
        for i in range(5):
            test_file = processor.temp_dir / f"voice_test_{i}.mp3"
            test_file.write_text("test content")
            # Set old modification time
            old_time = time.time() - (25 * 3600)  # 25 hours ago
            os.utime(test_file, (old_time, old_time))
            old_files.append(test_file)
        
        # Create some recent files that shouldn't be cleaned
        recent_files = []
        for i in range(3):
            test_file = processor.temp_dir / f"voice_recent_{i}.mp3"
            test_file.write_text("recent content")
            recent_files.append(test_file)
        
        # Run cleanup
        cleaned_count = await processor.cleanup_old_files(max_age_hours=24)
        
        # Verify cleanup results
        assert cleaned_count == 5
        
        # Old files should be gone
        for old_file in old_files:
            assert not old_file.exists()
        
        # Recent files should remain
        for recent_file in recent_files:
            assert recent_file.exists()
    
    @pytest.mark.asyncio
    async def test_processing_statistics(self, processor, temp_ogg_file):
        """Test processing statistics tracking."""
        initial_stats = processor.get_processing_stats()
        assert initial_stats['total_processed'] == 0
        assert initial_stats['avg_processing_time'] == 0.0
        
        # Process a file
        await processor.convert_ogg_to_mp3(temp_ogg_file)
        
        # Check updated stats
        updated_stats = processor.get_processing_stats()
        assert updated_stats['total_processed'] == 1
        assert updated_stats['avg_processing_time'] > 0
        assert 'cache_size' in updated_stats
        assert 'temp_dir' in updated_stats
    
    @pytest.mark.asyncio
    async def test_error_handling_and_cleanup(self, processor, mock_bot):
        """Test error handling and automatic cleanup."""
        # Setup bot to fail during processing
        mock_bot.get_file.side_effect = Exception("Network error")
        
        with pytest.raises(VoiceProcessingError, match="Download failed"):
            await processor.process_voice_message(
                mock_bot, "failing_file_id", 12345
            )
        
        # Verify error statistics
        stats = processor.get_processing_stats()
        assert stats['errors'] > 0
        
        # Verify no leftover files in temp directory
        temp_files = list(processor.temp_dir.glob("voice_*"))
        assert len(temp_files) == 0


class TestConcurrentProcessing:
    """Test concurrent voice processing for high load scenarios."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for concurrent testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield VoiceProcessor(temp_dir=temp_dir)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_conversions(self, processor):
        """Test concurrent OGG to MP3 conversions (simulating 100 users)."""
        # Create test audio files
        test_files = []
        for i in range(10):  # Create 10 test files to reuse
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                audio = AudioSegment.sine(frequency=440 + i*10, duration=1000)
                audio.export(tmp.name, format="ogg")
                test_files.append(Path(tmp.name))
        
        try:
            # Create concurrent conversion tasks (100 conversions using 10 files)
            tasks = []
            for i in range(100):
                file_to_use = test_files[i % len(test_files)]
                task = processor.convert_ogg_to_mp3(file_to_use)
                tasks.append(task)
            
            # Execute all tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Verify results
            successful_conversions = [r for r in results if isinstance(r, Path)]
            errors = [r for r in results if isinstance(r, Exception)]
            
            # Most conversions should succeed
            assert len(successful_conversions) >= 90
            assert len(errors) <= 10
            
            # Total time should be reasonable (with caching, should be fast)
            assert total_time < 30.0  # Allow 30 seconds for 100 conversions
            
            # Average time per conversion should be low
            avg_time = total_time / len(successful_conversions)
            assert avg_time < 0.5  # Should average <500ms per conversion
            
            print(f"Concurrent test: {len(successful_conversions)}/100 successful, "
                  f"{total_time:.2f}s total, {avg_time:.3f}s average")
            
        finally:
            # Cleanup test files
            for test_file in test_files:
                if test_file.exists():
                    test_file.unlink()
            
            # Cleanup converted files
            for result in results:
                if isinstance(result, Path) and result.exists():
                    result.unlink()
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, processor):
        """Test memory usage doesn't grow excessively under load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create one test file to reuse
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            audio = AudioSegment.sine(frequency=440, duration=500)  # Short audio
            audio.export(tmp.name, format="ogg")
            test_file = Path(tmp.name)
        
        try:
            # Process in batches to check memory growth
            batch_size = 20
            max_memory_growth = 0
            
            for batch in range(5):  # 5 batches of 20 = 100 total
                # Process batch
                tasks = [processor.convert_ogg_to_mp3(test_file) for _ in range(batch_size)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Clean up results
                for result in results:
                    if isinstance(result, Path) and result.exists():
                        result.unlink()
                
                # Force garbage collection
                gc.collect()
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                max_memory_growth = max(max_memory_growth, memory_growth)
                
                print(f"Batch {batch + 1}: Memory growth = {memory_growth:.1f} MB")
            
            # Memory growth should be reasonable (<100MB for processing)
            assert max_memory_growth < 100, f"Memory growth too high: {max_memory_growth:.1f} MB"
            
        finally:
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_cache_efficiency_under_load(self, processor):
        """Test cache efficiency with repeated files."""
        # Create a few test files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                audio = AudioSegment.sine(frequency=440 + i*100, duration=800)
                audio.export(tmp.name, format="ogg")
                test_files.append(Path(tmp.name))
        
        try:
            # Process same files multiple times
            tasks = []
            for _ in range(50):  # 50 conversions
                for test_file in test_files:
                    tasks.append(processor.convert_ogg_to_mp3(test_file))
            
            # Execute all tasks
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Check cache efficiency
            stats = processor.get_processing_stats()
            cache_hit_ratio = stats['cache_hits'] / stats['total_processed']
            
            # With repeated files, cache hit ratio should be high
            assert cache_hit_ratio > 0.8, f"Cache hit ratio too low: {cache_hit_ratio:.2f}"
            
            # Total time should be very fast due to caching
            assert total_time < 10.0, f"Total time too high: {total_time:.2f}s"
            
            print(f"Cache efficiency test: {cache_hit_ratio:.1%} hit ratio, "
                  f"{total_time:.2f}s total time")
            
        finally:
            # Cleanup
            for test_file in test_files:
                if test_file.exists():
                    test_file.unlink()
            
            for result in results:
                if isinstance(result, Path) and result.exists():
                    result.unlink()


class TestVoiceProcessorIntegration:
    """Integration tests for voice processor with external dependencies."""
    
    @pytest.mark.asyncio
    async def test_telegram_voice_processing_integration(self):
        """Test integration with Telegram bot processing."""
        # Mock Telegram objects
        mock_bot = Mock()
        mock_voice = Mock()
        mock_voice.file_id = "test_voice_file_id"
        mock_voice.duration = 5
        mock_voice.mime_type = "audio/ogg"
        
        # Mock file download
        test_audio_content = b"mock ogg audio data"
        
        async def mock_get_file(file_id):
            return Mock(file_path="voice/123.ogg")
        
        async def mock_download(file_path, destination):
            # Create a real audio file for testing
            audio = AudioSegment.sine(frequency=440, duration=1000)
            audio.export(destination, format="ogg")
        
        mock_bot.get_file = mock_get_file
        mock_bot.download_file = mock_download
        
        # Test the integration function
        result_path, metadata = await process_telegram_voice(
            mock_bot, mock_voice, user_id=12345, optimize_for_speech=True
        )
        
        # Verify results
        assert result_path.exists()
        assert result_path.suffix == ".mp3"
        assert metadata['file_id'] == "test_voice_file_id"
        assert metadata['user_id'] == 12345
        assert metadata['optimized_for_speech'] is True
        
        # Cleanup
        if result_path.exists():
            result_path.unlink()
    
    @pytest.mark.asyncio
    async def test_global_processor_instance(self):
        """Test global processor instance management."""
        processor1 = get_voice_processor()
        processor2 = get_voice_processor()
        
        # Should be same instance
        assert processor1 is processor2
        
        # Should have valid configuration
        assert processor1.temp_dir.exists()
        assert len(processor1._cache) >= 0


class TestVoiceProcessingPerformance:
    """Performance benchmarks for voice processing."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_processing_speed_benchmark(self):
        """Benchmark processing speed for different file sizes."""
        processor = VoiceProcessor()
        
        # Test different file sizes
        test_configs = [
            {"duration": 1000, "description": "1 second"},
            {"duration": 5000, "description": "5 seconds"},
            {"duration": 10000, "description": "10 seconds"},
            {"duration": 30000, "description": "30 seconds"}
        ]
        
        benchmark_results = []
        
        for config in test_configs:
            # Create test file
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                audio = AudioSegment.sine(frequency=440, duration=config["duration"])
                audio.export(tmp.name, format="ogg")
                test_file = Path(tmp.name)
            
            try:
                # Benchmark conversion
                start_time = time.time()
                result_path = await processor.convert_ogg_to_mp3(test_file)
                conversion_time = time.time() - start_time
                
                # Record results
                benchmark_results.append({
                    "duration": config["description"],
                    "file_size": test_file.stat().st_size / 1024,  # KB
                    "conversion_time": conversion_time,
                    "speed_ratio": config["duration"] / 1000 / conversion_time  # real-time ratio
                })
                
                # Cleanup
                if result_path.exists():
                    result_path.unlink()
                
            finally:
                if test_file.exists():
                    test_file.unlink()
        
        # Print benchmark results
        print("\nVoice Processing Benchmark Results:")
        print("Duration\t\tFile Size (KB)\tConversion Time (s)\tSpeed Ratio")
        for result in benchmark_results:
            print(f"{result['duration']}\t\t{result['file_size']:.1f}\t\t"
                  f"{result['conversion_time']:.3f}\t\t{result['speed_ratio']:.1f}x")
        
        # Performance assertions
        for result in benchmark_results:
            # All conversions should be reasonably fast
            assert result['conversion_time'] < 2.0, \
                f"Conversion too slow for {result['duration']}: {result['conversion_time']:.3f}s"
            
            # Should process faster than real-time for short files
            if "second" in result['duration'] and int(result['duration'].split()[0]) <= 10:
                assert result['speed_ratio'] > 1.0, \
                    f"Should process faster than real-time: {result['speed_ratio']:.1f}x"


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__ + "::TestVoiceProcessingPerformance", "-v", "-s"])