# Voice Message Processing System

A comprehensive voice processing system for Telegram bots with speech-to-text transcription, AI response generation, and text-to-speech synthesis.

## Features

### 🎤 Voice Processing
- **High-performance audio conversion**: OGG to MP3 with speech optimization
- **Audio validation**: File size, duration, and format validation
- **Performance optimization**: Target <2 second processing time
- **Temporary file management**: Automatic cleanup with configurable retention

### 🗣️ Speech-to-Text (Whisper)
- **OpenAI Whisper API integration**: Industry-leading speech recognition
- **Language detection**: Automatic language detection with 50+ languages supported
- **Caching system**: Redis-based caching for performance
- **Retry logic**: Exponential backoff with intelligent error handling
- **Confidence scoring**: Quality assessment for transcriptions

### 🔊 Text-to-Speech (gTTS)
- **Google Text-to-Speech**: Natural voice synthesis
- **Multi-language support**: 40+ languages supported
- **Smart text chunking**: Handles long text automatically
- **Audio optimization**: Telegram-optimized voice messages
- **Caching**: Efficient storage of generated voice files

### 🤖 Intelligent Integration
- **Complete pipeline**: End-to-end voice processing
- **AI response generation**: Integration with LLM services
- **User preferences**: Customizable voice response settings
- **Performance monitoring**: Detailed metrics and analytics
- **Error handling**: Graceful fallbacks and user feedback

## Installation

### 1. System Dependencies

Run the automated setup script:

```bash
python setup_voice.py
```

Or install manually:

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

**Windows:**
- Download FFmpeg from https://ffmpeg.org/download.html
- Add to PATH environment variable

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

The voice processing system requires:
- `pydub==0.25.1` - Audio processing
- `gtts==2.4.0` - Text-to-speech
- `ffmpeg-python==0.2.0` - FFmpeg integration
- `openai>=1.3.7` - Whisper API client
- `aiofiles>=23.2.1` - Async file operations

### 3. Configuration

Add to your `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Voice Processing Settings
ENABLE_VOICE_PROCESSING=true
ENABLE_VOICE_RESPONSES=true
MAX_AUDIO_FILE_SIZE=26214400        # 25MB
MAX_AUDIO_DURATION=600              # 10 minutes
TARGET_PROCESSING_TIME=2.0          # 2 seconds

# TTS Configuration
TTS_DEFAULT_LANGUAGE=en
TTS_MAX_TEXT_LENGTH=5000
VOICE_RESPONSE_MAX_LENGTH=500       # characters

# Caching (requires Redis)
ENABLE_TRANSCRIPTION_CACHE=true
ENABLE_TTS_CACHE=true
TRANSCRIPTION_CACHE_TTL=86400       # 24 hours
TTS_CACHE_TTL=604800                # 7 days

# Audio Quality
OUTPUT_SAMPLE_RATE=16000            # 16kHz for speech
OUTPUT_BITRATE=64k
ENABLE_AUDIO_COMPRESSION=true

# Voice Response Settings
VOICE_QUIET_HOURS_START=22          # 10 PM
VOICE_QUIET_HOURS_END=7             # 7 AM
ENABLE_VOICE_IN_GROUPS=false
```

## Usage

### Basic Voice Message Handling

The voice processing system is automatically integrated into the Telegram handlers. Users can:

1. **Send voice messages** - Automatically transcribed and processed
2. **Upload audio files** - Support for longer audio content
3. **Receive voice responses** - AI-generated voice replies (optional)

### Programmatic Usage

#### Voice Processor

```python
from app.services.voice_processor import get_voice_processor

# Initialize processor
processor = get_voice_processor()

# Process voice message
mp3_path, metadata = await processor.process_voice_message(
    bot=bot,
    file_id="voice_file_id",
    user_id=12345,
    optimize_for_speech=True
)

print(f"Processed in {metadata['processing_time']:.2f}s")
print(f"Duration: {metadata['original_duration']}s")
```

#### Whisper Client

```python
from app.services.whisper_client import get_whisper_client

# Initialize client
client = await get_whisper_client()

# Transcribe audio
result = await client.transcribe_audio(
    audio_file_path="path/to/audio.mp3",
    language="en",  # or None for auto-detection
    use_cache=True
)

print(f"Transcribed: {result.text}")
print(f"Language: {result.language}")
print(f"Confidence: {result.confidence}")
```

#### TTS Service

```python
from app.services.tts_service import get_tts_service

# Initialize service
tts = await get_tts_service()

# Generate speech
response = await tts.generate_speech(
    text="Hello, this is a test message!",
    language="en",
    optimize_for_telegram=True
)

print(f"Audio file: {response.audio_file_path}")
print(f"Duration: {response.duration}s")
```

#### Complete Integration

```python
from app.services.voice_integration import get_voice_integration_service

# Initialize integration service
service = await get_voice_integration_service()

# Process complete voice message pipeline
result = await service.process_voice_message_complete(
    bot=bot,
    voice_message=voice_message,
    user_id=user_id,
    chat_id=chat_id,
    generate_voice_response=True
)

if result['success']:
    print(f"Transcription: {result['transcription']['text']}")
    if 'voice_response' in result:
        print(f"Generated voice response: {result['voice_response']['voice_file_path']}")
```

## Performance Optimization

### Target Performance Metrics

- **Voice Processing**: <2 seconds for 1-minute audio
- **Transcription**: <1 second average response time (with caching)
- **TTS Generation**: <1 second for short responses
- **Memory Usage**: Efficient temporary file management
- **Cache Hit Rate**: >80% for frequently used content

### Optimization Features

1. **Caching Strategy**:
   - Transcription cache: 24-hour TTL
   - TTS cache: 7-day TTL
   - Smart cache keys based on audio content hash

2. **Audio Processing**:
   - Speech-optimized conversion (16kHz, mono)
   - Dynamic range compression
   - Noise reduction filters

3. **Concurrent Processing**:
   - Configurable concurrent session limits
   - Session tracking and management
   - Performance monitoring

### Monitoring

```python
# Get processing statistics
processor = get_voice_processor()
stats = processor.get_processing_stats()

print(f"Total processed: {stats['total_processed']}")
print(f"Average time: {stats['avg_processing_time']:.2f}s")
print(f"Cache hit rate: {stats['cache_hits']}/{stats['total_processed']}")

# Get integration metrics
integration = await get_voice_integration_service()
metrics = integration.get_metrics()

print(f"Success rate: {metrics['successful_transcriptions']}/{metrics['total_voice_messages']}")
print(f"Languages processed: {metrics['languages_processed']}")
```

## Error Handling

The system provides comprehensive error handling with user-friendly messages:

### Error Types

1. **VoiceProcessingError**: Audio conversion/processing issues
2. **WhisperError**: Speech recognition failures
3. **TTSError**: Text-to-speech generation problems
4. **Validation errors**: File size, format, or duration limits exceeded

### Fallback Mechanisms

1. **Transcription fallbacks**: Retry logic with exponential backoff
2. **TTS fallbacks**: Alternative voice generation methods
3. **Response fallbacks**: Text-only responses when voice generation fails
4. **Graceful degradation**: Service continues with reduced functionality

### User Feedback

- Clear error messages without technical details
- Processing status indicators
- Helpful suggestions for resolution

## Configuration Options

### Audio Quality Settings

```python
# In settings.py
class VoiceProcessingSettings(BaseSettings):
    # File limits
    max_audio_file_size: int = 25 * 1024 * 1024  # 25MB
    max_audio_duration: int = 600  # 10 minutes
    
    # Quality settings
    output_sample_rate: int = 16000  # 16kHz
    output_bitrate: str = "64k"
    enable_audio_compression: bool = True
    
    # Performance
    target_processing_time: float = 2.0
    max_concurrent_voice_processing: int = 10
```

### User Preferences

```python
# User-specific settings
user_preferences = {
    'language': 'en',  # Preferred language
    'voice_responses_enabled': True,  # Enable voice replies
    'transcription_quality': 'high',  # Quality vs speed tradeoff
}
```

## API Reference

### VoiceProcessor

#### Methods

- `process_voice_message(bot, file_id, user_id, optimize_for_speech=True)` - Complete voice processing
- `download_voice_message(bot, file_id, user_id)` - Download voice file
- `convert_ogg_to_mp3(input_path, optimize_for_speech=True)` - Audio conversion
- `cleanup_old_files(max_age_hours=24)` - File cleanup
- `get_processing_stats()` - Performance statistics

### WhisperClient

#### Methods

- `transcribe_audio(audio_file_path, language=None, use_cache=True)` - Speech-to-text
- `detect_language(audio_file_path)` - Language detection
- `transcribe_with_timestamps(audio_file_path, word_timestamps=False)` - Detailed transcription
- `get_supported_languages()` - Available languages
- `cleanup_cache()` - Cache maintenance

### TTSService

#### Methods

- `generate_speech(text, language=None, optimize_for_telegram=True)` - Text-to-speech
- `get_supported_languages()` - Available languages
- `cleanup_old_files(max_age_hours=24)` - File cleanup
- `get_stats()` - Service statistics

### VoiceIntegrationService

#### Methods

- `process_voice_message_complete(bot, voice_message, user_id, chat_id, user_preferences=None, generate_voice_response=True)` - Complete pipeline
- `generate_voice_response(text, response_language=None, user_preferences=None)` - Voice response generation
- `get_metrics()` - Processing metrics
- `get_active_sessions()` - Current processing sessions
- `cleanup_old_files(max_age_hours=2)` - Cleanup all temporary files

## Security Considerations

### Input Validation

- File size and duration limits
- Audio format validation
- Content security scanning
- Rate limiting per user

### Data Privacy

- Temporary file encryption
- Automatic file cleanup
- No persistent audio storage
- Configurable cache retention

### API Security

- Secure API key management
- Request authentication
- Error message sanitization
- Audit logging

## Testing

### Running Tests

```bash
# Run all voice processing tests
pytest tests/test_voice_processing.py -v

# Run specific test categories
pytest tests/test_voice_processing.py::TestVoiceProcessor -v
pytest tests/test_voice_processing.py::TestWhisperClient -v
pytest tests/test_voice_processing.py::TestTTSService -v

# Run performance benchmarks
pytest tests/test_voice_processing.py::TestPerformanceBenchmarks -v

# Run integration tests (requires API keys)
pytest tests/test_voice_processing.py::TestVoiceProcessingIntegration -v
```

### Test Coverage

The test suite covers:
- Unit tests for all service components
- Integration tests for complete pipeline
- Performance benchmarks
- Error handling scenarios
- Mock-based testing for CI/CD

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```bash
   # Check FFmpeg installation
   ffmpeg -version
   
   # Install if missing (macOS)
   brew install ffmpeg
   ```

2. **OpenAI API errors**:
   - Verify API key is set correctly
   - Check API usage limits
   - Ensure sufficient credits

3. **Audio processing failures**:
   - Check file format (supported: MP3, OGG, WAV, M4A)
   - Verify file size limits
   - Check audio file integrity

4. **Performance issues**:
   - Enable Redis caching
   - Check concurrent processing limits
   - Monitor system resources

### Debug Mode

Enable detailed logging:

```python
import structlog

# Enable debug logging
logger = structlog.get_logger(__name__)
logger.setLevel("DEBUG")
```

### Monitoring Endpoints

```python
# Health check
GET /api/v1/voice/health

# Performance metrics
GET /api/v1/voice/metrics

# Active sessions
GET /api/v1/voice/sessions
```

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies
3. Set up test environment with API keys
4. Run test suite
5. Follow code style guidelines

### Code Style

- Use `black` for formatting
- Follow `pylint` recommendations
- Add type hints for all functions
- Include comprehensive docstrings
- Write tests for new features

## License

This voice processing system is part of the Telegram Bot framework and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review test examples
3. Check system logs for errors
4. Create detailed issue reports with:
   - Error messages
   - System configuration
   - Reproduction steps
   - Performance metrics