#!/usr/bin/env python3
"""
Voice Processing Setup Script

Installs and configures all dependencies for voice message processing:
- System dependencies (ffmpeg)
- Python packages
- Configuration validation
- Service initialization
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json


def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return None


def check_system_dependencies():
    """Check if required system dependencies are installed."""
    print("🔍 Checking system dependencies...")
    
    dependencies = {
        'ffmpeg': 'ffmpeg -version',
        'git': 'git --version',
        'python3': 'python3 --version'
    }
    
    missing = []
    
    for name, cmd in dependencies.items():
        result = run_command(cmd, check=False)
        if result and result.returncode == 0:
            print(f"  ✅ {name} is installed")
        else:
            print(f"  ❌ {name} is missing")
            missing.append(name)
    
    return missing


def install_system_dependencies():
    """Install missing system dependencies based on platform."""
    print("\n📦 Installing system dependencies...")
    
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("Installing on macOS using Homebrew...")
        
        # Check if Homebrew is installed
        if not run_command("which brew", check=False):
            print("Installing Homebrew first...")
            install_brew_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            run_command(install_brew_cmd, check=False, capture_output=False)
        
        # Install ffmpeg
        run_command("brew install ffmpeg", check=False, capture_output=False)
        
    elif system == "linux":
        print("Installing on Linux...")
        
        # Detect Linux distribution
        if Path("/etc/debian_version").exists():  # Debian/Ubuntu
            run_command("sudo apt-get update", check=False, capture_output=False)
            run_command("sudo apt-get install -y ffmpeg", check=False, capture_output=False)
            
        elif Path("/etc/redhat-release").exists():  # RHEL/CentOS/Fedora
            run_command("sudo yum install -y ffmpeg", check=False, capture_output=False)
            
        else:
            print("  ⚠️  Unknown Linux distribution. Please install ffmpeg manually.")
            
    elif system == "windows":
        print("Windows detected.")
        print("  ⚠️  Please install ffmpeg manually from: https://ffmpeg.org/download.html")
        print("  📖 Add ffmpeg to your PATH environment variable")
        
    else:
        print(f"  ⚠️  Unknown operating system: {system}")
        print("  📖 Please install ffmpeg manually")


def install_python_dependencies():
    """Install Python packages for voice processing."""
    print("\n🐍 Installing Python dependencies...")
    
    # Voice processing specific packages
    voice_packages = [
        "pydub==0.25.1",
        "gtts==2.4.0",
        "ffmpeg-python==0.2.0",
        "openai>=1.3.7",
        "requests>=2.31.0",
        "aiofiles>=23.2.1"
    ]
    
    for package in voice_packages:
        print(f"  📦 Installing {package}...")
        result = run_command(f"pip install {package}", check=False, capture_output=False)
        if result and result.returncode != 0:
            print(f"    ❌ Failed to install {package}")
        else:
            print(f"    ✅ Installed {package}")


def create_voice_config():
    """Create default voice processing configuration."""
    print("\n⚙️  Creating voice processing configuration...")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    voice_config = {
        "voice_processing": {
            "enabled": True,
            "openai_api_key": "${OPENAI_API_KEY}",
            "max_audio_file_size_mb": 25,
            "max_audio_duration_seconds": 600,
            "target_processing_time_seconds": 2.0,
            "enable_caching": True,
            "enable_voice_responses": True,
            "default_language": "en",
            "quiet_hours": {
                "start": 22,
                "end": 7,
                "enabled": True
            },
            "audio_quality": {
                "sample_rate": 16000,
                "bitrate": "64k",
                "optimize_for_speech": True
            }
        },
        "tts_settings": {
            "default_language": "en",
            "max_text_length": 5000,
            "slow_speech": False,
            "chunk_long_text": True
        },
        "transcription_settings": {
            "model": "whisper-1",
            "enable_language_detection": True,
            "confidence_threshold": 0.7
        }
    }
    
    config_file = config_dir / "voice_config.json"
    
    with open(config_file, 'w') as f:
        json.dump(voice_config, f, indent=2)
    
    print(f"  ✅ Created voice configuration: {config_file}")
    return config_file


def create_environment_template():
    """Create environment variable template for voice processing."""
    print("\n📝 Creating environment template...")
    
    env_template = """
# Voice Processing Configuration
ENABLE_VOICE_PROCESSING=true
OPENAI_API_KEY=your_openai_api_key_here

# Audio Processing Limits
MAX_AUDIO_FILE_SIZE=26214400  # 25MB in bytes
MAX_AUDIO_DURATION=600        # 10 minutes in seconds
MAX_VOICE_MESSAGE_DURATION=300 # 5 minutes in seconds

# TTS Configuration  
ENABLE_VOICE_RESPONSES=true
TTS_DEFAULT_LANGUAGE=en
TTS_MAX_TEXT_LENGTH=5000

# Performance Settings
TARGET_PROCESSING_TIME=2.0    # 2 seconds target
MAX_CONCURRENT_VOICE_PROCESSING=10

# Caching Settings
ENABLE_TRANSCRIPTION_CACHE=true
ENABLE_TTS_CACHE=true
TRANSCRIPTION_CACHE_TTL=86400  # 24 hours
TTS_CACHE_TTL=604800           # 7 days

# Voice Response Settings
VOICE_RESPONSE_MAX_LENGTH=500  # characters
VOICE_QUIET_HOURS_START=22     # 10 PM
VOICE_QUIET_HOURS_END=7        # 7 AM
ENABLE_VOICE_IN_GROUPS=false

# Audio Quality
OUTPUT_SAMPLE_RATE=16000       # 16kHz for speech
OUTPUT_BITRATE=64k             # Good quality for voice
ENABLE_AUDIO_COMPRESSION=true

# Error Handling
MAX_TRANSCRIPTION_RETRIES=3
ENABLE_FALLBACK_TTS=true
ENABLE_SPEECH_RECOGNITION_FALLBACK=false

# Security
VALIDATE_AUDIO_CONTENT=true
SCAN_FOR_MALICIOUS_AUDIO=true

# Monitoring
ENABLE_VOICE_METRICS=true
LOG_VOICE_PROCESSING_STATS=true
"""
    
    env_file = Path(".env.voice.template")
    
    with open(env_file, 'w') as f:
        f.write(env_template.strip())
    
    print(f"  ✅ Created environment template: {env_file}")
    print(f"  📖 Copy this to your .env file and configure your API keys")
    
    return env_file


def test_voice_processing():
    """Test if voice processing components work correctly."""
    print("\n🧪 Testing voice processing components...")
    
    try:
        # Test pydub import
        from pydub import AudioSegment
        print("  ✅ pydub import successful")
        
        # Test gTTS import
        from gtts import gTTS
        print("  ✅ gTTS import successful")
        
        # Test ffmpeg integration
        try:
            # Create a simple test audio
            test_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
            print("  ✅ pydub AudioSegment creation successful")
            
            # Test export (this will use ffmpeg internally)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
                test_audio.export(tmp.name, format="mp3")
                print("  ✅ Audio export (ffmpeg) successful")
                
        except Exception as e:
            print(f"  ❌ Audio processing test failed: {e}")
            return False
        
        # Test OpenAI client import
        try:
            import openai
            print("  ✅ OpenAI client import successful")
        except ImportError:
            print("  ⚠️  OpenAI client not installed, but this is optional")
        
        print("\n✅ All voice processing components are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Voice processing test failed: {e}")
        return False


def show_next_steps():
    """Show next steps for completing voice setup."""
    print("\n🎯 Next Steps:")
    print("1. 🔑 Add your OpenAI API key to .env file:")
    print("   OPENAI_API_KEY=your_actual_api_key_here")
    print()
    print("2. 🗄️  Configure Redis for caching (optional but recommended):")
    print("   REDIS_HOST=localhost")
    print("   REDIS_PORT=6379")
    print()
    print("3. 🚀 Test voice processing in your bot:")
    print("   - Send a voice message to your bot")
    print("   - Check logs for processing confirmation")
    print()
    print("4. 🎛️  Fine-tune settings in config/voice_config.json")
    print()
    print("📚 Documentation:")
    print("   - Voice Processor: app/services/voice_processor.py")
    print("   - Whisper Client: app/services/whisper_client.py")
    print("   - TTS Service: app/services/tts_service.py")
    print("   - Integration: app/services/voice_integration.py")


def main():
    """Main setup function."""
    print("🎤 Voice Processing Setup for Telegram Bot")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version} detected")
    
    # Check system dependencies
    missing_deps = check_system_dependencies()
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        answer = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if answer == 'y':
            install_system_dependencies()
        else:
            print("❌ Cannot proceed without required dependencies")
            sys.exit(1)
    
    # Install Python packages
    install_python_dependencies()
    
    # Create configuration
    config_file = create_voice_config()
    env_file = create_environment_template()
    
    # Test installation
    if test_voice_processing():
        print("\n🎉 Voice processing setup completed successfully!")
        show_next_steps()
    else:
        print("\n❌ Setup completed with errors. Please check the installation.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())