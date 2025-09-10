"""
Application Configuration Management

Handles environment-specific settings using Pydantic Settings for
type safety and validation. Supports multiple environments:
development, staging, production.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from functools import lru_cache

from pydantic import Field, validator, AnyHttpUrl, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="telegram_bot", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    # Connection pool settings for high concurrency (1000+ users)
    pool_size: int = Field(default=100, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=200, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    pool_pre_ping: bool = Field(default=True, env="DB_POOL_PRE_PING")
    
    # Backup settings
    backup_dir: str = Field(default="./backups", env="DB_BACKUP_DIR")
    backup_retention_days: int = Field(default=30, env="DB_BACKUP_RETENTION_DAYS")
    
    @property
    def url(self) -> str:
        """Generate database URL for SQLAlchemy."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Generate synchronous database URL for migrations."""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching, rate limiting, and distributed operations."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    # Basic connection settings
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Clustering support
    cluster_enabled: bool = Field(default=False, env="REDIS_CLUSTER_ENABLED")
    cluster_nodes: List[str] = Field(default=[], env="REDIS_CLUSTER_NODES")
    cluster_skip_full_coverage_check: bool = Field(default=False, env="REDIS_CLUSTER_SKIP_FULL_COVERAGE")
    
    # Sentinel support for high availability
    sentinel_enabled: bool = Field(default=False, env="REDIS_SENTINEL_ENABLED")
    sentinel_hosts: List[str] = Field(default=[], env="REDIS_SENTINEL_HOSTS")
    sentinel_service_name: str = Field(default="mymaster", env="REDIS_SENTINEL_SERVICE_NAME")
    
    # Connection pool settings (increased for 1000+ users)
    max_connections: int = Field(default=500, env="REDIS_MAX_CONNECTIONS")
    min_connections: int = Field(default=50, env="REDIS_MIN_CONNECTIONS")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    socket_timeout: float = Field(default=30.0, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: float = Field(default=30.0, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    # Health check settings
    health_check_interval: int = Field(default=30, env="REDIS_HEALTH_CHECK_INTERVAL")
    connection_check_interval: int = Field(default=60, env="REDIS_CONNECTION_CHECK_INTERVAL")
    
    # Cache settings with different TTLs for different data types
    cache_ttl: int = Field(default=3600, env="REDIS_CACHE_TTL")  # 1 hour
    session_ttl: int = Field(default=86400, env="REDIS_SESSION_TTL")  # 24 hours
    user_profile_ttl: int = Field(default=604800, env="REDIS_USER_PROFILE_TTL")  # 7 days
    conversation_context_ttl: int = Field(default=7200, env="REDIS_CONVERSATION_CONTEXT_TTL")  # 2 hours
    rate_limit_ttl: int = Field(default=3600, env="REDIS_RATE_LIMIT_TTL")  # 1 hour
    
    # Memory optimization settings
    max_memory_policy: str = Field(default="allkeys-lru", env="REDIS_MAX_MEMORY_POLICY")
    memory_samples: int = Field(default=5, env="REDIS_MEMORY_SAMPLES")
    compression_enabled: bool = Field(default=True, env="REDIS_COMPRESSION_ENABLED")
    compression_threshold: int = Field(default=1024, env="REDIS_COMPRESSION_THRESHOLD")  # bytes
    
    # Pub/Sub settings
    pubsub_patterns: List[str] = Field(default=["bot:*", "session:*", "rate_limit:*"], env="REDIS_PUBSUB_PATTERNS")
    pubsub_buffer_size: int = Field(default=10000, env="REDIS_PUBSUB_BUFFER_SIZE")
    
    # Pipeline settings for batch operations
    pipeline_batch_size: int = Field(default=100, env="REDIS_PIPELINE_BATCH_SIZE")
    pipeline_timeout: float = Field(default=10.0, env="REDIS_PIPELINE_TIMEOUT")
    
    # Failover and recovery settings
    failover_enabled: bool = Field(default=True, env="REDIS_FAILOVER_ENABLED")
    failover_timeout: int = Field(default=30, env="REDIS_FAILOVER_TIMEOUT")
    auto_reconnect: bool = Field(default=True, env="REDIS_AUTO_RECONNECT")
    reconnect_retries: int = Field(default=5, env="REDIS_RECONNECT_RETRIES")
    reconnect_delay: float = Field(default=1.0, env="REDIS_RECONNECT_DELAY")
    
    # Performance monitoring
    enable_monitoring: bool = Field(default=True, env="REDIS_ENABLE_MONITORING")
    slow_log_threshold: int = Field(default=10000, env="REDIS_SLOW_LOG_THRESHOLD")  # microseconds
    metrics_collection_interval: int = Field(default=60, env="REDIS_METRICS_COLLECTION_INTERVAL")
    
    # Security settings
    ssl_enabled: bool = Field(default=False, env="REDIS_SSL_ENABLED")
    ssl_cert_file: Optional[str] = Field(default=None, env="REDIS_SSL_CERT_FILE")
    ssl_key_file: Optional[str] = Field(default=None, env="REDIS_SSL_KEY_FILE")
    ssl_ca_certs: Optional[str] = Field(default=None, env="REDIS_SSL_CA_CERTS")
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.ssl_enabled:
            scheme = "rediss"
        else:
            scheme = "redis"
        
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"
    
    @property
    def cluster_urls(self) -> List[str]:
        """Generate cluster node URLs."""
        if not self.cluster_nodes:
            return [self.url]
        
        urls = []
        for node in self.cluster_nodes:
            if ":" in node:
                host, port = node.split(":")
            else:
                host, port = node, str(self.port)
            
            scheme = "rediss" if self.ssl_enabled else "redis"
            auth = f":{self.password}@" if self.password else ""
            urls.append(f"{scheme}://{auth}{host}:{port}/{self.db}")
        
        return urls
    
    @property
    def sentinel_urls(self) -> List[str]:
        """Generate sentinel URLs."""
        urls = []
        for host_port in self.sentinel_hosts:
            if ":" in host_port:
                host, port = host_port.split(":")
            else:
                host, port = host_port, "26379"
            
            urls.append(f"{host}:{port}")
        
        return urls
    
    @validator("cluster_nodes", pre=True)
    def parse_cluster_nodes(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [node.strip() for node in v.split(",") if node.strip()]
        return v
    
    @validator("sentinel_hosts", pre=True)
    def parse_sentinel_hosts(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [host.strip() for host in v.split(",") if host.strip()]
        return v
    
    @validator("pubsub_patterns", pre=True)
    def parse_pubsub_patterns(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [pattern.strip() for pattern in v.split(",") if pattern.strip()]
        return v


class TelegramSettings(BaseSettings):
    """Telegram Bot API configuration."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    bot_token: str = Field(default="test_token", env="TELEGRAM_BOT_TOKEN")
    webhook_url: Optional[str] = Field(default=None, env="TELEGRAM_WEBHOOK_URL")
    webhook_secret: Optional[str] = Field(default=None, env="TELEGRAM_WEBHOOK_SECRET")
    
    # Rate limiting settings for Telegram API
    rate_limit_calls: int = Field(default=20, env="TELEGRAM_RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, env="TELEGRAM_RATE_LIMIT_PERIOD")
    
    # Bot behavior settings
    parse_mode: str = Field(default="HTML", env="TELEGRAM_PARSE_MODE")
    disable_web_page_preview: bool = Field(default=True)
    
    @validator("bot_token")
    def validate_bot_token(cls, v):
        if not v:
            raise ValueError("Bot token is required")
        
        # Allow test tokens during development
        if v.startswith("test_"):
            return v
        
        if not v.startswith(("bot", "BOT")):
            # Remove bot prefix if exists for validation
            token_part = v.replace("bot", "").replace("BOT", "")
            if ":" not in token_part:
                raise ValueError("Invalid Telegram bot token format")
        return v


class MLSettings(BaseSettings):
    """Machine Learning model configuration."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    model_path: Path = Field(default=Path("models"), env="ML_MODEL_PATH")
    device: str = Field(default="cpu", env="ML_DEVICE")  # cpu, cuda, mps
    batch_size: int = Field(default=32, env="ML_BATCH_SIZE")
    max_sequence_length: int = Field(default=512, env="ML_MAX_SEQUENCE_LENGTH")
    
    # Model-specific settings
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        env="ML_SENTIMENT_MODEL"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="ML_EMBEDDING_MODEL"
    )
    
    # Inference settings
    enable_gpu: bool = Field(default=False, env="ML_ENABLE_GPU")
    model_cache_size: int = Field(default=3, env="ML_MODEL_CACHE_SIZE")


class SecuritySettings(BaseSettings):
    """Enhanced security and authentication settings."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    jwt_secret: str = Field(default="dev-jwt-secret-change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION_SECONDS")
    
    # Enhanced rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    rate_limit_per_ip_per_minute: int = Field(default=10, env="RATE_LIMIT_PER_IP_PER_MINUTE")
    rate_limit_burst: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    # Security headers
    enable_security_headers: bool = Field(default=True, env="ENABLE_SECURITY_HEADERS")
    hsts_max_age: int = Field(default=31536000, env="HSTS_MAX_AGE")  # 1 year
    
    # Input validation
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB
    max_json_size: int = Field(default=1048576, env="MAX_JSON_SIZE")  # 1MB
    enable_input_sanitization: bool = Field(default=True, env="ENABLE_INPUT_SANITIZATION")
    
    # Admin security
    admin_users: List[int] = Field(default=[], env="TELEGRAM_ADMIN_USERS")
    require_admin_2fa: bool = Field(default=True, env="REQUIRE_ADMIN_2FA")
    admin_session_timeout: int = Field(default=3600, env="ADMIN_SESSION_TIMEOUT")  # 1 hour
    
    # IP security
    enable_ip_whitelist: bool = Field(default=False, env="ENABLE_IP_WHITELIST")
    ip_whitelist: List[str] = Field(default=[], env="IP_WHITELIST")
    enable_ip_blocking: bool = Field(default=True, env="ENABLE_IP_BLOCKING")
    max_failed_attempts: int = Field(default=10, env="MAX_FAILED_ATTEMPTS")
    block_duration: int = Field(default=3600, env="BLOCK_DURATION")  # 1 hour
    
    # Webhook security
    webhook_signature_required: bool = Field(default=True, env="WEBHOOK_SIGNATURE_REQUIRED")
    webhook_ip_validation: bool = Field(default=True, env="WEBHOOK_IP_VALIDATION")
    webhook_timeout: int = Field(default=30, env="WEBHOOK_TIMEOUT")
    
    # Content security
    enable_content_filtering: bool = Field(default=True, env="ENABLE_CONTENT_FILTERING")
    max_message_length: int = Field(default=10000, env="MAX_MESSAGE_LENGTH")
    enable_url_validation: bool = Field(default=True, env="ENABLE_URL_VALIDATION")
    
    # Encryption
    enable_data_encryption: bool = Field(default=True, env="ENABLE_DATA_ENCRYPTION")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # Security monitoring
    enable_security_monitoring: bool = Field(default=True, env="ENABLE_SECURITY_MONITORING")
    security_alert_threshold: int = Field(default=5, env="SECURITY_ALERT_THRESHOLD")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [url.strip() for url in v.split(",")]
        return v
    
    @validator("admin_users", pre=True)
    def parse_admin_users(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [int(uid.strip()) for uid in v.split(",") if uid.strip().isdigit()]
        return v
    
    @validator("ip_whitelist", pre=True)
    def parse_ip_whitelist(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [ip.strip() for ip in v.split(",") if ip.strip()]
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    @validator("jwt_secret")
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring, logging, and observability settings."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json, text
    
    # Sentry for error tracking
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")
    
    # Prometheus metrics
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")


class CelerySettings(BaseSettings):
    """Celery task queue configuration."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Worker settings
    worker_concurrency: int = Field(default=4, env="CELERY_WORKER_CONCURRENCY")
    task_time_limit: int = Field(default=300, env="CELERY_TASK_TIME_LIMIT")
    task_soft_time_limit: int = Field(default=240, env="CELERY_TASK_SOFT_TIME_LIMIT")
    
    # Queue configuration
    default_queue: str = Field(default="default", env="CELERY_DEFAULT_QUEUE")
    ml_queue: str = Field(default="ml_tasks", env="CELERY_ML_QUEUE")
    
    # Retry settings
    task_max_retries: int = Field(default=3, env="CELERY_TASK_MAX_RETRIES")
    task_retry_delay: int = Field(default=60, env="CELERY_TASK_RETRY_DELAY")


class StripeSettings(BaseSettings):
    """Stripe payment processing configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False, 
        extra='ignore',
        env_prefix="STRIPE_"
    )
    
    # API Keys
    publishable_key: str = Field(default="pk_test_...", alias="STRIPE_PUBLISHABLE_KEY")
    secret_key: str = Field(default="sk_test_...", alias="STRIPE_SECRET_KEY") 
    webhook_secret: str = Field(default="whsec_test_...", alias="STRIPE_WEBHOOK_SECRET")
    
    # API Configuration
    api_version: str = Field(default="2023-10-16", env="STRIPE_API_VERSION")
    test_mode: bool = Field(default=True, env="STRIPE_TEST_MODE")
    
    # Customer Portal
    portal_configuration_id: Optional[str] = Field(default=None, env="STRIPE_PORTAL_CONFIGURATION_ID")
    
    # Tax Settings
    tax_enabled: bool = Field(default=False, env="STRIPE_TAX_ENABLED")
    automatic_tax: bool = Field(default=False, env="STRIPE_AUTOMATIC_TAX")
    
    # Webhook Settings
    webhook_endpoint_url: Optional[str] = Field(default=None, env="STRIPE_WEBHOOK_ENDPOINT_URL")
    webhook_events: List[str] = Field(
        default=[
            "customer.subscription.created",
            "customer.subscription.updated", 
            "customer.subscription.deleted",
            "invoice.payment_succeeded",
            "invoice.payment_failed",
            "customer.created",
            "payment_method.attached"
        ],
        env="STRIPE_WEBHOOK_EVENTS"
    )
    
    # Retry and Timeout Settings
    request_timeout: int = Field(default=30, env="STRIPE_REQUEST_TIMEOUT")
    max_network_retries: int = Field(default=3, env="STRIPE_MAX_RETRIES")
    
    # Dunning Management
    max_payment_retries: int = Field(default=3, env="STRIPE_MAX_PAYMENT_RETRIES")
    retry_delay_days: List[int] = Field(default=[1, 3, 7], env="STRIPE_RETRY_DELAY_DAYS")
    
    # Security Settings
    require_webhook_signature: bool = Field(default=True, env="STRIPE_REQUIRE_WEBHOOK_SIGNATURE")
    enable_idempotency_keys: bool = Field(default=True, env="STRIPE_ENABLE_IDEMPOTENCY_KEYS")
    
    @validator("webhook_events", pre=True)
    def parse_webhook_events(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [event.strip() for event in v.split(",") if event.strip()]
        return v
    
    @validator("retry_delay_days", pre=True)
    def parse_retry_delay_days(cls, v):
        if isinstance(v, str):
            return [int(day.strip()) for day in v.split(",") if day.strip().isdigit()]
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if not v or v.startswith("sk_test_..."):
            return v  # Allow default test keys
        
        if not v.startswith(("sk_test_", "sk_live_")):
            raise ValueError("Invalid Stripe secret key format")
        return v
    
    @validator("publishable_key")
    def validate_publishable_key(cls, v):
        if not v or v.startswith("pk_test_..."):
            return v  # Allow default test keys
            
        if not v.startswith(("pk_test_", "pk_live_")):
            raise ValueError("Invalid Stripe publishable key format")
        return v


class VoiceProcessingSettings(BaseSettings):
    """Voice processing and TTS configuration."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    # OpenAI Whisper API settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    whisper_model: str = Field(default="whisper-1", env="WHISPER_MODEL")
    whisper_language: Optional[str] = Field(default=None, env="WHISPER_DEFAULT_LANGUAGE")  # Auto-detect by default
    
    # Audio file limits
    max_audio_file_size: int = Field(default=25 * 1024 * 1024, env="MAX_AUDIO_FILE_SIZE")  # 25MB
    max_audio_duration: int = Field(default=600, env="MAX_AUDIO_DURATION")  # 10 minutes
    max_voice_message_duration: int = Field(default=300, env="MAX_VOICE_MESSAGE_DURATION")  # 5 minutes
    
    # Voice processing settings
    enable_voice_processing: bool = Field(default=True, env="ENABLE_VOICE_PROCESSING")
    enable_voice_responses: bool = Field(default=True, env="ENABLE_VOICE_RESPONSES")
    optimize_for_speech: bool = Field(default=True, env="OPTIMIZE_FOR_SPEECH")
    voice_processing_timeout: int = Field(default=30, env="VOICE_PROCESSING_TIMEOUT")  # 30 seconds
    
    # TTS (Text-to-Speech) settings
    tts_default_language: str = Field(default="en", env="TTS_DEFAULT_LANGUAGE")
    tts_max_text_length: int = Field(default=5000, env="TTS_MAX_TEXT_LENGTH")
    tts_slow_speech: bool = Field(default=False, env="TTS_SLOW_SPEECH")
    tts_chunk_long_text: bool = Field(default=True, env="TTS_CHUNK_LONG_TEXT")
    
    # Voice response preferences
    voice_response_max_length: int = Field(default=500, env="VOICE_RESPONSE_MAX_LENGTH")  # chars
    voice_quiet_hours_start: int = Field(default=22, env="VOICE_QUIET_HOURS_START")  # 10 PM
    voice_quiet_hours_end: int = Field(default=7, env="VOICE_QUIET_HOURS_END")  # 7 AM
    enable_voice_in_groups: bool = Field(default=False, env="ENABLE_VOICE_IN_GROUPS")  # Usually disabled for groups
    
    # Caching settings
    enable_transcription_cache: bool = Field(default=True, env="ENABLE_TRANSCRIPTION_CACHE")
    enable_tts_cache: bool = Field(default=True, env="ENABLE_TTS_CACHE")
    transcription_cache_ttl: int = Field(default=3600 * 24, env="TRANSCRIPTION_CACHE_TTL")  # 24 hours
    tts_cache_ttl: int = Field(default=3600 * 24 * 7, env="TTS_CACHE_TTL")  # 7 days
    
    # Performance settings
    target_processing_time: float = Field(default=2.0, env="TARGET_PROCESSING_TIME")  # 2 seconds
    max_concurrent_voice_processing: int = Field(default=10, env="MAX_CONCURRENT_VOICE_PROCESSING")
    voice_processor_temp_dir: Optional[str] = Field(default=None, env="VOICE_PROCESSOR_TEMP_DIR")
    
    # Audio quality settings
    output_sample_rate: int = Field(default=16000, env="OUTPUT_SAMPLE_RATE")  # 16kHz for speech
    output_bitrate: str = Field(default="64k", env="OUTPUT_BITRATE")  # Good for voice messages
    enable_audio_compression: bool = Field(default=True, env="ENABLE_AUDIO_COMPRESSION")
    
    # Error handling and fallbacks
    enable_fallback_tts: bool = Field(default=True, env="ENABLE_FALLBACK_TTS")
    max_transcription_retries: int = Field(default=3, env="MAX_TRANSCRIPTION_RETRIES")
    enable_speech_recognition_fallback: bool = Field(default=False, env="ENABLE_SPEECH_RECOGNITION_FALLBACK")
    
    # Monitoring and metrics
    enable_voice_metrics: bool = Field(default=True, env="ENABLE_VOICE_METRICS")
    log_voice_processing_stats: bool = Field(default=True, env="LOG_VOICE_PROCESSING_STATS")
    
    # Security settings
    validate_audio_content: bool = Field(default=True, env="VALIDATE_AUDIO_CONTENT")
    scan_for_malicious_audio: bool = Field(default=True, env="SCAN_FOR_MALICIOUS_AUDIO")
    
    @validator("max_audio_file_size")
    def validate_file_size(cls, v):
        if v > 100 * 1024 * 1024:  # 100MB absolute max
            raise ValueError("Max audio file size cannot exceed 100MB")
        return v
    
    @validator("voice_quiet_hours_start")
    def validate_quiet_hours_start(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("Quiet hours start must be between 0 and 23")
        return v
    
    @validator("voice_quiet_hours_end")
    def validate_quiet_hours_end(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("Quiet hours end must be between 0 and 23")
        return v


class AdvancedTypingSettings(BaseSettings):
    """Advanced Typing Simulator configuration."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')
    
    # Core typing system settings
    enable_advanced_typing: bool = Field(default=True, env="ENABLE_ADVANCED_TYPING")
    enable_typing_caching: bool = Field(default=True, env="ENABLE_TYPING_CACHING")
    enable_typing_integration: bool = Field(default=True, env="ENABLE_TYPING_INTEGRATION")
    
    # Performance and scaling settings
    max_concurrent_typing_sessions: int = Field(default=1000, env="MAX_TYPING_SESSIONS")
    typing_cache_ttl: int = Field(default=300, env="TYPING_CACHE_TTL")  # 5 minutes
    typing_session_timeout: int = Field(default=60, env="TYPING_SESSION_TIMEOUT")  # 1 minute max
    
    # Simulation quality settings
    max_simulation_time: float = Field(default=30.0, env="MAX_SIMULATION_TIME")  # 30 seconds max
    min_simulation_time: float = Field(default=0.3, env="MIN_SIMULATION_TIME")   # 0.3 seconds min
    enable_error_simulation: bool = Field(default=True, env="ENABLE_ERROR_SIMULATION")
    enable_pause_simulation: bool = Field(default=True, env="ENABLE_PAUSE_SIMULATION")
    
    # Anti-detection settings
    enable_anti_detection: bool = Field(default=True, env="ENABLE_ANTI_DETECTION")
    pattern_randomization_level: float = Field(default=0.3, env="PATTERN_RANDOMIZATION_LEVEL")  # 0-1
    detection_prevention_enabled: bool = Field(default=True, env="DETECTION_PREVENTION_ENABLED")
    
    # Performance monitoring
    enable_typing_metrics: bool = Field(default=True, env="ENABLE_TYPING_METRICS")
    metrics_collection_interval: int = Field(default=60, env="TYPING_METRICS_INTERVAL")  # 1 minute
    
    # Rate limiting for typing indicators
    typing_rate_limit_per_user: int = Field(default=10, env="TYPING_RATE_LIMIT_PER_USER")  # per minute
    typing_burst_limit: int = Field(default=3, env="TYPING_BURST_LIMIT")
    
    # Fallback settings
    enable_simple_fallback: bool = Field(default=True, env="ENABLE_SIMPLE_FALLBACK")
    fallback_typing_speed: float = Field(default=120.0, env="FALLBACK_TYPING_SPEED")  # chars per minute
    
    # Personality integration
    enable_personality_typing: bool = Field(default=True, env="ENABLE_PERSONALITY_TYPING")
    personality_adaptation_strength: float = Field(default=0.7, env="PERSONALITY_ADAPTATION_STRENGTH")  # 0-1
    
    # Context awareness
    enable_context_adaptation: bool = Field(default=True, env="ENABLE_CONTEXT_ADAPTATION")
    enable_conversation_flow_analysis: bool = Field(default=True, env="ENABLE_CONVERSATION_FLOW")
    enable_emotional_state_modeling: bool = Field(default=True, env="ENABLE_EMOTIONAL_STATE_MODELING")
    
    # Debug and development
    enable_typing_debug_logs: bool = Field(default=False, env="ENABLE_TYPING_DEBUG_LOGS")
    typing_simulation_logging: bool = Field(default=False, env="TYPING_SIMULATION_LOGGING")
    
    @validator("pattern_randomization_level")
    def validate_randomization_level(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Pattern randomization level must be between 0 and 1")
        return v
    
    @validator("personality_adaptation_strength")
    def validate_adaptation_strength(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Personality adaptation strength must be between 0 and 1")
        return v
    
    @validator("max_simulation_time")
    def validate_max_simulation_time(cls, v):
        if v <= 0 or v > 300:  # Max 5 minutes
            raise ValueError("Max simulation time must be between 0 and 300 seconds")
        return v


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'  # Allow extra fields to be ignored instead of causing validation errors
    )
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Application settings
    app_name: str = Field(default="Telegram ML Bot", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Configuration sections
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    stripe: StripeSettings = Field(default_factory=StripeSettings)
    advanced_typing: AdvancedTypingSettings = Field(default_factory=AdvancedTypingSettings)
    voice_processing: VoiceProcessingSettings = Field(default_factory=VoiceProcessingSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.testing or self.environment.lower() == "testing"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are loaded only once
    and reused across the application.
    """
    return Settings()


# Export settings instance for easy import
settings = get_settings()