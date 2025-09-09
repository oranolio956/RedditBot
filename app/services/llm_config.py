"""
LLM Configuration Management

Secure configuration management for LLM API keys, model settings, and quotas.
Supports multiple environments and secure credential handling.

Features:
- Secure API key management
- Environment-based configuration
- Model quota and rate limiting configuration
- Cost management settings
- Provider failover configuration
- Configuration validation
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

import structlog
from pydantic import BaseSettings, Field, validator
from cryptography.fernet import Fernet
import base64

logger = structlog.get_logger(__name__)


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider."""
    name: str
    api_key: str
    base_url: Optional[str] = None
    
    # Quota and rate limits
    daily_quota_usd: float = 100.0
    hourly_quota_usd: float = 20.0
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    
    # Model preferences
    default_model: Optional[str] = None
    fallback_models: List[str] = field(default_factory=list)
    
    # Cost optimization
    cost_threshold_warning: float = 50.0  # USD
    cost_threshold_stop: float = 90.0     # USD
    
    # Performance settings
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    # Feature flags
    enabled: bool = True
    supports_streaming: bool = True
    supports_functions: bool = False


class LLMSettings(BaseSettings):
    """
    LLM service configuration with secure credential management.
    
    This class handles:
    - API key management and encryption
    - Provider configuration and failover
    - Cost management and quotas
    - Model selection and preferences
    - Environment-specific settings
    """
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug_mode: bool = Field(default=False, env="LLM_DEBUG_MODE")
    
    # Master encryption key for credentials (should be in secure storage)
    master_encryption_key: Optional[str] = Field(default=None, env="LLM_MASTER_KEY")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_base_url: Optional[str] = Field(default=None, env="ANTHROPIC_BASE_URL")
    
    # Default quotas (can be overridden per provider)
    default_daily_quota_usd: float = Field(default=100.0, env="LLM_DAILY_QUOTA")
    default_hourly_quota_usd: float = Field(default=20.0, env="LLM_HOURLY_QUOTA")
    default_requests_per_minute: int = Field(default=60, env="LLM_REQUESTS_PER_MINUTE")
    
    # Model preferences
    preferred_provider: str = Field(default="openai", env="LLM_PREFERRED_PROVIDER")
    preferred_model: str = Field(default="gpt-3.5-turbo", env="LLM_PREFERRED_MODEL")
    fallback_models: List[str] = Field(
        default=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
        env="LLM_FALLBACK_MODELS"
    )
    
    # Cost management
    global_daily_limit_usd: float = Field(default=500.0, env="LLM_GLOBAL_DAILY_LIMIT")
    cost_alert_threshold: float = Field(default=0.8, env="LLM_COST_ALERT_THRESHOLD")  # 80%
    enable_cost_alerts: bool = Field(default=True, env="LLM_ENABLE_COST_ALERTS")
    
    # Performance settings
    default_timeout_seconds: int = Field(default=30, env="LLM_DEFAULT_TIMEOUT")
    max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")
    enable_caching: bool = Field(default=True, env="LLM_ENABLE_CACHING")
    cache_ttl_seconds: int = Field(default=3600, env="LLM_CACHE_TTL")
    
    # Response settings
    max_response_tokens: int = Field(default=4096, env="LLM_MAX_RESPONSE_TOKENS")
    default_temperature: float = Field(default=0.7, env="LLM_DEFAULT_TEMPERATURE")
    enable_streaming: bool = Field(default=True, env="LLM_ENABLE_STREAMING")
    
    # Integration settings
    enable_personality_adaptation: bool = Field(default=True, env="LLM_ENABLE_PERSONALITY")
    enable_conversation_memory: bool = Field(default=True, env="LLM_ENABLE_MEMORY")
    enable_context_optimization: bool = Field(default=True, env="LLM_ENABLE_CONTEXT_OPT")
    
    # Monitoring and analytics
    enable_metrics_collection: bool = Field(default=True, env="LLM_ENABLE_METRICS")
    metrics_retention_days: int = Field(default=30, env="LLM_METRICS_RETENTION")
    enable_quality_scoring: bool = Field(default=True, env="LLM_ENABLE_QUALITY_SCORING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("fallback_models", pre=True)
    def parse_fallback_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v
    
    @validator("cost_alert_threshold")
    def validate_cost_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Cost alert threshold must be between 0 and 1")
        return v
    
    @validator("default_temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v


class LLMConfigManager:
    """
    Secure configuration manager for LLM services.
    
    Handles:
    - Secure API key storage and retrieval
    - Provider configuration management
    - Environment-specific settings
    - Configuration validation
    - Runtime configuration updates
    """
    
    def __init__(self):
        self.settings = LLMSettings()
        self.cipher_suite = None
        self.provider_configs: Dict[str, LLMProviderConfig] = {}
        
        # Initialize encryption if master key is available
        if self.settings.master_encryption_key:
            try:
                key = base64.urlsafe_b64decode(self.settings.master_encryption_key.encode())
                self.cipher_suite = Fernet(key)
                logger.info("Encryption enabled for LLM configuration")
            except Exception as e:
                logger.error("Failed to initialize encryption", error=str(e))
        
        # Load provider configurations
        self._load_provider_configs()
    
    def _load_provider_configs(self):
        """Load configuration for all LLM providers."""
        try:
            # OpenAI Configuration
            if self.settings.openai_api_key:
                openai_config = LLMProviderConfig(
                    name="openai",
                    api_key=self._decrypt_if_needed(self.settings.openai_api_key),
                    base_url=self.settings.openai_base_url,
                    daily_quota_usd=self.settings.default_daily_quota_usd,
                    hourly_quota_usd=self.settings.default_hourly_quota_usd,
                    requests_per_minute=self.settings.default_requests_per_minute,
                    default_model="gpt-3.5-turbo",
                    fallback_models=["gpt-4", "gpt-3.5-turbo"],
                    supports_streaming=True,
                    supports_functions=True
                )
                self.provider_configs["openai"] = openai_config
            
            # Anthropic Configuration
            if self.settings.anthropic_api_key:
                anthropic_config = LLMProviderConfig(
                    name="anthropic",
                    api_key=self._decrypt_if_needed(self.settings.anthropic_api_key),
                    base_url=self.settings.anthropic_base_url,
                    daily_quota_usd=self.settings.default_daily_quota_usd,
                    hourly_quota_usd=self.settings.default_hourly_quota_usd,
                    requests_per_minute=self.settings.default_requests_per_minute,
                    default_model="claude-3-sonnet-20240229",
                    fallback_models=["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
                    supports_streaming=True,
                    supports_functions=False
                )
                self.provider_configs["anthropic"] = anthropic_config
            
            # Configure environment-specific settings
            self._apply_environment_settings()
            
            logger.info(
                f"Loaded {len(self.provider_configs)} LLM provider configurations",
                providers=list(self.provider_configs.keys()),
                environment=self.settings.environment
            )
            
        except Exception as e:
            logger.error("Error loading provider configurations", error=str(e))
    
    def _apply_environment_settings(self):
        """Apply environment-specific configuration overrides."""
        try:
            if self.settings.environment == Environment.DEVELOPMENT:
                # Lower quotas for development
                for config in self.provider_configs.values():
                    config.daily_quota_usd = min(config.daily_quota_usd, 10.0)
                    config.hourly_quota_usd = min(config.hourly_quota_usd, 2.0)
                    config.requests_per_minute = min(config.requests_per_minute, 30)
            
            elif self.settings.environment == Environment.STAGING:
                # Medium quotas for staging
                for config in self.provider_configs.values():
                    config.daily_quota_usd = min(config.daily_quota_usd, 50.0)
                    config.hourly_quota_usd = min(config.hourly_quota_usd, 10.0)
            
            elif self.settings.environment == Environment.PRODUCTION:
                # Full quotas for production
                pass
                
        except Exception as e:
            logger.error("Error applying environment settings", error=str(e))
    
    def _decrypt_if_needed(self, value: str) -> str:
        """Decrypt value if encryption is enabled and value appears encrypted."""
        try:
            if not self.cipher_suite or not value:
                return value
            
            # Check if value looks like encrypted data (base64)
            if value.startswith("gAAAAAB"):  # Fernet token prefix
                decrypted = self.cipher_suite.decrypt(value.encode())
                return decrypted.decode()
            
            return value
            
        except Exception as e:
            logger.error("Error decrypting value", error=str(e))
            return value
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a value if encryption is enabled."""
        try:
            if not self.cipher_suite or not value:
                return value
            
            encrypted = self.cipher_suite.encrypt(value.encode())
            return encrypted.decode()
            
        except Exception as e:
            logger.error("Error encrypting value", error=str(e))
            return value
    
    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider."""
        return self.provider_configs.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available/enabled providers."""
        return [
            name for name, config in self.provider_configs.items()
            if config.enabled and config.api_key
        ]
    
    def get_preferred_provider(self) -> Optional[str]:
        """Get the preferred provider if available."""
        preferred = self.settings.preferred_provider
        if preferred in self.provider_configs and self.provider_configs[preferred].enabled:
            return preferred
        
        # Return first available provider
        available = self.get_available_providers()
        return available[0] if available else None
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration and return status."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'provider_status': {}
        }
        
        try:
            # Check if at least one provider is configured
            if not self.provider_configs:
                validation_result['valid'] = False
                validation_result['errors'].append("No LLM providers configured")
            
            # Validate each provider
            for name, config in self.provider_configs.items():
                provider_status = {
                    'enabled': config.enabled,
                    'has_api_key': bool(config.api_key),
                    'quota_configured': config.daily_quota_usd > 0,
                    'valid': True,
                    'issues': []
                }
                
                # Check API key
                if not config.api_key:
                    provider_status['valid'] = False
                    provider_status['issues'].append("Missing API key")
                
                # Check quotas
                if config.daily_quota_usd <= 0:
                    provider_status['issues'].append("Invalid daily quota")
                
                if config.hourly_quota_usd > config.daily_quota_usd:
                    provider_status['issues'].append("Hourly quota exceeds daily quota")
                
                validation_result['provider_status'][name] = provider_status
                
                if not provider_status['valid']:
                    validation_result['warnings'].append(f"Provider {name} has configuration issues")
            
            # Check global settings
            if self.settings.global_daily_limit_usd <= 0:
                validation_result['errors'].append("Invalid global daily limit")
            
            # Environment-specific checks
            if self.settings.environment == Environment.PRODUCTION:
                if not self.settings.master_encryption_key:
                    validation_result['warnings'].append("No encryption key configured for production")
                
                total_daily_quota = sum(
                    config.daily_quota_usd for config in self.provider_configs.values()
                )
                if total_daily_quota > self.settings.global_daily_limit_usd:
                    validation_result['warnings'].append("Provider quotas exceed global limit")
            
            if validation_result['errors']:
                validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error("Error validating configuration", error=str(e))
            return {
                'valid': False,
                'errors': [f"Configuration validation failed: {e}"],
                'warnings': [],
                'provider_status': {}
            }
    
    def update_provider_quota(self, provider_name: str, daily_quota: float, hourly_quota: Optional[float] = None):
        """Update quota limits for a provider."""
        try:
            if provider_name in self.provider_configs:
                config = self.provider_configs[provider_name]
                config.daily_quota_usd = daily_quota
                if hourly_quota is not None:
                    config.hourly_quota_usd = hourly_quota
                
                logger.info(
                    f"Updated quotas for provider {provider_name}",
                    daily_quota=daily_quota,
                    hourly_quota=hourly_quota
                )
            else:
                raise ValueError(f"Provider {provider_name} not found")
                
        except Exception as e:
            logger.error("Error updating provider quota", error=str(e))
            raise
    
    def enable_provider(self, provider_name: str, enabled: bool = True):
        """Enable or disable a provider."""
        try:
            if provider_name in self.provider_configs:
                self.provider_configs[provider_name].enabled = enabled
                logger.info(f"Provider {provider_name} {'enabled' if enabled else 'disabled'}")
            else:
                raise ValueError(f"Provider {provider_name} not found")
                
        except Exception as e:
            logger.error("Error updating provider status", error=str(e))
            raise
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        try:
            return {
                'environment': self.settings.environment,
                'debug_mode': self.settings.debug_mode,
                'encryption_enabled': self.cipher_suite is not None,
                'providers': {
                    name: {
                        'enabled': config.enabled,
                        'has_api_key': bool(config.api_key),
                        'daily_quota': config.daily_quota_usd,
                        'hourly_quota': config.hourly_quota_usd,
                        'default_model': config.default_model,
                        'supports_streaming': config.supports_streaming,
                        'supports_functions': config.supports_functions
                    }
                    for name, config in self.provider_configs.items()
                },
                'global_settings': {
                    'preferred_provider': self.settings.preferred_provider,
                    'preferred_model': self.settings.preferred_model,
                    'global_daily_limit': self.settings.global_daily_limit_usd,
                    'enable_caching': self.settings.enable_caching,
                    'enable_streaming': self.settings.enable_streaming,
                    'enable_personality': self.settings.enable_personality_adaptation,
                    'enable_memory': self.settings.enable_conversation_memory
                },
                'validation': self.validate_configuration()
            }
            
        except Exception as e:
            logger.error("Error getting configuration summary", error=str(e))
            return {'error': str(e)}
    
    @staticmethod
    def generate_master_key() -> str:
        """Generate a new master encryption key."""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()


# Global configuration instance
config_manager: Optional[LLMConfigManager] = None


def get_llm_config() -> LLMConfigManager:
    """Get the global LLM configuration manager."""
    global config_manager
    if config_manager is None:
        config_manager = LLMConfigManager()
    return config_manager


# Export main classes
__all__ = [
    'LLMConfigManager',
    'LLMProviderConfig',
    'LLMSettings',
    'Environment',
    'get_llm_config'
]