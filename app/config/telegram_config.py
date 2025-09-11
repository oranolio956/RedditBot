"""
Telegram Configuration
Configuration settings and constants for Telegram account management system.
"""

import os
from datetime import timedelta
from typing import Dict, List, Any
from pydantic import BaseSettings, Field
from enum import Enum


class TelegramEnvironment(str, Enum):
    """Telegram environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TelegramConfig(BaseSettings):
    """
    Telegram configuration settings with environment variable support
    """
    
    # Environment
    ENVIRONMENT: TelegramEnvironment = Field(
        TelegramEnvironment.DEVELOPMENT,
        env="TELEGRAM_ENVIRONMENT"
    )
    
    # Telegram API Configuration
    API_ID: int = Field(..., env="TELEGRAM_API_ID")
    API_HASH: str = Field(..., env="TELEGRAM_API_HASH")
    
    # Session Configuration
    SESSION_DIRECTORY: str = Field("sessions", env="TELEGRAM_SESSION_DIR")
    SESSION_ENCRYPTION_KEY: str = Field(..., env="TELEGRAM_SESSION_KEY")
    
    # Safety and Rate Limiting
    MAX_ACCOUNTS: int = Field(10, env="TELEGRAM_MAX_ACCOUNTS")
    DEFAULT_SAFETY_LEVEL: str = Field("conservative", env="TELEGRAM_DEFAULT_SAFETY_LEVEL")
    
    # Message Limits (Conservative defaults)
    DEFAULT_MAX_MESSAGES_PER_DAY: int = Field(50, env="TELEGRAM_MAX_MESSAGES_DAY")
    DEFAULT_MAX_GROUPS_PER_DAY: int = Field(2, env="TELEGRAM_MAX_GROUPS_DAY")
    DEFAULT_MAX_DMS_PER_DAY: int = Field(5, env="TELEGRAM_MAX_DMS_DAY")
    
    # Timing Configuration
    MIN_MESSAGE_DELAY: int = Field(30, env="TELEGRAM_MIN_DELAY")  # seconds
    MAX_MESSAGE_DELAY: int = Field(300, env="TELEGRAM_MAX_DELAY")  # seconds
    TYPING_SPEED_MIN: float = Field(0.03, env="TELEGRAM_TYPING_MIN")  # seconds per char
    TYPING_SPEED_MAX: float = Field(0.08, env="TELEGRAM_TYPING_MAX")  # seconds per char
    
    # Risk Management
    MAX_RISK_SCORE: float = Field(70.0, env="TELEGRAM_MAX_RISK_SCORE")
    CRITICAL_RISK_SCORE: float = Field(85.0, env="TELEGRAM_CRITICAL_RISK_SCORE")
    RISK_COOLDOWN_MULTIPLIER: float = Field(2.0, env="TELEGRAM_RISK_COOLDOWN_MULT")
    
    # Error Handling
    MAX_FLOOD_WAITS_PER_HOUR: int = Field(3, env="TELEGRAM_MAX_FLOOD_WAITS")
    MAX_CONSECUTIVE_ERRORS: int = Field(5, env="TELEGRAM_MAX_CONSECUTIVE_ERRORS")
    FLOOD_WAIT_BACKOFF_MULTIPLIER: float = Field(1.5, env="TELEGRAM_FLOOD_BACKOFF")
    
    # Legal and Compliance
    REQUIRE_AI_DISCLOSURE: bool = Field(True, env="TELEGRAM_REQUIRE_AI_DISCLOSURE")
    REQUIRE_GDPR_CONSENT: bool = Field(True, env="TELEGRAM_REQUIRE_GDPR_CONSENT")
    DATA_RETENTION_DAYS: int = Field(30, env="TELEGRAM_DATA_RETENTION_DAYS")
    
    # Monitoring and Health Checks
    HEALTH_CHECK_INTERVAL: int = Field(300, env="TELEGRAM_HEALTH_CHECK_INTERVAL")  # seconds
    SAFETY_MONITOR_INTERVAL: int = Field(60, env="TELEGRAM_SAFETY_MONITOR_INTERVAL")  # seconds
    ANALYTICS_UPDATE_INTERVAL: int = Field(3600, env="TELEGRAM_ANALYTICS_INTERVAL")  # seconds
    
    # Revolutionary Features Integration
    ENABLE_CONSCIOUSNESS_MIRROR: bool = Field(True, env="TELEGRAM_ENABLE_CONSCIOUSNESS")
    ENABLE_MEMORY_PALACE: bool = Field(True, env="TELEGRAM_ENABLE_MEMORY_PALACE")
    ENABLE_EMOTION_DETECTION: bool = Field(True, env="TELEGRAM_ENABLE_EMOTION_DETECTION")
    ENABLE_DIGITAL_TELEPATHY: bool = Field(True, env="TELEGRAM_ENABLE_DIGITAL_TELEPATHY")
    ENABLE_TEMPORAL_ARCHAEOLOGY: bool = Field(True, env="TELEGRAM_ENABLE_TEMPORAL_ARCHAEOLOGY")
    
    # AI Configuration
    DEFAULT_LLM_PROVIDER: str = Field("anthropic", env="TELEGRAM_LLM_PROVIDER")
    DEFAULT_MODEL_NAME: str = Field("claude-3-sonnet", env="TELEGRAM_MODEL_NAME")
    DEFAULT_TEMPERATURE: float = Field(0.7, env="TELEGRAM_LLM_TEMPERATURE")
    DEFAULT_MAX_TOKENS: int = Field(200, env="TELEGRAM_LLM_MAX_TOKENS")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Safety Configuration Templates
SAFETY_LEVELS = {
    "conservative": {
        "max_messages_per_day": 30,
        "max_groups_per_day": 1,
        "max_dms_per_day": 3,
        "min_delay_between_messages": 60,
        "response_probability": 0.2,
        "risk_threshold": 50.0,
        "flood_wait_tolerance": 1
    },
    "moderate": {
        "max_messages_per_day": 50,
        "max_groups_per_day": 2,
        "max_dms_per_day": 5,
        "min_delay_between_messages": 30,
        "response_probability": 0.3,
        "risk_threshold": 70.0,
        "flood_wait_tolerance": 2
    },
    "aggressive": {
        "max_messages_per_day": 80,
        "max_groups_per_day": 3,
        "max_dms_per_day": 8,
        "min_delay_between_messages": 15,
        "response_probability": 0.4,
        "risk_threshold": 80.0,
        "flood_wait_tolerance": 3
    }
}

# Community Type Configurations
COMMUNITY_CONFIGS = {
    "supergroup": {
        "default_strategy": "participant",
        "formality_level": "casual",
        "response_rate_target": 0.3,
        "max_daily_messages": 10,
        "lurk_period_days": 3
    },
    "channel": {
        "default_strategy": "lurker",
        "formality_level": "semi-formal",
        "response_rate_target": 0.1,
        "max_daily_messages": 2,
        "lurk_period_days": 7
    },
    "basic_group": {
        "default_strategy": "participant",
        "formality_level": "casual",
        "response_rate_target": 0.4,
        "max_daily_messages": 15,
        "lurk_period_days": 1
    },
    "private_chat": {
        "default_strategy": "contributor",
        "formality_level": "adaptive",
        "response_rate_target": 0.6,
        "max_daily_messages": 20,
        "lurk_period_days": 0
    }
}

# Error Pattern Configurations
ERROR_PATTERNS = {
    "FloodWait": {
        "severity": "high",
        "risk_increase": 15.0,
        "cooldown_multiplier": 2.0,
        "max_occurrences_per_hour": 3,
        "recovery_strategy": "exponential_backoff"
    },
    "SpamWait": {
        "severity": "critical",
        "risk_increase": 25.0,
        "cooldown_multiplier": 3.0,
        "max_occurrences_per_hour": 1,
        "recovery_strategy": "immediate_stop"
    },
    "PeerFlood": {
        "severity": "critical",
        "risk_increase": 30.0,
        "cooldown_multiplier": 4.0,
        "max_occurrences_per_hour": 1,
        "recovery_strategy": "emergency_stop"
    },
    "UserBannedInChannel": {
        "severity": "critical",
        "risk_increase": 50.0,
        "cooldown_multiplier": 0,  # Immediate stop
        "max_occurrences_per_hour": 0,
        "recovery_strategy": "manual_review_required"
    },
    "ChatWriteForbidden": {
        "severity": "medium",
        "risk_increase": 10.0,
        "cooldown_multiplier": 1.5,
        "max_occurrences_per_hour": 5,
        "recovery_strategy": "skip_and_continue"
    }
}

# Natural Behavior Patterns
BEHAVIOR_PATTERNS = {
    "typing_speeds": {
        "min_chars_per_second": 8,
        "max_chars_per_second": 25,
        "average": 15,
        "variation_factor": 0.3
    },
    "response_times": {
        "immediate": {"min": 2, "max": 10, "probability": 0.1},
        "quick": {"min": 10, "max": 60, "probability": 0.4},
        "normal": {"min": 60, "max": 300, "probability": 0.4},
        "delayed": {"min": 300, "max": 1800, "probability": 0.1}
    },
    "activity_hours": {
        "peak": [9, 12, 15, 18, 20, 21],
        "normal": [8, 10, 11, 13, 14, 16, 17, 19, 22],
        "low": [7, 23],
        "inactive": [0, 1, 2, 3, 4, 5, 6]
    },
    "weekend_adjustments": {
        "activity_multiplier": 0.7,
        "delayed_responses": 1.5,
        "casual_language_increase": 0.2
    }
}

# Content Safety Filters
CONTENT_FILTERS = {
    "forbidden_patterns": [
        r"(phone|email|address|password|ssn|credit card)",
        r"(hate|violence|explicit|illegal)",
        r"(buy now|click here|limited time|urgent|act fast)",
        r"(diagnose|treatment|medical|prescription|cure)",
        r"(invest|stock|crypto|trading|financial advice)"
    ],
    "warning_patterns": [
        r"(personal information|private details)",
        r"(spam|promotional|advertisement)",
        r"(off-topic|inappropriate|violation)"
    ],
    "safe_topics": [
        "technology", "science", "education", "entertainment",
        "sports", "books", "movies", "music", "travel",
        "food", "photography", "art", "gaming"
    ]
}

# Personality Configuration Templates
PERSONALITY_TEMPLATES = {
    "professional": {
        "formality": 0.8,
        "friendliness": 0.6,
        "humor": 0.3,
        "empathy": 0.7,
        "assertiveness": 0.7,
        "curiosity": 0.6,
        "supportiveness": 0.8
    },
    "casual": {
        "formality": 0.3,
        "friendliness": 0.8,
        "humor": 0.7,
        "empathy": 0.8,
        "assertiveness": 0.5,
        "curiosity": 0.7,
        "supportiveness": 0.7
    },
    "helpful": {
        "formality": 0.5,
        "friendliness": 0.9,
        "humor": 0.5,
        "empathy": 0.9,
        "assertiveness": 0.6,
        "curiosity": 0.8,
        "supportiveness": 0.9
    },
    "expert": {
        "formality": 0.7,
        "friendliness": 0.6,
        "humor": 0.4,
        "empathy": 0.6,
        "assertiveness": 0.8,
        "curiosity": 0.9,
        "supportiveness": 0.8
    }
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "health_check": {
        "interval_seconds": 300,
        "timeout_seconds": 30,
        "critical_metrics": [
            "risk_score", "flood_wait_count", "spam_warnings",
            "daily_message_count", "error_rate"
        ]
    },
    "safety_monitor": {
        "interval_seconds": 60,
        "alert_thresholds": {
            "risk_score": 70.0,
            "consecutive_errors": 5,
            "flood_waits_per_hour": 3,
            "message_rate_limit": 0.8  # 80% of daily limit
        }
    },
    "performance_monitor": {
        "interval_seconds": 3600,
        "metrics_retention_days": 30,
        "analytics_update_frequency": "hourly"
    }
}

# Legal and Compliance Configuration
COMPLIANCE_CONFIG = {
    "ai_disclosure": {
        "required": True,
        "bio_text": "🤖 AI Assistant - Here to help and learn!",
        "profile_indicators": ["AI", "Bot", "Assistant"],
        "transparency_level": "full"
    },
    "gdpr": {
        "data_retention_days": 30,
        "consent_required": True,
        "right_to_deletion": True,
        "data_portability": True,
        "privacy_by_design": True
    },
    "rate_limiting": {
        "respect_telegram_limits": True,
        "conservative_approach": True,
        "escalation_procedures": True,
        "manual_review_triggers": [
            "multiple_bans", "spam_detection", "unusual_patterns"
        ]
    }
}

# Integration Configuration
INTEGRATION_CONFIG = {
    "revolutionary_features": {
        "consciousness_mirror": {
            "enabled": True,
            "adaptation_frequency": "per_conversation",
            "personality_flexibility": 0.3
        },
        "memory_palace": {
            "enabled": True,
            "retention_period_days": 90,
            "importance_threshold": 0.5,
            "context_window_size": 50
        },
        "emotion_detection": {
            "enabled": True,
            "confidence_threshold": 0.6,
            "response_adaptation": True,
            "emotional_memory": True
        },
        "digital_telepathy": {
            "enabled": True,
            "deep_understanding": True,
            "context_inference": True,
            "predictive_responses": True
        },
        "temporal_archaeology": {
            "enabled": True,
            "timing_optimization": True,
            "pattern_recognition": True,
            "behavioral_prediction": True
        }
    },
    "llm_providers": {
        "anthropic": {
            "default_model": "claude-3-sonnet",
            "fallback_model": "claude-3-haiku",
            "max_tokens": 200,
            "temperature": 0.7
        },
        "openai": {
            "default_model": "gpt-4-turbo",
            "fallback_model": "gpt-3.5-turbo",
            "max_tokens": 200,
            "temperature": 0.7
        }
    }
}

# Development and Testing Configuration
DEVELOPMENT_CONFIG = {
    "test_mode": {
        "enabled": False,
        "mock_telegram_api": False,
        "accelerated_timing": False,
        "verbose_logging": True
    },
    "debugging": {
        "log_level": "INFO",
        "save_conversation_logs": True,
        "track_personality_changes": True,
        "monitor_api_calls": True
    },
    "testing": {
        "test_phone_numbers": ["+1234567890"],
        "test_chat_ids": [-1001234567890],
        "safety_overrides": False,
        "mock_responses": False
    }
}


def get_config() -> TelegramConfig:
    """Get Telegram configuration instance"""
    return TelegramConfig()


def get_safety_config(level: str) -> Dict[str, Any]:
    """Get safety configuration for specific level"""
    return SAFETY_LEVELS.get(level, SAFETY_LEVELS["conservative"])


def get_community_config(community_type: str) -> Dict[str, Any]:
    """Get community configuration for specific type"""
    return COMMUNITY_CONFIGS.get(community_type, COMMUNITY_CONFIGS["supergroup"])


def get_personality_template(template_name: str) -> Dict[str, float]:
    """Get personality template by name"""
    return PERSONALITY_TEMPLATES.get(template_name, PERSONALITY_TEMPLATES["helpful"])


# Environment-specific overrides
def apply_environment_overrides(config: TelegramConfig) -> TelegramConfig:
    """Apply environment-specific configuration overrides"""
    
    if config.ENVIRONMENT == TelegramEnvironment.DEVELOPMENT:
        # Development overrides
        config.DEFAULT_MAX_MESSAGES_PER_DAY = 20
        config.HEALTH_CHECK_INTERVAL = 60
        config.MIN_MESSAGE_DELAY = 10
        
    elif config.ENVIRONMENT == TelegramEnvironment.STAGING:
        # Staging overrides
        config.DEFAULT_MAX_MESSAGES_PER_DAY = 30
        config.MAX_RISK_SCORE = 60.0
        
    elif config.ENVIRONMENT == TelegramEnvironment.PRODUCTION:
        # Production overrides (most conservative)
        config.REQUIRE_AI_DISCLOSURE = True
        config.REQUIRE_GDPR_CONSENT = True
        config.DEFAULT_SAFETY_LEVEL = "conservative"
    
    return config


# Export configuration
__all__ = [
    'TelegramConfig',
    'SAFETY_LEVELS',
    'COMMUNITY_CONFIGS', 
    'ERROR_PATTERNS',
    'BEHAVIOR_PATTERNS',
    'CONTENT_FILTERS',
    'PERSONALITY_TEMPLATES',
    'MONITORING_CONFIG',
    'COMPLIANCE_CONFIG',
    'INTEGRATION_CONFIG',
    'DEVELOPMENT_CONFIG',
    'get_config',
    'get_safety_config',
    'get_community_config',
    'get_personality_template',
    'apply_environment_overrides'
]