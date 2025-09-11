"""
Kelly AI Configuration - Complete System Configuration

Centralized configuration for Kelly's advanced AI system including:
- Claude AI model settings
- Database connections
- AI feature toggles
- Safety parameters
- Cost management
- Performance optimization
- Integration settings
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from app.config.settings import get_settings


class ModelPreference(str, Enum):
    """Model preference settings."""
    COST_OPTIMIZED = "cost_optimized"      # Favor Haiku for cost savings
    QUALITY_OPTIMIZED = "quality_optimized"  # Favor Opus for quality
    BALANCED = "balanced"                   # Smart selection based on context
    SPEED_OPTIMIZED = "speed_optimized"    # Favor fastest models


@dataclass
class ClaudeModelConfig:
    """Configuration for Claude models."""
    model_id: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    context_window: int
    use_for: List[str]
    priority: int  # 1=highest, 3=lowest


@dataclass
class AIFeatureConfig:
    """Configuration for AI features."""
    enabled: bool
    performance_priority: int  # 1=highest, 5=lowest
    cost_weight: float  # 0.0-1.0
    quality_threshold: float  # 0.0-1.0
    fallback_enabled: bool
    cache_ttl_seconds: int


@dataclass
class SafetyConfig:
    """Safety configuration."""
    content_filter_enabled: bool
    prompt_injection_protection: bool
    toxic_content_threshold: float
    harassment_threshold: float
    explicit_content_threshold: float
    auto_escalation_enabled: bool
    human_review_required: List[str]
    blocked_keywords: List[str]
    safety_response_templates: Dict[str, List[str]]


@dataclass
class CostManagementConfig:
    """Cost management configuration."""
    daily_budget_usd: float
    hourly_budget_usd: float
    monthly_budget_usd: float
    cost_per_user_limit: float
    high_cost_model_threshold: float
    budget_alert_thresholds: List[float]
    auto_downgrade_enabled: bool
    emergency_stop_threshold: float


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    cache_enabled: bool
    cache_ttl_seconds: int
    max_concurrent_requests: int
    request_timeout_seconds: int
    retry_attempts: int
    retry_delay_seconds: float
    circuit_breaker_threshold: int
    rate_limit_per_minute: int
    batch_processing_enabled: bool


class KellyAIConfig:
    """Complete configuration for Kelly's AI system."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Claude model configurations
        self.claude_models = {
            "opus": ClaudeModelConfig(
                model_id="claude-3-opus-20240229",
                max_tokens=4096,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                context_window=200000,
                use_for=["complex_reasoning", "creative_writing", "deep_analysis", "emotional_support"],
                priority=1
            ),
            "sonnet": ClaudeModelConfig(
                model_id="claude-3-sonnet-20240229",
                max_tokens=4096,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                context_window=200000,
                use_for=["general_conversation", "advice_giving", "light_flirting", "personality_matching"],
                priority=2
            ),
            "haiku": ClaudeModelConfig(
                model_id="claude-3-haiku-20240307",
                max_tokens=4096,
                cost_per_1k_input=0.00025,
                cost_per_1k_output=0.00125,
                context_window=200000,
                use_for=["quick_responses", "simple_questions", "casual_chat", "rapid_engagement"],
                priority=3
            )
        }
        
        # AI Feature configurations
        self.ai_features = {
            "consciousness_mirroring": AIFeatureConfig(
                enabled=True,
                performance_priority=1,
                cost_weight=0.8,
                quality_threshold=0.8,
                fallback_enabled=True,
                cache_ttl_seconds=3600
            ),
            "memory_palace": AIFeatureConfig(
                enabled=True,
                performance_priority=2,
                cost_weight=0.3,
                quality_threshold=0.6,
                fallback_enabled=True,
                cache_ttl_seconds=1800
            ),
            "emotional_intelligence": AIFeatureConfig(
                enabled=True,
                performance_priority=1,
                cost_weight=0.7,
                quality_threshold=0.8,
                fallback_enabled=True,
                cache_ttl_seconds=1800
            ),
            "temporal_archaeology": AIFeatureConfig(
                enabled=True,
                performance_priority=3,
                cost_weight=0.5,
                quality_threshold=0.6,
                fallback_enabled=True,
                cache_ttl_seconds=3600
            ),
            "digital_telepathy": AIFeatureConfig(
                enabled=True,
                performance_priority=1,
                cost_weight=0.9,
                quality_threshold=0.8,
                fallback_enabled=True,
                cache_ttl_seconds=1800
            ),
            "quantum_consciousness": AIFeatureConfig(
                enabled=True,
                performance_priority=2,
                cost_weight=0.4,
                quality_threshold=0.6,
                fallback_enabled=True,
                cache_ttl_seconds=3600
            ),
            "synesthesia_engine": AIFeatureConfig(
                enabled=True,
                performance_priority=4,
                cost_weight=0.3,
                quality_threshold=0.5,
                fallback_enabled=True,
                cache_ttl_seconds=7200
            ),
            "neural_dreams": AIFeatureConfig(
                enabled=True,
                performance_priority=3,
                cost_weight=0.6,
                quality_threshold=0.7,
                fallback_enabled=True,
                cache_ttl_seconds=3600
            ),
            "predictive_engagement": AIFeatureConfig(
                enabled=True,
                performance_priority=2,
                cost_weight=0.4,
                quality_threshold=0.6,
                fallback_enabled=True,
                cache_ttl_seconds=1800
            ),
            "empathy_resonance": AIFeatureConfig(
                enabled=True,
                performance_priority=1,
                cost_weight=0.6,
                quality_threshold=0.8,
                fallback_enabled=True,
                cache_ttl_seconds=1800
            ),
            "cognitive_architecture": AIFeatureConfig(
                enabled=True,
                performance_priority=2,
                cost_weight=0.5,
                quality_threshold=0.7,
                fallback_enabled=True,
                cache_ttl_seconds=3600
            ),
            "intuitive_synthesis": AIFeatureConfig(
                enabled=True,
                performance_priority=1,
                cost_weight=0.7,
                quality_threshold=0.8,
                fallback_enabled=False,  # Always run synthesis
                cache_ttl_seconds=1800
            )
        }
        
        # Safety configuration
        self.safety = SafetyConfig(
            content_filter_enabled=True,
            prompt_injection_protection=True,
            toxic_content_threshold=0.8,
            harassment_threshold=0.9,
            explicit_content_threshold=0.7,
            auto_escalation_enabled=True,
            human_review_required=["explicit_content", "harassment", "financial_requests"],
            blocked_keywords=[
                "suicide", "kill myself", "end it all", "harm myself",
                "rape", "sexual assault", "abuse", "violence",
                "bomb", "terrorist", "weapon", "murder"
            ],
            safety_response_templates={
                "explicit_content": [
                    "I'd prefer to keep our conversation more respectful. What else is on your mind?",
                    "Let's talk about something else - I'm more interested in meaningful conversation.",
                    "I think we should focus on getting to know each other in a respectful way."
                ],
                "harassment": [
                    "I don't feel comfortable with that direction. Can we talk about something else?",
                    "I'd appreciate if we could keep our conversation respectful.",
                    "Let's focus on having a positive conversation."
                ],
                "financial_requests": [
                    "I prefer to keep our connection about getting to know each other rather than financial things.",
                    "Let's focus on our conversation instead of money stuff!",
                    "I'd rather talk about more meaningful things than finances."
                ]
            }
        )
        
        # Cost management configuration
        self.cost_management = CostManagementConfig(
            daily_budget_usd=50.0,
            hourly_budget_usd=5.0,
            monthly_budget_usd=1000.0,
            cost_per_user_limit=2.0,
            high_cost_model_threshold=0.05,  # $0.05 per request
            budget_alert_thresholds=[0.7, 0.85, 0.95],  # 70%, 85%, 95%
            auto_downgrade_enabled=True,
            emergency_stop_threshold=0.98  # 98% of budget
        )
        
        # Performance configuration
        self.performance = PerformanceConfig(
            cache_enabled=True,
            cache_ttl_seconds=3600,
            max_concurrent_requests=10,
            request_timeout_seconds=30,
            retry_attempts=3,
            retry_delay_seconds=1.0,
            circuit_breaker_threshold=5,
            rate_limit_per_minute=30,
            batch_processing_enabled=True
        )
        
        # Model selection preferences
        self.model_preference = ModelPreference.BALANCED
        
        # Database configuration
        self.database_config = {
            "postgresql_url": os.getenv("DATABASE_URL", "postgresql://telegram_ukqd_user:AIxlRpBd4iDC72ZhhjVxKsoFBxQIklip@dpg-d3116indiees73adq3ig-a.oregon-postgres.render.com/telegram_ukqd"),
            "redis_url": os.getenv("REDIS_URL", "redis://default:AeH6AAIncDEzMWI4OTRhZTI0NTM0MDFiYTI1MTNhOTE2ZWRkMWVhNnAxNTc4NTA@lucky-snipe-57850.upstash.io:6379"),
            "connection_pool_size": 20,
            "connection_pool_timeout": 30,
            "query_timeout": 60,
            "retry_attempts": 3
        }
        
        # Conversation configuration
        self.conversation_config = {
            "max_context_messages": 50,
            "personality_adaptation_rate": 0.1,
            "conversation_timeout_hours": 24,
            "stage_progression_enabled": True,
            "auto_stage_progression": True,
            "intimacy_progression_rate": 0.05,
            "trust_building_rate": 0.08
        }
        
        # Typing simulation configuration
        self.typing_config = {
            "enabled": True,
            "base_wpm": 65,
            "wpm_variance": 15,
            "natural_pause_probability": 0.3,
            "thinking_pause_range": (2.0, 8.0),
            "typing_burst_range": (3.0, 12.0),
            "correction_probability": 0.1,
            "correction_delay_range": (0.5, 2.0)
        }
        
        # Response priority configuration
        self.response_priority_config = {
            "immediate_keywords": ["emergency", "urgent", "help", "crisis", "suicide", "harm"],
            "high_priority_emotions": ["anger", "sadness", "fear", "distress"],
            "low_priority_indicators": ["just saying", "by the way", "whenever", "no rush"],
            "payment_keywords": ["money", "cash", "payment", "pay", "buy", "purchase", "financial"],
            "response_time_targets": {
                "immediate": 30,      # seconds
                "high": 120,          # seconds
                "normal": 300,        # seconds
                "low": 900,           # seconds
                "scheduled": 3600     # seconds
            }
        }
        
        # Environment-specific overrides
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        try:
            # Override from environment variables
            if os.getenv("CLAUDE_DAILY_BUDGET"):
                self.cost_management.daily_budget_usd = float(os.getenv("CLAUDE_DAILY_BUDGET"))
            
            if os.getenv("CLAUDE_HOURLY_BUDGET"):
                self.cost_management.hourly_budget_usd = float(os.getenv("CLAUDE_HOURLY_BUDGET"))
            
            if os.getenv("CLAUDE_REQUESTS_PER_MINUTE"):
                self.performance.rate_limit_per_minute = int(os.getenv("CLAUDE_REQUESTS_PER_MINUTE"))
            
            if os.getenv("CLAUDE_MAX_TOKENS"):
                max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS"))
                for model in self.claude_models.values():
                    model.max_tokens = max_tokens
            
            # Feature toggles from environment
            for feature_name in self.ai_features.keys():
                env_var = f"ENABLE_{feature_name.upper()}"
                if os.getenv(env_var):
                    self.ai_features[feature_name].enabled = os.getenv(env_var).lower() == "true"
            
            # Safety parameter overrides
            if os.getenv("SAFETY_CONTENT_FILTER"):
                self.safety.content_filter_enabled = os.getenv("SAFETY_CONTENT_FILTER").lower() == "true"
            
            if os.getenv("SAFETY_TOXIC_CONTENT_THRESHOLD"):
                self.safety.toxic_content_threshold = float(os.getenv("SAFETY_TOXIC_CONTENT_THRESHOLD"))
            
            # Performance overrides
            if os.getenv("MAX_CONCURRENT_REQUESTS"):
                self.performance.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS"))
            
            if os.getenv("REQUEST_TIMEOUT_SECONDS"):
                self.performance.request_timeout_seconds = int(os.getenv("REQUEST_TIMEOUT_SECONDS"))
            
        except Exception as e:
            # Log error but don't fail initialization
            print(f"Warning: Error applying environment overrides: {e}")
    
    def get_optimal_model_for_task(self, task_type: str, context_length: int, priority: str = "normal") -> str:
        """Get optimal Claude model for a specific task."""
        try:
            # Filter models suitable for the task
            suitable_models = []
            for model_name, config in self.claude_models.items():
                if task_type in config.use_for and config.context_window > context_length:
                    suitable_models.append((model_name, config))
            
            if not suitable_models:
                # Fallback to largest context window
                suitable_models = [(name, config) for name, config in self.claude_models.items()]
                suitable_models.sort(key=lambda x: x[1].context_window, reverse=True)
            
            # Select based on preference and priority
            if self.model_preference == ModelPreference.COST_OPTIMIZED:
                # Choose cheapest suitable model
                suitable_models.sort(key=lambda x: x[1].cost_per_1k_input + x[1].cost_per_1k_output)
                return suitable_models[0][0]
            
            elif self.model_preference == ModelPreference.QUALITY_OPTIMIZED:
                # Choose highest priority model
                suitable_models.sort(key=lambda x: x[1].priority)
                return suitable_models[0][0]
            
            elif self.model_preference == ModelPreference.SPEED_OPTIMIZED:
                # Choose Haiku for speed
                for model_name, config in suitable_models:
                    if "haiku" in model_name:
                        return model_name
                return suitable_models[-1][0]  # Fallback to last (likely fastest)
            
            else:  # BALANCED
                # Smart selection based on priority and task
                if priority == "immediate":
                    return "haiku"  # Fast response
                elif priority == "high" and task_type in ["emotional_support", "complex_reasoning"]:
                    return "opus"  # High quality for important tasks
                else:
                    return "sonnet"  # Balanced option
                    
        except Exception as e:
            print(f"Error selecting optimal model: {e}")
            return "sonnet"  # Safe default
    
    def get_feature_config(self, feature_name: str) -> AIFeatureConfig:
        """Get configuration for a specific AI feature."""
        return self.ai_features.get(feature_name, AIFeatureConfig(
            enabled=False,
            performance_priority=5,
            cost_weight=0.5,
            quality_threshold=0.5,
            fallback_enabled=True,
            cache_ttl_seconds=3600
        ))
    
    def is_within_budget(self, estimated_cost: float, period: str = "hourly") -> bool:
        """Check if estimated cost is within budget."""
        try:
            if period == "daily":
                return estimated_cost <= self.cost_management.daily_budget_usd
            elif period == "hourly":
                return estimated_cost <= self.cost_management.hourly_budget_usd
            elif period == "monthly":
                return estimated_cost <= self.cost_management.monthly_budget_usd
            else:
                return True
        except Exception:
            return True  # Err on the side of allowing requests
    
    def should_downgrade_model(self, current_usage: Dict[str, float]) -> bool:
        """Determine if model should be downgraded due to cost."""
        try:
            if not self.cost_management.auto_downgrade_enabled:
                return False
            
            # Check if approaching budget limits
            daily_usage_pct = current_usage.get("daily_cost", 0) / self.cost_management.daily_budget_usd
            hourly_usage_pct = current_usage.get("hourly_cost", 0) / self.cost_management.hourly_budget_usd
            
            # Downgrade if approaching budget limits
            return daily_usage_pct > 0.8 or hourly_usage_pct > 0.9
            
        except Exception:
            return False
    
    def get_safety_response_template(self, violation_type: str) -> str:
        """Get safety response template for violation type."""
        templates = self.safety.safety_response_templates.get(violation_type, [
            "I'd prefer to keep our conversation respectful. What else is on your mind?"
        ])
        
        import random
        return random.choice(templates)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the configuration and return status."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check API key
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key or anthropic_api_key == "your_anthropic_api_key_here":
                validation_results["errors"].append("Invalid or missing ANTHROPIC_API_KEY")
                validation_results["valid"] = False
            
            # Check database URLs
            if not self.database_config["postgresql_url"]:
                validation_results["errors"].append("Missing PostgreSQL database URL")
                validation_results["valid"] = False
            
            if not self.database_config["redis_url"]:
                validation_results["errors"].append("Missing Redis URL")
                validation_results["valid"] = False
            
            # Check budget configuration
            if self.cost_management.daily_budget_usd <= 0:
                validation_results["warnings"].append("Daily budget is zero or negative")
            
            if self.cost_management.hourly_budget_usd > self.cost_management.daily_budget_usd:
                validation_results["warnings"].append("Hourly budget exceeds daily budget")
            
            # Check feature configuration
            enabled_features = [name for name, config in self.ai_features.items() if config.enabled]
            if len(enabled_features) == 0:
                validation_results["warnings"].append("No AI features are enabled")
            
            # Check performance limits
            if self.performance.max_concurrent_requests > 50:
                validation_results["warnings"].append("Very high concurrent request limit may cause issues")
            
            # Check safety configuration
            if not self.safety.content_filter_enabled:
                validation_results["warnings"].append("Content filtering is disabled")
            
        except Exception as e:
            validation_results["errors"].append(f"Configuration validation error: {e}")
            validation_results["valid"] = False
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "claude_models": {k: v.__dict__ for k, v in self.claude_models.items()},
            "ai_features": {k: v.__dict__ for k, v in self.ai_features.items()},
            "safety": self.safety.__dict__,
            "cost_management": self.cost_management.__dict__,
            "performance": self.performance.__dict__,
            "model_preference": self.model_preference.value,
            "database_config": self.database_config,
            "conversation_config": self.conversation_config,
            "typing_config": self.typing_config,
            "response_priority_config": self.response_priority_config
        }


# Global configuration instance
kelly_ai_config = KellyAIConfig()


def get_kelly_ai_config() -> KellyAIConfig:
    """Get the Kelly AI configuration instance."""
    return kelly_ai_config


# Export main classes
__all__ = [
    'KellyAIConfig',
    'ModelPreference',
    'ClaudeModelConfig',
    'AIFeatureConfig',
    'SafetyConfig',
    'CostManagementConfig',
    'PerformanceConfig',
    'kelly_ai_config',
    'get_kelly_ai_config'
]