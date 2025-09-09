"""
System Configuration Models

Defines models for system configuration, feature flags, and runtime settings.
Supports dynamic configuration management and A/B testing capabilities.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON, 
    Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class ConfigurationScope(str, Enum):
    """Configuration scope enumeration."""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    CONVERSATION = "conversation"
    FEATURE = "feature"
    ENVIRONMENT = "environment"


class ConfigurationType(str, Enum):
    """Configuration type enumeration."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    ARRAY = "array"
    SECRET = "secret"


class FeatureFlagStrategy(str, Enum):
    """Feature flag rollout strategy enumeration."""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    USER_ATTRIBUTES = "user_attributes"
    GRADUAL_ROLLOUT = "gradual_rollout"
    A_B_TEST = "a_b_test"


class SystemConfiguration(FullAuditModel):
    """
    System configuration settings with hierarchical support.
    
    Manages configuration values across different scopes and environments
    with validation and versioning support.
    """
    
    __tablename__ = "system_configurations"
    
    # Configuration identification
    key = Column(
        String(200),
        nullable=False,
        index=True,
        comment="Configuration key (hierarchical dot notation supported)"
    )
    
    scope = Column(
        String(20),
        default=ConfigurationScope.GLOBAL,
        nullable=False,
        index=True,
        comment="Configuration scope"
    )
    
    environment = Column(
        String(20),
        default="production",
        nullable=False,
        index=True,
        comment="Environment this configuration applies to"
    )
    
    # Configuration value and metadata
    value_type = Column(
        String(20),
        nullable=False,
        comment="Type of configuration value"
    )
    
    string_value = Column(
        Text,
        nullable=True,
        comment="String value for string and secret types"
    )
    
    integer_value = Column(
        Integer,
        nullable=True,
        comment="Integer value"
    )
    
    float_value = Column(
        Float,
        nullable=True,
        comment="Float value"
    )
    
    boolean_value = Column(
        Boolean,
        nullable=True,
        comment="Boolean value"
    )
    
    json_value = Column(
        JSONB,
        nullable=True,
        comment="JSON value for complex configurations"
    )
    
    array_value = Column(
        ARRAY(String),
        nullable=True,
        comment="Array value"
    )
    
    # Configuration metadata
    description = Column(
        Text,
        nullable=False,
        comment="Description of this configuration"
    )
    
    default_value = Column(
        Text,
        nullable=True,
        comment="Default value (serialized as string)"
    )
    
    validation_rules = Column(
        JSONB,
        nullable=True,
        comment="Validation rules for this configuration"
    )
    
    # Configuration management
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this configuration is active"
    )
    
    is_secret = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this configuration contains sensitive data"
    )
    
    is_required = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this configuration is required"
    )
    
    # Versioning and history
    version = Column(
        Integer,
        default=1,
        nullable=False,
        comment="Configuration version number"
    )
    
    previous_value = Column(
        Text,
        nullable=True,
        comment="Previous value (for rollback capability)"
    )
    
    change_reason = Column(
        Text,
        nullable=True,
        comment="Reason for last configuration change"
    )
    
    # Usage tracking
    last_accessed_at = Column(
        "last_accessed_at",
        nullable=True,
        index=True,
        comment="When configuration was last accessed"
    )
    
    access_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times configuration has been accessed"
    )
    
    # Cache control
    cache_ttl_seconds = Column(
        Integer,
        default=3600,
        nullable=False,
        comment="Cache TTL in seconds"
    )
    
    last_cached_at = Column(
        "last_cached_at",
        nullable=True,
        comment="When configuration was last cached"
    )
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_config_key_scope_env', 'key', 'scope', 'environment'),
        Index('idx_config_active', 'is_active'),
        Index('idx_config_secret', 'is_secret'),
        Index('idx_config_version', 'version'),
        UniqueConstraint('key', 'scope', 'environment', name='uq_config_key_scope_env'),
        CheckConstraint('version >= 1', name='ck_version_positive'),
        CheckConstraint('access_count >= 0', name='ck_access_count_positive'),
        CheckConstraint('cache_ttl_seconds > 0', name='ck_cache_ttl_positive'),
    )
    
    def get_value(self) -> Any:
        """Get the configuration value based on its type."""
        if self.value_type == ConfigurationType.STRING:
            return self.string_value
        elif self.value_type == ConfigurationType.INTEGER:
            return self.integer_value
        elif self.value_type == ConfigurationType.FLOAT:
            return self.float_value
        elif self.value_type == ConfigurationType.BOOLEAN:
            return self.boolean_value
        elif self.value_type == ConfigurationType.JSON:
            return self.json_value
        elif self.value_type == ConfigurationType.ARRAY:
            return self.array_value
        elif self.value_type == ConfigurationType.SECRET:
            # For secrets, consider encryption/decryption here
            return self.string_value
        else:
            return None
    
    def set_value(self, value: Any, change_reason: Optional[str] = None) -> None:
        """Set the configuration value based on its type."""
        # Store previous value for rollback
        self.previous_value = str(self.get_value()) if self.get_value() is not None else None
        
        # Set new value based on type
        if self.value_type == ConfigurationType.STRING:
            self.string_value = str(value) if value is not None else None
        elif self.value_type == ConfigurationType.INTEGER:
            self.integer_value = int(value) if value is not None else None
        elif self.value_type == ConfigurationType.FLOAT:
            self.float_value = float(value) if value is not None else None
        elif self.value_type == ConfigurationType.BOOLEAN:
            self.boolean_value = bool(value) if value is not None else None
        elif self.value_type == ConfigurationType.JSON:
            self.json_value = value
        elif self.value_type == ConfigurationType.ARRAY:
            self.array_value = list(value) if value is not None else None
        elif self.value_type == ConfigurationType.SECRET:
            # For secrets, consider encryption here
            self.string_value = str(value) if value is not None else None
        
        # Update metadata
        self.version += 1
        self.change_reason = change_reason
        self.updated_at = datetime.utcnow()
    
    def validate_value(self, value: Any) -> bool:
        """Validate configuration value against validation rules."""
        if not self.validation_rules:
            return True
        
        rules = self.validation_rules
        
        # Required check
        if rules.get('required', False) and value is None:
            return False
        
        if value is None:
            return True  # Optional value
        
        # Type-specific validations
        if self.value_type == ConfigurationType.STRING:
            return self._validate_string(value, rules)
        elif self.value_type == ConfigurationType.INTEGER:
            return self._validate_integer(value, rules)
        elif self.value_type == ConfigurationType.FLOAT:
            return self._validate_float(value, rules)
        elif self.value_type == ConfigurationType.ARRAY:
            return self._validate_array(value, rules)
        
        return True
    
    def _validate_string(self, value: str, rules: Dict[str, Any]) -> bool:
        """Validate string value."""
        if 'min_length' in rules and len(value) < rules['min_length']:
            return False
        if 'max_length' in rules and len(value) > rules['max_length']:
            return False
        if 'pattern' in rules:
            import re
            if not re.match(rules['pattern'], value):
                return False
        if 'allowed_values' in rules and value not in rules['allowed_values']:
            return False
        return True
    
    def _validate_integer(self, value: int, rules: Dict[str, Any]) -> bool:
        """Validate integer value."""
        if 'min_value' in rules and value < rules['min_value']:
            return False
        if 'max_value' in rules and value > rules['max_value']:
            return False
        if 'allowed_values' in rules and value not in rules['allowed_values']:
            return False
        return True
    
    def _validate_float(self, value: float, rules: Dict[str, Any]) -> bool:
        """Validate float value."""
        if 'min_value' in rules and value < rules['min_value']:
            return False
        if 'max_value' in rules and value > rules['max_value']:
            return False
        return True
    
    def _validate_array(self, value: List, rules: Dict[str, Any]) -> bool:
        """Validate array value."""
        if 'min_items' in rules and len(value) < rules['min_items']:
            return False
        if 'max_items' in rules and len(value) > rules['max_items']:
            return False
        if 'allowed_values' in rules:
            for item in value:
                if item not in rules['allowed_values']:
                    return False
        return True
    
    def record_access(self) -> None:
        """Record that this configuration was accessed."""
        self.last_accessed_at = datetime.utcnow()
        self.access_count += 1
    
    def is_cache_valid(self) -> bool:
        """Check if cached configuration is still valid."""
        if not self.last_cached_at:
            return False
        
        cache_expires_at = self.last_cached_at + timedelta(seconds=self.cache_ttl_seconds)
        return datetime.utcnow() < cache_expires_at
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for management interfaces."""
        return {
            "key": self.key,
            "scope": self.scope,
            "environment": self.environment,
            "type": self.value_type,
            "value": "***REDACTED***" if self.is_secret else self.get_value(),
            "is_active": self.is_active,
            "is_required": self.is_required,
            "version": self.version,
            "last_accessed": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
        }


class FeatureFlag(FullAuditModel):
    """
    Feature flag system for controlled rollouts and A/B testing.
    
    Manages feature enablement with sophisticated targeting and
    gradual rollout capabilities.
    """
    
    __tablename__ = "feature_flags"
    
    # Feature identification
    flag_key = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique feature flag key"
    )
    
    name = Column(
        String(200),
        nullable=False,
        comment="Human-readable feature name"
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Description of the feature"
    )
    
    category = Column(
        String(50),
        nullable=True,
        index=True,
        comment="Feature category for organization"
    )
    
    # Feature status
    is_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether the feature is enabled"
    )
    
    is_permanent = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this is a permanent feature (not a flag)"
    )
    
    environment = Column(
        String(20),
        default="production",
        nullable=False,
        index=True,
        comment="Environment where flag is active"
    )
    
    # Rollout strategy
    strategy = Column(
        String(20),
        default=FeatureFlagStrategy.ALL_USERS,
        nullable=False,
        comment="Rollout strategy"
    )
    
    rollout_percentage = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Percentage of users to include (0-100)"
    )
    
    target_user_ids = Column(
        ARRAY(String),
        nullable=True,
        comment="Specific user IDs to target"
    )
    
    target_attributes = Column(
        JSONB,
        nullable=True,
        comment="User attributes to target"
    )
    
    exclusion_attributes = Column(
        JSONB,
        nullable=True,
        comment="User attributes to exclude"
    )
    
    # A/B testing configuration
    is_ab_test = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this is an A/B test"
    )
    
    ab_test_variants = Column(
        JSONB,
        nullable=True,
        comment="A/B test variant configurations"
    )
    
    control_percentage = Column(
        Float,
        default=50.0,
        nullable=False,
        comment="Percentage for control group in A/B test"
    )
    
    # Scheduling
    start_date = Column(
        "start_date",
        nullable=True,
        index=True,
        comment="When feature should become available"
    )
    
    end_date = Column(
        "end_date",
        nullable=True,
        index=True,
        comment="When feature should be disabled"
    )
    
    # Metrics and monitoring
    usage_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times feature has been evaluated"
    )
    
    enabled_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times feature was enabled"
    )
    
    conversion_metrics = Column(
        JSONB,
        nullable=True,
        comment="Conversion and success metrics"
    )
    
    # Performance impact
    performance_impact = Column(
        JSONB,
        nullable=True,
        comment="Performance impact measurements"
    )
    
    error_rate = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Error rate when feature is enabled"
    )
    
    # Dependency management
    depends_on = Column(
        ARRAY(String),
        nullable=True,
        comment="Other feature flags this depends on"
    )
    
    conflicts_with = Column(
        ARRAY(String),
        nullable=True,
        comment="Feature flags that conflict with this one"
    )
    
    # Emergency controls
    kill_switch_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Emergency kill switch status"
    )
    
    kill_switch_reason = Column(
        Text,
        nullable=True,
        comment="Reason for activating kill switch"
    )
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_flag_enabled_env', 'is_enabled', 'environment'),
        Index('idx_flag_category', 'category'),
        Index('idx_flag_schedule', 'start_date', 'end_date'),
        Index('idx_flag_ab_test', 'is_ab_test'),
        CheckConstraint('rollout_percentage >= 0 AND rollout_percentage <= 100', name='ck_rollout_percentage_range'),
        CheckConstraint('control_percentage >= 0 AND control_percentage <= 100', name='ck_control_percentage_range'),
        CheckConstraint('usage_count >= 0', name='ck_usage_count_positive'),
        CheckConstraint('enabled_count >= 0', name='ck_enabled_count_positive'),
        CheckConstraint('error_rate >= 0 AND error_rate <= 1', name='ck_error_rate_range'),
    )
    
    def is_enabled_for_user(self, user_id: str, user_attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if feature is enabled for a specific user.
        
        Args:
            user_id: User identifier
            user_attributes: User attributes for targeting
            
        Returns:
            Whether feature is enabled for this user
        """
        # Record usage
        self.usage_count += 1
        
        # Check if feature is globally disabled
        if not self.is_enabled or self.kill_switch_enabled:
            return False
        
        # Check scheduling
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        
        # Check dependencies
        if self.depends_on:
            # In a real implementation, you'd check other feature flags
            # For now, assume dependencies are met
            pass
        
        # Apply strategy
        enabled = False
        
        if self.strategy == FeatureFlagStrategy.ALL_USERS:
            enabled = True
        
        elif self.strategy == FeatureFlagStrategy.USER_LIST:
            enabled = user_id in (self.target_user_ids or [])
        
        elif self.strategy == FeatureFlagStrategy.PERCENTAGE:
            # Consistent hash-based percentage rollout
            hash_input = f"{self.flag_key}:{user_id}"
            hash_value = hash(hash_input) % 100
            enabled = hash_value < self.rollout_percentage
        
        elif self.strategy == FeatureFlagStrategy.USER_ATTRIBUTES:
            if user_attributes and self.target_attributes:
                enabled = self._match_attributes(user_attributes, self.target_attributes)
            
            # Apply exclusions
            if enabled and user_attributes and self.exclusion_attributes:
                if self._match_attributes(user_attributes, self.exclusion_attributes):
                    enabled = False
        
        elif self.strategy == FeatureFlagStrategy.A_B_TEST:
            # Hash-based A/B assignment
            hash_input = f"{self.flag_key}:ab:{user_id}"
            hash_value = hash(hash_input) % 100
            enabled = hash_value >= self.control_percentage
        
        # Record enablement
        if enabled:
            self.enabled_count += 1
        
        return enabled
    
    def _match_attributes(self, user_attrs: Dict[str, Any], target_attrs: Dict[str, Any]) -> bool:
        """Check if user attributes match targeting criteria."""
        for key, criteria in target_attrs.items():
            if key not in user_attrs:
                continue
            
            user_value = user_attrs[key]
            
            if isinstance(criteria, dict):
                # Complex criteria
                if 'equals' in criteria and user_value != criteria['equals']:
                    return False
                if 'in' in criteria and user_value not in criteria['in']:
                    return False
                if 'greater_than' in criteria and user_value <= criteria['greater_than']:
                    return False
                if 'less_than' in criteria and user_value >= criteria['less_than']:
                    return False
            else:
                # Simple equality check
                if user_value != criteria:
                    return False
        
        return True
    
    def get_variant_for_user(self, user_id: str) -> Optional[str]:
        """Get A/B test variant for user."""
        if not self.is_ab_test or not self.ab_test_variants:
            return None
        
        if not self.is_enabled_for_user(user_id):
            return "control"
        
        # Hash-based variant assignment
        variants = list(self.ab_test_variants.keys())
        hash_input = f"{self.flag_key}:variant:{user_id}"
        hash_value = hash(hash_input) % len(variants)
        
        return variants[hash_value]
    
    def record_conversion(self, user_id: str, conversion_type: str, value: float = 1.0) -> None:
        """Record a conversion event for this feature."""
        if not self.conversion_metrics:
            self.conversion_metrics = {}
        
        if conversion_type not in self.conversion_metrics:
            self.conversion_metrics[conversion_type] = {
                "count": 0,
                "total_value": 0.0,
                "users": set()
            }
        
        metrics = self.conversion_metrics[conversion_type]
        metrics["count"] += 1
        metrics["total_value"] += value
        metrics["users"] = list(set(metrics["users"]) | {user_id})  # Convert to list for JSON serialization
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'conversion_metrics')
    
    def get_flag_summary(self) -> Dict[str, Any]:
        """Get feature flag summary for management interfaces."""
        return {
            "flag_key": self.flag_key,
            "name": self.name,
            "category": self.category,
            "is_enabled": self.is_enabled,
            "environment": self.environment,
            "strategy": self.strategy,
            "rollout_percentage": self.rollout_percentage,
            "is_ab_test": self.is_ab_test,
            "usage_count": self.usage_count,
            "enabled_count": self.enabled_count,
            "conversion_rate": self.enabled_count / self.usage_count if self.usage_count > 0 else 0,
            "kill_switch_enabled": self.kill_switch_enabled,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


class RateLimitConfig(FullAuditModel):
    """
    Rate limiting configuration for different endpoints and users.
    
    Provides flexible rate limiting with multiple strategies and
    user-specific overrides.
    """
    
    __tablename__ = "rate_limit_configs"
    
    # Identifier and scope
    identifier = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique identifier for this rate limit"
    )
    
    scope = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Scope (global, user, ip, endpoint, etc.)"
    )
    
    resource = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Resource being rate limited (endpoint, feature, etc.)"
    )
    
    # Rate limit parameters
    requests_per_second = Column(
        Float,
        nullable=True,
        comment="Requests per second limit"
    )
    
    requests_per_minute = Column(
        Integer,
        nullable=True,
        comment="Requests per minute limit"
    )
    
    requests_per_hour = Column(
        Integer,
        nullable=True,
        comment="Requests per hour limit"
    )
    
    requests_per_day = Column(
        Integer,
        nullable=True,
        comment="Requests per day limit"
    )
    
    # Burst handling
    burst_capacity = Column(
        Integer,
        nullable=True,
        comment="Burst capacity (token bucket)"
    )
    
    burst_refill_rate = Column(
        Float,
        nullable=True,
        comment="Burst refill rate per second"
    )
    
    # Advanced configuration
    strategy = Column(
        String(20),
        default="fixed_window",
        nullable=False,
        comment="Rate limiting strategy (fixed_window, sliding_window, token_bucket)"
    )
    
    window_size_seconds = Column(
        Integer,
        default=60,
        nullable=False,
        comment="Window size for windowed strategies"
    )
    
    # Override and exemption settings
    user_overrides = Column(
        JSONB,
        nullable=True,
        comment="Per-user rate limit overrides"
    )
    
    ip_exemptions = Column(
        ARRAY(String),
        nullable=True,
        comment="IP addresses exempt from rate limiting"
    )
    
    user_exemptions = Column(
        ARRAY(String),
        nullable=True,
        comment="User IDs exempt from rate limiting"
    )
    
    # Enforcement settings
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this rate limit is active"
    )
    
    block_on_exceed = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether to block requests that exceed limits"
    )
    
    delay_on_exceed = Column(
        Float,
        nullable=True,
        comment="Delay in seconds for requests that exceed limits"
    )
    
    # Response configuration
    error_message = Column(
        Text,
        nullable=True,
        comment="Custom error message when rate limit exceeded"
    )
    
    retry_after_seconds = Column(
        Integer,
        nullable=True,
        comment="Retry-After header value in seconds"
    )
    
    # Monitoring and statistics
    violations_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of rate limit violations"
    )
    
    last_violation_at = Column(
        "last_violation_at",
        nullable=True,
        comment="When last violation occurred"
    )
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_rate_limit_scope_resource', 'scope', 'resource'),
        Index('idx_rate_limit_active', 'is_active'),
        CheckConstraint('requests_per_second > 0', name='ck_rps_positive'),
        CheckConstraint('requests_per_minute > 0', name='ck_rpm_positive'),
        CheckConstraint('requests_per_hour > 0', name='ck_rph_positive'),
        CheckConstraint('requests_per_day > 0', name='ck_rpd_positive'),
        CheckConstraint('burst_capacity > 0', name='ck_burst_capacity_positive'),
        CheckConstraint('burst_refill_rate > 0', name='ck_burst_refill_positive'),
        CheckConstraint('window_size_seconds > 0', name='ck_window_size_positive'),
        CheckConstraint('violations_count >= 0', name='ck_violations_positive'),
    )
    
    def get_limit_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get effective rate limit for a specific user."""
        # Check exemptions
        if self.user_exemptions and user_id in self.user_exemptions:
            return None  # No limit
        
        # Check user-specific overrides
        if self.user_overrides and user_id in self.user_overrides:
            override = self.user_overrides[user_id]
            return {
                "requests_per_second": override.get("requests_per_second", self.requests_per_second),
                "requests_per_minute": override.get("requests_per_minute", self.requests_per_minute),
                "requests_per_hour": override.get("requests_per_hour", self.requests_per_hour),
                "requests_per_day": override.get("requests_per_day", self.requests_per_day),
                "burst_capacity": override.get("burst_capacity", self.burst_capacity),
            }
        
        # Return default limits
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "burst_capacity": self.burst_capacity,
        }
    
    def is_ip_exempt(self, ip_address: str) -> bool:
        """Check if IP address is exempt from rate limiting."""
        return self.ip_exemptions and ip_address in self.ip_exemptions
    
    def record_violation(self) -> None:
        """Record a rate limit violation."""
        self.violations_count += 1
        self.last_violation_at = datetime.utcnow()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get rate limit configuration summary."""
        return {
            "identifier": self.identifier,
            "scope": self.scope,
            "resource": self.resource,
            "strategy": self.strategy,
            "limits": {
                "per_second": self.requests_per_second,
                "per_minute": self.requests_per_minute,
                "per_hour": self.requests_per_hour,
                "per_day": self.requests_per_day,
            },
            "burst_capacity": self.burst_capacity,
            "is_active": self.is_active,
            "violations_count": self.violations_count,
            "last_violation": self.last_violation_at.isoformat() if self.last_violation_at else None,
        }


# No need to update User model as system configuration models are standalone