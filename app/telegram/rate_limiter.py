"""
Advanced Rate Limiter for Telegram Bot

Provides sophisticated rate limiting capabilities using Redis with:
- Multiple rate limiting algorithms (sliding window, token bucket, fixed window)
- User-specific and global rate limits
- Priority-based rate limiting
- Anti-spam and abuse detection
- Dynamic rate limit adjustments
- Graceful degradation strategies
"""

import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

import structlog

from app.core.redis import (
    redis_manager, sliding_window_rate_limit, token_bucket_rate_limit,
    RateLimitInfo, CompressionType
)
from app.config import settings

logger = structlog.get_logger(__name__)


class RateLimitType(Enum):
    """Rate limit types."""
    GLOBAL = "global"
    USER = "user"
    CHAT = "chat"
    COMMAND = "command"
    MESSAGE = "message"
    CALLBACK = "callback"
    INLINE = "inline"


class Priority(Enum):
    """User priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VIP = "vip"
    ADMIN = "admin"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    limit: int
    window: int
    burst_limit: Optional[int] = None
    algorithm: str = "sliding_window"  # sliding_window, token_bucket, fixed_window
    priority_multipliers: Dict[str, float] = field(default_factory=lambda: {
        Priority.LOW.value: 0.5,
        Priority.NORMAL.value: 1.0,
        Priority.HIGH.value: 2.0,
        Priority.VIP.value: 5.0,
        Priority.ADMIN.value: 10.0
    })


@dataclass
class UserStats:
    """User interaction statistics."""
    total_requests: int = 0
    blocked_requests: int = 0
    last_request_time: float = 0
    average_interval: float = 0
    burst_count: int = 0
    priority: Priority = Priority.NORMAL
    trust_score: float = 1.0


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple algorithms and anti-abuse features.
    
    Features:
    - Multiple rate limiting algorithms
    - User-specific rate limits based on priority
    - Anti-spam detection and mitigation
    - Dynamic rate limit adjustments
    - Burst allowance for legitimate users
    - Comprehensive usage statistics
    - Graceful degradation on Redis failure
    """
    
    def __init__(self):
        # Rate limit configurations
        self.configs: Dict[RateLimitType, RateLimitConfig] = {
            RateLimitType.GLOBAL: RateLimitConfig(
                limit=1000, window=60, burst_limit=50
            ),
            RateLimitType.USER: RateLimitConfig(
                limit=30, window=60, burst_limit=10
            ),
            RateLimitType.CHAT: RateLimitConfig(
                limit=100, window=60, burst_limit=20
            ),
            RateLimitType.COMMAND: RateLimitConfig(
                limit=10, window=60, burst_limit=5
            ),
            RateLimitType.MESSAGE: RateLimitConfig(
                limit=20, window=60, burst_limit=5
            ),
            RateLimitType.CALLBACK: RateLimitConfig(
                limit=50, window=60, burst_limit=10
            ),
            RateLimitType.INLINE: RateLimitConfig(
                limit=15, window=60, burst_limit=5
            ),
        }
        
        # User statistics cache
        self.user_stats: Dict[int, UserStats] = {}
        
        # Anti-spam detection
        self.spam_patterns: List[Dict[str, Any]] = [
            {"name": "rapid_fire", "threshold": 10, "window": 5, "penalty": 0.5},
            {"name": "repeated_content", "threshold": 5, "window": 60, "penalty": 0.3},
            {"name": "command_spam", "threshold": 8, "window": 30, "penalty": 0.4},
        ]
        
        # Rate limit bypass conditions
        self.bypass_users: set = set()
        self.admin_users: set = set(settings.security.admin_users)
        
        # Performance monitoring
        self.total_checks: int = 0
        self.total_blocks: int = 0
        self.fallback_mode: bool = False
    
    async def check_rate_limit(
        self,
        user_id: int,
        rate_type: RateLimitType,
        identifier: Optional[str] = None,
        content: Optional[str] = None,
        priority: Optional[Priority] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: User identifier
            rate_type: Type of rate limit to check
            identifier: Additional identifier (chat_id, command_name, etc.)
            content: Message content for spam detection
            priority: User priority level override
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        start_time = time.time()
        self.total_checks += 1
        
        try:
            # Check for bypass conditions
            if await self._should_bypass(user_id, rate_type):
                return True, {
                    "bypassed": True,
                    "reason": "admin_or_whitelisted",
                    "remaining": float('inf'),
                    "reset_time": 0
                }
            
            # Get user priority and stats
            user_priority = priority or await self._get_user_priority(user_id)
            user_stats = await self._get_user_stats(user_id)
            
            # Update user statistics
            await self._update_user_stats(user_id, user_stats)
            
            # Perform anti-spam checks
            spam_penalty = await self._check_spam_patterns(user_id, rate_type, content)
            
            # Get rate limit configuration
            config = self.configs.get(rate_type, self.configs[RateLimitType.USER])
            
            # Apply priority and spam adjustments
            adjusted_limit = int(
                config.limit * 
                config.priority_multipliers.get(user_priority.value, 1.0) * 
                user_stats.trust_score * 
                (1.0 - spam_penalty)
            )
            
            # Create rate limit key
            rate_key = self._create_rate_key(user_id, rate_type, identifier)
            
            # Perform rate limit check based on algorithm
            if config.algorithm == "token_bucket":
                result = await self._check_token_bucket(
                    rate_key, adjusted_limit, config.window
                )
            elif config.algorithm == "fixed_window":
                result = await self._check_fixed_window(
                    rate_key, adjusted_limit, config.window
                )
            else:  # sliding_window (default)
                result = await self._check_sliding_window(
                    rate_key, adjusted_limit, config.window
                )
            
            # Handle burst allowance
            if not result.allowed and config.burst_limit:
                burst_result = await self._check_burst_allowance(
                    user_id, rate_type, config.burst_limit
                )
                if burst_result:
                    result.allowed = True
                    result.remaining = 0  # Indicate burst used
            
            # Update statistics
            if not result.allowed:
                self.total_blocks += 1
                user_stats.blocked_requests += 1
                user_stats.trust_score = max(0.1, user_stats.trust_score * 0.95)
            else:
                user_stats.trust_score = min(1.0, user_stats.trust_score * 1.001)
            
            # Store updated stats
            await self._store_user_stats(user_id, user_stats)
            
            # Create response info
            response_info = {
                "allowed": result.allowed,
                "remaining": result.remaining,
                "reset_time": result.reset_time,
                "current_count": result.current_count,
                "limit": adjusted_limit,
                "original_limit": config.limit,
                "priority": user_priority.value,
                "trust_score": user_stats.trust_score,
                "spam_penalty": spam_penalty,
                "algorithm": config.algorithm,
                "processing_time": time.time() - start_time,
                "burst_used": result.remaining == 0 and result.allowed
            }
            
            # Log rate limit events
            if not result.allowed:
                logger.warning(
                    "Rate limit exceeded",
                    user_id=user_id,
                    rate_type=rate_type.value,
                    limit=adjusted_limit,
                    current_count=result.current_count
                )
            
            return result.allowed, response_info
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e), user_id=user_id)
            
            # Fallback to permissive mode on Redis failure
            self.fallback_mode = True
            return True, {
                "allowed": True,
                "fallback": True,
                "error": str(e),
                "remaining": float('inf'),
                "reset_time": 0
            }
    
    async def _should_bypass(self, user_id: int, rate_type: RateLimitType) -> bool:
        """Check if user should bypass rate limits."""
        # Admin users bypass most limits
        if user_id in self.admin_users:
            return True
        
        # Explicitly whitelisted users
        if user_id in self.bypass_users:
            return True
        
        # Global rate limits apply to everyone
        if rate_type == RateLimitType.GLOBAL:
            return False
        
        return False
    
    async def _get_user_priority(self, user_id: int) -> Priority:
        """Get user priority level."""
        try:
            # Check if admin
            if user_id in self.admin_users:
                return Priority.ADMIN
            
            # Load from cache or database
            priority_key = f"user_priority:{user_id}"
            cached_priority = await redis_manager.get_cache(
                priority_key, default=Priority.NORMAL.value
            )
            
            return Priority(cached_priority)
            
        except Exception as e:
            logger.error("Failed to get user priority", error=str(e), user_id=user_id)
            return Priority.NORMAL
    
    async def _get_user_stats(self, user_id: int) -> UserStats:
        """Get user statistics."""
        try:
            # Check memory cache first
            if user_id in self.user_stats:
                return self.user_stats[user_id]
            
            # Load from Redis
            stats_key = f"user_stats:{user_id}"
            cached_stats = await redis_manager.get_cache(stats_key, default=None)
            
            if cached_stats:
                stats = UserStats(**cached_stats)
            else:
                stats = UserStats()
            
            # Cache in memory
            self.user_stats[user_id] = stats
            return stats
            
        except Exception as e:
            logger.error("Failed to get user stats", error=str(e), user_id=user_id)
            return UserStats()
    
    async def _update_user_stats(self, user_id: int, stats: UserStats) -> None:
        """Update user statistics."""
        try:
            current_time = time.time()
            
            # Calculate average interval
            if stats.last_request_time > 0:
                interval = current_time - stats.last_request_time
                if stats.total_requests > 0:
                    stats.average_interval = (
                        (stats.average_interval * stats.total_requests + interval) /
                        (stats.total_requests + 1)
                    )
                else:
                    stats.average_interval = interval
            
            # Detect burst behavior
            if stats.last_request_time > 0 and (current_time - stats.last_request_time) < 1.0:
                stats.burst_count += 1
            else:
                stats.burst_count = max(0, stats.burst_count - 1)
            
            stats.total_requests += 1
            stats.last_request_time = current_time
            
        except Exception as e:
            logger.error("Failed to update user stats", error=str(e), user_id=user_id)
    
    async def _store_user_stats(self, user_id: int, stats: UserStats) -> None:
        """Store user statistics in Redis."""
        try:
            stats_key = f"user_stats:{user_id}"
            stats_dict = {
                "total_requests": stats.total_requests,
                "blocked_requests": stats.blocked_requests,
                "last_request_time": stats.last_request_time,
                "average_interval": stats.average_interval,
                "burst_count": stats.burst_count,
                "priority": stats.priority.value,
                "trust_score": stats.trust_score
            }
            
            await redis_manager.set_cache(
                stats_key, stats_dict, ttl=3600  # 1 hour
            )
            
        except Exception as e:
            logger.error("Failed to store user stats", error=str(e), user_id=user_id)
    
    async def _check_spam_patterns(
        self,
        user_id: int,
        rate_type: RateLimitType,
        content: Optional[str]
    ) -> float:
        """Check for spam patterns and return penalty factor."""
        total_penalty = 0.0
        
        try:
            # Check rapid fire pattern
            rapid_fire_penalty = await self._check_rapid_fire(user_id)
            total_penalty += rapid_fire_penalty
            
            # Check repeated content if available
            if content:
                content_penalty = await self._check_repeated_content(user_id, content)
                total_penalty += content_penalty
            
            # Check command spam for command rate types
            if rate_type == RateLimitType.COMMAND:
                command_penalty = await self._check_command_spam(user_id)
                total_penalty += command_penalty
            
            # Cap total penalty
            return min(0.8, total_penalty)  # Max 80% penalty
            
        except Exception as e:
            logger.error("Spam pattern check failed", error=str(e), user_id=user_id)
            return 0.0
    
    async def _check_rapid_fire(self, user_id: int) -> float:
        """Check for rapid fire requests."""
        try:
            rapid_key = f"rapid_fire:{user_id}"
            
            # Count requests in last 5 seconds
            result = await sliding_window_rate_limit(rapid_key, 10, 5, 1)
            
            if result.current_count > 8:  # More than 8 requests in 5 seconds
                return 0.5  # 50% penalty
            elif result.current_count > 5:
                return 0.2  # 20% penalty
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _check_repeated_content(self, user_id: int, content: str) -> float:
        """Check for repeated message content."""
        try:
            # Create content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            content_key = f"content_repeat:{user_id}:{content_hash}"
            
            # Check how many times this content was sent recently
            result = await sliding_window_rate_limit(content_key, 3, 60, 1)
            
            if result.current_count > 2:  # Same content more than 2 times in 1 minute
                return 0.3  # 30% penalty
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _check_command_spam(self, user_id: int) -> float:
        """Check for command spam patterns."""
        try:
            command_key = f"command_spam:{user_id}"
            
            # Count commands in last 30 seconds
            result = await sliding_window_rate_limit(command_key, 5, 30, 1)
            
            if result.current_count > 4:  # More than 4 commands in 30 seconds
                return 0.4  # 40% penalty
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _create_rate_key(
        self,
        user_id: int,
        rate_type: RateLimitType,
        identifier: Optional[str]
    ) -> str:
        """Create rate limit key."""
        base_key = f"rate_limit:{rate_type.value}:{user_id}"
        
        if identifier:
            base_key += f":{identifier}"
        
        return base_key
    
    async def _check_sliding_window(
        self,
        key: str,
        limit: int,
        window: int
    ) -> RateLimitInfo:
        """Check sliding window rate limit."""
        return await sliding_window_rate_limit(key, limit, window, 1)
    
    async def _check_token_bucket(
        self,
        key: str,
        capacity: int,
        refill_period: int
    ) -> RateLimitInfo:
        """Check token bucket rate limit."""
        refill_rate = capacity / refill_period  # tokens per second
        return await token_bucket_rate_limit(key, capacity, refill_rate, 1)
    
    async def _check_fixed_window(
        self,
        key: str,
        limit: int,
        window: int
    ) -> RateLimitInfo:
        """Check fixed window rate limit."""
        try:
            # Get current window start time
            current_time = int(time.time())
            window_start = current_time - (current_time % window)
            fixed_key = f"{key}:{window_start}"
            
            # Increment counter for this window
            current_count = await redis_manager.atomic_increment(fixed_key, 1, window)
            
            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = window - (current_time % window)
            
            return RateLimitInfo(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                current_count=current_count,
                limit=limit,
                window_size=window,
                identifier=key
            )
            
        except Exception as e:
            logger.error("Fixed window rate limit check failed", error=str(e))
            return RateLimitInfo(
                allowed=True,
                remaining=limit,
                reset_time=window,
                current_count=0,
                limit=limit,
                window_size=window,
                identifier=key
            )
    
    async def _check_burst_allowance(
        self,
        user_id: int,
        rate_type: RateLimitType,
        burst_limit: int
    ) -> bool:
        """Check if user has burst allowance available."""
        try:
            burst_key = f"burst:{rate_type.value}:{user_id}"
            
            # Check burst usage in last hour
            result = await sliding_window_rate_limit(burst_key, burst_limit, 3600, 1)
            
            return result.allowed
            
        except Exception as e:
            logger.error("Burst allowance check failed", error=str(e))
            return False
    
    # Configuration and Management Methods
    
    async def set_user_priority(self, user_id: int, priority: Priority) -> bool:
        """Set user priority level."""
        try:
            priority_key = f"user_priority:{user_id}"
            await redis_manager.set_cache(priority_key, priority.value, ttl=86400)
            
            # Update local cache
            if user_id in self.user_stats:
                self.user_stats[user_id].priority = priority
            
            logger.info(f"Set user {user_id} priority to {priority.value}")
            return True
            
        except Exception as e:
            logger.error("Failed to set user priority", error=str(e))
            return False
    
    async def add_bypass_user(self, user_id: int) -> bool:
        """Add user to rate limit bypass list."""
        try:
            self.bypass_users.add(user_id)
            
            # Store in Redis
            bypass_key = "rate_limit_bypass_users"
            bypass_list = list(self.bypass_users)
            await redis_manager.set_cache(bypass_key, bypass_list, ttl=86400)
            
            logger.info(f"Added user {user_id} to rate limit bypass")
            return True
            
        except Exception as e:
            logger.error("Failed to add bypass user", error=str(e))
            return False
    
    async def remove_bypass_user(self, user_id: int) -> bool:
        """Remove user from rate limit bypass list."""
        try:
            self.bypass_users.discard(user_id)
            
            # Store in Redis
            bypass_key = "rate_limit_bypass_users"
            bypass_list = list(self.bypass_users)
            await redis_manager.set_cache(bypass_key, bypass_list, ttl=86400)
            
            logger.info(f"Removed user {user_id} from rate limit bypass")
            return True
            
        except Exception as e:
            logger.error("Failed to remove bypass user", error=str(e))
            return False
    
    async def update_rate_config(
        self,
        rate_type: RateLimitType,
        config: RateLimitConfig
    ) -> bool:
        """Update rate limit configuration."""
        try:
            self.configs[rate_type] = config
            
            # Store in Redis
            config_key = f"rate_config:{rate_type.value}"
            config_dict = {
                "limit": config.limit,
                "window": config.window,
                "burst_limit": config.burst_limit,
                "algorithm": config.algorithm,
                "priority_multipliers": config.priority_multipliers
            }
            await redis_manager.set_cache(config_key, config_dict, ttl=86400)
            
            logger.info(f"Updated rate config for {rate_type.value}")
            return True
            
        except Exception as e:
            logger.error("Failed to update rate config", error=str(e))
            return False
    
    async def get_user_rate_info(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive rate limit info for user."""
        try:
            user_stats = await self._get_user_stats(user_id)
            user_priority = await self._get_user_priority(user_id)
            
            # Get current rate limit status for each type
            rate_status = {}
            for rate_type in RateLimitType:
                config = self.configs.get(rate_type)
                if config:
                    rate_key = self._create_rate_key(user_id, rate_type, None)
                    
                    # Check current status without incrementing
                    if config.algorithm == "sliding_window":
                        result = await sliding_window_rate_limit(rate_key, config.limit, config.window, 0)
                    else:
                        result = await token_bucket_rate_limit(rate_key, config.limit, config.window / config.limit, 0)
                    
                    rate_status[rate_type.value] = {
                        "limit": config.limit,
                        "remaining": result.remaining,
                        "reset_time": result.reset_time,
                        "current_count": result.current_count
                    }
            
            return {
                "user_id": user_id,
                "priority": user_priority.value,
                "trust_score": user_stats.trust_score,
                "total_requests": user_stats.total_requests,
                "blocked_requests": user_stats.blocked_requests,
                "average_interval": user_stats.average_interval,
                "burst_count": user_stats.burst_count,
                "is_bypassed": user_id in self.bypass_users,
                "rate_status": rate_status
            }
            
        except Exception as e:
            logger.error("Failed to get user rate info", error=str(e))
            return {}
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiter statistics."""
        try:
            # Calculate success rate
            success_rate = (
                (self.total_checks - self.total_blocks) / max(1, self.total_checks) * 100
            )
            
            # Get Redis health
            redis_healthy = await redis_manager.health_check()
            redis_metrics = redis_manager.metrics
            
            return {
                "total_checks": self.total_checks,
                "total_blocks": self.total_blocks,
                "success_rate": success_rate,
                "fallback_mode": self.fallback_mode,
                "active_configs": len(self.configs),
                "bypass_users_count": len(self.bypass_users),
                "cached_user_stats": len(self.user_stats),
                "redis_healthy": redis_healthy,
                "redis_success_rate": redis_metrics.success_rate,
                "redis_cache_hit_rate": redis_metrics.cache_hit_rate
            }
            
        except Exception as e:
            logger.error("Failed to get global stats", error=str(e))
            return {}
    
    async def cleanup_expired_data(self) -> int:
        """Clean up expired rate limit data."""
        try:
            cleanup_count = 0
            
            # Clean up old user stats from memory
            current_time = time.time()
            expired_users = []
            
            for user_id, stats in self.user_stats.items():
                if current_time - stats.last_request_time > 3600:  # 1 hour
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.user_stats[user_id]
                cleanup_count += 1
            
            # Clean up Redis patterns
            patterns_to_clean = [
                "rate_limit:*",
                "rapid_fire:*",
                "content_repeat:*",
                "command_spam:*",
                "burst:*"
            ]
            
            for pattern in patterns_to_clean:
                deleted = await redis_manager.flush_cache(pattern, batch_size=100)
                if deleted > 0:
                    cleanup_count += deleted
            
            logger.info(f"Cleaned up {cleanup_count} expired rate limit entries")
            return cleanup_count
            
        except Exception as e:
            logger.error("Failed to cleanup expired data", error=str(e))
            return 0


# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()


# Convenience functions for common rate limiting scenarios
async def check_user_rate_limit(
    user_id: int,
    action_type: str = "message",
    identifier: Optional[str] = None,
    content: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function for checking user rate limits.
    
    Args:
        user_id: User identifier
        action_type: Type of action (message, command, callback, inline)
        identifier: Additional identifier
        content: Message content for spam detection
        
    Returns:
        Tuple of (allowed, rate_limit_info)
    """
    rate_type_map = {
        "message": RateLimitType.MESSAGE,
        "command": RateLimitType.COMMAND,
        "callback": RateLimitType.CALLBACK,
        "inline": RateLimitType.INLINE
    }
    
    rate_type = rate_type_map.get(action_type, RateLimitType.MESSAGE)
    
    return await rate_limiter.check_rate_limit(
        user_id=user_id,
        rate_type=rate_type,
        identifier=identifier,
        content=content
    )


async def check_global_rate_limit() -> Tuple[bool, Dict[str, Any]]:
    """Check global rate limits."""
    return await rate_limiter.check_rate_limit(
        user_id=0,  # Special user ID for global limits
        rate_type=RateLimitType.GLOBAL
    )


async def check_chat_rate_limit(
    chat_id: int,
    user_id: int
) -> Tuple[bool, Dict[str, Any]]:
    """Check chat-specific rate limits."""
    return await rate_limiter.check_rate_limit(
        user_id=user_id,
        rate_type=RateLimitType.CHAT,
        identifier=str(chat_id)
    )