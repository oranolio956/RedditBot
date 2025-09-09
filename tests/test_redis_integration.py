"""
Comprehensive Test Suite for Redis Integration

Tests all Redis functionality including:
- Advanced Redis manager with clustering and sentinel support
- Rate limiting with multiple algorithms
- Session management with distributed coordination
- Pub/sub for multi-instance communication
- Performance monitoring and health checks
- Failover and recovery mechanisms
"""

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from app.core.redis import (
    redis_manager, AdvancedRedisManager, RedisMetrics, RateLimitInfo,
    CompressionType, ConnectionType,
    cache_set, cache_get, cache_delete, cache_get_multiple, cache_set_multiple,
    sliding_window_rate_limit, token_bucket_rate_limit,
    set_session, get_session, update_session, delete_session,
    publish_message, subscribe_channel,
    get_redis_metrics, get_redis_info, is_redis_healthy
)
from app.telegram.rate_limiter import (
    rate_limiter, AdvancedRateLimiter, RateLimitType, Priority,
    RateLimitConfig, UserStats,
    check_user_rate_limit, check_global_rate_limit, check_chat_rate_limit
)
from app.telegram.session import session_manager, SessionManager
from app.telegram.middleware import (
    telegram_rate_limit_middleware, telegram_session_middleware,
    TelegramRateLimitMiddleware, TelegramSessionMiddleware,
    get_middleware_metrics, get_middleware_health
)


class TestAdvancedRedisManager:
    """Test advanced Redis manager functionality."""
    
    @pytest.fixture
    async def redis_mgr(self):
        """Create test Redis manager."""
        mgr = AdvancedRedisManager()
        # Mock Redis client for testing
        mgr._redis_client = AsyncMock()
        mgr._is_healthy = True
        return mgr
    
    @pytest.mark.asyncio
    async def test_redis_initialization(self, redis_mgr):
        """Test Redis manager initialization."""
        # Test standalone connection
        with patch.object(redis_mgr, '_connect_standalone') as mock_standalone:
            await redis_mgr._establish_connection()
            mock_standalone.assert_called_once()
        
        # Test cluster connection
        redis_mgr._redis_client = None
        with patch('app.config.settings.redis.cluster_enabled', True):
            with patch.object(redis_mgr, '_connect_cluster') as mock_cluster:
                await redis_mgr._establish_connection()
                mock_cluster.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_lua_script_loading(self, redis_mgr):
        """Test Lua script loading and caching."""
        redis_mgr._redis_client.script_load = AsyncMock(return_value="sha123")
        
        await redis_mgr._load_lua_scripts()
        
        assert "sliding_window_rate_limit" in redis_mgr._script_shas
        assert redis_mgr._script_shas["sliding_window_rate_limit"] == "sha123"
    
    @pytest.mark.asyncio
    async def test_health_check(self, redis_mgr):
        """Test comprehensive health check."""
        # Mock successful health check
        redis_mgr._redis_client.ping = AsyncMock()
        redis_mgr._redis_client.set = AsyncMock()
        redis_mgr._redis_client.get = AsyncMock(return_value="test")
        redis_mgr._redis_client.delete = AsyncMock()
        
        result = await redis_mgr.health_check()
        assert result is True
        assert redis_mgr._metrics.successful_commands > 0
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, redis_mgr):
        """Test cache operations with compression."""
        # Test set_cache
        redis_mgr._redis_client.setex = AsyncMock()
        
        success = await redis_mgr.set_cache(
            "test_key", 
            {"data": "test"}, 
            ttl=60, 
            compression=CompressionType.JSON
        )
        assert success is True
        
        # Test get_cache
        redis_mgr._redis_client.get = AsyncMock(return_value='{"data": "test"}')
        
        result = await redis_mgr.get_cache(
            "test_key", 
            compression=CompressionType.JSON
        )
        assert result == {"data": "test"}
        assert redis_mgr._metrics.cache_hits > 0
    
    @pytest.mark.asyncio
    async def test_sliding_window_rate_limit(self, redis_mgr):
        """Test sliding window rate limiting."""
        # Mock Lua script execution
        redis_mgr._script_shas["sliding_window_rate_limit"] = "sha123"
        redis_mgr._redis_client.evalsha = AsyncMock(return_value=[1, 5, 1])
        redis_mgr._redis_client.ttl = AsyncMock(return_value=60)
        
        result = await redis_mgr.sliding_window_rate_limit(
            "user123", 10, 60, 1
        )
        
        assert isinstance(result, RateLimitInfo)
        assert result.allowed is True
        assert result.remaining == 5
        assert result.current_count == 1
    
    @pytest.mark.asyncio
    async def test_token_bucket_rate_limit(self, redis_mgr):
        """Test token bucket rate limiting."""
        redis_mgr._script_shas["token_bucket_rate_limit"] = "sha123"
        redis_mgr._redis_client.evalsha = AsyncMock(return_value=[1, 9, 1])
        
        result = await redis_mgr.token_bucket_rate_limit(
            "user123", 10, 1.0, 1
        )
        
        assert isinstance(result, RateLimitInfo)
        assert result.allowed is True
        assert result.remaining == 9
    
    @pytest.mark.asyncio
    async def test_distributed_locking(self, redis_mgr):
        """Test distributed locking with auto-renewal."""
        redis_mgr._script_shas["acquire_lock"] = "sha123"
        redis_mgr._script_shas["release_lock"] = "sha456"
        redis_mgr._redis_client.evalsha = AsyncMock(side_effect=[1, 1])  # acquire, release
        
        # Test lock acquisition
        lock_value = await redis_mgr.acquire_lock("test_lock", timeout=5, expire=30)
        assert lock_value is not None
        
        # Test lock release
        success = await redis_mgr.release_lock("test_lock", lock_value)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_pubsub_operations(self, redis_mgr):
        """Test pub/sub operations."""
        # Mock pubsub
        mock_pubsub = AsyncMock()
        redis_mgr._redis_client.pubsub = MagicMock(return_value=mock_pubsub)
        redis_mgr._redis_client.publish = AsyncMock(return_value=3)
        
        # Test publish
        subscribers = await redis_mgr.publish("test_channel", {"msg": "test"})
        assert subscribers == 3
        
        # Test subscribe
        handler = AsyncMock()
        pubsub = await redis_mgr.subscribe("test_channel", handler=handler)
        assert pubsub is not None
    
    @pytest.mark.asyncio
    async def test_session_management(self, redis_mgr):
        """Test session management operations."""
        session_data = {
            "user_id": 123,
            "chat_id": 456,
            "data": {"key": "value"}
        }
        
        # Mock Redis operations
        redis_mgr.set_cache = AsyncMock(return_value=True)
        redis_mgr.get_cache = AsyncMock(return_value=session_data)
        redis_mgr.delete_cache = AsyncMock(return_value=1)
        
        # Test set_session
        success = await redis_mgr.set_session("session123", session_data)
        assert success is True
        
        # Test get_session
        result = await redis_mgr.get_session("session123")
        assert result == session_data
        
        # Test delete_session
        deleted = await redis_mgr.delete_session("session123")
        assert deleted is True
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, redis_mgr):
        """Test performance monitoring and metrics."""
        # Simulate some operations
        redis_mgr._metrics.total_commands = 100
        redis_mgr._metrics.successful_commands = 95
        redis_mgr._metrics.failed_commands = 5
        redis_mgr._metrics.total_latency = 0.5
        redis_mgr._metrics.cache_hits = 80
        redis_mgr._metrics.cache_misses = 20
        
        # Test metrics properties
        assert redis_mgr._metrics.success_rate == 95.0
        assert redis_mgr._metrics.average_latency == 0.5 / 95
        assert redis_mgr._metrics.cache_hit_rate == 80.0
        
        # Test info gathering
        redis_mgr._redis_client.info = AsyncMock(return_value={
            "redis_version": "7.0.0",
            "used_memory": 1024,
            "connected_clients": 5
        })
        
        info = await redis_mgr.get_info()
        assert info["server"]["version"] == "7.0.0"
        assert "client_metrics" in info


class TestAdvancedRateLimiter:
    """Test advanced rate limiter functionality."""
    
    @pytest.fixture
    def rate_limiter_instance(self):
        """Create test rate limiter."""
        limiter = AdvancedRateLimiter()
        return limiter
    
    @pytest.mark.asyncio
    async def test_rate_limit_check(self, rate_limiter_instance):
        """Test comprehensive rate limit check."""
        # Mock Redis operations
        with patch('app.telegram.rate_limiter.sliding_window_rate_limit') as mock_rate_limit:
            mock_rate_limit.return_value = RateLimitInfo(
                allowed=True,
                remaining=5,
                reset_time=60,
                current_count=1,
                limit=10,
                window_size=60,
                identifier="test"
            )
            
            allowed, info = await rate_limiter_instance.check_rate_limit(
                user_id=123,
                rate_type=RateLimitType.MESSAGE,
                content="test message"
            )
            
            assert allowed is True
            assert info["remaining"] == 5
            assert info["algorithm"] == "sliding_window"
    
    @pytest.mark.asyncio
    async def test_spam_detection(self, rate_limiter_instance):
        """Test anti-spam detection patterns."""
        # Test rapid fire detection
        with patch('app.telegram.rate_limiter.sliding_window_rate_limit') as mock_rate_limit:
            # Simulate rapid fire (high current count)
            mock_rate_limit.return_value = RateLimitInfo(
                allowed=True, remaining=0, reset_time=5,
                current_count=9, limit=10, window_size=5, identifier="test"
            )
            
            penalty = await rate_limiter_instance._check_rapid_fire(123)
            assert penalty == 0.5  # 50% penalty for rapid fire
    
    @pytest.mark.asyncio
    async def test_user_priority_system(self, rate_limiter_instance):
        """Test user priority and trust score system."""
        # Mock Redis operations
        with patch.object(rate_limiter_instance, '_get_user_priority') as mock_priority:
            with patch.object(rate_limiter_instance, '_get_user_stats') as mock_stats:
                mock_priority.return_value = Priority.VIP
                mock_stats.return_value = UserStats(trust_score=1.0)
                
                # Test priority multiplier application
                config = rate_limiter_instance.configs[RateLimitType.MESSAGE]
                expected_limit = int(config.limit * 5.0 * 1.0)  # VIP multiplier * trust score
                
                with patch('app.telegram.rate_limiter.sliding_window_rate_limit') as mock_rate_limit:
                    mock_rate_limit.return_value = RateLimitInfo(
                        allowed=True, remaining=expected_limit-1, reset_time=60,
                        current_count=1, limit=expected_limit, window_size=60, identifier="test"
                    )
                    
                    allowed, info = await rate_limiter_instance.check_rate_limit(
                        user_id=123,
                        rate_type=RateLimitType.MESSAGE,
                        priority=Priority.VIP
                    )
                    
                    assert allowed is True
                    assert info["limit"] == expected_limit
                    assert info["priority"] == "vip"
    
    @pytest.mark.asyncio
    async def test_burst_allowance(self, rate_limiter_instance):
        """Test burst allowance for legitimate users."""
        config = rate_limiter_instance.configs[RateLimitType.MESSAGE]
        
        with patch('app.telegram.rate_limiter.sliding_window_rate_limit') as mock_rate_limit:
            # First call: normal rate limit exceeded
            # Second call: burst allowance check (allowed)
            mock_rate_limit.side_effect = [
                RateLimitInfo(allowed=False, remaining=0, reset_time=60,
                             current_count=21, limit=20, window_size=60, identifier="test"),
                RateLimitInfo(allowed=True, remaining=4, reset_time=3600,
                             current_count=1, limit=5, window_size=3600, identifier="burst")
            ]
            
            allowed, info = await rate_limiter_instance.check_rate_limit(
                user_id=123,
                rate_type=RateLimitType.MESSAGE
            )
            
            # Should be allowed due to burst allowance
            assert allowed is True
            assert info["burst_used"] is True
    
    @pytest.mark.asyncio
    async def test_configuration_management(self, rate_limiter_instance):
        """Test rate limit configuration management."""
        # Test updating rate configuration
        new_config = RateLimitConfig(
            limit=50,
            window=120,
            burst_limit=15,
            algorithm="token_bucket"
        )
        
        with patch('app.core.redis.redis_manager.set_cache') as mock_set_cache:
            mock_set_cache.return_value = True
            
            success = await rate_limiter_instance.update_rate_config(
                RateLimitType.MESSAGE, new_config
            )
            
            assert success is True
            assert rate_limiter_instance.configs[RateLimitType.MESSAGE].limit == 50
            assert rate_limiter_instance.configs[RateLimitType.MESSAGE].algorithm == "token_bucket"
    
    @pytest.mark.asyncio
    async def test_user_management(self, rate_limiter_instance):
        """Test user priority and bypass management."""
        # Test setting user priority
        with patch('app.core.redis.redis_manager.set_cache') as mock_set_cache:
            mock_set_cache.return_value = True
            
            success = await rate_limiter_instance.set_user_priority(123, Priority.HIGH)
            assert success is True
        
        # Test adding bypass user
        with patch('app.core.redis.redis_manager.set_cache') as mock_set_cache:
            mock_set_cache.return_value = True
            
            success = await rate_limiter_instance.add_bypass_user(456)
            assert success is True
            assert 456 in rate_limiter_instance.bypass_users


class TestTelegramMiddleware:
    """Test Telegram bot middleware functionality."""
    
    @pytest.fixture
    def rate_limit_middleware(self):
        """Create test rate limit middleware."""
        return TelegramRateLimitMiddleware()
    
    @pytest.fixture
    def session_middleware(self):
        """Create test session middleware."""
        return TelegramSessionMiddleware()
    
    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = MagicMock()
        update.message = MagicMock()
        update.message.from_user = MagicMock()
        update.message.from_user.id = 123
        update.message.chat = MagicMock()
        update.message.chat.id = 456
        update.message.text = "Hello, bot!"
        return update
    
    @pytest.mark.asyncio
    async def test_rate_limit_middleware(self, rate_limit_middleware, mock_update):
        """Test rate limiting middleware."""
        handler = AsyncMock(return_value="success")
        
        with patch('app.telegram.middleware.check_global_rate_limit') as mock_global:
            with patch('app.telegram.middleware.check_user_rate_limit') as mock_user:
                mock_global.return_value = (True, {"remaining": 100})
                mock_user.return_value = (True, {"remaining": 10})
                
                result = await rate_limit_middleware(handler, mock_update, {})
                
                assert result == "success"
                assert rate_limit_middleware.metrics.successful_updates == 1
                handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocking(self, rate_limit_middleware, mock_update):
        """Test rate limit blocking."""
        handler = AsyncMock(return_value="success")
        
        with patch('app.telegram.middleware.check_global_rate_limit') as mock_global:
            with patch('app.telegram.middleware.check_user_rate_limit') as mock_user:
                mock_global.return_value = (True, {"remaining": 100})
                mock_user.return_value = (False, {"remaining": 0, "reset_time": 60})
                
                result = await rate_limit_middleware(handler, mock_update, {})
                
                assert result is None  # Blocked
                assert rate_limit_middleware.metrics.blocked_updates == 1
                assert rate_limit_middleware.metrics.rate_limit_blocks == 1
                handler.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_emergency_mode(self, rate_limit_middleware, mock_update):
        """Test emergency mode activation."""
        handler = AsyncMock(return_value="success")
        
        # Simulate high failure rate
        rate_limit_middleware.metrics.total_updates = 200
        rate_limit_middleware.metrics.successful_updates = 10
        rate_limit_middleware.metrics.blocked_updates = 190
        
        await rate_limit_middleware._check_emergency_mode()
        assert rate_limit_middleware.emergency_mode is True
        
        # In emergency mode, should bypass rate limits
        result = await rate_limit_middleware(handler, mock_update, {})
        assert result == "success"
        handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_middleware(self, session_middleware, mock_update):
        """Test session management middleware."""
        handler = AsyncMock(return_value="success")
        
        with patch.object(session_middleware, '_get_or_create_session') as mock_session:
            mock_session_obj = MagicMock()
            mock_session.return_value = mock_session_obj
            
            data = {}
            result = await session_middleware(handler, mock_update, data)
            
            assert result == "success"
            assert "session" in data
            assert data["session"] == mock_session_obj
    
    @pytest.mark.asyncio
    async def test_middleware_metrics(self, rate_limit_middleware):
        """Test middleware metrics collection."""
        # Simulate some activity
        rate_limit_middleware.metrics.total_updates = 100
        rate_limit_middleware.metrics.successful_updates = 95
        rate_limit_middleware.metrics.blocked_updates = 3
        rate_limit_middleware.metrics.error_updates = 2
        rate_limit_middleware.metrics.total_processing_time = 1.5
        
        metrics = await rate_limit_middleware.get_metrics()
        
        assert metrics["total_updates"] == 100
        assert metrics["success_rate"] == 95.0
        assert metrics["average_processing_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_health_status(self, rate_limit_middleware):
        """Test middleware health status."""
        # Test healthy status
        rate_limit_middleware.metrics.total_updates = 100
        rate_limit_middleware.metrics.successful_updates = 95
        rate_limit_middleware.metrics.error_updates = 0
        rate_limit_middleware.emergency_mode = False
        
        with patch('app.core.redis.redis_manager.is_healthy', True):
            health = await rate_limit_middleware.get_health_status()
            assert health["status"] == "healthy"
        
        # Test degraded status
        rate_limit_middleware.metrics.error_updates = 15  # 15% error rate
        health = await rate_limit_middleware.get_health_status()
        assert health["status"] == "degraded"
        
        # Test critical status (emergency mode)
        rate_limit_middleware.emergency_mode = True
        health = await rate_limit_middleware.get_health_status()
        assert health["status"] == "critical"


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_redis_failover_scenario(self):
        """Test Redis failover and recovery."""
        mgr = AdvancedRedisManager()
        mgr._redis_client = AsyncMock()
        mgr._is_healthy = True
        
        # Simulate Redis failure
        mgr._redis_client.ping.side_effect = Exception("Connection lost")
        
        health = await mgr.health_check()
        assert health is False
        assert mgr._circuit_breaker_failures > 0
        
        # Simulate recovery
        mgr._redis_client.ping.side_effect = None
        mgr._redis_client.ping.return_value = True
        mgr._redis_client.set = AsyncMock()
        mgr._redis_client.get = AsyncMock(return_value="test")
        mgr._redis_client.delete = AsyncMock()
        
        health = await mgr.health_check()
        assert health is True
        assert mgr._circuit_breaker_open is False
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self):
        """Test system behavior under high load."""
        limiter = AdvancedRateLimiter()
        
        # Simulate high load
        tasks = []
        for i in range(100):
            task = limiter.check_rate_limit(
                user_id=i % 10,  # 10 different users
                rate_type=RateLimitType.MESSAGE,
                content=f"message {i}"
            )
            tasks.append(task)
        
        with patch('app.telegram.rate_limiter.sliding_window_rate_limit') as mock_rate_limit:
            # Simulate some requests being rate limited
            mock_rate_limit.return_value = RateLimitInfo(
                allowed=True, remaining=5, reset_time=60,
                current_count=1, limit=10, window_size=60, identifier="test"
            )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle all requests without crashing
            assert len(results) == 100
            success_count = sum(1 for allowed, _ in results if isinstance(allowed, bool) and allowed)
            assert success_count > 0
    
    @pytest.mark.asyncio
    async def test_data_consistency(self):
        """Test data consistency across operations."""
        mgr = AdvancedRedisManager()
        mgr._redis_client = AsyncMock()
        mgr._is_healthy = True
        
        # Test session data consistency
        session_data = {
            "user_id": 123,
            "chat_id": 456,
            "created_at": time.time(),
            "data": {"complex": {"nested": "structure"}}
        }
        
        # Mock serialization and deserialization
        mgr._redis_client.setex = AsyncMock()
        serialized = json.dumps(session_data, default=str)
        mgr._redis_client.get = AsyncMock(return_value=serialized)
        
        # Set and get session
        await mgr.set_session("test_session", session_data)
        retrieved = await mgr.get_session("test_session")
        
        # Data should be consistent
        assert retrieved["user_id"] == session_data["user_id"]
        assert retrieved["chat_id"] == session_data["chat_id"]
    
    @pytest.mark.asyncio
    async def test_cleanup_operations(self):
        """Test cleanup and maintenance operations."""
        mgr = AdvancedRedisManager()
        mgr._redis_client = AsyncMock()
        mgr._is_healthy = True
        
        # Mock cleanup operations
        mgr._redis_client.scan = AsyncMock(return_value=(0, ["expired_key1", "expired_key2"]))
        mgr._redis_client.delete = AsyncMock(return_value=2)
        
        cleaned = await mgr.flush_cache("expired:*", batch_size=100)
        assert cleaned == 2
        
        # Test rate limiter cleanup
        limiter = AdvancedRateLimiter()
        limiter.user_stats = {
            123: UserStats(last_request_time=time.time() - 7200),  # 2 hours ago
            456: UserStats(last_request_time=time.time() - 1800)   # 30 minutes ago
        }
        
        with patch('app.core.redis.redis_manager.flush_cache') as mock_flush:
            mock_flush.return_value = 10
            
            cleaned = await limiter.cleanup_expired_data()
            assert cleaned > 0


# Integration test runner
@pytest.mark.asyncio
async def test_full_integration():
    """Test full integration of all Redis components."""
    # Initialize components
    mgr = AdvancedRedisManager()
    mgr._redis_client = AsyncMock()
    mgr._is_healthy = True
    
    limiter = AdvancedRateLimiter()
    middleware = TelegramRateLimitMiddleware()
    
    # Mock all Redis operations
    mgr._redis_client.ping = AsyncMock()
    mgr._redis_client.setex = AsyncMock()
    mgr._redis_client.get = AsyncMock(return_value='{"test": "data"}')
    mgr._redis_client.delete = AsyncMock(return_value=1)
    mgr._redis_client.evalsha = AsyncMock(return_value=[1, 5, 1])
    mgr._redis_client.publish = AsyncMock(return_value=2)
    
    # Test cache operations
    success = await mgr.set_cache("test", {"data": "test"}, ttl=60)
    assert success is True
    
    data = await mgr.get_cache("test")
    assert data == {"test": "data"}
    
    # Test rate limiting
    with patch('app.telegram.rate_limiter.sliding_window_rate_limit') as mock_rate:
        mock_rate.return_value = RateLimitInfo(
            allowed=True, remaining=5, reset_time=60,
            current_count=1, limit=10, window_size=60, identifier="test"
        )
        
        allowed, info = await limiter.check_rate_limit(
            user_id=123,
            rate_type=RateLimitType.MESSAGE
        )
        assert allowed is True
    
    # Test middleware
    handler = AsyncMock(return_value="success")
    mock_update = MagicMock()
    mock_update.message.from_user.id = 123
    mock_update.message.chat.id = 456
    mock_update.message.text = "test"
    
    with patch('app.telegram.middleware.check_global_rate_limit') as mock_global:
        with patch('app.telegram.middleware.check_user_rate_limit') as mock_user:
            mock_global.return_value = (True, {"remaining": 100})
            mock_user.return_value = (True, {"remaining": 10})
            
            result = await middleware(handler, mock_update, {})
            assert result == "success"
    
    # Test health and metrics
    health = await mgr.health_check()
    assert health is True
    
    metrics = await middleware.get_metrics()
    assert "total_updates" in metrics
    
    print("✅ Full integration test passed!")


if __name__ == "__main__":
    asyncio.run(test_full_integration())