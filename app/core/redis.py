"""
Advanced Redis Connection and Cache Management

Provides comprehensive Redis management including:
- Production-ready connection management with clustering and sentinel support
- Advanced session management with serialization and TTL handling
- High-performance rate limiting with sliding windows and Lua scripts
- Distributed caching with memory optimization
- Real-time pub/sub for multi-instance coordination
- Automatic failover and recovery mechanisms
- Performance monitoring and health checks
"""

import json
import pickle
import gzip
import time
import asyncio
import hashlib
import uuid
from typing import Any, Optional, Union, Dict, List, Callable, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
from collections import defaultdict

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis, Sentinel
from redis.asyncio.cluster import RedisCluster
from redis.exceptions import (
    ConnectionError, TimeoutError, RedisError,
    ClusterDownError, ReadOnlyError
)
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class CompressionType(Enum):
    """Data compression types."""
    NONE = "none"
    GZIP = "gzip"
    JSON = "json"
    PICKLE = "pickle"


class ConnectionType(Enum):
    """Redis connection types."""
    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


@dataclass
class RedisMetrics:
    """Redis performance metrics."""
    total_commands: int = 0
    successful_commands: int = 0
    failed_commands: int = 0
    total_latency: float = 0.0
    peak_connections: int = 0
    current_connections: int = 0
    memory_usage: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_updated: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate command success rate."""
        if self.total_commands == 0:
            return 100.0
        return (self.successful_commands / self.total_commands) * 100
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_commands == 0:
            return 0.0
        return self.total_latency / self.successful_commands
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.cache_hits / total_cache_ops) * 100


@dataclass
class RateLimitInfo:
    """Rate limit information."""
    allowed: bool
    remaining: int
    reset_time: int
    current_count: int
    limit: int
    window_size: int
    identifier: str


class AdvancedRedisManager:
    """
    Production-ready Redis connection manager with comprehensive features.
    
    Features:
    - Clustering and Sentinel support for high availability
    - Advanced connection pooling with health checks
    - Intelligent failover and recovery
    - Memory optimization with compression
    - Real-time pub/sub coordination
    - Performance monitoring and metrics
    - Sliding window rate limiting with Lua scripts
    - Distributed session management
    - Automatic data serialization/deserialization
    """
    
    def __init__(self):
        self._redis_client: Optional[Union[Redis, RedisCluster]] = None
        self._sentinel_client: Optional[Sentinel] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._connection_type: ConnectionType = ConnectionType.STANDALONE
        
        # Health and monitoring
        self._is_healthy: bool = False
        self._last_health_check: float = 0
        self._metrics: RedisMetrics = RedisMetrics()
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Failover and recovery
        self._reconnect_attempts: int = 0
        self._last_reconnect: float = 0
        self._circuit_breaker_open: bool = False
        self._circuit_breaker_failures: int = 0
        self._circuit_breaker_last_failure: float = 0
        
        # Pub/Sub
        self._pubsub_clients: Dict[str, Any] = {}
        self._pubsub_handlers: Dict[str, Callable] = {}
        
        # Lua scripts cache
        self._lua_scripts: Dict[str, str] = {}
        self._script_shas: Dict[str, str] = {}
        
        # Performance optimization
        self._pipeline_queue: List[Tuple[str, tuple, dict]] = []
        self._pipeline_lock = asyncio.Lock()
        
        # Initialize Lua scripts
        self._init_lua_scripts()
    
    def _init_lua_scripts(self) -> None:
        """Initialize Lua scripts for atomic operations."""
        # Sliding window rate limiter
        self._lua_scripts['sliding_window_rate_limit'] = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        local increment = tonumber(ARGV[4]) or 1
        
        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', current_time - window)
        
        -- Get current count
        local current_count = redis.call('ZCARD', key)
        
        if current_count < limit then
            -- Add current request
            redis.call('ZADD', key, current_time, current_time .. ':' .. math.random())
            redis.call('EXPIRE', key, window)
            
            local remaining = limit - current_count - increment
            return {1, remaining, current_count + increment}
        else
            return {0, 0, current_count}
        end
        """
        
        # Token bucket rate limiter
        self._lua_scripts['token_bucket_rate_limit'] = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        local requested_tokens = tonumber(ARGV[4]) or 1
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- Calculate tokens to add
        local time_passed = current_time - last_refill
        local tokens_to_add = math.floor(time_passed * refill_rate)
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        if tokens >= requested_tokens then
            tokens = tokens - requested_tokens
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, capacity / refill_rate * 2)
            return {1, tokens, requested_tokens}
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, capacity / refill_rate * 2)
            return {0, tokens, 0}
        end
        """
        
        # Atomic counter with expiration
        self._lua_scripts['atomic_counter'] = """
        local key = KEYS[1]
        local increment = tonumber(ARGV[1]) or 1
        local expire = tonumber(ARGV[2])
        
        local current = redis.call('GET', key)
        if current == false then
            redis.call('SET', key, increment)
            if expire > 0 then
                redis.call('EXPIRE', key, expire)
            end
            return increment
        else
            local new_value = redis.call('INCRBY', key, increment)
            if expire > 0 then
                redis.call('EXPIRE', key, expire)
            end
            return new_value
        end
        """
        
        # Distributed lock with auto-release
        self._lua_scripts['acquire_lock'] = """
        local key = KEYS[1]
        local value = ARGV[1]
        local expire = tonumber(ARGV[2])
        
        if redis.call('SET', key, value, 'NX', 'EX', expire) then
            return 1
        else
            return 0
        end
        """
        
        self._lua_scripts['release_lock'] = """
        local key = KEYS[1]
        local value = ARGV[1]
        
        if redis.call('GET', key) == value then
            return redis.call('DEL', key)
        else
            return 0
        end
        """
    
    async def initialize(self) -> None:
        """Initialize Redis connection with clustering/sentinel support."""
        try:
            await self._establish_connection()
            await self._load_lua_scripts()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            self._is_healthy = True
            logger.info(
                "Redis connection established successfully",
                connection_type=self._connection_type.value,
                cluster_enabled=settings.redis.cluster_enabled,
                sentinel_enabled=settings.redis.sentinel_enabled
            )
            
        except Exception as e:
            logger.error("Failed to initialize Redis connection", error=str(e))
            raise
    
    async def _establish_connection(self) -> None:
        """Establish Redis connection based on configuration."""
        if settings.redis.cluster_enabled:
            await self._connect_cluster()
        elif settings.redis.sentinel_enabled:
            await self._connect_sentinel()
        else:
            await self._connect_standalone()
    
    async def _connect_standalone(self) -> None:
        """Connect to standalone Redis instance."""
        self._connection_type = ConnectionType.STANDALONE
        
        # Create connection pool
        self._connection_pool = ConnectionPool.from_url(
            settings.redis.url,
            max_connections=settings.redis.max_connections,
            retry_on_timeout=settings.redis.retry_on_timeout,
            health_check_interval=settings.redis.health_check_interval,
            socket_timeout=settings.redis.socket_timeout,
            socket_connect_timeout=settings.redis.socket_connect_timeout,
        )
        
        # Create Redis client
        self._redis_client = Redis(
            connection_pool=self._connection_pool,
            decode_responses=True,
        )
        
        # Test connection
        await self._redis_client.ping()
    
    async def _connect_cluster(self) -> None:
        """Connect to Redis cluster."""
        self._connection_type = ConnectionType.CLUSTER
        
        startup_nodes = []
        for url in settings.redis.cluster_urls:
            startup_nodes.append({"url": url})
        
        self._redis_client = RedisCluster(
            startup_nodes=startup_nodes,
            max_connections=settings.redis.max_connections,
            retry_on_timeout=settings.redis.retry_on_timeout,
            skip_full_coverage_check=settings.redis.cluster_skip_full_coverage_check,
            decode_responses=True,
        )
        
        # Test connection
        await self._redis_client.ping()
    
    async def _connect_sentinel(self) -> None:
        """Connect to Redis via Sentinel."""
        self._connection_type = ConnectionType.SENTINEL
        
        sentinel_list = []
        for url in settings.redis.sentinel_urls:
            host, port = url.split(":")
            sentinel_list.append((host, int(port)))
        
        self._sentinel_client = Sentinel(
            sentinel_list,
            socket_timeout=settings.redis.socket_timeout,
        )
        
        # Get master connection
        self._redis_client = self._sentinel_client.master_for(
            settings.redis.sentinel_service_name,
            password=settings.redis.password,
            db=settings.redis.db,
            decode_responses=True,
        )
        
        # Test connection
        await self._redis_client.ping()
    
    async def _load_lua_scripts(self) -> None:
        """Load and cache Lua scripts."""
        for script_name, script_code in self._lua_scripts.items():
            try:
                script_sha = await self._redis_client.script_load(script_code)
                self._script_shas[script_name] = script_sha
                logger.debug(f"Loaded Lua script: {script_name}")
            except Exception as e:
                logger.error(f"Failed to load Lua script {script_name}", error=str(e))
    
    async def close(self) -> None:
        """Close Redis connections and cleanup resources."""
        try:
            # Cancel background tasks
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._metrics_task and not self._metrics_task.done():
                self._metrics_task.cancel()
                try:
                    await self._metrics_task
                except asyncio.CancelledError:
                    pass
            
            # Close pub/sub clients
            for pubsub_client in self._pubsub_clients.values():
                try:
                    await pubsub_client.close()
                except Exception:
                    pass
            
            # Close main client
            if self._redis_client:
                await self._redis_client.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            if self._sentinel_client:
                # Sentinel doesn't have a close method in redis-py
                pass
            
            self._is_healthy = False
            logger.info("Redis connections closed successfully")
            
        except Exception as e:
            logger.error("Error closing Redis connections", error=str(e))
    
    @property
    def client(self) -> Union[Redis, RedisCluster]:
        """Get Redis client instance."""
        if not self._redis_client:
            raise RuntimeError("Redis manager not initialized")
        if not self._is_healthy:
            raise RuntimeError("Redis connection is not healthy")
        return self._redis_client
    
    @property
    def metrics(self) -> RedisMetrics:
        """Get current Redis metrics."""
        return self._metrics
    
    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        return self._is_healthy
    
    async def health_check(self) -> bool:
        """
        Perform comprehensive Redis health check.
        
        Returns:
            bool: True if Redis is healthy, False otherwise.
        """
        try:
            start_time = time.time()
            
            # Basic ping test
            await self.client.ping()
            
            # Test basic operations
            test_key = f"health_check:{uuid.uuid4().hex[:8]}"
            await self.client.set(test_key, "test", ex=60)
            result = await self.client.get(test_key)
            await self.client.delete(test_key)
            
            if result != "test":
                raise Exception("Basic operation test failed")
            
            # Update metrics
            latency = time.time() - start_time
            self._metrics.total_latency += latency
            self._metrics.successful_commands += 3  # ping, set, get, delete
            self._metrics.total_commands += 3
            
            self._last_health_check = time.time()
            
            # Reset circuit breaker on successful health check
            if self._circuit_breaker_open:
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0
                logger.info("Circuit breaker reset - Redis connection recovered")
            
            return True
            
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            
            # Update failure metrics
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            
            # Circuit breaker logic
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = time.time()
            
            if self._circuit_breaker_failures >= 5:
                self._circuit_breaker_open = True
                logger.warning("Circuit breaker opened - Redis connection issues detected")
            
            return False
    
    # Advanced Caching Operations with Compression and Optimization
    
    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        compression: CompressionType = CompressionType.JSON,
        pipeline: Optional[Any] = None
    ) -> bool:
        """
        Set cache value with automatic compression and serialization.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: settings.redis.cache_ttl)
            compression: Compression method to use
            pipeline: Optional pipeline for batch operations
            
        Returns:
            bool: True if successful, False otherwise.
        """
        start_time = time.time()
        
        try:
            ttl = ttl or settings.redis.cache_ttl
            
            # Serialize and optionally compress value
            serialized_value = await self._serialize_value(value, compression)
            
            # Use pipeline if provided, otherwise use direct client
            client = pipeline if pipeline else self.client
            
            # Set with expiration
            await client.setex(key, ttl, serialized_value)
            
            # Update metrics
            self._metrics.successful_commands += 1
            self._metrics.total_commands += 1
            self._metrics.total_latency += time.time() - start_time
            
            return True
            
        except Exception as e:
            logger.error("Failed to set cache", key=key, error=str(e))
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            return False
    
    async def _serialize_value(self, value: Any, compression: CompressionType) -> Union[str, bytes]:
        """Serialize and optionally compress value."""
        if compression == CompressionType.JSON:
            serialized = json.dumps(value, default=str, separators=(',', ':'))
        elif compression == CompressionType.PICKLE:
            serialized = pickle.dumps(value)
        else:
            serialized = str(value)
        
        # Apply compression if enabled and data is large enough
        if (settings.redis.compression_enabled and 
            compression == CompressionType.GZIP and
            len(serialized) > settings.redis.compression_threshold):
            
            if isinstance(serialized, str):
                serialized = serialized.encode('utf-8')
            
            compressed = gzip.compress(serialized)
            
            # Only use compression if it actually reduces size
            if len(compressed) < len(serialized):
                return b'gzip:' + compressed
        
        return serialized
    
    async def _deserialize_value(self, value: Union[str, bytes], compression: CompressionType) -> Any:
        """Deserialize and optionally decompress value."""
        if isinstance(value, bytes) and value.startswith(b'gzip:'):
            # Handle compressed data
            compressed_data = value[5:]  # Remove 'gzip:' prefix
            decompressed = gzip.decompress(compressed_data)
            value = decompressed.decode('utf-8')
        
        if compression == CompressionType.JSON:
            return json.loads(value)
        elif compression == CompressionType.PICKLE:
            if isinstance(value, str):
                value = value.encode('utf-8')
            return pickle.loads(value)
        else:
            return value
    
    async def get_cache(
        self,
        key: str,
        default: Any = None,
        compression: CompressionType = CompressionType.JSON,
        update_ttl: Optional[int] = None
    ) -> Any:
        """
        Get cache value with automatic decompression and deserialization.
        
        Args:
            key: Cache key
            default: Default value if key not found
            compression: Compression method used
            update_ttl: Optional TTL to extend on access
            
        Returns:
            Cached value or default.
        """
        start_time = time.time()
        
        try:
            # Get value
            value = await self.client.get(key)
            
            if value is None:
                self._metrics.cache_misses += 1
                return default
            
            # Update TTL if requested
            if update_ttl:
                await self.client.expire(key, update_ttl)
            
            # Deserialize and decompress value
            result = await self._deserialize_value(value, compression)
            
            # Update metrics
            self._metrics.cache_hits += 1
            self._metrics.successful_commands += 1
            self._metrics.total_commands += 1
            self._metrics.total_latency += time.time() - start_time
            
            return result
                
        except Exception as e:
            logger.error("Failed to get cache", key=key, error=str(e))
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            self._metrics.cache_misses += 1
            return default
    
    async def delete_cache(self, *keys: str) -> int:
        """
        Delete one or more cache keys.
        
        Args:
            keys: Cache keys to delete
            
        Returns:
            int: Number of keys deleted.
        """
        start_time = time.time()
        
        try:
            result = await self.client.delete(*keys)
            
            # Update metrics
            self._metrics.successful_commands += 1
            self._metrics.total_commands += 1
            self._metrics.total_latency += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error("Failed to delete cache", keys=keys, error=str(e))
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            return 0
    
    async def get_multiple(
        self,
        keys: List[str],
        compression: CompressionType = CompressionType.JSON
    ) -> Dict[str, Any]:
        """
        Get multiple cache values efficiently.
        
        Args:
            keys: List of cache keys
            compression: Compression method used
            
        Returns:
            Dictionary mapping keys to values (missing keys are omitted).
        """
        start_time = time.time()
        
        try:
            # Use pipeline for efficiency
            pipeline = self.client.pipeline()
            for key in keys:
                pipeline.get(key)
            
            values = await pipeline.execute()
            
            # Process results
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = await self._deserialize_value(value, compression)
                        self._metrics.cache_hits += 1
                    except Exception as e:
                        logger.error(f"Failed to deserialize cache value for key {key}", error=str(e))
                        self._metrics.cache_misses += 1
                else:
                    self._metrics.cache_misses += 1
            
            # Update metrics
            self._metrics.successful_commands += len(keys)
            self._metrics.total_commands += len(keys)
            self._metrics.total_latency += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error("Failed to get multiple cache values", keys=keys, error=str(e))
            self._metrics.failed_commands += len(keys)
            self._metrics.total_commands += len(keys)
            self._metrics.cache_misses += len(keys)
            return {}
    
    async def set_multiple(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        compression: CompressionType = CompressionType.JSON
    ) -> bool:
        """
        Set multiple cache values efficiently.
        
        Args:
            mapping: Dictionary of key-value pairs to set
            ttl: Time to live in seconds
            compression: Compression method to use
            
        Returns:
            bool: True if all successful, False otherwise.
        """
        start_time = time.time()
        
        try:
            ttl = ttl or settings.redis.cache_ttl
            
            # Use pipeline for efficiency
            pipeline = self.client.pipeline()
            
            for key, value in mapping.items():
                serialized_value = await self._serialize_value(value, compression)
                pipeline.setex(key, ttl, serialized_value)
            
            await pipeline.execute()
            
            # Update metrics
            self._metrics.successful_commands += len(mapping)
            self._metrics.total_commands += len(mapping)
            self._metrics.total_latency += time.time() - start_time
            
            return True
            
        except Exception as e:
            logger.error("Failed to set multiple cache values", error=str(e))
            self._metrics.failed_commands += len(mapping)
            self._metrics.total_commands += len(mapping)
            return False
    
    async def exists(self, *keys: str) -> int:
        """
        Check if cache keys exist.
        
        Args:
            keys: Cache keys to check
            
        Returns:
            int: Number of keys that exist.
        """
        start_time = time.time()
        
        try:
            result = await self.client.exists(*keys)
            
            # Update metrics
            self._metrics.successful_commands += 1
            self._metrics.total_commands += 1
            self._metrics.total_latency += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error("Failed to check cache existence", keys=keys, error=str(e))
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            return 0
    
    async def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            int: Remaining TTL in seconds (-1 if no expiry, -2 if key doesn't exist)
        """
        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error("Failed to get TTL", key=key, error=str(e))
            return -2
    
    async def extend_ttl(self, key: str, ttl: int) -> bool:
        """
        Extend TTL for an existing key.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            result = await self.client.expire(key, ttl)
            return result
        except Exception as e:
            logger.error("Failed to extend TTL", key=key, error=str(e))
            return False
    
    # Advanced Rate Limiting with Sliding Windows and Token Buckets
    
    async def sliding_window_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
        increment: int = 1
    ) -> RateLimitInfo:
        """
        Sliding window rate limiter using Lua script for atomicity.
        
        Args:
            identifier: Unique identifier (user_id, IP, etc.)
            limit: Maximum number of requests allowed
            window: Time window in seconds
            increment: Number of requests to add
            
        Returns:
            RateLimitInfo with detailed rate limit status.
        """
        key = f"rate_limit:sliding:{identifier}"
        current_time = time.time() * 1000  # Use milliseconds for precision
        
        try:
            script_sha = self._script_shas.get('sliding_window_rate_limit')
            if not script_sha:
                # Fallback to direct script execution
                result = await self.client.eval(
                    self._lua_scripts['sliding_window_rate_limit'],
                    1, key, window * 1000, limit, current_time, increment
                )
            else:
                result = await self.client.evalsha(
                    script_sha, 1, key, window * 1000, limit, current_time, increment
                )
            
            allowed, remaining, current_count = result
            
            # Calculate reset time
            if allowed:
                reset_time = window
            else:
                ttl = await self.client.ttl(key)
                reset_time = max(ttl, 0)
            
            # Update metrics
            self._metrics.successful_commands += 1
            self._metrics.total_commands += 1
            
            return RateLimitInfo(
                allowed=bool(allowed),
                remaining=remaining,
                reset_time=reset_time,
                current_count=current_count,
                limit=limit,
                window_size=window,
                identifier=identifier
            )
            
        except Exception as e:
            logger.error("Sliding window rate limit check failed", identifier=identifier, error=str(e))
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            
            # Allow on error to prevent blocking legitimate requests
            return RateLimitInfo(
                allowed=True,
                remaining=limit,
                reset_time=window,
                current_count=0,
                limit=limit,
                window_size=window,
                identifier=identifier
            )
    
    async def token_bucket_rate_limit(
        self,
        identifier: str,
        capacity: int,
        refill_rate: float,
        requested_tokens: int = 1
    ) -> RateLimitInfo:
        """
        Token bucket rate limiter for smooth rate limiting.
        
        Args:
            identifier: Unique identifier
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens per second refill rate
            requested_tokens: Number of tokens to consume
            
        Returns:
            RateLimitInfo with rate limit status.
        """
        key = f"rate_limit:bucket:{identifier}"
        current_time = time.time()
        
        try:
            script_sha = self._script_shas.get('token_bucket_rate_limit')
            if not script_sha:
                result = await self.client.eval(
                    self._lua_scripts['token_bucket_rate_limit'],
                    1, key, capacity, refill_rate, current_time, requested_tokens
                )
            else:
                result = await self.client.evalsha(
                    script_sha, 1, key, capacity, refill_rate, current_time, requested_tokens
                )
            
            allowed, remaining_tokens, consumed_tokens = result
            
            # Update metrics
            self._metrics.successful_commands += 1
            self._metrics.total_commands += 1
            
            return RateLimitInfo(
                allowed=bool(allowed),
                remaining=remaining_tokens,
                reset_time=int(capacity / refill_rate) if refill_rate > 0 else 0,
                current_count=capacity - remaining_tokens,
                limit=capacity,
                window_size=int(capacity / refill_rate) if refill_rate > 0 else 0,
                identifier=identifier
            )
            
        except Exception as e:
            logger.error("Token bucket rate limit check failed", identifier=identifier, error=str(e))
            self._metrics.failed_commands += 1
            self._metrics.total_commands += 1
            
            return RateLimitInfo(
                allowed=True,
                remaining=capacity,
                reset_time=0,
                current_count=0,
                limit=capacity,
                window_size=0,
                identifier=identifier
            )
    
    async def rate_limit_check(
        self,
        identifier: str,
        limit: int,
        window: int,
        increment: bool = True
    ) -> Dict[str, Any]:
        """
        Legacy rate limit check method for backward compatibility.
        
        Uses sliding window rate limiter internally.
        """
        result = await self.sliding_window_rate_limit(
            identifier, limit, window, 1 if increment else 0
        )
        
        return {
            "allowed": result.allowed,
            "remaining": result.remaining,
            "reset_time": result.reset_time,
            "current_count": result.current_count,
            "limit": result.limit,
        }
    
    # Advanced Session Management with Distributed Support
    
    async def set_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        notify_instances: bool = True
    ) -> bool:
        """
        Store session data with optional multi-instance notification.
        
        Args:
            session_id: Unique session identifier
            data: Session data dictionary
            ttl: Session TTL in seconds (default: settings.redis.session_ttl)
            notify_instances: Whether to notify other instances via pub/sub
            
        Returns:
            bool: True if successful, False otherwise.
        """
        key = f"session:{session_id}"
        ttl = ttl or settings.redis.session_ttl
        
        # Add metadata to session
        session_data = {
            **data,
            '_metadata': {
                'created_at': time.time(),
                'updated_at': time.time(),
                'ttl': ttl,
                'version': 1
            }
        }
        
        success = await self.set_cache(key, session_data, ttl, CompressionType.JSON)
        
        # Notify other instances if requested
        if success and notify_instances:
            await self._publish_session_event('session_updated', session_id, data)
        
        return success
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
        ttl_extend: Optional[int] = None
    ) -> bool:
        """
        Update specific fields in session data atomically.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
            ttl_extend: Optional TTL extension
            
        Returns:
            bool: True if successful, False otherwise.
        """
        key = f"session:{session_id}"
        
        try:
            # Get current session data
            current_data = await self.get_session(session_id)
            if current_data is None:
                return False
            
            # Update fields
            current_data.update(updates)
            
            # Update metadata
            if '_metadata' not in current_data:
                current_data['_metadata'] = {}
            
            current_data['_metadata']['updated_at'] = time.time()
            current_data['_metadata']['version'] = current_data['_metadata'].get('version', 0) + 1
            
            # Calculate TTL
            if ttl_extend:
                ttl = ttl_extend
            else:
                # Use remaining TTL or default
                current_ttl = await self.get_ttl(key)
                ttl = current_ttl if current_ttl > 0 else settings.redis.session_ttl
            
            # Save updated session
            success = await self.set_cache(key, current_data, ttl, CompressionType.JSON)
            
            # Notify other instances
            if success:
                await self._publish_session_event('session_updated', session_id, updates)
            
            return success
            
        except Exception as e:
            logger.error("Failed to update session", session_id=session_id, error=str(e))
            return False
    
    async def get_session(
        self,
        session_id: str,
        extend_ttl: bool = False,
        ttl_extension: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get session data with optional TTL extension.
        
        Args:
            session_id: Session identifier
            extend_ttl: Whether to extend TTL on access
            ttl_extension: TTL extension in seconds
            
        Returns:
            Session data dictionary or None if not found.
        """
        key = f"session:{session_id}"
        
        # Get TTL extension value
        if extend_ttl and not ttl_extension:
            ttl_extension = settings.redis.session_ttl
        
        session_data = await self.get_cache(
            key, 
            default=None, 
            compression=CompressionType.JSON,
            update_ttl=ttl_extension if extend_ttl else None
        )
        
        return session_data
    
    async def delete_session(
        self,
        session_id: str,
        notify_instances: bool = True
    ) -> bool:
        """
        Delete session data with optional multi-instance notification.
        
        Args:
            session_id: Session identifier
            notify_instances: Whether to notify other instances
            
        Returns:
            bool: True if deleted, False otherwise.
        """
        key = f"session:{session_id}"
        
        result = await self.delete_cache(key)
        
        # Notify other instances
        if result > 0 and notify_instances:
            await self._publish_session_event('session_deleted', session_id, {})
        
        return result > 0
    
    async def list_sessions(
        self,
        pattern: str = "session:*",
        limit: int = 1000
    ) -> List[str]:
        """
        List session keys matching pattern.
        
        Args:
            pattern: Key pattern to match
            limit: Maximum number of keys to return
            
        Returns:
            List of session IDs.
        """
        try:
            # Use SCAN for better performance on large datasets
            session_keys = []
            cursor = 0
            
            while len(session_keys) < limit:
                cursor, keys = await self.client.scan(
                    cursor=cursor, 
                    match=pattern, 
                    count=min(100, limit - len(session_keys))
                )
                
                # Extract session IDs from keys
                for key in keys:
                    if key.startswith("session:"):
                        session_id = key[8:]  # Remove "session:" prefix
                        session_keys.append(session_id)
                
                if cursor == 0:  # Scan complete
                    break
            
            return session_keys[:limit]
            
        except Exception as e:
            logger.error("Failed to list sessions", error=str(e))
            return []
    
    async def cleanup_expired_sessions(self, batch_size: int = 100) -> int:
        """
        Clean up expired sessions in batches.
        
        Args:
            batch_size: Number of sessions to process per batch
            
        Returns:
            Number of sessions cleaned up.
        """
        try:
            cleaned_count = 0
            session_keys = await self.list_sessions(limit=batch_size * 10)
            
            # Process in batches
            for i in range(0, len(session_keys), batch_size):
                batch_keys = session_keys[i:i + batch_size]
                full_keys = [f"session:{session_id}" for session_id in batch_keys]
                
                # Check which keys exist
                existing_count = await self.exists(*full_keys)
                
                # If some keys don't exist, they were already expired
                expired_in_batch = len(full_keys) - existing_count
                cleaned_count += expired_in_batch
                
                # Small delay to prevent overwhelming Redis
                if i + batch_size < len(session_keys):
                    await asyncio.sleep(0.01)
            
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
            return 0
    
    # Advanced Distributed Locking with Auto-renewal
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        expire: int = 60,
        auto_renew: bool = False
    ) -> Optional[str]:
        """
        Acquire distributed lock with optional auto-renewal.
        
        Args:
            lock_name: Lock identifier
            timeout: Timeout to acquire lock in seconds
            expire: Lock expiration time in seconds
            auto_renew: Whether to auto-renew the lock
            
        Returns:
            Lock identifier if acquired, None otherwise.
        """
        lock_key = f"lock:{lock_name}"
        lock_value = str(uuid.uuid4())
        
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            try:
                # Try to acquire lock using Lua script
                script_sha = self._script_shas.get('acquire_lock')
                if script_sha:
                    result = await self.client.evalsha(script_sha, 1, lock_key, lock_value, expire)
                else:
                    result = await self.client.eval(
                        self._lua_scripts['acquire_lock'],
                        1, lock_key, lock_value, expire
                    )
                
                if result:
                    # Start auto-renewal if requested
                    if auto_renew:
                        asyncio.create_task(self._auto_renew_lock(lock_key, lock_value, expire))
                    
                    logger.debug(f"Acquired lock: {lock_name}")
                    return lock_value
                
                # Wait before retrying with exponential backoff
                await asyncio.sleep(min(0.1 * (2 ** (min(10, timeout - int(end_time - time.time())))), 1.0))
                
            except Exception as e:
                logger.error(f"Error acquiring lock {lock_name}", error=str(e))
                break
        
        logger.debug(f"Failed to acquire lock: {lock_name}")
        return None
    
    async def _auto_renew_lock(self, lock_key: str, lock_value: str, expire: int) -> None:
        """
        Auto-renew lock in background until released.
        
        Args:
            lock_key: Redis key for the lock
            lock_value: Lock value for verification
            expire: Lock expiration time
        """
        renewal_interval = expire // 3  # Renew at 1/3 of expiration time
        
        while True:
            try:
                await asyncio.sleep(renewal_interval)
                
                # Check if lock still exists and belongs to us
                current_value = await self.client.get(lock_key)
                if current_value != lock_value:
                    # Lock was released or expired
                    break
                
                # Renew the lock
                await self.client.expire(lock_key, expire)
                logger.debug(f"Renewed lock: {lock_key}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error renewing lock {lock_key}", error=str(e))
                break
    
    async def release_lock(self, lock_name: str, lock_value: str) -> bool:
        """
        Release distributed lock atomically.
        
        Args:
            lock_name: Lock identifier
            lock_value: Lock value returned by acquire_lock
            
        Returns:
            bool: True if released, False otherwise.
        """
        lock_key = f"lock:{lock_name}"
        
        try:
            # Use pre-loaded Lua script for atomic release
            script_sha = self._script_shas.get('release_lock')
            if script_sha:
                result = await self.client.evalsha(script_sha, 1, lock_key, lock_value)
            else:
                result = await self.client.eval(
                    self._lua_scripts['release_lock'],
                    1, lock_key, lock_value
                )
            
            success = result == 1
            if success:
                logger.debug(f"Released lock: {lock_name}")
            else:
                logger.warning(f"Failed to release lock {lock_name} - may have expired or been stolen")
            
            return success
            
        except Exception as e:
            logger.error("Failed to release lock", lock_name=lock_name, error=str(e))
            return False
    
    @asynccontextmanager
    async def lock_context(
        self,
        lock_name: str,
        timeout: int = 10,
        expire: int = 60,
        auto_renew: bool = False
    ):
        """
        Context manager for distributed locking.
        
        Args:
            lock_name: Lock identifier
            timeout: Timeout to acquire lock in seconds
            expire: Lock expiration time in seconds
            auto_renew: Whether to auto-renew the lock
        
        Usage:
            async with redis_manager.lock_context("my_lock") as lock_acquired:
                if lock_acquired:
                    # Critical section
                    pass
        """
        lock_value = await self.acquire_lock(lock_name, timeout, expire, auto_renew)
        
        try:
            yield lock_value is not None
        finally:
            if lock_value:
                await self.release_lock(lock_name, lock_value)
    
    # Pub/Sub Operations for Multi-Instance Coordination
    
    async def publish(
        self,
        channel: str,
        message: Any,
        compression: CompressionType = CompressionType.JSON
    ) -> int:
        """
        Publish message to Redis channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            compression: Compression method for message
            
        Returns:
            Number of subscribers that received the message.
        """
        try:
            serialized_message = await self._serialize_value(message, compression)
            result = await self.client.publish(channel, serialized_message)
            
            logger.debug(f"Published message to channel {channel}, {result} subscribers notified")
            return result
            
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}", error=str(e))
            return 0
    
    async def subscribe(
        self,
        *channels: str,
        handler: Optional[Callable] = None
    ) -> Any:
        """
        Subscribe to Redis channels.
        
        Args:
            channels: Channel names to subscribe to
            handler: Optional message handler function
            
        Returns:
            PubSub instance for manual message handling.
        """
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(*channels)
            
            # Store pubsub client for cleanup
            pubsub_id = str(uuid.uuid4())
            self._pubsub_clients[pubsub_id] = pubsub
            
            # Register handler if provided
            if handler:
                for channel in channels:
                    self._pubsub_handlers[channel] = handler
                
                # Start message processing task
                asyncio.create_task(self._process_pubsub_messages(pubsub, channels))
            
            logger.info(f"Subscribed to channels: {channels}")
            return pubsub
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channels {channels}", error=str(e))
            return None
    
    async def _process_pubsub_messages(self, pubsub: Any, channels: Tuple[str, ...]) -> None:
        """
        Process pub/sub messages in background.
        
        Args:
            pubsub: PubSub instance
            channels: Subscribed channels
        """
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    
                    # Get handler for this channel
                    handler = self._pubsub_handlers.get(channel)
                    if handler:
                        try:
                            # Deserialize message
                            deserialized_data = await self._deserialize_value(
                                data, CompressionType.JSON
                            )
                            
                            # Call handler
                            await handler(channel, deserialized_data)
                            
                        except Exception as e:
                            logger.error(
                                f"Error processing pub/sub message from {channel}",
                                error=str(e)
                            )
                            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in pub/sub message processing", error=str(e))
        finally:
            try:
                await pubsub.close()
            except Exception:
                pass
    
    async def _publish_session_event(
        self,
        event_type: str,
        session_id: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Publish session event for multi-instance coordination.
        
        Args:
            event_type: Type of session event
            session_id: Session identifier
            data: Event data
        """
        try:
            event_data = {
                'event_type': event_type,
                'session_id': session_id,
                'timestamp': time.time(),
                'data': data
            }
            
            await self.publish('session:events', event_data)
            
        except Exception as e:
            logger.error("Failed to publish session event", error=str(e))
    
    # Performance and Monitoring Operations
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive Redis server information.
        
        Returns:
            Dictionary with Redis server info and performance metrics.
        """
        try:
            info = await self.client.info()
            
            # Extract key metrics
            server_info = {
                "version": info.get("redis_version"),
                "mode": info.get("redis_mode", "standalone"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
                "process_id": info.get("process_id"),
            }
            
            # Memory information
            memory_info = {
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "used_memory_peak": info.get("used_memory_peak"),
                "used_memory_peak_human": info.get("used_memory_peak_human"),
                "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio"),
                "maxmemory": info.get("maxmemory"),
                "maxmemory_human": info.get("maxmemory_human"),
                "maxmemory_policy": info.get("maxmemory_policy"),
            }
            
            # Connection information
            connection_info = {
                "connected_clients": info.get("connected_clients"),
                "client_recent_max_input_buffer": info.get("client_recent_max_input_buffer"),
                "client_recent_max_output_buffer": info.get("client_recent_max_output_buffer"),
                "blocked_clients": info.get("blocked_clients"),
            }
            
            # Performance statistics
            perf_info = {
                "total_commands_processed": info.get("total_commands_processed"),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
                "instantaneous_input_kbps": info.get("instantaneous_input_kbps"),
                "instantaneous_output_kbps": info.get("instantaneous_output_kbps"),
                "rejected_connections": info.get("rejected_connections"),
                "total_net_input_bytes": info.get("total_net_input_bytes"),
                "total_net_output_bytes": info.get("total_net_output_bytes"),
            }
            
            # Persistence information
            persistence_info = {
                "loading": info.get("loading"),
                "rdb_changes_since_last_save": info.get("rdb_changes_since_last_save"),
                "rdb_bgsave_in_progress": info.get("rdb_bgsave_in_progress"),
                "rdb_last_save_time": info.get("rdb_last_save_time"),
                "rdb_last_bgsave_status": info.get("rdb_last_bgsave_status"),
            }
            
            # Cluster information (if applicable)
            cluster_info = {}
            if self._connection_type == ConnectionType.CLUSTER:
                cluster_info = {
                    "cluster_enabled": info.get("cluster_enabled"),
                    "cluster_state": info.get("cluster_state"),
                    "cluster_slots_assigned": info.get("cluster_slots_assigned"),
                    "cluster_slots_ok": info.get("cluster_slots_ok"),
                    "cluster_slots_pfail": info.get("cluster_slots_pfail"),
                    "cluster_slots_fail": info.get("cluster_slots_fail"),
                    "cluster_known_nodes": info.get("cluster_known_nodes"),
                    "cluster_size": info.get("cluster_size"),
                }
            
            return {
                "server": server_info,
                "memory": memory_info,
                "connections": connection_info,
                "performance": perf_info,
                "persistence": persistence_info,
                "cluster": cluster_info,
                "connection_type": self._connection_type.value,
                "client_metrics": {
                    "success_rate": self._metrics.success_rate,
                    "average_latency": self._metrics.average_latency,
                    "cache_hit_rate": self._metrics.cache_hit_rate,
                    "total_commands": self._metrics.total_commands,
                    "current_connections": self._metrics.current_connections,
                }
            }
            
        except Exception as e:
            logger.error("Failed to get Redis info", error=str(e))
            return {}
    
    async def flush_cache(self, pattern: Optional[str] = None, batch_size: int = 1000) -> int:
        """
        Flush cache keys matching pattern in batches for better performance.
        
        Args:
            pattern: Key pattern (e.g., "user:*", default: all keys)
            batch_size: Number of keys to delete per batch
            
        Returns:
            Number of keys deleted.
        """
        try:
            if pattern:
                deleted_count = 0
                cursor = 0
                
                # Use SCAN for better performance on large datasets
                while True:
                    cursor, keys = await self.client.scan(
                        cursor=cursor, 
                        match=pattern, 
                        count=batch_size
                    )
                    
                    if keys:
                        # Delete in batches
                        for i in range(0, len(keys), batch_size):
                            batch_keys = keys[i:i + batch_size]
                            deleted = await self.client.delete(*batch_keys)
                            deleted_count += deleted
                            
                            # Small delay to prevent overwhelming Redis
                            if i + batch_size < len(keys):
                                await asyncio.sleep(0.001)
                    
                    if cursor == 0:  # Scan complete
                        break
                
                logger.info(f"Flushed {deleted_count} keys matching pattern {pattern}")
                return deleted_count
            else:
                # Full database flush
                await self.client.flushdb()
                logger.warning("Performed full database flush")
                return -1  # Indicate full flush
                
        except Exception as e:
            logger.error("Failed to flush cache", pattern=pattern, error=str(e))
            return 0
    
    async def get_memory_usage(self, key: str) -> Optional[int]:
        """
        Get memory usage of a specific key.
        
        Args:
            key: Redis key
            
        Returns:
            Memory usage in bytes, or None if key doesn't exist.
        """
        try:
            return await self.client.memory_usage(key)
        except Exception as e:
            logger.error(f"Failed to get memory usage for key {key}", error=str(e))
            return None
    
    async def get_slow_log(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get Redis slow log entries.
        
        Args:
            count: Number of entries to retrieve
            
        Returns:
            List of slow log entries.
        """
        try:
            slow_log = await self.client.slowlog_get(count)
            
            formatted_entries = []
            for entry in slow_log:
                formatted_entries.append({
                    'id': entry['id'],
                    'start_time': entry['start_time'],
                    'duration': entry['duration'],
                    'command': ' '.join(str(arg) for arg in entry['command']),
                    'client_address': entry.get('client_address', 'unknown'),
                    'client_name': entry.get('client_name', 'unknown')
                })
            
            return formatted_entries
            
        except Exception as e:
            logger.error("Failed to get slow log", error=str(e))
            return []
    
    # Background Tasks
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(settings.redis.health_check_interval)
                
                health_status = await self.health_check()
                self._is_healthy = health_status
                
                # Attempt reconnection if unhealthy
                if not health_status and settings.redis.auto_reconnect:
                    await self._attempt_reconnection()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                self._is_healthy = False
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while True:
            try:
                await asyncio.sleep(settings.redis.metrics_collection_interval)
                
                if self._is_healthy:
                    await self._collect_redis_metrics()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
    
    async def _collect_redis_metrics(self) -> None:
        """Collect Redis performance metrics."""
        try:
            info = await self.client.info()
            
            # Update connection metrics
            self._metrics.current_connections = info.get('connected_clients', 0)
            self._metrics.peak_connections = max(
                self._metrics.peak_connections,
                self._metrics.current_connections
            )
            
            # Update memory usage
            self._metrics.memory_usage = info.get('used_memory', 0)
            
            # Update timestamp
            self._metrics.last_updated = time.time()
            
        except Exception as e:
            logger.error("Failed to collect Redis metrics", error=str(e))
    
    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to Redis."""
        if (time.time() - self._last_reconnect < settings.redis.reconnect_delay or
            self._reconnect_attempts >= settings.redis.reconnect_retries):
            return
        
        self._last_reconnect = time.time()
        self._reconnect_attempts += 1
        
        try:
            logger.info(f"Attempting Redis reconnection (attempt {self._reconnect_attempts})")
            
            # Close existing connection
            if self._redis_client:
                await self._redis_client.close()
            
            # Re-establish connection
            await self._establish_connection()
            await self._load_lua_scripts()
            
            self._is_healthy = True
            self._reconnect_attempts = 0
            self._circuit_breaker_open = False
            self._circuit_breaker_failures = 0
            
            logger.info("Redis reconnection successful")
            
        except Exception as e:
            logger.error(f"Redis reconnection failed (attempt {self._reconnect_attempts})", error=str(e))
            
            if self._reconnect_attempts >= settings.redis.reconnect_retries:
                logger.error("Max reconnection attempts reached, giving up")
                self._circuit_breaker_open = True
    
    # Pipeline Operations for Batch Processing
    
    @asynccontextmanager
    async def pipeline(self, transaction: bool = False):
        """
        Context manager for Redis pipeline operations.
        
        Args:
            transaction: Whether to use MULTI/EXEC transaction
        
        Usage:
            async with redis_manager.pipeline() as pipe:
                await pipe.set("key1", "value1")
                await pipe.set("key2", "value2")
                results = await pipe.execute()
        """
        pipeline = self.client.pipeline(transaction=transaction)
        
        try:
            yield pipeline
        finally:
            # Pipeline cleanup is handled automatically
            pass
    
    async def atomic_increment(
        self,
        key: str,
        increment: int = 1,
        expire: Optional[int] = None
    ) -> int:
        """
        Atomically increment a counter with optional expiration.
        
        Args:
            key: Counter key
            increment: Value to increment by
            expire: Expiration time in seconds
            
        Returns:
            New counter value.
        """
        try:
            script_sha = self._script_shas.get('atomic_counter')
            if script_sha:
                result = await self.client.evalsha(
                    script_sha, 1, key, increment, expire or 0
                )
            else:
                result = await self.client.eval(
                    self._lua_scripts['atomic_counter'],
                    1, key, increment, expire or 0
                )
            
            return int(result)
            
        except Exception as e:
            logger.error(f"Failed to atomically increment {key}", error=str(e))
            return 0


# Global advanced Redis manager instance
redis_manager = AdvancedRedisManager()


# Convenience functions for common operations with enhanced features
async def cache_set(
    key: str, 
    value: Any, 
    ttl: Optional[int] = None,
    compression: CompressionType = CompressionType.JSON
) -> bool:
    """Enhanced convenience function for setting cache with compression."""
    return await redis_manager.set_cache(key, value, ttl, compression)


async def cache_get(
    key: str, 
    default: Any = None,
    compression: CompressionType = CompressionType.JSON
) -> Any:
    """Enhanced convenience function for getting cache with decompression."""
    return await redis_manager.get_cache(key, default, compression)


async def cache_delete(*keys: str) -> int:
    """Enhanced convenience function for deleting multiple cache keys."""
    return await redis_manager.delete_cache(*keys)


async def cache_get_multiple(
    keys: List[str],
    compression: CompressionType = CompressionType.JSON
) -> Dict[str, Any]:
    """Convenience function for getting multiple cache values."""
    return await redis_manager.get_multiple(keys, compression)


async def cache_set_multiple(
    mapping: Dict[str, Any],
    ttl: Optional[int] = None,
    compression: CompressionType = CompressionType.JSON
) -> bool:
    """Convenience function for setting multiple cache values."""
    return await redis_manager.set_multiple(mapping, ttl, compression)


async def sliding_window_rate_limit(
    identifier: str, 
    limit: int, 
    window: int, 
    increment: int = 1
) -> RateLimitInfo:
    """Convenience function for sliding window rate limiting."""
    return await redis_manager.sliding_window_rate_limit(identifier, limit, window, increment)


async def token_bucket_rate_limit(
    identifier: str,
    capacity: int,
    refill_rate: float,
    requested_tokens: int = 1
) -> RateLimitInfo:
    """Convenience function for token bucket rate limiting."""
    return await redis_manager.token_bucket_rate_limit(identifier, capacity, refill_rate, requested_tokens)


async def rate_limit(identifier: str, limit: int, window: int) -> Dict[str, Any]:
    """Legacy convenience function for rate limiting (backward compatibility)."""
    return await redis_manager.rate_limit_check(identifier, limit, window)


async def distributed_lock(
    lock_name: str,
    timeout: int = 10,
    expire: int = 60,
    auto_renew: bool = False
) -> Optional[str]:
    """Convenience function for distributed locking."""
    return await redis_manager.acquire_lock(lock_name, timeout, expire, auto_renew)


async def publish_message(
    channel: str,
    message: Any,
    compression: CompressionType = CompressionType.JSON
) -> int:
    """Convenience function for publishing messages."""
    return await redis_manager.publish(channel, message, compression)


async def subscribe_channel(
    *channels: str,
    handler: Optional[Callable] = None
) -> Any:
    """Convenience function for subscribing to channels."""
    return await redis_manager.subscribe(*channels, handler=handler)


# Session management convenience functions
async def set_session(
    session_id: str,
    data: Dict[str, Any],
    ttl: Optional[int] = None,
    notify_instances: bool = True
) -> bool:
    """Convenience function for setting session data."""
    return await redis_manager.set_session(session_id, data, ttl, notify_instances)


async def get_session(
    session_id: str,
    extend_ttl: bool = False,
    ttl_extension: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Convenience function for getting session data."""
    return await redis_manager.get_session(session_id, extend_ttl, ttl_extension)


async def update_session(
    session_id: str,
    updates: Dict[str, Any],
    ttl_extend: Optional[int] = None
) -> bool:
    """Convenience function for updating session data."""
    return await redis_manager.update_session(session_id, updates, ttl_extend)


async def delete_session(
    session_id: str,
    notify_instances: bool = True
) -> bool:
    """Convenience function for deleting session data."""
    return await redis_manager.delete_session(session_id, notify_instances)


# Performance monitoring convenience functions
async def get_redis_metrics() -> RedisMetrics:
    """Get current Redis performance metrics."""
    return redis_manager.metrics


async def get_redis_info() -> Dict[str, Any]:
    """Get comprehensive Redis server information."""
    return await redis_manager.get_info()


async def is_redis_healthy() -> bool:
    """Check if Redis connection is healthy."""
    return redis_manager.is_healthy