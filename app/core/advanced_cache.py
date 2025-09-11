"""
Advanced Caching System

Multi-tier caching with LRU eviction, intelligent warming,
and performance optimization for AI operations.
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

from app.core.redis import redis_manager
from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int
    last_accessed: datetime
    size_bytes: int
    tags: List[str]

class LRUCache:
    """In-memory LRU cache with intelligent eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory = 0
        
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, dict)):
                return len(pickle.dumps(value))
            else:
                return len(str(value))
        except:
            return 100  # Default estimate
    
    def _update_access_order(self, key: str):
        """Update access order for LRU tracking"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        while (len(self.cache) >= self.max_size or 
               self.current_memory >= self.max_memory_bytes) and self.access_order:
            
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                entry = self.cache[lru_key]
                self.current_memory -= entry.size_bytes
                del self.cache[lru_key]
                logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        
        # Check expiration
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            self.delete(key)
            return None
        
        # Update access metadata
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        self._update_access_order(key)
        
        return entry.value
    
    def set(self, key: str, value: Any, expires_in: Optional[int] = None, tags: List[str] = None):
        """Set value in cache"""
        size_bytes = self._calculate_size(value)
        expires_at = None
        
        if expires_in:
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        # Remove existing entry if it exists
        if key in self.cache:
            self.current_memory -= self.cache[key].size_bytes
        
        # Create new entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            access_count=1,
            last_accessed=datetime.utcnow(),
            size_bytes=size_bytes,
            tags=tags or []
        )
        
        # Evict if necessary
        self.current_memory += size_bytes
        self._evict_lru()
        
        # Store entry
        self.cache[key] = entry
        self._update_access_order(key)
        
        logger.debug(f"Cached entry: {key} ({size_bytes} bytes)")
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key not in self.cache:
            return False
            
        entry = self.cache[key]
        self.current_memory -= entry.size_bytes
        del self.cache[key]
        
        if key in self.access_order:
            self.access_order.remove(key)
            
        return True
    
    def clear_by_tags(self, tags: List[str]):
        """Clear entries by tags"""
        keys_to_delete = []
        
        for key, entry in self.cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        return {
            'entries': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_bytes': self.current_memory,
            'memory_usage_mb': self.current_memory / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'total_accesses': total_accesses,
            'memory_utilization': self.current_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0
        }

class AdvancedCacheManager:
    """Multi-tier caching system with Redis and in-memory layers"""
    
    def __init__(self):
        self.l1_cache = LRUCache(max_size=1000, max_memory_mb=50)  # Fast in-memory
        self.l2_cache = LRUCache(max_size=10000, max_memory_mb=200)  # Larger in-memory
        self.cache_stats = {
            'hits': {'l1': 0, 'l2': 0, 'redis': 0},
            'misses': {'total': 0},
            'sets': {'l1': 0, 'l2': 0, 'redis': 0},
            'evictions': {'l1': 0, 'l2': 0}
        }
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the advanced cache manager"""
        if self.initialized:
            return
            
        logger.info("Initializing advanced cache manager...")
        
        # Start cache warming tasks
        await self._start_warming_tasks()
        
        self.initialized = True
        logger.info("Advanced cache manager initialized")
    
    def _generate_cache_key(self, namespace: str, key: str, params: Dict[str, Any] = None) -> str:
        """Generate a consistent cache key"""
        if params:
            # Sort params for consistent key generation
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{namespace}:{key}:{param_hash}"
        return f"{namespace}:{key}"
    
    async def get(self, namespace: str, key: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get value from multi-tier cache"""
        cache_key = self._generate_cache_key(namespace, key, params)
        
        # Try L1 cache first (fastest)
        value = self.l1_cache.get(cache_key)
        if value is not None:
            self.cache_stats['hits']['l1'] += 1
            return value
        
        # Try L2 cache
        value = self.l2_cache.get(cache_key)
        if value is not None:
            self.cache_stats['hits']['l2'] += 1
            # Promote to L1
            self.l1_cache.set(cache_key, value, expires_in=3600)
            return value
        
        # Try Redis cache
        try:
            value = await redis_manager.get_cached_data(cache_key)
            if value is not None:
                self.cache_stats['hits']['redis'] += 1
                # Promote to L2 and L1
                self.l2_cache.set(cache_key, value, expires_in=7200)
                self.l1_cache.set(cache_key, value, expires_in=3600)
                return value
        except Exception as e:
            logger.warning(f"Redis cache error: {str(e)}")
        
        # Cache miss
        self.cache_stats['misses']['total'] += 1
        return None
    
    async def set(self, namespace: str, key: str, value: Any, expires_in: int = 3600, 
                  params: Dict[str, Any] = None, tags: List[str] = None):
        """Set value in multi-tier cache"""
        cache_key = self._generate_cache_key(namespace, key, params)
        
        # Store in all cache tiers
        self.l1_cache.set(cache_key, value, expires_in=min(expires_in, 3600), tags=tags)
        self.cache_stats['sets']['l1'] += 1
        
        self.l2_cache.set(cache_key, value, expires_in=min(expires_in, 7200), tags=tags)
        self.cache_stats['sets']['l2'] += 1
        
        # Store in Redis with longer TTL
        try:
            await redis_manager.cache_data(cache_key, value, expires_in)
            self.cache_stats['sets']['redis'] += 1
        except Exception as e:
            logger.warning(f"Redis cache set error: {str(e)}")
    
    async def delete(self, namespace: str, key: str, params: Dict[str, Any] = None):
        """Delete from all cache tiers"""
        cache_key = self._generate_cache_key(namespace, key, params)
        
        self.l1_cache.delete(cache_key)
        self.l2_cache.delete(cache_key)
        
        try:
            await redis_manager.delete_cached_data(cache_key)
        except Exception as e:
            logger.warning(f"Redis cache delete error: {str(e)}")
    
    async def clear_namespace(self, namespace: str):
        """Clear all entries in a namespace"""
        # This would require pattern matching - simplified implementation
        logger.info(f"Clearing cache namespace: {namespace}")
        
        # Clear Redis keys with pattern
        try:
            pattern = f"{namespace}:*"
            await redis_manager.delete_pattern(pattern)
        except Exception as e:
            logger.warning(f"Redis namespace clear error: {str(e)}")
    
    async def clear_by_tags(self, tags: List[str]):
        """Clear entries by tags"""
        self.l1_cache.clear_by_tags(tags)
        self.l2_cache.clear_by_tags(tags)
        
        # Redis tag clearing would require additional metadata storage
        logger.info(f"Cleared cache entries with tags: {tags}")
    
    # AI-specific caching methods
    
    async def cache_ai_result(self, model_name: str, input_hash: str, result: Any, expires_in: int = 3600):
        """Cache AI model result"""
        await self.set(
            namespace="ai_results",
            key=f"{model_name}:{input_hash}",
            value=result,
            expires_in=expires_in,
            tags=["ai", model_name]
        )
    
    async def get_ai_result(self, model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached AI model result"""
        return await self.get(
            namespace="ai_results",
            key=f"{model_name}:{input_hash}"
        )
    
    async def cache_consciousness_state(self, session_id: str, state_data: Dict[str, Any]):
        """Cache consciousness session state"""
        await self.set(
            namespace="consciousness",
            key=f"session:{session_id}",
            value=state_data,
            expires_in=1800,  # 30 minutes
            tags=["consciousness", "session"]
        )
    
    async def get_consciousness_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached consciousness session state"""
        return await self.get(
            namespace="consciousness",
            key=f"session:{session_id}"
        )
    
    async def cache_emotional_profile(self, user_id: int, profile_data: Dict[str, Any]):
        """Cache user emotional profile"""
        await self.set(
            namespace="emotional_profiles",
            key=str(user_id),
            value=profile_data,
            expires_in=7200,  # 2 hours
            tags=["emotional", "profile"]
        )
    
    async def get_emotional_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached emotional profile"""
        return await self.get(
            namespace="emotional_profiles",
            key=str(user_id)
        )
    
    # Cache warming strategies
    
    async def _start_warming_tasks(self):
        """Start background cache warming tasks"""
        self.warming_tasks['ai_models'] = asyncio.create_task(self._warm_ai_models())
        self.warming_tasks['user_data'] = asyncio.create_task(self._warm_user_data())
        self.warming_tasks['system_data'] = asyncio.create_task(self._warm_system_data())
    
    async def _warm_ai_models(self):
        """Warm cache with AI model metadata"""
        while True:
            try:
                # Cache frequently used AI model data
                model_metadata = {
                    'bert': {'type': 'transformer', 'warm': True},
                    'gpt2': {'type': 'generative', 'warm': True},
                    'sentence_transformer': {'type': 'embedding', 'warm': True}
                }
                
                for model_name, metadata in model_metadata.items():
                    await self.set(
                        namespace="ai_metadata",
                        key=model_name,
                        value=metadata,
                        expires_in=3600
                    )
                
                await asyncio.sleep(1800)  # Warm every 30 minutes
                
            except Exception as e:
                logger.error(f"AI model warming error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _warm_user_data(self):
        """Warm cache with frequently accessed user data"""
        while True:
            try:
                # This would warm frequently accessed user profiles
                await asyncio.sleep(3600)  # Warm every hour
                
            except Exception as e:
                logger.error(f"User data warming error: {str(e)}")
                await asyncio.sleep(7200)
    
    async def _warm_system_data(self):
        """Warm cache with system configuration data"""
        while True:
            try:
                # Cache system configuration and health data
                system_config = {
                    'features_enabled': [
                        'consciousness_mirroring',
                        'emotional_intelligence',
                        'quantum_consciousness',
                        'digital_telepathy'
                    ],
                    'cache_warm': True
                }
                
                await self.set(
                    namespace="system",
                    key="config",
                    value=system_config,
                    expires_in=7200
                )
                
                await asyncio.sleep(7200)  # Warm every 2 hours
                
            except Exception as e:
                logger.error(f"System data warming error: {str(e)}")
                await asyncio.sleep(7200)
    
    # Monitoring and statistics
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        # Calculate hit ratios
        total_hits = sum(self.cache_stats['hits'].values())
        total_requests = total_hits + self.cache_stats['misses']['total']
        hit_ratio = total_hits / total_requests if total_requests > 0 else 0
        
        redis_stats = {}
        try:
            redis_stats = await redis_manager.get_cache_stats()
        except Exception as e:
            logger.warning(f"Could not get Redis stats: {str(e)}")
        
        return {
            'hit_ratio': hit_ratio,
            'total_requests': total_requests,
            'cache_stats': self.cache_stats,
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'redis_cache': redis_stats,
            'warming_tasks_active': len([t for t in self.warming_tasks.values() if not t.done()])
        }
    
    async def cleanup(self):
        """Clean up cache manager"""
        # Cancel warming tasks
        for task in self.warming_tasks.values():
            task.cancel()
        
        # Clear caches
        self.l1_cache.cache.clear()
        self.l2_cache.cache.clear()
        
        logger.info("Advanced cache manager cleaned up")

# Global cache manager instance
cache_manager = AdvancedCacheManager()