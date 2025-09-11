"""
Database Optimization Service

Provides advanced database optimizations including:
- Query optimization and caching
- Connection pool management
- Async session handling
- Query monitoring and analysis
- Database health monitoring
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import QueuePool

from app.database.manager import db_service
from app.core.advanced_cache import cache_manager
from app.core.performance_utils import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class QueryStats:
    """Statistics for a database query"""
    query_hash: str
    query_text: str
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: datetime
    cache_hits: int
    cache_misses: int

class DatabaseOptimizer:
    """Database optimization and monitoring service"""
    
    def __init__(self):
        self.query_stats: Dict[str, QueryStats] = {}
        self.slow_query_threshold = 1.0  # 1 second
        self.cache_query_threshold = 0.5  # 500ms
        self.monitoring_enabled = True
        
    async def initialize(self):
        """Initialize database optimizer"""
        logger.info("Initializing database optimizer...")
        
        # Set up query monitoring
        if self.monitoring_enabled:
            await self._setup_query_monitoring()
        
        # Start optimization tasks
        asyncio.create_task(self._query_analysis_task())
        asyncio.create_task(self._connection_monitoring_task())
        
        logger.info("Database optimizer initialized")
    
    async def _setup_query_monitoring(self):
        """Set up SQLAlchemy event listeners for query monitoring"""
        try:
            from sqlalchemy import event
            from sqlalchemy.engine import Engine
            
            @event.listens_for(Engine, "before_cursor_execute")
            def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                context._query_start_time = time.time()
                context._query_statement = statement
            
            @event.listens_for(Engine, "after_cursor_execute")
            def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                total_time = time.time() - context._query_start_time
                
                # Track query performance
                asyncio.create_task(self._track_query_performance(
                    statement, total_time, parameters
                ))
            
            logger.info("Query monitoring event listeners registered")
            
        except Exception as e:
            logger.error(f"Failed to setup query monitoring: {str(e)}")
    
    async def _track_query_performance(self, query: str, execution_time: float, parameters: Any):
        """Track performance of individual queries"""
        try:
            # Generate query hash for grouping
            import hashlib
            query_normalized = self._normalize_query(query)
            query_hash = hashlib.md5(query_normalized.encode()).hexdigest()
            
            # Update or create query stats
            if query_hash in self.query_stats:
                stats = self.query_stats[query_hash]
                stats.execution_count += 1
                stats.total_time += execution_time
                stats.avg_time = stats.total_time / stats.execution_count
                stats.min_time = min(stats.min_time, execution_time)
                stats.max_time = max(stats.max_time, execution_time)
                stats.last_executed = datetime.utcnow()
            else:
                self.query_stats[query_hash] = QueryStats(
                    query_hash=query_hash,
                    query_text=query_normalized,
                    execution_count=1,
                    total_time=execution_time,
                    avg_time=execution_time,
                    min_time=execution_time,
                    max_time=execution_time,
                    last_executed=datetime.utcnow(),
                    cache_hits=0,
                    cache_misses=0
                )
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected ({execution_time:.3f}s): {query_normalized[:200]}...")
                
                # Store slow query for analysis
                await cache_manager.set(
                    namespace="slow_queries",
                    key=f"{query_hash}_{int(time.time())}",
                    value={
                        "query": query_normalized,
                        "execution_time": execution_time,
                        "parameters": str(parameters)[:500],
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    expires_in=86400  # Keep for 24 hours
                )
            
        except Exception as e:
            logger.error(f"Error tracking query performance: {str(e)}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent tracking"""
        # Remove extra whitespace and normalize case
        normalized = ' '.join(query.split()).lower()
        
        # Replace parameter placeholders with generic markers
        import re
        normalized = re.sub(r'\$\d+', '$N', normalized)  # PostgreSQL parameters
        normalized = re.sub(r'\?', '?', normalized)      # Generic parameters
        normalized = re.sub(r'= \d+', '= N', normalized) # Numeric values
        normalized = re.sub(r"= '[^']*'", "= 'STRING'", normalized)  # String values
        
        return normalized
    
    async def execute_with_cache(self, session: AsyncSession, query: str, 
                                parameters: Dict[str, Any] = None, 
                                cache_ttl: int = 300) -> Any:
        """Execute query with intelligent caching"""
        try:
            # Generate cache key
            import hashlib
            cache_key_data = f"{query}_{parameters}"
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
            
            # Try cache first
            cached_result = await cache_manager.get("query_cache", cache_key)
            if cached_result is not None:
                # Update cache hit stats
                query_hash = hashlib.md5(self._normalize_query(query).encode()).hexdigest()
                if query_hash in self.query_stats:
                    self.query_stats[query_hash].cache_hits += 1
                
                logger.debug(f"Query cache hit: {query[:100]}...")
                return cached_result
            
            # Execute query
            with performance_monitor(f"db_query_{cache_key[:8]}"):
                if parameters:
                    result = await session.execute(text(query), parameters)
                else:
                    result = await session.execute(text(query))
                
                # Convert result to cacheable format
                if hasattr(result, 'fetchall'):
                    rows = result.fetchall()
                    cacheable_result = [dict(row._mapping) for row in rows]
                else:
                    cacheable_result = result
            
            # Cache the result if query is cacheable
            if self._is_query_cacheable(query):
                await cache_manager.set(
                    namespace="query_cache",
                    key=cache_key,
                    value=cacheable_result,
                    expires_in=cache_ttl
                )
                
                # Update cache miss stats
                query_hash = hashlib.md5(self._normalize_query(query).encode()).hexdigest()
                if query_hash in self.query_stats:
                    self.query_stats[query_hash].cache_misses += 1
            
            return cacheable_result
            
        except Exception as e:
            logger.error(f"Error executing cached query: {str(e)}")
            raise
    
    def _is_query_cacheable(self, query: str) -> bool:
        """Determine if a query result should be cached"""
        query_lower = query.lower().strip()
        
        # Don't cache write operations
        write_keywords = ['insert', 'update', 'delete', 'create', 'drop', 'alter']
        if any(query_lower.startswith(keyword) for keyword in write_keywords):
            return False
        
        # Don't cache queries with time-sensitive functions
        time_functions = ['now()', 'current_timestamp', 'random()', 'uuid_generate']
        if any(func in query_lower for func in time_functions):
            return False
        
        # Cache SELECT queries by default
        return query_lower.startswith('select')
    
    async def get_optimized_session(self) -> AsyncSession:
        """Get an optimized database session"""
        try:
            session = await db_service.get_session()
            
            # Configure session for optimal performance
            await session.execute(text("SET statement_timeout = '30s'"))
            await session.execute(text("SET lock_timeout = '10s'"))
            await session.execute(text("SET idle_in_transaction_session_timeout = '60s'"))
            
            return session
            
        except Exception as e:
            logger.error(f"Error getting optimized session: {str(e)}")
            raise
    
    async def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns and provide optimization recommendations"""
        try:
            total_queries = len(self.query_stats)
            if total_queries == 0:
                return {"message": "No queries tracked yet"}
            
            # Find slow queries
            slow_queries = [
                stats for stats in self.query_stats.values()
                if stats.avg_time > self.slow_query_threshold
            ]
            
            # Find frequently executed queries
            frequent_queries = sorted(
                self.query_stats.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )[:10]
            
            # Calculate cache efficiency
            total_cache_requests = sum(
                stats.cache_hits + stats.cache_misses
                for stats in self.query_stats.values()
            )
            total_cache_hits = sum(stats.cache_hits for stats in self.query_stats.values())
            cache_hit_ratio = total_cache_hits / total_cache_requests if total_cache_requests > 0 else 0
            
            # Generate recommendations
            recommendations = []
            
            if len(slow_queries) > 0:
                recommendations.append({
                    "type": "slow_queries",
                    "description": f"Found {len(slow_queries)} slow queries that should be optimized",
                    "priority": "high"
                })
            
            if cache_hit_ratio < 0.7:
                recommendations.append({
                    "type": "cache_optimization",
                    "description": f"Cache hit ratio is {cache_hit_ratio:.2%}, consider increasing cache TTL",
                    "priority": "medium"
                })
            
            # Check for queries that could benefit from indexing
            for stats in frequent_queries:
                if 'where' in stats.query_text.lower() and stats.avg_time > 0.1:
                    recommendations.append({
                        "type": "indexing",
                        "description": f"Frequent query with WHERE clause may benefit from indexing",
                        "query_hash": stats.query_hash,
                        "priority": "medium"
                    })
            
            return {
                "total_queries_tracked": total_queries,
                "slow_queries_count": len(slow_queries),
                "cache_hit_ratio": cache_hit_ratio,
                "most_frequent_queries": [
                    {
                        "query_hash": q.query_hash,
                        "execution_count": q.execution_count,
                        "avg_time": q.avg_time,
                        "query_preview": q.query_text[:100]
                    }
                    for q in frequent_queries
                ],
                "slow_queries": [
                    {
                        "query_hash": q.query_hash,
                        "avg_time": q.avg_time,
                        "execution_count": q.execution_count,
                        "query_preview": q.query_text[:100]
                    }
                    for q in slow_queries[:5]
                ],
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query patterns: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """Optimize database connection pool settings"""
        try:
            # Get current pool status
            pool_status = await db_service.get_pool_status()
            
            recommendations = []
            
            # Analyze pool usage
            active_connections = pool_status.get('active_connections', 0)
            pool_size = pool_status.get('pool_size', 10)
            max_overflow = pool_status.get('max_overflow', 20)
            
            utilization = active_connections / pool_size if pool_size > 0 else 0
            
            if utilization > 0.8:
                recommendations.append({
                    "type": "pool_size",
                    "description": "High connection pool utilization, consider increasing pool size",
                    "current_size": pool_size,
                    "recommended_size": min(pool_size * 2, 50),
                    "priority": "high"
                })
            
            if utilization < 0.3 and pool_size > 5:
                recommendations.append({
                    "type": "pool_size",
                    "description": "Low connection pool utilization, consider decreasing pool size",
                    "current_size": pool_size,
                    "recommended_size": max(pool_size // 2, 5),
                    "priority": "low"
                })
            
            return {
                "current_pool_status": pool_status,
                "utilization": utilization,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error optimizing connection pool: {str(e)}")
            return {"error": str(e)}
    
    async def _query_analysis_task(self):
        """Background task for continuous query analysis"""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                analysis = await self.analyze_query_patterns()
                
                # Log recommendations
                if 'recommendations' in analysis:
                    for rec in analysis['recommendations']:
                        if rec['priority'] == 'high':
                            logger.warning(f"Database optimization needed: {rec['description']}")
                        else:
                            logger.info(f"Database recommendation: {rec['description']}")
                
                # Clear old query stats to prevent memory growth
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.query_stats = {
                    hash_key: stats for hash_key, stats in self.query_stats.items()
                    if stats.last_executed > cutoff_time
                }
                
            except Exception as e:
                logger.error(f"Query analysis task error: {str(e)}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _connection_monitoring_task(self):
        """Background task for connection pool monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                pool_status = await db_service.get_pool_status()
                
                # Alert on connection issues
                active_connections = pool_status.get('active_connections', 0)
                pool_size = pool_status.get('pool_size', 10)
                
                if active_connections >= pool_size:
                    logger.warning(f"Connection pool exhausted: {active_connections}/{pool_size}")
                
                # Store metrics for monitoring
                await cache_manager.set(
                    namespace="db_metrics",
                    key="pool_status",
                    value=pool_status,
                    expires_in=300
                )
                
            except Exception as e:
                logger.error(f"Connection monitoring task error: {str(e)}")
                await asyncio.sleep(120)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        if not self.query_stats:
            return {"message": "No performance data available"}
        
        total_queries = sum(stats.execution_count for stats in self.query_stats.values())
        total_time = sum(stats.total_time for stats in self.query_stats.values())
        avg_query_time = total_time / total_queries if total_queries > 0 else 0
        
        slow_queries = len([
            stats for stats in self.query_stats.values()
            if stats.avg_time > self.slow_query_threshold
        ])
        
        return {
            "total_unique_queries": len(self.query_stats),
            "total_query_executions": total_queries,
            "average_query_time": avg_query_time,
            "slow_queries_count": slow_queries,
            "monitoring_enabled": self.monitoring_enabled,
            "slow_query_threshold": self.slow_query_threshold,
            "cache_query_threshold": self.cache_query_threshold
        }

# Global database optimizer instance
db_optimizer = DatabaseOptimizer()