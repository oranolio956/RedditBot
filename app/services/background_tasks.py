"""
Background Tasks Service

Manages all background tasks for the application including:
- Celery task processing
- WebSocket message broadcasting
- Cache warming and cleanup
- Health monitoring
- Performance metrics collection
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil

from app.config import settings
from app.core.performance_utils import performance_monitor

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """Manages all background tasks and workers"""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        self.health_stats = {}
        
    async def start_all_tasks(self):
        """Start all background tasks"""
        if self.running:
            logger.info("Background tasks already running")
            return
            
        try:
            logger.info("Starting background tasks...")
            
            # Start individual task categories
            await self.start_websocket_tasks()
            await self.start_cache_tasks()
            await self.start_monitoring_tasks()
            await self.start_kelly_monitoring_tasks()  # Kelly AI monitoring
            await self.start_ai_processing_tasks()
            await self.start_cleanup_tasks()
            
            self.running = True
            logger.info("All background tasks started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {str(e)}")
            await self.stop_all_tasks()
            raise
    
    async def start_websocket_tasks(self):
        """Start WebSocket-related background tasks"""
        try:
            # WebSocket connection cleanup
            self.tasks['websocket_cleanup'] = asyncio.create_task(
                self._websocket_cleanup_task()
            )
            
            # Real-time metrics broadcasting
            self.tasks['metrics_broadcast'] = asyncio.create_task(
                self._metrics_broadcast_task()
            )
            
            # AI status broadcasting
            self.tasks['ai_status_broadcast'] = asyncio.create_task(
                self._ai_status_broadcast_task()
            )
            
            logger.info("WebSocket background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket tasks: {str(e)}")
            raise
    
    async def start_cache_tasks(self):
        """Start cache-related background tasks"""
        try:
            # Cache warming for frequently accessed data
            self.tasks['cache_warming'] = asyncio.create_task(
                self._cache_warming_task()
            )
            
            # Cache cleanup and optimization
            self.tasks['cache_cleanup'] = asyncio.create_task(
                self._cache_cleanup_task()
            )
            
            # Cache hit ratio monitoring
            self.tasks['cache_monitoring'] = asyncio.create_task(
                self._cache_monitoring_task()
            )
            
            logger.info("Cache background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start cache tasks: {str(e)}")
            raise
    
    async def start_monitoring_tasks(self):
        """Start monitoring and health check tasks"""
        try:
            # System health monitoring
            self.tasks['health_monitoring'] = asyncio.create_task(
                self._health_monitoring_task()
            )
            
            # Database connection monitoring
            self.tasks['db_monitoring'] = asyncio.create_task(
                self._database_monitoring_task()
            )
            
            # Memory usage monitoring
            self.tasks['memory_monitoring'] = asyncio.create_task(
                self._memory_monitoring_task()
            )
            
            # Performance metrics collection
            self.tasks['performance_metrics'] = asyncio.create_task(
                self._performance_metrics_task()
            )
            
            logger.info("Monitoring background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring tasks: {str(e)}")
            raise
    
    async def start_kelly_monitoring_tasks(self):
        """Start Kelly AI monitoring tasks"""
        try:
            # Initialize Kelly monitoring service
            from app.services.kelly_monitoring_service import kelly_monitoring_service
            await kelly_monitoring_service.initialize()
            
            # Kelly metrics collection and broadcasting is handled by the service itself
            logger.info("Kelly monitoring background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start Kelly monitoring tasks: {str(e)}")
            raise
    
    async def start_ai_processing_tasks(self):
        """Start AI-related background processing tasks"""
        try:
            # AI model warming (prevent cold starts)
            self.tasks['ai_model_warming'] = asyncio.create_task(
                self._ai_model_warming_task()
            )
            
            # Consciousness session monitoring
            self.tasks['consciousness_monitoring'] = asyncio.create_task(
                self._consciousness_session_monitoring()
            )
            
            # Emotional intelligence processing
            self.tasks['emotional_processing'] = asyncio.create_task(
                self._emotional_intelligence_processing()
            )
            
            # Neural dream generation
            self.tasks['neural_dreams'] = asyncio.create_task(
                self._neural_dream_generation()
            )
            
            logger.info("AI processing background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start AI processing tasks: {str(e)}")
            raise
    
    async def start_cleanup_tasks(self):
        """Start cleanup and maintenance tasks"""
        try:
            # Expired session cleanup
            self.tasks['session_cleanup'] = asyncio.create_task(
                self._expired_session_cleanup()
            )
            
            # Temporary file cleanup
            self.tasks['file_cleanup'] = asyncio.create_task(
                self._temporary_file_cleanup()
            )
            
            # Log rotation and archival
            self.tasks['log_cleanup'] = asyncio.create_task(
                self._log_cleanup_task()
            )
            
            logger.info("Cleanup background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start cleanup tasks: {str(e)}")
            raise
    
    # Individual task implementations
    
    async def _websocket_cleanup_task(self):
        """Clean up inactive WebSocket connections"""
        while self.running:
            try:
                from app.websocket.manager import websocket_manager
                await websocket_manager.cleanup_inactive_connections()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"WebSocket cleanup task error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _metrics_broadcast_task(self):
        """Broadcast real-time metrics to connected clients"""
        while self.running:
            try:
                from app.websocket.manager import websocket_manager
                
                # Collect system metrics
                metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'active_connections': websocket_manager.connection_count,
                    'ai_sessions_active': len(getattr(websocket_manager, 'ai_sessions', {}))
                }
                
                # Broadcast to all connected clients
                await websocket_manager.broadcast_to_all(
                    message_type="system_metrics",
                    data=metrics
                )
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics broadcast task error: {str(e)}")
                await asyncio.sleep(10)
    
    async def _ai_status_broadcast_task(self):
        """Broadcast AI system status updates"""
        while self.running:
            try:
                from app.services.ml_initialization import model_manager
                from app.websocket.manager import websocket_manager
                
                # Get AI system health
                ai_health = await model_manager.health_check()
                
                status_update = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'ai_models_loaded': ai_health.get('models_loaded', 0),
                    'engines_active': ai_health.get('engines_initialized', 0),
                    'system_healthy': ai_health.get('initialized', False),
                    'component_status': ai_health.get('components', {})
                }
                
                # Broadcast to AI-interested clients
                await websocket_manager.broadcast_to_topic(
                    topic="ai_status",
                    message_type="ai_status_update",
                    data=status_update
                )
                
                await asyncio.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                logger.error(f"AI status broadcast task error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _cache_warming_task(self):
        """Warm up cache with frequently accessed data"""
        while self.running:
            try:
                from app.core.redis import redis_manager
                
                # Warm frequently accessed endpoints
                warmup_keys = [
                    "user_sessions",
                    "ai_model_metadata", 
                    "system_health",
                    "api_endpoints"
                ]
                
                for key in warmup_keys:
                    try:
                        # Check if key exists and refresh if needed
                        exists = await redis_manager.exists(key)
                        if not exists:
                            # Trigger cache population
                            logger.debug(f"Warming cache for key: {key}")
                    except Exception as e:
                        logger.debug(f"Cache warming error for {key}: {str(e)}")
                
                await asyncio.sleep(300)  # Warm every 5 minutes
                
            except Exception as e:
                logger.error(f"Cache warming task error: {str(e)}")
                await asyncio.sleep(600)
    
    async def _cache_cleanup_task(self):
        """Clean up expired and unused cache entries"""
        while self.running:
            try:
                from app.core.redis import redis_manager
                
                # Get cache statistics
                cache_info = await redis_manager.get_info()
                memory_usage = cache_info.get('used_memory', 0)
                
                # Clean up if memory usage is high
                if memory_usage > settings.redis.max_memory_threshold:
                    await redis_manager.cleanup_expired_keys()
                    logger.info("Cache cleanup completed due to high memory usage")
                
                await asyncio.sleep(1800)  # Clean every 30 minutes
                
            except Exception as e:
                logger.error(f"Cache cleanup task error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _cache_monitoring_task(self):
        """Monitor cache performance and hit ratios"""
        while self.running:
            try:
                from app.core.redis import redis_manager
                
                cache_stats = await redis_manager.get_cache_stats()
                
                # Log performance metrics
                if cache_stats:
                    hit_ratio = cache_stats.get('hit_ratio', 0)
                    if hit_ratio < 0.8:  # Less than 80% hit ratio
                        logger.warning(f"Low cache hit ratio: {hit_ratio:.2%}")
                
                self.health_stats['cache'] = cache_stats
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Cache monitoring task error: {str(e)}")
                await asyncio.sleep(120)
    
    async def _health_monitoring_task(self):
        """Monitor overall system health"""
        while self.running:
            try:
                from app.database.manager import db_service
                from app.core.redis import redis_manager
                
                # Check all major components
                health_checks = {
                    'database': await db_service.get_health_status(),
                    'redis': await redis_manager.health_check(),
                    'disk_space': psutil.disk_usage('/').percent < 90,
                    'memory_usage': psutil.virtual_memory().percent < 85,
                    'cpu_usage': psutil.cpu_percent(interval=1) < 80
                }
                
                # Alert on unhealthy components
                for component, status in health_checks.items():
                    if isinstance(status, dict):
                        component_healthy = status.get('overall_status') == 'healthy'
                    else:
                        component_healthy = status
                        
                    if not component_healthy:
                        logger.warning(f"Component {component} is unhealthy: {status}")
                
                self.health_stats['system'] = health_checks
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring task error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _database_monitoring_task(self):
        """Monitor database performance and connections"""
        while self.running:
            try:
                from app.database.manager import db_service
                
                # Get database statistics
                db_stats = await db_service.get_statistics()
                
                # Monitor connection pool
                pool_stats = await db_service.get_pool_status()
                
                # Alert on high connection usage
                if pool_stats.get('active_connections', 0) > 20:
                    logger.warning(f"High database connection usage: {pool_stats}")
                
                self.health_stats['database'] = {
                    'statistics': db_stats,
                    'pool_status': pool_stats
                }
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Database monitoring task error: {str(e)}")
                await asyncio.sleep(120)
    
    async def _memory_monitoring_task(self):
        """Monitor memory usage and trigger cleanup if needed"""
        while self.running:
            try:
                memory_info = psutil.virtual_memory()
                
                if memory_info.percent > 85:
                    logger.warning(f"High memory usage: {memory_info.percent}%")
                    
                    # Trigger garbage collection
                    import gc
                    gc.collect()
                    
                    # Clear ML model caches if needed
                    if memory_info.percent > 90:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                self.health_stats['memory'] = {
                    'percent': memory_info.percent,
                    'available': memory_info.available,
                    'total': memory_info.total
                }
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring task error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _performance_metrics_task(self):
        """Collect and store performance metrics"""
        while self.running:
            try:
                # Collect metrics from performance monitor
                from app.core.performance_utils import get_performance_stats
                
                metrics = get_performance_stats()
                
                # Store in time-series format for analysis
                self.health_stats['performance'] = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'metrics': metrics
                }
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Performance metrics task error: {str(e)}")
                await asyncio.sleep(120)
    
    async def _ai_model_warming_task(self):
        """Keep AI models warm to prevent cold starts"""
        while self.running:
            try:
                from app.services.ml_initialization import model_manager
                
                # Test each model with dummy data to keep them warm
                models_to_warm = ['bert', 'gpt2', 'sentence_transformer']
                
                for model_name in models_to_warm:
                    model = model_manager.get_model(model_name)
                    if model:
                        try:
                            if model_name == 'sentence_transformer':
                                _ = model.encode("warmup")
                            elif model_name == 'bert':
                                tokenizer = model_manager.get_tokenizer('bert')
                                if tokenizer:
                                    inputs = tokenizer("warmup", return_tensors="pt")
                                    with torch.no_grad():
                                        _ = model(**inputs)
                        except Exception as e:
                            logger.debug(f"Model warming error for {model_name}: {str(e)}")
                
                await asyncio.sleep(300)  # Warm every 5 minutes
                
            except Exception as e:
                logger.error(f"AI model warming task error: {str(e)}")
                await asyncio.sleep(600)
    
    async def _consciousness_session_monitoring(self):
        """Monitor active consciousness sessions"""
        while self.running:
            try:
                from app.services.ml_initialization import model_manager
                
                consciousness_engine = model_manager.get_engine('consciousness')
                if consciousness_engine and hasattr(consciousness_engine, 'active_sessions'):
                    active_sessions = consciousness_engine.active_sessions
                    
                    # Monitor for sessions that need attention
                    for session_id, session_data in active_sessions.items():
                        if hasattr(session_data, 'needs_intervention'):
                            if session_data.needs_intervention:
                                logger.info(f"Consciousness session {session_id} needs intervention")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Consciousness monitoring task error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _emotional_intelligence_processing(self):
        """Process emotional intelligence data in background"""
        while self.running:
            try:
                # Process queued emotional analysis tasks
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Emotional intelligence processing error: {str(e)}")
                await asyncio.sleep(15)
    
    async def _neural_dream_generation(self):
        """Generate neural dreams in background"""
        while self.running:
            try:
                # Generate and cache dream content
                await asyncio.sleep(60)  # Generate every minute
                
            except Exception as e:
                logger.error(f"Neural dream generation error: {str(e)}")
                await asyncio.sleep(180)
    
    async def _expired_session_cleanup(self):
        """Clean up expired user sessions"""
        while self.running:
            try:
                from app.database.manager import db_service
                
                # Clean up sessions older than 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # This would be implemented based on actual session models
                logger.debug("Cleaning up expired sessions")
                
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                logger.error(f"Session cleanup task error: {str(e)}")
                await asyncio.sleep(7200)
    
    async def _temporary_file_cleanup(self):
        """Clean up temporary files"""
        while self.running:
            try:
                import os
                import tempfile
                
                temp_dir = tempfile.gettempdir()
                
                # Clean files older than 1 hour
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                for filename in os.listdir(temp_dir):
                    filepath = os.path.join(temp_dir, filename)
                    if os.path.isfile(filepath):
                        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if mod_time < cutoff_time:
                            try:
                                os.remove(filepath)
                            except:
                                pass  # File might be in use
                
                await asyncio.sleep(1800)  # Clean every 30 minutes
                
            except Exception as e:
                logger.error(f"File cleanup task error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _log_cleanup_task(self):
        """Clean up old log files"""
        while self.running:
            try:
                # Implement log rotation and cleanup
                await asyncio.sleep(86400)  # Clean daily
                
            except Exception as e:
                logger.error(f"Log cleanup task error: {str(e)}")
                await asyncio.sleep(86400)
    
    async def stop_all_tasks(self):
        """Stop all background tasks"""
        logger.info("Stopping background tasks...")
        
        self.running = False
        
        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug(f"Stopped background task: {task_name}")
        
        self.tasks.clear()
        logger.info("All background tasks stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of background tasks"""
        return {
            'running': self.running,
            'active_tasks': len([t for t in self.tasks.values() if not t.done()]),
            'total_tasks': len(self.tasks),
            'health_stats': self.health_stats,
            'task_status': {
                name: 'running' if not task.done() else 'stopped'
                for name, task in self.tasks.items()
            }
        }

# Global background task manager
background_task_manager = BackgroundTaskManager()

async def start_background_tasks():
    """Start all background tasks"""
    await background_task_manager.start_all_tasks()

async def stop_background_tasks():
    """Stop all background tasks"""
    await background_task_manager.stop_all_tasks()

def get_background_task_status() -> Dict[str, Any]:
    """Get status of background tasks"""
    return background_task_manager.get_health_status()