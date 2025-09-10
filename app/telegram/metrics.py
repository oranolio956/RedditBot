"""
Telegram Bot Metrics and Monitoring

Comprehensive metrics collection and monitoring for the Telegram bot
with Prometheus integration and real-time analytics.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest
)
import psutil
import aioredis

from app.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughput_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    concurrent_connections: int = 0
    peak_concurrent_connections: int = 0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Network metrics
    network_sent: int = 0
    network_received: int = 0


@dataclass
class BehaviorMetrics:
    """User behavior metrics."""
    active_users: int = 0
    new_users_today: int = 0
    returning_users: int = 0
    
    # Interaction patterns
    messages_per_user: Dict[int, int] = field(default_factory=dict)
    commands_usage: Dict[str, int] = field(default_factory=dict)
    conversation_lengths: List[int] = field(default_factory=list)
    
    # Time-based patterns
    hourly_activity: List[int] = field(default_factory=lambda: [0] * 24)
    daily_activity: List[int] = field(default_factory=lambda: [0] * 7)
    
    # Geographic/linguistic
    language_distribution: Dict[str, int] = field(default_factory=dict)
    timezone_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class AntiDetectionMetrics:
    """Anti-detection system metrics."""
    suspicious_patterns_detected: int = 0
    risk_mitigation_applied: int = 0
    false_positives: int = 0
    successful_evasions: int = 0
    
    # Pattern analysis
    typing_pattern_variations: int = 0
    timing_adjustments: int = 0
    behavior_randomizations: int = 0
    
    # Effectiveness
    detection_accuracy: float = 0.0
    evasion_success_rate: float = 0.0


class TelegramMetrics:
    """
    Comprehensive metrics system for Telegram bot.
    
    Features:
    - Prometheus metrics integration
    - Real-time performance tracking
    - User behavior analytics
    - Anti-detection metrics
    - System resource monitoring
    - Custom business metrics
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.redis: Optional[aioredis.Redis] = None
        
        # Performance tracking
        self.performance = PerformanceMetrics()
        self.behavior = BehaviorMetrics()
        self.anti_detection = AntiDetectionMetrics()
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Memory leak prevention
        self._memory_stats = {
            'peak_memory': 0,
            'cleanup_runs': 0,
            'last_cleanup_time': 0
        }
        
        # Real-time tracking with memory management
        self._start_time = time.time()
        self._message_timestamps = deque(maxlen=5000)  # Reduced from 10000
        self._error_timestamps = deque(maxlen=500)     # Reduced from 1000
        
        # Memory management settings
        self._max_memory_mb = 256  # Maximum memory usage in MB
        self._cleanup_interval = 3600  # Cleanup every hour
        self._last_cleanup = time.time()
        
        # Background tasks
        self._metrics_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        # Message metrics
        self.messages_total = Counter(
            'telegram_messages_total',
            'Total messages processed',
            ['message_type', 'status'],
            registry=self.registry
        )
        
        self.message_processing_duration = Histogram(
            'telegram_message_processing_seconds',
            'Time spent processing messages',
            ['message_type'],
            registry=self.registry,
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # User metrics
        self.active_users = Gauge(
            'telegram_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        self.user_sessions = Gauge(
            'telegram_user_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
        
        # Command metrics
        self.commands_total = Counter(
            'telegram_commands_total',
            'Total commands executed',
            ['command', 'status'],
            registry=self.registry
        )
        
        self.command_processing_duration = Histogram(
            'telegram_command_processing_seconds',
            'Time spent processing commands',
            ['command'],
            registry=self.registry
        )
        
        # API metrics
        self.api_requests_total = Counter(
            'telegram_api_requests_total',
            'Total Telegram API requests',
            ['method', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'telegram_api_request_seconds',
            'Telegram API request duration',
            ['method'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'telegram_rate_limit_hits_total',
            'Rate limit hits',
            ['limit_type'],
            registry=self.registry
        )
        
        self.rate_limit_remaining = Gauge(
            'telegram_rate_limit_remaining',
            'Remaining rate limit capacity',
            ['limit_type'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'telegram_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['breaker_name'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'telegram_circuit_breaker_failures_total',
            'Circuit breaker failures',
            ['breaker_name', 'failure_type'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'telegram_system_cpu_usage_percent',
            'System CPU usage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'telegram_system_memory_usage_bytes',
            'System memory usage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'telegram_system_disk_usage_percent',
            'System disk usage',
            registry=self.registry
        )
        
        # Anti-detection metrics
        self.anti_detection_patterns = Counter(
            'telegram_anti_detection_patterns_total',
            'Anti-detection patterns applied',
            ['pattern_type'],
            registry=self.registry
        )
        
        self.risk_assessments = Histogram(
            'telegram_risk_assessments',
            'Risk assessment scores',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            registry=self.registry
        )
        
        # Bot health metrics
        self.bot_uptime = Gauge(
            'telegram_bot_uptime_seconds',
            'Bot uptime in seconds',
            registry=self.registry
        )
        
        self.bot_health_status = Gauge(
            'telegram_bot_health_status',
            'Bot health status (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
    
    async def initialize(self) -> None:
        """Initialize metrics system."""
        try:
            # Initialize Redis connection for metrics storage
            self.redis = await aioredis.from_url(
                settings.redis.url,
                max_connections=10,
                decode_responses=True
            )
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Telegram metrics system initialized")
            
        except Exception as e:
            logger.error("Failed to initialize metrics system", error=str(e))
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background metrics collection tasks."""
        # System metrics updater
        self._metrics_tasks.append(
            asyncio.create_task(self._system_metrics_loop())
        )
        
        # Performance calculator
        self._metrics_tasks.append(
            asyncio.create_task(self._performance_metrics_loop())
        )
        
        # Behavior analyzer
        self._metrics_tasks.append(
            asyncio.create_task(self._behavior_analysis_loop())
        )
        
        # Metrics persistence
        self._metrics_tasks.append(
            asyncio.create_task(self._metrics_persistence_loop())
        )
    
    # Message tracking methods
    async def record_message_received(
        self,
        message_type: str = "text",
        processing_time: float = 0.0
    ) -> None:
        """Record message received with memory management."""
        timestamp = time.time()
        
        # Check memory usage periodically
        await self._check_and_cleanup_memory()
        
        self._message_timestamps.append(timestamp)
        self.messages_total.labels(message_type=message_type, status='received').inc()
        
        if processing_time > 0:
            self.message_processing_duration.labels(message_type=message_type).observe(processing_time)
        
        # Update performance tracking with bounds checking
        if len(self.performance.response_times) >= self.performance.response_times.maxlen:
            # Force dequeue to maintain memory bounds
            pass  # deque automatically handles maxlen
        self.performance.response_times.append(processing_time)
    
    async def record_message_sent(
        self,
        message_type: str = "text",
        success: bool = True
    ) -> None:
        """Record message sent."""
        status = 'sent' if success else 'failed'
        self.messages_total.labels(message_type=message_type, status=status).inc()
        
        if not success:
            self._error_timestamps.append(time.time())
    
    async def record_command_executed(
        self,
        command: str,
        processing_time: float,
        success: bool = True
    ) -> None:
        """Record command execution."""
        status = 'success' if success else 'error'
        
        self.commands_total.labels(command=command, status=status).inc()
        self.command_processing_duration.labels(command=command).observe(processing_time)
        
        # Update behavior tracking
        self.behavior.commands_usage[command] = self.behavior.commands_usage.get(command, 0) + 1
    
    async def record_api_request(
        self,
        method: str,
        duration: float,
        status_code: int
    ) -> None:
        """Record Telegram API request."""
        status = 'success' if 200 <= status_code < 300 else 'error'
        
        self.api_requests_total.labels(method=method, status=status).inc()
        self.api_request_duration.labels(method=method).observe(duration)
    
    async def record_rate_limit_hit(
        self,
        limit_type: str,
        remaining: int = 0
    ) -> None:
        """Record rate limit hit."""
        self.rate_limit_hits.labels(limit_type=limit_type).inc()
        self.rate_limit_remaining.labels(limit_type=limit_type).set(remaining)
    
    async def record_circuit_breaker_event(
        self,
        breaker_name: str,
        state: str,
        failure_type: Optional[str] = None
    ) -> None:
        """Record circuit breaker event."""
        state_value = {'closed': 0, 'open': 1, 'half_open': 2}.get(state, 0)
        self.circuit_breaker_state.labels(breaker_name=breaker_name).set(state_value)
        
        if failure_type:
            self.circuit_breaker_failures.labels(
                breaker_name=breaker_name,
                failure_type=failure_type
            ).inc()
    
    async def record_anti_detection_event(
        self,
        pattern_type: str,
        risk_score: float = 0.0
    ) -> None:
        """Record anti-detection event."""
        self.anti_detection_patterns.labels(pattern_type=pattern_type).inc()
        
        if risk_score > 0:
            self.risk_assessments.observe(risk_score)
        
        # Update anti-detection metrics
        if pattern_type == 'suspicious_detected':
            self.anti_detection.suspicious_patterns_detected += 1
        elif pattern_type == 'risk_mitigation':
            self.anti_detection.risk_mitigation_applied += 1
        elif pattern_type == 'typing_variation':
            self.anti_detection.typing_pattern_variations += 1
        elif pattern_type == 'timing_adjustment':
            self.anti_detection.timing_adjustments += 1
        elif pattern_type == 'behavior_randomization':
            self.anti_detection.behavior_randomizations += 1
    
    async def record_user_activity(
        self,
        user_id: int,
        language_code: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> None:
        """Record user activity with memory bounds checking."""
        # Update message count for user with memory management
        self.behavior.messages_per_user[user_id] = (
            self.behavior.messages_per_user.get(user_id, 0) + 1
        )
        
        # Limit the number of users tracked to prevent memory leaks
        if len(self.behavior.messages_per_user) > 10000:
            # Keep only the most active users
            sorted_users = sorted(
                self.behavior.messages_per_user.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5000]
            self.behavior.messages_per_user = dict(sorted_users)
        
        # Update language distribution with bounds
        if language_code:
            self.behavior.language_distribution[language_code] = (
                self.behavior.language_distribution.get(language_code, 0) + 1
            )
            
            # Limit language distribution size
            if len(self.behavior.language_distribution) > 1000:
                sorted_langs = sorted(
                    self.behavior.language_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:500]
                self.behavior.language_distribution = dict(sorted_langs)
        
        # Update timezone distribution with bounds
        if timezone:
            self.behavior.timezone_distribution[timezone] = (
                self.behavior.timezone_distribution.get(timezone, 0) + 1
            )
            
            # Limit timezone distribution size
            if len(self.behavior.timezone_distribution) > 500:
                sorted_zones = sorted(
                    self.behavior.timezone_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:250]
                self.behavior.timezone_distribution = dict(sorted_zones)
        
        # Update hourly activity (bounded array)
        current_hour = datetime.now().hour
        self.behavior.hourly_activity[current_hour] += 1
        
        # Update daily activity (bounded array)
        current_day = datetime.now().weekday()
        self.behavior.daily_activity[current_day] += 1
    
    async def update_active_users(self, count: int) -> None:
        """Update active users count."""
        self.active_users.set(count)
        self.behavior.active_users = count
    
    async def update_user_sessions(self, count: int) -> None:
        """Update user sessions count."""
        self.user_sessions.set(count)
    
    async def update_concurrent_connections(self, count: int) -> None:
        """Update concurrent connections."""
        self.performance.concurrent_connections = count
        self.performance.peak_concurrent_connections = max(
            self.performance.peak_concurrent_connections,
            count
        )
    
    async def increment_messages_sent(self) -> None:
        """Increment messages sent counter."""
        await self.record_message_sent()
    
    async def increment_errors(self) -> None:
        """Increment error counter."""
        self._error_timestamps.append(time.time())
        await self.record_message_sent(success=False)
    
    async def update_health_status(
        self,
        bot_healthy: bool = True,
        redis_healthy: bool = True
    ) -> None:
        """Update bot health status."""
        overall_health = 1 if (bot_healthy and redis_healthy) else 0
        self.bot_health_status.set(overall_health)
    
    async def update_bot_status(self, status: Any) -> None:
        """Update bot status metrics."""
        if hasattr(status, 'current_connections'):
            await self.update_concurrent_connections(status.current_connections)
        
        # Update uptime
        uptime = time.time() - self._start_time
        self.bot_uptime.set(uptime)
    
    async def _system_metrics_loop(self) -> None:
        """Background task for system metrics collection."""
        while not self._shutdown_event.is_set():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage.set(cpu_percent)
                self.performance.cpu_usage = cpu_percent
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_memory_usage.set(memory.used)
                self.performance.memory_usage = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.system_disk_usage.set(disk_percent)
                self.performance.disk_usage = disk_percent
                
                # Network I/O
                network = psutil.net_io_counters()
                self.performance.network_sent = network.bytes_sent
                self.performance.network_received = network.bytes_recv
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
                await asyncio.sleep(30)
    
    async def _performance_metrics_loop(self) -> None:
        """Background task for performance metrics calculation."""
        while not self._shutdown_event.is_set():
            try:
                now = time.time()
                
                # Calculate throughput per minute
                recent_messages = sum(
                    1 for ts in self._message_timestamps
                    if now - ts <= 60
                )
                self.performance.throughput_per_minute.append(recent_messages)
                
                # Calculate error rate
                recent_errors = sum(
                    1 for ts in self._error_timestamps
                    if now - ts <= 300  # Last 5 minutes
                )
                total_recent = sum(
                    1 for ts in self._message_timestamps
                    if now - ts <= 300
                )
                
                error_rate = (recent_errors / max(1, total_recent)) * 100
                self.performance.error_rates.append(error_rate)
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error calculating performance metrics", error=str(e))
                await asyncio.sleep(60)
    
    async def _behavior_analysis_loop(self) -> None:
        """Background task for behavior analysis."""
        while not self._shutdown_event.is_set():
            try:
                # Analyze conversation lengths
                # This would typically involve database queries
                # For now, we'll update based on current session data
                
                # Calculate anti-detection effectiveness
                total_detections = self.anti_detection.suspicious_patterns_detected
                total_mitigations = self.anti_detection.risk_mitigation_applied
                
                if total_detections > 0:
                    self.anti_detection.detection_accuracy = (
                        total_mitigations / total_detections
                    )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in behavior analysis", error=str(e))
                await asyncio.sleep(300)
    
    async def _metrics_persistence_loop(self) -> None:
        """Background task for metrics persistence."""
        while not self._shutdown_event.is_set():
            try:
                # Persist key metrics to Redis for historical analysis
                metrics_data = {
                    'timestamp': time.time(),
                    'performance': {
                        'cpu_usage': self.performance.cpu_usage,
                        'memory_usage': self.performance.memory_usage,
                        'concurrent_connections': self.performance.concurrent_connections,
                        'avg_response_time': (
                            sum(self.performance.response_times) / 
                            max(1, len(self.performance.response_times))
                        ),
                        'throughput': list(self.performance.throughput_per_minute)[-1] if self.performance.throughput_per_minute else 0,
                        'error_rate': list(self.performance.error_rates)[-1] if self.performance.error_rates else 0,
                    },
                    'behavior': {
                        'active_users': self.behavior.active_users,
                        'total_interactions': len(self._message_timestamps),
                        'top_commands': dict(list(self.behavior.commands_usage.items())[:10]),
                    },
                    'anti_detection': {
                        'patterns_applied': self.anti_detection.typing_pattern_variations,
                        'risk_mitigations': self.anti_detection.risk_mitigation_applied,
                        'detection_accuracy': self.anti_detection.detection_accuracy,
                    }
                }
                
                # Store with hourly keys
                hour_key = f"metrics:hourly:{int(time.time() // 3600)}"
                await self.redis.setex(
                    hour_key,
                    86400 * 7,  # Keep for 7 days
                    json.dumps(metrics_data)
                )
                
                await asyncio.sleep(3600)  # Update every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error persisting metrics", error=str(e))
                await asyncio.sleep(3600)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        now = time.time()
        
        # Calculate recent throughput
        recent_messages = sum(
            1 for ts in self._message_timestamps
            if now - ts <= 60
        )
        
        # Calculate recent error rate
        recent_errors = sum(
            1 for ts in self._error_timestamps
            if now - ts <= 300
        )
        
        return {
            'timestamp': now,
            'uptime': now - self._start_time,
            'performance': {
                'cpu_usage': self.performance.cpu_usage,
                'memory_usage': self.performance.memory_usage,
                'disk_usage': self.performance.disk_usage,
                'concurrent_connections': self.performance.concurrent_connections,
                'peak_connections': self.performance.peak_concurrent_connections,
                'avg_response_time': (
                    sum(self.performance.response_times) / 
                    max(1, len(self.performance.response_times))
                ),
                'recent_throughput_per_minute': recent_messages,
                'recent_error_count': recent_errors,
                'network_sent': self.performance.network_sent,
                'network_received': self.performance.network_received,
            },
            'behavior': {
                'active_users': self.behavior.active_users,
                'new_users_today': self.behavior.new_users_today,
                'total_messages': len(self._message_timestamps),
                'total_errors': len(self._error_timestamps),
                'top_commands': dict(list(self.behavior.commands_usage.items())[:10]),
                'hourly_activity': self.behavior.hourly_activity,
                'language_distribution': dict(list(self.behavior.language_distribution.items())[:10]),
            },
            'anti_detection': {
                'suspicious_patterns': self.anti_detection.suspicious_patterns_detected,
                'risk_mitigations': self.anti_detection.risk_mitigation_applied,
                'typing_variations': self.anti_detection.typing_pattern_variations,
                'timing_adjustments': self.anti_detection.timing_adjustments,
                'detection_accuracy': self.anti_detection.detection_accuracy,
                'evasion_success_rate': self.anti_detection.evasion_success_rate,
            }
        }
    
    async def get_historical_metrics(
        self,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        try:
            historical_data = []
            current_hour = int(time.time() // 3600)
            
            for i in range(hours_back):
                hour_key = f"metrics:hourly:{current_hour - i}"
                data = await self.redis.get(hour_key)
                
                if data:
                    try:
                        historical_data.append(json.loads(data))
                    except json.JSONDecodeError:
                        continue
            
            return historical_data
            
        except Exception as e:
            logger.error("Failed to get historical metrics", error=str(e))
            return []
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def cleanup(self) -> None:
        """Clean up metrics system."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._metrics_tasks:
                if not task.done():
                    task.cancel()
            
            if self._metrics_tasks:
                await asyncio.gather(*self._metrics_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis:
                await self.redis.close()
            
            logger.info("Telegram metrics cleanup completed")
            
        except Exception as e:
            logger.error("Error during metrics cleanup", error=str(e))
    
    async def _check_and_cleanup_memory(self) -> None:
        """Check memory usage and cleanup if necessary."""
        try:
            now = time.time()
            
            # Only check memory periodically to avoid performance impact
            if now - self._last_cleanup < self._cleanup_interval:
                return
            
            self._last_cleanup = now
            
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._max_memory_mb:
                logger.warning(f"High memory usage detected: {memory_mb:.1f}MB, cleaning up")
                await self._cleanup_memory()
            
        except Exception as e:
            logger.error("Error checking memory usage", error=str(e))
    
    async def _cleanup_memory(self) -> None:
        """Cleanup memory-intensive data structures."""
        try:
            # Reduce message timestamps
            if len(self._message_timestamps) > 1000:
                # Keep only recent timestamps
                recent_cutoff = time.time() - 3600  # Last hour
                self._message_timestamps = deque(
                    [ts for ts in self._message_timestamps if ts > recent_cutoff],
                    maxlen=self._message_timestamps.maxlen
                )
            
            # Reduce error timestamps
            if len(self._error_timestamps) > 100:
                recent_cutoff = time.time() - 1800  # Last 30 minutes
                self._error_timestamps = deque(
                    [ts for ts in self._error_timestamps if ts > recent_cutoff],
                    maxlen=self._error_timestamps.maxlen
                )
            
            # Cleanup behavior metrics
            if len(self.behavior.messages_per_user) > 5000:
                # Keep only top 2000 most active users
                sorted_users = sorted(
                    self.behavior.messages_per_user.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:2000]
                self.behavior.messages_per_user = dict(sorted_users)
            
            # Cleanup conversation lengths to prevent unbounded growth
            if len(self.behavior.conversation_lengths) > 1000:
                self.behavior.conversation_lengths = self.behavior.conversation_lengths[-500:]
            
            # Cleanup response times
            if len(self.performance.response_times) > 500:
                # Keep only recent response times
                self.performance.response_times = deque(
                    list(self.performance.response_times)[-250:],
                    maxlen=self.performance.response_times.maxlen
                )
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error("Error during memory cleanup", error=str(e))
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'message_timestamps_count': len(self._message_timestamps),
                'error_timestamps_count': len(self._error_timestamps),
                'tracked_users_count': len(self.behavior.messages_per_user),
                'response_times_count': len(self.performance.response_times),
                'conversation_lengths_count': len(self.behavior.conversation_lengths),
                'last_cleanup': self._last_cleanup,
            }
            
        except Exception as e:
            logger.error("Error getting memory stats", error=str(e))
            return {}
            
        except Exception as e:
            logger.error("Error during metrics cleanup", error=str(e))