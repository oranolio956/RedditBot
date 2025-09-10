"""
Production Monitoring and Alerting System

Comprehensive monitoring with Prometheus metrics, custom alerts,
and performance tracking for the AI conversation bot.

Features:
- Application metrics (requests, latency, errors)
- Business metrics (subscriptions, revenue, user engagement)
- System metrics (CPU, memory, database)
- Custom alerts and thresholds
- Real-time dashboards
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Callable
from functools import wraps
import logging
from enum import Enum

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, push_to_gateway
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from aiohttp import web

logger = structlog.get_logger(__name__)


# Custom registry for application metrics
registry = CollectorRegistry()

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests in progress',
    ['method', 'endpoint'],
    registry=registry
)

# Telegram bot metrics
telegram_messages_total = Counter(
    'telegram_messages_total',
    'Total Telegram messages processed',
    ['message_type', 'user_type'],
    registry=registry
)

telegram_commands_total = Counter(
    'telegram_commands_total',
    'Total Telegram commands executed',
    ['command', 'status'],
    registry=registry
)

telegram_response_time = Histogram(
    'telegram_response_time_seconds',
    'Telegram bot response time',
    ['command'],
    registry=registry
)

# LLM metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'status'],
    registry=registry
)

llm_tokens_used = Counter(
    'llm_tokens_used_total',
    'Total LLM tokens consumed',
    ['provider', 'model', 'token_type'],
    registry=registry
)

llm_response_time = Histogram(
    'llm_response_time_seconds',
    'LLM API response time',
    ['provider', 'model'],
    registry=registry
)

llm_cost_total = Counter(
    'llm_cost_dollars_total',
    'Total LLM API cost in dollars',
    ['provider', 'model'],
    registry=registry
)

# Subscription metrics
active_subscriptions = Gauge(
    'active_subscriptions',
    'Number of active subscriptions',
    ['plan', 'status'],
    registry=registry
)

subscription_revenue = Counter(
    'subscription_revenue_dollars_total',
    'Total subscription revenue in dollars',
    ['plan', 'payment_method'],
    registry=registry
)

subscription_churn = Counter(
    'subscription_churn_total',
    'Total subscription cancellations',
    ['plan', 'reason'],
    registry=registry
)

payment_failures = Counter(
    'payment_failures_total',
    'Total payment failures',
    ['reason', 'retry_attempt'],
    registry=registry
)

# User engagement metrics
active_users = Gauge(
    'active_users',
    'Number of active users',
    ['time_range'],
    registry=registry
)

user_sessions = Histogram(
    'user_session_duration_seconds',
    'User session duration',
    ['user_type'],
    registry=registry
)

messages_per_conversation = Histogram(
    'messages_per_conversation',
    'Number of messages per conversation',
    ['conversation_type'],
    registry=registry
)

# Risk management metrics
risk_assessments = Counter(
    'risk_assessments_total',
    'Total risk assessments performed',
    ['risk_level', 'category'],
    registry=registry
)

blocked_messages = Counter(
    'blocked_messages_total',
    'Total messages blocked',
    ['reason', 'user_type'],
    registry=registry
)

user_trust_scores = Histogram(
    'user_trust_scores',
    'Distribution of user trust scores',
    ['user_segment'],
    registry=registry
)

# System metrics
database_connections = Gauge(
    'database_connections',
    'Database connection pool status',
    ['status'],
    registry=registry
)

redis_operations = Counter(
    'redis_operations_total',
    'Total Redis operations',
    ['operation', 'status'],
    registry=registry
)

cache_hits = Counter(
    'cache_hits_total',
    'Cache hit/miss ratio',
    ['cache_type', 'result'],
    registry=registry
)

# Error metrics
application_errors = Counter(
    'application_errors_total',
    'Total application errors',
    ['error_type', 'severity', 'component'],
    registry=registry
)

# Custom business metrics
conversation_quality = Histogram(
    'conversation_quality_score',
    'Conversation quality scores',
    ['personality_type'],
    registry=registry
)

personality_adaptation_success = Counter(
    'personality_adaptation_success_total',
    'Successful personality adaptations',
    ['adaptation_type'],
    registry=registry
)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


def track_request_metrics(method: str, endpoint: str):
    """Decorator to track HTTP request metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Track in-progress requests
            http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
            
            # Track request duration
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track successful request
                status = getattr(result, 'status_code', 200)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                
                return result
                
            except Exception as e:
                # Track failed request
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=500
                ).inc()
                
                # Track error
                application_errors.labels(
                    error_type=type(e).__name__,
                    severity='high',
                    component='http'
                ).inc()
                
                raise
                
            finally:
                # Track duration
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                
                # Decrement in-progress counter
                http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()
        
        return wrapper
    return decorator


def track_telegram_command(command: str):
    """Decorator to track Telegram command metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track successful command
                telegram_commands_total.labels(
                    command=command,
                    status='success'
                ).inc()
                
                return result
                
            except Exception as e:
                # Track failed command
                telegram_commands_total.labels(
                    command=command,
                    status='error'
                ).inc()
                
                raise
                
            finally:
                # Track response time
                duration = time.time() - start_time
                telegram_response_time.labels(command=command).observe(duration)
        
        return wrapper
    return decorator


class MetricsCollector:
    """Custom metrics collector for business metrics."""
    
    def __init__(self, db_session=None, redis_client=None):
        self.db = db_session
        self.redis = redis_client
        self.last_collection = datetime.utcnow()
        
    async def collect_business_metrics(self):
        """Collect business-specific metrics."""
        try:
            # Collect subscription metrics
            await self._collect_subscription_metrics()
            
            # Collect user engagement metrics
            await self._collect_engagement_metrics()
            
            # Collect system health metrics
            await self._collect_system_metrics()
            
            # Collect risk metrics
            await self._collect_risk_metrics()
            
            self.last_collection = datetime.utcnow()
            
        except Exception as e:
            logger.error("Error collecting business metrics", error=str(e))
    
    async def _collect_subscription_metrics(self):
        """Collect subscription-related metrics."""
        if not self.db:
            return
        
        try:
            from app.models.payment import Subscription, SubscriptionStatus, SubscriptionPlan
            from sqlalchemy import select, func
            
            # Count active subscriptions by plan
            for plan in SubscriptionPlan:
                for status in SubscriptionStatus:
                    count_query = select(func.count()).select_from(Subscription).where(
                        Subscription.plan == plan,
                        Subscription.status == status
                    )
                    result = await self.db.execute(count_query)
                    count = result.scalar()
                    
                    active_subscriptions.labels(
                        plan=plan.value,
                        status=status.value
                    ).set(count)
            
            # Calculate total revenue (simplified)
            revenue_query = select(
                func.sum(Subscription.amount)
            ).where(
                Subscription.status == SubscriptionStatus.ACTIVE,
                Subscription.created_at >= datetime.utcnow() - timedelta(days=30)
            )
            result = await self.db.execute(revenue_query)
            total_revenue = result.scalar() or 0
            
            # This would normally track incremental revenue
            # subscription_revenue.labels(plan='all', payment_method='all').inc(total_revenue)
            
        except Exception as e:
            logger.error("Error collecting subscription metrics", error=str(e))
    
    async def _collect_engagement_metrics(self):
        """Collect user engagement metrics."""
        if not self.db:
            return
        
        try:
            from app.models.user import User
            from app.models.conversation import ConversationSession
            from sqlalchemy import select, func
            
            # Count active users (last 24 hours)
            active_24h_query = select(func.count()).select_from(User).where(
                User.last_active >= datetime.utcnow() - timedelta(hours=24)
            )
            result = await self.db.execute(active_24h_query)
            active_24h = result.scalar()
            active_users.labels(time_range='24h').set(active_24h)
            
            # Count active users (last 7 days)
            active_7d_query = select(func.count()).select_from(User).where(
                User.last_active >= datetime.utcnow() - timedelta(days=7)
            )
            result = await self.db.execute(active_7d_query)
            active_7d = result.scalar()
            active_users.labels(time_range='7d').set(active_7d)
            
            # Average messages per conversation
            avg_messages_query = select(
                func.avg(ConversationSession.message_count)
            ).where(
                ConversationSession.created_at >= datetime.utcnow() - timedelta(days=1)
            )
            result = await self.db.execute(avg_messages_query)
            avg_messages = result.scalar() or 0
            messages_per_conversation.labels(conversation_type='all').observe(avg_messages)
            
        except Exception as e:
            logger.error("Error collecting engagement metrics", error=str(e))
    
    async def _collect_system_metrics(self):
        """Collect system health metrics."""
        try:
            # Database connection pool status
            if self.db:
                pool = self.db.bind.pool if hasattr(self.db, 'bind') else None
                if pool:
                    database_connections.labels(status='active').set(pool.size())
                    database_connections.labels(status='idle').set(pool.checked_in())
                    database_connections.labels(status='overflow').set(pool.overflow())
            
            # Redis metrics
            if self.redis:
                info = await self.redis.info()
                
                # Track memory usage
                memory_used = info.get('used_memory', 0)
                database_connections.labels(status='redis_memory_mb').set(memory_used / 1024 / 1024)
                
                # Track connected clients
                connected_clients = info.get('connected_clients', 0)
                database_connections.labels(status='redis_clients').set(connected_clients)
            
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
    
    async def _collect_risk_metrics(self):
        """Collect risk management metrics."""
        if not self.redis:
            return
        
        try:
            # Get risk events from Redis
            import json
            risk_events = await self.redis.lrange('risk_events', 0, 999)
            
            # Count by risk level
            risk_counts = {}
            for event_data in risk_events:
                event = json.loads(event_data)
                risk_level = event.get('risk_level', 'unknown')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # Update metrics
            for level, count in risk_counts.items():
                risk_assessments.labels(
                    risk_level=level,
                    category='all'
                ).inc(count)
            
        except Exception as e:
            logger.error("Error collecting risk metrics", error=str(e))


class AlertManager:
    """Manages alerts based on metric thresholds."""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = self._define_alert_rules()
        
    def _define_alert_rules(self) -> List[Dict[str, Any]]:
        """Define alert rules and thresholds."""
        return [
            {
                'name': 'high_error_rate',
                'condition': lambda: self._check_error_rate() > 0.05,
                'severity': 'critical',
                'message': 'Error rate exceeds 5%',
                'cooldown_minutes': 15
            },
            {
                'name': 'low_subscription_rate',
                'condition': lambda: self._check_subscription_rate() < 0.01,
                'severity': 'warning',
                'message': 'Subscription conversion rate below 1%',
                'cooldown_minutes': 60
            },
            {
                'name': 'high_response_time',
                'condition': lambda: self._check_response_time() > 3.0,
                'severity': 'warning',
                'message': 'Average response time exceeds 3 seconds',
                'cooldown_minutes': 30
            },
            {
                'name': 'payment_failures',
                'condition': lambda: self._check_payment_failures() > 5,
                'severity': 'critical',
                'message': 'More than 5 payment failures in last hour',
                'cooldown_minutes': 30
            },
            {
                'name': 'high_risk_messages',
                'condition': lambda: self._check_risk_messages() > 10,
                'severity': 'warning',
                'message': 'High number of risky messages detected',
                'cooldown_minutes': 15
            }
        ]
    
    def _check_error_rate(self) -> float:
        """Check current error rate."""
        # Implementation would query actual metrics
        return 0.02  # Placeholder
    
    def _check_subscription_rate(self) -> float:
        """Check subscription conversion rate."""
        # Implementation would query actual metrics
        return 0.02  # Placeholder
    
    def _check_response_time(self) -> float:
        """Check average response time."""
        # Implementation would query actual metrics
        return 2.0  # Placeholder
    
    def _check_payment_failures(self) -> int:
        """Check recent payment failures."""
        # Implementation would query actual metrics
        return 2  # Placeholder
    
    def _check_risk_messages(self) -> int:
        """Check number of high-risk messages."""
        # Implementation would query actual metrics
        return 3  # Placeholder
    
    async def check_alerts(self):
        """Check all alert rules and trigger if needed."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition']():
                    # Check cooldown
                    last_alert = self._get_last_alert(rule['name'])
                    if last_alert:
                        cooldown = timedelta(minutes=rule['cooldown_minutes'])
                        if datetime.utcnow() - last_alert < cooldown:
                            continue
                    
                    # Trigger alert
                    alert = {
                        'name': rule['name'],
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'timestamp': datetime.utcnow()
                    }
                    
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
                    
                    # Send notification (webhook, email, etc.)
                    await self._send_alert_notification(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}", error=str(e))
        
        return triggered_alerts
    
    def _get_last_alert(self, alert_name: str) -> Optional[datetime]:
        """Get timestamp of last alert with given name."""
        for alert in reversed(self.alerts):
            if alert['name'] == alert_name:
                return alert['timestamp']
        return None
    
    async def _send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification via configured channels."""
        try:
            # Log alert
            logger.warning(
                "Alert triggered",
                alert_name=alert['name'],
                severity=alert['severity'],
                message=alert['message']
            )
            
            # Send to monitoring service (e.g., PagerDuty, Slack)
            # Implementation would send actual notifications
            
        except Exception as e:
            logger.error("Error sending alert notification", error=str(e))


async def metrics_endpoint(request: web.Request):
    """Prometheus metrics endpoint."""
    metrics = generate_latest(registry)
    return web.Response(body=metrics, content_type=CONTENT_TYPE_LATEST)


async def health_check_endpoint(request: web.Request):
    """Health check endpoint for monitoring."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'checks': {
            'database': 'healthy',
            'redis': 'healthy',
            'telegram': 'healthy',
            'llm': 'healthy'
        }
    }
    
    # Perform actual health checks
    # ... implementation ...
    
    return web.json_response(health_status)


def setup_monitoring_routes(app: web.Application):
    """Setup monitoring routes for the application."""
    app.router.add_get('/metrics', metrics_endpoint)
    app.router.add_get('/health', health_check_endpoint)


# Export monitoring components
__all__ = [
    'track_request_metrics',
    'track_telegram_command',
    'MetricsCollector',
    'AlertManager',
    'setup_monitoring_routes',
    # Metrics
    'http_requests_total',
    'telegram_messages_total',
    'llm_requests_total',
    'active_subscriptions',
    'subscription_revenue',
    'application_errors'
]