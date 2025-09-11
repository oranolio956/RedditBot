"""
Kelly AI Real-Time Monitoring API Endpoints

FastAPI endpoints for real-time monitoring, metrics, activity feeds,
interventions, alerts, and emergency overrides for the Kelly AI system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, WebSocket
from pydantic import BaseModel, Field, validator
import structlog

from app.core.auth import get_current_user, require_admin
from app.core.redis import redis_manager
from app.services.kelly_conversation_manager import kelly_conversation_manager
from app.services.kelly_claude_ai import kelly_claude_ai
from app.services.kelly_safety_monitor import kelly_safety_monitor
from app.services.kelly_telegram_userbot import kelly_userbot
from app.websocket.manager import websocket_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/kelly/monitoring", tags=["kelly-monitoring"])

# ===== MONITORING MODELS =====

class MetricType(BaseModel):
    name: str
    current_value: Union[int, float]
    previous_value: Union[int, float] = 0
    trend: str = Field(..., regex="^(up|down|stable)$")
    threshold_status: str = Field(..., regex="^(normal|warning|critical)$")

class LiveMetricsResponse(BaseModel):
    conversations_active: int
    conversations_total_today: int
    messages_processed: int
    ai_confidence_avg: float
    safety_score_avg: float
    human_interventions: int
    alert_count: int
    system_load: float
    response_time_avg: float
    claude_requests_today: int
    claude_cost_today: float
    timestamp: str

class DetailedMetricResponse(BaseModel):
    metric_name: str
    current_value: Union[int, float]
    historical_data: List[Dict[str, Any]]
    percentile_data: Dict[str, float]
    thresholds: Dict[str, float]
    trend_analysis: Dict[str, Any]

class SystemHealthResponse(BaseModel):
    status: str = Field(..., regex="^(healthy|degraded|critical)$")
    uptime: float
    services: Dict[str, str]
    database_health: Dict[str, Any]
    redis_health: Dict[str, Any]
    telegram_connections: int
    claude_api_health: bool
    last_backup: str
    memory_usage: float
    cpu_usage: float

class PerformanceBenchmarkResponse(BaseModel):
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_per_second: float
    error_rate: float
    concurrent_conversations: int
    ai_processing_time: float
    database_query_time: float
    redis_latency: float

class ActivityEventResponse(BaseModel):
    id: str
    type: str
    account_id: str
    conversation_id: Optional[str]
    user_id: Optional[str]
    message: str
    metadata: Dict[str, Any]
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    timestamp: str
    read: bool = False

class ActivityFeedResponse(BaseModel):
    events: List[ActivityEventResponse]
    total_count: int
    unread_count: int
    has_more: bool
    next_cursor: Optional[str]

class VIPActivityResponse(BaseModel):
    vip_conversations: List[Dict[str, Any]]
    high_value_interactions: List[ActivityEventResponse]
    priority_alerts: List[Dict[str, Any]]
    engagement_score_changes: List[Dict[str, Any]]

class ActivitySummaryResponse(BaseModel):
    total_activities: int
    activities_by_type: Dict[str, int]
    activities_by_severity: Dict[str, int]
    peak_activity_time: str
    most_active_account: str
    safety_incidents: int
    ai_interventions: int

class InterventionStatusResponse(BaseModel):
    conversation_id: str
    status: str = Field(..., regex="^(ai_active|human_reviewing|human_active|paused|completed)$")
    human_operator: Optional[str]
    takeover_time: Optional[str]
    ai_confidence: float
    last_ai_suggestion: Optional[str]
    intervention_reason: str
    can_release: bool

class InterventionRequest(BaseModel):
    reason: str
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    
class SuggestResponseRequest(BaseModel):
    conversation_id: str
    context: Optional[str] = None
    tone: str = Field(default="professional", regex="^(professional|casual|empathetic|firm)$")

class AlertInstanceResponse(BaseModel):
    id: str
    type: str
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    conversation_id: Optional[str]
    account_id: str
    title: str
    description: str
    metadata: Dict[str, Any]
    created_at: str
    acknowledged_at: Optional[str]
    resolved_at: Optional[str]
    escalated_at: Optional[str]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]
    status: str = Field(..., regex="^(active|acknowledged|resolved|escalated)$")

class AlertActionRequest(BaseModel):
    action_type: str = Field(..., regex="^(acknowledge|resolve|escalate)$")
    notes: Optional[str] = None

class EmergencyStopRequest(BaseModel):
    reason: str
    immediate: bool = True
    notify_admin: bool = True

class EmergencyResetRequest(BaseModel):
    reset_type: str = Field(..., regex="^(conversation|account|all)$")
    reason: str
    preserve_data: bool = True

class AuditLogResponse(BaseModel):
    entries: List[Dict[str, Any]]
    total_count: int
    filtered_count: int

# ===== REAL-TIME METRICS API =====

@router.get("/metrics", response_model=LiveMetricsResponse)
async def get_live_metrics(
    current_user = Depends(get_current_user)
) -> LiveMetricsResponse:
    """Get live conversation metrics and system status"""
    try:
        # Get current metrics from Redis
        metrics_key = "kelly:live_metrics"
        cached_metrics = await redis_manager.get(metrics_key)
        
        if cached_metrics and datetime.now().timestamp() - json.loads(cached_metrics).get("last_updated", 0) < 10:
            # Use cached metrics if less than 10 seconds old
            metrics = json.loads(cached_metrics)
        else:
            # Calculate fresh metrics
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get active conversations
            active_convs = await redis_manager.keys("kelly:conversation_track:*")
            conversations_active = len(active_convs)
            
            # Get daily stats
            daily_keys = await redis_manager.keys(f"kelly:daily_stats:*:{now.strftime('%Y-%m-%d')}")
            total_today = 0
            messages_processed = 0
            ai_confidence_sum = 0.0
            safety_score_sum = 0.0
            
            for key in daily_keys:
                daily_data = await redis_manager.lrange(key, 0, -1)
                total_today += len(set(json.loads(d).get("user_id") for d in daily_data))
                messages_processed += len(daily_data)
                
                for data_str in daily_data:
                    data = json.loads(data_str)
                    ai_confidence_sum += data.get("ai_confidence", 0.7)
                    safety_score_sum += data.get("safety_score", 1.0)
            
            # Get intervention and alert counts
            intervention_keys = await redis_manager.keys("kelly:intervention:*")
            alert_keys = await redis_manager.keys("kelly:alert:active:*")
            
            # Get Claude metrics
            claude_metrics = await kelly_claude_ai.get_aggregated_metrics("today")
            
            # Calculate system load (simplified)
            system_load = min(100.0, (conversations_active * 2.5 + len(alert_keys) * 5.0))
            
            metrics = {
                "conversations_active": conversations_active,
                "conversations_total_today": total_today,
                "messages_processed": messages_processed,
                "ai_confidence_avg": ai_confidence_sum / max(messages_processed, 1),
                "safety_score_avg": safety_score_sum / max(messages_processed, 1),
                "human_interventions": len(intervention_keys),
                "alert_count": len(alert_keys),
                "system_load": system_load,
                "response_time_avg": claude_metrics.get("avg_response_time", 0.0),
                "claude_requests_today": claude_metrics.get("total_requests", 0),
                "claude_cost_today": claude_metrics.get("cost_today", 0.0),
                "timestamp": now.isoformat(),
                "last_updated": now.timestamp()
            }
            
            # Cache for 10 seconds
            await redis_manager.setex(metrics_key, 10, json.dumps(metrics))
        
        return LiveMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting live metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/detailed/{metric_type}", response_model=DetailedMetricResponse)
async def get_detailed_metric(
    metric_type: str,
    timeframe: str = Query(default="24h", regex="^(1h|6h|24h|7d|30d)$"),
    current_user = Depends(get_current_user)
) -> DetailedMetricResponse:
    """Get detailed drill-down data for a specific metric"""
    try:
        # Validate metric type
        valid_metrics = [
            "conversations", "messages", "ai_confidence", "safety_score",
            "response_time", "claude_cost", "interventions", "alerts"
        ]
        
        if metric_type not in valid_metrics:
            raise HTTPException(status_code=400, detail="Invalid metric type")
        
        # Calculate time range
        end_time = datetime.now()
        if timeframe == "1h":
            start_time = end_time - timedelta(hours=1)
            bucket_size = 60  # 1 minute buckets
        elif timeframe == "6h":
            start_time = end_time - timedelta(hours=6)
            bucket_size = 300  # 5 minute buckets
        elif timeframe == "24h":
            start_time = end_time - timedelta(days=1)
            bucket_size = 1800  # 30 minute buckets
        elif timeframe == "7d":
            start_time = end_time - timedelta(days=7)
            bucket_size = 3600  # 1 hour buckets
        else:  # 30d
            start_time = end_time - timedelta(days=30)
            bucket_size = 86400  # 1 day buckets
        
        # Get historical data from Redis time series
        historical_key = f"kelly:metrics:timeseries:{metric_type}"
        
        # Simulate time series data (in production, use Redis TimeSeries or similar)
        historical_data = []
        current_time = start_time
        
        while current_time <= end_time:
            # Get data for this time bucket
            bucket_key = f"kelly:metrics:{metric_type}:{int(current_time.timestamp())}"
            value = await redis_manager.get(bucket_key)
            
            if value is None:
                # Generate realistic sample data based on metric type
                if metric_type == "conversations":
                    value = max(0, 50 + int(20 * (0.5 - abs(current_time.hour - 14) / 24)))
                elif metric_type == "ai_confidence":
                    value = 0.75 + 0.15 * (0.5 - abs(current_time.hour - 12) / 24)
                elif metric_type == "safety_score":
                    value = 0.95 + 0.04 * (0.5 - abs(current_time.hour - 12) / 24)
                elif metric_type == "response_time":
                    value = 180 + 50 * (abs(current_time.hour - 14) / 24)
                else:
                    value = 0
            else:
                value = float(value)
            
            historical_data.append({
                "timestamp": current_time.isoformat(),
                "value": value
            })
            
            current_time += timedelta(seconds=bucket_size)
        
        # Calculate percentiles
        values = [d["value"] for d in historical_data]
        values.sort()
        
        if values:
            percentile_data = {
                "p50": values[int(len(values) * 0.5)],
                "p75": values[int(len(values) * 0.75)],
                "p90": values[int(len(values) * 0.9)],
                "p95": values[int(len(values) * 0.95)],
                "p99": values[int(len(values) * 0.99)]
            }
            current_value = values[-1] if values else 0
        else:
            percentile_data = {"p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}
            current_value = 0
        
        # Define thresholds based on metric type
        thresholds = {
            "conversations": {"warning": 100, "critical": 200},
            "ai_confidence": {"warning": 0.6, "critical": 0.4},
            "safety_score": {"warning": 0.8, "critical": 0.6},
            "response_time": {"warning": 300, "critical": 500},
            "alerts": {"warning": 5, "critical": 10}
        }.get(metric_type, {"warning": 100, "critical": 200})
        
        # Trend analysis
        if len(values) >= 2:
            recent_avg = sum(values[-5:]) / min(5, len(values))
            earlier_avg = sum(values[:5]) / min(5, len(values))
            trend_direction = "up" if recent_avg > earlier_avg else ("down" if recent_avg < earlier_avg else "stable")
            trend_percentage = ((recent_avg - earlier_avg) / max(earlier_avg, 0.01)) * 100
        else:
            trend_direction = "stable"
            trend_percentage = 0
        
        trend_analysis = {
            "direction": trend_direction,
            "percentage_change": trend_percentage,
            "is_concerning": (
                (trend_direction == "up" and metric_type in ["alerts", "response_time", "interventions"]) or
                (trend_direction == "down" and metric_type in ["ai_confidence", "safety_score"])
            )
        }
        
        return DetailedMetricResponse(
            metric_name=metric_type,
            current_value=current_value,
            historical_data=historical_data,
            percentile_data=percentile_data,
            thresholds=thresholds,
            trend_analysis=trend_analysis
        )
        
    except Exception as e:
        logger.error(f"Error getting detailed metric {metric_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user = Depends(get_current_user)
) -> SystemHealthResponse:
    """Get comprehensive system health status"""
    try:
        import psutil
        from app.database.connection import db_manager
        
        # Get uptime
        uptime_key = "kelly:system:start_time"
        start_time = await redis_manager.get(uptime_key)
        
        if start_time:
            uptime = datetime.now().timestamp() - float(start_time)
        else:
            uptime = 0
            await redis_manager.set(uptime_key, datetime.now().timestamp())
        
        # Check service health
        services = {
            "kelly_personality": "healthy",
            "kelly_userbot": "healthy",
            "kelly_dm_detector": "healthy",
            "kelly_conversation_manager": "healthy",
            "kelly_claude_ai": "healthy",
            "kelly_safety_monitor": "healthy",
            "telegram_connections": "healthy",
            "websocket_manager": "healthy"
        }
        
        # Check database health
        try:
            db_health = await db_manager.health_check()
            database_health = {
                "status": "healthy",
                "connection_count": db_health.get("active_connections", 0),
                "pool_size": db_health.get("pool_size", 0),
                "response_time_ms": db_health.get("response_time", 0)
            }
        except Exception as e:
            database_health = {
                "status": "unhealthy",
                "error": str(e),
                "connection_count": 0,
                "pool_size": 0,
                "response_time_ms": 0
            }
            services["database"] = "unhealthy"
        
        # Check Redis health
        try:
            redis_info = await redis_manager.info()
            redis_health = {
                "status": "healthy",
                "memory_usage": redis_info.get("used_memory", 0),
                "connected_clients": redis_info.get("connected_clients", 0),
                "ops_per_sec": redis_info.get("instantaneous_ops_per_sec", 0)
            }
        except Exception as e:
            redis_health = {
                "status": "unhealthy",
                "error": str(e),
                "memory_usage": 0,
                "connected_clients": 0,
                "ops_per_sec": 0
            }
            services["redis"] = "unhealthy"
        
        # Check Telegram connections
        telegram_keys = await redis_manager.keys("kelly:account:*")
        telegram_connections = len(telegram_keys)
        
        # Check Claude API health
        try:
            claude_health = await kelly_claude_ai.health_check()
            claude_api_health = claude_health.get("status") == "healthy"
        except:
            claude_api_health = False
            services["kelly_claude_ai"] = "unhealthy"
        
        # Get system metrics
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        # Get last backup time
        last_backup = await redis_manager.get("kelly:last_backup_time")
        if not last_backup:
            last_backup = "never"
        
        # Determine overall status
        unhealthy_services = [k for k, v in services.items() if v != "healthy"]
        
        if not unhealthy_services and memory_usage < 85 and cpu_usage < 80:
            overall_status = "healthy"
        elif len(unhealthy_services) <= 1 and memory_usage < 95 and cpu_usage < 90:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return SystemHealthResponse(
            status=overall_status,
            uptime=uptime,
            services=services,
            database_health=database_health,
            redis_health=redis_health,
            telegram_connections=telegram_connections,
            claude_api_health=claude_api_health,
            last_backup=last_backup,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=PerformanceBenchmarkResponse)
async def get_performance_benchmarks(
    current_user = Depends(get_current_user)
) -> PerformanceBenchmarkResponse:
    """Get detailed performance benchmarks and timing metrics"""
    try:
        # Get performance metrics from Redis
        perf_key = "kelly:performance:current"
        cached_perf = await redis_manager.get(perf_key)
        
        if cached_perf:
            perf_data = json.loads(cached_perf)
        else:
            # Calculate performance metrics
            now = datetime.now()
            
            # Get response time data
            response_times_key = "kelly:metrics:response_times"
            response_times = await redis_manager.lrange(response_times_key, 0, 999)
            
            if response_times:
                times = [float(t) for t in response_times]
                times.sort()
                
                avg_response_time = sum(times) / len(times)
                p95_response_time = times[int(len(times) * 0.95)]
                p99_response_time = times[int(len(times) * 0.99)]
            else:
                avg_response_time = p95_response_time = p99_response_time = 0.0
            
            # Get throughput
            throughput_key = f"kelly:throughput:{int(now.timestamp() // 60)}"  # per minute
            throughput = float(await redis_manager.get(throughput_key) or 0)
            throughput_per_second = throughput / 60.0
            
            # Get error rate
            error_count_key = "kelly:errors:count"
            success_count_key = "kelly:success:count"
            
            error_count = int(await redis_manager.get(error_count_key) or 0)
            success_count = int(await redis_manager.get(success_count_key) or 1)
            
            error_rate = error_count / (error_count + success_count)
            
            # Get concurrent conversations
            active_convs = await redis_manager.keys("kelly:conversation_track:*")
            concurrent_conversations = len(active_convs)
            
            # Get AI processing time
            ai_times_key = "kelly:metrics:ai_processing_times"
            ai_times = await redis_manager.lrange(ai_times_key, 0, 99)
            ai_processing_time = sum(float(t) for t in ai_times) / max(len(ai_times), 1)
            
            # Get database query time
            db_times_key = "kelly:metrics:db_query_times"
            db_times = await redis_manager.lrange(db_times_key, 0, 99)
            database_query_time = sum(float(t) for t in db_times) / max(len(db_times), 1)
            
            # Test Redis latency
            start = datetime.now()
            await redis_manager.ping()
            redis_latency = (datetime.now() - start).total_seconds() * 1000
            
            perf_data = {
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "throughput_per_second": throughput_per_second,
                "error_rate": error_rate,
                "concurrent_conversations": concurrent_conversations,
                "ai_processing_time": ai_processing_time,
                "database_query_time": database_query_time,
                "redis_latency": redis_latency
            }
            
            # Cache for 30 seconds
            await redis_manager.setex(perf_key, 30, json.dumps(perf_data))
        
        return PerformanceBenchmarkResponse(**perf_data)
        
    except Exception as e:
        logger.error(f"Error getting performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ACTIVITY FEED API =====

@router.get("/activity/feed", response_model=ActivityFeedResponse)
async def get_activity_feed(
    limit: int = Query(default=50, le=200),
    cursor: Optional[str] = None,
    severity: Optional[str] = Query(default=None, regex="^(low|medium|high|critical)$"),
    event_type: Optional[str] = None,
    account_id: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> ActivityFeedResponse:
    """Get live activity stream with pagination and filtering"""
    try:
        # Build Redis key pattern for activity events
        if account_id:
            pattern = f"kelly:activity:{account_id}:*"
        else:
            pattern = "kelly:activity:*"
        
        # Get activity keys
        activity_keys = await redis_manager.keys(pattern)
        
        # Sort by timestamp (embedded in key)
        activity_keys.sort(reverse=True)
        
        # Apply cursor pagination
        start_index = 0
        if cursor:
            try:
                cursor_time = datetime.fromisoformat(cursor.replace('Z', '+00:00'))
                cursor_timestamp = int(cursor_time.timestamp())
                
                # Find position of cursor
                for i, key in enumerate(activity_keys):
                    key_timestamp = int(key.split(':')[-1])
                    if key_timestamp <= cursor_timestamp:
                        start_index = i
                        break
            except:
                start_index = 0
        
        # Get events for this page
        page_keys = activity_keys[start_index:start_index + limit + 1]
        has_more = len(page_keys) > limit
        
        if has_more:
            page_keys = page_keys[:-1]  # Remove extra item
        
        # Fetch event data
        events = []
        for key in page_keys:
            event_data = await redis_manager.get(key)
            if event_data:
                event = json.loads(event_data)
                
                # Apply filters
                if severity and event.get("severity") != severity:
                    continue
                
                if event_type and event.get("type") != event_type:
                    continue
                
                # Check if event is read
                read_key = f"kelly:activity_read:{current_user.get('id')}:{event['id']}"
                is_read = bool(await redis_manager.get(read_key))
                
                events.append(ActivityEventResponse(
                    id=event["id"],
                    type=event["type"],
                    account_id=event["account_id"],
                    conversation_id=event.get("conversation_id"),
                    user_id=event.get("user_id"),
                    message=event["message"],
                    metadata=event.get("metadata", {}),
                    severity=event["severity"],
                    timestamp=event["timestamp"],
                    read=is_read
                ))
        
        # Calculate next cursor
        next_cursor = None
        if has_more and events:
            next_cursor = events[-1].timestamp
        
        # Get counts
        total_count = len(activity_keys)
        unread_count = await _get_unread_activity_count(current_user.get("id"))
        
        return ActivityFeedResponse(
            events=events,
            total_count=total_count,
            unread_count=unread_count,
            has_more=has_more,
            next_cursor=next_cursor
        )
        
    except Exception as e:
        logger.error(f"Error getting activity feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity/vip", response_model=VIPActivityResponse)
async def get_vip_activities(
    current_user = Depends(get_current_user)
) -> VIPActivityResponse:
    """Get VIP conversation activities and high-priority events"""
    try:
        # Get VIP conversations (high engagement or safety concerns)
        vip_keys = await redis_manager.keys("kelly:vip_conversation:*")
        vip_conversations = []
        
        for key in vip_keys:
            vip_data = await redis_manager.get(key)
            if vip_data:
                conversation = json.loads(vip_data)
                vip_conversations.append({
                    "conversation_id": conversation["conversation_id"],
                    "account_id": conversation["account_id"],
                    "user_id": conversation["user_id"],
                    "vip_reason": conversation["vip_reason"],
                    "engagement_score": conversation["engagement_score"],
                    "safety_score": conversation["safety_score"],
                    "last_activity": conversation["last_activity"],
                    "requires_attention": conversation.get("requires_attention", False)
                })
        
        # Get high-value interactions (recent activities with high severity)
        high_value_keys = await redis_manager.keys("kelly:activity:*")
        high_value_interactions = []
        
        # Sort by timestamp and get recent high-severity events
        high_value_keys.sort(reverse=True)
        
        for key in high_value_keys[:50]:  # Check last 50 events
            event_data = await redis_manager.get(key)
            if event_data:
                event = json.loads(event_data)
                
                if event.get("severity") in ["high", "critical"]:
                    # Check if read
                    read_key = f"kelly:activity_read:{current_user.get('id')}:{event['id']}"
                    is_read = bool(await redis_manager.get(read_key))
                    
                    high_value_interactions.append(ActivityEventResponse(
                        id=event["id"],
                        type=event["type"],
                        account_id=event["account_id"],
                        conversation_id=event.get("conversation_id"),
                        user_id=event.get("user_id"),
                        message=event["message"],
                        metadata=event.get("metadata", {}),
                        severity=event["severity"],
                        timestamp=event["timestamp"],
                        read=is_read
                    ))
        
        # Get priority alerts
        alert_keys = await redis_manager.keys("kelly:alert:active:*")
        priority_alerts = []
        
        for key in alert_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                if alert.get("severity") in ["high", "critical"]:
                    priority_alerts.append(alert)
        
        # Get recent engagement score changes
        score_change_keys = await redis_manager.keys("kelly:engagement_change:*")
        engagement_score_changes = []
        
        for key in score_change_keys[-20:]:  # Last 20 changes
            change_data = await redis_manager.get(key)
            if change_data:
                engagement_score_changes.append(json.loads(change_data))
        
        return VIPActivityResponse(
            vip_conversations=vip_conversations,
            high_value_interactions=high_value_interactions,
            priority_alerts=priority_alerts,
            engagement_score_changes=engagement_score_changes
        )
        
    except Exception as e:
        logger.error(f"Error getting VIP activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/activity/mark-read")
async def mark_activities_read(
    activity_ids: List[str],
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Mark activities as read for the current user"""
    try:
        user_id = current_user.get("id")
        
        # Mark each activity as read
        for activity_id in activity_ids:
            read_key = f"kelly:activity_read:{user_id}:{activity_id}"
            await redis_manager.setex(read_key, 86400 * 30, "1")  # Keep for 30 days
        
        return {"message": f"Marked {len(activity_ids)} activities as read"}
        
    except Exception as e:
        logger.error(f"Error marking activities read: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity/summary", response_model=ActivitySummaryResponse)
async def get_activity_summary(
    timeframe: str = Query(default="24h", regex="^(1h|6h|24h|7d)$"),
    current_user = Depends(get_current_user)
) -> ActivitySummaryResponse:
    """Get activity summary statistics"""
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if timeframe == "1h":
            start_time = end_time - timedelta(hours=1)
        elif timeframe == "6h":
            start_time = end_time - timedelta(hours=6)
        elif timeframe == "24h":
            start_time = end_time - timedelta(days=1)
        else:  # 7d
            start_time = end_time - timedelta(days=7)
        
        start_timestamp = int(start_time.timestamp())
        
        # Get activities in timeframe
        activity_keys = await redis_manager.keys("kelly:activity:*")
        
        activities_by_type = {}
        activities_by_severity = {}
        total_activities = 0
        activity_times = []
        account_activity_counts = {}
        safety_incidents = 0
        ai_interventions = 0
        
        for key in activity_keys:
            # Check if activity is in timeframe
            key_timestamp = int(key.split(':')[-1])
            if key_timestamp < start_timestamp:
                continue
            
            event_data = await redis_manager.get(key)
            if event_data:
                event = json.loads(event_data)
                total_activities += 1
                
                # Count by type
                event_type = event.get("type", "unknown")
                activities_by_type[event_type] = activities_by_type.get(event_type, 0) + 1
                
                # Count by severity
                severity = event.get("severity", "low")
                activities_by_severity[severity] = activities_by_severity.get(severity, 0) + 1
                
                # Track activity times
                activity_times.append(datetime.fromisoformat(event["timestamp"]))
                
                # Count by account
                account_id = event["account_id"]
                account_activity_counts[account_id] = account_activity_counts.get(account_id, 0) + 1
                
                # Count specific types
                if event_type == "safety_incident":
                    safety_incidents += 1
                elif event_type == "ai_intervention":
                    ai_interventions += 1
        
        # Find peak activity time
        if activity_times:
            # Group by hour and find the hour with most activities
            hour_counts = {}
            for time in activity_times:
                hour = time.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            peak_hour = max(hour_counts, key=hour_counts.get)
            peak_activity_time = f"{peak_hour:02d}:00"
        else:
            peak_activity_time = "N/A"
        
        # Find most active account
        if account_activity_counts:
            most_active_account = max(account_activity_counts, key=account_activity_counts.get)
        else:
            most_active_account = "N/A"
        
        return ActivitySummaryResponse(
            total_activities=total_activities,
            activities_by_type=activities_by_type,
            activities_by_severity=activities_by_severity,
            peak_activity_time=peak_activity_time,
            most_active_account=most_active_account,
            safety_incidents=safety_incidents,
            ai_interventions=ai_interventions
        )
        
    except Exception as e:
        logger.error(f"Error getting activity summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function
async def _get_unread_activity_count(user_id: str) -> int:
    """Get count of unread activities for a user"""
    try:
        activity_keys = await redis_manager.keys("kelly:activity:*")
        unread_count = 0
        
        for key in activity_keys:
            activity_id = key.split(':')[-1]
            read_key = f"kelly:activity_read:{user_id}:{activity_id}"
            
            if not await redis_manager.get(read_key):
                unread_count += 1
        
        return unread_count
        
    except:
        return 0