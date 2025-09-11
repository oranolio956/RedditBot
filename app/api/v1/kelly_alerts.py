"""
Kelly AI Alert Management API Endpoints

FastAPI endpoints for managing alerts, acknowledgments, resolutions,
and escalations in the Kelly AI monitoring system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
import structlog

from app.core.auth import get_current_user, require_admin
from app.core.redis import redis_manager
from app.services.kelly_safety_monitor import kelly_safety_monitor
from app.services.kelly_conversation_manager import kelly_conversation_manager
from app.websocket.manager import websocket_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/kelly/alerts", tags=["kelly-alerts"])

# ===== ALERT MODELS =====

class AlertInstanceResponse(BaseModel):
    id: str
    type: str = Field(..., regex="^(safety|performance|security|system|conversation)$")
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    status: str = Field(..., regex="^(active|acknowledged|investigating|resolved|escalated)$")
    conversation_id: Optional[str]
    account_id: Optional[str]
    title: str
    description: str
    metadata: Dict[str, Any]
    created_at: str
    acknowledged_at: Optional[str]
    resolved_at: Optional[str]
    escalated_at: Optional[str]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]
    escalated_by: Optional[str]
    resolution_notes: Optional[str]
    escalation_reason: Optional[str]
    auto_generated: bool
    requires_human_review: bool
    impact_score: float

class AlertActionRequest(BaseModel):
    action_type: str = Field(..., regex="^(acknowledge|investigate|resolve|escalate)$")
    notes: Optional[str] = None
    escalation_level: Optional[str] = Field(None, regex="^(supervisor|admin|emergency)$")
    resolution_type: Optional[str] = Field(None, regex="^(fixed|false_positive|deferred|escalated)$")

class AlertFilterRequest(BaseModel):
    severity: Optional[str] = Field(None, regex="^(low|medium|high|critical)$")
    type: Optional[str] = Field(None, regex="^(safety|performance|security|system|conversation)$")
    status: Optional[str] = Field(None, regex="^(active|acknowledged|investigating|resolved|escalated)$")
    account_id: Optional[str] = None
    conversation_id: Optional[str] = None
    time_range: str = Field(default="24h", regex="^(1h|6h|24h|7d|30d)$")
    assigned_to: Optional[str] = None

class AlertSummaryResponse(BaseModel):
    total_active: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    avg_resolution_time: float
    escalation_rate: float
    false_positive_rate: float
    top_alert_sources: List[Dict[str, Any]]

class AlertTrendResponse(BaseModel):
    timeframe: str
    data_points: List[Dict[str, Any]]
    trend_direction: str
    percentage_change: float
    peak_times: List[str]
    patterns: List[Dict[str, Any]]

class EscalationRequest(BaseModel):
    escalation_level: str = Field(..., regex="^(supervisor|admin|emergency)$")
    reason: str
    urgency: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    notify_immediately: bool = True
    include_context: bool = True

# ===== ALERT ENDPOINTS =====

@router.get("/active", response_model=List[AlertInstanceResponse])
async def get_active_alerts(
    limit: int = Query(default=50, le=200),
    severity: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    alert_type: Optional[str] = Query(None, regex="^(safety|performance|security|system|conversation)$"),
    account_id: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> List[AlertInstanceResponse]:
    """Get all active alerts with optional filtering"""
    try:
        # Build key pattern based on filters
        if account_id:
            pattern = f"kelly:alert:active:{account_id}:*"
        else:
            pattern = "kelly:alert:active:*"
        
        # Get alert keys
        alert_keys = await redis_manager.keys(pattern)
        
        # Sort by creation time (most recent first)
        alert_keys.sort(reverse=True)
        
        alerts = []
        for key in alert_keys[:limit * 2]:  # Get extra to account for filtering
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                
                # Apply filters
                if severity and alert.get("severity") != severity:
                    continue
                
                if alert_type and alert.get("type") != alert_type:
                    continue
                
                # Check user permissions for account-specific alerts
                if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
                    continue
                
                alerts.append(AlertInstanceResponse(**alert))
                
                if len(alerts) >= limit:
                    break
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/acknowledge/{alert_id}")
async def acknowledge_alert(
    alert_id: str,
    request: AlertActionRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Acknowledge an alert"""
    try:
        # Find the alert
        alert_key = await _find_alert_key(alert_id)
        if not alert_key:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert_data = await redis_manager.get(alert_key)
        if not alert_data:
            raise HTTPException(status_code=404, detail="Alert data not found")
        
        alert = json.loads(alert_data)
        
        # Check permissions
        if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
            raise HTTPException(status_code=403, detail="Access denied to this alert")
        
        # Prevent double acknowledgment
        if alert.get("status") in ["acknowledged", "investigating", "resolved"]:
            raise HTTPException(status_code=400, detail=f"Alert already {alert['status']}")
        
        # Update alert
        now = datetime.now()
        alert.update({
            "status": "acknowledged",
            "acknowledged_at": now.isoformat(),
            "acknowledged_by": current_user.get("id"),
            "acknowledgment_notes": request.notes
        })
        
        # Store updated alert
        await redis_manager.setex(alert_key, 86400 * 30, json.dumps(alert))  # 30 days
        
        # Log action
        background_tasks.add_task(
            _log_alert_action,
            alert_id,
            "acknowledged",
            current_user.get("id"),
            request.notes
        )
        
        # Broadcast update
        background_tasks.add_task(
            _broadcast_alert_update,
            alert_id,
            "acknowledged",
            {
                "acknowledged_by": current_user.get("name"),
                "notes": request.notes,
                "timestamp": now.isoformat()
            }
        )
        
        # Start auto-investigation if it's a critical alert
        if alert.get("severity") == "critical":
            background_tasks.add_task(
                _auto_investigate_critical_alert,
                alert_id,
                alert
            )
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resolve/{alert_id}")
async def resolve_alert(
    alert_id: str,
    request: AlertActionRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Resolve an alert"""
    try:
        # Find the alert
        alert_key = await _find_alert_key(alert_id)
        if not alert_key:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert_data = await redis_manager.get(alert_key)
        if not alert_data:
            raise HTTPException(status_code=404, detail="Alert data not found")
        
        alert = json.loads(alert_data)
        
        # Check permissions
        if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
            raise HTTPException(status_code=403, detail="Access denied to this alert")
        
        # Prevent resolving already resolved alerts
        if alert.get("status") == "resolved":
            raise HTTPException(status_code=400, detail="Alert already resolved")
        
        # Require resolution type and notes for high/critical alerts
        if alert.get("severity") in ["high", "critical"]:
            if not request.resolution_type or not request.notes:
                raise HTTPException(
                    status_code=400, 
                    detail="Resolution type and notes required for high/critical alerts"
                )
        
        # Calculate resolution time
        created_at = datetime.fromisoformat(alert["created_at"])
        resolved_at = datetime.now()
        resolution_time = (resolved_at - created_at).total_seconds()
        
        # Update alert
        alert.update({
            "status": "resolved",
            "resolved_at": resolved_at.isoformat(),
            "resolved_by": current_user.get("id"),
            "resolution_notes": request.notes,
            "resolution_type": request.resolution_type,
            "resolution_time_seconds": resolution_time
        })
        
        # Move to resolved alerts
        resolved_key = alert_key.replace(":active:", ":resolved:")
        await redis_manager.setex(resolved_key, 86400 * 90, json.dumps(alert))  # 90 days
        
        # Remove from active alerts
        await redis_manager.delete(alert_key)
        
        # Log action
        background_tasks.add_task(
            _log_alert_action,
            alert_id,
            "resolved",
            current_user.get("id"),
            request.notes,
            {"resolution_type": request.resolution_type, "resolution_time": resolution_time}
        )
        
        # Broadcast update
        background_tasks.add_task(
            _broadcast_alert_update,
            alert_id,
            "resolved",
            {
                "resolved_by": current_user.get("name"),
                "resolution_type": request.resolution_type,
                "notes": request.notes,
                "resolution_time": resolution_time,
                "timestamp": resolved_at.isoformat()
            }
        )
        
        # Update alert statistics
        background_tasks.add_task(
            _update_alert_statistics,
            alert["type"],
            alert["severity"],
            resolution_time,
            request.resolution_type
        )
        
        # If this was a safety alert, update safety monitoring
        if alert.get("type") == "safety" and alert.get("conversation_id"):
            background_tasks.add_task(
                kelly_safety_monitor.mark_alert_resolved,
                alert["conversation_id"],
                alert_id,
                request.resolution_type
            )
        
        return {"message": "Alert resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/escalate/{alert_id}")
async def escalate_alert(
    alert_id: str,
    request: EscalationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Escalate an alert to higher level support"""
    try:
        # Find the alert
        alert_key = await _find_alert_key(alert_id)
        if not alert_key:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert_data = await redis_manager.get(alert_key)
        if not alert_data:
            raise HTTPException(status_code=404, detail="Alert data not found")
        
        alert = json.loads(alert_data)
        
        # Check permissions
        if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
            raise HTTPException(status_code=403, detail="Access denied to this alert")
        
        # Prevent escalating resolved alerts
        if alert.get("status") == "resolved":
            raise HTTPException(status_code=400, detail="Cannot escalate resolved alert")
        
        # Check escalation permissions based on level
        if request.escalation_level == "emergency" and not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required for emergency escalation")
        
        # Update alert
        now = datetime.now()
        escalation_data = {
            "escalation_level": request.escalation_level,
            "escalation_reason": request.reason,
            "escalation_urgency": request.urgency,
            "escalated_by": current_user.get("id"),
            "escalated_at": now.isoformat()
        }
        
        # Add to escalation history
        if "escalation_history" not in alert:
            alert["escalation_history"] = []
        
        alert["escalation_history"].append(escalation_data)
        alert.update({
            "status": "escalated",
            "escalated_at": now.isoformat(),
            "escalated_by": current_user.get("id"),
            "escalation_reason": request.reason,
            "escalation_level": request.escalation_level
        })
        
        # Store updated alert
        await redis_manager.setex(alert_key, 86400 * 30, json.dumps(alert))
        
        # Create escalation notification
        escalation_id = str(uuid4())
        escalation_key = f"kelly:escalation:{escalation_id}"
        
        escalation_notification = {
            "id": escalation_id,
            "alert_id": alert_id,
            "escalation_level": request.escalation_level,
            "urgency": request.urgency,
            "reason": request.reason,
            "escalated_by": current_user.get("id"),
            "escalated_by_name": current_user.get("name"),
            "created_at": now.isoformat(),
            "alert_data": alert if request.include_context else {"id": alert_id, "severity": alert["severity"]},
            "requires_immediate_attention": request.notify_immediately and request.urgency in ["high", "urgent"]
        }
        
        await redis_manager.setex(escalation_key, 86400 * 7, json.dumps(escalation_notification))  # 7 days
        
        # Log action
        background_tasks.add_task(
            _log_alert_action,
            alert_id,
            "escalated",
            current_user.get("id"),
            request.reason,
            {
                "escalation_level": request.escalation_level,
                "urgency": request.urgency
            }
        )
        
        # Broadcast update
        background_tasks.add_task(
            _broadcast_alert_update,
            alert_id,
            "escalated",
            {
                "escalated_by": current_user.get("name"),
                "escalation_level": request.escalation_level,
                "reason": request.reason,
                "urgency": request.urgency,
                "timestamp": now.isoformat()
            }
        )
        
        # Send notifications based on escalation level
        background_tasks.add_task(
            _send_escalation_notifications,
            escalation_notification
        )
        
        return {"message": f"Alert escalated to {request.escalation_level} level successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error escalating alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary", response_model=AlertSummaryResponse)
async def get_alert_summary(
    timeframe: str = Query(default="24h", regex="^(1h|6h|24h|7d|30d)$"),
    current_user = Depends(get_current_user)
) -> AlertSummaryResponse:
    """Get alert summary statistics and metrics"""
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if timeframe == "1h":
            start_time = end_time - timedelta(hours=1)
        elif timeframe == "6h":
            start_time = end_time - timedelta(hours=6)
        elif timeframe == "24h":
            start_time = end_time - timedelta(days=1)
        elif timeframe == "7d":
            start_time = end_time - timedelta(days=7)
        else:  # 30d
            start_time = end_time - timedelta(days=30)
        
        start_timestamp = start_time.timestamp()
        
        # Get active alerts
        active_keys = await redis_manager.keys("kelly:alert:active:*")
        active_alerts = []
        
        for key in active_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                
                # Check permissions
                if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
                    continue
                
                active_alerts.append(alert)
        
        # Get resolved alerts in timeframe
        resolved_keys = await redis_manager.keys("kelly:alert:resolved:*")
        resolved_alerts = []
        
        for key in resolved_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                
                # Check if in timeframe
                created_at = datetime.fromisoformat(alert["created_at"]).timestamp()
                if created_at < start_timestamp:
                    continue
                
                # Check permissions
                if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
                    continue
                
                resolved_alerts.append(alert)
        
        # Calculate statistics
        total_active = len(active_alerts)
        
        # Count by severity
        by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for alert in active_alerts:
            severity = alert.get("severity", "low")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by type
        by_type = {}
        for alert in active_alerts:
            alert_type = alert.get("type", "unknown")
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        # Count by status
        by_status = {}
        for alert in active_alerts:
            status = alert.get("status", "active")
            by_status[status] = by_status.get(status, 0) + 1
        
        # Calculate resolution time
        resolution_times = []
        escalated_count = 0
        false_positive_count = 0
        
        for alert in resolved_alerts:
            if alert.get("resolution_time_seconds"):
                resolution_times.append(alert["resolution_time_seconds"])
            
            if alert.get("status") == "escalated":
                escalated_count += 1
            
            if alert.get("resolution_type") == "false_positive":
                false_positive_count += 1
        
        avg_resolution_time = sum(resolution_times) / max(len(resolution_times), 1)
        total_processed = len(resolved_alerts) + escalated_count
        escalation_rate = (escalated_count / max(total_processed, 1)) * 100
        false_positive_rate = (false_positive_count / max(len(resolved_alerts), 1)) * 100
        
        # Find top alert sources
        source_counts = {}
        for alert in active_alerts + resolved_alerts:
            source = alert.get("account_id", "system")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        top_alert_sources = [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        ][:5]
        
        return AlertSummaryResponse(
            total_active=total_active,
            by_severity=by_severity,
            by_type=by_type,
            by_status=by_status,
            avg_resolution_time=avg_resolution_time,
            escalation_rate=escalation_rate,
            false_positive_rate=false_positive_rate,
            top_alert_sources=top_alert_sources
        )
        
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends", response_model=AlertTrendResponse)
async def get_alert_trends(
    timeframe: str = Query(default="24h", regex="^(1h|6h|24h|7d|30d)$"),
    metric: str = Query(default="count", regex="^(count|resolution_time|escalation_rate)$"),
    current_user = Depends(get_current_user)
) -> AlertTrendResponse:
    """Get alert trend analysis over time"""
    try:
        # Calculate time range and bucket size
        end_time = datetime.now()
        
        if timeframe == "1h":
            start_time = end_time - timedelta(hours=1)
            bucket_minutes = 5
        elif timeframe == "6h":
            start_time = end_time - timedelta(hours=6)
            bucket_minutes = 30
        elif timeframe == "24h":
            start_time = end_time - timedelta(days=1)
            bucket_minutes = 60
        elif timeframe == "7d":
            start_time = end_time - timedelta(days=7)
            bucket_minutes = 360  # 6 hours
        else:  # 30d
            start_time = end_time - timedelta(days=30)
            bucket_minutes = 1440  # 24 hours
        
        # Generate time buckets
        data_points = []
        current_time = start_time
        
        while current_time <= end_time:
            bucket_start = current_time
            bucket_end = current_time + timedelta(minutes=bucket_minutes)
            
            # Get alerts in this bucket
            if metric == "count":
                value = await _get_alert_count_in_bucket(bucket_start, bucket_end, current_user)
            elif metric == "resolution_time":
                value = await _get_avg_resolution_time_in_bucket(bucket_start, bucket_end, current_user)
            else:  # escalation_rate
                value = await _get_escalation_rate_in_bucket(bucket_start, bucket_end, current_user)
            
            data_points.append({
                "timestamp": current_time.isoformat(),
                "value": value,
                "bucket_start": bucket_start.isoformat(),
                "bucket_end": bucket_end.isoformat()
            })
            
            current_time = bucket_end
        
        # Calculate trend
        if len(data_points) >= 2:
            recent_values = [dp["value"] for dp in data_points[-3:]]
            earlier_values = [dp["value"] for dp in data_points[:3]]
            
            recent_avg = sum(recent_values) / len(recent_values)
            earlier_avg = sum(earlier_values) / len(earlier_values)
            
            if earlier_avg > 0:
                percentage_change = ((recent_avg - earlier_avg) / earlier_avg) * 100
            else:
                percentage_change = 0
            
            if percentage_change > 5:
                trend_direction = "increasing"
            elif percentage_change < -5:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
            percentage_change = 0
        
        # Find peak times
        sorted_points = sorted(data_points, key=lambda x: x["value"], reverse=True)
        peak_times = [dp["timestamp"] for dp in sorted_points[:3]]
        
        # Identify patterns
        patterns = []
        
        # Check for daily patterns (if timeframe allows)
        if timeframe in ["7d", "30d"]:
            hour_values = {}
            for dp in data_points:
                hour = datetime.fromisoformat(dp["timestamp"]).hour
                if hour not in hour_values:
                    hour_values[hour] = []
                hour_values[hour].append(dp["value"])
            
            # Find peak hours
            hour_averages = {
                hour: sum(values) / len(values)
                for hour, values in hour_values.items()
            }
            
            peak_hour = max(hour_averages, key=hour_averages.get)
            patterns.append({
                "type": "daily_peak",
                "description": f"Peak activity typically occurs around {peak_hour:02d}:00",
                "confidence": 0.8
            })
        
        return AlertTrendResponse(
            timeframe=timeframe,
            data_points=data_points,
            trend_direction=trend_direction,
            percentage_change=percentage_change,
            peak_times=peak_times,
            patterns=patterns
        )
        
    except Exception as e:
        logger.error(f"Error getting alert trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{alert_id}", response_model=AlertInstanceResponse)
async def get_alert_details(
    alert_id: str,
    current_user = Depends(get_current_user)
) -> AlertInstanceResponse:
    """Get detailed information about a specific alert"""
    try:
        # Find the alert (check both active and resolved)
        alert_key = await _find_alert_key(alert_id)
        if not alert_key:
            # Check resolved alerts
            resolved_keys = await redis_manager.keys("kelly:alert:resolved:*")
            for key in resolved_keys:
                alert_data = await redis_manager.get(key)
                if alert_data:
                    alert = json.loads(alert_data)
                    if alert.get("id") == alert_id:
                        alert_key = key
                        break
        
        if not alert_key:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert_data = await redis_manager.get(alert_key)
        if not alert_data:
            raise HTTPException(status_code=404, detail="Alert data not found")
        
        alert = json.loads(alert_data)
        
        # Check permissions
        if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
            raise HTTPException(status_code=403, detail="Access denied to this alert")
        
        return AlertInstanceResponse(**alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert details for {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HELPER FUNCTIONS =====

async def _find_alert_key(alert_id: str) -> Optional[str]:
    """Find the Redis key for a given alert ID"""
    try:
        # Check active alerts
        active_keys = await redis_manager.keys("kelly:alert:active:*")
        for key in active_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                if alert.get("id") == alert_id:
                    return key
        return None
        
    except Exception:
        return None

def _user_can_access_account(user: Dict[str, Any], account_id: str) -> bool:
    """Check if user has permission to access alerts for an account"""
    # Admin users can access all accounts
    if user.get("is_admin", False):
        return True
    
    # Users can access their own accounts
    user_accounts = user.get("kelly_accounts", [])
    return account_id in user_accounts

async def _log_alert_action(
    alert_id: str,
    action: str,
    user_id: str,
    notes: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log alert action to audit trail"""
    try:
        log_entry = {
            "id": str(uuid4()),
            "alert_id": alert_id,
            "action": action,
            "user_id": user_id,
            "notes": notes,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        log_key = f"kelly:alert_log:{alert_id}:{int(datetime.now().timestamp())}"
        await redis_manager.setex(log_key, 86400 * 90, json.dumps(log_entry))  # 90 days
        
    except Exception as e:
        logger.error(f"Error logging alert action: {e}")

async def _broadcast_alert_update(
    alert_id: str,
    action: str,
    data: Dict[str, Any]
):
    """Broadcast alert update via WebSocket"""
    try:
        message = {
            "type": "alert_update",
            "alert_id": alert_id,
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_to_room("monitoring_dashboard", message)
        await websocket_manager.broadcast_to_room(f"alert_{alert_id}", message)
        
    except Exception as e:
        logger.error(f"Error broadcasting alert update: {e}")

async def _update_alert_statistics(
    alert_type: str,
    severity: str,
    resolution_time: float,
    resolution_type: str
):
    """Update alert statistics for analytics"""
    try:
        stats_key = f"kelly:alert_stats:{datetime.now().strftime('%Y-%m-%d')}"
        
        # Get existing stats
        stats_data = await redis_manager.get(stats_key)
        if stats_data:
            stats = json.loads(stats_data)
        else:
            stats = {
                "total_resolved": 0,
                "avg_resolution_time": 0,
                "by_type": {},
                "by_severity": {},
                "by_resolution_type": {}
            }
        
        # Update stats
        stats["total_resolved"] += 1
        stats["avg_resolution_time"] = (
            (stats["avg_resolution_time"] * (stats["total_resolved"] - 1) + resolution_time) /
            stats["total_resolved"]
        )
        
        # Update counts
        stats["by_type"][alert_type] = stats["by_type"].get(alert_type, 0) + 1
        stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        stats["by_resolution_type"][resolution_type] = stats["by_resolution_type"].get(resolution_type, 0) + 1
        
        # Store updated stats
        await redis_manager.setex(stats_key, 86400 * 7, json.dumps(stats))  # 7 days
        
    except Exception as e:
        logger.error(f"Error updating alert statistics: {e}")

async def _send_escalation_notifications(escalation_data: Dict[str, Any]):
    """Send notifications for escalated alerts"""
    try:
        escalation_level = escalation_data.get("escalation_level")
        
        # Create notification for escalation queue
        notification = {
            "id": str(uuid4()),
            "type": "alert_escalation",
            "escalation_level": escalation_level,
            "alert_id": escalation_data["alert_id"],
            "urgency": escalation_data.get("urgency", "normal"),
            "reason": escalation_data.get("reason"),
            "escalated_by": escalation_data.get("escalated_by_name"),
            "created_at": datetime.now().isoformat(),
            "requires_immediate_attention": escalation_data.get("requires_immediate_attention", False)
        }
        
        # Store in escalation queue
        queue_key = f"kelly:escalation_queue:{escalation_level}"
        await redis_manager.lpush(queue_key, json.dumps(notification))
        await redis_manager.expire(queue_key, 86400 * 7)  # 7 days
        
        # Broadcast to monitoring dashboard
        await websocket_manager.broadcast_to_room(
            "monitoring_dashboard",
            {
                "type": "escalation_notification",
                "data": notification
            }
        )
        
    except Exception as e:
        logger.error(f"Error sending escalation notifications: {e}")

async def _get_alert_count_in_bucket(
    start_time: datetime,
    end_time: datetime,
    current_user: Dict[str, Any]
) -> int:
    """Get alert count in time bucket"""
    try:
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        # Check both active and resolved alerts
        all_keys = await redis_manager.keys("kelly:alert:*")
        
        count = 0
        for key in all_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                
                # Check permissions
                if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
                    continue
                
                # Check if in time bucket
                created_at = datetime.fromisoformat(alert["created_at"]).timestamp()
                if start_ts <= created_at < end_ts:
                    count += 1
        
        return count
        
    except Exception:
        return 0

async def _get_avg_resolution_time_in_bucket(
    start_time: datetime,
    end_time: datetime,
    current_user: Dict[str, Any]
) -> float:
    """Get average resolution time in time bucket"""
    try:
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        resolved_keys = await redis_manager.keys("kelly:alert:resolved:*")
        
        resolution_times = []
        for key in resolved_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                
                # Check permissions
                if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
                    continue
                
                # Check if resolved in time bucket
                if alert.get("resolved_at"):
                    resolved_at = datetime.fromisoformat(alert["resolved_at"]).timestamp()
                    if start_ts <= resolved_at < end_ts and alert.get("resolution_time_seconds"):
                        resolution_times.append(alert["resolution_time_seconds"])
        
        return sum(resolution_times) / max(len(resolution_times), 1)
        
    except Exception:
        return 0.0

async def _get_escalation_rate_in_bucket(
    start_time: datetime,
    end_time: datetime,
    current_user: Dict[str, Any]
) -> float:
    """Get escalation rate in time bucket"""
    try:
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        all_keys = await redis_manager.keys("kelly:alert:*")
        
        total_alerts = 0
        escalated_alerts = 0
        
        for key in all_keys:
            alert_data = await redis_manager.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                
                # Check permissions
                if alert.get("account_id") and not _user_can_access_account(current_user, alert["account_id"]):
                    continue
                
                # Check if created in time bucket
                created_at = datetime.fromisoformat(alert["created_at"]).timestamp()
                if start_ts <= created_at < end_ts:
                    total_alerts += 1
                    
                    if alert.get("status") == "escalated":
                        escalated_alerts += 1
        
        return (escalated_alerts / max(total_alerts, 1)) * 100
        
    except Exception:
        return 0.0

async def _auto_investigate_critical_alert(alert_id: str, alert: Dict[str, Any]):
    """Auto-investigate critical alerts"""
    try:
        # Start investigation process
        alert["status"] = "investigating"
        alert["auto_investigation_started"] = datetime.now().isoformat()
        
        # Store updated alert
        alert_key = await _find_alert_key(alert_id)
        if alert_key:
            await redis_manager.setex(alert_key, 86400 * 30, json.dumps(alert))
        
        # Broadcast investigation start
        await websocket_manager.broadcast_to_room(
            "monitoring_dashboard",
            {
                "type": "auto_investigation_started",
                "alert_id": alert_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error auto-investigating critical alert {alert_id}: {e}")