"""
Telegram Account Monitoring API
Real-time monitoring and analytics for account health and performance
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import json
import asyncio

from app.database import get_session
from app.models.telegram_account import TelegramAccount, AccountSafetyEvent
from app.models.telegram_community import TelegramCommunity, CommunityEngagementEvent
from app.models.telegram_conversation import TelegramConversation, ConversationMessage
from app.services.telegram_safety_monitor import TelegramSafetyMonitor
from app.schemas.telegram_schemas import (
    AccountHealthResponse,
    SafetyMetricsResponse,
    EngagementAnalyticsResponse,
    RealTimeMetric
)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.get("/health/{account_id}", response_model=AccountHealthResponse)
async def get_account_health(
    account_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get comprehensive health metrics for a specific account"""
    
    # Get account
    result = await session.execute(
        select(TelegramAccount).where(TelegramAccount.id == account_id)
    )
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Get recent safety events
    safety_events_result = await session.execute(
        select(AccountSafetyEvent)
        .where(AccountSafetyEvent.account_id == account_id)
        .where(AccountSafetyEvent.created_at >= datetime.utcnow() - timedelta(days=7))
        .order_by(AccountSafetyEvent.created_at.desc())
        .limit(10)
    )
    recent_safety_events = safety_events_result.scalars().all()
    
    # Calculate health metrics
    health_metrics = {
        "overall_health": account.health_score,
        "risk_score": account.risk_score,
        "status": account.status,
        "uptime_percentage": calculate_uptime(account),
        "daily_limits": {
            "messages": {
                "used": account.daily_message_count,
                "limit": account.configuration.get("max_daily_messages", 50),
                "percentage": (account.daily_message_count / max(account.configuration.get("max_daily_messages", 50), 1)) * 100
            },
            "groups": {
                "used": account.daily_group_joins,
                "limit": account.configuration.get("max_daily_groups", 2),
                "percentage": (account.daily_group_joins / max(account.configuration.get("max_daily_groups", 2), 1)) * 100
            },
            "dms": {
                "used": account.daily_dm_count,
                "limit": account.configuration.get("max_daily_dms", 5),
                "percentage": (account.daily_dm_count / max(account.configuration.get("max_daily_dms", 5), 1)) * 100
            }
        },
        "safety_indicators": {
            "flood_waits_today": account.flood_wait_count,
            "spam_warnings": account.spam_warning_count,
            "consecutive_errors": account.consecutive_errors,
            "last_safety_event": recent_safety_events[0].created_at if recent_safety_events else None,
            "critical_events_7d": len([e for e in recent_safety_events if e.severity == "critical"])
        },
        "performance_metrics": {
            "avg_response_time": account.metrics.get("avg_response_time", 0),
            "successful_messages": account.metrics.get("successful_messages", 0),
            "failed_messages": account.metrics.get("failed_messages", 0),
            "success_rate": calculate_success_rate(account)
        },
        "last_active": account.last_active,
        "account_age_days": (datetime.utcnow() - account.created_at).days
    }
    
    return AccountHealthResponse(**health_metrics)

@router.get("/safety/{account_id}", response_model=SafetyMetricsResponse)
async def get_safety_metrics(
    account_id: int,
    days: int = 7,
    session: AsyncSession = Depends(get_session)
):
    """Get detailed safety metrics and risk analysis"""
    
    # Get safety events for period
    start_date = datetime.utcnow() - timedelta(days=days)
    
    result = await session.execute(
        select(AccountSafetyEvent)
        .where(AccountSafetyEvent.account_id == account_id)
        .where(AccountSafetyEvent.created_at >= start_date)
        .order_by(AccountSafetyEvent.created_at.desc())
    )
    safety_events = result.scalars().all()
    
    # Group events by type and severity
    event_analysis = {
        "by_type": {},
        "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
        "by_day": {},
        "patterns": []
    }
    
    for event in safety_events:
        # By type
        if event.event_type not in event_analysis["by_type"]:
            event_analysis["by_type"][event.event_type] = 0
        event_analysis["by_type"][event.event_type] += 1
        
        # By severity
        event_analysis["by_severity"][event.severity] += 1
        
        # By day
        day_key = event.created_at.date().isoformat()
        if day_key not in event_analysis["by_day"]:
            event_analysis["by_day"][day_key] = []
        event_analysis["by_day"][day_key].append({
            "type": event.event_type,
            "severity": event.severity,
            "time": event.created_at.isoformat()
        })
    
    # Detect patterns
    if len(safety_events) > 3:
        # Check for flood wait escalation
        flood_events = [e for e in safety_events if "flood" in e.event_type.lower()]
        if len(flood_events) > 2:
            event_analysis["patterns"].append({
                "pattern": "flood_wait_escalation",
                "severity": "high",
                "description": "Multiple flood wait events detected",
                "recommendation": "Reduce message frequency and increase delays"
            })
        
        # Check for time-based patterns
        hour_distribution = {}
        for event in safety_events:
            hour = event.created_at.hour
            if hour not in hour_distribution:
                hour_distribution[hour] = 0
            hour_distribution[hour] += 1
        
        # Find peak problem hours
        if hour_distribution:
            peak_hour = max(hour_distribution, key=hour_distribution.get)
            if hour_distribution[peak_hour] > len(safety_events) * 0.3:
                event_analysis["patterns"].append({
                    "pattern": "time_concentration",
                    "severity": "medium",
                    "description": f"Most issues occur around {peak_hour}:00",
                    "recommendation": "Avoid heavy activity during this time"
                })
    
    # Calculate risk trend
    if len(event_analysis["by_day"]) > 1:
        days_sorted = sorted(event_analysis["by_day"].keys())
        first_half = days_sorted[:len(days_sorted)//2]
        second_half = days_sorted[len(days_sorted)//2:]
        
        first_half_events = sum(len(event_analysis["by_day"][d]) for d in first_half)
        second_half_events = sum(len(event_analysis["by_day"][d]) for d in second_half)
        
        if second_half_events > first_half_events * 1.5:
            risk_trend = "increasing"
        elif second_half_events < first_half_events * 0.5:
            risk_trend = "decreasing"
        else:
            risk_trend = "stable"
    else:
        risk_trend = "insufficient_data"
    
    return SafetyMetricsResponse(
        account_id=account_id,
        period_days=days,
        total_events=len(safety_events),
        event_analysis=event_analysis,
        risk_trend=risk_trend,
        recommendations=generate_safety_recommendations(event_analysis)
    )

@router.get("/engagement/{account_id}", response_model=EngagementAnalyticsResponse)
async def get_engagement_analytics(
    account_id: int,
    days: int = 7,
    session: AsyncSession = Depends(get_session)
):
    """Get engagement analytics and community performance"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get communities
    communities_result = await session.execute(
        select(TelegramCommunity)
        .where(TelegramCommunity.account_id == account_id)
        .where(TelegramCommunity.status == "active")
    )
    communities = communities_result.scalars().all()
    
    # Get engagement events
    engagement_result = await session.execute(
        select(CommunityEngagementEvent)
        .join(TelegramCommunity)
        .where(TelegramCommunity.account_id == account_id)
        .where(CommunityEngagementEvent.created_at >= start_date)
    )
    engagement_events = engagement_result.scalars().all()
    
    # Get conversations
    conversations_result = await session.execute(
        select(TelegramConversation)
        .where(TelegramConversation.account_id == account_id)
        .where(TelegramConversation.created_at >= start_date)
    )
    conversations = conversations_result.scalars().all()
    
    # Calculate analytics
    analytics = {
        "communities": {
            "total": len(communities),
            "active": len([c for c in communities if c.last_activity and c.last_activity >= start_date]),
            "top_performers": []
        },
        "engagement": {
            "total_events": len(engagement_events),
            "messages_sent": len([e for e in engagement_events if e.event_type == "message_sent"]),
            "messages_received": len([e for e in engagement_events if e.event_type == "message_received"]),
            "reactions_given": len([e for e in engagement_events if e.event_type == "reaction"]),
            "mentions_received": len([e for e in engagement_events if e.event_type == "mention"])
        },
        "conversations": {
            "total": len(conversations),
            "active": len([c for c in conversations if c.last_message_at and c.last_message_at >= datetime.utcnow() - timedelta(days=1)]),
            "avg_messages": sum(c.message_count for c in conversations) / max(len(conversations), 1),
            "avg_duration": calculate_avg_conversation_duration(conversations)
        },
        "response_metrics": {
            "avg_response_time": calculate_avg_response_time(engagement_events),
            "response_rate": calculate_response_rate(engagement_events),
            "engagement_score": calculate_engagement_score(communities, engagement_events)
        }
    }
    
    # Get top performing communities
    for community in sorted(communities, key=lambda c: c.reputation_score, reverse=True)[:5]:
        analytics["communities"]["top_performers"].append({
            "name": community.chat_title,
            "reputation": community.reputation_score,
            "engagement_level": community.engagement_level,
            "messages": community.metrics.get("total_messages", 0)
        })
    
    return EngagementAnalyticsResponse(**analytics)

@router.get("/dashboard/{account_id}")
async def get_dashboard_data(
    account_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get comprehensive dashboard data for account monitoring"""
    
    # Get all metrics concurrently
    health_task = get_account_health(account_id, session)
    safety_task = get_safety_metrics(account_id, 7, session)
    engagement_task = get_engagement_analytics(account_id, 7, session)
    
    health, safety, engagement = await asyncio.gather(
        health_task, safety_task, engagement_task
    )
    
    # Compile dashboard data
    dashboard = {
        "account_id": account_id,
        "timestamp": datetime.utcnow().isoformat(),
        "health": health.dict(),
        "safety": safety.dict(),
        "engagement": engagement.dict(),
        "alerts": generate_alerts(health, safety, engagement),
        "recommendations": generate_dashboard_recommendations(health, safety, engagement)
    }
    
    return dashboard

@router.websocket("/ws/{account_id}")
async def websocket_monitoring(
    websocket: WebSocket,
    account_id: int,
    session: AsyncSession = Depends(get_session)
):
    """WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Get real-time metrics
            metrics = await get_realtime_metrics(account_id, session)
            
            # Send to client
            await websocket.send_json(metrics)
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
        raise e

# Helper functions
def calculate_uptime(account: TelegramAccount) -> float:
    """Calculate account uptime percentage"""
    if not account.created_at:
        return 0.0
    
    total_time = (datetime.utcnow() - account.created_at).total_seconds()
    if total_time == 0:
        return 0.0
    
    # Estimate downtime from safety events
    downtime = account.metrics.get("total_downtime_seconds", 0)
    uptime = max(0, total_time - downtime)
    
    return min(100.0, (uptime / total_time) * 100)

def calculate_success_rate(account: TelegramAccount) -> float:
    """Calculate message success rate"""
    successful = account.metrics.get("successful_messages", 0)
    failed = account.metrics.get("failed_messages", 0)
    total = successful + failed
    
    if total == 0:
        return 100.0
    
    return (successful / total) * 100

def calculate_avg_conversation_duration(conversations: List[TelegramConversation]) -> float:
    """Calculate average conversation duration in minutes"""
    if not conversations:
        return 0.0
    
    durations = []
    for conv in conversations:
        if conv.created_at and conv.last_message_at:
            duration = (conv.last_message_at - conv.created_at).total_seconds() / 60
            durations.append(duration)
    
    return sum(durations) / max(len(durations), 1)

def calculate_avg_response_time(events: List[CommunityEngagementEvent]) -> float:
    """Calculate average response time in seconds"""
    response_times = []
    
    # Group events by conversation
    conversations = {}
    for event in sorted(events, key=lambda e: e.created_at):
        conv_id = event.metadata.get("conversation_id")
        if conv_id:
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(event)
    
    # Calculate response times
    for conv_events in conversations.values():
        for i in range(1, len(conv_events)):
            if conv_events[i].event_type == "message_sent" and conv_events[i-1].event_type == "message_received":
                response_time = (conv_events[i].created_at - conv_events[i-1].created_at).total_seconds()
                if response_time < 3600:  # Ignore responses over 1 hour
                    response_times.append(response_time)
    
    return sum(response_times) / max(len(response_times), 1) if response_times else 0

def calculate_response_rate(events: List[CommunityEngagementEvent]) -> float:
    """Calculate response rate percentage"""
    received = len([e for e in events if e.event_type == "message_received"])
    sent = len([e for e in events if e.event_type == "message_sent"])
    
    if received == 0:
        return 0.0
    
    return min(100.0, (sent / received) * 100)

def calculate_engagement_score(communities: List[TelegramCommunity], events: List[CommunityEngagementEvent]) -> float:
    """Calculate overall engagement score"""
    if not communities:
        return 0.0
    
    # Factors: reputation, activity, diversity
    avg_reputation = sum(c.reputation_score for c in communities) / len(communities)
    activity_score = min(100, len(events) / 10)  # Scale to 100
    diversity_score = min(100, len(communities) * 10)  # Scale to 100
    
    # Weighted average
    engagement_score = (
        avg_reputation * 0.4 +
        activity_score * 0.3 +
        diversity_score * 0.3
    )
    
    return min(100.0, engagement_score)

def generate_safety_recommendations(event_analysis: Dict[str, Any]) -> List[str]:
    """Generate safety recommendations based on event analysis"""
    recommendations = []
    
    # Check for high-severity events
    if event_analysis["by_severity"]["critical"] > 0:
        recommendations.append("⚠️ Critical events detected - Immediate review required")
    
    if event_analysis["by_severity"]["high"] > 2:
        recommendations.append("Reduce activity frequency to avoid detection")
    
    # Check for specific event types
    if "FloodWait" in event_analysis["by_type"]:
        recommendations.append("Increase delays between messages (minimum 60 seconds)")
    
    if "SpamWarning" in event_analysis["by_type"]:
        recommendations.append("Review message content for spam-like patterns")
    
    # Check patterns
    for pattern in event_analysis.get("patterns", []):
        recommendations.append(pattern["recommendation"])
    
    if not recommendations:
        recommendations.append("✅ No immediate safety concerns")
    
    return recommendations

def generate_alerts(health: AccountHealthResponse, safety: SafetyMetricsResponse, engagement: EngagementAnalyticsResponse) -> List[Dict[str, Any]]:
    """Generate alerts based on current metrics"""
    alerts = []
    
    # Health alerts
    if health.risk_score > 70:
        alerts.append({
            "level": "critical",
            "type": "risk",
            "message": f"High risk score: {health.risk_score:.1f}",
            "action": "Reduce activity immediately"
        })
    
    if health.overall_health < 50:
        alerts.append({
            "level": "warning",
            "type": "health",
            "message": f"Low health score: {health.overall_health:.1f}",
            "action": "Review recent activities and errors"
        })
    
    # Safety alerts
    if safety.total_events > 10:
        alerts.append({
            "level": "warning",
            "type": "safety",
            "message": f"High number of safety events: {safety.total_events}",
            "action": "Review safety event log"
        })
    
    # Engagement alerts
    if engagement.response_metrics["response_rate"] < 20:
        alerts.append({
            "level": "info",
            "type": "engagement",
            "message": "Low response rate detected",
            "action": "Increase engagement in active communities"
        })
    
    return alerts

def generate_dashboard_recommendations(health: AccountHealthResponse, safety: SafetyMetricsResponse, engagement: EngagementAnalyticsResponse) -> List[str]:
    """Generate comprehensive dashboard recommendations"""
    recommendations = []
    
    # Combine all recommendation sources
    recommendations.extend(safety.recommendations)
    
    # Add health-based recommendations
    if health.daily_limits["messages"]["percentage"] > 80:
        recommendations.append("Approaching daily message limit - pace activity")
    
    # Add engagement-based recommendations
    if engagement.communities["active"] < engagement.communities["total"] * 0.5:
        recommendations.append("Increase activity in dormant communities")
    
    if engagement.response_metrics["avg_response_time"] > 300:
        recommendations.append("Improve response time for better engagement")
    
    return list(set(recommendations))  # Remove duplicates

async def get_realtime_metrics(account_id: int, session: AsyncSession) -> Dict[str, Any]:
    """Get real-time metrics for WebSocket updates"""
    
    # Get account
    result = await session.execute(
        select(TelegramAccount).where(TelegramAccount.id == account_id)
    )
    account = result.scalar_one_or_none()
    
    if not account:
        return {"error": "Account not found"}
    
    # Get recent events (last 5 minutes)
    recent_time = datetime.utcnow() - timedelta(minutes=5)
    
    # Get recent safety events
    safety_result = await session.execute(
        select(func.count(AccountSafetyEvent.id))
        .where(AccountSafetyEvent.account_id == account_id)
        .where(AccountSafetyEvent.created_at >= recent_time)
    )
    recent_safety_events = safety_result.scalar() or 0
    
    # Get recent messages
    message_result = await session.execute(
        select(func.count(ConversationMessage.id))
        .join(TelegramConversation)
        .where(TelegramConversation.account_id == account_id)
        .where(ConversationMessage.created_at >= recent_time)
    )
    recent_messages = message_result.scalar() or 0
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "account_id": account_id,
        "status": account.status,
        "health_score": account.health_score,
        "risk_score": account.risk_score,
        "recent_activity": {
            "messages": recent_messages,
            "safety_events": recent_safety_events
        },
        "daily_usage": {
            "messages": f"{account.daily_message_count}/{account.configuration.get('max_daily_messages', 50)}",
            "groups": f"{account.daily_group_joins}/{account.configuration.get('max_daily_groups', 2)}",
            "dms": f"{account.daily_dm_count}/{account.configuration.get('max_daily_dms', 5)}"
        },
        "alerts": []  # Add any immediate alerts
    }