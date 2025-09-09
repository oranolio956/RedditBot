"""
Telegram Bot API Endpoints

REST API endpoints for managing the Telegram bot, retrieving metrics,
and controlling bot operations.
"""

from typing import Dict, Any, List, Optional
import json
import time

import structlog
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from pydantic import BaseModel, Field

from app.telegram.bot import get_bot, telegram_bot
from app.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


# Pydantic models for request/response

class BotStatusResponse(BaseModel):
    """Bot status response model."""
    is_running: bool
    uptime: float
    message_count: int
    error_count: int
    current_connections: int
    queue_size: int
    components: Dict[str, bool]
    health: Dict[str, bool]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    timestamp: float
    uptime: float
    performance: Dict[str, Any]
    behavior: Dict[str, Any]
    anti_detection: Dict[str, Any]


class WebhookInfoResponse(BaseModel):
    """Webhook info response model."""
    url: Optional[str]
    has_custom_certificate: bool
    pending_update_count: int
    last_error_date: Optional[float]
    last_error_message: Optional[str]
    max_connections: Optional[int]
    allowed_updates: Optional[List[str]]


class SendMessageRequest(BaseModel):
    """Send message request model."""
    chat_id: int
    text: str
    parse_mode: Optional[str] = "HTML"
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    reply_to_message_id: Optional[int] = None


class RateLimitRequest(BaseModel):
    """Rate limit management request."""
    user_id: int
    rule_name: Optional[str] = None


# Dependency to get bot instance
async def get_bot_instance():
    """Get bot instance dependency."""
    bot = await get_bot()
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram bot not available"
        )
    return bot


# Status and monitoring endpoints

@router.get("/status", response_model=BotStatusResponse, tags=["Bot Management"])
async def get_bot_status(bot=Depends(get_bot_instance)) -> BotStatusResponse:
    """
    Get comprehensive bot status.
    
    Returns detailed information about the bot's current state,
    including health status, metrics, and component status.
    """
    try:
        status_data = await bot.get_status()
        return BotStatusResponse(**status_data)
        
    except Exception as e:
        logger.error("Failed to get bot status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve bot status: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_bot_metrics(bot=Depends(get_bot_instance)) -> MetricsResponse:
    """
    Get bot metrics.
    
    Returns comprehensive metrics including performance data,
    user behavior analytics, and anti-detection statistics.
    """
    try:
        if not bot.metrics:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Metrics system not available"
            )
        
        metrics_data = await bot.metrics.get_current_metrics()
        return MetricsResponse(**metrics_data)
        
    except Exception as e:
        logger.error("Failed to get bot metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.get("/metrics/historical", tags=["Monitoring"])
async def get_historical_metrics(
    hours: int = 24,
    bot=Depends(get_bot_instance)
) -> List[Dict[str, Any]]:
    """
    Get historical metrics data.
    
    Args:
        hours: Number of hours of historical data to retrieve (default: 24)
        
    Returns:
        List of historical metrics data points
    """
    try:
        if not bot.metrics:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Metrics system not available"
            )
        
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 168 (1 week)"
            )
        
        historical_data = await bot.metrics.get_historical_metrics(hours)
        return historical_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get historical metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve historical metrics: {str(e)}"
        )


# Webhook management endpoints

@router.get("/webhook/info", response_model=WebhookInfoResponse, tags=["Webhook"])
async def get_webhook_info(bot=Depends(get_bot_instance)) -> WebhookInfoResponse:
    """
    Get current webhook information.
    
    Returns detailed information about the configured webhook,
    including URL, certificate status, and error information.
    """
    try:
        if not bot.webhook_manager:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook not configured"
            )
        
        webhook_info = await bot.webhook_manager.get_webhook_info()
        return WebhookInfoResponse(**webhook_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get webhook info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve webhook info: {str(e)}"
        )


@router.post("/webhook/test", tags=["Webhook"])
async def test_webhook(bot=Depends(get_bot_instance)) -> Dict[str, Any]:
    """
    Test webhook connectivity and configuration.
    
    Performs comprehensive webhook testing including connectivity,
    security validation, and Telegram API integration.
    """
    try:
        if not bot.webhook_manager:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook not configured"
            )
        
        test_result = await bot.webhook_manager.test_webhook_connectivity()
        return test_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to test webhook", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test webhook: {str(e)}"
        )


@router.post("/webhook/restart", tags=["Webhook"])
async def restart_webhook(
    background_tasks: BackgroundTasks,
    bot=Depends(get_bot_instance)
) -> Dict[str, str]:
    """
    Restart webhook configuration.
    
    Removes current webhook and sets it up again.
    This operation is performed in the background.
    """
    try:
        if not bot.webhook_manager:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook not configured"
            )
        
        # Schedule webhook restart in background
        background_tasks.add_task(bot.webhook_manager.restart_webhook)
        
        return {"message": "Webhook restart initiated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restart webhook", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart webhook: {str(e)}"
        )


# Message sending endpoints

@router.post("/send-message", tags=["Messages"])
async def send_message(
    request: SendMessageRequest,
    bot=Depends(get_bot_instance)
) -> Dict[str, Any]:
    """
    Send message to a chat.
    
    Sends a message through the bot with anti-ban measures
    and natural timing simulation.
    """
    try:
        await bot.send_message_queued(
            chat_id=request.chat_id,
            text=request.text,
            simulate_typing=True,
            parse_mode=request.parse_mode,
            disable_web_page_preview=request.disable_web_page_preview,
            disable_notification=request.disable_notification,
            reply_to_message_id=request.reply_to_message_id
        )
        
        return {
            "message": "Message queued for sending",
            "chat_id": request.chat_id,
            "queued_at": time.time()
        }
        
    except Exception as e:
        logger.error("Failed to send message", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )


# Session management endpoints

@router.get("/sessions", tags=["Sessions"])
async def get_active_sessions(bot=Depends(get_bot_instance)) -> Dict[str, Any]:
    """
    Get information about active user sessions.
    
    Returns summary of currently active sessions and their metrics.
    """
    try:
        if not bot.session_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session manager not available"
            )
        
        session_metrics = await bot.session_manager.get_metrics()
        return session_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session info: {str(e)}"
        )


@router.get("/sessions/{user_id}", tags=["Sessions"])
async def get_user_sessions(
    user_id: int,
    bot=Depends(get_bot_instance)
) -> List[Dict[str, Any]]:
    """
    Get sessions for a specific user.
    
    Args:
        user_id: Telegram user ID
        
    Returns:
        List of active sessions for the user
    """
    try:
        if not bot.session_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session manager not available"
            )
        
        sessions = await bot.session_manager.get_user_sessions(user_id)
        
        # Convert to serializable format
        session_data = []
        for session in sessions:
            session_dict = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "chat_id": session.chat_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "state": session.state.value,
                "conversation_mode": session.conversation_mode.value,
                "total_interactions": session.total_interactions,
                "error_count": session.error_count,
            }
            session_data.append(session_dict)
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user sessions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user sessions: {str(e)}"
        )


@router.delete("/sessions/{session_id}", tags=["Sessions"])
async def expire_session(
    session_id: str,
    bot=Depends(get_bot_instance)
) -> Dict[str, str]:
    """
    Manually expire a user session.
    
    Args:
        session_id: Session ID to expire
        
    Returns:
        Success message
    """
    try:
        if not bot.session_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session manager not available"
            )
        
        success = await bot.session_manager.expire_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {"message": f"Session {session_id} expired successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to expire session", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to expire session: {str(e)}"
        )


# Rate limiting management

@router.get("/rate-limits/{user_id}", tags=["Rate Limiting"])
async def get_rate_limit_status(
    user_id: int,
    bot=Depends(get_bot_instance)
) -> Dict[str, Any]:
    """
    Get rate limit status for a user.
    
    Args:
        user_id: Telegram user ID
        
    Returns:
        Current rate limit status for all applicable rules
    """
    try:
        if not bot.rate_limiter:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter not available"
            )
        
        status_data = await bot.rate_limiter.get_status(str(user_id))
        return status_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get rate limit status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve rate limit status: {str(e)}"
        )


@router.post("/rate-limits/reset", tags=["Rate Limiting"])
async def reset_rate_limits(
    request: RateLimitRequest,
    bot=Depends(get_bot_instance)
) -> Dict[str, str]:
    """
    Reset rate limits for a user.
    
    Resets either all rate limits or a specific rule for the user.
    """
    try:
        if not bot.rate_limiter:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter not available"
            )
        
        await bot.rate_limiter.reset_limits(str(request.user_id), request.rule_name)
        
        if request.rule_name:
            message = f"Reset rate limit rule '{request.rule_name}' for user {request.user_id}"
        else:
            message = f"Reset all rate limits for user {request.user_id}"
        
        return {"message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reset rate limits", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset rate limits: {str(e)}"
        )


# Circuit breaker management

@router.get("/circuit-breakers", tags=["Circuit Breakers"])
async def get_circuit_breaker_status(bot=Depends(get_bot_instance)) -> Dict[str, Any]:
    """
    Get status of all circuit breakers.
    
    Returns comprehensive status information for all circuit breakers
    including state, failure counts, and performance metrics.
    """
    try:
        if not bot.circuit_breaker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Circuit breaker not available"
            )
        
        stats = await bot.circuit_breaker.get_all_stats()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get circuit breaker status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve circuit breaker status: {str(e)}"
        )


@router.post("/circuit-breakers/{breaker_name}/reset", tags=["Circuit Breakers"])
async def reset_circuit_breaker(
    breaker_name: str,
    bot=Depends(get_bot_instance)
) -> Dict[str, str]:
    """
    Reset a specific circuit breaker.
    
    Args:
        breaker_name: Name of the circuit breaker to reset
        
    Returns:
        Success message
    """
    try:
        if not bot.circuit_breaker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Circuit breaker not available"
            )
        
        success = await bot.circuit_breaker.reset_breaker(breaker_name)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Circuit breaker '{breaker_name}' not found"
            )
        
        return {"message": f"Circuit breaker '{breaker_name}' reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reset circuit breaker", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset circuit breaker: {str(e)}"
        )


# Anti-ban system management

@router.get("/anti-ban/metrics", tags=["Anti-Ban"])
async def get_anti_ban_metrics(bot=Depends(get_bot_instance)) -> Dict[str, Any]:
    """
    Get anti-ban system metrics.
    
    Returns comprehensive metrics about anti-detection measures,
    risk assessments, and evasion effectiveness.
    """
    try:
        if not bot.anti_ban:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Anti-ban system not available"
            )
        
        metrics = await bot.anti_ban.get_metrics()
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get anti-ban metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve anti-ban metrics: {str(e)}"
        )


# Administrative endpoints

@router.post("/maintenance/cleanup", tags=["Maintenance"])
async def run_maintenance_cleanup(
    background_tasks: BackgroundTasks,
    bot=Depends(get_bot_instance)
) -> Dict[str, str]:
    """
    Run maintenance cleanup tasks.
    
    Triggers cleanup of expired sessions, old metrics data,
    and other maintenance tasks in the background.
    """
    try:
        async def cleanup_task():
            """Background cleanup task."""
            try:
                # Session cleanup
                if bot.session_manager:
                    await bot.session_manager.cleanup_expired_sessions()
                
                # Rate limiter cleanup
                if bot.rate_limiter:
                    await bot.rate_limiter.cleanup_expired_data()
                
                # Anti-ban cleanup
                if bot.anti_ban:
                    await bot.anti_ban.cleanup_old_data()
                
                logger.info("Maintenance cleanup completed successfully")
                
            except Exception as e:
                logger.error("Error during maintenance cleanup", error=str(e))
        
        background_tasks.add_task(cleanup_task)
        
        return {"message": "Maintenance cleanup initiated"}
        
    except Exception as e:
        logger.error("Failed to start maintenance cleanup", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start maintenance cleanup: {str(e)}"
        )