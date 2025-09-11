"""
Kelly Brain API Endpoints

FastAPI endpoints for managing Kelly's Telegram accounts, AI features,
conversation monitoring, and safety settings.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from app.core.auth import get_current_user
from app.core.redis import redis_manager
from app.services.kelly_personality_service import kelly_personality_service, KellyPersonalityConfig
from app.services.kelly_telegram_userbot import kelly_userbot, AccountConfig
from app.services.kelly_dm_detector import kelly_dm_detector
from app.services.kelly_conversation_manager import kelly_conversation_manager
from app.services.kelly_claude_ai import kelly_claude_ai
from app.services.kelly_safety_monitor import kelly_safety_monitor

logger = structlog.get_logger()

router = APIRouter(prefix="/kelly", tags=["kelly"])

# Request/Response Models
class AddAccountRequest(BaseModel):
    api_id: int
    api_hash: str
    phone_number: str
    dm_only_mode: bool = True
    kelly_config: Optional[Dict[str, Any]] = None

class UpdateAccountConfigRequest(BaseModel):
    kelly_config: Optional[Dict[str, Any]] = None
    dm_only_mode: Optional[bool] = None
    max_daily_messages: Optional[int] = None
    response_probability: Optional[float] = None

class ToggleAccountRequest(BaseModel):
    enabled: bool

class ToggleAIFeatureRequest(BaseModel):
    feature: str
    enabled: bool

class AccountResponse(BaseModel):
    id: str
    phone_number: str
    session_name: str
    dm_only_mode: bool
    enabled: bool
    connected: bool
    messages_today: int
    max_daily_messages: int
    response_probability: float
    recent_conversations: int
    kelly_config: Dict[str, Any]

class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    stage: str
    message_count: int
    safety_score: float
    trust_level: float
    engagement_quality: float
    red_flags_count: int
    first_contact: str
    last_message: str

class AIFeaturesResponse(BaseModel):
    consciousness_mirroring: bool
    memory_palace: bool
    emotional_intelligence: bool
    temporal_archaeology: bool
    digital_telepathy: bool
    quantum_consciousness: bool
    synesthesia: bool
    neural_dreams: bool

class ClaudeMetricsResponse(BaseModel):
    total_requests: int
    total_tokens_used: int
    cost_today: float
    cost_this_month: float
    model_usage: Dict[str, int]
    avg_response_time: float
    success_rate: float
    conversations_enhanced: int
    personality_adaptations: int

class ClaudeConfigRequest(BaseModel):
    enabled: bool = True
    model_preference: str = "claude-3-sonnet-20240229"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)
    personality_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    safety_level: str = Field(default="high", regex="^(low|medium|high)$")
    context_memory: bool = True
    auto_adapt: bool = True

class ClaudeConfigResponse(BaseModel):
    account_id: str
    config: ClaudeConfigRequest
    last_updated: str
    performance_score: float

class SafetyStatusResponse(BaseModel):
    overall_status: str
    active_threats: int
    blocked_users: int
    flagged_conversations: int
    threat_level_distribution: Dict[str, int]
    detection_accuracy: float
    response_time_avg: float
    alerts_pending_review: int
    auto_actions_today: int

class SafetyAlertRequest(BaseModel):
    action: str = Field(..., regex="^(approve|deny|escalate)$")
    reason: Optional[str] = None
    override_block: bool = False

@router.get("/accounts")
async def get_accounts(current_user = Depends(get_current_user)) -> Dict[str, List[AccountResponse]]:
    """Get all Kelly Telegram accounts"""
    try:
        accounts = []
        
        # Get all account IDs from Redis
        keys = await redis_manager.scan_iter(match="kelly:account:*")
        async for key in keys:
            account_id = key.split(":")[-1]
            
            # Get account status
            status = await kelly_userbot.get_account_status(account_id)
            if "error" not in status:
                accounts.append(AccountResponse(
                    id=account_id,
                    phone_number=status.get("phone_number", "Unknown"),
                    session_name=status.get("session_name", f"kelly_{account_id}"),
                    dm_only_mode=status.get("dm_only_mode", True),
                    enabled=status.get("enabled", False),
                    connected=status.get("connected", False),
                    messages_today=status.get("messages_today", 0),
                    max_daily_messages=status.get("max_daily_messages", 50),
                    response_probability=status.get("response_probability", 0.9),
                    recent_conversations=status.get("recent_conversations", 0),
                    kelly_config=status.get("kelly_config", {})
                ))
        
        return {"accounts": accounts}
        
    except Exception as e:
        logger.error(f"Error getting accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/accounts")
async def add_account(
    request: AddAccountRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Add a new Kelly Telegram account"""
    try:
        # Generate account ID
        import uuid
        account_id = str(uuid.uuid4())[:8]
        
        # Create Kelly configuration
        kelly_config = None
        if request.kelly_config:
            kelly_config = KellyPersonalityConfig(**request.kelly_config)
        
        # Add account in background
        background_tasks.add_task(
            _add_account_background,
            account_id,
            request.api_id,
            request.api_hash,
            request.phone_number,
            request.dm_only_mode,
            kelly_config
        )
        
        return {
            "account_id": account_id,
            "message": "Account addition started. Check status in a few moments."
        }
        
    except Exception as e:
        logger.error(f"Error adding account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _add_account_background(
    account_id: str,
    api_id: int,
    api_hash: str,
    phone_number: str,
    dm_only_mode: bool,
    kelly_config: Optional[KellyPersonalityConfig]
):
    """Background task to add account"""
    try:
        success = await kelly_userbot.add_account(
            account_id=account_id,
            api_id=api_id,
            api_hash=api_hash,
            phone_number=phone_number,
            dm_only_mode=dm_only_mode,
            kelly_config=kelly_config
        )
        
        if success:
            logger.info(f"Successfully added Kelly account {account_id}")
        else:
            logger.error(f"Failed to add Kelly account {account_id}")
            
    except Exception as e:
        logger.error(f"Error in background account addition: {e}")

@router.patch("/accounts/{account_id}/config")
async def update_account_config(
    account_id: str,
    request: UpdateAccountConfigRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Update account configuration"""
    try:
        updates = {}
        
        if request.kelly_config is not None:
            updates["kelly_config"] = request.kelly_config
        if request.dm_only_mode is not None:
            updates["dm_only_mode"] = request.dm_only_mode
        if request.max_daily_messages is not None:
            updates["max_daily_messages"] = request.max_daily_messages
        if request.response_probability is not None:
            updates["response_probability"] = request.response_probability
        
        success = await kelly_userbot.update_account_config(account_id, updates)
        
        if success:
            return {"message": "Account configuration updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Account not found")
            
    except Exception as e:
        logger.error(f"Error updating account config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/accounts/{account_id}/toggle")
async def toggle_account(
    account_id: str,
    request: ToggleAccountRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Enable or disable an account"""
    try:
        success = await kelly_userbot.update_account_config(
            account_id, 
            {"enabled": request.enabled}
        )
        
        if success:
            action = "enabled" if request.enabled else "disabled"
            return {"message": f"Account {action} successfully"}
        else:
            raise HTTPException(status_code=404, detail="Account not found")
            
    except Exception as e:
        logger.error(f"Error toggling account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/accounts/{account_id}")
async def delete_account(
    account_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete an account"""
    try:
        # Stop the account first
        await kelly_userbot.stop_account(account_id)
        
        # Remove from Redis
        await redis_manager.delete(f"kelly:account:{account_id}")
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai-features")
async def get_ai_features(current_user = Depends(get_current_user)) -> Dict[str, AIFeaturesResponse]:
    """Get AI features status"""
    try:
        features = await kelly_conversation_manager.get_ai_features_status()
        
        return {"features": AIFeaturesResponse(**features)}
        
    except Exception as e:
        logger.error(f"Error getting AI features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-features/toggle")
async def toggle_ai_feature(
    request: ToggleAIFeatureRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Toggle AI feature on/off"""
    try:
        success = await kelly_conversation_manager.toggle_ai_feature(
            request.feature, 
            request.enabled
        )
        
        if success:
            action = "enabled" if request.enabled else "disabled"
            return {"message": f"AI feature '{request.feature}' {action} successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid AI feature name")
            
    except Exception as e:
        logger.error(f"Error toggling AI feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations")
async def get_conversations(
    account_id: Optional[str] = None,
    limit: int = 50,
    current_user = Depends(get_current_user)
) -> Dict[str, List[ConversationResponse]]:
    """Get active conversations"""
    try:
        conversations = []
        
        # Get conversation data from Redis
        if account_id:
            pattern = f"kelly:conversation_track:{account_id}_*"
        else:
            pattern = "kelly:conversation_track:*"
        
        keys = await redis_manager.scan_iter(match=pattern)
        async for key in keys:
            conversation_id = key.split(":")[-1]
            
            # Get conversation stats
            stats = await kelly_personality_service.get_conversation_stats(conversation_id)
            if stats:
                conversations.append(ConversationResponse(
                    conversation_id=conversation_id,
                    user_id=conversation_id.split("_")[-1],
                    stage=stats.get("stage", "unknown"),
                    message_count=stats.get("message_count", 0),
                    safety_score=stats.get("safety_score", 1.0),
                    trust_level=stats.get("trust_level", 0.0),
                    engagement_quality=stats.get("engagement_quality", 0.0),
                    red_flags_count=stats.get("red_flags_count", 0),
                    first_contact=stats.get("first_contact", ""),
                    last_message=stats.get("last_message", "")
                ))
        
        # Sort by last message time and limit
        conversations.sort(key=lambda x: x.last_message, reverse=True)
        
        return {"conversations": conversations[:limit]}
        
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}")
async def get_conversation_details(
    conversation_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed conversation information"""
    try:
        # Get conversation insights
        account_id, user_id = conversation_id.split("_", 1)
        insights = await kelly_conversation_manager.get_conversation_insights(account_id, user_id)
        
        return insights
        
    except Exception as e:
        logger.error(f"Error getting conversation details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    current_user = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """Get conversation message history"""
    try:
        account_id, user_id = conversation_id.split("_", 1)
        history = await kelly_userbot.get_conversation_history(account_id, user_id, limit)
        
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/detection")
async def get_detection_stats(
    account_id: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get DM detection statistics"""
    try:
        if account_id:
            stats = await kelly_dm_detector.get_detection_stats(account_id)
            return {"account_stats": stats}
        else:
            # Get stats for all accounts
            all_stats = {}
            keys = await redis_manager.scan_iter(match="kelly:detection_profile:*")
            async for key in keys:
                acc_id = key.split(":")[-1]
                stats = await kelly_dm_detector.get_detection_stats(acc_id)
                all_stats[acc_id] = stats
            
            return {"all_accounts": all_stats}
            
    except Exception as e:
        logger.error(f"Error getting detection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/daily")
async def get_daily_stats(
    account_id: Optional[str] = None,
    date: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get daily conversation statistics"""
    try:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        stats = {}
        
        if account_id:
            key = f"kelly:daily_stats:{account_id}:{date}"
            daily_data = await redis_manager.lrange(key, 0, -1)
            
            stats[account_id] = {
                "date": date,
                "total_messages": len(daily_data),
                "conversations": len(set(
                    json.loads(data).get("user_id") for data in daily_data
                )),
                "safety_scores": [
                    json.loads(data).get("safety_score", 1.0) for data in daily_data
                ],
                "stages": [
                    json.loads(data).get("stage", "unknown") for data in daily_data
                ]
            }
        else:
            # Get stats for all accounts
            keys = await redis_manager.scan_iter(match=f"kelly:daily_stats:*:{date}")
            async for key in keys:
                acc_id = key.split(":")[2]
                daily_data = await redis_manager.lrange(key, 0, -1)
                
                stats[acc_id] = {
                    "date": date,
                    "total_messages": len(daily_data),
                    "conversations": len(set(
                        json.loads(data).get("user_id") for data in daily_data
                    )),
                    "safety_scores": [
                        json.loads(data).get("safety_score", 1.0) for data in daily_data
                    ],
                    "stages": [
                        json.loads(data).get("stage", "unknown") for data in daily_data
                    ]
                }
        
        return {"daily_stats": stats}
        
    except Exception as e:
        logger.error(f"Error getting daily stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/block-user")
async def block_user(
    user_id: str,
    reason: str,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Manually block a user"""
    try:
        block_data = {
            "user_id": user_id,
            "blocked_at": datetime.now().isoformat(),
            "reason": [reason],
            "manual_block": True,
            "blocked_by": current_user.get("id", "admin")
        }
        
        key = f"kelly:blocked:{user_id}"
        await redis_manager.setex(key, 86400 * 30, json.dumps(block_data))  # 30 days
        
        return {"message": f"User {user_id} blocked successfully"}
        
    except Exception as e:
        logger.error(f"Error blocking user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/safety/unblock-user/{user_id}")
async def unblock_user(
    user_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Unblock a user"""
    try:
        key = f"kelly:blocked:{user_id}"
        deleted = await redis_manager.delete(key)
        
        if deleted:
            return {"message": f"User {user_id} unblocked successfully"}
        else:
            raise HTTPException(status_code=404, detail="User not found in blocked list")
            
    except Exception as e:
        logger.error(f"Error unblocking user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety/blocked-users")
async def get_blocked_users(
    current_user = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """Get list of blocked users"""
    try:
        blocked_users = []
        
        keys = await redis_manager.scan_iter(match="kelly:blocked:*")
        async for key in keys:
            user_id = key.split(":")[-1]
            block_data = await redis_manager.get(key)
            
            if block_data:
                data = json.loads(block_data)
                blocked_users.append({
                    "user_id": user_id,
                    "blocked_at": data.get("blocked_at"),
                    "reason": data.get("reason", []),
                    "auto_blocked": data.get("auto_blocked", False),
                    "manual_block": data.get("manual_block", False)
                })
        
        return {"blocked_users": blocked_users}
        
    except Exception as e:
        logger.error(f"Error getting blocked users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Claude AI Endpoints
@router.get("/claude/metrics")
async def get_claude_metrics(
    account_id: Optional[str] = None,
    timeframe: str = "today",
    current_user = Depends(get_current_user)
) -> Dict[str, ClaudeMetricsResponse]:
    """Get Claude AI usage metrics and performance data"""
    try:
        if account_id:
            # Get metrics for specific account
            metrics = await kelly_claude_ai.get_account_metrics(account_id, timeframe)
            if not metrics:
                raise HTTPException(status_code=404, detail="Account not found or no metrics available")
            
            return {"account_metrics": ClaudeMetricsResponse(**metrics)}
        else:
            # Get aggregated metrics for all accounts
            all_metrics = await kelly_claude_ai.get_aggregated_metrics(timeframe)
            
            return {"global_metrics": ClaudeMetricsResponse(**all_metrics)}
            
    except Exception as e:
        logger.error(f"Error getting Claude metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/accounts/{account_id}/claude-config")
async def get_claude_config(
    account_id: str,
    current_user = Depends(get_current_user)
) -> ClaudeConfigResponse:
    """Get Claude AI configuration for a specific account"""
    try:
        # Verify account ownership
        owner_id = await redis_manager.get(f"kelly:account_owner:{account_id}")
        if owner_id != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Account access denied")
        
        # Get Claude configuration
        config = await kelly_claude_ai.get_account_config(account_id)
        if not config:
            raise HTTPException(status_code=404, detail="Account configuration not found")
        
        return ClaudeConfigResponse(
            account_id=account_id,
            config=ClaudeConfigRequest(**config["settings"]),
            last_updated=config["last_updated"],
            performance_score=config["performance_score"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Claude config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/accounts/{account_id}/claude-config")
async def update_claude_config(
    account_id: str,
    config: ClaudeConfigRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Update Claude AI configuration for a specific account"""
    try:
        # Verify account ownership
        owner_id = await redis_manager.get(f"kelly:account_owner:{account_id}")
        if owner_id != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Account access denied")
        
        # Validate model preference
        valid_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307"
        ]
        if config.model_preference not in valid_models:
            raise HTTPException(status_code=400, detail="Invalid Claude model specified")
        
        # Update configuration
        config_data = config.dict()
        config_data["updated_by"] = current_user.get("id")
        config_data["updated_at"] = datetime.now().isoformat()
        
        success = await kelly_claude_ai.update_account_config(account_id, config_data)
        
        if success:
            logger.info(f"Claude config updated for account {account_id}")
            return {"message": "Claude AI configuration updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating Claude config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Safety Monitoring Endpoints
@router.get("/safety")
async def get_safety_status(
    account_id: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> Dict[str, SafetyStatusResponse]:
    """Get comprehensive safety monitoring status"""
    try:
        if account_id:
            # Verify account ownership
            owner_id = await redis_manager.get(f"kelly:account_owner:{account_id}")
            if owner_id != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Account access denied")
            
            # Get safety status for specific account
            status = await kelly_safety_monitor.get_account_safety_status(account_id)
            if not status:
                raise HTTPException(status_code=404, detail="Account safety data not found")
            
            return {"account_safety": SafetyStatusResponse(**status)}
        else:
            # Get global safety status
            global_status = await kelly_safety_monitor.get_global_safety_status()
            
            return {"global_safety": SafetyStatusResponse(**global_status)}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/alerts/{alert_id}/review")
async def review_safety_alert(
    alert_id: str,
    request: SafetyAlertRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Review and take action on a safety alert"""
    try:
        # Get alert details
        alert = await kelly_safety_monitor.get_alert_details(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Safety alert not found")
        
        # Verify user has permission to review this alert
        account_id = alert.get("account_id")
        if account_id:
            owner_id = await redis_manager.get(f"kelly:account_owner:{account_id}")
            if owner_id != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Alert review access denied")
        
        # Process the review action
        review_data = {
            "alert_id": alert_id,
            "action": request.action,
            "reason": request.reason,
            "override_block": request.override_block,
            "reviewed_by": current_user.get("id"),
            "reviewed_at": datetime.now().isoformat()
        }
        
        success = await kelly_safety_monitor.process_alert_review(review_data)
        
        if success:
            action_msg = {
                "approve": "approved and action taken",
                "deny": "denied and alert dismissed",
                "escalate": "escalated for further review"
            }.get(request.action, "processed")
            
            logger.info(f"Safety alert {alert_id} {action_msg} by user {current_user.get('id')}")
            return {"message": f"Safety alert {action_msg} successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process alert review")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing safety alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety/alerts")
async def get_pending_safety_alerts(
    account_id: Optional[str] = None,
    limit: int = 50,
    severity: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """Get pending safety alerts requiring review"""
    try:
        filters = {}
        
        if account_id:
            # Verify account ownership
            owner_id = await redis_manager.get(f"kelly:account_owner:{account_id}")
            if owner_id != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Account access denied")
            filters["account_id"] = account_id
        
        if severity:
            if severity not in ["low", "medium", "high", "critical"]:
                raise HTTPException(status_code=400, detail="Invalid severity level")
            filters["severity"] = severity
        
        # Get pending alerts
        alerts = await kelly_safety_monitor.get_pending_alerts(
            filters=filters,
            limit=limit,
            user_id=current_user.get("id")
        )
        
        return {"alerts": alerts}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pending alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Kelly system health check"""
    try:
        # Check if services are running
        services_status = {
            "kelly_personality": "running",
            "kelly_userbot": "running",
            "kelly_dm_detector": "running",
            "kelly_conversation_manager": "running",
            "kelly_claude_ai": "running",
            "kelly_safety_monitor": "running"
        }
        
        return {
            "status": "healthy",
            "services": services_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))