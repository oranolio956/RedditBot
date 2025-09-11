"""
Kelly AI Emergency Override API Endpoints

FastAPI endpoints for emergency controls including conversation stops,
system resets, audit logging, and system-wide emergency actions.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from app.core.auth import get_current_user, require_admin
from app.core.redis import redis_manager
from app.services.kelly_conversation_manager import kelly_conversation_manager
from app.services.kelly_safety_monitor import kelly_safety_monitor
from app.services.kelly_telegram_userbot import kelly_userbot
from app.websocket.manager import websocket_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/kelly/emergency", tags=["kelly-emergency"])

# ===== EMERGENCY MODELS =====

class EmergencyStopRequest(BaseModel):
    reason: str
    immediate: bool = True
    notify_user: bool = False
    block_future_messages: bool = False
    escalate_to_admin: bool = False

class EmergencyResetRequest(BaseModel):
    reset_type: str = Field(..., regex="^(conversation|account|system)$")
    reason: str
    preserve_data: bool = True
    backup_before_reset: bool = True
    confirmation_code: Optional[str] = None

class SystemWideStopRequest(BaseModel):
    reason: str
    duration_minutes: Optional[int] = Field(None, ge=1, le=1440)  # Max 24 hours
    notify_all_users: bool = False
    maintenance_mode: bool = False
    confirmation_code: str

class EmergencyActionResponse(BaseModel):
    action_id: str
    action_type: str
    status: str = Field(..., regex="^(initiated|in_progress|completed|failed)$")
    affected_items: List[str]
    initiated_by: str
    initiated_at: str
    completed_at: Optional[str]
    error_message: Optional[str]
    rollback_available: bool

class AuditLogEntry(BaseModel):
    id: str
    action_type: str
    performed_by: str
    performed_by_name: str
    target_type: str  # conversation, account, system
    target_id: str
    reason: str
    timestamp: str
    details: Dict[str, Any]
    severity: str = Field(..., regex="^(low|medium|high|critical)$")

class AuditLogResponse(BaseModel):
    entries: List[AuditLogEntry]
    total_count: int
    filtered_count: int
    date_range: Dict[str, str]
    summary: Dict[str, Any]

class EmergencyStatusResponse(BaseModel):
    emergency_mode_active: bool
    system_wide_stop_active: bool
    maintenance_mode_active: bool
    active_emergency_actions: List[EmergencyActionResponse]
    recent_emergency_actions: List[AuditLogEntry]
    system_health_during_emergency: Dict[str, Any]

# ===== EMERGENCY ENDPOINTS =====

@router.post("/stop/{conversation_id}", response_model=EmergencyActionResponse)
async def emergency_stop_conversation(
    conversation_id: str,
    request: EmergencyStopRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> EmergencyActionResponse:
    """Emergency stop a specific conversation"""
    try:
        # Verify conversation exists
        conv_key = f"kelly:conversation_track:{conversation_id}"
        conversation_data = await redis_manager.get(conv_key)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation = json.loads(conversation_data)
        account_id = conversation.get("account_id")
        user_id = conversation.get("user_id")
        
        # Generate action ID
        action_id = str(uuid4())
        action_start = datetime.now()
        
        # Create emergency action record
        emergency_action = {
            "action_id": action_id,
            "action_type": "emergency_stop_conversation",
            "status": "initiated",
            "conversation_id": conversation_id,
            "account_id": account_id,
            "user_id": user_id,
            "initiated_by": current_user.get("id"),
            "initiated_by_name": current_user.get("name"),
            "initiated_at": action_start.isoformat(),
            "reason": request.reason,
            "immediate": request.immediate,
            "notify_user": request.notify_user,
            "block_future_messages": request.block_future_messages,
            "affected_items": [conversation_id]
        }
        
        # Store action record
        action_key = f"kelly:emergency_action:{action_id}"
        await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))  # 7 days
        
        try:
            # Update action status
            emergency_action["status"] = "in_progress"
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            # Stop AI for this conversation immediately
            await kelly_conversation_manager.emergency_stop_conversation(conversation_id)
            
            # Update conversation status
            conversation.update({
                "emergency_stopped": True,
                "emergency_stop_reason": request.reason,
                "emergency_stopped_by": current_user.get("id"),
                "emergency_stopped_at": action_start.isoformat(),
                "ai_paused": True,
                "human_controlled": True
            })
            
            await redis_manager.setex(conv_key, 86400, json.dumps(conversation))
            
            # Block future messages if requested
            if request.block_future_messages and account_id and user_id:
                block_data = {
                    "user_id": user_id,
                    "blocked_at": action_start.isoformat(),
                    "reason": [f"Emergency stop: {request.reason}"],
                    "emergency_block": True,
                    "blocked_by": current_user.get("id")
                }
                
                block_key = f"kelly:blocked:{user_id}"
                await redis_manager.setex(block_key, 86400 * 30, json.dumps(block_data))  # 30 days
            
            # Notify user if requested
            if request.notify_user and account_id and user_id:
                background_tasks.add_task(
                    _send_emergency_stop_notification,
                    account_id,
                    user_id,
                    current_user.get("name", "Support Team")
                )
            
            # Escalate to admin if requested
            if request.escalate_to_admin:
                background_tasks.add_task(
                    _escalate_emergency_to_admin,
                    action_id,
                    conversation_id,
                    request.reason,
                    current_user.get("id")
                )
            
            # Complete action
            emergency_action.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "rollback_available": True
            })
            
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            # Log to audit trail
            background_tasks.add_task(
                _log_emergency_action,
                emergency_action,
                "critical"
            )
            
            # Broadcast emergency update
            background_tasks.add_task(
                _broadcast_emergency_update,
                "conversation_emergency_stop",
                {
                    "conversation_id": conversation_id,
                    "action_id": action_id,
                    "reason": request.reason,
                    "performed_by": current_user.get("name")
                }
            )
            
            return EmergencyActionResponse(
                action_id=action_id,
                action_type="emergency_stop_conversation",
                status="completed",
                affected_items=[conversation_id],
                initiated_by=current_user.get("name"),
                initiated_at=action_start.isoformat(),
                completed_at=emergency_action["completed_at"],
                rollback_available=True
            )
            
        except Exception as e:
            # Mark action as failed
            emergency_action.update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat(),
                "rollback_available": False
            })
            
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            logger.error(f"Emergency stop failed for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in emergency stop for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset/{conversation_id}", response_model=EmergencyActionResponse)
async def emergency_reset_conversation(
    conversation_id: str,
    request: EmergencyResetRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> EmergencyActionResponse:
    """Emergency reset conversation state"""
    try:
        # Verify conversation exists
        conv_key = f"kelly:conversation_track:{conversation_id}"
        conversation_data = await redis_manager.get(conv_key)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation = json.loads(conversation_data)
        
        # Generate action ID
        action_id = str(uuid4())
        action_start = datetime.now()
        
        # Create backup if requested
        backup_key = None
        if request.backup_before_reset:
            backup_key = f"kelly:conversation_backup:{conversation_id}:{int(action_start.timestamp())}"
            await redis_manager.setex(backup_key, 86400 * 30, json.dumps(conversation))  # 30 days
        
        # Create emergency action record
        emergency_action = {
            "action_id": action_id,
            "action_type": f"emergency_reset_{request.reset_type}",
            "status": "initiated",
            "conversation_id": conversation_id,
            "initiated_by": current_user.get("id"),
            "initiated_by_name": current_user.get("name"),
            "initiated_at": action_start.isoformat(),
            "reason": request.reason,
            "reset_type": request.reset_type,
            "preserve_data": request.preserve_data,
            "backup_key": backup_key,
            "affected_items": [conversation_id]
        }
        
        # Store action record
        action_key = f"kelly:emergency_action:{action_id}"
        await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
        
        try:
            # Update action status
            emergency_action["status"] = "in_progress"
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            if request.reset_type == "conversation":
                # Reset conversation state
                reset_conversation = {
                    "conversation_id": conversation_id,
                    "account_id": conversation.get("account_id"),
                    "user_id": conversation.get("user_id"),
                    "stage": "initial",
                    "message_count": 0,
                    "safety_score": 1.0,
                    "trust_level": 0.0,
                    "engagement_quality": 0.0,
                    "red_flags": [],
                    "ai_confidence": 0.7,
                    "human_controlled": False,
                    "ai_paused": False,
                    "reset_at": action_start.isoformat(),
                    "reset_by": current_user.get("id"),
                    "reset_reason": request.reason
                }
                
                # Preserve certain data if requested
                if request.preserve_data:
                    reset_conversation.update({
                        "original_first_contact": conversation.get("first_contact"),
                        "original_start_time": conversation.get("start_time"),
                        "total_messages_before_reset": conversation.get("message_count", 0)
                    })
                
                await redis_manager.setex(conv_key, 86400, json.dumps(reset_conversation))
                
                # Reset conversation in AI systems
                await kelly_conversation_manager.reset_conversation_state(conversation_id)
                
            elif request.reset_type == "account":
                account_id = conversation.get("account_id")
                if account_id:
                    # Reset entire account state
                    await kelly_userbot.emergency_reset_account(account_id, request.preserve_data)
                    emergency_action["affected_items"] = [f"account_{account_id}"]
                else:
                    raise HTTPException(status_code=400, detail="Account ID not found in conversation")
                
            elif request.reset_type == "system":
                # Require admin permissions for system reset
                if not current_user.get("is_admin", False):
                    raise HTTPException(status_code=403, detail="Admin access required for system reset")
                
                # Require confirmation code for system reset
                if not request.confirmation_code or request.confirmation_code != "EMERGENCY_SYSTEM_RESET":
                    raise HTTPException(status_code=400, detail="Invalid confirmation code for system reset")
                
                # Perform system-wide reset
                await _perform_system_reset(request.preserve_data)
                emergency_action["affected_items"] = ["system_wide"]
            
            # Complete action
            emergency_action.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "rollback_available": backup_key is not None
            })
            
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            # Log to audit trail
            background_tasks.add_task(
                _log_emergency_action,
                emergency_action,
                "high"
            )
            
            # Broadcast emergency update
            background_tasks.add_task(
                _broadcast_emergency_update,
                f"emergency_reset_{request.reset_type}",
                {
                    "action_id": action_id,
                    "reset_type": request.reset_type,
                    "affected_items": emergency_action["affected_items"],
                    "reason": request.reason,
                    "performed_by": current_user.get("name")
                }
            )
            
            return EmergencyActionResponse(
                action_id=action_id,
                action_type=f"emergency_reset_{request.reset_type}",
                status="completed",
                affected_items=emergency_action["affected_items"],
                initiated_by=current_user.get("name"),
                initiated_at=action_start.isoformat(),
                completed_at=emergency_action["completed_at"],
                rollback_available=emergency_action["rollback_available"]
            )
            
        except Exception as e:
            # Mark action as failed
            emergency_action.update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat(),
                "rollback_available": backup_key is not None
            })
            
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            logger.error(f"Emergency reset failed for {request.reset_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Emergency reset failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in emergency reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system-wide-stop", response_model=EmergencyActionResponse)
async def system_wide_emergency_stop(
    request: SystemWideStopRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_admin)  # Admin only
) -> EmergencyActionResponse:
    """Emergency stop all conversations system-wide (admin only)"""
    try:
        # Verify confirmation code
        if request.confirmation_code != "EMERGENCY_SYSTEM_STOP":
            raise HTTPException(status_code=400, detail="Invalid confirmation code")
        
        # Generate action ID
        action_id = str(uuid4())
        action_start = datetime.now()
        
        # Calculate stop duration
        stop_until = None
        if request.duration_minutes:
            stop_until = action_start + timedelta(minutes=request.duration_minutes)
        
        # Create emergency action record
        emergency_action = {
            "action_id": action_id,
            "action_type": "system_wide_emergency_stop",
            "status": "initiated",
            "initiated_by": current_user.get("id"),
            "initiated_by_name": current_user.get("name"),
            "initiated_at": action_start.isoformat(),
            "reason": request.reason,
            "duration_minutes": request.duration_minutes,
            "stop_until": stop_until.isoformat() if stop_until else None,
            "notify_all_users": request.notify_all_users,
            "maintenance_mode": request.maintenance_mode,
            "affected_items": ["system_wide"]
        }
        
        # Store action record
        action_key = f"kelly:emergency_action:{action_id}"
        await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
        
        try:
            # Update action status
            emergency_action["status"] = "in_progress"
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            # Set system-wide emergency stop flag
            stop_data = {
                "active": True,
                "action_id": action_id,
                "started_at": action_start.isoformat(),
                "stop_until": stop_until.isoformat() if stop_until else None,
                "reason": request.reason,
                "initiated_by": current_user.get("id"),
                "maintenance_mode": request.maintenance_mode
            }
            
            system_stop_key = "kelly:system_emergency_stop"
            ttl = int(request.duration_minutes * 60) if request.duration_minutes else 86400  # Default 24h
            await redis_manager.setex(system_stop_key, ttl, json.dumps(stop_data))
            
            # Stop all active conversations
            active_conv_keys = await redis_manager.keys("kelly:conversation_track:*")
            stopped_conversations = []
            
            for conv_key in active_conv_keys:
                try:
                    conversation_data = await redis_manager.get(conv_key)
                    if conversation_data:
                        conversation = json.loads(conversation_data)
                        conversation_id = conversation.get("conversation_id")
                        
                        if conversation_id:
                            # Stop the conversation
                            await kelly_conversation_manager.emergency_stop_conversation(conversation_id)
                            
                            # Update conversation status
                            conversation.update({
                                "system_emergency_stopped": True,
                                "system_stop_action_id": action_id,
                                "system_stopped_at": action_start.isoformat(),
                                "ai_paused": True
                            })
                            
                            await redis_manager.setex(conv_key, 86400, json.dumps(conversation))
                            stopped_conversations.append(conversation_id)
                            
                except Exception as e:
                    logger.warning(f"Failed to stop conversation {conv_key}: {e}")
            
            # Set maintenance mode if requested
            if request.maintenance_mode:
                maintenance_data = {
                    "active": True,
                    "started_at": action_start.isoformat(),
                    "reason": request.reason,
                    "initiated_by": current_user.get("name")
                }
                
                maintenance_key = "kelly:maintenance_mode"
                await redis_manager.setex(maintenance_key, ttl, json.dumps(maintenance_data))
            
            # Notify all users if requested
            if request.notify_all_users:
                background_tasks.add_task(
                    _notify_all_users_emergency_stop,
                    request.reason,
                    request.maintenance_mode,
                    stop_until
                )
            
            # Complete action
            emergency_action.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "stopped_conversations_count": len(stopped_conversations),
                "rollback_available": True
            })
            
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            # Log to audit trail
            background_tasks.add_task(
                _log_emergency_action,
                emergency_action,
                "critical"
            )
            
            # Broadcast emergency update
            background_tasks.add_task(
                _broadcast_emergency_update,
                "system_wide_emergency_stop",
                {
                    "action_id": action_id,
                    "reason": request.reason,
                    "duration_minutes": request.duration_minutes,
                    "maintenance_mode": request.maintenance_mode,
                    "stopped_conversations": len(stopped_conversations),
                    "performed_by": current_user.get("name")
                }
            )
            
            return EmergencyActionResponse(
                action_id=action_id,
                action_type="system_wide_emergency_stop",
                status="completed",
                affected_items=[f"system_wide_{len(stopped_conversations)}_conversations"],
                initiated_by=current_user.get("name"),
                initiated_at=action_start.isoformat(),
                completed_at=emergency_action["completed_at"],
                rollback_available=True
            )
            
        except Exception as e:
            # Mark action as failed
            emergency_action.update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat(),
                "rollback_available": False
            })
            
            await redis_manager.setex(action_key, 86400 * 7, json.dumps(emergency_action))
            
            logger.error(f"System-wide emergency stop failed: {e}")
            raise HTTPException(status_code=500, detail=f"System-wide emergency stop failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in system-wide emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit-log", response_model=AuditLogResponse)
async def get_emergency_audit_log(
    limit: int = 100,
    days: int = 7,
    action_type: Optional[str] = None,
    severity: Optional[str] = None,
    performed_by: Optional[str] = None,
    current_user = Depends(get_current_user)
) -> AuditLogResponse:
    """Get emergency action audit log"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get audit log entries
        log_keys = await redis_manager.keys("kelly:emergency_audit:*")
        
        entries = []
        total_count = 0
        
        for key in log_keys:
            log_data = await redis_manager.get(key)
            if log_data:
                entry_data = json.loads(log_data)
                entry_timestamp = datetime.fromisoformat(entry_data["timestamp"])
                
                # Check if in date range
                if start_date <= entry_timestamp <= end_date:
                    total_count += 1
                    
                    # Apply filters
                    if action_type and entry_data.get("action_type") != action_type:
                        continue
                    
                    if severity and entry_data.get("severity") != severity:
                        continue
                    
                    if performed_by and entry_data.get("performed_by") != performed_by:
                        continue
                    
                    # Check permissions (non-admin users only see their own actions)
                    if not current_user.get("is_admin", False):
                        if entry_data.get("performed_by") != current_user.get("id"):
                            continue
                    
                    entries.append(AuditLogEntry(**entry_data))
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        filtered_count = len(entries)
        entries = entries[:limit]
        
        # Generate summary
        summary = {
            "total_actions": total_count,
            "filtered_actions": filtered_count,
            "actions_by_type": {},
            "actions_by_severity": {},
            "most_active_user": None,
            "peak_activity_day": None
        }
        
        # Calculate summary stats
        for entry in entries:
            # Count by type
            action_type = entry.action_type
            summary["actions_by_type"][action_type] = summary["actions_by_type"].get(action_type, 0) + 1
            
            # Count by severity
            severity = entry.severity
            summary["actions_by_severity"][severity] = summary["actions_by_severity"].get(severity, 0) + 1
        
        return AuditLogResponse(
            entries=entries,
            total_count=total_count,
            filtered_count=filtered_count,
            date_range={
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error getting emergency audit log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=EmergencyStatusResponse)
async def get_emergency_status(
    current_user = Depends(get_current_user)
) -> EmergencyStatusResponse:
    """Get current emergency system status"""
    try:
        # Check system-wide emergency stop
        system_stop_data = await redis_manager.get("kelly:system_emergency_stop")
        system_wide_stop_active = bool(system_stop_data)
        
        # Check maintenance mode
        maintenance_data = await redis_manager.get("kelly:maintenance_mode")
        maintenance_mode_active = bool(maintenance_data)
        
        # Emergency mode is active if either stop or maintenance is active
        emergency_mode_active = system_wide_stop_active or maintenance_mode_active
        
        # Get active emergency actions
        action_keys = await redis_manager.keys("kelly:emergency_action:*")
        active_actions = []
        
        for key in action_keys:
            action_data = await redis_manager.get(key)
            if action_data:
                action = json.loads(action_data)
                
                # Check if action is still active/in progress
                if action.get("status") in ["initiated", "in_progress"]:
                    active_actions.append(EmergencyActionResponse(
                        action_id=action["action_id"],
                        action_type=action["action_type"],
                        status=action["status"],
                        affected_items=action.get("affected_items", []),
                        initiated_by=action.get("initiated_by_name", "Unknown"),
                        initiated_at=action["initiated_at"],
                        completed_at=action.get("completed_at"),
                        error_message=action.get("error_message"),
                        rollback_available=action.get("rollback_available", False)
                    ))
        
        # Get recent emergency actions
        recent_audit_keys = await redis_manager.keys("kelly:emergency_audit:*")
        recent_actions = []
        
        # Sort and get most recent
        recent_audit_keys.sort(reverse=True)
        
        for key in recent_audit_keys[:10]:  # Last 10 actions
            audit_data = await redis_manager.get(key)
            if audit_data:
                audit_entry = json.loads(audit_data)
                
                # Check permissions
                if not current_user.get("is_admin", False):
                    if audit_entry.get("performed_by") != current_user.get("id"):
                        continue
                
                recent_actions.append(AuditLogEntry(**audit_entry))
        
        # Get system health during emergency
        system_health = {}
        if emergency_mode_active:
            try:
                # Basic health metrics during emergency
                active_conversations = await redis_manager.keys("kelly:conversation_track:*")
                redis_info = await redis_manager.info()
                
                system_health = {
                    "active_conversations": len(active_conversations),
                    "redis_memory_usage": redis_info.get("used_memory", 0),
                    "redis_connected_clients": redis_info.get("connected_clients", 0),
                    "emergency_start_time": system_stop_data and json.loads(system_stop_data).get("started_at"),
                    "maintenance_start_time": maintenance_data and json.loads(maintenance_data).get("started_at")
                }
            except Exception as e:
                logger.warning(f"Could not get system health during emergency: {e}")
                system_health = {"error": "Unable to retrieve system health"}
        
        return EmergencyStatusResponse(
            emergency_mode_active=emergency_mode_active,
            system_wide_stop_active=system_wide_stop_active,
            maintenance_mode_active=maintenance_mode_active,
            active_emergency_actions=active_actions,
            recent_emergency_actions=recent_actions[:5],  # Most recent 5
            system_health_during_emergency=system_health
        )
        
    except Exception as e:
        logger.error(f"Error getting emergency status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HELPER FUNCTIONS =====

async def _send_emergency_stop_notification(
    account_id: str,
    user_id: str,
    operator_name: str
):
    """Send emergency stop notification to user"""
    try:
        message = f"⚠️ This conversation has been temporarily paused by {operator_name} for your safety. Please wait while we review the situation."
        
        await kelly_userbot.send_message(
            account_id=account_id,
            user_id=user_id,
            message=message,
            is_emergency=True
        )
        
    except Exception as e:
        logger.error(f"Error sending emergency stop notification: {e}")

async def _escalate_emergency_to_admin(
    action_id: str,
    conversation_id: str,
    reason: str,
    operator_id: str
):
    """Escalate emergency action to admin"""
    try:
        escalation_data = {
            "id": str(uuid4()),
            "type": "emergency_escalation",
            "action_id": action_id,
            "conversation_id": conversation_id,
            "reason": reason,
            "escalated_by": operator_id,
            "escalated_at": datetime.now().isoformat(),
            "severity": "critical"
        }
        
        escalation_key = f"kelly:admin_escalation:{escalation_data['id']}"
        await redis_manager.setex(escalation_key, 86400 * 3, json.dumps(escalation_data))  # 3 days
        
        # Broadcast to admin dashboard
        await websocket_manager.broadcast_to_room(
            "admin_dashboard",
            {
                "type": "emergency_escalation",
                "data": escalation_data
            }
        )
        
    except Exception as e:
        logger.error(f"Error escalating emergency to admin: {e}")

async def _perform_system_reset(preserve_data: bool):
    """Perform system-wide emergency reset"""
    try:
        if preserve_data:
            # Selective reset - keep user data, reset AI state
            await kelly_conversation_manager.reset_all_ai_states()
            
            # Reset conversation tracking but preserve history
            conv_keys = await redis_manager.keys("kelly:conversation_track:*")
            for key in conv_keys:
                conv_data = await redis_manager.get(key)
                if conv_data:
                    conversation = json.loads(conv_data)
                    
                    # Reset AI state but preserve core data
                    conversation.update({
                        "ai_confidence": 0.7,
                        "stage": "initial",
                        "ai_paused": False,
                        "human_controlled": False,
                        "emergency_stopped": False,
                        "system_reset_at": datetime.now().isoformat()
                    })
                    
                    await redis_manager.setex(key, 86400, json.dumps(conversation))
        else:
            # Full reset - clear all conversation data
            await redis_manager.flushdb()
            logger.warning("FULL SYSTEM RESET PERFORMED - ALL DATA CLEARED")
            
    except Exception as e:
        logger.error(f"Error performing system reset: {e}")
        raise

async def _notify_all_users_emergency_stop(
    reason: str,
    maintenance_mode: bool,
    stop_until: Optional[datetime]
):
    """Notify all active users about emergency stop"""
    try:
        # Get all active conversations
        conv_keys = await redis_manager.keys("kelly:conversation_track:*")
        
        for key in conv_keys:
            try:
                conv_data = await redis_manager.get(key)
                if conv_data:
                    conversation = json.loads(conv_data)
                    account_id = conversation.get("account_id")
                    user_id = conversation.get("user_id")
                    
                    if account_id and user_id:
                        if maintenance_mode:
                            message = f"🔧 We're performing emergency maintenance. Service will be restored as soon as possible. Reason: {reason}"
                        else:
                            message = f"⚠️ Emergency system pause activated. "
                            if stop_until:
                                message += f"Service will resume at {stop_until.strftime('%H:%M')}. "
                            message += f"Reason: {reason}"
                        
                        await kelly_userbot.send_message(
                            account_id=account_id,
                            user_id=user_id,
                            message=message,
                            is_emergency=True
                        )
                        
            except Exception as e:
                logger.warning(f"Failed to notify user in conversation {key}: {e}")
                
    except Exception as e:
        logger.error(f"Error notifying all users of emergency stop: {e}")

async def _log_emergency_action(
    action_data: Dict[str, Any],
    severity: str
):
    """Log emergency action to audit trail"""
    try:
        audit_entry = {
            "id": str(uuid4()),
            "action_type": action_data["action_type"],
            "performed_by": action_data["initiated_by"],
            "performed_by_name": action_data["initiated_by_name"],
            "target_type": _get_target_type(action_data["action_type"]),
            "target_id": action_data.get("conversation_id", action_data.get("action_id")),
            "reason": action_data["reason"],
            "timestamp": action_data["initiated_at"],
            "details": {
                "action_id": action_data["action_id"],
                "status": action_data["status"],
                "affected_items": action_data.get("affected_items", []),
                "completed_at": action_data.get("completed_at"),
                "error_message": action_data.get("error_message")
            },
            "severity": severity
        }
        
        audit_key = f"kelly:emergency_audit:{int(datetime.now().timestamp())}:{audit_entry['id']}"
        await redis_manager.setex(audit_key, 86400 * 90, json.dumps(audit_entry))  # 90 days
        
    except Exception as e:
        logger.error(f"Error logging emergency action to audit trail: {e}")

def _get_target_type(action_type: str) -> str:
    """Get target type from action type"""
    if "conversation" in action_type:
        return "conversation"
    elif "account" in action_type:
        return "account"
    elif "system" in action_type:
        return "system"
    else:
        return "unknown"

async def _broadcast_emergency_update(
    update_type: str,
    data: Dict[str, Any]
):
    """Broadcast emergency update via WebSocket"""
    try:
        message = {
            "type": "emergency_update",
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "severity": "critical"
        }
        
        # Broadcast to all monitoring rooms
        await websocket_manager.broadcast_to_room("monitoring_dashboard", message)
        await websocket_manager.broadcast_to_room("admin_dashboard", message)
        await websocket_manager.broadcast_to_room("emergency_monitoring", message)
        
    except Exception as e:
        logger.error(f"Error broadcasting emergency update: {e}")