"""
Kelly AI Intervention API Endpoints

FastAPI endpoints for human intervention management, including taking control,
releasing control, AI suggestions, and intervention status tracking.
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

from app.core.auth import get_current_user
from app.core.redis import redis_manager
from app.services.kelly_conversation_manager import kelly_conversation_manager
from app.services.kelly_claude_ai import kelly_claude_ai
from app.services.kelly_safety_monitor import kelly_safety_monitor
from app.websocket.manager import websocket_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/kelly/intervention", tags=["kelly-intervention"])

# ===== INTERVENTION MODELS =====

class InterventionStatusResponse(BaseModel):
    conversation_id: str
    status: str = Field(..., regex="^(ai_active|human_reviewing|human_active|paused|completed)$")
    human_operator: Optional[str]
    takeover_time: Optional[str]
    release_time: Optional[str]
    ai_confidence: float
    last_ai_suggestion: Optional[str]
    intervention_reason: str
    can_release: bool
    messages_during_intervention: int
    intervention_duration: Optional[float]  # in seconds
    handoff_context: Dict[str, Any]

class TakeControlRequest(BaseModel):
    reason: str
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    notify_user: bool = False
    preserve_ai_suggestions: bool = True

class ReleaseControlRequest(BaseModel):
    summary: str
    ai_handoff_instructions: Optional[str] = None
    mark_resolved: bool = False

class SuggestResponseRequest(BaseModel):
    context: Optional[str] = None
    tone: str = Field(default="professional", regex="^(professional|casual|empathetic|firm)$")
    max_length: int = Field(default=500, ge=50, le=2000)
    include_reasoning: bool = True

class AIResponseSuggestion(BaseModel):
    suggested_response: str
    confidence_score: float
    reasoning: str
    alternative_suggestions: List[str]
    safety_notes: List[str]
    tone_analysis: Dict[str, float]

class InterventionHistoryResponse(BaseModel):
    interventions: List[Dict[str, Any]]
    total_count: int
    success_rate: float
    avg_duration: float
    common_reasons: List[Dict[str, Any]]

# ===== INTERVENTION ENDPOINTS =====

@router.post("/take-control/{conversation_id}", response_model=InterventionStatusResponse)
async def take_control(
    conversation_id: str,
    request: TakeControlRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> InterventionStatusResponse:
    """Human operator takes control of a conversation"""
    try:
        # Validate conversation exists
        conv_key = f"kelly:conversation_track:{conversation_id}"
        conversation_data = await redis_manager.get(conv_key)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation = json.loads(conversation_data)
        
        # Check if already under human control
        intervention_key = f"kelly:intervention:{conversation_id}"
        existing_intervention = await redis_manager.get(intervention_key)
        
        if existing_intervention:
            intervention = json.loads(existing_intervention)
            if intervention.get("status") in ["human_reviewing", "human_active"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Conversation already under control by {intervention.get('human_operator')}"
                )
        
        # Get current AI confidence and context
        ai_confidence = conversation.get("ai_confidence", 0.0)
        
        # Create intervention record
        intervention_id = str(uuid4())
        takeover_time = datetime.now()
        
        intervention_data = {
            "id": intervention_id,
            "conversation_id": conversation_id,
            "account_id": conversation.get("account_id"),
            "user_id": conversation.get("user_id"),
            "human_operator": current_user.get("id"),
            "human_operator_name": current_user.get("name", "Unknown"),
            "status": "human_active",
            "takeover_time": takeover_time.isoformat(),
            "intervention_reason": request.reason,
            "priority": request.priority,
            "ai_confidence_at_takeover": ai_confidence,
            "preserve_ai_suggestions": request.preserve_ai_suggestions,
            "notify_user": request.notify_user,
            "messages_during_intervention": 0,
            "handoff_context": {
                "conversation_stage": conversation.get("stage", "unknown"),
                "last_ai_response": conversation.get("last_ai_response"),
                "safety_score": conversation.get("safety_score", 1.0),
                "engagement_quality": conversation.get("engagement_quality", 0.0),
                "red_flags": conversation.get("red_flags", [])
            }
        }
        
        # Store intervention data
        await redis_manager.setex(intervention_key, 86400 * 7, json.dumps(intervention_data))  # 7 days
        
        # Update conversation status
        conversation["human_controlled"] = True
        conversation["intervention_id"] = intervention_id
        conversation["ai_paused"] = True
        
        await redis_manager.setex(conv_key, 86400, json.dumps(conversation))
        
        # Pause AI for this conversation
        await kelly_conversation_manager.pause_ai_for_conversation(conversation_id)
        
        # Log activity
        background_tasks.add_task(
            _log_intervention_activity,
            "intervention_started",
            conversation_id,
            current_user.get("id"),
            {
                "reason": request.reason,
                "priority": request.priority,
                "ai_confidence": ai_confidence
            }
        )
        
        # Notify via WebSocket
        background_tasks.add_task(
            _broadcast_intervention_update,
            conversation_id,
            "human_takeover",
            {
                "operator": current_user.get("name"),
                "reason": request.reason,
                "timestamp": takeover_time.isoformat()
            }
        )
        
        # Optionally notify the user in conversation
        if request.notify_user:
            background_tasks.add_task(
                _notify_user_of_intervention,
                conversation_id,
                current_user.get("name", "Support")
            )
        
        # Get AI suggestion for immediate context if requested
        last_ai_suggestion = None
        if request.preserve_ai_suggestions:
            try:
                suggestion = await kelly_claude_ai.get_intervention_suggestion(
                    conversation_id,
                    context="Human operator taking control",
                    include_reasoning=True
                )
                last_ai_suggestion = suggestion.get("suggested_response")
                
                # Store the suggestion
                suggestion_key = f"kelly:ai_suggestion:{conversation_id}:latest"
                await redis_manager.setex(suggestion_key, 3600, json.dumps(suggestion))  # 1 hour
                
            except Exception as e:
                logger.warning(f"Could not get AI suggestion during takeover: {e}")
        
        return InterventionStatusResponse(
            conversation_id=conversation_id,
            status="human_active",
            human_operator=current_user.get("name"),
            takeover_time=takeover_time.isoformat(),
            ai_confidence=ai_confidence,
            last_ai_suggestion=last_ai_suggestion,
            intervention_reason=request.reason,
            can_release=True,
            messages_during_intervention=0,
            handoff_context=intervention_data["handoff_context"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error taking control of conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/release-control/{conversation_id}")
async def release_control(
    conversation_id: str,
    request: ReleaseControlRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Release conversation back to AI control"""
    try:
        # Get intervention data
        intervention_key = f"kelly:intervention:{conversation_id}"
        intervention_data = await redis_manager.get(intervention_key)
        
        if not intervention_data:
            raise HTTPException(status_code=404, detail="No active intervention found")
        
        intervention = json.loads(intervention_data)
        
        # Verify operator is the one who took control or has admin rights
        if (intervention.get("human_operator") != current_user.get("id") and 
            not current_user.get("is_admin", False)):
            raise HTTPException(status_code=403, detail="Not authorized to release this intervention")
        
        # Update intervention record
        release_time = datetime.now()
        takeover_time = datetime.fromisoformat(intervention["takeover_time"])
        intervention_duration = (release_time - takeover_time).total_seconds()
        
        intervention.update({
            "status": "completed" if request.mark_resolved else "ai_active",
            "release_time": release_time.isoformat(),
            "intervention_duration": intervention_duration,
            "release_summary": request.summary,
            "ai_handoff_instructions": request.ai_handoff_instructions,
            "marked_resolved": request.mark_resolved
        })
        
        # Store updated intervention
        await redis_manager.setex(intervention_key, 86400 * 30, json.dumps(intervention))  # 30 days
        
        # Update conversation status
        conv_key = f"kelly:conversation_track:{conversation_id}"
        conversation_data = await redis_manager.get(conv_key)
        
        if conversation_data:
            conversation = json.loads(conversation_data)
            conversation["human_controlled"] = False
            conversation["ai_paused"] = False
            conversation["last_intervention_summary"] = request.summary
            
            if request.ai_handoff_instructions:
                conversation["ai_handoff_instructions"] = request.ai_handoff_instructions
            
            await redis_manager.setex(conv_key, 86400, json.dumps(conversation))
        
        # Resume AI if not marked as resolved
        if not request.mark_resolved:
            await kelly_conversation_manager.resume_ai_for_conversation(
                conversation_id,
                handoff_instructions=request.ai_handoff_instructions
            )
        
        # Log activity
        background_tasks.add_task(
            _log_intervention_activity,
            "intervention_completed",
            conversation_id,
            current_user.get("id"),
            {
                "duration": intervention_duration,
                "summary": request.summary,
                "marked_resolved": request.mark_resolved,
                "messages_handled": intervention.get("messages_during_intervention", 0)
            }
        )
        
        # Broadcast update
        background_tasks.add_task(
            _broadcast_intervention_update,
            conversation_id,
            "human_release",
            {
                "operator": current_user.get("name"),
                "duration": intervention_duration,
                "summary": request.summary,
                "timestamp": release_time.isoformat()
            }
        )
        
        # Archive intervention to history
        background_tasks.add_task(
            _archive_intervention,
            intervention
        )
        
        status = "resolved" if request.mark_resolved else "returned to AI"
        return {"message": f"Conversation {status} successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error releasing control of conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{conversation_id}", response_model=InterventionStatusResponse)
async def get_intervention_status(
    conversation_id: str,
    current_user = Depends(get_current_user)
) -> InterventionStatusResponse:
    """Get current intervention status for a conversation"""
    try:
        # Get intervention data
        intervention_key = f"kelly:intervention:{conversation_id}"
        intervention_data = await redis_manager.get(intervention_key)
        
        if not intervention_data:
            # No intervention, check if conversation exists
            conv_key = f"kelly:conversation_track:{conversation_id}"
            if not await redis_manager.get(conv_key):
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Return AI-active status
            return InterventionStatusResponse(
                conversation_id=conversation_id,
                status="ai_active",
                human_operator=None,
                takeover_time=None,
                ai_confidence=0.7,  # Default
                last_ai_suggestion=None,
                intervention_reason="No intervention",
                can_release=False,
                messages_during_intervention=0,
                handoff_context={}
            )
        
        intervention = json.loads(intervention_data)
        
        # Calculate duration if active
        intervention_duration = None
        if intervention.get("takeover_time") and intervention.get("status") in ["human_reviewing", "human_active"]:
            takeover_time = datetime.fromisoformat(intervention["takeover_time"])
            intervention_duration = (datetime.now() - takeover_time).total_seconds()
        elif intervention.get("release_time"):
            takeover_time = datetime.fromisoformat(intervention["takeover_time"])
            release_time = datetime.fromisoformat(intervention["release_time"])
            intervention_duration = (release_time - takeover_time).total_seconds()
        
        # Get latest AI suggestion if available
        suggestion_key = f"kelly:ai_suggestion:{conversation_id}:latest"
        suggestion_data = await redis_manager.get(suggestion_key)
        last_ai_suggestion = None
        
        if suggestion_data:
            suggestion = json.loads(suggestion_data)
            last_ai_suggestion = suggestion.get("suggested_response")
        
        # Check if user can release (must be the operator or admin)
        can_release = (
            intervention.get("human_operator") == current_user.get("id") or
            current_user.get("is_admin", False)
        ) and intervention.get("status") in ["human_reviewing", "human_active"]
        
        return InterventionStatusResponse(
            conversation_id=conversation_id,
            status=intervention.get("status", "unknown"),
            human_operator=intervention.get("human_operator_name"),
            takeover_time=intervention.get("takeover_time"),
            release_time=intervention.get("release_time"),
            ai_confidence=intervention.get("ai_confidence_at_takeover", 0.0),
            last_ai_suggestion=last_ai_suggestion,
            intervention_reason=intervention.get("intervention_reason", ""),
            can_release=can_release,
            messages_during_intervention=intervention.get("messages_during_intervention", 0),
            intervention_duration=intervention_duration,
            handoff_context=intervention.get("handoff_context", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting intervention status for {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/suggest-response", response_model=AIResponseSuggestion)
async def suggest_response(
    conversation_id: str,
    request: SuggestResponseRequest,
    current_user = Depends(get_current_user)
) -> AIResponseSuggestion:
    """Get AI suggestion for human operator review"""
    try:
        # Verify conversation exists and is under human control
        intervention_key = f"kelly:intervention:{conversation_id}"
        intervention_data = await redis_manager.get(intervention_key)
        
        if not intervention_data:
            raise HTTPException(status_code=404, detail="No active intervention found")
        
        intervention = json.loads(intervention_data)
        
        # Verify user has access
        if (intervention.get("human_operator") != current_user.get("id") and 
            not current_user.get("is_admin", False)):
            raise HTTPException(status_code=403, detail="Not authorized to access this intervention")
        
        # Get conversation context
        conv_key = f"kelly:conversation_track:{conversation_id}"
        conversation_data = await redis_manager.get(conv_key)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation data not found")
        
        conversation = json.loads(conversation_data)
        
        # Get recent message history for context
        account_id = conversation.get("account_id")
        user_id = conversation.get("user_id")
        
        if not account_id or not user_id:
            raise HTTPException(status_code=400, detail="Invalid conversation data")
        
        # Get conversation history
        history = await kelly_userbot.get_conversation_history(account_id, user_id, limit=10)
        
        # Prepare context for AI
        context_data = {
            "conversation_history": history,
            "intervention_reason": intervention.get("intervention_reason"),
            "safety_score": conversation.get("safety_score", 1.0),
            "engagement_quality": conversation.get("engagement_quality", 0.0),
            "user_context": request.context,
            "desired_tone": request.tone,
            "max_length": request.max_length
        }
        
        # Get AI suggestion
        suggestion_response = await kelly_claude_ai.get_intervention_suggestion(
            conversation_id,
            context=json.dumps(context_data),
            tone=request.tone,
            max_length=request.max_length,
            include_reasoning=request.include_reasoning
        )
        
        # Analyze safety and tone
        safety_notes = []
        
        # Check for potential safety issues
        suggested_text = suggestion_response.get("suggested_response", "")
        
        if any(word in suggested_text.lower() for word in ["meet", "phone", "personal", "address"]):
            safety_notes.append("Contains potentially personal information requests")
        
        if len(suggested_text) > request.max_length:
            safety_notes.append(f"Response exceeds requested length ({len(suggested_text)} > {request.max_length})")
        
        # Tone analysis (simplified)
        tone_analysis = {
            "professional": 0.8 if request.tone == "professional" else 0.3,
            "casual": 0.8 if request.tone == "casual" else 0.2,
            "empathetic": 0.7 if request.tone == "empathetic" else 0.4,
            "firm": 0.8 if request.tone == "firm" else 0.3
        }
        
        # Generate alternative suggestions
        alternatives = []
        if suggestion_response.get("alternatives"):
            alternatives = suggestion_response["alternatives"][:3]  # Max 3 alternatives
        
        # Store suggestion for audit trail
        suggestion_id = str(uuid4())
        suggestion_audit = {
            "id": suggestion_id,
            "conversation_id": conversation_id,
            "requested_by": current_user.get("id"),
            "request_context": request.dict(),
            "suggestion": suggestion_response,
            "timestamp": datetime.now().isoformat()
        }
        
        audit_key = f"kelly:suggestion_audit:{conversation_id}:{suggestion_id}"
        await redis_manager.setex(audit_key, 86400 * 7, json.dumps(suggestion_audit))  # 7 days
        
        # Update latest suggestion
        latest_key = f"kelly:ai_suggestion:{conversation_id}:latest"
        await redis_manager.setex(latest_key, 3600, json.dumps(suggestion_response))  # 1 hour
        
        return AIResponseSuggestion(
            suggested_response=suggested_text,
            confidence_score=suggestion_response.get("confidence_score", 0.7),
            reasoning=suggestion_response.get("reasoning", "AI analysis not available"),
            alternative_suggestions=alternatives,
            safety_notes=safety_notes,
            tone_analysis=tone_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI suggestion for {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{conversation_id}", response_model=List[Dict[str, Any]])
async def get_intervention_history(
    conversation_id: str,
    current_user = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get intervention history for a conversation"""
    try:
        # Get archived interventions for this conversation
        history_keys = await redis_manager.keys(f"kelly:intervention_history:{conversation_id}:*")
        
        interventions = []
        for key in history_keys:
            intervention_data = await redis_manager.get(key)
            if intervention_data:
                intervention = json.loads(intervention_data)
                
                # Remove sensitive information if user is not admin
                if not current_user.get("is_admin", False):
                    intervention.pop("human_operator", None)
                    intervention["human_operator_name"] = "Operator"
                
                interventions.append(intervention)
        
        # Sort by takeover time
        interventions.sort(key=lambda x: x.get("takeover_time", ""), reverse=True)
        
        return interventions
        
    except Exception as e:
        logger.error(f"Error getting intervention history for {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=InterventionHistoryResponse)
async def get_intervention_statistics(
    timeframe: str = "7d",
    current_user = Depends(get_current_user)
) -> InterventionHistoryResponse:
    """Get intervention statistics and analytics"""
    try:
        # Calculate time range
        end_time = datetime.now()
        
        if timeframe == "24h":
            start_time = end_time - timedelta(days=1)
        elif timeframe == "7d":
            start_time = end_time - timedelta(days=7)
        elif timeframe == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=7)  # Default to 7 days
        
        # Get all intervention history
        history_keys = await redis_manager.keys("kelly:intervention_history:*")
        
        interventions = []
        total_duration = 0
        successful_interventions = 0
        reason_counts = {}
        
        for key in history_keys:
            intervention_data = await redis_manager.get(key)
            if intervention_data:
                intervention = json.loads(intervention_data)
                
                # Check if in timeframe
                takeover_time = datetime.fromisoformat(intervention.get("takeover_time", ""))
                if takeover_time < start_time:
                    continue
                
                interventions.append(intervention)
                
                # Calculate statistics
                if intervention.get("intervention_duration"):
                    total_duration += intervention["intervention_duration"]
                
                if intervention.get("status") == "completed":
                    successful_interventions += 1
                
                # Count reasons
                reason = intervention.get("intervention_reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Calculate metrics
        total_count = len(interventions)
        success_rate = (successful_interventions / max(total_count, 1)) * 100
        avg_duration = total_duration / max(total_count, 1)
        
        # Format common reasons
        common_reasons = [
            {"reason": reason, "count": count, "percentage": (count / total_count) * 100}
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        ][:5]  # Top 5 reasons
        
        return InterventionHistoryResponse(
            interventions=interventions[-20:],  # Last 20 interventions
            total_count=total_count,
            success_rate=success_rate,
            avg_duration=avg_duration,
            common_reasons=common_reasons
        )
        
    except Exception as e:
        logger.error(f"Error getting intervention statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HELPER FUNCTIONS =====

async def _log_intervention_activity(
    activity_type: str,
    conversation_id: str,
    operator_id: str,
    metadata: Dict[str, Any]
):
    """Log intervention activity to the activity feed"""
    try:
        activity_id = str(uuid4())
        timestamp = datetime.now()
        
        activity_data = {
            "id": activity_id,
            "type": activity_type,
            "account_id": metadata.get("account_id", "unknown"),
            "conversation_id": conversation_id,
            "user_id": operator_id,
            "message": f"Intervention {activity_type.replace('_', ' ')} by operator",
            "metadata": metadata,
            "severity": "medium" if activity_type == "intervention_started" else "low",
            "timestamp": timestamp.isoformat()
        }
        
        activity_key = f"kelly:activity:intervention:{int(timestamp.timestamp())}"
        await redis_manager.setex(activity_key, 86400 * 7, json.dumps(activity_data))  # 7 days
        
    except Exception as e:
        logger.error(f"Error logging intervention activity: {e}")

async def _broadcast_intervention_update(
    conversation_id: str,
    update_type: str,
    data: Dict[str, Any]
):
    """Broadcast intervention update via WebSocket"""
    try:
        message = {
            "type": "intervention_update",
            "conversation_id": conversation_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_to_room(f"intervention_{conversation_id}", message)
        await websocket_manager.broadcast_to_room("monitoring_dashboard", message)
        
    except Exception as e:
        logger.error(f"Error broadcasting intervention update: {e}")

async def _notify_user_of_intervention(
    conversation_id: str,
    operator_name: str
):
    """Send notification to user about intervention"""
    try:
        # Get conversation details
        conv_key = f"kelly:conversation_track:{conversation_id}"
        conversation_data = await redis_manager.get(conv_key)
        
        if conversation_data:
            conversation = json.loads(conversation_data)
            account_id = conversation.get("account_id")
            user_id = conversation.get("user_id")
            
            if account_id and user_id:
                # Send message through userbot
                message = f"👋 Hi! {operator_name} from our support team is now helping with your conversation to provide the best assistance possible."
                
                await kelly_userbot.send_message(
                    account_id=account_id,
                    user_id=user_id,
                    message=message,
                    is_intervention=True
                )
                
    except Exception as e:
        logger.error(f"Error notifying user of intervention: {e}")

async def _archive_intervention(intervention: Dict[str, Any]):
    """Archive completed intervention to history"""
    try:
        conversation_id = intervention.get("conversation_id")
        intervention_id = intervention.get("id")
        
        if conversation_id and intervention_id:
            history_key = f"kelly:intervention_history:{conversation_id}:{intervention_id}"
            await redis_manager.setex(history_key, 86400 * 90, json.dumps(intervention))  # 90 days
            
            # Remove from active interventions
            active_key = f"kelly:intervention:{conversation_id}"
            await redis_manager.delete(active_key)
            
    except Exception as e:
        logger.error(f"Error archiving intervention: {e}")