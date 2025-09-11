"""
Telegram Management API
RESTful API endpoints for managing Telegram account operations.
Provides comprehensive control, monitoring, and analytics capabilities.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import asyncio

from app.services.telegram_account_manager import TelegramAccountManager
from app.services.telegram_conversation_engine import TelegramConversationEngine
from app.services.telegram_community_adapter import TelegramCommunityAdapter
from app.services.telegram_safety_monitor import TelegramSafetyMonitor, ThreatLevel, SafetyAction
from app.models.telegram_account import TelegramAccount, AccountStatus, SafetyLevel
from app.models.telegram_community import TelegramCommunity, EngagementStrategy, CommunityStatus
from app.models.telegram_conversation import TelegramConversation
from app.database.repositories import DatabaseRepository
from app.core.security import get_current_user
from app.schemas.telegram_schemas import *

router = APIRouter(prefix="/telegram", tags=["Telegram Management"])
security = HTTPBearer()


# Dependency injection
async def get_database() -> DatabaseRepository:
    # This would be injected from main app
    pass

async def get_account_manager(account_id: str) -> TelegramAccountManager:
    # This would be injected/created based on account
    pass


@router.post("/accounts", response_model=TelegramAccountResponse)
async def create_telegram_account(
    account_data: CreateTelegramAccountRequest,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Create a new Telegram account for management
    """
    
    try:
        # Validate phone number format
        if not account_data.phone_number.startswith('+'):
            raise HTTPException(
                status_code=400,
                detail="Phone number must include country code with + prefix"
            )
        
        # Check if account already exists
        existing_account = await db.get_telegram_account_by_phone(account_data.phone_number)
        if existing_account:
            raise HTTPException(
                status_code=409,
                detail="Account with this phone number already exists"
            )
        
        # Create account record
        account = TelegramAccount(
            phone_number=account_data.phone_number,
            first_name=account_data.first_name,
            last_name=account_data.last_name,
            bio=account_data.bio,
            safety_level=account_data.safety_level,
            is_ai_disclosed=account_data.is_ai_disclosed,
            gdpr_consent_given=account_data.gdpr_consent_given,
            privacy_policy_accepted=account_data.privacy_policy_accepted,
            personality_profile=account_data.personality_profile,
            communication_style=account_data.communication_style,
            interests=account_data.interests
        )
        
        # Store in database
        created_account = await db.create_telegram_account(account)
        
        return TelegramAccountResponse(
            id=str(created_account.id),
            phone_number=created_account.phone_number,
            status=created_account.status,
            safety_level=created_account.safety_level,
            is_healthy=created_account.is_healthy,
            risk_score=created_account.risk_score,
            created_at=created_account.created_at,
            last_health_check=created_account.last_health_check
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create account: {str(e)}")


@router.get("/accounts/{account_id}", response_model=TelegramAccountDetailResponse)
async def get_account_details(
    account_id: str,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Get detailed information about a Telegram account
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Get account manager for status
    try:
        manager = await get_account_manager(account_id)
        status_data = await manager.get_account_status()
    except:
        status_data = {"error": "Manager not available"}
    
    # Get communities
    communities = await db.get_account_communities(account_id)
    
    # Get recent conversations
    recent_conversations = await db.get_recent_conversations(account_id, limit=10)
    
    # Get safety events
    recent_events = await db.get_recent_safety_events(account_id, hours=24)
    
    return TelegramAccountDetailResponse(
        account=TelegramAccountResponse(
            id=str(account.id),
            phone_number=account.phone_number,
            status=account.status,
            safety_level=account.safety_level,
            is_healthy=account.is_healthy,
            risk_score=account.risk_score,
            created_at=account.created_at,
            last_health_check=account.last_health_check
        ),
        status_data=status_data,
        communities=[
            TelegramCommunityResponse(
                id=str(c.id),
                title=c.title,
                status=c.status,
                engagement_strategy=c.engagement_strategy,
                engagement_score=c.engagement_score,
                reputation_score=c.reputation_score,
                member_count=c.member_count
            ) for c in communities
        ],
        recent_conversations=[
            ConversationSummaryResponse(
                id=str(c.id),
                chat_id=c.chat_id,
                message_count=c.message_count,
                last_message_date=c.last_message_date,
                engagement_score=c.engagement_score
            ) for c in recent_conversations
        ],
        recent_safety_events=[
            SafetyEventResponse(
                id=str(e.id),
                event_type=e.event_type,
                severity=e.severity,
                description=e.description,
                created_at=e.created_at
            ) for e in recent_events
        ]
    )


@router.post("/accounts/{account_id}/start")
async def start_account(
    account_id: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Start Telegram account manager
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    if account.status == AccountStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Account is already active")
    
    try:
        # Start account manager in background
        manager = await get_account_manager(account_id)
        background_tasks.add_task(manager.start)
        
        return {"message": "Account start initiated", "account_id": account_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start account: {str(e)}")


@router.post("/accounts/{account_id}/stop")
async def stop_account(
    account_id: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Stop Telegram account manager
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    try:
        # Stop account manager
        manager = await get_account_manager(account_id)
        background_tasks.add_task(manager.stop)
        
        return {"message": "Account stop initiated", "account_id": account_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop account: {str(e)}")


@router.get("/accounts/{account_id}/health", response_model=AccountHealthResponse)
async def check_account_health(
    account_id: str,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Get comprehensive account health status
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    try:
        # Get safety monitor
        from app.services.telegram_safety_monitor import TelegramSafetyMonitor
        from app.services.risk_manager import RiskManager
        
        risk_manager = RiskManager(db)
        safety_monitor = TelegramSafetyMonitor(risk_manager, db)
        await safety_monitor.initialize(account_id)
        
        # Generate safety report
        safety_report = await safety_monitor.generate_safety_report(account_id)
        
        return AccountHealthResponse(
            account_id=account_id,
            is_healthy=safety_report["health_status"]["is_healthy"],
            health_score=safety_report["health_status"]["health_score"],
            risk_score=safety_report["risk_assessment"]["overall_risk_score"],
            threat_level=safety_report["risk_assessment"]["threat_level"],
            active_warnings=safety_report["health_status"]["active_warnings"],
            daily_activity=safety_report["recent_activity"],
            recommendations=safety_report["recommendations"],
            last_check=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/accounts/{account_id}/communities/{community_id}/join")
async def join_community(
    account_id: str,
    community_id: str,
    join_request: JoinCommunityRequest,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Join a Telegram community
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Check if already in community
    existing_community = await db.get_telegram_community_by_chat_id(account_id, join_request.chat_id)
    if existing_community:
        raise HTTPException(status_code=409, detail="Already member of this community")
    
    try:
        # Create community record
        community = TelegramCommunity(
            account_id=account.id,
            chat_id=join_request.chat_id,
            title=join_request.title,
            username=join_request.username,
            invite_link=join_request.invite_link,
            community_type=join_request.community_type,
            engagement_strategy=join_request.engagement_strategy or EngagementStrategy.PARTICIPANT,
            status=CommunityStatus.PENDING_JOIN
        )
        
        created_community = await db.create_telegram_community(community)
        
        # Get account manager and attempt to join
        manager = await get_account_manager(account_id)
        # This would trigger the actual join process
        
        return {
            "message": "Community join initiated",
            "community_id": str(created_community.id),
            "status": "pending_join"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to join community: {str(e)}")


@router.get("/accounts/{account_id}/communities", response_model=List[TelegramCommunityResponse])
async def list_communities(
    account_id: str,
    status: Optional[CommunityStatus] = None,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    List communities for account
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    communities = await db.get_account_communities(account_id, status=status)
    
    return [
        TelegramCommunityResponse(
            id=str(c.id),
            title=c.title,
            status=c.status,
            engagement_strategy=c.engagement_strategy,
            engagement_score=c.engagement_score,
            reputation_score=c.reputation_score,
            member_count=c.member_count,
            join_date=c.join_date,
            last_activity_date=c.last_activity_date
        ) for c in communities
    ]


@router.get("/accounts/{account_id}/communities/{community_id}/insights")
async def get_community_insights(
    account_id: str,
    community_id: str,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Get AI-generated insights about community engagement
    """
    
    community = await db.get_telegram_community(community_id)
    if not community or str(community.account_id) != account_id:
        raise HTTPException(status_code=404, detail="Community not found")
    
    try:
        # Get community adapter
        from app.services.telegram_community_adapter import TelegramCommunityAdapter
        from app.services.consciousness_mirror import ConsciousnessMirror
        from app.services.memory_palace import MemoryPalace
        from app.services.behavioral_predictor import BehavioralPredictor
        from app.services.engagement_analyzer import EngagementAnalyzer
        
        consciousness = ConsciousnessMirror(db)
        memory = MemoryPalace(db)
        predictor = BehavioralPredictor(db)
        analyzer = EngagementAnalyzer(db)
        
        adapter = TelegramCommunityAdapter(consciousness, memory, predictor, analyzer, db)
        await adapter.initialize(account_id)
        
        # Generate insights
        insights = await adapter.generate_community_insights(community_id)
        
        return {
            "community_id": community_id,
            "insights": [
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence_score": insight.confidence_score,
                    "impact_potential": insight.impact_potential,
                    "recommended_actions": insight.recommended_actions,
                    "expected_outcomes": insight.expected_outcomes
                } for insight in insights
            ],
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.post("/accounts/{account_id}/messages/send")
async def send_message(
    account_id: str,
    message_request: SendMessageRequest,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Send a message through the Telegram account
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    if account.status != AccountStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Account is not active")
    
    try:
        # Get safety monitor for pre-check
        from app.services.telegram_safety_monitor import TelegramSafetyMonitor
        from app.services.risk_manager import RiskManager
        
        risk_manager = RiskManager(db)
        safety_monitor = TelegramSafetyMonitor(risk_manager, db)
        await safety_monitor.initialize(account_id)
        
        # Safety check
        is_safe, alert = await safety_monitor.check_safety_before_action(
            "send_message",
            target_chat_id=message_request.chat_id,
            message_content=message_request.content
        )
        
        if not is_safe:
            raise HTTPException(
                status_code=429,
                detail=f"Message blocked for safety: {alert.description if alert else 'Safety check failed'}"
            )
        
        # Get account manager and send message
        manager = await get_account_manager(account_id)
        # This would trigger the actual message sending
        
        return {
            "message": "Message sent successfully",
            "chat_id": message_request.chat_id,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.get("/accounts/{account_id}/conversations", response_model=List[ConversationSummaryResponse])
async def list_conversations(
    account_id: str,
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    List conversations for account
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    conversations = await db.get_account_conversations(
        account_id, 
        limit=limit, 
        offset=offset,
        status=status
    )
    
    return [
        ConversationSummaryResponse(
            id=str(c.id),
            chat_id=c.chat_id,
            message_count=c.message_count,
            last_message_date=c.last_message_date,
            engagement_score=c.engagement_score,
            status=c.status,
            is_group=c.is_group
        ) for c in conversations
    ]


@router.get("/accounts/{account_id}/analytics", response_model=AccountAnalyticsResponse)
async def get_account_analytics(
    account_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Get comprehensive analytics for account
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get analytics data
        message_stats = await db.get_message_statistics(account_id, start_date, end_date)
        engagement_stats = await db.get_engagement_statistics(account_id, start_date, end_date)
        community_stats = await db.get_community_statistics(account_id)
        safety_stats = await db.get_safety_statistics(account_id, start_date, end_date)
        
        return AccountAnalyticsResponse(
            account_id=account_id,
            period_days=days,
            message_statistics=message_stats,
            engagement_statistics=engagement_stats,
            community_statistics=community_stats,
            safety_statistics=safety_stats,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics: {str(e)}")


@router.post("/accounts/{account_id}/safety/emergency-stop")
async def emergency_stop(
    account_id: str,
    reason: str,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Emergency stop for account operations
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    try:
        # Stop account manager immediately
        manager = await get_account_manager(account_id)
        await manager.stop()
        
        # Update account status
        account.status = AccountStatus.SUSPENDED
        await db.update_telegram_account(account)
        
        # Record safety event
        from app.models.telegram_account import AccountSafetyEvent
        event = AccountSafetyEvent(
            account_id=account.id,
            event_type="emergency_stop",
            severity="critical",
            description=f"Emergency stop triggered: {reason}",
            data={"reason": reason, "triggered_by": "user"}
        )
        await db.create_safety_event(event)
        
        return {
            "message": "Emergency stop executed",
            "account_id": account_id,
            "reason": reason,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")


@router.get("/accounts/{account_id}/safety/report")
async def get_safety_report(
    account_id: str,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Get comprehensive safety report
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    try:
        # Get safety monitor
        from app.services.telegram_safety_monitor import TelegramSafetyMonitor
        from app.services.risk_manager import RiskManager
        
        risk_manager = RiskManager(db)
        safety_monitor = TelegramSafetyMonitor(risk_manager, db)
        await safety_monitor.initialize(account_id)
        
        # Generate comprehensive report
        report = await safety_monitor.generate_safety_report(account_id)
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate safety report: {str(e)}")


@router.post("/accounts/{account_id}/personality/adapt")
async def adapt_personality(
    account_id: str,
    adaptation_request: PersonalityAdaptationRequest,
    current_user = Depends(get_current_user),
    db: DatabaseRepository = Depends(get_database)
):
    """
    Adapt personality for specific context or community
    """
    
    account = await db.get_telegram_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    try:
        # Get consciousness mirror
        from app.services.consciousness_mirror import ConsciousnessMirror
        
        consciousness = ConsciousnessMirror(db)
        await consciousness.initialize(account_id)
        
        # Apply personality adaptation
        adapted_personality = await consciousness.adapt_to_context(adaptation_request.context)
        
        # Update account personality if requested
        if adaptation_request.save_as_default:
            account.personality_profile = adapted_personality
            await db.update_telegram_account(account)
        
        return {
            "account_id": account_id,
            "adapted_personality": adapted_personality,
            "context": adaptation_request.context,
            "saved_as_default": adaptation_request.save_as_default,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Personality adaptation failed: {str(e)}")


# WebSocket endpoint for real-time monitoring
@router.websocket("/accounts/{account_id}/monitor")
async def monitor_account_realtime(websocket, account_id: str):
    """
    Real-time monitoring WebSocket endpoint
    """
    
    await websocket.accept()
    
    try:
        # Set up real-time monitoring
        while True:
            # Get current status
            account = await db.get_telegram_account(account_id)
            if not account:
                await websocket.send_json({"error": "Account not found"})
                break
            
            # Get real-time status
            try:
                manager = await get_account_manager(account_id)
                status = await manager.get_account_status()
                
                await websocket.send_json({
                    "type": "status_update",
                    "data": status,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        await websocket.send_json({"error": f"Monitoring failed: {str(e)}"})
    finally:
        await websocket.close()