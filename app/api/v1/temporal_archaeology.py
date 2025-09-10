"""
Temporal Archaeology API Endpoints

Provides RESTful API for conversation archaeology, pattern discovery,
and message reconstruction from behavioral patterns.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import json

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from app.database.session import get_db
from app.core.redis import get_redis
from app.models.user import User
from app.models.temporal_archaeology import (
    LinguisticFingerprint, TemporalPattern, ReconstructedMessage,
    GhostConversation, ConversationFragment, ArchaeologySession
)
from app.models.message import Message
from app.services.temporal_archaeology import ArchaeologyEngine
from app.api.deps import get_current_user
from app.core.security_utils import RateLimiter, ConsentManager
from app.config.settings import get_settings
from app.core.security import verify_jwt_token
import jwt

router = APIRouter()
security = HTTPBearer()
rate_limiter = RateLimiter()
consent_manager = ConsentManager()


# JWT verification is now handled by the centralized security module
# Use verify_jwt_token from app.core.security instead


# Request/Response Models
class CreateFingerprintRequest(BaseModel):
    message_limit: int = Field(default=100, ge=10, le=1000)
    time_range_days: Optional[int] = Field(default=30, ge=1, le=365)
    update_existing: bool = Field(default=True)
    

class FingerprintResponse(BaseModel):
    id: UUID
    user_id: UUID
    vocabulary_size: int
    avg_message_length: float
    confidence_score: float
    message_sample_size: int
    time_span_days: int
    unique_phrases: List[str]
    last_updated: datetime
    
    class Config:
        orm_mode = True
        

class PatternDiscoveryRequest(BaseModel):
    time_range_days: Optional[int] = Field(default=90, ge=1, le=365)
    pattern_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by pattern types: temporal, conversational, behavioral, evolution"
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    

class PatternResponse(BaseModel):
    id: UUID
    pattern_type: str
    confidence: float
    frequency: int
    discovered_at: datetime
    pattern_summary: Dict[str, Any]
    
    class Config:
        orm_mode = True
        

class ReconstructGapRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    max_messages: int = Field(default=10, ge=1, le=50)
    use_context: bool = Field(default=True)
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v
        

class ReconstructedMessageResponse(BaseModel):
    id: UUID
    content: str
    estimated_timestamp: datetime
    confidence_score: float
    reconstruction_method: str
    linguistic_match_score: Optional[float]
    contextual_coherence: Optional[float]
    evidence_markers: List[str]
    
    class Config:
        orm_mode = True
        

class GhostConversationResponse(BaseModel):
    id: UUID
    time_range_start: datetime
    time_range_end: datetime
    message_count: int
    confidence_score: float
    completeness_score: Optional[float]
    reconstruction_method: str
    topics_identified: List[str]
    messages: List[ReconstructedMessageResponse]
    
    class Config:
        orm_mode = True
        

class ArchaeologySessionRequest(BaseModel):
    session_type: str = Field(default="manual", regex="^(manual|automated|scheduled)$")
    time_range_days: Optional[int] = Field(default=30, ge=1, le=365)
    gap_threshold_hours: float = Field(default=1.0, ge=0.5, le=24.0)
    auto_reconstruct: bool = Field(default=False)
    

class SessionResponse(BaseModel):
    id: UUID
    session_status: str
    patterns_discovered: List[UUID]
    gaps_identified: int
    messages_reconstructed: int
    total_messages_analyzed: int
    overall_confidence: Optional[float]
    created_at: datetime
    
    class Config:
        orm_mode = True


# Fingerprint Endpoints
@router.post("/fingerprints", response_model=FingerprintResponse)
async def create_linguistic_fingerprint(
    request: CreateFingerprintRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Create or update linguistic fingerprint for user."""
    engine = ArchaeologyEngine(db, redis_client)
    
    # Get user's message history
    since = datetime.utcnow() - timedelta(days=request.time_range_days) if request.time_range_days else None
    
    query = select(Message).where(Message.user_id == current_user.id)
    if since:
        query = query.where(Message.created_at >= since)
    query = query.order_by(Message.created_at.desc()).limit(request.message_limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    if len(messages) < 10:
        raise HTTPException(
            status_code=400,
            detail="Insufficient message history. Need at least 10 messages."
        )
    
    # Check for existing fingerprint
    if request.update_existing:
        result = await db.execute(
            select(LinguisticFingerprint)
            .where(LinguisticFingerprint.user_id == current_user.id)
            .order_by(LinguisticFingerprint.created_at.desc())
            .limit(1)
        )
        existing = result.scalar_one_or_none()
    else:
        existing = None
    
    # Create or update fingerprint
    fingerprint = await engine.profiler.create_fingerprint(messages)
    fingerprint.user_id = current_user.id
    fingerprint.message_sample_size = len(messages)
    
    if messages:
        time_span = (messages[0].created_at - messages[-1].created_at).days
        fingerprint.time_span_days = max(1, time_span)
    
    if existing:
        # Update existing
        fingerprint.id = existing.id
        fingerprint.update_count = existing.update_count + 1
        await db.merge(fingerprint)
    else:
        db.add(fingerprint)
    
    await db.commit()
    await db.refresh(fingerprint)
    
    return FingerprintResponse.from_orm(fingerprint)


@router.get("/fingerprints", response_model=List[FingerprintResponse])
async def list_fingerprints(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's linguistic fingerprints."""
    result = await db.execute(
        select(LinguisticFingerprint)
        .where(LinguisticFingerprint.user_id == current_user.id)
        .order_by(LinguisticFingerprint.created_at.desc())
        .limit(limit)
    )
    
    fingerprints = result.scalars().all()
    return [FingerprintResponse.from_orm(f) for f in fingerprints]


# Pattern Discovery Endpoints
@router.post("/patterns/discover", response_model=List[PatternResponse])
async def discover_patterns(
    request: PatternDiscoveryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Discover temporal patterns in conversation history."""
    engine = ArchaeologyEngine(db, redis_client)
    
    # Set time range
    time_range = None
    if request.time_range_days:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=request.time_range_days)
        time_range = (start_time, end_time)
    
    # Discover patterns
    patterns = await engine.archaeologist.discover_patterns(
        user_id=current_user.id,
        time_range=time_range
    )
    
    # Filter by confidence
    patterns = [p for p in patterns if p.confidence >= request.min_confidence]
    
    # Filter by type if specified
    if request.pattern_types:
        patterns = [p for p in patterns if p.pattern_type in request.pattern_types]
    
    # Save patterns
    for pattern in patterns:
        db.add(pattern)
    
    await db.commit()
    
    # Convert to response
    responses = []
    for pattern in patterns[:20]:  # Limit to 20 patterns
        responses.append(PatternResponse(
            id=pattern.id,
            pattern_type=pattern.pattern_type,
            confidence=pattern.confidence,
            frequency=pattern.frequency,
            discovered_at=pattern.discovered_at,
            pattern_summary=pattern.pattern_signature
        ))
    
    return responses


@router.get("/patterns", response_model=List[PatternResponse])
async def list_patterns(
    pattern_type: Optional[str] = Query(None),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List discovered patterns for user."""
    query = select(TemporalPattern).where(
        TemporalPattern.user_id == current_user.id
    )
    
    if pattern_type:
        query = query.where(TemporalPattern.pattern_type == pattern_type)
    
    query = query.where(TemporalPattern.confidence >= min_confidence)
    query = query.order_by(TemporalPattern.confidence.desc()).limit(limit)
    
    result = await db.execute(query)
    patterns = result.scalars().all()
    
    responses = []
    for pattern in patterns:
        responses.append(PatternResponse(
            id=pattern.id,
            pattern_type=pattern.pattern_type,
            confidence=pattern.confidence,
            frequency=pattern.frequency,
            discovered_at=pattern.discovered_at,
            pattern_summary=pattern.pattern_signature
        ))
    
    return responses


# Gap Analysis Endpoints
@router.get("/gaps")
async def analyze_conversation_gaps(
    threshold_hours: float = Query(1.0, ge=0.5, le=24.0),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Identify gaps in conversation history."""
    engine = ArchaeologyEngine(db, redis_client)
    
    gaps = await engine.analyze_conversation_gaps(
        user_id=current_user.id,
        threshold=timedelta(hours=threshold_hours)
    )
    
    # Format response
    gap_list = []
    for gap in gaps[:limit]:
        gap_list.append({
            "start": gap['start'],
            "end": gap['end'],
            "duration_hours": gap['duration'].total_seconds() / 3600,
            "before_preview": gap['before_message'].content[:100] if gap['before_message'] else None,
            "after_preview": gap['after_message'].content[:100] if gap['after_message'] else None
        })
    
    return {
        "total_gaps": len(gaps),
        "threshold_hours": threshold_hours,
        "gaps": gap_list
    }


# Reconstruction Endpoints
@router.post("/reconstruct", response_model=GhostConversationResponse)
async def reconstruct_conversation(
    request: ReconstructGapRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Reconstruct lost conversation in time gap."""
    engine = ArchaeologyEngine(db, redis_client)
    
    # Validate time range
    gap_duration = request.end_time - request.start_time
    if gap_duration > timedelta(days=7):
        raise HTTPException(
            status_code=400,
            detail="Gap too large. Maximum 7 days."
        )
    
    # Get context messages if requested
    context_messages = None
    if request.use_context:
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == current_user.id,
                    or_(
                        Message.created_at < request.start_time,
                        Message.created_at > request.end_time
                    )
                )
            )
            .order_by(Message.created_at)
            .limit(50)
        )
        context_messages = result.scalars().all()
    
    # Reconstruct conversation
    ghost_conversation = await engine.reconstruct_conversation(
        user_id=current_user.id,
        start_time=request.start_time,
        end_time=request.end_time,
        context_messages=context_messages
    )
    
    # Get reconstructed messages
    result = await db.execute(
        select(ReconstructedMessage)
        .where(
            ReconstructedMessage.id.in_(ghost_conversation.reconstructed_messages)
        )
        .order_by(ReconstructedMessage.estimated_timestamp)
    )
    messages = result.scalars().all()
    
    # Filter by confidence
    messages = [m for m in messages if m.confidence_score >= request.min_confidence]
    messages = messages[:request.max_messages]
    
    # Build response
    response = GhostConversationResponse(
        id=ghost_conversation.id,
        time_range_start=ghost_conversation.time_range_start,
        time_range_end=ghost_conversation.time_range_end,
        message_count=len(messages),
        confidence_score=ghost_conversation.confidence_score,
        completeness_score=ghost_conversation.completeness_score,
        reconstruction_method=ghost_conversation.reconstruction_method,
        topics_identified=ghost_conversation.topics_identified or [],
        messages=[ReconstructedMessageResponse.from_orm(m) for m in messages]
    )
    
    return response


@router.get("/reconstructed", response_model=List[ReconstructedMessageResponse])
async def list_reconstructed_messages(
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    validation_status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List reconstructed messages for user."""
    query = select(ReconstructedMessage).where(
        ReconstructedMessage.user_id == current_user.id
    )
    
    query = query.where(ReconstructedMessage.confidence_score >= min_confidence)
    
    if validation_status:
        query = query.where(ReconstructedMessage.validation_status == validation_status)
    
    query = query.order_by(ReconstructedMessage.estimated_timestamp.desc()).limit(limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    return [ReconstructedMessageResponse.from_orm(m) for m in messages]


@router.post("/reconstructed/{message_id}/validate")
async def validate_reconstruction(
    message_id: UUID,
    validation: str = Query(..., regex="^(confirmed|rejected|partial)$"),
    feedback: Optional[str] = Query(None, max_length=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Validate a reconstructed message."""
    result = await db.execute(
        select(ReconstructedMessage)
        .where(
            and_(
                ReconstructedMessage.id == message_id,
                ReconstructedMessage.user_id == current_user.id
            )
        )
    )
    
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Reconstructed message not found")
    
    # Update validation
    message.validation_status = validation
    
    # Update ghost conversation if linked
    if message.ghost_conversation_id:
        result = await db.execute(
            select(GhostConversation)
            .where(GhostConversation.id == message.ghost_conversation_id)
        )
        ghost = result.scalar_one_or_none()
        if ghost:
            ghost.user_validation = validation
            if feedback:
                ghost.user_feedback = feedback
    
    await db.commit()
    
    return {
        "message_id": str(message_id),
        "validation_status": validation,
        "feedback_recorded": bool(feedback)
    }


# Session Management Endpoints
@router.post("/sessions", response_model=SessionResponse)
async def create_archaeology_session(
    request: ArchaeologySessionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Create new archaeology session for exploration."""
    engine = ArchaeologyEngine(db, redis_client)
    
    # Set time range
    target_period = None
    if request.time_range_days:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=request.time_range_days)
        target_period = (start_time, end_time)
    
    # Create session
    session = await engine.create_archaeology_session(
        user_id=current_user.id,
        target_period=target_period
    )
    
    # Auto-reconstruct if requested
    if request.auto_reconstruct:
        background_tasks.add_task(
            auto_reconstruct_gaps,
            session.id,
            current_user.id,
            request.gap_threshold_hours,
            db,
            redis_client
        )
    
    return SessionResponse.from_orm(session)


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    status: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List archaeology sessions for user."""
    query = select(ArchaeologySession).where(
        ArchaeologySession.user_id == current_user.id
    )
    
    if status:
        query = query.where(ArchaeologySession.session_status == status)
    
    query = query.order_by(ArchaeologySession.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    return [SessionResponse.from_orm(s) for s in sessions]


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get detailed archaeology session information."""
    result = await db.execute(
        select(ArchaeologySession)
        .where(
            and_(
                ArchaeologySession.id == session_id,
                ArchaeologySession.user_id == current_user.id
            )
        )
    )
    
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get cached session data
    key = f"archaeology:session:{session_id}"
    cached_data = await redis_client.get(key)
    
    if cached_data:
        cached = json.loads(cached_data)
    else:
        cached = None
    
    return {
        "session": SessionResponse.from_orm(session),
        "cached_data": cached,
        "results_summary": session.results_summary
    }


# Analytics Endpoints
@router.get("/analytics")
async def get_archaeology_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get analytics for temporal archaeology usage."""
    since = datetime.utcnow() - timedelta(days=days)
    
    # Count reconstructed messages
    result = await db.execute(
        select(
            func.count(ReconstructedMessage.id).label("total_reconstructed"),
            func.avg(ReconstructedMessage.confidence_score).label("avg_confidence"),
            func.count(
                func.distinct(ReconstructedMessage.validation_status)
            ).filter(
                ReconstructedMessage.validation_status == "confirmed"
            ).label("confirmed_count")
        )
        .where(
            and_(
                ReconstructedMessage.user_id == current_user.id,
                ReconstructedMessage.created_at >= since
            )
        )
    )
    
    message_stats = result.one()
    
    # Count patterns discovered
    result = await db.execute(
        select(
            func.count(TemporalPattern.id).label("total_patterns"),
            func.count(func.distinct(TemporalPattern.pattern_type)).label("pattern_types"),
            func.avg(TemporalPattern.confidence).label("avg_pattern_confidence")
        )
        .where(
            and_(
                TemporalPattern.user_id == current_user.id,
                TemporalPattern.discovered_at >= since
            )
        )
    )
    
    pattern_stats = result.one()
    
    # Count sessions
    result = await db.execute(
        select(
            func.count(ArchaeologySession.id).label("total_sessions"),
            func.avg(ArchaeologySession.messages_reconstructed).label("avg_messages_per_session"),
            func.sum(ArchaeologySession.gaps_identified).label("total_gaps_found")
        )
        .where(
            and_(
                ArchaeologySession.user_id == current_user.id,
                ArchaeologySession.created_at >= since
            )
        )
    )
    
    session_stats = result.one()
    
    return {
        "period_days": days,
        "message_reconstruction": {
            "total_reconstructed": message_stats.total_reconstructed or 0,
            "average_confidence": float(message_stats.avg_confidence or 0),
            "confirmed_messages": message_stats.confirmed_count or 0
        },
        "pattern_discovery": {
            "total_patterns": pattern_stats.total_patterns or 0,
            "unique_pattern_types": pattern_stats.pattern_types or 0,
            "average_confidence": float(pattern_stats.avg_pattern_confidence or 0)
        },
        "archaeology_sessions": {
            "total_sessions": session_stats.total_sessions or 0,
            "average_messages_per_session": float(session_stats.avg_messages_per_session or 0),
            "total_gaps_identified": session_stats.total_gaps_found or 0
        }
    }


# Background task for auto-reconstruction
async def auto_reconstruct_gaps(
    session_id: UUID,
    user_id: UUID,
    gap_threshold_hours: float,
    db: AsyncSession,
    redis_client: redis.Redis
):
    """Background task to automatically reconstruct gaps."""
    engine = ArchaeologyEngine(db, redis_client)
    
    # Find gaps
    gaps = await engine.analyze_conversation_gaps(
        user_id=user_id,
        threshold=timedelta(hours=gap_threshold_hours)
    )
    
    # Reconstruct top gaps
    reconstructed_count = 0
    for gap in gaps[:5]:  # Limit to 5 gaps
        try:
            await engine.reconstruct_conversation(
                user_id=user_id,
                start_time=gap['start'],
                end_time=gap['end']
            )
            reconstructed_count += 1
        except Exception as e:
            # Log error but continue
            print(f"Failed to reconstruct gap: {e}")
    
    # Update session
    result = await db.execute(
        select(ArchaeologySession).where(ArchaeologySession.id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if session:
        session.messages_reconstructed = reconstructed_count
        session.session_status = "completed"
        session.completed_at = datetime.utcnow()
        await db.commit()