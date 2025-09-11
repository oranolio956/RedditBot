"""
Viral Sharing API Endpoints

RESTful API for viral content generation, sharing, and referral tracking.
Enables users to create shareable content, track referrals, and view
growth analytics for maximum viral potential.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.connection import get_db_session as get_db
from app.models.sharing import (
    ShareableContent, ContentShare, UserReferral, ReferralProgram,
    ViralMetrics, ShareableContentType, SocialPlatform
)
from app.models.user import User
from app.models.conversation import Conversation
from app.services.viral_engine import ViralEngine
from app.services.referral_tracker import ReferralTracker
from app.services.llm_service import LLMService
from app.core.deps import get_current_user
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sharing", tags=["viral-sharing"])


# Pydantic models for request/response
class ShareableContentResponse(BaseModel):
    """Response model for shareable content."""
    id: str
    content_type: str
    title: str
    description: Optional[str]
    viral_score: float
    hashtags: Optional[List[str]]
    optimal_platforms: Optional[List[str]]
    view_count: int
    share_count: int
    like_count: int
    is_trending: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ContentShareRequest(BaseModel):
    """Request model for sharing content."""
    content_id: str
    platform: str
    share_text: Optional[str] = None
    custom_hashtags: Optional[List[str]] = None


class ReferralStatsResponse(BaseModel):
    """Response model for referral statistics."""
    referrals_sent: int
    referrals_converted: int
    referrals_pending: int
    conversion_rate: float
    content_shares: int
    total_points: int
    rank: int
    achievements: List[Dict[str, Any]]
    credits: int
    premium_expires_at: Optional[str]


class LeaderboardResponse(BaseModel):
    """Response model for referral leaderboard."""
    rank: int
    user_id: str
    display_name: str
    referral_count: int
    conversion_count: int
    total_points: int
    badges: List[str]


class ViralContentGenerationRequest(BaseModel):
    """Request model for generating viral content."""
    conversation_id: str
    content_types: Optional[List[str]] = None
    target_platforms: Optional[List[str]] = None


class SocialProofResponse(BaseModel):
    """Response model for social proof statistics."""
    total_users: int
    total_referrals: int
    successful_conversions: int
    conversion_rate: float
    viral_coefficient: float
    recent_growth: Dict[str, Any]


@router.get("/content/trending", response_model=List[ShareableContentResponse])
async def get_trending_content(
    limit: int = Query(10, ge=1, le=50),
    content_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get currently trending viral content."""
    
    llm_service = LLMService()
    viral_engine = ViralEngine(db, llm_service)
    
    content_type_enum = None
    if content_type:
        try:
            content_type_enum = ShareableContentType(content_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {content_type}"
            )
    
    trending_content = await viral_engine.get_trending_content(
        limit=limit,
        content_type=content_type_enum
    )
    
    response_content = []
    for content in trending_content:
        response_content.append(ShareableContentResponse(
            id=str(content.id),
            content_type=content.content_type,
            title=content.title,
            description=content.description,
            viral_score=content.viral_score,
            hashtags=content.hashtags,
            optimal_platforms=content.optimal_platforms,
            view_count=content.view_count,
            share_count=content.share_count,
            like_count=content.like_count,
            is_trending=content.is_trending(),
            created_at=content.created_at
        ))
    
    return response_content


@router.post("/content/generate")
async def generate_viral_content(
    request: ViralContentGenerationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate viral content from a conversation."""
    
    # Verify user owns the conversation
    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found or access denied"
        )
    
    llm_service = LLMService()
    viral_engine = ViralEngine(db, llm_service)
    
    # Generate viral content
    generated_content = await viral_engine.analyze_conversation_for_viral_content(
        conversation=conversation,
        real_time=False
    )
    
    if not generated_content:
        raise HTTPException(
            status_code=400,
            detail="No viral content could be generated from this conversation"
        )
    
    # Convert to response format
    response_content = []
    for content in generated_content:
        response_content.append(ShareableContentResponse(
            id=str(content.id),
            content_type=content.content_type,
            title=content.title,
            description=content.description,
            viral_score=content.viral_score,
            hashtags=content.hashtags,
            optimal_platforms=content.optimal_platforms,
            view_count=content.view_count,
            share_count=content.share_count,
            like_count=content.like_count,
            is_trending=content.is_trending(),
            created_at=content.created_at
        ))
    
    logger.info(
        "viral_content_generated_api",
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        content_count=len(generated_content)
    )
    
    return {
        "message": "Viral content generated successfully",
        "content": response_content
    }


@router.get("/content/{content_id}")
async def get_shareable_content(
    content_id: str,
    platform: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get shareable content optimized for a platform."""
    
    content = db.query(ShareableContent).filter(
        ShareableContent.id == content_id,
        ShareableContent.is_published == True
    ).first()
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    # Increment view count
    content.view_count += 1
    db.commit()
    
    # Platform optimization
    if platform:
        try:
            platform_enum = SocialPlatform(platform)
            llm_service = LLMService()
            viral_engine = ViralEngine(db, llm_service)
            
            optimized_content = await viral_engine.optimize_content_for_platform(
                content=content,
                platform=platform_enum
            )
            
            return optimized_content
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid platform: {platform}"
            )
    
    # Return standard content
    return ShareableContentResponse(
        id=str(content.id),
        content_type=content.content_type,
        title=content.title,
        description=content.description,
        viral_score=content.viral_score,
        hashtags=content.hashtags,
        optimal_platforms=content.optimal_platforms,
        view_count=content.view_count,
        share_count=content.share_count,
        like_count=content.like_count,
        is_trending=content.is_trending(),
        created_at=content.created_at
    )


@router.post("/content/{content_id}/share")
async def share_content(
    content_id: str,
    request: ContentShareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Share viral content on social platform."""
    
    content = db.query(ShareableContent).filter(
        ShareableContent.id == content_id,
        ShareableContent.is_published == True
    ).first()
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    try:
        platform_enum = SocialPlatform(request.platform)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform: {request.platform}"
        )
    
    # Create content share record
    content_share = ContentShare(
        content_id=content.id,
        sharer_user_id=current_user.id,
        platform=request.platform,
        share_format="standard",
        share_text=request.share_text,
        sharer_anonymous_id=f"anon_{str(current_user.id)[:8]}"
    )
    
    db.add(content_share)
    
    # Update content share count
    content.share_count += 1
    db.commit()
    
    logger.info(
        "content_shared",
        content_id=content_id,
        user_id=current_user.id,
        platform=request.platform,
        total_shares=content.share_count
    )
    
    return {
        "message": "Content shared successfully",
        "share_id": str(content_share.id),
        "total_shares": content.share_count
    }


@router.get("/referrals/stats", response_model=ReferralStatsResponse)
async def get_referral_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's referral statistics and achievements."""
    
    referral_tracker = ReferralTracker(db)
    stats = await referral_tracker.get_user_referral_stats(current_user.id)
    
    return ReferralStatsResponse(
        referrals_sent=stats["referrals_sent"],
        referrals_converted=stats["referrals_converted"],
        referrals_pending=stats["referrals_pending"],
        conversion_rate=stats["conversion_rate"],
        content_shares=stats["content_shares"],
        total_points=stats["total_points"],
        rank=stats["rank"],
        achievements=stats["achievements"],
        credits=stats["credits"],
        premium_expires_at=stats["premium_expires_at"]
    )


@router.post("/referrals/generate")
async def generate_referral_code(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a new referral code for the user."""
    
    referral_tracker = ReferralTracker(db)
    referral = await referral_tracker.generate_referral_code(current_user.id)
    
    # Create shareable content for referral
    shareable_content = await referral_tracker.create_shareable_referral_content(
        current_user.id
    )
    
    return {
        "referral_code": referral.referral_code,
        "share_url": f"https://yourbot.com/join?ref={referral.referral_code}",
        "shareable_content": shareable_content
    }


@router.get("/referrals/leaderboard", response_model=List[LeaderboardResponse])
async def get_referral_leaderboard(
    period: str = Query("all_time", regex="^(weekly|monthly|all_time)$"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get referral leaderboard with rankings."""
    
    referral_tracker = ReferralTracker(db)
    leaderboard = await referral_tracker.get_referral_leaderboard(
        limit=limit,
        period=period
    )
    
    response_entries = []
    for entry in leaderboard:
        response_entries.append(LeaderboardResponse(
            rank=entry.rank,
            user_id=entry.user_id,
            display_name=entry.display_name,
            referral_count=entry.referral_count,
            conversion_count=entry.conversion_count,
            total_points=entry.total_points,
            badges=entry.badges
        ))
    
    return response_entries


@router.get("/referrals/{referral_code}/click")
async def track_referral_click(
    referral_code: str,
    visitor_ip: Optional[str] = Query(None),
    user_agent: Optional[str] = Query(None),
    platform: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Track referral link click and redirect to app."""
    
    referral_tracker = ReferralTracker(db)
    
    visitor_data = {
        "ip": visitor_ip,
        "user_agent": user_agent,
        "platform": platform,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Track the click
    success = await referral_tracker.track_referral_click(
        referral_code=referral_code,
        visitor_data=visitor_data
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Invalid referral code"
        )
    
    # Redirect to app with referral code
    redirect_url = f"https://yourbot.com/welcome?ref={referral_code}"
    return RedirectResponse(url=redirect_url, status_code=302)


@router.post("/referrals/{referral_code}/signup")
async def process_referral_signup(
    referral_code: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process referral signup when new user joins."""
    
    referral_tracker = ReferralTracker(db)
    referral = await referral_tracker.process_referral_signup(
        referral_code=referral_code,
        new_user=current_user
    )
    
    if not referral:
        raise HTTPException(
            status_code=404,
            detail="Invalid referral code"
        )
    
    return {
        "message": "Referral signup processed successfully",
        "welcome_reward": referral.program.referee_reward,
        "referrer_name": referral.referrer.get_display_name()
    }


@router.get("/analytics/social-proof", response_model=SocialProofResponse)
async def get_social_proof_stats(
    db: Session = Depends(get_db)
):
    """Get social proof statistics for marketing purposes."""
    
    referral_tracker = ReferralTracker(db)
    stats = await referral_tracker.get_social_proof_stats()
    
    return SocialProofResponse(
        total_users=stats["total_users"],
        total_referrals=stats["total_referrals"],
        successful_conversions=stats["successful_conversions"],
        conversion_rate=stats["conversion_rate"],
        viral_coefficient=stats["viral_coefficient"],
        recent_growth=stats["growth_metrics"]
    )


@router.get("/analytics/viral-metrics")
async def get_viral_metrics(
    period: str = Query("daily", regex="^(daily|weekly|monthly)$"),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get viral performance metrics over time."""
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    metrics = db.query(ViralMetrics).filter(
        ViralMetrics.period_type == period,
        ViralMetrics.date >= start_date,
        ViralMetrics.date <= end_date
    ).order_by(ViralMetrics.date.desc()).all()
    
    metrics_data = []
    for metric in metrics:
        metrics_data.append({
            "date": metric.date.isoformat(),
            "period_type": metric.period_type,
            "total_content_created": metric.total_content_created,
            "total_content_shared": metric.total_content_shared,
            "total_views": metric.total_views,
            "total_shares": metric.total_shares,
            "total_referrals_sent": metric.total_referrals_sent,
            "total_referrals_converted": metric.total_referrals_converted,
            "viral_coefficient": metric.viral_coefficient,
            "top_content_type": metric.top_performing_content_type,
            "top_platform": metric.top_performing_platform
        })
    
    return {
        "period": period,
        "days": days,
        "metrics": metrics_data,
        "summary": {
            "total_content": sum(m.total_content_created for m in metrics),
            "total_shares": sum(m.total_shares for m in metrics),
            "avg_viral_coefficient": sum(m.viral_coefficient for m in metrics) / len(metrics) if metrics else 0,
            "growth_trend": "increasing" if len(metrics) >= 2 and metrics[0].viral_coefficient > metrics[-1].viral_coefficient else "stable"
        }
    }


@router.post("/content/{content_id}/track-performance")
async def track_content_performance(
    content_id: str,
    platform: str,
    metrics: Dict[str, int] = Body(...),
    db: Session = Depends(get_db)
):
    """Track viral content performance from external platforms."""
    
    try:
        platform_enum = SocialPlatform(platform)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform: {platform}"
        )
    
    llm_service = LLMService()
    viral_engine = ViralEngine(db, llm_service)
    
    await viral_engine.track_content_performance(
        content_id=content_id,
        platform=platform_enum,
        metrics=metrics
    )
    
    return {
        "message": "Performance metrics tracked successfully",
        "content_id": content_id,
        "platform": platform,
        "metrics": metrics
    }


@router.get("/content/{content_id}/share-formats")
async def get_share_formats(
    content_id: str,
    db: Session = Depends(get_db)
):
    """Get optimized share formats for different platforms."""
    
    content = db.query(ShareableContent).filter(
        ShareableContent.id == content_id,
        ShareableContent.is_published == True
    ).first()
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    llm_service = LLMService()
    viral_engine = ViralEngine(db, llm_service)
    
    # Generate formats for all major platforms
    formats = {}
    platforms = [
        SocialPlatform.TWITTER,
        SocialPlatform.INSTAGRAM,
        SocialPlatform.TIKTOK,
        SocialPlatform.LINKEDIN,
        SocialPlatform.FACEBOOK
    ]
    
    for platform in platforms:
        formats[platform.value] = await viral_engine.optimize_content_for_platform(
            content=content,
            platform=platform
        )
    
    return {
        "content_id": content_id,
        "title": content.title,
        "viral_score": content.viral_score,
        "formats": formats
    }