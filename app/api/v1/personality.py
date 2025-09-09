"""
Personality API Endpoints

FastAPI endpoints for personality system management and interaction:
- Personality analysis and insights
- Profile management and customization
- Testing and experimentation
- Performance monitoring and analytics
- Real-time personality adaptation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, func, desc
import logging

# Internal imports
from app.database.connection import get_db_session
from app.core.redis import get_redis_client
from app.models.personality import (
    PersonalityProfile, UserPersonalityMapping, PersonalityTrait,
    AdaptationStrategy, PersonalityDimension
)
from app.models.user import User
from app.models.conversation import ConversationSession, Message
from app.services.personality_manager import (
    PersonalityManager, PersonalityResponse, InteractionOutcome
)
from app.services.personality_testing import (
    PersonalityTestingFramework, PersonalityTest, TestType
)
from app.middleware.rate_limiting import RateLimitMiddleware
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/personality", tags=["personality"])
settings = get_settings()


# Pydantic models for API

class PersonalityTraitModel(BaseModel):
    """Personality trait model for API."""
    name: str
    value: float = Field(..., ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class CreatePersonalityProfileRequest(BaseModel):
    """Request model for creating personality profile."""
    name: str = Field(..., min_length=3, max_length=100)
    display_name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)
    category: Optional[str] = Field("custom", max_length=50)
    trait_scores: Dict[str, float] = Field(..., description="Personality trait scores (0-1)")
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCE
    adaptation_sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    behavioral_patterns: Optional[Dict[str, Any]] = None
    communication_style: Optional[Dict[str, Any]] = None

    @validator('trait_scores')
    def validate_trait_scores(cls, v):
        for trait_name, score in v.items():
            if trait_name not in [t.value for t in PersonalityDimension]:
                raise ValueError(f"Invalid trait name: {trait_name}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Trait score must be between 0 and 1: {trait_name}={score}")
        return v


class UpdatePersonalityProfileRequest(BaseModel):
    """Request model for updating personality profile."""
    display_name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, min_length=10, max_length=500)
    trait_scores: Optional[Dict[str, float]] = None
    adaptation_strategy: Optional[AdaptationStrategy] = None
    adaptation_sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    behavioral_patterns: Optional[Dict[str, Any]] = None
    communication_style: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

    @validator('trait_scores')
    def validate_trait_scores(cls, v):
        if v:
            for trait_name, score in v.items():
                if trait_name not in [t.value for t in PersonalityDimension]:
                    raise ValueError(f"Invalid trait name: {trait_name}")
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"Trait score must be between 0 and 1: {trait_name}={score}")
        return v


class ProcessMessageRequest(BaseModel):
    """Request model for processing user message."""
    user_id: str = Field(..., description="User ID")
    message_content: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    message_metadata: Optional[Dict[str, Any]] = None
    context_override: Optional[Dict[str, Any]] = None


class UserFeedbackRequest(BaseModel):
    """Request model for user feedback."""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    feedback_type: str = Field(..., description="Type of feedback")
    feedback_value: float = Field(..., ge=0.0, le=1.0, description="Feedback value (0-1)")
    feedback_details: Optional[Dict[str, Any]] = None


class CreateABTestRequest(BaseModel):
    """Request model for creating A/B test."""
    name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)
    control_config: Dict[str, Any] = Field(..., description="Control variant configuration")
    treatment_config: Dict[str, Any] = Field(..., description="Treatment variant configuration")
    primary_metric: str = Field(..., description="Primary metric to optimize")
    target_population: Optional[Dict[str, Any]] = None
    duration_days: int = Field(14, ge=1, le=90)
    traffic_split: float = Field(0.5, ge=0.1, le=0.9)
    minimum_sample_size: int = Field(100, ge=10, le=10000)


class PersonalityInsightsResponse(BaseModel):
    """Response model for personality insights."""
    user_id: str
    personality_profile: Dict[str, Any]
    current_state: Dict[str, Any]
    personality_trends: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    performance_indicators: Dict[str, Any]
    recommendations: List[str]


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""
    system_overview: Dict[str, Any]
    matching_performance: Dict[str, Any]
    satisfaction_trends: Dict[str, Any]
    profile_effectiveness: Dict[str, Any]
    system_health: Dict[str, Any]
    recommendations: List[str]


# Dependency injection

async def get_personality_manager(db: AsyncSession = Depends(get_db_session)):
    """Get personality manager instance."""
    redis_client = await get_redis_client()
    manager = PersonalityManager(db, redis_client)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.cleanup()


async def get_testing_framework(db: AsyncSession = Depends(get_db_session)):
    """Get personality testing framework instance."""
    redis_client = await get_redis_client()
    framework = PersonalityTestingFramework(db, redis_client)
    yield framework


# API Endpoints

@router.post("/process-message", response_model=Dict[str, Any])
async def process_user_message(
    request: ProcessMessageRequest,
    background_tasks: BackgroundTasks,
    manager: PersonalityManager = Depends(get_personality_manager)
):
    """
    Process user message with personality adaptation.
    
    This is the main endpoint for personality-driven conversation handling.
    """
    try:
        response = await manager.process_user_message(
            user_id=request.user_id,
            message_content=request.message_content,
            session_id=request.session_id,
            message_metadata=request.message_metadata
        )
        
        return {
            "status": "success",
            "response": response.content,
            "personality_info": {
                "adapted_traits": response.personality_state.adapted_traits,
                "confidence_score": response.confidence_score,
                "adaptation_strategy": response.adaptation_info.get('adaptation_strategy'),
                "processing_time_ms": response.processing_time_ms
            },
            "context_factors": response.adaptation_info.get('context_factors', {}),
            "user_traits_detected": response.adaptation_info.get('user_traits_detected', {})
        }
        
    except Exception as e:
        logger.error(f"Error processing user message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_user_feedback(
    request: UserFeedbackRequest,
    manager: PersonalityManager = Depends(get_personality_manager)
):
    """Submit user feedback for personality learning."""
    try:
        await manager.provide_user_feedback(
            user_id=request.user_id,
            session_id=request.session_id,
            feedback_type=request.feedback_type,
            feedback_value=request.feedback_value,
            feedback_details=request.feedback_details
        )
        
        return {"status": "success", "message": "Feedback recorded successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{user_id}", response_model=PersonalityInsightsResponse)
async def get_user_personality_insights(
    user_id: str,
    manager: PersonalityManager = Depends(get_personality_manager)
):
    """Get comprehensive personality insights for a user."""
    try:
        insights = await manager.get_user_personality_insights(user_id)
        
        if insights.get('status') == 'error':
            raise HTTPException(status_code=400, detail=insights.get('message'))
        
        return PersonalityInsightsResponse(**insights)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting personality insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles", response_model=List[Dict[str, Any]])
async def list_personality_profiles(
    active_only: bool = Query(True, description="Filter to active profiles only"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """List available personality profiles."""
    try:
        query = select(PersonalityProfile)
        
        if active_only:
            query = query.where(PersonalityProfile.is_active == True)
        
        if category:
            query = query.where(PersonalityProfile.category == category)
        
        query = query.order_by(desc(PersonalityProfile.usage_count)).limit(limit).offset(offset)
        
        result = await db.execute(query)
        profiles = result.scalars().all()
        
        return [
            {
                "id": str(profile.id),
                "name": profile.name,
                "display_name": profile.display_name,
                "description": profile.description,
                "category": profile.category,
                "trait_scores": profile.trait_scores,
                "adaptation_strategy": profile.adaptation_strategy.value,
                "usage_count": profile.usage_count,
                "average_satisfaction": profile.average_satisfaction_score,
                "is_active": profile.is_active,
                "is_default": profile.is_default,
                "created_at": profile.created_at.isoformat()
            }
            for profile in profiles
        ]
        
    except Exception as e:
        logger.error(f"Error listing personality profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles", response_model=Dict[str, Any])
async def create_personality_profile(
    request: CreatePersonalityProfileRequest,
    manager: PersonalityManager = Depends(get_personality_manager)
):
    """Create a new personality profile."""
    try:
        profile = await manager.create_custom_personality_profile(
            name=request.name,
            description=request.description,
            trait_scores=request.trait_scores,
            behavioral_patterns=request.behavioral_patterns,
            communication_style=request.communication_style
        )
        
        return {
            "status": "success",
            "profile_id": str(profile.id),
            "name": profile.name,
            "message": "Personality profile created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating personality profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{profile_id}", response_model=Dict[str, Any])
async def get_personality_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get detailed information about a personality profile."""
    try:
        query = select(PersonalityProfile).where(PersonalityProfile.id == profile_id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Personality profile not found")
        
        return {
            "id": str(profile.id),
            "name": profile.name,
            "display_name": profile.display_name,
            "description": profile.description,
            "category": profile.category,
            "trait_scores": profile.trait_scores,
            "behavioral_patterns": profile.behavioral_patterns,
            "communication_style": profile.communication_style,
            "adaptation_strategy": profile.adaptation_strategy.value,
            "adaptation_sensitivity": profile.adaptation_sensitivity,
            "adaptation_limits": profile.adaptation_limits,
            "usage_count": profile.usage_count,
            "average_satisfaction_score": profile.average_satisfaction_score,
            "performance_metrics": profile.performance_metrics,
            "is_active": profile.is_active,
            "is_default": profile.is_default,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting personality profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profiles/{profile_id}", response_model=Dict[str, Any])
async def update_personality_profile(
    profile_id: str,
    request: UpdatePersonalityProfileRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Update a personality profile."""
    try:
        query = select(PersonalityProfile).where(PersonalityProfile.id == profile_id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Personality profile not found")
        
        # Update fields
        update_data = request.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(profile, field, value)
        
        profile.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(profile)
        
        return {
            "status": "success",
            "profile_id": str(profile.id),
            "message": "Personality profile updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating personality profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profiles/{profile_id}")
async def delete_personality_profile(
    profile_id: str,
    soft_delete: bool = Query(True, description="Soft delete (deactivate) instead of hard delete"),
    db: AsyncSession = Depends(get_db_session)
):
    """Delete or deactivate a personality profile."""
    try:
        query = select(PersonalityProfile).where(PersonalityProfile.id == profile_id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Personality profile not found")
        
        if profile.is_default:
            raise HTTPException(status_code=400, detail="Cannot delete default personality profile")
        
        if soft_delete:
            profile.is_active = False
            profile.updated_at = datetime.utcnow()
            await db.commit()
            message = "Personality profile deactivated successfully"
        else:
            await db.delete(profile)
            await db.commit()
            message = "Personality profile deleted successfully"
        
        return {
            "status": "success",
            "profile_id": profile_id,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting personality profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    manager: PersonalityManager = Depends(get_personality_manager)
):
    """Get system-wide personality performance metrics."""
    try:
        metrics = await manager.get_system_performance_metrics()
        
        if metrics.get('status') == 'error':
            raise HTTPException(status_code=500, detail=metrics.get('message'))
        
        return SystemMetricsResponse(**metrics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/optimize")
async def optimize_personality_profiles(
    background_tasks: BackgroundTasks,
    manager: PersonalityManager = Depends(get_personality_manager)
):
    """Run personality profile optimization."""
    try:
        # Run optimization in background
        background_tasks.add_task(manager.optimize_personality_profiles)
        
        return {
            "status": "success",
            "message": "Personality optimization started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting personality optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/testing/ab-test", response_model=Dict[str, Any])
async def create_ab_test(
    request: CreateABTestRequest,
    framework: PersonalityTestingFramework = Depends(get_testing_framework)
):
    """Create a new A/B test for personality optimization."""
    try:
        test = await framework.create_ab_test(
            name=request.name,
            description=request.description,
            control_config=request.control_config,
            treatment_config=request.treatment_config,
            primary_metric=request.primary_metric,
            target_population=request.target_population,
            duration_days=request.duration_days,
            traffic_split=request.traffic_split,
            minimum_sample_size=request.minimum_sample_size
        )
        
        return {
            "status": "success",
            "test_id": test.id,
            "test_name": test.name,
            "message": "A/B test created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/testing/{test_id}/start")
async def start_test(
    test_id: str,
    framework: PersonalityTestingFramework = Depends(get_testing_framework)
):
    """Start a personality test."""
    try:
        success = await framework.start_test(test_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start test")
        
        return {
            "status": "success",
            "test_id": test_id,
            "message": "Test started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/testing/{test_id}/stop")
async def stop_test(
    test_id: str,
    reason: str = Body("Manual stop", description="Reason for stopping test"),
    framework: PersonalityTestingFramework = Depends(get_testing_framework)
):
    """Stop a running personality test."""
    try:
        success = await framework.stop_test(test_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop test")
        
        return {
            "status": "success",
            "test_id": test_id,
            "message": "Test stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/testing/{test_id}/results", response_model=Dict[str, Any])
async def get_test_results(
    test_id: str,
    metric_name: Optional[str] = Query(None, description="Specific metric to analyze"),
    framework: PersonalityTestingFramework = Depends(get_testing_framework)
):
    """Get results for a personality test."""
    try:
        results = await framework.get_test_results(test_id, metric_name)
        
        if 'error' in results:
            raise HTTPException(status_code=404, detail=results['error'])
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/testing/algorithms/performance", response_model=Dict[str, Any])
async def analyze_algorithm_performance(
    algorithms: List[str] = Query(..., description="Algorithms to compare"),
    metric: str = Query("user_satisfaction", description="Metric to analyze"),
    days_back: int = Query(30, ge=1, le=365, description="Days of data to analyze"),
    framework: PersonalityTestingFramework = Depends(get_testing_framework)
):
    """Analyze and compare personality algorithm performance."""
    try:
        analysis = await framework.analyze_personality_algorithm_performance(
            algorithm_names=algorithms,
            metric_name=metric,
            days_back=days_back
        )
        
        if 'error' in analysis:
            raise HTTPException(status_code=400, detail=analysis['error'])
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing algorithm performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/testing/benchmark", response_model=Dict[str, Any])
async def run_effectiveness_benchmark(
    personality_profiles: List[str] = Body(..., description="Personality profiles to benchmark"),
    test_scenarios: List[Dict[str, Any]] = Body(..., description="Test scenarios"),
    duration_hours: int = Body(24, ge=1, le=168, description="Benchmark duration in hours"),
    background_tasks: BackgroundTasks,
    framework: PersonalityTestingFramework = Depends(get_testing_framework)
):
    """Run comprehensive personality effectiveness benchmark."""
    try:
        # Start benchmark in background
        background_tasks.add_task(
            framework.run_personality_effectiveness_benchmark,
            personality_profiles,
            test_scenarios,
            duration_hours
        )
        
        return {
            "status": "success",
            "message": "Personality effectiveness benchmark started in background",
            "profiles_count": len(personality_profiles),
            "scenarios_count": len(test_scenarios),
            "estimated_duration_hours": duration_hours
        }
        
    except Exception as e:
        logger.error(f"Error starting effectiveness benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def personality_health_check():
    """Health check endpoint for personality system."""
    try:
        redis_client = await get_redis_client()
        
        # Test Redis connection
        await redis_client.ping()
        redis_status = "healthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": redis_status,
                "personality_system": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# WebSocket endpoint for real-time personality adaptation
@router.websocket("/ws/real-time-adaptation/{user_id}")
async def websocket_real_time_adaptation(websocket, user_id: str):
    """WebSocket endpoint for real-time personality adaptation monitoring."""
    try:
        await websocket.accept()
        
        # This would implement real-time streaming of personality adaptations
        # For now, sending periodic updates
        while True:
            # In practice, this would stream real personality adaptation events
            update = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "personality_update",
                "data": {
                    "current_traits": {"openness": 0.7, "extraversion": 0.6},
                    "adaptation_confidence": 0.85,
                    "last_interaction": "2 minutes ago"
                }
            }
            
            await websocket.send_json(update)
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Add rate limiting to sensitive endpoints
router.dependencies.append(Depends(RateLimitMiddleware(calls=100, period=60)))

# Export router
__all__ = ["router"]