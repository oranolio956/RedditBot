"""
Reality Synthesis API Endpoints

RESTful API for reality layer manipulation and immersive environment creation.
Provides comprehensive endpoints for creating, managing, and transitioning between
reality layers with spatial computing integration and safety monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from app.database.connection import get_db
from app.models.reality_synthesis import (
    RealityProfile, RealitySession, SpatialEnvironment,
    RealitySafetyMonitoring, CollaborativeRealitySession, PortalSystem,
    RealityLayer, SpatialComputingPlatform, RealityRenderingEngine,
    RealityTransitionType, TherapeuticRealityProtocol, SafetyLevel
)
from app.schemas.temporal_reality import (
    RealityProfileCreate, RealityProfileResponse,
    RealitySessionCreate, RealitySessionResponse,
    RealityExperienceRequest, RealityStateResponse,
    SpatialEnvironmentRequest, SpatialEnvironmentResponse,
    PortalTransitionRequest, CollaborativeSessionRequest,
    RealitySafetyResponse, EmergencyResetRequest
)
from app.services.reality_matrix import (
    RealityMatrix, RealityExperienceConfig, RealityStateReading, PortalTransitionConfig
)
from app.services.neural_environment_generator import NeuralEnvironmentGenerator
from app.core.redis import RedisManager
from app.core.monitoring import log_api_request
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/reality", tags=["Reality Synthesis"])

# Initialize reality matrix
reality_matrix = None
environment_generator = None

def get_reality_matrix():
    global reality_matrix
    if reality_matrix is None:
        redis_manager = RedisManager()
        reality_matrix = RealityMatrix(redis_manager)
    return reality_matrix

def get_environment_generator():
    global environment_generator
    if environment_generator is None:
        environment_generator = NeuralEnvironmentGenerator()
    return environment_generator

@router.post("/profiles", response_model=RealityProfileResponse)
async def create_reality_profile(
    profile_data: RealityProfileCreate,
    db: Session = Depends(get_db)
) -> RealityProfileResponse:
    """
    Create a new reality synthesis profile with platform capabilities,
    preferences, and safety parameters.
    """
    try:
        await log_api_request("create_reality_profile", {
            'user_id': profile_data.user_id,
            'supported_platforms_count': len(profile_data.supported_platforms or [])
        })
        
        # Create reality profile
        reality_profile = RealityProfile(
            user_id=profile_data.user_id,
            supported_platforms=profile_data.supported_platforms,
            platform_preferences=profile_data.platform_preferences,
            hardware_performance_profile=profile_data.hardware_performance_profile,
            spatial_awareness_score=profile_data.spatial_awareness_score,
            depth_perception_accuracy=profile_data.depth_perception_accuracy,
            motion_sickness_susceptibility=profile_data.motion_sickness_susceptibility,
            preferred_reality_layers=profile_data.preferred_reality_layers,
            reality_immersion_tolerance=profile_data.reality_immersion_tolerance,
            maximum_session_duration=profile_data.maximum_session_duration,
            comfort_zone_boundaries=profile_data.comfort_zone_boundaries,
            therapeutic_goals=profile_data.therapeutic_goals,
            contraindications=profile_data.contraindications,
            accessibility_accommodations=profile_data.accessibility_accommodations,
            created_at=datetime.utcnow()
        )
        
        db.add(reality_profile)
        db.commit()
        db.refresh(reality_profile)
        
        logger.info(
            "Reality profile created",
            profile_id=reality_profile.id,
            user_id=profile_data.user_id
        )
        
        return RealityProfileResponse.from_orm(reality_profile)
        
    except Exception as e:
        logger.error("Failed to create reality profile", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles/{profile_id}", response_model=RealityProfileResponse)
async def get_reality_profile(
    profile_id: int,
    db: Session = Depends(get_db)
) -> RealityProfileResponse:
    """
    Retrieve reality synthesis profile with current settings and capabilities.
    """
    try:
        reality_profile = db.query(RealityProfile).filter(
            RealityProfile.id == profile_id
        ).first()
        
        if not reality_profile:
            raise HTTPException(status_code=404, detail="Reality profile not found")
        
        await log_api_request("get_reality_profile", {
            'profile_id': profile_id,
            'user_id': reality_profile.user_id
        })
        
        return RealityProfileResponse.from_orm(reality_profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get reality profile", profile_id=profile_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions", response_model=RealitySessionResponse)
async def create_reality_session(
    session_request: RealityExperienceRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    matrix: RealityMatrix = Depends(get_reality_matrix)
) -> RealitySessionResponse:
    """
    Create and initiate a comprehensive reality synthesis experience
    with multi-layer support and safety monitoring.
    """
    try:
        # Get reality profile
        reality_profile = db.query(RealityProfile).filter(
            RealityProfile.id == session_request.profile_id
        ).first()
        
        if not reality_profile:
            raise HTTPException(status_code=404, detail="Reality profile not found")
        
        await log_api_request("create_reality_session", {
            'profile_id': session_request.profile_id,
            'primary_layer': session_request.primary_reality_layer.value,
            'platform': session_request.platform.value
        })
        
        # Create reality experience configuration
        config = RealityExperienceConfig(
            primary_layer=session_request.primary_reality_layer,
            target_layers=session_request.target_reality_layers,
            rendering_engine=session_request.rendering_engine,
            platform=session_request.platform,
            therapeutic_protocol=session_request.therapeutic_protocol,
            transition_types=session_request.transition_types or [],
            safety_thresholds=session_request.safety_thresholds or {},
            collaborative_enabled=session_request.collaborative_enabled,
            duration_minutes=session_request.duration_minutes
        )
        
        # Create reality experience
        reality_session = await matrix.create_reality_experience(
            reality_profile, config
        )
        
        # Store session in database
        db.add(reality_session)
        db.commit()
        db.refresh(reality_session)
        
        # Start background monitoring
        background_tasks.add_task(
            _monitor_reality_session,
            matrix,
            reality_session.session_uuid
        )
        
        logger.info(
            "Reality session created",
            session_uuid=reality_session.session_uuid,
            profile_id=session_request.profile_id
        )
        
        return RealitySessionResponse.from_orm(reality_session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create reality session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_uuid}/state", response_model=RealityStateResponse)
async def get_reality_state(
    session_uuid: str,
    matrix: RealityMatrix = Depends(get_reality_matrix)
) -> RealityStateResponse:
    """
    Get current reality synthesis state with presence analysis,
    technical performance, and safety monitoring.
    """
    try:
        state_reading = await matrix.monitor_reality_state(session_uuid)
        
        await log_api_request("get_reality_state", {
            'session_uuid': session_uuid,
            'current_layer': state_reading.current_layer.value,
            'presence_score': state_reading.presence_score
        })
        
        return RealityStateResponse(
            timestamp=state_reading.timestamp,
            session_uuid=session_uuid,
            current_layer=state_reading.current_layer,
            presence_score=state_reading.presence_score,
            immersion_depth=state_reading.immersion_depth,
            spatial_awareness=state_reading.spatial_awareness,
            safety_level=state_reading.safety_level,
            technical_performance=state_reading.technical_performance,
            user_comfort=state_reading.user_comfort
        )
        
    except Exception as e:
        logger.error("Failed to get reality state", session_uuid=session_uuid, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_uuid}/transition")
async def transition_reality_layer(
    session_uuid: str,
    transition_request: PortalTransitionRequest,
    matrix: RealityMatrix = Depends(get_reality_matrix)
) -> Dict[str, Any]:
    """
    Execute seamless transition between reality layers using portal system
    with comprehensive safety monitoring and spatial continuity.
    """
    try:
        await log_api_request("transition_reality_layer", {
            'session_uuid': session_uuid,
            'target_layer': transition_request.target_layer.value,
            'transition_type': transition_request.transition_type.value
        })
        
        # Create transition configuration
        transition_config = PortalTransitionConfig(
            source_layer=transition_request.source_layer,
            destination_layer=transition_request.target_layer,
            transition_type=transition_request.transition_type,
            transition_duration=transition_request.transition_duration,
            spatial_anchors=transition_request.spatial_anchors or [],
            safety_protocols=transition_request.safety_protocols or []
        )
        
        # Execute transition
        transition_result = await matrix.transition_reality_layer(
            session_uuid, transition_request.target_layer, transition_config
        )
        
        logger.info(
            "Reality layer transition executed",
            session_uuid=session_uuid,
            target_layer=transition_request.target_layer.value,
            success=transition_result['success']
        )
        
        return {
            'success': transition_result['success'],
            'transition_result': transition_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to transition reality layer",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/environments", response_model=SpatialEnvironmentResponse)
async def create_spatial_environment(
    environment_request: SpatialEnvironmentRequest,
    db: Session = Depends(get_db),
    generator: NeuralEnvironmentGenerator = Depends(get_environment_generator)
) -> SpatialEnvironmentResponse:
    """
    Generate immersive 3D environment using Neural Radiance Fields
    and advanced spatial computing techniques.
    """
    try:
        await log_api_request("create_spatial_environment", {
            'prompt_length': len(environment_request.generation_prompt),
            'layer_type': environment_request.layer_type.value,
            'therapeutic_purpose': environment_request.therapeutic_purpose.value if environment_request.therapeutic_purpose else None
        })
        
        # Generate environment
        environment_data = await generator.generate_nerf_environment(
            prompt=environment_request.generation_prompt,
            layer_type=environment_request.layer_type,
            therapeutic_context=environment_request.therapeutic_purpose,
            quality_preset=environment_request.quality_preset,
            platform_targets=environment_request.platform_targets
        )
        
        # Create spatial environment record
        spatial_environment = SpatialEnvironment(
            environment_uuid=f"env_{datetime.utcnow().timestamp()}",
            environment_name=environment_request.environment_name or f"Generated: {environment_request.generation_prompt[:50]}...",
            environment_category="ai_generated",
            reality_layer_compatibility=[environment_request.layer_type.value],
            target_use_cases=[environment_request.therapeutic_purpose.value] if environment_request.therapeutic_purpose else ["general"],
            neural_radiance_field_data=environment_data['nerf_model_data'],
            spatial_anchors=environment_data['spatial_anchors'],
            generation_prompt=environment_request.generation_prompt,
            ai_model_used="NeuralRadianceField_v3.2.0",
            webxr_compatibility=environment_data.get('platform_compatibility', {}).get('webxr_browser', {}),
            mobile_ar_optimization=environment_data.get('platform_compatibility', {}).get('mobile_ar', {}),
            desktop_vr_configuration=environment_data.get('platform_compatibility', {}).get('desktop_vr', {}),
            interactive_objects=environment_data.get('interaction_capabilities', {}),
            therapeutic_applications=[environment_request.therapeutic_purpose.value] if environment_request.therapeutic_purpose else [],
            content_safety_rating="approved",
            created_at=datetime.utcnow()
        )
        
        db.add(spatial_environment)
        db.commit()
        db.refresh(spatial_environment)
        
        logger.info(
            "Spatial environment created",
            environment_uuid=spatial_environment.environment_uuid,
            layer_type=environment_request.layer_type.value
        )
        
        return SpatialEnvironmentResponse.from_orm(spatial_environment)
        
    except Exception as e:
        logger.error("Failed to create spatial environment", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environments/{environment_uuid}", response_model=SpatialEnvironmentResponse)
async def get_spatial_environment(
    environment_uuid: str,
    db: Session = Depends(get_db)
) -> SpatialEnvironmentResponse:
    """
    Retrieve spatial environment with all configuration data.
    """
    try:
        environment = db.query(SpatialEnvironment).filter(
            SpatialEnvironment.environment_uuid == environment_uuid
        ).first()
        
        if not environment:
            raise HTTPException(status_code=404, detail="Spatial environment not found")
        
        await log_api_request("get_spatial_environment", {
            'environment_uuid': environment_uuid,
            'layer_compatibility': environment.reality_layer_compatibility
        })
        
        return SpatialEnvironmentResponse.from_orm(environment)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get spatial environment",
            environment_uuid=environment_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collaborative-sessions", response_model=Dict[str, Any])
async def create_collaborative_session(
    collab_request: CollaborativeSessionRequest,
    db: Session = Depends(get_db),
    matrix: RealityMatrix = Depends(get_reality_matrix)
) -> Dict[str, Any]:
    """
    Create multi-user collaborative reality session with
    personal object awareness and social safety monitoring.
    """
    try:
        # Get host profile
        host_profile = db.query(RealityProfile).filter(
            RealityProfile.id == collab_request.host_profile_id
        ).first()
        
        if not host_profile:
            raise HTTPException(status_code=404, detail="Host profile not found")
        
        await log_api_request("create_collaborative_session", {
            'host_profile_id': collab_request.host_profile_id,
            'max_participants': collab_request.max_participants
        })
        
        # Create session configuration
        session_config = RealityExperienceConfig(
            primary_layer=collab_request.primary_reality_layer,
            target_layers=collab_request.target_reality_layers or [collab_request.primary_reality_layer],
            rendering_engine=collab_request.rendering_engine,
            platform=collab_request.platform,
            therapeutic_protocol=collab_request.therapeutic_protocol,
            transition_types=[],
            safety_thresholds={},
            collaborative_enabled=True,
            duration_minutes=collab_request.duration_minutes
        )
        
        # Create collaborative session
        collaborative_session = await matrix.create_collaborative_session(
            host_profile, session_config, collab_request.max_participants
        )
        
        # Store in database
        db.add(collaborative_session)
        db.commit()
        db.refresh(collaborative_session)
        
        logger.info(
            "Collaborative session created",
            session_uuid=collaborative_session.collaborative_session_uuid,
            max_participants=collab_request.max_participants
        )
        
        return {
            'success': True,
            'collaborative_session_uuid': collaborative_session.collaborative_session_uuid,
            'session_host_user_id': collaborative_session.session_host_user_id,
            'maximum_participants': collaborative_session.maximum_participants,
            'session_privacy_level': collaborative_session.session_privacy_level,
            'created_at': collaborative_session.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create collaborative session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_uuid}/emergency-reset")
async def emergency_reality_reset(
    session_uuid: str,
    reset_request: EmergencyResetRequest,
    matrix: RealityMatrix = Depends(get_reality_matrix)
) -> Dict[str, Any]:
    """
    Emergency protocol to immediately return user to safe reality state
    in case of disorientation, motion sickness, or technical failures.
    """
    try:
        await log_api_request("emergency_reality_reset", {
            'session_uuid': session_uuid,
            'reason': reset_request.reason
        })
        
        reset_result = await matrix.emergency_reality_reset(
            session_uuid, reset_request.reason
        )
        
        logger.critical(
            "Emergency reality reset executed",
            session_uuid=session_uuid,
            reason=reset_request.reason,
            success=reset_result['success']
        )
        
        return {
            'success': reset_result['success'],
            'reset_result': reset_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to execute emergency reality reset",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_uuid}/safety", response_model=RealitySafetyResponse)
async def get_reality_safety_status(
    session_uuid: str,
    db: Session = Depends(get_db)
) -> RealitySafetyResponse:
    """
    Get comprehensive safety status for reality synthesis session
    including dissociation risk and intervention history.
    """
    try:
        # Get latest safety monitoring record
        safety_record = db.query(RealitySafetyMonitoring).filter(
            RealitySafetyMonitoring.reality_session_id == session_uuid
        ).order_by(RealitySafetyMonitoring.monitoring_started_at.desc()).first()
        
        if not safety_record:
            raise HTTPException(status_code=404, detail="Safety monitoring record not found")
        
        await log_api_request("get_reality_safety_status", {
            'session_uuid': session_uuid,
            'current_safety_level': safety_record.current_safety_level.value
        })
        
        return RealitySafetyResponse(
            session_uuid=session_uuid,
            current_safety_level=safety_record.current_safety_level,
            safety_trajectory=safety_record.safety_trajectory,
            reality_confusion_markers=safety_record.reality_confusion_markers,
            motion_sickness_events=safety_record.motion_sickness_events,
            safety_interventions_triggered=safety_record.safety_interventions_triggered,
            intervention_effectiveness=safety_record.intervention_effectiveness_scores,
            monitoring_started_at=safety_record.monitoring_started_at,
            last_updated=safety_record.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get reality safety status",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=List[RealitySessionResponse])
async def list_reality_sessions(
    profile_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[RealitySessionResponse]:
    """
    List reality synthesis sessions with optional filtering by profile.
    """
    try:
        query = db.query(RealitySession)
        
        if profile_id:
            query = query.filter(RealitySession.reality_profile_id == profile_id)
        
        sessions = query.order_by(
            RealitySession.started_at.desc()
        ).offset(offset).limit(limit).all()
        
        await log_api_request("list_reality_sessions", {
            'profile_id': profile_id,
            'count': len(sessions),
            'limit': limit,
            'offset': offset
        })
        
        return [RealitySessionResponse.from_orm(session) for session in sessions]
        
    except Exception as e:
        logger.error("Failed to list reality sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environments")
async def list_spatial_environments(
    layer_type: Optional[RealityLayer] = None,
    therapeutic_purpose: Optional[TherapeuticRealityProtocol] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    List available spatial environments with optional filtering.
    """
    try:
        query = db.query(SpatialEnvironment)
        
        if layer_type:
            # Filter by layer compatibility (JSON contains check)
            query = query.filter(
                SpatialEnvironment.reality_layer_compatibility.contains([layer_type.value])
            )
        
        if therapeutic_purpose:
            query = query.filter(
                SpatialEnvironment.therapeutic_applications.contains([therapeutic_purpose.value])
            )
        
        environments = query.order_by(
            SpatialEnvironment.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        await log_api_request("list_spatial_environments", {
            'layer_type': layer_type.value if layer_type else None,
            'therapeutic_purpose': therapeutic_purpose.value if therapeutic_purpose else None,
            'count': len(environments),
            'limit': limit,
            'offset': offset
        })
        
        return {
            'environments': [
                {
                    'environment_uuid': env.environment_uuid,
                    'environment_name': env.environment_name,
                    'environment_category': env.environment_category,
                    'reality_layer_compatibility': env.reality_layer_compatibility,
                    'target_use_cases': env.target_use_cases,
                    'therapeutic_applications': env.therapeutic_applications,
                    'content_safety_rating': env.content_safety_rating,
                    'created_at': env.created_at.isoformat(),
                    'generation_prompt': env.generation_prompt[:100] + "..." if len(env.generation_prompt or "") > 100 else env.generation_prompt
                } for env in environments
            ],
            'total_count': len(environments),
            'filters_applied': {
                'layer_type': layer_type.value if layer_type else None,
                'therapeutic_purpose': therapeutic_purpose.value if therapeutic_purpose else None
            }
        }
        
    except Exception as e:
        logger.error("Failed to list spatial environments", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Background task for session monitoring
async def _monitor_reality_session(
    matrix: RealityMatrix,
    session_uuid: str
):
    """Background monitoring task for reality session"""
    try:
        while session_uuid in matrix.active_sessions:
            state_reading = await matrix.monitor_reality_state(session_uuid)
            
            # Check if intervention is needed
            if state_reading.safety_level != SafetyLevel.SAFE:
                # Apply safety adaptations
                await matrix._apply_safety_adaptations(session_uuid, state_reading)
            
            await asyncio.sleep(3)  # Monitor every 3 seconds
            
    except Exception as e:
        logger.error(
            "Background monitoring failed",
            session_uuid=session_uuid,
            error=str(e)
        )

@router.get("/health")
async def reality_health_check() -> Dict[str, Any]:
    """Health check endpoint for reality synthesis system"""
    return {
        'status': 'healthy',
        'service': 'reality_synthesis',
        'version': '2.1.0',
        'timestamp': datetime.utcnow().isoformat(),
        'features': {
            'reality_layer_synthesis': True,
            'neural_radiance_fields': True,
            'portal_transitions': True,
            'collaborative_sessions': True,
            'spatial_computing': True,
            'safety_monitoring': True,
            'webxr_compatibility': True
        }
    }
