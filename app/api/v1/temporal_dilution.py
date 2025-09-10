"""
Temporal Dilution API Endpoints

RESTful API for temporal perception manipulation and flow state induction.
Provides comprehensive endpoints for creating, monitoring, and managing
temporal dilution experiences with safety monitoring and therapeutic protocols.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from app.database.connection import get_db
from app.models.temporal_dilution import (
    TemporalProfile, TemporalSession, FlowStateSession,
    TemporalBiometricReading, TemporalSafetyMonitoring,
    TemporalState, FlowStateType, TemporalCueType, SafetyLevel
)
from app.schemas.temporal_reality import (
    TemporalProfileCreate, TemporalProfileResponse,
    TemporalSessionCreate, TemporalSessionResponse,
    TemporalExperienceRequest, TemporalStateResponse,
    FlowStateInductionRequest, FlowStateResponse,
    TemporalSafetyResponse, EmergencyResetRequest
)
from app.services.temporal_engine import (
    TemporalEngine, TemporalExperienceConfig, TemporalStateReading
)
from app.core.redis import RedisManager
from app.core.monitoring import log_api_request
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/temporal", tags=["Temporal Dilution"])

# Initialize temporal engine
temporal_engine = None

def get_temporal_engine():
    global temporal_engine
    if temporal_engine is None:
        redis_manager = RedisManager()
        temporal_engine = TemporalEngine(redis_manager)
    return temporal_engine

@router.post("/profiles", response_model=TemporalProfileResponse)
async def create_temporal_profile(
    profile_data: TemporalProfileCreate,
    db: Session = Depends(get_db)
) -> TemporalProfileResponse:
    """
    Create a new temporal dilution profile with personalized parameters,
    baseline measurements, and safety configurations.
    """
    try:
        await log_api_request("create_temporal_profile", {
            'user_id': profile_data.user_id,
            'chronotype': profile_data.chronotype_classification
        })
        
        # Create temporal profile
        temporal_profile = TemporalProfile(
            user_id=profile_data.user_id,
            baseline_time_perception_accuracy=profile_data.baseline_time_perception_accuracy,
            natural_rhythm_period=profile_data.natural_rhythm_period,
            temporal_sensitivity_score=profile_data.temporal_sensitivity_score,
            chronotype_classification=profile_data.chronotype_classification,
            flow_state_frequency=profile_data.flow_state_frequency,
            optimal_flow_duration=profile_data.optimal_flow_duration,
            preferred_temporal_state=profile_data.preferred_temporal_state,
            optimal_learning_time_ratio=profile_data.optimal_learning_time_ratio,
            maximum_safe_temporal_distortion=profile_data.maximum_safe_temporal_distortion,
            disorientation_sensitivity=profile_data.disorientation_sensitivity,
            preferred_temporal_cue_types=profile_data.preferred_temporal_cue_types,
            contraindications=profile_data.contraindications,
            created_at=datetime.utcnow()
        )
        
        db.add(temporal_profile)
        db.commit()
        db.refresh(temporal_profile)
        
        logger.info(
            "Temporal profile created",
            profile_id=temporal_profile.id,
            user_id=profile_data.user_id
        )
        
        return TemporalProfileResponse.from_orm(temporal_profile)
        
    except Exception as e:
        logger.error("Failed to create temporal profile", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles/{profile_id}", response_model=TemporalProfileResponse)
async def get_temporal_profile(
    profile_id: int,
    db: Session = Depends(get_db)
) -> TemporalProfileResponse:
    """
    Retrieve temporal dilution profile with current settings and capabilities.
    """
    try:
        temporal_profile = db.query(TemporalProfile).filter(
            TemporalProfile.id == profile_id
        ).first()
        
        if not temporal_profile:
            raise HTTPException(status_code=404, detail="Temporal profile not found")
        
        await log_api_request("get_temporal_profile", {
            'profile_id': profile_id,
            'user_id': temporal_profile.user_id
        })
        
        return TemporalProfileResponse.from_orm(temporal_profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get temporal profile", profile_id=profile_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions", response_model=TemporalSessionResponse)
async def create_temporal_session(
    session_request: TemporalExperienceRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    engine: TemporalEngine = Depends(get_temporal_engine)
) -> TemporalSessionResponse:
    """
    Create and initiate a comprehensive temporal dilution experience
    with real-time monitoring and safety protocols.
    """
    try:
        # Get temporal profile
        temporal_profile = db.query(TemporalProfile).filter(
            TemporalProfile.id == session_request.profile_id
        ).first()
        
        if not temporal_profile:
            raise HTTPException(status_code=404, detail="Temporal profile not found")
        
        await log_api_request("create_temporal_session", {
            'profile_id': session_request.profile_id,
            'target_state': session_request.target_temporal_state.value,
            'duration_minutes': session_request.duration_minutes
        })
        
        # Create temporal experience configuration
        config = TemporalExperienceConfig(
            target_state=session_request.target_temporal_state,
            target_dilation_ratio=session_request.target_dilation_ratio,
            duration_minutes=session_request.duration_minutes,
            flow_state_type=session_request.flow_state_type,
            temporal_cues=session_request.temporal_cues,
            safety_thresholds=session_request.safety_thresholds or {},
            personalization_level=session_request.personalization_level,
            circadian_optimization=session_request.circadian_optimization
        )
        
        # Create temporal experience
        temporal_session = await engine.create_temporal_experience(
            temporal_profile, config
        )
        
        # Store session in database
        db.add(temporal_session)
        db.commit()
        db.refresh(temporal_session)
        
        # Start background monitoring
        background_tasks.add_task(
            _monitor_temporal_session,
            engine,
            temporal_session.session_uuid
        )
        
        logger.info(
            "Temporal session created",
            session_uuid=temporal_session.session_uuid,
            profile_id=session_request.profile_id
        )
        
        return TemporalSessionResponse.from_orm(temporal_session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create temporal session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_uuid}/state", response_model=TemporalStateResponse)
async def get_temporal_state(
    session_uuid: str,
    engine: TemporalEngine = Depends(get_temporal_engine)
) -> TemporalStateResponse:
    """
    Get current temporal perception state with biometric analysis,
    flow depth assessment, and safety monitoring.
    """
    try:
        state_reading = await engine.monitor_temporal_state(session_uuid)
        
        await log_api_request("get_temporal_state", {
            'session_uuid': session_uuid,
            'detected_state': state_reading.detected_state.value,
            'safety_level': state_reading.safety_level.value
        })
        
        return TemporalStateResponse(
            timestamp=state_reading.timestamp,
            session_uuid=session_uuid,
            detected_state=state_reading.detected_state,
            confidence=state_reading.confidence,
            time_dilation_ratio=state_reading.time_dilation_ratio,
            flow_depth=state_reading.flow_depth,
            safety_level=state_reading.safety_level,
            biometric_data=state_reading.biometric_data,
            intervention_needed=state_reading.intervention_needed
        )
        
    except Exception as e:
        logger.error("Failed to get temporal state", session_uuid=session_uuid, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_uuid}/flow-state", response_model=FlowStateResponse)
async def induce_flow_state(
    session_uuid: str,
    flow_request: FlowStateInductionRequest,
    engine: TemporalEngine = Depends(get_temporal_engine)
) -> FlowStateResponse:
    """
    Induce specific flow state using temporal dilution and
    multi-modal neural entrainment techniques.
    """
    try:
        await log_api_request("induce_flow_state", {
            'session_uuid': session_uuid,
            'flow_type': flow_request.flow_type.value
        })
        
        flow_result = await engine.induce_flow_state(
            session_uuid, flow_request.flow_type
        )
        
        logger.info(
            "Flow state induction attempted",
            session_uuid=session_uuid,
            flow_type=flow_request.flow_type.value,
            achieved=flow_result['flow_state_achieved']
        )
        
        return FlowStateResponse(
            session_uuid=session_uuid,
            flow_type=flow_request.flow_type,
            flow_state_achieved=flow_result['flow_state_achieved'],
            time_to_flow_entry=flow_result.get('time_to_flow_entry'),
            flow_depth_score=flow_result['flow_depth_score'],
            neural_entrainment_success=flow_result['neural_entrainment_success'],
            temporal_dilation_applied=flow_result['temporal_dilation_applied'],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(
            "Failed to induce flow state",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/sessions/{session_uuid}/adapt")
async def adapt_temporal_experience(
    session_uuid: str,
    engine: TemporalEngine = Depends(get_temporal_engine)
) -> Dict[str, Any]:
    """
    Trigger real-time adaptation of temporal experience based on current state,
    biometric feedback, and safety considerations.
    """
    try:
        # Get current state
        state_reading = await engine.monitor_temporal_state(session_uuid)
        
        # Apply adaptations
        adaptations = await engine.adapt_temporal_experience(
            session_uuid, state_reading
        )
        
        await log_api_request("adapt_temporal_experience", {
            'session_uuid': session_uuid,
            'changes_made': len(adaptations['changes_made']),
            'safety_level': state_reading.safety_level.value
        })
        
        logger.info(
            "Temporal experience adapted",
            session_uuid=session_uuid,
            changes_made=len(adaptations['changes_made'])
        )
        
        return {
            'success': True,
            'adaptations_applied': adaptations,
            'current_state': {
                'detected_state': state_reading.detected_state.value,
                'safety_level': state_reading.safety_level.value,
                'flow_depth': state_reading.flow_depth
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to adapt temporal experience",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_uuid}/emergency-reset")
async def emergency_temporal_reset(
    session_uuid: str,
    reset_request: EmergencyResetRequest,
    engine: TemporalEngine = Depends(get_temporal_engine)
) -> Dict[str, Any]:
    """
    Emergency protocol to immediately return user to baseline temporal state
    in case of disorientation, safety concerns, or adverse effects.
    """
    try:
        await log_api_request("emergency_temporal_reset", {
            'session_uuid': session_uuid,
            'reason': reset_request.reason
        })
        
        reset_result = await engine.emergency_temporal_reset(
            session_uuid, reset_request.reason
        )
        
        logger.critical(
            "Emergency temporal reset executed",
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
            "Failed to execute emergency temporal reset",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_uuid}/safety", response_model=TemporalSafetyResponse)
async def get_safety_status(
    session_uuid: str,
    db: Session = Depends(get_db)
) -> TemporalSafetyResponse:
    """
    Get comprehensive safety status for temporal dilution session
    including risk indicators and intervention history.
    """
    try:
        # Get latest safety monitoring record
        safety_record = db.query(TemporalSafetyMonitoring).filter(
            TemporalSafetyMonitoring.temporal_session_id == session_uuid
        ).order_by(TemporalSafetyMonitoring.monitoring_started_at.desc()).first()
        
        if not safety_record:
            raise HTTPException(status_code=404, detail="Safety monitoring record not found")
        
        await log_api_request("get_safety_status", {
            'session_uuid': session_uuid,
            'current_safety_level': safety_record.current_safety_level.value
        })
        
        return TemporalSafetyResponse(
            session_uuid=session_uuid,
            current_safety_level=safety_record.current_safety_level,
            safety_trajectory=safety_record.safety_trajectory,
            temporal_confusion_markers=safety_record.temporal_confusion_markers,
            safety_interventions_triggered=safety_record.safety_interventions_triggered,
            intervention_effectiveness=safety_record.intervention_effectiveness_scores,
            monitoring_started_at=safety_record.monitoring_started_at,
            last_updated=safety_record.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get safety status",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=List[TemporalSessionResponse])
async def list_temporal_sessions(
    profile_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[TemporalSessionResponse]:
    """
    List temporal dilution sessions with optional filtering by profile.
    """
    try:
        query = db.query(TemporalSession)
        
        if profile_id:
            query = query.filter(TemporalSession.temporal_profile_id == profile_id)
        
        sessions = query.order_by(
            TemporalSession.started_at.desc()
        ).offset(offset).limit(limit).all()
        
        await log_api_request("list_temporal_sessions", {
            'profile_id': profile_id,
            'count': len(sessions),
            'limit': limit,
            'offset': offset
        })
        
        return [TemporalSessionResponse.from_orm(session) for session in sessions]
        
    except Exception as e:
        logger.error("Failed to list temporal sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/biometrics/{session_uuid}")
async def get_biometric_data(
    session_uuid: str,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get biometric readings for temporal dilution session.
    """
    try:
        # Get session
        session = db.query(TemporalSession).filter(
            TemporalSession.session_uuid == session_uuid
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get biometric readings
        readings = db.query(TemporalBiometricReading).filter(
            TemporalBiometricReading.temporal_session_id == session.id
        ).order_by(
            TemporalBiometricReading.timestamp.desc()
        ).limit(limit).all()
        
        await log_api_request("get_biometric_data", {
            'session_uuid': session_uuid,
            'readings_count': len(readings)
        })
        
        return {
            'session_uuid': session_uuid,
            'readings_count': len(readings),
            'readings': [
                {
                    'timestamp': reading.timestamp.isoformat(),
                    'device_type': reading.device_type.value,
                    'eeg_data': {
                        'alpha_power': reading.eeg_alpha_power,
                        'beta_power': reading.eeg_beta_power,
                        'theta_power': reading.eeg_theta_power,
                        'delta_power': reading.eeg_delta_power,
                        'gamma_power': reading.eeg_gamma_power
                    },
                    'temporal_indicators': {
                        'intrinsic_neural_timescales': reading.intrinsic_neural_timescales,
                        'temporal_prediction_error': reading.temporal_prediction_error
                    },
                    'flow_indicators': {
                        'frontal_alpha_asymmetry': reading.frontal_alpha_asymmetry
                    },
                    'signal_quality': reading.signal_quality_score
                } for reading in readings
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get biometric data",
            session_uuid=session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

# Background task for session monitoring
async def _monitor_temporal_session(
    engine: TemporalEngine,
    session_uuid: str
):
    """Background monitoring task for temporal session"""
    try:
        while session_uuid in engine.active_sessions:
            state_reading = await engine.monitor_temporal_state(session_uuid)
            
            # Check if adaptation is needed
            if (state_reading.safety_level != SafetyLevel.SAFE or
                state_reading.intervention_needed):
                await engine.adapt_temporal_experience(session_uuid, state_reading)
            
            await asyncio.sleep(5)  # Monitor every 5 seconds
            
    except Exception as e:
        logger.error(
            "Background monitoring failed",
            session_uuid=session_uuid,
            error=str(e)
        )

@router.get("/health")
async def temporal_health_check() -> Dict[str, Any]:
    """Health check endpoint for temporal dilution system"""
    return {
        'status': 'healthy',
        'service': 'temporal_dilution',
        'version': '2.1.0',
        'timestamp': datetime.utcnow().isoformat(),
        'features': {
            'temporal_manipulation': True,
            'flow_state_induction': True,
            'neural_monitoring': True,
            'safety_protocols': True,
            'circadian_integration': True
        }
    }
