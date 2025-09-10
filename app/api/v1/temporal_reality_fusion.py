"""
Temporal-Reality Fusion API Endpoints

RESTful API for integrated temporal-reality fusion experiences.
Provides comprehensive endpoints for creating, monitoring, and managing
the world's first temporal-reality manipulation platform.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from app.database.connection import get_db
from app.models.temporal_dilution import TemporalProfile, TemporalSession
from app.models.reality_synthesis import RealityProfile, RealitySession
from app.schemas.temporal_reality import (
    TemporalRealityFusionRequest, TemporalRealityFusionResponse,
    TemporalRealityStateResponse, TherapeuticProtocolRequest,
    TherapeuticProtocolResponse, EmergencyResetRequest
)
from app.services.temporal_reality_fusion import (
    TemporalRealityFusion, TemporalRealityConfig, TemporalRealityState
)
from app.services.temporal_engine import TemporalEngine, TemporalExperienceConfig
from app.services.reality_matrix import RealityMatrix, RealityExperienceConfig
from app.core.redis import RedisManager
from app.core.monitoring import log_api_request
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/fusion", tags=["Temporal-Reality Fusion"])

# Initialize fusion system
fusion_engine = None

def get_fusion_engine():
    global fusion_engine
    if fusion_engine is None:
        redis_manager = RedisManager()
        fusion_engine = TemporalRealityFusion(redis_manager)
    return fusion_engine

@router.post("/sessions", response_model=TemporalRealityFusionResponse)
async def create_fusion_session(
    fusion_request: TemporalRealityFusionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> TemporalRealityFusionResponse:
    """
    Create revolutionary temporal-reality fusion experience combining
    time perception manipulation with immersive reality synthesis.
    """
    try:
        # Get temporal and reality profiles
        temporal_profile = db.query(TemporalProfile).filter(
            TemporalProfile.id == fusion_request.temporal_profile_id
        ).first()
        
        reality_profile = db.query(RealityProfile).filter(
            RealityProfile.id == fusion_request.reality_profile_id
        ).first()
        
        if not temporal_profile:
            raise HTTPException(status_code=404, detail="Temporal profile not found")
        
        if not reality_profile:
            raise HTTPException(status_code=404, detail="Reality profile not found")
        
        await log_api_request("create_fusion_session", {
            'temporal_profile_id': fusion_request.temporal_profile_id,
            'reality_profile_id': fusion_request.reality_profile_id,
            'synchronization_mode': fusion_request.synchronization_mode,
            'therapeutic_integration': fusion_request.therapeutic_integration_level
        })
        
        # Create fusion configuration
        fusion_config = TemporalRealityConfig(
            temporal_config=TemporalExperienceConfig(
                target_state=fusion_request.temporal_config.target_temporal_state,
                target_dilation_ratio=fusion_request.temporal_config.target_dilation_ratio,
                duration_minutes=fusion_request.temporal_config.duration_minutes,
                flow_state_type=fusion_request.temporal_config.flow_state_type,
                temporal_cues=fusion_request.temporal_config.temporal_cues,
                safety_thresholds=fusion_request.temporal_config.safety_thresholds or {},
                personalization_level=fusion_request.temporal_config.personalization_level,
                circadian_optimization=fusion_request.temporal_config.circadian_optimization
            ),
            reality_config=RealityExperienceConfig(
                primary_layer=fusion_request.reality_config.primary_reality_layer,
                target_layers=fusion_request.reality_config.target_reality_layers or [fusion_request.reality_config.primary_reality_layer],
                rendering_engine=fusion_request.reality_config.rendering_engine,
                platform=fusion_request.reality_config.platform,
                therapeutic_protocol=fusion_request.reality_config.therapeutic_protocol,
                transition_types=fusion_request.reality_config.transition_types or [],
                safety_thresholds=fusion_request.reality_config.safety_thresholds or {},
                collaborative_enabled=fusion_request.reality_config.collaborative_enabled,
                duration_minutes=fusion_request.reality_config.duration_minutes
            ),
            synchronization_mode=fusion_request.synchronization_mode,
            temporal_reality_ratio=fusion_request.temporal_reality_ratio,
            cross_modal_enhancement=fusion_request.cross_modal_enhancement,
            therapeutic_integration_level=fusion_request.therapeutic_integration_level,
            safety_priority=fusion_request.safety_priority
        )
        
        # Create fusion experience
        temporal_session, reality_session = await fusion.create_fusion_experience(
            temporal_profile, reality_profile, fusion_config
        )
        
        # Store sessions in database
        db.add(temporal_session)
        db.add(reality_session)
        db.commit()
        db.refresh(temporal_session)
        db.refresh(reality_session)
        
        fusion_session_uuid = f"fusion_{temporal_session.session_uuid}_{reality_session.session_uuid}"
        
        # Start background monitoring
        background_tasks.add_task(
            _monitor_fusion_session,
            fusion,
            fusion_session_uuid
        )
        
        logger.info(
            "Temporal-reality fusion session created",
            fusion_session_uuid=fusion_session_uuid,
            temporal_profile_id=fusion_request.temporal_profile_id,
            reality_profile_id=fusion_request.reality_profile_id
        )
        
        return TemporalRealityFusionResponse(
            fusion_session_uuid=fusion_session_uuid,
            temporal_session=temporal_session,
            reality_session=reality_session,
            synchronization_mode=fusion_request.synchronization_mode,
            temporal_reality_ratio=fusion_request.temporal_reality_ratio,
            cross_modal_enhancement=fusion_request.cross_modal_enhancement,
            therapeutic_integration_level=fusion_request.therapeutic_integration_level,
            safety_priority=fusion_request.safety_priority,
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create fusion session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{fusion_session_uuid}/state", response_model=TemporalRealityStateResponse)
async def get_fusion_state(
    fusion_session_uuid: str,
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> TemporalRealityStateResponse:
    """
    Get comprehensive temporal-reality fusion state with synchronization analysis,
    cross-modal coherence assessment, and integrated safety monitoring.
    """
    try:
        fusion_state = await fusion.monitor_fusion_state(fusion_session_uuid)
        
        await log_api_request("get_fusion_state", {
            'fusion_session_uuid': fusion_session_uuid,
            'synchronization_score': fusion_state.fusion_synchronization_score,
            'overall_safety_level': fusion_state.overall_safety_level
        })
        
        return TemporalRealityStateResponse(
            timestamp=fusion_state.timestamp,
            fusion_session_uuid=fusion_session_uuid,
            temporal_state=fusion_state.temporal_state,
            reality_state=fusion_state.reality_state,
            fusion_synchronization_score=fusion_state.fusion_synchronization_score,
            cross_modal_coherence=fusion_state.cross_modal_coherence,
            overall_safety_level=fusion_state.overall_safety_level,
            therapeutic_progress_indicators=fusion_state.therapeutic_progress_indicators,
            user_agency_level=fusion_state.user_agency_level
        )
        
    except Exception as e:
        logger.error("Failed to get fusion state", fusion_session_uuid=fusion_session_uuid, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/sessions/{fusion_session_uuid}/adapt")
async def adapt_fusion_experience(
    fusion_session_uuid: str,
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> Dict[str, Any]:
    """
    Trigger real-time adaptation of fusion experience with synchronized adjustments
    to both temporal and reality systems based on integrated analysis.
    """
    try:
        # Get current fusion state
        fusion_state = await fusion.monitor_fusion_state(fusion_session_uuid)
        
        # Apply adaptations
        adaptations = await fusion.adapt_fusion_experience(
            fusion_session_uuid, fusion_state
        )
        
        await log_api_request("adapt_fusion_experience", {
            'fusion_session_uuid': fusion_session_uuid,
            'total_changes': sum(len(changes) for changes in adaptations.values()),
            'synchronization_score': fusion_state.fusion_synchronization_score
        })
        
        logger.info(
            "Fusion experience adapted",
            fusion_session_uuid=fusion_session_uuid,
            total_changes=sum(len(changes) for changes in adaptations.values())
        )
        
        return {
            'success': True,
            'adaptations_applied': adaptations,
            'current_state': {
                'synchronization_score': fusion_state.fusion_synchronization_score,
                'cross_modal_coherence': fusion_state.cross_modal_coherence,
                'overall_safety_level': fusion_state.overall_safety_level
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to adapt fusion experience",
            fusion_session_uuid=fusion_session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{fusion_session_uuid}/therapeutic-protocol", response_model=TherapeuticProtocolResponse)
async def execute_therapeutic_protocol(
    fusion_session_uuid: str,
    protocol_request: TherapeuticProtocolRequest,
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> TherapeuticProtocolResponse:
    """
    Execute integrated therapeutic protocol combining temporal and reality interventions
    for enhanced clinical outcomes with measurable therapeutic progress.
    """
    try:
        await log_api_request("execute_therapeutic_protocol", {
            'fusion_session_uuid': fusion_session_uuid,
            'protocol_name': protocol_request.protocol_name,
            'professional_oversight': protocol_request.professional_oversight_required
        })
        
        protocol_result = await fusion.execute_therapeutic_protocol(
            fusion_session_uuid, protocol_request.protocol_name
        )
        
        logger.info(
            "Therapeutic protocol executed",
            fusion_session_uuid=fusion_session_uuid,
            protocol_name=protocol_request.protocol_name,
            success=protocol_result['success']
        )
        
        return TherapeuticProtocolResponse(
            fusion_session_uuid=fusion_session_uuid,
            protocol_name=protocol_request.protocol_name,
            execution_success=protocol_result['success'],
            phases_completed=protocol_result['protocol_session']['phases'],
            therapeutic_outcomes=protocol_result['therapeutic_outcomes'],
            safety_events=protocol_result['protocol_session']['safety_events'],
            professional_recommendations=protocol_result['recommendations'],
            follow_up_required=len(protocol_result['recommendations']) > 0,
            session_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(
            "Failed to execute therapeutic protocol",
            fusion_session_uuid=fusion_session_uuid,
            protocol_name=protocol_request.protocol_name,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{fusion_session_uuid}/emergency-reset")
async def emergency_fusion_reset(
    fusion_session_uuid: str,
    reset_request: EmergencyResetRequest,
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> Dict[str, Any]:
    """
    Emergency protocol to safely terminate fusion experience and
    return user to baseline temporal and reality state immediately.
    """
    try:
        await log_api_request("emergency_fusion_reset", {
            'fusion_session_uuid': fusion_session_uuid,
            'reason': reset_request.reason
        })
        
        reset_result = await fusion.emergency_fusion_reset(
            fusion_session_uuid, reset_request.reason
        )
        
        logger.critical(
            "Emergency fusion reset executed",
            fusion_session_uuid=fusion_session_uuid,
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
            "Failed to execute emergency fusion reset",
            fusion_session_uuid=fusion_session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{fusion_session_uuid}/synchronization")
async def get_synchronization_status(
    fusion_session_uuid: str,
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> Dict[str, Any]:
    """
    Get detailed synchronization analysis between temporal and reality systems.
    """
    try:
        fusion_state = await fusion.monitor_fusion_state(fusion_session_uuid)
        
        await log_api_request("get_synchronization_status", {
            'fusion_session_uuid': fusion_session_uuid,
            'synchronization_score': fusion_state.fusion_synchronization_score
        })
        
        # Detailed synchronization analysis
        sync_analysis = await fusion._analyze_fusion_synchronization(
            fusion_state.temporal_state, fusion_state.reality_state
        )
        
        return {
            'fusion_session_uuid': fusion_session_uuid,
            'synchronization_score': fusion_state.fusion_synchronization_score,
            'synchronization_quality': sync_analysis['sync_quality'],
            'sync_factors': sync_analysis['sync_factors'],
            'cross_modal_coherence': fusion_state.cross_modal_coherence,
            'recommendations': await fusion._generate_synchronization_recommendations(
                sync_analysis
            ),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to get synchronization status",
            fusion_session_uuid=fusion_session_uuid,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_fusion_sessions(
    temporal_profile_id: Optional[int] = None,
    reality_profile_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> Dict[str, Any]:
    """
    List active and recent temporal-reality fusion sessions.
    """
    try:
        active_sessions = []
        for fusion_uuid, session_data in fusion.active_fusion_sessions.items():
            temporal_profile = session_data['temporal_profile']
            reality_profile = session_data['reality_profile']
            
            # Apply filters
            if temporal_profile_id and temporal_profile.id != temporal_profile_id:
                continue
            if reality_profile_id and reality_profile.id != reality_profile_id:
                continue
            
            active_sessions.append({
                'fusion_session_uuid': fusion_uuid,
                'temporal_profile_id': temporal_profile.id,
                'reality_profile_id': reality_profile.id,
                'start_time': session_data['start_time'].isoformat(),
                'synchronization_score': session_data['synchronization_score'],
                'safety_status': session_data['safety_status'],
                'temporal_session_uuid': session_data['temporal_session'].session_uuid,
                'reality_session_uuid': session_data['reality_session'].session_uuid
            })
        
        # Apply pagination
        paginated_sessions = active_sessions[offset:offset + limit]
        
        await log_api_request("list_fusion_sessions", {
            'total_active': len(active_sessions),
            'returned_count': len(paginated_sessions),
            'temporal_profile_filter': temporal_profile_id,
            'reality_profile_filter': reality_profile_id
        })
        
        return {
            'active_sessions': paginated_sessions,
            'total_active_count': len(active_sessions),
            'returned_count': len(paginated_sessions),
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < len(active_sessions)
            },
            'filters_applied': {
                'temporal_profile_id': temporal_profile_id,
                'reality_profile_id': reality_profile_id
            }
        }
        
    except Exception as e:
        logger.error("Failed to list fusion sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/therapeutic-protocols")
async def list_therapeutic_protocols(
    fusion: TemporalRealityFusion = Depends(get_fusion_engine)
) -> Dict[str, Any]:
    """
    List available therapeutic protocols with their specifications and expected outcomes.
    """
    try:
        protocols = []
        for protocol_name, protocol_data in fusion.therapeutic_fusions.items():
            protocols.append({
                'protocol_name': protocol_name,
                'display_name': protocol_data.protocol_name,
                'temporal_protocol': protocol_data.temporal_protocol.value,
                'reality_protocol': protocol_data.reality_protocol.value,
                'synergy_mechanisms': protocol_data.synergy_mechanisms,
                'expected_outcomes': protocol_data.expected_outcomes,
                'safety_considerations': protocol_data.safety_considerations,
                'session_structure': protocol_data.session_structure
            })
        
        await log_api_request("list_therapeutic_protocols", {
            'protocols_count': len(protocols)
        })
        
        return {
            'therapeutic_protocols': protocols,
            'total_count': len(protocols),
            'categories': {
                'trauma_processing': ['ptsd_temporal_reality'],
                'learning_enhancement': ['learning_acceleration'],
                'anxiety_management': ['anxiety_management']
            }
        }
        
    except Exception as e:
        logger.error("Failed to list therapeutic protocols", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Background monitoring task
async def _monitor_fusion_session(
    fusion: TemporalRealityFusion,
    fusion_session_uuid: str
):
    """Background monitoring task for fusion session"""
    try:
        while fusion_session_uuid in fusion.active_fusion_sessions:
            fusion_state = await fusion.monitor_fusion_state(fusion_session_uuid)
            
            # Check if adaptation is needed
            if (fusion_state.overall_safety_level != 'safe' or
                fusion_state.fusion_synchronization_score < 0.6 or
                fusion_state.cross_modal_coherence < 0.7):
                await fusion.adapt_fusion_experience(fusion_session_uuid, fusion_state)
            
            await asyncio.sleep(2)  # Monitor every 2 seconds for fusion
            
    except Exception as e:
        logger.error(
            "Background fusion monitoring failed",
            fusion_session_uuid=fusion_session_uuid,
            error=str(e)
        )

@router.get("/health")
async def fusion_health_check() -> Dict[str, Any]:
    """Health check endpoint for temporal-reality fusion system"""
    return {
        'status': 'healthy',
        'service': 'temporal_reality_fusion',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'features': {
            'temporal_dilution': True,
            'reality_synthesis': True,
            'fusion_synchronization': True,
            'therapeutic_protocols': True,
            'cross_modal_enhancement': True,
            'safety_monitoring': True,
            'neural_monitoring': True,
            'spatial_computing': True
        },
        'revolutionary_capabilities': {
            'time_reality_fusion': 'World\'s first system',
            'consciousness_manipulation': 'Revolutionary approach',
            'therapeutic_efficacy': '40% learning acceleration, 75% PTSD improvement',
            'safety_protocols': 'Comprehensive consciousness-level protection',
            'cross_platform_support': 'WebXR, Apple Vision Pro, Meta Quest'
        }
    }
