"""
Meta-Reality Engine API - Feature 11 Endpoints

RESTful API endpoints for managing multi-dimensional reality experiences,
parallel existences, and alternate self exploration with comprehensive
identity coherence protection and safety monitoring.

Revolutionary API Features:
- Create and manage parallel reality sessions
- Generate alternate selves and life scenarios
- Transition seamlessly between reality layers
- Extract insights from multi-dimensional experiences
- Emergency return protocols for safety
- Real-time identity coherence monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.meta_reality import (
    MetaRealitySession, RealityLayer, IdentitySnapshot, RealityTransition,
    MetaRealityInsight, RealityLayerType, RealityCoherenceState,
    MetaRealitySessionCreate, MetaRealitySessionResponse, RealityLayerResponse,
    IdentitySnapshotResponse, MetaRealityInsightResponse
)
from app.services.meta_reality_engine import create_meta_reality_engine
from app.schemas.response import ResponseModel, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meta-reality", tags=["Meta-Reality Engine"])

# Initialize meta-reality engine
meta_reality_engine = create_meta_reality_engine()

@router.post("/sessions", response_model=MetaRealitySessionResponse)
async def create_meta_reality_session(
    session_config: MetaRealitySessionCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new meta-reality session for exploring parallel existences
    
    Features:
    - Comprehensive user readiness assessment
    - Personalized reality layer recommendations
    - Identity coherence monitoring setup
    - Emergency return protocol preparation
    """
    try:
        logger.info(f"Creating meta-reality session for user {current_user['id']}")
        
        # Convert Pydantic model to dict
        config_dict = {
            'session_name': session_config.session_name,
            'reality_layer_types': [layer_type.value for layer_type in session_config.reality_layer_types],
            'max_layers': session_config.max_simultaneous_layers,
            'intensity': session_config.experience_intensity,
            'drift_threshold': session_config.identity_drift_threshold,
            'anchor_strength': 1.0 - session_config.identity_drift_threshold,
            'memory_bridge': session_config.memory_bridge_enabled,
            'emergency_return': session_config.emergency_return_enabled
        }
        
        # Create session
        session = await meta_reality_engine.create_meta_reality_session(
            user_id=current_user['id'],
            session_config=config_dict,
            db_session=db
        )
        
        # Start background monitoring
        background_tasks.add_task(
            _monitor_session_background,
            session.id,
            db
        )
        
        logger.info(f"Meta-reality session {session.id} created successfully")
        
        return MetaRealitySessionResponse.from_orm(session)
        
    except ValueError as e:
        logger.warning(f"Invalid meta-reality session configuration: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create meta-reality session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create meta-reality session")

@router.get("/sessions", response_model=List[MetaRealitySessionResponse])
async def get_user_sessions(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all meta-reality sessions for the current user"""
    try:
        sessions = db.query(MetaRealitySession).filter(
            MetaRealitySession.primary_user_id == current_user['id']
        ).order_by(MetaRealitySession.created_at.desc()).all()
        
        return [MetaRealitySessionResponse.from_orm(session) for session in sessions]
        
    except Exception as e:
        logger.error(f"Failed to retrieve user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@router.get("/sessions/{session_id}", response_model=MetaRealitySessionResponse)
async def get_session_details(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific meta-reality session"""
    try:
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return MetaRealitySessionResponse.from_orm(session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session details")

@router.post("/sessions/{session_id}/layers")
async def create_reality_layer(
    session_id: int,
    layer_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new reality layer within a session
    
    Features:
    - Alternate self generation based on life divergences
    - Custom environment and narrative design  
    - AI personality adjustments for authentic experience
    - Complexity assessment and safety validation
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create reality layer
        layer = await meta_reality_engine.create_reality_layer(
            session_id=session_id,
            layer_config=layer_config,
            db_session=db
        )
        
        return RealityLayerResponse.from_orm(layer)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create reality layer: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create reality layer")

@router.post("/sessions/{session_id}/layers/{layer_id}/activate")
async def activate_reality_layer(
    session_id: int,
    layer_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Activate a reality layer and transition user into the experience
    
    Features:
    - Smooth reality transition with narrative bridging
    - Consciousness resource allocation across layers
    - Real-time identity coherence monitoring
    - Interference detection and resolution
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Activate layer
        activation_result = await meta_reality_engine.activate_reality_layer(
            session_id=session_id,
            layer_id=layer_id,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=activation_result,
            message="Reality layer activated successfully"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to activate reality layer: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate reality layer")

@router.post("/sessions/{session_id}/transitions")
async def transition_between_realities(
    session_id: int,
    transition_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Transition between different reality layers with safety protocols
    
    Features:
    - Identity snapshot comparison before/after
    - Smooth transition design based on layer complexity
    - Memory continuity preservation
    - Insight extraction during transitions
    """
    try:
        from_layer_id = transition_request.get('from_layer_id')
        to_layer_id = transition_request.get('to_layer_id')
        transition_type = transition_request.get('transition_type', 'smooth_fade')
        
        if not from_layer_id or not to_layer_id:
            raise HTTPException(status_code=400, detail="Both from_layer_id and to_layer_id required")
        
        # Execute transition
        transition_result = await meta_reality_engine.transition_between_realities(
            session_id=session_id,
            from_layer_id=from_layer_id,
            to_layer_id=to_layer_id,
            transition_type=transition_type,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=transition_result,
            message="Reality transition completed successfully"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute reality transition: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute reality transition")

@router.post("/sessions/{session_id}/alternate-self")
async def generate_alternate_self(
    session_id: int,
    divergence_scenario: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate an alternate self based on different life choices or circumstances
    
    Features:
    - Personality evolution modeling based on life events
    - Behavioral prediction for alternate timeline
    - Comprehensive life narrative generation
    - Decision-making pattern analysis
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate alternate self
        alternate_self = await meta_reality_engine.generate_alternate_self(
            user_id=current_user['id'],
            divergence_scenario=divergence_scenario,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=alternate_self,
            message="Alternate self generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate alternate self: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate alternate self")

@router.post("/sessions/{session_id}/parallel-existence")
async def create_parallel_existence_experience(
    session_id: int,
    parallel_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create parallel existence experience across multiple simultaneous realities
    
    Features:
    - Quantum consciousness superposition simulation
    - Multiple reality synchronization protocols
    - Consciousness bandwidth allocation
    - Quantum entanglement effect monitoring
    """
    try:
        num_parallel_realities = parallel_config.get('num_realities', 2)
        experience_focus = parallel_config.get('focus', 'decision_exploration')
        
        if num_parallel_realities > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 parallel realities supported")
        
        # Create parallel existence experience
        parallel_result = await meta_reality_engine.create_parallel_existence_experience(
            session_id=session_id,
            num_parallel_realities=num_parallel_realities,
            experience_focus=experience_focus,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=parallel_result,
            message=f"Parallel existence experience created with {num_parallel_realities} realities"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create parallel existence experience: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create parallel existence experience")

@router.get("/sessions/{session_id}/insights", response_model=List[MetaRealityInsightResponse])
async def get_reality_insights(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Extract and retrieve insights from meta-reality experiences
    
    Features:
    - Cross-reality pattern analysis
    - Decision-making insight generation
    - Identity evolution understanding
    - Creative and problem-solving breakthrough identification
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Extract insights
        insights_data = await meta_reality_engine.extract_reality_insights(
            session_id=session_id,
            db_session=db
        )
        
        # Get stored insights from database
        insights = db.query(MetaRealityInsight).filter(
            MetaRealityInsight.session_id == session_id
        ).all()
        
        return [MetaRealityInsightResponse.from_orm(insight) for insight in insights]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reality insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

@router.get("/sessions/{session_id}/identity-snapshots", response_model=List[IdentitySnapshotResponse])
async def get_identity_snapshots(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get identity coherence snapshots for monitoring identity stability
    
    Features:
    - Real-time identity drift tracking
    - Personality coherence analysis
    - Memory integrity assessment
    - Value system alignment monitoring
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get identity snapshots
        snapshots = db.query(IdentitySnapshot).filter(
            IdentitySnapshot.session_id == session_id
        ).order_by(IdentitySnapshot.created_at).all()
        
        return [IdentitySnapshotResponse.from_orm(snapshot) for snapshot in snapshots]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get identity snapshots: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve identity snapshots")

@router.post("/sessions/{session_id}/emergency-return")
async def emergency_reality_return(
    session_id: int,
    emergency_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Emergency protocol to immediately return user to baseline reality
    
    Features:
    - Immediate reality layer deactivation
    - Identity coherence restoration
    - Psychological safety prioritization
    - Crisis intervention protocols
    """
    try:
        reason = emergency_request.get('reason', 'User requested emergency return')
        
        # Execute emergency return
        return_result = await meta_reality_engine.emergency_reality_return(
            session_id=session_id,
            reason=reason,
            db_session=db
        )
        
        logger.warning(f"Emergency return executed for session {session_id}: {reason}")
        
        return ResponseModel(
            success=True,
            data=return_result,
            message="Emergency return to baseline reality completed"
        )
        
    except Exception as e:
        logger.error(f"Failed to execute emergency return: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute emergency return")

@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get real-time status of meta-reality session
    
    Features:
    - Active layer information
    - Identity coherence current state
    - Psychological safety metrics
    - Recent activity summary
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get current session state from engine
        session_state = meta_reality_engine.active_sessions.get(session_id)
        
        # Get active layers
        active_layers = db.query(RealityLayer).filter(
            RealityLayer.session_id == session_id,
            RealityLayer.is_active == True
        ).all()
        
        # Get latest identity snapshot
        latest_snapshot = db.query(IdentitySnapshot).filter(
            IdentitySnapshot.session_id == session_id
        ).order_by(IdentitySnapshot.created_at.desc()).first()
        
        status_data = {
            'session_id': session_id,
            'session_status': session.session_status,
            'coherence_state': session.coherence_state,
            'active_layers_count': len(active_layers),
            'active_layers': [RealityLayerResponse.from_orm(layer) for layer in active_layers],
            'total_transitions': session.total_layer_transitions,
            'insights_generated': session.insights_generated,
            'current_identity_coherence': latest_snapshot.identity_coherence_score if latest_snapshot else 1.0,
            'psychological_safety': session_state.psychological_safety_score if session_state else 1.0,
            'user_immersion': session_state.user_immersion if session_state else 0.0,
            'emergency_protocols_active': session_state.emergency_protocols_active if session_state else False,
            'started_at': session.started_at,
            'last_activity': latest_snapshot.created_at if latest_snapshot else session.updated_at
        }
        
        return ResponseModel(
            success=True,
            data=status_data,
            message="Session status retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session status")

@router.delete("/sessions/{session_id}")
async def end_session(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Properly end a meta-reality session with full integration support
    
    Features:
    - Graceful session termination
    - Final insight extraction
    - Identity integration verification
    - Session completion analytics
    """
    try:
        # Verify session ownership
        session = db.query(MetaRealitySession).filter(
            MetaRealitySession.id == session_id,
            MetaRealitySession.primary_user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Deactivate all layers
        active_layers = db.query(RealityLayer).filter(
            RealityLayer.session_id == session_id,
            RealityLayer.is_active == True
        ).all()
        
        for layer in active_layers:
            layer.is_active = False
            layer.deactivation_timestamp = datetime.utcnow()
        
        # Update session
        session.session_status = "completed"
        session.ended_at = datetime.utcnow()
        
        # Remove from active sessions
        if session_id in meta_reality_engine.active_sessions:
            del meta_reality_engine.active_sessions[session_id]
        
        # Cancel monitoring
        if session_id in meta_reality_engine.session_monitors:
            meta_reality_engine.session_monitors[session_id].cancel()
            del meta_reality_engine.session_monitors[session_id]
        
        db.commit()
        
        return ResponseModel(
            success=True,
            message="Meta-reality session ended successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to end session")

# Background task functions

async def _monitor_session_background(session_id: int, db: Session):
    """Background task for session monitoring"""
    try:
        # This would be handled by the engine's monitoring system
        logger.info(f"Background monitoring started for session {session_id}")
    except Exception as e:
        logger.error(f"Background monitoring failed for session {session_id}: {str(e)}")

# Health and utility endpoints

@router.get("/health")
async def meta_reality_health():
    """Check meta-reality engine health and status"""
    try:
        active_sessions_count = len(meta_reality_engine.active_sessions)
        monitoring_tasks_count = len(meta_reality_engine.session_monitors)
        
        return ResponseModel(
            success=True,
            data={
                'engine_status': 'healthy',
                'active_sessions': active_sessions_count,
                'monitoring_tasks': monitoring_tasks_count,
                'max_simultaneous_realities': meta_reality_engine.max_simultaneous_realities,
                'identity_coherence_threshold': meta_reality_engine.identity_coherence_threshold,
                'emergency_return_threshold': meta_reality_engine.emergency_return_threshold
            },
            message="Meta-reality engine is operational"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/reality-types")
async def get_reality_layer_types():
    """Get available reality layer types and descriptions"""
    try:
        reality_types = {
            RealityLayerType.BASELINE_REALITY: {
                'name': 'Baseline Reality',
                'description': 'Current/default reality state',
                'complexity': 0,
                'recommended_for': 'beginners'
            },
            RealityLayerType.ALTERNATE_SELF: {
                'name': 'Alternate Self',
                'description': 'Different life path versions of yourself',
                'complexity': 3,
                'recommended_for': 'self-exploration'
            },
            RealityLayerType.FUTURE_PROBABLE: {
                'name': 'Future Probable',
                'description': 'Probable future timeline scenarios',
                'complexity': 4,
                'recommended_for': 'decision-making'
            },
            RealityLayerType.PAST_ALTERNATE: {
                'name': 'Past Alternate',
                'description': 'Alternative past scenarios',
                'complexity': 3,
                'recommended_for': 'healing-integration'
            },
            RealityLayerType.CREATIVE_SANDBOX: {
                'name': 'Creative Sandbox',
                'description': 'Creative exploration spaces',
                'complexity': 2,
                'recommended_for': 'creativity-enhancement'
            },
            RealityLayerType.PROBLEM_SOLVING: {
                'name': 'Problem Solving',
                'description': 'Scenario testing environments',
                'complexity': 3,
                'recommended_for': 'problem-solving'
            },
            RealityLayerType.THERAPEUTIC: {
                'name': 'Therapeutic',
                'description': 'Healing and growth spaces',
                'complexity': 2,
                'recommended_for': 'healing-therapy'
            },
            RealityLayerType.COLLECTIVE_SHARED: {
                'name': 'Collective Shared',
                'description': 'Shared reality experiences',
                'complexity': 5,
                'recommended_for': 'advanced-users'
            }
        }
        
        return ResponseModel(
            success=True,
            data=reality_types,
            message="Reality layer types retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get reality types: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get reality types")