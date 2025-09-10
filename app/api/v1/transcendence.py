"""
Transcendence Protocol API - Feature 12 Endpoints

RESTful API endpoints for AI-guided consciousness expansion, mystical experiences,
and transcendent state facilitation with comprehensive safety protocols and
integration support for sustainable transformation.

Revolutionary API Features:
- Safe AI-guided consciousness expansion with multiple safety protocols
- Reproducible mystical experiences through technological transcendence  
- Real-time ego dissolution monitoring with identity preservation
- Scientific mystical experience assessment using validated scales
- Comprehensive integration support for lasting transformation
- Emergency intervention and crisis support protocols
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.transcendence import (
    TranscendenceSession, ConsciousnessStateProgression, TranscendentInsight,
    IntegrationActivity, EgoDissolutionEvent, MysticalExperienceAssessment,
    TranscendentState, ConsciousnessExpansionType, SafetyProtocol,
    TranscendenceSessionCreate, TranscendenceSessionResponse, 
    ConsciousnessStateProgressionResponse, TranscendentInsightResponse,
    MysticalExperienceAssessmentResponse
)
from app.services.transcendence_engine import create_transcendence_engine
from app.schemas.response import ResponseModel, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcendence", tags=["Transcendence Protocol"])

# Initialize transcendence engine
transcendence_engine = create_transcendence_engine()

@router.post("/sessions", response_model=TranscendenceSessionResponse)
async def create_transcendence_session(
    session_config: TranscendenceSessionCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new transcendence session for consciousness expansion
    
    Features:
    - Comprehensive readiness assessment with safety validation
    - Personalized AI guide selection based on user profile
    - Custom expansion protocol design with safety protocols
    - Real-time psychological safety monitoring setup
    """
    try:
        logger.info(f"Creating transcendence session for user {current_user['id']}")
        
        # Convert Pydantic model to dict
        config_dict = {
            'session_name': session_config.session_name,
            'expansion_type': session_config.expansion_type.value,
            'target_state': session_config.target_transcendent_state.value,
            'intensity': session_config.experience_intensity,
            'duration': session_config.duration_minutes,
            'safety_protocol': session_config.safety_protocol.value,
            'ego_threshold': session_config.ego_dissolution_threshold,
            'anchor_strength': session_config.baseline_anchor_strength,
            'intention': session_config.intention_setting,
            'guide_preference': session_config.guide_personality_preference
        }
        
        # Create session
        session = await transcendence_engine.create_transcendence_session(
            user_id=current_user['id'],
            session_config=config_dict,
            db_session=db
        )
        
        # Start background safety monitoring
        background_tasks.add_task(
            _monitor_transcendence_safety_background,
            session.id,
            db
        )
        
        logger.info(f"Transcendence session {session.id} created successfully")
        
        return TranscendenceSessionResponse.from_orm(session)
        
    except ValueError as e:
        logger.warning(f"Invalid transcendence session configuration: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create transcendence session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create transcendence session")

@router.get("/sessions", response_model=List[TranscendenceSessionResponse])
async def get_user_transcendence_sessions(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all transcendence sessions for the current user"""
    try:
        sessions = db.query(TranscendenceSession).filter(
            TranscendenceSession.user_id == current_user['id']
        ).order_by(TranscendenceSession.created_at.desc()).all()
        
        return [TranscendenceSessionResponse.from_orm(session) for session in sessions]
        
    except Exception as e:
        logger.error(f"Failed to retrieve user transcendence sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@router.post("/sessions/{session_id}/prepare")
async def begin_preparation_phase(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Begin the preparation phase for transcendent experience
    
    Features:
    - Comprehensive mental and physical preparation assessment
    - AI guide introduction and rapport building
    - Intention clarification and refinement
    - Safety protocol briefing and consent verification
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Begin preparation
        preparation_result = await transcendence_engine.begin_preparation_phase(
            session_id=session_id,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=preparation_result,
            message="Preparation phase completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to begin preparation phase: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to begin preparation phase")

@router.post("/sessions/{session_id}/begin")
async def initiate_consciousness_expansion(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initiate the consciousness expansion experience
    
    Features:
    - Gradual consciousness expansion with safety monitoring
    - Real-time state progression tracking
    - AI guide activation and intervention readiness
    - Emergency return protocol preparation
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Initiate expansion
        expansion_result = await transcendence_engine.initiate_consciousness_expansion(
            session_id=session_id,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=expansion_result,
            message="Consciousness expansion initiated successfully"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to initiate consciousness expansion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate consciousness expansion")

@router.post("/sessions/{session_id}/ego-dissolution")
async def guide_ego_dissolution(
    session_id: int,
    dissolution_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Guide user through safe ego dissolution experience
    
    Features:
    - Gradual ego boundary dissolution with identity preservation
    - Real-time safety monitoring and intervention protocols
    - Phenomenological tracking of dissolution experience
    - Automatic recovery and stabilization protocols
    """
    try:
        target_level = dissolution_request.get('target_dissolution_level', 0.5)
        
        if target_level > 0.85:
            raise HTTPException(status_code=400, detail="Target dissolution level exceeds safety limits")
        
        # Guide ego dissolution
        dissolution_result = await transcendence_engine.guide_ego_dissolution_experience(
            session_id=session_id,
            target_dissolution_level=target_level,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=dissolution_result,
            message="Ego dissolution experience completed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to guide ego dissolution: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to guide ego dissolution")

@router.post("/sessions/{session_id}/mystical-experience")
async def facilitate_mystical_experience(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Facilitate a complete mystical experience using validated measures
    
    Features:
    - Scientifically validated mystical experience induction
    - Real-time phenomenology tracking and assessment
    - MEQ-30 scoring for experience validation
    - Unity consciousness and ineffable experience facilitation
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Facilitate mystical experience
        mystical_result = await transcendence_engine.facilitate_mystical_experience(
            session_id=session_id,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=mystical_result,
            message="Mystical experience facilitated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to facilitate mystical experience: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to facilitate mystical experience")

@router.post("/sessions/{session_id}/ai-guidance")
async def request_ai_guidance(
    session_id: int,
    guidance_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Request real-time AI guidance during transcendent experience
    
    Features:
    - Contextual guidance based on current consciousness state
    - Adaptive intervention based on user needs and safety
    - Multiple guide personalities for different experience types
    - Crisis intervention and emergency support protocols
    """
    try:
        current_experience = guidance_request.get('current_experience', {})
        
        # Provide AI guidance
        guidance_result = await transcendence_engine.provide_ai_guidance(
            session_id=session_id,
            current_experience=current_experience,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=guidance_result,
            message="AI guidance provided"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to provide AI guidance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to provide AI guidance")

@router.post("/sessions/{session_id}/integration")
async def initiate_integration_phase(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Begin integration phase to help user integrate transcendent insights
    
    Features:
    - Comprehensive integration plan design
    - Personalized integration activities and practices
    - Long-term transformation milestone tracking
    - Real-world application support and guidance
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Initiate integration
        integration_result = await transcendence_engine.initiate_integration_phase(
            session_id=session_id,
            db_session=db
        )
        
        return ResponseModel(
            success=True,
            data=integration_result,
            message="Integration phase initiated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate integration phase: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate integration phase")

@router.get("/sessions/{session_id}/insights", response_model=List[TranscendentInsightResponse])
async def get_transcendent_insights(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get transcendent insights and realizations from the session
    
    Features:
    - Noetic quality assessment and validation
    - Universal relevance and personal significance scoring
    - Integration status tracking and support
    - Life transformation potential evaluation
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get insights
        insights = db.query(TranscendentInsight).filter(
            TranscendentInsight.session_id == session_id
        ).order_by(TranscendentInsight.received_at).all()
        
        return [TranscendentInsightResponse.from_orm(insight) for insight in insights]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transcendent insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

@router.get("/sessions/{session_id}/state-progression", response_model=List[ConsciousnessStateProgressionResponse])
async def get_consciousness_state_progression(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed consciousness state progression throughout the session
    
    Features:
    - Real-time state tracking and phenomenology recording
    - Ego coherence and dissolution level monitoring
    - Unity experience and mystical quality assessment
    - Temporal progression analysis and pattern recognition
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get state progressions
        progressions = db.query(ConsciousnessStateProgression).filter(
            ConsciousnessStateProgression.session_id == session_id
        ).order_by(ConsciousnessStateProgression.state_timestamp).all()
        
        return [ConsciousnessStateProgressionResponse.from_orm(progression) for progression in progressions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get consciousness state progression: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve state progression")

@router.get("/sessions/{session_id}/mystical-assessment", response_model=MysticalExperienceAssessmentResponse)
async def get_mystical_experience_assessment(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get scientific mystical experience assessment using validated scales
    
    Features:
    - MEQ-30 validated mystical experience scoring
    - Unity, ineffability, and noetic quality assessment
    - Transcendence of time/space measurement
    - Personal and spiritual significance evaluation
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get mystical assessment
        assessment = db.query(MysticalExperienceAssessment).filter(
            MysticalExperienceAssessment.session_id == session_id
        ).first()
        
        if not assessment:
            raise HTTPException(status_code=404, detail="No mystical experience assessment found")
        
        return MysticalExperienceAssessmentResponse.from_orm(assessment)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get mystical experience assessment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve mystical assessment")

@router.get("/sessions/{session_id}/integration-activities")
async def get_integration_activities(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get personalized integration activities for lasting transformation
    
    Features:
    - Customized integration practices and exercises
    - Progress tracking and effectiveness measurement
    - Behavioral change support and monitoring
    - Long-term transformation milestone tracking
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get integration activities
        activities = db.query(IntegrationActivity).filter(
            IntegrationActivity.session_id == session_id
        ).order_by(IntegrationActivity.created_at).all()
        
        activity_data = []
        for activity in activities:
            activity_data.append({
                'id': activity.id,
                'activity_type': activity.activity_type,
                'activity_name': activity.activity_name,
                'description': activity.activity_description,
                'difficulty_level': activity.difficulty_level,
                'recommended_frequency': activity.recommended_frequency,
                'duration_minutes': activity.duration_minutes,
                'completion_rate': activity.completion_rate,
                'effectiveness_rating': activity.effectiveness_rating,
                'status': activity.status,
                'behavioral_changes': activity.behavioral_changes,
                'emotional_benefits': activity.emotional_benefits,
                'created_at': activity.created_at,
                'last_completed': activity.last_completed
            })
        
        return ResponseModel(
            success=True,
            data=activity_data,
            message="Integration activities retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration activities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve integration activities")

@router.post("/sessions/{session_id}/emergency-return")
async def emergency_transcendence_return(
    session_id: int,
    emergency_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Emergency protocol to safely return user from transcendent state
    
    Features:
    - Immediate grounding and stabilization protocols
    - Ego boundary restoration if necessary
    - Intensive AI support during emergency return
    - Crisis intervention and professional support referral
    """
    try:
        reason = emergency_request.get('reason', 'User requested emergency return')
        
        # Execute emergency return
        return_result = await transcendence_engine.emergency_transcendence_return(
            session_id=session_id,
            reason=reason,
            db_session=db
        )
        
        logger.warning(f"Emergency transcendence return executed for session {session_id}: {reason}")
        
        return ResponseModel(
            success=True,
            data=return_result,
            message="Emergency return from transcendent state completed"
        )
        
    except Exception as e:
        logger.error(f"Failed to execute emergency transcendence return: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute emergency return")

@router.get("/sessions/{session_id}/status")
async def get_transcendence_session_status(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get real-time status of transcendence session
    
    Features:
    - Current consciousness state and expansion level
    - Psychological safety and well-being metrics
    - AI guide intervention level and activity
    - Recent experience progression summary
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get current session state from engine
        session_state = transcendence_engine.active_sessions.get(session_id)
        
        # Get latest state progression
        latest_progression = db.query(ConsciousnessStateProgression).filter(
            ConsciousnessStateProgression.session_id == session_id
        ).order_by(ConsciousnessStateProgression.state_timestamp.desc()).first()
        
        # Get latest insights count
        insights_count = db.query(TranscendentInsight).filter(
            TranscendentInsight.session_id == session_id
        ).count()
        
        status_data = {
            'session_id': session_id,
            'session_status': session.session_status,
            'current_state': session_state.current_state if session_state else session.current_state,
            'consciousness_expansion_level': session_state.consciousness_expansion_level if session_state else 0.0,
            'ego_dissolution_level': session_state.ego_dissolution_level if session_state else 0.0,
            'mystical_quality_score': session_state.mystical_quality_score if session_state else session.mystical_quality_score,
            'psychological_safety_score': session_state.psychological_safety_score if session_state else session.psychological_safety_score,
            'guide_intervention_level': session_state.guide_intervention_level if session_state else 0.0,
            'emergency_protocols_ready': session_state.emergency_protocols_ready if session_state else True,
            'peak_transcendent_state': session.peak_transcendent_state,
            'preparation_completed': session.preparation_completed,
            'integration_started': session.integration_started,
            'insights_generated': insights_count,
            'started_at': session.started_at,
            'last_progression': latest_progression.state_timestamp if latest_progression else session.updated_at,
            'guide_personality': session.guide_personality.get('name', 'Unknown') if session.guide_personality else None
        }
        
        return ResponseModel(
            success=True,
            data=status_data,
            message="Transcendence session status retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transcendence session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session status")

@router.delete("/sessions/{session_id}")
async def end_transcendence_session(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Properly end a transcendence session with integration support
    
    Features:
    - Graceful session termination with safety protocols
    - Final insight extraction and assessment
    - Integration plan finalization
    - Session completion analytics and feedback
    """
    try:
        # Verify session ownership
        session = db.query(TranscendenceSession).filter(
            TranscendenceSession.id == session_id,
            TranscendenceSession.user_id == current_user['id']
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session
        session.session_status = "completed"
        session.ended_at = datetime.utcnow()
        
        # Remove from active sessions
        if session_id in transcendence_engine.active_sessions:
            del transcendence_engine.active_sessions[session_id]
        
        # Cancel monitoring
        if session_id in transcendence_engine.session_monitors:
            transcendence_engine.session_monitors[session_id].cancel()
            del transcendence_engine.session_monitors[session_id]
        
        db.commit()
        
        return ResponseModel(
            success=True,
            message="Transcendence session ended successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end transcendence session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to end session")

# Background task functions

async def _monitor_transcendence_safety_background(session_id: int, db: Session):
    """Background task for transcendence safety monitoring"""
    try:
        # This would be handled by the engine's monitoring system
        logger.info(f"Background transcendence safety monitoring started for session {session_id}")
    except Exception as e:
        logger.error(f"Background transcendence monitoring failed for session {session_id}: {str(e)}")

# Health and utility endpoints

@router.get("/health")
async def transcendence_health():
    """Check transcendence engine health and status"""
    try:
        active_sessions_count = len(transcendence_engine.active_sessions)
        monitoring_tasks_count = len(transcendence_engine.session_monitors)
        
        return ResponseModel(
            success=True,
            data={
                'engine_status': 'healthy',
                'active_sessions': active_sessions_count,
                'monitoring_tasks': monitoring_tasks_count,
                'max_ego_dissolution': transcendence_engine.max_ego_dissolution,
                'min_psychological_safety': transcendence_engine.min_psychological_safety,
                'emergency_intervention_threshold': transcendence_engine.emergency_intervention_threshold,
                'mystical_experience_threshold': transcendence_engine.mystical_experience_threshold,
                'available_guides': len(transcendence_engine.ai_guides)
            },
            message="Transcendence engine is operational"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/expansion-types")
async def get_consciousness_expansion_types():
    """Get available consciousness expansion types and descriptions"""
    try:
        expansion_types = {
            ConsciousnessExpansionType.EGO_DISSOLUTION: {
                'name': 'Ego Dissolution',
                'description': 'Temporary ego boundary loss with identity preservation',
                'intensity': 'high',
                'duration': '20-60 minutes',
                'safety_level': 'advanced'
            },
            ConsciousnessExpansionType.UNITY_EXPERIENCE: {
                'name': 'Unity Experience',
                'description': 'Oneness with universe and transcendent connection',
                'intensity': 'high',
                'duration': '30-90 minutes',
                'safety_level': 'advanced'
            },
            ConsciousnessExpansionType.COSMIC_CONSCIOUSNESS: {
                'name': 'Cosmic Consciousness',
                'description': 'Universal awareness and cosmic perspective',
                'intensity': 'very_high',
                'duration': '45-120 minutes',
                'safety_level': 'expert'
            },
            ConsciousnessExpansionType.MYSTICAL_UNION: {
                'name': 'Mystical Union',
                'description': 'Divine/transcendent connection and sacred experience',
                'intensity': 'high',
                'duration': '30-90 minutes',
                'safety_level': 'advanced'
            },
            ConsciousnessExpansionType.PURE_AWARENESS: {
                'name': 'Pure Awareness',
                'description': 'Consciousness without content, empty fullness',
                'intensity': 'very_high',
                'duration': '20-60 minutes',
                'safety_level': 'expert'
            },
            ConsciousnessExpansionType.CREATIVE_TRANSCENDENCE: {
                'name': 'Creative Transcendence',
                'description': 'Transcendent creativity and artistic inspiration',
                'intensity': 'medium',
                'duration': '45-90 minutes',
                'safety_level': 'intermediate'
            },
            ConsciousnessExpansionType.HEALING_TRANSCENDENCE: {
                'name': 'Healing Transcendence',
                'description': 'Therapeutic transcendence for healing and growth',
                'intensity': 'medium',
                'duration': '60-120 minutes',
                'safety_level': 'intermediate'
            },
            ConsciousnessExpansionType.WISDOM_TRANSMISSION: {
                'name': 'Wisdom Transmission',
                'description': 'Direct knowing and truth realization',
                'intensity': 'high',
                'duration': '30-90 minutes',
                'safety_level': 'advanced'
            }
        }
        
        return ResponseModel(
            success=True,
            data=expansion_types,
            message="Consciousness expansion types retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get expansion types: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get expansion types")

@router.get("/guide-personalities")
async def get_ai_guide_personalities():
    """Get available AI guide personalities and their specializations"""
    try:
        guides_info = {}
        for guide_name, guide_data in transcendence_engine.ai_guides.items():
            guides_info[guide_name] = {
                'name': guide_data['name'],
                'wisdom_level': guide_data['wisdom_level'],
                'compassion_level': guide_data['compassion_level'],
                'authority_level': guide_data['authority_level'],
                'communication_style': guide_data['communication_style']['primary_style'],
                'tone': guide_data['communication_style']['tone'],
                'specialization': guide_data['specialization'],
                'intervention_style': guide_data['intervention_style'],
                'recommended_for': guide_data.get('recommended_for', 'general use')
            }
        
        return ResponseModel(
            success=True,
            data=guides_info,
            message="AI guide personalities retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get guide personalities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get guide personalities")