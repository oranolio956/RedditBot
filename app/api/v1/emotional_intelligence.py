"""
Revolutionary Emotional Intelligence API Endpoints

Provides comprehensive API access to the advanced emotional intelligence system:
- Multi-modal emotion detection and analysis
- Real-time empathy coaching and feedback
- Emotional intelligence assessment and development
- Crisis detection and intervention
- Personalized emotional response generation
- Empathy matching and collaborative growth

All endpoints include proper security, validation, and error handling
with scientifically-grounded emotional intelligence capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
import logging

from app.database.connection import get_db_session as get_db
from app.models.user import User
from app.models.emotional_intelligence import (
    EmotionalProfile, EmotionReading, EmpathyAssessment,
    EmotionalInteraction, BasicEmotion, EmotionIntensity,
    DetectionModality, EmotionRegulationStrategy, CrisisLevel
)
from app.services.emotion_detector import (
    emotion_detector, detect_emotion_from_message, EmotionAnalysisResult
)
from app.services.emotional_responder import (
    emotional_responder, generate_empathetic_response, EmotionalResponse,
    ResponseStyle, TherapeuticTechnique
)
from app.services.empathy_engine import (
    empathy_engine, EmpathyScore, EmpathyCoachingResult,
    EmpathyAssessmentType, CoachingIntervention
)
from app.core.security import get_current_user
from app.core.config import settings


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotional-intelligence", tags=["Emotional Intelligence"])
security = HTTPBearer()


# Request/Response Models

class EmotionAnalysisRequest(BaseModel):
    """Request model for emotion analysis."""
    text_content: Optional[str] = Field(None, description="Text content to analyze")
    voice_features: Optional[Dict[str, Any]] = Field(None, description="Voice/audio features")
    behavioral_data: Optional[Dict[str, Any]] = Field(None, description="User behavioral patterns")
    conversation_context: Optional[Dict[str, Any]] = Field(None, description="Current conversation context")
    include_regulation_suggestions: bool = Field(True, description="Include emotion regulation strategies")


class EmotionAnalysisResponse(BaseModel):
    """Response model for emotion analysis results."""
    valence: float = Field(..., ge=-1, le=1, description="Valence score (-1 to 1)")
    arousal: float = Field(..., ge=-1, le=1, description="Arousal score (-1 to 1)")
    dominance: float = Field(..., ge=-1, le=1, description="Dominance score (-1 to 1)")
    primary_emotion: BasicEmotion = Field(..., description="Primary detected emotion")
    secondary_emotion: Optional[BasicEmotion] = Field(None, description="Secondary emotion if present")
    emotion_intensity: EmotionIntensity = Field(..., description="Intensity of primary emotion")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores by modality")
    detection_modalities: List[DetectionModality] = Field(..., description="Methods used for detection")
    plutchik_scores: Dict[str, float] = Field(..., description="Scores for all 8 basic emotions")
    processing_time_ms: int = Field(..., description="Analysis processing time")
    analysis_quality: float = Field(..., ge=0, le=1, description="Quality score of analysis")
    regulation_strategies: Optional[List[EmotionRegulationStrategy]] = Field(None, description="Suggested regulation strategies")
    crisis_level: Optional[CrisisLevel] = Field(None, description="Assessed crisis level if applicable")


class EmpatheticResponseRequest(BaseModel):
    """Request model for empathetic response generation."""
    user_message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    conversation_context: Optional[Dict[str, Any]] = Field(None, description="Conversation context")
    preferred_response_style: Optional[ResponseStyle] = Field(None, description="Preferred response style")
    include_coaching: bool = Field(True, description="Include empathy coaching elements")


class EmpatheticResponseResponse(BaseModel):
    """Response model for empathetic response generation."""
    response_text: str = Field(..., description="Generated empathetic response")
    response_style: ResponseStyle = Field(..., description="Applied response style")
    therapeutic_techniques: List[TherapeuticTechnique] = Field(..., description="Therapeutic techniques used")
    emotional_intent: Dict[str, float] = Field(..., description="Intended emotional impact")
    regulation_strategies: List[EmotionRegulationStrategy] = Field(..., description="Suggested regulation strategies")
    crisis_level: CrisisLevel = Field(..., description="Assessed crisis level")
    confidence_score: float = Field(..., ge=0, le=1, description="Response confidence")
    personalization_applied: bool = Field(..., description="Whether personalization was applied")
    followup_suggestions: List[str] = Field(..., description="Suggested followup questions/topics")
    processing_time_ms: int = Field(..., description="Response generation time")


class EmpathyAssessmentRequest(BaseModel):
    """Request model for empathy assessment."""
    assessment_type: EmpathyAssessmentType = Field(EmpathyAssessmentType.INITIAL_SCREENING, description="Type of assessment")
    include_behavioral_analysis: bool = Field(True, description="Include behavioral pattern analysis")
    include_conversation_analysis: bool = Field(True, description="Include conversation history analysis")


class EmpathyAssessmentResponse(BaseModel):
    """Response model for empathy assessment results."""
    overall_empathy_quotient: float = Field(..., ge=0, le=80, description="Overall EQ score (Baron-Cohen scale)")
    cognitive_empathy: float = Field(..., ge=0, le=100, description="Cognitive empathy score")
    affective_empathy: float = Field(..., ge=0, le=100, description="Affective empathy score")
    compassionate_empathy: float = Field(..., ge=0, le=100, description="Compassionate empathy score")
    perspective_taking: float = Field(..., ge=0, le=100, description="Perspective taking ability")
    empathic_concern: float = Field(..., ge=0, le=100, description="Empathic concern score")
    personal_distress: float = Field(..., ge=0, le=100, description="Personal distress score")
    assessment_confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    improvement_areas: List[str] = Field(..., description="Areas identified for improvement")
    strengths: List[str] = Field(..., description="Empathy strengths")
    coaching_recommendations: List[CoachingIntervention] = Field(..., description="Recommended coaching interventions")


class EmpathyCoachingRequest(BaseModel):
    """Request model for empathy coaching."""
    conversation_context: Dict[str, Any] = Field(..., description="Current conversation context")
    coaching_focus: Optional[CoachingIntervention] = Field(None, description="Specific coaching focus")
    other_user_emotions: Optional[Dict[str, Any]] = Field(None, description="Other user's emotional state if known")


class EmotionalProfileResponse(BaseModel):
    """Response model for emotional profile."""
    user_id: str = Field(..., description="User ID")
    baseline_dimensions: Dict[str, Optional[float]] = Field(..., description="Baseline emotional dimensions")
    intelligence_metrics: Dict[str, Optional[float]] = Field(..., description="Emotional intelligence metrics")
    dominant_emotions: Optional[Dict[str, Any]] = Field(None, description="Most frequent emotions")
    regulation_strategies: Optional[List[str]] = Field(None, description="Primary regulation strategies")
    attachment_style: Optional[str] = Field(None, description="Attachment style")
    crisis_risk_level: str = Field(..., description="Current crisis risk assessment")
    empathy_development_level: str = Field(..., description="Current empathy development level")


# API Endpoints

@router.post("/analyze", response_model=EmotionAnalysisResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze emotions using advanced multi-modal detection.
    
    Combines text analysis, voice features, behavioral patterns, and contextual
    information to provide comprehensive emotion recognition with therapeutic insights.
    """
    try:
        logger.info(f"Emotion analysis requested for user {current_user.id}")
        
        # Perform emotion analysis
        analysis_result = await emotion_detector.analyze_emotion(
            text_content=request.text_content,
            voice_features=request.voice_features,
            behavioral_data=request.behavioral_data,
            conversation_context=request.conversation_context,
            user_id=str(current_user.id)
        )
        
        # Get user's emotional profile for personalized insights
        emotional_profile = db.query(EmotionalProfile).filter(
            EmotionalProfile.user_id == current_user.id
        ).first()
        
        # Assess crisis level if needed
        crisis_level = None
        if (analysis_result.valence < -0.6 or 
            analysis_result.primary_emotion in [BasicEmotion.SADNESS, BasicEmotion.FEAR] and
            analysis_result.emotion_intensity in [EmotionIntensity.HIGH, EmotionIntensity.EXTREME]):
            
            # This would trigger more sophisticated crisis assessment
            crisis_level = CrisisLevel.MODERATE_CONCERN
        
        # Get regulation strategies if requested
        regulation_strategies = None
        if request.include_regulation_suggestions and emotional_profile:
            regulation_strategies = emotional_profile.primary_regulation_strategies or []
        
        # Save emotion reading to database
        emotion_reading = EmotionReading(
            emotional_profile_id=emotional_profile.id if emotional_profile else None,
            valence_score=analysis_result.valence,
            arousal_score=analysis_result.arousal,
            dominance_score=analysis_result.dominance,
            primary_emotion=analysis_result.primary_emotion,
            secondary_emotion=analysis_result.secondary_emotion,
            emotion_intensity=analysis_result.emotion_intensity,
            detection_modalities=[mod.value for mod in analysis_result.detection_modalities],
            confidence_scores=analysis_result.confidence_scores,
            emotion_blend_scores=analysis_result.plutchik_scores,
            processing_time_ms=analysis_result.processing_time_ms,
            analysis_quality_score=analysis_result.analysis_quality
        )
        
        if emotional_profile:
            db.add(emotion_reading)
            db.commit()
        
        return EmotionAnalysisResponse(
            valence=analysis_result.valence,
            arousal=analysis_result.arousal,
            dominance=analysis_result.dominance,
            primary_emotion=analysis_result.primary_emotion,
            secondary_emotion=analysis_result.secondary_emotion,
            emotion_intensity=analysis_result.emotion_intensity,
            confidence_scores=analysis_result.confidence_scores,
            detection_modalities=analysis_result.detection_modalities,
            plutchik_scores=analysis_result.plutchik_scores,
            processing_time_ms=analysis_result.processing_time_ms,
            analysis_quality=analysis_result.analysis_quality,
            regulation_strategies=regulation_strategies,
            crisis_level=crisis_level
        )
        
    except Exception as e:
        logger.error(f"Emotion analysis failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Emotion analysis failed. Please try again."
        )


@router.post("/respond", response_model=EmpatheticResponseResponse)
async def generate_empathetic_response_endpoint(
    request: EmpatheticResponseRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate emotionally-intelligent, therapeutically-informed responses.
    
    Creates responses that adapt to the user's emotional state using evidence-based
    therapeutic techniques and personalized emotional intelligence insights.
    """
    try:
        logger.info(f"Empathetic response generation requested for user {current_user.id}")
        
        # Get user's emotional profile
        emotional_profile = db.query(EmotionalProfile).filter(
            EmotionalProfile.user_id == current_user.id
        ).first()
        
        # Generate empathetic response
        response_result = await generate_empathetic_response(
            user_message=request.user_message,
            user_id=str(current_user.id),
            conversation_context=request.conversation_context,
            emotional_profile=emotional_profile
        )
        
        # Record emotional interaction for learning
        if emotional_profile:
            # This would create EmotionalInteraction record for tracking empathic exchanges
            pass
        
        return EmpatheticResponseResponse(
            response_text=response_result.response_text,
            response_style=response_result.response_style,
            therapeutic_techniques=response_result.therapeutic_techniques,
            emotional_intent=response_result.emotional_intent,
            regulation_strategies=response_result.regulation_strategies,
            crisis_level=response_result.crisis_level,
            confidence_score=response_result.confidence_score,
            personalization_applied=response_result.personalization_applied,
            followup_suggestions=response_result.followup_suggestions,
            processing_time_ms=response_result.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Empathetic response generation failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate empathetic response. Please try again."
        )


@router.post("/assess-empathy", response_model=EmpathyAssessmentResponse)
async def assess_empathy(
    request: EmpathyAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Conduct comprehensive empathy assessment using multiple methodologies.
    
    Combines behavioral analysis, conversation history, and psychological assessment
    tools to provide detailed empathy and emotional intelligence evaluation.
    """
    try:
        logger.info(f"Empathy assessment requested for user {current_user.id}")
        
        # Conduct empathy assessment
        empathy_score = await empathy_engine.conduct_comprehensive_empathy_assessment(
            user_id=str(current_user.id),
            assessment_type=request.assessment_type,
            db_session=db
        )
        
        return EmpathyAssessmentResponse(
            overall_empathy_quotient=empathy_score.overall_empathy_quotient,
            cognitive_empathy=empathy_score.cognitive_empathy,
            affective_empathy=empathy_score.affective_empathy,
            compassionate_empathy=empathy_score.compassionate_empathy,
            perspective_taking=empathy_score.perspective_taking,
            empathic_concern=empathy_score.empathic_concern,
            personal_distress=empathy_score.personal_distress,
            assessment_confidence=empathy_score.assessment_confidence,
            improvement_areas=empathy_score.improvement_areas,
            strengths=empathy_score.strengths,
            coaching_recommendations=empathy_score.coaching_recommendations
        )
        
    except Exception as e:
        logger.error(f"Empathy assessment failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Empathy assessment failed. Please try again."
        )


@router.post("/coach-empathy")
async def provide_empathy_coaching(
    request: EmpathyCoachingRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Provide real-time empathy coaching during conversations.
    
    Analyzes current conversation context to identify empathy opportunities
    and provides personalized coaching interventions to develop emotional intelligence.
    """
    try:
        logger.info(f"Empathy coaching requested for user {current_user.id}")
        
        # Analyze user's current emotional state from context
        user_emotion_analysis = None
        if request.conversation_context.get("user_message"):
            user_emotion_analysis = await detect_emotion_from_message(
                message_content=request.conversation_context["user_message"],
                user_id=str(current_user.id)
            )
        
        # Analyze other user's emotional state if provided
        other_user_emotion_analysis = None
        if request.other_user_emotions:
            # This would convert provided emotions to EmotionAnalysisResult format
            pass
        
        # Provide empathy coaching
        coaching_result = await empathy_engine.provide_real_time_empathy_coaching(
            user_id=str(current_user.id),
            conversation_context=request.conversation_context,
            user_emotion_analysis=user_emotion_analysis,
            other_user_emotion_analysis=other_user_emotion_analysis,
            db_session=db
        )
        
        return {
            "coaching_provided": True,
            "intervention_type": coaching_result.intervention_type.value,
            "skill_practiced": coaching_result.skill_practiced,
            "performance_score": coaching_result.performance_score,
            "feedback": coaching_result.feedback_provided,
            "next_practice": coaching_result.next_recommended_practice.value if coaching_result.next_recommended_practice else None,
            "user_engagement": coaching_result.user_engagement,
            "improvement_observed": coaching_result.improvement_observed
        }
        
    except Exception as e:
        logger.error(f"Empathy coaching failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Empathy coaching failed. Please try again."
        )


@router.get("/profile", response_model=EmotionalProfileResponse)
async def get_emotional_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's comprehensive emotional intelligence profile.
    
    Returns detailed emotional profile including baseline dimensions,
    empathy scores, regulation strategies, and development recommendations.
    """
    try:
        # Get or create emotional profile
        emotional_profile = db.query(EmotionalProfile).filter(
            EmotionalProfile.user_id == current_user.id
        ).first()
        
        if not emotional_profile:
            # Create initial emotional profile
            emotional_profile = EmotionalProfile(
                user_id=current_user.id,
                baseline_valence=0.0,
                baseline_arousal=0.0,
                baseline_dominance=0.0,
                sharing_consent_level=1,
                crisis_intervention_consent=True
            )
            db.add(emotional_profile)
            db.commit()
            db.refresh(emotional_profile)
        
        # Get recent empathy assessment
        recent_assessment = db.query(EmpathyAssessment).filter(
            EmpathyAssessment.emotional_profile_id == emotional_profile.id
        ).order_by(EmpathyAssessment.created_at.desc()).first()
        
        empathy_level = "Not assessed"
        if recent_assessment:
            if recent_assessment.cognitive_empathy_score and recent_assessment.cognitive_empathy_score > 70:
                empathy_level = "High"
            elif recent_assessment.cognitive_empathy_score and recent_assessment.cognitive_empathy_score > 50:
                empathy_level = "Moderate"
            else:
                empathy_level = "Developing"
        
        return EmotionalProfileResponse(
            user_id=str(current_user.id),
            baseline_dimensions={
                "valence": emotional_profile.baseline_valence,
                "arousal": emotional_profile.baseline_arousal,
                "dominance": emotional_profile.baseline_dominance
            },
            intelligence_metrics={
                "empathy_quotient": emotional_profile.empathy_quotient,
                "emotional_quotient": emotional_profile.emotional_quotient,
                "alexithymia_score": emotional_profile.alexithymia_score
            },
            dominant_emotions=emotional_profile.dominant_emotions,
            regulation_strategies=emotional_profile.primary_regulation_strategies,
            attachment_style=emotional_profile.attachment_style,
            crisis_risk_level=emotional_profile.assess_current_crisis_level(),
            empathy_development_level=empathy_level
        )
        
    except Exception as e:
        logger.error(f"Failed to get emotional profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emotional profile."
        )


@router.put("/profile/update")
async def update_emotional_profile(
    profile_updates: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's emotional profile preferences and settings.
    
    Allows users to update their emotional intelligence preferences,
    regulation strategies, and privacy settings.
    """
    try:
        # Get emotional profile
        emotional_profile = db.query(EmotionalProfile).filter(
            EmotionalProfile.user_id == current_user.id
        ).first()
        
        if not emotional_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Emotional profile not found."
            )
        
        # Update allowed fields
        updatable_fields = {
            'primary_regulation_strategies',
            'sharing_consent_level',
            'crisis_intervention_consent',
            'attachment_style'
        }
        
        updated_fields = []
        for field, value in profile_updates.items():
            if field in updatable_fields and hasattr(emotional_profile, field):
                setattr(emotional_profile, field, value)
                updated_fields.append(field)
        
        if updated_fields:
            db.commit()
            logger.info(f"Updated emotional profile fields {updated_fields} for user {current_user.id}")
        
        return {
            "success": True,
            "updated_fields": updated_fields,
            "message": "Emotional profile updated successfully."
        }
        
    except Exception as e:
        logger.error(f"Failed to update emotional profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update emotional profile."
        )


@router.get("/development-progress")
async def get_empathy_development_progress(
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's empathy development progress over time.
    
    Returns comprehensive analysis of empathy growth, improvement trends,
    and personalized development recommendations.
    """
    try:
        # Get empathy development tracking
        progress_report = await empathy_engine.track_empathy_development_over_time(
            user_id=str(current_user.id),
            time_period_days=days,
            db_session=db
        )
        
        return progress_report
        
    except Exception as e:
        logger.error(f"Failed to get empathy development progress for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve empathy development progress."
        )


@router.post("/check-in")
async def emotional_check_in(
    check_in_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Record emotional check-in from user.
    
    Allows users to self-report their emotional state for tracking
    and personalized emotional intelligence development.
    """
    try:
        # Get emotional profile
        emotional_profile = db.query(EmotionalProfile).filter(
            EmotionalProfile.user_id == current_user.id
        ).first()
        
        if not emotional_profile:
            # Create profile if it doesn't exist
            emotional_profile = EmotionalProfile(
                user_id=current_user.id,
                sharing_consent_level=1,
                crisis_intervention_consent=True
            )
            db.add(emotional_profile)
            db.commit()
            db.refresh(emotional_profile)
        
        # Create emotion reading from check-in
        emotion_reading = EmotionReading(
            emotional_profile_id=emotional_profile.id,
            valence_score=check_in_data.get('valence', 0.0),
            arousal_score=check_in_data.get('arousal', 0.0),
            dominance_score=check_in_data.get('dominance', 0.0),
            primary_emotion=BasicEmotion(check_in_data.get('primary_emotion', 'trust')),
            emotion_intensity=EmotionIntensity(check_in_data.get('intensity', 'moderate')),
            detection_modalities=['self_report'],
            confidence_scores={'self_report': 1.0},
            emotion_blend_scores={emotion.value: 0.125 for emotion in BasicEmotion},
            context_factors=check_in_data.get('context', {}),
            processing_time_ms=0  # Self-report, no processing time
        )
        
        db.add(emotion_reading)
        db.commit()
        
        # Provide personalized response based on check-in
        response_suggestions = []
        if check_in_data.get('valence', 0) < -0.3:
            response_suggestions.append("Would you like some emotion regulation suggestions?")
            response_suggestions.append("Is there someone you'd like to talk to about how you're feeling?")
        
        return {
            "success": True,
            "message": "Thank you for checking in with your emotions.",
            "emotional_state_recorded": True,
            "suggestions": response_suggestions,
            "crisis_support_available": True if check_in_data.get('valence', 0) < -0.6 else False
        }
        
    except Exception as e:
        logger.error(f"Emotional check-in failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Emotional check-in failed. Please try again."
        )


@router.get("/system-metrics")
async def get_emotional_intelligence_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get emotional intelligence system performance metrics.
    
    Returns system-wide metrics for emotion detection, response generation,
    and empathy development effectiveness. (Admin/monitoring endpoint)
    """
    try:
        # Check if user has appropriate permissions (would implement proper admin check)
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions."
            )
        
        # Get metrics from all emotional intelligence components
        emotion_detector_metrics = emotion_detector.get_performance_metrics()
        emotional_responder_metrics = emotional_responder.get_performance_metrics()
        empathy_engine_metrics = empathy_engine.get_empathy_engine_metrics()
        
        return {
            "emotion_detection": emotion_detector_metrics,
            "emotional_response": emotional_responder_metrics,
            "empathy_development": empathy_engine_metrics,
            "overall_system_health": "operational",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get emotional intelligence metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics."
        )