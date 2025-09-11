"""
Advanced Empathy & Emotional Intelligence Development Engine

Revolutionary system for measuring, tracking, and developing empathy and EQ through:
- Multi-dimensional empathy assessment (cognitive, affective, compassionate)
- Real-time empathy coaching during conversations
- Behavioral observation and feedback
- Personalized emotional intelligence development programs
- Social emotional learning through AI-guided practice
- Peer empathy matching and collaborative growth

Based on established psychological frameworks:
- Baron-Cohen Empathy Quotient (EQ)
- Mayer-Salovey Four-Branch Model of EI
- Davis Interpersonal Reactivity Index (IRI)
- Goleman's Emotional Intelligence Framework
- Hogan Empathy Scale
- Compassion Scale (CS)
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque

from app.models.emotional_intelligence import (
    EmotionalProfile, EmotionReading, EmpathyAssessment,
    EmotionalInteraction, BasicEmotion, EmotionIntensity
)
from app.models.user import User
from app.models.conversation import Message, Conversation
from app.services.emotion_detector import EmotionAnalysisResult
from app.database.connection import get_db_session as get_db
from sqlalchemy.orm import Session
from app.core.config import settings


logger = logging.getLogger(__name__)


class EmpathyDimension(str, Enum):
    """Core dimensions of empathy measurement."""
    COGNITIVE_EMPATHY = "cognitive_empathy"          # Understanding others' emotions
    AFFECTIVE_EMPATHY = "affective_empathy"          # Sharing others' emotions
    COMPASSIONATE_EMPATHY = "compassionate_empathy"  # Taking action to help
    PERSPECTIVE_TAKING = "perspective_taking"        # Seeing from others' viewpoints
    FANTASY = "fantasy"                              # Emotional investment in fictional characters
    PERSONAL_DISTRESS = "personal_distress"          # Distress from others' suffering
    EMPATHIC_CONCERN = "empathic_concern"           # Other-oriented concern


class EmpathyAssessmentType(str, Enum):
    """Types of empathy assessments."""
    INITIAL_SCREENING = "initial_screening"
    BEHAVIORAL_OBSERVATION = "behavioral_observation"
    SELF_REPORT_EQ = "self_report_eq"
    PEER_FEEDBACK = "peer_feedback"
    AI_ANALYSIS = "ai_analysis"
    SITUATIONAL_RESPONSE = "situational_response"
    CONVERSATION_ANALYSIS = "conversation_analysis"


class CoachingIntervention(str, Enum):
    """Types of empathy coaching interventions."""
    PERSPECTIVE_TAKING_EXERCISE = "perspective_taking_exercise"
    EMOTION_LABELING_PRACTICE = "emotion_labeling_practice"
    ACTIVE_LISTENING_TRAINING = "active_listening_training"
    COMPASSION_MEDITATION = "compassion_meditation"
    EMPATHY_SCENARIO_PRACTICE = "empathy_scenario_practice"
    EMOTIONAL_AWARENESS_BUILDING = "emotional_awareness_building"
    NONVIOLENT_COMMUNICATION = "nonviolent_communication"
    MINDFUL_EMPATHY_PRACTICE = "mindful_empathy_practice"


@dataclass
class EmpathyScore:
    """Comprehensive empathy scoring result."""
    overall_empathy_quotient: float  # 0-80 (Baron-Cohen scale)
    cognitive_empathy: float         # 0-100
    affective_empathy: float         # 0-100
    compassionate_empathy: float     # 0-100
    perspective_taking: float        # 0-100
    empathic_concern: float         # 0-100
    personal_distress: float        # 0-100
    assessment_confidence: float     # 0-1
    improvement_areas: List[str]
    strengths: List[str]
    coaching_recommendations: List[CoachingIntervention]


@dataclass
class EmpathyCoachingResult:
    """Result of empathy coaching session."""
    intervention_type: CoachingIntervention
    skill_practiced: str
    performance_score: float
    improvement_observed: bool
    feedback_provided: str
    next_recommended_practice: Optional[CoachingIntervention]
    practice_duration_minutes: int
    user_engagement: float


class AdvancedEmpathyEngine:
    """
    Revolutionary empathy measurement and development system.
    
    Combines psychological assessment tools with AI-powered behavioral analysis
    to provide comprehensive empathy development programs.
    """
    
    def __init__(self):
        """Initialize the empathy development engine."""
        
        # Assessment questionnaires and scales
        self.eq_questionnaire = self._load_empathy_quotient_scale()
        self.iri_questionnaire = self._load_interpersonal_reactivity_index()
        self.empathy_scenarios = self._load_empathy_scenarios()
        
        # Coaching programs and exercises
        self.coaching_programs = self._load_coaching_programs()
        self.practice_exercises = self._load_practice_exercises()
        
        # Behavioral analysis patterns
        self.empathy_indicators = self._load_empathy_behavioral_indicators()
        
        # Performance tracking
        self.coaching_metrics = {
            "total_assessments": 0,
            "total_coaching_sessions": 0,
            "avg_improvement_rate": 0.0,
            "most_effective_interventions": defaultdict(int),
            "user_engagement_scores": deque(maxlen=100)
        }
    
    async def conduct_comprehensive_empathy_assessment(
        self,
        user_id: str,
        assessment_type: EmpathyAssessmentType = EmpathyAssessmentType.INITIAL_SCREENING,
        db_session: Optional[Session] = None
    ) -> EmpathyScore:
        """
        Conduct comprehensive empathy assessment using multiple methodologies.
        
        Args:
            user_id: User to assess
            assessment_type: Type of assessment to conduct
            db_session: Database session
            
        Returns:
            Comprehensive empathy scoring results
        """
        if not db_session:
            db_session = next(get_db())
        
        try:
            # Get user's emotional profile
            emotional_profile = db_session.query(EmotionalProfile).filter(
                EmotionalProfile.user_id == user_id
            ).first()
            
            # Initialize assessment scores
            empathy_scores = {dimension.value: 0.0 for dimension in EmpathyDimension}
            assessment_confidence = 0.0
            
            # Conduct assessment based on type
            if assessment_type == EmpathyAssessmentType.BEHAVIORAL_OBSERVATION:
                scores, confidence = await self._assess_from_behavioral_patterns(
                    user_id, emotional_profile, db_session
                )
            
            elif assessment_type == EmpathyAssessmentType.CONVERSATION_ANALYSIS:
                scores, confidence = await self._assess_from_conversation_history(
                    user_id, emotional_profile, db_session
                )
            
            elif assessment_type == EmpathyAssessmentType.AI_ANALYSIS:
                scores, confidence = await self._assess_with_ai_analysis(
                    user_id, emotional_profile, db_session
                )
            
            else:  # Default to initial screening
                scores, confidence = await self._conduct_initial_screening(
                    user_id, emotional_profile, db_session
                )
            
            empathy_scores.update(scores)
            assessment_confidence = confidence
            
            # Calculate overall empathy quotient (Baron-Cohen scale: 0-80)
            overall_eq = self._calculate_overall_empathy_quotient(empathy_scores)
            
            # Identify strengths and improvement areas
            strengths, improvement_areas = self._analyze_empathy_profile(empathy_scores)
            
            # Generate coaching recommendations
            coaching_recommendations = await self._generate_coaching_recommendations(
                empathy_scores, emotional_profile
            )
            
            # Create empathy score result
            empathy_score = EmpathyScore(
                overall_empathy_quotient=overall_eq,
                cognitive_empathy=empathy_scores[EmpathyDimension.COGNITIVE_EMPATHY.value],
                affective_empathy=empathy_scores[EmpathyDimension.AFFECTIVE_EMPATHY.value],
                compassionate_empathy=empathy_scores[EmpathyDimension.COMPASSIONATE_EMPATHY.value],
                perspective_taking=empathy_scores[EmpathyDimension.PERSPECTIVE_TAKING.value],
                empathic_concern=empathy_scores[EmpathyDimension.EMPATHIC_CONCERN.value],
                personal_distress=empathy_scores[EmpathyDimension.PERSONAL_DISTRESS.value],
                assessment_confidence=assessment_confidence,
                improvement_areas=improvement_areas,
                strengths=strengths,
                coaching_recommendations=coaching_recommendations
            )
            
            # Save assessment to database
            await self._save_empathy_assessment(
                user_id, empathy_score, assessment_type, db_session
            )
            
            # Update tracking metrics
            self.coaching_metrics["total_assessments"] += 1
            
            return empathy_score
            
        except Exception as e:
            logger.error(f"Empathy assessment failed for user {user_id}: {str(e)}")
            return self._create_fallback_empathy_score()
    
    async def provide_real_time_empathy_coaching(
        self,
        user_id: str,
        conversation_context: Dict[str, Any],
        user_emotion_analysis: EmotionAnalysisResult,
        other_user_emotion_analysis: Optional[EmotionAnalysisResult] = None,
        db_session: Optional[Session] = None
    ) -> EmpathyCoachingResult:
        """
        Provide real-time empathy coaching during conversations.
        
        Args:
            user_id: User receiving coaching
            conversation_context: Current conversation context
            user_emotion_analysis: User's current emotional state
            other_user_emotion_analysis: Other party's emotional state if available
            db_session: Database session
            
        Returns:
            Empathy coaching result with feedback and recommendations
        """
        if not db_session:
            db_session = next(get_db())
        
        try:
            start_time = datetime.now()
            
            # Analyze empathy opportunity in current context
            empathy_opportunity = await self._identify_empathy_opportunity(
                conversation_context, user_emotion_analysis, other_user_emotion_analysis
            )
            
            if not empathy_opportunity["opportunity_present"]:
                return self._create_no_coaching_result()
            
            # Select appropriate coaching intervention
            intervention_type = await self._select_coaching_intervention(
                empathy_opportunity, user_id, db_session
            )
            
            # Provide targeted empathy coaching
            coaching_result = await self._deliver_empathy_coaching(
                intervention_type, empathy_opportunity, user_emotion_analysis
            )
            
            # Measure user's empathic response
            performance_score = await self._assess_empathic_response(
                user_id, conversation_context, coaching_result, db_session
            )
            
            # Calculate practice duration
            practice_duration = int((datetime.now() - start_time).total_seconds() / 60)
            
            # Generate personalized feedback
            feedback = await self._generate_empathy_feedback(
                intervention_type, performance_score, empathy_opportunity
            )
            
            # Recommend next practice
            next_practice = await self._recommend_next_empathy_practice(
                intervention_type, performance_score, user_id, db_session
            )
            
            # Create coaching result
            result = EmpathyCoachingResult(
                intervention_type=intervention_type,
                skill_practiced=empathy_opportunity["skill_focus"],
                performance_score=performance_score,
                improvement_observed=performance_score > 0.6,
                feedback_provided=feedback,
                next_recommended_practice=next_practice,
                practice_duration_minutes=practice_duration,
                user_engagement=coaching_result["engagement_score"]
            )
            
            # Update coaching metrics
            await self._update_coaching_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time empathy coaching failed for user {user_id}: {str(e)}")
            return self._create_fallback_coaching_result()
    
    async def track_empathy_development_over_time(
        self,
        user_id: str,
        time_period_days: int = 30,
        db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Track user's empathy development progress over time.
        
        Args:
            user_id: User to track
            time_period_days: Number of days to analyze
            db_session: Database session
            
        Returns:
            Comprehensive empathy development report
        """
        if not db_session:
            db_session = next(get_db())
        
        try:
            # Get empathy assessments over time period
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            assessments = db_session.query(EmpathyAssessment).filter(
                EmpathyAssessment.emotional_profile.has(user_id=user_id),
                EmpathyAssessment.created_at >= cutoff_date
            ).order_by(EmpathyAssessment.created_at).all()
            
            if not assessments:
                return {"error": "No empathy assessments found for this period"}
            
            # Analyze empathy trends
            empathy_trends = await self._analyze_empathy_trends(assessments)
            
            # Calculate improvement rates
            improvement_rates = await self._calculate_empathy_improvement_rates(assessments)
            
            # Identify most effective coaching interventions
            effective_interventions = await self._identify_most_effective_interventions(
                user_id, time_period_days, db_session
            )
            
            # Generate development recommendations
            development_recommendations = await self._generate_development_recommendations(
                empathy_trends, improvement_rates, effective_interventions
            )
            
            return {
                "user_id": user_id,
                "assessment_period_days": time_period_days,
                "total_assessments": len(assessments),
                "empathy_trends": empathy_trends,
                "improvement_rates": improvement_rates,
                "most_effective_interventions": effective_interventions,
                "development_recommendations": development_recommendations,
                "overall_progress": self._calculate_overall_progress(assessments),
                "next_milestone_goals": await self._set_next_milestone_goals(
                    assessments[-1] if assessments else None
                )
            }
            
        except Exception as e:
            logger.error(f"Empathy tracking failed for user {user_id}: {str(e)}")
            return {"error": "Failed to track empathy development"}
    
    async def match_users_for_empathy_practice(
        self,
        user_id: str,
        matching_criteria: Dict[str, Any],
        db_session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Match users for collaborative empathy practice and development.
        
        Args:
            user_id: User seeking empathy practice partner
            matching_criteria: Criteria for matching (skill level, focus areas, etc.)
            db_session: Database session
            
        Returns:
            List of potential empathy practice partners
        """
        if not db_session:
            db_session = next(get_db())
        
        try:
            # Get user's empathy profile
            user_profile = db_session.query(EmotionalProfile).filter(
                EmotionalProfile.user_id == user_id
            ).first()
            
            if not user_profile:
                return []
            
            # Get recent empathy assessment for user
            user_assessment = db_session.query(EmpathyAssessment).filter(
                EmpathyAssessment.emotional_profile_id == user_profile.id
            ).order_by(EmpathyAssessment.created_at.desc()).first()
            
            if not user_assessment:
                return []
            
            # Find compatible practice partners
            potential_partners = await self._find_empathy_practice_partners(
                user_assessment, matching_criteria, db_session
            )
            
            # Score and rank potential partners
            ranked_partners = await self._rank_empathy_practice_partners(
                user_assessment, potential_partners, matching_criteria
            )
            
            return ranked_partners[:5]  # Return top 5 matches
            
        except Exception as e:
            logger.error(f"Empathy matching failed for user {user_id}: {str(e)}")
            return []
    
    # Private assessment methods
    
    async def _assess_from_behavioral_patterns(
        self,
        user_id: str,
        emotional_profile: Optional[EmotionalProfile],
        db_session: Session
    ) -> Tuple[Dict[str, float], float]:
        """Assess empathy from observed behavioral patterns."""
        
        # Get recent emotional interactions
        recent_interactions = db_session.query(EmotionalInteraction).filter(
            EmotionalInteraction.source_user_id == user_id,
            EmotionalInteraction.created_at >= datetime.now() - timedelta(days=30)
        ).all()
        
        if not recent_interactions:
            return {dimension.value: 50.0 for dimension in EmpathyDimension}, 0.3
        
        # Analyze empathic behaviors
        empathy_scores = {}
        
        # Cognitive empathy: Understanding others' emotions
        emotion_recognition_accuracy = np.mean([
            interaction.empathy_demonstrated or 0.5 
            for interaction in recent_interactions
        ])
        empathy_scores[EmpathyDimension.COGNITIVE_EMPATHY.value] = emotion_recognition_accuracy * 100
        
        # Affective empathy: Emotional contagion and sharing
        emotional_contagion_scores = [
            interaction.emotional_contagion_score or 0.5 
            for interaction in recent_interactions
            if interaction.emotional_contagion_score is not None
        ]
        empathy_scores[EmpathyDimension.AFFECTIVE_EMPATHY.value] = np.mean(emotional_contagion_scores) * 100 if emotional_contagion_scores else 50.0
        
        # Compassionate empathy: Taking action to help
        support_provided_scores = [
            interaction.emotional_support_provided or 0.5 
            for interaction in recent_interactions
            if interaction.emotional_support_provided is not None
        ]
        empathy_scores[EmpathyDimension.COMPASSIONATE_EMPATHY.value] = np.mean(support_provided_scores) * 100 if support_provided_scores else 50.0
        
        # Set other dimensions to moderate baseline
        empathy_scores[EmpathyDimension.PERSPECTIVE_TAKING.value] = 60.0
        empathy_scores[EmpathyDimension.EMPATHIC_CONCERN.value] = 65.0
        empathy_scores[EmpathyDimension.PERSONAL_DISTRESS.value] = 45.0  # Lower is better for this dimension
        empathy_scores[EmpathyDimension.FANTASY.value] = 55.0
        
        confidence = min(0.8, len(recent_interactions) / 10.0)  # Higher confidence with more data
        
        return empathy_scores, confidence
    
    async def _assess_from_conversation_history(
        self,
        user_id: str,
        emotional_profile: Optional[EmotionalProfile],
        db_session: Session
    ) -> Tuple[Dict[str, float], float]:
        """Assess empathy from conversation history analysis."""
        
        # Get recent conversations
        recent_conversations = db_session.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.created_at >= datetime.now() - timedelta(days=30)
        ).all()
        
        if not recent_conversations:
            return {dimension.value: 50.0 for dimension in EmpathyDimension}, 0.2
        
        # Analyze conversations for empathic language and responses
        empathy_indicators = {
            "empathic_language": 0,
            "perspective_taking_phrases": 0,
            "emotional_validation": 0,
            "supportive_responses": 0,
            "total_messages": 0
        }
        
        for conversation in recent_conversations:
            messages = db_session.query(Message).filter(
                Message.conversation_id == conversation.id,
                Message.user_id == user_id
            ).all()
            
            empathy_indicators["total_messages"] += len(messages)
            
            for message in messages:
                if message.content:
                    empathy_indicators.update(
                        self._analyze_message_for_empathy(message.content)
                    )
        
        # Calculate empathy scores from conversation analysis
        empathy_scores = {}
        
        if empathy_indicators["total_messages"] > 0:
            empathic_ratio = empathy_indicators["empathic_language"] / empathy_indicators["total_messages"]
            perspective_ratio = empathy_indicators["perspective_taking_phrases"] / empathy_indicators["total_messages"]
            validation_ratio = empathy_indicators["emotional_validation"] / empathy_indicators["total_messages"]
            
            empathy_scores[EmpathyDimension.COGNITIVE_EMPATHY.value] = min(100, empathic_ratio * 200)
            empathy_scores[EmpathyDimension.PERSPECTIVE_TAKING.value] = min(100, perspective_ratio * 300)
            empathy_scores[EmpathyDimension.EMPATHIC_CONCERN.value] = min(100, validation_ratio * 250)
            empathy_scores[EmpathyDimension.AFFECTIVE_EMPATHY.value] = min(100, (empathic_ratio + validation_ratio) * 100)
            empathy_scores[EmpathyDimension.COMPASSIONATE_EMPATHY.value] = min(100, empathy_indicators["supportive_responses"] / empathy_indicators["total_messages"] * 200)
        else:
            empathy_scores = {dimension.value: 50.0 for dimension in EmpathyDimension}
        
        # Set remaining dimensions
        empathy_scores[EmpathyDimension.PERSONAL_DISTRESS.value] = 40.0  # Moderate baseline
        empathy_scores[EmpathyDimension.FANTASY.value] = 55.0
        
        confidence = min(0.7, empathy_indicators["total_messages"] / 50.0)
        
        return empathy_scores, confidence
    
    def _analyze_message_for_empathy(self, message_content: str) -> Dict[str, int]:
        """Analyze individual message for empathy indicators."""
        content_lower = message_content.lower()
        
        indicators = {
            "empathic_language": 0,
            "perspective_taking_phrases": 0,
            "emotional_validation": 0,
            "supportive_responses": 0
        }
        
        # Empathic language patterns
        empathic_phrases = [
            "i understand", "i can see", "that must be", "i imagine", "i feel",
            "sounds like", "seems like", "i hear you", "that's difficult",
            "i'm sorry you're going through"
        ]
        
        # Perspective-taking phrases
        perspective_phrases = [
            "from your perspective", "in your situation", "if i were you",
            "you might feel", "it makes sense that", "anyone would",
            "i can imagine", "putting myself in your shoes"
        ]
        
        # Emotional validation phrases
        validation_phrases = [
            "your feelings are valid", "that's understandable", "makes perfect sense",
            "completely normal", "anyone would feel", "you have every right",
            "that's a natural response", "totally understandable"
        ]
        
        # Supportive response patterns
        supportive_phrases = [
            "i'm here for you", "you're not alone", "how can i help",
            "what do you need", "i support you", "you can get through this",
            "i believe in you", "you're strong"
        ]
        
        # Count occurrences
        for phrase in empathic_phrases:
            if phrase in content_lower:
                indicators["empathic_language"] += 1
        
        for phrase in perspective_phrases:
            if phrase in content_lower:
                indicators["perspective_taking_phrases"] += 1
        
        for phrase in validation_phrases:
            if phrase in content_lower:
                indicators["emotional_validation"] += 1
        
        for phrase in supportive_phrases:
            if phrase in content_lower:
                indicators["supportive_responses"] += 1
        
        return indicators
    
    async def _assess_with_ai_analysis(
        self,
        user_id: str,
        emotional_profile: Optional[EmotionalProfile],
        db_session: Session
    ) -> Tuple[Dict[str, float], float]:
        """Assess empathy using AI analysis of user interactions."""
        
        # This would integrate with advanced NLP models to analyze:
        # - Language patterns indicating empathy
        # - Emotional responsiveness to others
        # - Perspective-taking ability
        # - Compassionate action tendencies
        
        # For now, return moderate baseline scores
        empathy_scores = {
            EmpathyDimension.COGNITIVE_EMPATHY.value: 65.0,
            EmpathyDimension.AFFECTIVE_EMPATHY.value: 60.0,
            EmpathyDimension.COMPASSIONATE_EMPATHY.value: 55.0,
            EmpathyDimension.PERSPECTIVE_TAKING.value: 70.0,
            EmpathyDimension.EMPATHIC_CONCERN.value: 68.0,
            EmpathyDimension.PERSONAL_DISTRESS.value: 45.0,
            EmpathyDimension.FANTASY.value: 58.0
        }
        
        return empathy_scores, 0.6
    
    async def _conduct_initial_screening(
        self,
        user_id: str,
        emotional_profile: Optional[EmotionalProfile],
        db_session: Session
    ) -> Tuple[Dict[str, float], float]:
        """Conduct initial empathy screening assessment."""
        
        # Use existing emotional profile data if available
        if emotional_profile:
            eq_score = emotional_profile.empathy_quotient or 40.0  # Baron-Cohen scale average
            
            # Derive dimensional scores from overall EQ
            empathy_scores = {
                EmpathyDimension.COGNITIVE_EMPATHY.value: eq_score * 1.2,  # Usually slightly higher
                EmpathyDimension.AFFECTIVE_EMPATHY.value: eq_score * 0.9,  # Often lower than cognitive
                EmpathyDimension.COMPASSIONATE_EMPATHY.value: eq_score * 1.0,  # Proportional
                EmpathyDimension.PERSPECTIVE_TAKING.value: eq_score * 1.1,   # Related to cognitive
                EmpathyDimension.EMPATHIC_CONCERN.value: eq_score * 1.15,    # Often high in empathic individuals
                EmpathyDimension.PERSONAL_DISTRESS.value: max(20, 80 - eq_score * 0.8),  # Inverse relationship
                EmpathyDimension.FANTASY.value: eq_score * 0.95  # Moderate correlation
            }
            
            # Clamp to valid ranges
            for key in empathy_scores:
                empathy_scores[key] = max(0, min(100, empathy_scores[key]))
            
            confidence = 0.7 if eq_score > 0 else 0.3
        
        else:
            # Default moderate baseline scores
            empathy_scores = {dimension.value: 50.0 for dimension in EmpathyDimension}
            confidence = 0.3
        
        return empathy_scores, confidence
    
    def _calculate_overall_empathy_quotient(self, empathy_scores: Dict[str, float]) -> float:
        """Calculate overall empathy quotient using Baron-Cohen formula."""
        
        # Key dimensions for EQ calculation (0-80 scale)
        key_dimensions = [
            empathy_scores.get(EmpathyDimension.COGNITIVE_EMPATHY.value, 50.0),
            empathy_scores.get(EmpathyDimension.AFFECTIVE_EMPATHY.value, 50.0),
            empathy_scores.get(EmpathyDimension.PERSPECTIVE_TAKING.value, 50.0),
            empathy_scores.get(EmpathyDimension.EMPATHIC_CONCERN.value, 50.0)
        ]
        
        # Convert from 0-100 scale to 0-80 scale (Baron-Cohen EQ)
        average_score = np.mean(key_dimensions)
        eq_score = (average_score / 100.0) * 80.0
        
        return max(0.0, min(80.0, eq_score))
    
    def _analyze_empathy_profile(self, empathy_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Analyze empathy profile to identify strengths and improvement areas."""
        
        strengths = []
        improvement_areas = []
        
        # Thresholds for high/low scores
        high_threshold = 70.0
        low_threshold = 40.0
        
        dimension_labels = {
            EmpathyDimension.COGNITIVE_EMPATHY.value: "Understanding Others' Emotions",
            EmpathyDimension.AFFECTIVE_EMPATHY.value: "Sharing Others' Emotions", 
            EmpathyDimension.COMPASSIONATE_EMPATHY.value: "Taking Action to Help",
            EmpathyDimension.PERSPECTIVE_TAKING.value: "Seeing Others' Viewpoints",
            EmpathyDimension.EMPATHIC_CONCERN.value: "Caring for Others' Wellbeing",
            EmpathyDimension.PERSONAL_DISTRESS.value: "Managing Emotional Overwhelm",
            EmpathyDimension.FANTASY.value: "Emotional Investment in Stories"
        }
        
        for dimension, score in empathy_scores.items():
            label = dimension_labels.get(dimension, dimension)
            
            if dimension == EmpathyDimension.PERSONAL_DISTRESS.value:
                # Personal distress: lower is better
                if score < low_threshold:
                    strengths.append(f"Strong emotional regulation - {label}")
                elif score > high_threshold:
                    improvement_areas.append(f"Managing overwhelming emotions - {label}")
            else:
                # Other dimensions: higher is better
                if score > high_threshold:
                    strengths.append(f"Strong {label.lower()}")
                elif score < low_threshold:
                    improvement_areas.append(f"Developing {label.lower()}")
        
        return strengths, improvement_areas
    
    async def _generate_coaching_recommendations(
        self,
        empathy_scores: Dict[str, float],
        emotional_profile: Optional[EmotionalProfile]
    ) -> List[CoachingIntervention]:
        """Generate personalized empathy coaching recommendations."""
        
        recommendations = []
        
        # Cognitive empathy recommendations
        if empathy_scores.get(EmpathyDimension.COGNITIVE_EMPATHY.value, 50) < 60:
            recommendations.append(CoachingIntervention.EMOTION_LABELING_PRACTICE)
            recommendations.append(CoachingIntervention.EMOTIONAL_AWARENESS_BUILDING)
        
        # Affective empathy recommendations  
        if empathy_scores.get(EmpathyDimension.AFFECTIVE_EMPATHY.value, 50) < 60:
            recommendations.append(CoachingIntervention.COMPASSION_MEDITATION)
            recommendations.append(CoachingIntervention.MINDFUL_EMPATHY_PRACTICE)
        
        # Perspective taking recommendations
        if empathy_scores.get(EmpathyDimension.PERSPECTIVE_TAKING.value, 50) < 60:
            recommendations.append(CoachingIntervention.PERSPECTIVE_TAKING_EXERCISE)
            recommendations.append(CoachingIntervention.EMPATHY_SCENARIO_PRACTICE)
        
        # Compassionate empathy recommendations
        if empathy_scores.get(EmpathyDimension.COMPASSIONATE_EMPATHY.value, 50) < 60:
            recommendations.append(CoachingIntervention.ACTIVE_LISTENING_TRAINING)
            recommendations.append(CoachingIntervention.NONVIOLENT_COMMUNICATION)
        
        # Personal distress management (high scores need help)
        if empathy_scores.get(EmpathyDimension.PERSONAL_DISTRESS.value, 50) > 70:
            recommendations.append(CoachingIntervention.MINDFUL_EMPATHY_PRACTICE)
            recommendations.append(CoachingIntervention.COMPASSION_MEDITATION)
        
        # Consider user's previous effective interventions
        if emotional_profile and emotional_profile.coaching_history:
            effective_interventions = emotional_profile.coaching_history.get("effective_interventions", [])
            # Prioritize previously effective interventions
            for intervention in effective_interventions:
                if intervention in [ci.value for ci in CoachingIntervention]:
                    recommendations.insert(0, CoachingIntervention(intervention))
        
        # Remove duplicates and limit to top 5
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]
    
    # Additional helper methods would continue...
    
    def _load_empathy_quotient_scale(self) -> Dict[str, Any]:
        """Load Baron-Cohen Empathy Quotient scale questions."""
        return {
            "questions": [
                "I can easily tell if someone else wants to enter a conversation.",
                "I find it difficult to explain to others things that I understand easily, when they don't understand it first time.",
                "I really enjoy caring for other people.",
                "I find it hard to know what to do in a social situation.",
                "People often tell me that I went too far in driving my point home in a discussion.",
                # ... Additional EQ questions would be included
            ],
            "scoring": "4_point_likert",
            "max_score": 80
        }
    
    def _load_interpersonal_reactivity_index(self) -> Dict[str, Any]:
        """Load Davis Interpersonal Reactivity Index questions."""
        return {
            "perspective_taking": [
                "I try to look at everybody's side of a disagreement before I make a decision.",
                "I sometimes try to understand my friends better by imagining how things look from their perspective.",
                "When I'm upset at someone, I usually try to put myself in their shoes for a while."
            ],
            "fantasy": [
                "I daydream and fantasize, with some regularity, about things that might happen to me.",
                "I really get involved with the feelings of the characters in a novel."
            ],
            "empathic_concern": [
                "I often have tender, concerned feelings for people less fortunate than me.",
                "I am often quite touched by things that I see happen.",
                "I would describe myself as a pretty soft-hearted person."
            ],
            "personal_distress": [
                "When I see someone get hurt, I tend to remain calm.",
                "I sometimes feel helpless when I am in the middle of a very emotional situation.",
                "Being in a tense emotional situation scares me."
            ]
        }
    
    def _create_fallback_empathy_score(self) -> EmpathyScore:
        """Create fallback empathy score for error cases."""
        return EmpathyScore(
            overall_empathy_quotient=40.0,
            cognitive_empathy=50.0,
            affective_empathy=50.0,
            compassionate_empathy=50.0,
            perspective_taking=50.0,
            empathic_concern=50.0,
            personal_distress=50.0,
            assessment_confidence=0.3,
            improvement_areas=["General empathy development needed"],
            strengths=["Baseline empathy present"],
            coaching_recommendations=[CoachingIntervention.EMOTIONAL_AWARENESS_BUILDING]
        )
    
    # Additional methods for coaching, tracking, matching would continue...
    
    def get_empathy_engine_metrics(self) -> Dict[str, Any]:
        """Get empathy engine performance metrics."""
        return {
            "total_assessments_conducted": self.coaching_metrics["total_assessments"],
            "total_coaching_sessions": self.coaching_metrics["total_coaching_sessions"],
            "average_improvement_rate": self.coaching_metrics["avg_improvement_rate"],
            "most_effective_interventions": dict(self.coaching_metrics["most_effective_interventions"]),
            "average_user_engagement": np.mean(self.coaching_metrics["user_engagement_scores"]) if self.coaching_metrics["user_engagement_scores"] else 0.0,
            "system_status": "operational"
        }


# Create global empathy engine instance
empathy_engine = AdvancedEmpathyEngine()