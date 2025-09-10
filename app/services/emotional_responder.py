"""
Revolutionary Emotional Response Engine

AI-powered system that adapts communication style and content based on:
- Real-time emotion detection results
- User's emotional profile and history
- Therapeutic communication principles
- Contextual appropriateness
- Crisis detection and intervention protocols
- Emotional regulation assistance

Evidence-based approaches include:
- Cognitive Behavioral Therapy (CBT) techniques
- Dialectical Behavior Therapy (DBT) skills
- Person-Centered Therapy principles  
- Trauma-informed communication
- Crisis intervention protocols
- Motivational Interviewing techniques
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

from app.models.emotional_intelligence import (
    EmotionalProfile, EmotionReading, BasicEmotion, 
    EmotionIntensity, CrisisLevel, EmotionRegulationStrategy
)
from app.models.user import User
from app.models.conversation import Message, Conversation
from app.services.emotion_detector import EmotionAnalysisResult, emotion_detector
from app.services.llm_service import LLMService
from app.core.config import settings


logger = logging.getLogger(__name__)


class ResponseStyle(str, Enum):
    """Different emotional response styles."""
    EMPATHETIC = "empathetic"
    SUPPORTIVE = "supportive" 
    VALIDATING = "validating"
    COACHING = "coaching"
    CALMING = "calming"
    ENERGIZING = "energizing"
    ANALYTICAL = "analytical"
    PLAYFUL = "playful"
    PROFESSIONAL = "professional"
    CRISIS_INTERVENTION = "crisis_intervention"


class TherapeuticTechnique(str, Enum):
    """Evidence-based therapeutic communication techniques."""
    ACTIVE_LISTENING = "active_listening"
    REFLECTION = "reflection"
    VALIDATION = "validation"
    COGNITIVE_REFRAMING = "cognitive_reframing"
    MINDFULNESS_GUIDANCE = "mindfulness_guidance"
    GROUNDING_TECHNIQUE = "grounding_technique"
    EMOTION_REGULATION = "emotion_regulation"
    MOTIVATIONAL_INTERVIEWING = "motivational_interviewing"
    CRISIS_SUPPORT = "crisis_support"
    PSYCHOEDUCATION = "psychoeducation"


@dataclass
class EmotionalResponse:
    """Result of emotionally-aware response generation."""
    response_text: str
    response_style: ResponseStyle
    therapeutic_techniques: List[TherapeuticTechnique]
    emotional_intent: Dict[str, float]  # Intended emotional impact
    regulation_strategies: List[EmotionRegulationStrategy]
    crisis_level: CrisisLevel
    confidence_score: float
    personalization_applied: bool
    followup_suggestions: List[str]
    processing_time_ms: int


class AdvancedEmotionalResponder:
    """
    Revolutionary emotionally-intelligent response system.
    
    Generates contextually appropriate, therapeutically informed responses
    that adapt to the user's current emotional state and long-term patterns.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        
        # Therapeutic response templates
        self.response_templates = self._load_response_templates()
        
        # Crisis intervention resources
        self.crisis_resources = self._load_crisis_resources()
        
        # Emotion regulation strategies
        self.regulation_strategies = self._load_regulation_strategies()
        
        # Performance tracking
        self.response_metrics = {
            "total_responses": 0,
            "avg_processing_time": 0.0,
            "style_usage": {},
            "technique_effectiveness": {},
            "crisis_interventions": 0
        }
    
    async def generate_emotional_response(
        self,
        user_message: str,
        emotion_analysis: EmotionAnalysisResult,
        user_id: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        emotional_profile: Optional[EmotionalProfile] = None
    ) -> EmotionalResponse:
        """
        Generate emotionally-aware response based on comprehensive analysis.
        
        Args:
            user_message: The user's message
            emotion_analysis: Results from emotion detection
            user_id: User identifier
            conversation_context: Current conversation context
            emotional_profile: User's emotional profile if available
            
        Returns:
            Emotionally intelligent response with therapeutic elements
        """
        start_time = datetime.now()
        
        try:
            # Assess crisis level and intervention needs
            crisis_level = await self._assess_crisis_level(
                emotion_analysis, user_message, emotional_profile
            )
            
            # Handle crisis situations with priority
            if crisis_level in [CrisisLevel.CRISIS, CrisisLevel.EMERGENCY]:
                return await self._generate_crisis_response(
                    user_message, emotion_analysis, crisis_level, user_id, start_time
                )
            
            # Determine appropriate response style
            response_style = await self._determine_response_style(
                emotion_analysis, emotional_profile, conversation_context
            )
            
            # Select therapeutic techniques
            therapeutic_techniques = await self._select_therapeutic_techniques(
                emotion_analysis, response_style, emotional_profile
            )
            
            # Generate base response using LLM with emotional context
            base_response = await self._generate_base_response(
                user_message, emotion_analysis, response_style, therapeutic_techniques
            )
            
            # Apply personalization based on emotional profile
            personalized_response, personalization_applied = await self._apply_personalization(
                base_response, emotional_profile, emotion_analysis
            )
            
            # Add emotion regulation strategies if needed
            regulation_strategies = await self._suggest_regulation_strategies(
                emotion_analysis, emotional_profile
            )
            
            # Generate followup suggestions
            followup_suggestions = await self._generate_followup_suggestions(
                emotion_analysis, response_style, conversation_context
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create final response
            emotional_response = EmotionalResponse(
                response_text=personalized_response,
                response_style=response_style,
                therapeutic_techniques=therapeutic_techniques,
                emotional_intent=await self._calculate_emotional_intent(response_style, emotion_analysis),
                regulation_strategies=regulation_strategies,
                crisis_level=crisis_level,
                confidence_score=self._calculate_confidence_score(emotion_analysis, personalization_applied),
                personalization_applied=personalization_applied,
                followup_suggestions=followup_suggestions,
                processing_time_ms=int(processing_time)
            )
            
            # Update metrics
            await self._update_response_metrics(emotional_response)
            
            return emotional_response
            
        except Exception as e:
            logger.error(f"Error generating emotional response: {str(e)}")
            return await self._create_fallback_response(user_message, start_time)
    
    async def _assess_crisis_level(
        self,
        emotion_analysis: EmotionAnalysisResult,
        user_message: str,
        emotional_profile: Optional[EmotionalProfile]
    ) -> CrisisLevel:
        """Assess if user is in emotional crisis requiring intervention."""
        crisis_indicators = 0
        
        # Check emotion analysis for crisis patterns
        if (emotion_analysis.valence < -0.7 and 
            emotion_analysis.arousal > 0.5 and
            emotion_analysis.emotion_intensity in [EmotionIntensity.HIGH, EmotionIntensity.EXTREME]):
            crisis_indicators += 2
        
        # Check for crisis keywords in message
        crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'no point', 'give up',
            'hurt myself', 'self harm', 'can\'t go on', 'hopeless', 'worthless'
        ]
        
        message_lower = user_message.lower()
        crisis_keyword_count = sum(1 for keyword in crisis_keywords if keyword in message_lower)
        crisis_indicators += crisis_keyword_count
        
        # Check emotional profile for known risk factors
        if emotional_profile and emotional_profile.crisis_indicators:
            recent_flags = emotional_profile.crisis_indicators.get("recent_flags", [])
            crisis_indicators += len(recent_flags)
        
        # Assess crisis level based on indicators
        if crisis_indicators >= 3:
            return CrisisLevel.EMERGENCY
        elif crisis_indicators >= 2:
            return CrisisLevel.CRISIS
        elif crisis_indicators >= 1:
            return CrisisLevel.HIGH_RISK
        elif emotion_analysis.valence < -0.5:
            return CrisisLevel.MODERATE_CONCERN
        elif emotion_analysis.valence < -0.2:
            return CrisisLevel.MILD_CONCERN
        else:
            return CrisisLevel.NONE
    
    async def _generate_crisis_response(
        self,
        user_message: str,
        emotion_analysis: EmotionAnalysisResult,
        crisis_level: CrisisLevel,
        user_id: str,
        start_time: datetime
    ) -> EmotionalResponse:
        """Generate specialized crisis intervention response."""
        logger.warning(f"Crisis intervention triggered for user {user_id}: {crisis_level}")
        
        # Immediate safety and validation
        crisis_responses = {
            CrisisLevel.EMERGENCY: [
                "I'm genuinely concerned about you right now. You're not alone, and there are people who want to help.",
                "What you're feeling is incredibly difficult, and I want you to know that reaching out shows tremendous strength.",
                "Please know that this pain you're experiencing, while overwhelming, is temporary. There are trained professionals who can help right now."
            ],
            CrisisLevel.CRISIS: [
                "I hear that you're going through something really challenging right now. That takes courage to share.",
                "Your feelings are completely valid, and it's understandable that you're struggling with this.",
                "You don't have to navigate this alone. There are people and resources specifically trained to help."
            ]
        }
        
        # Select appropriate crisis response
        base_responses = crisis_responses.get(crisis_level, crisis_responses[CrisisLevel.CRISIS])
        crisis_response = random.choice(base_responses)
        
        # Add crisis resources
        crisis_resources_text = await self._get_crisis_resources_text()
        
        # Combine response with resources
        full_response = f"{crisis_response}\n\n{crisis_resources_text}"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Track crisis intervention
        self.response_metrics["crisis_interventions"] += 1
        
        return EmotionalResponse(
            response_text=full_response,
            response_style=ResponseStyle.CRISIS_INTERVENTION,
            therapeutic_techniques=[TherapeuticTechnique.CRISIS_SUPPORT, TherapeuticTechnique.VALIDATION],
            emotional_intent={"safety": 1.0, "hope": 0.8, "connection": 0.9},
            regulation_strategies=[EmotionRegulationStrategy.SOCIAL_SHARING],
            crisis_level=crisis_level,
            confidence_score=0.95,  # High confidence in crisis protocols
            personalization_applied=False,  # Crisis responses prioritize safety over personalization
            followup_suggestions=[
                "Would you like to talk about what's been making you feel this way?",
                "Are there trusted people in your life you could reach out to?",
                "Would it help to explore some immediate coping strategies?"
            ],
            processing_time_ms=int(processing_time)
        )
    
    async def _determine_response_style(
        self,
        emotion_analysis: EmotionAnalysisResult,
        emotional_profile: Optional[EmotionalProfile],
        conversation_context: Optional[Dict[str, Any]]
    ) -> ResponseStyle:
        """Determine the most appropriate response style."""
        
        # Primary emotion-based style selection
        emotion_style_mapping = {
            BasicEmotion.SADNESS: ResponseStyle.SUPPORTIVE,
            BasicEmotion.ANGER: ResponseStyle.VALIDATING,
            BasicEmotion.FEAR: ResponseStyle.CALMING,
            BasicEmotion.JOY: ResponseStyle.ENERGIZING,
            BasicEmotion.SURPRISE: ResponseStyle.ANALYTICAL,
            BasicEmotion.DISGUST: ResponseStyle.VALIDATING,
            BasicEmotion.TRUST: ResponseStyle.EMPATHETIC,
            BasicEmotion.ANTICIPATION: ResponseStyle.COACHING
        }
        
        base_style = emotion_style_mapping.get(emotion_analysis.primary_emotion, ResponseStyle.EMPATHETIC)
        
        # Adjust based on emotion intensity
        if emotion_analysis.emotion_intensity in [EmotionIntensity.HIGH, EmotionIntensity.EXTREME]:
            if base_style in [ResponseStyle.ENERGIZING, ResponseStyle.PLAYFUL]:
                base_style = ResponseStyle.CALMING  # High intensity needs calming
        
        # Adjust based on valence and arousal
        if emotion_analysis.valence < -0.5:  # Strong negative emotions
            if emotion_analysis.arousal > 0.5:  # High arousal negative
                base_style = ResponseStyle.CALMING
            else:  # Low arousal negative
                base_style = ResponseStyle.SUPPORTIVE
        
        # Apply user preferences from emotional profile
        if emotional_profile and emotional_profile.support_preferences:
            preferred_styles = emotional_profile.support_preferences.get("response_styles", [])
            if preferred_styles and base_style.value not in preferred_styles:
                # Try to find a preferred style that's appropriate
                appropriate_preferences = [style for style in preferred_styles 
                                         if self._is_style_appropriate(style, emotion_analysis)]
                if appropriate_preferences:
                    base_style = ResponseStyle(appropriate_preferences[0])
        
        return base_style
    
    def _is_style_appropriate(self, style: str, emotion_analysis: EmotionAnalysisResult) -> bool:
        """Check if a response style is appropriate for current emotional state."""
        # Don't use playful style for negative emotions
        if style == ResponseStyle.PLAYFUL.value and emotion_analysis.valence < -0.2:
            return False
        
        # Don't use energizing style for high arousal negative states
        if (style == ResponseStyle.ENERGIZING.value and 
            emotion_analysis.arousal > 0.5 and emotion_analysis.valence < -0.2):
            return False
        
        return True
    
    async def _select_therapeutic_techniques(
        self,
        emotion_analysis: EmotionAnalysisResult,
        response_style: ResponseStyle,
        emotional_profile: Optional[EmotionalProfile]
    ) -> List[TherapeuticTechnique]:
        """Select appropriate therapeutic techniques for the response."""
        techniques = []
        
        # Always include active listening
        techniques.append(TherapeuticTechnique.ACTIVE_LISTENING)
        
        # Emotion-specific techniques
        if emotion_analysis.primary_emotion == BasicEmotion.SADNESS:
            techniques.extend([TherapeuticTechnique.VALIDATION, TherapeuticTechnique.REFLECTION])
            if emotion_analysis.emotion_intensity in [EmotionIntensity.HIGH, EmotionIntensity.EXTREME]:
                techniques.append(TherapeuticTechnique.GROUNDING_TECHNIQUE)
        
        elif emotion_analysis.primary_emotion == BasicEmotion.ANGER:
            techniques.extend([TherapeuticTechnique.VALIDATION, TherapeuticTechnique.EMOTION_REGULATION])
            if emotion_analysis.arousal > 0.5:
                techniques.append(TherapeuticTechnique.GROUNDING_TECHNIQUE)
        
        elif emotion_analysis.primary_emotion == BasicEmotion.FEAR:
            techniques.extend([TherapeuticTechnique.GROUNDING_TECHNIQUE, TherapeuticTechnique.MINDFULNESS_GUIDANCE])
        
        elif emotion_analysis.primary_emotion == BasicEmotion.JOY:
            techniques.append(TherapeuticTechnique.REFLECTION)
        
        # Response style-specific techniques
        if response_style == ResponseStyle.COACHING:
            techniques.extend([TherapeuticTechnique.MOTIVATIONAL_INTERVIEWING, TherapeuticTechnique.COGNITIVE_REFRAMING])
        
        elif response_style == ResponseStyle.CALMING:
            techniques.extend([TherapeuticTechnique.MINDFULNESS_GUIDANCE, TherapeuticTechnique.GROUNDING_TECHNIQUE])
        
        elif response_style == ResponseStyle.SUPPORTIVE:
            techniques.extend([TherapeuticTechnique.VALIDATION, TherapeuticTechnique.PSYCHOEDUCATION])
        
        # Adjust based on emotional profile effectiveness data
        if emotional_profile and emotional_profile.coaching_history:
            effective_techniques = emotional_profile.coaching_history.get("effective_techniques", [])
            # Prioritize techniques that have been effective for this user
            techniques = [tech for tech in effective_techniques if tech in [t.value for t in TherapeuticTechnique]] + techniques
        
        # Remove duplicates while preserving order
        seen = set()
        unique_techniques = []
        for tech in techniques:
            if tech not in seen:
                unique_techniques.append(tech)
                seen.add(tech)
        
        return unique_techniques[:4]  # Limit to 4 techniques to avoid overwhelming response
    
    async def _generate_base_response(
        self,
        user_message: str,
        emotion_analysis: EmotionAnalysisResult,
        response_style: ResponseStyle,
        therapeutic_techniques: List[TherapeuticTechnique]
    ) -> str:
        """Generate base response using LLM with emotional intelligence context."""
        
        # Create detailed prompt for emotionally-intelligent response
        emotional_context = f"""
        User's emotional state analysis:
        - Primary emotion: {emotion_analysis.primary_emotion.value}
        - Secondary emotion: {emotion_analysis.secondary_emotion.value if emotion_analysis.secondary_emotion else 'None'}
        - Intensity: {emotion_analysis.emotion_intensity.value}
        - Valence: {emotion_analysis.valence:.2f} (positive/negative scale)
        - Arousal: {emotion_analysis.arousal:.2f} (energy level)
        - Dominance: {emotion_analysis.dominance:.2f} (control/submission)
        
        Response parameters:
        - Style: {response_style.value}
        - Therapeutic techniques to incorporate: {[tech.value for tech in therapeutic_techniques]}
        
        User message: "{user_message}"
        
        Generate a response that:
        1. Acknowledges and validates their emotional state
        2. Uses the specified response style appropriately
        3. Incorporates the therapeutic techniques naturally
        4. Is empathetic and supportive
        5. Provides genuine value and insight
        6. Is conversational and warm, not clinical
        7. Is 1-3 sentences unless more is needed for support
        
        Response:
        """
        
        try:
            # Generate response using LLM
            response = await self.llm_service.generate_response(
                prompt=emotional_context,
                max_tokens=200,
                temperature=0.7,
                context_type="emotional_support"
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {str(e)}")
            return await self._get_fallback_template_response(emotion_analysis, response_style)
    
    async def _get_fallback_template_response(
        self,
        emotion_analysis: EmotionAnalysisResult,
        response_style: ResponseStyle
    ) -> str:
        """Get template-based response as fallback."""
        
        emotion_templates = {
            BasicEmotion.SADNESS: [
                "I can hear that you're going through a difficult time right now. That sounds really challenging.",
                "It's completely understandable that you're feeling this way. Your emotions are valid.",
                "I'm sorry you're experiencing this pain. You don't have to go through it alone."
            ],
            BasicEmotion.ANGER: [
                "I can sense your frustration, and that's completely understandable given what you're dealing with.",
                "Your anger makes perfect sense in this situation. It sounds like something important to you has been affected.",
                "That does sound incredibly frustrating. Your feelings about this are totally valid."
            ],
            BasicEmotion.FEAR: [
                "It's natural to feel anxious about this. Uncertainty can be really uncomfortable.",
                "Fear in situations like this is completely normal. You're not alone in feeling this way.",
                "I hear the worry in what you're sharing. That anxiety makes complete sense."
            ],
            BasicEmotion.JOY: [
                "I can feel your enthusiasm, and that's wonderful! It sounds like something really positive is happening.",
                "Your excitement is contagious! I'm so glad you're experiencing something that brings you joy.",
                "That's fantastic! It's beautiful to see you so happy about this."
            ]
        }
        
        templates = emotion_templates.get(emotion_analysis.primary_emotion, emotion_templates[BasicEmotion.SADNESS])
        return random.choice(templates)
    
    async def _apply_personalization(
        self,
        base_response: str,
        emotional_profile: Optional[EmotionalProfile],
        emotion_analysis: EmotionAnalysisResult
    ) -> Tuple[str, bool]:
        """Apply personalization based on user's emotional profile."""
        if not emotional_profile:
            return base_response, False
        
        personalized_response = base_response
        personalization_applied = False
        
        # Adjust language based on communication preferences
        if emotional_profile.support_preferences:
            comm_style = emotional_profile.support_preferences.get("communication_style", "casual")
            
            if comm_style == "formal":
                # Make response more formal
                personalized_response = personalized_response.replace("you're", "you are")
                personalized_response = personalized_response.replace("I'm", "I am")
                personalization_applied = True
            
            elif comm_style == "casual":
                # Ensure casual tone
                if "." in personalized_response and not any(punct in personalized_response for punct in ["!", "?"]):
                    # Add some warmth to purely declarative statements
                    personalized_response = personalized_response.replace(".", " 💙", 1)
                    personalization_applied = True
        
        # Add personalized coping suggestions based on effective strategies
        if (emotion_analysis.valence < -0.3 and 
            emotional_profile.regulation_effectiveness):
            
            effective_strategies = [
                strategy for strategy, effectiveness 
                in emotional_profile.regulation_effectiveness.items() 
                if effectiveness > 0.7
            ]
            
            if effective_strategies:
                strategy_name = random.choice(effective_strategies)
                strategy_text = self._get_strategy_suggestion_text(strategy_name)
                if strategy_text:
                    personalized_response += f"\n\n{strategy_text}"
                    personalization_applied = True
        
        return personalized_response, personalization_applied
    
    def _get_strategy_suggestion_text(self, strategy_name: str) -> Optional[str]:
        """Get suggestion text for regulation strategy."""
        strategy_texts = {
            "mindfulness": "You mentioned mindfulness has helped you before - might be worth trying a few deep breaths or a quick body scan.",
            "deep_breathing": "I know deep breathing has worked well for you in the past. Even just three slow breaths might help right now.",
            "social_sharing": "You've found it helpful to talk things through before. Is there someone you trust you could share this with?",
            "physical_exercise": "You've mentioned movement helps you process emotions. Maybe a short walk or some stretching could help?",
        }
        
        return strategy_texts.get(strategy_name)
    
    async def _suggest_regulation_strategies(
        self,
        emotion_analysis: EmotionAnalysisResult,
        emotional_profile: Optional[EmotionalProfile]
    ) -> List[EmotionRegulationStrategy]:
        """Suggest appropriate emotion regulation strategies."""
        strategies = []
        
        # Base strategies on current emotional state
        if emotion_analysis.valence < -0.3:  # Negative emotions
            if emotion_analysis.arousal > 0.5:  # High arousal negative (anger, fear)
                strategies.extend([
                    EmotionRegulationStrategy.DEEP_BREATHING,
                    EmotionRegulationStrategy.PROGRESSIVE_MUSCLE_RELAXATION,
                    EmotionRegulationStrategy.MINDFULNESS
                ])
            else:  # Low arousal negative (sadness)
                strategies.extend([
                    EmotionRegulationStrategy.SOCIAL_SHARING,
                    EmotionRegulationStrategy.EXPRESSIVE_WRITING,
                    EmotionRegulationStrategy.PHYSICAL_EXERCISE
                ])
        
        # Add cognitive strategies for appropriate situations
        if emotion_analysis.primary_emotion in [BasicEmotion.ANGER, BasicEmotion.FEAR]:
            strategies.append(EmotionRegulationStrategy.COGNITIVE_REAPPRAISAL)
        
        # Personalize based on user's effective strategies
        if emotional_profile and emotional_profile.primary_regulation_strategies:
            # Prioritize user's preferred strategies
            user_strategies = [
                EmotionRegulationStrategy(strategy) 
                for strategy in emotional_profile.primary_regulation_strategies
                if strategy in [s.value for s in EmotionRegulationStrategy]
            ]
            strategies = user_strategies + [s for s in strategies if s not in user_strategies]
        
        return strategies[:3]  # Return top 3 strategies
    
    async def _generate_followup_suggestions(
        self,
        emotion_analysis: EmotionAnalysisResult,
        response_style: ResponseStyle,
        conversation_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate appropriate followup conversation suggestions."""
        suggestions = []
        
        # Emotion-specific followup suggestions
        if emotion_analysis.primary_emotion == BasicEmotion.SADNESS:
            suggestions.extend([
                "Would you like to tell me more about what's been weighing on you?",
                "How long have you been feeling this way?",
                "Are there things that usually help when you're feeling down?"
            ])
        
        elif emotion_analysis.primary_emotion == BasicEmotion.ANGER:
            suggestions.extend([
                "What would feel most helpful right now - talking through what happened or finding ways to cool down?",
                "Is this something that's been building up, or did it happen suddenly?",
                "How would you most like to handle this situation?"
            ])
        
        elif emotion_analysis.primary_emotion == BasicEmotion.FEAR:
            suggestions.extend([
                "Would it help to break down what you're worried about into smaller pieces?",
                "Have you dealt with something like this before?",
                "What would help you feel more secure about this situation?"
            ])
        
        elif emotion_analysis.primary_emotion == BasicEmotion.JOY:
            suggestions.extend([
                "What made this so special for you?",
                "Who else would you want to share this good news with?",
                "How are you planning to celebrate or build on this?"
            ])
        
        # Generic supportive followups
        suggestions.extend([
            "Is there anything specific I can help you with right now?",
            "What's been most on your mind lately?",
            "How are you taking care of yourself through this?"
        ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def _calculate_emotional_intent(
        self,
        response_style: ResponseStyle,
        emotion_analysis: EmotionAnalysisResult
    ) -> Dict[str, float]:
        """Calculate the intended emotional impact of the response."""
        intent = {}
        
        # Base intent on response style
        style_intents = {
            ResponseStyle.EMPATHETIC: {"understanding": 0.9, "connection": 0.8, "validation": 0.7},
            ResponseStyle.SUPPORTIVE: {"support": 0.9, "hope": 0.7, "strength": 0.6},
            ResponseStyle.VALIDATING: {"validation": 1.0, "acceptance": 0.8, "safety": 0.7},
            ResponseStyle.COACHING: {"growth": 0.8, "empowerment": 0.7, "clarity": 0.6},
            ResponseStyle.CALMING: {"peace": 0.9, "safety": 0.8, "grounding": 0.7},
            ResponseStyle.ENERGIZING: {"motivation": 0.8, "joy": 0.7, "enthusiasm": 0.6},
            ResponseStyle.CRISIS_INTERVENTION: {"safety": 1.0, "hope": 0.9, "connection": 0.8}
        }
        
        intent = style_intents.get(response_style, {"support": 0.5, "understanding": 0.5})
        
        # Adjust based on user's current emotional state
        if emotion_analysis.valence < -0.5:
            intent["hope"] = intent.get("hope", 0.0) + 0.2
            intent["support"] = intent.get("support", 0.0) + 0.3
        
        # Normalize values to 0-1 range
        for key, value in intent.items():
            intent[key] = max(0.0, min(1.0, value))
        
        return intent
    
    def _calculate_confidence_score(
        self,
        emotion_analysis: EmotionAnalysisResult,
        personalization_applied: bool
    ) -> float:
        """Calculate confidence score for the response."""
        base_confidence = 0.7
        
        # Boost confidence for high-quality emotion analysis
        if emotion_analysis.analysis_quality > 0.8:
            base_confidence += 0.15
        elif emotion_analysis.analysis_quality > 0.6:
            base_confidence += 0.1
        
        # Boost confidence for personalization
        if personalization_applied:
            base_confidence += 0.1
        
        # Boost confidence for multi-modal emotion detection
        if len(emotion_analysis.detection_modalities) > 1:
            base_confidence += 0.05 * len(emotion_analysis.detection_modalities)
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _create_fallback_response(
        self,
        user_message: str,
        start_time: datetime
    ) -> EmotionalResponse:
        """Create fallback response for error cases."""
        fallback_text = ("I want to acknowledge what you've shared with me. "
                        "While I'm having some technical difficulties right now, "
                        "I want you to know that I'm here and listening.")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return EmotionalResponse(
            response_text=fallback_text,
            response_style=ResponseStyle.EMPATHETIC,
            therapeutic_techniques=[TherapeuticTechnique.ACTIVE_LISTENING],
            emotional_intent={"support": 0.5, "understanding": 0.5},
            regulation_strategies=[],
            crisis_level=CrisisLevel.NONE,
            confidence_score=0.3,
            personalization_applied=False,
            followup_suggestions=["Is there anything specific I can help you with?"],
            processing_time_ms=int(processing_time)
        )
    
    async def _update_response_metrics(self, response: EmotionalResponse):
        """Update response generation metrics."""
        self.response_metrics["total_responses"] += 1
        
        # Update average processing time
        current_avg = self.response_metrics["avg_processing_time"]
        total = self.response_metrics["total_responses"]
        self.response_metrics["avg_processing_time"] = (
            (current_avg * (total - 1) + response.processing_time_ms) / total
        )
        
        # Track style usage
        style_key = response.response_style.value
        self.response_metrics["style_usage"][style_key] = self.response_metrics["style_usage"].get(style_key, 0) + 1
    
    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load therapeutic response templates."""
        return {
            "validation": {
                "sadness": [
                    "I can hear how difficult this is for you, and your feelings are completely valid.",
                    "It makes perfect sense that you'd feel this way given what you're going through.",
                    "Anyone would struggle with what you're experiencing right now."
                ],
                "anger": [
                    "Your frustration is completely understandable in this situation.",
                    "That would be infuriating - your anger makes total sense.",
                    "I can feel how upset you are, and that's entirely justified."
                ]
            },
            "support": {
                "general": [
                    "You don't have to go through this alone.",
                    "I'm here with you through this difficult time.",
                    "You're showing incredible strength by reaching out."
                ]
            }
        }
    
    def _load_crisis_resources(self) -> Dict[str, str]:
        """Load crisis intervention resources."""
        return {
            "suicide_prevention": "National Suicide Prevention Lifeline: 988 or 1-800-273-8255",
            "crisis_text": "Crisis Text Line: Text HOME to 741741",
            "emergency": "If you're in immediate danger, please call 911",
            "international": "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
        }
    
    def _load_regulation_strategies(self) -> Dict[EmotionRegulationStrategy, Dict[str, str]]:
        """Load emotion regulation strategy descriptions and techniques."""
        return {
            EmotionRegulationStrategy.DEEP_BREATHING: {
                "description": "Slow, deep breathing to activate the parasympathetic nervous system",
                "technique": "Try the 4-7-8 technique: breathe in for 4, hold for 7, out for 8"
            },
            EmotionRegulationStrategy.MINDFULNESS: {
                "description": "Present-moment awareness to observe emotions without judgment",
                "technique": "Notice 5 things you can see, 4 you can hear, 3 you can touch, 2 you can smell, 1 you can taste"
            },
            EmotionRegulationStrategy.COGNITIVE_REAPPRAISAL: {
                "description": "Reframing thoughts to change emotional response",
                "technique": "Ask yourself: Is this thought helpful? What would I tell a friend in this situation?"
            }
        }
    
    async def _get_crisis_resources_text(self) -> str:
        """Get formatted crisis resources text."""
        return """🆘 Immediate Support Resources:
        
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741  
• Emergency Services: 911
• International Crisis Lines: https://findahelpline.com

You deserve support, and trained professionals are available 24/7 to help. Please don't hesitate to reach out to them."""
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get response system performance metrics."""
        total_responses = self.response_metrics["total_responses"]
        
        return {
            "total_responses_generated": total_responses,
            "avg_processing_time_ms": self.response_metrics["avg_processing_time"],
            "style_usage_distribution": self.response_metrics["style_usage"],
            "crisis_interventions_triggered": self.response_metrics["crisis_interventions"],
            "crisis_intervention_rate": (
                self.response_metrics["crisis_interventions"] / total_responses 
                if total_responses > 0 else 0.0
            ),
            "system_status": "operational"
        }


# Create global emotional responder instance
emotional_responder = AdvancedEmotionalResponder()


async def generate_empathetic_response(
    user_message: str,
    user_id: str,
    conversation_context: Optional[Dict[str, Any]] = None,
    emotional_profile: Optional[EmotionalProfile] = None
) -> EmotionalResponse:
    """
    Convenience function to generate an emotionally-intelligent response.
    
    Args:
        user_message: The user's message to respond to
        user_id: User identifier
        conversation_context: Optional conversation context
        emotional_profile: Optional user emotional profile
        
    Returns:
        Emotionally intelligent response with therapeutic elements
    """
    # First, analyze the emotion in the user's message
    emotion_analysis = await emotion_detector.analyze_emotion(
        text_content=user_message,
        conversation_context=conversation_context,
        user_id=user_id
    )
    
    # Generate emotionally-aware response
    return await emotional_responder.generate_emotional_response(
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        user_id=user_id,
        conversation_context=conversation_context,
        emotional_profile=emotional_profile
    )