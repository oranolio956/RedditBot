"""
Advanced Multi-Modal Emotion Detection Service

Revolutionary emotion recognition system that combines:
- Text sentiment and emotion analysis using transformers
- Voice prosodic feature analysis  
- Behavioral pattern recognition from typing/interaction patterns
- Contextual emotion inference from conversation flow
- Temporal emotion tracking and pattern recognition

Scientifically grounded using:
- BERT/RoBERTa for contextual text emotion analysis
- Prosodic feature extraction for voice emotion detection
- Behavioral biometrics for emotional state inference
- Ensemble methods for robust multi-modal fusion
"""

import asyncio
import json
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict
import math
import re
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, AutoModel
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch.nn.functional as F

from app.models.emotional_intelligence import (
    EmotionalProfile, EmotionReading, BasicEmotion, 
    EmotionIntensity, DetectionModality, EmotionDimension
)
from app.models.user import User
from app.models.conversation import Message, Conversation
from app.database.connection import get_db_session as get_db
from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class EmotionAnalysisResult:
    """Result of multi-modal emotion analysis."""
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # -1 (low energy) to 1 (high energy) 
    dominance: float  # -1 (submissive) to 1 (dominant)
    primary_emotion: BasicEmotion
    secondary_emotion: Optional[BasicEmotion]
    emotion_intensity: EmotionIntensity
    confidence_scores: Dict[str, float]
    detection_modalities: List[DetectionModality]
    plutchik_scores: Dict[str, float]
    processing_time_ms: int
    analysis_quality: float


class AdvancedEmotionDetector:
    """
    Revolutionary multi-modal emotion detection engine.
    
    Combines multiple AI models and analysis techniques for
    unprecedented accuracy in understanding human emotions.
    """
    
    def __init__(self):
        """Initialize the emotion detection system."""
        self.text_analyzer = TextEmotionAnalyzer()
        self.voice_analyzer = VoiceProsodyAnalyzer()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()
        self.contextual_analyzer = ContextualEmotionAnalyzer()
        self.temporal_analyzer = TemporalEmotionAnalyzer()
        self.ensemble_fusion = EmotionEnsembleFusion()
        
        # Cache for frequently accessed models
        self.model_cache = {}
        self.scaler_cache = {}
        
        # Performance tracking
        self.analysis_metrics = {
            "total_analyses": 0,
            "avg_processing_time": 0.0,
            "modality_usage": defaultdict(int),
            "accuracy_scores": []
        }
    
    async def analyze_emotion(
        self,
        text_content: Optional[str] = None,
        voice_features: Optional[Dict[str, Any]] = None,
        behavioral_data: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> EmotionAnalysisResult:
        """
        Perform comprehensive multi-modal emotion analysis.
        
        Args:
            text_content: Text message to analyze
            voice_features: Voice/audio features if available
            behavioral_data: User behavioral patterns
            conversation_context: Current conversation context
            user_id: User ID for personalized analysis
            message_id: Message ID for context
            
        Returns:
            Complete emotion analysis result
        """
        start_time = datetime.now()
        
        try:
            # Collect analysis results from each modality
            modality_results = {}
            used_modalities = []
            
            # Text emotion analysis
            if text_content:
                text_result = await self.text_analyzer.analyze_text_emotion(
                    text_content, conversation_context
                )
                modality_results['text'] = text_result
                used_modalities.append(DetectionModality.TEXT_ANALYSIS)
            
            # Voice prosody analysis
            if voice_features:
                voice_result = await self.voice_analyzer.analyze_voice_emotion(
                    voice_features, conversation_context
                )
                modality_results['voice'] = voice_result
                used_modalities.append(DetectionModality.VOICE_PROSODY)
            
            # Behavioral pattern analysis
            if behavioral_data:
                behavioral_result = await self.behavioral_analyzer.analyze_behavioral_emotion(
                    behavioral_data, user_id
                )
                modality_results['behavioral'] = behavioral_result
                used_modalities.append(DetectionModality.BEHAVIORAL_PATTERN)
            
            # Contextual emotion inference
            if conversation_context:
                contextual_result = await self.contextual_analyzer.analyze_contextual_emotion(
                    conversation_context, user_id
                )
                modality_results['contextual'] = contextual_result
                used_modalities.append(DetectionModality.CONTEXTUAL_INFERENCE)
            
            # Temporal pattern analysis (if user history available)
            if user_id:
                temporal_result = await self.temporal_analyzer.analyze_temporal_patterns(
                    user_id, text_content
                )
                modality_results['temporal'] = temporal_result
            
            # Ensemble fusion to combine all modalities
            final_result = await self.ensemble_fusion.fuse_emotions(
                modality_results, used_modalities
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            await self._update_metrics(processing_time, used_modalities, final_result)
            
            return EmotionAnalysisResult(
                valence=final_result['valence'],
                arousal=final_result['arousal'],
                dominance=final_result['dominance'],
                primary_emotion=BasicEmotion(final_result['primary_emotion']),
                secondary_emotion=BasicEmotion(final_result['secondary_emotion']) if final_result.get('secondary_emotion') else None,
                emotion_intensity=EmotionIntensity(final_result['intensity']),
                confidence_scores=final_result['confidence_scores'],
                detection_modalities=used_modalities,
                plutchik_scores=final_result['plutchik_scores'],
                processing_time_ms=int(processing_time),
                analysis_quality=final_result['quality_score']
            )
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            # Return neutral emotion on error
            return self._create_neutral_result(used_modalities, int((datetime.now() - start_time).total_seconds() * 1000))
    
    async def _update_metrics(self, processing_time: float, modalities: List[DetectionModality], result: Dict):
        """Update performance metrics."""
        self.analysis_metrics["total_analyses"] += 1
        
        # Update average processing time
        current_avg = self.analysis_metrics["avg_processing_time"]
        total = self.analysis_metrics["total_analyses"]
        self.analysis_metrics["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total
        
        # Update modality usage
        for modality in modalities:
            self.analysis_metrics["modality_usage"][modality.value] += 1
        
        # Track quality scores
        if "quality_score" in result:
            self.analysis_metrics["accuracy_scores"].append(result["quality_score"])
            # Keep only last 100 scores for memory efficiency
            if len(self.analysis_metrics["accuracy_scores"]) > 100:
                self.analysis_metrics["accuracy_scores"] = self.analysis_metrics["accuracy_scores"][-100:]
    
    def _create_neutral_result(self, modalities: List[DetectionModality], processing_time: int) -> EmotionAnalysisResult:
        """Create neutral emotion result for error cases."""
        return EmotionAnalysisResult(
            valence=0.0,
            arousal=0.0,
            dominance=0.0,
            primary_emotion=BasicEmotion.TRUST,  # Neutral baseline
            secondary_emotion=None,
            emotion_intensity=EmotionIntensity.MINIMAL,
            confidence_scores={"overall": 0.1},
            detection_modalities=modalities,
            plutchik_scores={emotion.value: 0.125 for emotion in BasicEmotion},  # Equal distribution
            processing_time_ms=processing_time,
            analysis_quality=0.1
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        avg_quality = np.mean(self.analysis_metrics["accuracy_scores"]) if self.analysis_metrics["accuracy_scores"] else 0.0
        
        return {
            "total_analyses": self.analysis_metrics["total_analyses"],
            "avg_processing_time_ms": self.analysis_metrics["avg_processing_time"],
            "modality_usage": dict(self.analysis_metrics["modality_usage"]),
            "avg_analysis_quality": avg_quality,
            "system_status": "operational" if avg_quality > 0.6 else "degraded"
        }


class TextEmotionAnalyzer:
    """Advanced text-based emotion analysis using transformer models."""
    
    def __init__(self):
        self.emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Load models
        self.emotion_classifier = None
        self.sentiment_classifier = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models asynchronously
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize ML models asynchronously."""
        try:
            # Emotion classification model
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                tokenizer=self.emotion_model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Sentiment analysis model
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                tokenizer=self.sentiment_model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("Text emotion models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize text models: {str(e)}")
    
    async def analyze_text_emotion(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze emotions in text using advanced NLP models."""
        if not self.emotion_classifier:
            await self._initialize_models()
        
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Get emotion scores
            emotion_results = self.emotion_classifier(cleaned_text)
            sentiment_results = self.sentiment_classifier(cleaned_text)
            
            # Convert to standard format
            emotion_scores = {}
            for result in emotion_results[0]:
                emotion_label = result['label'].lower()
                emotion_scores[emotion_label] = result['score']
            
            # Map to Plutchik's 8 emotions
            plutchik_scores = self._map_to_plutchik_emotions(emotion_scores)
            
            # Calculate dimensional scores
            valence = self._calculate_valence_from_sentiment(sentiment_results[0])
            arousal = self._calculate_arousal_from_emotions(emotion_scores)
            dominance = self._calculate_dominance_from_text(cleaned_text, emotion_scores)
            
            # Determine primary and secondary emotions
            primary_emotion = max(plutchik_scores, key=plutchik_scores.get)
            secondary_candidates = sorted(plutchik_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_emotion = secondary_candidates[1][0] if len(secondary_candidates) > 1 and secondary_candidates[1][1] > 0.3 else None
            
            # Calculate intensity
            max_score = max(plutchik_scores.values())
            intensity = self._score_to_intensity(max_score)
            
            # Apply contextual adjustments
            if context:
                valence, arousal, dominance = self._apply_contextual_adjustments(
                    valence, arousal, dominance, context
                )
            
            return {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "primary_emotion": primary_emotion,
                "secondary_emotion": secondary_emotion,
                "intensity": intensity,
                "plutchik_scores": plutchik_scores,
                "confidence": max_score,
                "raw_emotion_scores": emotion_scores,
                "raw_sentiment_scores": {r['label']: r['score'] for r in sentiment_results[0]}
            }
            
        except Exception as e:
            logger.error(f"Text emotion analysis failed: {str(e)}")
            return self._get_neutral_text_result()
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common abbreviations and emoticons
        emotion_patterns = {
            r':\)': ' happy ',
            r':\(': ' sad ',
            r':D': ' very happy ',
            r':\|': ' neutral ',
            r':\*': ' love ',
            r'<3': ' love ',
            r'>:\(': ' angry ',
        }
        
        for pattern, replacement in emotion_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _map_to_plutchik_emotions(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Map detected emotions to Plutchik's 8 basic emotions."""
        # Mapping from common emotion labels to Plutchik's emotions
        emotion_mapping = {
            'joy': 'joy',
            'happiness': 'joy',
            'love': 'joy',
            'optimism': 'joy',
            'sadness': 'sadness',
            'grief': 'sadness',
            'disappointment': 'sadness',
            'anger': 'anger',
            'annoyance': 'anger',
            'rage': 'anger',
            'fear': 'fear',
            'anxiety': 'fear',
            'nervousness': 'fear',
            'trust': 'trust',
            'approval': 'trust',
            'admiration': 'trust',
            'disgust': 'disgust',
            'disapproval': 'disgust',
            'surprise': 'surprise',
            'amazement': 'surprise',
            'confusion': 'surprise',
            'anticipation': 'anticipation',
            'excitement': 'anticipation',
            'curiosity': 'anticipation',
        }
        
        plutchik_scores = {emotion.value: 0.0 for emotion in BasicEmotion}
        
        for detected_emotion, score in emotion_scores.items():
            plutchik_emotion = emotion_mapping.get(detected_emotion, 'trust')  # Default to trust for unknown
            plutchik_scores[plutchik_emotion] += score
        
        # Normalize scores to sum to 1
        total_score = sum(plutchik_scores.values())
        if total_score > 0:
            plutchik_scores = {k: v / total_score for k, v in plutchik_scores.items()}
        
        return plutchik_scores
    
    def _calculate_valence_from_sentiment(self, sentiment_results: List[Dict]) -> float:
        """Calculate valence from sentiment analysis results."""
        positive_score = 0.0
        negative_score = 0.0
        
        for result in sentiment_results:
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                positive_score = score
            elif 'negative' in label:
                negative_score = score
        
        # Convert to valence scale (-1 to 1)
        return positive_score - negative_score
    
    def _calculate_arousal_from_emotions(self, emotion_scores: Dict[str, float]) -> float:
        """Calculate arousal from emotion scores."""
        # High arousal emotions
        high_arousal_emotions = ['anger', 'fear', 'joy', 'surprise', 'excitement', 'rage', 'amazement']
        # Low arousal emotions  
        low_arousal_emotions = ['sadness', 'trust', 'disgust', 'boredom', 'contentment']
        
        high_arousal_score = sum(emotion_scores.get(emotion, 0.0) for emotion in high_arousal_emotions)
        low_arousal_score = sum(emotion_scores.get(emotion, 0.0) for emotion in low_arousal_emotions)
        
        # Convert to arousal scale (-1 to 1)
        return (high_arousal_score - low_arousal_score) * 2.0 - 1.0
    
    def _calculate_dominance_from_text(self, text: str, emotion_scores: Dict[str, float]) -> float:
        """Calculate dominance from text features and emotions."""
        # Dominant language patterns
        dominant_patterns = [
            r'\b(will|shall|must|should|need to|have to)\b',
            r'\b(command|order|demand|insist)\b',
            r'\b(I|me|my|mine)\b',  # First person assertiveness
            r'!+',  # Exclamation marks
        ]
        
        # Submissive language patterns
        submissive_patterns = [
            r'\b(please|sorry|excuse me|pardon)\b',
            r'\b(maybe|perhaps|might|could|would)\b',
            r'\?+',  # Question marks
            r'\b(you|your|yours)\b',  # Other-focused
        ]
        
        dominant_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in dominant_patterns)
        submissive_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in submissive_patterns)
        
        # Add emotion-based dominance
        dominant_emotions = emotion_scores.get('anger', 0.0) + emotion_scores.get('disgust', 0.0)
        submissive_emotions = emotion_scores.get('fear', 0.0) + emotion_scores.get('sadness', 0.0)
        
        # Normalize and combine
        text_dominance = (dominant_score - submissive_score) / max(len(text.split()), 1)
        emotion_dominance = dominant_emotions - submissive_emotions
        
        # Weighted combination
        return (0.6 * text_dominance + 0.4 * emotion_dominance)
    
    def _apply_contextual_adjustments(self, valence: float, arousal: float, dominance: float, context: Dict) -> Tuple[float, float, float]:
        """Apply contextual adjustments to dimensional scores."""
        # Time of day adjustments
        if context.get('time_of_day'):
            hour = context['time_of_day']
            if 22 <= hour or hour <= 6:  # Night hours
                arousal *= 0.8  # Lower arousal at night
                valence *= 0.9  # Slight mood dampening
        
        # Social context adjustments
        if context.get('is_group_conversation'):
            dominance *= 0.9  # Slightly less dominant in groups
            
        if context.get('conversation_length', 0) > 10:  # Long conversations
            arousal *= 0.95  # Slight fatigue effect
        
        # Clamp values to valid range
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))
        
        return valence, arousal, dominance
    
    def _score_to_intensity(self, score: float) -> str:
        """Convert confidence score to emotion intensity level."""
        if score >= 0.8:
            return EmotionIntensity.EXTREME.value
        elif score >= 0.6:
            return EmotionIntensity.HIGH.value
        elif score >= 0.4:
            return EmotionIntensity.MODERATE.value
        elif score >= 0.2:
            return EmotionIntensity.LOW.value
        else:
            return EmotionIntensity.MINIMAL.value
    
    def _get_neutral_text_result(self) -> Dict[str, Any]:
        """Get neutral result for error cases."""
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "primary_emotion": "trust",
            "secondary_emotion": None,
            "intensity": "minimal",
            "plutchik_scores": {emotion.value: 0.125 for emotion in BasicEmotion},
            "confidence": 0.1,
            "raw_emotion_scores": {},
            "raw_sentiment_scores": {}
        }


class VoiceProsodyAnalyzer:
    """Voice prosody analysis for emotion detection."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = 1024
        self.hop_length = 512
    
    async def analyze_voice_emotion(self, voice_features: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze emotion from voice prosodic features."""
        try:
            # Extract key prosodic features
            pitch_mean = voice_features.get('pitch_mean', 0.0)
            pitch_std = voice_features.get('pitch_std', 0.0)
            intensity_mean = voice_features.get('intensity_mean', 0.0)
            speaking_rate = voice_features.get('speaking_rate', 0.0)
            pause_duration = voice_features.get('pause_duration', 0.0)
            
            # Map prosodic features to emotions
            valence = self._calculate_valence_from_prosody(pitch_mean, intensity_mean, speaking_rate)
            arousal = self._calculate_arousal_from_prosody(pitch_std, intensity_mean, speaking_rate)
            dominance = self._calculate_dominance_from_prosody(intensity_mean, pitch_mean, pause_duration)
            
            # Determine primary emotion based on prosodic patterns
            primary_emotion = self._classify_emotion_from_prosody(valence, arousal, dominance)
            
            return {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "primary_emotion": primary_emotion,
                "confidence": 0.7,  # Prosody analysis confidence
                "prosodic_features": voice_features
            }
            
        except Exception as e:
            logger.error(f"Voice prosody analysis failed: {str(e)}")
            return self._get_neutral_voice_result()
    
    def _calculate_valence_from_prosody(self, pitch_mean: float, intensity_mean: float, speaking_rate: float) -> float:
        """Calculate valence from prosodic features."""
        # Higher pitch and intensity often correlate with positive emotions
        # Faster speaking rate can indicate excitement or anxiety
        valence_score = 0.0
        
        if pitch_mean > 0:  # Normalized pitch
            valence_score += 0.3 * min(pitch_mean, 1.0)
        
        if intensity_mean > 0:  # Normalized intensity
            valence_score += 0.2 * min(intensity_mean, 1.0)
        
        if speaking_rate > 1.0:  # Normal speaking rate = 1.0
            valence_score += 0.1 * min(speaking_rate - 1.0, 1.0)
        
        return max(-1.0, min(1.0, valence_score))
    
    def _calculate_arousal_from_prosody(self, pitch_std: float, intensity_mean: float, speaking_rate: float) -> float:
        """Calculate arousal from prosodic features."""
        # Higher pitch variation, intensity, and speaking rate indicate high arousal
        arousal_score = 0.0
        
        if pitch_std > 0:
            arousal_score += 0.4 * min(pitch_std, 1.0)
        
        if intensity_mean > 0:
            arousal_score += 0.3 * min(intensity_mean, 1.0)
        
        if speaking_rate > 1.0:
            arousal_score += 0.3 * min(speaking_rate - 1.0, 2.0)
        
        return max(-1.0, min(1.0, arousal_score * 2.0 - 1.0))
    
    def _calculate_dominance_from_prosody(self, intensity_mean: float, pitch_mean: float, pause_duration: float) -> float:
        """Calculate dominance from prosodic features."""
        # Higher intensity and controlled pauses indicate dominance
        # Very high pitch might indicate submission
        dominance_score = 0.0
        
        if intensity_mean > 0:
            dominance_score += 0.4 * min(intensity_mean, 1.0)
        
        if pitch_mean > 0.8:  # Very high pitch might indicate submission
            dominance_score -= 0.2
        
        if pause_duration < 0.5:  # Controlled, brief pauses
            dominance_score += 0.2
        
        return max(-1.0, min(1.0, dominance_score))
    
    def _classify_emotion_from_prosody(self, valence: float, arousal: float, dominance: float) -> str:
        """Classify basic emotion from dimensional scores."""
        # Use Russell's circumplex model for emotion classification
        if valence > 0.2 and arousal > 0.2:
            return BasicEmotion.JOY.value
        elif valence < -0.2 and arousal > 0.2:
            return BasicEmotion.ANGER.value if dominance > 0 else BasicEmotion.FEAR.value
        elif valence < -0.2 and arousal < -0.2:
            return BasicEmotion.SADNESS.value
        elif valence > 0.2 and arousal < -0.2:
            return BasicEmotion.TRUST.value
        else:
            return BasicEmotion.TRUST.value  # Neutral default
    
    def _get_neutral_voice_result(self) -> Dict[str, Any]:
        """Get neutral result for error cases."""
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "primary_emotion": "trust",
            "confidence": 0.1,
            "prosodic_features": {}
        }


class BehavioralPatternAnalyzer:
    """Analyzes behavioral patterns to infer emotional states."""
    
    async def analyze_behavioral_emotion(self, behavioral_data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze emotion from behavioral patterns."""
        try:
            # Extract behavioral indicators
            typing_speed = behavioral_data.get('typing_speed', 0.0)
            pause_patterns = behavioral_data.get('pause_patterns', [])
            message_length_variance = behavioral_data.get('message_length_variance', 0.0)
            response_time = behavioral_data.get('response_time_seconds', 0.0)
            
            # Calculate emotional dimensions from behavior
            arousal = self._calculate_arousal_from_behavior(typing_speed, pause_patterns)
            valence = self._calculate_valence_from_behavior(message_length_variance, response_time)
            dominance = self._calculate_dominance_from_behavior(typing_speed, response_time)
            
            # Infer primary emotion
            primary_emotion = self._infer_emotion_from_behavior(arousal, valence, dominance)
            
            return {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "primary_emotion": primary_emotion,
                "confidence": 0.6,  # Behavioral inference confidence
                "behavioral_indicators": behavioral_data
            }
            
        except Exception as e:
            logger.error(f"Behavioral emotion analysis failed: {str(e)}")
            return self._get_neutral_behavioral_result()
    
    def _calculate_arousal_from_behavior(self, typing_speed: float, pause_patterns: List[float]) -> float:
        """Calculate arousal from typing and pause patterns."""
        arousal = 0.0
        
        # Fast typing indicates high arousal
        if typing_speed > 1.2:  # Normalized typing speed
            arousal += 0.4
        elif typing_speed < 0.8:
            arousal -= 0.2
        
        # Irregular pauses indicate emotional disturbance (high arousal)
        if pause_patterns:
            pause_variance = np.var(pause_patterns) if len(pause_patterns) > 1 else 0
            if pause_variance > 0.5:
                arousal += 0.3
        
        return max(-1.0, min(1.0, arousal))
    
    def _calculate_valence_from_behavior(self, message_length_variance: float, response_time: float) -> float:
        """Calculate valence from message patterns."""
        valence = 0.0
        
        # Consistent message lengths might indicate positive state
        if message_length_variance < 0.3:
            valence += 0.2
        
        # Very quick responses might indicate engagement (positive)
        if 1.0 <= response_time <= 5.0:  # Optimal response time
            valence += 0.3
        elif response_time > 30.0:  # Very slow responses might indicate negative state
            valence -= 0.2
        
        return max(-1.0, min(1.0, valence))
    
    def _calculate_dominance_from_behavior(self, typing_speed: float, response_time: float) -> float:
        """Calculate dominance from behavioral assertiveness indicators."""
        dominance = 0.0
        
        # Fast, confident typing indicates dominance
        if typing_speed > 1.1:
            dominance += 0.3
        
        # Quick responses indicate confidence/dominance
        if response_time < 3.0:
            dominance += 0.2
        
        return max(-1.0, min(1.0, dominance))
    
    def _infer_emotion_from_behavior(self, arousal: float, valence: float, dominance: float) -> str:
        """Infer emotion from behavioral dimensional scores."""
        if arousal > 0.2 and valence > 0.2:
            return BasicEmotion.JOY.value
        elif arousal > 0.2 and valence < -0.2:
            return BasicEmotion.ANGER.value if dominance > 0 else BasicEmotion.FEAR.value
        elif arousal < -0.2 and valence < -0.2:
            return BasicEmotion.SADNESS.value
        else:
            return BasicEmotion.TRUST.value
    
    def _get_neutral_behavioral_result(self) -> Dict[str, Any]:
        """Get neutral result for error cases."""
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "primary_emotion": "trust",
            "confidence": 0.1,
            "behavioral_indicators": {}
        }


class ContextualEmotionAnalyzer:
    """Analyzes conversational context to infer emotions."""
    
    async def analyze_contextual_emotion(self, context: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze emotion from conversational context."""
        try:
            # Extract contextual factors
            conversation_topic = context.get('topic', '')
            social_context = context.get('social_context', 'private')
            time_context = context.get('time_of_day', 12)
            conversation_length = context.get('message_count', 0)
            
            # Calculate contextual emotional influence
            valence = self._calculate_contextual_valence(conversation_topic, social_context)
            arousal = self._calculate_contextual_arousal(conversation_length, time_context)
            dominance = self._calculate_contextual_dominance(social_context)
            
            # Determine contextually influenced emotion
            primary_emotion = self._infer_contextual_emotion(valence, arousal, context)
            
            return {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "primary_emotion": primary_emotion,
                "confidence": 0.5,  # Context inference confidence
                "context_factors": context
            }
            
        except Exception as e:
            logger.error(f"Contextual emotion analysis failed: {str(e)}")
            return self._get_neutral_contextual_result()
    
    def _calculate_contextual_valence(self, topic: str, social_context: str) -> float:
        """Calculate valence from conversation context."""
        valence = 0.0
        
        # Topic-based valence
        positive_topics = ['celebration', 'achievement', 'good news', 'success', 'love', 'friendship']
        negative_topics = ['problem', 'complaint', 'sad', 'loss', 'failure', 'conflict']
        
        topic_lower = topic.lower()
        if any(pos_topic in topic_lower for pos_topic in positive_topics):
            valence += 0.4
        elif any(neg_topic in topic_lower for neg_topic in negative_topics):
            valence -= 0.4
        
        # Social context influence
        if social_context == 'group':
            valence += 0.1  # Group conversations tend to be more positive
        
        return max(-1.0, min(1.0, valence))
    
    def _calculate_contextual_arousal(self, conversation_length: int, time_of_day: int) -> float:
        """Calculate arousal from temporal and conversational context."""
        arousal = 0.0
        
        # Long conversations might indicate high engagement (arousal)
        if conversation_length > 20:
            arousal += 0.3
        
        # Time of day influence
        if 9 <= time_of_day <= 17:  # Workday hours
            arousal += 0.1
        elif 18 <= time_of_day <= 22:  # Evening social hours
            arousal += 0.2
        elif 22 <= time_of_day or time_of_day <= 6:  # Night hours
            arousal -= 0.3
        
        return max(-1.0, min(1.0, arousal))
    
    def _calculate_contextual_dominance(self, social_context: str) -> float:
        """Calculate dominance from social context."""
        if social_context == 'private':
            return 0.1  # Slightly more open in private
        elif social_context == 'group':
            return -0.1  # Slightly less dominant in groups
        else:
            return 0.0
    
    def _infer_contextual_emotion(self, valence: float, arousal: float, context: Dict) -> str:
        """Infer emotion from contextual factors."""
        # Use context to bias emotion classification
        topic = context.get('topic', '').lower()
        
        if 'celebration' in topic or 'achievement' in topic:
            return BasicEmotion.JOY.value
        elif 'problem' in topic or 'issue' in topic:
            return BasicEmotion.SADNESS.value if valence < 0 else BasicEmotion.ANGER.value
        elif 'surprise' in topic or 'news' in topic:
            return BasicEmotion.SURPRISE.value
        else:
            # Fall back to dimensional classification
            if valence > 0.2:
                return BasicEmotion.JOY.value
            elif valence < -0.2 and arousal > 0:
                return BasicEmotion.ANGER.value
            elif valence < -0.2:
                return BasicEmotion.SADNESS.value
            else:
                return BasicEmotion.TRUST.value
    
    def _get_neutral_contextual_result(self) -> Dict[str, Any]:
        """Get neutral result for error cases."""
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "primary_emotion": "trust",
            "confidence": 0.1,
            "context_factors": {}
        }


class TemporalEmotionAnalyzer:
    """Analyzes temporal emotional patterns."""
    
    async def analyze_temporal_patterns(self, user_id: str, current_text: Optional[str] = None) -> Dict[str, Any]:
        """Analyze emotional patterns over time for this user."""
        try:
            # This would query the database for historical emotion data
            # For now, return a basic temporal analysis
            
            return {
                "temporal_trend": "stable",
                "pattern_confidence": 0.3,
                "historical_baseline": {
                    "valence": 0.0,
                    "arousal": 0.0,
                    "dominance": 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Temporal emotion analysis failed: {str(e)}")
            return {
                "temporal_trend": "unknown",
                "pattern_confidence": 0.1,
                "historical_baseline": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            }


class EmotionEnsembleFusion:
    """Fuses emotion predictions from multiple modalities."""
    
    async def fuse_emotions(self, modality_results: Dict[str, Dict], used_modalities: List[DetectionModality]) -> Dict[str, Any]:
        """Fuse emotion predictions from multiple modalities using weighted ensemble."""
        try:
            # Modality confidence weights (can be learned/optimized over time)
            modality_weights = {
                'text': 0.4,
                'voice': 0.3,
                'behavioral': 0.2,
                'contextual': 0.1,
                'temporal': 0.05
            }
            
            # Initialize aggregated scores
            fused_valence = 0.0
            fused_arousal = 0.0
            fused_dominance = 0.0
            fused_plutchik = {emotion.value: 0.0 for emotion in BasicEmotion}
            total_weight = 0.0
            
            # Aggregate predictions from each modality
            for modality_name, result in modality_results.items():
                weight = modality_weights.get(modality_name, 0.1)
                confidence = result.get('confidence', 0.5)
                adjusted_weight = weight * confidence
                
                fused_valence += result.get('valence', 0.0) * adjusted_weight
                fused_arousal += result.get('arousal', 0.0) * adjusted_weight
                fused_dominance += result.get('dominance', 0.0) * adjusted_weight
                
                # Aggregate Plutchik scores if available
                if 'plutchik_scores' in result:
                    for emotion, score in result['plutchik_scores'].items():
                        fused_plutchik[emotion] += score * adjusted_weight
                
                total_weight += adjusted_weight
            
            # Normalize by total weight
            if total_weight > 0:
                fused_valence /= total_weight
                fused_arousal /= total_weight
                fused_dominance /= total_weight
                fused_plutchik = {k: v / total_weight for k, v in fused_plutchik.items()}
            
            # Determine final emotion classification
            primary_emotion = max(fused_plutchik, key=fused_plutchik.get)
            sorted_emotions = sorted(fused_plutchik.items(), key=lambda x: x[1], reverse=True)
            secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0.2 else None
            
            # Calculate intensity from maximum emotion score
            max_emotion_score = max(fused_plutchik.values())
            if max_emotion_score >= 0.7:
                intensity = EmotionIntensity.HIGH.value
            elif max_emotion_score >= 0.5:
                intensity = EmotionIntensity.MODERATE.value
            elif max_emotion_score >= 0.3:
                intensity = EmotionIntensity.LOW.value
            else:
                intensity = EmotionIntensity.MINIMAL.value
            
            # Calculate confidence scores for each modality
            confidence_scores = {}
            for modality_name, result in modality_results.items():
                confidence_scores[modality_name] = result.get('confidence', 0.1)
            
            overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.1
            
            # Calculate quality score based on agreement between modalities
            quality_score = self._calculate_quality_score(modality_results, fused_valence, fused_arousal)
            
            return {
                "valence": max(-1.0, min(1.0, fused_valence)),
                "arousal": max(-1.0, min(1.0, fused_arousal)),
                "dominance": max(-1.0, min(1.0, fused_dominance)),
                "primary_emotion": primary_emotion,
                "secondary_emotion": secondary_emotion,
                "intensity": intensity,
                "plutchik_scores": fused_plutchik,
                "confidence_scores": confidence_scores,
                "overall_confidence": overall_confidence,
                "quality_score": quality_score,
                "modalities_used": len(used_modalities),
                "fusion_method": "weighted_ensemble"
            }
            
        except Exception as e:
            logger.error(f"Emotion fusion failed: {str(e)}")
            return self._get_neutral_fusion_result()
    
    def _calculate_quality_score(self, modality_results: Dict, fused_valence: float, fused_arousal: float) -> float:
        """Calculate quality score based on inter-modality agreement."""
        if len(modality_results) < 2:
            return 0.5  # Medium quality for single modality
        
        # Calculate agreement in valence and arousal predictions
        valence_scores = [result.get('valence', 0.0) for result in modality_results.values()]
        arousal_scores = [result.get('arousal', 0.0) for result in modality_results.values()]
        
        # Calculate standard deviation (lower = better agreement)
        valence_agreement = 1.0 - (np.std(valence_scores) / 2.0)  # Normalize by max possible std
        arousal_agreement = 1.0 - (np.std(arousal_scores) / 2.0)
        
        # Average agreement score
        quality = (valence_agreement + arousal_agreement) / 2.0
        
        # Boost quality if multiple modalities agree on primary emotion
        primary_emotions = [result.get('primary_emotion', 'trust') for result in modality_results.values()]
        most_common_emotion = max(set(primary_emotions), key=primary_emotions.count)
        emotion_agreement = primary_emotions.count(most_common_emotion) / len(primary_emotions)
        
        # Weighted combination
        final_quality = 0.7 * quality + 0.3 * emotion_agreement
        
        return max(0.0, min(1.0, final_quality))
    
    def _get_neutral_fusion_result(self) -> Dict[str, Any]:
        """Get neutral result for fusion errors."""
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "primary_emotion": "trust",
            "secondary_emotion": None,
            "intensity": "minimal",
            "plutchik_scores": {emotion.value: 0.125 for emotion in BasicEmotion},
            "confidence_scores": {"overall": 0.1},
            "overall_confidence": 0.1,
            "quality_score": 0.1,
            "modalities_used": 0,
            "fusion_method": "error_fallback"
        }


# Create global emotion detector instance
emotion_detector = AdvancedEmotionDetector()


async def detect_emotion_from_message(
    message_content: str,
    user_id: Optional[str] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    voice_features: Optional[Dict[str, Any]] = None,
    behavioral_data: Optional[Dict[str, Any]] = None
) -> EmotionAnalysisResult:
    """
    Convenience function to detect emotion from a message.
    
    Args:
        message_content: The text content to analyze
        user_id: User ID for personalized analysis
        conversation_context: Context from the current conversation
        voice_features: Voice/audio features if available
        behavioral_data: User behavioral patterns
        
    Returns:
        Complete emotion analysis result
    """
    return await emotion_detector.analyze_emotion(
        text_content=message_content,
        voice_features=voice_features,
        behavioral_data=behavioral_data,
        conversation_context=conversation_context,
        user_id=user_id
    )