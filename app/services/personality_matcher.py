"""
Personality Matching Service

Advanced personality matching and compatibility system that:
- Matches optimal personality profiles to users
- Predicts personality compatibility and rapport potential
- Optimizes personality selection using ML algorithms
- A/B tests different personality approaches
- Learns from user interactions to improve matching
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# ML imports
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# Redis for caching
from redis.asyncio import Redis

# Internal imports
from app.models.personality import (
    PersonalityProfile, UserPersonalityMapping, PersonalityTrait,
    AdaptationStrategy, PersonalityDimension
)
from app.models.user import User
from app.services.personality_engine import ConversationContext, PersonalityState
from app.services.conversation_analyzer import ConversationMetrics, EmotionalState
from app.database.repository import Repository
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PersonalityMatch:
    """Result of personality matching algorithm."""
    profile_id: str
    profile_name: str
    compatibility_score: float
    confidence_level: float
    matching_reasons: List[str]
    adaptation_strategy: AdaptationStrategy
    expected_effectiveness: float
    risk_factors: List[str]
    optimization_suggestions: List[str]


@dataclass
class MatchingContext:
    """Context information for personality matching."""
    user_id: str
    user_traits: Dict[str, float]
    conversation_context: ConversationContext
    emotional_state: EmotionalState
    interaction_history: List[Dict[str, Any]]
    current_performance: Optional[Dict[str, float]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    situational_factors: Optional[Dict[str, Any]] = None


class PersonalityMatcher:
    """
    Advanced personality matching system using ML algorithms.
    
    This system optimally matches personality profiles to users by:
    1. Analyzing user personality traits and preferences
    2. Predicting compatibility using trained models
    3. Considering situational context and conversation state
    4. Learning from interaction outcomes to improve matching
    5. A/B testing different matching approaches
    """
    
    def __init__(self, db_session, redis_client: Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        
        # ML models for compatibility prediction
        self.compatibility_predictor = None
        self.effectiveness_predictor = None
        self.clustering_model = None
        
        # User and profile embeddings
        self.user_embeddings = {}
        self.profile_embeddings = {}
        
        # Matching algorithms
        self.matching_algorithms = {
            'similarity_based': self._similarity_based_matching,
            'complementary': self._complementary_matching,
            'ml_predicted': self._ml_predicted_matching,
            'contextual': self._contextual_matching,
            'hybrid': self._hybrid_matching
        }
        
        # A/B testing framework
        self.ab_test_groups = {}
        self.matching_performance = defaultdict(list)
        
        # Caching
        self.match_cache = {}
        self.profile_cache = {}
        
        logger.info("Personality matcher initialized")
    
    async def initialize_models(self) -> None:
        """Initialize ML models for personality matching."""
        try:
            logger.info("Initializing personality matching models...")
            
            # Initialize compatibility prediction model
            self.compatibility_predictor = CompatibilityPredictor()
            await self._load_or_train_compatibility_model()
            
            # Initialize effectiveness prediction model
            self.effectiveness_predictor = EffectivenessPredictor()
            await self._load_or_train_effectiveness_model()
            
            # Initialize clustering model for user segmentation
            self.clustering_model = KMeans(n_clusters=8, random_state=42)
            await self._initialize_user_clustering()
            
            # Load pre-computed embeddings
            await self._load_embeddings()
            
            logger.info("Personality matching models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing personality matching models: {e}")
            # Continue with rule-based fallbacks
    
    async def _load_or_train_compatibility_model(self) -> None:
        """Load existing compatibility model or train new one."""
        try:
            model_path = self.settings.ml.model_path / "compatibility_predictor.pt"
            if model_path.exists():
                self.compatibility_predictor.load_state_dict(torch.load(model_path))
                logger.info("Loaded existing compatibility prediction model")
            else:
                # Train new model with synthetic data initially
                await self._train_compatibility_model()
                
        except Exception as e:
            logger.error(f"Error loading compatibility model: {e}")
            # Use random forest as fallback
            self.compatibility_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    async def _load_or_train_effectiveness_model(self) -> None:
        """Load existing effectiveness model or train new one."""
        try:
            model_path = self.settings.ml.model_path / "effectiveness_predictor.pt"
            if model_path.exists():
                self.effectiveness_predictor.load_state_dict(torch.load(model_path))
                logger.info("Loaded existing effectiveness prediction model")
            else:
                # Train new model
                await self._train_effectiveness_model()
                
        except Exception as e:
            logger.error(f"Error loading effectiveness model: {e}")
            # Use random forest as fallback
            self.effectiveness_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    async def _initialize_user_clustering(self) -> None:
        """Initialize user clustering for segmentation."""
        try:
            # Get user personality data for clustering
            user_data = await self._get_user_personality_data()
            
            if len(user_data) > 10:  # Need minimum data for clustering
                features = np.array([list(traits.values()) for traits in user_data])
                self.clustering_model.fit(features)
                logger.info("User clustering model trained")
            else:
                logger.warning("Insufficient user data for clustering, using default segments")
                
        except Exception as e:
            logger.error(f"Error initializing user clustering: {e}")
    
    async def _load_embeddings(self) -> None:
        """Load pre-computed user and profile embeddings."""
        try:
            # Try to load from Redis cache
            user_embeddings_data = await self.redis.get("user_embeddings")
            if user_embeddings_data:
                self.user_embeddings = json.loads(user_embeddings_data)
            
            profile_embeddings_data = await self.redis.get("profile_embeddings")
            if profile_embeddings_data:
                self.profile_embeddings = json.loads(profile_embeddings_data)
            
            logger.info(f"Loaded {len(self.user_embeddings)} user embeddings and {len(self.profile_embeddings)} profile embeddings")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
    
    async def find_optimal_personality_match(
        self,
        matching_context: MatchingContext,
        available_profiles: Optional[List[PersonalityProfile]] = None,
        algorithm: str = 'hybrid'
    ) -> PersonalityMatch:
        """
        Find the optimal personality profile match for a user.
        
        Args:
            matching_context: Complete context for personality matching
            available_profiles: Optional list of profiles to consider
            algorithm: Matching algorithm to use
            
        Returns:
            PersonalityMatch with optimal profile and compatibility info
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(matching_context, algorithm)
            cached_match = await self._get_cached_match(cache_key)
            if cached_match:
                return cached_match
            
            # Get available profiles
            if not available_profiles:
                available_profiles = await self._get_available_profiles()
            
            if not available_profiles:
                return self._get_default_match()
            
            # Apply selected matching algorithm
            matching_func = self.matching_algorithms.get(algorithm, self._hybrid_matching)
            matches = await matching_func(matching_context, available_profiles)
            
            # Select best match
            best_match = max(matches, key=lambda m: m.compatibility_score)
            
            # Enhance match with additional analysis
            enhanced_match = await self._enhance_match(best_match, matching_context)
            
            # Cache the result
            await self._cache_match(cache_key, enhanced_match)
            
            # Log matching decision
            logger.info(f"Found personality match for user {matching_context.user_id}: "
                       f"{enhanced_match.profile_name} (score: {enhanced_match.compatibility_score:.3f})")
            
            return enhanced_match
            
        except Exception as e:
            logger.error(f"Error finding personality match: {e}")
            return self._get_default_match()
    
    async def _similarity_based_matching(
        self,
        context: MatchingContext,
        profiles: List[PersonalityProfile]
    ) -> List[PersonalityMatch]:
        """Match based on personality trait similarity."""
        matches = []
        
        for profile in profiles:
            if not profile.trait_scores:
                continue
            
            # Calculate cosine similarity between user and profile traits
            user_vector = np.array([context.user_traits.get(trait, 0.5) 
                                   for trait in PersonalityDimension])
            profile_vector = np.array([profile.trait_scores.get(trait.value, 0.5) 
                                     for trait in PersonalityDimension])
            
            similarity = cosine_similarity([user_vector], [profile_vector])[0][0]
            
            # Adjust for situational factors
            adjusted_score = await self._adjust_for_context(similarity, context, profile)
            
            matches.append(PersonalityMatch(
                profile_id=str(profile.id),
                profile_name=profile.name,
                compatibility_score=adjusted_score,
                confidence_level=0.7,  # Moderate confidence for similarity-based
                matching_reasons=[f"High personality similarity: {similarity:.2f}"],
                adaptation_strategy=AdaptationStrategy.MIRROR,
                expected_effectiveness=adjusted_score * 0.8,
                risk_factors=[],
                optimization_suggestions=[]
            ))
        
        return sorted(matches, key=lambda m: m.compatibility_score, reverse=True)
    
    async def _complementary_matching(
        self,
        context: MatchingContext,
        profiles: List[PersonalityProfile]
    ) -> List[PersonalityMatch]:
        """Match based on complementary personality traits."""
        matches = []
        
        for profile in profiles:
            if not profile.trait_scores:
                continue
            
            complement_score = 0.0
            trait_count = 0
            reasons = []
            
            for trait_enum in PersonalityDimension:
                trait = trait_enum.value
                user_value = context.user_traits.get(trait, 0.5)
                profile_value = profile.trait_scores.get(trait, 0.5)
                
                # Calculate complement based on trait type
                if trait == PersonalityDimension.NEUROTICISM.value:
                    # Low neuroticism complements high neuroticism
                    complement = 1.0 - abs(user_value - (1.0 - profile_value))
                elif trait in [PersonalityDimension.EXTRAVERSION.value]:
                    # Moderate complement for social traits
                    optimal_diff = 0.3  # Slightly different is good
                    complement = 1.0 - abs(abs(user_value - profile_value) - optimal_diff)
                else:
                    # For other traits, moderate similarity is complementary
                    complement = 1.0 - abs(user_value - profile_value) * 0.5
                
                complement_score += complement
                trait_count += 1
                
                if complement > 0.8:
                    reasons.append(f"Complementary {trait}: {complement:.2f}")
            
            overall_score = complement_score / max(trait_count, 1)
            adjusted_score = await self._adjust_for_context(overall_score, context, profile)
            
            matches.append(PersonalityMatch(
                profile_id=str(profile.id),
                profile_name=profile.name,
                compatibility_score=adjusted_score,
                confidence_level=0.6,  # Lower confidence for complementary matching
                matching_reasons=reasons[:3],  # Top 3 reasons
                adaptation_strategy=AdaptationStrategy.COMPLEMENT,
                expected_effectiveness=adjusted_score * 0.75,
                risk_factors=["Complementary matching can be unpredictable"],
                optimization_suggestions=["Monitor user reaction carefully"]
            ))
        
        return sorted(matches, key=lambda m: m.compatibility_score, reverse=True)
    
    async def _ml_predicted_matching(
        self,
        context: MatchingContext,
        profiles: List[PersonalityProfile]
    ) -> List[PersonalityMatch]:
        """Match using ML prediction models."""
        matches = []
        
        if not self.compatibility_predictor:
            # Fall back to similarity-based matching
            return await self._similarity_based_matching(context, profiles)
        
        try:
            for profile in profiles:
                if not profile.trait_scores:
                    continue
                
                # Prepare features for ML model
                features = await self._prepare_ml_features(context, profile)
                
                # Predict compatibility
                if isinstance(self.compatibility_predictor, torch.nn.Module):
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        compatibility_score = float(torch.sigmoid(self.compatibility_predictor(features_tensor)))
                else:
                    # Sklearn model
                    compatibility_score = float(self.compatibility_predictor.predict([features])[0])
                    compatibility_score = max(0.0, min(1.0, compatibility_score))
                
                # Predict effectiveness
                effectiveness_score = compatibility_score * 0.9  # Simplified
                if self.effectiveness_predictor and hasattr(self.effectiveness_predictor, 'predict'):
                    try:
                        effectiveness_score = float(self.effectiveness_predictor.predict([features])[0])
                        effectiveness_score = max(0.0, min(1.0, effectiveness_score))
                    except:
                        pass
                
                # Determine adaptation strategy based on prediction
                strategy = AdaptationStrategy.BALANCE
                if compatibility_score > 0.8:
                    strategy = AdaptationStrategy.MIRROR
                elif compatibility_score < 0.4:
                    strategy = AdaptationStrategy.COMPLEMENT
                
                matches.append(PersonalityMatch(
                    profile_id=str(profile.id),
                    profile_name=profile.name,
                    compatibility_score=compatibility_score,
                    confidence_level=0.85,  # High confidence for ML predictions
                    matching_reasons=[f"ML predicted compatibility: {compatibility_score:.2f}"],
                    adaptation_strategy=strategy,
                    expected_effectiveness=effectiveness_score,
                    risk_factors=[],
                    optimization_suggestions=[]
                ))
            
        except Exception as e:
            logger.error(f"Error in ML prediction matching: {e}")
            # Fall back to similarity-based matching
            return await self._similarity_based_matching(context, profiles)
        
        return sorted(matches, key=lambda m: m.compatibility_score, reverse=True)
    
    async def _contextual_matching(
        self,
        context: MatchingContext,
        profiles: List[PersonalityProfile]
    ) -> List[PersonalityMatch]:
        """Match based on conversation context and situational factors."""
        matches = []
        
        for profile in profiles:
            if not profile.trait_scores:
                continue
            
            context_score = 0.0
            reasons = []
            risk_factors = []
            
            # Consider conversation phase
            if context.conversation_context.conversation_phase == 'opening':
                # For opening phase, prefer welcoming and engaging personalities
                extraversion = profile.trait_scores.get(PersonalityDimension.EXTRAVERSION.value, 0.5)
                agreeableness = profile.trait_scores.get(PersonalityDimension.AGREEABLENESS.value, 0.5)
                context_score += (extraversion + agreeableness) * 0.4
                if extraversion > 0.6:
                    reasons.append("High extraversion good for opening")
            
            elif context.conversation_context.conversation_phase == 'deep':
                # For deep conversations, prefer thoughtful and empathetic personalities
                openness = profile.trait_scores.get(PersonalityDimension.OPENNESS.value, 0.5)
                empathy = profile.trait_scores.get(PersonalityDimension.EMPATHY.value, 0.5)
                context_score += (openness + empathy) * 0.4
                if openness > 0.6:
                    reasons.append("High openness good for deep conversations")
            
            # Consider emotional state
            if context.emotional_state.dominant_emotion:
                emotion = context.emotional_state.dominant_emotion
                if emotion in ['sadness', 'anger', 'fear']:
                    # Need empathetic and stable personality
                    empathy = profile.trait_scores.get(PersonalityDimension.EMPATHY.value, 0.5)
                    stability = 1.0 - profile.trait_scores.get(PersonalityDimension.NEUROTICISM.value, 0.5)
                    context_score += (empathy + stability) * 0.3
                    if empathy > 0.7:
                        reasons.append("High empathy for emotional support")
                
                elif emotion == 'joy':
                    # Can match user's positive energy
                    enthusiasm = profile.trait_scores.get(PersonalityDimension.ENTHUSIASM.value, 0.5)
                    context_score += enthusiasm * 0.2
                    if enthusiasm > 0.6:
                        reasons.append("High enthusiasm matches positive mood")
            
            # Consider urgency indicators
            urgency_level = max(context.conversation_context.urgency_indicators.values()) if context.conversation_context.urgency_indicators else 0.0
            if urgency_level > 0.7:
                # Need direct and efficient personality
                directness = profile.trait_scores.get(PersonalityDimension.DIRECTNESS.value, 0.5)
                conscientiousness = profile.trait_scores.get(PersonalityDimension.CONSCIENTIOUSNESS.value, 0.5)
                context_score += (directness + conscientiousness) * 0.3
                if directness > 0.6:
                    reasons.append("High directness for urgent situations")
                else:
                    risk_factors.append("May be too indirect for urgent needs")
            
            # Consider conversation goals
            for goal in context.conversation_context.conversation_goals:
                if goal == 'learning':
                    openness = profile.trait_scores.get(PersonalityDimension.OPENNESS.value, 0.5)
                    context_score += openness * 0.2
                elif goal == 'support':
                    empathy = profile.trait_scores.get(PersonalityDimension.EMPATHY.value, 0.5)
                    context_score += empathy * 0.2
                elif goal == 'entertainment':
                    humor = profile.trait_scores.get(PersonalityDimension.HUMOR.value, 0.5)
                    context_score += humor * 0.2
            
            # Normalize context score
            context_score = max(0.0, min(1.0, context_score))
            
            matches.append(PersonalityMatch(
                profile_id=str(profile.id),
                profile_name=profile.name,
                compatibility_score=context_score,
                confidence_level=0.75,
                matching_reasons=reasons,
                adaptation_strategy=AdaptationStrategy.BALANCE,
                expected_effectiveness=context_score * 0.85,
                risk_factors=risk_factors,
                optimization_suggestions=[]
            ))
        
        return sorted(matches, key=lambda m: m.compatibility_score, reverse=True)
    
    async def _hybrid_matching(
        self,
        context: MatchingContext,
        profiles: List[PersonalityProfile]
    ) -> List[PersonalityMatch]:
        """Hybrid matching combining multiple approaches."""
        # Get results from different algorithms
        similarity_matches = await self._similarity_based_matching(context, profiles)
        contextual_matches = await self._contextual_matching(context, profiles)
        
        # Try ML matching if available
        ml_matches = []
        if self.compatibility_predictor:
            ml_matches = await self._ml_predicted_matching(context, profiles)
        
        # Combine scores with weights
        combined_matches = {}
        
        # Weight configuration
        weights = {
            'similarity': 0.3,
            'contextual': 0.4,
            'ml': 0.3 if ml_matches else 0.0
        }
        
        # Normalize weights if no ML
        if not ml_matches:
            weights['similarity'] = 0.5
            weights['contextual'] = 0.5
        
        # Process similarity matches
        for match in similarity_matches:
            profile_id = match.profile_id
            if profile_id not in combined_matches:
                combined_matches[profile_id] = {
                    'profile_name': match.profile_name,
                    'scores': {},
                    'reasons': [],
                    'strategies': [],
                    'risk_factors': [],
                    'effectiveness_scores': []
                }
            combined_matches[profile_id]['scores']['similarity'] = match.compatibility_score
            combined_matches[profile_id]['reasons'].extend(match.matching_reasons)
            combined_matches[profile_id]['strategies'].append(match.adaptation_strategy)
            combined_matches[profile_id]['effectiveness_scores'].append(match.expected_effectiveness)
        
        # Process contextual matches
        for match in contextual_matches:
            profile_id = match.profile_id
            if profile_id not in combined_matches:
                combined_matches[profile_id] = {
                    'profile_name': match.profile_name,
                    'scores': {},
                    'reasons': [],
                    'strategies': [],
                    'risk_factors': [],
                    'effectiveness_scores': []
                }
            combined_matches[profile_id]['scores']['contextual'] = match.compatibility_score
            combined_matches[profile_id]['reasons'].extend(match.matching_reasons)
            combined_matches[profile_id]['risk_factors'].extend(match.risk_factors)
            combined_matches[profile_id]['effectiveness_scores'].append(match.expected_effectiveness)
        
        # Process ML matches if available
        for match in ml_matches:
            profile_id = match.profile_id
            if profile_id in combined_matches:
                combined_matches[profile_id]['scores']['ml'] = match.compatibility_score
                combined_matches[profile_id]['strategies'].append(match.adaptation_strategy)
                combined_matches[profile_id]['effectiveness_scores'].append(match.expected_effectiveness)
        
        # Calculate combined scores
        final_matches = []
        for profile_id, data in combined_matches.items():
            combined_score = 0.0
            for method, weight in weights.items():
                if method in data['scores']:
                    combined_score += data['scores'][method] * weight
            
            # Choose most common strategy
            strategies = data['strategies']
            if strategies:
                strategy_counts = {s: strategies.count(s) for s in set(strategies)}
                best_strategy = max(strategy_counts, key=strategy_counts.get)
            else:
                best_strategy = AdaptationStrategy.BALANCE
            
            # Average effectiveness
            avg_effectiveness = np.mean(data['effectiveness_scores']) if data['effectiveness_scores'] else combined_score * 0.8
            
            # Calculate confidence based on agreement between methods
            confidence = self._calculate_method_agreement(data['scores'])
            
            final_matches.append(PersonalityMatch(
                profile_id=profile_id,
                profile_name=data['profile_name'],
                compatibility_score=combined_score,
                confidence_level=confidence,
                matching_reasons=list(set(data['reasons'][:5])),  # Top 5 unique reasons
                adaptation_strategy=best_strategy,
                expected_effectiveness=avg_effectiveness,
                risk_factors=list(set(data['risk_factors'])),
                optimization_suggestions=[]
            ))
        
        return sorted(final_matches, key=lambda m: m.compatibility_score, reverse=True)
    
    def _calculate_method_agreement(self, scores: Dict[str, float]) -> float:
        """Calculate confidence based on agreement between different methods."""
        if len(scores) < 2:
            return 0.7  # Default confidence for single method
        
        score_values = list(scores.values())
        # Calculate standard deviation - lower std means higher agreement
        std_dev = np.std(score_values)
        # Convert to confidence (0-1 range)
        confidence = max(0.3, min(0.95, 1.0 - std_dev))
        return confidence
    
    async def _adjust_for_context(
        self,
        base_score: float,
        context: MatchingContext,
        profile: PersonalityProfile
    ) -> float:
        """Adjust compatibility score based on context."""
        adjusted_score = base_score
        
        # Adjust for conversation phase
        phase = context.conversation_context.conversation_phase
        if phase == 'opening':
            # Boost extraverted profiles for openings
            extraversion = profile.trait_scores.get(PersonalityDimension.EXTRAVERSION.value, 0.5)
            adjusted_score *= (1.0 + (extraversion - 0.5) * 0.2)
        
        elif phase == 'closing':
            # Boost agreeable profiles for closings
            agreeableness = profile.trait_scores.get(PersonalityDimension.AGREEABLENESS.value, 0.5)
            adjusted_score *= (1.0 + (agreeableness - 0.5) * 0.2)
        
        # Adjust for user engagement level
        if context.conversation_context.user_engagement_level < 0.3:
            # Low engagement - boost engaging personalities
            enthusiasm = profile.trait_scores.get(PersonalityDimension.ENTHUSIASM.value, 0.5)
            humor = profile.trait_scores.get(PersonalityDimension.HUMOR.value, 0.5)
            adjusted_score *= (1.0 + (enthusiasm + humor - 1.0) * 0.15)
        
        # Adjust for emotional volatility
        if hasattr(context.emotional_state, 'emotional_volatility') and context.emotional_state.emotional_volatility > 0.7:
            # High volatility - prefer stable personalities
            stability = 1.0 - profile.trait_scores.get(PersonalityDimension.NEUROTICISM.value, 0.5)
            adjusted_score *= (1.0 + (stability - 0.5) * 0.3)
        
        return max(0.0, min(1.0, adjusted_score))
    
    async def _prepare_ml_features(self, context: MatchingContext, profile: PersonalityProfile) -> List[float]:
        """Prepare features for ML models."""
        features = []
        
        # User personality traits (10 dimensions)
        for trait_enum in PersonalityDimension:
            features.append(context.user_traits.get(trait_enum.value, 0.5))
        
        # Profile personality traits (10 dimensions)
        for trait_enum in PersonalityDimension:
            features.append(profile.trait_scores.get(trait_enum.value, 0.5))
        
        # Trait differences (10 dimensions)
        for trait_enum in PersonalityDimension:
            user_val = context.user_traits.get(trait_enum.value, 0.5)
            profile_val = profile.trait_scores.get(trait_enum.value, 0.5)
            features.append(abs(user_val - profile_val))
        
        # Context features
        features.extend([
            context.conversation_context.user_engagement_level,
            context.conversation_context.time_in_conversation / 3600.0,  # Normalize to hours
            len(context.conversation_context.conversation_goals) / 5.0,  # Normalize
            context.conversation_context.context_switches / 10.0,  # Normalize
            max(context.conversation_context.urgency_indicators.values()) if context.conversation_context.urgency_indicators else 0.0
        ])
        
        # Emotional state features
        if hasattr(context.emotional_state, 'emotional_volatility'):
            features.append(context.emotional_state.emotional_volatility)
        else:
            features.append(0.5)
        
        # Add dominant emotion encoding (simplified)
        dominant_emotion = getattr(context.emotional_state, 'dominant_emotion', None)
        emotion_encoding = {
            'joy': [1, 0, 0, 0, 0],
            'sadness': [0, 1, 0, 0, 0],
            'anger': [0, 0, 1, 0, 0],
            'fear': [0, 0, 0, 1, 0],
            'surprise': [0, 0, 0, 0, 1]
        }
        features.extend(emotion_encoding.get(dominant_emotion, [0.2, 0.2, 0.2, 0.2, 0.2]))
        
        # Historical performance if available
        if context.current_performance:
            features.extend([
                context.current_performance.get('effectiveness', 0.5),
                context.current_performance.get('satisfaction', 0.5),
                context.current_performance.get('engagement', 0.5)
            ])
        else:
            features.extend([0.5, 0.5, 0.5])
        
        return features
    
    async def _enhance_match(self, match: PersonalityMatch, context: MatchingContext) -> PersonalityMatch:
        """Enhance match with additional analysis and suggestions."""
        # Add optimization suggestions
        suggestions = []
        
        if match.compatibility_score < 0.6:
            suggestions.append("Consider more adaptive personality traits")
            suggestions.append("Monitor user reaction closely")
        
        if context.conversation_context.user_engagement_level < 0.4:
            suggestions.append("Increase enthusiasm and interactivity")
        
        if context.conversation_context.urgency_indicators:
            max_urgency = max(context.conversation_context.urgency_indicators.values())
            if max_urgency > 0.6:
                suggestions.append("Prioritize directness and efficiency")
        
        # Add risk factors based on context
        risk_factors = list(match.risk_factors)
        
        if match.confidence_level < 0.5:
            risk_factors.append("Low confidence in personality match")
        
        if context.emotional_state.emotional_volatility > 0.7:
            risk_factors.append("User emotional volatility may affect interaction")
        
        # Create enhanced match
        enhanced_match = PersonalityMatch(
            profile_id=match.profile_id,
            profile_name=match.profile_name,
            compatibility_score=match.compatibility_score,
            confidence_level=match.confidence_level,
            matching_reasons=match.matching_reasons,
            adaptation_strategy=match.adaptation_strategy,
            expected_effectiveness=match.expected_effectiveness,
            risk_factors=risk_factors,
            optimization_suggestions=suggestions
        )
        
        return enhanced_match
    
    async def learn_from_interaction_outcome(
        self,
        match: PersonalityMatch,
        context: MatchingContext,
        outcome: Dict[str, Any]
    ) -> None:
        """Learn from interaction outcomes to improve matching."""
        try:
            # Extract performance metrics
            actual_effectiveness = outcome.get('effectiveness_score', 0.5)
            user_satisfaction = outcome.get('satisfaction_score', 0.5)
            engagement_score = outcome.get('engagement_score', 0.5)
            
            # Calculate prediction error
            effectiveness_error = abs(match.expected_effectiveness - actual_effectiveness)
            
            # Update matching algorithm performance
            algorithm_used = outcome.get('algorithm_used', 'hybrid')
            self.matching_performance[algorithm_used].append({
                'predicted_effectiveness': match.expected_effectiveness,
                'actual_effectiveness': actual_effectiveness,
                'error': effectiveness_error,
                'satisfaction': user_satisfaction,
                'engagement': engagement_score,
                'timestamp': datetime.now()
            })
            
            # Update ML models if available
            if isinstance(self.compatibility_predictor, torch.nn.Module):
                await self._update_compatibility_model(match, context, outcome)
            
            if isinstance(self.effectiveness_predictor, torch.nn.Module):
                await self._update_effectiveness_model(match, context, outcome)
            
            # Update user and profile embeddings
            await self._update_embeddings(match, context, outcome)
            
            logger.info(f"Learned from interaction: predicted {match.expected_effectiveness:.3f}, "
                       f"actual {actual_effectiveness:.3f}, error {effectiveness_error:.3f}")
            
        except Exception as e:
            logger.error(f"Error learning from interaction outcome: {e}")
    
    async def _update_compatibility_model(
        self,
        match: PersonalityMatch,
        context: MatchingContext,
        outcome: Dict[str, Any]
    ) -> None:
        """Update compatibility prediction model with new data."""
        # This would implement online learning for the neural network
        # For now, we'll collect data for periodic retraining
        pass
    
    async def _update_effectiveness_model(
        self,
        match: PersonalityMatch,
        context: MatchingContext,
        outcome: Dict[str, Any]
    ) -> None:
        """Update effectiveness prediction model with new data."""
        # Similar to compatibility model update
        pass
    
    async def _update_embeddings(
        self,
        match: PersonalityMatch,
        context: MatchingContext,
        outcome: Dict[str, Any]
    ) -> None:
        """Update user and profile embeddings based on interaction outcome."""
        try:
            user_id = context.user_id
            profile_id = match.profile_id
            
            # Simple embedding update based on outcome
            effectiveness = outcome.get('effectiveness_score', 0.5)
            
            # Update user embedding
            if user_id in self.user_embeddings:
                # Weighted update towards successful interactions
                current_embedding = np.array(self.user_embeddings[user_id])
                profile_traits = np.array([context.user_traits.get(t.value, 0.5) for t in PersonalityDimension])
                
                # Learning rate based on effectiveness
                learning_rate = 0.1 * effectiveness
                updated_embedding = current_embedding + learning_rate * (profile_traits - current_embedding)
                self.user_embeddings[user_id] = updated_embedding.tolist()
            
            # Cache updated embeddings
            await self.redis.setex(
                "user_embeddings",
                86400,  # 24 hours
                json.dumps(self.user_embeddings)
            )
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
    
    async def get_matching_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for different matching algorithms."""
        stats = {}
        
        for algorithm, performances in self.matching_performance.items():
            if not performances:
                continue
            
            recent_performances = [p for p in performances 
                                 if datetime.now() - p['timestamp'] < timedelta(days=7)]
            
            if recent_performances:
                stats[algorithm] = {
                    'total_matches': len(performances),
                    'recent_matches': len(recent_performances),
                    'avg_effectiveness': np.mean([p['actual_effectiveness'] for p in recent_performances]),
                    'avg_error': np.mean([p['error'] for p in recent_performances]),
                    'avg_satisfaction': np.mean([p['satisfaction'] for p in recent_performances]),
                    'avg_engagement': np.mean([p['engagement'] for p in recent_performances])
                }
        
        return stats
    
    # Utility methods
    async def _get_available_profiles(self) -> List[PersonalityProfile]:
        """Get all available personality profiles."""
        try:
            from sqlalchemy import select
            query = select(PersonalityProfile).where(PersonalityProfile.is_active == True)
            result = await self.db.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting available profiles: {e}")
            return []
    
    async def _get_user_personality_data(self) -> List[Dict[str, float]]:
        """Get user personality data for clustering."""
        try:
            from sqlalchemy import select
            query = select(UserPersonalityMapping).where(
                UserPersonalityMapping.is_active == True,
                UserPersonalityMapping.measured_user_traits.is_not(None)
            )
            result = await self.db.execute(query)
            mappings = result.scalars().all()
            
            return [mapping.measured_user_traits for mapping in mappings 
                   if mapping.measured_user_traits]
        except Exception as e:
            logger.error(f"Error getting user personality data: {e}")
            return []
    
    def _get_cache_key(self, context: MatchingContext, algorithm: str) -> str:
        """Generate cache key for personality match."""
        # Create a hash of the context for caching
        import hashlib
        context_str = f"{context.user_id}_{algorithm}_{context.conversation_context.conversation_phase}_{context.conversation_context.user_engagement_level:.2f}"
        return f"personality_match:{hashlib.md5(context_str.encode()).hexdigest()}"
    
    async def _get_cached_match(self, cache_key: str) -> Optional[PersonalityMatch]:
        """Get cached personality match."""
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return PersonalityMatch(
                    profile_id=data['profile_id'],
                    profile_name=data['profile_name'],
                    compatibility_score=data['compatibility_score'],
                    confidence_level=data['confidence_level'],
                    matching_reasons=data['matching_reasons'],
                    adaptation_strategy=AdaptationStrategy(data['adaptation_strategy']),
                    expected_effectiveness=data['expected_effectiveness'],
                    risk_factors=data['risk_factors'],
                    optimization_suggestions=data['optimization_suggestions']
                )
        except Exception as e:
            logger.error(f"Error getting cached match: {e}")
        return None
    
    async def _cache_match(self, cache_key: str, match: PersonalityMatch) -> None:
        """Cache personality match."""
        try:
            cache_data = {
                'profile_id': match.profile_id,
                'profile_name': match.profile_name,
                'compatibility_score': match.compatibility_score,
                'confidence_level': match.confidence_level,
                'matching_reasons': match.matching_reasons,
                'adaptation_strategy': match.adaptation_strategy.value,
                'expected_effectiveness': match.expected_effectiveness,
                'risk_factors': match.risk_factors,
                'optimization_suggestions': match.optimization_suggestions,
                'cached_at': datetime.now().isoformat()
            }
            
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.error(f"Error caching match: {e}")
    
    def _get_default_match(self) -> PersonalityMatch:
        """Get default personality match as fallback."""
        return PersonalityMatch(
            profile_id="default",
            profile_name="Balanced Assistant",
            compatibility_score=0.7,
            confidence_level=0.5,
            matching_reasons=["Default balanced personality"],
            adaptation_strategy=AdaptationStrategy.BALANCE,
            expected_effectiveness=0.65,
            risk_factors=[],
            optimization_suggestions=["Consider more personalized matching when data is available"]
        )
    
    async def _train_compatibility_model(self) -> None:
        """Train compatibility prediction model with available data."""
        # This would implement model training
        # For now, we'll use a pre-initialized model
        logger.info("Compatibility model training would be implemented here")
    
    async def _train_effectiveness_model(self) -> None:
        """Train effectiveness prediction model with available data."""
        # This would implement model training
        logger.info("Effectiveness model training would be implemented here")


class CompatibilityPredictor(nn.Module):
    """Neural network for predicting personality compatibility."""
    
    def __init__(self, input_dim: int = 46):  # Based on feature engineering
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class EffectivenessPredictor(nn.Module):
    """Neural network for predicting interaction effectiveness."""
    
    def __init__(self, input_dim: int = 46):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Export main classes
__all__ = ['PersonalityMatcher', 'PersonalityMatch', 'MatchingContext']