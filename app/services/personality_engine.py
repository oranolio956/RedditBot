"""
Advanced Personality Engine
 
A comprehensive AI personality system that uses machine learning to create dynamic,
adaptive personalities for each user. Features include:

- Master personality framework with trait definitions
- Sub-personality adaptation using reinforcement learning 
- Real-time conversation context analysis
- Psychological profiling and rapport building
- Natural conversation flow with personality consistency
- A/B testing for personality optimization
- Memory integration for long-term personality development
"""

import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

# ML and NLP imports
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

# Redis for caching and real-time data
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, func

# Internal imports
from app.database.repository import Repository
from app.models.personality import (
    PersonalityTrait, PersonalityProfile, UserPersonalityMapping,
    PersonalityDimension, AdaptationStrategy
)
from app.models.user import User
from app.models.conversation import Message, ConversationSession
from app.core.redis import get_redis_client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class PersonalityMetrics(Enum):
    """Metrics for personality effectiveness tracking."""
    ENGAGEMENT_SCORE = "engagement_score"
    SATISFACTION_SCORE = "satisfaction_score"
    CONVERSATION_LENGTH = "conversation_length"
    RESPONSE_TIME = "response_time"
    SENTIMENT_IMPROVEMENT = "sentiment_improvement"
    USER_RETENTION = "user_retention"
    MESSAGE_QUALITY = "message_quality"


@dataclass
class ConversationContext:
    """Context information for personality adaptation."""
    user_id: str
    session_id: str
    message_history: List[Dict[str, Any]]
    current_sentiment: float
    topic: Optional[str] = None
    urgency_level: float = 0.5
    conversation_phase: str = "ongoing"  # greeting, ongoing, closing
    user_emotional_state: Dict[str, float] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalityState:
    """Current personality state for a user interaction."""
    base_traits: Dict[str, float]
    adapted_traits: Dict[str, float]
    confidence_level: float
    adaptation_history: List[Dict[str, Any]]
    effectiveness_metrics: Dict[str, float]
    last_updated: datetime


class AdvancedPersonalityEngine:
    """
    Advanced ML-driven personality engine for AI conversation bots.
    
    This engine creates unique, adaptive personalities for each user by:
    1. Analyzing conversation patterns and user behavior
    2. Adapting personality traits in real-time
    3. Using reinforcement learning to optimize interactions
    4. Building psychological rapport through mirroring and complementing
    5. A/B testing different personality approaches
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        
        # ML Models - initialized asynchronously
        self.sentiment_analyzer = None
        self.emotion_classifier = None
        self.personality_classifier = None
        self.rapport_predictor = None
        
        # Tokenizer and embedding model
        self.tokenizer = None
        self.embedding_model = None
        
        # Cache for frequently accessed data
        self.trait_cache = {}
        self.profile_cache = {}
        self.user_personality_cache = {}
        
        # Personality adaptation algorithms
        self.adaptation_algorithms = {
            AdaptationStrategy.MIRROR: self._mirror_adaptation,
            AdaptationStrategy.COMPLEMENT: self._complement_adaptation,
            AdaptationStrategy.BALANCE: self._balanced_adaptation,
            AdaptationStrategy.CUSTOM: self._custom_adaptation,
        }
        
        # Reinforcement learning components
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        
        # A/B testing framework
        self.ab_tests = {}
        self.ab_results = defaultdict(list)
        
        logger.info("Personality engine initialized")
    
    async def initialize_models(self) -> None:
        """Initialize ML models asynchronously."""
        try:
            logger.info("Loading ML models for personality engine...")
            
            # Sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Emotion classification model  
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Initialize tokenizer and embedding model for similarity calculations
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embedding_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize personality trait classifier (custom model)
            await self._initialize_personality_classifier()
            
            # Initialize rapport prediction model
            await self._initialize_rapport_predictor()
            
            logger.info("All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            raise
    
    async def _initialize_personality_classifier(self) -> None:
        """Initialize or load personality trait classification model."""
        try:
            # Try to load pre-trained model, otherwise create new one
            try:
                # Load from saved state
                model_path = self.settings.ml.model_path / "personality_classifier.pt"
                if model_path.exists():
                    self.personality_classifier = torch.load(model_path)
                    logger.info("Loaded existing personality classifier")
                else:
                    raise FileNotFoundError("Model not found, creating new one")
            except:
                # Create new model
                self.personality_classifier = PersonalityClassifierNN()
                logger.info("Created new personality classifier model")
                
        except Exception as e:
            logger.error(f"Error initializing personality classifier: {e}")
            # Fallback to rule-based system
            self.personality_classifier = RuleBasedPersonalityClassifier()
    
    async def _initialize_rapport_predictor(self) -> None:
        """Initialize rapport prediction model."""
        try:
            # This model predicts how well a personality will work with a user
            self.rapport_predictor = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Try to load existing model
            model_path = self.settings.ml.model_path / "rapport_predictor.pkl"
            if model_path.exists():
                import pickle
                with open(model_path, 'rb') as f:
                    self.rapport_predictor = pickle.load(f)
                logger.info("Loaded existing rapport predictor")
            else:
                logger.info("Created new rapport predictor model")
                
        except Exception as e:
            logger.error(f"Error initializing rapport predictor: {e}")
            # Continue without rapport prediction
            self.rapport_predictor = None
    
    async def analyze_user_personality(
        self, 
        user_id: str, 
        conversation_context: ConversationContext
    ) -> Dict[str, float]:
        """
        Analyze user's personality from conversation history and current context.
        
        Uses multiple ML models to extract personality traits from:
        - Message content and style
        - Conversation patterns  
        - Emotional responses
        - Topic preferences
        - Response timing and engagement
        """
        try:
            cache_key = f"user_personality:{user_id}"
            
            # Check cache first
            cached_result = await self.redis.get(cache_key)
            if cached_result:
                cached_data = json.loads(cached_result)
                # Check if cache is recent enough
                if datetime.fromisoformat(cached_data['timestamp']) > datetime.now() - timedelta(hours=1):
                    return cached_data['traits']
            
            # Analyze message content for personality indicators
            content_traits = await self._analyze_message_content(conversation_context.message_history)
            
            # Analyze conversation patterns
            pattern_traits = await self._analyze_conversation_patterns(user_id, conversation_context)
            
            # Analyze emotional responses
            emotion_traits = await self._analyze_emotional_patterns(conversation_context.message_history)
            
            # Analyze linguistic style
            linguistic_traits = await self._analyze_linguistic_style(conversation_context.message_history)
            
            # Analyze timing and engagement patterns
            engagement_traits = await self._analyze_engagement_patterns(user_id, conversation_context)
            
            # Combine all trait analyses with weighted averaging
            combined_traits = self._combine_trait_analyses([
                (content_traits, 0.25),
                (pattern_traits, 0.20),
                (emotion_traits, 0.20),
                (linguistic_traits, 0.15),
                (engagement_traits, 0.20)
            ])
            
            # Apply confidence scoring
            confidence_score = self._calculate_analysis_confidence(
                conversation_context.message_history,
                combined_traits
            )
            
            result = {
                'traits': combined_traits,
                'confidence': confidence_score,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(result)
            )
            
            logger.info(f"Analyzed personality for user {user_id} with {confidence_score:.2f} confidence")
            return combined_traits
            
        except Exception as e:
            logger.error(f"Error analyzing user personality for {user_id}: {e}")
            # Return neutral personality as fallback
            return self._get_neutral_personality()
    
    async def _analyze_message_content(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze message content for personality indicators."""
        if not messages:
            return self._get_neutral_personality()
        
        # Extract text content
        texts = [msg.get('content', '') for msg in messages if msg.get('content')]
        if not texts:
            return self._get_neutral_personality()
        
        # Combine all texts for analysis
        combined_text = ' '.join(texts)
        
        # Use personality classifier if available
        if self.personality_classifier:
            try:
                traits = await self._classify_personality_from_text(combined_text)
                return traits
            except Exception as e:
                logger.error(f"Error in personality classification: {e}")
        
        # Fallback to rule-based analysis
        return await self._rule_based_personality_analysis(texts)
    
    async def _classify_personality_from_text(self, text: str) -> Dict[str, float]:
        """Use ML model to classify personality from text."""
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Classify personality traits
        if isinstance(self.personality_classifier, PersonalityClassifierNN):
            with torch.no_grad():
                trait_scores = self.personality_classifier(embeddings.unsqueeze(0))
                trait_scores = torch.sigmoid(trait_scores).squeeze().numpy()
        else:
            # Rule-based classifier
            trait_scores = self.personality_classifier.predict(embeddings.numpy().reshape(1, -1))[0]
        
        # Map to personality dimensions
        return {
            PersonalityDimension.OPENNESS.value: float(trait_scores[0]),
            PersonalityDimension.CONSCIENTIOUSNESS.value: float(trait_scores[1]),
            PersonalityDimension.EXTRAVERSION.value: float(trait_scores[2]),
            PersonalityDimension.AGREEABLENESS.value: float(trait_scores[3]),
            PersonalityDimension.NEUROTICISM.value: float(trait_scores[4]),
            PersonalityDimension.HUMOR.value: float(trait_scores[5]) if len(trait_scores) > 5 else 0.5,
            PersonalityDimension.EMPATHY.value: float(trait_scores[6]) if len(trait_scores) > 6 else 0.5,
            PersonalityDimension.FORMALITY.value: float(trait_scores[7]) if len(trait_scores) > 7 else 0.5,
            PersonalityDimension.DIRECTNESS.value: float(trait_scores[8]) if len(trait_scores) > 8 else 0.5,
            PersonalityDimension.ENTHUSIASM.value: float(trait_scores[9]) if len(trait_scores) > 9 else 0.5,
        }
    
    async def _rule_based_personality_analysis(self, texts: List[str]) -> Dict[str, float]:
        """Rule-based personality analysis as fallback."""
        combined_text = ' '.join(texts).lower()
        text_length = len(combined_text)
        
        # Initialize traits with neutral values
        traits = self._get_neutral_personality()
        
        # Analyze openness indicators
        openness_indicators = ['creative', 'innovative', 'curious', 'artistic', 'imagine', 'novel']
        openness_score = sum(combined_text.count(word) for word in openness_indicators) / max(text_length / 1000, 1)
        traits[PersonalityDimension.OPENNESS.value] = min(0.9, 0.3 + openness_score * 0.6)
        
        # Analyze conscientiousness indicators
        conscientiousness_indicators = ['organized', 'plan', 'schedule', 'careful', 'detail', 'systematic']
        conscientiousness_score = sum(combined_text.count(word) for word in conscientiousness_indicators) / max(text_length / 1000, 1)
        traits[PersonalityDimension.CONSCIENTIOUSNESS.value] = min(0.9, 0.3 + conscientiousness_score * 0.6)
        
        # Analyze extraversion indicators
        extraversion_indicators = ['social', 'party', 'friends', 'outgoing', 'energetic', 'talkative']
        extraversion_score = sum(combined_text.count(word) for word in extraversion_indicators) / max(text_length / 1000, 1)
        traits[PersonalityDimension.EXTRAVERSION.value] = min(0.9, 0.3 + extraversion_score * 0.6)
        
        # Analyze agreeableness indicators
        agreeableness_indicators = ['kind', 'helpful', 'trust', 'cooperative', 'sympathetic', 'considerate']
        agreeableness_score = sum(combined_text.count(word) for word in agreeableness_indicators) / max(text_length / 1000, 1)
        traits[PersonalityDimension.AGREEABLENESS.value] = min(0.9, 0.3 + agreeableness_score * 0.6)
        
        # Analyze neuroticism indicators
        neuroticism_indicators = ['worry', 'stress', 'anxious', 'nervous', 'moody', 'tense']
        neuroticism_score = sum(combined_text.count(word) for word in neuroticism_indicators) / max(text_length / 1000, 1)
        traits[PersonalityDimension.NEUROTICISM.value] = min(0.9, 0.1 + neuroticism_score * 0.8)
        
        # Analyze humor indicators
        humor_indicators = ['haha', 'lol', 'funny', 'joke', 'hilarious', '😂', '😄', '😆']
        humor_score = sum(combined_text.count(word) for word in humor_indicators) / max(text_length / 1000, 1)
        traits[PersonalityDimension.HUMOR.value] = min(0.9, 0.2 + humor_score * 0.7)
        
        # Analyze formality from language patterns
        formal_indicators = ['please', 'thank you', 'would you', 'could you', 'appreciate']
        informal_indicators = ['hey', 'gonna', 'wanna', 'yeah', 'nope', 'yep']
        formal_score = sum(combined_text.count(word) for word in formal_indicators)
        informal_score = sum(combined_text.count(word) for word in informal_indicators)
        
        if formal_score + informal_score > 0:
            formality_ratio = formal_score / (formal_score + informal_score)
            traits[PersonalityDimension.FORMALITY.value] = formality_ratio
        
        return traits
    
    async def _analyze_conversation_patterns(
        self, 
        user_id: str, 
        context: ConversationContext
    ) -> Dict[str, float]:
        """Analyze conversation patterns for personality insights."""
        patterns = self._get_neutral_personality()
        
        try:
            # Analyze message timing patterns
            message_times = []
            for msg in context.message_history:
                if 'timestamp' in msg:
                    message_times.append(datetime.fromisoformat(msg['timestamp']))
            
            if len(message_times) > 1:
                # Calculate response time patterns
                response_times = []
                for i in range(1, len(message_times)):
                    time_diff = (message_times[i] - message_times[i-1]).total_seconds()
                    response_times.append(time_diff)
                
                avg_response_time = np.mean(response_times)
                response_time_variance = np.var(response_times)
                
                # Quick responders might be more extraverted
                if avg_response_time < 30:  # Quick responder
                    patterns[PersonalityDimension.EXTRAVERSION.value] += 0.1
                elif avg_response_time > 300:  # Thoughtful responder
                    patterns[PersonalityDimension.CONSCIENTIOUSNESS.value] += 0.1
                
                # Consistent timing suggests conscientiousness
                if response_time_variance < 100:
                    patterns[PersonalityDimension.CONSCIENTIOUSNESS.value] += 0.1
            
            # Analyze conversation topic diversity
            topics = set()
            for msg in context.message_history:
                if 'topic' in msg and msg['topic']:
                    topics.add(msg['topic'])
            
            topic_diversity = len(topics) / max(len(context.message_history), 1)
            patterns[PersonalityDimension.OPENNESS.value] = min(0.9, 0.3 + topic_diversity * 0.6)
            
            # Analyze message length patterns
            message_lengths = [len(msg.get('content', '')) for msg in context.message_history if msg.get('content')]
            if message_lengths:
                avg_length = np.mean(message_lengths)
                if avg_length > 100:  # Verbose users might be more open
                    patterns[PersonalityDimension.OPENNESS.value] += 0.1
                if avg_length > 200:  # Very verbose might indicate conscientiousness
                    patterns[PersonalityDimension.CONSCIENTIOUSNESS.value] += 0.1
            
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
        
        return patterns
    
    async def _analyze_emotional_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze emotional patterns in messages."""
        traits = self._get_neutral_personality()
        
        if not messages:
            return traits
        
        try:
            # Extract sentiment for each message
            sentiments = []
            emotions = []
            
            for msg in messages:
                content = msg.get('content', '')
                if not content:
                    continue
                
                # Get sentiment
                if self.sentiment_analyzer:
                    sentiment_results = self.sentiment_analyzer(content)
                    # Convert to numerical score
                    sentiment_score = 0.0
                    for result in sentiment_results[0]:
                        if result['label'] == 'LABEL_2':  # Positive
                            sentiment_score += result['score']
                        elif result['label'] == 'LABEL_0':  # Negative
                            sentiment_score -= result['score']
                    sentiments.append(sentiment_score)
                
                # Get emotions
                if self.emotion_classifier:
                    emotion_results = self.emotion_classifier(content)
                    emotions.append(emotion_results[0])
            
            # Analyze sentiment patterns
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                sentiment_variance = np.var(sentiments)
                
                # Positive sentiment suggests low neuroticism, high agreeableness
                if avg_sentiment > 0.2:
                    traits[PersonalityDimension.NEUROTICISM.value] = max(0.1, traits[PersonalityDimension.NEUROTICISM.value] - 0.2)
                    traits[PersonalityDimension.AGREEABLENESS.value] = min(0.9, traits[PersonalityDimension.AGREEABLENESS.value] + 0.2)
                
                # High sentiment variance might indicate neuroticism
                if sentiment_variance > 0.5:
                    traits[PersonalityDimension.NEUROTICISM.value] = min(0.9, traits[PersonalityDimension.NEUROTICISM.value] + 0.2)
            
            # Analyze emotion patterns
            if emotions:
                emotion_counts = defaultdict(float)
                for emotion_set in emotions:
                    for emotion in emotion_set:
                        emotion_counts[emotion['label']] += emotion['score']
                
                # Map emotions to personality traits
                if emotion_counts['joy'] > emotion_counts['sadness']:
                    traits[PersonalityDimension.NEUROTICISM.value] = max(0.1, traits[PersonalityDimension.NEUROTICISM.value] - 0.1)
                    traits[PersonalityDimension.EXTRAVERSION.value] = min(0.9, traits[PersonalityDimension.EXTRAVERSION.value] + 0.1)
                
                if emotion_counts['anger'] > 0.3:
                    traits[PersonalityDimension.AGREEABLENESS.value] = max(0.1, traits[PersonalityDimension.AGREEABLENESS.value] - 0.1)
                    traits[PersonalityDimension.NEUROTICISM.value] = min(0.9, traits[PersonalityDimension.NEUROTICISM.value] + 0.1)
                
                if emotion_counts['fear'] > 0.3:
                    traits[PersonalityDimension.NEUROTICISM.value] = min(0.9, traits[PersonalityDimension.NEUROTICISM.value] + 0.2)
        
        except Exception as e:
            logger.error(f"Error analyzing emotional patterns: {e}")
        
        return traits
    
    async def _analyze_linguistic_style(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze linguistic style for personality indicators."""
        traits = self._get_neutral_personality()
        
        if not messages:
            return traits
        
        try:
            texts = [msg.get('content', '') for msg in messages if msg.get('content')]
            combined_text = ' '.join(texts)
            
            if not combined_text:
                return traits
            
            # Analyze vocabulary complexity
            words = combined_text.lower().split()
            unique_words = set(words)
            vocabulary_richness = len(unique_words) / max(len(words), 1)
            
            # Rich vocabulary suggests openness and conscientiousness
            if vocabulary_richness > 0.7:
                traits[PersonalityDimension.OPENNESS.value] = min(0.9, traits[PersonalityDimension.OPENNESS.value] + 0.2)
                traits[PersonalityDimension.CONSCIENTIOUSNESS.value] = min(0.9, traits[PersonalityDimension.CONSCIENTIOUSNESS.value] + 0.1)
            
            # Analyze sentence complexity
            sentences = combined_text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Complex sentences suggest conscientiousness and openness
            if avg_sentence_length > 15:
                traits[PersonalityDimension.CONSCIENTIOUSNESS.value] = min(0.9, traits[PersonalityDimension.CONSCIENTIOUSNESS.value] + 0.1)
                traits[PersonalityDimension.OPENNESS.value] = min(0.9, traits[PersonalityDimension.OPENNESS.value] + 0.1)
            
            # Analyze punctuation and capitalization patterns
            exclamation_count = combined_text.count('!')
            question_count = combined_text.count('?')
            caps_ratio = sum(1 for c in combined_text if c.isupper()) / max(len(combined_text), 1)
            
            # Exclamation marks might indicate extraversion and enthusiasm
            if exclamation_count / len(sentences) > 0.3:
                traits[PersonalityDimension.EXTRAVERSION.value] = min(0.9, traits[PersonalityDimension.EXTRAVERSION.value] + 0.1)
                traits[PersonalityDimension.ENTHUSIASM.value] = min(0.9, traits[PersonalityDimension.ENTHUSIASM.value] + 0.2)
            
            # Many questions might indicate openness and curiosity
            if question_count / len(sentences) > 0.2:
                traits[PersonalityDimension.OPENNESS.value] = min(0.9, traits[PersonalityDimension.OPENNESS.value] + 0.1)
            
            # High caps ratio might indicate neuroticism or low agreeableness
            if caps_ratio > 0.1:
                traits[PersonalityDimension.NEUROTICISM.value] = min(0.9, traits[PersonalityDimension.NEUROTICISM.value] + 0.1)
                traits[PersonalityDimension.AGREEABLENESS.value] = max(0.1, traits[PersonalityDimension.AGREEABLENESS.value] - 0.1)
        
        except Exception as e:
            logger.error(f"Error analyzing linguistic style: {e}")
        
        return traits
    
    async def _analyze_engagement_patterns(
        self, 
        user_id: str, 
        context: ConversationContext
    ) -> Dict[str, float]:
        """Analyze user engagement patterns."""
        traits = self._get_neutral_personality()
        
        try:
            # Analyze conversation frequency and duration
            session_data = await self._get_user_session_data(user_id)
            
            if session_data:
                avg_session_duration = np.mean([s['duration'] for s in session_data])
                session_frequency = len(session_data) / max((datetime.now() - datetime.fromisoformat(session_data[0]['created_at'])).days, 1)
                
                # Long sessions might indicate conscientiousness
                if avg_session_duration > 600:  # 10 minutes
                    traits[PersonalityDimension.CONSCIENTIOUSNESS.value] = min(0.9, traits[PersonalityDimension.CONSCIENTIOUSNESS.value] + 0.1)
                
                # Frequent engagement might indicate extraversion
                if session_frequency > 1:  # More than once per day
                    traits[PersonalityDimension.EXTRAVERSION.value] = min(0.9, traits[PersonalityDimension.EXTRAVERSION.value] + 0.1)
            
            # Analyze message engagement within conversation
            message_count = len(context.message_history)
            if message_count > 20:  # High engagement
                traits[PersonalityDimension.EXTRAVERSION.value] = min(0.9, traits[PersonalityDimension.EXTRAVERSION.value] + 0.1)
                traits[PersonalityDimension.AGREEABLENESS.value] = min(0.9, traits[PersonalityDimension.AGREEABLENESS.value] + 0.1)
        
        except Exception as e:
            logger.error(f"Error analyzing engagement patterns: {e}")
        
        return traits
    
    async def _get_user_session_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's recent session data."""
        try:
            # Query last 30 days of sessions
            cutoff_date = datetime.now() - timedelta(days=30)
            
            query = select(ConversationSession).where(
                ConversationSession.user_id == user_id,
                ConversationSession.created_at >= cutoff_date
            )
            
            result = await self.db.execute(query)
            sessions = result.scalars().all()
            
            return [
                {
                    'duration': session.duration_seconds,
                    'message_count': session.message_count,
                    'created_at': session.created_at.isoformat()
                }
                for session in sessions
            ]
        
        except Exception as e:
            logger.error(f"Error getting user session data: {e}")
            return []
    
    def _combine_trait_analyses(self, analyses: List[Tuple[Dict[str, float], float]]) -> Dict[str, float]:
        """Combine multiple trait analyses with weighted averaging."""
        combined = defaultdict(float)
        total_weight = sum(weight for _, weight in analyses)
        
        if total_weight == 0:
            return self._get_neutral_personality()
        
        for traits, weight in analyses:
            normalized_weight = weight / total_weight
            for trait, value in traits.items():
                combined[trait] += value * normalized_weight
        
        # Ensure all traits are present and normalized
        neutral = self._get_neutral_personality()
        for trait in neutral:
            if trait not in combined:
                combined[trait] = neutral[trait]
            else:
                combined[trait] = max(0.0, min(1.0, combined[trait]))
        
        return dict(combined)
    
    def _calculate_analysis_confidence(
        self, 
        messages: List[Dict[str, Any]], 
        traits: Dict[str, float]
    ) -> float:
        """Calculate confidence in personality analysis."""
        # Base confidence on amount of data
        message_count = len(messages)
        data_confidence = min(1.0, message_count / 50)  # Max confidence at 50 messages
        
        # Calculate text amount
        total_text_length = sum(len(msg.get('content', '')) for msg in messages)
        text_confidence = min(1.0, total_text_length / 5000)  # Max confidence at 5000 chars
        
        # Calculate trait consistency (how far from neutral)
        neutral_traits = self._get_neutral_personality()
        trait_deviations = []
        for trait, value in traits.items():
            neutral_value = neutral_traits.get(trait, 0.5)
            deviation = abs(value - neutral_value)
            trait_deviations.append(deviation)
        
        trait_confidence = np.mean(trait_deviations) * 2  # Amplify deviations
        
        # Combined confidence
        overall_confidence = (data_confidence * 0.4 + text_confidence * 0.3 + trait_confidence * 0.3)
        return min(1.0, max(0.1, overall_confidence))
    
    def _get_neutral_personality(self) -> Dict[str, float]:
        """Get neutral personality traits (all 0.5)."""
        return {
            PersonalityDimension.OPENNESS.value: 0.5,
            PersonalityDimension.CONSCIENTIOUSNESS.value: 0.5,
            PersonalityDimension.EXTRAVERSION.value: 0.5,
            PersonalityDimension.AGREEABLENESS.value: 0.5,
            PersonalityDimension.NEUROTICISM.value: 0.5,
            PersonalityDimension.HUMOR.value: 0.5,
            PersonalityDimension.EMPATHY.value: 0.5,
            PersonalityDimension.FORMALITY.value: 0.5,
            PersonalityDimension.DIRECTNESS.value: 0.5,
            PersonalityDimension.ENTHUSIASM.value: 0.5,
        }
    
    async def adapt_personality(
        self,
        user_id: str,
        user_traits: Dict[str, float],
        base_profile: PersonalityProfile,
        context: ConversationContext
    ) -> PersonalityState:
        """
        Adapt personality based on user traits and context.
        
        Uses reinforcement learning and multiple adaptation strategies
        to create optimal personality for each user interaction.
        """
        try:
            # Get or create user personality mapping
            mapping = await self._get_or_create_personality_mapping(user_id, base_profile.id)
            
            # Choose adaptation strategy based on performance history
            strategy = await self._select_optimal_strategy(user_id, mapping, context)
            
            # Apply adaptation algorithm
            adaptation_func = self.adaptation_algorithms.get(
                strategy, 
                self._balanced_adaptation
            )
            
            adapted_traits = await adaptation_func(
                user_traits, 
                base_profile.trait_scores, 
                mapping,
                context
            )
            
            # Apply reinforcement learning adjustments
            rl_adjusted_traits = await self._apply_reinforcement_learning(
                user_id,
                adapted_traits,
                mapping,
                context
            )
            
            # Create personality state
            personality_state = PersonalityState(
                base_traits=base_profile.trait_scores.copy(),
                adapted_traits=rl_adjusted_traits,
                confidence_level=mapping.adaptation_confidence,
                adaptation_history=mapping.interaction_history_summary.get('adaptations', []) if mapping.interaction_history_summary else [],
                effectiveness_metrics=mapping.engagement_metrics if mapping.engagement_metrics else {},
                last_updated=datetime.now()
            )
            
            # Cache the personality state
            await self._cache_personality_state(user_id, personality_state)
            
            logger.info(f"Adapted personality for user {user_id} using strategy {strategy}")
            return personality_state
            
        except Exception as e:
            logger.error(f"Error adapting personality for user {user_id}: {e}")
            # Return base personality as fallback
            return PersonalityState(
                base_traits=base_profile.trait_scores.copy(),
                adapted_traits=base_profile.trait_scores.copy(),
                confidence_level=0.5,
                adaptation_history=[],
                effectiveness_metrics={},
                last_updated=datetime.now()
            )
    
    async def _get_or_create_personality_mapping(
        self, 
        user_id: str, 
        profile_id: str
    ) -> UserPersonalityMapping:
        """Get existing personality mapping or create new one."""
        try:
            # Try to get existing mapping
            query = select(UserPersonalityMapping).where(
                UserPersonalityMapping.user_id == user_id,
                UserPersonalityMapping.profile_id == profile_id,
                UserPersonalityMapping.is_active == True
            )
            result = await self.db.execute(query)
            mapping = result.scalar_one_or_none()
            
            if mapping:
                return mapping
            
            # Create new mapping
            mapping = UserPersonalityMapping(
                user_id=user_id,
                profile_id=profile_id,
                measured_user_traits={},
                adapted_profile_traits={},
                interaction_history_summary={},
                adaptation_confidence=0.1,
                learning_iterations=0,
                satisfaction_scores={},
                engagement_metrics={},
                is_active=True,
                is_primary=True,  # First mapping for user
                usage_count=0
            )
            
            self.db.add(mapping)
            await self.db.commit()
            await self.db.refresh(mapping)
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error getting/creating personality mapping: {e}")
            raise
    
    async def _select_optimal_strategy(
        self,
        user_id: str,
        mapping: UserPersonalityMapping,
        context: ConversationContext
    ) -> AdaptationStrategy:
        """Select optimal adaptation strategy based on historical performance."""
        try:
            # If we don't have enough data, use balanced approach
            if mapping.learning_iterations < 5:
                return AdaptationStrategy.BALANCE
            
            # Get strategy performance history
            strategy_performance = {}
            if mapping.interaction_history_summary and 'strategy_performance' in mapping.interaction_history_summary:
                strategy_performance = mapping.interaction_history_summary['strategy_performance']
            
            if not strategy_performance:
                return AdaptationStrategy.BALANCE
            
            # Select strategy with highest average effectiveness
            best_strategy = AdaptationStrategy.BALANCE
            best_score = 0.0
            
            for strategy_name, performance_data in strategy_performance.items():
                if performance_data.get('usage_count', 0) > 0:
                    avg_effectiveness = performance_data.get('avg_effectiveness', 0.0)
                    if avg_effectiveness > best_score:
                        best_score = avg_effectiveness
                        best_strategy = AdaptationStrategy(strategy_name)
            
            # Add some exploration (epsilon-greedy)
            if np.random.random() < 0.1:  # 10% exploration
                strategies = list(AdaptationStrategy)
                return np.random.choice(strategies)
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error selecting adaptation strategy: {e}")
            return AdaptationStrategy.BALANCE
    
    async def _mirror_adaptation(
        self,
        user_traits: Dict[str, float],
        base_traits: Dict[str, float],
        mapping: UserPersonalityMapping,
        context: ConversationContext
    ) -> Dict[str, float]:
        """Mirror user's personality traits."""
        adapted_traits = base_traits.copy()
        adaptation_strength = min(0.8, 0.3 + mapping.adaptation_confidence * 0.5)
        
        for trait, base_value in base_traits.items():
            user_value = user_traits.get(trait, 0.5)
            # Move towards user's trait value
            adapted_value = base_value + (user_value - base_value) * adaptation_strength
            adapted_traits[trait] = max(0.1, min(0.9, adapted_value))
        
        return adapted_traits
    
    async def _complement_adaptation(
        self,
        user_traits: Dict[str, float],
        base_traits: Dict[str, float],
        mapping: UserPersonalityMapping,
        context: ConversationContext
    ) -> Dict[str, float]:
        """Complement user's personality traits."""
        adapted_traits = base_traits.copy()
        adaptation_strength = min(0.6, 0.2 + mapping.adaptation_confidence * 0.4)
        
        for trait, base_value in base_traits.items():
            user_value = user_traits.get(trait, 0.5)
            
            # Determine complement based on context
            if trait in [PersonalityDimension.EXTRAVERSION.value]:
                # If user is very introverted, be more extraverted
                complement_value = 1.0 - user_value
            elif trait in [PersonalityDimension.NEUROTICISM.value]:
                # If user is neurotic, be more stable
                complement_value = max(0.1, 0.8 - user_value)
            elif trait in [PersonalityDimension.AGREEABLENESS.value]:
                # Match agreeableness rather than complement
                complement_value = user_value
            else:
                # For other traits, provide moderate complement
                complement_value = 0.5 + (0.5 - user_value) * 0.3
            
            adapted_value = base_value + (complement_value - base_value) * adaptation_strength
            adapted_traits[trait] = max(0.1, min(0.9, adapted_value))
        
        return adapted_traits
    
    async def _balanced_adaptation(
        self,
        user_traits: Dict[str, float],
        base_traits: Dict[str, float],
        mapping: UserPersonalityMapping,
        context: ConversationContext
    ) -> Dict[str, float]:
        """Balanced adaptation combining mirroring and complementing."""
        adapted_traits = base_traits.copy()
        adaptation_strength = min(0.7, 0.25 + mapping.adaptation_confidence * 0.45)
        
        for trait, base_value in base_traits.items():
            user_value = user_traits.get(trait, 0.5)
            
            # Calculate mirror and complement values
            mirror_value = user_value
            if trait == PersonalityDimension.NEUROTICISM.value:
                complement_value = max(0.1, 0.8 - user_value)
            elif trait == PersonalityDimension.AGREEABLENESS.value:
                complement_value = user_value  # Match rather than complement
            else:
                complement_value = 0.5 + (0.5 - user_value) * 0.2
            
            # Weighted combination based on trait type
            if trait in [PersonalityDimension.AGREEABLENESS.value, PersonalityDimension.EMPATHY.value]:
                # Mirror more for social traits
                balanced_value = mirror_value * 0.8 + complement_value * 0.2
            elif trait == PersonalityDimension.NEUROTICISM.value:
                # Complement more for emotional stability
                balanced_value = mirror_value * 0.2 + complement_value * 0.8
            else:
                # Equal balance for other traits
                balanced_value = mirror_value * 0.6 + complement_value * 0.4
            
            adapted_value = base_value + (balanced_value - base_value) * adaptation_strength
            adapted_traits[trait] = max(0.1, min(0.9, adapted_value))
        
        return adapted_traits
    
    async def _custom_adaptation(
        self,
        user_traits: Dict[str, float],
        base_traits: Dict[str, float],
        mapping: UserPersonalityMapping,
        context: ConversationContext
    ) -> Dict[str, float]:
        """Custom adaptation based on learned user preferences."""
        # Start with balanced adaptation as base
        adapted_traits = await self._balanced_adaptation(user_traits, base_traits, mapping, context)
        
        # Apply custom adjustments based on interaction history
        if mapping.interaction_history_summary and 'preferences' in mapping.interaction_history_summary:
            preferences = mapping.interaction_history_summary['preferences']
            
            # Adjust traits based on learned preferences
            for trait, preference_data in preferences.items():
                if trait in adapted_traits:
                    preferred_value = preference_data.get('preferred_value', 0.5)
                    confidence = preference_data.get('confidence', 0.0)
                    
                    # Apply preference with confidence weighting
                    current_value = adapted_traits[trait]
                    adjustment = (preferred_value - current_value) * confidence * 0.3
                    adapted_traits[trait] = max(0.1, min(0.9, current_value + adjustment))
        
        return adapted_traits
    
    async def _apply_reinforcement_learning(
        self,
        user_id: str,
        adapted_traits: Dict[str, float],
        mapping: UserPersonalityMapping,
        context: ConversationContext
    ) -> Dict[str, float]:
        """Apply reinforcement learning adjustments to personality traits."""
        try:
            # Get Q-values for this user
            user_q_table = self.q_table[user_id]
            
            # Define state representation
            state_key = self._get_state_key(context, adapted_traits)
            
            # For each trait, consider small adjustments based on Q-learning
            final_traits = adapted_traits.copy()
            
            for trait, value in adapted_traits.items():
                # Define possible actions (small adjustments)
                actions = ['increase', 'decrease', 'maintain']
                action_values = {
                    'increase': user_q_table.get(f"{state_key}_{trait}_increase", 0.0),
                    'decrease': user_q_table.get(f"{state_key}_{trait}_decrease", 0.0),
                    'maintain': user_q_table.get(f"{state_key}_{trait}_maintain", 0.0),
                }
                
                # Epsilon-greedy action selection
                if np.random.random() < 0.1:  # 10% exploration
                    action = np.random.choice(actions)
                else:
                    action = max(action_values, key=action_values.get)
                
                # Apply action
                if action == 'increase':
                    final_traits[trait] = min(0.9, value + 0.05)
                elif action == 'decrease':
                    final_traits[trait] = max(0.1, value - 0.05)
                # 'maintain' does nothing
                
                # Store action for future reward learning
                action_key = f"{state_key}_{trait}_{action}"
                if user_id not in self.action_history:
                    self.action_history[user_id] = []
                self.action_history[user_id].append({
                    'action_key': action_key,
                    'timestamp': datetime.now(),
                    'trait': trait,
                    'action': action,
                    'value_before': value,
                    'value_after': final_traits[trait]
                })
            
            return final_traits
            
        except Exception as e:
            logger.error(f"Error applying reinforcement learning: {e}")
            return adapted_traits
    
    def _get_state_key(self, context: ConversationContext, traits: Dict[str, float]) -> str:
        """Generate state key for reinforcement learning."""
        # Simplify state representation for manageability
        sentiment_bucket = 'positive' if context.current_sentiment > 0.1 else 'negative' if context.current_sentiment < -0.1 else 'neutral'
        phase = context.conversation_phase
        urgency_bucket = 'high' if context.urgency_level > 0.7 else 'low' if context.urgency_level < 0.3 else 'medium'
        
        # Discretize key personality traits
        extraversion_bucket = 'high' if traits.get(PersonalityDimension.EXTRAVERSION.value, 0.5) > 0.6 else 'low' if traits.get(PersonalityDimension.EXTRAVERSION.value, 0.5) < 0.4 else 'medium'
        agreeableness_bucket = 'high' if traits.get(PersonalityDimension.AGREEABLENESS.value, 0.5) > 0.6 else 'low' if traits.get(PersonalityDimension.AGREEABLENESS.value, 0.5) < 0.4 else 'medium'
        
        return f"{sentiment_bucket}_{phase}_{urgency_bucket}_{extraversion_bucket}_{agreeableness_bucket}"
    
    async def learn_from_interaction(
        self,
        user_id: str,
        personality_state: PersonalityState,
        context: ConversationContext,
        interaction_outcome: Dict[str, Any]
    ) -> None:
        """
        Learn from interaction outcomes to improve personality adaptation.
        
        Uses the interaction outcome to update:
        - Reinforcement learning Q-values
        - User personality measurements
        - Adaptation strategy performance
        - A/B test results
        """
        try:
            # Calculate reward from interaction outcome
            reward = self._calculate_interaction_reward(interaction_outcome)
            
            # Update Q-learning values
            await self._update_q_values(user_id, reward, context, personality_state)
            
            # Update user personality measurements
            await self._update_user_personality_measurements(user_id, context, interaction_outcome)
            
            # Update adaptation strategy performance
            await self._update_strategy_performance(user_id, personality_state, reward)
            
            # Record A/B test results
            await self._record_ab_test_results(user_id, personality_state, interaction_outcome)
            
            # Update personality mapping in database
            await self._update_personality_mapping(user_id, personality_state, interaction_outcome)
            
            logger.info(f"Learned from interaction for user {user_id} with reward {reward}")
            
        except Exception as e:
            logger.error(f"Error learning from interaction for user {user_id}: {e}")
    
    def _calculate_interaction_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward signal from interaction outcome."""
        reward = 0.0
        
        # Engagement metrics
        if 'engagement_score' in outcome:
            reward += outcome['engagement_score'] * 0.3
        
        # User satisfaction
        if 'satisfaction_score' in outcome:
            reward += outcome['satisfaction_score'] * 0.4
        
        # Conversation continuation
        if outcome.get('continued_conversation', False):
            reward += 0.2
        
        # Response time (faster is better, up to a point)
        if 'response_time_ms' in outcome:
            response_time = outcome['response_time_ms']
            if response_time < 1000:  # Under 1 second
                reward += 0.1
            elif response_time > 5000:  # Over 5 seconds
                reward -= 0.1
        
        # User feedback
        if 'user_feedback' in outcome:
            feedback = outcome['user_feedback'].lower()
            if feedback in ['positive', 'good', 'helpful']:
                reward += 0.3
            elif feedback in ['negative', 'bad', 'unhelpful']:
                reward -= 0.3
        
        # Normalize reward to [-1, 1] range
        return max(-1.0, min(1.0, reward))
    
    async def _update_q_values(
        self,
        user_id: str,
        reward: float,
        context: ConversationContext,
        personality_state: PersonalityState
    ) -> None:
        """Update Q-learning values based on reward."""
        if user_id not in self.action_history:
            return
        
        # Get recent actions for this user
        recent_actions = [
            action for action in self.action_history[user_id]
            if datetime.now() - action['timestamp'] < timedelta(minutes=30)
        ]
        
        # Update Q-values for recent actions
        learning_rate = 0.1
        discount_factor = 0.9
        
        for action in recent_actions:
            action_key = action['action_key']
            current_q = self.q_table[user_id].get(action_key, 0.0)
            
            # Temporal difference learning
            # For simplicity, we don't estimate next state value
            new_q = current_q + learning_rate * (reward - current_q)
            self.q_table[user_id][action_key] = new_q
        
        # Record reward for analysis
        if user_id not in self.reward_history:
            self.reward_history[user_id] = deque(maxlen=100)
        self.reward_history[user_id].append({
            'reward': reward,
            'timestamp': datetime.now(),
            'actions': recent_actions.copy()
        })
        
        # Clear processed actions
        self.action_history[user_id] = [
            action for action in self.action_history[user_id]
            if datetime.now() - action['timestamp'] >= timedelta(minutes=30)
        ]
    
    async def _update_user_personality_measurements(
        self,
        user_id: str,
        context: ConversationContext,
        outcome: Dict[str, Any]
    ) -> None:
        """Update user personality measurements based on interaction."""
        try:
            # Get current mapping
            query = select(UserPersonalityMapping).where(
                UserPersonalityMapping.user_id == user_id,
                UserPersonalityMapping.is_active == True
            ).options(selectinload(UserPersonalityMapping.profile))
            
            result = await self.db.execute(query)
            mapping = result.scalar_one_or_none()
            
            if not mapping:
                return
            
            # Analyze latest messages for personality updates
            if context.message_history:
                latest_traits = await self.analyze_user_personality(user_id, context)
                
                # Update measurements with confidence weighting
                confidence_boost = min(0.1, outcome.get('engagement_score', 0.5) * 0.2)
                mapping.update_user_traits(latest_traits, confidence_boost)
            
            # Update interaction outcome in history
            if not mapping.interaction_history_summary:
                mapping.interaction_history_summary = {}
            
            history = mapping.interaction_history_summary
            if 'recent_outcomes' not in history:
                history['recent_outcomes'] = []
            
            history['recent_outcomes'].append({
                'timestamp': datetime.now().isoformat(),
                'outcome': outcome,
                'context_summary': {
                    'sentiment': context.current_sentiment,
                    'topic': context.topic,
                    'phase': context.conversation_phase,
                    'message_count': len(context.message_history)
                }
            })
            
            # Keep only recent outcomes
            if len(history['recent_outcomes']) > 50:
                history['recent_outcomes'] = history['recent_outcomes'][-50:]
            
            # Mark field as modified
            from sqlalchemy.orm import attributes
            attributes.flag_modified(mapping, 'interaction_history_summary')
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating user personality measurements: {e}")
    
    async def _update_strategy_performance(
        self,
        user_id: str,
        personality_state: PersonalityState,
        reward: float
    ) -> None:
        """Update adaptation strategy performance metrics."""
        try:
            query = select(UserPersonalityMapping).where(
                UserPersonalityMapping.user_id == user_id,
                UserPersonalityMapping.is_active == True
            )
            
            result = await self.db.execute(query)
            mapping = result.scalar_one_or_none()
            
            if not mapping:
                return
            
            # Determine which strategy was used (simplified)
            # In practice, you'd track this more precisely
            current_strategy = AdaptationStrategy.BALANCE  # Default assumption
            
            if not mapping.interaction_history_summary:
                mapping.interaction_history_summary = {}
            
            history = mapping.interaction_history_summary
            if 'strategy_performance' not in history:
                history['strategy_performance'] = {}
            
            strategy_name = current_strategy.value
            if strategy_name not in history['strategy_performance']:
                history['strategy_performance'][strategy_name] = {
                    'total_reward': 0.0,
                    'usage_count': 0,
                    'avg_effectiveness': 0.0
                }
            
            perf = history['strategy_performance'][strategy_name]
            perf['total_reward'] += reward
            perf['usage_count'] += 1
            perf['avg_effectiveness'] = perf['total_reward'] / perf['usage_count']
            
            # Mark field as modified
            from sqlalchemy.orm import attributes
            attributes.flag_modified(mapping, 'interaction_history_summary')
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    async def _record_ab_test_results(
        self,
        user_id: str,
        personality_state: PersonalityState,
        outcome: Dict[str, Any]
    ) -> None:
        """Record A/B test results for personality optimization."""
        # Implementation for A/B testing framework
        # This would track different personality variations and their performance
        pass
    
    async def _update_personality_mapping(
        self,
        user_id: str,
        personality_state: PersonalityState,
        outcome: Dict[str, Any]
    ) -> None:
        """Update personality mapping with interaction results."""
        try:
            query = select(UserPersonalityMapping).where(
                UserPersonalityMapping.user_id == user_id,
                UserPersonalityMapping.is_active == True
            )
            
            result = await self.db.execute(query)
            mapping = result.scalar_one_or_none()
            
            if not mapping:
                return
            
            # Update adapted traits
            mapping.adapted_profile_traits = personality_state.adapted_traits
            
            # Record interaction
            engagement_data = {
                'engagement_score': outcome.get('engagement_score', 0.5),
                'response_time': outcome.get('response_time_ms', 0),
                'satisfaction': outcome.get('satisfaction_score', 0.5)
            }
            
            satisfaction_score = outcome.get('satisfaction_score', 0.5)
            mapping.record_interaction(satisfaction_score, engagement_data)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating personality mapping: {e}")
    
    async def _cache_personality_state(self, user_id: str, state: PersonalityState) -> None:
        """Cache personality state in Redis."""
        try:
            cache_key = f"personality_state:{user_id}"
            cache_data = {
                'base_traits': state.base_traits,
                'adapted_traits': state.adapted_traits,
                'confidence_level': state.confidence_level,
                'last_updated': state.last_updated.isoformat()
            }
            
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching personality state: {e}")
    
    async def generate_personality_response(
        self,
        personality_state: PersonalityState,
        context: ConversationContext,
        base_response: str
    ) -> str:
        """
        Generate personality-adapted response based on current personality state.
        
        Modifies the base response to match the adapted personality traits.
        """
        try:
            # Get personality adjustments
            traits = personality_state.adapted_traits
            
            # Apply trait-based modifications
            modified_response = base_response
            
            # Adjust for extraversion
            extraversion = traits.get(PersonalityDimension.EXTRAVERSION.value, 0.5)
            if extraversion > 0.7:
                # More enthusiastic and expressive
                modified_response = self._make_more_extraverted(modified_response)
            elif extraversion < 0.3:
                # More reserved and concise
                modified_response = self._make_more_introverted(modified_response)
            
            # Adjust for agreeableness
            agreeableness = traits.get(PersonalityDimension.AGREEABLENESS.value, 0.5)
            if agreeableness > 0.7:
                modified_response = self._make_more_agreeable(modified_response)
            elif agreeableness < 0.3:
                modified_response = self._make_more_direct(modified_response)
            
            # Adjust for humor
            humor = traits.get(PersonalityDimension.HUMOR.value, 0.5)
            if humor > 0.7 and context.conversation_phase != 'urgent':
                modified_response = self._add_humor(modified_response)
            
            # Adjust for formality
            formality = traits.get(PersonalityDimension.FORMALITY.value, 0.5)
            if formality > 0.7:
                modified_response = self._make_more_formal(modified_response)
            elif formality < 0.3:
                modified_response = self._make_more_casual(modified_response)
            
            # Adjust for empathy
            empathy = traits.get(PersonalityDimension.EMPATHY.value, 0.5)
            if empathy > 0.7 and context.current_sentiment < 0:
                modified_response = self._add_empathy(modified_response)
            
            return modified_response
            
        except Exception as e:
            logger.error(f"Error generating personality response: {e}")
            return base_response
    
    def _make_more_extraverted(self, response: str) -> str:
        """Make response more extraverted."""
        # Add excitement and enthusiasm
        if not response.endswith(('!', '?', '.')):
            response += '!'
        
        # Add energetic words
        energetic_replacements = {
            'good': 'great',
            'ok': 'awesome',
            'fine': 'fantastic',
            'yes': 'absolutely',
            'sure': 'definitely'
        }
        
        for old, new in energetic_replacements.items():
            response = response.replace(old, new)
        
        return response
    
    def _make_more_introverted(self, response: str) -> str:
        """Make response more introverted."""
        # Make more concise and thoughtful
        words = response.split()
        if len(words) > 20:
            # Shorten response
            key_sentences = response.split('.')[:2]
            response = '. '.join(key_sentences) + '.'
        
        # Remove excessive enthusiasm
        response = response.replace('!!', '.').replace('!', '.')
        
        return response
    
    def _make_more_agreeable(self, response: str) -> str:
        """Make response more agreeable."""
        # Add supportive language
        supportive_phrases = [
            "I understand",
            "That makes sense",
            "I can see why you'd feel that way"
        ]
        
        # Add validation if addressing concerns
        if any(word in response.lower() for word in ['problem', 'issue', 'concern', 'worry']):
            validation = np.random.choice(supportive_phrases)
            response = f"{validation}. {response}"
        
        return response
    
    def _make_more_direct(self, response: str) -> str:
        """Make response more direct."""
        # Remove hedging language
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        for hedge in hedging_words:
            response = response.replace(hedge, '')
        
        # Clean up extra spaces
        response = ' '.join(response.split())
        
        return response
    
    def _add_humor(self, response: str) -> str:
        """Add appropriate humor to response."""
        # Simple humor injection - in practice, this would be more sophisticated
        humor_additions = [
            "😄",
            "😊", 
            "(just kidding!)",
            "haha"
        ]
        
        # Add humor if response is not too serious
        if not any(serious in response.lower() for serious in ['error', 'problem', 'sorry', 'urgent']):
            humor = np.random.choice(humor_additions)
            response += f" {humor}"
        
        return response
    
    def _make_more_formal(self, response: str) -> str:
        """Make response more formal."""
        # Replace informal contractions
        formal_replacements = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not",
            "aren't": "are not",
            "you're": "you are",
            "we're": "we are",
            "it's": "it is"
        }
        
        for informal, formal in formal_replacements.items():
            response = response.replace(informal, formal)
        
        return response
    
    def _make_more_casual(self, response: str) -> str:
        """Make response more casual."""
        # Add casual contractions
        casual_replacements = {
            "cannot": "can't",
            "will not": "won't", 
            "do not": "don't",
            "is not": "isn't",
            "are not": "aren't",
            "you are": "you're",
            "we are": "we're",
            "it is": "it's"
        }
        
        for formal, casual in casual_replacements.items():
            response = response.replace(formal, casual)
        
        return response
    
    def _add_empathy(self, response: str) -> str:
        """Add empathetic language to response."""
        empathetic_phrases = [
            "I'm here to help",
            "I understand this might be frustrating",
            "Let me help you with this",
            "I can see this is important to you"
        ]
        
        # Prepend with empathetic phrase
        empathy = np.random.choice(empathetic_phrases)
        response = f"{empathy}. {response}"
        
        return response


class PersonalityClassifierNN(nn.Module):
    """Neural network for personality trait classification."""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class RuleBasedPersonalityClassifier:
    """Fallback rule-based personality classifier."""
    
    def predict(self, embeddings):
        # Simple rule-based fallback
        # Returns neutral personality with slight random variation
        base_traits = np.array([0.5] * 10)
        noise = np.random.normal(0, 0.1, 10)
        traits = np.clip(base_traits + noise, 0.1, 0.9)
        return [traits]


# Export the main class
__all__ = ['AdvancedPersonalityEngine', 'ConversationContext', 'PersonalityState', 'PersonalityMetrics']