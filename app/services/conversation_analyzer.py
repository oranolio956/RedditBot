"""
Conversation Analysis Service

Real-time conversation analysis for personality adaptation and user understanding.
Provides comprehensive analysis of:
- Conversation flow and context
- Emotional state tracking
- Topic detection and transitions
- User engagement patterns
- Conversation quality metrics
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging

# ML and NLP imports
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Redis for real-time data
from redis.asyncio import Redis

# Internal imports
from app.models.conversation import Message, ConversationSession, Conversation
from app.models.user import User
from app.core.redis import get_redis_client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Comprehensive conversation quality and engagement metrics."""
    engagement_score: float = 0.5
    sentiment_trajectory: List[float] = field(default_factory=list)
    topic_coherence: float = 0.5
    conversation_depth: int = 0
    user_satisfaction_indicators: Dict[str, float] = field(default_factory=dict)
    response_quality: float = 0.5
    emotional_intelligence: float = 0.5
    rapport_building: float = 0.5


@dataclass
class TopicAnalysis:
    """Analysis of conversation topics and transitions."""
    current_topic: Optional[str] = None
    topic_confidence: float = 0.0
    topic_history: List[Dict[str, Any]] = field(default_factory=list)
    topic_transitions: List[Dict[str, Any]] = field(default_factory=list)
    topic_depth: int = 0
    topic_engagement: float = 0.5


@dataclass
class EmotionalState:
    """Current and historical emotional state of the conversation."""
    current_emotions: Dict[str, float] = field(default_factory=dict)
    emotion_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    emotional_volatility: float = 0.0
    dominant_emotion: Optional[str] = None
    emotional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Complete conversation context for personality adaptation."""
    session_id: str
    user_id: str
    conversation_phase: str  # 'opening', 'building', 'deep', 'closing'
    time_in_conversation: int  # seconds
    message_count: int
    user_engagement_level: float
    urgency_indicators: Dict[str, float] = field(default_factory=dict)
    context_switches: int = 0
    conversation_goals: List[str] = field(default_factory=list)


class ConversationAnalyzer:
    """
    Advanced conversation analysis engine for AI personality systems.
    
    Provides real-time analysis of conversations to enable:
    - Intelligent personality adaptation
    - Context-aware responses
    - Emotional intelligence
    - Conversation flow optimization
    - User satisfaction tracking
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.settings = get_settings()
        
        # NLP models - initialized asynchronously
        self.sentiment_analyzer = None
        self.emotion_classifier = None
        self.topic_classifier = None
        self.intent_classifier = None
        self.toxicity_classifier = None
        
        # Text processing tools
        self.nlp = None  # spaCy model
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Real-time conversation state
        self.active_conversations = {}  # session_id -> ConversationState
        self.conversation_cache = {}    # session_id -> cached analysis
        
        # Topic modeling and clustering
        self.topic_clusters = None
        self.topic_keywords = {}
        
        # Quality assessment models
        self.quality_metrics = {
            'coherence': self._assess_coherence,
            'relevance': self._assess_relevance,
            'helpfulness': self._assess_helpfulness,
            'engagement': self._assess_engagement,
            'empathy': self._assess_empathy
        }
        
        logger.info("Conversation analyzer initialized")
    
    async def initialize_models(self) -> None:
        """Initialize ML models and NLP tools."""
        try:
            logger.info("Loading conversation analysis models...")
            
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Emotion classification
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Intent classification
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Topic classification
            self.topic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Toxicity detection
            try:
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="martin-ha/toxic-comment-model",
                    return_all_scores=True
                )
            except:
                logger.warning("Toxicity classifier not available, using fallback")
                self.toxicity_classifier = None
            
            # Load spaCy model for NER and linguistic analysis
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, some features will be limited")
                self.nlp = None
            
            # Initialize topic modeling
            await self._initialize_topic_modeling()
            
            logger.info("All conversation analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing conversation analysis models: {e}")
            # Continue with limited functionality
    
    async def _initialize_topic_modeling(self) -> None:
        """Initialize topic modeling and clustering."""
        try:
            # Pre-defined topic categories for zero-shot classification
            self.topic_categories = [
                "technology", "business", "health", "entertainment", "sports",
                "politics", "science", "education", "travel", "food",
                "relationships", "career", "finance", "hobbies", "news",
                "shopping", "weather", "transportation", "support", "general"
            ]
            
            # Intent categories
            self.intent_categories = [
                "question", "request", "complaint", "compliment", "information_seeking",
                "help_needed", "casual_chat", "urgent", "feedback", "goodbye",
                "greeting", "clarification", "agreement", "disagreement", "suggestion"
            ]
            
        except Exception as e:
            logger.error(f"Error initializing topic modeling: {e}")
    
    async def analyze_conversation_context(
        self,
        session_id: str,
        messages: List[Message],
        user_data: Optional[User] = None
    ) -> ConversationContext:
        """
        Analyze complete conversation context for personality adaptation.
        
        Provides comprehensive analysis of conversation state including:
        - Current conversation phase
        - User engagement patterns
        - Topic flow and coherence
        - Emotional trajectory
        - Urgency indicators
        """
        try:
            if not messages:
                return self._get_default_context(session_id, user_data.id if user_data else None)
            
            # Basic conversation metrics
            message_count = len(messages)
            conversation_start = messages[0].created_at if messages else datetime.now()
            time_in_conversation = int((datetime.now() - conversation_start).total_seconds())
            
            # Determine conversation phase
            phase = await self._determine_conversation_phase(messages, time_in_conversation)
            
            # Calculate user engagement
            engagement_level = await self._calculate_user_engagement(messages, user_data)
            
            # Detect urgency indicators
            urgency_indicators = await self._detect_urgency_indicators(messages)
            
            # Count context switches
            context_switches = await self._count_context_switches(messages)
            
            # Identify conversation goals
            goals = await self._identify_conversation_goals(messages)
            
            context = ConversationContext(
                session_id=session_id,
                user_id=user_data.id if user_data else "unknown",
                conversation_phase=phase,
                time_in_conversation=time_in_conversation,
                message_count=message_count,
                user_engagement_level=engagement_level,
                urgency_indicators=urgency_indicators,
                context_switches=context_switches,
                conversation_goals=goals
            )
            
            # Cache the context
            await self._cache_conversation_context(session_id, context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            return self._get_default_context(session_id, user_data.id if user_data else None)
    
    async def _determine_conversation_phase(self, messages: List[Message], duration: int) -> str:
        """Determine current phase of conversation."""
        message_count = len(messages)
        
        # Opening phase: First few exchanges
        if message_count <= 4 or duration < 120:  # Less than 2 minutes
            return "opening"
        
        # Deep phase: Extended engagement with complex topics
        if message_count >= 20 and duration > 600:  # More than 10 minutes
            # Check for deep engagement indicators
            recent_messages = messages[-10:]
            avg_length = np.mean([len(msg.content or "") for msg in recent_messages])
            if avg_length > 100:  # Longer, more detailed messages
                return "deep"
        
        # Closing phase: Conversation winding down
        if message_count > 10:
            recent_messages = messages[-5:]
            closing_indicators = [
                "goodbye", "bye", "thanks", "thank you", "see you",
                "take care", "have a good", "talk later", "gotta go"
            ]
            
            recent_text = " ".join([msg.content or "" for msg in recent_messages]).lower()
            if any(indicator in recent_text for indicator in closing_indicators):
                return "closing"
        
        # Default: building phase
        return "building"
    
    async def _calculate_user_engagement(self, messages: List[Message], user_data: Optional[User]) -> float:
        """Calculate user engagement level based on conversation patterns."""
        if not messages:
            return 0.5
        
        engagement_score = 0.0
        total_factors = 0
        
        # Message frequency
        if len(messages) > 1:
            timestamps = [msg.created_at for msg in messages if msg.created_at]
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                             for i in range(1, len(timestamps))]
                avg_response_time = np.mean(time_diffs)
                
                # Quick responses indicate high engagement
                if avg_response_time < 30:
                    engagement_score += 0.8
                elif avg_response_time < 120:
                    engagement_score += 0.6
                else:
                    engagement_score += 0.3
                total_factors += 1
        
        # Message length and complexity
        user_messages = [msg for msg in messages if msg.direction.value == "incoming"]
        if user_messages:
            avg_length = np.mean([len(msg.content or "") for msg in user_messages])
            if avg_length > 100:
                engagement_score += 0.7
            elif avg_length > 50:
                engagement_score += 0.5
            else:
                engagement_score += 0.3
            total_factors += 1
        
        # Question asking (indicates curiosity/engagement)
        question_count = sum(1 for msg in user_messages if "?" in (msg.content or ""))
        question_ratio = question_count / max(len(user_messages), 1)
        if question_ratio > 0.3:
            engagement_score += 0.8
        elif question_ratio > 0.1:
            engagement_score += 0.6
        total_factors += 1
        
        # Emotional expression
        emotional_messages = 0
        for msg in user_messages:
            content = (msg.content or "").lower()
            if any(indicator in content for indicator in ["!", "wow", "amazing", "terrible", "love", "hate"]):
                emotional_messages += 1
        
        emotion_ratio = emotional_messages / max(len(user_messages), 1)
        engagement_score += min(0.8, emotion_ratio * 2)
        total_factors += 1
        
        return engagement_score / max(total_factors, 1)
    
    async def _detect_urgency_indicators(self, messages: List[Message]) -> Dict[str, float]:
        """Detect urgency indicators in conversation."""
        urgency_indicators = {
            'time_pressure': 0.0,
            'emotional_urgency': 0.0,
            'explicit_urgency': 0.0,
            'situational_urgency': 0.0
        }
        
        if not messages:
            return urgency_indicators
        
        recent_messages = messages[-5:]  # Focus on recent messages
        combined_text = " ".join([msg.content or "" for msg in recent_messages]).lower()
        
        # Explicit urgency keywords
        urgent_keywords = [
            "urgent", "emergency", "asap", "immediately", "right now",
            "quickly", "fast", "hurry", "rush", "deadline", "critical"
        ]
        urgency_indicators['explicit_urgency'] = min(1.0, sum(
            combined_text.count(keyword) for keyword in urgent_keywords
        ) / 10)
        
        # Time pressure indicators
        time_keywords = [
            "soon", "today", "now", "waiting", "delayed", "late",
            "tomorrow", "deadline", "expires", "due"
        ]
        urgency_indicators['time_pressure'] = min(1.0, sum(
            combined_text.count(keyword) for keyword in time_keywords
        ) / 10)
        
        # Emotional urgency (high emotion + negative sentiment)
        if self.sentiment_analyzer and self.emotion_classifier:
            try:
                sentiment_results = self.sentiment_analyzer(combined_text)
                emotion_results = self.emotion_classifier(combined_text)
                
                # Check for negative sentiment with high intensity
                negative_score = 0.0
                for result in sentiment_results[0]:
                    if result['label'] == 'LABEL_0':  # Negative
                        negative_score = result['score']
                
                # Check for stress-related emotions
                stress_emotions = ['anger', 'fear', 'sadness']
                stress_score = sum(
                    emotion['score'] for emotion in emotion_results[0]
                    if emotion['label'] in stress_emotions
                )
                
                urgency_indicators['emotional_urgency'] = min(1.0, negative_score * stress_score * 2)
                
            except Exception as e:
                logger.error(f"Error analyzing emotional urgency: {e}")
        
        # Situational urgency (context-based)
        situation_keywords = [
            "problem", "issue", "error", "broken", "failed", "stuck",
            "help", "support", "trouble", "difficulty"
        ]
        urgency_indicators['situational_urgency'] = min(1.0, sum(
            combined_text.count(keyword) for keyword in situation_keywords
        ) / 15)
        
        return urgency_indicators
    
    async def _count_context_switches(self, messages: List[Message]) -> int:
        """Count number of topic/context switches in conversation."""
        if len(messages) < 4:
            return 0
        
        try:
            # Get topics for each message
            topics = []
            for msg in messages:
                if msg.content and len(msg.content.strip()) > 10:
                    topic = await self._classify_message_topic(msg.content)
                    topics.append(topic)
            
            # Count switches
            switches = 0
            for i in range(1, len(topics)):
                if topics[i] != topics[i-1]:
                    switches += 1
            
            return switches
            
        except Exception as e:
            logger.error(f"Error counting context switches: {e}")
            return 0
    
    async def _classify_message_topic(self, text: str) -> str:
        """Classify the topic of a message."""
        try:
            if self.topic_classifier and len(text.strip()) > 10:
                result = self.topic_classifier(text, self.topic_categories)
                return result['labels'][0] if result['scores'][0] > 0.3 else 'general'
            return 'general'
        except Exception as e:
            logger.error(f"Error classifying message topic: {e}")
            return 'general'
    
    async def _identify_conversation_goals(self, messages: List[Message]) -> List[str]:
        """Identify apparent goals of the conversation."""
        goals = []
        
        if not messages:
            return goals
        
        # Analyze user messages for goal indicators
        user_messages = [msg for msg in messages if msg.direction.value == "incoming"]
        combined_text = " ".join([msg.content or "" for msg in user_messages]).lower()
        
        # Goal detection patterns
        goal_patterns = {
            'information_seeking': ['how', 'what', 'when', 'where', 'why', 'explain', 'tell me'],
            'problem_solving': ['problem', 'issue', 'fix', 'solve', 'help', 'broken', 'error'],
            'learning': ['learn', 'understand', 'teach', 'show me', 'tutorial', 'guide'],
            'decision_making': ['should', 'choose', 'decide', 'recommend', 'suggest', 'advice'],
            'entertainment': ['fun', 'joke', 'story', 'chat', 'talk', 'bored'],
            'support': ['frustrated', 'upset', 'difficult', 'stressed', 'worried', 'concern']
        }
        
        for goal, keywords in goal_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                goals.append(goal)
        
        return goals[:3]  # Limit to top 3 goals
    
    def _get_default_context(self, session_id: str, user_id: Optional[str]) -> ConversationContext:
        """Get default conversation context."""
        return ConversationContext(
            session_id=session_id,
            user_id=user_id or "unknown",
            conversation_phase="opening",
            time_in_conversation=0,
            message_count=0,
            user_engagement_level=0.5,
            urgency_indicators={},
            context_switches=0,
            conversation_goals=[]
        )
    
    async def analyze_emotional_state(self, messages: List[Message]) -> EmotionalState:
        """Analyze emotional state trajectory of conversation."""
        try:
            if not messages or not self.emotion_classifier:
                return EmotionalState()
            
            # Analyze emotions for recent messages
            recent_messages = messages[-10:]  # Last 10 messages
            emotion_trajectory = []
            all_emotions = defaultdict(list)
            
            for msg in recent_messages:
                if not msg.content or len(msg.content.strip()) < 10:
                    continue
                
                try:
                    emotion_results = self.emotion_classifier(msg.content)
                    emotions = {emotion['label']: emotion['score'] 
                              for emotion in emotion_results[0]}
                    
                    emotion_trajectory.append({
                        'timestamp': msg.created_at.isoformat() if msg.created_at else None,
                        'emotions': emotions,
                        'direction': msg.direction.value
                    })
                    
                    # Collect for overall analysis
                    for emotion, score in emotions.items():
                        all_emotions[emotion].append(score)
                        
                except Exception as e:
                    logger.error(f"Error analyzing emotion for message: {e}")
                    continue
            
            # Calculate current dominant emotions
            current_emotions = {}
            dominant_emotion = None
            max_emotion_score = 0.0
            
            for emotion, scores in all_emotions.items():
                if scores:
                    avg_score = np.mean(scores)
                    current_emotions[emotion] = avg_score
                    if avg_score > max_emotion_score:
                        max_emotion_score = avg_score
                        dominant_emotion = emotion
            
            # Calculate emotional volatility
            emotional_volatility = 0.0
            if len(emotion_trajectory) > 1:
                volatilities = []
                for emotion in current_emotions.keys():
                    emotion_values = [
                        entry['emotions'].get(emotion, 0.0) 
                        for entry in emotion_trajectory
                    ]
                    if len(emotion_values) > 1:
                        volatilities.append(np.std(emotion_values))
                
                emotional_volatility = np.mean(volatilities) if volatilities else 0.0
            
            return EmotionalState(
                current_emotions=current_emotions,
                emotion_trajectory=emotion_trajectory,
                emotional_volatility=emotional_volatility,
                dominant_emotion=dominant_emotion,
                emotional_context=self._analyze_emotional_context(emotion_trajectory)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing emotional state: {e}")
            return EmotionalState()
    
    def _analyze_emotional_context(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional context from trajectory."""
        if not trajectory:
            return {}
        
        context = {
            'emotional_trend': 'stable',
            'emotion_triggers': [],
            'recovery_patterns': {},
            'emotional_peaks': []
        }
        
        try:
            # Analyze emotional trend
            if len(trajectory) >= 3:
                positive_emotions = ['joy', 'love', 'surprise']
                negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
                
                positive_trend = []
                negative_trend = []
                
                for entry in trajectory:
                    pos_score = sum(entry['emotions'].get(em, 0) for em in positive_emotions)
                    neg_score = sum(entry['emotions'].get(em, 0) for em in negative_emotions)
                    
                    positive_trend.append(pos_score)
                    negative_trend.append(neg_score)
                
                # Determine trend
                pos_slope = np.polyfit(range(len(positive_trend)), positive_trend, 1)[0]
                neg_slope = np.polyfit(range(len(negative_trend)), negative_trend, 1)[0]
                
                if pos_slope > 0.05:
                    context['emotional_trend'] = 'improving'
                elif neg_slope > 0.05:
                    context['emotional_trend'] = 'declining'
                else:
                    context['emotional_trend'] = 'stable'
            
        except Exception as e:
            logger.error(f"Error analyzing emotional context: {e}")
        
        return context
    
    async def analyze_topic_flow(self, messages: List[Message]) -> TopicAnalysis:
        """Analyze topic flow and coherence in conversation."""
        try:
            if not messages:
                return TopicAnalysis()
            
            # Get topics for messages
            topic_history = []
            current_topic = None
            topic_transitions = []
            
            for i, msg in enumerate(messages):
                if not msg.content or len(msg.content.strip()) < 10:
                    continue
                
                topic = await self._classify_message_topic(msg.content)
                confidence = 0.7  # Simplified confidence
                
                topic_entry = {
                    'message_index': i,
                    'topic': topic,
                    'confidence': confidence,
                    'timestamp': msg.created_at.isoformat() if msg.created_at else None,
                    'direction': msg.direction.value
                }
                topic_history.append(topic_entry)
                
                # Track transitions
                if current_topic and current_topic != topic:
                    topic_transitions.append({
                        'from_topic': current_topic,
                        'to_topic': topic,
                        'message_index': i,
                        'transition_type': await self._classify_transition_type(
                            messages[i-1] if i > 0 else None, msg
                        )
                    })
                
                current_topic = topic
            
            # Calculate topic coherence
            topic_coherence = self._calculate_topic_coherence(topic_history)
            
            # Calculate topic depth
            topic_depth = self._calculate_topic_depth(topic_history, messages)
            
            # Calculate topic engagement
            topic_engagement = self._calculate_topic_engagement(topic_history, messages)
            
            return TopicAnalysis(
                current_topic=current_topic,
                topic_confidence=topic_history[-1]['confidence'] if topic_history else 0.0,
                topic_history=topic_history,
                topic_transitions=topic_transitions,
                topic_depth=topic_depth,
                topic_engagement=topic_engagement
            )
            
        except Exception as e:
            logger.error(f"Error analyzing topic flow: {e}")
            return TopicAnalysis()
    
    async def _classify_transition_type(
        self, 
        prev_msg: Optional[Message], 
        curr_msg: Message
    ) -> str:
        """Classify the type of topic transition."""
        if not prev_msg:
            return 'initial'
        
        # Simple heuristics for transition types
        prev_content = (prev_msg.content or "").lower()
        curr_content = (curr_msg.content or "").lower()
        
        # Question-driven transition
        if '?' in curr_content:
            return 'question_driven'
        
        # Natural flow transition
        transition_words = ['by the way', 'speaking of', 'that reminds me', 'also', 'additionally']
        if any(word in curr_content for word in transition_words):
            return 'natural_flow'
        
        # Abrupt change
        if len(set(prev_content.split()) & set(curr_content.split())) < 2:
            return 'abrupt_change'
        
        return 'gradual_shift'
    
    def _calculate_topic_coherence(self, topic_history: List[Dict[str, Any]]) -> float:
        """Calculate topic coherence score."""
        if len(topic_history) < 2:
            return 1.0
        
        # Simple coherence: fewer topic switches = higher coherence
        topic_switches = 0
        for i in range(1, len(topic_history)):
            if topic_history[i]['topic'] != topic_history[i-1]['topic']:
                topic_switches += 1
        
        # Normalize by conversation length
        coherence = 1.0 - (topic_switches / len(topic_history))
        return max(0.0, min(1.0, coherence))
    
    def _calculate_topic_depth(self, topic_history: List[Dict[str, Any]], messages: List[Message]) -> int:
        """Calculate how deep the conversation goes into topics."""
        if not topic_history:
            return 0
        
        # Count consecutive messages on same topic
        max_depth = 0
        current_depth = 1
        current_topic = topic_history[0]['topic']
        
        for entry in topic_history[1:]:
            if entry['topic'] == current_topic:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            else:
                current_depth = 1
                current_topic = entry['topic']
        
        return max_depth
    
    def _calculate_topic_engagement(
        self, 
        topic_history: List[Dict[str, Any]], 
        messages: List[Message]
    ) -> float:
        """Calculate engagement level for different topics."""
        if not topic_history or not messages:
            return 0.5
        
        # Calculate average message length by topic
        topic_engagement = {}
        topic_message_counts = defaultdict(int)
        topic_total_lengths = defaultdict(int)
        
        for i, entry in enumerate(topic_history):
            if i < len(messages):
                msg = messages[entry['message_index']]
                topic = entry['topic']
                
                if msg.content:
                    topic_message_counts[topic] += 1
                    topic_total_lengths[topic] += len(msg.content)
        
        # Calculate average engagement
        total_engagement = 0.0
        topic_count = 0
        
        for topic in topic_message_counts:
            if topic_message_counts[topic] > 0:
                avg_length = topic_total_lengths[topic] / topic_message_counts[topic]
                # Normalize engagement score
                engagement = min(1.0, avg_length / 100)  # 100 chars = full engagement
                topic_engagement[topic] = engagement
                total_engagement += engagement
                topic_count += 1
        
        return total_engagement / max(topic_count, 1)
    
    async def calculate_conversation_metrics(
        self,
        messages: List[Message],
        emotional_state: EmotionalState,
        topic_analysis: TopicAnalysis,
        context: ConversationContext
    ) -> ConversationMetrics:
        """Calculate comprehensive conversation quality metrics."""
        try:
            metrics = ConversationMetrics()
            
            # Calculate engagement score
            metrics.engagement_score = context.user_engagement_level
            
            # Build sentiment trajectory
            if messages:
                metrics.sentiment_trajectory = await self._build_sentiment_trajectory(messages)
            
            # Topic coherence from topic analysis
            metrics.topic_coherence = topic_analysis.topic_coherence
            
            # Conversation depth
            metrics.conversation_depth = topic_analysis.topic_depth
            
            # User satisfaction indicators
            metrics.user_satisfaction_indicators = await self._assess_user_satisfaction(messages)
            
            # Response quality assessment
            metrics.response_quality = await self._assess_overall_response_quality(messages)
            
            # Emotional intelligence score
            metrics.emotional_intelligence = await self._assess_emotional_intelligence(
                messages, emotional_state
            )
            
            # Rapport building score
            metrics.rapport_building = await self._assess_rapport_building(messages, context)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating conversation metrics: {e}")
            return ConversationMetrics()
    
    async def _build_sentiment_trajectory(self, messages: List[Message]) -> List[float]:
        """Build sentiment trajectory for conversation."""
        trajectory = []
        
        if not self.sentiment_analyzer:
            return trajectory
        
        try:
            for msg in messages:
                if not msg.content or len(msg.content.strip()) < 5:
                    continue
                
                sentiment_results = self.sentiment_analyzer(msg.content)
                
                # Convert to numerical score
                sentiment_score = 0.0
                for result in sentiment_results[0]:
                    if result['label'] == 'LABEL_2':  # Positive
                        sentiment_score += result['score']
                    elif result['label'] == 'LABEL_0':  # Negative
                        sentiment_score -= result['score']
                
                trajectory.append(sentiment_score)
            
        except Exception as e:
            logger.error(f"Error building sentiment trajectory: {e}")
        
        return trajectory
    
    async def _assess_user_satisfaction(self, messages: List[Message]) -> Dict[str, float]:
        """Assess various indicators of user satisfaction."""
        indicators = {
            'positive_feedback': 0.0,
            'engagement_maintenance': 0.5,
            'goal_achievement': 0.5,
            'emotional_improvement': 0.5
        }
        
        if not messages:
            return indicators
        
        try:
            user_messages = [msg for msg in messages if msg.direction.value == "incoming"]
            
            # Positive feedback indicators
            positive_words = [
                'thank', 'thanks', 'helpful', 'great', 'excellent', 'perfect',
                'amazing', 'wonderful', 'good', 'nice', 'appreciate'
            ]
            
            combined_text = " ".join([msg.content or "" for msg in user_messages]).lower()
            positive_count = sum(combined_text.count(word) for word in positive_words)
            indicators['positive_feedback'] = min(1.0, positive_count / max(len(user_messages), 1))
            
            # Engagement maintenance
            if len(messages) > 10:
                early_engagement = self._calculate_engagement_subset(messages[:5])
                late_engagement = self._calculate_engagement_subset(messages[-5:])
                
                if early_engagement > 0:
                    indicators['engagement_maintenance'] = late_engagement / early_engagement
                else:
                    indicators['engagement_maintenance'] = late_engagement
            
        except Exception as e:
            logger.error(f"Error assessing user satisfaction: {e}")
        
        return indicators
    
    def _calculate_engagement_subset(self, messages: List[Message]) -> float:
        """Calculate engagement for a subset of messages."""
        if not messages:
            return 0.0
        
        user_messages = [msg for msg in messages if msg.direction.value == "incoming"]
        if not user_messages:
            return 0.0
        
        avg_length = np.mean([len(msg.content or "") for msg in user_messages])
        return min(1.0, avg_length / 100)  # Normalize to 100 characters
    
    async def _assess_overall_response_quality(self, messages: List[Message]) -> float:
        """Assess overall quality of bot responses."""
        if not messages:
            return 0.5
        
        bot_messages = [msg for msg in messages if msg.direction.value == "outgoing"]
        if not bot_messages:
            return 0.5
        
        quality_scores = []
        
        for msg in bot_messages:
            if not msg.content:
                continue
            
            # Assess individual message quality
            message_quality = 0.0
            factors = 0
            
            # Length appropriateness
            length = len(msg.content)
            if 50 <= length <= 300:  # Appropriate length
                message_quality += 0.8
            elif 20 <= length < 50 or 300 < length <= 500:
                message_quality += 0.6
            else:
                message_quality += 0.3
            factors += 1
            
            # Readability
            try:
                readability = flesch_reading_ease(msg.content)
                if 60 <= readability <= 80:  # Good readability
                    message_quality += 0.8
                elif 40 <= readability < 60 or 80 < readability <= 90:
                    message_quality += 0.6
                else:
                    message_quality += 0.4
                factors += 1
            except:
                pass
            
            # Helpfulness indicators
            helpful_words = ['help', 'assist', 'support', 'guide', 'explain', 'show']
            if any(word in msg.content.lower() for word in helpful_words):
                message_quality += 0.7
                factors += 1
            
            quality_scores.append(message_quality / max(factors, 1))
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    async def _assess_emotional_intelligence(
        self, 
        messages: List[Message], 
        emotional_state: EmotionalState
    ) -> float:
        """Assess emotional intelligence of bot responses."""
        if not messages or not emotional_state.emotion_trajectory:
            return 0.5
        
        try:
            bot_messages = [msg for msg in messages if msg.direction.value == "outgoing"]
            if not bot_messages:
                return 0.5
            
            # Look for empathetic responses
            empathy_indicators = [
                'understand', 'feel', 'sorry', 'empathize', 'recognize',
                'appreciate', 'realize', 'acknowledge', 'relate'
            ]
            
            empathetic_responses = 0
            total_responses = 0
            
            for msg in bot_messages:
                if msg.content:
                    total_responses += 1
                    if any(indicator in msg.content.lower() for indicator in empathy_indicators):
                        empathetic_responses += 1
            
            empathy_ratio = empathetic_responses / max(total_responses, 1)
            
            # Adjust based on emotional volatility
            if emotional_state.emotional_volatility > 0.5:  # User is emotionally volatile
                # Bot should show more emotional intelligence
                return min(1.0, empathy_ratio * 1.5)
            
            return empathy_ratio
            
        except Exception as e:
            logger.error(f"Error assessing emotional intelligence: {e}")
            return 0.5
    
    async def _assess_rapport_building(
        self, 
        messages: List[Message], 
        context: ConversationContext
    ) -> float:
        """Assess how well rapport is being built with the user."""
        if not messages:
            return 0.5
        
        try:
            # Factors that indicate good rapport building
            rapport_score = 0.0
            factors = 0
            
            # Conversation length (longer conversations suggest good rapport)
            if context.time_in_conversation > 300:  # 5 minutes
                rapport_score += 0.8
            elif context.time_in_conversation > 120:  # 2 minutes
                rapport_score += 0.6
            else:
                rapport_score += 0.3
            factors += 1
            
            # User engagement level
            rapport_score += context.user_engagement_level
            factors += 1
            
            # Personal touches in bot responses
            bot_messages = [msg for msg in messages if msg.direction.value == "outgoing"]
            personal_indicators = [
                'you', 'your', 'i understand', 'that sounds', 'i can see',
                'it seems like', 'i hear', 'that must', 'you might'
            ]
            
            personal_responses = 0
            for msg in bot_messages:
                if msg.content and any(indicator in msg.content.lower() for indicator in personal_indicators):
                    personal_responses += 1
            
            personalization_ratio = personal_responses / max(len(bot_messages), 1)
            rapport_score += personalization_ratio
            factors += 1
            
            # Topic consistency (staying on topics user cares about)
            if context.context_switches < len(messages) / 10:  # Low context switching
                rapport_score += 0.7
            else:
                rapport_score += 0.4
            factors += 1
            
            return rapport_score / max(factors, 1)
            
        except Exception as e:
            logger.error(f"Error assessing rapport building: {e}")
            return 0.5
    
    async def _cache_conversation_context(self, session_id: str, context: ConversationContext) -> None:
        """Cache conversation context in Redis."""
        try:
            cache_key = f"conversation_context:{session_id}"
            cache_data = {
                'session_id': context.session_id,
                'user_id': context.user_id,
                'conversation_phase': context.conversation_phase,
                'time_in_conversation': context.time_in_conversation,
                'message_count': context.message_count,
                'user_engagement_level': context.user_engagement_level,
                'urgency_indicators': context.urgency_indicators,
                'context_switches': context.context_switches,
                'conversation_goals': context.conversation_goals,
                'cached_at': datetime.now().isoformat()
            }
            
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching conversation context: {e}")
    
    # Quality assessment methods for specific aspects
    async def _assess_coherence(self, messages: List[Message]) -> float:
        """Assess conversation coherence."""
        # Implementation for coherence assessment
        return 0.7  # Placeholder
    
    async def _assess_relevance(self, messages: List[Message]) -> float:
        """Assess response relevance."""
        # Implementation for relevance assessment
        return 0.8  # Placeholder
    
    async def _assess_helpfulness(self, messages: List[Message]) -> float:
        """Assess response helpfulness."""
        # Implementation for helpfulness assessment
        return 0.7  # Placeholder
    
    async def _assess_engagement(self, messages: List[Message]) -> float:
        """Assess conversation engagement."""
        # Implementation for engagement assessment
        return 0.6  # Placeholder
    
    async def _assess_empathy(self, messages: List[Message]) -> float:
        """Assess empathetic response quality."""
        # Implementation for empathy assessment
        return 0.7  # Placeholder


# Export main classes
__all__ = [
    'ConversationAnalyzer', 'ConversationMetrics', 'TopicAnalysis', 
    'EmotionalState', 'ConversationContext'
]