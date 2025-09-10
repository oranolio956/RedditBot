"""
Engagement Analyzer Service

Analyzes user interaction patterns, sentiment trends, and behavioral data
to enable proactive engagement and personalized outreach campaigns.
"""

import asyncio
import statistics
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import re
import json

import numpy as np
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.orm import sessionmaker
import structlog

from app.database.connection import get_database_session
from app.models.user import User
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, EngagementType, SentimentType,
    EngagementMilestone, UserMilestoneProgress
)
from app.services.llm_service import LLMService

logger = structlog.get_logger(__name__)


class EngagementAnalyzer:
    """
    Analyzes user engagement patterns and behavioral data for proactive outreach.
    
    Implements research-proven techniques:
    - Behavioral pattern recognition
    - Sentiment trend analysis
    - Optimal timing detection
    - Churn risk prediction
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        
        # Sentiment analysis patterns
        self.positive_patterns = [
            r'\b(great|awesome|amazing|fantastic|love|perfect|excellent|wonderful)\b',
            r':\)|😊|😁|😍|🔥|👍|❤️|💕',
            r'\b(thank you|thanks|appreciate|helpful)\b'
        ]
        
        self.negative_patterns = [
            r'\b(hate|terrible|awful|bad|worst|horrible|stupid|sucks)\b',
            r':\(|😞|😢|😭|😠|😡|💔',
            r'\b(frustrated|annoyed|disappointed|angry|upset)\b'
        ]
        
        self.question_patterns = [
            r'\?',
            r'\b(how|what|when|where|why|which|who|can|could|would|should|is|are|do|does)\b.*\?',
            r'\b(help|assist|support)\b'
        ]
    
    async def analyze_user_interaction(
        self,
        user_id: str,
        telegram_id: int,
        engagement_type: EngagementType,
        message_text: Optional[str] = None,
        command_name: Optional[str] = None,
        session_id: Optional[str] = None,
        previous_bot_message: Optional[str] = None,
        response_time_seconds: Optional[int] = None
    ) -> UserEngagement:
        """
        Analyze and store a single user interaction.
        
        Args:
            user_id: User UUID
            telegram_id: Telegram user ID
            engagement_type: Type of engagement
            message_text: User message content
            command_name: Command name if applicable
            session_id: Session identifier
            previous_bot_message: Previous bot message for context
            response_time_seconds: Time user took to respond
            
        Returns:
            UserEngagement record with analysis results
        """
        try:
            # Analyze sentiment if text available
            sentiment_score = None
            sentiment_type = SentimentType.UNKNOWN
            mood_indicators = {}
            
            if message_text:
                sentiment_score, sentiment_type = await self._analyze_sentiment(message_text)
                mood_indicators = await self._detect_mood_indicators(message_text)
            
            # Analyze message content
            message_length = len(message_text) if message_text else 0
            contains_emoji = bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', message_text or ''))
            contains_question = any(re.search(pattern, message_text or '', re.IGNORECASE) for pattern in self.question_patterns)
            
            # Detect topics and intent
            detected_topics = []
            user_intent = None
            
            if message_text and len(message_text) > 10:
                detected_topics = await self._extract_topics(message_text)
                user_intent = await self._classify_intent(message_text, command_name)
            
            # Calculate engagement quality
            engagement_quality_score = self._calculate_engagement_quality(
                sentiment_score=sentiment_score,
                message_length=message_length,
                contains_question=contains_question,
                response_time=response_time_seconds,
                engagement_type=engagement_type
            )
            
            # Determine if interaction is meaningful
            is_meaningful = self._is_meaningful_interaction(
                engagement_type=engagement_type,
                message_length=message_length,
                sentiment_type=sentiment_type,
                contains_question=contains_question
            )
            
            # Create engagement record
            engagement = UserEngagement(
                user_id=user_id,
                telegram_id=telegram_id,
                engagement_type=engagement_type,
                interaction_timestamp=datetime.utcnow(),
                message_text=message_text,
                command_name=command_name,
                session_id=session_id,
                sentiment_score=sentiment_score,
                sentiment_type=sentiment_type,
                response_time_seconds=response_time_seconds,
                message_length=message_length,
                contains_emoji=contains_emoji,
                contains_question=contains_question,
                detected_topics=detected_topics,
                user_intent=user_intent,
                mood_indicators=mood_indicators,
                engagement_quality_score=engagement_quality_score,
                is_meaningful_interaction=is_meaningful,
                previous_bot_message=previous_bot_message
            )
            
            # Save to database
            async with get_database_session() as session:
                session.add(engagement)
                await session.commit()
                await session.refresh(engagement)
            
            # Trigger pattern analysis if significant interaction
            if is_meaningful and engagement_quality_score and engagement_quality_score > 0.6:
                asyncio.create_task(self._trigger_pattern_update(user_id, telegram_id))
            
            logger.info(
                "Interaction analyzed",
                user_id=user_id,
                engagement_type=engagement_type.value,
                sentiment_type=sentiment_type.value if sentiment_type else None,
                quality_score=engagement_quality_score,
                is_meaningful=is_meaningful
            )
            
            return engagement
            
        except Exception as e:
            logger.error("Error analyzing user interaction", error=str(e), user_id=user_id)
            raise
    
    async def update_user_behavior_patterns(self, user_id: str, telegram_id: int) -> UserBehaviorPattern:
        """
        Update comprehensive behavioral patterns for a user.
        
        Args:
            user_id: User UUID
            telegram_id: Telegram user ID
            
        Returns:
            Updated UserBehaviorPattern record
        """
        try:
            async with get_database_session() as session:
                # Get recent interactions (last 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                recent_interactions = await session.execute(
                    select(UserEngagement)
                    .where(
                        and_(
                            UserEngagement.user_id == user_id,
                            UserEngagement.interaction_timestamp >= cutoff_date,
                            UserEngagement.is_meaningful_interaction == True
                        )
                    )
                    .order_by(desc(UserEngagement.interaction_timestamp))
                )
                interactions = recent_interactions.scalars().all()
                
                if not interactions:
                    logger.info("No recent interactions found for pattern analysis", user_id=user_id)
                    return None
                
                # Calculate activity patterns
                activity_patterns = self._analyze_activity_patterns(interactions)
                
                # Calculate engagement metrics
                engagement_metrics = self._calculate_engagement_metrics(interactions)
                
                # Analyze interaction preferences
                interaction_preferences = self._analyze_interaction_preferences(interactions)
                
                # Calculate behavioral indicators
                behavioral_indicators = self._calculate_behavioral_indicators(interactions, engagement_metrics)
                
                # Calculate churn risk
                churn_risk_score = self._calculate_churn_risk(interactions, engagement_metrics)
                
                # Find or create behavior pattern record
                existing_pattern = await session.execute(
                    select(UserBehaviorPattern).where(UserBehaviorPattern.user_id == user_id)
                )
                pattern = existing_pattern.scalar_one_or_none()
                
                if not pattern:
                    pattern = UserBehaviorPattern(
                        user_id=user_id,
                        telegram_id=telegram_id
                    )
                    session.add(pattern)
                
                # Update pattern data
                self._update_pattern_data(pattern, activity_patterns, engagement_metrics, 
                                       interaction_preferences, behavioral_indicators, churn_risk_score)
                
                # Update milestone progress
                await self._update_milestone_progress(session, user_id, telegram_id, interactions)
                
                await session.commit()
                await session.refresh(pattern)
                
                logger.info(
                    "Behavior patterns updated",
                    user_id=user_id,
                    total_interactions=pattern.total_interactions,
                    churn_risk=pattern.churn_risk_score,
                    engagement_trend=pattern.engagement_quality_trend
                )
                
                return pattern
                
        except Exception as e:
            logger.error("Error updating behavior patterns", error=str(e), user_id=user_id)
            raise
    
    async def find_users_needing_engagement(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find users who need proactive engagement based on behavioral patterns.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            List of users with engagement recommendations
        """
        try:
            async with get_database_session() as session:
                # Find users needing re-engagement or at risk of churning
                query = select(UserBehaviorPattern, User).join(
                    User, UserBehaviorPattern.user_id == User.id
                ).where(
                    and_(
                        User.is_active == True,
                        User.is_blocked == False,
                        or_(
                            UserBehaviorPattern.needs_re_engagement == True,
                            UserBehaviorPattern.churn_risk_score > 0.6,
                            UserBehaviorPattern.shows_declining_engagement == True,
                            UserBehaviorPattern.days_since_last_interaction > 3
                        )
                    )
                ).order_by(
                    desc(UserBehaviorPattern.churn_risk_score),
                    desc(UserBehaviorPattern.days_since_last_interaction)
                ).limit(limit)
                
                result = await session.execute(query)
                user_patterns = result.all()
                
                engagement_candidates = []
                
                for pattern, user in user_patterns:
                    # Determine best engagement strategy
                    recommendations = await self._generate_engagement_recommendations(pattern, user)
                    
                    engagement_candidates.append({
                        'user_id': str(user.id),
                        'telegram_id': user.telegram_id,
                        'user': {
                            'display_name': user.get_display_name(),
                            'username': user.username,
                            'language_code': user.language_code
                        },
                        'patterns': {
                            'churn_risk_score': pattern.churn_risk_score,
                            'days_since_last_interaction': pattern.days_since_last_interaction,
                            'engagement_quality_trend': pattern.engagement_quality_trend,
                            'needs_re_engagement': pattern.needs_re_engagement,
                            'optimal_outreach_hour': pattern.optimal_outreach_hour,
                            'dominant_sentiment': pattern.dominant_sentiment.value if pattern.dominant_sentiment else None,
                            'topic_interests': pattern.topic_interests
                        },
                        'recommendations': recommendations
                    })
                
                logger.info(
                    "Found users needing engagement",
                    total_candidates=len(engagement_candidates),
                    high_risk_count=len([c for c in engagement_candidates if c['patterns']['churn_risk_score'] > 0.8])
                )
                
                return engagement_candidates
                
        except Exception as e:
            logger.error("Error finding users needing engagement", error=str(e))
            raise
    
    async def get_optimal_outreach_timing(self, user_id: str) -> Optional[datetime]:
        """
        Calculate optimal timing for proactive outreach to a user.
        
        Args:
            user_id: User UUID
            
        Returns:
            Optimal datetime for outreach, or None if insufficient data
        """
        try:
            async with get_database_session() as session:
                # Get user's behavior pattern
                pattern_result = await session.execute(
                    select(UserBehaviorPattern).where(UserBehaviorPattern.user_id == user_id)
                )
                pattern = pattern_result.scalar_one_or_none()
                
                if not pattern or not pattern.optimal_outreach_hour:
                    # Default to afternoon if no pattern available
                    next_outreach = datetime.utcnow().replace(hour=14, minute=0, second=0, microsecond=0)
                    if next_outreach <= datetime.utcnow():
                        next_outreach += timedelta(days=1)
                    return next_outreach
                
                # Calculate next optimal time based on user's patterns
                optimal_hour = pattern.optimal_outreach_hour
                
                # Consider user's most active day
                today = datetime.utcnow()
                target_date = today.date()
                
                if pattern.most_active_day is not None:
                    days_until_optimal = (pattern.most_active_day - today.weekday()) % 7
                    if days_until_optimal == 0 and today.hour >= optimal_hour:
                        days_until_optimal = 7  # Next week if optimal time passed today
                    target_date = today.date() + timedelta(days=days_until_optimal)
                
                optimal_time = datetime.combine(target_date, time(hour=optimal_hour))
                
                # Ensure it's in the future
                if optimal_time <= datetime.utcnow():
                    optimal_time += timedelta(days=1)
                
                return optimal_time
                
        except Exception as e:
            logger.error("Error calculating optimal outreach timing", error=str(e), user_id=user_id)
            return None
    
    # Private helper methods
    
    async def _analyze_sentiment(self, text: str) -> Tuple[float, SentimentType]:
        """Analyze sentiment of text using patterns and LLM."""
        if not text:
            return 0.0, SentimentType.NEUTRAL
        
        text_lower = text.lower()
        
        # Pattern-based sentiment analysis
        positive_score = sum(1 for pattern in self.positive_patterns 
                           if re.search(pattern, text_lower, re.IGNORECASE))
        negative_score = sum(1 for pattern in self.negative_patterns 
                           if re.search(pattern, text_lower, re.IGNORECASE))
        
        # Simple scoring
        if positive_score > negative_score:
            score = min(0.8, positive_score * 0.3)
            sentiment_type = SentimentType.POSITIVE if score > 0.3 else SentimentType.NEUTRAL
        elif negative_score > positive_score:
            score = max(-0.8, -negative_score * 0.3)
            sentiment_type = SentimentType.NEGATIVE if score < -0.3 else SentimentType.NEUTRAL
        else:
            score = 0.0
            sentiment_type = SentimentType.NEUTRAL
        
        # Enhanced analysis for longer texts using LLM
        if len(text) > 30 and self.llm_service:
            try:
                llm_sentiment = await self._get_llm_sentiment(text)
                if llm_sentiment:
                    # Blend pattern and LLM results
                    score = (score + llm_sentiment) / 2
                    
                    if score > 0.6:
                        sentiment_type = SentimentType.VERY_POSITIVE
                    elif score > 0.2:
                        sentiment_type = SentimentType.POSITIVE
                    elif score < -0.6:
                        sentiment_type = SentimentType.VERY_NEGATIVE
                    elif score < -0.2:
                        sentiment_type = SentimentType.NEGATIVE
                    else:
                        sentiment_type = SentimentType.NEUTRAL
            except Exception as e:
                logger.warning("LLM sentiment analysis failed", error=str(e))
        
        return score, sentiment_type
    
    async def _get_llm_sentiment(self, text: str) -> Optional[float]:
        """Get sentiment score from LLM."""
        try:
            prompt = f"""
            Analyze the sentiment of this message on a scale from -1 (very negative) to 1 (very positive).
            Return only a decimal number.
            
            Message: {text}
            """
            
            response = await self.llm_service.get_completion(prompt, max_tokens=10)
            
            # Extract number from response
            import re
            number_match = re.search(r'-?\d*\.?\d+', response.strip())
            if number_match:
                return float(number_match.group())
                
        except Exception as e:
            logger.warning("Error getting LLM sentiment", error=str(e))
            
        return None
    
    async def _detect_mood_indicators(self, text: str) -> Dict[str, Any]:
        """Detect mood indicators in text."""
        mood_indicators = {
            'excitement_level': 0.0,
            'frustration_level': 0.0,
            'confusion_level': 0.0,
            'gratitude_level': 0.0
        }
        
        text_lower = text.lower()
        
        # Excitement indicators
        excitement_patterns = [r'!{2,}', r'\b(wow|amazing|awesome|excited)\b', r'🔥|⚡|🎉']
        mood_indicators['excitement_level'] = min(1.0, sum(
            len(re.findall(pattern, text_lower)) for pattern in excitement_patterns
        ) * 0.3)
        
        # Frustration indicators
        frustration_patterns = [r'\b(ugh|argh|frustrated|annoying)\b', r'😠|😡|💢']
        mood_indicators['frustration_level'] = min(1.0, sum(
            len(re.findall(pattern, text_lower)) for pattern in frustration_patterns
        ) * 0.4)
        
        # Confusion indicators
        confusion_patterns = [r'\?{2,}', r'\b(confused|what|huh|unclear)\b', r'🤔|😕']
        mood_indicators['confusion_level'] = min(1.0, sum(
            len(re.findall(pattern, text_lower)) for pattern in confusion_patterns
        ) * 0.4)
        
        # Gratitude indicators
        gratitude_patterns = [r'\b(thank|thanks|appreciate|grateful)\b', r'🙏|💝|❤️']
        mood_indicators['gratitude_level'] = min(1.0, sum(
            len(re.findall(pattern, text_lower)) for pattern in gratitude_patterns
        ) * 0.5)
        
        return mood_indicators
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple keyword-based topic extraction
        topic_keywords = {
            'help': ['help', 'support', 'assist', 'guide', 'how'],
            'technical': ['bug', 'error', 'problem', 'issue', 'crash'],
            'feature': ['feature', 'functionality', 'option', 'setting'],
            'feedback': ['feedback', 'opinion', 'review', 'rating'],
            'social': ['friend', 'share', 'community', 'group'],
            'personal': ['personal', 'private', 'me', 'my', 'myself']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    async def _classify_intent(self, text: str, command_name: Optional[str] = None) -> Optional[str]:
        """Classify user intent from text and command."""
        if command_name:
            return f"command_{command_name}"
        
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Intent classification patterns
        if any(word in text_lower for word in ['help', 'how', 'what', 'guide']):
            return 'help_seeking'
        elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'gratitude'
        elif any(word in text_lower for word in ['bug', 'error', 'problem', 'issue']):
            return 'problem_reporting'
        elif '?' in text:
            return 'question'
        elif any(word in text_lower for word in ['good', 'great', 'love', 'awesome']):
            return 'positive_feedback'
        elif any(word in text_lower for word in ['bad', 'hate', 'terrible', 'awful']):
            return 'negative_feedback'
        else:
            return 'general_conversation'
    
    def _calculate_engagement_quality(
        self,
        sentiment_score: Optional[float],
        message_length: int,
        contains_question: bool,
        response_time: Optional[int],
        engagement_type: EngagementType
    ) -> float:
        """Calculate engagement quality score."""
        quality = 0.5  # Base score
        
        # Sentiment contribution (30%)
        if sentiment_score is not None:
            quality += (sentiment_score + 1) * 0.15  # Normalize to 0-0.3
        
        # Message length contribution (20%)
        if message_length > 0:
            length_score = min(1.0, message_length / 100) * 0.2
            quality += length_score
        
        # Question bonus (15%)
        if contains_question:
            quality += 0.15
        
        # Response time contribution (15%)
        if response_time is not None:
            # Faster responses get higher scores (up to 2 minutes)
            time_score = max(0, 1 - (response_time / 120)) * 0.15
            quality += time_score
        
        # Engagement type bonus (20%)
        type_scores = {
            EngagementType.COMMAND: 0.2,
            EngagementType.MESSAGE: 0.15,
            EngagementType.VOICE_MESSAGE: 0.2,
            EngagementType.CALLBACK: 0.1,
            EngagementType.REACTION: 0.05
        }
        quality += type_scores.get(engagement_type, 0.1)
        
        return min(1.0, quality)
    
    def _is_meaningful_interaction(
        self,
        engagement_type: EngagementType,
        message_length: int,
        sentiment_type: SentimentType,
        contains_question: bool
    ) -> bool:
        """Determine if interaction is meaningful for analysis."""
        # Commands are always meaningful
        if engagement_type == EngagementType.COMMAND:
            return True
        
        # Voice messages are meaningful
        if engagement_type == EngagementType.VOICE_MESSAGE:
            return True
        
        # Short messages with no clear intent are less meaningful
        if message_length < 5:
            return False
        
        # Questions are meaningful
        if contains_question:
            return True
        
        # Messages with clear sentiment are meaningful
        if sentiment_type in [SentimentType.VERY_POSITIVE, SentimentType.VERY_NEGATIVE]:
            return True
        
        # Longer messages are generally meaningful
        if message_length > 30:
            return True
        
        return message_length > 10  # Default threshold
    
    def _analyze_activity_patterns(self, interactions: List[UserEngagement]) -> Dict[str, Any]:
        """Analyze user activity patterns from interactions."""
        if not interactions:
            return {}
        
        # Activity by hour
        hours = [interaction.interaction_timestamp.hour for interaction in interactions]
        most_active_hour = Counter(hours).most_common(1)[0][0] if hours else None
        
        # Activity by day of week
        days = [interaction.interaction_timestamp.weekday() for interaction in interactions]
        most_active_day = Counter(days).most_common(1)[0][0] if days else None
        
        # Calculate session patterns
        session_lengths = self._calculate_session_lengths(interactions)
        avg_session_length = statistics.mean(session_lengths) if session_lengths else 0
        
        # Daily interaction frequency
        interaction_dates = [interaction.interaction_timestamp.date() for interaction in interactions]
        unique_dates = len(set(interaction_dates))
        daily_average = len(interactions) / max(1, unique_dates)
        
        return {
            'most_active_hour': most_active_hour,
            'most_active_day': most_active_day,
            'average_session_length': avg_session_length,
            'daily_interaction_average': daily_average,
            'total_interactions': len(interactions)
        }
    
    def _calculate_session_lengths(self, interactions: List[UserEngagement]) -> List[float]:
        """Calculate session lengths from interactions."""
        sessions = defaultdict(list)
        
        for interaction in interactions:
            if interaction.session_id:
                sessions[interaction.session_id].append(interaction.interaction_timestamp)
        
        session_lengths = []
        for session_times in sessions.values():
            if len(session_times) > 1:
                session_times.sort()
                duration = (session_times[-1] - session_times[0]).total_seconds() / 60
                session_lengths.append(duration)
        
        return session_lengths
    
    def _calculate_engagement_metrics(self, interactions: List[UserEngagement]) -> Dict[str, Any]:
        """Calculate engagement quality metrics."""
        if not interactions:
            return {}
        
        # Sentiment metrics
        sentiments = [i.sentiment_score for i in interactions if i.sentiment_score is not None]
        avg_sentiment = statistics.mean(sentiments) if sentiments else 0
        
        # Sentiment trend (recent vs older)
        if len(sentiments) >= 4:
            recent_sentiment = statistics.mean(sentiments[:len(sentiments)//2])
            older_sentiment = statistics.mean(sentiments[len(sentiments)//2:])
            sentiment_trend = recent_sentiment - older_sentiment
        else:
            sentiment_trend = 0
        
        # Dominant sentiment
        sentiment_types = [i.sentiment_type for i in interactions if i.sentiment_type]
        dominant_sentiment = Counter(sentiment_types).most_common(1)[0][0] if sentiment_types else SentimentType.NEUTRAL
        
        # Quality metrics
        quality_scores = [i.engagement_quality_score for i in interactions if i.engagement_quality_score is not None]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        # Quality trend
        if len(quality_scores) >= 4:
            recent_quality = statistics.mean(quality_scores[:len(quality_scores)//2])
            older_quality = statistics.mean(quality_scores[len(quality_scores)//2:])
            quality_trend = recent_quality - older_quality
        else:
            quality_trend = 0
        
        # Response time metrics
        response_times = [i.response_time_seconds for i in interactions if i.response_time_seconds is not None]
        avg_response_time = statistics.mean(response_times) if response_times else None
        
        return {
            'average_sentiment_score': avg_sentiment,
            'dominant_sentiment': dominant_sentiment,
            'engagement_quality_trend': quality_trend,
            'response_time_average_seconds': avg_response_time,
            'sentiment_trend': sentiment_trend
        }
    
    def _analyze_interaction_preferences(self, interactions: List[UserEngagement]) -> Dict[str, Any]:
        """Analyze user interaction preferences."""
        if not interactions:
            return {}
        
        # Preferred interaction types
        interaction_types = [i.engagement_type for i in interactions]
        type_counts = Counter(interaction_types)
        preferred_types = [{'type': str(t), 'count': c} for t, c in type_counts.most_common(5)]
        
        # Favorite commands
        commands = [i.command_name for i in interactions if i.command_name]
        command_counts = Counter(commands)
        favorite_commands = [{'command': c, 'count': count} for c, count in command_counts.most_common(5)]
        
        # Topic interests
        all_topics = []
        for interaction in interactions:
            if interaction.detected_topics:
                all_topics.extend(interaction.detected_topics)
        
        topic_counts = Counter(all_topics)
        topic_interests = [{'topic': t, 'relevance': c/len(interactions)} for t, c in topic_counts.most_common(5)]
        
        return {
            'preferred_interaction_types': preferred_types,
            'favorite_commands': favorite_commands,
            'topic_interests': topic_interests
        }
    
    def _calculate_behavioral_indicators(
        self,
        interactions: List[UserEngagement],
        engagement_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate behavioral health indicators."""
        if not interactions:
            return {}
        
        # High engagement indicators
        high_engagement_threshold = 0.7
        quality_scores = [i.engagement_quality_score for i in interactions if i.engagement_quality_score is not None]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        is_highly_engaged = avg_quality >= high_engagement_threshold
        
        # Declining engagement
        quality_trend = engagement_metrics.get('engagement_quality_trend', 0)
        shows_declining_engagement = quality_trend < -0.1
        
        # Re-engagement needs
        days_since_last = (datetime.utcnow() - interactions[0].interaction_timestamp).days if interactions else 999
        needs_re_engagement = (
            days_since_last > 3 or 
            shows_declining_engagement or 
            avg_quality < 0.4
        )
        
        return {
            'is_highly_engaged': is_highly_engaged,
            'shows_declining_engagement': shows_declining_engagement,
            'needs_re_engagement': needs_re_engagement,
            'days_since_last_interaction': days_since_last
        }
    
    def _calculate_churn_risk(
        self,
        interactions: List[UserEngagement],
        engagement_metrics: Dict[str, Any]
    ) -> float:
        """Calculate churn risk score using multiple factors."""
        if not interactions:
            return 1.0
        
        risk_score = 0.0
        
        # Days since last interaction (40% weight)
        days_since_last = (datetime.utcnow() - interactions[0].interaction_timestamp).days
        if days_since_last > 7:
            risk_score += 0.4
        elif days_since_last > 3:
            risk_score += 0.2
        
        # Engagement quality trend (30% weight)
        quality_trend = engagement_metrics.get('engagement_quality_trend', 0)
        if quality_trend < -0.2:
            risk_score += 0.3
        elif quality_trend < -0.1:
            risk_score += 0.15
        
        # Average sentiment (20% weight)
        avg_sentiment = engagement_metrics.get('average_sentiment_score', 0)
        if avg_sentiment < -0.3:
            risk_score += 0.2
        elif avg_sentiment < 0:
            risk_score += 0.1
        
        # Interaction frequency (10% weight)
        if len(interactions) < 3:  # Very few interactions in 30 days
            risk_score += 0.1
        elif len(interactions) < 10:
            risk_score += 0.05
        
        return min(1.0, risk_score)
    
    def _update_pattern_data(
        self,
        pattern: UserBehaviorPattern,
        activity_patterns: Dict[str, Any],
        engagement_metrics: Dict[str, Any],
        interaction_preferences: Dict[str, Any],
        behavioral_indicators: Dict[str, Any],
        churn_risk_score: float
    ) -> None:
        """Update behavior pattern record with calculated data."""
        # Activity patterns
        pattern.total_interactions = activity_patterns.get('total_interactions', 0)
        pattern.daily_interaction_average = activity_patterns.get('daily_interaction_average')
        pattern.most_active_hour = activity_patterns.get('most_active_hour')
        pattern.most_active_day = activity_patterns.get('most_active_day')
        pattern.average_session_length_minutes = activity_patterns.get('average_session_length', 0)
        
        # Engagement metrics
        pattern.average_sentiment_score = engagement_metrics.get('average_sentiment_score')
        pattern.dominant_sentiment = engagement_metrics.get('dominant_sentiment')
        pattern.engagement_quality_trend = engagement_metrics.get('engagement_quality_trend')
        pattern.response_time_average_seconds = engagement_metrics.get('response_time_average_seconds')
        
        # Preferences
        pattern.preferred_interaction_types = interaction_preferences.get('preferred_interaction_types')
        pattern.favorite_commands = interaction_preferences.get('favorite_commands')
        pattern.topic_interests = interaction_preferences.get('topic_interests')
        
        # Behavioral indicators
        pattern.is_highly_engaged = behavioral_indicators.get('is_highly_engaged', False)
        pattern.shows_declining_engagement = behavioral_indicators.get('shows_declining_engagement', False)
        pattern.needs_re_engagement = behavioral_indicators.get('needs_re_engagement', False)
        pattern.days_since_last_interaction = behavioral_indicators.get('days_since_last_interaction')
        pattern.churn_risk_score = churn_risk_score
        
        # Set optimal outreach timing
        pattern.optimal_outreach_hour = pattern.most_active_hour or 14  # Default to 2 PM
        
        # Update analysis metadata
        pattern.last_pattern_analysis = datetime.utcnow()
        pattern.pattern_analysis_version = "1.0"
    
    async def _update_milestone_progress(
        self,
        session,
        user_id: str,
        telegram_id: int,
        interactions: List[UserEngagement]
    ) -> None:
        """Update user milestone progress based on interactions."""
        # Get active milestones
        milestones_result = await session.execute(
            select(EngagementMilestone).where(EngagementMilestone.is_active == True)
        )
        milestones = milestones_result.scalars().all()
        
        for milestone in milestones:
            # Calculate current value based on metric
            current_value = self._calculate_milestone_value(milestone, interactions)
            
            # Get or create progress record
            progress_result = await session.execute(
                select(UserMilestoneProgress).where(
                    and_(
                        UserMilestoneProgress.user_id == user_id,
                        UserMilestoneProgress.milestone_id == milestone.id
                    )
                )
            )
            progress = progress_result.scalar_one_or_none()
            
            if not progress:
                progress = UserMilestoneProgress(
                    user_id=user_id,
                    milestone_id=milestone.id,
                    telegram_id=telegram_id,
                    target_value=milestone.target_value
                )
                session.add(progress)
            
            # Update progress
            milestone_achieved = progress.update_progress(current_value)
            
            if milestone_achieved:
                # Update milestone statistics
                milestone.total_achievements += 1
                milestone.last_achieved_at = datetime.utcnow()
                
                logger.info(
                    "Milestone achieved",
                    user_id=user_id,
                    milestone_name=milestone.milestone_name,
                    value=current_value
                )
    
    def _calculate_milestone_value(
        self,
        milestone: EngagementMilestone,
        interactions: List[UserEngagement]
    ) -> float:
        """Calculate current milestone value for a user."""
        metric_name = milestone.metric_name
        
        if metric_name == "total_interactions":
            return float(len(interactions))
        elif metric_name == "message_count":
            return float(len([i for i in interactions if i.engagement_type == EngagementType.MESSAGE]))
        elif metric_name == "command_count":
            return float(len([i for i in interactions if i.engagement_type == EngagementType.COMMAND]))
        elif metric_name == "positive_interactions":
            return float(len([i for i in interactions if i.sentiment_type in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]]))
        elif metric_name == "session_days":
            unique_dates = len(set(i.interaction_timestamp.date() for i in interactions))
            return float(unique_dates)
        elif metric_name == "average_quality_score":
            quality_scores = [i.engagement_quality_score for i in interactions if i.engagement_quality_score is not None]
            return statistics.mean(quality_scores) if quality_scores else 0.0
        
        return 0.0
    
    async def _generate_engagement_recommendations(
        self,
        pattern: UserBehaviorPattern,
        user: User
    ) -> List[Dict[str, Any]]:
        """Generate engagement recommendations for a user."""
        recommendations = []
        
        # High churn risk - immediate re-engagement
        if pattern.churn_risk_score > 0.8:
            recommendations.append({
                'type': 're_engagement',
                'priority': 0.9,
                'reason': 'high_churn_risk',
                'suggested_message': 'personalized_checkin',
                'timing': 'immediate'
            })
        
        # Declining engagement - supportive outreach
        if pattern.shows_declining_engagement:
            recommendations.append({
                'type': 'mood_support',
                'priority': 0.7,
                'reason': 'declining_engagement',
                'suggested_message': 'supportive_checkin',
                'timing': 'optimal_hour'
            })
        
        # Long absence - re-engagement campaign
        if pattern.days_since_last_interaction and pattern.days_since_last_interaction > 5:
            recommendations.append({
                'type': 're_engagement',
                'priority': 0.6,
                'reason': 'long_absence',
                'suggested_message': 'welcome_back',
                'timing': 'optimal_hour'
            })
        
        # Milestone approaching
        if pattern.next_milestone_target and pattern.milestone_progress_percent and pattern.milestone_progress_percent > 80:
            recommendations.append({
                'type': 'milestone_celebration',
                'priority': 0.8,
                'reason': 'milestone_approaching',
                'suggested_message': 'milestone_encouragement',
                'timing': 'optimal_hour'
            })
        
        # Topic-based follow-up
        if pattern.topic_interests:
            recommendations.append({
                'type': 'topic_follow_up',
                'priority': 0.5,
                'reason': 'topic_interest',
                'suggested_message': 'topic_continuation',
                'timing': 'optimal_hour',
                'topics': pattern.topic_interests[:3]
            })
        
        return recommendations
    
    async def _trigger_pattern_update(self, user_id: str, telegram_id: int) -> None:
        """Trigger background pattern update for a user."""
        try:
            # Import here to avoid circular imports
            from app.services.celery_service import update_user_patterns_task
            
            # Schedule pattern update task
            update_user_patterns_task.delay(user_id, telegram_id)
            
        except ImportError:
            logger.warning("Celery service not available for pattern updates")
        except Exception as e:
            logger.error("Error triggering pattern update", error=str(e))