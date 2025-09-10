"""
Group Manager Service

Advanced group conversation logic and management system for handling
multi-user interactions, conversation states, analytics, and engagement tracking.
"""

import asyncio
import time
import hashlib
import json
import re
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from ..models.group_session import (
    GroupSession, GroupMember, GroupConversation, GroupAnalytics,
    GroupType, MemberRole, GroupStatus, MessageFrequency
)
from ..models.user import User
from ..models.conversation import Message, MessageType, MessageDirection
from ..database.connection import get_async_session

logger = structlog.get_logger(__name__)


class ConversationState(str, Enum):
    """Conversation state enumeration."""
    STARTING = "starting"
    ACTIVE = "active"
    WAITING_RESPONSE = "waiting_response"
    MULTI_TURN = "multi_turn"
    WINDING_DOWN = "winding_down"
    ENDED = "ended"


class ThreadImportance(str, Enum):
    """Thread importance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConversationContext:
    """Context data for ongoing conversations."""
    thread_id: str
    participants: Set[int] = field(default_factory=set)
    topic: Optional[str] = None
    keywords: Set[str] = field(default_factory=set)
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment_scores: List[float] = field(default_factory=list)
    message_count: int = 0
    bot_interactions: int = 0
    last_activity: Optional[datetime] = None
    state: ConversationState = ConversationState.STARTING
    importance: ThreadImportance = ThreadImportance.MEDIUM
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemberEngagementData:
    """Member engagement tracking data."""
    user_id: int
    telegram_user_id: int
    message_count: int = 0
    mention_count: int = 0
    last_activity: Optional[datetime] = None
    engagement_score: float = 0.0
    conversation_threads: Set[str] = field(default_factory=set)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    sentiment_trend: List[float] = field(default_factory=list)


class ConversationThreadManager:
    """Manages conversation threads and context within groups."""
    
    def __init__(self, max_threads: int = 50, thread_timeout: int = 3600):
        self.max_threads = max_threads
        self.thread_timeout = thread_timeout
        
        # Active conversation threads
        self.active_threads: Dict[str, ConversationContext] = {}
        self.thread_participants: Dict[int, Set[str]] = defaultdict(set)
        self.thread_timestamps: deque = deque()
        
        # Topic detection and clustering
        self.topic_keywords: Dict[str, Set[str]] = defaultdict(set)
        self.topic_clustering: Dict[str, List[str]] = defaultdict(list)
    
    def generate_thread_id(self, message_content: str, participants: List[int]) -> str:
        """Generate unique thread ID based on content and participants."""
        # Create hash from content keywords and participants
        content_hash = hashlib.md5(message_content.lower().encode()).hexdigest()[:8]
        participant_hash = hashlib.md5(
            str(sorted(participants)).encode()
        ).hexdigest()[:8]
        
        return f"thread_{content_hash}_{participant_hash}_{int(time.time() // 300)}"  # 5-minute buckets
    
    def detect_thread_continuation(
        self, 
        message_content: str, 
        user_id: int, 
        reply_to_message_id: Optional[int] = None
    ) -> Optional[str]:
        """Detect if message continues an existing thread."""
        # If replying to a message, find the thread it belongs to
        if reply_to_message_id:
            for thread_id, context in self.active_threads.items():
                if user_id in context.participants:
                    return thread_id
        
        # Content-based thread detection
        message_keywords = set(re.findall(r'\b\w+\b', message_content.lower()))
        
        best_match = None
        best_similarity = 0.0
        
        for thread_id, context in self.active_threads.items():
            if user_id in context.participants:
                # Calculate keyword similarity
                similarity = len(message_keywords & context.keywords) / max(
                    len(message_keywords | context.keywords), 1
                )
                
                if similarity > best_similarity and similarity > 0.3:
                    best_similarity = similarity
                    best_match = thread_id
        
        return best_match
    
    def create_thread(
        self, 
        message_content: str, 
        participants: List[int],
        reply_to_message_id: Optional[int] = None
    ) -> str:
        """Create new conversation thread."""
        # Check if continuing existing thread
        existing_thread = self.detect_thread_continuation(
            message_content, participants[0], reply_to_message_id
        )
        
        if existing_thread:
            return existing_thread
        
        # Create new thread
        thread_id = self.generate_thread_id(message_content, participants)
        
        # Extract keywords and entities
        keywords = set(re.findall(r'\b\w+\b', message_content.lower()))
        keywords = {word for word in keywords if len(word) > 3}  # Filter short words
        
        context = ConversationContext(
            thread_id=thread_id,
            participants=set(participants),
            keywords=keywords,
            last_activity=datetime.utcnow(),
            message_count=1
        )
        
        self.active_threads[thread_id] = context
        
        # Track participants
        for participant in participants:
            self.thread_participants[participant].add(thread_id)
        
        # Cleanup old threads if needed
        self._cleanup_old_threads()
        
        return thread_id
    
    def update_thread(
        self, 
        thread_id: str, 
        message_content: str, 
        user_id: int,
        is_bot_interaction: bool = False
    ) -> None:
        """Update existing thread with new message."""
        if thread_id not in self.active_threads:
            return
        
        context = self.active_threads[thread_id]
        
        # Update context
        context.participants.add(user_id)
        context.keywords.update(
            set(re.findall(r'\b\w+\b', message_content.lower()))
        )
        context.message_count += 1
        context.last_activity = datetime.utcnow()
        
        if is_bot_interaction:
            context.bot_interactions += 1
        
        # Update state based on activity
        if context.message_count < 3:
            context.state = ConversationState.STARTING
        elif context.bot_interactions > 0:
            context.state = ConversationState.ACTIVE
        elif context.message_count > 20:
            context.state = ConversationState.WINDING_DOWN
        else:
            context.state = ConversationState.ACTIVE
        
        # Track participant
        self.thread_participants[user_id].add(thread_id)
    
    def get_thread_context(self, thread_id: str) -> Optional[ConversationContext]:
        """Get conversation context for thread."""
        return self.active_threads.get(thread_id)
    
    def end_thread(self, thread_id: str) -> Optional[ConversationContext]:
        """End conversation thread and return final context."""
        if thread_id not in self.active_threads:
            return None
        
        context = self.active_threads[thread_id]
        context.state = ConversationState.ENDED
        
        # Remove from tracking
        del self.active_threads[thread_id]
        
        # Clean up participant tracking
        for participant in context.participants:
            self.thread_participants[participant].discard(thread_id)
        
        return context
    
    def _cleanup_old_threads(self) -> None:
        """Clean up old or inactive threads."""
        now = datetime.utcnow()
        cutoff_time = now - timedelta(seconds=self.thread_timeout)
        
        threads_to_remove = []
        
        for thread_id, context in self.active_threads.items():
            if context.last_activity and context.last_activity < cutoff_time:
                threads_to_remove.append(thread_id)
        
        # Remove old threads
        for thread_id in threads_to_remove:
            self.end_thread(thread_id)
        
        # Remove excess threads if over limit
        if len(self.active_threads) > self.max_threads:
            # Sort by last activity and remove oldest
            sorted_threads = sorted(
                self.active_threads.items(),
                key=lambda x: x[1].last_activity or datetime.min
            )
            
            excess_count = len(self.active_threads) - self.max_threads
            for thread_id, _ in sorted_threads[:excess_count]:
                self.end_thread(thread_id)


class MemberEngagementTracker:
    """Tracks individual member engagement within groups."""
    
    def __init__(self, max_members: int = 1000):
        self.max_members = max_members
        self.member_data: Dict[Tuple[int, int], MemberEngagementData] = {}  # (group_id, user_id)
        self.engagement_history: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
    
    def track_member_activity(
        self,
        group_session: GroupSession,
        user_id: int,
        telegram_user_id: int,
        message_content: str,
        thread_id: str,
        is_bot_interaction: bool = False
    ) -> None:
        """Track member activity and update engagement data."""
        member_key = (group_session.id, user_id)
        
        # Get or create member data
        if member_key not in self.member_data:
            self.member_data[member_key] = MemberEngagementData(
                user_id=user_id,
                telegram_user_id=telegram_user_id
            )
        
        member_data = self.member_data[member_key]
        
        # Update activity
        member_data.message_count += 1
        member_data.last_activity = datetime.utcnow()
        member_data.conversation_threads.add(thread_id)
        
        if is_bot_interaction:
            member_data.mention_count += 1
        
        # Analyze message content for patterns
        self._analyze_interaction_patterns(member_data, message_content, is_bot_interaction)
        
        # Calculate engagement score
        member_data.engagement_score = self._calculate_member_engagement(
            member_data, group_session
        )
        
        # Record engagement history
        self.engagement_history[member_key].append({
            'timestamp': time.time(),
            'engagement_score': member_data.engagement_score,
            'thread_id': thread_id,
            'is_bot_interaction': is_bot_interaction
        })
    
    def get_member_engagement(
        self, 
        group_id: int, 
        user_id: int
    ) -> Optional[MemberEngagementData]:
        """Get member engagement data."""
        return self.member_data.get((group_id, user_id))
    
    def get_top_engaged_members(
        self, 
        group_session: GroupSession, 
        limit: int = 10
    ) -> List[MemberEngagementData]:
        """Get top engaged members in group."""
        group_members = [
            data for (gid, uid), data in self.member_data.items() 
            if gid == group_session.id
        ]
        
        return sorted(
            group_members, 
            key=lambda x: x.engagement_score, 
            reverse=True
        )[:limit]
    
    def _analyze_interaction_patterns(
        self, 
        member_data: MemberEngagementData, 
        message_content: str,
        is_bot_interaction: bool
    ) -> None:
        """Analyze member interaction patterns."""
        patterns = member_data.interaction_patterns
        
        # Message length analysis
        msg_length = len(message_content)
        patterns['avg_message_length'] = (
            patterns.get('avg_message_length', 0) * 0.9 + msg_length * 0.1
        )
        
        # Question asking pattern
        question_count = message_content.count('?')
        patterns['questions_per_message'] = (
            patterns.get('questions_per_message', 0) * 0.9 + question_count * 0.1
        )
        
        # Bot interaction frequency
        patterns['bot_interaction_ratio'] = (
            member_data.mention_count / max(member_data.message_count, 1)
        )
        
        # Activity timing (hour of day)
        current_hour = datetime.now().hour
        hourly_activity = patterns.get('hourly_activity', [0] * 24)
        hourly_activity[current_hour] += 1
        patterns['hourly_activity'] = hourly_activity
        patterns['peak_activity_hour'] = hourly_activity.index(max(hourly_activity))
    
    def _calculate_member_engagement(
        self, 
        member_data: MemberEngagementData, 
        group_session: GroupSession
    ) -> float:
        """Calculate member engagement score."""
        try:
            # Base factors
            factors = {
                'message_frequency': min(
                    member_data.message_count / max(group_session.total_messages / max(group_session.member_count, 1), 1),
                    2.0
                ),
                'thread_participation': min(
                    len(member_data.conversation_threads) / max(member_data.message_count / 5, 1),
                    1.0
                ),
                'bot_interaction': min(
                    member_data.mention_count / max(member_data.message_count, 1) * 5,
                    1.0
                ),
                'recent_activity': 1.0 if (
                    member_data.last_activity and 
                    (datetime.utcnow() - member_data.last_activity).days < 1
                ) else 0.5
            }
            
            # Weighted calculation
            weights = {
                'message_frequency': 0.3,
                'thread_participation': 0.25,
                'bot_interaction': 0.25,
                'recent_activity': 0.2
            }
            
            score = sum(factors[key] * weights[key] for key in factors)
            
            # Pattern bonuses
            patterns = member_data.interaction_patterns
            if patterns.get('questions_per_message', 0) > 0.1:
                score += 0.05  # Bonus for asking questions
            if patterns.get('avg_message_length', 0) > 50:
                score += 0.05  # Bonus for detailed messages
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating member engagement: {e}")
            return 0.0


class GroupAnalyticsEngine:
    """Advanced analytics engine for group behavior analysis."""
    
    def __init__(self):
        self.sentiment_analyzer = None  # Could integrate with external service
        self.topic_modeler = None
        
        # Analytics caches
        self.hourly_stats: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.daily_stats: Dict[int, Dict[str, Any]] = defaultdict(dict)
        
    async def analyze_group_activity(
        self, 
        group_session: GroupSession,
        time_period: str = "daily"
    ) -> Dict[str, Any]:
        """Comprehensive group activity analysis."""
        try:
            async with get_async_session() as session:
                # Define time range
                now = datetime.utcnow()
                if time_period == "hourly":
                    start_time = now - timedelta(hours=1)
                elif time_period == "daily":
                    start_time = now - timedelta(days=1)
                elif time_period == "weekly":
                    start_time = now - timedelta(weeks=1)
                else:
                    start_time = now - timedelta(days=1)
                
                # Get conversation data
                conversations = await session.execute(
                    select(GroupConversation)
                    .filter(
                        and_(
                            GroupConversation.group_id == group_session.id,
                            GroupConversation.started_at >= start_time
                        )
                    )
                )
                conversations = conversations.scalars().all()
                
                # Get member activity
                members = await session.execute(
                    select(GroupMember)
                    .filter(
                        and_(
                            GroupMember.group_id == group_session.id,
                            GroupMember.is_active == True
                        )
                    )
                )
                members = members.scalars().all()
                
                # Calculate metrics
                analytics = {
                    'time_period': time_period,
                    'start_time': start_time.isoformat(),
                    'end_time': now.isoformat(),
                    
                    # Activity metrics
                    'total_conversations': len(conversations),
                    'active_members': len([m for m in members if m.last_seen_at and m.last_seen_at >= start_time]),
                    'total_messages': sum(c.message_count for c in conversations),
                    'bot_interactions': sum(c.bot_interactions for c in conversations),
                    
                    # Engagement metrics
                    'avg_conversation_length': sum(c.message_count for c in conversations) / max(len(conversations), 1),
                    'avg_participants_per_conversation': sum(c.participant_count for c in conversations) / max(len(conversations), 1),
                    'conversation_duration_avg': sum(c.duration_seconds for c in conversations) / max(len(conversations), 1),
                    
                    # Content analysis
                    'topic_distribution': self._analyze_topic_distribution(conversations),
                    'sentiment_summary': self._analyze_sentiment_distribution(conversations),
                    'language_usage': self._analyze_language_usage(conversations),
                    
                    # Temporal patterns
                    'activity_timeline': self._analyze_activity_timeline(conversations),
                    'peak_activity_hours': self._identify_peak_hours(conversations),
                    
                    # Member insights
                    'member_engagement_distribution': self._analyze_member_engagement(members),
                    'new_vs_returning_ratio': self._analyze_member_retention(members, start_time),
                    
                    # Quality metrics
                    'avg_engagement_score': sum(c.engagement_score or 0 for c in conversations) / max(len(conversations), 1),
                    'toxicity_indicators': self._analyze_toxicity_indicators(conversations)
                }
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error analyzing group activity: {e}")
            return {}
    
    def _analyze_topic_distribution(self, conversations: List[GroupConversation]) -> Dict[str, Any]:
        """Analyze distribution of conversation topics."""
        topics = {}
        total_conversations = len(conversations)
        
        for conv in conversations:
            if conv.topic:
                topics[conv.topic] = topics.get(conv.topic, 0) + 1
            
            # Also analyze keywords
            if conv.keywords:
                for keyword in conv.keywords:
                    if keyword not in topics:
                        topics[keyword] = 0
                    topics[keyword] += 0.1  # Lighter weight for keywords
        
        # Calculate percentages and return top topics
        if total_conversations > 0:
            topic_percentages = {
                topic: (count / total_conversations) * 100 
                for topic, count in topics.items()
            }
            
            # Return top 10 topics
            sorted_topics = sorted(
                topic_percentages.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                'top_topics': sorted_topics,
                'total_unique_topics': len(topics),
                'topic_diversity': len(topics) / max(total_conversations, 1)
            }
        
        return {'top_topics': [], 'total_unique_topics': 0, 'topic_diversity': 0}
    
    def _analyze_sentiment_distribution(self, conversations: List[GroupConversation]) -> Dict[str, Any]:
        """Analyze sentiment distribution across conversations."""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = []
        
        for conv in conversations:
            if conv.sentiment_summary:
                summary = conv.sentiment_summary
                if isinstance(summary, dict):
                    sentiment_counts['positive'] += summary.get('positive', 0)
                    sentiment_counts['negative'] += summary.get('negative', 0)
                    sentiment_counts['neutral'] += summary.get('neutral', 0)
                    
                    # If we have a score, track it
                    if 'score' in summary:
                        sentiment_scores.append(summary['score'])
        
        total_sentiments = sum(sentiment_counts.values())
        if total_sentiments > 0:
            return {
                'distribution': {
                    sentiment: (count / total_sentiments) * 100
                    for sentiment, count in sentiment_counts.items()
                },
                'average_score': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                'sentiment_trend': 'positive' if sentiment_counts['positive'] > sentiment_counts['negative'] else 'negative'
            }
        
        return {'distribution': {}, 'average_score': 0, 'sentiment_trend': 'neutral'}
    
    def _analyze_language_usage(self, conversations: List[GroupConversation]) -> Dict[str, Any]:
        """Analyze language usage patterns."""
        language_counts = defaultdict(int)
        
        for conv in conversations:
            if conv.language_distribution:
                if isinstance(conv.language_distribution, dict):
                    for lang, count in conv.language_distribution.items():
                        language_counts[lang] += count
        
        total_usage = sum(language_counts.values())
        if total_usage > 0:
            return {
                'languages': {
                    lang: (count / total_usage) * 100 
                    for lang, count in language_counts.items()
                },
                'primary_language': max(language_counts.items(), key=lambda x: x[1])[0],
                'language_diversity': len(language_counts)
            }
        
        return {'languages': {}, 'primary_language': None, 'language_diversity': 0}
    
    def _analyze_activity_timeline(self, conversations: List[GroupConversation]) -> List[Dict[str, Any]]:
        """Analyze activity timeline over the period."""
        timeline = []
        
        # Group conversations by hour
        hourly_activity = defaultdict(lambda: {'conversations': 0, 'messages': 0})
        
        for conv in conversations:
            if conv.started_at:
                hour_key = conv.started_at.replace(minute=0, second=0, microsecond=0)
                hourly_activity[hour_key]['conversations'] += 1
                hourly_activity[hour_key]['messages'] += conv.message_count
        
        # Convert to sorted timeline
        for hour, activity in sorted(hourly_activity.items()):
            timeline.append({
                'timestamp': hour.isoformat(),
                'conversations': activity['conversations'],
                'messages': activity['messages']
            })
        
        return timeline
    
    def _identify_peak_hours(self, conversations: List[GroupConversation]) -> Dict[str, Any]:
        """Identify peak activity hours."""
        hourly_counts = [0] * 24
        
        for conv in conversations:
            if conv.started_at:
                hour = conv.started_at.hour
                hourly_counts[hour] += conv.message_count
        
        peak_hour = hourly_counts.index(max(hourly_counts))
        
        return {
            'peak_hour': peak_hour,
            'hourly_distribution': hourly_counts,
            'peak_activity_level': max(hourly_counts),
            'activity_variance': max(hourly_counts) - min(hourly_counts)
        }
    
    def _analyze_member_engagement(self, members: List[GroupMember]) -> Dict[str, Any]:
        """Analyze member engagement distribution."""
        if not members:
            return {}
        
        engagement_scores = [m.engagement_score for m in members if m.engagement_score is not None]
        
        if not engagement_scores:
            return {}
        
        engagement_scores.sort()
        n = len(engagement_scores)
        
        return {
            'average_engagement': sum(engagement_scores) / n,
            'median_engagement': engagement_scores[n // 2],
            'high_engagement_members': len([s for s in engagement_scores if s > 0.7]),
            'low_engagement_members': len([s for s in engagement_scores if s < 0.3]),
            'engagement_distribution': {
                'high': len([s for s in engagement_scores if s > 0.7]) / n * 100,
                'medium': len([s for s in engagement_scores if 0.3 <= s <= 0.7]) / n * 100,
                'low': len([s for s in engagement_scores if s < 0.3]) / n * 100
            }
        }
    
    def _analyze_member_retention(self, members: List[GroupMember], start_time: datetime) -> Dict[str, Any]:
        """Analyze member retention patterns."""
        new_members = len([m for m in members if m.joined_at and m.joined_at >= start_time])
        returning_members = len([m for m in members if m.joined_at and m.joined_at < start_time])
        
        total_members = len(members)
        
        return {
            'new_members': new_members,
            'returning_members': returning_members,
            'new_member_ratio': (new_members / total_members * 100) if total_members > 0 else 0,
            'retention_indicators': {
                'avg_days_active': sum(
                    (datetime.utcnow() - m.joined_at).days 
                    for m in members if m.joined_at
                ) / max(total_members, 1)
            }
        }
    
    def _analyze_toxicity_indicators(self, conversations: List[GroupConversation]) -> Dict[str, Any]:
        """Analyze toxicity and conflict indicators."""
        toxicity_scores = [c.toxicity_score for c in conversations if c.toxicity_score is not None]
        
        if not toxicity_scores:
            return {'average_toxicity': 0, 'high_toxicity_conversations': 0}
        
        return {
            'average_toxicity': sum(toxicity_scores) / len(toxicity_scores),
            'high_toxicity_conversations': len([s for s in toxicity_scores if s > 0.6]),
            'toxicity_trend': 'improving' if toxicity_scores[-10:] < toxicity_scores[-20:-10] else 'stable'
        }


class GroupManager:
    """
    Main group management service that coordinates all group-related functionality.
    
    Integrates conversation threading, member engagement tracking, analytics,
    and provides high-level APIs for group interaction management.
    """
    
    def __init__(self):
        self.thread_manager = ConversationThreadManager()
        self.engagement_tracker = MemberEngagementTracker()
        self.analytics_engine = GroupAnalyticsEngine()
        
        # Group-level caches and state
        self.group_states: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.active_group_sessions: Dict[int, GroupSession] = {}
        
        # Performance monitoring
        self.operation_times: deque = deque(maxlen=1000)
        
    async def handle_group_message(
        self,
        group_session: GroupSession,
        user_id: int,
        telegram_user_id: int,
        message_content: str,
        message_id: int,
        reply_to_message_id: Optional[int] = None,
        is_bot_mentioned: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for handling group messages with full context management.
        
        Returns:
            Dictionary with conversation context, thread information, and response guidance
        """
        start_time = time.time()
        
        try:
            # Create or continue conversation thread
            thread_id = self.thread_manager.create_thread(
                message_content, [user_id], reply_to_message_id
            )
            
            # Update thread with new message
            self.thread_manager.update_thread(
                thread_id, message_content, user_id, is_bot_mentioned
            )
            
            # Track member engagement
            self.engagement_tracker.track_member_activity(
                group_session, user_id, telegram_user_id, message_content, 
                thread_id, is_bot_mentioned
            )
            
            # Get conversation context
            thread_context = self.thread_manager.get_thread_context(thread_id)
            member_engagement = self.engagement_tracker.get_member_engagement(
                group_session.id, user_id
            )
            
            # Determine response strategy
            response_strategy = await self._determine_response_strategy(
                group_session, thread_context, member_engagement, is_bot_mentioned
            )
            
            # Update group state
            self._update_group_state(group_session, thread_id, user_id, is_bot_mentioned)
            
            # Record operation time
            operation_time = time.time() - start_time
            self.operation_times.append(operation_time)
            
            return {
                'thread_id': thread_id,
                'thread_context': thread_context,
                'member_engagement': member_engagement,
                'response_strategy': response_strategy,
                'processing_time': operation_time,
                'group_state': self.group_states[group_session.id]
            }
            
        except Exception as e:
            logger.error(f"Error handling group message: {e}")
            return {'error': str(e), 'processing_time': time.time() - start_time}
    
    async def get_conversation_summary(
        self, 
        group_session: GroupSession, 
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive conversation summary for a thread."""
        thread_context = self.thread_manager.get_thread_context(thread_id)
        
        if not thread_context:
            return None
        
        # Get participant engagement data
        participant_data = []
        for participant_id in thread_context.participants:
            engagement = self.engagement_tracker.get_member_engagement(
                group_session.id, participant_id
            )
            if engagement:
                participant_data.append({
                    'user_id': participant_id,
                    'telegram_user_id': engagement.telegram_user_id,
                    'message_count': engagement.message_count,
                    'engagement_score': engagement.engagement_score,
                    'last_activity': engagement.last_activity.isoformat() if engagement.last_activity else None
                })
        
        return {
            'thread_id': thread_id,
            'state': thread_context.state,
            'importance': thread_context.importance,
            'topic': thread_context.topic,
            'keywords': list(thread_context.keywords),
            'message_count': thread_context.message_count,
            'bot_interactions': thread_context.bot_interactions,
            'participant_count': len(thread_context.participants),
            'participants': participant_data,
            'last_activity': thread_context.last_activity.isoformat() if thread_context.last_activity else None,
            'duration_seconds': (
                (datetime.utcnow() - thread_context.last_activity).total_seconds() 
                if thread_context.last_activity else 0
            )
        }
    
    async def get_group_analytics(
        self, 
        group_session: GroupSession, 
        time_period: str = "daily"
    ) -> Dict[str, Any]:
        """Get comprehensive group analytics."""
        # Get analytics from engine
        analytics = await self.analytics_engine.analyze_group_activity(
            group_session, time_period
        )
        
        # Add real-time data from thread manager
        analytics['real_time_data'] = {
            'active_threads': len(self.thread_manager.active_threads),
            'total_participants': len(self.thread_manager.thread_participants),
            'avg_thread_participants': (
                sum(len(ctx.participants) for ctx in self.thread_manager.active_threads.values()) / 
                max(len(self.thread_manager.active_threads), 1)
            )
        }
        
        # Add performance metrics
        analytics['performance_metrics'] = {
            'avg_processing_time': sum(self.operation_times) / max(len(self.operation_times), 1),
            'operations_per_minute': len([
                t for t in self.operation_times 
                if time.time() - t < 60
            ]),
            'system_load': len(self.active_group_sessions)
        }
        
        return analytics
    
    async def get_member_insights(
        self, 
        group_session: GroupSession, 
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get member insights for group or specific user."""
        if user_id:
            # Individual member insights
            engagement_data = self.engagement_tracker.get_member_engagement(
                group_session.id, user_id
            )
            
            if not engagement_data:
                return {'error': 'Member not found'}
            
            return {
                'user_id': user_id,
                'telegram_user_id': engagement_data.telegram_user_id,
                'engagement_score': engagement_data.engagement_score,
                'message_count': engagement_data.message_count,
                'mention_count': engagement_data.mention_count,
                'active_threads': len(engagement_data.conversation_threads),
                'interaction_patterns': engagement_data.interaction_patterns,
                'last_activity': engagement_data.last_activity.isoformat() if engagement_data.last_activity else None
            }
        else:
            # Group member insights
            top_members = self.engagement_tracker.get_top_engaged_members(
                group_session, limit=20
            )
            
            return {
                'total_tracked_members': len(self.engagement_tracker.member_data),
                'top_engaged_members': [
                    {
                        'user_id': member.user_id,
                        'telegram_user_id': member.telegram_user_id,
                        'engagement_score': member.engagement_score,
                        'message_count': member.message_count,
                        'thread_participation': len(member.conversation_threads)
                    }
                    for member in top_members
                ],
                'engagement_distribution': self._calculate_engagement_distribution(top_members)
            }
    
    async def cleanup_inactive_data(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up inactive threads and old data."""
        cleanup_stats = {
            'threads_cleaned': 0,
            'members_cleaned': 0,
            'groups_cleaned': 0
        }
        
        # Clean up old threads
        self.thread_manager._cleanup_old_threads()
        
        # Clean up inactive member data
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        members_to_remove = []
        for member_key, member_data in self.engagement_tracker.member_data.items():
            if (member_data.last_activity and 
                member_data.last_activity < cutoff_time):
                members_to_remove.append(member_key)
        
        for member_key in members_to_remove:
            del self.engagement_tracker.member_data[member_key]
            if member_key in self.engagement_tracker.engagement_history:
                del self.engagement_tracker.engagement_history[member_key]
        
        cleanup_stats['members_cleaned'] = len(members_to_remove)
        
        # Clean up inactive group states
        groups_to_clean = []
        for group_id, state in self.group_states.items():
            last_activity = state.get('last_activity')
            if (isinstance(last_activity, datetime) and 
                last_activity < cutoff_time):
                groups_to_clean.append(group_id)
        
        for group_id in groups_to_clean:
            del self.group_states[group_id]
            if group_id in self.active_group_sessions:
                del self.active_group_sessions[group_id]
        
        cleanup_stats['groups_cleaned'] = len(groups_to_clean)
        cleanup_stats['threads_cleaned'] = len(self.thread_manager.active_threads)
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    # Private helper methods
    
    async def _determine_response_strategy(
        self,
        group_session: GroupSession,
        thread_context: Optional[ConversationContext],
        member_engagement: Optional[MemberEngagementData],
        is_bot_mentioned: bool
    ) -> Dict[str, Any]:
        """Determine appropriate response strategy based on context."""
        strategy = {
            'should_respond': False,
            'response_type': 'none',
            'priority': 'low',
            'context_aware': False,
            'personalized': False
        }
        
        # Bot mentioned - always consider responding
        if is_bot_mentioned:
            strategy['should_respond'] = True
            strategy['response_type'] = 'direct_mention'
            strategy['priority'] = 'high'
        
        # High engagement member
        if member_engagement and member_engagement.engagement_score > 0.7:
            strategy['personalized'] = True
            strategy['priority'] = 'high'
        
        # Active conversation thread
        if thread_context and thread_context.state == ConversationState.ACTIVE:
            strategy['context_aware'] = True
            if thread_context.bot_interactions > 0:
                strategy['should_respond'] = True
                strategy['response_type'] = 'conversation_continuation'
        
        # Group settings influence
        if group_session.get_setting('proactive_responses', False):
            if thread_context and thread_context.importance == ThreadImportance.HIGH:
                strategy['should_respond'] = True
                strategy['response_type'] = 'proactive'
        
        return strategy
    
    def _update_group_state(
        self, 
        group_session: GroupSession, 
        thread_id: str, 
        user_id: int, 
        is_bot_mentioned: bool
    ) -> None:
        """Update group-level state tracking."""
        group_id = group_session.id
        state = self.group_states[group_id]
        
        # Update activity tracking
        state['last_activity'] = datetime.utcnow()
        state['recent_threads'] = state.get('recent_threads', set())
        state['recent_threads'].add(thread_id)
        
        # Keep only recent threads (last 10)
        if len(state['recent_threads']) > 10:
            state['recent_threads'] = set(list(state['recent_threads'])[-10:])
        
        # Update participant tracking
        state['recent_participants'] = state.get('recent_participants', set())
        state['recent_participants'].add(user_id)
        
        # Bot interaction tracking
        if is_bot_mentioned:
            state['last_bot_interaction'] = datetime.utcnow()
            state['bot_mentions_today'] = state.get('bot_mentions_today', 0) + 1
        
        # Cache group session
        self.active_group_sessions[group_id] = group_session
    
    def _calculate_engagement_distribution(
        self, 
        members: List[MemberEngagementData]
    ) -> Dict[str, Any]:
        """Calculate engagement score distribution statistics."""
        if not members:
            return {}
        
        scores = [m.engagement_score for m in members]
        scores.sort()
        
        return {
            'count': len(scores),
            'mean': sum(scores) / len(scores),
            'median': scores[len(scores) // 2],
            'quartiles': {
                'q1': scores[len(scores) // 4],
                'q3': scores[3 * len(scores) // 4]
            },
            'distribution': {
                'high': len([s for s in scores if s > 0.7]),
                'medium': len([s for s in scores if 0.3 <= s <= 0.7]),
                'low': len([s for s in scores if s < 0.3])
            }
        }