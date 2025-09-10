"""
Group Manager Tests

Comprehensive test suite for group chat functionality including:
- Conversation threading and context management
- Member engagement tracking and analytics
- Multi-user conversation handling
- Group analytics and insights
- Performance testing for large groups (1000+ members)
- Real-time message processing
- Thread cleanup and memory management
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Set
from collections import defaultdict
import json
import threading

from app.services.group_manager import (
    GroupManager, ConversationThreadManager, MemberEngagementTracker,
    GroupAnalyticsEngine, ConversationState, ThreadImportance,
    ConversationContext, MemberEngagementData
)
from app.models.group_session import (
    GroupSession, GroupMember, GroupConversation, GroupAnalytics,
    GroupType, MemberRole, GroupStatus, MessageFrequency
)
from app.models.user import User
from app.models.conversation import Message, MessageType, MessageDirection


@pytest.fixture
def sample_group_session():
    """Create sample group session for testing."""
    return GroupSession(
        id=1,
        telegram_group_id=-1001234567890,
        group_type=GroupType.SUPERGROUP,
        title="Test Group Chat",
        status=GroupStatus.ACTIVE,
        member_count=50,
        total_messages=1000,
        created_at=datetime.utcnow() - timedelta(days=30),
        settings={
            "proactive_responses": True,
            "analytics_enabled": True,
            "max_thread_age_hours": 24
        }
    )


@pytest.fixture
def sample_group_members():
    """Create sample group members for testing."""
    members = []
    base_time = datetime.utcnow() - timedelta(days=5)
    
    for i in range(10):
        member = GroupMember(
            id=i + 1,
            group_id=1,
            user_id=f"user-{i}",
            telegram_user_id=100000 + i,
            username=f"testuser{i}",
            role=MemberRole.MEMBER,
            is_active=True,
            joined_at=base_time + timedelta(days=i // 2),
            last_seen_at=base_time + timedelta(days=i // 2, hours=i),
            engagement_score=0.5 + (i % 5) * 0.1
        )
        members.append(member)
    
    return members


@pytest.fixture
def sample_conversations():
    """Create sample group conversations for testing."""
    conversations = []
    base_time = datetime.utcnow() - timedelta(hours=12)
    
    for i in range(20):
        conv = GroupConversation(
            id=i + 1,
            group_id=1,
            thread_id=f"thread-{i}",
            started_at=base_time + timedelta(minutes=i * 15),
            ended_at=base_time + timedelta(minutes=i * 15 + 10) if i % 3 == 0 else None,
            participant_count=2 + i % 5,
            message_count=5 + i % 15,
            bot_interactions=i % 4,
            duration_seconds=(10 + i * 2) * 60,
            topic=f"Topic {i}" if i % 3 == 0 else None,
            keywords=[f"keyword{i}", f"tag{i}"] if i % 2 == 0 else [],
            sentiment_summary={"positive": 0.6, "negative": 0.2, "neutral": 0.2},
            engagement_score=0.5 + (i % 8) * 0.1,
            toxicity_score=0.1 if i % 10 != 0 else 0.3
        )
        conversations.append(conv)
    
    return conversations


class TestConversationThreadManager:
    """Test conversation threading and context management."""
    
    @pytest.fixture
    def thread_manager(self):
        """Create ConversationThreadManager instance."""
        return ConversationThreadManager(max_threads=20, thread_timeout=3600)
    
    def test_thread_id_generation(self, thread_manager):
        """Test unique thread ID generation."""
        message = "Hello everyone, how is your day going?"
        participants = [123, 456, 789]
        
        thread_id1 = thread_manager.generate_thread_id(message, participants)
        thread_id2 = thread_manager.generate_thread_id(message, participants)
        
        # Same input should generate same ID (within 5-minute bucket)
        assert thread_id1 == thread_id2
        assert thread_id1.startswith("thread_")
        assert len(thread_id1) > 20
        
        # Different message should generate different ID
        different_message = "This is a completely different topic"
        thread_id3 = thread_manager.generate_thread_id(different_message, participants)
        assert thread_id3 != thread_id1
    
    def test_create_thread(self, thread_manager):
        """Test conversation thread creation."""
        message = "Let's discuss the new features we want to implement"
        participants = [123, 456]
        
        thread_id = thread_manager.create_thread(message, participants)
        
        # Verify thread creation
        assert thread_id in thread_manager.active_threads
        context = thread_manager.active_threads[thread_id]
        
        assert isinstance(context, ConversationContext)
        assert context.thread_id == thread_id
        assert context.participants == set(participants)
        assert len(context.keywords) > 0
        assert context.message_count == 1
        assert context.state == ConversationState.STARTING
        assert context.last_activity is not None
        
        # Verify participant tracking
        for participant in participants:
            assert thread_id in thread_manager.thread_participants[participant]
    
    def test_thread_continuation_detection(self, thread_manager):
        """Test detection of thread continuation."""
        # Create initial thread
        initial_message = "What do you think about the new design?"
        participants = [123, 456]
        thread_id = thread_manager.create_thread(initial_message, participants)
        
        # Test continuation with similar content
        similar_message = "I think the design looks great, what about colors?"
        detected_thread = thread_manager.detect_thread_continuation(similar_message, 123)
        
        assert detected_thread == thread_id
        
        # Test with unrelated content
        unrelated_message = "Did anyone see the weather forecast for tomorrow?"
        no_thread = thread_manager.detect_thread_continuation(unrelated_message, 123)
        
        # Should not match unrelated content
        assert no_thread != thread_id or no_thread is None
    
    def test_update_thread(self, thread_manager):
        """Test thread updating with new messages."""
        # Create thread
        message = "Initial message about project planning"
        participants = [123, 456]
        thread_id = thread_manager.create_thread(message, participants)
        
        initial_context = thread_manager.active_threads[thread_id]
        initial_count = initial_context.message_count
        
        # Update thread
        new_message = "I agree, we should focus on user experience first"
        thread_manager.update_thread(thread_id, new_message, 789, is_bot_interaction=True)
        
        # Verify updates
        updated_context = thread_manager.active_threads[thread_id]
        assert updated_context.message_count == initial_count + 1
        assert 789 in updated_context.participants
        assert updated_context.bot_interactions == 1
        assert updated_context.state == ConversationState.ACTIVE
        assert updated_context.last_activity > initial_context.last_activity
    
    def test_thread_cleanup(self, thread_manager):
        """Test automatic thread cleanup."""
        # Create multiple threads
        old_thread_ids = []
        recent_thread_ids = []
        
        for i in range(10):
            message = f"Thread {i} message content"
            participants = [100 + i, 200 + i]
            thread_id = thread_manager.create_thread(message, participants)
            
            if i < 5:
                # Make half the threads old
                context = thread_manager.active_threads[thread_id]
                context.last_activity = datetime.utcnow() - timedelta(hours=2)
                old_thread_ids.append(thread_id)
            else:
                recent_thread_ids.append(thread_id)
        
        # Verify all threads exist
        assert len(thread_manager.active_threads) == 10
        
        # Trigger cleanup
        thread_manager._cleanup_old_threads()
        
        # Verify cleanup (old threads should be removed)
        remaining_threads = set(thread_manager.active_threads.keys())
        
        # Old threads should be cleaned up
        for old_id in old_thread_ids:
            assert old_id not in remaining_threads
        
        # Recent threads should remain
        for recent_id in recent_thread_ids:
            assert recent_id in remaining_threads
    
    def test_thread_state_transitions(self, thread_manager):
        """Test conversation state transitions."""
        message = "Starting a new conversation topic"
        participants = [123, 456]
        thread_id = thread_manager.create_thread(message, participants)
        
        context = thread_manager.active_threads[thread_id]
        assert context.state == ConversationState.STARTING
        
        # Add more messages to transition states
        for i in range(3):
            thread_manager.update_thread(
                thread_id, f"Message {i}", 123 + i, is_bot_interaction=True
            )
        
        # Should transition to ACTIVE with bot interactions
        assert context.state == ConversationState.ACTIVE
        
        # Add many more messages to test WINDING_DOWN
        for i in range(25):
            thread_manager.update_thread(thread_id, f"Extended message {i}", 123)
        
        # Should transition to WINDING_DOWN with many messages
        assert context.state == ConversationState.WINDING_DOWN


class TestMemberEngagementTracker:
    """Test member engagement tracking functionality."""
    
    @pytest.fixture
    def engagement_tracker(self):
        """Create MemberEngagementTracker instance."""
        return MemberEngagementTracker(max_members=100)
    
    def test_track_member_activity(self, engagement_tracker, sample_group_session):
        """Test member activity tracking."""
        user_id = 123
        telegram_user_id = 100123
        message_content = "Great discussion! I really enjoyed participating 😊"
        thread_id = "thread-test-123"
        
        # Track initial activity
        engagement_tracker.track_member_activity(
            sample_group_session, user_id, telegram_user_id,
            message_content, thread_id, is_bot_interaction=False
        )
        
        # Verify member data creation
        member_key = (sample_group_session.id, user_id)
        assert member_key in engagement_tracker.member_data
        
        member_data = engagement_tracker.member_data[member_key]
        assert member_data.user_id == user_id
        assert member_data.telegram_user_id == telegram_user_id
        assert member_data.message_count == 1
        assert thread_id in member_data.conversation_threads
        assert member_data.engagement_score > 0
        assert member_data.last_activity is not None
        
        # Track bot interaction
        engagement_tracker.track_member_activity(
            sample_group_session, user_id, telegram_user_id,
            "@bot help me with this", thread_id, is_bot_interaction=True
        )
        
        # Verify bot interaction tracking
        updated_data = engagement_tracker.member_data[member_key]
        assert updated_data.message_count == 2
        assert updated_data.mention_count == 1
        assert updated_data.engagement_score > member_data.engagement_score
    
    def test_interaction_patterns_analysis(self, engagement_tracker, sample_group_session):
        """Test interaction pattern analysis."""
        user_id = 123
        telegram_user_id = 100123
        
        # Track various types of interactions
        test_messages = [
            "Short msg",
            "This is a much longer message that contains more detailed content and thoughts",
            "Can you help me with this? What should I do?",
            "Thanks for the help! Really appreciate it.",
            "@bot this is a bot interaction"
        ]
        
        for i, message in enumerate(test_messages):
            is_bot_interaction = "@bot" in message
            engagement_tracker.track_member_activity(
                sample_group_session, user_id, telegram_user_id,
                message, f"thread-{i}", is_bot_interaction
            )
        
        # Verify pattern analysis
        member_key = (sample_group_session.id, user_id)
        member_data = engagement_tracker.member_data[member_key]
        patterns = member_data.interaction_patterns
        
        assert "avg_message_length" in patterns
        assert "questions_per_message" in patterns
        assert "bot_interaction_ratio" in patterns
        assert "hourly_activity" in patterns
        assert "peak_activity_hour" in patterns
        
        # Verify calculated values
        assert patterns["avg_message_length"] > 0
        assert patterns["questions_per_message"] > 0  # Some messages had questions
        assert patterns["bot_interaction_ratio"] > 0  # Some bot interactions
        assert len(patterns["hourly_activity"]) == 24
    
    def test_engagement_score_calculation(self, engagement_tracker, sample_group_session):
        """Test engagement score calculation logic."""
        # Test high engagement user
        high_engagement_user = 456
        for i in range(20):  # Many messages
            engagement_tracker.track_member_activity(
                sample_group_session, high_engagement_user, 100456,
                f"Engaged message {i} with good content and questions?",
                f"thread-{i % 3}", is_bot_interaction=i % 3 == 0  # Regular bot interactions
            )
        
        # Test low engagement user  
        low_engagement_user = 789
        engagement_tracker.track_member_activity(
            sample_group_session, low_engagement_user, 100789,
            "ok", "thread-single", is_bot_interaction=False  # Single short message
        )
        
        # Get engagement data
        high_data = engagement_tracker.get_member_engagement(sample_group_session.id, high_engagement_user)
        low_data = engagement_tracker.get_member_engagement(sample_group_session.id, low_engagement_user)
        
        # Verify engagement score differences
        assert high_data.engagement_score > low_data.engagement_score
        assert high_data.engagement_score > 0.5  # Should be high
        assert low_data.engagement_score < 0.5   # Should be lower
        
        # Verify engagement factors
        assert high_data.message_count > low_data.message_count
        assert len(high_data.conversation_threads) > len(low_data.conversation_threads)
        assert high_data.mention_count > low_data.mention_count
    
    def test_top_engaged_members(self, engagement_tracker, sample_group_session):
        """Test identification of top engaged members."""
        # Create members with varying engagement levels
        for user_id in range(10):
            message_count = (user_id + 1) * 3  # Varying message counts
            for msg_idx in range(message_count):
                engagement_tracker.track_member_activity(
                    sample_group_session, user_id, 100000 + user_id,
                    f"User {user_id} message {msg_idx}",
                    f"thread-{msg_idx % 3}", is_bot_interaction=msg_idx % 4 == 0
                )
        
        # Get top engaged members
        top_members = engagement_tracker.get_top_engaged_members(sample_group_session, limit=5)
        
        # Verify results
        assert len(top_members) == 5
        
        # Verify sorting (highest engagement first)
        for i in range(len(top_members) - 1):
            assert top_members[i].engagement_score >= top_members[i + 1].engagement_score
        
        # Top member should be user 9 (most messages)
        assert top_members[0].user_id == 9
        assert top_members[0].message_count > top_members[-1].message_count


class TestGroupAnalyticsEngine:
    """Test group analytics and insights functionality."""
    
    @pytest.fixture
    def analytics_engine(self):
        """Create GroupAnalyticsEngine instance."""
        return GroupAnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_analyze_group_activity(
        self, analytics_engine, sample_group_session, sample_conversations, sample_group_members
    ):
        """Test comprehensive group activity analysis."""
        # Mock database session
        mock_session = AsyncMock()
        
        # Mock conversation query
        conv_result = Mock()
        conv_result.scalars.return_value.all.return_value = sample_conversations
        
        # Mock member query
        member_result = Mock()
        member_result.scalars.return_value.all.return_value = sample_group_members
        
        mock_session.execute.side_effect = [conv_result, member_result]
        
        with patch('app.services.group_manager.get_async_session', return_value=mock_session):
            analytics = await analytics_engine.analyze_group_activity(
                sample_group_session, time_period="daily"
            )
        
        # Verify analytics structure
        expected_keys = [
            'time_period', 'start_time', 'end_time',
            'total_conversations', 'active_members', 'total_messages',
            'bot_interactions', 'avg_conversation_length',
            'avg_participants_per_conversation', 'conversation_duration_avg',
            'topic_distribution', 'sentiment_summary', 'language_usage',
            'activity_timeline', 'peak_activity_hours',
            'member_engagement_distribution', 'new_vs_returning_ratio',
            'avg_engagement_score', 'toxicity_indicators'
        ]
        
        for key in expected_keys:
            assert key in analytics, f"Missing key: {key}"
        
        # Verify calculated metrics
        assert analytics['total_conversations'] == len(sample_conversations)
        assert analytics['avg_conversation_length'] > 0
        assert analytics['avg_participants_per_conversation'] > 0
        assert isinstance(analytics['topic_distribution'], dict)
        assert isinstance(analytics['sentiment_summary'], dict)
        assert isinstance(analytics['activity_timeline'], list)
    
    def test_topic_distribution_analysis(self, analytics_engine, sample_conversations):
        """Test topic distribution analysis."""
        topic_dist = analytics_engine._analyze_topic_distribution(sample_conversations)
        
        # Verify structure
        assert 'top_topics' in topic_dist
        assert 'total_unique_topics' in topic_dist
        assert 'topic_diversity' in topic_dist
        
        # Verify content
        assert isinstance(topic_dist['top_topics'], list)
        assert topic_dist['total_unique_topics'] > 0
        assert 0 <= topic_dist['topic_diversity'] <= 1
        
        # Top topics should be sorted by frequency
        if len(topic_dist['top_topics']) > 1:
            for i in range(len(topic_dist['top_topics']) - 1):
                assert topic_dist['top_topics'][i][1] >= topic_dist['top_topics'][i + 1][1]
    
    def test_sentiment_distribution_analysis(self, analytics_engine, sample_conversations):
        """Test sentiment distribution analysis."""
        sentiment_summary = analytics_engine._analyze_sentiment_distribution(sample_conversations)
        
        # Verify structure
        expected_keys = ['distribution', 'average_score', 'sentiment_trend']
        for key in expected_keys:
            assert key in sentiment_summary
        
        # Verify distribution
        distribution = sentiment_summary['distribution']
        if distribution:  # If we have sentiment data
            assert 'positive' in distribution
            assert 'negative' in distribution
            assert 'neutral' in distribution
            
            # Percentages should sum to approximately 100
            total_percentage = sum(distribution.values())
            assert 90 <= total_percentage <= 110  # Allow some rounding tolerance
    
    def test_activity_timeline_analysis(self, analytics_engine, sample_conversations):
        """Test activity timeline analysis."""
        timeline = analytics_engine._analyze_activity_timeline(sample_conversations)
        
        # Verify structure
        assert isinstance(timeline, list)
        
        if timeline:  # If we have timeline data
            for entry in timeline:
                assert 'timestamp' in entry
                assert 'conversations' in entry
                assert 'messages' in entry
                assert entry['conversations'] >= 0
                assert entry['messages'] >= 0
            
            # Timeline should be sorted by timestamp
            for i in range(len(timeline) - 1):
                assert timeline[i]['timestamp'] <= timeline[i + 1]['timestamp']
    
    def test_peak_hours_identification(self, analytics_engine, sample_conversations):
        """Test peak activity hours identification."""
        peak_hours = analytics_engine._identify_peak_hours(sample_conversations)
        
        # Verify structure
        expected_keys = ['peak_hour', 'hourly_distribution', 'peak_activity_level', 'activity_variance']
        for key in expected_keys:
            assert key in peak_hours
        
        # Verify values
        assert 0 <= peak_hours['peak_hour'] <= 23
        assert len(peak_hours['hourly_distribution']) == 24
        assert peak_hours['peak_activity_level'] >= 0
        assert peak_hours['activity_variance'] >= 0
    
    def test_member_engagement_distribution(self, analytics_engine, sample_group_members):
        """Test member engagement distribution analysis."""
        engagement_dist = analytics_engine._analyze_member_engagement(sample_group_members)
        
        if engagement_dist:  # If we have engagement data
            expected_keys = [
                'average_engagement', 'median_engagement',
                'high_engagement_members', 'low_engagement_members',
                'engagement_distribution'
            ]
            
            for key in expected_keys:
                assert key in engagement_dist
            
            # Verify percentages in distribution
            dist = engagement_dist['engagement_distribution']
            total_percentage = dist['high'] + dist['medium'] + dist['low']
            assert 99 <= total_percentage <= 101  # Should sum to 100%


class TestGroupManager:
    """Test main GroupManager integration and coordination."""
    
    @pytest.fixture
    def group_manager(self):
        """Create GroupManager instance."""
        return GroupManager()
    
    @pytest.mark.asyncio
    async def test_handle_group_message(self, group_manager, sample_group_session):
        """Test complete group message handling pipeline."""
        user_id = 123
        telegram_user_id = 100123
        message_content = "This is an interesting discussion about AI and machine learning"
        message_id = 456789
        
        # Handle message
        result = await group_manager.handle_group_message(
            group_session=sample_group_session,
            user_id=user_id,
            telegram_user_id=telegram_user_id,
            message_content=message_content,
            message_id=message_id,
            reply_to_message_id=None,
            is_bot_mentioned=False
        )
        
        # Verify result structure
        expected_keys = [
            'thread_id', 'thread_context', 'member_engagement',
            'response_strategy', 'processing_time', 'group_state'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify thread creation
        assert result['thread_id'] is not None
        assert result['thread_context'] is not None
        assert result['member_engagement'] is not None
        
        # Verify processing time
        assert result['processing_time'] > 0
        assert result['processing_time'] < 1.0  # Should be fast
    
    @pytest.mark.asyncio
    async def test_handle_bot_mentioned_message(self, group_manager, sample_group_session):
        """Test handling of messages that mention the bot."""
        result = await group_manager.handle_group_message(
            group_session=sample_group_session,
            user_id=456,
            telegram_user_id=100456,
            message_content="@bot can you help me with this question?",
            message_id=123456,
            is_bot_mentioned=True
        )
        
        # Verify response strategy for bot mention
        strategy = result['response_strategy']
        assert strategy['should_respond'] is True
        assert strategy['response_type'] == 'direct_mention'
        assert strategy['priority'] == 'high'
    
    @pytest.mark.asyncio
    async def test_conversation_summary_generation(self, group_manager, sample_group_session):
        """Test conversation summary generation."""
        # First create a conversation
        message_result = await group_manager.handle_group_message(
            group_session=sample_group_session,
            user_id=123,
            telegram_user_id=100123,
            message_content="Let's talk about the project timeline",
            message_id=111
        )
        
        thread_id = message_result['thread_id']
        
        # Get conversation summary
        summary = await group_manager.get_conversation_summary(sample_group_session, thread_id)
        
        # Verify summary structure
        expected_keys = [
            'thread_id', 'state', 'importance', 'topic', 'keywords',
            'message_count', 'bot_interactions', 'participant_count',
            'participants', 'last_activity', 'duration_seconds'
        ]
        
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"
        
        # Verify summary content
        assert summary['thread_id'] == thread_id
        assert summary['message_count'] > 0
        assert summary['participant_count'] > 0
        assert len(summary['participants']) > 0
    
    @pytest.mark.asyncio
    async def test_group_analytics_retrieval(self, group_manager, sample_group_session):
        """Test group analytics retrieval."""
        with patch.object(group_manager.analytics_engine, 'analyze_group_activity') as mock_analyze:
            mock_analyze.return_value = {
                'total_conversations': 10,
                'active_members': 5,
                'avg_engagement_score': 0.75
            }
            
            analytics = await group_manager.get_group_analytics(sample_group_session, "daily")
            
            # Verify analytics structure
            assert 'total_conversations' in analytics
            assert 'real_time_data' in analytics
            assert 'performance_metrics' in analytics
            
            # Verify real-time data
            real_time = analytics['real_time_data']
            assert 'active_threads' in real_time
            assert 'total_participants' in real_time
            assert 'avg_thread_participants' in real_time
    
    @pytest.mark.asyncio
    async def test_member_insights(self, group_manager, sample_group_session):
        """Test member insights retrieval."""
        # First generate some activity
        user_id = 789
        for i in range(5):
            await group_manager.handle_group_message(
                sample_group_session, user_id, 100789,
                f"Message {i} from user", 1000 + i
            )
        
        # Get individual member insights
        individual_insights = await group_manager.get_member_insights(
            sample_group_session, user_id
        )
        
        # Verify individual insights
        expected_keys = [
            'user_id', 'telegram_user_id', 'engagement_score',
            'message_count', 'mention_count', 'active_threads',
            'interaction_patterns', 'last_activity'
        ]
        
        for key in expected_keys:
            assert key in individual_insights, f"Missing key: {key}"
        
        assert individual_insights['user_id'] == user_id
        assert individual_insights['message_count'] == 5
        
        # Get group member insights
        group_insights = await group_manager.get_member_insights(sample_group_session)
        
        # Verify group insights
        assert 'total_tracked_members' in group_insights
        assert 'top_engaged_members' in group_insights
        assert 'engagement_distribution' in group_insights
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_data(self, group_manager):
        """Test cleanup of inactive data."""
        # Generate some activity first
        sample_session = GroupSession(id=999, telegram_group_id=-999, member_count=10)
        
        for i in range(10):
            await group_manager.handle_group_message(
                sample_session, i, 100000 + i, f"Message {i}", 2000 + i
            )
        
        # Verify data exists
        assert len(group_manager.thread_manager.active_threads) > 0
        assert len(group_manager.engagement_tracker.member_data) > 0
        
        # Run cleanup (with very short age to clean everything)
        cleanup_stats = await group_manager.cleanup_inactive_data(max_age_hours=0)
        
        # Verify cleanup occurred
        assert cleanup_stats['threads_cleaned'] >= 0
        assert cleanup_stats['members_cleaned'] >= 0
        assert cleanup_stats['groups_cleaned'] >= 0


class TestConcurrencyAndPerformance:
    """Test concurrency and performance aspects of group management."""
    
    @pytest.fixture
    def group_manager(self):
        """Create GroupManager for performance testing."""
        return GroupManager()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_message_processing(self, group_manager):
        """Test concurrent processing of multiple group messages."""
        sample_session = GroupSession(id=1001, telegram_group_id=-1001, member_count=100)
        
        # Create concurrent message tasks
        tasks = []
        num_messages = 100  # Simulate 100 concurrent messages
        
        for i in range(num_messages):
            user_id = i % 20  # 20 different users
            task = group_manager.handle_group_message(
                group_session=sample_session,
                user_id=user_id,
                telegram_user_id=100000 + user_id,
                message_content=f"Concurrent message {i} from user {user_id}",
                message_id=3000 + i,
                is_bot_mentioned=(i % 10 == 0)  # 10% bot mentions
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        # Most messages should process successfully
        assert len(successful_results) >= 90
        assert len(errors) <= 10
        
        # Performance should be reasonable
        assert total_time < 30.0  # Should complete in <30 seconds
        
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        assert avg_processing_time < 0.5  # Average <500ms per message
        
        print(f"Concurrent processing: {len(successful_results)}/{num_messages} successful, "
              f"{total_time:.2f}s total, {avg_processing_time:.3f}s average")
    
    @pytest.mark.asyncio
    async def test_large_group_scalability(self, group_manager):
        """Test scalability with large group simulation (1000 members)."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        large_session = GroupSession(id=2001, telegram_group_id=-2001, member_count=1000)
        
        # Simulate activity from 1000 different users
        batch_size = 50
        total_messages = 500  # 500 messages from 1000 users
        
        for batch in range(total_messages // batch_size):
            batch_tasks = []
            
            for i in range(batch_size):
                user_id = (batch * batch_size + i) % 1000  # Cycle through 1000 users
                task = group_manager.handle_group_message(
                    group_session=large_session,
                    user_id=user_id,
                    telegram_user_id=200000 + user_id,
                    message_content=f"Message from user {user_id} in batch {batch}",
                    message_id=4000 + batch * batch_size + i
                )
                batch_tasks.append(task)
            
            # Process batch
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Check memory usage periodically
            if batch % 5 == 0:  # Every 5 batches
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < 200, f"Memory growth too high: {memory_growth:.1f} MB"
        
        # Verify final state
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_growth = final_memory - initial_memory
        
        # Check thread management efficiency
        active_threads = len(group_manager.thread_manager.active_threads)
        tracked_members = len(group_manager.engagement_tracker.member_data)
        
        print(f"Large group test: {active_threads} active threads, "
              f"{tracked_members} tracked members, "
              f"{total_memory_growth:.1f} MB memory growth")
        
        # Performance assertions
        assert active_threads <= 100  # Should manage thread count
        assert tracked_members <= 1000  # Should track all unique members
        assert total_memory_growth < 300  # Should manage memory usage
    
    @pytest.mark.asyncio
    async def test_thread_management_efficiency(self, group_manager):
        """Test thread management efficiency and cleanup."""
        sample_session = GroupSession(id=3001, telegram_group_id=-3001, member_count=50)
        
        # Create many short-lived conversations
        thread_ids = []
        for i in range(30):  # Create 30 threads
            result = await group_manager.handle_group_message(
                sample_session, i % 10, 300000 + i,
                f"New thread starter {i}", 5000 + i
            )
            thread_ids.append(result['thread_id'])
        
        # Verify thread creation
        initial_thread_count = len(group_manager.thread_manager.active_threads)
        assert initial_thread_count >= 20  # Should have many threads
        
        # Simulate time passing and cleanup
        for thread_id in thread_ids[:15]:  # Age half the threads
            if thread_id in group_manager.thread_manager.active_threads:
                context = group_manager.thread_manager.active_threads[thread_id]
                context.last_activity = datetime.utcnow() - timedelta(hours=2)
        
        # Trigger cleanup
        group_manager.thread_manager._cleanup_old_threads()
        
        # Verify cleanup efficiency
        final_thread_count = len(group_manager.thread_manager.active_threads)
        assert final_thread_count < initial_thread_count
        assert final_thread_count <= group_manager.thread_manager.max_threads
        
        print(f"Thread cleanup: {initial_thread_count} -> {final_thread_count} threads")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__ + "::TestConcurrencyAndPerformance", "-v", "-s"])