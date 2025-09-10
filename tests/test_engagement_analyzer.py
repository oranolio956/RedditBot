"""
Engagement Analyzer Tests

Comprehensive test suite for engagement analysis functionality including:
- User interaction analysis with sentiment detection
- Behavioral pattern recognition and tracking
- Proactive engagement identification
- Churn risk calculation and prediction
- Milestone tracking and progress analysis
- Performance testing for real-time analysis
- LLM integration for sentiment analysis
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.engagement_analyzer import EngagementAnalyzer
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, EngagementType, SentimentType,
    EngagementMilestone, UserMilestoneProgress
)
from app.models.user import User
from app.services.llm_service import LLMService


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = Mock(spec=LLMService)
    service.get_completion = AsyncMock()
    service.generate_completion = AsyncMock()
    return service


@pytest.fixture
def engagement_analyzer(mock_llm_service):
    """Create EngagementAnalyzer instance with mocked dependencies."""
    return EngagementAnalyzer(llm_service=mock_llm_service)


@pytest.fixture
def sample_user():
    """Create sample user for testing."""
    return User(
        id="user-123",
        telegram_id=123456789,
        username="testuser",
        first_name="Test",
        last_name="User",
        language_code="en",
        is_active=True,
        is_blocked=False
    )


@pytest.fixture
def sample_engagement_data():
    """Sample engagement data for testing."""
    return {
        "user_id": "user-123",
        "telegram_id": 123456789,
        "engagement_type": EngagementType.MESSAGE,
        "message_text": "This is a great feature! I love using this bot 😊",
        "command_name": None,
        "session_id": "session-456",
        "previous_bot_message": "How are you enjoying the bot?",
        "response_time_seconds": 45
    }


class TestEngagementAnalyzer:
    """Test core engagement analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_user_interaction_positive_sentiment(
        self, engagement_analyzer, db_session, sample_engagement_data
    ):
        """Test analysis of positive user interaction."""
        with patch('app.database.connection.get_database_session', return_value=db_session):
            engagement = await engagement_analyzer.analyze_user_interaction(
                **sample_engagement_data
            )
        
        # Verify engagement record creation
        assert engagement.user_id == "user-123"
        assert engagement.telegram_id == 123456789
        assert engagement.engagement_type == EngagementType.MESSAGE
        assert engagement.message_text == sample_engagement_data["message_text"]
        assert engagement.sentiment_type in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]
        assert engagement.sentiment_score > 0
        assert engagement.contains_emoji is True
        assert engagement.is_meaningful_interaction is True
        assert engagement.engagement_quality_score > 0.5
        
        # Verify analysis data
        assert engagement.message_length > 0
        assert engagement.response_time_seconds == 45
        assert len(engagement.detected_topics) > 0
        assert engagement.user_intent is not None
        assert engagement.mood_indicators is not None
    
    @pytest.mark.asyncio
    async def test_analyze_user_interaction_negative_sentiment(
        self, engagement_analyzer, db_session
    ):
        """Test analysis of negative user interaction."""
        negative_data = {
            "user_id": "user-123",
            "telegram_id": 123456789,
            "engagement_type": EngagementType.MESSAGE,
            "message_text": "This bot is terrible and frustrating! I hate it 😡",
            "response_time_seconds": 120
        }
        
        with patch('app.database.connection.get_database_session', return_value=db_session):
            engagement = await engagement_analyzer.analyze_user_interaction(**negative_data)
        
        # Verify negative sentiment detection
        assert engagement.sentiment_type in [SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE]
        assert engagement.sentiment_score < 0
        assert engagement.contains_emoji is True
        assert engagement.mood_indicators["frustration_level"] > 0
        assert "negative_feedback" in engagement.user_intent
    
    @pytest.mark.asyncio
    async def test_analyze_user_interaction_with_questions(
        self, engagement_analyzer, db_session
    ):
        """Test analysis of interaction containing questions."""
        question_data = {
            "user_id": "user-123",
            "telegram_id": 123456789,
            "engagement_type": EngagementType.MESSAGE,
            "message_text": "How do I use this feature? What are the options available?",
            "response_time_seconds": 30
        }
        
        with patch('app.database.connection.get_database_session', return_value=db_session):
            engagement = await engagement_analyzer.analyze_user_interaction(**question_data)
        
        # Verify question detection
        assert engagement.contains_question is True
        assert engagement.user_intent == "question"
        assert "help" in engagement.detected_topics
        assert engagement.engagement_quality_score > 0.6  # Questions boost engagement
        assert engagement.is_meaningful_interaction is True
    
    @pytest.mark.asyncio
    async def test_analyze_command_interaction(
        self, engagement_analyzer, db_session
    ):
        """Test analysis of command-based interaction."""
        command_data = {
            "user_id": "user-123",
            "telegram_id": 123456789,
            "engagement_type": EngagementType.COMMAND,
            "message_text": "/help",
            "command_name": "help",
            "response_time_seconds": 5
        }
        
        with patch('app.database.connection.get_database_session', return_value=db_session):
            engagement = await engagement_analyzer.analyze_user_interaction(**command_data)
        
        # Verify command analysis
        assert engagement.engagement_type == EngagementType.COMMAND
        assert engagement.command_name == "help"
        assert engagement.user_intent == "command_help"
        assert engagement.is_meaningful_interaction is True  # Commands are always meaningful
        assert engagement.engagement_quality_score >= 0.7  # Commands have high base score
    
    @pytest.mark.asyncio
    async def test_analyze_voice_message(
        self, engagement_analyzer, db_session
    ):
        """Test analysis of voice message interaction."""
        voice_data = {
            "user_id": "user-123",
            "telegram_id": 123456789,
            "engagement_type": EngagementType.VOICE_MESSAGE,
            "message_text": None,
            "response_time_seconds": 60
        }
        
        with patch('app.database.connection.get_database_session', return_value=db_session):
            engagement = await engagement_analyzer.analyze_user_interaction(**voice_data)
        
        # Verify voice message handling
        assert engagement.engagement_type == EngagementType.VOICE_MESSAGE
        assert engagement.is_meaningful_interaction is True  # Voice messages are always meaningful
        assert engagement.engagement_quality_score >= 0.7  # Voice messages have high engagement
        assert engagement.message_length == 0


class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_pattern_based_sentiment_positive(self, engagement_analyzer):
        """Test pattern-based positive sentiment detection."""
        text = "This is absolutely amazing! I love it so much! 😍🔥"
        
        sentiment_score, sentiment_type = await engagement_analyzer._analyze_sentiment(text)
        
        assert sentiment_score > 0
        assert sentiment_type in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]
    
    @pytest.mark.asyncio
    async def test_pattern_based_sentiment_negative(self, engagement_analyzer):
        """Test pattern-based negative sentiment detection."""
        text = "This is terrible and awful! I hate this stupid thing! 😠💔"
        
        sentiment_score, sentiment_type = await engagement_analyzer._analyze_sentiment(text)
        
        assert sentiment_score < 0
        assert sentiment_type in [SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE]
    
    @pytest.mark.asyncio
    async def test_llm_sentiment_enhancement(self, engagement_analyzer, mock_llm_service):
        """Test LLM-enhanced sentiment analysis for longer texts."""
        long_text = "I've been using this service for a while now and I have to say " \
                   "it's been quite a journey. Initially I was skeptical but over time " \
                   "it has really grown on me and become an essential part of my workflow."
        
        # Mock LLM response
        mock_llm_service.get_completion.return_value = "0.7"
        
        sentiment_score, sentiment_type = await engagement_analyzer._analyze_sentiment(long_text)
        
        # Verify LLM was called for long text
        mock_llm_service.get_completion.assert_called_once()
        
        # Verify sentiment blending occurred
        assert sentiment_score > 0
        assert sentiment_type == SentimentType.POSITIVE
    
    @pytest.mark.asyncio
    async def test_mood_indicators_detection(self, engagement_analyzer):
        """Test mood indicator detection in text."""
        test_cases = [
            {
                "text": "Wow this is amazing!!! So excited! 🔥⚡",
                "expected_mood": "excitement_level",
                "expected_value": 0.3
            },
            {
                "text": "Ugh this is so frustrating and annoying 😠💢",
                "expected_mood": "frustration_level",
                "expected_value": 0.4
            },
            {
                "text": "I'm so confused??? What does this mean? 🤔😕",
                "expected_mood": "confusion_level",
                "expected_value": 0.4
            },
            {
                "text": "Thank you so much! I really appreciate your help 🙏❤️",
                "expected_mood": "gratitude_level",
                "expected_value": 0.5
            }
        ]
        
        for case in test_cases:
            mood_indicators = await engagement_analyzer._detect_mood_indicators(case["text"])
            
            assert mood_indicators[case["expected_mood"]] >= case["expected_value"], \
                f"Failed for text: {case['text']}"


class TestBehaviorPatterns:
    """Test behavioral pattern analysis and tracking."""
    
    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction history."""
        base_time = datetime.utcnow() - timedelta(days=15)
        interactions = []
        
        for i in range(20):
            interaction = UserEngagement(
                user_id="user-123",
                telegram_id=123456789,
                engagement_type=EngagementType.MESSAGE,
                interaction_timestamp=base_time + timedelta(days=i // 2, hours=i % 24),
                message_text=f"Test message {i}",
                sentiment_score=0.5 + (i % 5 - 2) * 0.2,  # Varying sentiment
                sentiment_type=SentimentType.POSITIVE if i % 3 == 0 else SentimentType.NEUTRAL,
                message_length=50 + i * 10,
                contains_question=i % 4 == 0,
                engagement_quality_score=0.6 + (i % 3) * 0.1,
                is_meaningful_interaction=True,
                session_id=f"session-{i // 3}"
            )
            interactions.append(interaction)
        
        return interactions
    
    @pytest.mark.asyncio
    async def test_update_behavior_patterns(
        self, engagement_analyzer, db_session, sample_interactions
    ):
        """Test comprehensive behavior pattern update."""
        # Mock database session and interactions
        mock_session = AsyncMock()
        mock_execute_result = Mock()
        mock_execute_result.scalars.return_value.all.return_value = sample_interactions
        mock_session.execute.return_value = mock_execute_result
        
        # Mock existing pattern lookup
        mock_pattern_result = Mock()
        mock_pattern_result.scalar_one_or_none.return_value = None
        mock_session.execute.side_effect = [mock_execute_result, mock_pattern_result]
        
        with patch('app.database.connection.get_database_session', return_value=mock_session):
            pattern = await engagement_analyzer.update_user_behavior_patterns(
                "user-123", 123456789
            )
        
        # Verify pattern creation and data
        assert pattern.user_id == "user-123"
        assert pattern.telegram_id == 123456789
        assert pattern.total_interactions > 0
        assert pattern.daily_interaction_average > 0
        assert pattern.most_active_hour is not None
        assert pattern.engagement_quality_trend is not None
        assert pattern.churn_risk_score is not None
    
    def test_activity_pattern_analysis(self, engagement_analyzer, sample_interactions):
        """Test activity pattern analysis from interactions."""
        patterns = engagement_analyzer._analyze_activity_patterns(sample_interactions)
        
        # Verify activity analysis
        assert "most_active_hour" in patterns
        assert "most_active_day" in patterns
        assert "average_session_length" in patterns
        assert "daily_interaction_average" in patterns
        assert "total_interactions" in patterns
        
        # Verify reasonable values
        assert patterns["total_interactions"] == len(sample_interactions)
        assert 0 <= patterns["most_active_hour"] <= 23
        assert 0 <= patterns["most_active_day"] <= 6
        assert patterns["daily_interaction_average"] > 0
    
    def test_engagement_metrics_calculation(self, engagement_analyzer, sample_interactions):
        """Test engagement metrics calculation."""
        metrics = engagement_analyzer._calculate_engagement_metrics(sample_interactions)
        
        # Verify metrics
        assert "average_sentiment_score" in metrics
        assert "dominant_sentiment" in metrics
        assert "engagement_quality_trend" in metrics
        assert "sentiment_trend" in metrics
        
        # Verify sentiment analysis
        assert isinstance(metrics["average_sentiment_score"], float)
        assert isinstance(metrics["dominant_sentiment"], SentimentType)
        assert isinstance(metrics["engagement_quality_trend"], float)
    
    def test_churn_risk_calculation(self, engagement_analyzer, sample_interactions):
        """Test churn risk calculation."""
        # Test with recent interactions
        recent_interactions = sample_interactions[-5:]  # Last 5 interactions
        engagement_metrics = engagement_analyzer._calculate_engagement_metrics(recent_interactions)
        
        churn_risk = engagement_analyzer._calculate_churn_risk(recent_interactions, engagement_metrics)
        
        # Verify churn risk calculation
        assert 0.0 <= churn_risk <= 1.0
        
        # Test with old interactions (should have higher churn risk)
        old_interactions = []
        old_time = datetime.utcnow() - timedelta(days=10)
        for interaction in recent_interactions:
            interaction.interaction_timestamp = old_time
            old_interactions.append(interaction)
        
        old_churn_risk = engagement_analyzer._calculate_churn_risk(old_interactions, engagement_metrics)
        assert old_churn_risk > churn_risk  # Older interactions should have higher churn risk


class TestProactiveEngagement:
    """Test proactive engagement identification and recommendations."""
    
    @pytest.fixture
    def high_risk_user_pattern(self):
        """Create high churn risk user pattern."""
        pattern = UserBehaviorPattern(
            user_id="user-high-risk",
            telegram_id=987654321,
            churn_risk_score=0.9,
            days_since_last_interaction=7,
            shows_declining_engagement=True,
            needs_re_engagement=True,
            engagement_quality_trend=-0.3
        )
        return pattern
    
    @pytest.fixture
    def sample_user_for_engagement(self):
        """Create user for engagement testing."""
        return User(
            id="user-high-risk",
            telegram_id=987654321,
            username="highriskuser",
            first_name="High",
            last_name="Risk",
            is_active=True,
            is_blocked=False
        )
    
    @pytest.mark.asyncio
    async def test_find_users_needing_engagement(
        self, engagement_analyzer, db_session, high_risk_user_pattern, sample_user_for_engagement
    ):
        """Test identification of users needing proactive engagement."""
        # Mock database query results
        mock_session = AsyncMock()
        mock_execute_result = Mock()
        mock_execute_result.all.return_value = [(high_risk_user_pattern, sample_user_for_engagement)]
        mock_session.execute.return_value = mock_execute_result
        
        with patch('app.database.connection.get_database_session', return_value=mock_session):
            candidates = await engagement_analyzer.find_users_needing_engagement(limit=10)
        
        # Verify candidates
        assert len(candidates) == 1
        candidate = candidates[0]
        
        assert candidate["user_id"] == "user-high-risk"
        assert candidate["telegram_id"] == 987654321
        assert candidate["patterns"]["churn_risk_score"] == 0.9
        assert candidate["patterns"]["needs_re_engagement"] is True
        assert len(candidate["recommendations"]) > 0
        
        # Verify high-priority recommendation exists
        high_priority_recs = [r for r in candidate["recommendations"] if r["priority"] > 0.8]
        assert len(high_priority_recs) > 0
    
    @pytest.mark.asyncio
    async def test_generate_engagement_recommendations(self, engagement_analyzer, sample_user_for_engagement):
        """Test engagement recommendation generation."""
        # Create pattern with various issues
        pattern = UserBehaviorPattern(
            churn_risk_score=0.85,
            days_since_last_interaction=6,
            shows_declining_engagement=True,
            needs_re_engagement=True,
            topic_interests=[{"topic": "help", "relevance": 0.8}]
        )
        
        recommendations = await engagement_analyzer._generate_engagement_recommendations(
            pattern, sample_user_for_engagement
        )
        
        # Verify recommendations
        assert len(recommendations) > 0
        
        # Check for high churn risk recommendation
        high_risk_recs = [r for r in recommendations if r["reason"] == "high_churn_risk"]
        assert len(high_risk_recs) > 0
        assert high_risk_recs[0]["priority"] == 0.9
        assert high_risk_recs[0]["timing"] == "immediate"
        
        # Check for topic-based recommendation
        topic_recs = [r for r in recommendations if r["type"] == "topic_follow_up"]
        assert len(topic_recs) > 0
        assert "topics" in topic_recs[0]
    
    @pytest.mark.asyncio
    async def test_optimal_outreach_timing(self, engagement_analyzer, db_session):
        """Test optimal outreach timing calculation."""
        # Mock user pattern with optimal timing data
        pattern = UserBehaviorPattern(
            user_id="user-123",
            optimal_outreach_hour=14,  # 2 PM
            most_active_day=1  # Tuesday
        )
        
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = pattern
        mock_session.execute.return_value = mock_result
        
        with patch('app.database.connection.get_database_session', return_value=mock_session):
            optimal_time = await engagement_analyzer.get_optimal_outreach_timing("user-123")
        
        # Verify optimal timing
        assert optimal_time is not None
        assert optimal_time > datetime.utcnow()
        assert optimal_time.hour == 14
        
        # Test with no pattern (should default)
        mock_result.scalar_one_or_none.return_value = None
        with patch('app.database.connection.get_database_session', return_value=mock_session):
            default_time = await engagement_analyzer.get_optimal_outreach_timing("user-456")
        
        assert default_time is not None
        assert default_time.hour == 14  # Default afternoon time


class TestMilestoneTracking:
    """Test milestone tracking and progress analysis."""
    
    @pytest.fixture
    def sample_milestones(self):
        """Create sample milestones for testing."""
        return [
            EngagementMilestone(
                milestone_name="First 10 Messages",
                metric_name="message_count",
                target_value=10,
                reward_points=100,
                is_active=True
            ),
            EngagementMilestone(
                milestone_name="Positive Vibes",
                metric_name="positive_interactions",
                target_value=5,
                reward_points=50,
                is_active=True
            ),
            EngagementMilestone(
                milestone_name="Weekly Champion",
                metric_name="session_days",
                target_value=7,
                reward_points=200,
                is_active=True
            )
        ]
    
    def test_calculate_milestone_values(self, engagement_analyzer, sample_interactions, sample_milestones):
        """Test milestone value calculation for different metrics."""
        for milestone in sample_milestones:
            value = engagement_analyzer._calculate_milestone_value(milestone, sample_interactions)
            
            if milestone.metric_name == "message_count":
                # Count MESSAGE type interactions
                expected = len([i for i in sample_interactions if i.engagement_type == EngagementType.MESSAGE])
                assert value == expected
            
            elif milestone.metric_name == "positive_interactions":
                # Count positive sentiment interactions
                expected = len([
                    i for i in sample_interactions 
                    if i.sentiment_type in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]
                ])
                assert value == expected
            
            elif milestone.metric_name == "session_days":
                # Count unique interaction dates
                unique_dates = len(set(i.interaction_timestamp.date() for i in sample_interactions))
                assert value == unique_dates
    
    @pytest.mark.asyncio
    async def test_milestone_progress_update(self, engagement_analyzer, sample_interactions, sample_milestones):
        """Test milestone progress update logic."""
        mock_session = AsyncMock()
        
        # Mock milestone query
        milestone_result = Mock()
        milestone_result.scalars.return_value.all.return_value = sample_milestones
        
        # Mock progress query (no existing progress)
        progress_result = Mock()
        progress_result.scalar_one_or_none.return_value = None
        
        mock_session.execute.side_effect = [milestone_result, progress_result] * len(sample_milestones)
        
        await engagement_analyzer._update_milestone_progress(
            mock_session, "user-123", 123456789, sample_interactions
        )
        
        # Verify session.add was called for new progress records
        assert mock_session.add.call_count == len(sample_milestones)


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects of engagement analysis."""
    
    @pytest.mark.asyncio
    async def test_concurrent_interaction_analysis(self, engagement_analyzer, db_session):
        """Test concurrent processing of multiple interactions."""
        # Create multiple interaction data sets
        interaction_tasks = []
        
        for i in range(50):  # 50 concurrent analyses
            interaction_data = {
                "user_id": f"user-{i}",
                "telegram_id": 100000 + i,
                "engagement_type": EngagementType.MESSAGE,
                "message_text": f"Test message {i} with sentiment analysis",
                "response_time_seconds": 30 + i % 60
            }
            
            with patch('app.database.connection.get_database_session', return_value=db_session):
                task = engagement_analyzer.analyze_user_interaction(**interaction_data)
                interaction_tasks.append(task)
        
        # Execute all analyses concurrently
        import time
        start_time = time.time()
        results = await asyncio.gather(*interaction_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify results
        successful_analyses = [r for r in results if isinstance(r, UserEngagement)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        # Most analyses should succeed
        assert len(successful_analyses) >= 45
        assert len(errors) <= 5
        
        # Total time should be reasonable for concurrent processing
        assert total_time < 10.0  # Should complete in <10 seconds
        
        # Average time per analysis
        avg_time = total_time / len(successful_analyses) if successful_analyses else float('inf')
        assert avg_time < 0.5  # Should average <500ms per analysis
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_performance(self, engagement_analyzer):
        """Test sentiment analysis performance for different text lengths."""
        test_texts = [
            "Short text",
            "Medium length text with some emotional content that expresses positive feelings",
            "Very long text that contains multiple sentences with various emotional indicators " +
            "including positive words like amazing, fantastic, wonderful and some negative words " +
            "like terrible, awful, disappointing to create a mixed sentiment that requires " +
            "more sophisticated analysis to determine the overall emotional tone and classification",
            "💀" * 10,  # Emoji-heavy text
            "?" * 50   # Question-heavy text
        ]
        
        performance_results = []
        
        for text in test_texts:
            start_time = time.time()
            sentiment_score, sentiment_type = await engagement_analyzer._analyze_sentiment(text)
            analysis_time = time.time() - start_time
            
            performance_results.append({
                "text_length": len(text),
                "analysis_time": analysis_time,
                "sentiment_score": sentiment_score,
                "sentiment_type": sentiment_type
            })
        
        # Verify performance requirements
        for result in performance_results:
            # All sentiment analyses should complete quickly
            assert result["analysis_time"] < 1.0, \
                f"Sentiment analysis too slow for {result['text_length']} chars: {result['analysis_time']:.3f}s"
            
            # Sentiment score should be valid
            assert -1.0 <= result["sentiment_score"] <= 1.0
            assert isinstance(result["sentiment_type"], SentimentType)
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_memory_usage(self, engagement_analyzer):
        """Test memory usage during pattern analysis."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large interaction dataset
        large_interaction_set = []
        base_time = datetime.utcnow() - timedelta(days=30)
        
        for i in range(1000):  # 1000 interactions
            interaction = UserEngagement(
                user_id=f"user-{i % 10}",  # 10 different users
                telegram_id=100000 + i,
                engagement_type=EngagementType.MESSAGE,
                interaction_timestamp=base_time + timedelta(hours=i),
                message_text=f"Test interaction {i} with various content",
                sentiment_score=(i % 11 - 5) / 5.0,  # Range from -1 to 1
                engagement_quality_score=0.5 + (i % 5) * 0.1,
                is_meaningful_interaction=True
            )
            large_interaction_set.append(interaction)
        
        # Analyze patterns for memory usage
        patterns = engagement_analyzer._analyze_activity_patterns(large_interaction_set)
        metrics = engagement_analyzer._calculate_engagement_metrics(large_interaction_set)
        preferences = engagement_analyzer._analyze_interaction_preferences(large_interaction_set)
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable for processing 1000 interactions
        assert memory_growth < 50, f"Memory growth too high: {memory_growth:.1f} MB"
        
        # Verify results are valid
        assert patterns["total_interactions"] == 1000
        assert len(metrics) > 0
        assert len(preferences) > 0


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([__file__, "-v"])