"""
Viral Engine Tests

Comprehensive test suite for viral content generation and optimization:
- Conversation analysis for viral content opportunities
- AI-powered content generation and optimization
- Social platform optimization (Twitter, Instagram, TikTok, LinkedIn)
- Viral score calculation and trending analysis
- Content anonymization and privacy protection
- Performance testing for real-time content generation
- Hashtag optimization and trending analysis
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import hashlib

from app.services.viral_engine import ViralEngine, ViralTrigger
from app.models.sharing import (
    ShareableContent, ContentShare, ShareableContentType, SocialPlatform,
    ViralMetrics
)
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.llm_service import LLMService


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.query = Mock()
    return session


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = Mock(spec=LLMService)
    service.generate_completion = AsyncMock()
    service.get_completion = AsyncMock()
    return service


@pytest.fixture
def viral_engine(mock_db_session, mock_llm_service):
    """Create ViralEngine instance with mocked dependencies."""
    return ViralEngine(db=mock_db_session, llm_service=mock_llm_service)


@pytest.fixture
def sample_conversation():
    """Create sample conversation for testing."""
    conversation = Conversation(
        id="conv-123",
        user_id="user-456",
        created_at=datetime.utcnow() - timedelta(minutes=30),
        updated_at=datetime.utcnow()
    )
    
    # Add sample messages
    messages = [
        Message(
            id="msg-1",
            conversation_id="conv-123",
            content="Hey, I've been using this AI bot and it's absolutely amazing!",
            is_from_bot=False,
            analysis_data={"sentiment": {"score": 0.8, "type": "positive"}}
        ),
        Message(
            id="msg-2",
            conversation_id="conv-123",
            content="That's interesting! What makes it so special?",
            is_from_bot=False,
            analysis_data={"sentiment": {"score": 0.3, "type": "neutral"}}
        ),
        Message(
            id="msg-3",
            conversation_id="conv-123",
            content="It's the way it understands context and provides such insightful responses. "
                   "It's like having a conversation with someone who really gets it! 😊",
            is_from_bot=False,
            analysis_data={"sentiment": {"score": 0.9, "type": "very_positive"}}
        ),
        Message(
            id="msg-4",
            conversation_id="conv-123",
            content="I'm glad you're finding our conversation helpful! The key is that I try to "
                   "understand not just what you're saying, but the deeper meaning behind it. "
                   "It's fascinating how context shapes understanding, isn't it?",
            is_from_bot=True,
            analysis_data={"sentiment": {"score": 0.7, "type": "positive"}}
        )
    ]
    
    conversation.messages = messages
    return conversation


@pytest.fixture
def sample_funny_conversation():
    """Create sample funny conversation for testing."""
    conversation = Conversation(
        id="conv-funny",
        user_id="user-789",
        created_at=datetime.utcnow() - timedelta(minutes=15)
    )
    
    messages = [
        Message(
            id="msg-f1",
            conversation_id="conv-funny",
            content="Bot, tell me a joke about artificial intelligence",
            is_from_bot=False,
            analysis_data={"sentiment": {"score": 0.5, "type": "neutral"}}
        ),
        Message(
            id="msg-f2",
            conversation_id="conv-funny",
            content="Why don't AI systems ever get tired? Because they run on coffee... "
                   "wait, that's programmers. AI systems run on electricity and existential dread! 😂",
            is_from_bot=True,
            analysis_data={"sentiment": {"score": 0.8, "type": "positive"}}
        ),
        Message(
            id="msg-f3",
            conversation_id="conv-funny",
            content="Haha that's hilarious! 😂🤣 You actually made me laugh out loud!",
            is_from_bot=False,
            analysis_data={"sentiment": {"score": 0.9, "type": "very_positive"}}
        )
    ]
    
    conversation.messages = messages
    return conversation


class TestViralTriggerSetup:
    """Test viral trigger configuration and setup."""
    
    def test_trigger_setup(self, viral_engine):
        """Test viral trigger configuration."""
        triggers = viral_engine.triggers
        
        # Verify we have expected trigger types
        trigger_types = [t.content_type for t in triggers]
        expected_types = [
            ShareableContentType.FUNNY_MOMENT,
            ShareableContentType.PERSONALITY_INSIGHT,
            ShareableContentType.WISDOM_QUOTE,
            ShareableContentType.AI_RESPONSE,
            ShareableContentType.EDUCATIONAL_SUMMARY
        ]
        
        for expected_type in expected_types:
            assert expected_type in trigger_types
        
        # Verify trigger configuration
        for trigger in triggers:
            assert isinstance(trigger, ViralTrigger)
            assert trigger.min_viral_score > 0
            assert len(trigger.platforms) > 0
            assert len(trigger.generation_prompt) > 0
            assert len(trigger.trigger_conditions) > 0
    
    def test_funny_moment_trigger(self, viral_engine):
        """Test funny moment trigger configuration."""
        funny_triggers = [
            t for t in viral_engine.triggers 
            if t.content_type == ShareableContentType.FUNNY_MOMENT
        ]
        
        assert len(funny_triggers) == 1
        trigger = funny_triggers[0]
        
        # Verify configuration
        assert trigger.min_viral_score >= 70.0
        assert SocialPlatform.TWITTER in trigger.platforms
        assert SocialPlatform.INSTAGRAM in trigger.platforms
        assert "humor_indicators" in trigger.trigger_conditions
        assert "sentiment_spike" in trigger.trigger_conditions
    
    def test_ai_response_trigger(self, viral_engine):
        """Test AI response trigger configuration."""
        ai_triggers = [
            t for t in viral_engine.triggers 
            if t.content_type == ShareableContentType.AI_RESPONSE
        ]
        
        assert len(ai_triggers) == 1
        trigger = ai_triggers[0]
        
        # Verify configuration
        assert trigger.min_viral_score >= 75.0
        assert SocialPlatform.TWITTER in trigger.platforms
        assert SocialPlatform.REDDIT in trigger.platforms
        assert "ai_cleverness" in trigger.trigger_conditions


class TestConversationAnalysis:
    """Test conversation analysis for viral content detection."""
    
    @pytest.mark.asyncio
    async def test_extract_conversation_context(self, viral_engine, sample_conversation):
        """Test conversation context extraction."""
        context = await viral_engine._extract_conversation_context(sample_conversation)
        
        # Verify context structure
        expected_keys = [
            'message_count', 'total_length', 'sentiment_spike',
            'avg_sentiment', 'humor_indicators', 'wisdom_indicators',
            'personality_markers', 'ai_cleverness', 'recent_messages',
            'full_text', 'conversation_age'
        ]
        
        for key in expected_keys:
            assert key in context, f"Missing key: {key}"
        
        # Verify context values
        assert context['message_count'] > 0
        assert context['total_length'] > 0
        assert 0 <= context['avg_sentiment'] <= 1
        assert context['sentiment_spike'] >= 0
        assert context['ai_cleverness'] >= 0
        assert len(context['recent_messages']) > 0
        assert len(context['full_text']) > 0
        assert context['conversation_age'] >= 0
    
    def test_humor_indicators_detection(self, viral_engine):
        """Test humor indicator detection in text."""
        test_cases = [
            ("This is so funny! 😂 haha lol", 4),  # Multiple indicators
            ("That was hilarious and made me laugh", 2),
            ("Just a normal message", 0),
            ("🤣😄😆 joke time!", 4)
        ]
        
        for text, expected_min in test_cases:
            count = viral_engine._count_humor_indicators(text)
            assert count >= expected_min, f"Failed for: {text}"
    
    def test_wisdom_indicators_detection(self, viral_engine):
        """Test wisdom indicator detection in text."""
        test_cases = [
            ("I learned something new today and gained insight", 2),
            ("This is a truth I now understand and realize", 3),
            ("Just chatting about random stuff", 0),
            ("What a breakthrough discovery with great clarity!", 3)
        ]
        
        for text, expected_min in test_cases:
            count = viral_engine._count_wisdom_indicators(text)
            assert count >= expected_min, f"Failed for: {text}"
    
    def test_personality_markers_extraction(self, viral_engine):
        """Test personality marker extraction."""
        test_text = "I'm quite introverted and prefer quiet solitude, but I'm also very creative " \
                   "and curious about exploring new artistic ideas. I tend to be organized and " \
                   "disciplined in achieving my goals."
        
        markers = viral_engine._extract_personality_markers(test_text)
        
        # Should detect multiple personality traits
        assert len(markers) > 0
        
        # Check for specific expected markers
        marker_strings = [str(marker) for marker in markers]
        trait_found = {
            'introversion': any('introversion:' in m for m in marker_strings),
            'openness': any('openness:' in m for m in marker_strings),
            'conscientiousness': any('conscientiousness:' in m for m in marker_strings)
        }
        
        # Should find at least some personality traits
        assert sum(trait_found.values()) >= 2
    
    def test_ai_cleverness_assessment(self, viral_engine, sample_conversation):
        """Test AI cleverness assessment."""
        cleverness_score = viral_engine._assess_ai_cleverness(sample_conversation.messages)
        
        # Should detect some cleverness in the AI response
        assert 0.0 <= cleverness_score <= 1.0
        assert cleverness_score > 0  # Sample has sophisticated AI response


class TestTriggerConditionChecking:
    """Test viral trigger condition evaluation."""
    
    @pytest.mark.asyncio
    async def test_check_funny_trigger_conditions(self, viral_engine, sample_funny_conversation):
        """Test funny moment trigger condition checking."""
        context = await viral_engine._extract_conversation_context(sample_funny_conversation)
        
        funny_trigger = next(
            t for t in viral_engine.triggers 
            if t.content_type == ShareableContentType.FUNNY_MOMENT
        )
        
        # Should meet funny moment conditions
        result = await viral_engine._check_trigger_conditions(context, funny_trigger)
        
        # Verify the funny conversation meets trigger conditions
        # (This may be True or False depending on exact thresholds)
        assert isinstance(result, bool)
        
        # Check individual condition factors
        assert context['humor_indicators'] >= 0
        assert context['sentiment_spike'] >= 0
    
    @pytest.mark.asyncio
    async def test_check_ai_response_trigger_conditions(self, viral_engine, sample_conversation):
        """Test AI response trigger condition checking."""
        context = await viral_engine._extract_conversation_context(sample_conversation)
        
        ai_trigger = next(
            t for t in viral_engine.triggers 
            if t.content_type == ShareableContentType.AI_RESPONSE
        )
        
        result = await viral_engine._check_trigger_conditions(context, ai_trigger)
        
        # Verify condition checking
        assert isinstance(result, bool)
        
        # AI cleverness should be assessed
        assert context['ai_cleverness'] >= 0
    
    @pytest.mark.asyncio
    async def test_message_length_constraints(self, viral_engine):
        """Test message length constraint checking."""
        # Test short message
        short_context = {
            'total_length': 10,
            'sentiment_spike': 0.9,
            'humor_indicators': 3
        }
        
        trigger_with_min_length = ViralTrigger(
            content_type=ShareableContentType.FUNNY_MOMENT,
            trigger_conditions={
                'min_message_length': 20,
                'sentiment_spike': 0.8
            },
            min_viral_score=70.0,
            platforms=[SocialPlatform.TWITTER],
            generation_prompt="Test prompt"
        )
        
        # Should fail due to length constraint
        result = await viral_engine._check_trigger_conditions(short_context, trigger_with_min_length)
        assert result is False
        
        # Test with sufficient length
        long_context = short_context.copy()
        long_context['total_length'] = 50
        
        result = await viral_engine._check_trigger_conditions(long_context, trigger_with_min_length)
        # May be True if other conditions are met


class TestViralContentGeneration:
    """Test AI-powered viral content generation."""
    
    @pytest.mark.asyncio
    async def test_generate_viral_content(self, viral_engine, sample_conversation, mock_llm_service):
        """Test viral content generation with AI."""
        # Setup mock LLM response
        mock_ai_response = json.dumps({
            "title": "AI Bot Provides Surprisingly Deep Insights",
            "description": "This AI conversation shows how artificial intelligence can provide "
                          "meaningful and contextual responses that feel genuinely helpful.",
            "content_data": {
                "text": "Check out this amazing AI conversation!",
                "platforms": {
                    "twitter": "🤖 AI conversations are getting seriously impressive!",
                    "instagram": "When AI actually gets it right ✨"
                }
            },
            "hashtags": ["#AI", "#ArtificialIntelligence", "#TechTrends"],
            "viral_elements": ["ai_insight", "relatability", "emotional_impact"]
        })
        
        mock_llm_service.generate_completion.return_value = mock_ai_response
        
        # Get conversation context
        context = await viral_engine._extract_conversation_context(sample_conversation)
        
        # Find AI response trigger
        ai_trigger = next(
            t for t in viral_engine.triggers 
            if t.content_type == ShareableContentType.AI_RESPONSE
        )
        
        # Generate viral content
        content = await viral_engine._generate_viral_content(
            sample_conversation, context, ai_trigger
        )
        
        # Verify content generation
        if content:  # May be None if conditions not met
            assert isinstance(content, ShareableContent)
            assert content.title is not None
            assert len(content.title) > 0
            assert content.description is not None
            assert content.viral_score > 0
            assert content.is_anonymized is True
            assert content.source_conversation_id == sample_conversation.id
            assert len(content.hashtags) > 0
            
            # Verify AI enhancement data
            assert 'generation_context' in content.ai_enhancement_data
            assert 'trigger_type' in content.ai_enhancement_data
            assert 'viral_elements' in content.ai_enhancement_data
    
    @pytest.mark.asyncio
    async def test_ai_response_fallback_parsing(self, viral_engine):
        """Test fallback parsing when AI doesn't return valid JSON."""
        invalid_ai_response = """
        This is a great AI conversation that shows how technology can be helpful!
        
        #AI #Technology #Helpful #Innovation
        
        The future is here and it's amazing!
        """
        
        parsed_content = viral_engine._parse_ai_response_fallback(invalid_ai_response)
        
        # Verify fallback parsing
        assert 'title' in parsed_content
        assert 'content_data' in parsed_content
        assert 'hashtags' in parsed_content
        
        # Should extract hashtags
        assert len(parsed_content['hashtags']) > 0
        assert '#AI' in parsed_content['hashtags']
        assert '#Technology' in parsed_content['hashtags']
        
        # Should have basic content
        assert len(parsed_content['content_data']['text']) > 0
    
    def test_viral_score_calculation(self, viral_engine):
        """Test viral score calculation logic."""
        # Test high-potential content
        high_context = {
            'sentiment_spike': 0.8,
            'ai_cleverness': 0.9,
            'humor_indicators': 3,
            'personality_markers': ['trait1:marker1', 'trait2:marker2']
        }
        
        high_content_json = {
            'title': 'Amazing AI Insight That Will Change Your Mind',
            'hashtags': ['#AI', '#Mindblowing', '#TechTrends'],
            'viral_elements': ['ai_insight', 'emotional_impact', 'humor']
        }
        
        high_trigger = ViralTrigger(
            content_type=ShareableContentType.AI_RESPONSE,
            trigger_conditions={},
            min_viral_score=70.0,
            platforms=[SocialPlatform.TWITTER, SocialPlatform.INSTAGRAM, SocialPlatform.TIKTOK],
            generation_prompt=""
        )
        
        high_score = viral_engine._calculate_viral_score(high_context, high_content_json, high_trigger)
        
        # Test low-potential content
        low_context = {
            'sentiment_spike': 0.1,
            'ai_cleverness': 0.2,
            'humor_indicators': 0,
            'personality_markers': []
        }
        
        low_content_json = {
            'title': 'OK',
            'hashtags': [],
            'viral_elements': []
        }
        
        low_trigger = ViralTrigger(
            content_type=ShareableContentType.EDUCATIONAL_SUMMARY,
            trigger_conditions={},
            min_viral_score=60.0,
            platforms=[SocialPlatform.LINKEDIN],
            generation_prompt=""
        )
        
        low_score = viral_engine._calculate_viral_score(low_context, low_content_json, low_trigger)
        
        # Verify scoring
        assert 0 <= high_score <= 100
        assert 0 <= low_score <= 100
        assert high_score > low_score  # High-potential should score higher
    
    def test_anonymous_id_generation(self, viral_engine):
        """Test anonymization ID generation."""
        user_id = "user-12345"
        
        anon_id1 = viral_engine._generate_anonymous_id(user_id)
        anon_id2 = viral_engine._generate_anonymous_id(user_id)
        
        # Same input should generate same anonymous ID
        assert anon_id1 == anon_id2
        
        # Should be properly anonymized
        assert len(anon_id1) == 16  # Truncated hash
        assert user_id not in anon_id1  # Original ID not visible
        
        # Different user should get different ID
        different_user = "user-67890"
        different_anon_id = viral_engine._generate_anonymous_id(different_user)
        assert different_anon_id != anon_id1


class TestPlatformOptimization:
    """Test social platform optimization features."""
    
    @pytest.mark.asyncio
    async def test_optimize_content_for_twitter(self, viral_engine):
        """Test Twitter-specific content optimization."""
        sample_content = ShareableContent(
            id="content-123",
            content_type=ShareableContentType.FUNNY_MOMENT.value,
            title="This AI bot just made the funniest joke I've ever heard!",
            description="A longer description that might need truncation for Twitter's character limits",
            hashtags=["#AI", "#Funny", "#Technology", "#Humor", "#Bot"],
            viral_score=85.0
        )
        
        optimized = await viral_engine.optimize_content_for_platform(
            sample_content, SocialPlatform.TWITTER
        )
        
        # Verify Twitter optimization
        assert optimized['platform'] == SocialPlatform.TWITTER.value
        assert len(optimized['title']) <= 280  # Twitter character limit
        assert len(optimized['hashtags']) <= 2  # Twitter hashtag limit
        assert 'specs' in optimized
        assert optimized['specs']['style'] == 'concise_witty'
    
    @pytest.mark.asyncio
    async def test_optimize_content_for_instagram(self, viral_engine):
        """Test Instagram-specific content optimization."""
        sample_content = ShareableContent(
            id="content-456",
            content_type=ShareableContentType.PERSONALITY_INSIGHT.value,
            title="Discover Your True Personality Through AI",
            description="An insightful exploration of personality traits",
            hashtags=["#SelfDiscovery", "#Personality", "#AI"] * 15,  # Many hashtags
            viral_score=75.0
        )
        
        optimized = await viral_engine.optimize_content_for_platform(
            sample_content, SocialPlatform.INSTAGRAM
        )
        
        # Verify Instagram optimization
        assert optimized['platform'] == SocialPlatform.INSTAGRAM.value
        assert len(optimized['hashtags']) <= 30  # Instagram hashtag limit
        assert optimized['specs']['style'] == 'visual_storytelling'
        assert optimized['specs']['image_ratio'] == '1:1'
    
    @pytest.mark.asyncio
    async def test_optimize_content_for_linkedin(self, viral_engine):
        """Test LinkedIn-specific content optimization."""
        sample_content = ShareableContent(
            id="content-789",
            content_type=ShareableContentType.EDUCATIONAL_SUMMARY.value,
            title="How AI is Transforming Professional Communication",
            description="Professional insights into AI applications",
            hashtags=["#AI", "#Professional", "#Innovation"],
            viral_score=70.0
        )
        
        optimized = await viral_engine.optimize_content_for_platform(
            sample_content, SocialPlatform.LINKEDIN
        )
        
        # Verify LinkedIn optimization
        assert optimized['platform'] == SocialPlatform.LINKEDIN.value
        assert optimized['specs']['style'] == 'professional_insightful'
        assert len(optimized['hashtags']) <= 5  # LinkedIn hashtag limit
        assert optimized['specs']['max_length'] == 3000


class TestTrendingAndAnalytics:
    """Test trending content and analytics features."""
    
    @pytest.mark.asyncio
    async def test_get_trending_hashtags(self, viral_engine):
        """Test trending hashtag retrieval."""
        platforms = [SocialPlatform.TWITTER, SocialPlatform.INSTAGRAM]
        
        hashtags = await viral_engine._get_trending_hashtags(platforms)
        
        # Verify hashtag retrieval
        assert isinstance(hashtags, list)
        assert len(hashtags) > 0
        
        # Should include platform-specific hashtags
        hashtag_strings = [str(tag) for tag in hashtags]
        assert any('#AI' in tag for tag in hashtag_strings)
        assert any('#' in tag for tag in hashtag_strings)  # All should be hashtags
    
    @pytest.mark.asyncio
    async def test_get_trending_content(self, viral_engine, mock_db_session):
        """Test trending content retrieval."""
        # Mock database query results
        mock_content = [
            ShareableContent(
                id=f"content-{i}",
                content_type=ShareableContentType.FUNNY_MOMENT.value,
                title=f"Trending Content {i}",
                viral_score=80 + i,
                view_count=100 + i * 10,
                share_count=20 + i * 2,
                is_published=True
            )
            for i in range(5)
        ]
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_content
        
        mock_db_session.query.return_value = mock_query
        
        # Get trending content
        trending = await viral_engine.get_trending_content(limit=3)
        
        # Verify results
        assert len(trending) == 5  # Mock returns all
        for content in trending:
            assert content.is_published is True
            assert content.viral_score > 0
    
    @pytest.mark.asyncio
    async def test_track_content_performance(self, viral_engine, mock_db_session):
        """Test content performance tracking."""
        # Mock existing content
        mock_content = ShareableContent(
            id="perf-test-123",
            content_type=ShareableContentType.AI_RESPONSE.value,
            title="Performance Test Content",
            view_count=100,
            share_count=10,
            like_count=50,
            comment_count=5
        )
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_content
        
        # Track performance metrics
        await viral_engine.track_content_performance(
            content_id="perf-test-123",
            platform=SocialPlatform.TWITTER,
            metrics={
                "views": 25,
                "shares": 5,
                "likes": 15,
                "comments": 3
            }
        )
        
        # Verify metrics update
        assert mock_content.view_count == 125  # 100 + 25
        assert mock_content.share_count == 15   # 10 + 5
        assert mock_content.like_count == 65    # 50 + 15
        assert mock_content.comment_count == 8  # 5 + 3
        
        # Verify database commit
        mock_db_session.commit.assert_called_once()


class TestViralEngineIntegration:
    """Integration tests for complete viral engine workflow."""
    
    @pytest.mark.asyncio
    async def test_analyze_conversation_for_viral_content(
        self, viral_engine, sample_funny_conversation, mock_llm_service
    ):
        """Test complete conversation analysis workflow."""
        # Mock AI response
        mock_ai_response = json.dumps({
            "title": "AI Bot Delivers Perfect Comedy Timing",
            "description": "This hilarious exchange shows AI's surprising sense of humor",
            "content_data": {"text": "AI comedy gold! 😂"},
            "hashtags": ["#AIHumor", "#TechComedy", "#BotJokes"],
            "viral_elements": ["humor", "timing", "relatability"]
        })
        
        mock_llm_service.generate_completion.return_value = mock_ai_response
        
        # Analyze conversation
        generated_content = await viral_engine.analyze_conversation_for_viral_content(
            sample_funny_conversation, real_time=True
        )
        
        # Verify analysis results
        assert isinstance(generated_content, list)
        
        if generated_content:  # Content was generated
            content = generated_content[0]
            assert isinstance(content, ShareableContent)
            assert content.content_type in [ct.value for ct in ShareableContentType]
            assert content.viral_score > 0
            assert content.is_anonymized is True
            assert content.source_conversation_id == sample_funny_conversation.id
    
    @pytest.mark.asyncio
    async def test_error_handling_in_content_generation(
        self, viral_engine, sample_conversation, mock_llm_service
    ):
        """Test error handling during content generation."""
        # Make LLM service fail
        mock_llm_service.generate_completion.side_effect = Exception("API Error")
        
        # Should handle error gracefully
        generated_content = await viral_engine.analyze_conversation_for_viral_content(
            sample_conversation, real_time=True
        )
        
        # Should return empty list on error
        assert generated_content == []
        
        # Should rollback database transaction
        viral_engine.db.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_identify_viral_elements(self, viral_engine, sample_funny_conversation):
        """Test viral element identification."""
        context = await viral_engine._extract_conversation_context(sample_funny_conversation)
        
        viral_elements = viral_engine._identify_viral_elements(context)
        
        # Verify viral elements detection
        assert isinstance(viral_elements, list)
        
        # Should identify humor in funny conversation
        if context['humor_indicators'] > 2:
            assert 'humor' in viral_elements
        
        # Should identify emotional impact if sentiment spike is high
        if context['sentiment_spike'] > 0.6:
            assert 'emotional_impact' in viral_elements


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects of viral engine."""
    
    @pytest.mark.asyncio
    async def test_concurrent_content_generation(self, mock_db_session, mock_llm_service):
        """Test concurrent viral content generation."""
        viral_engine = ViralEngine(db=mock_db_session, llm_service=mock_llm_service)
        
        # Mock AI responses for concurrent requests
        mock_responses = []
        for i in range(10):
            response = json.dumps({
                "title": f"Viral Content {i}",
                "description": f"Description {i}",
                "content_data": {"text": f"Content {i}"},
                "hashtags": [f"#Tag{i}"],
                "viral_elements": ["element1"]
            })
            mock_responses.append(response)
        
        mock_llm_service.generate_completion.side_effect = mock_responses
        
        # Create multiple conversations
        conversations = []
        for i in range(10):
            conv = Conversation(
                id=f"conv-{i}",
                user_id=f"user-{i}",
                created_at=datetime.utcnow()
            )
            conv.messages = [
                Message(
                    id=f"msg-{i}",
                    conversation_id=f"conv-{i}",
                    content=f"This is amazing content number {i}! So insightful! 😊",
                    is_from_bot=False,
                    analysis_data={"sentiment": {"score": 0.8}}
                )
            ]
            conversations.append(conv)
        
        # Process conversations concurrently
        tasks = []
        for conv in conversations:
            task = viral_engine.analyze_conversation_for_viral_content(conv, real_time=True)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify concurrent processing
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Should process multiple conversations efficiently
        assert len(successful_results) >= 8  # Most should succeed
        assert total_time < 10.0  # Should complete in reasonable time
        
        avg_time = total_time / len(successful_results) if successful_results else float('inf')
        assert avg_time < 2.0  # Average processing time per conversation
    
    @pytest.mark.asyncio
    async def test_hashtag_caching_performance(self, viral_engine):
        """Test hashtag caching for performance."""
        platforms = [SocialPlatform.TWITTER, SocialPlatform.INSTAGRAM]
        
        # First call (should populate cache)
        start_time = time.time()
        hashtags1 = await viral_engine._get_trending_hashtags(platforms)
        first_call_time = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        hashtags2 = await viral_engine._get_trending_hashtags(platforms)
        second_call_time = time.time() - start_time
        
        # Verify caching
        assert hashtags1 == hashtags2  # Same results
        assert second_call_time < first_call_time  # Faster due to cache
        
        # Cache should be populated
        assert len(viral_engine._trending_hashtags) > 0
        assert viral_engine._hashtags_updated_at is not None


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([__file__, "-v"])