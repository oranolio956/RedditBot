"""
Unit Tests for Conversation Engine

Tests for conversation management including:
- Natural message timing and typing simulation
- Conversation context management
- Message sentiment analysis
- Response quality assessment
- Conversation state tracking
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from app.models.telegram_conversation import (
    TelegramConversation, 
    ConversationMessage, 
    MessageDirection,
    ConversationType
)
from app.services.conversation_engine import ConversationEngine
from app.services.emotion_detector import EmotionDetector
from app.services.memory_palace import MemoryPalace


class TestConversationModel:
    """Test TelegramConversation model functionality"""
    
    def test_conversation_creation(self):
        """Test conversation model creation"""
        conversation = TelegramConversation(
            account_id="account123",
            community_id="community456",
            user_id=987654321,
            chat_id=-1001234567890,
            conversation_type=ConversationType.GROUP,
            is_active=True
        )
        
        assert conversation.account_id == "account123"
        assert conversation.community_id == "community456"
        assert conversation.user_id == 987654321
        assert conversation.chat_id == -1001234567890
        assert conversation.conversation_type == ConversationType.GROUP
        assert conversation.is_active is True
        assert conversation.engagement_score == 0.0
        assert conversation.message_count == 0
        assert conversation.last_interaction is None
    
    def test_conversation_context_management(self):
        """Test conversation context updates"""
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Initial context
        assert conversation.conversation_context == {}
        
        # Update context
        new_context = {
            "recent_topics": ["DeFi", "yield farming"],
            "user_interests": ["crypto", "trading"],
            "interaction_count": 5
        }
        
        conversation.conversation_context = new_context
        assert conversation.conversation_context == new_context
    
    def test_sentiment_history_management(self):
        """Test sentiment history tracking"""
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Start with empty history
        assert conversation.sentiment_history == []
        
        # Add sentiment entries
        conversation.update_sentiment_history("positive", 0.8)
        conversation.update_sentiment_history("neutral", 0.6)
        
        assert len(conversation.sentiment_history) == 2
        assert conversation.sentiment_history[0]["emotion"] == "positive"
        assert conversation.sentiment_history[0]["confidence"] == 0.8
        assert conversation.sentiment_history[1]["emotion"] == "neutral"
    
    def test_engagement_score_calculation(self):
        """Test engagement score updates"""
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890,
            engagement_score=50.0
        )
        
        # Positive engagement
        conversation.update_engagement_score(75.0)
        assert conversation.engagement_score == 75.0
        
        # Cap at 100
        conversation.update_engagement_score(120.0)
        assert conversation.engagement_score == 100.0
        
        # Floor at 0
        conversation.update_engagement_score(-10.0)
        assert conversation.engagement_score == 0.0
    
    def test_get_recent_topics(self):
        """Test recent topics extraction"""
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890,
            conversation_context={
                "recent_topics": ["DeFi", "NFTs", "yield farming"],
                "topic_timestamps": [
                    datetime.utcnow().isoformat(),
                    (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                    (datetime.utcnow() - timedelta(hours=2)).isoformat()
                ]
            }
        )
        
        topics = conversation.get_recent_topics(limit=2)
        assert len(topics) == 2
        assert topics == ["DeFi", "NFTs"]
    
    def test_conversation_activity_tracking(self):
        """Test conversation activity metrics"""
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Track message activity
        conversation.increment_message_count()
        assert conversation.message_count == 1
        
        conversation.increment_message_count()
        assert conversation.message_count == 2
        
        # Update last interaction
        now = datetime.utcnow()
        conversation.update_last_interaction(now)
        assert conversation.last_interaction == now


class TestConversationMessage:
    """Test ConversationMessage model"""
    
    def test_message_creation(self):
        """Test conversation message creation"""
        message = ConversationMessage(
            conversation_id="conv123",
            message_id=12345,
            direction=MessageDirection.INCOMING,
            content="Hello, how are you?",
            user_id=987654321,
            timestamp=datetime.utcnow()
        )
        
        assert message.conversation_id == "conv123"
        assert message.message_id == 12345
        assert message.direction == MessageDirection.INCOMING
        assert message.content == "Hello, how are you?"
        assert message.user_id == 987654321
    
    def test_message_metadata_handling(self):
        """Test message metadata storage"""
        metadata = {
            "emotions": {"positive": 0.8, "neutral": 0.2},
            "entities": ["greeting"],
            "response_time": 5.2
        }
        
        message = ConversationMessage(
            conversation_id="conv123",
            message_id=12345,
            direction=MessageDirection.OUTGOING,
            content="Hi! I'm doing well, thanks for asking!",
            metadata=metadata
        )
        
        assert message.metadata == metadata
        assert message.metadata["emotions"]["positive"] == 0.8
        assert message.metadata["response_time"] == 5.2
    
    def test_message_direction_types(self):
        """Test different message directions"""
        incoming = ConversationMessage(
            conversation_id="conv123",
            message_id=1,
            direction=MessageDirection.INCOMING,
            content="User message"
        )
        
        outgoing = ConversationMessage(
            conversation_id="conv123",
            message_id=2,
            direction=MessageDirection.OUTGOING,
            content="Bot response"
        )
        
        assert incoming.direction == MessageDirection.INCOMING
        assert outgoing.direction == MessageDirection.OUTGOING
        assert incoming.direction != outgoing.direction


class TestConversationEngine:
    """Test conversation engine functionality"""
    
    @pytest.fixture
    def mock_emotion_detector(self):
        """Mock emotion detector"""
        detector = Mock(spec=EmotionDetector)
        detector.analyze_text = AsyncMock(return_value={
            "primary_emotion": "positive",
            "confidence": 0.85,
            "emotions": {"joy": 0.6, "excitement": 0.25, "neutral": 0.15}
        })
        return detector
    
    @pytest.fixture
    def mock_memory_palace(self):
        """Mock memory palace"""
        memory = Mock(spec=MemoryPalace)
        memory.retrieve_conversation_context = AsyncMock(return_value={
            "previous_topics": ["crypto", "DeFi"],
            "user_preferences": {"technical_level": "intermediate"},
            "conversation_summary": "Discussion about DeFi protocols"
        })
        return memory
    
    @pytest.fixture
    def conversation_engine(self, mock_emotion_detector, mock_memory_palace):
        """Create conversation engine with mocks"""
        return ConversationEngine(
            emotion_detector=mock_emotion_detector,
            memory_palace=mock_memory_palace
        )
    
    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation for testing"""
        return TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890,
            conversation_type=ConversationType.PRIVATE,
            is_active=True,
            engagement_score=60.0,
            conversation_context={
                "recent_topics": ["yield farming", "liquidity pools"],
                "user_interests": ["DeFi", "crypto"],
                "formality_level": "casual"
            }
        )
    
    @pytest.mark.asyncio
    async def test_message_analysis(self, conversation_engine, sample_conversation):
        """Test comprehensive message analysis"""
        message_text = "What's the best yield farming strategy right now?"
        
        analysis = await conversation_engine.analyze_message(
            message_text, 
            sample_conversation
        )
        
        # Verify analysis components
        assert "emotion" in analysis
        assert "context" in analysis
        assert "intent" in analysis
        assert "complexity" in analysis
        
        # Verify emotion analysis was called
        conversation_engine.emotion_detector.analyze_text.assert_called_once_with(message_text)
        
        # Verify memory context retrieval
        conversation_engine.memory_palace.retrieve_conversation_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_timing_calculation(self, conversation_engine):
        """Test natural response timing calculation"""
        # Short, simple message
        short_analysis = {
            "complexity": "low",
            "emotion": {"intensity": 0.3},
            "requires_research": False
        }
        
        short_timing = await conversation_engine.calculate_response_timing(short_analysis)
        
        # Complex, emotional message
        complex_analysis = {
            "complexity": "high",
            "emotion": {"intensity": 0.9},
            "requires_research": True
        }
        
        complex_timing = await conversation_engine.calculate_response_timing(complex_analysis)
        
        # Complex messages should take longer
        assert complex_timing["thinking_time"] > short_timing["thinking_time"]
        assert complex_timing["typing_time"] > short_timing["typing_time"]
        
        # All times should be reasonable
        assert 1 <= short_timing["total_time"] <= 60
        assert 5 <= complex_timing["total_time"] <= 300
    
    @pytest.mark.asyncio
    async def test_typing_simulation(self, conversation_engine):
        """Test realistic typing simulation"""
        responses = [
            "Yes",
            "That's a great question about DeFi!",
            "The best yield farming strategies depend on your risk tolerance and capital allocation preferences..."
        ]
        
        for response in responses:
            typing_time = await conversation_engine.simulate_typing_time(response)
            
            # Longer responses should take more time
            expected_base_time = len(response) / 15  # ~15 chars per second
            
            # Should be within reasonable variance (±50%)
            assert expected_base_time * 0.5 <= typing_time <= expected_base_time * 2.0
            
            # Minimum and maximum bounds
            assert 0.5 <= typing_time <= 30.0
    
    @pytest.mark.asyncio
    async def test_conversation_context_building(self, conversation_engine, sample_conversation):
        """Test conversation context building"""
        recent_messages = [
            {"content": "I'm interested in DeFi", "timestamp": datetime.utcnow() - timedelta(minutes=5)},
            {"content": "What about yield farming?", "timestamp": datetime.utcnow() - timedelta(minutes=3)},
            {"content": "How risky is it?", "timestamp": datetime.utcnow() - timedelta(minutes=1)}
        ]
        
        context = await conversation_engine.build_conversation_context(
            sample_conversation, 
            recent_messages
        )
        
        # Verify context structure
        assert "conversation_flow" in context
        assert "topic_evolution" in context
        assert "user_sentiment_trend" in context
        assert "engagement_level" in context
        
        # Verify topic extraction
        assert "DeFi" in str(context["topic_evolution"])
        assert "yield farming" in str(context["topic_evolution"])
    
    @pytest.mark.asyncio
    async def test_engagement_assessment(self, conversation_engine, sample_conversation):
        """Test engagement level assessment"""
        # High engagement scenario
        high_engagement_data = {
            "response_rate": 0.9,
            "message_length_avg": 50,
            "question_frequency": 0.4,
            "emotion_intensity": 0.8,
            "topic_consistency": 0.7
        }
        
        high_score = await conversation_engine.assess_engagement_level(
            sample_conversation, 
            high_engagement_data
        )
        
        # Low engagement scenario
        low_engagement_data = {
            "response_rate": 0.2,
            "message_length_avg": 10,
            "question_frequency": 0.1,
            "emotion_intensity": 0.3,
            "topic_consistency": 0.3
        }
        
        low_score = await conversation_engine.assess_engagement_level(
            sample_conversation,
            low_engagement_data
        )
        
        # High engagement should score higher
        assert high_score > low_score
        assert 0.0 <= low_score <= 100.0
        assert 0.0 <= high_score <= 100.0
    
    @pytest.mark.asyncio
    async def test_conversation_state_tracking(self, conversation_engine, sample_conversation):
        """Test conversation state tracking"""
        # Start conversation tracking
        await conversation_engine.start_tracking(sample_conversation)
        
        # Simulate message exchange
        user_message = "Hi there! I'm new to DeFi"
        await conversation_engine.process_user_message(user_message, sample_conversation)
        
        bot_response = "Welcome! DeFi is an exciting space. What interests you most?"
        await conversation_engine.record_bot_response(bot_response, sample_conversation)
        
        # Get conversation state
        state = await conversation_engine.get_conversation_state(sample_conversation)
        
        # Verify state tracking
        assert "message_count" in state
        assert "last_activity" in state
        assert "current_topics" in state
        assert "engagement_trend" in state
        
        assert state["message_count"] >= 2  # User message + bot response
    
    @pytest.mark.asyncio
    async def test_context_memory_integration(self, conversation_engine, sample_conversation):
        """Test integration with memory palace for context"""
        message_text = "Remember when we talked about impermanent loss?"
        
        # Mock memory retrieval with relevant context
        conversation_engine.memory_palace.retrieve_conversation_context.return_value = {
            "previous_discussion": "Impermanent loss in AMM pools",
            "key_points": ["risk assessment", "mitigation strategies"],
            "user_understanding_level": "intermediate"
        }
        
        analysis = await conversation_engine.analyze_message(message_text, sample_conversation)
        
        # Should incorporate memory context
        assert "context" in analysis
        assert "previous_discussion" in str(analysis["context"])
    
    @pytest.mark.asyncio
    async def test_emotional_continuity(self, conversation_engine, sample_conversation):
        """Test emotional continuity in conversations"""
        # Set up conversation with emotional history
        sample_conversation.sentiment_history = [
            {"emotion": "excited", "confidence": 0.8, "timestamp": datetime.utcnow() - timedelta(minutes=10)},
            {"emotion": "curious", "confidence": 0.7, "timestamp": datetime.utcnow() - timedelta(minutes=5)},
            {"emotion": "positive", "confidence": 0.9, "timestamp": datetime.utcnow() - timedelta(minutes=2)}
        ]
        
        current_message = "This is getting really interesting!"
        
        analysis = await conversation_engine.analyze_message(current_message, sample_conversation)
        
        # Should consider emotional trajectory
        assert "emotional_continuity" in analysis
        assert analysis["emotional_continuity"]["trend"] in ["increasing", "stable", "decreasing"]
    
    @pytest.mark.asyncio
    async def test_conversation_quality_metrics(self, conversation_engine, sample_conversation):
        """Test conversation quality assessment"""
        # Simulate conversation with various metrics
        conversation_data = {
            "total_messages": 20,
            "user_messages": 12,
            "bot_messages": 8,
            "average_response_time": 15.5,
            "topic_consistency": 0.8,
            "emotional_engagement": 0.7,
            "information_value": 0.9
        }
        
        quality_score = await conversation_engine.calculate_conversation_quality(
            sample_conversation,
            conversation_data
        )
        
        # Verify quality score
        assert isinstance(quality_score, dict)
        assert "overall_score" in quality_score
        assert "dimensions" in quality_score
        
        # Score should be reasonable
        assert 0.0 <= quality_score["overall_score"] <= 100.0
        
        # Should have multiple quality dimensions
        dimensions = quality_score["dimensions"]
        assert "engagement" in dimensions
        assert "relevance" in dimensions
        assert "naturalness" in dimensions


class TestAdvancedConversationFeatures:
    """Test advanced conversation features"""
    
    @pytest.mark.asyncio
    async def test_multi_turn_context_management(self):
        """Test context management across multiple turns"""
        engine = ConversationEngine()
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Turn 1
        await engine.process_turn(
            "I want to learn about DeFi",
            "Great! DeFi offers many opportunities. What specific area interests you?",
            conversation
        )
        
        # Turn 2 
        await engine.process_turn(
            "Yield farming sounds interesting",
            "Yield farming can be profitable but has risks like impermanent loss. Want to know more?",
            conversation
        )
        
        # Turn 3
        await engine.process_turn(
            "What's impermanent loss?",
            "Impermanent loss occurs when the price ratio of your deposited tokens changes...",
            conversation
        )
        
        # Context should build across turns
        context = conversation.conversation_context
        assert "DeFi" in str(context)
        assert "yield farming" in str(context)
        assert "impermanent loss" in str(context)
    
    @pytest.mark.parametrize("message_type,expected_timing", [
        ("greeting", (1, 5)),      # Quick greeting response
        ("question", (3, 15)),     # Questions need thought
        ("complex_tech", (10, 30)), # Technical topics take time
        ("emotional", (2, 8)),     # Emotional responses are quicker
    ])
    @pytest.mark.asyncio
    async def test_message_type_timing(self, message_type, expected_timing):
        """Test timing based on message type"""
        engine = ConversationEngine()
        
        message_examples = {
            "greeting": "Hello there!",
            "question": "How does automated market making work?", 
            "complex_tech": "Can you explain the mathematical model behind constant product AMMs and how they handle price discovery?",
            "emotional": "I'm so excited about this project!"
        }
        
        analysis = await engine.analyze_message_type(message_examples[message_type])
        timing = await engine.calculate_response_timing(analysis)
        
        min_time, max_time = expected_timing
        assert min_time <= timing["total_time"] <= max_time
    
    @pytest.mark.asyncio
    async def test_conversation_recovery_from_errors(self):
        """Test conversation recovery from errors"""
        engine = ConversationEngine()
        conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Simulate error in processing
        with patch.object(engine, 'analyze_message', side_effect=Exception("Analysis failed")):
            # Should handle gracefully
            result = await engine.safe_process_message("Test message", conversation)
            
            # Should return fallback response
            assert result is not None
            assert "error" not in result.lower()  # Should not expose error to user
    
    @pytest.mark.asyncio
    async def test_conversation_warmth_adaptation(self):
        """Test conversation warmth adaptation based on relationship"""
        engine = ConversationEngine()
        
        # New conversation (cold)
        new_conversation = TelegramConversation(
            account_id="account123",
            user_id=987654321,
            chat_id=-1001234567890,
            message_count=1,
            engagement_score=0.0
        )
        
        # Established conversation (warm)
        established_conversation = TelegramConversation(
            account_id="account123", 
            user_id=987654321,
            chat_id=-1001234567890,
            message_count=50,
            engagement_score=85.0,
            created_at=datetime.utcnow() - timedelta(weeks=2)
        )
        
        new_warmth = await engine.calculate_conversation_warmth(new_conversation)
        established_warmth = await engine.calculate_conversation_warmth(established_conversation)
        
        # Established conversation should be warmer
        assert established_warmth > new_warmth
        assert 0.0 <= new_warmth <= 1.0
        assert 0.0 <= established_warmth <= 1.0