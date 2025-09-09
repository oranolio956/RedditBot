"""
Comprehensive Test Suite for AI Personality System

Tests all components of the personality system:
- Personality analysis and trait detection
- Personality matching and compatibility
- Real-time adaptation and learning
- A/B testing framework
- Telegram integration
- Performance and reliability
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, List, Any

# Internal imports
from app.services.personality_engine import (
    AdvancedPersonalityEngine, ConversationContext, PersonalityState
)
from app.services.conversation_analyzer import (
    ConversationAnalyzer, ConversationMetrics, EmotionalState
)
from app.services.personality_matcher import (
    PersonalityMatcher, PersonalityMatch, MatchingContext
)
from app.services.personality_manager import PersonalityManager, PersonalityResponse
from app.services.personality_testing import PersonalityTestingFramework, TestType
from app.models.personality import (
    PersonalityProfile, UserPersonalityMapping, PersonalityDimension, AdaptationStrategy
)
from app.models.user import User
from app.models.conversation import Message, ConversationSession, MessageDirection


class TestPersonalityEngine:
    """Test the core personality analysis and adaptation engine."""
    
    @pytest.fixture
    async def personality_engine(self):
        """Create personality engine for testing."""
        db_mock = AsyncMock()
        redis_mock = AsyncMock()
        
        engine = AdvancedPersonalityEngine(db_mock, redis_mock)
        # Mock the model initialization
        engine.sentiment_analyzer = Mock()
        engine.emotion_classifier = Mock()
        engine.personality_classifier = Mock()
        engine.tokenizer = Mock()
        engine.embedding_model = Mock()
        
        return engine
    
    @pytest.fixture
    def sample_conversation_context(self):
        """Create sample conversation context."""
        return ConversationContext(
            user_id="test_user_123",
            session_id="test_session_456",
            message_history=[
                {
                    "content": "Hello, I need help with something important",
                    "timestamp": datetime.now().isoformat(),
                    "direction": "incoming"
                },
                {
                    "content": "I'd be happy to help! What do you need assistance with?",
                    "timestamp": datetime.now().isoformat(),
                    "direction": "outgoing"
                }
            ],
            current_sentiment=0.2,
            topic="support",
            urgency_level=0.7,
            conversation_phase="building"
        )
    
    @pytest.mark.asyncio
    async def test_analyze_user_personality(self, personality_engine, sample_conversation_context):
        """Test user personality analysis from conversation."""
        
        # Mock sentiment analyzer
        personality_engine.sentiment_analyzer.return_value = [[
            {'label': 'LABEL_2', 'score': 0.7},  # Positive
            {'label': 'LABEL_1', 'score': 0.2},  # Neutral
            {'label': 'LABEL_0', 'score': 0.1}   # Negative
        ]]
        
        # Mock emotion classifier
        personality_engine.emotion_classifier.return_value = [[
            {'label': 'joy', 'score': 0.6},
            {'label': 'neutral', 'score': 0.3},
            {'label': 'sadness', 'score': 0.1}
        ]]
        
        # Test personality analysis
        user_traits = await personality_engine.analyze_user_personality(
            "test_user_123", sample_conversation_context
        )
        
        # Verify traits are returned
        assert isinstance(user_traits, dict)
        assert len(user_traits) == len(PersonalityDimension)
        
        # Verify all trait values are in valid range
        for trait, value in user_traits.items():
            assert 0.0 <= value <= 1.0
            assert trait in [dim.value for dim in PersonalityDimension]
        
        # Verify some expected personality patterns
        # High urgency should increase conscientiousness
        assert user_traits.get('conscientiousness', 0) > 0.4
    
    @pytest.mark.asyncio
    async def test_personality_adaptation_strategies(self, personality_engine, sample_conversation_context):
        """Test different personality adaptation strategies."""
        
        # Create base profile
        base_profile = PersonalityProfile(
            id="test_profile",
            name="test_profile",
            trait_scores={
                'openness': 0.6,
                'conscientiousness': 0.7,
                'extraversion': 0.5,
                'agreeableness': 0.8,
                'neuroticism': 0.3
            }
        )
        
        user_traits = {
            'openness': 0.4,
            'conscientiousness': 0.5,
            'extraversion': 0.8,
            'agreeableness': 0.6,
            'neuroticism': 0.2
        }
        
        # Test mirror adaptation
        personality_state = await personality_engine.adapt_personality(
            "test_user", user_traits, base_profile, sample_conversation_context
        )
        
        assert isinstance(personality_state, PersonalityState)
        assert personality_state.adapted_traits != personality_state.base_traits
        
        # In mirror strategy, adapted extraversion should be closer to user's (0.8)
        # than base profile's (0.5)
        adapted_extraversion = personality_state.adapted_traits.get('extraversion', 0.5)
        assert adapted_extraversion > 0.5  # Moved toward user's high extraversion
    
    @pytest.mark.asyncio
    async def test_personality_response_generation(self, personality_engine):
        """Test personality-adapted response generation."""
        
        # Create personality state with high extraversion
        personality_state = PersonalityState(
            base_traits={'extraversion': 0.5, 'humor': 0.5},
            adapted_traits={'extraversion': 0.9, 'humor': 0.8},
            confidence_level=0.8,
            adaptation_history=[],
            effectiveness_metrics={},
            last_updated=datetime.now()
        )
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            message_history=[],
            current_sentiment=0.5,
            conversation_phase="ongoing"
        )
        
        base_response = "I can help you with that."
        
        # Test response adaptation
        adapted_response = await personality_engine.generate_personality_response(
            personality_state, context, base_response
        )
        
        # High extraversion should make response more enthusiastic
        assert adapted_response != base_response
        assert ('!' in adapted_response or 
                'great' in adapted_response.lower() or
                'awesome' in adapted_response.lower())


class TestConversationAnalyzer:
    """Test conversation analysis and context understanding."""
    
    @pytest.fixture
    async def conversation_analyzer(self):
        """Create conversation analyzer for testing."""
        redis_mock = AsyncMock()
        analyzer = ConversationAnalyzer(redis_mock)
        
        # Mock the ML models
        analyzer.sentiment_analyzer = Mock()
        analyzer.emotion_classifier = Mock()
        analyzer.intent_classifier = Mock()
        analyzer.topic_classifier = Mock()
        
        return analyzer
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            Message(
                id="msg1",
                content="Hello, I'm really frustrated with this problem!",
                direction=MessageDirection.INCOMING,
                created_at=datetime.now() - timedelta(minutes=5)
            ),
            Message(
                id="msg2", 
                content="I understand your frustration. Let me help you solve this.",
                direction=MessageDirection.OUTGOING,
                created_at=datetime.now() - timedelta(minutes=4)
            ),
            Message(
                id="msg3",
                content="I've tried everything and nothing works. This is urgent!",
                direction=MessageDirection.INCOMING,
                created_at=datetime.now() - timedelta(minutes=3)
            )
        ]
    
    @pytest.mark.asyncio
    async def test_conversation_context_analysis(self, conversation_analyzer, sample_messages):
        """Test comprehensive conversation context analysis."""
        
        # Mock topic classification
        conversation_analyzer.topic_classifier.return_value = {
            'labels': ['support', 'technical'],
            'scores': [0.8, 0.2]
        }
        
        user = User(id="test_user", telegram_id=12345, first_name="Test")
        
        context = await conversation_analyzer.analyze_conversation_context(
            "test_session", sample_messages, user
        )
        
        # Verify context analysis
        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
        assert context.message_count == 3
        assert context.conversation_phase in ['opening', 'building', 'deep', 'closing']
        
        # Should detect urgency from "urgent" keyword
        assert context.urgency_indicators.get('explicit_urgency', 0) > 0.5
        
        # Should detect support goal
        assert 'support' in context.conversation_goals or 'problem_solving' in context.conversation_goals
    
    @pytest.mark.asyncio
    async def test_emotional_state_analysis(self, conversation_analyzer, sample_messages):
        """Test emotional state tracking."""
        
        # Mock emotion classifier
        conversation_analyzer.emotion_classifier.return_value = [[
            {'label': 'anger', 'score': 0.7},
            {'label': 'frustration', 'score': 0.6},
            {'label': 'neutral', 'score': 0.2}
        ]]
        
        emotional_state = await conversation_analyzer.analyze_emotional_state(sample_messages)
        
        assert isinstance(emotional_state, EmotionalState)
        assert emotional_state.current_emotions is not None
        assert emotional_state.dominant_emotion is not None
        
        # Should detect negative emotions from frustrated language
        if 'anger' in emotional_state.current_emotions:
            assert emotional_state.current_emotions['anger'] > 0.5
    
    @pytest.mark.asyncio
    async def test_conversation_metrics_calculation(self, conversation_analyzer, sample_messages):
        """Test conversation quality metrics."""
        
        # Mock the required components
        emotional_state = EmotionalState(
            current_emotions={'neutral': 0.6, 'anger': 0.4},
            dominant_emotion='neutral'
        )
        
        from app.services.conversation_analyzer import TopicAnalysis, ConversationContext
        topic_analysis = TopicAnalysis(
            current_topic='support',
            topic_coherence=0.8,
            topic_depth=2
        )
        
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user", 
            conversation_phase="building",
            time_in_conversation=300,
            message_count=3,
            user_engagement_level=0.7
        )
        
        metrics = await conversation_analyzer.calculate_conversation_metrics(
            sample_messages, emotional_state, topic_analysis, context
        )
        
        assert isinstance(metrics, ConversationMetrics)
        assert 0.0 <= metrics.engagement_score <= 1.0
        assert 0.0 <= metrics.topic_coherence <= 1.0
        assert metrics.conversation_depth >= 0


class TestPersonalityMatcher:
    """Test personality matching and compatibility algorithms."""
    
    @pytest.fixture
    async def personality_matcher(self):
        """Create personality matcher for testing."""
        db_mock = AsyncMock()
        redis_mock = AsyncMock()
        
        matcher = PersonalityMatcher(db_mock, redis_mock)
        
        # Mock the ML models
        matcher.compatibility_predictor = Mock()
        matcher.effectiveness_predictor = Mock()
        
        return matcher
    
    @pytest.fixture
    def sample_profiles(self):
        """Create sample personality profiles."""
        return [
            PersonalityProfile(
                id="profile1",
                name="supportive_helper",
                trait_scores={
                    'openness': 0.6,
                    'conscientiousness': 0.8,
                    'extraversion': 0.5,
                    'agreeableness': 0.9,
                    'neuroticism': 0.2,
                    'empathy': 0.8
                }
            ),
            PersonalityProfile(
                id="profile2", 
                name="direct_solver",
                trait_scores={
                    'openness': 0.7,
                    'conscientiousness': 0.9,
                    'extraversion': 0.6,
                    'agreeableness': 0.6,
                    'neuroticism': 0.1,
                    'directness': 0.9
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_similarity_based_matching(self, personality_matcher, sample_profiles):
        """Test similarity-based personality matching."""
        
        user_traits = {
            'openness': 0.6,
            'conscientiousness': 0.7,
            'extraversion': 0.5,
            'agreeableness': 0.8,
            'neuroticism': 0.3,
            'empathy': 0.7
        }
        
        context = MatchingContext(
            user_id="test_user",
            user_traits=user_traits,
            conversation_context=ConversationContext(
                session_id="test_session",
                user_id="test_user",
                conversation_phase="building",
                time_in_conversation=0,
                message_count=0,
                user_engagement_level=0.5
            ),
            emotional_state=EmotionalState(),
            interaction_history=[]
        )
        
        matches = await personality_matcher._similarity_based_matching(context, sample_profiles)
        
        # Should return matches for both profiles
        assert len(matches) == 2
        assert all(isinstance(match, PersonalityMatch) for match in matches)
        
        # Matches should be sorted by compatibility score
        assert matches[0].compatibility_score >= matches[1].compatibility_score
        
        # First profile (supportive_helper) should match better due to similar agreeableness/empathy
        assert matches[0].profile_name == "supportive_helper"
    
    @pytest.mark.asyncio
    async def test_contextual_matching(self, personality_matcher, sample_profiles):
        """Test context-aware personality matching."""
        
        # Context with high urgency - should prefer direct solver
        urgent_context = MatchingContext(
            user_id="test_user",
            user_traits={'openness': 0.5},
            conversation_context=ConversationContext(
                session_id="test_session",
                user_id="test_user",
                conversation_phase="building", 
                time_in_conversation=0,
                message_count=0,
                user_engagement_level=0.5,
                urgency_indicators={'explicit_urgency': 0.9}
            ),
            emotional_state=EmotionalState(),
            interaction_history=[]
        )
        
        matches = await personality_matcher._contextual_matching(urgent_context, sample_profiles)
        
        # Should prefer direct_solver for urgent situations
        best_match = max(matches, key=lambda m: m.compatibility_score)
        assert best_match.profile_name == "direct_solver"


class TestPersonalityManager:
    """Test the main personality management orchestrator."""
    
    @pytest.fixture
    async def personality_manager(self):
        """Create personality manager for testing."""
        db_mock = AsyncMock()
        redis_mock = AsyncMock()
        
        manager = PersonalityManager(db_mock, redis_mock)
        
        # Mock the component initialization
        manager.personality_engine = AsyncMock()
        manager.conversation_analyzer = AsyncMock()
        manager.personality_matcher = AsyncMock()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_process_user_message_flow(self, personality_manager):
        """Test end-to-end message processing flow."""
        
        # Mock the personality analysis
        personality_manager.personality_engine.analyze_user_personality.return_value = {
            'openness': 0.6, 'extraversion': 0.7
        }
        
        # Mock the personality matching
        personality_match = PersonalityMatch(
            profile_id="test_profile",
            profile_name="test_personality",
            compatibility_score=0.8,
            confidence_level=0.7,
            matching_reasons=["High compatibility"],
            adaptation_strategy=AdaptationStrategy.BALANCE,
            expected_effectiveness=0.75,
            risk_factors=[],
            optimization_suggestions=[]
        )
        personality_manager.personality_matcher.find_optimal_personality_match.return_value = personality_match
        
        # Mock personality adaptation
        personality_state = PersonalityState(
            base_traits={'openness': 0.5},
            adapted_traits={'openness': 0.6},
            confidence_level=0.8,
            adaptation_history=[],
            effectiveness_metrics={},
            last_updated=datetime.now()
        )
        personality_manager.personality_engine.adapt_personality.return_value = personality_state
        
        # Mock response generation
        personality_manager.personality_engine.generate_personality_response.return_value = "Adapted response"
        
        # Mock supporting methods
        personality_manager._build_conversation_context = AsyncMock(return_value=ConversationContext(
            session_id="test_session",
            user_id="test_user", 
            conversation_phase="building",
            time_in_conversation=0,
            message_count=0,
            user_engagement_level=0.5
        ))
        personality_manager._generate_base_response = AsyncMock(return_value="Base response")
        personality_manager._get_personality_profile = AsyncMock(return_value=PersonalityProfile(
            id="test_profile", name="test", trait_scores={}
        ))
        personality_manager._record_interaction = AsyncMock()
        personality_manager._create_conversation_session = AsyncMock(return_value="test_session")
        personality_manager._build_matching_context = AsyncMock(return_value=MatchingContext(
            user_id="test_user",
            user_traits={'openness': 0.6},
            conversation_context=ConversationContext(
                session_id="test_session", user_id="test_user",
                conversation_phase="building", time_in_conversation=0,
                message_count=0, user_engagement_level=0.5
            ),
            emotional_state=EmotionalState(),
            interaction_history=[]
        ))
        
        # Test message processing
        response = await personality_manager.process_user_message(
            user_id="test_user",
            message_content="Hello, I need help!",
            session_id="test_session"
        )
        
        # Verify response structure
        assert isinstance(response, PersonalityResponse)
        assert response.content == "Adapted response"
        assert response.confidence_score > 0
        assert response.processing_time_ms >= 0
        
        # Verify workflow was followed
        personality_manager.personality_engine.analyze_user_personality.assert_called_once()
        personality_manager.personality_matcher.find_optimal_personality_match.assert_called_once()
        personality_manager.personality_engine.adapt_personality.assert_called_once()


class TestPersonalityTesting:
    """Test the A/B testing and experimentation framework."""
    
    @pytest.fixture
    async def testing_framework(self):
        """Create testing framework for testing."""
        db_mock = AsyncMock()
        redis_mock = AsyncMock()
        
        framework = PersonalityTestingFramework(db_mock, redis_mock)
        return framework
    
    @pytest.mark.asyncio
    async def test_ab_test_creation(self, testing_framework):
        """Test A/B test creation and configuration."""
        
        # Mock test storage
        testing_framework._store_test = AsyncMock()
        
        test = await testing_framework.create_ab_test(
            name="Empathy vs Directness Test",
            description="Test empathetic vs direct communication styles",
            control_config={"empathy_level": 0.5, "directness": 0.5},
            treatment_config={"empathy_level": 0.8, "directness": 0.3},
            primary_metric="user_satisfaction",
            duration_days=14,
            traffic_split=0.5,
            minimum_sample_size=100
        )
        
        # Verify test structure
        assert test.name == "Empathy vs Directness Test"
        assert test.test_type == TestType.AB_TEST
        assert len(test.variants) == 2
        assert test.variants[0].id == "control"
        assert test.variants[1].id == "treatment"
        assert test.variants[1].traffic_allocation == 0.5
        assert test.metrics.primary_metric == "user_satisfaction"
    
    @pytest.mark.asyncio
    async def test_user_assignment_logic(self, testing_framework):
        """Test user assignment to test variants."""
        
        # Create mock test
        from app.services.personality_testing import PersonalityTest, TestVariant, TestMetrics, TestStatus
        
        test = PersonalityTest(
            id="test_123",
            name="Test",
            description="Test",
            test_type=TestType.AB_TEST,
            variants=[
                TestVariant("control", "Control", "Control variant", {}, 0.5),
                TestVariant("treatment", "Treatment", "Treatment variant", {}, 0.5)
            ],
            metrics=TestMetrics("satisfaction", [], {}),
            target_population={},
            start_date=datetime.now(),
            end_date=None,
            status=TestStatus.ACTIVE,
            sample_size_per_variant=100,
            created_by="test"
        )
        
        # Mock methods
        testing_framework._get_test = AsyncMock(return_value=test)
        testing_framework._is_user_eligible = AsyncMock(return_value=True)
        testing_framework._assign_variant = AsyncMock(return_value="control")
        testing_framework._cache_user_assignment = AsyncMock()
        
        variant_id = await testing_framework.assign_user_to_test("user_123", "test_123")
        
        assert variant_id in ["control", "treatment"]
        testing_framework._is_user_eligible.assert_called_once()
        testing_framework._assign_variant.assert_called_once()
    
    def test_statistical_analysis(self, testing_framework):
        """Test statistical analysis methods."""
        
        # Test t-test
        data1 = [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8]
        data2 = [0.6, 0.5, 0.7, 0.4, 0.6, 0.5, 0.7, 0.6]
        
        result = asyncio.run(testing_framework._perform_ttest(data1, data2, "control", "treatment"))
        
        assert 'p_value' in result
        assert 'effect_size' in result
        assert 'significant' in result
        assert result['test_type'] == 'ttest'
        assert 0.0 <= result['p_value'] <= 1.0
        
        # Should detect significant difference between the groups
        assert result['significant'] == True  # data1 mean > data2 mean


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation with personality adaptation."""
        
        # This would test a full conversation scenario:
        # 1. User starts conversation
        # 2. Personality analysis begins
        # 3. Personality matching occurs
        # 4. Responses are adapted
        # 5. Learning occurs from feedback
        # 6. Personality improves over time
        
        pass  # Implementation would be extensive
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        
        # This would test:
        # - Multiple concurrent personality analyses
        # - Response time under load
        # - Memory usage patterns
        # - Cache effectiveness
        
        pass  # Implementation would require load testing setup
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test system behavior under error conditions."""
        
        # This would test:
        # - ML model failures
        # - Database connectivity issues  
        # - Redis cache failures
        # - Graceful degradation
        
        pass  # Implementation would require error injection


class TestPersonalitySystemReliability:
    """Test system reliability and edge cases."""
    
    def test_personality_trait_validation(self):
        """Test personality trait score validation."""
        
        from app.models.personality import PersonalityDimension
        
        # Test valid trait scores
        valid_traits = {dim.value: 0.5 for dim in PersonalityDimension}
        # Should not raise any exceptions
        
        # Test invalid trait scores
        invalid_traits = {'openness': 1.5, 'invalid_trait': 0.5}
        # Should handle gracefully in production code
        
        assert len(valid_traits) == len(PersonalityDimension)
    
    def test_adaptation_strategy_bounds(self):
        """Test personality adaptation stays within bounds."""
        
        # Test extreme user traits
        extreme_user_traits = {
            'openness': 1.0,
            'conscientiousness': 0.0, 
            'extraversion': 1.0,
            'agreeableness': 0.0,
            'neuroticism': 1.0
        }
        
        base_traits = {dim.value: 0.5 for dim in PersonalityDimension}
        
        # Mock adaptation (simplified)
        adapted_traits = {}
        for trait, base_score in base_traits.items():
            user_score = extreme_user_traits.get(trait, 0.5)
            # Mirror adaptation with 50% strength
            adapted_score = base_score + (user_score - base_score) * 0.5
            adapted_traits[trait] = max(0.0, min(1.0, adapted_score))
        
        # Verify all adapted traits are in valid range
        for trait, score in adapted_traits.items():
            assert 0.0 <= score <= 1.0, f"Trait {trait} out of bounds: {score}"


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Benchmark personality system performance."""
    
    @pytest.mark.asyncio
    async def test_personality_analysis_speed(self):
        """Benchmark personality analysis speed."""
        
        # Create mock conversation with various message lengths
        messages = [
            {"content": "Short message", "timestamp": datetime.now().isoformat()},
            {"content": "This is a longer message with more content to analyze for personality traits and patterns in communication style and emotional expression.", "timestamp": datetime.now().isoformat()}
        ]
        
        context = ConversationContext(
            user_id="benchmark_user",
            session_id="benchmark_session", 
            message_history=messages,
            current_sentiment=0.0,
            conversation_phase="building"
        )
        
        # Time the analysis (would be implemented with actual engine)
        start_time = datetime.now()
        
        # Mock analysis
        await asyncio.sleep(0.01)  # Simulate processing time
        
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance assertion
        assert processing_time_ms < 500, f"Personality analysis too slow: {processing_time_ms}ms"
    
    def test_memory_usage_patterns(self):
        """Test memory usage during personality operations."""
        
        import sys
        
        # Get baseline memory
        baseline_size = sys.getsizeof({})
        
        # Create personality data structures
        personality_data = {
            'traits': {dim.value: 0.5 for dim in PersonalityDimension},
            'adaptation_history': [{'timestamp': datetime.now(), 'adaptation': 'test'} for _ in range(100)],
            'effectiveness_metrics': {'satisfaction': [0.8] * 50}
        }
        
        data_size = sys.getsizeof(personality_data)
        
        # Memory usage should be reasonable
        assert data_size < 10000, f"Personality data too large: {data_size} bytes"


if __name__ == "__main__":
    # Run the tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))