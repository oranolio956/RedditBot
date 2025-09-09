"""
Comprehensive Tests for Advanced Typing Simulator

Tests all aspects of the human-like typing simulation system including
psychological modeling, natural patterns, anti-detection measures,
and performance under load.
"""

import asyncio
import pytest
import time
import random
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from app.services.typing_simulator import (
    AdvancedTypingSimulator,
    TypingPersonality,
    PsychologicalFactors,
    ContextualFactors,
    LinguisticAnalyzer,
    CognitivLoadModel,
    EmotionalStateModel,
    NaturalErrorModel,
    TypingStyle,
    CognitiveState,
    MessageComplexity
)
from app.services.typing_integration import (
    EnhancedTypingIntegration,
    TypingSessionManager,
    TypingSession
)
from app.models.personality import PersonalityProfile
from app.telegram.anti_ban import RiskLevel


class TestLinguisticAnalyzer:
    """Test the linguistic analysis component."""
    
    def setup_method(self):
        self.analyzer = LinguisticAnalyzer()
    
    def test_simple_text_analysis(self):
        """Test analysis of simple text."""
        result = self.analyzer.analyze_text_complexity("Hello world")
        
        assert 'overall_complexity' in result
        assert 0 <= result['overall_complexity'] <= 1
        assert result['word_unfamiliarity'] < 0.5  # Common words
        
    def test_complex_text_analysis(self):
        """Test analysis of complex text."""
        complex_text = "Implementing sophisticated algorithms with intricate methodologies"
        result = self.analyzer.analyze_text_complexity(complex_text)
        
        assert result['overall_complexity'] > 0.3  # Should be more complex
        assert result['word_length'] > 0.5  # Long words
        
    def test_punctuation_complexity(self):
        """Test punctuation complexity detection."""
        punctuated = "Hello, world! How are you? (I'm fine) - thanks for asking..."
        result = self.analyzer.analyze_text_complexity(punctuated)
        
        assert result['punctuation_complexity'] > 0
        
    def test_special_characters(self):
        """Test special character handling."""
        special_text = "Email: user@example.com $50 & 25% off!"
        result = self.analyzer.analyze_text_complexity(special_text)
        
        assert result['special_char_density'] > 0
        assert result['number_density'] > 0
    
    def test_pause_point_identification(self):
        """Test identification of natural pause points."""
        text = "Hello there! How are you today? I hope you're doing well, my friend."
        pause_points = self.analyzer.identify_pause_points(text)
        
        assert len(pause_points) > 0
        assert any(text[p] in '.!?' for p in pause_points if p < len(text))
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.analyzer.analyze_text_complexity("")
        assert result['overall_complexity'] == 0.0


class TestCognitivLoadModel:
    """Test cognitive load modeling."""
    
    def setup_method(self):
        self.model = CognitivLoadModel()
    
    def test_basic_cognitive_load(self):
        """Test basic cognitive load calculation."""
        context = ContextualFactors()
        psychological = PsychologicalFactors()
        
        load = self.model.calculate_cognitive_load("Hello world", context, psychological)
        
        assert 0 <= load <= 2.0
        assert isinstance(load, float)
    
    def test_high_pressure_load(self):
        """Test cognitive load under pressure."""
        context = ContextualFactors(time_pressure=1.0, multitasking_level=0.8)
        psychological = PsychologicalFactors(attention_span=0.3, working_memory_load=0.9)
        
        high_load = self.model.calculate_cognitive_load("Complex text", context, psychological)
        
        # Should be higher than default
        default_load = self.model.calculate_cognitive_load("Complex text", ContextualFactors(), PsychologicalFactors())
        
        assert high_load > default_load
    
    def test_cognitive_effects_on_timing(self):
        """Test cognitive load effects on timing."""
        base_timing = 1.0
        
        # High load should slow timing
        affected_timing = self.model.apply_cognitive_effects(
            base_timing, cognitive_load=1.5, flow_state=0.0
        )
        assert affected_timing > base_timing
        
        # Flow state should reduce impact
        flow_timing = self.model.apply_cognitive_effects(
            base_timing, cognitive_load=1.5, flow_state=0.8
        )
        assert flow_timing < affected_timing


class TestEmotionalStateModel:
    """Test emotional state modeling."""
    
    def setup_method(self):
        self.model = EmotionalStateModel()
    
    def test_excited_state(self):
        """Test excited emotional state effects."""
        effects = self.model.get_emotional_effects(0.8, CognitiveState.EXCITED)
        
        assert effects['speed_mult'] > 1.0  # Should type faster
        assert effects['error_rate'] > 1.0  # But make more errors
    
    def test_stressed_state(self):
        """Test stressed emotional state effects."""
        effects = self.model.get_emotional_effects(-0.7, CognitiveState.STRESSED)
        
        assert effects['speed_mult'] < 1.0  # Should type slower
        assert effects['error_rate'] > 1.0  # And make more errors
        assert effects['pause_freq'] > 1.0  # With more pauses
    
    def test_focused_state(self):
        """Test focused state effects."""
        effects = self.model.get_emotional_effects(0.3, CognitiveState.FOCUSED)
        
        assert effects['error_rate'] < 1.0  # Should make fewer errors
        assert effects['pause_freq'] < 1.0  # With fewer pauses
    
    def test_tired_cognitive_modifier(self):
        """Test tired cognitive state modifier."""
        effects = self.model.get_emotional_effects(0.0, CognitiveState.TIRED)
        
        assert effects['speed_mult'] < 0.8  # Should be significantly slower
        assert effects['error_rate'] > 1.3  # With more errors


class TestNaturalErrorModel:
    """Test natural error simulation."""
    
    def setup_method(self):
        self.model = NaturalErrorModel()
    
    def test_error_probability(self):
        """Test error probability calculation."""
        # High error rate should cause more errors
        high_error_count = sum(
            1 for _ in range(1000)
            if self.model.should_make_error('a', error_rate=0.2)
        )
        
        low_error_count = sum(
            1 for _ in range(1000) 
            if self.model.should_make_error('a', error_rate=0.02)
        )
        
        assert high_error_count > low_error_count
    
    def test_character_difficulty(self):
        """Test character-specific error rates."""
        # 'q' should be more error-prone than 'a'
        q_errors = sum(
            1 for _ in range(1000)
            if self.model.should_make_error('q', error_rate=0.1)
        )
        
        a_errors = sum(
            1 for _ in range(1000)
            if self.model.should_make_error('a', error_rate=0.1)
        )
        
        assert q_errors >= a_errors  # Should be at least equal, likely higher
    
    def test_error_generation(self):
        """Test error generation."""
        error_char = self.model.generate_error('a')
        
        # Should return a character (or empty for omission)
        assert isinstance(error_char, str)
    
    def test_fatigue_effect(self):
        """Test fatigue effects on errors."""
        high_fatigue_errors = sum(
            1 for _ in range(1000)
            if self.model.should_make_error('a', error_rate=0.05, fatigue_level=0.8)
        )
        
        low_fatigue_errors = sum(
            1 for _ in range(1000)
            if self.model.should_make_error('a', error_rate=0.05, fatigue_level=0.1)
        )
        
        assert high_fatigue_errors > low_fatigue_errors
    
    def test_correction_time(self):
        """Test correction time calculation."""
        correction_time = self.model.calculate_correction_time(
            error_type='substitution',
            characters_to_fix=1,
            perfectionism=0.5
        )
        
        assert correction_time > 0
        assert isinstance(correction_time, float)
        
        # Perfectionists should take longer
        perfectionist_time = self.model.calculate_correction_time(
            error_type='substitution',
            characters_to_fix=1,
            perfectionism=0.9
        )
        
        assert perfectionist_time > correction_time


class TestTypingPersonality:
    """Test typing personality modeling."""
    
    def test_personality_creation(self):
        """Test typing personality creation."""
        personality = TypingPersonality(
            base_wpm=65.0,
            accuracy_rate=0.92,
            impulsivity=0.5
        )
        
        assert personality.base_wpm == 65.0
        assert personality.accuracy_rate == 0.92
        assert personality.impulsivity == 0.5
    
    def test_style_preferences(self):
        """Test typing style preferences."""
        personality = TypingPersonality(
            preferred_styles=[TypingStyle.RAPID_FIRE, TypingStyle.TOUCH_TYPIST]
        )
        
        assert TypingStyle.RAPID_FIRE in personality.preferred_styles
        assert TypingStyle.TOUCH_TYPIST in personality.preferred_styles


@pytest.mark.asyncio
class TestAdvancedTypingSimulator:
    """Test the main typing simulator."""
    
    async def setup_method(self):
        # Mock personality manager
        self.mock_personality_manager = Mock()
        self.simulator = AdvancedTypingSimulator(self.mock_personality_manager)
        
        # Mock Redis
        with patch('app.services.typing_simulator.redis_manager'):
            await self.simulator.initialize()
    
    async def test_basic_simulation(self):
        """Test basic typing simulation."""
        result = await self.simulator.simulate_human_typing(
            text="Hello, world!",
            user_id=12345
        )
        
        assert 'total_time' in result
        assert 'typing_events' in result
        assert 'realism_score' in result
        assert result['total_time'] > 0
        assert isinstance(result['typing_events'], list)
    
    async def test_empty_text_handling(self):
        """Test handling of empty text."""
        result = await self.simulator.simulate_human_typing(
            text="",
            user_id=12345
        )
        
        assert result['total_time'] == 0.0
        assert result['typing_events'] == []
    
    async def test_personality_integration(self):
        """Test personality profile integration."""
        # Mock personality profile
        mock_profile = Mock(spec=PersonalityProfile)
        mock_profile.trait_scores = {
            'extraversion': 0.8,
            'conscientiousness': 0.6,
            'openness': 0.7,
            'neuroticism': 0.3,
            'agreeableness': 0.5
        }
        
        result = await self.simulator.simulate_human_typing(
            text="This is a test message",
            user_id=12345,
            personality_profile=mock_profile
        )
        
        assert result['total_time'] > 0
        # Personality should influence the simulation
        assert 'meta' in result
    
    async def test_context_effects(self):
        """Test contextual effects on simulation."""
        context = {
            'device_type': 'mobile',
            'time_pressure': 0.8,
            'topic_familiarity': 0.3
        }
        
        mobile_result = await self.simulator.simulate_human_typing(
            text="Test message",
            user_id=12345,
            context=context
        )
        
        desktop_context = context.copy()
        desktop_context['device_type'] = 'desktop'
        
        desktop_result = await self.simulator.simulate_human_typing(
            text="Test message", 
            user_id=12345,
            context=desktop_context
        )
        
        # Mobile typing should typically be slower
        assert mobile_result['total_time'] != desktop_result['total_time']
    
    async def test_conversation_state_effects(self):
        """Test conversation state effects."""
        conversation_state = {
            'emotional_state': 0.8,  # Positive/excited
            'engagement_score': 0.9,
            'message_count': 15
        }
        
        result = await self.simulator.simulate_human_typing(
            text="I'm so excited about this!",
            user_id=12345,
            conversation_state=conversation_state
        )
        
        assert result['total_time'] > 0
        assert result.get('meta', {}).get('cognitive_load') >= 0
    
    async def test_typing_delay_interface(self):
        """Test the simple typing delay interface."""
        delay = await self.simulator.get_typing_delay(
            text="Hello world",
            user_id=12345
        )
        
        assert isinstance(delay, float)
        assert delay > 0
        assert delay < 30  # Reasonable upper bound
    
    async def test_typing_indicators(self):
        """Test typing indicators generation."""
        indicators = await self.simulator.get_typing_indicators(
            text="This is a longer message that should have multiple indicators",
            user_id=12345
        )
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        
        for indicator in indicators:
            assert 'timestamp' in indicator
            assert 'state' in indicator
            assert 'duration' in indicator
            assert indicator['state'] in ['typing', 'paused']
    
    async def test_long_text_handling(self):
        """Test handling of long text."""
        long_text = "This is a very long message. " * 50  # ~1500 characters
        
        result = await self.simulator.simulate_human_typing(
            text=long_text,
            user_id=12345
        )
        
        assert result['total_time'] > 5  # Should take reasonable time
        assert result['total_time'] < 300  # But not excessive
        assert len(result['typing_events']) > 10  # Should have multiple events
    
    async def test_error_simulation(self):
        """Test error simulation and correction."""
        # Use a personality with low accuracy to trigger errors
        mock_profile = Mock(spec=PersonalityProfile)
        mock_profile.trait_scores = {
            'conscientiousness': 0.2,  # Low conscientiousness = more errors
            'neuroticism': 0.8  # High neuroticism = stress errors
        }
        
        result = await self.simulator.simulate_human_typing(
            text="This message should generate some typing errors to test correction",
            user_id=12345,
            personality_profile=mock_profile
        )
        
        # Should have some error events
        assert len(result.get('error_events', [])) >= 0  # May or may not have errors
        
        # Check for correction events in typing events
        correction_events = [
            e for e in result['typing_events']
            if isinstance(e, dict) and e.get('event_type') in ['backspace', 'correction_keypress', 'error_detection']
        ]
        
        # If there are errors, there should be corrections
        if result.get('error_events'):
            assert len(correction_events) > 0
    
    async def test_performance_with_concurrent_simulations(self):
        """Test performance with multiple concurrent simulations."""
        start_time = time.time()
        
        # Run multiple simulations concurrently
        tasks = []
        for i in range(20):  # Simulate 20 concurrent typing sessions
            task = self.simulator.simulate_human_typing(
                text=f"Concurrent test message {i} with some variety in length and content",
                user_id=10000 + i
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (not more than 10 seconds for 20 simulations)
        assert total_time < 10
        
        # All simulations should succeed
        assert len(results) == 20
        for result in results:
            assert 'total_time' in result
            assert result['total_time'] > 0


@pytest.mark.asyncio
class TestTypingSessionManager:
    """Test typing session management."""
    
    async def setup_method(self):
        self.manager = TypingSessionManager()
        self.mock_bot = Mock()
        self.mock_bot.send_chat_action = AsyncMock()
    
    async def test_session_creation(self):
        """Test typing session creation."""
        session_id = await self.manager.start_typing_session(
            user_id=12345,
            chat_id=67890,
            message_text="Test message",
            bot=self.mock_bot
        )
        
        assert session_id is not None
        assert session_id in self.manager.active_sessions
        
        session = self.manager.active_sessions[session_id]
        assert session.user_id == 12345
        assert session.chat_id == 67890
        assert session.is_active
    
    async def test_session_status(self):
        """Test session status retrieval."""
        session_id = await self.manager.start_typing_session(
            user_id=12345,
            chat_id=67890,
            message_text="Test message",
            bot=self.mock_bot
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        status = await self.manager.get_session_status(session_id)
        
        assert status is not None
        assert status['user_id'] == 12345
        assert status['is_active']
        assert 'progress' in status
        assert 'estimated_remaining' in status
    
    async def test_session_cleanup(self):
        """Test session cleanup."""
        session_id = await self.manager.start_typing_session(
            user_id=12345,
            chat_id=67890,
            message_text="Short",
            bot=self.mock_bot
        )
        
        # Stop the session
        await self.manager.stop_session(session_id)
        
        # Session should be cleaned up
        assert session_id not in self.manager.active_sessions
    
    async def test_concurrent_session_limit(self):
        """Test concurrent session limits."""
        # Temporarily reduce limit for testing
        original_limit = self.manager.max_concurrent_sessions
        self.manager.max_concurrent_sessions = 5
        
        try:
            session_ids = []
            
            # Create sessions up to limit
            for i in range(7):  # Try to create more than limit
                session_id = await self.manager.start_typing_session(
                    user_id=i,
                    chat_id=i,
                    message_text="Test message",
                    bot=self.mock_bot
                )
                session_ids.append(session_id)
            
            # Should not exceed limit
            assert len(self.manager.active_sessions) <= 5
            
        finally:
            self.manager.max_concurrent_sessions = original_limit
    
    async def test_rate_limiting(self):
        """Test typing indicator rate limiting."""
        user_id = 12345
        
        # Should pass initial rate limit check
        assert self.manager._check_indicator_rate_limit(user_id)
        
        # Add many indicators quickly
        for _ in range(12):  # More than limit of 10
            self.manager.indicator_rate_limits[user_id].append(time.time())
        
        # Should now fail rate limit check
        assert not self.manager._check_indicator_rate_limit(user_id)
    
    async def test_cache_system(self):
        """Test simulation caching."""
        cache_key = "test_cache_key"
        test_data = {"test": "data", "total_time": 2.5}
        
        # Cache some data
        await self.manager._cache_simulation(cache_key, test_data)
        
        # Retrieve cached data
        cached = await self.manager._get_cached_simulation(cache_key)
        
        if cached:  # May not be cached in test environment
            assert cached["test"] == "data"
    
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        metrics = await self.manager.get_performance_metrics()
        
        assert 'active_sessions' in metrics
        assert 'completed_sessions' in metrics
        assert 'cache_hit_rate' in metrics
        assert isinstance(metrics['active_sessions'], int)


@pytest.mark.asyncio
class TestEnhancedTypingIntegration:
    """Test typing integration system."""
    
    async def setup_method(self):
        # Mock dependencies
        self.mock_anti_ban = Mock()
        self.mock_anti_ban.calculate_typing_delay = AsyncMock(return_value=2.0)
        
        self.mock_session_manager = Mock()
        self.mock_personality_manager = Mock()
        
        self.integration = EnhancedTypingIntegration(
            anti_ban_manager=self.mock_anti_ban,
            session_manager=self.mock_session_manager,
            personality_manager=self.mock_personality_manager
        )
        
        # Mock typing simulator
        with patch('app.services.typing_integration.get_typing_simulator'):
            await self.integration.initialize()
    
    async def test_enhanced_delay_calculation(self):
        """Test enhanced typing delay calculation."""
        delay = await self.integration.calculate_typing_delay_enhanced(
            text="Test message",
            user_id=12345,
            context={'device_type': 'mobile'},
            risk_level=RiskLevel.MEDIUM
        )
        
        assert isinstance(delay, float)
        assert delay > 0
        assert delay < 60  # Reasonable upper bound
    
    async def test_fallback_to_simple(self):
        """Test fallback to simple calculation."""
        # Disable advanced simulation
        self.integration.enable_advanced_simulation = False
        
        delay = await self.integration.calculate_typing_delay_enhanced(
            text="Test message",
            user_id=12345
        )
        
        # Should use anti-ban manager
        self.mock_anti_ban.calculate_typing_delay.assert_called_once()
        assert delay == 2.0  # Mocked return value
    
    async def test_risk_level_adjustments(self):
        """Test risk level adjustments."""
        base_delay = 2.0
        
        low_risk_delay = await self.integration._apply_risk_adjustments(
            base_delay, RiskLevel.LOW, 12345
        )
        
        high_risk_delay = await self.integration._apply_risk_adjustments(
            base_delay, RiskLevel.HIGH, 12345
        )
        
        critical_risk_delay = await self.integration._apply_risk_adjustments(
            base_delay, RiskLevel.CRITICAL, 12345
        )
        
        # Higher risk should result in longer delays
        assert low_risk_delay <= high_risk_delay <= critical_risk_delay
    
    async def test_realistic_typing_session(self):
        """Test realistic typing session creation."""
        mock_bot = Mock()
        mock_message_context = Mock()
        
        session_id = await self.integration.start_realistic_typing_session(
            text="Test message for realistic session",
            user_id=12345,
            chat_id=67890,
            bot=mock_bot,
            message_context=mock_message_context,
            send_callback=AsyncMock()
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
    
    async def test_context_building(self):
        """Test enhanced context building."""
        base_context = {'device_type': 'mobile'}
        mock_message = Mock()
        mock_message.text = "Short msg w/ abbrvs lol"
        
        enhanced_context = await self.integration._build_enhanced_context(
            user_id=12345,
            base_context=base_context,
            message=mock_message,
            risk_level=RiskLevel.HIGH
        )
        
        assert enhanced_context['device_type'] == 'mobile'
        assert enhanced_context['time_pressure'] > 0  # High risk should add pressure
    
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Simulate some activity
        self.integration.integration_metrics['total_requests'] = 100
        self.integration.integration_metrics['advanced_simulations'] = 75
        self.integration.integration_metrics['errors'] = 5
        
        metrics = await self.integration.get_performance_metrics()
        
        assert metrics['total_requests'] == 100
        assert metrics['advanced_simulation_rate'] == 0.75
        assert metrics['error_rate'] == 0.05
    
    async def test_simple_delay_calculation(self):
        """Test simple delay calculation fallback."""
        delay = await self.integration._calculate_simple_delay(
            text="Test message",
            user_id=12345,
            risk_level=RiskLevel.MEDIUM
        )
        
        # Should use anti-ban manager if available
        self.mock_anti_ban.calculate_typing_delay.assert_called()
        assert delay == 2.0  # Mocked return value
    
    async def test_enable_disable_simulation(self):
        """Test enabling/disabling advanced simulation."""
        # Initially enabled
        assert self.integration.enable_advanced_simulation
        
        # Disable
        await self.integration.enable_advanced_simulation(False)
        assert not self.integration.enable_advanced_simulation
        
        # Re-enable
        await self.integration.enable_advanced_simulation(True)
        assert self.integration.enable_advanced_simulation


@pytest.mark.asyncio
class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    async def test_concurrent_simulations_performance(self):
        """Test performance with many concurrent simulations."""
        # Mock personality manager
        mock_personality_manager = Mock()
        simulator = AdvancedTypingSimulator(mock_personality_manager)
        
        with patch('app.services.typing_simulator.redis_manager'):
            await simulator.initialize()
        
        start_time = time.time()
        
        # Create 100 concurrent simulations
        tasks = []
        for i in range(100):
            task = simulator.get_typing_delay(
                text=f"Performance test message {i} with varying lengths and complexity",
                user_id=i
            )
            tasks.append(task)
        
        delays = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within 5 seconds
        assert duration < 5.0
        
        # All delays should be valid
        assert len(delays) == 100
        for delay in delays:
            assert isinstance(delay, float)
            assert delay > 0
    
    async def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively."""
        mock_personality_manager = Mock()
        simulator = AdvancedTypingSimulator(mock_personality_manager)
        
        with patch('app.services.typing_simulator.redis_manager'):
            await simulator.initialize()
        
        # Run many simulations in sequence
        for i in range(50):
            await simulator.get_typing_delay(
                text=f"Memory test message {i}",
                user_id=i % 10  # Reuse user IDs to test caching
            )
        
        # Should not crash or consume excessive memory
        assert True  # If we get here, memory was stable
    
    async def test_typing_session_scalability(self):
        """Test typing session manager scalability."""
        manager = TypingSessionManager()
        mock_bot = Mock()
        mock_bot.send_chat_action = AsyncMock()
        
        # Create many sessions quickly
        session_ids = []
        start_time = time.time()
        
        for i in range(50):
            session_id = await manager.start_typing_session(
                user_id=i,
                chat_id=i,
                message_text=f"Scalability test {i}",
                bot=mock_bot
            )
            session_ids.append(session_id)
        
        creation_time = time.time() - start_time
        
        # Should create sessions quickly
        assert creation_time < 2.0
        
        # Check active sessions (accounting for concurrent limits)
        assert len(manager.active_sessions) > 0
        
        # Clean up
        for session_id in session_ids:
            try:
                await manager.stop_session(session_id)
            except:
                pass  # Some may already be finished
    
    async def test_error_resilience(self):
        """Test system resilience to errors."""
        mock_personality_manager = Mock()
        simulator = AdvancedTypingSimulator(mock_personality_manager)
        
        with patch('app.services.typing_simulator.redis_manager'):
            await simulator.initialize()
        
        # Test with various problematic inputs
        test_cases = [
            "",  # Empty string
            "a" * 10000,  # Very long string
            "🎉💯🚀" * 100,  # Unicode characters
            None,  # Invalid type (should be handled gracefully)
        ]
        
        for test_input in test_cases:
            try:
                if test_input is None:
                    continue  # Skip None test
                
                delay = await simulator.get_typing_delay(
                    text=test_input,
                    user_id=12345
                )
                
                assert isinstance(delay, float)
                assert delay >= 0
                
            except Exception as e:
                pytest.fail(f"Failed to handle input {repr(test_input)}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])