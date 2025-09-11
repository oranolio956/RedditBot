"""
Integration Tests for Anti-Detection System

Tests for anti-detection mechanisms including:
- Human behavior pattern simulation
- Activity timing randomization
- Detection avoidance strategies
- Behavioral pattern analysis
- Natural conversation flow
- Typing speed variation
- Response delay algorithms
"""

import pytest
import asyncio
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import random

from app.telegram.anti_ban import AntiBanSystem, BehaviorPattern, ActivityType
from app.services.temporal_archaeology import TemporalArchaeology
from app.models.telegram_account import TelegramAccount, SafetyLevel


class TestHumanBehaviorSimulation:
    """Test human behavior pattern simulation"""
    
    @pytest.fixture
    def anti_ban_system(self):
        """Create anti-ban system for testing"""
        return AntiBanSystem(
            detection_threshold=0.7,
            cooling_period=300,
            natural_variance_factor=0.3
        )
    
    @pytest.fixture
    def test_account(self):
        """Create test account"""
        return TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            safety_level=SafetyLevel.MODERATE
        )
    
    @pytest.mark.asyncio
    async def test_typing_speed_humanization(self, anti_ban_system):
        """Test human-like typing speed variation"""
        base_speed = 15  # chars per second
        
        # Generate multiple typing speeds
        speeds = []
        for _ in range(100):
            speed = await anti_ban_system.humanize_typing_speed(base_speed)
            speeds.append(speed)
        
        # Should have natural variation
        mean_speed = statistics.mean(speeds)
        std_dev = statistics.stdev(speeds)
        
        # Mean should be close to base speed
        assert abs(mean_speed - base_speed) < 2.0
        
        # Should have reasonable variation (not too uniform)
        assert std_dev > 1.0
        assert std_dev < 5.0
        
        # All speeds should be within human range
        assert all(8 <= speed <= 25 for speed in speeds)
    
    @pytest.mark.asyncio
    async def test_response_delay_naturalization(self, anti_ban_system):
        """Test natural response delay patterns"""
        message_complexity_levels = [
            ("simple", 0.2),    # "yes", "ok"
            ("medium", 0.5),    # "That's interesting"  
            ("complex", 0.8),   # "I think the implications of..."
            ("very_complex", 1.0)  # Long technical explanation
        ]
        
        for complexity_type, complexity_score in message_complexity_levels:
            delays = []
            for _ in range(50):
                delay = await anti_ban_system.calculate_natural_delay(
                    base_delay=30,
                    complexity_factor=complexity_score,
                    emotion_intensity=0.5
                )
                delays.append(delay)
            
            mean_delay = statistics.mean(delays)
            
            # More complex messages should generally take longer
            if complexity_score > 0.5:
                assert mean_delay > 20  # At least 20 seconds for complex
            
            # All delays should be reasonable
            assert all(2 <= delay <= 300 for delay in delays)
            
            # Should have variation (not all the same)
            assert statistics.stdev(delays) > 2.0
    
    @pytest.mark.asyncio
    async def test_activity_pattern_randomization(self, anti_ban_system):
        """Test activity pattern randomization"""
        # Generate activity schedule for a day
        activities = await anti_ban_system.generate_daily_activity_pattern(
            account_timezone="UTC",
            activity_level="moderate",
            preferred_hours=[9, 10, 11, 14, 15, 16, 19, 20, 21]
        )
        
        # Should have activities scheduled
        assert len(activities) >= 5
        assert len(activities) <= 20
        
        # Activities should be during preferred hours mostly
        preferred_count = sum(
            1 for activity in activities 
            if activity["hour"] in [9, 10, 11, 14, 15, 16, 19, 20, 21]
        )
        
        # At least 70% should be in preferred hours
        assert preferred_count / len(activities) >= 0.7
        
        # Should have different types of activities
        activity_types = {activity["type"] for activity in activities}
        assert len(activity_types) >= 3  # At least 3 different types
    
    @pytest.mark.asyncio
    async def test_message_timing_variance(self, anti_ban_system):
        """Test message timing variance to avoid patterns"""
        base_interval = 120  # 2 minutes
        
        # Generate message intervals
        intervals = []
        for _ in range(20):
            interval = await anti_ban_system.vary_message_interval(
                base_interval=base_interval,
                randomization_factor=0.4
            )
            intervals.append(interval)
        
        # Should have significant variance
        mean_interval = statistics.mean(intervals)
        assert abs(mean_interval - base_interval) < 30  # Within 30 seconds of base
        
        # Standard deviation should indicate good randomness
        std_dev = statistics.stdev(intervals)
        assert std_dev > 20  # At least 20 seconds variation
        
        # No two consecutive intervals should be identical
        consecutive_identical = sum(
            1 for i in range(len(intervals) - 1)
            if abs(intervals[i] - intervals[i + 1]) < 1
        )
        assert consecutive_identical == 0


class TestDetectionAvoidance:
    """Test detection avoidance strategies"""
    
    @pytest.fixture
    def anti_ban_system(self):
        return AntiBanSystem(detection_threshold=0.6)
    
    @pytest.mark.asyncio
    async def test_bot_pattern_detection(self, anti_ban_system):
        """Test detection of bot-like patterns"""
        # Bot-like pattern (too regular)
        bot_activities = [
            {
                "timestamp": datetime.utcnow() - timedelta(seconds=i * 60),  # Exactly every minute
                "activity": "message",
                "duration": 5  # Always 5 seconds
            }
            for i in range(20)
        ]
        
        bot_score = await anti_ban_system.detect_bot_patterns(bot_activities)
        assert bot_score > 0.7  # Should detect as bot-like
        
        # Human-like pattern (irregular)
        human_activities = [
            {
                "timestamp": datetime.utcnow() - timedelta(seconds=i * random.randint(30, 180)),
                "activity": random.choice(["message", "read", "typing"]),
                "duration": random.randint(2, 30)
            }
            for i in range(20)
        ]
        
        human_score = await anti_ban_system.detect_bot_patterns(human_activities)
        assert human_score < 0.4  # Should detect as human-like
    
    @pytest.mark.asyncio
    async def test_regularity_breaking(self, anti_ban_system):
        """Test breaking regular patterns"""
        # Start with regular pattern
        regular_schedule = [
            datetime.utcnow() + timedelta(minutes=i * 10)  # Every 10 minutes
            for i in range(12)
        ]
        
        # Apply pattern breaking
        irregular_schedule = await anti_ban_system.break_regularity(regular_schedule)
        
        # Should maintain similar number of activities
        assert abs(len(irregular_schedule) - len(regular_schedule)) <= 2
        
        # Should have different timings
        identical_count = sum(
            1 for orig, irreg in zip(regular_schedule, irregular_schedule)
            if abs((orig - irreg).total_seconds()) < 60  # Within 1 minute
        )
        
        # Most should be different
        assert identical_count < len(regular_schedule) * 0.3  # Less than 30% identical
    
    @pytest.mark.asyncio
    async def test_adaptive_behavior_modification(self, anti_ban_system):
        """Test adaptive behavior based on risk level"""
        # Low risk - normal behavior
        low_risk_behavior = await anti_ban_system.adapt_behavior_to_risk(
            risk_score=0.2,
            current_behavior={
                "message_frequency": 0.1,  # 10% chance per opportunity
                "response_speed": "normal",
                "activity_level": "moderate"
            }
        )
        
        # High risk - more cautious behavior
        high_risk_behavior = await anti_ban_system.adapt_behavior_to_risk(
            risk_score=0.8,
            current_behavior={
                "message_frequency": 0.1,
                "response_speed": "normal", 
                "activity_level": "moderate"
            }
        )
        
        # High risk should be more cautious
        assert high_risk_behavior["message_frequency"] < low_risk_behavior["message_frequency"]
        assert high_risk_behavior["response_speed"] == "slow"
        assert high_risk_behavior["activity_level"] in ["low", "minimal"]
    
    @pytest.mark.asyncio
    async def test_platform_mimicry(self, anti_ban_system):
        """Test platform-specific behavior mimicry"""
        platforms = ["mobile", "desktop", "web"]
        
        for platform in platforms:
            behavior_profile = await anti_ban_system.generate_platform_behavior(platform)
            
            if platform == "mobile":
                # Mobile users tend to have shorter messages, more voice notes
                assert behavior_profile["message_length_preference"] == "short"
                assert behavior_profile["voice_note_probability"] > 0.1
                assert behavior_profile["typo_probability"] > 0.05
            
            elif platform == "desktop":
                # Desktop users can type longer messages
                assert behavior_profile["message_length_preference"] in ["medium", "long"]
                assert behavior_profile["typing_speed_range"][1] > 20  # Can type faster
            
            elif platform == "web":
                # Web users have mixed behavior
                assert behavior_profile["platform_switches"] > 0
    
    @pytest.mark.asyncio
    async def test_timezone_behavior_adaptation(self, anti_ban_system):
        """Test timezone-appropriate behavior"""
        timezones = ["US/Eastern", "Europe/London", "Asia/Tokyo"]
        
        for timezone in timezones:
            behavior = await anti_ban_system.adapt_to_timezone(timezone)
            
            # Should have appropriate active hours for the timezone
            active_hours = behavior["preferred_active_hours"]
            assert len(active_hours) >= 8  # At least 8 active hours
            assert len(active_hours) <= 16  # At most 16 active hours
            
            # Should have sleep period
            sleep_hours = behavior["sleep_hours"]
            assert len(sleep_hours) >= 6  # At least 6 hours sleep
            
            # Active and sleep hours shouldn't overlap
            assert not set(active_hours).intersection(set(sleep_hours))


class TestBehaviorPatternAnalysis:
    """Test behavior pattern analysis and scoring"""
    
    @pytest.fixture
    def temporal_archaeology(self):
        """Mock temporal archaeology service"""
        mock_temporal = Mock(spec=TemporalArchaeology)
        mock_temporal.analyze_user_patterns = AsyncMock(return_value={
            "activity_peaks": [9, 12, 18, 21],  # 9am, 12pm, 6pm, 9pm
            "response_time_distribution": {"mean": 45, "std": 30},
            "message_length_distribution": {"mean": 25, "std": 15}
        })
        return mock_temporal
    
    @pytest.mark.asyncio
    async def test_behavior_pattern_scoring(self, temporal_archaeology):
        """Test behavior pattern analysis and scoring"""
        anti_ban = AntiBanSystem(temporal_archaeology=temporal_archaeology)
        
        # Sample activity data
        activities = [
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=i * random.randint(5, 45)),
                "activity_type": random.choice(["message", "read", "typing", "online"]),
                "duration": random.randint(1, 120),
                "metadata": {
                    "message_length": random.randint(5, 200) if random.random() > 0.3 else None
                }
            }
            for i in range(50)
        ]
        
        score = await anti_ban.analyze_behavior_naturalness(activities)
        
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
        
        # Verify temporal analysis was used
        temporal_archaeology.analyze_user_patterns.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_activity_clustering_detection(self):
        """Test detection of suspicious activity clustering"""
        anti_ban = AntiBanSystem()
        
        # Clustered activities (suspicious)
        clustered_activities = [
            {
                "timestamp": datetime.utcnow() - timedelta(seconds=i * 5),  # Every 5 seconds
                "activity_type": "message"
            }
            for i in range(20)
        ]
        
        clustering_score = await anti_ban.detect_activity_clustering(clustered_activities)
        assert clustering_score > 0.7  # High clustering detected
        
        # Distributed activities (natural)
        distributed_activities = [
            {
                "timestamp": datetime.utcnow() - timedelta(seconds=i * random.randint(60, 600)),
                "activity_type": random.choice(["message", "read", "online"])
            }
            for i in range(20)
        ]
        
        distribution_score = await anti_ban.detect_activity_clustering(distributed_activities)
        assert distribution_score < 0.4  # Low clustering detected
    
    @pytest.mark.asyncio
    async def test_vocabulary_pattern_analysis(self):
        """Test vocabulary and language pattern analysis"""
        anti_ban = AntiBanSystem()
        
        # Bot-like vocabulary (repetitive, formal)
        bot_messages = [
            "Thank you for your message.",
            "I understand your concern.",
            "Please let me know if you need assistance.",
            "I appreciate your feedback.",
        ] * 5  # Repeat to show pattern
        
        bot_vocab_score = await anti_ban.analyze_vocabulary_patterns(bot_messages)
        assert bot_vocab_score > 0.6  # Should detect bot-like patterns
        
        # Human-like vocabulary (varied, casual)
        human_messages = [
            "hey what's up?",
            "lol that's so funny 😂",
            "idk, maybe we should check it out",
            "omg really?? that's crazy!",
            "yeah I think so too",
            "nah not really my thing",
            "sounds good to me 👍",
            "wait what happened?",
            "nice! congrats dude",
            "ugh this is so annoying"
        ]
        
        human_vocab_score = await anti_ban.analyze_vocabulary_patterns(human_messages)
        assert human_vocab_score < 0.4  # Should detect human-like patterns
    
    @pytest.mark.asyncio
    async def test_response_pattern_analysis(self):
        """Test response timing and pattern analysis"""
        anti_ban = AntiBanSystem()
        
        # Bot-like responses (too consistent)
        bot_responses = [
            {"delay": 5.0, "message_length": 50},
            {"delay": 5.2, "message_length": 52},
            {"delay": 4.8, "message_length": 48},
            {"delay": 5.1, "message_length": 51},
        ] * 10  # Very consistent pattern
        
        bot_pattern_score = await anti_ban.analyze_response_patterns(bot_responses)
        assert bot_pattern_score > 0.7  # Should detect bot-like consistency
        
        # Human-like responses (varied)
        human_responses = [
            {"delay": random.uniform(2, 60), "message_length": random.randint(5, 200)}
            for _ in range(40)
        ]
        
        human_pattern_score = await anti_ban.analyze_response_patterns(human_responses)
        assert human_pattern_score < 0.5  # Should detect human-like variation


class TestCoolingAndRecovery:
    """Test cooling period and recovery mechanisms"""
    
    @pytest.fixture
    def anti_ban_system(self):
        return AntiBanSystem(
            detection_threshold=0.7,
            cooling_period=300,  # 5 minutes
            recovery_period=900  # 15 minutes
        )
    
    @pytest.mark.asyncio
    async def test_cooling_period_activation(self, anti_ban_system):
        """Test cooling period activation"""
        account_id = "test_account_123"
        
        # Trigger cooling period
        await anti_ban_system.activate_cooling_period(
            account_id,
            reason="High bot detection score",
            severity="high"
        )
        
        # Should be in cooling period
        is_cooling = await anti_ban_system.is_in_cooling_period(account_id)
        assert is_cooling is True
        
        # Should restrict activities
        activity_allowed = await anti_ban_system.is_activity_allowed(account_id, "message")
        assert activity_allowed is False
    
    @pytest.mark.asyncio
    async def test_cooling_period_expiry(self, anti_ban_system):
        """Test cooling period expiry"""
        account_id = "test_account_456"
        
        # Manually set expired cooling period
        await anti_ban_system._set_cooling_start_time(
            account_id, 
            datetime.utcnow() - timedelta(seconds=400)  # 400 seconds ago (> 300)
        )
        
        # Should not be in cooling period anymore
        is_cooling = await anti_ban_system.is_in_cooling_period(account_id)
        assert is_cooling is False
        
        # Activities should be allowed again
        activity_allowed = await anti_ban_system.is_activity_allowed(account_id, "message")
        assert activity_allowed is True
    
    @pytest.mark.asyncio
    async def test_graduated_activity_recovery(self, anti_ban_system):
        """Test graduated activity recovery after cooling"""
        account_id = "test_account_789"
        
        # Complete cooling period
        await anti_ban_system._set_cooling_start_time(
            account_id,
            datetime.utcnow() - timedelta(seconds=400)
        )
        
        # Should start with limited activity
        activity_level = await anti_ban_system.get_recommended_activity_level(account_id)
        assert activity_level in ["minimal", "limited"]
        
        # Simulate time passage for recovery
        await anti_ban_system._set_cooling_start_time(
            account_id,
            datetime.utcnow() - timedelta(seconds=1000)  # Long time passed
        )
        
        # Should allow more activity
        activity_level = await anti_ban_system.get_recommended_activity_level(account_id)
        assert activity_level in ["moderate", "normal"]
    
    @pytest.mark.asyncio
    async def test_risk_based_cooling_duration(self, anti_ban_system):
        """Test cooling duration based on risk level"""
        high_risk_account = "high_risk_account"
        low_risk_account = "low_risk_account"
        
        # High risk should have longer cooling
        await anti_ban_system.activate_cooling_period(
            high_risk_account,
            reason="Multiple violations",
            severity="critical"
        )
        
        high_risk_duration = await anti_ban_system.get_cooling_duration(high_risk_account)
        
        # Low risk should have shorter cooling
        await anti_ban_system.activate_cooling_period(
            low_risk_account,
            reason="Minor pattern detected",
            severity="low"
        )
        
        low_risk_duration = await anti_ban_system.get_cooling_duration(low_risk_account)
        
        # High risk should have longer duration
        assert high_risk_duration > low_risk_duration
        assert high_risk_duration >= 300  # At least base cooling period
        assert low_risk_duration >= 60   # At least 1 minute


class TestRealWorldScenarios:
    """Test real-world anti-detection scenarios"""
    
    @pytest.mark.asyncio
    async def test_conversation_flow_naturalness(self):
        """Test natural conversation flow patterns"""
        anti_ban = AntiBanSystem()
        
        # Simulate natural conversation
        conversation_events = [
            {"type": "user_message", "timestamp": datetime.utcnow() - timedelta(seconds=300)},
            {"type": "typing_start", "timestamp": datetime.utcnow() - timedelta(seconds=285)},
            {"type": "typing_stop", "timestamp": datetime.utcnow() - timedelta(seconds=275)},
            {"type": "bot_response", "timestamp": datetime.utcnow() - timedelta(seconds=270)},
            {"type": "user_seen", "timestamp": datetime.utcnow() - timedelta(seconds=240)},
            {"type": "user_message", "timestamp": datetime.utcnow() - timedelta(seconds=180)},
            {"type": "typing_start", "timestamp": datetime.utcnow() - timedelta(seconds=150)},
            {"type": "bot_response", "timestamp": datetime.utcnow() - timedelta(seconds=130)},
        ]
        
        naturalness_score = await anti_ban.analyze_conversation_naturalness(conversation_events)
        
        # Should detect as natural conversation
        assert naturalness_score > 0.6
        assert naturalness_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_multi_community_behavior_consistency(self):
        """Test consistent behavior across multiple communities"""
        anti_ban = AntiBanSystem()
        
        # Behavior in different communities should be consistent but not identical
        crypto_community_behavior = await anti_ban.generate_community_adapted_behavior(
            community_type="crypto",
            community_culture="technical",
            user_role="contributor"
        )
        
        general_community_behavior = await anti_ban.generate_community_adapted_behavior(
            community_type="general",
            community_culture="casual",
            user_role="participant"
        )
        
        # Should have some consistency (same base personality)
        assert crypto_community_behavior["base_personality"] == general_community_behavior["base_personality"]
        
        # But adapted communication style
        assert crypto_community_behavior["communication_style"] != general_community_behavior["communication_style"]
        assert crypto_community_behavior["vocabulary_preference"] != general_community_behavior["vocabulary_preference"]
    
    @pytest.mark.asyncio
    async def test_long_term_behavior_evolution(self):
        """Test long-term behavior evolution to avoid detection"""
        anti_ban = AntiBanSystem()
        
        # Initial behavior profile
        initial_behavior = await anti_ban.generate_behavior_profile(
            account_age_days=1,
            experience_level="beginner"
        )
        
        # Evolved behavior after time
        evolved_behavior = await anti_ban.generate_behavior_profile(
            account_age_days=90,
            experience_level="experienced"
        )
        
        # Should show evolution
        assert evolved_behavior["confidence_level"] > initial_behavior["confidence_level"]
        assert evolved_behavior["vocabulary_complexity"] > initial_behavior["vocabulary_complexity"]
        assert evolved_behavior["interaction_frequency"] > initial_behavior["interaction_frequency"]
    
    @pytest.mark.parametrize("activity_type,expected_detection_score", [
        ("human_casual", 0.2),        # Very human-like
        ("human_engaged", 0.3),       # Engaged human
        ("slightly_robotic", 0.6),    # Borderline suspicious
        ("clearly_bot", 0.9),         # Obviously bot
    ])
    @pytest.mark.asyncio
    async def test_detection_score_calibration(self, activity_type, expected_detection_score):
        """Test detection score calibration for different behavior types"""
        anti_ban = AntiBanSystem()
        
        # Generate activity patterns for each type
        if activity_type == "human_casual":
            activities = await anti_ban._generate_human_casual_pattern()
        elif activity_type == "human_engaged": 
            activities = await anti_ban._generate_human_engaged_pattern()
        elif activity_type == "slightly_robotic":
            activities = await anti_ban._generate_slightly_robotic_pattern()
        else:  # clearly_bot
            activities = await anti_ban._generate_clearly_bot_pattern()
        
        detection_score = await anti_ban.calculate_detection_probability(activities)
        
        # Should be within expected range (±0.2)
        assert abs(detection_score - expected_detection_score) < 0.2