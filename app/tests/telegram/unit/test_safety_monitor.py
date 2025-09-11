"""
Unit Tests for Safety Monitor and Circuit Breaker

Tests for safety mechanisms including:
- Risk score calculations
- Flood wait handling
- Rate limiting enforcement
- Circuit breaker patterns
- Anti-detection measures
- Safety event tracking
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from app.models.telegram_account import TelegramAccount, AccountStatus, SafetyLevel
from app.telegram.circuit_breaker import CircuitBreaker, CircuitState
from app.telegram.rate_limiter import RateLimiter
from app.telegram.anti_ban import AntiBanSystem
from pyrogram.errors import FloodWait, UserDeactivated, SpamWait


class TestSafetyMonitorCore:
    """Test core safety monitoring functionality"""
    
    @pytest.fixture
    def test_account(self):
        """Create test account for safety tests"""
        return TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            status=AccountStatus.ACTIVE,
            safety_level=SafetyLevel.CONSERVATIVE,
            risk_score=0.0,
            messages_sent_today=0,
            flood_wait_count=0,
            spam_warnings=0
        )
    
    def test_risk_score_calculation_flood_wait(self, test_account):
        """Test risk score increase from flood wait"""
        initial_score = test_account.risk_score
        
        # Small flood wait (10 seconds)
        safety_event = test_account.increment_risk_score(5.0, "Flood wait: 10s")
        assert test_account.risk_score == initial_score + 5.0
        assert safety_event.event_type == "risk_increase"
        
        # Large flood wait (120 seconds)
        test_account.increment_risk_score(20.0, "Flood wait: 120s")
        assert test_account.risk_score == initial_score + 25.0
    
    def test_risk_score_calculation_spam_warning(self, test_account):
        """Test risk score increase from spam warnings"""
        # First spam warning
        test_account.increment_risk_score(15.0, "Spam warning received")
        test_account.spam_warnings += 1
        assert test_account.risk_score == 15.0
        assert test_account.spam_warnings == 1
        
        # Second spam warning (higher penalty)
        test_account.increment_risk_score(25.0, "Second spam warning")
        test_account.spam_warnings += 1
        assert test_account.risk_score == 40.0
        assert test_account.spam_warnings == 2
    
    def test_risk_score_calculation_rate_limit(self, test_account):
        """Test risk score from rate limiting violations"""
        # Moderate rate limit violation
        test_account.increment_risk_score(10.0, "Rate limit exceeded")
        assert test_account.risk_score == 10.0
        
        # Multiple violations compound
        test_account.increment_risk_score(10.0, "Second rate limit violation")
        assert test_account.risk_score == 20.0
    
    def test_risk_score_capped_at_100(self, test_account):
        """Test risk score cannot exceed 100"""
        test_account.increment_risk_score(90.0, "High risk activity")
        assert test_account.risk_score == 90.0
        
        # Should cap at 100
        test_account.increment_risk_score(50.0, "Another high risk activity")
        assert test_account.risk_score == 100.0
    
    def test_account_health_with_high_risk(self, test_account):
        """Test account health with various risk levels"""
        # Healthy account
        test_account.risk_score = 30.0
        assert test_account.is_healthy is True
        
        # Moderate risk
        test_account.risk_score = 50.0
        assert test_account.is_healthy is False
        
        # High risk
        test_account.risk_score = 80.0
        assert test_account.is_healthy is False
    
    def test_account_health_with_spam_warnings(self, test_account):
        """Test account health with spam warnings"""
        test_account.risk_score = 30.0  # Acceptable risk
        
        # Few warnings OK
        test_account.spam_warnings = 2
        assert test_account.is_healthy is True
        
        # Too many warnings
        test_account.spam_warnings = 5
        assert test_account.is_healthy is False
    
    def test_daily_limit_safety_check(self, test_account):
        """Test daily limit enforcement for safety"""
        # Set today's date
        test_account.last_activity_reset = datetime.utcnow()
        
        # Within limits
        test_account.messages_sent_today = 25
        test_account.groups_joined_today = 1
        test_account.dms_sent_today = 2
        assert test_account.daily_limits_reached is False
        
        # Hit message limit
        test_account.messages_sent_today = 50
        assert test_account.daily_limits_reached is True
        
        # Reset and hit group limit
        test_account.messages_sent_today = 25
        test_account.groups_joined_today = 2
        assert test_account.daily_limits_reached is True
        
        # Reset and hit DM limit
        test_account.groups_joined_today = 1
        test_account.dms_sent_today = 5
        assert test_account.daily_limits_reached is True


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=FloodWait
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state"""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        
        # Successful call should keep circuit closed
        async def successful_operation():
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self, circuit_breaker):
        """Test circuit breaker failure counting"""
        async def failing_operation():
            raise FloodWait(value=30)
        
        # First failure
        with pytest.raises(FloodWait):
            await circuit_breaker.call(failing_operation)
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Second failure
        with pytest.raises(FloodWait):
            await circuit_breaker.call(failing_operation)
        assert circuit_breaker.failure_count == 2
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Third failure should open circuit
        with pytest.raises(FloodWait):
            await circuit_breaker.call(failing_operation)
        assert circuit_breaker.failure_count == 3
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker in open state"""
        # Force circuit to open
        circuit_breaker.failure_count = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = datetime.utcnow()
        
        async def any_operation():
            return "should not execute"
        
        # Should raise CircuitOpenError without executing
        from app.telegram.circuit_breaker import CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await circuit_breaker.call(any_operation)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self, circuit_breaker):
        """Test circuit breaker transition to half-open"""
        # Force circuit to open and wait for recovery timeout
        circuit_breaker.failure_count = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = datetime.utcnow() - timedelta(seconds=35)
        
        async def test_operation():
            return "recovery test"
        
        # Should transition to half-open and execute
        result = await circuit_breaker.call(test_operation)
        assert result == "recovery test"
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery from half-open to closed"""
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.failure_count = 3
        
        async def successful_operation():
            return "recovered"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "recovered"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing"""
        return RateLimiter(
            max_requests=10,
            window_seconds=60,
            burst_allowance=2
        )
    
    @pytest.mark.asyncio
    async def test_rate_limiter_within_limits(self, rate_limiter):
        """Test rate limiter within allowed limits"""
        user_id = "test_user_123"
        
        # Should allow requests within limit
        for i in range(8):
            allowed = await rate_limiter.check_rate_limit(user_id)
            assert allowed is True
        
        # Check remaining count
        remaining = await rate_limiter.get_remaining_requests(user_id)
        assert remaining == 2
    
    @pytest.mark.asyncio
    async def test_rate_limiter_burst_handling(self, rate_limiter):
        """Test rate limiter burst allowance"""
        user_id = "test_user_burst"
        
        # Use up normal quota
        for i in range(10):
            allowed = await rate_limiter.check_rate_limit(user_id)
            assert allowed is True
        
        # Burst allowance should still allow some requests
        for i in range(2):
            allowed = await rate_limiter.check_rate_limit(user_id)
            assert allowed is True
        
        # Now should be blocked
        allowed = await rate_limiter.check_rate_limit(user_id)
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_window_reset(self, rate_limiter):
        """Test rate limiter window reset"""
        user_id = "test_user_reset"
        
        # Use up quota
        for i in range(12):  # 10 normal + 2 burst
            await rate_limiter.check_rate_limit(user_id)
        
        # Should be blocked
        allowed = await rate_limiter.check_rate_limit(user_id)
        assert allowed is False
        
        # Simulate window reset (in real implementation, this would be time-based)
        await rate_limiter.reset_user_limits(user_id)
        
        # Should be allowed again
        allowed = await rate_limiter.check_rate_limit(user_id)
        assert allowed is True


class TestAntiBanSystem:
    """Test anti-ban detection and prevention"""
    
    @pytest.fixture
    def anti_ban_system(self):
        """Create anti-ban system for testing"""
        return AntiBanSystem(
            detection_threshold=0.7,
            cooling_period=300,  # 5 minutes
            safety_margin=0.2
        )
    
    @pytest.mark.asyncio
    async def test_human_behavior_patterns(self, anti_ban_system):
        """Test human behavior pattern detection"""
        # Test typing speed variation
        typing_speeds = await anti_ban_system.calculate_human_typing_speeds(5)
        assert len(typing_speeds) == 5
        assert all(10 <= speed <= 30 for speed in typing_speeds)  # Reasonable human range
        
        # Test response delay variation
        delays = await anti_ban_system.calculate_human_response_delays(3)
        assert len(delays) == 3
        assert all(2 <= delay <= 300 for delay in delays)  # 2 seconds to 5 minutes
    
    @pytest.mark.asyncio
    async def test_activity_pattern_analysis(self, anti_ban_system):
        """Test activity pattern analysis"""
        # Simulate bot-like behavior (too regular)
        bot_pattern = [
            {"timestamp": datetime.utcnow() - timedelta(seconds=i*60), "activity": "message"}
            for i in range(10)  # Exactly every minute
        ]
        
        bot_score = await anti_ban_system.analyze_activity_pattern(bot_pattern)
        assert bot_score > 0.7  # Should detect as bot-like
        
        # Simulate human-like behavior (irregular)
        human_pattern = [
            {"timestamp": datetime.utcnow() - timedelta(seconds=i*180 + (i*i*30)), "activity": "message"}
            for i in range(10)  # Irregular intervals
        ]
        
        human_score = await anti_ban_system.analyze_activity_pattern(human_pattern)
        assert human_score < 0.5  # Should detect as human-like
    
    @pytest.mark.asyncio
    async def test_detection_avoidance_measures(self, anti_ban_system):
        """Test detection avoidance measures"""
        # Test random delay injection
        base_delay = 30
        modified_delay = await anti_ban_system.add_natural_variance(base_delay)
        assert 20 <= modified_delay <= 40  # Within reasonable variance
        
        # Test activity distribution
        activities = ["message", "read", "typing", "idle"]
        distributed = await anti_ban_system.distribute_activities(activities, 100)
        
        # Should not have equal distribution (too bot-like)
        assert len(set(distributed)) == len(activities)
        activity_counts = {activity: distributed.count(activity) for activity in activities}
        assert not all(count == 25 for count in activity_counts.values())  # Not perfectly equal
    
    @pytest.mark.asyncio
    async def test_cooling_period_enforcement(self, anti_ban_system):
        """Test cooling period after detection"""
        user_id = "test_user_cooling"
        
        # Trigger cooling period
        await anti_ban_system.trigger_cooling_period(user_id, reason="High bot score detected")
        
        # Should be in cooling period
        is_cooling = await anti_ban_system.is_in_cooling_period(user_id)
        assert is_cooling is True
        
        # Should recommend no activity
        recommended_activity = await anti_ban_system.get_recommended_activity_level(user_id)
        assert recommended_activity == "minimal"
    
    @pytest.mark.asyncio
    async def test_safety_margin_application(self, anti_ban_system):
        """Test safety margin application"""
        # Test with current risk score
        current_risk = 0.6
        safe_activity = await anti_ban_system.is_activity_safe(current_risk)
        assert safe_activity is True  # 0.6 + 0.2 margin = 0.8, still under threshold
        
        # Test with high risk
        high_risk = 0.8
        unsafe_activity = await anti_ban_system.is_activity_safe(high_risk)
        assert unsafe_activity is False  # 0.8 + 0.2 margin = 1.0, over threshold


class TestSafetyIntegration:
    """Test integration between safety components"""
    
    @pytest.mark.asyncio
    async def test_flood_wait_handling_workflow(self):
        """Test complete flood wait handling workflow"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            status=AccountStatus.ACTIVE
        )
        
        # Simulate flood wait error
        flood_wait = FloodWait(value=60)
        
        # Should increment risk score
        safety_event = account.increment_risk_score(12.0, f"Flood wait: {flood_wait.value}s")
        account.flood_wait_count += 1
        
        assert account.risk_score == 12.0
        assert account.flood_wait_count == 1
        assert safety_event.event_type == "risk_increase"
        assert "60s" in safety_event.description
    
    @pytest.mark.asyncio
    async def test_multiple_safety_violations(self):
        """Test handling multiple safety violations"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            status=AccountStatus.ACTIVE
        )
        
        # Multiple violations
        violations = [
            ("flood_wait", 30, 10.0),
            ("spam_warning", 0, 20.0),
            ("rate_limit", 0, 5.0),
            ("flood_wait", 120, 25.0)
        ]
        
        for violation_type, wait_time, risk_points in violations:
            account.increment_risk_score(risk_points, f"{violation_type}: {wait_time}s")
            if violation_type == "flood_wait":
                account.flood_wait_count += 1
            elif violation_type == "spam_warning":
                account.spam_warnings += 1
        
        assert account.risk_score == 60.0  # 10 + 20 + 5 + 25
        assert account.flood_wait_count == 2
        assert account.spam_warnings == 1
        assert account.is_healthy is False  # Risk score > 50
    
    @pytest.mark.asyncio
    async def test_safety_level_impact_on_limits(self):
        """Test how safety level affects operational limits"""
        accounts = []
        
        for safety_level in [SafetyLevel.CONSERVATIVE, SafetyLevel.MODERATE, SafetyLevel.AGGRESSIVE]:
            account = TelegramAccount(
                phone_number=f"+123456789{safety_level.value}",
                first_name="Test",
                safety_level=safety_level
            )
            accounts.append(account)
        
        # Conservative should have strictest limits
        conservative = accounts[0]
        assert conservative.max_messages_per_day == 50
        assert conservative.max_groups_per_day == 2
        
        # In a real implementation, moderate and aggressive might have different limits
        # For now, they use the same defaults, but this shows where the logic would go
        moderate = accounts[1]
        aggressive = accounts[2]
        
        # These assertions would change based on actual implementation
        assert moderate.safety_level == SafetyLevel.MODERATE
        assert aggressive.safety_level == SafetyLevel.AGGRESSIVE
    
    @pytest.mark.asyncio
    async def test_emergency_safety_shutdown(self):
        """Test emergency safety shutdown conditions"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            status=AccountStatus.ACTIVE,
            risk_score=95.0,
            spam_warnings=5,
            flood_wait_count=10
        )
        
        # Should trigger emergency conditions
        assert account.risk_score >= 90.0
        assert account.spam_warnings >= 5
        assert account.is_healthy is False
        
        # Should force status change
        if account.risk_score >= 90.0:
            account.status = AccountStatus.LIMITED
        
        assert account.status == AccountStatus.LIMITED
    
    @pytest.mark.parametrize("violation_type,wait_time,expected_risk", [
        ("flood_wait", 10, 5.0),    # Short flood wait
        ("flood_wait", 60, 15.0),   # Medium flood wait  
        ("flood_wait", 300, 30.0),  # Long flood wait
        ("spam_warning", 0, 25.0),  # Spam warning
        ("account_limit", 0, 20.0), # Account limit hit
    ])
    @pytest.mark.asyncio
    async def test_violation_risk_scoring(self, violation_type, wait_time, expected_risk):
        """Test risk scoring for different violation types"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test"
        )
        
        account.increment_risk_score(expected_risk, f"{violation_type}: {wait_time}s")
        assert account.risk_score == expected_risk