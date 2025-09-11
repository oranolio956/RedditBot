"""
Unit Tests for TelegramAccount Model

Tests for the TelegramAccount SQLAlchemy model including:
- Model creation and validation
- Property methods and business logic
- Safety mechanisms and limits
- Risk score calculations
- Daily counter management
- Account health checks
"""

import pytest
from datetime import datetime, timedelta
from app.models.telegram_account import (
    TelegramAccount, 
    AccountStatus, 
    SafetyLevel,
    AccountSafetyEvent,
    AccountConfiguration
)


class TestTelegramAccountModel:
    """Test TelegramAccount model functionality"""
    
    def test_account_creation_with_defaults(self):
        """Test creating account with default values"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test"
        )
        
        assert account.phone_number == "+12345678901"
        assert account.first_name == "Test"
        assert account.status == AccountStatus.INACTIVE
        assert account.safety_level == SafetyLevel.CONSERVATIVE
        assert account.is_ai_disclosed is True
        assert account.warming_progress == 0.0
        assert account.risk_score == 0.0
        assert account.messages_sent_today == 0
        assert account.groups_joined_today == 0
        assert account.dms_sent_today == 0
        assert account.max_messages_per_day == 50
        assert account.max_groups_per_day == 2
        assert account.max_dms_per_day == 5
    
    def test_account_creation_with_full_data(self):
        """Test creating account with full configuration"""
        personality = {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.6
        }
        
        communication = {
            "formality": "casual",
            "enthusiasm": "high"
        }
        
        account = TelegramAccount(
            phone_number="+12345678901",
            telegram_id=123456789,
            username="testuser",
            first_name="Test",
            last_name="User",
            bio="Test bio",
            status=AccountStatus.ACTIVE,
            safety_level=SafetyLevel.MODERATE,
            personality_profile=personality,
            communication_style=communication,
            interests=["crypto", "AI"]
        )
        
        assert account.telegram_id == 123456789
        assert account.username == "testuser"
        assert account.status == AccountStatus.ACTIVE
        assert account.safety_level == SafetyLevel.MODERATE
        assert account.personality_profile == personality
        assert account.communication_style == communication
        assert account.interests == ["crypto", "AI"]
    
    def test_is_healthy_property(self):
        """Test is_healthy property logic"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            status=AccountStatus.ACTIVE,
            risk_score=30.0,
            spam_warnings=1
        )
        
        # Healthy account
        assert account.is_healthy is True
        
        # High risk score
        account.risk_score = 60.0
        assert account.is_healthy is False
        
        # Reset risk, add spam warnings
        account.risk_score = 30.0
        account.spam_warnings = 5
        assert account.is_healthy is False
        
        # Inactive status
        account.spam_warnings = 1
        account.status = AccountStatus.SUSPENDED
        assert account.is_healthy is False
        
        # Warming up is still healthy
        account.status = AccountStatus.WARMING_UP
        assert account.is_healthy is True
    
    def test_daily_limits_reached_property(self):
        """Test daily_limits_reached property"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            messages_sent_today=25,
            groups_joined_today=1,
            dms_sent_today=2,
            max_messages_per_day=50,
            max_groups_per_day=2,
            max_dms_per_day=5,
            last_activity_reset=datetime.utcnow()
        )
        
        # Within limits
        assert account.daily_limits_reached is False
        
        # Messages limit reached
        account.messages_sent_today = 50
        assert account.daily_limits_reached is True
        
        # Reset messages, check groups
        account.messages_sent_today = 25
        account.groups_joined_today = 2
        assert account.daily_limits_reached is True
        
        # Reset groups, check DMs
        account.groups_joined_today = 1
        account.dms_sent_today = 5
        assert account.daily_limits_reached is True
        
        # Old reset time should return False (needs reset)
        account.dms_sent_today = 2
        account.last_activity_reset = datetime.utcnow() - timedelta(days=1)
        assert account.daily_limits_reached is False
    
    def test_warming_completed_property(self):
        """Test warming_completed property"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test"
        )
        
        # Not completed
        assert account.warming_completed is False
        
        # Progress but no completion time
        account.warming_progress = 100.0
        assert account.warming_completed is False
        
        # Completion time but not full progress
        account.warming_progress = 80.0
        account.warming_completed_at = datetime.utcnow()
        assert account.warming_completed is False
        
        # Both conditions met
        account.warming_progress = 100.0
        assert account.warming_completed is True
    
    def test_reset_daily_counters(self):
        """Test daily counter reset functionality"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            messages_sent_today=25,
            groups_joined_today=2,
            dms_sent_today=3,
            last_activity_reset=datetime.utcnow() - timedelta(hours=12)
        )
        
        old_reset_time = account.last_activity_reset
        
        account.reset_daily_counters()
        
        assert account.messages_sent_today == 0
        assert account.groups_joined_today == 0
        assert account.dms_sent_today == 0
        assert account.last_activity_reset > old_reset_time
    
    def test_increment_risk_score(self):
        """Test risk score increment with safety event creation"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            risk_score=20.0
        )
        
        # Normal increment
        safety_event = account.increment_risk_score(15.0, "Flood wait detected")
        
        assert account.risk_score == 35.0
        assert isinstance(safety_event, AccountSafetyEvent)
        assert safety_event.event_type == "risk_increase"
        assert safety_event.severity == "high"
        assert "15.0 points" in safety_event.description
        assert safety_event.data["points"] == 15.0
        assert safety_event.data["new_score"] == 35.0
        
        # Cap at 100
        account.increment_risk_score(80.0, "Major violation")
        assert account.risk_score == 100.0
        
        # Small increment
        account.risk_score = 10.0
        safety_event = account.increment_risk_score(5.0, "Minor issue")
        assert safety_event.severity == "medium"


class TestAccountSafetyEvent:
    """Test AccountSafetyEvent model"""
    
    def test_safety_event_creation(self):
        """Test creating safety events"""
        account_id = "123e4567-e89b-12d3-a456-426614174000"
        
        event = AccountSafetyEvent(
            account_id=account_id,
            event_type="flood_wait",
            severity="high",
            description="Flood wait of 60 seconds",
            data={"wait_time": 60},
            telegram_error_code=420,
            telegram_error_message="Too Many Requests"
        )
        
        assert event.account_id == account_id
        assert event.event_type == "flood_wait"
        assert event.severity == "high"
        assert event.resolved is False
        assert event.data["wait_time"] == 60
        assert event.telegram_error_code == 420
    
    def test_safety_event_resolution(self):
        """Test marking safety event as resolved"""
        event = AccountSafetyEvent(
            account_id="123e4567-e89b-12d3-a456-426614174000",
            event_type="spam_warning",
            severity="medium",
            description="Spam detection triggered"
        )
        
        assert event.resolved is False
        assert event.resolved_at is None
        
        # Simulate resolution
        event.resolved = True
        event.resolved_at = datetime.utcnow()
        event.action_taken = "Reduced activity rate"
        
        assert event.resolved is True
        assert event.resolved_at is not None
        assert event.action_taken == "Reduced activity rate"


class TestAccountConfiguration:
    """Test AccountConfiguration model"""
    
    def test_configuration_defaults(self):
        """Test configuration with default values"""
        account_id = "123e4567-e89b-12d3-a456-426614174000"
        
        config = AccountConfiguration(account_id=account_id)
        
        assert config.account_id == account_id
        assert config.min_response_delay == 2
        assert config.max_response_delay == 30
        assert config.typing_speed == 0.05
        assert config.response_probability == 0.3
        assert config.proactive_messaging is False
        assert config.engagement_style == "helpful"
        assert config.use_random_delays is True
        assert config.simulate_typing is True
        assert config.llm_provider == "anthropic"
        assert config.model_name == "claude-3-sonnet"
        assert config.temperature == 0.7
        assert config.max_tokens == 150
    
    def test_configuration_custom_values(self):
        """Test configuration with custom values"""
        account_id = "123e4567-e89b-12d3-a456-426614174000"
        
        config = AccountConfiguration(
            account_id=account_id,
            min_response_delay=5,
            max_response_delay=60,
            typing_speed=0.08,
            response_probability=0.5,
            proactive_messaging=True,
            engagement_style="professional",
            use_random_delays=False,
            simulate_typing=False,
            blocked_keywords=["spam", "scam"],
            required_keywords=["crypto", "DeFi"],
            llm_provider="openai",
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=200
        )
        
        assert config.min_response_delay == 5
        assert config.max_response_delay == 60
        assert config.typing_speed == 0.08
        assert config.response_probability == 0.5
        assert config.proactive_messaging is True
        assert config.engagement_style == "professional"
        assert config.use_random_delays is False
        assert config.simulate_typing is False
        assert config.blocked_keywords == ["spam", "scam"]
        assert config.required_keywords == ["crypto", "DeFi"]
        assert config.llm_provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 200


class TestAccountBusinessLogic:
    """Test business logic and edge cases"""
    
    def test_account_status_transitions(self):
        """Test valid account status transitions"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test"
        )
        
        # Start inactive
        assert account.status == AccountStatus.INACTIVE
        
        # Can move to warming up
        account.status = AccountStatus.WARMING_UP
        assert account.status == AccountStatus.WARMING_UP
        
        # Can move to active
        account.status = AccountStatus.ACTIVE
        assert account.status == AccountStatus.ACTIVE
        
        # Can be limited
        account.status = AccountStatus.LIMITED
        assert account.status == AccountStatus.LIMITED
        
        # Can be suspended
        account.status = AccountStatus.SUSPENDED
        assert account.status == AccountStatus.SUSPENDED
        
        # Can be banned
        account.status = AccountStatus.BANNED
        assert account.status == AccountStatus.BANNED
    
    def test_safety_level_impact(self):
        """Test how safety level affects limits"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test"
        )
        
        # Conservative safety (default limits)
        assert account.safety_level == SafetyLevel.CONSERVATIVE
        assert account.max_messages_per_day == 50
        
        # Test with different safety levels
        account.safety_level = SafetyLevel.MODERATE
        # Note: In real implementation, this might adjust limits automatically
        
        account.safety_level = SafetyLevel.AGGRESSIVE
        # Note: In real implementation, this might allow higher limits
    
    def test_personality_profile_validation(self):
        """Test personality profile structure"""
        valid_personality = {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.3
        }
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            personality_profile=valid_personality
        )
        
        assert account.personality_profile == valid_personality
        
        # Test personality traits are within expected range
        for trait, value in valid_personality.items():
            assert 0.0 <= value <= 1.0
    
    def test_communication_style_validation(self):
        """Test communication style structure"""
        valid_style = {
            "formality": "casual",
            "enthusiasm": "moderate",
            "verbosity": "concise",
            "humor": "light",
            "emoji_usage": "minimal"
        }
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            communication_style=valid_style
        )
        
        assert account.communication_style == valid_style
    
    def test_risk_score_boundaries(self):
        """Test risk score boundary conditions"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test"
        )
        
        # Start at 0
        assert account.risk_score == 0.0
        
        # Increment to boundary
        account.increment_risk_score(100.0, "Max risk test")
        assert account.risk_score == 100.0
        
        # Try to exceed boundary
        account.increment_risk_score(50.0, "Over max test")
        assert account.risk_score == 100.0  # Should stay at max
    
    def test_account_repr(self):
        """Test string representation"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            status=AccountStatus.ACTIVE
        )
        
        repr_str = repr(account)
        assert "+12345678901" in repr_str
        assert "active" in repr_str.lower()
    
    @pytest.mark.parametrize("messages,groups,dms,expected", [
        (0, 0, 0, False),      # No activity
        (49, 1, 4, False),     # Under all limits
        (50, 1, 4, True),      # Messages at limit
        (49, 2, 4, True),      # Groups at limit
        (49, 1, 5, True),      # DMs at limit
        (50, 2, 5, True),      # All at limit
    ])
    def test_daily_limits_combinations(self, messages, groups, dms, expected):
        """Test various combinations of daily limits"""
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Test",
            messages_sent_today=messages,
            groups_joined_today=groups,
            dms_sent_today=dms,
            last_activity_reset=datetime.utcnow()
        )
        
        assert account.daily_limits_reached == expected