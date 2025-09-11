"""
Telegram-specific test fixtures and configuration
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, Mock, MagicMock
import pytest
import pytest_asyncio
from pyrogram import types
from pyrogram.errors import FloodWait, UserDeactivated

from app.models.telegram_account import TelegramAccount, AccountStatus, SafetyLevel, AccountSafetyEvent
from app.models.telegram_community import TelegramCommunity, CommunityStatus, EngagementStrategy
from app.models.telegram_conversation import TelegramConversation, ConversationMessage, MessageDirection
from app.services.telegram_account_manager import TelegramAccountManager
from app.database.repositories import DatabaseRepository


@pytest.fixture
def sample_telegram_account():
    """Create sample TelegramAccount for testing"""
    return TelegramAccount(
        id=uuid.uuid4(),
        phone_number="+12345678901",
        telegram_id=123456789,
        username="testuser",
        first_name="Test",
        last_name="User",
        bio="Test user bio",
        status=AccountStatus.ACTIVE,
        safety_level=SafetyLevel.CONSERVATIVE,
        is_verified=False,
        is_ai_disclosed=True,
        warming_progress=100.0,
        warming_phase="active",
        messages_sent_today=5,
        groups_joined_today=1,
        dms_sent_today=2,
        risk_score=15.0,
        max_messages_per_day=50,
        max_groups_per_day=2,
        max_dms_per_day=5,
        personality_profile={
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.3
        },
        communication_style={
            "formality": "casual",
            "enthusiasm": "moderate",
            "verbosity": "concise"
        },
        interests=["technology", "crypto", "AI"]
    )


@pytest.fixture
def sample_telegram_community():
    """Create sample TelegramCommunity for testing"""
    return TelegramCommunity(
        id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        chat_id=-1001234567890,
        community_name="Test Community",
        community_type="crypto",
        engagement_strategy=EngagementStrategy.HELPFUL_CONTRIBUTOR,
        status=CommunityStatus.ACTIVE,
        member_count=1500,
        daily_message_volume=50,
        formality_level="casual",
        primary_language="en",
        moderation_level="moderate",
        allowed_topics=["crypto", "trading", "defi"],
        community_metrics={
            "engagement_rate": 0.35,
            "response_rate": 0.8,
            "average_response_time": 120
        }
    )


@pytest.fixture
def sample_telegram_conversation(sample_telegram_account, sample_telegram_community):
    """Create sample TelegramConversation for testing"""
    return TelegramConversation(
        id=uuid.uuid4(),
        account_id=sample_telegram_account.id,
        community_id=sample_telegram_community.id,
        user_id=987654321,
        chat_id=-1001234567890,
        conversation_type="group",
        is_active=True,
        engagement_score=75.0,
        sentiment_history=[
            {"emotion": "positive", "confidence": 0.8, "timestamp": datetime.utcnow().isoformat()},
            {"emotion": "neutral", "confidence": 0.6, "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat()}
        ],
        conversation_context={
            "recent_topics": ["DeFi protocols", "yield farming"],
            "user_interests": ["crypto", "trading"],
            "interaction_count": 5
        }
    )


@pytest.fixture
def mock_pyrogram_client():
    """Mock Pyrogram client for testing"""
    client_mock = Mock()
    
    # Mock user info
    me_mock = Mock()
    me_mock.id = 123456789
    me_mock.is_bot = False
    me_mock.first_name = "Test"
    me_mock.username = "testuser"
    
    client_mock.get_me = AsyncMock(return_value=me_mock)
    client_mock.start = AsyncMock()
    client_mock.stop = AsyncMock()
    client_mock.send_message = AsyncMock()
    client_mock.send_chat_action = AsyncMock()
    client_mock.get_chat = AsyncMock()
    client_mock.get_chat_member = AsyncMock()
    client_mock.join_chat = AsyncMock()
    client_mock.leave_chat = AsyncMock()
    
    # Mock context manager for send_chat_action
    chat_action_mock = AsyncMock()
    chat_action_mock.__aenter__ = AsyncMock(return_value=chat_action_mock)
    chat_action_mock.__aexit__ = AsyncMock(return_value=None)
    client_mock.send_chat_action.return_value = chat_action_mock
    
    return client_mock


@pytest.fixture
def mock_telegram_message():
    """Create mock Telegram message"""
    message_mock = Mock(spec=types.Message)
    
    # User mock
    user_mock = Mock()
    user_mock.id = 987654321
    user_mock.is_bot = False
    user_mock.first_name = "John"
    user_mock.last_name = "Doe"
    user_mock.username = "johndoe"
    
    # Chat mock
    chat_mock = Mock()
    chat_mock.id = -1001234567890
    chat_mock.type = "supergroup"
    chat_mock.title = "Test Group"
    
    # Message properties
    message_mock.id = 12345
    message_mock.from_user = user_mock
    message_mock.chat = chat_mock
    message_mock.date = datetime.utcnow()
    message_mock.text = "Hello, this is a test message about DeFi protocols!"
    message_mock.reply_to_message = None
    
    return message_mock


@pytest.fixture
def mock_database_repository():
    """Mock DatabaseRepository for testing"""
    repo_mock = Mock(spec=DatabaseRepository)
    
    repo_mock.get_telegram_account = AsyncMock()
    repo_mock.update_telegram_account = AsyncMock()
    repo_mock.create_telegram_account = AsyncMock()
    repo_mock.delete_telegram_account = AsyncMock()
    
    repo_mock.get_telegram_community = AsyncMock()
    repo_mock.create_telegram_community = AsyncMock()
    repo_mock.update_telegram_community = AsyncMock()
    
    repo_mock.get_telegram_conversation = AsyncMock()
    repo_mock.create_telegram_conversation = AsyncMock()
    repo_mock.update_telegram_conversation = AsyncMock()
    
    repo_mock.create_conversation_message = AsyncMock()
    repo_mock.get_conversation_messages = AsyncMock(return_value=[])
    
    repo_mock.create_safety_event = AsyncMock()
    repo_mock.count_recent_safety_events = AsyncMock(return_value=0)
    repo_mock.count_active_communities = AsyncMock(return_value=3)
    repo_mock.count_recent_conversations = AsyncMock(return_value=10)
    
    return repo_mock


@pytest.fixture
def mock_consciousness_mirror():
    """Mock ConsciousnessMirror service"""
    consciousness_mock = Mock()
    
    consciousness_mock.initialize = AsyncMock()
    consciousness_mock.adapt_to_context = AsyncMock(return_value={
        "personality_adjustment": 0.1,
        "communication_style": "casual",
        "response_tone": "helpful"
    })
    consciousness_mock.generate_response = AsyncMock(return_value="This is a helpful response about DeFi!")
    
    return consciousness_mock


@pytest.fixture
def mock_memory_palace():
    """Mock MemoryPalace service"""
    memory_mock = Mock()
    
    memory_mock.initialize = AsyncMock()
    memory_mock.store_memory = AsyncMock()
    memory_mock.retrieve_relevant_memories = AsyncMock(return_value=[
        {
            "content": "Previous discussion about yield farming",
            "importance": 0.8,
            "timestamp": datetime.utcnow().isoformat()
        }
    ])
    memory_mock.consolidate_memories = AsyncMock()
    
    return memory_mock


@pytest.fixture
def mock_emotion_detector():
    """Mock EmotionDetector service"""
    emotion_mock = Mock()
    
    emotion_mock.analyze_text = AsyncMock(return_value={
        "primary_emotion": "positive",
        "confidence": 0.85,
        "intensity": 0.7,
        "emotions": {
            "joy": 0.6,
            "excitement": 0.25,
            "neutral": 0.15
        }
    })
    
    return emotion_mock


@pytest.fixture
def mock_temporal_archaeology():
    """Mock TemporalArchaeology service"""
    temporal_mock = Mock()
    
    temporal_mock.analyze_interaction_timing = AsyncMock(return_value={
        "optimal_timing": True,
        "user_activity_pattern": "active",
        "recommended_delay": 30,
        "confidence": 0.9
    })
    
    return temporal_mock


@pytest.fixture
def mock_telepathy_engine():
    """Mock DigitalTelepathyEngine service"""
    telepathy_mock = Mock()
    
    telepathy_mock.initialize = AsyncMock()
    telepathy_mock.process_communication = AsyncMock(return_value={
        "intent": "seeking_information",
        "context_understanding": 0.9,
        "response_suggestion": "Provide helpful DeFi information",
        "confidence": 0.85
    })
    
    return telepathy_mock


@pytest.fixture
def telegram_account_manager(
    sample_telegram_account,
    mock_database_repository,
    mock_consciousness_mirror,
    mock_memory_palace,
    mock_emotion_detector,
    mock_temporal_archaeology,
    mock_telepathy_engine
):
    """Create TelegramAccountManager with mocked dependencies"""
    
    manager = TelegramAccountManager(
        account_id=str(sample_telegram_account.id),
        api_id=12345,
        api_hash="test_api_hash",
        session_name="test_session",
        database=mock_database_repository,
        consciousness_mirror=mock_consciousness_mirror,
        memory_palace=mock_memory_palace,
        emotion_detector=mock_emotion_detector,
        temporal_archaeology=mock_temporal_archaeology,
        telepathy_engine=mock_telepathy_engine
    )
    
    # Set the account directly for testing
    manager.account = sample_telegram_account
    
    return manager


@pytest.fixture
def flood_wait_error():
    """Create FloodWait error for testing"""
    return FloodWait(value=30)  # 30 second flood wait


@pytest.fixture
def safety_event_data():
    """Sample safety event data"""
    return {
        "event_type": "flood_wait",
        "severity": "medium",
        "description": "Flood wait of 30 seconds",
        "data": {"wait_time": 30},
        "telegram_error_code": 420,
        "telegram_error_message": "Too Many Requests: retry after 30"
    }


class MockTelegramUpdate:
    """Mock Telegram update for testing webhooks"""
    
    def __init__(self, message_data: Dict[str, Any]):
        self.update_id = message_data.get("update_id", 1)
        self.message = MockTelegramMessage(message_data.get("message", {}))


class MockTelegramMessage:
    """Mock Telegram message for testing"""
    
    def __init__(self, message_data: Dict[str, Any]):
        self.id = message_data.get("id", 1)
        self.text = message_data.get("text", "")
        self.from_user = MockTelegramUser(message_data.get("from", {}))
        self.chat = MockTelegramChat(message_data.get("chat", {}))
        self.date = datetime.fromtimestamp(message_data.get("date", int(datetime.utcnow().timestamp())))
        self.reply_to_message = message_data.get("reply_to_message")


class MockTelegramUser:
    """Mock Telegram user for testing"""
    
    def __init__(self, user_data: Dict[str, Any]):
        self.id = user_data.get("id", 123456789)
        self.is_bot = user_data.get("is_bot", False)
        self.first_name = user_data.get("first_name", "Test")
        self.last_name = user_data.get("last_name", "User")
        self.username = user_data.get("username", "testuser")
        self.language_code = user_data.get("language_code", "en")


class MockTelegramChat:
    """Mock Telegram chat for testing"""
    
    def __init__(self, chat_data: Dict[str, Any]):
        self.id = chat_data.get("id", -1001234567890)
        self.type = chat_data.get("type", "supergroup")
        self.title = chat_data.get("title", "Test Group")
        self.username = chat_data.get("username")
        self.description = chat_data.get("description")


@pytest.fixture
def performance_test_scenarios():
    """Performance testing scenarios"""
    return {
        "message_processing": {
            "target_time": 0.1,  # 100ms
            "max_time": 0.5,     # 500ms
            "test_messages": 100
        },
        "response_generation": {
            "target_time": 2.0,   # 2 seconds
            "max_time": 10.0,     # 10 seconds
            "test_scenarios": 50
        },
        "safety_check": {
            "target_time": 0.05,  # 50ms
            "max_time": 0.2,      # 200ms
            "test_iterations": 1000
        },
        "memory_retrieval": {
            "target_time": 0.3,   # 300ms
            "max_time": 1.0,      # 1 second
            "test_queries": 200
        }
    }


@pytest.fixture
def anti_detection_test_patterns():
    """Anti-detection test patterns"""
    return {
        "typing_speeds": [10, 15, 20, 25, 30],  # chars per second
        "response_delays": [2, 5, 10, 30, 60, 120, 300],  # seconds
        "message_intervals": [60, 300, 600, 1800, 3600],  # seconds between messages
        "activity_patterns": [
            "morning_active",
            "evening_active",
            "random_bursts",
            "consistent_low"
        ],
        "conversation_styles": [
            "helpful_contributor",
            "casual_participant",
            "question_asker",
            "supportive_member"
        ]
    }


@pytest.fixture(autouse=True)
def setup_telegram_test_environment(
    mock_pyrogram_client,
    mock_database_repository,
    mock_consciousness_mirror,
    mock_memory_palace,
    mock_emotion_detector,
    mock_temporal_archaeology,
    mock_telepathy_engine
):
    """Setup test environment with all mocked Telegram dependencies"""
    # This fixture automatically sets up the testing environment
    # for all Telegram-related tests
    pass