"""
Telegram Test Data Fixtures

Comprehensive test data for Telegram account management testing:
- Sample messages and conversations
- Mock Telegram API responses
- Test scenarios and edge cases
- Performance test datasets
- Safety violation scenarios
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import uuid
from dataclasses import dataclass

from app.models.telegram_account import TelegramAccount, AccountStatus, SafetyLevel
from app.models.telegram_community import TelegramCommunity, CommunityStatus, EngagementStrategy
from app.models.telegram_conversation import TelegramConversation, ConversationType


@dataclass
class TestMessage:
    """Test message data structure"""
    id: int
    text: str
    user_id: int
    chat_id: int
    date: datetime
    reply_to_message_id: Optional[int] = None
    is_bot: bool = False
    username: Optional[str] = None
    first_name: str = "TestUser"
    chat_type: str = "private"
    chat_title: Optional[str] = None


@dataclass
class TestUpdate:
    """Test update data structure"""
    update_id: int
    message: TestMessage


class TelegramTestDataFactory:
    """Factory for generating Telegram test data"""
    
    def __init__(self):
        self.message_counter = 1
        self.update_counter = 1
        self.user_counter = 123456789
        self.chat_counter = -1001234567890
    
    def create_test_account(
        self, 
        status: AccountStatus = AccountStatus.ACTIVE,
        safety_level: SafetyLevel = SafetyLevel.CONSERVATIVE,
        risk_score: float = 0.0,
        **kwargs
    ) -> TelegramAccount:
        """Create test TelegramAccount"""
        defaults = {
            "phone_number": f"+1234567890{random.randint(0, 9)}",
            "first_name": "Test",
            "last_name": "User", 
            "username": f"testuser{random.randint(100, 999)}",
            "telegram_id": self.user_counter,
            "status": status,
            "safety_level": safety_level,
            "risk_score": risk_score,
            "max_messages_per_day": 50,
            "max_groups_per_day": 2,
            "max_dms_per_day": 5
        }
        defaults.update(kwargs)
        
        self.user_counter += 1
        return TelegramAccount(**defaults)
    
    def create_test_community(
        self,
        account_id: str,
        community_type: str = "crypto",
        engagement_strategy: EngagementStrategy = EngagementStrategy.HELPFUL_CONTRIBUTOR,
        **kwargs
    ) -> TelegramCommunity:
        """Create test TelegramCommunity"""
        defaults = {
            "account_id": account_id,
            "chat_id": self.chat_counter,
            "community_name": f"Test {community_type.title()} Community",
            "community_type": community_type,
            "engagement_strategy": engagement_strategy,
            "status": CommunityStatus.ACTIVE,
            "member_count": random.randint(100, 5000),
            "daily_message_volume": random.randint(10, 200),
            "formality_level": "casual",
            "primary_language": "en",
            "moderation_level": "moderate"
        }
        defaults.update(kwargs)
        
        self.chat_counter -= 1
        return TelegramCommunity(**defaults)
    
    def create_test_conversation(
        self,
        account_id: str,
        community_id: Optional[str] = None,
        conversation_type: ConversationType = ConversationType.PRIVATE,
        **kwargs
    ) -> TelegramConversation:
        """Create test TelegramConversation"""
        defaults = {
            "account_id": account_id,
            "community_id": community_id,
            "user_id": self.user_counter,
            "chat_id": self.chat_counter,
            "conversation_type": conversation_type,
            "is_active": True,
            "engagement_score": random.uniform(0, 100),
            "conversation_context": {
                "recent_topics": ["crypto", "DeFi"],
                "interaction_count": random.randint(1, 20)
            }
        }
        defaults.update(kwargs)
        
        self.user_counter += 1
        self.chat_counter -= 1
        return TelegramConversation(**defaults)
    
    def create_test_message(
        self,
        text: str,
        chat_type: str = "private",
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        **kwargs
    ) -> TestMessage:
        """Create test message"""
        message_id = self.message_counter
        self.message_counter += 1
        
        defaults = {
            "id": message_id,
            "text": text,
            "user_id": user_id or self.user_counter,
            "chat_id": chat_id or self.chat_counter,
            "date": datetime.utcnow(),
            "chat_type": chat_type,
            "first_name": "TestUser",
            "username": f"user{random.randint(100, 999)}"
        }
        defaults.update(kwargs)
        
        if chat_type in ["group", "supergroup"]:
            defaults["chat_title"] = f"Test {chat_type.title()}"
        
        return TestMessage(**defaults)
    
    def create_test_update(self, message: TestMessage) -> TestUpdate:
        """Create test update containing message"""
        update_id = self.update_counter
        self.update_counter += 1
        
        return TestUpdate(update_id=update_id, message=message)


# Pre-defined test data sets
class TelegramTestDatasets:
    """Pre-defined datasets for testing"""
    
    @staticmethod
    def get_sample_messages() -> List[Dict[str, Any]]:
        """Sample messages for testing"""
        return [
            {
                "text": "Hello! How's everyone doing today?",
                "emotion": "positive",
                "complexity": "low",
                "expected_response_time": (2, 10)
            },
            {
                "text": "What do you think about the new DeFi protocol that launched?",
                "emotion": "curious", 
                "complexity": "medium",
                "expected_response_time": (10, 30)
            },
            {
                "text": "Can someone explain the mathematical model behind constant product AMMs and how they handle impermanent loss in different market conditions?",
                "emotion": "analytical",
                "complexity": "high", 
                "expected_response_time": (30, 120)
            },
            {
                "text": "OMG the market is crashing! What should I do?!",
                "emotion": "panic",
                "complexity": "low",
                "expected_response_time": (2, 15)
            },
            {
                "text": "Thanks for the helpful explanation earlier 😊",
                "emotion": "grateful",
                "complexity": "low",
                "expected_response_time": (2, 8)
            }
        ]
    
    @staticmethod
    def get_conversation_scenarios() -> List[Dict[str, Any]]:
        """Conversation scenarios for testing"""
        return [
            {
                "scenario": "new_user_onboarding",
                "messages": [
                    "Hi, I'm new to crypto. Where should I start?",
                    "I've heard about DeFi but don't understand it",
                    "Is it safe to invest in these protocols?",
                    "What's a good amount to start with?"
                ],
                "expected_flow": "educational_support"
            },
            {
                "scenario": "technical_discussion",
                "messages": [
                    "The new Uniswap V4 hooks look interesting",
                    "How do you think they'll affect gas costs?",
                    "MEV protection is crucial for retail users",
                    "The architecture seems well thought out"
                ],
                "expected_flow": "collaborative_analysis"
            },
            {
                "scenario": "market_crisis",
                "messages": [
                    "Bitcoin just dropped 20%!",
                    "Should I sell everything?",
                    "This looks like 2018 all over again",
                    "When do you think it'll recover?"
                ],
                "expected_flow": "calm_guidance"
            },
            {
                "scenario": "casual_chat",
                "messages": [
                    "How's everyone's weekend going?",
                    "Anyone watching the football game?",
                    "Just got back from vacation!",
                    "Weather's been crazy lately"
                ],
                "expected_flow": "social_engagement"
            }
        ]
    
    @staticmethod
    def get_safety_test_scenarios() -> List[Dict[str, Any]]:
        """Safety violation scenarios for testing"""
        return [
            {
                "scenario": "flood_wait_progression",
                "events": [
                    {"type": "flood_wait", "duration": 10, "risk_increase": 5},
                    {"type": "flood_wait", "duration": 30, "risk_increase": 10},
                    {"type": "flood_wait", "duration": 60, "risk_increase": 15},
                    {"type": "flood_wait", "duration": 300, "risk_increase": 25}
                ]
            },
            {
                "scenario": "spam_detection",
                "events": [
                    {"type": "similar_message_warning", "risk_increase": 8},
                    {"type": "rapid_messaging", "risk_increase": 12},
                    {"type": "spam_warning", "risk_increase": 20},
                    {"type": "account_restriction", "risk_increase": 30}
                ]
            },
            {
                "scenario": "daily_limit_testing",
                "events": [
                    {"type": "approach_message_limit", "current": 45, "max": 50},
                    {"type": "exceed_message_limit", "current": 52, "max": 50},
                    {"type": "approach_group_limit", "current": 2, "max": 2},
                    {"type": "exceed_dm_limit", "current": 6, "max": 5}
                ]
            }
        ]
    
    @staticmethod
    def get_performance_test_data() -> Dict[str, List[Any]]:
        """Performance testing datasets"""
        return {
            "message_volumes": [10, 50, 100, 500, 1000],
            "concurrent_conversations": [1, 5, 10, 20, 50],
            "message_lengths": [
                "Hi",
                "Hello there!",
                "This is a medium length message for testing",
                "This is a very long message that simulates detailed explanations about complex topics like DeFi protocols, yield farming strategies, and risk management techniques",
                "Short messages mixed with " + "very " * 50 + "long content to test variable processing times"
            ],
            "sustained_load_durations": [30, 60, 300, 600],  # seconds
            "memory_test_iterations": [100, 500, 1000, 2000, 5000]
        }
    
    @staticmethod
    def get_anti_detection_patterns() -> List[Dict[str, Any]]:
        """Anti-detection test patterns"""
        return [
            {
                "pattern_type": "human_casual",
                "characteristics": {
                    "response_delays": [5, 12, 8, 25, 3, 45, 7, 18],
                    "typing_speeds": [12, 18, 15, 20, 14, 16, 19, 13],
                    "message_intervals": [180, 420, 95, 850, 234, 670],
                    "activity_times": [9, 12, 14, 18, 20, 22]
                }
            },
            {
                "pattern_type": "bot_like",
                "characteristics": {
                    "response_delays": [5, 5, 6, 5, 5, 6, 5, 5],
                    "typing_speeds": [15, 15, 15, 15, 15, 15, 15, 15],
                    "message_intervals": [60, 60, 60, 60, 60, 60],
                    "activity_times": [10, 11, 12, 13, 14, 15, 16, 17]
                }
            },
            {
                "pattern_type": "human_engaged", 
                "characteristics": {
                    "response_delays": [3, 8, 15, 5, 22, 1, 35, 12],
                    "typing_speeds": [18, 22, 16, 25, 14, 20, 19, 17],
                    "message_intervals": [45, 120, 30, 300, 180, 90],
                    "activity_times": [8, 9, 12, 13, 18, 19, 20, 21, 22]
                }
            }
        ]


class MockTelegramAPI:
    """Mock Telegram API responses for testing"""
    
    @staticmethod
    def get_user_response(user_id: int) -> Dict[str, Any]:
        """Mock getChat response for user"""
        return {
            "ok": True,
            "result": {
                "id": user_id,
                "first_name": "Test",
                "last_name": "User",
                "username": f"testuser{user_id}",
                "type": "private",
                "active_usernames": [f"testuser{user_id}"],
                "is_bot": False
            }
        }
    
    @staticmethod
    def get_chat_response(chat_id: int, chat_type: str = "supergroup") -> Dict[str, Any]:
        """Mock getChat response for group/channel"""
        return {
            "ok": True,
            "result": {
                "id": chat_id,
                "title": f"Test {chat_type.title()}",
                "type": chat_type,
                "member_count": random.randint(100, 5000),
                "description": f"Test {chat_type} for testing purposes",
                "username": f"test_{abs(chat_id)}"
            }
        }
    
    @staticmethod
    def get_chat_member_response(user_id: int, status: str = "member") -> Dict[str, Any]:
        """Mock getChatMember response"""
        return {
            "ok": True,
            "result": {
                "user": {
                    "id": user_id,
                    "is_bot": False,
                    "first_name": "Test",
                    "username": f"testuser{user_id}"
                },
                "status": status,
                "until_date": None
            }
        }
    
    @staticmethod
    def get_send_message_response(message_id: int, chat_id: int) -> Dict[str, Any]:
        """Mock sendMessage response"""
        return {
            "ok": True,
            "result": {
                "message_id": message_id,
                "from": {
                    "id": 123456789,
                    "is_bot": True,
                    "first_name": "Test Bot",
                    "username": "testbot"
                },
                "chat": {
                    "id": chat_id,
                    "type": "private" if chat_id > 0 else "supergroup"
                },
                "date": int(datetime.utcnow().timestamp()),
                "text": "Test response message"
            }
        }
    
    @staticmethod
    def get_error_response(error_code: int, description: str) -> Dict[str, Any]:
        """Mock error response"""
        return {
            "ok": False,
            "error_code": error_code,
            "description": description
        }
    
    @staticmethod
    def get_flood_wait_error(seconds: int) -> Dict[str, Any]:
        """Mock flood wait error response"""
        return MockTelegramAPI.get_error_response(
            429, 
            f"Too Many Requests: retry after {seconds}"
        )


class TestScenarioGenerator:
    """Generate test scenarios for comprehensive testing"""
    
    def __init__(self):
        self.factory = TelegramTestDataFactory()
    
    def generate_multi_community_scenario(self) -> Dict[str, Any]:
        """Generate multi-community engagement scenario"""
        account = self.factory.create_test_account()
        
        communities = [
            self.factory.create_test_community(
                account.id, 
                "crypto",
                EngagementStrategy.HELPFUL_CONTRIBUTOR
            ),
            self.factory.create_test_community(
                account.id,
                "defi", 
                EngagementStrategy.ACTIVE_PARTICIPANT
            ),
            self.factory.create_test_community(
                account.id,
                "general",
                EngagementStrategy.CASUAL_OBSERVER
            )
        ]
        
        conversations = []
        for community in communities:
            for _ in range(random.randint(1, 3)):
                conv = self.factory.create_test_conversation(
                    account.id,
                    community.id,
                    ConversationType.GROUP
                )
                conversations.append(conv)
        
        return {
            "account": account,
            "communities": communities,
            "conversations": conversations,
            "expected_daily_messages": sum(
                int(community.engagement_strategy.value.split("_")[0] == "helpful") * 15 +
                int(community.engagement_strategy.value.split("_")[0] == "active") * 10 +
                int(community.engagement_strategy.value.split("_")[0] == "casual") * 3
                for community in communities
            )
        }
    
    def generate_crisis_response_scenario(self) -> Dict[str, Any]:
        """Generate market crisis response scenario"""
        account = self.factory.create_test_account(
            safety_level=SafetyLevel.MODERATE
        )
        
        community = self.factory.create_test_community(
            account.id,
            "crypto",
            EngagementStrategy.HELPFUL_CONTRIBUTOR,
            member_count=5000,
            daily_message_volume=500  # High activity during crisis
        )
        
        crisis_messages = [
            "Bitcoin just crashed 30%! What's happening?",
            "Should I sell everything right now?", 
            "Is this the end of crypto?",
            "My portfolio is down 50%!",
            "When do you think it'll recover?",
            "Are we going to zero?",
            "This is worse than 2018",
            "Should I buy the dip?",
            "What caused this crash?",
            "Is DeFi still safe?"
        ]
        
        return {
            "account": account,
            "community": community,
            "crisis_messages": crisis_messages,
            "expected_behavior": {
                "response_rate": 0.6,  # Higher response rate during crisis
                "message_tone": "calm_and_educational",
                "avoid_speculation": True,
                "provide_resources": True
            }
        }
    
    def generate_account_warming_scenario(self) -> Dict[str, Any]:
        """Generate new account warming scenario"""
        account = self.factory.create_test_account(
            status=AccountStatus.INACTIVE,
            safety_level=SafetyLevel.CONSERVATIVE,
            warming_progress=0.0
        )
        
        warming_phases = [
            {
                "phase": "profile_setup",
                "duration_days": 1,
                "activities": ["complete_profile", "add_bio", "upload_photo"],
                "progress_target": 25.0
            },
            {
                "phase": "contact_building", 
                "duration_days": 2,
                "activities": ["add_contacts", "join_channels", "read_messages"],
                "progress_target": 50.0
            },
            {
                "phase": "group_joining",
                "duration_days": 3,
                "activities": ["join_groups", "observe_conversations", "occasional_reactions"],
                "progress_target": 75.0
            },
            {
                "phase": "engagement_start",
                "duration_days": 7,
                "activities": ["first_messages", "helpful_responses", "build_reputation"],
                "progress_target": 100.0
            }
        ]
        
        return {
            "account": account,
            "warming_phases": warming_phases,
            "total_warming_duration": sum(phase["duration_days"] for phase in warming_phases),
            "safety_constraints": {
                "max_messages_per_day": 5,  # Very conservative during warming
                "max_groups_per_day": 1,
                "no_proactive_messaging": True
            }
        }
    
    def generate_performance_stress_scenario(self) -> Dict[str, Any]:
        """Generate performance stress testing scenario"""
        account = self.factory.create_test_account(
            max_messages_per_day=1000  # High limits for stress testing
        )
        
        # Multiple communities with high activity
        communities = []
        conversations = []
        
        for i in range(10):  # 10 communities
            community = self.factory.create_test_community(
                account.id,
                f"community_{i}",
                random.choice(list(EngagementStrategy)),
                member_count=random.randint(1000, 10000),
                daily_message_volume=random.randint(100, 1000)
            )
            communities.append(community)
            
            # Multiple conversations per community
            for j in range(random.randint(5, 15)):
                conv = self.factory.create_test_conversation(
                    account.id,
                    community.id,
                    ConversationType.GROUP
                )
                conversations.append(conv)
        
        return {
            "account": account,
            "communities": communities,
            "conversations": conversations,
            "stress_parameters": {
                "concurrent_messages": 100,
                "messages_per_second": 50,
                "sustained_duration_minutes": 30,
                "memory_limit_mb": 500
            }
        }