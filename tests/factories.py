"""
Test Factories using Factory Boy

Comprehensive factory classes for generating test data objects:
- User and authentication factories
- Conversation and message factories
- Engagement and behavioral pattern factories
- Group session and member factories
- Viral content and sharing factories
- Payment and subscription factories
- Support for realistic test data generation
- Trait-based factory customization
"""

import factory
from factory import fuzzy
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any
import random

from app.models.user import User
from app.models.conversation import Conversation, Message, MessageType, MessageDirection
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, EngagementType, SentimentType,
    EngagementMilestone, UserMilestoneProgress
)
from app.models.group_session import (
    GroupSession, GroupMember, GroupConversation, GroupAnalytics,
    GroupType, MemberRole, GroupStatus, MessageFrequency
)
from app.models.sharing import (
    ShareableContent, ContentShare, ShareableContentType, SocialPlatform,
    ViralMetrics
)


class BaseFactory(factory.Factory):
    """Base factory with common utilities."""
    
    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        """Override create to handle UUID generation."""
        if hasattr(model_class, 'id') and 'id' not in kwargs:
            kwargs['id'] = str(uuid.uuid4())
        return super()._create(model_class, *args, **kwargs)


class UserFactory(BaseFactory):
    """Factory for User model."""
    
    class Meta:
        model = User
    
    # Core user data
    telegram_id = fuzzy.FuzzyInteger(100000000, 999999999)
    username = factory.Sequence(lambda n: f"testuser{n}")
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')
    language_code = fuzzy.FuzzyChoice(['en', 'es', 'fr', 'de', 'it', 'ru'])
    
    # User properties
    is_bot = False
    is_premium = factory.Faker('boolean', chance_of_getting_true=20)
    is_active = True
    is_blocked = False
    
    # Timestamps
    created_at = factory.Faker('date_time_between', start_date='-1y', end_date='now')
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(hours=random.randint(1, 24)))
    last_seen_at = factory.Faker('date_time_between', start_date='-30d', end_date='now')
    
    # User settings
    timezone = factory.Faker('timezone')
    notification_settings = factory.Dict({
        'email_notifications': factory.Faker('boolean'),
        'push_notifications': True,
        'marketing_emails': factory.Faker('boolean', chance_of_getting_true=30)
    })
    
    class Params:
        # Traits for different user types
        premium_user = factory.Trait(
            is_premium=True,
            notification_settings=factory.Dict({
                'email_notifications': True,
                'push_notifications': True,
                'marketing_emails': True,
                'priority_support': True
            })
        )
        
        inactive_user = factory.Trait(
            is_active=False,
            last_seen_at=factory.Faker('date_time_between', start_date='-90d', end_date='-30d')
        )
        
        blocked_user = factory.Trait(
            is_blocked=True,
            is_active=False
        )


class ConversationFactory(BaseFactory):
    """Factory for Conversation model."""
    
    class Meta:
        model = Conversation
    
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    context = factory.Faker('sentence', nb_words=6)
    is_active = True
    
    # Timestamps
    created_at = factory.Faker('date_time_between', start_date='-30d', end_date='now')
    updated_at = factory.LazyAttribute(
        lambda obj: obj.created_at + timedelta(minutes=random.randint(1, 120))
    )
    
    # Conversation metadata
    metadata = factory.Dict({
        'source': fuzzy.FuzzyChoice(['telegram', 'web', 'api']),
        'type': fuzzy.FuzzyChoice(['private', 'group']),
        'priority': fuzzy.FuzzyChoice(['low', 'medium', 'high'])
    })
    
    class Params:
        # Traits for different conversation types
        with_user = factory.Trait(
            user=factory.SubFactory(UserFactory)
        )
        
        long_running = factory.Trait(
            created_at=factory.Faker('date_time_between', start_date='-7d', end_date='-1d'),
            updated_at=factory.Faker('date_time_between', start_date='-1d', end_date='now')
        )
        
        inactive_conversation = factory.Trait(
            is_active=False,
            updated_at=factory.Faker('date_time_between', start_date='-30d', end_date='-7d')
        )


class MessageFactory(BaseFactory):
    """Factory for Message model."""
    
    class Meta:
        model = Message
    
    conversation_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    content = factory.Faker('paragraph', nb_sentences=3)
    message_type = fuzzy.FuzzyChoice([
        MessageType.TEXT, MessageType.VOICE, MessageType.IMAGE, MessageType.COMMAND
    ])
    direction = fuzzy.FuzzyChoice([MessageDirection.INCOMING, MessageDirection.OUTGOING])
    is_from_bot = factory.LazyAttribute(lambda obj: obj.direction == MessageDirection.OUTGOING)
    
    # Message metadata
    timestamp = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    token_count = fuzzy.FuzzyInteger(10, 200)
    processing_time = fuzzy.FuzzyFloat(0.1, 2.0)
    
    # Analysis data
    analysis_data = factory.Dict({
        'sentiment': factory.Dict({
            'score': fuzzy.FuzzyFloat(-1.0, 1.0),
            'type': fuzzy.FuzzyChoice(['positive', 'negative', 'neutral'])
        }),
        'topics': factory.List([
            factory.Faker('word') for _ in range(random.randint(1, 3))
        ]),
        'confidence': fuzzy.FuzzyFloat(0.7, 1.0)
    })
    
    class Params:
        # Message type traits
        user_message = factory.Trait(
            direction=MessageDirection.INCOMING,
            is_from_bot=False,
            content=factory.Faker('sentence', nb_words=10)
        )
        
        bot_message = factory.Trait(
            direction=MessageDirection.OUTGOING,
            is_from_bot=True,
            content=factory.Faker('paragraph', nb_sentences=2)
        )
        
        voice_message = factory.Trait(
            message_type=MessageType.VOICE,
            content="[Voice Message]",
            analysis_data=factory.Dict({
                'duration': fuzzy.FuzzyFloat(1.0, 30.0),
                'transcription': factory.Faker('sentence'),
                'audio_quality': fuzzy.FuzzyFloat(0.6, 1.0)
            })
        )
        
        command_message = factory.Trait(
            message_type=MessageType.COMMAND,
            content=factory.LazyAttribute(lambda obj: f"/{factory.Faker('word').generate()}"),
            direction=MessageDirection.INCOMING,
            is_from_bot=False
        )


class UserEngagementFactory(BaseFactory):
    """Factory for UserEngagement model."""
    
    class Meta:
        model = UserEngagement
    
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    telegram_id = fuzzy.FuzzyInteger(100000000, 999999999)
    engagement_type = fuzzy.FuzzyChoice([
        EngagementType.MESSAGE, EngagementType.COMMAND, EngagementType.VOICE_MESSAGE,
        EngagementType.CALLBACK, EngagementType.REACTION
    ])
    
    # Interaction data
    interaction_timestamp = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    message_text = factory.Faker('sentence', nb_words=12)
    command_name = factory.Maybe(
        'engagement_type',
        yes_declaration=factory.Faker('word'),
        no_declaration=None
    )
    session_id = factory.LazyFunction(lambda: f"session-{uuid.uuid4().hex[:8]}")
    
    # Sentiment analysis
    sentiment_score = fuzzy.FuzzyFloat(-1.0, 1.0)
    sentiment_type = fuzzy.FuzzyChoice([
        SentimentType.VERY_POSITIVE, SentimentType.POSITIVE, SentimentType.NEUTRAL,
        SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE
    ])
    
    # Message characteristics
    response_time_seconds = fuzzy.FuzzyInteger(5, 300)
    message_length = factory.LazyAttribute(lambda obj: len(obj.message_text) if obj.message_text else 0)
    contains_emoji = factory.Faker('boolean', chance_of_getting_true=40)
    contains_question = factory.Faker('boolean', chance_of_getting_true=25)
    
    # Analysis results
    detected_topics = factory.List([
        factory.Faker('word') for _ in range(random.randint(0, 3))
    ])
    user_intent = fuzzy.FuzzyChoice([
        'help_seeking', 'gratitude', 'question', 'general_conversation',
        'problem_reporting', 'positive_feedback', 'negative_feedback'
    ])
    mood_indicators = factory.Dict({
        'excitement_level': fuzzy.FuzzyFloat(0.0, 1.0),
        'frustration_level': fuzzy.FuzzyFloat(0.0, 1.0),
        'confusion_level': fuzzy.FuzzyFloat(0.0, 1.0),
        'gratitude_level': fuzzy.FuzzyFloat(0.0, 1.0)
    })
    
    engagement_quality_score = fuzzy.FuzzyFloat(0.1, 1.0)
    is_meaningful_interaction = factory.Faker('boolean', chance_of_getting_true=70)
    
    class Params:
        # Engagement type traits
        positive_engagement = factory.Trait(
            sentiment_type=SentimentType.POSITIVE,
            sentiment_score=fuzzy.FuzzyFloat(0.3, 1.0),
            engagement_quality_score=fuzzy.FuzzyFloat(0.6, 1.0),
            is_meaningful_interaction=True
        )
        
        negative_engagement = factory.Trait(
            sentiment_type=SentimentType.NEGATIVE,
            sentiment_score=fuzzy.FuzzyFloat(-1.0, -0.3),
            mood_indicators=factory.Dict({
                'frustration_level': fuzzy.FuzzyFloat(0.5, 1.0),
                'excitement_level': fuzzy.FuzzyFloat(0.0, 0.2)
            })
        )
        
        high_quality = factory.Trait(
            engagement_quality_score=fuzzy.FuzzyFloat(0.8, 1.0),
            is_meaningful_interaction=True,
            message_length=fuzzy.FuzzyInteger(50, 200)
        )


class UserBehaviorPatternFactory(BaseFactory):
    """Factory for UserBehaviorPattern model."""
    
    class Meta:
        model = UserBehaviorPattern
    
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    telegram_id = fuzzy.FuzzyInteger(100000000, 999999999)
    
    # Activity metrics
    total_interactions = fuzzy.FuzzyInteger(10, 500)
    daily_interaction_average = fuzzy.FuzzyFloat(1.0, 20.0)
    most_active_hour = fuzzy.FuzzyInteger(8, 22)
    most_active_day = fuzzy.FuzzyInteger(0, 6)  # 0=Monday, 6=Sunday
    average_session_length_minutes = fuzzy.FuzzyFloat(5.0, 60.0)
    
    # Sentiment patterns
    average_sentiment_score = fuzzy.FuzzyFloat(-0.5, 0.8)
    dominant_sentiment = fuzzy.FuzzyChoice([
        SentimentType.POSITIVE, SentimentType.NEUTRAL, SentimentType.NEGATIVE
    ])
    engagement_quality_trend = fuzzy.FuzzyFloat(-0.3, 0.3)
    response_time_average_seconds = fuzzy.FuzzyFloat(30.0, 180.0)
    
    # Interaction preferences
    preferred_interaction_types = factory.List([
        factory.Dict({
            'type': fuzzy.FuzzyChoice(['message', 'command', 'voice']),
            'count': fuzzy.FuzzyInteger(5, 100)
        }) for _ in range(random.randint(1, 3))
    ])
    favorite_commands = factory.List([
        factory.Dict({
            'command': factory.Faker('word'),
            'count': fuzzy.FuzzyInteger(1, 20)
        }) for _ in range(random.randint(0, 3))
    ])
    topic_interests = factory.List([
        factory.Dict({
            'topic': factory.Faker('word'),
            'relevance': fuzzy.FuzzyFloat(0.1, 1.0)
        }) for _ in range(random.randint(1, 5))
    ])
    
    # Behavioral indicators
    is_highly_engaged = factory.Faker('boolean', chance_of_getting_true=30)
    shows_declining_engagement = factory.Faker('boolean', chance_of_getting_true=20)
    needs_re_engagement = factory.Faker('boolean', chance_of_getting_true=25)
    days_since_last_interaction = fuzzy.FuzzyInteger(0, 14)
    churn_risk_score = fuzzy.FuzzyFloat(0.0, 1.0)
    
    # Outreach optimization
    optimal_outreach_hour = factory.LazyAttribute(lambda obj: obj.most_active_hour)
    
    # Analysis metadata
    last_pattern_analysis = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    pattern_analysis_version = "1.0"
    
    class Params:
        # Behavioral pattern traits
        highly_engaged_user = factory.Trait(
            is_highly_engaged=True,
            engagement_quality_trend=fuzzy.FuzzyFloat(0.1, 0.5),
            churn_risk_score=fuzzy.FuzzyFloat(0.0, 0.3),
            needs_re_engagement=False,
            total_interactions=fuzzy.FuzzyInteger(100, 500)
        )
        
        at_risk_user = factory.Trait(
            churn_risk_score=fuzzy.FuzzyFloat(0.7, 1.0),
            shows_declining_engagement=True,
            needs_re_engagement=True,
            days_since_last_interaction=fuzzy.FuzzyInteger(3, 14),
            engagement_quality_trend=fuzzy.FuzzyFloat(-0.5, -0.1)
        )
        
        new_user = factory.Trait(
            total_interactions=fuzzy.FuzzyInteger(1, 10),
            days_since_last_interaction=fuzzy.FuzzyInteger(0, 3),
            last_pattern_analysis=factory.Faker('date_time_between', start_date='-3d', end_date='now')
        )


class GroupSessionFactory(BaseFactory):
    """Factory for GroupSession model."""
    
    class Meta:
        model = GroupSession
    
    telegram_group_id = fuzzy.FuzzyInteger(-1001999999999, -1001000000000)
    title = factory.Faker('catch_phrase')
    description = factory.Faker('text', max_nb_chars=200)
    group_type = fuzzy.FuzzyChoice([GroupType.GROUP, GroupType.SUPERGROUP, GroupType.CHANNEL])
    status = GroupStatus.ACTIVE
    
    # Group metrics
    member_count = fuzzy.FuzzyInteger(5, 1000)
    total_messages = fuzzy.FuzzyInteger(100, 50000)
    active_conversations = fuzzy.FuzzyInteger(1, 20)
    
    # Timestamps
    created_at = factory.Faker('date_time_between', start_date='-1y', end_date='-1w')
    updated_at = factory.Faker('date_time_between', start_date='-1w', end_date='now')
    last_activity_at = factory.Faker('date_time_between', start_date='-1d', end_date='now')
    
    # Group settings
    settings = factory.Dict({
        'proactive_responses': factory.Faker('boolean', chance_of_getting_true=60),
        'analytics_enabled': True,
        'max_thread_age_hours': fuzzy.FuzzyInteger(12, 72),
        'auto_moderation': factory.Faker('boolean', chance_of_getting_true=40)
    })
    
    class Params:
        # Group size traits
        small_group = factory.Trait(
            member_count=fuzzy.FuzzyInteger(5, 50),
            total_messages=fuzzy.FuzzyInteger(100, 5000)
        )
        
        large_group = factory.Trait(
            member_count=fuzzy.FuzzyInteger(100, 1000),
            total_messages=fuzzy.FuzzyInteger(5000, 50000),
            group_type=GroupType.SUPERGROUP
        )
        
        inactive_group = factory.Trait(
            status=GroupStatus.INACTIVE,
            last_activity_at=factory.Faker('date_time_between', start_date='-30d', end_date='-7d')
        )


class GroupMemberFactory(BaseFactory):
    """Factory for GroupMember model."""
    
    class Meta:
        model = GroupMember
    
    group_id = fuzzy.FuzzyInteger(1, 1000)
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    telegram_user_id = fuzzy.FuzzyInteger(100000000, 999999999)
    username = factory.Sequence(lambda n: f"groupuser{n}")
    display_name = factory.Faker('name')
    
    # Member status
    role = fuzzy.FuzzyChoice([MemberRole.MEMBER, MemberRole.ADMIN, MemberRole.OWNER])
    is_active = True
    is_muted = factory.Faker('boolean', chance_of_getting_true=5)
    
    # Activity tracking
    joined_at = factory.Faker('date_time_between', start_date='-1y', end_date='-1w')
    last_seen_at = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    last_message_at = factory.Faker('date_time_between', start_date='-3d', end_date='now')
    
    # Engagement metrics
    message_count = fuzzy.FuzzyInteger(10, 1000)
    engagement_score = fuzzy.FuzzyFloat(0.1, 1.0)
    average_message_length = fuzzy.FuzzyFloat(20.0, 150.0)
    
    class Params:
        # Role-based traits
        admin_member = factory.Trait(
            role=MemberRole.ADMIN,
            engagement_score=fuzzy.FuzzyFloat(0.6, 1.0),
            message_count=fuzzy.FuzzyInteger(100, 1000)
        )
        
        lurker_member = factory.Trait(
            message_count=fuzzy.FuzzyInteger(0, 10),
            engagement_score=fuzzy.FuzzyFloat(0.0, 0.3),
            last_message_at=factory.Faker('date_time_between', start_date='-30d', end_date='-7d')
        )
        
        active_member = factory.Trait(
            message_count=fuzzy.FuzzyInteger(50, 500),
            engagement_score=fuzzy.FuzzyFloat(0.5, 1.0),
            last_message_at=factory.Faker('date_time_between', start_date='-1d', end_date='now')
        )


class GroupConversationFactory(BaseFactory):
    """Factory for GroupConversation model."""
    
    class Meta:
        model = GroupConversation
    
    group_id = fuzzy.FuzzyInteger(1, 1000)
    thread_id = factory.LazyFunction(lambda: f"thread-{uuid.uuid4().hex[:12]}")
    
    # Conversation timeline
    started_at = factory.Faker('date_time_between', start_date='-7d', end_date='-1h')
    ended_at = factory.Maybe(
        'is_active',
        yes_declaration=None,
        no_declaration=factory.LazyAttribute(
            lambda obj: obj.started_at + timedelta(minutes=random.randint(5, 120))
        )
    )
    
    # Participation metrics
    participant_count = fuzzy.FuzzyInteger(2, 10)
    message_count = fuzzy.FuzzyInteger(5, 100)
    bot_interactions = fuzzy.FuzzyInteger(0, 10)
    duration_seconds = factory.LazyAttribute(
        lambda obj: random.randint(300, 7200) if obj.ended_at else None
    )
    
    # Content analysis
    topic = factory.Maybe(
        factory.Faker('boolean', chance_of_getting_true=40),
        yes_declaration=factory.Faker('words', nb=3),
        no_declaration=None
    )
    keywords = factory.List([
        factory.Faker('word') for _ in range(random.randint(1, 5))
    ])
    sentiment_summary = factory.Dict({
        'positive': fuzzy.FuzzyFloat(0.0, 1.0),
        'negative': fuzzy.FuzzyFloat(0.0, 1.0),
        'neutral': fuzzy.FuzzyFloat(0.0, 1.0),
        'score': fuzzy.FuzzyFloat(-1.0, 1.0)
    })
    language_distribution = factory.Dict({
        'en': fuzzy.FuzzyFloat(0.5, 1.0),
        'es': fuzzy.FuzzyFloat(0.0, 0.3)
    })
    
    # Quality metrics
    engagement_score = fuzzy.FuzzyFloat(0.1, 1.0)
    toxicity_score = fuzzy.FuzzyFloat(0.0, 0.3)
    educational_value = fuzzy.FuzzyFloat(0.0, 1.0)
    
    class Params:
        # Conversation traits
        active_conversation = factory.Trait(
            ended_at=None,
            started_at=factory.Faker('date_time_between', start_date='-2h', end_date='now')
        )
        
        long_conversation = factory.Trait(
            message_count=fuzzy.FuzzyInteger(50, 200),
            participant_count=fuzzy.FuzzyInteger(5, 15),
            duration_seconds=fuzzy.FuzzyInteger(3600, 14400)  # 1-4 hours
        )
        
        high_engagement = factory.Trait(
            engagement_score=fuzzy.FuzzyFloat(0.7, 1.0),
            bot_interactions=fuzzy.FuzzyInteger(3, 10),
            toxicity_score=fuzzy.FuzzyFloat(0.0, 0.2)
        )


class ShareableContentFactory(BaseFactory):
    """Factory for ShareableContent model."""
    
    class Meta:
        model = ShareableContent
    
    content_type = fuzzy.FuzzyChoice([
        ShareableContentType.FUNNY_MOMENT,
        ShareableContentType.PERSONALITY_INSIGHT,
        ShareableContentType.WISDOM_QUOTE,
        ShareableContentType.AI_RESPONSE,
        ShareableContentType.EDUCATIONAL_SUMMARY
    ])
    
    # Content data
    title = factory.Faker('catch_phrase')
    description = factory.Faker('paragraph', nb_sentences=3)
    content_data = factory.Dict({
        'text': factory.Faker('paragraph'),
        'platforms': factory.Dict({
            'twitter': factory.Faker('sentence', nb_words=15),
            'instagram': factory.Faker('paragraph', nb_sentences=2)
        })
    })
    
    # Viral metrics
    viral_score = fuzzy.FuzzyFloat(60.0, 95.0)
    view_count = fuzzy.FuzzyInteger(100, 10000)
    share_count = fuzzy.FuzzyInteger(10, 1000)
    like_count = fuzzy.FuzzyInteger(50, 5000)
    comment_count = fuzzy.FuzzyInteger(5, 500)
    
    # Social media optimization
    hashtags = factory.List([
        f"#{factory.Faker('word').generate()}" for _ in range(random.randint(3, 8))
    ])
    optimal_platforms = factory.List([
        platform.value for platform in random.sample(list(SocialPlatform), k=random.randint(2, 4))
    ])
    
    # Privacy and source
    is_anonymized = True
    anonymization_level = fuzzy.FuzzyChoice(['medium', 'high'])
    source_conversation_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    source_user_anonymous_id = factory.LazyFunction(lambda: uuid.uuid4().hex[:16])
    
    # Publishing status
    is_published = factory.Faker('boolean', chance_of_getting_true=70)
    published_at = factory.Maybe(
        'is_published',
        yes_declaration=factory.Faker('date_time_between', start_date='-7d', end_date='now'),
        no_declaration=None
    )
    
    # AI enhancement data
    ai_enhancement_data = factory.Dict({
        'generation_context': factory.Dict({
            'conversation_excerpt': factory.Faker('paragraph'),
            'viral_elements': factory.List([
                factory.Faker('word') for _ in range(random.randint(2, 5))
            ])
        }),
        'trigger_type': factory.LazyAttribute(lambda obj: obj.content_type),
        'model_version': 'gpt-4'
    })
    
    class Params:
        # Content type traits
        high_viral_potential = factory.Trait(
            viral_score=fuzzy.FuzzyFloat(85.0, 95.0),
            view_count=fuzzy.FuzzyInteger(5000, 50000),
            share_count=fuzzy.FuzzyInteger(500, 5000),
            is_published=True
        )
        
        funny_content = factory.Trait(
            content_type=ShareableContentType.FUNNY_MOMENT,
            hashtags=['#Funny', '#AI', '#Humor', '#TechComedy'],
            optimal_platforms=[SocialPlatform.TWITTER.value, SocialPlatform.TIKTOK.value]
        )
        
        educational_content = factory.Trait(
            content_type=ShareableContentType.EDUCATIONAL_SUMMARY,
            hashtags=['#Learning', '#AI', '#Education', '#Knowledge'],
            optimal_platforms=[SocialPlatform.LINKEDIN.value, SocialPlatform.TWITTER.value],
            viral_score=fuzzy.FuzzyFloat(60.0, 80.0)
        )


class EngagementMilestoneFactory(BaseFactory):
    """Factory for EngagementMilestone model."""
    
    class Meta:
        model = EngagementMilestone
    
    milestone_name = factory.Faker('catch_phrase')
    description = factory.Faker('sentence', nb_words=10)
    metric_name = fuzzy.FuzzyChoice([
        'total_interactions', 'message_count', 'command_count',
        'positive_interactions', 'session_days', 'average_quality_score'
    ])
    target_value = fuzzy.FuzzyFloat(5.0, 100.0)
    
    # Gamification
    reward_points = fuzzy.FuzzyInteger(10, 500)
    badge_icon = factory.Faker('word')
    unlock_message = factory.Faker('sentence', nb_words=8)
    
    # Milestone configuration
    is_active = True
    is_repeatable = factory.Faker('boolean', chance_of_getting_true=30)
    difficulty_level = fuzzy.FuzzyChoice(['easy', 'medium', 'hard'])
    category = fuzzy.FuzzyChoice(['engagement', 'social', 'learning', 'achievement'])
    
    # Statistics
    total_achievements = fuzzy.FuzzyInteger(0, 1000)
    last_achieved_at = factory.Maybe(
        factory.Faker('boolean', chance_of_getting_true=50),
        yes_declaration=factory.Faker('date_time_between', start_date='-30d', end_date='now'),
        no_declaration=None
    )
    
    class Params:
        # Milestone difficulty traits
        easy_milestone = factory.Trait(
            difficulty_level='easy',
            target_value=fuzzy.FuzzyFloat(5.0, 20.0),
            reward_points=fuzzy.FuzzyInteger(10, 50),
            total_achievements=fuzzy.FuzzyInteger(100, 1000)
        )
        
        hard_milestone = factory.Trait(
            difficulty_level='hard',
            target_value=fuzzy.FuzzyFloat(50.0, 100.0),
            reward_points=fuzzy.FuzzyInteger(200, 500),
            total_achievements=fuzzy.FuzzyInteger(0, 50)
        )


class UserMilestoneProgressFactory(BaseFactory):
    """Factory for UserMilestoneProgress model."""
    
    class Meta:
        model = UserMilestoneProgress
    
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    milestone_id = fuzzy.FuzzyInteger(1, 100)
    telegram_id = fuzzy.FuzzyInteger(100000000, 999999999)
    
    # Progress tracking
    current_value = fuzzy.FuzzyFloat(0.0, 100.0)
    target_value = factory.LazyAttribute(
        lambda obj: obj.current_value + random.uniform(5.0, 50.0)
    )
    progress_percentage = factory.LazyAttribute(
        lambda obj: min(100.0, (obj.current_value / obj.target_value) * 100)
    )
    
    # Achievement status
    is_achieved = factory.LazyAttribute(lambda obj: obj.progress_percentage >= 100.0)
    achieved_at = factory.Maybe(
        'is_achieved',
        yes_declaration=factory.Faker('date_time_between', start_date='-30d', end_date='now'),
        no_declaration=None
    )
    
    # Progress metadata
    started_at = factory.Faker('date_time_between', start_date='-60d', end_date='-30d')
    last_updated_at = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    achievement_count = factory.LazyAttribute(lambda obj: 1 if obj.is_achieved else 0)
    
    class Params:
        # Progress state traits
        completed_milestone = factory.Trait(
            is_achieved=True,
            progress_percentage=100.0,
            current_value=factory.LazyAttribute(lambda obj: obj.target_value),
            achieved_at=factory.Faker('date_time_between', start_date='-30d', end_date='now')
        )
        
        in_progress = factory.Trait(
            is_achieved=False,
            progress_percentage=fuzzy.FuzzyFloat(10.0, 90.0),
            achieved_at=None
        )
        
        just_started = factory.Trait(
            progress_percentage=fuzzy.FuzzyFloat(0.0, 15.0),
            started_at=factory.Faker('date_time_between', start_date='-7d', end_date='now'),
            is_achieved=False
        )


# Batch factory for creating multiple related objects
class ConversationWithMessagesFactory(ConversationFactory):
    """Factory that creates a conversation with related messages."""
    
    @factory.post_generation
    def messages(self, create, extracted, **kwargs):
        if not create:
            return
        
        if extracted:
            # Use provided messages
            for message_data in extracted:
                MessageFactory(conversation_id=self.id, **message_data)
        else:
            # Create random number of messages
            message_count = random.randint(3, 15)
            for i in range(message_count):
                is_from_bot = i % 2 == 1  # Alternate between user and bot
                MessageFactory(
                    conversation_id=self.id,
                    is_from_bot=is_from_bot,
                    direction=MessageDirection.OUTGOING if is_from_bot else MessageDirection.INCOMING
                )


class UserWithEngagementFactory(UserFactory):
    """Factory that creates a user with engagement history."""
    
    @factory.post_generation
    def engagements(self, create, extracted, **kwargs):
        if not create:
            return
        
        engagement_count = extracted if extracted else random.randint(5, 30)
        for _ in range(engagement_count):
            UserEngagementFactory(
                user_id=str(self.id),
                telegram_id=self.telegram_id
            )


class GroupWithMembersFactory(GroupSessionFactory):
    """Factory that creates a group session with members."""
    
    @factory.post_generation
    def members(self, create, extracted, **kwargs):
        if not create:
            return
        
        member_count = extracted if extracted else random.randint(5, 20)
        for _ in range(member_count):
            GroupMemberFactory(group_id=self.id)


# Utility functions for test data generation
def create_test_scenario(scenario_type: str, count: int = 1, **kwargs):
    """Create test scenarios with predefined configurations."""
    
    scenarios = {
        'active_users': lambda: UserFactory.create_batch(count, is_active=True, **kwargs),
        'premium_users': lambda: UserFactory.create_batch(count, premium_user=True, **kwargs),
        'at_risk_users': lambda: UserWithEngagementFactory.create_batch(count, **kwargs),
        'group_conversations': lambda: GroupWithMembersFactory.create_batch(count, **kwargs),
        'viral_content': lambda: ShareableContentFactory.create_batch(count, high_viral_potential=True, **kwargs),
        'recent_conversations': lambda: ConversationWithMessagesFactory.create_batch(
            count, created_at=factory.Faker('date_time_between', start_date='-1d', end_date='now'), **kwargs
        )
    }
    
    if scenario_type not in scenarios:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    return scenarios[scenario_type]()


def create_realistic_test_data():
    """Create a realistic dataset for comprehensive testing."""
    # Create users with different characteristics
    users = {
        'active_users': UserFactory.create_batch(20, is_active=True),
        'premium_users': UserFactory.create_batch(5, premium_user=True),
        'inactive_users': UserFactory.create_batch(3, inactive_user=True)
    }
    
    # Create conversations and messages
    conversations = []
    for user in users['active_users'][:10]:
        conv = ConversationWithMessagesFactory(user_id=str(user.id))
        conversations.append(conv)
    
    # Create group sessions with members
    groups = GroupWithMembersFactory.create_batch(3, members=10)
    
    # Create engagement data
    engagements = []
    for user in users['active_users']:
        user_engagements = UserEngagementFactory.create_batch(
            random.randint(5, 25),
            user_id=str(user.id),
            telegram_id=user.telegram_id
        )
        engagements.extend(user_engagements)
    
    # Create viral content
    viral_content = ShareableContentFactory.create_batch(10)
    
    return {
        'users': users,
        'conversations': conversations,
        'groups': groups,
        'engagements': engagements,
        'viral_content': viral_content
    }