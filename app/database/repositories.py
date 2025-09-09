"""
Specific Repository Implementations

Concrete repository implementations for all model types with specialized
querying methods and business logic integration.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload, joinedload
import structlog

from app.database.repository import BaseRepository, QueryFilter, SortOrder, PaginationParams, QueryResult, FilterOperator
from app.database.connection import db_manager
from app.models import (
    User, Conversation, Message, ConversationSession,
    PersonalityProfile, PersonalityTrait, UserPersonalityMapping,
    RiskAssessment, RiskFactor, ConversationRisk,
    UserActivity, ConversationAnalytics, SystemMetrics,
    SystemConfiguration, FeatureFlag, RateLimitConfig,
    AuditLog, SecurityEvent, PerformanceMetric
)

logger = structlog.get_logger(__name__)


class UserRepository(BaseRepository[User]):
    """Repository for User model with specialized user operations."""
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_telegram_id(self, telegram_id: int, use_cache: bool = True) -> Optional[User]:
        """Get user by Telegram ID."""
        return await self.get_by_field("telegram_id", telegram_id, use_cache)
    
    async def get_by_username(self, username: str, use_cache: bool = True) -> Optional[User]:
        """Get user by username."""
        return await self.get_by_field("username", username, use_cache)
    
    async def find_active_users(
        self,
        pagination: PaginationParams = None,
        last_activity_days: int = 30
    ) -> QueryResult[User]:
        """Find users active within specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=last_activity_days)
        filters = [
            QueryFilter("is_active", FilterOperator.EQ, True),
            QueryFilter("updated_at", FilterOperator.GE, cutoff_date)
        ]
        
        return await self.find_by_filters(filters, pagination=pagination)
    
    async def get_user_statistics(self, user_id: Any) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        async with db_manager.get_async_session() as session:
            # Get user with related data
            result = await session.execute(
                select(User)
                .options(
                    selectinload(User.conversation_sessions),
                    selectinload(User.messages),
                    selectinload(User.activities)
                )
                .where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return {}
            
            # Calculate statistics
            total_sessions = len(user.conversation_sessions) if user.conversation_sessions else 0
            total_messages = len(user.messages) if user.messages else 0
            total_activities = len(user.activities) if user.activities else 0
            
            # Recent activity
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_messages = sum(
                1 for msg in (user.messages or [])
                if msg.created_at and msg.created_at >= recent_cutoff
            )
            
            return {
                "user_id": str(user.id),
                "telegram_id": user.telegram_id,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_activities": total_activities,
                "recent_messages_7d": recent_messages,
                "account_age_days": (datetime.utcnow() - user.created_at).days if user.created_at else 0,
                "is_active": user.is_active,
                "last_activity": user.updated_at.isoformat() if user.updated_at else None,
            }
    
    async def update_user_activity(self, telegram_id: int, activity_type: str) -> Optional[User]:
        """Update user activity tracking."""
        user = await self.get_by_telegram_id(telegram_id, use_cache=False)
        if not user:
            return None
        
        # Update activity tracking
        if not user.first_interaction:
            user.first_interaction = activity_type
        
        user.last_activity = activity_type
        user.updated_at = datetime.utcnow()
        
        if activity_type == "command":
            user.command_count += 1
        else:
            user.message_count += 1
        
        return await self.update(user.id, user.to_dict())


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for Conversation model with conversation-specific operations."""
    
    def __init__(self):
        super().__init__(Conversation)
    
    async def get_user_conversations(
        self,
        user_id: Any,
        status: str = None,
        pagination: PaginationParams = None
    ) -> QueryResult[Conversation]:
        """Get conversations for a user."""
        filters = [QueryFilter("user_id", FilterOperator.EQ, user_id)]
        
        if status:
            filters.append(QueryFilter("status", FilterOperator.EQ, status))
        
        sort_orders = [SortOrder("last_message_at", "desc")]
        
        return await self.find_by_filters(
            filters=filters,
            sort_orders=sort_orders,
            pagination=pagination
        )
    
    async def get_active_conversations(self, hours_threshold: int = 24) -> List[Conversation]:
        """Get conversations active within specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours_threshold)
        filters = [
            QueryFilter("status", FilterOperator.EQ, "active"),
            QueryFilter("last_message_at", FilterOperator.GE, cutoff)
        ]
        
        result = await self.find_by_filters(filters, use_cache=False)
        return result.items
    
    async def get_conversation_with_messages(self, conversation_id: Any, message_limit: int = 100) -> Optional[Conversation]:
        """Get conversation with recent messages."""
        async with db_manager.get_async_session() as session:
            result = await session.execute(
                select(Conversation)
                .options(
                    selectinload(Conversation.messages).options(
                        selectinload(Message.user)
                    ).limit(message_limit)
                )
                .where(Conversation.id == conversation_id)
            )
            return result.scalar_one_or_none()


class MessageRepository(BaseRepository[Message]):
    """Repository for Message model with message-specific operations."""
    
    def __init__(self):
        super().__init__(Message)
    
    async def get_conversation_messages(
        self,
        conversation_id: Any,
        pagination: PaginationParams = None,
        message_type: str = None
    ) -> QueryResult[Message]:
        """Get messages for a conversation."""
        filters = [QueryFilter("conversation_id", FilterOperator.EQ, conversation_id)]
        
        if message_type:
            filters.append(QueryFilter("message_type", FilterOperator.EQ, message_type))
        
        sort_orders = [SortOrder("created_at", "asc")]
        
        return await self.find_by_filters(
            filters=filters,
            sort_orders=sort_orders,
            pagination=pagination
        )
    
    async def get_user_recent_messages(
        self,
        user_id: Any,
        hours: int = 24,
        limit: int = 50
    ) -> List[Message]:
        """Get user's recent messages."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("created_at", FilterOperator.GE, cutoff)
        ]
        
        pagination = PaginationParams(page=1, size=limit)
        result = await self.find_by_filters(
            filters=filters,
            sort_orders=[SortOrder("created_at", "desc")],
            pagination=pagination
        )
        
        return result.items
    
    async def find_duplicate_messages(self, user_id: Any, content_hash: str) -> List[Message]:
        """Find potential duplicate messages."""
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("content_hash", FilterOperator.EQ, content_hash)
        ]
        
        result = await self.find_by_filters(filters)
        return result.items
    
    async def get_message_analytics(self, conversation_id: Any) -> Dict[str, Any]:
        """Get analytics for messages in a conversation."""
        async with db_manager.get_async_session() as session:
            # Get message statistics
            result = await session.execute(
                select([
                    func.count(Message.id).label('total_messages'),
                    func.count().filter(Message.direction == 'incoming').label('incoming_messages'),
                    func.count().filter(Message.direction == 'outgoing').label('outgoing_messages'),
                    func.avg(Message.response_time_ms).label('avg_response_time'),
                    func.avg(Message.sentiment_score).label('avg_sentiment'),
                    func.count().filter(Message.sentiment_label == 'positive').label('positive_messages'),
                    func.count().filter(Message.sentiment_label == 'negative').label('negative_messages'),
                    func.count().filter(Message.sentiment_label == 'neutral').label('neutral_messages'),
                ])
                .where(Message.conversation_id == conversation_id)
            )
            
            row = result.first()
            
            return {
                'total_messages': row.total_messages or 0,
                'incoming_messages': row.incoming_messages or 0,
                'outgoing_messages': row.outgoing_messages or 0,
                'avg_response_time_ms': float(row.avg_response_time) if row.avg_response_time else 0.0,
                'avg_sentiment': float(row.avg_sentiment) if row.avg_sentiment else 0.0,
                'sentiment_distribution': {
                    'positive': row.positive_messages or 0,
                    'negative': row.negative_messages or 0,
                    'neutral': row.neutral_messages or 0
                }
            }


class ConversationSessionRepository(BaseRepository[ConversationSession]):
    """Repository for ConversationSession model."""
    
    def __init__(self):
        super().__init__(ConversationSession)
    
    async def get_user_active_session(self, user_id: Any) -> Optional[ConversationSession]:
        """Get user's active session."""
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("status", FilterOperator.EQ, "active")
        ]
        
        result = await self.find_by_filters(filters, use_cache=True)
        return result.items[0] if result.items else None
    
    async def end_inactive_sessions(self, inactive_minutes: int = 30) -> int:
        """End sessions that have been inactive."""
        cutoff = datetime.utcnow() - timedelta(minutes=inactive_minutes)
        filters = [
            QueryFilter("status", FilterOperator.EQ, "active"),
            QueryFilter("last_activity_at", FilterOperator.LT, cutoff)
        ]
        
        # Get sessions to end
        result = await self.find_by_filters(filters, use_cache=False)
        
        # End each session
        ended_count = 0
        for session in result.items:
            session.end_session()
            await self.update(session.id, session.to_dict())
            ended_count += 1
        
        return ended_count


class PersonalityProfileRepository(BaseRepository[PersonalityProfile]):
    """Repository for PersonalityProfile model."""
    
    def __init__(self):
        super().__init__(PersonalityProfile)
    
    async def get_active_profiles(self) -> List[PersonalityProfile]:
        """Get all active personality profiles."""
        filters = [QueryFilter("is_active", FilterOperator.EQ, True)]
        result = await self.find_by_filters(filters, use_cache=True)
        return result.items
    
    async def get_default_profile(self) -> Optional[PersonalityProfile]:
        """Get the default personality profile."""
        filters = [
            QueryFilter("is_active", FilterOperator.EQ, True),
            QueryFilter("is_default", FilterOperator.EQ, True)
        ]
        
        result = await self.find_by_filters(filters, use_cache=True)
        return result.items[0] if result.items else None
    
    async def get_profiles_by_category(self, category: str) -> List[PersonalityProfile]:
        """Get profiles by category."""
        filters = [
            QueryFilter("is_active", FilterOperator.EQ, True),
            QueryFilter("category", FilterOperator.EQ, category)
        ]
        
        result = await self.find_by_filters(filters, use_cache=True)
        return result.items


class UserPersonalityMappingRepository(BaseRepository[UserPersonalityMapping]):
    """Repository for UserPersonalityMapping model."""
    
    def __init__(self):
        super().__init__(UserPersonalityMapping)
    
    async def get_user_primary_personality(self, user_id: Any) -> Optional[UserPersonalityMapping]:
        """Get user's primary personality mapping."""
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("is_primary", FilterOperator.EQ, True),
            QueryFilter("is_active", FilterOperator.EQ, True)
        ]
        
        result = await self.find_by_filters(filters, use_cache=True)
        return result.items[0] if result.items else None
    
    async def get_user_personalities(self, user_id: Any) -> List[UserPersonalityMapping]:
        """Get all personality mappings for a user."""
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("is_active", FilterOperator.EQ, True)
        ]
        
        sort_orders = [SortOrder("effectiveness_score", "desc")]
        result = await self.find_by_filters(filters, sort_orders=sort_orders)
        return result.items


class RiskAssessmentRepository(BaseRepository[RiskAssessment]):
    """Repository for RiskAssessment model."""
    
    def __init__(self):
        super().__init__(RiskAssessment)
    
    async def get_high_risk_assessments(
        self,
        hours: int = 24,
        min_risk_score: float = 0.7
    ) -> List[RiskAssessment]:
        """Get recent high-risk assessments."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filters = [
            QueryFilter("created_at", FilterOperator.GE, cutoff),
            QueryFilter("overall_risk_score", FilterOperator.GE, min_risk_score)
        ]
        
        sort_orders = [SortOrder("overall_risk_score", "desc")]
        result = await self.find_by_filters(filters, sort_orders=sort_orders)
        return result.items
    
    async def get_assessments_requiring_escalation(self) -> List[RiskAssessment]:
        """Get assessments that require escalation."""
        filters = [
            QueryFilter("requires_escalation", FilterOperator.EQ, True),
            QueryFilter("escalated_at", FilterOperator.IS_NULL)
        ]
        
        result = await self.find_by_filters(filters, use_cache=False)
        return result.items
    
    async def get_assessment_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get risk assessment statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        filters = [QueryFilter("created_at", FilterOperator.GE, cutoff)]
        
        async with db_manager.get_async_session() as session:
            result = await session.execute(
                select([
                    func.count(RiskAssessment.id).label('total_assessments'),
                    func.avg(RiskAssessment.overall_risk_score).label('avg_risk_score'),
                    func.count().filter(RiskAssessment.overall_risk_level == 'critical').label('critical_count'),
                    func.count().filter(RiskAssessment.overall_risk_level == 'high').label('high_count'),
                    func.count().filter(RiskAssessment.requires_escalation == True).label('escalation_count'),
                ])
                .where(RiskAssessment.created_at >= cutoff)
            )
            
            row = result.first()
            
            return {
                'total_assessments': row.total_assessments or 0,
                'avg_risk_score': float(row.avg_risk_score) if row.avg_risk_score else 0.0,
                'critical_count': row.critical_count or 0,
                'high_count': row.high_count or 0,
                'escalation_count': row.escalation_count or 0,
                'period_days': days
            }


class UserActivityRepository(BaseRepository[UserActivity]):
    """Repository for UserActivity model."""
    
    def __init__(self):
        super().__init__(UserActivity)
    
    async def get_user_activity_timeline(
        self,
        user_id: Any,
        days: int = 7
    ) -> List[UserActivity]:
        """Get user activity timeline."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("activity_timestamp", FilterOperator.GE, cutoff)
        ]
        
        sort_orders = [SortOrder("activity_timestamp", "desc")]
        result = await self.find_by_filters(filters, sort_orders=sort_orders)
        return result.items
    
    async def get_activity_statistics(
        self,
        user_id: Any = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get activity statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        filters = [QueryFilter("activity_timestamp", FilterOperator.GE, cutoff)]
        
        if user_id:
            filters.append(QueryFilter("user_id", FilterOperator.EQ, user_id))
        
        # Get activity type distribution
        activity_groups = await self.group_by_field(
            group_field="activity_type",
            agg_func="count",
            filters=filters
        )
        
        # Get success rate
        total_activities = await self.count_by_filters(filters)
        success_filters = filters + [QueryFilter("success", FilterOperator.EQ, True)]
        successful_activities = await self.count_by_filters(success_filters)
        
        success_rate = (successful_activities / total_activities) if total_activities > 0 else 0.0
        
        return {
            'total_activities': total_activities,
            'successful_activities': successful_activities,
            'success_rate': success_rate,
            'activity_breakdown': {item['group_value']: item['agg_value'] for item in activity_groups},
            'period_days': days
        }


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for AuditLog model."""
    
    def __init__(self):
        super().__init__(AuditLog)
    
    async def log_event(
        self,
        event_type: str,
        action: str,
        description: str,
        user_id: Any = None,
        **kwargs
    ) -> AuditLog:
        """Create an audit log entry."""
        audit_data = {
            'event_type': event_type,
            'event_category': kwargs.get('event_category', event_type),
            'action': action,
            'description': description,
            'user_id': user_id,
            'severity': kwargs.get('severity', 'info'),
            'success': kwargs.get('success', True),
            'ip_address': kwargs.get('ip_address'),
            'user_agent': kwargs.get('user_agent'),
            'event_data': kwargs.get('event_data'),
            'metadata': kwargs.get('metadata'),
            'resource_type': kwargs.get('resource_type'),
            'resource_id': kwargs.get('resource_id'),
        }
        
        return await self.create(audit_data)
    
    async def get_user_audit_trail(
        self,
        user_id: Any,
        days: int = 30,
        pagination: PaginationParams = None
    ) -> QueryResult[AuditLog]:
        """Get audit trail for a user."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        filters = [
            QueryFilter("user_id", FilterOperator.EQ, user_id),
            QueryFilter("timestamp", FilterOperator.GE, cutoff)
        ]
        
        sort_orders = [SortOrder("timestamp", "desc")]
        
        return await self.find_by_filters(
            filters=filters,
            sort_orders=sort_orders,
            pagination=pagination
        )
    
    async def get_security_events(
        self,
        hours: int = 24,
        severity: str = None
    ) -> List[AuditLog]:
        """Get security-related audit events."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filters = [
            QueryFilter("timestamp", FilterOperator.GE, cutoff),
            QueryFilter("event_category", FilterOperator.EQ, "security_event")
        ]
        
        if severity:
            filters.append(QueryFilter("severity", FilterOperator.EQ, severity))
        
        sort_orders = [SortOrder("timestamp", "desc")]
        result = await self.find_by_filters(filters, sort_orders=sort_orders)
        return result.items


class SecurityEventRepository(BaseRepository[SecurityEvent]):
    """Repository for SecurityEvent model."""
    
    def __init__(self):
        super().__init__(SecurityEvent)
    
    async def get_open_security_events(self) -> List[SecurityEvent]:
        """Get open security events requiring attention."""
        filters = [
            QueryFilter("status", FilterOperator.IN, ["open", "investigating"]),
            QueryFilter("false_positive", FilterOperator.EQ, False)
        ]
        
        sort_orders = [SortOrder("threat_level", "desc"), SortOrder("first_seen", "desc")]
        result = await self.find_by_filters(filters, sort_orders=sort_orders)
        return result.items
    
    async def get_events_by_source_ip(self, source_ip: str, hours: int = 24) -> List[SecurityEvent]:
        """Get security events from a specific IP."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filters = [
            QueryFilter("source_ip", FilterOperator.EQ, source_ip),
            QueryFilter("first_seen", FilterOperator.GE, cutoff)
        ]
        
        result = await self.find_by_filters(filters)
        return result.items
    
    async def get_threat_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get threat statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        filters = [QueryFilter("first_seen", FilterOperator.GE, cutoff)]
        
        # Get threat level distribution
        threat_groups = await self.group_by_field(
            group_field="threat_level",
            agg_func="count",
            filters=filters
        )
        
        # Get event type distribution
        event_groups = await self.group_by_field(
            group_field="event_type",
            agg_func="count",
            filters=filters
        )
        
        total_events = await self.count_by_filters(filters)
        
        return {
            'total_events': total_events,
            'threat_level_distribution': {item['group_value']: item['agg_value'] for item in threat_groups},
            'event_type_distribution': {item['group_value']: item['agg_value'] for item in event_groups},
            'period_days': days
        }


class SystemConfigurationRepository(BaseRepository[SystemConfiguration]):
    """Repository for SystemConfiguration model."""
    
    def __init__(self):
        super().__init__(SystemConfiguration)
    
    async def get_config_value(
        self,
        key: str,
        scope: str = "global",
        environment: str = "production"
    ) -> Any:
        """Get configuration value."""
        filters = [
            QueryFilter("key", FilterOperator.EQ, key),
            QueryFilter("scope", FilterOperator.EQ, scope),
            QueryFilter("environment", FilterOperator.EQ, environment),
            QueryFilter("is_active", FilterOperator.EQ, True)
        ]
        
        result = await self.find_by_filters(filters, use_cache=True)
        if result.items:
            config = result.items[0]
            config.record_access()
            await self.update(config.id, {"access_count": config.access_count, "last_accessed_at": config.last_accessed_at})
            return config.get_value()
        
        return None
    
    async def set_config_value(
        self,
        key: str,
        value: Any,
        scope: str = "global",
        environment: str = "production",
        change_reason: str = None
    ) -> SystemConfiguration:
        """Set configuration value."""
        # Try to find existing config
        existing = await self.get_by_field("key", key)
        
        if existing:
            existing.set_value(value, change_reason)
            return await self.update(existing.id, existing.to_dict())
        else:
            # Create new config
            config_data = {
                "key": key,
                "scope": scope,
                "environment": environment,
                "value_type": self._determine_value_type(value),
                "description": f"Configuration for {key}",
                "change_reason": change_reason or "Initial configuration"
            }
            
            config = await self.create(config_data)
            config.set_value(value, change_reason)
            return await self.update(config.id, config.to_dict())
    
    def _determine_value_type(self, value: Any) -> str:
        """Determine configuration value type."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, (dict, list)):
            return "json"
        else:
            return "string"


class FeatureFlagRepository(BaseRepository[FeatureFlag]):
    """Repository for FeatureFlag model."""
    
    def __init__(self):
        super().__init__(FeatureFlag)
    
    async def get_active_flags(self, environment: str = "production") -> List[FeatureFlag]:
        """Get active feature flags for environment."""
        filters = [
            QueryFilter("is_enabled", FilterOperator.EQ, True),
            QueryFilter("environment", FilterOperator.EQ, environment),
            QueryFilter("kill_switch_enabled", FilterOperator.EQ, False)
        ]
        
        result = await self.find_by_filters(filters, use_cache=True)
        return result.items
    
    async def is_flag_enabled(
        self,
        flag_key: str,
        user_id: str,
        user_attributes: Dict[str, Any] = None,
        environment: str = "production"
    ) -> bool:
        """Check if feature flag is enabled for user."""
        flag = await self.get_by_field("flag_key", flag_key, use_cache=True)
        
        if not flag or flag.environment != environment:
            return False
        
        return flag.is_enabled_for_user(user_id, user_attributes)
    
    async def get_flag_statistics(self) -> Dict[str, Any]:
        """Get feature flag usage statistics."""
        total_flags = await self.count_by_filters([])
        enabled_flags = await self.count_by_filters([QueryFilter("is_enabled", FilterOperator.EQ, True)])
        
        # Get flags by category
        category_groups = await self.group_by_field("category", "count")
        
        return {
            'total_flags': total_flags,
            'enabled_flags': enabled_flags,
            'disabled_flags': total_flags - enabled_flags,
            'category_distribution': {item['group_value']: item['agg_value'] for item in category_groups}
        }


# Repository factory for easy access
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    _repositories = {
        'user': UserRepository,
        'conversation': ConversationRepository,
        'message': MessageRepository,
        'conversation_session': ConversationSessionRepository,
        'personality_profile': PersonalityProfileRepository,
        'user_personality_mapping': UserPersonalityMappingRepository,
        'risk_assessment': RiskAssessmentRepository,
        'user_activity': UserActivityRepository,
        'audit_log': AuditLogRepository,
        'security_event': SecurityEventRepository,
        'system_configuration': SystemConfigurationRepository,
        'feature_flag': FeatureFlagRepository,
    }
    
    @classmethod
    def get_repository(cls, repo_name: str):
        """Get repository instance by name."""
        if repo_name not in cls._repositories:
            raise ValueError(f"Unknown repository: {repo_name}")
        
        return cls._repositories[repo_name]()
    
    @classmethod
    def get_all_repositories(cls) -> Dict[str, BaseRepository]:
        """Get all repository instances."""
        return {name: repo_cls() for name, repo_cls in cls._repositories.items()}


# Global repository instances for easy import
user_repo = UserRepository()
conversation_repo = ConversationRepository()
message_repo = MessageRepository()
session_repo = ConversationSessionRepository()
personality_repo = PersonalityProfileRepository()
user_personality_repo = UserPersonalityMappingRepository()
risk_repo = RiskAssessmentRepository()
activity_repo = UserActivityRepository()
audit_repo = AuditLogRepository()
security_repo = SecurityEventRepository()
config_repo = SystemConfigurationRepository()
feature_flag_repo = FeatureFlagRepository()