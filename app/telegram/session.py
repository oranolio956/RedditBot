"""
Session Management System

Handles user sessions, conversation state, and context tracking
with Redis persistence and automatic cleanup.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from datetime import datetime, timedelta

import structlog
from app.config import settings
from app.core.redis import (
    redis_manager, set_session, get_session, update_session, delete_session,
    publish_message, subscribe_channel, CompressionType
)

logger = structlog.get_logger(__name__)


class SessionState(Enum):
    """Session states."""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ConversationMode(Enum):
    """Conversation modes."""
    NORMAL = "normal"
    COMMAND = "command"
    MULTI_STEP = "multi_step"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class MessageContext:
    """Message context information."""
    message_id: int
    timestamp: float
    message_type: str
    content_type: str
    text_length: int = 0
    has_entities: bool = False
    has_media: bool = False
    is_command: bool = False
    command_name: Optional[str] = None
    reply_to_message_id: Optional[int] = None
    forward_from_user_id: Optional[int] = None


@dataclass
class UserProfile:
    """User profile data."""
    user_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    language_code: Optional[str] = None
    is_bot: bool = False
    is_premium: bool = False
    
    # Behavioral data
    preferred_response_style: str = "normal"
    timezone: Optional[str] = None
    active_hours: List[int] = field(default_factory=lambda: list(range(9, 22)))
    interaction_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationHistory:
    """Conversation history tracking."""
    messages: List[MessageContext] = field(default_factory=list)
    last_message_time: float = 0
    message_count: int = 0
    average_response_time: float = 0
    topics: List[str] = field(default_factory=list)
    sentiment_scores: List[float] = field(default_factory=list)
    
    # Limits
    max_history_size: int = 100
    history_ttl: int = 3600  # 1 hour


@dataclass
class SessionData:
    """Complete session data."""
    session_id: str
    user_id: int
    chat_id: int
    
    # Session metadata
    created_at: float
    last_activity: float
    state: SessionState = SessionState.ACTIVE
    conversation_mode: ConversationMode = ConversationMode.NORMAL
    
    # User data
    user_profile: UserProfile = None
    
    # Conversation tracking
    conversation_history: ConversationHistory = field(default_factory=ConversationHistory)
    
    # Context data
    current_context: Dict[str, Any] = field(default_factory=dict)
    persistent_context: Dict[str, Any] = field(default_factory=dict)
    temp_data: Dict[str, Any] = field(default_factory=dict)
    
    # Multi-step operations
    pending_operations: List[Dict[str, Any]] = field(default_factory=list)
    waiting_for_input: Optional[str] = None
    
    # Metrics
    total_interactions: int = 0
    total_response_time: float = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Rate limiting context
    rate_limit_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.user_profile is None:
            self.user_profile = UserProfile(user_id=self.user_id)


class SessionManager:
    """
    Comprehensive session management system.
    
    Features:
    - Redis-backed session persistence
    - Automatic session cleanup
    - Context tracking across conversations
    - User profile management
    - Multi-step operation support
    - Session analytics and metrics
    """
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.active_sessions: Dict[str, SessionData] = {}
        self.user_session_map: Dict[int, Set[str]] = {}  # user_id -> session_ids
        
        # Configuration
        self.session_ttl = settings.redis.session_ttl
        self.max_sessions_per_user = 5
        self.cleanup_interval = 300  # 5 minutes
        
        # Metrics
        self._total_sessions_created = 0
        self._total_sessions_expired = 0
        self._average_session_duration = 0
        
        # Background task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize session manager with Redis connection."""
        try:
            # Redis connection is managed by the redis_manager
            # Just verify it's healthy
            if not redis_manager.is_healthy:
                await redis_manager.initialize()
            
            # Subscribe to session events for multi-instance coordination
            await subscribe_channel(
                "session:events",
                handler=self._handle_session_event
            )
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Session manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize session manager", error=str(e))
            raise
    
    async def get_or_create_session(
        self,
        user_id: int,
        chat_id: int,
        user_data: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """Get existing session or create new one."""
        try:
            # Try to get existing session for this user/chat combination
            session = await self.get_session_by_user_chat(user_id, chat_id)
            
            if session:
                # Update last activity
                session.last_activity = time.time()
                session.state = SessionState.ACTIVE
                
                await self._save_session(session)
                return session
            
            # Create new session
            return await self.create_session(user_id, chat_id, user_data)
            
        except Exception as e:
            logger.error("Failed to get or create session", error=str(e))
            # Return minimal session as fallback
            return SessionData(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                chat_id=chat_id,
                created_at=time.time(),
                last_activity=time.time()
            )
    
    async def create_session(
        self,
        user_id: int,
        chat_id: int,
        user_data: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """Create new session."""
        try:
            # Generate session ID
            session_id = f"{user_id}:{chat_id}:{uuid.uuid4().hex[:8]}"
            
            # Create user profile
            user_profile = UserProfile(user_id=user_id)
            if user_data:
                for key, value in user_data.items():
                    if hasattr(user_profile, key):
                        setattr(user_profile, key, value)
            
            # Create session data
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "chat_id": chat_id,
                "created_at": time.time(),
                "last_activity": time.time(),
                "state": SessionState.ACTIVE.value,
                "conversation_mode": ConversationMode.NORMAL.value,
                "user_profile": {
                    "user_id": user_profile.user_id,
                    "username": user_profile.username,
                    "first_name": user_profile.first_name,
                    "last_name": user_profile.last_name,
                    "language_code": user_profile.language_code,
                    "is_bot": user_profile.is_bot,
                    "is_premium": user_profile.is_premium,
                    "preferred_response_style": user_profile.preferred_response_style,
                    "timezone": user_profile.timezone,
                    "active_hours": user_profile.active_hours,
                    "interaction_preferences": user_profile.interaction_preferences
                },
                "conversation_history": {
                    "messages": [],
                    "last_message_time": 0,
                    "message_count": 0,
                    "average_response_time": 0,
                    "topics": [],
                    "sentiment_scores": [],
                    "max_history_size": 100,
                    "history_ttl": 3600
                },
                "current_context": {},
                "persistent_context": {},
                "temp_data": {},
                "pending_operations": [],
                "waiting_for_input": None,
                "total_interactions": 0,
                "total_response_time": 0,
                "error_count": 0,
                "last_error": None,
                "rate_limit_data": {}
            }
            
            # Store session using Redis manager
            await set_session(
                session_id, 
                session_data, 
                ttl=self.session_ttl,
                notify_instances=True
            )
            
            # Create session object for local cache
            session = SessionData(
                session_id=session_id,
                user_id=user_id,
                chat_id=chat_id,
                created_at=session_data["created_at"],
                last_activity=session_data["last_activity"],
                user_profile=user_profile
            )
            
            # Store in memory cache
            self.active_sessions[session_id] = session
            
            # Update user session mapping
            if user_id not in self.user_session_map:
                self.user_session_map[user_id] = set()
            
            self.user_session_map[user_id].add(session_id)
            
            # Enforce session limit per user
            await self._enforce_session_limit(user_id)
            
            self._total_sessions_created += 1
            
            logger.info(f"Created new session {session_id} for user {user_id}")
            
            return session
            
        except Exception as e:
            logger.error("Failed to create session", error=str(e))
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        try:
            # Check memory cache first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Check if session is still valid
                if await self._is_session_valid(session):
                    return session
                else:
                    await self._expire_session(session)
                    return None
            
            # Load from Redis using enhanced session manager
            session_data = await get_session(
                session_id, 
                extend_ttl=True,
                ttl_extension=self.session_ttl
            )
            
            if session_data:
                # Convert to SessionData object
                session = await self._dict_to_session(session_data)
                
                if await self._is_session_valid(session):
                    self.active_sessions[session_id] = session
                    return session
                else:
                    await self._expire_session(session)
            
            return None
            
        except Exception as e:
            logger.error("Failed to get session", session_id=session_id, error=str(e))
            return None
    
    async def get_session_by_user_chat(
        self,
        user_id: int,
        chat_id: int
    ) -> Optional[SessionData]:
        """Get session by user and chat ID."""
        try:
            # Check user sessions
            if user_id in self.user_session_map:
                for session_id in self.user_session_map[user_id]:
                    session = await self.get_session(session_id)
                    if session and session.chat_id == chat_id:
                        return session
            
            # Search in Redis if not in memory
            # Use the enhanced session listing functionality
            session_keys = await redis_manager.list_sessions(
                pattern=f"session:{user_id}:{chat_id}:*",
                limit=10
            )
            
            for session_id in session_keys:
                session_data = await get_session(session_id)
                if session_data:
                    session = await self._dict_to_session(session_data)
                    if await self._is_session_valid(session):
                        self.active_sessions[session_id] = session
                        return session
            
            return None
            
        except Exception as e:
            logger.error("Failed to get session by user/chat", error=str(e))
            return None
    
    async def update_session_activity(
        self,
        session_id: str,
        message_context: Optional[MessageContext] = None,
        context_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update session activity and context."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Update activity timestamp
            session.last_activity = time.time()
            session.total_interactions += 1
            
            # Add message to history
            if message_context:
                session.conversation_history.messages.append(message_context)
                session.conversation_history.message_count += 1
                session.conversation_history.last_message_time = time.time()
                
                # Maintain history size limit
                max_size = session.conversation_history.max_history_size
                if len(session.conversation_history.messages) > max_size:
                    session.conversation_history.messages = (
                        session.conversation_history.messages[-max_size:]
                    )
            
            # Update context
            if context_updates:
                session.current_context.update(context_updates)
            
            # Save changes
            await self._save_session(session)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update session activity", error=str(e))
            return False
    
    async def set_conversation_mode(
        self,
        session_id: str,
        mode: ConversationMode,
        waiting_for: Optional[str] = None
    ) -> bool:
        """Set conversation mode for session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.conversation_mode = mode
            session.waiting_for_input = waiting_for
            
            await self._save_session(session)
            
            logger.info(
                f"Set conversation mode {mode.value} for session {session_id}",
                waiting_for=waiting_for
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to set conversation mode", error=str(e))
            return False
    
    async def add_pending_operation(
        self,
        session_id: str,
        operation: Dict[str, Any]
    ) -> bool:
        """Add pending operation to session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Add operation with timestamp
            operation["added_at"] = time.time()
            operation["operation_id"] = str(uuid.uuid4())
            
            session.pending_operations.append(operation)
            
            await self._save_session(session)
            
            return True
            
        except Exception as e:
            logger.error("Failed to add pending operation", error=str(e))
            return False
    
    async def get_pending_operations(
        self,
        session_id: str,
        operation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending operations for session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return []
            
            operations = session.pending_operations
            
            if operation_type:
                operations = [
                    op for op in operations
                    if op.get("type") == operation_type
                ]
            
            return operations
            
        except Exception as e:
            logger.error("Failed to get pending operations", error=str(e))
            return []
    
    async def complete_operation(
        self,
        session_id: str,
        operation_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Complete and remove a pending operation."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Find and remove operation
            for i, operation in enumerate(session.pending_operations):
                if operation.get("operation_id") == operation_id:
                    completed_op = session.pending_operations.pop(i)
                    
                    # Store result in context if provided
                    if result:
                        session.current_context[f"completed_{operation_id}"] = result
                    
                    await self._save_session(session)
                    
                    logger.info(f"Completed operation {operation_id} in session {session_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to complete operation", error=str(e))
            return False
    
    async def set_persistent_context(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> bool:
        """Set persistent context data."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.persistent_context[key] = value
            await self._save_session(session)
            
            return True
            
        except Exception as e:
            logger.error("Failed to set persistent context", error=str(e))
            return False
    
    async def get_persistent_context(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get persistent context data."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return default
            
            return session.persistent_context.get(key, default)
            
        except Exception as e:
            logger.error("Failed to get persistent context", error=str(e))
            return default
    
    async def get_user_sessions(self, user_id: int) -> List[SessionData]:
        """Get all active sessions for a user."""
        try:
            sessions = []
            
            if user_id in self.user_session_map:
                for session_id in self.user_session_map[user_id].copy():
                    session = await self.get_session(session_id)
                    if session:
                        sessions.append(session)
                    else:
                        # Clean up invalid session reference
                        self.user_session_map[user_id].discard(session_id)
            
            return sessions
            
        except Exception as e:
            logger.error("Failed to get user sessions", error=str(e))
            return []
    
    async def expire_session(self, session_id: str) -> bool:
        """Manually expire a session."""
        try:
            session = await self.get_session(session_id)
            if session:
                await self._expire_session(session)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to expire session", error=str(e))
            return False
    
    async def _save_session(self, session: SessionData) -> None:
        """Save session to Redis."""
        try:
            # Convert session to dict for Redis storage
            session_dict = await self._session_to_dict(session)
            
            # Use enhanced Redis session manager
            await set_session(
                session.session_id,
                session_dict,
                ttl=self.session_ttl,
                notify_instances=True
            )
            
        except Exception as e:
            logger.error("Failed to save session to Redis", error=str(e))
            raise
    
    async def _session_to_dict(self, session: SessionData) -> Dict[str, Any]:
        """Convert SessionData to dictionary for Redis storage."""
        try:
            session_dict = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "chat_id": session.chat_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "state": session.state.value if hasattr(session.state, 'value') else str(session.state),
                "conversation_mode": session.conversation_mode.value if hasattr(session.conversation_mode, 'value') else str(session.conversation_mode),
                "total_interactions": session.total_interactions,
                "total_response_time": session.total_response_time,
                "error_count": session.error_count,
                "last_error": session.last_error,
                "waiting_for_input": session.waiting_for_input,
                "current_context": session.current_context,
                "persistent_context": session.persistent_context,
                "temp_data": session.temp_data,
                "pending_operations": session.pending_operations,
                "rate_limit_data": session.rate_limit_data
            }
            
            # Handle user profile
            if session.user_profile:
                session_dict["user_profile"] = {
                    "user_id": session.user_profile.user_id,
                    "username": session.user_profile.username,
                    "first_name": session.user_profile.first_name,
                    "last_name": session.user_profile.last_name,
                    "language_code": session.user_profile.language_code,
                    "is_bot": session.user_profile.is_bot,
                    "is_premium": session.user_profile.is_premium,
                    "preferred_response_style": session.user_profile.preferred_response_style,
                    "timezone": session.user_profile.timezone,
                    "active_hours": session.user_profile.active_hours,
                    "interaction_preferences": session.user_profile.interaction_preferences
                }
            
            # Handle conversation history
            if session.conversation_history:
                messages = []
                for msg in session.conversation_history.messages:
                    if hasattr(msg, '__dict__'):
                        messages.append({
                            "message_id": msg.message_id,
                            "timestamp": msg.timestamp,
                            "message_type": msg.message_type,
                            "content_type": msg.content_type,
                            "text_length": msg.text_length,
                            "has_entities": msg.has_entities,
                            "has_media": msg.has_media,
                            "is_command": msg.is_command,
                            "command_name": msg.command_name,
                            "reply_to_message_id": msg.reply_to_message_id,
                            "forward_from_user_id": msg.forward_from_user_id
                        })
                    else:
                        messages.append(msg)
                
                session_dict["conversation_history"] = {
                    "messages": messages,
                    "last_message_time": session.conversation_history.last_message_time,
                    "message_count": session.conversation_history.message_count,
                    "average_response_time": session.conversation_history.average_response_time,
                    "topics": session.conversation_history.topics,
                    "sentiment_scores": session.conversation_history.sentiment_scores,
                    "max_history_size": session.conversation_history.max_history_size,
                    "history_ttl": session.conversation_history.history_ttl
                }
            
            return session_dict
            
        except Exception as e:
            logger.error("Failed to convert session to dict", error=str(e))
            raise
    
    async def _dict_to_session(self, session_dict: Dict[str, Any]) -> SessionData:
        """Convert dictionary to SessionData object."""
        try:
            # Create user profile
            user_profile = None
            if "user_profile" in session_dict and session_dict["user_profile"]:
                profile_data = session_dict["user_profile"]
                user_profile = UserProfile(
                    user_id=profile_data.get("user_id", 0),
                    username=profile_data.get("username"),
                    first_name=profile_data.get("first_name"),
                    last_name=profile_data.get("last_name"),
                    language_code=profile_data.get("language_code"),
                    is_bot=profile_data.get("is_bot", False),
                    is_premium=profile_data.get("is_premium", False),
                    preferred_response_style=profile_data.get("preferred_response_style", "normal"),
                    timezone=profile_data.get("timezone"),
                    active_hours=profile_data.get("active_hours", list(range(9, 22))),
                    interaction_preferences=profile_data.get("interaction_preferences", {})
                )
            
            # Create conversation history
            conversation_history = ConversationHistory()
            if "conversation_history" in session_dict and session_dict["conversation_history"]:
                hist_data = session_dict["conversation_history"]
                
                # Convert messages
                messages = []
                for msg_data in hist_data.get("messages", []):
                    if isinstance(msg_data, dict):
                        messages.append(MessageContext(
                            message_id=msg_data.get("message_id", 0),
                            timestamp=msg_data.get("timestamp", time.time()),
                            message_type=msg_data.get("message_type", "text"),
                            content_type=msg_data.get("content_type", "text"),
                            text_length=msg_data.get("text_length", 0),
                            has_entities=msg_data.get("has_entities", False),
                            has_media=msg_data.get("has_media", False),
                            is_command=msg_data.get("is_command", False),
                            command_name=msg_data.get("command_name"),
                            reply_to_message_id=msg_data.get("reply_to_message_id"),
                            forward_from_user_id=msg_data.get("forward_from_user_id")
                        ))
                
                conversation_history = ConversationHistory(
                    messages=messages,
                    last_message_time=hist_data.get("last_message_time", 0),
                    message_count=hist_data.get("message_count", 0),
                    average_response_time=hist_data.get("average_response_time", 0),
                    topics=hist_data.get("topics", []),
                    sentiment_scores=hist_data.get("sentiment_scores", []),
                    max_history_size=hist_data.get("max_history_size", 100),
                    history_ttl=hist_data.get("history_ttl", 3600)
                )
            
            # Create session
            session = SessionData(
                session_id=session_dict.get("session_id", ""),
                user_id=session_dict.get("user_id", 0),
                chat_id=session_dict.get("chat_id", 0),
                created_at=session_dict.get("created_at", time.time()),
                last_activity=session_dict.get("last_activity", time.time()),
                state=SessionState(session_dict.get("state", SessionState.ACTIVE.value)),
                conversation_mode=ConversationMode(session_dict.get("conversation_mode", ConversationMode.NORMAL.value)),
                user_profile=user_profile,
                conversation_history=conversation_history,
                current_context=session_dict.get("current_context", {}),
                persistent_context=session_dict.get("persistent_context", {}),
                temp_data=session_dict.get("temp_data", {}),
                pending_operations=session_dict.get("pending_operations", []),
                waiting_for_input=session_dict.get("waiting_for_input"),
                total_interactions=session_dict.get("total_interactions", 0),
                total_response_time=session_dict.get("total_response_time", 0),
                error_count=session_dict.get("error_count", 0),
                last_error=session_dict.get("last_error"),
                rate_limit_data=session_dict.get("rate_limit_data", {})
            )
            
            return session
            
        except Exception as e:
            logger.error("Failed to convert dict to session", error=str(e))
            raise
    
    async def _is_session_valid(self, session: SessionData) -> bool:
        """Check if session is still valid."""
        now = time.time()
        
        # Check if expired
        if now - session.last_activity > self.session_ttl:
            return False
        
        # Check state
        if session.state in [SessionState.EXPIRED, SessionState.TERMINATED]:
            return False
        
        return True
    
    async def _expire_session(self, session: SessionData) -> None:
        """Expire a session."""
        try:
            session.state = SessionState.EXPIRED
            
            # Remove from active sessions
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            # Remove from user mapping
            if session.user_id in self.user_session_map:
                self.user_session_map[session.user_id].discard(session.session_id)
                
                # Clean up empty user mapping
                if not self.user_session_map[session.user_id]:
                    del self.user_session_map[session.user_id]
            
            # Remove from Redis using enhanced session manager
            await delete_session(session.session_id, notify_instances=True)
            
            # Update metrics
            self._total_sessions_expired += 1
            duration = time.time() - session.created_at
            self._average_session_duration = (
                (self._average_session_duration * (self._total_sessions_expired - 1) + duration)
                / self._total_sessions_expired
            )
            
            logger.info(f"Expired session {session.session_id}")
            
        except Exception as e:
            logger.error("Failed to expire session", error=str(e))
    
    async def _enforce_session_limit(self, user_id: int) -> None:
        """Enforce maximum sessions per user."""
        try:
            if user_id not in self.user_session_map:
                return
            
            user_sessions = list(self.user_session_map[user_id])
            
            if len(user_sessions) > self.max_sessions_per_user:
                # Get sessions with activity times
                session_activities = []
                for session_id in user_sessions:
                    session = await self.get_session(session_id)
                    if session:
                        session_activities.append((session_id, session.last_activity))
                
                # Sort by activity (oldest first)
                session_activities.sort(key=lambda x: x[1])
                
                # Expire oldest sessions
                sessions_to_expire = len(session_activities) - self.max_sessions_per_user
                for i in range(sessions_to_expire):
                    session_id, _ = session_activities[i]
                    session = await self.get_session(session_id)
                    if session:
                        await self._expire_session(session)
                
                logger.info(f"Expired {sessions_to_expire} old sessions for user {user_id}")
            
        except Exception as e:
            logger.error("Failed to enforce session limit", error=str(e))
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            cleaned_count = 0
            now = time.time()
            
            # Check active sessions in memory
            expired_session_ids = []
            for session_id, session in self.active_sessions.items():
                if not await self._is_session_valid(session):
                    expired_session_ids.append(session_id)
            
            # Expire them
            for session_id in expired_session_ids:
                session = self.active_sessions.get(session_id)
                if session:
                    await self._expire_session(session)
                    cleaned_count += 1
            
            # Use Redis manager cleanup for expired sessions
            redis_cleaned = await redis_manager.cleanup_expired_sessions(batch_size=50)
            cleaned_count += redis_cleaned
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
            return 0
    
    async def update_session(self, session: SessionData) -> bool:
        """Update session data."""
        try:
            session.last_activity = time.time()
            
            # Update in memory
            self.active_sessions[session.session_id] = session
            
            # Save to Redis using enhanced session manager
            session_dict = await self._session_to_dict(session)
            await update_session(
                session.session_id,
                session_dict,
                ttl_extend=self.session_ttl
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to update session", session_id=session.session_id, error=str(e))
            return False
    
    async def _handle_session_event(self, channel: str, event_data: Dict[str, Any]) -> None:
        """Handle session events from other instances."""
        try:
            event_type = event_data.get("event_type")
            session_id = event_data.get("session_id")
            
            if event_type == "session_updated":
                # Invalidate local cache to force reload
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                    logger.debug(f"Invalidated local cache for session {session_id}")
            
            elif event_type == "session_deleted":
                # Remove from local cache
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    del self.active_sessions[session_id]
                    
                    # Update user session mapping
                    if session.user_id in self.user_session_map:
                        self.user_session_map[session.user_id].discard(session_id)
                        if not self.user_session_map[session.user_id]:
                            del self.user_session_map[session.user_id]
                    
                    logger.debug(f"Removed session {session_id} from local cache")
            
        except Exception as e:
            logger.error("Failed to handle session event", error=str(e), event_data=event_data)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get session manager metrics."""
        try:
            active_sessions_count = len(self.active_sessions)
            total_users = len(self.user_session_map)
            
            # Session state distribution
            state_distribution = {}
            conversation_mode_distribution = {}
            
            for session in self.active_sessions.values():
                state = session.state.value
                mode = session.conversation_mode.value
                
                state_distribution[state] = state_distribution.get(state, 0) + 1
                conversation_mode_distribution[mode] = conversation_mode_distribution.get(mode, 0) + 1
            
            # Average session metrics
            total_interactions = sum(s.total_interactions for s in self.active_sessions.values())
            avg_interactions_per_session = (
                total_interactions / max(1, active_sessions_count)
            )
            
            return {
                "active_sessions": active_sessions_count,
                "total_users": total_users,
                "total_sessions_created": self._total_sessions_created,
                "total_sessions_expired": self._total_sessions_expired,
                "average_session_duration": self._average_session_duration,
                "average_interactions_per_session": avg_interactions_per_session,
                "total_interactions": total_interactions,
                "state_distribution": state_distribution,
                "conversation_mode_distribution": conversation_mode_distribution,
                "session_ttl": self.session_ttl,
                "max_sessions_per_user": self.max_sessions_per_user,
            }
            
        except Exception as e:
            logger.error("Failed to get session metrics", error=str(e))
            return {}
    
    async def cleanup(self) -> None:
        """Clean up session manager resources."""
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Redis connection is managed by redis_manager, no need to close
            
            # Clear memory
            self.active_sessions.clear()
            self.user_session_map.clear()
            
            logger.info("Session manager cleanup completed")
            
        except Exception as e:
            logger.error("Error during session manager cleanup", error=str(e))