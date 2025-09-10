"""
Group Chat Handlers

Comprehensive group message handling system with multi-user context management,
admin commands, permissions, and intelligent conversation flow for group chats.
"""

import asyncio
import time
import re
import json
import uuid
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict

import structlog
from aiogram import Router, F, Bot
from aiogram.types import (
    Message, CallbackQuery, ChatMemberUpdated, 
    InlineKeyboardMarkup, InlineKeyboardButton,
    BotCommand, BotCommandScopeChat, BotCommandScopeAllGroupChats,
    ChatPermissions, ChatMemberOwner, ChatMemberAdministrator,
    ChatMemberMember, ChatMemberRestricted, ChatMemberLeft, ChatMemberBanned
)
from aiogram.filters import Command, StateFilter, ChatMemberUpdatedFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from .session import SessionManager, MessageContext
from .anti_ban import AntiBanManager, RiskLevel
from .metrics import TelegramMetrics
from .rate_limiter import AdvancedRateLimiter
from ..models.group_session import (
    GroupSession, GroupMember, GroupConversation, GroupAnalytics,
    GroupType, MemberRole, GroupStatus, MessageFrequency
)
from ..models.user import User
from ..database.connection import get_async_session

logger = structlog.get_logger(__name__)


class GroupStates(StatesGroup):
    """Group-specific FSM states."""
    idle = State()
    admin_action = State()
    moderation_review = State()
    settings_config = State()
    bulk_operation = State()


class GroupMentionDetector:
    """Utility class for detecting and parsing bot mentions in group messages."""
    
    def __init__(self, bot_username: str, bot_id: int):
        self.bot_username = bot_username.lower().replace('@', '')
        self.bot_id = bot_id
        
        # Compile regex patterns for efficient mention detection
        self.username_pattern = re.compile(
            rf'@{re.escape(self.bot_username)}\b', 
            re.IGNORECASE
        )
        self.reply_pattern = re.compile(r'^(/\w+)(@\w+)?\s*', re.IGNORECASE)
    
    def is_bot_mentioned(self, message: Message) -> bool:
        """Check if the bot is mentioned in the message."""
        # Check if replying to bot
        if (message.reply_to_message and 
            message.reply_to_message.from_user and 
            message.reply_to_message.from_user.id == self.bot_id):
            return True
        
        # Check for username mentions in text
        if message.text and self.username_pattern.search(message.text):
            return True
        
        # Check for username mentions in entities
        if message.entities:
            for entity in message.entities:
                if entity.type == "mention":
                    mentioned_username = message.text[entity.offset:entity.offset + entity.length]
                    if mentioned_username.lower() == f"@{self.bot_username}":
                        return True
                elif entity.type == "text_mention" and entity.user:
                    if entity.user.id == self.bot_id:
                        return True
        
        return False
    
    def extract_command_from_mention(self, message: Message) -> Optional[str]:
        """Extract command from message that mentions the bot."""
        if not message.text:
            return None
        
        # Check for command pattern with optional bot username
        match = self.reply_pattern.match(message.text)
        if match:
            command = match.group(1)[1:]  # Remove '/' prefix
            mentioned_bot = match.group(2)
            
            # If bot is specifically mentioned or no bot mentioned in group
            if not mentioned_bot or mentioned_bot.lower() == f"@{self.bot_username}":
                return command
        
        # Check for commands after mentions
        text = message.text.lower()
        if f"@{self.bot_username}" in text:
            # Look for commands after the mention
            parts = text.split()
            for i, part in enumerate(parts):
                if f"@{self.bot_username}" in part and i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part.startswith('/'):
                        return next_part[1:]  # Remove '/' prefix
        
        return None
    
    def clean_message_text(self, message: Message) -> str:
        """Remove bot mentions from message text for cleaner processing."""
        if not message.text:
            return ""
        
        # Remove username mentions
        cleaned = self.username_pattern.sub('', message.text)
        
        # Remove command mentions
        cleaned = self.reply_pattern.sub('', cleaned)
        
        # Clean up extra whitespace
        return re.sub(r'\s+', ' ', cleaned).strip()


class GroupPermissionManager:
    """Manages group permissions and role-based access control."""
    
    def __init__(self):
        self.permission_cache: Dict[int, Dict[str, Any]] = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def check_admin_permissions(
        self, 
        bot: Bot, 
        chat_id: int, 
        user_id: int,
        required_permission: str = None
    ) -> bool:
        """Check if user has admin permissions in the group."""
        try:
            # Check cache first
            cache_key = f"{chat_id}:{user_id}"
            cached = self.permission_cache.get(cache_key)
            
            if cached and time.time() - cached['timestamp'] < self.cache_timeout:
                permissions = cached['permissions']
            else:
                # Fetch fresh permissions
                member = await bot.get_chat_member(chat_id, user_id)
                permissions = self._extract_permissions(member)
                
                # Cache the result
                self.permission_cache[cache_key] = {
                    'permissions': permissions,
                    'timestamp': time.time()
                }
            
            # Check general admin status
            if not permissions.get('is_admin', False):
                return False
            
            # Check specific permission if required
            if required_permission:
                return permissions.get(required_permission, False)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check admin permissions: {e}")
            return False
    
    def _extract_permissions(self, member) -> Dict[str, Any]:
        """Extract permissions from ChatMember object."""
        permissions = {'is_admin': False}
        
        if isinstance(member, (ChatMemberOwner, ChatMemberAdministrator)):
            permissions['is_admin'] = True
            permissions['can_delete_messages'] = getattr(member, 'can_delete_messages', False)
            permissions['can_restrict_members'] = getattr(member, 'can_restrict_members', False)
            permissions['can_promote_members'] = getattr(member, 'can_promote_members', False)
            permissions['can_change_info'] = getattr(member, 'can_change_info', False)
            permissions['can_invite_users'] = getattr(member, 'can_invite_users', False)
            permissions['can_pin_messages'] = getattr(member, 'can_pin_messages', False)
            permissions['can_manage_voice_chats'] = getattr(member, 'can_manage_voice_chats', False)
            permissions['is_owner'] = isinstance(member, ChatMemberOwner)
        
        return permissions
    
    def clear_cache_for_chat(self, chat_id: int) -> None:
        """Clear permission cache for a specific chat."""
        keys_to_remove = [key for key in self.permission_cache.keys() if key.startswith(f"{chat_id}:")]
        for key in keys_to_remove:
            del self.permission_cache[key]


class GroupRateLimiter:
    """Advanced rate limiting specifically designed for group chats."""
    
    def __init__(self):
        self.group_counters: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.user_counters: Dict[tuple, List[float]] = defaultdict(list)
        self.violation_counts: Dict[int, int] = defaultdict(int)
        
        # Rate limit configurations
        self.limits = {
            MessageFrequency.LOW: {
                "messages_per_hour": 10,
                "mentions_per_hour": 3,
                "commands_per_hour": 5
            },
            MessageFrequency.MODERATE: {
                "messages_per_hour": 30,
                "mentions_per_hour": 10,
                "commands_per_hour": 15
            },
            MessageFrequency.HIGH: {
                "messages_per_hour": 60,
                "mentions_per_hour": 20,
                "commands_per_hour": 30
            },
            MessageFrequency.VERY_HIGH: {
                "messages_per_hour": 120,
                "mentions_per_hour": 40,
                "commands_per_hour": 60
            }
        }
    
    async def check_rate_limit(
        self,
        group_session: GroupSession,
        user_id: int,
        action_type: str = "message"
    ) -> tuple[bool, Optional[str]]:
        """Check if action is within rate limits."""
        try:
            now = time.time()
            chat_id = group_session.telegram_chat_id
            
            # Get current limits for the group
            frequency = group_session.message_frequency
            limits = self.limits.get(frequency, self.limits[MessageFrequency.MODERATE])
            
            # Clean old entries (older than 1 hour)
            self._clean_old_entries(now)
            
            # Check group-wide limits
            if not await self._check_group_limits(chat_id, action_type, limits, now):
                return False, f"Group {action_type} limit exceeded"
            
            # Check user-specific limits in this group
            user_key = (chat_id, user_id)
            if not await self._check_user_limits(user_key, action_type, limits, now):
                return False, f"User {action_type} limit exceeded"
            
            # Record the action
            self.group_counters[chat_id][action_type].append(now)
            self.user_counters[user_key].append(now)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True, None  # Allow on error to avoid breaking functionality
    
    async def _check_group_limits(
        self,
        chat_id: int,
        action_type: str,
        limits: Dict[str, int],
        now: float
    ) -> bool:
        """Check group-wide rate limits."""
        if action_type == "mention":
            limit_key = "mentions_per_hour"
        elif action_type == "command":
            limit_key = "commands_per_hour"
        else:
            limit_key = "messages_per_hour"
        
        limit = limits.get(limit_key, 30)  # Default limit
        
        # Count recent actions
        recent_actions = [t for t in self.group_counters[chat_id][action_type] if now - t < 3600]
        
        return len(recent_actions) < limit
    
    async def _check_user_limits(
        self,
        user_key: tuple,
        action_type: str,
        limits: Dict[str, int],
        now: float
    ) -> bool:
        """Check user-specific rate limits."""
        # User limits are 50% of group limits for fairness
        if action_type == "mention":
            limit = max(1, limits.get("mentions_per_hour", 10) // 2)
        elif action_type == "command":
            limit = max(1, limits.get("commands_per_hour", 15) // 2)
        else:
            limit = max(2, limits.get("messages_per_hour", 30) // 2)
        
        # Count recent actions
        recent_actions = [t for t in self.user_counters[user_key] if now - t < 3600]
        
        return len(recent_actions) < limit
    
    def _clean_old_entries(self, now: float) -> None:
        """Clean entries older than 1 hour."""
        cutoff = now - 3600  # 1 hour ago
        
        # Clean group counters
        for chat_id in list(self.group_counters.keys()):
            for action_type in list(self.group_counters[chat_id].keys()):
                self.group_counters[chat_id][action_type] = [
                    t for t in self.group_counters[chat_id][action_type] if t > cutoff
                ]
                
                # Remove empty action lists
                if not self.group_counters[chat_id][action_type]:
                    del self.group_counters[chat_id][action_type]
            
            # Remove empty chat entries
            if not self.group_counters[chat_id]:
                del self.group_counters[chat_id]
        
        # Clean user counters
        for user_key in list(self.user_counters.keys()):
            self.user_counters[user_key] = [
                t for t in self.user_counters[user_key] if t > cutoff
            ]
            
            # Remove empty user entries
            if not self.user_counters[user_key]:
                del self.user_counters[user_key]
    
    def add_violation(self, chat_id: int) -> None:
        """Record a rate limit violation."""
        self.violation_counts[chat_id] += 1
    
    def get_violation_count(self, chat_id: int) -> int:
        """Get violation count for a group."""
        return self.violation_counts.get(chat_id, 0)


class GroupHandlers:
    """
    Comprehensive group chat handling system.
    
    Features:
    - Multi-user context management
    - @mention detection and intelligent responses
    - Admin commands and permissions
    - Group settings and configuration
    - Advanced rate limiting and anti-spam
    - Member analytics and behavior tracking
    - Thread-aware conversation management
    """
    
    def __init__(self, bot: Bot):
        self.bot = bot
        self.session_manager: Optional[SessionManager] = None
        self.anti_ban: Optional[AntiBanManager] = None
        self.metrics: Optional[TelegramMetrics] = None
        self.rate_limiter: Optional[AdvancedRateLimiter] = None
        
        # Group-specific components
        self.mention_detector: Optional[GroupMentionDetector] = None
        self.permission_manager = GroupPermissionManager()
        self.group_rate_limiter = GroupRateLimiter()
        
        # Router for organizing handlers
        self.router = Router()
        
        # In-memory caches for performance
        self.group_cache: Dict[int, GroupSession] = {}
        self.member_cache: Dict[tuple, GroupMember] = {}
        self.conversation_cache: Dict[str, GroupConversation] = {}
        
        # Active conversation tracking
        self.active_conversations: Dict[int, Set[str]] = defaultdict(set)
        
        # Response templates
        self.templates = {
            'group_welcome': "👋 Hello {group_title}! I'm ready to help. Mention me (@{bot_username}) to interact!",
            'admin_only': "🔒 This command is only available to group administrators.",
            'rate_limited': "⏳ Slow down! You're sending messages too quickly.",
            'group_rate_limited': "⏳ This group has reached its message limit. Please wait a moment.",
            'mention_response': "👋 Hi {user_name}! How can I help you?",
            'group_settings_updated': "✅ Group settings have been updated successfully.",
            'member_added': "👋 Welcome {user_name} to {group_title}!",
            'member_left': "👋 {user_name} has left the group. We'll miss you!",
            'conversation_started': "💬 New conversation topic detected: {topic}",
            'help_group': """
🤖 **Group Commands:**

**For Everyone:**
• Mention me (@{bot_username}) to start a conversation
• /help - Show this help message
• /status - Show group statistics (if enabled)

**For Admins:**
• /group_settings - Configure group preferences
• /moderation - Moderation settings
• /analytics - View group analytics
• /export_data - Export group conversation data

Just mention me in any message and I'll respond naturally!
""",
            'group_stats': """
📊 **Group Statistics**

👥 **Members:** {member_count} ({active_count} active)
💬 **Messages:** {total_messages} total
🎯 **Bot Mentions:** {bot_mentions}
📈 **Engagement Score:** {engagement_score:.1%}
⚡ **Activity Level:** {message_frequency}

📈 **Recent Activity:** {recent_messages} messages today
""",
        }
        
        # Setup handlers
        self._setup_handlers()
    
    async def initialize_mention_detector(self) -> None:
        """Initialize mention detector with bot information."""
        try:
            bot_info = await self.bot.get_me()
            self.mention_detector = GroupMentionDetector(
                bot_username=bot_info.username or "bot",
                bot_id=bot_info.id
            )
            logger.info(f"Initialized mention detector for bot @{bot_info.username}")
        except Exception as e:
            logger.error(f"Failed to initialize mention detector: {e}")
    
    def _setup_handlers(self) -> None:
        """Setup group message handlers with proper routing."""
        # Chat member updates (joins, leaves, role changes)
        self.router.chat_member()(self.handle_member_update)
        
        # Admin commands
        self.router.message(Command("group_settings"))(self.handle_group_settings_command)
        self.router.message(Command("moderation"))(self.handle_moderation_command)
        self.router.message(Command("analytics"))(self.handle_analytics_command)
        self.router.message(Command("export_data"))(self.handle_export_command)
        
        # Group-specific commands
        self.router.message(Command("help"))(self.handle_group_help)
        self.router.message(Command("status"))(self.handle_group_status)
        
        # Group message handler (mentions and regular messages)
        self.router.message(F.chat.type.in_(["group", "supergroup"]))(self.handle_group_message)
        
        # Callback query handler for group interactions
        self.router.callback_query()(self.handle_group_callback)
        
        # Error handler
        self.router.error()(self.handle_group_error)
        
        logger.info("Group handlers setup completed")
    
    async def handle_member_update(self, update: ChatMemberUpdated) -> None:
        """Handle group member updates (joins, leaves, promotions, etc.)."""
        try:
            chat_id = update.chat.id
            user = update.new_chat_member.user
            old_status = update.old_chat_member.status if update.old_chat_member else None
            new_status = update.new_chat_member.status
            
            logger.info(
                f"Member update in group {chat_id}: {user.id} ({old_status} -> {new_status})"
            )
            
            # Get or create group session
            group_session = await self._get_or_create_group_session(update.chat)
            
            # Handle different types of member updates
            if old_status in [None, "left", "kicked"] and new_status == "member":
                await self._handle_member_joined(group_session, user, update.chat)
            elif old_status == "member" and new_status in ["left", "kicked"]:
                await self._handle_member_left(group_session, user, new_status)
            elif new_status in ["administrator", "creator"]:
                await self._handle_member_promoted(group_session, user, new_status)
            elif old_status in ["administrator", "creator"] and new_status == "member":
                await self._handle_member_demoted(group_session, user)
            
            # Clear permission cache for this chat
            self.permission_manager.clear_cache_for_chat(chat_id)
            
        except Exception as e:
            logger.error(f"Error handling member update: {e}", exc_info=True)
    
    async def handle_group_message(self, message: Message, state: FSMContext) -> None:
        """Handle messages in group chats with comprehensive context management."""
        try:
            start_time = time.time()
            chat_id = message.chat.id
            user_id = message.from_user.id
            
            # Skip if no mention detector available
            if not self.mention_detector:
                await self.initialize_mention_detector()
                if not self.mention_detector:
                    return
            
            # Get or create group session
            group_session = await self._get_or_create_group_session(message.chat)
            
            # Check if bot is mentioned or if this is a reply to bot
            is_bot_mentioned = self.mention_detector.is_bot_mentioned(message)
            
            # Get or create group member
            member = await self._get_or_create_group_member(group_session, message.from_user)
            
            # Check rate limits
            action_type = "mention" if is_bot_mentioned else "message"
            is_within_limits, limit_message = await self.group_rate_limiter.check_rate_limit(
                group_session, user_id, action_type
            )
            
            if not is_within_limits:
                # Record violation
                group_session.add_rate_limit_violation(
                    "rate_limit_exceeded",
                    {"user_id": user_id, "action_type": action_type, "message": limit_message}
                )
                
                # Send rate limit message (only for mentions to avoid spam)
                if is_bot_mentioned:
                    await message.reply(
                        self.templates['rate_limited'],
                        reply_to_message_id=message.message_id
                    )
                return
            
            # Update group and member statistics
            group_session.update_message_stats(is_bot_mentioned)
            member.add_message(is_bot_mentioned)
            
            # Detect conversation thread
            conversation_thread = await self._detect_or_create_conversation_thread(
                group_session, message, is_bot_mentioned
            )
            
            # Process message based on mention status
            if is_bot_mentioned:
                await self._handle_mentioned_message(
                    message, group_session, member, conversation_thread, state
                )
            else:
                # Passive monitoring for context and analytics
                await self._handle_passive_message(
                    message, group_session, member, conversation_thread
                )
            
            # Update caches
            self.group_cache[chat_id] = group_session
            self.member_cache[(chat_id, user_id)] = member
            
            # Record metrics
            processing_time = time.time() - start_time
            await self.metrics.record_message_received(
                message_type="group_message",
                processing_time=processing_time
            )
            
            # Save to database (async to avoid blocking)
            asyncio.create_task(self._save_group_session_async(group_session))
            asyncio.create_task(self._save_group_member_async(member))
            
        except Exception as e:
            logger.error(f"Error handling group message: {e}", exc_info=True)
    
    async def handle_group_help(self, message: Message, state: FSMContext) -> None:
        """Handle /help command in groups."""
        try:
            if not self.mention_detector:
                await self.initialize_mention_detector()
            
            bot_info = await self.bot.get_me()
            help_text = self.templates['help_group'].format(
                bot_username=bot_info.username or "bot"
            )
            
            await message.reply(help_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in group help command: {e}")
            await message.reply("❌ Error displaying help information.")
    
    async def handle_group_status(self, message: Message, state: FSMContext) -> None:
        """Handle /status command in groups."""
        try:
            group_session = await self._get_group_session(message.chat.id)
            if not group_session:
                await message.reply("❌ Group data not found.")
                return
            
            # Check if status display is enabled
            if not group_session.get_setting('show_status', True):
                return
            
            status_text = self.templates['group_stats'].format(
                member_count=group_session.member_count,
                active_count=group_session.active_member_count,
                total_messages=group_session.total_messages,
                bot_mentions=group_session.bot_mentions,
                engagement_score=group_session.engagement_score,
                message_frequency=group_session.message_frequency.value,
                recent_messages=await self._get_recent_message_count(group_session)
            )
            
            await message.reply(status_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in group status command: {e}")
            await message.reply("❌ Error retrieving group status.")
    
    async def handle_group_settings_command(self, message: Message, state: FSMContext) -> None:
        """Handle /group_settings command (admin only)."""
        try:
            chat_id = message.chat.id
            user_id = message.from_user.id
            
            # Check admin permissions
            if not await self.permission_manager.check_admin_permissions(
                self.bot, chat_id, user_id
            ):
                await message.reply(self.templates['admin_only'])
                return
            
            # Get group session
            group_session = await self._get_group_session(chat_id)
            if not group_session:
                await message.reply("❌ Group data not found.")
                return
            
            # Create settings keyboard
            keyboard = await self._create_group_settings_keyboard(group_session)
            
            settings_text = f"""
⚙️ **Group Settings for {group_session.title}**

Current Configuration:
• Status Display: {'✅' if group_session.get_setting('show_status', True) else '❌'}
• Analytics: {'✅' if group_session.get_setting('enable_analytics', True) else '❌'}
• Auto Moderation: {'✅' if group_session.get_setting('auto_moderation', False) else '❌'}
• Welcome Messages: {'✅' if group_session.get_setting('welcome_messages', True) else '❌'}
• Message Frequency: {group_session.message_frequency.value.title()}
• Rate Limiting: {'✅' if group_session.get_setting('rate_limiting', True) else '❌'}

Use the buttons below to modify settings.
"""
            
            await message.reply(
                settings_text,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
            await state.set_state(GroupStates.settings_config)
            
        except Exception as e:
            logger.error(f"Error in group settings command: {e}")
            await message.reply("❌ Error accessing group settings.")
    
    async def handle_moderation_command(self, message: Message, state: FSMContext) -> None:
        """Handle /moderation command (admin only)."""
        try:
            chat_id = message.chat.id
            user_id = message.from_user.id
            
            # Check admin permissions
            if not await self.permission_manager.check_admin_permissions(
                self.bot, chat_id, user_id, 'can_restrict_members'
            ):
                await message.reply(self.templates['admin_only'])
                return
            
            group_session = await self._get_group_session(chat_id)
            if not group_session:
                await message.reply("❌ Group data not found.")
                return
            
            # Get recent violations
            recent_violations = len(group_session.recent_violations or [])
            rate_limit_status = group_session.get_rate_limit_status()
            
            moderation_text = f"""
🛡️ **Moderation Dashboard**

**Current Status:**
• Group Status: {group_session.status.value.title()}
• Recent Violations: {recent_violations}
• Rate Limit Level: {rate_limit_status['current_frequency'].value.title()}

**Auto-Moderation Settings:**
• Spam Detection: {'✅' if group_session.get_setting('spam_detection', True) else '❌'}
• Rate Limiting: {'✅' if group_session.get_setting('rate_limiting', True) else '❌'}
• Content Filtering: {'✅' if group_session.get_setting('content_filter', False) else '❌'}

**Recent Activity:**
• Messages/Hour: {rate_limit_status['limits']['messages_per_hour']}
• Mentions/Hour: {rate_limit_status['limits']['mentions_per_hour']}
• Commands/Hour: {rate_limit_status['limits']['commands_per_hour']}
"""
            
            keyboard = await self._create_moderation_keyboard()
            
            await message.reply(
                moderation_text,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
            await state.set_state(GroupStates.moderation_review)
            
        except Exception as e:
            logger.error(f"Error in moderation command: {e}")
            await message.reply("❌ Error accessing moderation settings.")
    
    async def handle_analytics_command(self, message: Message, state: FSMContext) -> None:
        """Handle /analytics command (admin only)."""
        try:
            chat_id = message.chat.id
            user_id = message.from_user.id
            
            # Check admin permissions
            if not await self.permission_manager.check_admin_permissions(
                self.bot, chat_id, user_id
            ):
                await message.reply(self.templates['admin_only'])
                return
            
            group_session = await self._get_group_session(chat_id)
            if not group_session:
                await message.reply("❌ Group data not found.")
                return
            
            # Generate analytics summary
            analytics = await self._generate_group_analytics(group_session)
            
            analytics_text = f"""
📊 **Group Analytics Dashboard**

**Activity Summary (Last 24h):**
• Messages: {analytics['recent_messages']}
• Active Members: {analytics['active_members']}
• Conversations: {analytics['conversation_count']}
• Bot Interactions: {analytics['bot_interactions']}

**Engagement Metrics:**
• Overall Score: {group_session.engagement_score:.1%}
• Top Active Hour: {analytics['peak_hour']}:00
• Average Response Time: {analytics['avg_response_time']:.1f}s

**Member Statistics:**
• Total Members: {group_session.member_count}
• Active Today: {analytics['active_today']}
• New This Week: {analytics['new_members_week']}

**Content Insights:**
• Top Topics: {', '.join(analytics['top_topics'][:3])}
• Sentiment: {analytics['sentiment_summary']}
"""
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📈 Detailed Report",
                        callback_data=json.dumps({"action": "detailed_analytics", "group_id": str(group_session.id)})
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="📊 Export Data",
                        callback_data=json.dumps({"action": "export_analytics", "group_id": str(group_session.id)})
                    )
                ]
            ])
            
            await message.reply(
                analytics_text,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in analytics command: {e}")
            await message.reply("❌ Error generating analytics.")
    
    async def handle_export_command(self, message: Message, state: FSMContext) -> None:
        """Handle /export_data command (admin only)."""
        try:
            chat_id = message.chat.id
            user_id = message.from_user.id
            
            # Check admin permissions
            if not await self.permission_manager.check_admin_permissions(
                self.bot, chat_id, user_id
            ):
                await message.reply(self.templates['admin_only'])
                return
            
            group_session = await self._get_group_session(chat_id)
            if not group_session:
                await message.reply("❌ Group data not found.")
                return
            
            # Generate export data
            export_data = await self._generate_export_data(group_session)
            
            # Create temporary file
            filename = f"group_data_{chat_id}_{int(time.time())}.json"
            
            # Send as document
            from io import BytesIO
            buffer = BytesIO()
            buffer.write(json.dumps(export_data, indent=2, default=str).encode('utf-8'))
            buffer.seek(0)
            
            await message.reply_document(
                document=buffer,
                filename=filename,
                caption=f"📊 Group data export for {group_session.title}"
            )
            
        except Exception as e:
            logger.error(f"Error in export command: {e}")
            await message.reply("❌ Error exporting group data.")
    
    async def handle_group_callback(self, callback: CallbackQuery, state: FSMContext) -> None:
        """Handle callback queries from group inline keyboards."""
        try:
            await callback.answer()
            
            callback_data = json.loads(callback.data)
            action = callback_data.get('action')
            
            # Route to appropriate handler
            if action == 'toggle_setting':
                await self._handle_setting_toggle(callback, callback_data, state)
            elif action == 'moderation_action':
                await self._handle_moderation_action(callback, callback_data, state)
            elif action == 'detailed_analytics':
                await self._handle_detailed_analytics(callback, callback_data)
            elif action == 'export_analytics':
                await self._handle_export_analytics(callback, callback_data)
            else:
                await callback.message.edit_text("❌ Unknown action.")
                
        except json.JSONDecodeError:
            await callback.message.edit_text("❌ Invalid callback data.")
        except Exception as e:
            logger.error(f"Error handling group callback: {e}")
            await callback.message.edit_text("❌ Error processing request.")
    
    async def handle_group_error(self, update, exception) -> None:
        """Global error handler for group operations."""
        try:
            logger.error(
                "Unhandled error in group processing",
                update_type=type(update).__name__,
                error=str(exception),
                exc_info=True
            )
            
            # Try to send error message if possible
            if hasattr(update, 'message') and update.message:
                try:
                    await update.message.reply(
                        "❌ An error occurred while processing your request. Please try again.",
                        reply_to_message_id=update.message.message_id
                    )
                except Exception:
                    pass  # Can't send error message, just log
                    
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
    
    # Private helper methods
    
    async def _get_or_create_group_session(self, chat) -> GroupSession:
        """Get or create group session from chat info."""
        chat_id = chat.id
        
        # Check cache first
        if chat_id in self.group_cache:
            return self.group_cache[chat_id]
        
        try:
            async with get_async_session() as session:
                # Try to find existing group
                result = await session.execute(
                    "SELECT * FROM group_sessions WHERE telegram_chat_id = %s",
                    (chat_id,)
                )
                
                if result.fetchone():
                    # Load existing group
                    group_session = await session.query(GroupSession).filter(
                        GroupSession.telegram_chat_id == chat_id
                    ).first()
                else:
                    # Create new group
                    group_type = self._determine_group_type(chat)
                    
                    group_session = GroupSession(
                        telegram_chat_id=chat_id,
                        group_type=group_type,
                        title=chat.title or f"Group {chat_id}",
                        username=getattr(chat, 'username', None),
                        description=getattr(chat, 'description', None),
                        status=GroupStatus.ACTIVE,
                        first_interaction=datetime.utcnow()
                    )
                    
                    session.add(group_session)
                    await session.commit()
                
                # Cache the result
                self.group_cache[chat_id] = group_session
                return group_session
                
        except Exception as e:
            logger.error(f"Error getting/creating group session: {e}")
            raise
    
    async def _get_group_session(self, chat_id: int) -> Optional[GroupSession]:
        """Get existing group session."""
        # Check cache first
        if chat_id in self.group_cache:
            return self.group_cache[chat_id]
        
        try:
            async with get_async_session() as session:
                group_session = await session.query(GroupSession).filter(
                    GroupSession.telegram_chat_id == chat_id
                ).first()
                
                if group_session:
                    self.group_cache[chat_id] = group_session
                
                return group_session
                
        except Exception as e:
            logger.error(f"Error getting group session: {e}")
            return None
    
    async def _get_or_create_group_member(
        self, 
        group_session: GroupSession, 
        user
    ) -> GroupMember:
        """Get or create group member from user info."""
        cache_key = (group_session.telegram_chat_id, user.id)
        
        # Check cache first
        if cache_key in self.member_cache:
            return self.member_cache[cache_key]
        
        try:
            async with get_async_session() as session:
                # Try to find existing member
                member = await session.query(GroupMember).filter(
                    GroupMember.group_id == group_session.id,
                    GroupMember.telegram_user_id == user.id
                ).first()
                
                if not member:
                    # Get or create user record
                    user_record = await session.query(User).filter(
                        User.telegram_id == user.id
                    ).first()
                    
                    if not user_record:
                        user_record = User(
                            telegram_id=user.id,
                            username=user.username,
                            first_name=user.first_name,
                            last_name=user.last_name,
                            language_code=user.language_code
                        )
                        session.add(user_record)
                        await session.flush()
                    
                    # Create new member
                    member = GroupMember(
                        group_id=group_session.id,
                        user_id=user_record.id,
                        telegram_user_id=user.id,
                        role=MemberRole.MEMBER,
                        joined_at=datetime.utcnow()
                    )
                    
                    session.add(member)
                    await session.commit()
                
                # Cache the result
                self.member_cache[cache_key] = member
                return member
                
        except Exception as e:
            logger.error(f"Error getting/creating group member: {e}")
            raise
    
    def _determine_group_type(self, chat) -> GroupType:
        """Determine group type from chat object."""
        chat_type = getattr(chat, 'type', 'private')
        
        if chat_type == 'supergroup':
            return GroupType.SUPERGROUP
        elif chat_type == 'group':
            return GroupType.PRIVATE_GROUP
        elif chat_type == 'channel':
            return GroupType.CHANNEL
        else:
            return GroupType.PRIVATE_GROUP  # Default fallback
    
    # Additional methods would continue here...
    # (Implementation of remaining private methods for conversation handling,
    # analytics generation, callback handling, etc.)
    
    # Placeholder implementations for remaining methods
    async def _handle_mentioned_message(self, message, group_session, member, conversation_thread, state):
        """Handle message that mentions the bot."""
        # Implementation for bot mention handling
        pass
    
    async def _handle_passive_message(self, message, group_session, member, conversation_thread):
        """Handle regular group message (passive monitoring)."""
        # Implementation for passive message monitoring
        pass
    
    async def _detect_or_create_conversation_thread(self, group_session, message, is_bot_mentioned):
        """Detect or create conversation thread."""
        # Implementation for conversation thread detection
        pass
    
    async def _handle_member_joined(self, group_session, user, chat):
        """Handle new member joining."""
        # Implementation for member join handling
        pass
    
    async def _handle_member_left(self, group_session, user, status):
        """Handle member leaving."""
        # Implementation for member leave handling
        pass
    
    async def _handle_member_promoted(self, group_session, user, new_role):
        """Handle member promotion."""
        # Implementation for member promotion handling
        pass
    
    async def _handle_member_demoted(self, group_session, user):
        """Handle member demotion."""
        # Implementation for member demotion handling
        pass
    
    async def _create_group_settings_keyboard(self, group_session):
        """Create group settings inline keyboard."""
        # Implementation for settings keyboard creation
        return InlineKeyboardMarkup(inline_keyboard=[])
    
    async def _create_moderation_keyboard(self):
        """Create moderation inline keyboard."""
        # Implementation for moderation keyboard creation
        return InlineKeyboardMarkup(inline_keyboard=[])
    
    async def _generate_group_analytics(self, group_session):
        """Generate comprehensive group analytics."""
        # Implementation for analytics generation
        return {}
    
    async def _generate_export_data(self, group_session):
        """Generate export data for the group."""
        # Implementation for data export
        return {}
    
    async def _get_recent_message_count(self, group_session):
        """Get recent message count."""
        # Implementation for recent message counting
        return 0
    
    async def _save_group_session_async(self, group_session):
        """Save group session to database asynchronously."""
        # Implementation for async database save
        pass
    
    async def _save_group_member_async(self, member):
        """Save group member to database asynchronously."""
        # Implementation for async database save
        pass
    
    async def _handle_setting_toggle(self, callback, callback_data, state):
        """Handle settings toggle callback."""
        # Implementation for settings toggle
        pass
    
    async def _handle_moderation_action(self, callback, callback_data, state):
        """Handle moderation action callback."""
        # Implementation for moderation actions
        pass
    
    async def _handle_detailed_analytics(self, callback, callback_data):
        """Handle detailed analytics callback."""
        # Implementation for detailed analytics
        pass
    
    async def _handle_export_analytics(self, callback, callback_data):
        """Handle analytics export callback."""
        # Implementation for analytics export
        pass


async def setup_group_handlers(dp: "Dispatcher", bot: Bot) -> None:
    """Setup group handlers with the dispatcher."""
    try:
        # Initialize group handlers
        handlers = GroupHandlers(bot)
        
        # Set component references
        handlers.session_manager = bot.session_manager
        handlers.anti_ban = bot.anti_ban
        handlers.metrics = bot.metrics
        handlers.rate_limiter = bot.rate_limiter
        
        # Initialize mention detector
        await handlers.initialize_mention_detector()
        
        # Include router
        dp.include_router(handlers.router)
        
        logger.info("Group handlers setup completed")
        
    except Exception as e:
        logger.error("Failed to setup group handlers", error=str(e))
        raise