"""
Kelly Telegram Userbot Service

Pyrogram-based userbot for real Telegram account management with DM-only focus,
anti-detection measures, and Kelly personality integration.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

import structlog
from pyrogram import Client, filters, types
from pyrogram.errors import FloodWait, UserDeactivated, UserDeactivatedBan, AuthKeyUnregistered
from pyrogram.enums import ChatType

from app.config import settings
from app.core.redis import redis_manager
from app.services.kelly_personality_service import kelly_personality_service, KellyPersonalityConfig
from app.services.typing_simulator import TypingSimulator
from app.services.anti_ban import AntiDetectionManager

logger = structlog.get_logger()

class MessageType(Enum):
    """Types of messages for processing"""
    PRIVATE_DM = "private_dm"
    GROUP_MENTION = "group_mention"
    GROUP_REGULAR = "group_regular"
    CHANNEL_MESSAGE = "channel_message"

@dataclass
class AccountConfig:
    """Configuration for each Telegram account"""
    api_id: int
    api_hash: str
    phone_number: str
    session_name: str
    dm_only_mode: bool = True
    max_daily_messages: int = 50
    response_probability: float = 0.9
    kelly_config: Optional[KellyPersonalityConfig] = None
    enabled: bool = True

@dataclass
class MessageContext:
    """Context for incoming messages"""
    message: types.Message
    chat_type: ChatType
    message_type: MessageType
    is_dm: bool
    from_user: types.User
    text: str
    should_respond: bool

class KellyTelegramUserbot:
    """Advanced Telegram userbot with Kelly personality integration"""
    
    def __init__(self):
        self.clients: Dict[str, Client] = {}
        self.account_configs: Dict[str, AccountConfig] = {}
        self.typing_simulator = TypingSimulator()
        self.anti_detection = AntiDetectionManager()
        
        # Message queues for rate limiting
        self.message_queues: Dict[str, List[datetime]] = {}
        
        # Active conversations tracking
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self.message_handlers: List[Callable] = []
        
    async def initialize(self):
        """Initialize the userbot system"""
        try:
            await kelly_personality_service.initialize()
            await self.typing_simulator.initialize()
            await self.anti_detection.initialize()
            
            # Load account configurations
            await self._load_account_configs()
            
            logger.info("Kelly Telegram userbot system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly userbot system: {e}")
            raise

    async def _load_account_configs(self):
        """Load account configurations from Redis"""
        try:
            keys = await redis_manager.scan_iter(match="kelly:account:*")
            async for key in keys:
                data = await redis_manager.get(key)
                if data:
                    config_data = json.loads(data)
                    account_id = key.split(":")[-1]
                    
                    # Create AccountConfig object
                    kelly_config_data = config_data.pop("kelly_config", {})
                    kelly_config = KellyPersonalityConfig(**kelly_config_data) if kelly_config_data else None
                    
                    config = AccountConfig(**config_data, kelly_config=kelly_config)
                    self.account_configs[account_id] = config
                    
        except Exception as e:
            logger.error(f"Error loading account configs: {e}")

    async def add_account(
        self, 
        account_id: str, 
        api_id: int, 
        api_hash: str, 
        phone_number: str,
        dm_only_mode: bool = True,
        kelly_config: Optional[KellyPersonalityConfig] = None
    ) -> bool:
        """Add a new Telegram account to the userbot system"""
        try:
            # Create account configuration
            config = AccountConfig(
                api_id=api_id,
                api_hash=api_hash,
                phone_number=phone_number,
                session_name=f"kelly_{account_id}",
                dm_only_mode=dm_only_mode,
                kelly_config=kelly_config or KellyPersonalityConfig()
            )
            
            # Create Pyrogram client
            client = Client(
                name=config.session_name,
                api_id=config.api_id,
                api_hash=config.api_hash,
                phone_number=config.phone_number,
                workdir="sessions/"
            )
            
            # Start the client
            await client.start()
            
            # Store client and config
            self.clients[account_id] = client
            self.account_configs[account_id] = config
            
            # Setup message handlers for this client
            await self._setup_message_handlers(account_id, client)
            
            # Save configuration to Redis
            await self._save_account_config(account_id, config)
            
            logger.info(f"Added Telegram account {account_id} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add account {account_id}: {e}")
            return False

    async def _setup_message_handlers(self, account_id: str, client: Client):
        """Setup message handlers for a specific client"""
        
        @client.on_message(filters.private & filters.incoming & ~filters.bot)
        async def handle_private_message(client, message):
            await self._handle_incoming_message(account_id, message, MessageType.PRIVATE_DM)
        
        @client.on_message(filters.group & filters.incoming & filters.mentioned)
        async def handle_group_mention(client, message):
            if not self.account_configs[account_id].dm_only_mode:
                await self._handle_incoming_message(account_id, message, MessageType.GROUP_MENTION)
        
        @client.on_message(filters.group & filters.incoming & ~filters.mentioned)
        async def handle_group_message(client, message):
            if not self.account_configs[account_id].dm_only_mode:
                await self._handle_incoming_message(account_id, message, MessageType.GROUP_REGULAR)

    async def _handle_incoming_message(self, account_id: str, message: types.Message, msg_type: MessageType):
        """Handle incoming messages with Kelly personality"""
        try:
            config = self.account_configs.get(account_id)
            if not config or not config.enabled:
                return
            
            # Create message context
            context = await self._create_message_context(message, msg_type)
            
            # Check if we should respond
            if not await self._should_respond(account_id, context, config):
                return
            
            # Check rate limits
            if not await self._check_rate_limits(account_id, config):
                logger.warning(f"Rate limit exceeded for account {account_id}")
                return
            
            # Anti-detection delay
            await self._apply_anti_detection_delay(account_id)
            
            # Generate Kelly response
            response, response_metadata = await kelly_personality_service.generate_kelly_response(
                user_id=str(context.from_user.id),
                conversation_id=f"{account_id}_{context.from_user.id}",
                message=context.text,
                config=config.kelly_config
            )
            
            # Check if user was blocked
            if response_metadata.get("blocked", False):
                logger.warning(f"User {context.from_user.id} auto-blocked by Kelly AI")
                return
            
            # Apply typing simulation
            typing_delay = response_metadata.get("typing_delay", 5.0)
            await self._simulate_typing(account_id, message.chat.id, typing_delay)
            
            # Send response
            await self._send_response(account_id, message, response, response_metadata)
            
            # Log conversation stats
            await self._log_conversation_stats(account_id, context, response_metadata)
            
        except FloodWait as e:
            logger.warning(f"Flood wait for {e.value} seconds on account {account_id}")
            await asyncio.sleep(e.value)
            
        except (UserDeactivated, UserDeactivatedBan, AuthKeyUnregistered) as e:
            logger.error(f"Account {account_id} deactivated: {e}")
            await self._disable_account(account_id)
            
        except Exception as e:
            logger.error(f"Error handling message for account {account_id}: {e}")

    async def _create_message_context(self, message: types.Message, msg_type: MessageType) -> MessageContext:
        """Create message context for processing"""
        return MessageContext(
            message=message,
            chat_type=message.chat.type,
            message_type=msg_type,
            is_dm=message.chat.type == ChatType.PRIVATE,
            from_user=message.from_user,
            text=message.text or "",
            should_respond=False  # Will be determined later
        )

    async def _should_respond(self, account_id: str, context: MessageContext, config: AccountConfig) -> bool:
        """Determine if we should respond to this message"""
        
        # Always respond to DMs (unless blocked)
        if context.is_dm:
            # Check if user is blocked
            is_blocked = await kelly_personality_service.is_user_blocked(str(context.from_user.id))
            if is_blocked:
                return False
            
            # Check response probability
            return random.random() < config.response_probability
        
        # For group messages (if not DM-only mode)
        if not config.dm_only_mode:
            if context.message_type == MessageType.GROUP_MENTION:
                return random.random() < config.response_probability * 0.8  # Lower probability for groups
            elif context.message_type == MessageType.GROUP_REGULAR:
                return random.random() < config.response_probability * 0.1  # Very low for regular group messages
        
        return False

    async def _check_rate_limits(self, account_id: str, config: AccountConfig) -> bool:
        """Check if account has exceeded daily message limits"""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        # Get message history for last 24 hours
        if account_id not in self.message_queues:
            self.message_queues[account_id] = []
        
        # Remove old messages
        self.message_queues[account_id] = [
            msg_time for msg_time in self.message_queues[account_id] 
            if msg_time > cutoff
        ]
        
        # Check if under limit
        if len(self.message_queues[account_id]) >= config.max_daily_messages:
            return False
        
        # Add current message to queue
        self.message_queues[account_id].append(now)
        return True

    async def _apply_anti_detection_delay(self, account_id: str):
        """Apply anti-detection delays using advanced algorithms"""
        try:
            # Get account-specific delay pattern
            delay = await self.anti_detection.calculate_response_delay(account_id)
            
            # Apply minimum delay for natural behavior
            min_delay = random.uniform(1.0, 3.0)
            total_delay = max(min_delay, delay)
            
            logger.debug(f"Applying anti-detection delay of {total_delay:.2f}s for account {account_id}")
            await asyncio.sleep(total_delay)
            
        except Exception as e:
            logger.error(f"Error applying anti-detection delay: {e}")
            # Fallback to simple delay
            await asyncio.sleep(random.uniform(2.0, 5.0))

    async def _simulate_typing(self, account_id: str, chat_id: int, duration: float):
        """Simulate natural typing behavior"""
        try:
            client = self.clients.get(account_id)
            if not client:
                return
            
            # Apply typing simulation with Kelly's patterns
            await self.typing_simulator.simulate_natural_typing(
                client, chat_id, duration
            )
            
        except Exception as e:
            logger.error(f"Error simulating typing: {e}")

    async def _send_response(
        self, 
        account_id: str, 
        original_message: types.Message, 
        response: str,
        metadata: Dict[str, Any]
    ):
        """Send Kelly's response with natural behavior patterns"""
        try:
            client = self.clients.get(account_id)
            if not client:
                return
            
            # Add final anti-detection measures
            await self._apply_final_anti_detection(account_id, response)
            
            # Send the message
            sent_message = await client.send_message(
                chat_id=original_message.chat.id,
                text=response,
                reply_to_message_id=original_message.id if random.random() < 0.3 else None
            )
            
            # Log successful send
            logger.info(f"Kelly response sent from account {account_id} to {original_message.from_user.id}")
            
            # Store conversation tracking
            await self._track_conversation(account_id, original_message, sent_message, metadata)
            
        except Exception as e:
            logger.error(f"Error sending response from account {account_id}: {e}")

    async def _apply_final_anti_detection(self, account_id: str, response: str):
        """Apply final anti-detection measures before sending"""
        try:
            # Random final delay
            final_delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(final_delay)
            
            # Mark as online (natural behavior)
            client = self.clients.get(account_id)
            if client and random.random() < 0.7:
                try:
                    await client.set_chat_presence(chat_id="me", presence="online")
                except:
                    pass  # Ignore errors for presence
            
        except Exception as e:
            logger.error(f"Error applying final anti-detection: {e}")

    async def _track_conversation(
        self, 
        account_id: str, 
        original_message: types.Message, 
        sent_message: types.Message,
        metadata: Dict[str, Any]
    ):
        """Track conversation for analytics and learning"""
        try:
            conversation_key = f"kelly:conversation_track:{account_id}_{original_message.from_user.id}"
            
            track_data = {
                "account_id": account_id,
                "user_id": original_message.from_user.id,
                "username": original_message.from_user.username,
                "original_message": original_message.text,
                "kelly_response": sent_message.text,
                "timestamp": datetime.now().isoformat(),
                "stage": metadata.get("stage", "unknown"),
                "safety_score": metadata.get("safety_score", 1.0),
                "typing_delay": metadata.get("typing_delay", 0)
            }
            
            await redis_manager.lpush(conversation_key, json.dumps(track_data))
            await redis_manager.ltrim(conversation_key, 0, 999)  # Keep last 1000 messages
            await redis_manager.expire(conversation_key, 86400 * 30)  # 30 days
            
        except Exception as e:
            logger.error(f"Error tracking conversation: {e}")

    async def _log_conversation_stats(
        self, 
        account_id: str, 
        context: MessageContext, 
        metadata: Dict[str, Any]
    ):
        """Log conversation statistics for monitoring"""
        try:
            stats = {
                "account_id": account_id,
                "user_id": context.from_user.id,
                "message_type": context.message_type.value,
                "stage": metadata.get("stage", "unknown"),
                "safety_score": metadata.get("safety_score", 1.0),
                "red_flags": metadata.get("red_flags", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            stats_key = f"kelly:daily_stats:{account_id}:{today}"
            await redis_manager.lpush(stats_key, json.dumps(stats))
            await redis_manager.expire(stats_key, 86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Error logging conversation stats: {e}")

    async def _disable_account(self, account_id: str):
        """Disable account due to errors or bans"""
        try:
            if account_id in self.account_configs:
                self.account_configs[account_id].enabled = False
                await self._save_account_config(account_id, self.account_configs[account_id])
            
            if account_id in self.clients:
                try:
                    await self.clients[account_id].stop()
                except:
                    pass
                del self.clients[account_id]
            
            logger.warning(f"Disabled account {account_id}")
            
        except Exception as e:
            logger.error(f"Error disabling account {account_id}: {e}")

    async def _save_account_config(self, account_id: str, config: AccountConfig):
        """Save account configuration to Redis"""
        try:
            config_data = {
                "api_id": config.api_id,
                "api_hash": config.api_hash,
                "phone_number": config.phone_number,
                "session_name": config.session_name,
                "dm_only_mode": config.dm_only_mode,
                "max_daily_messages": config.max_daily_messages,
                "response_probability": config.response_probability,
                "enabled": config.enabled,
                "kelly_config": config.kelly_config.__dict__ if config.kelly_config else {}
            }
            
            key = f"kelly:account:{account_id}"
            await redis_manager.setex(key, 86400 * 30, json.dumps(config_data))
            
        except Exception as e:
            logger.error(f"Error saving account config: {e}")

    async def get_account_status(self, account_id: str) -> Dict[str, Any]:
        """Get account status and statistics"""
        try:
            config = self.account_configs.get(account_id)
            client = self.clients.get(account_id)
            
            if not config:
                return {"error": "Account not found"}
            
            # Get daily message count
            today = datetime.now().strftime("%Y-%m-%d")
            messages_today = len(self.message_queues.get(account_id, []))
            
            # Get recent conversation stats
            stats_key = f"kelly:daily_stats:{account_id}:{today}"
            recent_stats = await redis_manager.lrange(stats_key, 0, -1)
            
            return {
                "account_id": account_id,
                "enabled": config.enabled,
                "dm_only_mode": config.dm_only_mode,
                "connected": client is not None and client.is_connected,
                "messages_today": messages_today,
                "max_daily_messages": config.max_daily_messages,
                "response_probability": config.response_probability,
                "recent_conversations": len(recent_stats),
                "kelly_config": config.kelly_config.__dict__ if config.kelly_config else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return {"error": str(e)}

    async def update_account_config(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """Update account configuration"""
        try:
            config = self.account_configs.get(account_id)
            if not config:
                return False
            
            # Update basic config fields
            for field, value in updates.items():
                if field == "kelly_config" and isinstance(value, dict):
                    # Update Kelly configuration
                    if config.kelly_config:
                        for kelly_field, kelly_value in value.items():
                            if hasattr(config.kelly_config, kelly_field):
                                setattr(config.kelly_config, kelly_field, kelly_value)
                    else:
                        config.kelly_config = KellyPersonalityConfig(**value)
                elif hasattr(config, field):
                    setattr(config, field, value)
            
            # Save updated configuration
            await self._save_account_config(account_id, config)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating account config: {e}")
            return False

    async def get_conversation_history(self, account_id: str, user_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history between account and user"""
        try:
            conversation_key = f"kelly:conversation_track:{account_id}_{user_id}"
            messages = await redis_manager.lrange(conversation_key, 0, limit - 1)
            
            return [json.loads(msg) for msg in messages]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def stop_account(self, account_id: str):
        """Stop a specific account"""
        try:
            if account_id in self.clients:
                await self.clients[account_id].stop()
                del self.clients[account_id]
            
            if account_id in self.account_configs:
                self.account_configs[account_id].enabled = False
                await self._save_account_config(account_id, self.account_configs[account_id])
            
            logger.info(f"Stopped account {account_id}")
            
        except Exception as e:
            logger.error(f"Error stopping account {account_id}: {e}")

    async def stop_all_accounts(self):
        """Stop all accounts"""
        for account_id in list(self.clients.keys()):
            await self.stop_account(account_id)
    
    async def send_verification_code(self, api_id: int, api_hash: str, phone_number: str) -> Dict[str, Any]:
        """Send verification code for Telegram authentication."""
        try:
            # Create temporary client for authentication
            temp_client = Client(
                name=f"temp_auth_{phone_number.replace('+', '')}",
                api_id=api_id,
                api_hash=api_hash,
                phone_number=phone_number,
                in_memory=True  # Don't save session to disk yet
            )
            
            # Connect and send code
            await temp_client.connect()
            
            # Send verification code
            sent_code = await temp_client.send_code(phone_number)
            
            # Get session string for later use
            session_string = await temp_client.export_session_string()
            
            # Disconnect temporary client
            await temp_client.disconnect()
            
            return {
                "success": True,
                "phone_code_hash": sent_code.phone_code_hash,
                "session_string": session_string,
                "message": "Verification code sent successfully"
            }
            
        except Exception as e:
            logger.error(f"Error sending verification code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def verify_authentication_code(
        self, 
        session_string: str, 
        phone_code_hash: str, 
        verification_code: str
    ) -> Dict[str, Any]:
        """Verify the authentication code."""
        try:
            # Create client from session string
            temp_client = Client(
                name="temp_verify",
                session_string=session_string,
                in_memory=True
            )
            
            await temp_client.connect()
            
            # Verify the code
            try:
                signed_in = await temp_client.sign_in(
                    phone_code_hash=phone_code_hash,
                    phone_code=verification_code
                )
                
                # Get user info
                user_info = {
                    "id": signed_in.id,
                    "first_name": signed_in.first_name,
                    "last_name": signed_in.last_name,
                    "username": signed_in.username,
                    "phone_number": signed_in.phone_number
                }
                
                await temp_client.disconnect()
                
                return {
                    "success": True,
                    "requires_2fa": False,
                    "user_info": user_info,
                    "message": "Authentication successful"
                }
                
            except Exception as inner_e:
                # Check if 2FA is required
                if "Two-step verification" in str(inner_e) or "PASSWORD_HASH_INVALID" in str(inner_e):
                    await temp_client.disconnect()
                    return {
                        "success": True,
                        "requires_2fa": True,
                        "message": "Two-factor authentication required"
                    }
                else:
                    await temp_client.disconnect()
                    raise inner_e
                    
        except Exception as e:
            logger.error(f"Error verifying authentication code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def verify_2fa_password(self, session_string: str, password: str) -> Dict[str, Any]:
        """Verify two-factor authentication password."""
        try:
            # Create client from session string
            temp_client = Client(
                name="temp_2fa",
                session_string=session_string,
                in_memory=True
            )
            
            await temp_client.connect()
            
            # Verify 2FA password
            signed_in = await temp_client.check_password(password)
            
            # Get user info
            user_info = {
                "id": signed_in.id,
                "first_name": signed_in.first_name,
                "last_name": signed_in.last_name,
                "username": signed_in.username,
                "phone_number": signed_in.phone_number
            }
            
            await temp_client.disconnect()
            
            return {
                "success": True,
                "user_info": user_info,
                "message": "Two-factor authentication successful"
            }
            
        except Exception as e:
            logger.error(f"Error verifying 2FA password: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_authenticated_account(
        self, 
        account_id: str, 
        account_config: AccountConfig, 
        session_string: str, 
        user_info: Dict[str, Any]
    ) -> bool:
        """Add an authenticated account to the Kelly system."""
        try:
            # Create client with the authenticated session
            client = Client(
                name=account_config.session_name,
                api_id=account_config.api_id,
                api_hash=account_config.api_hash,
                session_string=session_string
            )
            
            # Store client and config
            self.clients[account_id] = client
            self.account_configs[account_id] = account_config
            
            # Save account configuration to Redis
            await self._save_account_config(account_id, account_config)
            
            # Store user info
            user_key = f"kelly:account:user_info:{account_id}"
            await redis_manager.setex(user_key, 86400 * 30, json.dumps(user_info, default=str))
            
            # Start the client if enabled
            if account_config.enabled:
                await self._start_account_client(account_id)
            
            logger.info(f"Added authenticated account {account_id} for user {user_info.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding authenticated account: {e}")
            return False

# Global instance
kelly_userbot = KellyTelegramUserbot()