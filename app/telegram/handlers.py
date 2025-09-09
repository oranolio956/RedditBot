"""
Telegram Bot Message Handlers

Comprehensive message handling system with conversation flow management,
command processing, and intelligent response generation.
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional, List, Union
import json
import re
import html
import urllib.parse

import structlog
from aiogram import Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineQuery, Update,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton,
    BotCommand, BotCommandScopeDefault
)
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from .session import SessionManager, MessageContext, ConversationMode
from .anti_ban import AntiBanManager, RiskLevel
from .metrics import TelegramMetrics
from .rate_limiter import AdvancedRateLimiter

logger = structlog.get_logger(__name__)


# FSM States for conversation flow
class ConversationStates(StatesGroup):
    """Conversation flow states."""
    idle = State()
    waiting_input = State()
    processing = State()
    multi_step_operation = State()
    error_recovery = State()


class TelegramHandlers:
    """
    Comprehensive message handling system.
    
    Features:
    - Command processing with permissions
    - Natural conversation flow
    - Multi-step operations
    - Error handling and recovery
    - Anti-ban integrated responses
    - Metrics and analytics
    - User session management
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.session_manager: Optional[SessionManager] = None
        self.anti_ban: Optional[AntiBanManager] = None
        self.metrics: Optional[TelegramMetrics] = None
        self.rate_limiter: Optional[AdvancedRateLimiter] = None
        
        # Router for organizing handlers
        self.router = Router()
        
        # Command registry
        self.commands: Dict[str, Dict[str, Any]] = {}
        
        # Response templates
        self.templates = {
            'start': "👋 Welcome! I'm your intelligent assistant. How can I help you today?",
            'help': "🔧 Available commands:\n{commands}\n\nJust type naturally and I'll understand!",
            'error': "❌ Something went wrong. Let me try again...",
            'rate_limited': "⏳ Please slow down a bit. I'm here when you're ready!",
            'processing': "🤔 Let me think about that...",
            'unknown_command': "🤷‍♂️ I'm not sure what you mean. Try /help to see what I can do!",
        }
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup message handlers with proper routing."""
        # Start command
        self.router.message(CommandStart())(self.handle_start)
        
        # Help command
        self.router.message(Command("help"))(self.handle_help)
        
        # Status command (admin only)
        self.router.message(Command("status"))(self.handle_status)
        
        # Settings command
        self.router.message(Command("settings"))(self.handle_settings)
        
        # Generic command handler
        self.router.message(Command())(self.handle_command)
        
        # Text message handler
        self.router.message(F.text)(self.handle_text_message)
        
        # Callback query handler
        self.router.callback_query()(self.handle_callback_query)
        
        # Inline query handler
        self.router.inline_query()(self.handle_inline_query)
        
        # Error handler
        self.router.error()(self.handle_error)
        
        logger.info("Message handlers setup completed")
    
    async def handle_start(self, message: Message, state: FSMContext) -> None:
        """Handle /start command."""
        try:
            # Create message context
            context = await self._create_message_context(message)
            
            # Get or create user session
            session = await self.session_manager.get_or_create_session(
                user_id=message.from_user.id,
                chat_id=message.chat.id,
                user_data={
                    'username': message.from_user.username,
                    'first_name': message.from_user.first_name,
                    'last_name': message.from_user.last_name,
                    'language_code': message.from_user.language_code,
                    'is_premium': message.from_user.is_premium or False,
                }
            )
            
            # Risk assessment
            risk_level, risk_score, _ = await self.anti_ban.assess_risk_level(
                user_id=message.from_user.id,
                action="start_command",
                context={'is_command': True}
            )
            
            # Apply anti-ban measures
            typing_delay = await self.anti_ban.calculate_typing_delay(
                text=self.templates['start'],
                user_id=message.from_user.id,
                context={'is_command': True}
            )
            
            # Show typing
            await message.chat.do('typing')
            await asyncio.sleep(typing_delay)
            
            # Send welcome message
            await message.reply(
                self.templates['start'],
                reply_markup=self._get_start_keyboard()
            )
            
            # Update session
            await self.session_manager.update_session_activity(
                session.session_id,
                message_context=context,
                context_updates={'last_command': 'start'}
            )
            
            # Record metrics
            await self.metrics.record_command_executed(
                command="start",
                processing_time=typing_delay,
                success=True
            )
            
            await self.metrics.record_user_activity(
                user_id=message.from_user.id,
                language_code=message.from_user.language_code
            )
            
            # Set state
            await state.set_state(ConversationStates.idle)
            
            logger.info(f"Start command processed for user {message.from_user.id}")
    
    async def _validate_and_sanitize_message(self, message: Message) -> bool:
        """Validate and sanitize incoming message content."""
        try:
            if not message.text:
                return True  # Non-text messages are handled elsewhere
            
            text = message.text
            
            # Length validation
            if len(text) > 10000:  # Reasonable limit for text messages
                logger.warning(f"Message too long: {len(text)} characters")
                return False
            
            # Check for malicious patterns
            malicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'data:text/html',
                r'<iframe[^>]*>.*?</iframe>',
                r'<object[^>]*>.*?</object>',
                r'<embed[^>]*>.*?</embed>',
                r'<link[^>]*rel=["\']?stylesheet',
                r'<meta[^>]*http-equiv',
                r'expression\s*\(',
                r'@import',
                r'binding:\s*url',
            ]
            
            for pattern in malicious_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                    logger.error(f"Malicious pattern detected: {pattern}")
                    return False
            
            # Check for excessive special characters (potential obfuscation)
            special_char_count = len(re.findall(r'[^\w\s\.,!?;:()\[\]{}"\'-]', text))
            if special_char_count > len(text) * 0.3:  # More than 30% special chars
                logger.warning(f"Excessive special characters: {special_char_count}/{len(text)}")
                return False
            
            # Check for potential SQL injection patterns
            sql_patterns = [
                r'(union|select|insert|update|delete|drop|create|alter)\s+',
                r'[\'"][^\'"]*(union|select|insert|update|delete)[^\'"]*(\-\-|#)',
                r'\b(or|and)\s+[\'"]?\d+[\'"]?\s*=\s*[\'"]?\d+[\'"]?',
                r';\s*(drop|delete|truncate|insert|update)',
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"Potential SQL injection pattern: {pattern}")
                    # Don't block, but log for monitoring
            
            # URL validation - check for suspicious domains
            urls = re.findall(r'https?://[^\s]+', text)
            for url in urls:
                if not await self._validate_url(url):
                    logger.warning(f"Suspicious URL detected: {url}")
                    return False
            
            # Sanitize HTML entities and control characters
            text = html.unescape(text)
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
            
            # Update the message text with sanitized version
            message.text = text
            
            return True
            
        except Exception as e:
            logger.error("Error validating message", error=str(e))
            return False
    
    async def _validate_url(self, url: str) -> bool:
        """Validate URL for suspicious patterns."""
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Block suspicious domains
            suspicious_domains = [
                'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',
                'localhost', '127.0.0.1', '0.0.0.0',
                # Add more suspicious domains as needed
            ]
            
            domain = parsed.netloc.lower()
            for suspicious in suspicious_domains:
                if suspicious in domain:
                    return False
            
            # Block suspicious file extensions
            path = parsed.path.lower()
            suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
            for ext in suspicious_extensions:
                if path.endswith(ext):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _contains_urls(self, text: str) -> bool:
        """Check if text contains URLs."""
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        return bool(re.search(url_pattern, text, re.IGNORECASE))
    
    def _contains_mentions(self, text: str) -> bool:
        """Check if text contains user mentions."""
        mention_pattern = r'@\w+'
        return bool(re.search(mention_pattern, text))
    
    def _detect_suspicious_patterns(self, text: str) -> int:
        """Detect and count suspicious patterns in text."""
        suspicious_count = 0
        
        patterns = [
            r'(free|win|winner|prize|congratulations).*\$\d+',
            r'(click|visit).*https?://',
            r'urgent.*act now',
            r'\b(bitcoin|crypto|investment).*guaranteed',
            r'(password|login|account).*suspended',
            r'verify.*account.*immediately',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious_count += 1
        
        return suspicious_count
            
        except Exception as e:
            logger.error("Error handling start command", error=str(e))
            await self._handle_command_error(message, "start", str(e))
    
    async def handle_help(self, message: Message, state: FSMContext) -> None:
        """Handle /help command."""
        try:
            # Get available commands
            commands_text = await self._get_commands_text()
            help_text = self.templates['help'].format(commands=commands_text)
            
            # Anti-ban measures
            typing_delay = await self.anti_ban.calculate_typing_delay(
                text=help_text,
                user_id=message.from_user.id,
                context={'is_command': True}
            )
            
            await message.chat.do('typing')
            await asyncio.sleep(typing_delay)
            
            await message.reply(
                help_text,
                reply_markup=self._get_help_keyboard()
            )
            
            # Update session and metrics
            await self._update_session_and_metrics(
                message, "help", typing_delay
            )
            
        except Exception as e:
            logger.error("Error handling help command", error=str(e))
            await self._handle_command_error(message, "help", str(e))
    
    async def handle_status(self, message: Message, state: FSMContext) -> None:
        """Handle /status command (admin only)."""
        try:
            # Check if user is admin (implement your admin check logic)
            if not await self._is_admin_user(message.from_user.id):
                await message.reply("❌ You don't have permission to use this command.")
                return
            
            # Get system status
            bot_status = await self.bot.get_status() if hasattr(self.bot, 'get_status') else {}
            metrics = await self.metrics.get_current_metrics()
            
            status_text = self._format_status_message(bot_status, metrics)
            
            # Send status with typing simulation
            typing_delay = await self.anti_ban.calculate_typing_delay(
                text=status_text,
                user_id=message.from_user.id,
                context={'is_command': True}
            )
            
            await message.chat.do('typing')
            await asyncio.sleep(typing_delay)
            
            await message.reply(
                status_text,
                parse_mode='HTML',
                reply_markup=self._get_status_keyboard()
            )
            
            await self._update_session_and_metrics(
                message, "status", typing_delay
            )
            
        except Exception as e:
            logger.error("Error handling status command", error=str(e))
            await self._handle_command_error(message, "status", str(e))
    
    async def handle_settings(self, message: Message, state: FSMContext) -> None:
        """Handle /settings command."""
        try:
            # Get user session
            session = await self.session_manager.get_session_by_user_chat(
                message.from_user.id, message.chat.id
            )
            
            if not session:
                await message.reply("❌ Session not found. Please use /start first.")
                return
            
            settings_text = await self._format_user_settings(session)
            
            # Anti-ban measures
            typing_delay = await self.anti_ban.calculate_typing_delay(
                text=settings_text,
                user_id=message.from_user.id,
                context={'is_command': True}
            )
            
            await message.chat.do('typing')
            await asyncio.sleep(typing_delay)
            
            await message.reply(
                settings_text,
                reply_markup=self._get_settings_keyboard(),
                parse_mode='HTML'
            )
            
            # Set state for settings interaction
            await state.set_state(ConversationStates.waiting_input)
            await state.update_data(waiting_for="settings_selection")
            
            await self._update_session_and_metrics(
                message, "settings", typing_delay
            )
            
        except Exception as e:
            logger.error("Error handling settings command", error=str(e))
            await self._handle_command_error(message, "settings", str(e))
    
    async def handle_command(self, message: Message, state: FSMContext) -> None:
        """Handle generic commands."""
        try:
            command = message.text[1:]  # Remove '/' prefix
            command_name = command.split()[0].lower()
            
            # Check if command exists in registry
            if command_name in self.commands:
                command_config = self.commands[command_name]
                await self._execute_registered_command(
                    message, command_name, command_config, state
                )
            else:
                # Unknown command
                await self._handle_unknown_command(message)
            
        except Exception as e:
            logger.error(f"Error handling command {message.text}", error=str(e))
            await self._handle_command_error(message, "unknown", str(e))
    
    async def handle_text_message(self, message: Message, state: FSMContext) -> None:
        """Handle regular text messages with comprehensive input validation."""
        try:
            start_time = time.time()
            
            # Input validation and sanitization
            if not await self._validate_and_sanitize_message(message):
                logger.warning(f"Invalid message from user {message.from_user.id}")
                await message.reply("❌ Your message contains invalid content.")
                return
            
            # Create message context
            context = await self._create_message_context(message)
            
            # Get user session
            session = await self.session_manager.get_session_by_user_chat(
                message.from_user.id, message.chat.id
            )
            
            if not session:
                # Create session if doesn't exist
                session = await self.session_manager.get_or_create_session(
                    user_id=message.from_user.id,
                    chat_id=message.chat.id
                )
            
            # Enhanced risk assessment with input analysis
            risk_level, risk_score, risk_factors = await self.anti_ban.assess_risk_level(
                user_id=message.from_user.id,
                action="text_message",
                context={
                    'message_length': len(message.text),
                    'response_time': time.time() - start_time,
                    'session_mode': session.conversation_mode.value,
                    'contains_urls': self._contains_urls(message.text),
                    'contains_mentions': self._contains_mentions(message.text),
                    'suspicious_patterns': self._detect_suspicious_patterns(message.text)
                }
            )
            
            # Apply risk mitigation if needed
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                mitigation = await self.anti_ban.apply_risk_mitigation(
                    message.from_user.id, risk_level, risk_score, "text_message"
                )
                
                if mitigation.get('force_delay', 0) > 0:
                    await asyncio.sleep(mitigation['force_delay'])
            
            # Check conversation state
            current_state = await state.get_state()
            state_data = await state.get_data()
            
            if current_state == ConversationStates.waiting_input:
                await self._handle_waiting_input(message, state, state_data, session)
            elif current_state == ConversationStates.multi_step_operation:
                await self._handle_multi_step_input(message, state, state_data, session)
            else:
                await self._handle_normal_conversation(message, session, risk_level)
            
            # Update session
            await self.session_manager.update_session_activity(
                session.session_id,
                message_context=context
            )
            
            # Record metrics
            processing_time = time.time() - start_time
            await self.metrics.record_message_received(
                message_type="text",
                processing_time=processing_time
            )
            
            await self.metrics.record_user_activity(
                user_id=message.from_user.id,
                language_code=message.from_user.language_code
            )
            
        except Exception as e:
            logger.error("Error handling text message", error=str(e))
            await self._handle_message_error(message, str(e))
            # Record security incident if this looks like an attack
            if await self._is_potential_attack(e):
                await self._record_security_incident(message.from_user.id, "message_processing_attack")
    
    async def _is_potential_attack(self, exception: Exception) -> bool:
        """Determine if exception indicates potential attack."""
        attack_indicators = [
            "malicious", "injection", "overflow", "invalid",
            "suspicious", "blocked", "security"
        ]
        
        error_str = str(exception).lower()
        return any(indicator in error_str for indicator in attack_indicators)
    
    async def _record_security_incident(self, user_id: int, incident_type: str) -> None:
        """Record security incident for monitoring."""
        try:
            incident_data = {
                'timestamp': time.time(),
                'user_id': user_id,
                'incident_type': incident_type,
                'severity': 'medium'
            }
            
            # Store in session context for analysis
            if hasattr(self, 'security_incidents'):
                self.security_incidents.append(incident_data)
            else:
                self.security_incidents = [incident_data]
            
            # Keep only recent incidents (last 1000)
            if len(self.security_incidents) > 1000:
                self.security_incidents = self.security_incidents[-500:]
                
            logger.warning(f"Security incident recorded: {incident_type} for user {user_id}")
            
        except Exception as e:
            logger.error("Error recording security incident", error=str(e))
    
    async def handle_callback_query(self, callback: CallbackQuery, state: FSMContext) -> None:
        """Handle inline keyboard callbacks."""
        try:
            await callback.answer()  # Acknowledge the callback
            
            # Parse callback data
            callback_data = json.loads(callback.data)
            action = callback_data.get('action')
            
            # Route to appropriate handler
            if action == 'settings':
                await self._handle_settings_callback(callback, callback_data, state)
            elif action == 'help':
                await self._handle_help_callback(callback, callback_data)
            elif action == 'status':
                await self._handle_status_callback(callback, callback_data)
            else:
                await callback.message.edit_text("❌ Unknown action.")
            
        except json.JSONDecodeError:
            await callback.message.edit_text("❌ Invalid callback data.")
        except Exception as e:
            logger.error("Error handling callback query", error=str(e))
            await callback.message.edit_text("❌ Error processing your request.")
    
    async def handle_inline_query(self, inline_query: InlineQuery) -> None:
        """Handle inline queries."""
        try:
            query = inline_query.query.strip()
            
            # Generate inline results based on query
            results = await self._generate_inline_results(query, inline_query.from_user.id)
            
            await inline_query.answer(
                results=results,
                cache_time=300,  # 5 minutes cache
                is_personal=True
            )
            
            # Record metrics
            await self.metrics.record_command_executed(
                command="inline_query",
                processing_time=0.1,  # Inline queries should be fast
                success=True
            )
            
        except Exception as e:
            logger.error("Error handling inline query", error=str(e))
            await inline_query.answer([])  # Return empty results on error
    
    async def handle_error(self, update: Update, exception: Exception) -> None:
        """Global error handler."""
        try:
            logger.error(
                "Unhandled error in update processing",
                update_type=type(update).__name__,
                error=str(exception),
                exc_info=True
            )
            
            # Record error metrics
            await self.metrics.increment_errors()
            
            # Try to send error message to user if possible
            if hasattr(update, 'message') and update.message:
                try:
                    await update.message.reply(
                        self.templates['error'],
                        reply_markup=self._get_error_recovery_keyboard()
                    )
                except Exception:
                    pass  # Can't send error message, just log
            
        except Exception as e:
            logger.error("Error in error handler", error=str(e))
    
    # Helper methods
    
    async def _create_message_context(self, message: Message) -> MessageContext:
        """Create message context for session tracking."""
        return MessageContext(
            message_id=message.message_id,
            timestamp=time.time(),
            message_type=message.content_type,
            content_type=message.content_type,
            text_length=len(message.text) if message.text else 0,
            has_entities=bool(message.entities),
            has_media=message.content_type != 'text',
            is_command=message.text and message.text.startswith('/') if message.text else False,
            command_name=message.text[1:].split()[0] if message.text and message.text.startswith('/') else None,
            reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None
        )
    
    async def _handle_normal_conversation(
        self,
        message: Message,
        session,
        risk_level: RiskLevel
    ) -> None:
        """Handle normal conversation flow."""
        try:
            # Generate response based on message content
            response = await self._generate_response(message.text, session)
            
            # Use enhanced typing simulation for message responses
            try:
                from app.services.typing_integration import typing_integration
                
                if typing_integration and typing_integration.enable_advanced_simulation:
                    # Get message context for enhanced simulation
                    message_context = None
                    if self.session_manager:
                        try:
                            session = await self.session_manager.get_session(message.from_user.id)
                            if session and hasattr(session, 'current_context'):
                                message_context = session.current_context
                        except Exception:
                            pass
                    
                    # Enhanced typing with full context
                    session_id = await typing_integration.start_realistic_typing_session(
                        text=response,
                        user_id=message.from_user.id,
                        chat_id=message.chat.id,
                        bot=message.bot,
                        message_context=message_context,
                        send_callback=lambda: message.reply(response)
                    )
                    
                    logger.debug(
                        "Enhanced conversation typing started",
                        session_id=session_id,
                        risk_level=risk_level.value,
                        response_length=len(response)
                    )
                    
                else:
                    raise ImportError("Enhanced typing not available")
                    
            except Exception as e:
                logger.debug("Using fallback conversation typing", error=str(e))
                
                # Fallback to original typing calculation
                typing_delay = await self.anti_ban.calculate_typing_delay(
                    text=response,
                    user_id=message.from_user.id,
                    context={
                        'is_command': False,
                        'risk_level': risk_level.value,
                        'message_length': len(response)
                    }
                )
                
                # Apply additional delay for high risk
                if risk_level == RiskLevel.HIGH:
                    typing_delay *= 1.5
                elif risk_level == RiskLevel.CRITICAL:
                    typing_delay *= 2.0
                
                # Show typing and send response
                await message.chat.do('typing')
                await asyncio.sleep(typing_delay)
                
                await message.reply(response)
            
            # Record successful interaction
            await self.metrics.record_anti_detection_event(
                pattern_type="natural_response",
                risk_score=0.0
            )
            
        except Exception as e:
            logger.error("Error in normal conversation", error=str(e))
            raise
    
    async def _generate_response(self, text: str, session) -> str:
        """Generate contextual response to user message."""
        try:
            # Simple response generation (in production, this would use ML/AI)
            text_lower = text.lower()
            
            # Greeting patterns
            if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
                responses = [
                    f"Hello {session.user_profile.first_name or 'there'}! How can I help you?",
                    "Hi! What can I do for you today?",
                    "Hey there! I'm here to assist you.",
                ]
                return responses[hash(text) % len(responses)]
            
            # Question patterns
            if '?' in text:
                return "That's an interesting question! Let me think about it... 🤔"
            
            # Default responses
            responses = [
                "I understand. Can you tell me more about that?",
                "That's interesting! What would you like to know?",
                "I'm here to help. What specific assistance do you need?",
                "Got it! How can I assist you further?",
            ]
            
            return responses[hash(text) % len(responses)]
            
        except Exception as e:
            logger.error("Error generating response", error=str(e))
            return "I'm here to help! What can I do for you?"
    
    async def _handle_waiting_input(
        self,
        message: Message,
        state: FSMContext,
        state_data: Dict[str, Any],
        session
    ) -> None:
        """Handle input when waiting for specific user response."""
        waiting_for = state_data.get('waiting_for')
        
        if waiting_for == 'settings_selection':
            await self._process_settings_input(message, state, session)
        else:
            # Generic input processing
            await self._process_generic_input(message, state, state_data, session)
    
    async def _handle_multi_step_input(
        self,
        message: Message,
        state: FSMContext,
        state_data: Dict[str, Any],
        session
    ) -> None:
        """Handle multi-step operation input."""
        operation = state_data.get('operation')
        step = state_data.get('step', 0)
        
        # Process based on operation type
        if operation == 'user_onboarding':
            await self._process_onboarding_step(message, state, step, session)
        else:
            await message.reply("❌ Unknown operation. Returning to normal mode.")
            await state.set_state(ConversationStates.idle)
    
    async def _get_commands_text(self) -> str:
        """Get formatted commands text."""
        commands = [
            "/start - Start the bot",
            "/help - Show this help message",
            "/settings - Manage your preferences",
            "/status - Show bot status (admin only)",
        ]
        
        return "\n".join(commands)
    
    async def _is_admin_user(self, user_id: int) -> bool:
        """Check if user is admin with secure validation."""
        try:
            # Get admin users from environment variable or database
            # This should be configured via environment variables in production
            from app.config import settings
            
            # Check environment variable for admin users
            admin_users_env = os.getenv('TELEGRAM_ADMIN_USERS', '')
            if admin_users_env:
                admin_users = [int(uid.strip()) for uid in admin_users_env.split(',') if uid.strip().isdigit()]
            else:
                # Fallback to hardcoded admin users (should be avoided in production)
                admin_users = []  # Configure this properly in production
            
            # Additional security: check if user has recent activity
            if user_id in admin_users:
                # Verify the user is actually an admin by checking recent interactions
                session = await self.session_manager.get_session_by_user_chat(
                    user_id, user_id  # For admin checks, use user_id as chat_id
                )
                
                if session:
                    # Admin users must have had recent activity (within last 24 hours)
                    now = time.time()
                    if now - session.last_activity < 86400:  # 24 hours
                        return True
                    else:
                        logger.warning(f"Admin user {user_id} has stale session")
                        return False
                
                # If no session, require admin to start the bot first
                logger.warning(f"Admin user {user_id} has no active session")
                return False
            
            return False
            
        except Exception as e:
            logger.error("Error checking admin status", error=str(e), user_id=user_id)
            return False
    
    def _format_status_message(self, bot_status: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Format status message."""
        try:
            uptime = metrics.get('uptime', 0)
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
            
            performance = metrics.get('performance', {})
            behavior = metrics.get('behavior', {})
            
            status_text = f"""
🤖 <b>Bot Status</b>

⏱️ <b>Uptime:</b> {uptime_str}
👥 <b>Active Users:</b> {behavior.get('active_users', 0)}
💬 <b>Total Messages:</b> {behavior.get('total_messages', 0)}
⚡ <b>Avg Response Time:</b> {performance.get('avg_response_time', 0):.2f}s

🖥️ <b>System Resources:</b>
• CPU: {performance.get('cpu_usage', 0):.1f}%
• Memory: {performance.get('memory_usage', 0):.1f}%
• Connections: {performance.get('concurrent_connections', 0)}

🔒 <b>Anti-Detection:</b>
• Risk Mitigations: {metrics.get('anti_detection', {}).get('risk_mitigations', 0)}
• Pattern Variations: {metrics.get('anti_detection', {}).get('patterns_applied', 0)}
"""
            
            return status_text.strip()
            
        except Exception as e:
            logger.error("Error formatting status message", error=str(e))
            return "❌ Error retrieving status information."
    
    async def _format_user_settings(self, session) -> str:
        """Format user settings display."""
        profile = session.user_profile
        
        settings_text = f"""
⚙️ <b>Your Settings</b>

👤 <b>Profile:</b>
• Name: {profile.first_name or 'Not set'}
• Username: @{profile.username or 'Not set'}
• Language: {profile.language_code or 'en'}

🎨 <b>Preferences:</b>
• Response Style: {profile.preferred_response_style}
• Timezone: {profile.timezone or 'Auto-detect'}

📊 <b>Activity:</b>
• Total Interactions: {session.total_interactions}
• Session Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.created_at))}

Use the buttons below to modify your settings.
"""
        
        return settings_text
    
    # Keyboard helpers
    
    def _get_start_keyboard(self) -> InlineKeyboardMarkup:
        """Get start command keyboard."""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔧 Settings",
                    callback_data=json.dumps({"action": "settings"})
                ),
                InlineKeyboardButton(
                    text="ℹ️ Help",
                    callback_data=json.dumps({"action": "help"})
                )
            ]
        ])
    
    def _get_help_keyboard(self) -> InlineKeyboardMarkup:
        """Get help command keyboard."""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🚀 Get Started",
                    callback_data=json.dumps({"action": "start"})
                )
            ]
        ])
    
    def _get_settings_keyboard(self) -> InlineKeyboardMarkup:
        """Get settings keyboard."""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎨 Response Style",
                    callback_data=json.dumps({"action": "settings", "type": "style"})
                ),
                InlineKeyboardButton(
                    text="🌍 Language",
                    callback_data=json.dumps({"action": "settings", "type": "language"})
                )
            ],
            [
                InlineKeyboardButton(
                    text="🕒 Timezone",
                    callback_data=json.dumps({"action": "settings", "type": "timezone"})
                )
            ]
        ])
    
    def _get_status_keyboard(self) -> InlineKeyboardMarkup:
        """Get status command keyboard."""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Refresh",
                    callback_data=json.dumps({"action": "status", "refresh": True})
                )
            ]
        ])
    
    def _get_error_recovery_keyboard(self) -> InlineKeyboardMarkup:
        """Get error recovery keyboard."""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Try Again",
                    callback_data=json.dumps({"action": "retry"})
                ),
                InlineKeyboardButton(
                    text="🏠 Start Over",
                    callback_data=json.dumps({"action": "start"})
                )
            ]
        ])
    
    # Error handling
    
    async def _handle_command_error(self, message: Message, command: str, error: str) -> None:
        """Handle command execution errors."""
        try:
            await message.reply(
                f"❌ Error executing /{command}: {error[:100]}...",
                reply_markup=self._get_error_recovery_keyboard()
            )
            
            # Record error metrics
            await self.metrics.record_command_executed(
                command=command,
                processing_time=0,
                success=False
            )
            
        except Exception as e:
            logger.error("Error in command error handler", error=str(e))
    
    async def _handle_message_error(self, message: Message, error: str) -> None:
        """Handle message processing errors."""
        try:
            await message.reply(self.templates['error'])
            
        except Exception as e:
            logger.error("Error in message error handler", error=str(e))
    
    async def _handle_unknown_command(self, message: Message) -> None:
        """Handle unknown commands."""
        try:
            await message.reply(
                self.templates['unknown_command'],
                reply_markup=self._get_help_keyboard()
            )
            
        except Exception as e:
            logger.error("Error handling unknown command", error=str(e))
    
    # Callback handlers
    
    async def _handle_settings_callback(
        self,
        callback: CallbackQuery,
        callback_data: Dict[str, Any],
        state: FSMContext
    ) -> None:
        """Handle settings-related callbacks."""
        setting_type = callback_data.get('type')
        
        if setting_type == 'style':
            await callback.message.edit_text(
                "🎨 Choose your preferred response style:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="📝 Formal",
                            callback_data=json.dumps({"action": "set_style", "value": "formal"})
                        ),
                        InlineKeyboardButton(
                            text="😊 Casual",
                            callback_data=json.dumps({"action": "set_style", "value": "casual"})
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            text="🔙 Back",
                            callback_data=json.dumps({"action": "settings"})
                        )
                    ]
                ])
            )
        # Add more setting type handlers as needed
    
    async def _handle_help_callback(self, callback: CallbackQuery, callback_data: Dict[str, Any]) -> None:
        """Handle help-related callbacks."""
        # Implement help callback logic
        pass
    
    async def _handle_status_callback(self, callback: CallbackQuery, callback_data: Dict[str, Any]) -> None:
        """Handle status-related callbacks."""
        if callback_data.get('refresh'):
            # Refresh status
            bot_status = await self.bot.get_status() if hasattr(self.bot, 'get_status') else {}
            metrics = await self.metrics.get_current_metrics()
            status_text = self._format_status_message(bot_status, metrics)
            
            await callback.message.edit_text(
                status_text,
                parse_mode='HTML',
                reply_markup=self._get_status_keyboard()
            )
    
    # Utility methods
    
    async def _update_session_and_metrics(
        self,
        message: Message,
        command: str,
        processing_time: float
    ) -> None:
        """Update session and metrics for command execution."""
        try:
            # Get session
            session = await self.session_manager.get_session_by_user_chat(
                message.from_user.id, message.chat.id
            )
            
            if session:
                # Update session
                context = await self._create_message_context(message)
                await self.session_manager.update_session_activity(
                    session.session_id,
                    message_context=context,
                    context_updates={'last_command': command}
                )
            
            # Update metrics
            await self.metrics.record_command_executed(
                command=command,
                processing_time=processing_time,
                success=True
            )
            
            await self.metrics.record_user_activity(
                user_id=message.from_user.id,
                language_code=message.from_user.language_code
            )
            
        except Exception as e:
            logger.error("Error updating session and metrics", error=str(e))
    
    async def _generate_inline_results(self, query: str, user_id: int) -> List:
        """Generate inline query results."""
        # Implement inline result generation
        # This is a placeholder implementation
        return []
    
    # Additional handler methods would be implemented here...
    
    async def _execute_registered_command(
        self,
        message: Message,
        command_name: str,
        command_config: Dict[str, Any],
        state: FSMContext
    ) -> None:
        """Execute a registered command."""
        # Implement registered command execution
        pass
    
    async def _process_settings_input(self, message: Message, state: FSMContext, session) -> None:
        """Process settings-related input."""
        # Implement settings input processing
        pass
    
    async def _process_generic_input(
        self,
        message: Message,
        state: FSMContext,
        state_data: Dict[str, Any],
        session
    ) -> None:
        """Process generic input when waiting."""
        # Implement generic input processing
        pass
    
    async def _process_onboarding_step(
        self,
        message: Message,
        state: FSMContext,
        step: int,
        session
    ) -> None:
        """Process onboarding step."""
        # Implement onboarding step processing
        pass


async def setup_handlers(dp: "Dispatcher", bot) -> None:
    """Setup message handlers with the dispatcher."""
    try:
        # Initialize handlers
        handlers = TelegramHandlers(bot)
        
        # Set component references
        handlers.session_manager = bot.session_manager
        handlers.anti_ban = bot.anti_ban
        handlers.metrics = bot.metrics
        handlers.rate_limiter = bot.rate_limiter
        
        # Include router
        dp.include_router(handlers.router)
        
        logger.info("Telegram handlers setup completed")
        
    except Exception as e:
        logger.error("Failed to setup handlers", error=str(e))
        raise