"""
Telegram Personality Handler

Integrates the personality system with Telegram bot handlers:
- Real-time personality adaptation for Telegram messages
- User personality learning from Telegram interactions
- Context-aware response generation
- Personality-based conversation management
- Feedback collection and learning
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
import logging

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import (
    ContextTypes, MessageHandler, CallbackQueryHandler, 
    ConversationHandler, filters
)

# Internal imports
from app.services.personality_manager import (
    PersonalityManager, PersonalityResponse, InteractionOutcome
)
from app.services.conversation_analyzer import ConversationContext
from app.models.user import User
from app.models.conversation import ConversationSession, Message, MessageDirection
from app.database.connection import get_db_session
from app.core.redis import get_redis_client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class TelegramPersonalityHandler:
    """
    Telegram bot handler with integrated personality system.
    
    This handler processes Telegram messages through the personality engine
    to provide adaptive, context-aware responses that learn from user interactions.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.personality_managers = {}  # Cache personality managers by session
        
        # Conversation states for feedback collection
        self.FEEDBACK_RATING = 1
        self.FEEDBACK_DETAILS = 2
        
        # Rate limiting
        self.user_message_counts = {}
        self.rate_limit_window = 60  # seconds
        self.max_messages_per_window = 30
        
        logger.info("Telegram personality handler initialized")
    
    async def get_personality_manager(self, session_key: str = "default") -> PersonalityManager:
        """Get or create personality manager for session."""
        if session_key not in self.personality_managers:
            db_session = await get_db_session().__anext__()
            redis_client = await get_redis_client()
            
            manager = PersonalityManager(db_session, redis_client)
            await manager.initialize()
            
            self.personality_managers[session_key] = manager
        
        return self.personality_managers[session_key]
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle incoming Telegram message with personality adaptation.
        
        This is the main entry point for personality-driven message processing.
        """
        try:
            user = update.effective_user
            message = update.message
            
            if not user or not message or not message.text:
                return
            
            # Rate limiting
            if not await self._check_rate_limit(str(user.id)):
                await message.reply_text(
                    "You're sending messages too quickly. Please slow down a bit! 😊"
                )
                return
            
            # Get or create user session
            session_id = await self._get_or_create_session(user.id)
            
            # Get personality manager
            manager = await self.get_personality_manager()
            
            # Process message through personality system
            personality_response = await manager.process_user_message(
                user_id=str(user.id),
                message_content=message.text,
                session_id=session_id,
                message_metadata=self._extract_telegram_metadata(message)
            )
            
            # Send personality-adapted response
            sent_message = await self._send_personality_response(
                message, personality_response
            )
            
            # Add feedback options periodically
            if await self._should_request_feedback(str(user.id), session_id):
                await self._add_feedback_buttons(sent_message)
            
            # Log interaction for analytics
            await self._log_interaction(
                user_id=str(user.id),
                session_id=session_id,
                user_message=message.text,
                bot_response=personality_response.content,
                personality_info=personality_response.adaptation_info
            )
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error_response(update.message)
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries for personality feedback and interactions."""
        try:
            query = update.callback_query
            await query.answer()
            
            user = update.effective_user
            if not user:
                return
            
            callback_data = json.loads(query.data)
            action = callback_data.get('action')
            
            if action == 'feedback_rating':
                await self._handle_feedback_rating(query, callback_data)
            elif action == 'feedback_details':
                await self._handle_feedback_details(query, callback_data)
            elif action == 'personality_info':
                await self._handle_personality_info_request(query, callback_data)
            elif action == 'change_personality':
                await self._handle_personality_change_request(query, callback_data)
            else:
                logger.warning(f"Unknown callback action: {action}")
                
        except Exception as e:
            logger.error(f"Error handling callback query: {e}")
    
    async def _send_personality_response(
        self, 
        original_message, 
        personality_response: PersonalityResponse
    ) -> Any:
        """Send personality-adapted response to user."""
        try:
            # Basic response
            response_text = personality_response.content
            
            # Add personality indicators for high confidence adaptations
            if personality_response.confidence_score > 0.8:
                adaptation_info = personality_response.adaptation_info
                if adaptation_info.get('adaptation_strategy') == 'mirror':
                    response_text += "\n\n*I'm matching your communication style* 🎭"
                elif adaptation_info.get('adaptation_strategy') == 'complement':
                    response_text += "\n\n*I'm adapting to complement your preferences* 🤝"
            
            # Send response
            sent_message = await original_message.reply_text(
                response_text,
                parse_mode='Markdown'
            )
            
            return sent_message
            
        except Exception as e:
            logger.error(f"Error sending personality response: {e}")
            return await original_message.reply_text(personality_response.content)
    
    async def _add_feedback_buttons(self, message) -> None:
        """Add feedback collection buttons to message."""
        try:
            keyboard = [
                [
                    InlineKeyboardButton("👍 Helpful", callback_data=json.dumps({
                        'action': 'feedback_rating',
                        'rating': 0.9,
                        'message_id': message.message_id
                    })),
                    InlineKeyboardButton("👎 Not helpful", callback_data=json.dumps({
                        'action': 'feedback_rating',
                        'rating': 0.2,
                        'message_id': message.message_id
                    }))
                ],
                [
                    InlineKeyboardButton("ℹ️ Personality Info", callback_data=json.dumps({
                        'action': 'personality_info',
                        'message_id': message.message_id
                    }))
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await message.reply_text(
                "How was my response? Your feedback helps me learn! 🤖",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error adding feedback buttons: {e}")
    
    async def _handle_feedback_rating(self, query: CallbackQuery, callback_data: Dict) -> None:
        """Handle feedback rating from user."""
        try:
            user = query.from_user
            rating = callback_data.get('rating', 0.5)
            message_id = callback_data.get('message_id')
            
            # Get session
            session_id = await self._get_current_session(user.id)
            if not session_id:
                await query.edit_message_text("Session not found. Thank you for the feedback!")
                return
            
            # Submit feedback
            manager = await self.get_personality_manager()
            await manager.provide_user_feedback(
                user_id=str(user.id),
                session_id=session_id,
                feedback_type='satisfaction',
                feedback_value=rating,
                feedback_details={'message_id': message_id, 'source': 'telegram_inline'}
            )
            
            # Update message
            if rating > 0.7:
                await query.edit_message_text("Thanks for the positive feedback! 😊 I'll keep learning from our interactions.")
            elif rating < 0.4:
                await query.edit_message_text(
                    "Thanks for letting me know. I'll work on improving! 🔧\n\n"
                    "Would you like to provide more specific feedback?",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("Add details", callback_data=json.dumps({
                            'action': 'feedback_details',
                            'rating': rating,
                            'message_id': message_id
                        }))
                    ]])
                )
            else:
                await query.edit_message_text("Thank you for your feedback! 👍")
            
        except Exception as e:
            logger.error(f"Error handling feedback rating: {e}")
            await query.edit_message_text("Thanks for your feedback!")
    
    async def _handle_feedback_details(self, query: CallbackQuery, callback_data: Dict) -> None:
        """Handle detailed feedback collection."""
        try:
            await query.edit_message_text(
                "Please type what I could do better in your next message. "
                "I'll use this to improve our conversations! 💬"
            )
            
            # Store callback data for next message
            user_id = str(query.from_user.id)
            await self._store_pending_feedback_context(user_id, callback_data)
            
        except Exception as e:
            logger.error(f"Error handling feedback details request: {e}")
    
    async def _handle_personality_info_request(self, query: CallbackQuery, callback_data: Dict) -> None:
        """Handle personality information request."""
        try:
            user = query.from_user
            manager = await self.get_personality_manager()
            
            # Get personality insights
            insights = await manager.get_user_personality_insights(str(user.id))
            
            if insights.get('status') == 'error':
                await query.edit_message_text("Sorry, I couldn't retrieve your personality information right now.")
                return
            
            # Format personality info for Telegram
            personality_text = self._format_personality_info(insights)
            
            # Add personality change option
            keyboard = [[
                InlineKeyboardButton("🔄 Try Different Personality", callback_data=json.dumps({
                    'action': 'change_personality'
                }))
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                personality_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error handling personality info request: {e}")
            await query.edit_message_text("Sorry, I couldn't retrieve your personality information.")
    
    async def _handle_personality_change_request(self, query: CallbackQuery, callback_data: Dict) -> None:
        """Handle request to change personality approach."""
        try:
            # Show available personality options
            keyboard = [
                [InlineKeyboardButton("🎭 Mirror my style", callback_data=json.dumps({
                    'action': 'set_strategy', 'strategy': 'mirror'
                }))],
                [InlineKeyboardButton("🤝 Complement my style", callback_data=json.dumps({
                    'action': 'set_strategy', 'strategy': 'complement'
                }))],
                [InlineKeyboardButton("⚖️ Balanced approach", callback_data=json.dumps({
                    'action': 'set_strategy', 'strategy': 'balance'
                }))],
                [InlineKeyboardButton("🔙 Back", callback_data=json.dumps({
                    'action': 'personality_info'
                }))]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "How would you like me to adapt my personality? 🤖\n\n"
                "• **Mirror**: I'll match your communication style\n"
                "• **Complement**: I'll adapt to balance your preferences\n"
                "• **Balanced**: I'll use a mix of approaches",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error handling personality change request: {e}")
    
    def _format_personality_info(self, insights: Dict[str, Any]) -> str:
        """Format personality insights for Telegram display."""
        try:
            personality_profile = insights.get('personality_profile', {})
            current_state = insights.get('current_state', {})
            engagement_metrics = insights.get('engagement_metrics', {})
            
            text = "🧠 **Your Personality Profile**\n\n"
            
            # Current personality traits
            if current_state.get('active_personality'):
                traits = current_state['active_personality']
                text += "**Current traits I've detected:**\n"
                
                for trait, score in traits.items():
                    if score > 0.6:
                        level = "High"
                        emoji = "🔥"
                    elif score < 0.4:
                        level = "Low"
                        emoji = "❄️"
                    else:
                        level = "Moderate"
                        emoji = "⚖️"
                    
                    trait_name = trait.replace('_', ' ').title()
                    text += f"• {emoji} {trait_name}: {level}\n"
                
                text += "\n"
            
            # Confidence and adaptation info
            confidence = current_state.get('confidence_level', 0)
            text += f"**Adaptation Confidence:** {confidence:.0%}\n"
            
            # Engagement metrics
            if engagement_metrics:
                engagement = engagement_metrics.get('overall_engagement', 0)
                text += f"**Engagement Score:** {engagement:.0%}\n"
            
            text += "\n*I use this information to adapt my responses to work better for you!* 🎯"
            
            return text
            
        except Exception as e:
            logger.error(f"Error formatting personality info: {e}")
            return "🧠 **Personality Information**\n\nI'm learning about your communication style to provide better responses!"
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        try:
            now = datetime.now()
            
            if user_id not in self.user_message_counts:
                self.user_message_counts[user_id] = []
            
            # Clean old timestamps
            cutoff = now - timedelta(seconds=self.rate_limit_window)
            self.user_message_counts[user_id] = [
                timestamp for timestamp in self.user_message_counts[user_id]
                if timestamp > cutoff
            ]
            
            # Check limit
            if len(self.user_message_counts[user_id]) >= self.max_messages_per_window:
                return False
            
            # Add current timestamp
            self.user_message_counts[user_id].append(now)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    async def _get_or_create_session(self, user_id: int) -> str:
        """Get or create conversation session for user."""
        try:
            # For simplicity, using a basic session management
            # In production, you'd want more sophisticated session handling
            redis_client = await get_redis_client()
            session_key = f"telegram_session:{user_id}"
            
            session_id = await redis_client.get(session_key)
            if session_id:
                return session_id.decode('utf-8')
            
            # Create new session
            import uuid
            session_id = str(uuid.uuid4())
            
            # Cache for 24 hours
            await redis_client.setex(session_key, 86400, session_id)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error managing session: {e}")
            return f"telegram_session_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    
    async def _get_current_session(self, user_id: int) -> Optional[str]:
        """Get current session ID for user."""
        try:
            redis_client = await get_redis_client()
            session_key = f"telegram_session:{user_id}"
            session_id = await redis_client.get(session_key)
            return session_id.decode('utf-8') if session_id else None
            
        except Exception as e:
            logger.error(f"Error getting current session: {e}")
            return None
    
    def _extract_telegram_metadata(self, message) -> Dict[str, Any]:
        """Extract metadata from Telegram message."""
        return {
            'telegram_message_id': message.message_id,
            'telegram_chat_id': message.chat_id,
            'telegram_user_id': message.from_user.id if message.from_user else None,
            'message_type': 'text',
            'timestamp': message.date.isoformat() if message.date else None,
            'platform': 'telegram'
        }
    
    async def _should_request_feedback(self, user_id: str, session_id: str) -> bool:
        """Determine if we should request feedback from user."""
        try:
            # Request feedback every 10 messages or 15 minutes
            redis_client = await get_redis_client()
            
            # Check message count
            msg_count_key = f"feedback_msg_count:{user_id}"
            msg_count = await redis_client.incr(msg_count_key)
            await redis_client.expire(msg_count_key, 3600)  # Reset hourly
            
            if msg_count % 10 == 0:
                return True
            
            # Check time since last feedback
            last_feedback_key = f"last_feedback:{user_id}"
            last_feedback = await redis_client.get(last_feedback_key)
            
            if not last_feedback:
                # First time - request feedback after 5 messages
                if msg_count >= 5:
                    await redis_client.setex(last_feedback_key, 3600, str(datetime.now().timestamp()))
                    return True
                return False
            
            last_time = datetime.fromtimestamp(float(last_feedback.decode('utf-8')))
            if datetime.now() - last_time > timedelta(minutes=15):
                await redis_client.setex(last_feedback_key, 3600, str(datetime.now().timestamp()))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking feedback timing: {e}")
            return False
    
    async def _store_pending_feedback_context(self, user_id: str, context_data: Dict) -> None:
        """Store context for pending feedback collection."""
        try:
            redis_client = await get_redis_client()
            context_key = f"pending_feedback:{user_id}"
            await redis_client.setex(context_key, 300, json.dumps(context_data))  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error storing feedback context: {e}")
    
    async def _get_pending_feedback_context(self, user_id: str) -> Optional[Dict]:
        """Get pending feedback context."""
        try:
            redis_client = await get_redis_client()
            context_key = f"pending_feedback:{user_id}"
            context_data = await redis_client.get(context_key)
            
            if context_data:
                await redis_client.delete(context_key)  # Clean up
                return json.loads(context_data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting feedback context: {e}")
            return None
    
    async def _log_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        bot_response: str,
        personality_info: Dict[str, Any]
    ) -> None:
        """Log interaction for analytics and monitoring."""
        try:
            interaction_log = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'session_id': session_id,
                'user_message_length': len(user_message),
                'bot_response_length': len(bot_response),
                'personality_profile': personality_info.get('personality_match', {}).get('profile_name'),
                'adaptation_strategy': personality_info.get('adaptation_strategy'),
                'confidence_score': personality_info.get('context_factors', {}).get('confidence', 0),
                'processing_time_ms': personality_info.get('processing_time_ms', 0),
                'platform': 'telegram'
            }
            
            redis_client = await get_redis_client()
            log_key = f"interaction_log:{datetime.now().strftime('%Y%m%d')}"
            await redis_client.lpush(log_key, json.dumps(interaction_log))
            await redis_client.expire(log_key, 86400 * 7)  # Keep for 7 days
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    async def _send_error_response(self, message) -> None:
        """Send error response to user."""
        try:
            error_responses = [
                "I'm having a bit of trouble right now. Could you try again? 🤖",
                "Something went wrong on my end. Please give me another chance! 😅",
                "Oops, I encountered an issue. Let's try that again! 🔧"
            ]
            
            import random
            response = random.choice(error_responses)
            await message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error sending error response: {e}")
    
    async def handle_detailed_feedback_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle detailed feedback message from user."""
        try:
            user = update.effective_user
            message = update.message
            
            if not user or not message or not message.text:
                return
            
            # Check if this is a feedback response
            feedback_context = await self._get_pending_feedback_context(str(user.id))
            if not feedback_context:
                # Not a feedback message, handle normally
                await self.handle_message(update, context)
                return
            
            # Process detailed feedback
            session_id = await self._get_current_session(user.id)
            if session_id:
                manager = await self.get_personality_manager()
                await manager.provide_user_feedback(
                    user_id=str(user.id),
                    session_id=session_id,
                    feedback_type='detailed_feedback',
                    feedback_value=feedback_context.get('rating', 0.5),
                    feedback_details={
                        'detailed_feedback': message.text,
                        'original_message_id': feedback_context.get('message_id'),
                        'source': 'telegram_text'
                    }
                )
            
            await message.reply_text(
                "Thank you for the detailed feedback! 🙏 "
                "I'll use this to improve our future conversations."
            )
            
        except Exception as e:
            logger.error(f"Error handling detailed feedback: {e}")
            await message.reply_text("Thank you for your feedback!")
    
    def get_handlers(self) -> List[Any]:
        """Get Telegram handlers for the personality system."""
        return [
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_detailed_feedback_message
            ),
            CallbackQueryHandler(self.handle_callback_query)
        ]
    
    async def cleanup(self) -> None:
        """Cleanup personality managers and resources."""
        for manager in self.personality_managers.values():
            await manager.cleanup()
        self.personality_managers.clear()
        
        logger.info("Telegram personality handler cleanup completed")


# Create global handler instance
telegram_personality_handler = TelegramPersonalityHandler()

# Export handler functions for use in main bot
async def handle_personality_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle message with personality adaptation."""
    await telegram_personality_handler.handle_message(update, context)

async def handle_personality_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle personality-related callbacks."""
    await telegram_personality_handler.handle_callback_query(update, context)

# Export for use in bot setup
__all__ = [
    'TelegramPersonalityHandler', 
    'telegram_personality_handler',
    'handle_personality_message',
    'handle_personality_callback'
]