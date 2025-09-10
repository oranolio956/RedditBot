"""
Engagement Integration Example

Example integration showing how to integrate the proactive engagement system
with existing Telegram bot message handlers and conversation flows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import uuid4

import structlog
from aiogram import types
from aiogram.dispatcher import FSMContext

from app.models.user import User
from app.models.engagement import EngagementType, OutreachType
from app.services.engagement_analyzer import EngagementAnalyzer
from app.services.behavioral_predictor import BehavioralPredictor
from app.services.proactive_outreach import ProactiveOutreachService
from app.services.engagement_tasks import analyze_user_patterns_single
from app.database.connection import get_database_session

logger = structlog.get_logger(__name__)


class EngagementIntegration:
    """
    Integration layer for proactive engagement system.
    
    Provides easy-to-use methods for integrating engagement tracking
    and proactive outreach with existing bot message handlers.
    """
    
    def __init__(self):
        self.analyzer = EngagementAnalyzer()
        self.predictor = BehavioralPredictor()
        self.outreach_service = ProactiveOutreachService()
        
        # Session tracking for conversation context
        self.user_sessions = {}
        
        # Configuration
        self.enable_realtime_analysis = True
        self.enable_predictive_interventions = True
        self.enable_milestone_tracking = True
    
    async def track_message_interaction(
        self,
        user: User,
        message: types.Message,
        bot_response: Optional[str] = None,
        response_time_seconds: Optional[int] = None
    ) -> None:
        """
        Track a user message interaction with engagement analysis.
        
        Args:
            user: User model instance
            message: Telegram message object
            bot_response: Bot's response to the message
            response_time_seconds: Time user took to respond to previous bot message
        """
        try:
            # Get or create session ID
            session_id = self._get_session_id(user.telegram_id)
            
            # Get previous bot message for context
            previous_bot_message = self.user_sessions.get(user.telegram_id, {}).get('last_bot_message')
            
            # Analyze the interaction
            engagement = await self.analyzer.analyze_user_interaction(
                user_id=str(user.id),
                telegram_id=user.telegram_id,
                engagement_type=EngagementType.MESSAGE,
                message_text=message.text,
                session_id=session_id,
                previous_bot_message=previous_bot_message,
                response_time_seconds=response_time_seconds
            )
            
            # Update session data
            self.user_sessions[user.telegram_id] = {
                'session_id': session_id,
                'last_bot_message': bot_response,
                'last_interaction': datetime.utcnow(),
                'interaction_count': self.user_sessions.get(user.telegram_id, {}).get('interaction_count', 0) + 1
            }
            
            # Trigger proactive interventions if enabled
            if self.enable_predictive_interventions:
                await self._check_intervention_triggers(user, engagement)
            
            # Schedule background pattern analysis for significant interactions
            if engagement.engagement_quality_score and engagement.engagement_quality_score > 0.6:
                analyze_user_patterns_single.delay(str(user.id), user.telegram_id)
            
            logger.info(
                "Message interaction tracked",
                user_id=str(user.id),
                engagement_quality=engagement.engagement_quality_score,
                sentiment=engagement.sentiment_type.value if engagement.sentiment_type else None,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error("Error tracking message interaction", error=str(e), user_id=str(user.id))
    
    async def track_command_interaction(
        self,
        user: User,
        command_name: str,
        command_args: Optional[str] = None
    ) -> None:
        """
        Track a command interaction.
        
        Args:
            user: User model instance
            command_name: Name of the executed command
            command_args: Command arguments if any
        """
        try:
            session_id = self._get_session_id(user.telegram_id)
            
            engagement = await self.analyzer.analyze_user_interaction(
                user_id=str(user.id),
                telegram_id=user.telegram_id,
                engagement_type=EngagementType.COMMAND,
                command_name=command_name,
                message_text=command_args,
                session_id=session_id
            )
            
            # Commands always trigger pattern analysis (higher value interactions)
            analyze_user_patterns_single.delay(str(user.id), user.telegram_id)
            
            logger.info(
                "Command interaction tracked",
                user_id=str(user.id),
                command=command_name,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error("Error tracking command interaction", error=str(e), user_id=str(user.id))
    
    async def track_callback_interaction(
        self,
        user: User,
        callback_data: str,
        response_time_seconds: Optional[int] = None
    ) -> None:
        """
        Track a callback button interaction.
        
        Args:
            user: User model instance
            callback_data: Callback button data
            response_time_seconds: Time to respond to the callback
        """
        try:
            session_id = self._get_session_id(user.telegram_id)
            
            engagement = await self.analyzer.analyze_user_interaction(
                user_id=str(user.id),
                telegram_id=user.telegram_id,
                engagement_type=EngagementType.CALLBACK,
                message_text=callback_data,
                session_id=session_id,
                response_time_seconds=response_time_seconds
            )
            
            logger.info(
                "Callback interaction tracked",
                user_id=str(user.id),
                callback_data=callback_data,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error("Error tracking callback interaction", error=str(e), user_id=str(user.id))
    
    async def handle_outreach_response(
        self,
        user: User,
        message: types.Message,
        outreach_message_id: Optional[str] = None
    ) -> None:
        """
        Handle user response to a proactive outreach message.
        
        Args:
            user: User model instance
            message: User's response message
            outreach_message_id: ID of the outreach message being responded to
        """
        try:
            if not outreach_message_id:
                # Try to find recent outreach for this user
                outreach_message_id = await self._find_recent_outreach(user.telegram_id)
            
            if outreach_message_id:
                # Analyze sentiment of response
                _, sentiment_type = await self.analyzer._analyze_sentiment(message.text)
                
                # Calculate response time (simplified - would need to track send time)
                response_time_minutes = 30  # Placeholder - implement proper tracking
                
                # Track the response
                await self.outreach_service.track_outreach_response(
                    outreach_id=outreach_message_id,
                    user_response=message.text,
                    response_sentiment=sentiment_type,
                    response_time_minutes=response_time_minutes
                )
                
                logger.info(
                    "Outreach response tracked",
                    user_id=str(user.id),
                    outreach_id=outreach_message_id,
                    sentiment=sentiment_type.value
                )
            
            # Track as regular interaction too
            await self.track_message_interaction(user, message)
            
        except Exception as e:
            logger.error("Error handling outreach response", error=str(e), user_id=str(user.id))
    
    async def schedule_custom_outreach(
        self,
        user: User,
        outreach_type: OutreachType,
        context_data: Dict[str, Any],
        priority_score: float = 0.5
    ) -> bool:
        """
        Schedule a custom proactive outreach.
        
        Args:
            user: User model instance
            outreach_type: Type of outreach to schedule
            context_data: Context data for personalization
            priority_score: Priority score (0-1)
            
        Returns:
            True if outreach was scheduled successfully
        """
        try:
            outreach = await self.outreach_service.schedule_proactive_outreach(
                user_id=str(user.id),
                outreach_type=outreach_type,
                priority_score=priority_score,
                context_data=context_data
            )
            
            success = outreach is not None
            
            logger.info(
                "Custom outreach scheduled",
                user_id=str(user.id),
                outreach_type=outreach_type.value,
                success=success,
                outreach_id=str(outreach.id) if outreach else None
            )
            
            return success
            
        except Exception as e:
            logger.error("Error scheduling custom outreach", error=str(e), user_id=str(user.id))
            return False
    
    async def get_user_engagement_insights(self, user: User) -> Dict[str, Any]:
        """
        Get engagement insights for a specific user.
        
        Args:
            user: User model instance
            
        Returns:
            Dictionary with engagement insights
        """
        try:
            async with get_database_session() as session:
                # Get behavior pattern
                pattern_result = await session.execute(
                    "SELECT * FROM user_behavior_patterns WHERE user_id = :user_id",
                    {'user_id': str(user.id)}
                )
                pattern = pattern_result.fetchone()
                
                # Get recent predictions
                churn_prediction = await self.predictor.predict_churn_risk(str(user.id))
                mood_prediction = await self.predictor.predict_mood_trend(str(user.id))
                timing_prediction = await self.predictor.predict_optimal_timing(str(user.id))
                
                # Get milestone progress
                milestones_result = await session.execute(
                    """SELECT em.display_name, ump.progress_percentage, ump.is_achieved
                       FROM user_milestone_progress ump
                       JOIN engagement_milestones em ON ump.milestone_id = em.id
                       WHERE ump.user_id = :user_id
                       ORDER BY ump.progress_percentage DESC
                       LIMIT 5""",
                    {'user_id': str(user.id)}
                )
                milestones = [
                    {
                        'name': row.display_name,
                        'progress': row.progress_percentage,
                        'achieved': row.is_achieved
                    }
                    for row in milestones_result.fetchall()
                ]
                
                return {
                    'user_id': str(user.id),
                    'behavior_pattern': {
                        'total_interactions': pattern.total_interactions if pattern else 0,
                        'churn_risk_score': pattern.churn_risk_score if pattern else 0,
                        'engagement_trend': pattern.engagement_quality_trend if pattern else 0,
                        'days_since_last_interaction': pattern.days_since_last_interaction if pattern else 0,
                        'needs_re_engagement': pattern.needs_re_engagement if pattern else False
                    },
                    'predictions': {
                        'churn_risk': churn_prediction.prediction,
                        'churn_confidence': churn_prediction.confidence,
                        'mood_trend': mood_prediction.prediction,
                        'optimal_outreach_hour': timing_prediction.prediction
                    },
                    'milestones': milestones,
                    'session_data': self.user_sessions.get(user.telegram_id, {})
                }
                
        except Exception as e:
            logger.error("Error getting user engagement insights", error=str(e), user_id=str(user.id))
            return {'error': str(e)}
    
    # Private helper methods
    
    def _get_session_id(self, telegram_id: int) -> str:
        """Get or create session ID for user."""
        session_data = self.user_sessions.get(telegram_id, {})
        
        # Create new session if none exists or if too much time has passed
        if not session_data.get('session_id') or self._is_session_expired(session_data):
            session_id = str(uuid4())
            self.user_sessions[telegram_id] = {
                'session_id': session_id,
                'started_at': datetime.utcnow(),
                'interaction_count': 0
            }
            return session_id
        
        return session_data['session_id']
    
    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """Check if session has expired (30 minutes of inactivity)."""
        if not session_data.get('last_interaction'):
            return True
        
        time_since_last = datetime.utcnow() - session_data['last_interaction']
        return time_since_last > timedelta(minutes=30)
    
    async def _check_intervention_triggers(self, user: User, engagement) -> None:
        """Check if immediate interventions are needed based on interaction."""
        # Negative sentiment intervention
        if engagement.sentiment_type and engagement.sentiment_type.value in ['negative', 'very_negative']:
            await self.outreach_service.create_mood_support_outreach(
                user_id=str(user.id),
                mood_trend=-0.6,
                recent_sentiment=engagement.sentiment_type
            )
        
        # High engagement celebration (milestone-like)
        elif engagement.engagement_quality_score and engagement.engagement_quality_score > 0.9:
            await self.schedule_custom_outreach(
                user=user,
                outreach_type=OutreachType.MILESTONE_CELEBRATION,
                context_data={
                    'trigger_event': 'high_quality_interaction',
                    'quality_score': engagement.engagement_quality_score,
                    'achievement': 'Excellent conversation quality!'
                },
                priority_score=0.7
            )
    
    async def _find_recent_outreach(self, telegram_id: int) -> Optional[str]:
        """Find recent outreach message for response tracking."""
        try:
            async with get_database_session() as session:
                result = await session.execute(
                    """SELECT id FROM proactive_outreaches 
                       WHERE telegram_id = :telegram_id 
                       AND status IN ('sent', 'delivered') 
                       AND sent_at >= :cutoff_time
                       ORDER BY sent_at DESC 
                       LIMIT 1""",
                    {
                        'telegram_id': telegram_id,
                        'cutoff_time': datetime.utcnow() - timedelta(hours=2)
                    }
                )
                
                row = result.fetchone()
                return str(row.id) if row else None
                
        except Exception as e:
            logger.error("Error finding recent outreach", error=str(e), telegram_id=telegram_id)
            return None


# Example integration with aiogram handlers

# Global integration instance
engagement_integration = EngagementIntegration()


async def example_message_handler(message: types.Message, state: FSMContext):
    """
    Example message handler with engagement tracking.
    
    This shows how to integrate the engagement system with existing
    Telegram bot message handlers using aiogram.
    """
    try:
        # Get user from database (your existing user retrieval logic)
        user = await get_user_from_database(message.from_user.id)
        if not user:
            # Handle new user creation
            user = await create_new_user(message.from_user)
        
        # Calculate response time if this is a response to bot message
        user_data = await state.get_data()
        response_time = None
        if user_data.get('last_bot_message_time'):
            time_diff = datetime.utcnow() - user_data['last_bot_message_time']
            response_time = int(time_diff.total_seconds())
        
        # Your existing message processing logic
        response_text = await process_user_message(message.text, user, state)
        
        # Send response
        bot_message = await message.reply(response_text)
        
        # Track the interaction with engagement system
        await engagement_integration.track_message_interaction(
            user=user,
            message=message,
            bot_response=response_text,
            response_time_seconds=response_time
        )
        
        # Update state for next interaction
        await state.update_data(
            last_bot_message=response_text,
            last_bot_message_time=datetime.utcnow(),
            last_bot_message_id=bot_message.message_id
        )
        
    except Exception as e:
        logger.error("Error in message handler", error=str(e))


async def example_command_handler(message: types.Message, state: FSMContext, command: str):
    """
    Example command handler with engagement tracking.
    """
    try:
        user = await get_user_from_database(message.from_user.id)
        if not user:
            return
        
        # Your existing command processing logic
        response_text = await process_command(command, message.text, user, state)
        await message.reply(response_text)
        
        # Track command interaction
        await engagement_integration.track_command_interaction(
            user=user,
            command_name=command,
            command_args=message.text
        )
        
    except Exception as e:
        logger.error("Error in command handler", error=str(e))


async def example_callback_handler(callback_query: types.CallbackQuery, state: FSMContext):
    """
    Example callback handler with engagement tracking.
    """
    try:
        user = await get_user_from_database(callback_query.from_user.id)
        if not user:
            return
        
        # Calculate response time
        user_data = await state.get_data()
        response_time = None
        if user_data.get('last_bot_message_time'):
            time_diff = datetime.utcnow() - user_data['last_bot_message_time']
            response_time = int(time_diff.total_seconds())
        
        # Your existing callback processing logic
        await process_callback(callback_query.data, user, state)
        await callback_query.answer()
        
        # Track callback interaction
        await engagement_integration.track_callback_interaction(
            user=user,
            callback_data=callback_query.data,
            response_time_seconds=response_time
        )
        
    except Exception as e:
        logger.error("Error in callback handler", error=str(e))


# Placeholder functions (implement with your existing logic)
async def get_user_from_database(telegram_id: int) -> Optional[User]:
    """Retrieve user from database."""
    # Your implementation here
    pass

async def create_new_user(telegram_user) -> User:
    """Create new user from Telegram user object."""
    # Your implementation here
    pass

async def process_user_message(text: str, user: User, state: FSMContext) -> str:
    """Process user message and generate response."""
    # Your implementation here
    return "Response text"

async def process_command(command: str, args: str, user: User, state: FSMContext) -> str:
    """Process command and generate response."""
    # Your implementation here
    return "Command response"

async def process_callback(callback_data: str, user: User, state: FSMContext) -> None:
    """Process callback query."""
    # Your implementation here
    pass