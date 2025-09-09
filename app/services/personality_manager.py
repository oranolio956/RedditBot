"""
Personality Management Service

Central orchestrator for the personality system that coordinates:
- Personality analysis and adaptation
- Real-time conversation monitoring
- Personality matching and optimization
- Performance tracking and learning
- Integration with chat handlers and Telegram bot
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from contextlib import asynccontextmanager

# Database and Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update
from redis.asyncio import Redis

# Internal imports
from app.models.personality import (
    PersonalityProfile, UserPersonalityMapping, PersonalityTrait,
    AdaptationStrategy, PersonalityDimension
)
from app.models.user import User
from app.models.conversation import Message, ConversationSession, Conversation, MessageDirection
from app.services.personality_engine import (
    AdvancedPersonalityEngine, ConversationContext, PersonalityState
)
from app.services.conversation_analyzer import (
    ConversationAnalyzer, ConversationMetrics, EmotionalState, TopicAnalysis
)
from app.services.personality_matcher import (
    PersonalityMatcher, PersonalityMatch, MatchingContext
)
from app.database.repository import Repository
from app.core.redis import get_redis_client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PersonalityResponse:
    """Response generated with personality adaptation."""
    content: str
    personality_state: PersonalityState
    adaptation_info: Dict[str, Any]
    confidence_score: float
    processing_time_ms: int


@dataclass
class InteractionOutcome:
    """Outcome of a personality-driven interaction."""
    user_id: str
    session_id: str
    personality_match: PersonalityMatch
    conversation_metrics: ConversationMetrics
    user_satisfaction: float
    engagement_score: float
    effectiveness_score: float
    learning_feedback: Dict[str, Any]


class PersonalityManager:
    """
    Central coordinator for the AI personality system.
    
    This manager orchestrates all personality-related functionality:
    1. Analyzes user personality from conversations
    2. Matches optimal personality profiles
    3. Adapts responses in real-time
    4. Monitors conversation quality and user satisfaction
    5. Learns from interactions to improve performance
    6. Provides analytics and optimization insights
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: Optional[Redis] = None
    ):
        self.db = db_session
        self.redis = redis_client or get_redis_client()
        self.settings = get_settings()
        
        # Core personality system components
        self.personality_engine = AdvancedPersonalityEngine(db_session, self.redis)
        self.conversation_analyzer = ConversationAnalyzer(self.redis)
        self.personality_matcher = PersonalityMatcher(db_session, self.redis)
        
        # Real-time state management
        self.active_sessions = {}  # session_id -> session state
        self.user_personalities = {}  # user_id -> current personality state
        
        # Performance monitoring
        self.interaction_history = {}  # user_id -> interaction history
        self.system_metrics = {
            'total_interactions': 0,
            'successful_adaptations': 0,
            'user_satisfaction_average': 0.0,
            'performance_trends': []
        }
        
        # Background tasks
        self._background_tasks = set()
        self._monitoring_active = False
        
        logger.info("Personality manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the personality management system."""
        try:
            logger.info("Initializing personality management system...")
            
            # Initialize ML models and NLP tools
            await self.personality_engine.initialize_models()
            await self.conversation_analyzer.initialize_models()
            await self.personality_matcher.initialize_models()
            
            # Load existing personality profiles and mappings
            await self._load_personality_data()
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            logger.info("Personality management system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing personality manager: {e}")
            raise
    
    async def process_user_message(
        self,
        user_id: str,
        message_content: str,
        session_id: Optional[str] = None,
        message_metadata: Optional[Dict[str, Any]] = None
    ) -> PersonalityResponse:
        """
        Process user message and generate personality-adapted response.
        
        This is the main entry point for personality-driven conversation handling.
        """
        start_time = datetime.now()
        
        try:
            # Get or create session
            if not session_id:
                session_id = await self._create_conversation_session(user_id)
            
            # Get conversation context
            context = await self._build_conversation_context(user_id, session_id, message_content)
            
            # Analyze user personality
            user_traits = await self.personality_engine.analyze_user_personality(user_id, context)
            
            # Find optimal personality match
            matching_context = await self._build_matching_context(user_id, user_traits, context)
            personality_match = await self.personality_matcher.find_optimal_personality_match(matching_context)
            
            # Get personality profile
            base_profile = await self._get_personality_profile(personality_match.profile_id)
            if not base_profile:
                base_profile = await self._get_default_personality_profile()
            
            # Adapt personality for this interaction
            personality_state = await self.personality_engine.adapt_personality(
                user_id, user_traits, base_profile, context
            )
            
            # Store current personality state
            self.user_personalities[user_id] = personality_state
            
            # Generate base response (this would come from your chat engine)
            base_response = await self._generate_base_response(message_content, context)
            
            # Apply personality adaptation to response
            adapted_response = await self.personality_engine.generate_personality_response(
                personality_state, context, base_response
            )
            
            # Calculate processing time
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create response object
            response = PersonalityResponse(
                content=adapted_response,
                personality_state=personality_state,
                adaptation_info={
                    'personality_match': asdict(personality_match),
                    'adaptation_strategy': personality_match.adaptation_strategy.value,
                    'user_traits_detected': user_traits,
                    'context_factors': {
                        'phase': context.conversation_phase,
                        'engagement': context.user_engagement_level,
                        'urgency': max(context.urgency_indicators.values()) if context.urgency_indicators else 0.0
                    }
                },
                confidence_score=personality_state.confidence_level,
                processing_time_ms=processing_time
            )
            
            # Store message and track interaction
            await self._record_interaction(
                user_id, session_id, message_content, adapted_response, 
                personality_match, personality_state, context
            )
            
            # Update system metrics
            self.system_metrics['total_interactions'] += 1
            
            logger.info(f"Processed message for user {user_id} with personality {personality_match.profile_name} "
                       f"in {processing_time}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            
            # Fallback response
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return PersonalityResponse(
                content="I apologize, but I'm having trouble processing your message right now. Could you please try again?",
                personality_state=PersonalityState(
                    base_traits=self._get_neutral_traits(),
                    adapted_traits=self._get_neutral_traits(),
                    confidence_level=0.5,
                    adaptation_history=[],
                    effectiveness_metrics={},
                    last_updated=datetime.now()
                ),
                adaptation_info={'error': str(e)},
                confidence_score=0.3,
                processing_time_ms=processing_time
            )
    
    async def provide_user_feedback(
        self,
        user_id: str,
        session_id: str,
        feedback_type: str,
        feedback_value: float,
        feedback_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Process user feedback to improve personality adaptation.
        
        Args:
            user_id: User providing feedback
            session_id: Current conversation session
            feedback_type: Type of feedback (satisfaction, engagement, etc.)
            feedback_value: Numeric feedback value (0-1)
            feedback_details: Additional feedback details
        """
        try:
            # Get current personality state
            personality_state = self.user_personalities.get(user_id)
            if not personality_state:
                logger.warning(f"No personality state found for user {user_id}")
                return
            
            # Get conversation context
            context = await self._get_current_context(user_id, session_id)
            if not context:
                logger.warning(f"No conversation context found for session {session_id}")
                return
            
            # Create interaction outcome
            outcome = {
                'feedback_type': feedback_type,
                'feedback_value': feedback_value,
                'feedback_details': feedback_details or {},
                'satisfaction_score': feedback_value if feedback_type == 'satisfaction' else 0.7,
                'engagement_score': feedback_value if feedback_type == 'engagement' else 0.7,
                'effectiveness_score': feedback_value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Learn from feedback
            await self._learn_from_interaction(user_id, personality_state, context, outcome)
            
            # Update user personality mapping
            await self._update_user_personality_mapping(user_id, outcome)
            
            logger.info(f"Processed {feedback_type} feedback from user {user_id}: {feedback_value}")
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
    
    async def get_user_personality_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive personality insights for a user."""
        try:
            # Get user personality mapping
            mapping = await self._get_user_personality_mapping(user_id)
            if not mapping:
                return {'status': 'no_data', 'message': 'No personality data available for user'}
            
            # Get conversation history
            conversations = await self._get_user_conversation_history(user_id, days=30)
            
            # Analyze personality trends
            personality_trends = await self._analyze_personality_trends(user_id, conversations)
            
            # Get current personality state
            current_state = self.user_personalities.get(user_id)
            
            # Calculate engagement metrics
            engagement_metrics = await self._calculate_user_engagement_metrics(user_id, conversations)
            
            insights = {
                'user_id': user_id,
                'personality_profile': {
                    'measured_traits': mapping.measured_user_traits or {},
                    'adapted_traits': mapping.adapted_profile_traits or {},
                    'confidence_level': mapping.adaptation_confidence,
                    'learning_iterations': mapping.learning_iterations
                },
                'current_state': {
                    'active_personality': current_state.adapted_traits if current_state else None,
                    'confidence_level': current_state.confidence_level if current_state else None,
                    'last_updated': current_state.last_updated.isoformat() if current_state else None
                },
                'personality_trends': personality_trends,
                'engagement_metrics': engagement_metrics,
                'performance_indicators': {
                    'effectiveness_score': mapping.effectiveness_score,
                    'usage_count': mapping.usage_count,
                    'satisfaction_history': mapping.satisfaction_scores
                },
                'recommendations': await self._generate_personality_recommendations(mapping, conversations)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting personality insights for user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system-wide personality performance metrics."""
        try:
            # Get matching algorithm performance
            matching_stats = await self.personality_matcher.get_matching_performance_stats()
            
            # Calculate user satisfaction trends
            satisfaction_trends = await self._calculate_satisfaction_trends()
            
            # Get personality profile effectiveness
            profile_effectiveness = await self._calculate_profile_effectiveness()
            
            # System health metrics
            health_metrics = await self._calculate_system_health_metrics()
            
            metrics = {
                'system_overview': {
                    'total_users_with_personalities': len(self.user_personalities),
                    'active_sessions': len(self.active_sessions),
                    'total_interactions': self.system_metrics['total_interactions'],
                    'successful_adaptations': self.system_metrics['successful_adaptations'],
                    'average_satisfaction': self.system_metrics['user_satisfaction_average']
                },
                'matching_performance': matching_stats,
                'satisfaction_trends': satisfaction_trends,
                'profile_effectiveness': profile_effectiveness,
                'system_health': health_metrics,
                'recommendations': await self._generate_system_recommendations()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system performance metrics: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def optimize_personality_profiles(self) -> Dict[str, Any]:
        """Optimize personality profiles based on performance data."""
        try:
            logger.info("Starting personality profile optimization...")
            
            # Get all profiles and their performance data
            profiles = await self._get_all_personality_profiles()
            optimization_results = []
            
            for profile in profiles:
                # Analyze profile performance
                performance = await self._analyze_profile_performance(profile)
                
                # Generate optimization suggestions
                suggestions = await self._generate_profile_optimizations(profile, performance)
                
                if suggestions:
                    optimization_results.append({
                        'profile_id': str(profile.id),
                        'profile_name': profile.name,
                        'current_performance': performance,
                        'optimizations': suggestions,
                        'priority': self._calculate_optimization_priority(performance)
                    })
            
            # Sort by priority
            optimization_results.sort(key=lambda x: x['priority'], reverse=True)
            
            return {
                'status': 'success',
                'profiles_analyzed': len(profiles),
                'optimizations_found': len(optimization_results),
                'results': optimization_results[:10],  # Top 10 priorities
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing personality profiles: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def create_custom_personality_profile(
        self,
        name: str,
        description: str,
        trait_scores: Dict[str, float],
        behavioral_patterns: Optional[Dict[str, Any]] = None,
        communication_style: Optional[Dict[str, Any]] = None
    ) -> PersonalityProfile:
        """Create a new custom personality profile."""
        try:
            # Validate trait scores
            validated_traits = {}
            for trait_enum in PersonalityDimension:
                trait_name = trait_enum.value
                score = trait_scores.get(trait_name, 0.5)
                validated_traits[trait_name] = max(0.0, min(1.0, float(score)))
            
            # Create new profile
            profile = PersonalityProfile(
                name=name,
                display_name=name,
                description=description,
                category="custom",
                trait_scores=validated_traits,
                behavioral_patterns=behavioral_patterns or {},
                communication_style=communication_style or {},
                adaptation_strategy=AdaptationStrategy.BALANCE,
                adaptation_sensitivity=0.5,
                is_active=True,
                is_default=False
            )
            
            self.db.add(profile)
            await self.db.commit()
            await self.db.refresh(profile)
            
            logger.info(f"Created custom personality profile: {name}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating custom personality profile: {e}")
            raise
    
    # Private methods
    
    async def _load_personality_data(self) -> None:
        """Load existing personality data from database."""
        try:
            # Load personality profiles
            profiles_query = select(PersonalityProfile).where(PersonalityProfile.is_active == True)
            profiles_result = await self.db.execute(profiles_query)
            profiles = profiles_result.scalars().all()
            
            logger.info(f"Loaded {len(profiles)} personality profiles")
            
            # Load user personality mappings
            mappings_query = select(UserPersonalityMapping).where(
                UserPersonalityMapping.is_active == True
            ).options(selectinload(UserPersonalityMapping.profile))
            mappings_result = await self.db.execute(mappings_query)
            mappings = mappings_result.scalars().all()
            
            logger.info(f"Loaded {len(mappings)} user personality mappings")
            
        except Exception as e:
            logger.error(f"Error loading personality data: {e}")
    
    async def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start performance monitoring task
        task = asyncio.create_task(self._performance_monitoring_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        # Start personality optimization task
        task = asyncio.create_task(self._personality_optimization_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.info("Background monitoring tasks started")
    
    async def _performance_monitoring_task(self) -> None:
        """Background task for monitoring system performance."""
        while self._monitoring_active:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Update system metrics
                await self._update_system_metrics()
                
                # Check for performance issues
                await self._check_performance_alerts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _personality_optimization_task(self) -> None:
        """Background task for personality optimization."""
        while self._monitoring_active:
            try:
                await asyncio.sleep(3600)  # 1 hour
                
                # Run incremental optimizations
                await self._run_incremental_optimizations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in personality optimization task: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _build_conversation_context(
        self, 
        user_id: str, 
        session_id: str, 
        current_message: str
    ) -> ConversationContext:
        """Build conversation context from session history."""
        try:
            # Get recent messages
            messages = await self._get_session_messages(session_id)
            
            # Get user data
            user = await self._get_user_data(user_id)
            
            # Use conversation analyzer to build context
            context = await self.conversation_analyzer.analyze_conversation_context(
                session_id, messages, user
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            # Return minimal context
            return ConversationContext(
                session_id=session_id,
                user_id=user_id,
                conversation_phase="ongoing",
                time_in_conversation=0,
                message_count=0,
                user_engagement_level=0.5
            )
    
    async def _build_matching_context(
        self,
        user_id: str,
        user_traits: Dict[str, float],
        conversation_context: ConversationContext
    ) -> MatchingContext:
        """Build context for personality matching."""
        try:
            # Get emotional state
            messages = await self._get_session_messages(conversation_context.session_id)
            emotional_state = await self.conversation_analyzer.analyze_emotional_state(messages)
            
            # Get interaction history
            interaction_history = await self._get_user_interaction_history(user_id)
            
            # Get current performance
            current_performance = await self._get_current_performance(user_id)
            
            return MatchingContext(
                user_id=user_id,
                user_traits=user_traits,
                conversation_context=conversation_context,
                emotional_state=emotional_state,
                interaction_history=interaction_history,
                current_performance=current_performance
            )
            
        except Exception as e:
            logger.error(f"Error building matching context: {e}")
            # Return minimal context
            return MatchingContext(
                user_id=user_id,
                user_traits=user_traits,
                conversation_context=conversation_context,
                emotional_state=EmotionalState(),
                interaction_history=[]
            )
    
    async def _generate_base_response(
        self,
        message_content: str,
        context: ConversationContext
    ) -> str:
        """Generate base response before personality adaptation."""
        # This is a placeholder - in practice, this would integrate with your main chat engine
        # The personality system enhances whatever response your chat system generates
        
        if "hello" in message_content.lower() or "hi" in message_content.lower():
            return "Hello! How can I help you today?"
        elif "?" in message_content:
            return "That's a great question. Let me help you with that."
        elif any(word in message_content.lower() for word in ["help", "support", "assist"]):
            return "I'm here to help! What do you need assistance with?"
        elif any(word in message_content.lower() for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        else:
            return "I understand what you're saying. How would you like me to help?"
    
    async def _record_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        bot_response: str,
        personality_match: PersonalityMatch,
        personality_state: PersonalityState,
        context: ConversationContext
    ) -> None:
        """Record interaction for learning and analytics."""
        try:
            # Store messages in database
            user_msg = Message(
                session_id=session_id,
                user_id=user_id,
                content=user_message,
                message_type="text",
                direction=MessageDirection.INCOMING
            )
            
            bot_msg = Message(
                session_id=session_id,
                user_id=user_id,
                content=bot_response,
                message_type="text",
                direction=MessageDirection.OUTGOING,
                response_generated=True,
                response_model_used=f"personality_{personality_match.profile_name}"
            )
            
            self.db.add(user_msg)
            self.db.add(bot_msg)
            await self.db.commit()
            
            # Cache interaction data
            interaction_data = {
                'user_id': user_id,
                'session_id': session_id,
                'personality_profile': personality_match.profile_name,
                'adaptation_strategy': personality_match.adaptation_strategy.value,
                'confidence_score': personality_state.confidence_level,
                'context_phase': context.conversation_phase,
                'timestamp': datetime.now().isoformat()
            }
            
            cache_key = f"interaction:{user_id}:{session_id}:{datetime.now().timestamp()}"
            await self.redis.setex(cache_key, 3600, json.dumps(interaction_data))
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
    
    async def _learn_from_interaction(
        self,
        user_id: str,
        personality_state: PersonalityState,
        context: ConversationContext,
        outcome: Dict[str, Any]
    ) -> None:
        """Process interaction outcome for learning."""
        try:
            # Learn in personality engine
            await self.personality_engine.learn_from_interaction(
                user_id, personality_state, context, outcome
            )
            
            # Learn in personality matcher
            if 'personality_match' in outcome:
                match = outcome['personality_match']
                await self.personality_matcher.learn_from_interaction_outcome(
                    match, context, outcome
                )
            
            # Update system metrics
            self.system_metrics['successful_adaptations'] += 1
            
            # Update user satisfaction average
            if 'satisfaction_score' in outcome:
                current_avg = self.system_metrics['user_satisfaction_average']
                total = self.system_metrics['total_interactions']
                new_avg = ((current_avg * (total - 1)) + outcome['satisfaction_score']) / total
                self.system_metrics['user_satisfaction_average'] = new_avg
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    # Database helper methods
    
    async def _create_conversation_session(self, user_id: str) -> str:
        """Create new conversation session."""
        try:
            from app.models.conversation import ConversationSession
            import uuid
            
            session = ConversationSession(
                user_id=user_id,
                session_token=str(uuid.uuid4()),
                started_at=datetime.now()
            )
            
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            
            return str(session.id)
            
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            return str(uuid.uuid4())  # Fallback
    
    async def _get_personality_profile(self, profile_id: str) -> Optional[PersonalityProfile]:
        """Get personality profile by ID."""
        try:
            query = select(PersonalityProfile).where(PersonalityProfile.id == profile_id)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting personality profile {profile_id}: {e}")
            return None
    
    async def _get_default_personality_profile(self) -> PersonalityProfile:
        """Get default personality profile."""
        try:
            query = select(PersonalityProfile).where(
                PersonalityProfile.is_default == True,
                PersonalityProfile.is_active == True
            )
            result = await self.db.execute(query)
            profile = result.scalar_one_or_none()
            
            if profile:
                return profile
            
            # Create default profile if none exists
            return await self._create_default_profile()
            
        except Exception as e:
            logger.error(f"Error getting default personality profile: {e}")
            return await self._create_default_profile()
    
    async def _create_default_profile(self) -> PersonalityProfile:
        """Create default personality profile."""
        default_traits = self._get_neutral_traits()
        
        profile = PersonalityProfile(
            name="balanced_assistant",
            display_name="Balanced Assistant",
            description="A balanced, helpful AI assistant personality",
            category="default",
            trait_scores=default_traits,
            adaptation_strategy=AdaptationStrategy.BALANCE,
            adaptation_sensitivity=0.5,
            is_active=True,
            is_default=True
        )
        
        self.db.add(profile)
        await self.db.commit()
        await self.db.refresh(profile)
        
        return profile
    
    def _get_neutral_traits(self) -> Dict[str, float]:
        """Get neutral personality traits."""
        return {trait.value: 0.5 for trait in PersonalityDimension}
    
    # Additional helper methods would be implemented here...
    # For brevity, I'm including key method signatures:
    
    async def _get_session_messages(self, session_id: str) -> List[Message]:
        """Get messages for a session."""
        # Implementation would query Message table
        pass
    
    async def _get_user_data(self, user_id: str) -> Optional[User]:
        """Get user data."""
        # Implementation would query User table
        pass
    
    async def _get_user_personality_mapping(self, user_id: str) -> Optional[UserPersonalityMapping]:
        """Get user personality mapping."""
        # Implementation would query UserPersonalityMapping table
        pass
    
    async def _update_user_personality_mapping(self, user_id: str, outcome: Dict[str, Any]) -> None:
        """Update user personality mapping with outcome."""
        # Implementation would update mapping with new data
        pass
    
    async def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        # Implementation would calculate and update metrics
        pass
    
    async def _check_performance_alerts(self) -> None:
        """Check for performance issues and alerts."""
        # Implementation would monitor for problems
        pass
    
    async def _run_incremental_optimizations(self) -> None:
        """Run incremental personality optimizations."""
        # Implementation would perform optimization steps
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        self._monitoring_active = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Personality manager cleanup completed")


# Context manager for personality manager
@asynccontextmanager
async def personality_manager(db_session: AsyncSession, redis_client: Optional[Redis] = None):
    """Context manager for personality manager lifecycle."""
    manager = PersonalityManager(db_session, redis_client)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.cleanup()


# Export main classes
__all__ = [
    'PersonalityManager', 'PersonalityResponse', 'InteractionOutcome', 
    'personality_manager'
]