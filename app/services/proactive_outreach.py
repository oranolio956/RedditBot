"""
Proactive Outreach Service

Smart messaging service for context-aware proactive engagement campaigns.
Implements personalized outreach strategies based on behavioral analysis.
"""

import asyncio
import random
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json

from sqlalchemy import select, and_, or_, func, desc, asc
import structlog

from app.database.connection import get_database_session
from app.models.user import User
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, OutreachType, OutreachStatus,
    ProactiveOutreach, SentimentType, EngagementMilestone, UserMilestoneProgress
)
from app.services.engagement_analyzer import EngagementAnalyzer
from app.services.behavioral_predictor import BehavioralPredictor
from app.services.llm_service import LLMService

logger = structlog.get_logger(__name__)


class MessageTemplate:
    """Message template with dynamic content generation."""
    
    def __init__(self, template_id: str, base_message: str, personalization_slots: List[str]):
        self.template_id = template_id
        self.base_message = base_message
        self.personalization_slots = personalization_slots
    
    def generate(self, user_data: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Generate personalized message from template."""
        message = self.base_message
        context = context or {}
        
        # Replace personalization slots
        replacements = {
            '{name}': user_data.get('name', 'there'),
            '{first_name}': user_data.get('first_name', 'there'),
            '{days_absent}': str(context.get('days_absent', '0')),
            '{milestone_name}': context.get('milestone_name', 'milestone'),
            '{progress_percent}': str(context.get('progress_percent', '0')),
            '{topic}': context.get('topic', 'topic'),
            '{achievement}': context.get('achievement', 'achievement'),
            '{streak_days}': str(context.get('streak_days', '0')),
            '{total_interactions}': str(context.get('total_interactions', '0')),
            '{favorite_feature}': context.get('favorite_feature', 'feature')
        }
        
        for placeholder, value in replacements.items():
            message = message.replace(placeholder, str(value))
        
        return message


class ProactiveOutreachService:
    """
    Proactive engagement service with AI-powered personalization.
    
    Manages context-aware conversation starters, milestone celebrations,
    re-engagement campaigns, and personalized check-ins.
    """
    
    def __init__(
        self,
        analyzer: Optional[EngagementAnalyzer] = None,
        predictor: Optional[BehavioralPredictor] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.analyzer = analyzer or EngagementAnalyzer()
        self.predictor = predictor or BehavioralPredictor()
        self.llm_service = llm_service or LLMService()
        
        # Initialize message templates
        self.templates = self._initialize_templates()
        
        # Outreach configuration
        self.max_outreaches_per_user_per_day = 1
        self.min_hours_between_outreaches = 12
        self.optimal_hours = [9, 10, 11, 14, 15, 16, 19, 20]  # Generally good times
        
        # Effectiveness tracking
        self.template_effectiveness = {}
    
    async def schedule_proactive_outreach(
        self,
        user_id: str,
        outreach_type: OutreachType,
        priority_score: float = 0.5,
        context_data: Optional[Dict[str, Any]] = None,
        force_timing: Optional[datetime] = None
    ) -> Optional[ProactiveOutreach]:
        """
        Schedule a proactive outreach campaign for a user.
        
        Args:
            user_id: Target user UUID
            outreach_type: Type of outreach campaign
            priority_score: Priority score (0-1)
            context_data: Additional context for personalization
            force_timing: Override optimal timing calculation
            
        Returns:
            Created ProactiveOutreach record or None if skipped
        """
        try:
            async with get_database_session() as session:
                # Get user data
                user_result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = user_result.scalar_one_or_none()
                
                if not user or not user.is_active or user.is_blocked:
                    logger.info("Skipping outreach for inactive/blocked user", user_id=user_id)
                    return None
                
                # Check recent outreaches to avoid spam
                recent_cutoff = datetime.utcnow() - timedelta(hours=self.min_hours_between_outreaches)
                recent_outreaches = await session.execute(
                    select(func.count(ProactiveOutreach.id))
                    .where(
                        and_(
                            ProactiveOutreach.user_id == user_id,
                            ProactiveOutreach.created_at >= recent_cutoff,
                            ProactiveOutreach.status != OutreachStatus.CANCELLED
                        )
                    )
                )
                
                if recent_outreaches.scalar() > 0:
                    logger.info("Skipping outreach due to recent activity", user_id=user_id)
                    return None
                
                # Calculate optimal timing
                if force_timing:
                    scheduled_time = force_timing
                else:
                    optimal_timing = await self.analyzer.get_optimal_outreach_timing(user_id)
                    scheduled_time = optimal_timing or self._calculate_default_timing()
                
                # Generate personalized content
                message_content = await self._generate_personalized_message(
                    user=user,
                    outreach_type=outreach_type,
                    context=context_data or {}
                )
                
                if not message_content:
                    logger.warning("Failed to generate message content", user_id=user_id, outreach_type=outreach_type.value)
                    return None
                
                # Create outreach record
                outreach = ProactiveOutreach(
                    user_id=user_id,
                    telegram_id=user.telegram_id,
                    outreach_type=outreach_type,
                    priority_score=priority_score,
                    scheduled_for=scheduled_time,
                    optimal_timing_used=force_timing is None,
                    message_content=message_content['content'],
                    message_template=message_content['template_id'],
                    personalization_data=context_data,
                    trigger_event=context_data.get('trigger_event') if context_data else None,
                    trigger_data=context_data,
                    status=OutreachStatus.SCHEDULED
                )
                
                session.add(outreach)
                await session.commit()
                await session.refresh(outreach)
                
                logger.info(
                    "Proactive outreach scheduled",
                    user_id=user_id,
                    outreach_type=outreach_type.value,
                    scheduled_for=scheduled_time.isoformat(),
                    priority=priority_score,
                    outreach_id=str(outreach.id)
                )
                
                return outreach
                
        except Exception as e:
            logger.error("Error scheduling proactive outreach", error=str(e), user_id=user_id)
            raise
    
    async def process_scheduled_outreaches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Process scheduled outreaches that are ready to send.
        
        Args:
            limit: Maximum number of outreaches to process
            
        Returns:
            List of processing results
        """
        try:
            async with get_database_session() as session:
                # Get outreaches ready to send
                now = datetime.utcnow()
                
                outreaches_result = await session.execute(
                    select(ProactiveOutreach)
                    .where(
                        and_(
                            ProactiveOutreach.status == OutreachStatus.SCHEDULED,
                            ProactiveOutreach.scheduled_for <= now,
                            ProactiveOutreach.retry_count < ProactiveOutreach.max_retries
                        )
                    )
                    .order_by(desc(ProactiveOutreach.priority_score), asc(ProactiveOutreach.scheduled_for))
                    .limit(limit)
                )
                outreaches = outreaches_result.scalars().all()
                
                if not outreaches:
                    logger.info("No scheduled outreaches ready to process")
                    return []
                
                results = []
                
                for outreach in outreaches:
                    try:
                        result = await self._send_outreach(session, outreach)
                        results.append(result)
                        
                        # Small delay between sends to avoid rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(
                            "Error processing individual outreach",
                            error=str(e),
                            outreach_id=str(outreach.id)
                        )
                        
                        # Mark as failed
                        outreach.status = OutreachStatus.FAILED
                        outreach.failure_reason = str(e)
                        outreach.retry_count += 1
                        
                        results.append({
                            'outreach_id': str(outreach.id),
                            'status': 'failed',
                            'error': str(e)
                        })
                
                await session.commit()
                
                logger.info(
                    "Processed scheduled outreaches",
                    total_processed=len(results),
                    successful=len([r for r in results if r['status'] == 'sent']),
                    failed=len([r for r in results if r['status'] == 'failed'])
                )
                
                return results
                
        except Exception as e:
            logger.error("Error processing scheduled outreaches", error=str(e))
            raise
    
    async def create_milestone_celebration(
        self,
        user_id: str,
        milestone_id: str,
        achievement_data: Dict[str, Any]
    ) -> Optional[ProactiveOutreach]:
        """
        Create a milestone celebration outreach.
        
        Args:
            user_id: User UUID
            milestone_id: Milestone UUID
            achievement_data: Data about the achieved milestone
            
        Returns:
            Created outreach record or None
        """
        try:
            context = {
                'trigger_event': 'milestone_achieved',
                'milestone_name': achievement_data.get('name', 'milestone'),
                'achievement': achievement_data.get('description', ''),
                'milestone_id': milestone_id,
                **achievement_data
            }
            
            return await self.schedule_proactive_outreach(
                user_id=user_id,
                outreach_type=OutreachType.MILESTONE_CELEBRATION,
                priority_score=0.8,  # High priority for celebrations
                context_data=context
            )
            
        except Exception as e:
            logger.error("Error creating milestone celebration", error=str(e), user_id=user_id)
            return None
    
    async def create_re_engagement_campaign(
        self,
        user_id: str,
        absence_days: int,
        last_activity: Optional[str] = None
    ) -> Optional[ProactiveOutreach]:
        """
        Create a re-engagement campaign for dormant users.
        
        Args:
            user_id: User UUID
            absence_days: Days since last meaningful interaction
            last_activity: Description of last activity
            
        Returns:
            Created outreach record or None
        """
        try:
            # Calculate priority based on absence duration
            if absence_days > 30:
                priority = 0.9  # High priority for very dormant users
            elif absence_days > 14:
                priority = 0.7
            elif absence_days > 7:
                priority = 0.5
            else:
                priority = 0.3
            
            context = {
                'trigger_event': 're_engagement_needed',
                'days_absent': absence_days,
                'last_activity': last_activity,
                'campaign_type': 'win_back' if absence_days > 30 else 'check_in'
            }
            
            return await self.schedule_proactive_outreach(
                user_id=user_id,
                outreach_type=OutreachType.RE_ENGAGEMENT,
                priority_score=priority,
                context_data=context
            )
            
        except Exception as e:
            logger.error("Error creating re-engagement campaign", error=str(e), user_id=user_id)
            return None
    
    async def create_mood_support_outreach(
        self,
        user_id: str,
        mood_trend: float,
        recent_sentiment: SentimentType
    ) -> Optional[ProactiveOutreach]:
        """
        Create a mood support outreach for users showing negative trends.
        
        Args:
            user_id: User UUID
            mood_trend: Mood trend (-1 to 1)
            recent_sentiment: Recent sentiment classification
            
        Returns:
            Created outreach record or None
        """
        try:
            # Only create if mood is notably negative
            if mood_trend > -0.2 and recent_sentiment not in [SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE]:
                return None
            
            priority = 0.7 if recent_sentiment == SentimentType.VERY_NEGATIVE else 0.5
            
            context = {
                'trigger_event': 'negative_mood_detected',
                'mood_trend': mood_trend,
                'recent_sentiment': recent_sentiment.value,
                'support_type': 'emotional' if mood_trend < -0.5 else 'gentle'
            }
            
            return await self.schedule_proactive_outreach(
                user_id=user_id,
                outreach_type=OutreachType.MOOD_SUPPORT,
                priority_score=priority,
                context_data=context
            )
            
        except Exception as e:
            logger.error("Error creating mood support outreach", error=str(e), user_id=user_id)
            return None
    
    async def create_topic_follow_up(
        self,
        user_id: str,
        topic: str,
        last_mention_days: int,
        relevance_score: float
    ) -> Optional[ProactiveOutreach]:
        """
        Create a topic follow-up outreach based on user interests.
        
        Args:
            user_id: User UUID
            topic: Topic of interest
            last_mention_days: Days since last mention
            relevance_score: How relevant this topic is to the user
            
        Returns:
            Created outreach record or None
        """
        try:
            # Only follow up on highly relevant topics after reasonable time
            if relevance_score < 0.3 or last_mention_days < 2:
                return None
            
            priority = min(0.6, relevance_score)
            
            context = {
                'trigger_event': 'topic_follow_up',
                'topic': topic,
                'last_mention_days': last_mention_days,
                'relevance_score': relevance_score
            }
            
            return await self.schedule_proactive_outreach(
                user_id=user_id,
                outreach_type=OutreachType.TOPIC_FOLLOW_UP,
                priority_score=priority,
                context_data=context
            )
            
        except Exception as e:
            logger.error("Error creating topic follow-up", error=str(e), user_id=user_id)
            return None
    
    async def track_outreach_response(
        self,
        outreach_id: str,
        user_response: str,
        response_sentiment: SentimentType,
        response_time_minutes: int
    ) -> None:
        """
        Track user response to proactive outreach.
        
        Args:
            outreach_id: Outreach UUID
            user_response: User's response content
            response_sentiment: Sentiment of response
            response_time_minutes: Minutes between outreach and response
        """
        try:
            async with get_database_session() as session:
                outreach_result = await session.execute(
                    select(ProactiveOutreach).where(ProactiveOutreach.id == outreach_id)
                )
                outreach = outreach_result.scalar_one_or_none()
                
                if not outreach:
                    logger.warning("Outreach not found for response tracking", outreach_id=outreach_id)
                    return
                
                # Update response tracking
                outreach.user_responded = True
                outreach.response_time_minutes = response_time_minutes
                outreach.response_sentiment = response_sentiment
                outreach.response_content = user_response
                outreach.status = OutreachStatus.RESPONDED
                
                # Calculate effectiveness
                effectiveness = self._calculate_outreach_effectiveness(
                    outreach_type=outreach.outreach_type,
                    response_time=response_time_minutes,
                    response_sentiment=response_sentiment
                )
                outreach.effectiveness_score = effectiveness
                
                # Check if led to extended session
                # This would be determined by subsequent interactions
                # For now, positive responses suggest engagement
                if response_sentiment in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]:
                    outreach.led_to_extended_session = True
                    outreach.engagement_improvement = True
                
                await session.commit()
                
                # Update template effectiveness tracking
                await self._update_template_effectiveness(outreach)
                
                logger.info(
                    "Outreach response tracked",
                    outreach_id=outreach_id,
                    response_sentiment=response_sentiment.value,
                    response_time=response_time_minutes,
                    effectiveness=effectiveness
                )
                
        except Exception as e:
            logger.error("Error tracking outreach response", error=str(e), outreach_id=outreach_id)
    
    async def get_outreach_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get analytics on outreach campaign performance.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        try:
            async with get_database_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Overall metrics
                total_outreaches = await session.execute(
                    select(func.count(ProactiveOutreach.id))
                    .where(ProactiveOutreach.created_at >= cutoff_date)
                )
                
                successful_sends = await session.execute(
                    select(func.count(ProactiveOutreach.id))
                    .where(
                        and_(
                            ProactiveOutreach.created_at >= cutoff_date,
                            ProactiveOutreach.status.in_([OutreachStatus.SENT, OutreachStatus.DELIVERED, OutreachStatus.READ, OutreachStatus.RESPONDED])
                        )
                    )
                )
                
                responses = await session.execute(
                    select(func.count(ProactiveOutreach.id))
                    .where(
                        and_(
                            ProactiveOutreach.created_at >= cutoff_date,
                            ProactiveOutreach.user_responded == True
                        )
                    )
                )
                
                # Response rates by type
                type_stats_result = await session.execute(
                    select(
                        ProactiveOutreach.outreach_type,
                        func.count(ProactiveOutreach.id).label('total'),
                        func.sum(
                            func.case(
                                (ProactiveOutreach.user_responded == True, 1),
                                else_=0
                            )
                        ).label('responses'),
                        func.avg(ProactiveOutreach.effectiveness_score).label('avg_effectiveness')
                    )
                    .where(ProactiveOutreach.created_at >= cutoff_date)
                    .group_by(ProactiveOutreach.outreach_type)
                )
                type_stats = type_stats_result.all()
                
                # Template effectiveness
                template_stats_result = await session.execute(
                    select(
                        ProactiveOutreach.message_template,
                        func.count(ProactiveOutreach.id).label('usage_count'),
                        func.avg(ProactiveOutreach.effectiveness_score).label('avg_effectiveness'),
                        func.avg(ProactiveOutreach.response_time_minutes).label('avg_response_time')
                    )
                    .where(
                        and_(
                            ProactiveOutreach.created_at >= cutoff_date,
                            ProactiveOutreach.message_template.isnot(None)
                        )
                    )
                    .group_by(ProactiveOutreach.message_template)
                )
                template_stats = template_stats_result.all()
                
                total_count = total_outreaches.scalar() or 0
                sent_count = successful_sends.scalar() or 0
                response_count = responses.scalar() or 0
                
                analytics = {
                    'period_days': days,
                    'overall_metrics': {
                        'total_outreaches': total_count,
                        'successful_sends': sent_count,
                        'total_responses': response_count,
                        'send_rate': sent_count / max(1, total_count) * 100,
                        'response_rate': response_count / max(1, sent_count) * 100
                    },
                    'by_outreach_type': [
                        {
                            'type': row.outreach_type.value,
                            'total_sent': row.total,
                            'responses': row.responses or 0,
                            'response_rate': (row.responses or 0) / max(1, row.total) * 100,
                            'avg_effectiveness': round(row.avg_effectiveness or 0, 3)
                        }
                        for row in type_stats
                    ],
                    'template_performance': [
                        {
                            'template': row.message_template,
                            'usage_count': row.usage_count,
                            'avg_effectiveness': round(row.avg_effectiveness or 0, 3),
                            'avg_response_time_minutes': round(row.avg_response_time or 0, 1)
                        }
                        for row in template_stats
                        if row.message_template
                    ]
                }
                
                logger.info(
                    "Outreach analytics generated",
                    period_days=days,
                    total_outreaches=total_count,
                    response_rate=analytics['overall_metrics']['response_rate']
                )
                
                return analytics
                
        except Exception as e:
            logger.error("Error getting outreach analytics", error=str(e))
            raise
    
    # Private helper methods
    
    def _initialize_templates(self) -> Dict[OutreachType, List[MessageTemplate]]:
        """Initialize message templates for different outreach types."""
        return {
            OutreachType.MILESTONE_CELEBRATION: [
                MessageTemplate(
                    "celebration_basic",
                    "🎉 Congratulations {name}! You just achieved {milestone_name}! {achievement}",
                    ["name", "milestone_name", "achievement"]
                ),
                MessageTemplate(
                    "celebration_progress", 
                    "Amazing work {first_name}! 🌟 You've reached {milestone_name} after {total_interactions} interactions. Keep up the great progress!",
                    ["first_name", "milestone_name", "total_interactions"]
                ),
                MessageTemplate(
                    "celebration_streak",
                    "Incredible! 🔥 {name}, you've been active for {streak_days} days in a row and just unlocked {milestone_name}!",
                    ["name", "streak_days", "milestone_name"]
                )
            ],
            
            OutreachType.RE_ENGAGEMENT: [
                MessageTemplate(
                    "welcome_back",
                    "Hey {name}! 👋 I noticed you haven't been around for {days_absent} days. Hope everything's going well! Anything I can help you with?",
                    ["name", "days_absent"]
                ),
                MessageTemplate(
                    "gentle_check",
                    "Hi {first_name}! 😊 Just wanted to check in - you've been missed! What's been keeping you busy lately?",
                    ["first_name"]
                ),
                MessageTemplate(
                    "feature_update",
                    "Hello {name}! 🆕 I've learned some new tricks since we last talked {days_absent} days ago. Want to see what's new?",
                    ["name", "days_absent"]
                )
            ],
            
            OutreachType.PERSONALIZED_CHECKIN: [
                MessageTemplate(
                    "friendly_checkin",
                    "Hey {name}! 😊 How are things going today? Anything exciting happening?",
                    ["name"]
                ),
                MessageTemplate(
                    "activity_based",
                    "Hi {first_name}! I remember you were interested in {topic}. How's that going?",
                    ["first_name", "topic"]
                ),
                MessageTemplate(
                    "time_based",
                    "Good {time_of_day} {name}! 🌟 Hope your day is going wonderfully. What are you up to?",
                    ["time_of_day", "name"]
                )
            ],
            
            OutreachType.MOOD_SUPPORT: [
                MessageTemplate(
                    "supportive_gentle",
                    "Hey {name} 💙 I'm here if you need to chat about anything. Sometimes talking helps!",
                    ["name"]
                ),
                MessageTemplate(
                    "encouraging",
                    "Hi {first_name}! 🌈 Remember that tough times don't last, but resilient people like you do. I'm here to listen.",
                    ["first_name"]
                ),
                MessageTemplate(
                    "light_distraction",
                    "Hey {name}! 😊 Want to hear something that might brighten your day? I've got some fun facts or jokes ready!",
                    ["name"]
                )
            ],
            
            OutreachType.TOPIC_FOLLOW_UP: [
                MessageTemplate(
                    "topic_continuation",
                    "Hi {name}! 💭 I remember you mentioning {topic} a few days ago. Any updates on that?",
                    ["name", "topic"]
                ),
                MessageTemplate(
                    "related_suggestion",
                    "Hey {first_name}! Since you're interested in {topic}, I thought you might like to know about something related I learned!",
                    ["first_name", "topic"]
                ),
                MessageTemplate(
                    "expertise_offer",
                    "Hello {name}! 🎯 I've been thinking about your interest in {topic}. Want to dive deeper into it together?",
                    ["name", "topic"]
                )
            ],
            
            OutreachType.FEATURE_SUGGESTION: [
                MessageTemplate(
                    "new_feature",
                    "Hey {name}! 🚀 Based on how you use me, I think you'd love this new feature: {favorite_feature}. Want to try it?",
                    ["name", "favorite_feature"]
                ),
                MessageTemplate(
                    "personalized_tip",
                    "Hi {first_name}! 💡 Here's a tip that might help with what you're working on: try asking me about {topic}!",
                    ["first_name", "topic"]
                )
            ]
        }
    
    def _calculate_default_timing(self) -> datetime:
        """Calculate default timing for outreach."""
        now = datetime.utcnow()
        
        # Choose a random optimal hour
        optimal_hour = random.choice(self.optimal_hours)
        
        # If the optimal hour has passed today, schedule for tomorrow
        if now.hour >= optimal_hour:
            target_date = now.date() + timedelta(days=1)
        else:
            target_date = now.date()
        
        return datetime.combine(target_date, time(hour=optimal_hour))
    
    async def _generate_personalized_message(
        self,
        user: User,
        outreach_type: OutreachType,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Generate personalized message content."""
        try:
            # Get templates for this outreach type
            templates = self.templates.get(outreach_type, [])
            if not templates:
                logger.warning("No templates found for outreach type", outreach_type=outreach_type.value)
                return None
            
            # Choose template based on context or randomly
            template = self._choose_best_template(templates, context)
            
            # Prepare user data for personalization
            user_data = {
                'name': user.get_display_name(),
                'first_name': user.first_name or user.get_display_name(),
                'username': user.username
            }
            
            # Add time-based context
            current_hour = datetime.utcnow().hour
            if current_hour < 12:
                context['time_of_day'] = 'morning'
            elif current_hour < 17:
                context['time_of_day'] = 'afternoon'
            else:
                context['time_of_day'] = 'evening'
            
            # Generate base message
            base_message = template.generate(user_data, context)
            
            # Enhance with AI if available and message is important
            enhanced_message = base_message
            if self.llm_service and outreach_type in [OutreachType.MOOD_SUPPORT, OutreachType.MILESTONE_CELEBRATION]:
                try:
                    enhanced_message = await self._enhance_message_with_ai(base_message, user, context)
                except Exception as e:
                    logger.warning("AI enhancement failed, using base message", error=str(e))
            
            return {
                'content': enhanced_message,
                'template_id': template.template_id,
                'base_message': base_message
            }
            
        except Exception as e:
            logger.error("Error generating personalized message", error=str(e))
            return None
    
    def _choose_best_template(self, templates: List[MessageTemplate], context: Dict[str, Any]) -> MessageTemplate:
        """Choose the best template based on context and effectiveness."""
        if len(templates) == 1:
            return templates[0]
        
        # Choose based on context preferences
        context_type = context.get('campaign_type', '')
        
        if context_type == 'win_back':
            # Prefer feature update templates for win-back campaigns
            for template in templates:
                if 'feature' in template.template_id:
                    return template
        
        # Check template effectiveness if available
        best_template = None
        best_score = -1
        
        for template in templates:
            effectiveness = self.template_effectiveness.get(template.template_id, {})
            score = effectiveness.get('avg_effectiveness', 0.5)
            
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template or random.choice(templates)
    
    async def _enhance_message_with_ai(
        self,
        base_message: str,
        user: User,
        context: Dict[str, Any]
    ) -> str:
        """Enhance message with AI for better personalization."""
        try:
            prompt = f"""
            Enhance this message to be more personalized and engaging while keeping it concise and natural.
            
            Base message: {base_message}
            
            User context:
            - Display name: {user.get_display_name()}
            - Language: {user.language_code or 'en'}
            
            Situation context:
            {json.dumps(context, default=str, indent=2)}
            
            Guidelines:
            - Keep the same tone and intent
            - Make it feel personal but not intrusive
            - Maximum 2-3 sentences
            - Include appropriate emoji
            - Sound natural and conversational
            
            Enhanced message:
            """
            
            enhanced = await self.llm_service.get_completion(prompt, max_tokens=150)
            
            # Basic validation
            if enhanced and len(enhanced.strip()) > 10 and len(enhanced) < 500:
                return enhanced.strip()
            else:
                return base_message
                
        except Exception as e:
            logger.warning("AI message enhancement failed", error=str(e))
            return base_message
    
    async def _send_outreach(self, session, outreach: ProactiveOutreach) -> Dict[str, Any]:
        """Send a single outreach message."""
        try:
            # Mark as being sent
            outreach.status = OutreachStatus.SENT
            outreach.sent_at = datetime.utcnow()
            
            # Here you would integrate with your actual message sending system
            # For now, we'll simulate the send
            await self._simulate_message_send(outreach)
            
            # Mark as delivered (in real implementation, this would be updated by delivery confirmation)
            outreach.delivered_at = datetime.utcnow()
            outreach.status = OutreachStatus.DELIVERED
            
            logger.info(
                "Outreach sent successfully",
                outreach_id=str(outreach.id),
                user_id=str(outreach.user_id),
                outreach_type=outreach.outreach_type.value
            )
            
            return {
                'outreach_id': str(outreach.id),
                'status': 'sent',
                'sent_at': outreach.sent_at.isoformat(),
                'message': outreach.message_content
            }
            
        except Exception as e:
            raise e
    
    async def _simulate_message_send(self, outreach: ProactiveOutreach) -> None:
        """Simulate sending a message (replace with actual sending logic)."""
        # In a real implementation, this would:
        # 1. Send via Telegram Bot API
        # 2. Handle rate limits
        # 3. Handle delivery failures
        # 4. Update status based on API response
        
        logger.info(
            "SIMULATED MESSAGE SEND",
            telegram_id=outreach.telegram_id,
            message=outreach.message_content,
            outreach_type=outreach.outreach_type.value
        )
        
        # Simulate small delay
        await asyncio.sleep(0.1)
    
    def _calculate_outreach_effectiveness(
        self,
        outreach_type: OutreachType,
        response_time: int,
        response_sentiment: SentimentType
    ) -> float:
        """Calculate effectiveness score for an outreach."""
        effectiveness = 0.5  # Base score
        
        # Response time bonus (faster = better, up to 60 minutes)
        if response_time <= 60:
            effectiveness += 0.3
        elif response_time <= 240:  # 4 hours
            effectiveness += 0.2
        elif response_time <= 1440:  # 24 hours
            effectiveness += 0.1
        
        # Sentiment bonus
        if response_sentiment == SentimentType.VERY_POSITIVE:
            effectiveness += 0.3
        elif response_sentiment == SentimentType.POSITIVE:
            effectiveness += 0.2
        elif response_sentiment == SentimentType.NEGATIVE:
            effectiveness -= 0.1
        elif response_sentiment == SentimentType.VERY_NEGATIVE:
            effectiveness -= 0.2
        
        # Type-specific adjustments
        if outreach_type == OutreachType.MILESTONE_CELEBRATION:
            effectiveness += 0.1  # Celebrations are usually well-received
        elif outreach_type == OutreachType.MOOD_SUPPORT:
            # Mood support is effective if it gets any response
            effectiveness += 0.1
        
        return min(1.0, max(0.0, effectiveness))
    
    async def _update_template_effectiveness(self, outreach: ProactiveOutreach) -> None:
        """Update template effectiveness tracking."""
        if not outreach.message_template or outreach.effectiveness_score is None:
            return
        
        template_id = outreach.message_template
        
        if template_id not in self.template_effectiveness:
            self.template_effectiveness[template_id] = {
                'total_uses': 0,
                'total_effectiveness': 0.0,
                'response_count': 0
            }
        
        stats = self.template_effectiveness[template_id]
        stats['total_uses'] += 1
        stats['total_effectiveness'] += outreach.effectiveness_score
        
        if outreach.user_responded:
            stats['response_count'] += 1
        
        # Calculate averages
        stats['avg_effectiveness'] = stats['total_effectiveness'] / stats['total_uses']
        stats['response_rate'] = stats['response_count'] / stats['total_uses']