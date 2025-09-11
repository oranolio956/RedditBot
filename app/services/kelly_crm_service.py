"""
Kelly CRM Service

Customer relationship management service with contact profiles, lead scoring,
deal management, and customer journey tracking.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
import uuid
from sqlalchemy import func, and_, or_, desc, asc
from sqlalchemy.orm import Session, joinedload

from app.database.connection import get_session
from app.models.kelly_crm import Contact, LeadScore, Deal, TouchPoint, CustomerJourney
from app.models.kelly_analytics import ConversationAnalytics, RevenueAttribution
from app.models.kelly_intelligence import ConversationInsight, TopicAnalysis
from app.core.redis import redis_manager
from app.core.claude_ai import claude_client

logger = logging.getLogger(__name__)

@dataclass
class ContactProfile:
    """Enhanced contact profile with AI insights"""
    contact_id: str
    basic_info: Dict[str, Any]
    ai_insights: Dict[str, Any]
    relationship_metrics: Dict[str, Any]
    communication_preferences: Dict[str, Any]
    business_context: Dict[str, Any]

@dataclass
class LeadScoringResult:
    """Lead scoring assessment result"""
    overall_score: float
    grade: str
    qualification_status: str
    component_scores: Dict[str, float]
    conversion_probability: float
    recommendations: List[str]

@dataclass
class CustomerJourneyAnalysis:
    """Customer journey analysis result"""
    journey_id: str
    current_stage: str
    stage_history: List[Dict[str, Any]]
    engagement_trend: str
    next_best_actions: List[str]
    risk_indicators: List[str]

class KellyCrmService:
    """
    Advanced CRM service for Kelly AI system.
    
    Provides contact management, lead scoring, deal tracking, and customer
    journey analysis with AI-powered insights and recommendations.
    """
    
    def __init__(self):
        self.cache_ttl = 1800  # 30 minutes cache for CRM data
        self.lead_score_weights = {
            'demographic': 0.20,
            'behavioral': 0.25,
            'engagement': 0.20,
            'fit': 0.15,
            'intent': 0.15,
            'timing': 0.05
        }
        
        # BANT scoring weights
        self.bant_weights = {
            'budget': 0.30,
            'authority': 0.25,
            'need': 0.25,
            'timeline': 0.20
        }
    
    async def get_contact_profile(
        self, 
        user_id: str,
        include_ai_insights: bool = True
    ) -> Optional[ContactProfile]:
        """
        Get comprehensive contact profile with AI insights.
        
        Args:
            user_id: User identifier
            include_ai_insights: Whether to include AI-generated insights
            
        Returns:
            Complete contact profile or None if not found
        """
        try:
            # Check cache first
            cache_key = f"contact_profile:{user_id}"
            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                return ContactProfile(**json.loads(cached_result))
            
            async with get_session() as session:
                contact = session.query(Contact).options(
                    joinedload(Contact.lead_scores),
                    joinedload(Contact.deals),
                    joinedload(Contact.touchpoints),
                    joinedload(Contact.customer_journey)
                ).filter(Contact.user_id == user_id).first()
                
                if not contact:
                    return None
                
                # Build basic info
                basic_info = {
                    'id': str(contact.id),
                    'user_id': contact.user_id,
                    'name': f"{contact.first_name or ''} {contact.last_name or ''}".strip(),
                    'display_name': contact.display_name,
                    'email': contact.email,
                    'phone': contact.phone,
                    'company': contact.company,
                    'job_title': contact.job_title,
                    'industry': contact.industry,
                    'country': contact.country,
                    'status': contact.status,
                    'created_at': contact.created_at.isoformat(),
                    'last_contact_at': contact.last_contact_at.isoformat() if contact.last_contact_at else None
                }
                
                # Build relationship metrics
                relationship_metrics = {
                    'relationship_strength': contact.relationship_strength,
                    'engagement_score': contact.engagement_score,
                    'trust_level': contact.trust_level,
                    'satisfaction_score': contact.satisfaction_score
                }
                
                # Build communication preferences
                communication_preferences = {
                    'preferred_channel': contact.preferred_communication_channel,
                    'frequency': contact.communication_frequency,
                    'best_contact_time': contact.best_contact_time,
                    'do_not_contact': contact.do_not_contact
                }
                
                # Build business context
                business_context = {
                    'authority_level': contact.authority_level,
                    'budget_range': contact.budget_range,
                    'need_urgency': contact.need_urgency,
                    'timeline': contact.timeline,
                    'decision_making_style': contact.decision_making_style
                }
                
                # Generate AI insights if requested
                ai_insights = {}
                if include_ai_insights:
                    ai_insights = await self._generate_contact_ai_insights(contact, session)
                
                profile = ContactProfile(
                    contact_id=str(contact.id),
                    basic_info=basic_info,
                    ai_insights=ai_insights,
                    relationship_metrics=relationship_metrics,
                    communication_preferences=communication_preferences,
                    business_context=business_context
                )
                
                # Cache the profile
                await redis_manager.set(
                    cache_key,
                    json.dumps(profile.__dict__, default=str),
                    expire=self.cache_ttl
                )
                
                return profile
                
        except Exception as e:
            logger.error(f"Error getting contact profile: {str(e)}")
            raise
    
    async def enrich_contact_from_conversations(
        self,
        user_id: str,
        conversation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enrich contact profile using conversation data and AI analysis.
        
        Args:
            user_id: User identifier
            conversation_data: List of conversation data
            
        Returns:
            Enrichment results and updated profile data
        """
        try:
            if not conversation_data:
                return {'message': 'No conversation data provided'}
            
            # Combine all conversations for analysis
            all_messages = []
            total_interactions = 0
            
            for conv in conversation_data:
                messages = conv.get('messages', [])
                all_messages.extend(messages)
                total_interactions += len(messages)
            
            if not all_messages:
                return {'message': 'No messages found in conversations'}
            
            # Use Claude AI to extract insights
            enrichment_prompt = f"""
            Analyze these conversations to extract customer insights:
            
            Total interactions: {total_interactions}
            Recent conversations: {json.dumps(conversation_data, indent=2)}
            
            Extract and infer:
            1. Personality traits (communication style, decision-making style)
            2. Business context (company size, industry, role, authority level)
            3. Pain points and challenges mentioned
            4. Interests and motivators
            5. Communication preferences (tone, frequency, channel)
            6. Budget indicators and timeline hints
            7. Relationship building opportunities
            
            Return as structured JSON with confidence scores for each insight.
            """
            
            claude_response = await claude_client.analyze_text(enrichment_prompt)
            enrichment_data = json.loads(claude_response)
            
            # Update contact profile with enriched data
            async with get_session() as session:
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
                
                if not contact:
                    # Create new contact if doesn't exist
                    contact = Contact(
                        user_id=user_id,
                        status='active',
                        contact_source='conversation_enrichment'
                    )
                    session.add(contact)
                
                # Update personality profile
                personality_insights = enrichment_data.get('personality', {})
                contact.personality_profile = personality_insights
                contact.communication_style = personality_insights.get('communication_style')
                contact.decision_making_style = personality_insights.get('decision_making_style')
                
                # Update business context
                business_insights = enrichment_data.get('business_context', {})
                if business_insights.get('company') and not contact.company:
                    contact.company = business_insights['company']
                if business_insights.get('industry') and not contact.industry:
                    contact.industry = business_insights['industry']
                if business_insights.get('job_title') and not contact.job_title:
                    contact.job_title = business_insights['job_title']
                if business_insights.get('authority_level'):
                    contact.authority_level = business_insights['authority_level']
                if business_insights.get('company_size'):
                    contact.company_size = business_insights['company_size']
                
                # Update interests and pain points
                contact.interests = enrichment_data.get('interests', [])
                contact.pain_points = enrichment_data.get('pain_points', [])
                contact.motivators = enrichment_data.get('motivators', [])
                
                # Update communication preferences
                comm_prefs = enrichment_data.get('communication_preferences', {})
                if comm_prefs.get('preferred_tone'):
                    contact.communication_style = comm_prefs['preferred_tone']
                if comm_prefs.get('frequency'):
                    contact.communication_frequency = comm_prefs['frequency']
                
                # Update business indicators
                budget_indicators = enrichment_data.get('budget_indicators', {})
                if budget_indicators.get('range'):
                    contact.budget_range = budget_indicators['range']
                if budget_indicators.get('urgency'):
                    contact.need_urgency = budget_indicators['urgency']
                if budget_indicators.get('timeline'):
                    contact.timeline = budget_indicators['timeline']
                
                # Calculate behavioral metrics
                contact.avg_response_time_minutes = await self._calculate_avg_response_time(conversation_data)
                contact.preferred_conversation_length = await self._determine_conversation_length_preference(conversation_data)
                
                # Update relationship strength based on conversation quality
                quality_scores = [conv.get('quality_score', 0.5) for conv in conversation_data]
                if quality_scores:
                    avg_quality = statistics.mean(quality_scores)
                    # Adjust relationship strength based on conversation quality
                    contact.relationship_strength = min(
                        contact.relationship_strength + (avg_quality - 0.5) * 0.2,
                        1.0
                    )
                
                contact.updated_at = datetime.utcnow()
                contact.last_activity_at = datetime.utcnow()
                
                await session.commit()
                
                # Trigger lead scoring update
                await self.calculate_lead_score(user_id)
                
                return {
                    'enrichment_success': True,
                    'insights_extracted': len(enrichment_data.keys()),
                    'profile_updates': {
                        'personality_insights': len(personality_insights.keys()) if personality_insights else 0,
                        'business_context_updates': len(business_insights.keys()) if business_insights else 0,
                        'interests_identified': len(contact.interests or []),
                        'pain_points_identified': len(contact.pain_points or [])
                    },
                    'enrichment_details': enrichment_data
                }
                
        except Exception as e:
            logger.error(f"Error enriching contact from conversations: {str(e)}")
            raise
    
    async def calculate_lead_score(
        self,
        user_id: str,
        force_recalculate: bool = False
    ) -> LeadScoringResult:
        """
        Calculate comprehensive AI-powered lead score.
        
        Args:
            user_id: User identifier
            force_recalculate: Force recalculation even if recent score exists
            
        Returns:
            Detailed lead scoring result
        """
        try:
            async with get_session() as session:
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
                
                if not contact:
                    raise ValueError(f"Contact not found for user_id: {user_id}")
                
                # Check for recent lead score
                if not force_recalculate:
                    recent_score = session.query(LeadScore).filter(
                        and_(
                            LeadScore.contact_id == contact.id,
                            LeadScore.scored_at >= datetime.utcnow() - timedelta(hours=24)
                        )
                    ).first()
                    
                    if recent_score:
                        return LeadScoringResult(
                            overall_score=recent_score.overall_score,
                            grade=recent_score.score_grade,
                            qualification_status=recent_score.qualification_status,
                            component_scores={
                                'demographic': recent_score.demographic_score,
                                'behavioral': recent_score.behavioral_score,
                                'engagement': recent_score.engagement_score,
                                'fit': recent_score.fit_score,
                                'intent': recent_score.intent_score,
                                'timing': recent_score.timing_score
                            },
                            conversion_probability=recent_score.conversion_probability or 0.0,
                            recommendations=recent_score.recommendations or []
                        )
                
                # Calculate component scores
                demographic_score = await self._calculate_demographic_score(contact)
                behavioral_score = await self._calculate_behavioral_score(contact, session)
                engagement_score = await self._calculate_engagement_score(contact, session)
                fit_score = await self._calculate_fit_score(contact)
                intent_score = await self._calculate_intent_score(contact, session)
                timing_score = await self._calculate_timing_score(contact)
                
                # Calculate BANT scores
                bant_scores = await self._calculate_bant_scores(contact)
                
                # Calculate weighted overall score
                component_scores = {
                    'demographic': demographic_score,
                    'behavioral': behavioral_score,
                    'engagement': engagement_score,
                    'fit': fit_score,
                    'intent': intent_score,
                    'timing': timing_score
                }
                
                overall_score = sum(
                    score * self.lead_score_weights.get(component, 0)
                    for component, score in component_scores.items()
                )
                
                # Determine grade and qualification status
                if overall_score >= 85:
                    grade = 'A+'
                    qualification_status = 'qualified'
                elif overall_score >= 80:
                    grade = 'A'
                    qualification_status = 'qualified'
                elif overall_score >= 75:
                    grade = 'B+'
                    qualification_status = 'qualified'
                elif overall_score >= 70:
                    grade = 'B'
                    qualification_status = 'nurturing'
                elif overall_score >= 60:
                    grade = 'C+'
                    qualification_status = 'nurturing'
                elif overall_score >= 50:
                    grade = 'C'
                    qualification_status = 'investigating'
                else:
                    grade = 'D'
                    qualification_status = 'unqualified'
                
                # Calculate conversion probability using AI model
                conversion_probability = await self._predict_conversion_probability(
                    contact, component_scores, bant_scores
                )
                
                # Generate recommendations
                recommendations = await self._generate_lead_recommendations(
                    contact, component_scores, qualification_status
                )
                
                # Store lead score
                lead_score = LeadScore(
                    contact_id=contact.id,
                    overall_score=overall_score,
                    score_grade=grade,
                    qualification_status=qualification_status,
                    demographic_score=demographic_score,
                    behavioral_score=behavioral_score,
                    engagement_score=engagement_score,
                    fit_score=fit_score,
                    intent_score=intent_score,
                    timing_score=timing_score,
                    budget_score=bant_scores['budget'],
                    authority_score=bant_scores['authority'],
                    need_score=bant_scores['need'],
                    timeline_score=bant_scores['timeline'],
                    scoring_confidence=0.85,  # High confidence for AI-based scoring
                    scoring_method='ai_model',
                    model_version='kelly_v1',
                    conversion_probability=conversion_probability,
                    recommendations=recommendations
                )
                
                session.add(lead_score)
                await session.commit()
                
                return LeadScoringResult(
                    overall_score=overall_score,
                    grade=grade,
                    qualification_status=qualification_status,
                    component_scores=component_scores,
                    conversion_probability=conversion_probability,
                    recommendations=recommendations
                )
                
        except Exception as e:
            logger.error(f"Error calculating lead score: {str(e)}")
            raise
    
    async def track_customer_journey(
        self,
        user_id: str,
        touchpoint_data: Dict[str, Any]
    ) -> CustomerJourneyAnalysis:
        """
        Track and analyze customer journey with AI insights.
        
        Args:
            user_id: User identifier
            touchpoint_data: New touchpoint information
            
        Returns:
            Customer journey analysis with recommendations
        """
        try:
            async with get_session() as session:
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
                
                if not contact:
                    raise ValueError(f"Contact not found for user_id: {user_id}")
                
                # Get or create customer journey
                journey = session.query(CustomerJourney).filter(
                    CustomerJourney.contact_id == contact.id
                ).first()
                
                if not journey:
                    journey = CustomerJourney(
                        contact_id=contact.id,
                        journey_status='active',
                        current_stage='awareness',
                        journey_started_at=datetime.utcnow()
                    )
                    session.add(journey)
                
                # Create new touchpoint
                touchpoint = TouchPoint(
                    contact_id=contact.id,
                    conversation_id=touchpoint_data.get('conversation_id'),
                    touchpoint_type=touchpoint_data.get('type', 'conversation'),
                    channel=touchpoint_data.get('channel', 'unknown'),
                    direction=touchpoint_data.get('direction', 'inbound'),
                    subject=touchpoint_data.get('subject'),
                    summary=touchpoint_data.get('summary'),
                    sentiment=touchpoint_data.get('sentiment'),
                    sentiment_score=touchpoint_data.get('sentiment_score'),
                    journey_stage=journey.current_stage,
                    engagement_quality=touchpoint_data.get('engagement_quality', 0.5),
                    ai_handled=touchpoint_data.get('ai_handled', False),
                    ai_confidence=touchpoint_data.get('ai_confidence'),
                    occurred_at=datetime.fromisoformat(
                        touchpoint_data.get('timestamp', datetime.utcnow().isoformat())
                    )
                )
                
                session.add(touchpoint)
                
                # Update journey metrics
                journey.total_touchpoints += 1
                journey.journey_updated_at = datetime.utcnow()
                
                # Determine if stage progression occurred
                previous_stage = journey.current_stage
                new_stage = await self._determine_journey_stage(contact, touchpoint_data)
                
                if new_stage != previous_stage:
                    journey.previous_stage = previous_stage
                    journey.current_stage = new_stage
                    journey.stage_entered_at = datetime.utcnow()
                    
                    # Update stage-specific timestamps
                    if new_stage == 'consideration' and not journey.consideration_entered_at:
                        journey.consideration_entered_at = datetime.utcnow()
                    elif new_stage == 'decision' and not journey.decision_entered_at:
                        journey.decision_entered_at = datetime.utcnow()
                    elif new_stage == 'retention' and not journey.retention_entered_at:
                        journey.retention_entered_at = datetime.utcnow()
                
                # Calculate engagement metrics
                all_touchpoints = session.query(TouchPoint).filter(
                    TouchPoint.contact_id == contact.id
                ).all()
                
                if len(all_touchpoints) > 1:
                    # Calculate average time between touchpoints
                    touchpoint_times = [tp.occurred_at for tp in all_touchpoints]
                    touchpoint_times.sort()
                    
                    intervals = [
                        (touchpoint_times[i] - touchpoint_times[i-1]).total_seconds() / 3600
                        for i in range(1, len(touchpoint_times))
                    ]
                    
                    journey.avg_time_between_touchpoints_hours = statistics.mean(intervals)
                    journey.longest_gap_hours = max(intervals) if intervals else 0
                
                # Update engagement trend
                recent_touchpoints = [tp for tp in all_touchpoints[-5:]]  # Last 5 touchpoints
                if recent_touchpoints:
                    engagement_scores = [tp.engagement_quality for tp in recent_touchpoints]
                    if len(engagement_scores) >= 3:
                        trend_slope = np.polyfit(range(len(engagement_scores)), engagement_scores, 1)[0]
                        if trend_slope > 0.1:
                            journey.engagement_trend = 'increasing'
                        elif trend_slope < -0.1:
                            journey.engagement_trend = 'decreasing'
                        else:
                            journey.engagement_trend = 'stable'
                
                # Generate AI recommendations
                journey_recommendations = await self._generate_journey_recommendations(
                    journey, all_touchpoints, touchpoint_data
                )
                
                journey.ai_recommendations = journey_recommendations
                journey.next_best_actions = journey_recommendations.get('next_actions', [])
                journey.risk_indicators = journey_recommendations.get('risk_indicators', [])
                
                await session.commit()
                
                # Build stage history
                stage_history = []
                if journey.awareness_entered_at:
                    stage_history.append({
                        'stage': 'awareness',
                        'entered_at': journey.awareness_entered_at.isoformat(),
                        'duration_hours': journey.awareness_duration_hours
                    })
                if journey.consideration_entered_at:
                    stage_history.append({
                        'stage': 'consideration',
                        'entered_at': journey.consideration_entered_at.isoformat(),
                        'duration_hours': journey.consideration_duration_hours
                    })
                if journey.decision_entered_at:
                    stage_history.append({
                        'stage': 'decision',
                        'entered_at': journey.decision_entered_at.isoformat(),
                        'duration_hours': journey.decision_duration_hours
                    })
                if journey.retention_entered_at:
                    stage_history.append({
                        'stage': 'retention',
                        'entered_at': journey.retention_entered_at.isoformat(),
                        'duration_hours': journey.retention_duration_hours
                    })
                
                return CustomerJourneyAnalysis(
                    journey_id=str(journey.id),
                    current_stage=journey.current_stage,
                    stage_history=stage_history,
                    engagement_trend=journey.engagement_trend or 'stable',
                    next_best_actions=journey.next_best_actions or [],
                    risk_indicators=journey.risk_indicators or []
                )
                
        except Exception as e:
            logger.error(f"Error tracking customer journey: {str(e)}")
            raise
    
    async def manage_deal_pipeline(
        self,
        deal_data: Dict[str, Any],
        action: str = 'create'  # create, update, advance, close
    ) -> Dict[str, Any]:
        """
        Manage deals in the sales pipeline with AI insights.
        
        Args:
            deal_data: Deal information
            action: Action to perform on the deal
            
        Returns:
            Deal management result with insights
        """
        try:
            async with get_session() as session:
                if action == 'create':
                    # Create new deal
                    contact = session.query(Contact).filter(
                        Contact.user_id == deal_data['user_id']
                    ).first()
                    
                    if not contact:
                        raise ValueError(f"Contact not found for user_id: {deal_data['user_id']}")
                    
                    deal = Deal(
                        deal_name=deal_data['name'],
                        contact_id=contact.id,
                        account_id=deal_data['account_id'],
                        deal_value=deal_data['value'],
                        currency=deal_data.get('currency', 'USD'),
                        probability=deal_data.get('probability', 0.5),
                        expected_close_date=datetime.fromisoformat(deal_data['expected_close_date']),
                        stage=deal_data.get('stage', 'qualification'),
                        source=deal_data.get('source', 'conversation'),
                        source_conversation_id=deal_data.get('conversation_id'),
                        attribution_type=deal_data.get('attribution_type', 'direct'),
                        ai_assisted=deal_data.get('ai_assisted', True),
                        owner_id=deal_data['owner_id']
                    )
                    
                    session.add(deal)
                    await session.commit()
                    
                    # Calculate AI contribution score
                    ai_contribution = await self._calculate_ai_contribution_score(deal, session)
                    deal.ai_contribution_score = ai_contribution
                    
                    await session.commit()
                    
                    return {
                        'action': 'created',
                        'deal_id': str(deal.id),
                        'ai_contribution_score': ai_contribution,
                        'pipeline_stage': deal.stage
                    }
                
                elif action == 'update':
                    # Update existing deal
                    deal = session.query(Deal).filter(Deal.id == deal_data['deal_id']).first()
                    
                    if not deal:
                        raise ValueError(f"Deal not found: {deal_data['deal_id']}")
                    
                    # Update fields
                    for field, value in deal_data.get('updates', {}).items():
                        if hasattr(deal, field):
                            setattr(deal, field, value)
                    
                    deal.updated_at = datetime.utcnow()
                    await session.commit()
                    
                    return {
                        'action': 'updated',
                        'deal_id': str(deal.id),
                        'updated_fields': list(deal_data.get('updates', {}).keys())
                    }
                
                elif action == 'advance':
                    # Advance deal to next stage
                    deal = session.query(Deal).filter(Deal.id == deal_data['deal_id']).first()
                    
                    if not deal:
                        raise ValueError(f"Deal not found: {deal_data['deal_id']}")
                    
                    # Determine next stage
                    stage_progression = {
                        'qualification': 'needs_analysis',
                        'needs_analysis': 'proposal',
                        'proposal': 'negotiation',
                        'negotiation': 'closed_won'
                    }
                    
                    current_stage = deal.stage
                    next_stage = stage_progression.get(current_stage, current_stage)
                    
                    if next_stage != current_stage:
                        deal.previous_stage = current_stage
                        deal.stage = next_stage
                        deal.stage_entered_at = datetime.utcnow()
                        
                        # Update probability based on stage
                        stage_probabilities = {
                            'qualification': 0.1,
                            'needs_analysis': 0.25,
                            'proposal': 0.5,
                            'negotiation': 0.75,
                            'closed_won': 1.0
                        }
                        deal.probability = stage_probabilities.get(next_stage, deal.probability)
                    
                    await session.commit()
                    
                    return {
                        'action': 'advanced',
                        'deal_id': str(deal.id),
                        'previous_stage': current_stage,
                        'current_stage': next_stage,
                        'new_probability': deal.probability
                    }
                
                elif action == 'close':
                    # Close deal (won or lost)
                    deal = session.query(Deal).filter(Deal.id == deal_data['deal_id']).first()
                    
                    if not deal:
                        raise ValueError(f"Deal not found: {deal_data['deal_id']}")
                    
                    deal.status = deal_data.get('status', 'won')  # won or lost
                    deal.closed_at = datetime.utcnow()
                    deal.actual_close_date = datetime.utcnow()
                    deal.close_reason = deal_data.get('reason')
                    
                    if deal.status == 'won':
                        deal.actual_deal_value = deal_data.get('actual_value', deal.deal_value)
                        
                        # Create revenue attribution if deal won
                        if deal.source_conversation_id:
                            from app.services.kelly_analytics_engine import kelly_analytics_engine
                            await kelly_analytics_engine.calculate_revenue_attribution(
                                deal.source_conversation_id,
                                {
                                    'amount': deal.actual_deal_value,
                                    'currency': deal.currency,
                                    'deal_id': str(deal.id),
                                    'journey_stage': 'decision'
                                }
                            )
                    
                    await session.commit()
                    
                    return {
                        'action': 'closed',
                        'deal_id': str(deal.id),
                        'status': deal.status,
                        'value': deal.actual_deal_value if deal.status == 'won' else 0,
                        'ai_contribution': deal.ai_contribution_score
                    }
                
                else:
                    raise ValueError(f"Unknown action: {action}")
                    
        except Exception as e:
            logger.error(f"Error managing deal pipeline: {str(e)}")
            raise
    
    # Helper methods for scoring and analysis
    
    async def _calculate_demographic_score(self, contact: Contact) -> float:
        """Calculate demographic score component"""
        score = 50.0  # Base score
        
        # Company size bonus
        company_size_scores = {
            'enterprise': 40,
            'large': 30,
            'medium': 20,
            'small': 10,
            'startup': 5
        }
        score += company_size_scores.get(contact.company_size, 0)
        
        # Industry relevance (would be configured based on business)
        high_value_industries = ['technology', 'finance', 'healthcare', 'manufacturing']
        if contact.industry and contact.industry.lower() in high_value_industries:
            score += 10
        
        return min(score, 100.0)
    
    async def _calculate_behavioral_score(self, contact: Contact, session: Session) -> float:
        """Calculate behavioral score based on interaction patterns"""
        score = 50.0  # Base score
        
        # Response time bonus (faster responders score higher)
        if contact.avg_response_time_minutes:
            if contact.avg_response_time_minutes <= 15:
                score += 20
            elif contact.avg_response_time_minutes <= 60:
                score += 15
            elif contact.avg_response_time_minutes <= 240:
                score += 10
        
        # Engagement consistency
        touchpoints = session.query(TouchPoint).filter(
            TouchPoint.contact_id == contact.id
        ).order_by(TouchPoint.occurred_at.desc()).limit(10).all()
        
        if touchpoints:
            avg_engagement = statistics.mean([tp.engagement_quality for tp in touchpoints])
            score += avg_engagement * 30  # Up to 30 points for high engagement
        
        return min(score, 100.0)
    
    async def _calculate_engagement_score(self, contact: Contact, session: Session) -> float:
        """Calculate engagement score based on interaction quality"""
        score = contact.engagement_score * 100  # Base from stored engagement score
        
        # Recent activity bonus
        if contact.last_activity_at:
            days_since_activity = (datetime.utcnow() - contact.last_activity_at).days
            if days_since_activity <= 7:
                score += 20
            elif days_since_activity <= 30:
                score += 10
        
        return min(score, 100.0)
    
    async def _calculate_fit_score(self, contact: Contact) -> float:
        """Calculate product/service fit score"""
        score = 50.0  # Base score
        
        # Authority level bonus
        authority_scores = {
            'decision_maker': 30,
            'influencer': 20,
            'user': 10,
            'blocker': 0
        }
        score += authority_scores.get(contact.authority_level, 10)
        
        # Budget range bonus
        if contact.budget_range:
            # This would be configured based on product pricing
            if 'high' in contact.budget_range.lower() or 'enterprise' in contact.budget_range.lower():
                score += 20
            elif 'medium' in contact.budget_range.lower():
                score += 15
        
        return min(score, 100.0)
    
    async def _calculate_intent_score(self, contact: Contact, session: Session) -> float:
        """Calculate purchase intent score based on conversation analysis"""
        score = 50.0  # Base score
        
        # Analyze recent topics for purchase intent
        recent_topics = session.query(TopicAnalysis).filter(
            and_(
                TopicAnalysis.user_id == contact.user_id,
                TopicAnalysis.analyzed_at >= datetime.utcnow() - timedelta(days=30)
            )
        ).all()
        
        if recent_topics:
            # High business relevance topics indicate intent
            business_relevant_topics = [t for t in recent_topics if t.business_relevance > 0.7]
            score += len(business_relevant_topics) * 10  # Up to significant bonus for relevant topics
            
            # Product-related topics
            product_relevant_topics = [t for t in recent_topics if t.product_relevance > 0.7]
            score += len(product_relevant_topics) * 15
        
        # Need urgency bonus
        urgency_scores = {
            'urgent': 30,
            'high': 20,
            'medium': 10,
            'low': 0
        }
        score += urgency_scores.get(contact.need_urgency, 5)
        
        return min(score, 100.0)
    
    async def _calculate_timing_score(self, contact: Contact) -> float:
        """Calculate timing score based on buying timeline"""
        score = 50.0  # Base score
        
        # Timeline urgency
        if contact.timeline:
            if 'immediate' in contact.timeline.lower() or 'now' in contact.timeline.lower():
                score += 30
            elif 'month' in contact.timeline.lower():
                score += 20
            elif 'quarter' in contact.timeline.lower():
                score += 15
            elif 'year' in contact.timeline.lower():
                score += 5
        
        return min(score, 100.0)
    
    async def _calculate_bant_scores(self, contact: Contact) -> Dict[str, float]:
        """Calculate BANT (Budget, Authority, Need, Timeline) scores"""
        bant_scores = {}
        
        # Budget score
        budget_score = 50.0
        if contact.budget_range:
            if 'high' in contact.budget_range.lower() or 'unlimited' in contact.budget_range.lower():
                budget_score = 100.0
            elif 'medium' in contact.budget_range.lower():
                budget_score = 75.0
            elif 'low' in contact.budget_range.lower():
                budget_score = 25.0
        bant_scores['budget'] = budget_score
        
        # Authority score
        authority_score = 50.0
        if contact.authority_level:
            authority_scores = {
                'decision_maker': 100.0,
                'influencer': 75.0,
                'user': 50.0,
                'blocker': 0.0
            }
            authority_score = authority_scores.get(contact.authority_level, 50.0)
        bant_scores['authority'] = authority_score
        
        # Need score (based on pain points and interests)
        need_score = 50.0
        if contact.pain_points:
            need_score += len(contact.pain_points) * 10  # More pain points = higher need
        if contact.interests:
            need_score += len(contact.interests) * 5  # Relevant interests indicate need
        bant_scores['need'] = min(need_score, 100.0)
        
        # Timeline score
        timeline_score = 50.0
        if contact.timeline:
            if 'immediate' in contact.timeline.lower():
                timeline_score = 100.0
            elif 'month' in contact.timeline.lower():
                timeline_score = 80.0
            elif 'quarter' in contact.timeline.lower():
                timeline_score = 60.0
            elif 'year' in contact.timeline.lower():
                timeline_score = 30.0
        bant_scores['timeline'] = timeline_score
        
        return bant_scores
    
    async def _predict_conversion_probability(
        self,
        contact: Contact,
        component_scores: Dict[str, float],
        bant_scores: Dict[str, float]
    ) -> float:
        """Use AI to predict conversion probability"""
        # Simplified model - in production, this would use a trained ML model
        
        # Weighted combination of scores
        weighted_score = (
            component_scores['engagement'] * 0.25 +
            component_scores['intent'] * 0.25 +
            component_scores['fit'] * 0.20 +
            bant_scores['budget'] * 0.15 +
            bant_scores['authority'] * 0.15
        )
        
        # Apply sigmoid function to convert to probability
        probability = 1 / (1 + np.exp(-(weighted_score - 50) / 20))
        
        return min(max(probability, 0.0), 1.0)
    
    async def _generate_lead_recommendations(
        self,
        contact: Contact,
        component_scores: Dict[str, float],
        qualification_status: str
    ) -> List[str]:
        """Generate AI-powered lead recommendations"""
        recommendations = []
        
        # Low engagement recommendations
        if component_scores['engagement'] < 60:
            recommendations.append("Increase engagement through personalized follow-ups")
            recommendations.append("Share relevant case studies or content")
        
        # Low intent recommendations
        if component_scores['intent'] < 60:
            recommendations.append("Conduct needs discovery call to understand pain points")
            recommendations.append("Provide product demonstration or trial")
        
        # Low fit recommendations
        if component_scores['fit'] < 60:
            recommendations.append("Qualify budget and decision-making process")
            recommendations.append("Identify key stakeholders and influencers")
        
        # Timing recommendations
        if component_scores['timing'] < 60:
            recommendations.append("Understand project timeline and urgency")
            recommendations.append("Create sense of urgency with limited-time offers")
        
        # Status-specific recommendations
        if qualification_status == 'qualified':
            recommendations.append("Schedule sales presentation or proposal meeting")
            recommendations.append("Prepare detailed ROI analysis")
        elif qualification_status == 'nurturing':
            recommendations.append("Enroll in nurture campaign with valuable content")
            recommendations.append("Schedule regular check-ins to monitor progress")
        
        return recommendations
    
    async def _determine_journey_stage(
        self,
        contact: Contact,
        touchpoint_data: Dict[str, Any]
    ) -> str:
        """Determine customer journey stage based on touchpoint data"""
        
        # Analyze touchpoint content for stage indicators
        content = touchpoint_data.get('summary', '') + ' ' + touchpoint_data.get('subject', '')
        content_lower = content.lower()
        
        # Decision stage indicators
        decision_keywords = ['price', 'cost', 'proposal', 'contract', 'timeline', 'implementation']
        if any(keyword in content_lower for keyword in decision_keywords):
            return 'decision'
        
        # Consideration stage indicators
        consideration_keywords = ['demo', 'features', 'compare', 'evaluation', 'requirements']
        if any(keyword in content_lower for keyword in consideration_keywords):
            return 'consideration'
        
        # Retention stage indicators (existing customers)
        retention_keywords = ['support', 'training', 'upgrade', 'renewal', 'expansion']
        if any(keyword in content_lower for keyword in retention_keywords):
            return 'retention'
        
        # Default to awareness if no specific indicators
        return 'awareness'
    
    async def _generate_journey_recommendations(
        self,
        journey: CustomerJourney,
        touchpoints: List[TouchPoint],
        latest_touchpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI recommendations for customer journey optimization"""
        
        recommendations = {
            'next_actions': [],
            'risk_indicators': [],
            'opportunities': []
        }
        
        # Stage-specific recommendations
        if journey.current_stage == 'awareness':
            recommendations['next_actions'].extend([
                "Share educational content about industry challenges",
                "Invite to webinar or educational event",
                "Conduct discovery call to understand needs"
            ])
        elif journey.current_stage == 'consideration':
            recommendations['next_actions'].extend([
                "Provide product demonstration",
                "Share relevant case studies",
                "Connect with product specialist"
            ])
        elif journey.current_stage == 'decision':
            recommendations['next_actions'].extend([
                "Prepare customized proposal",
                "Schedule stakeholder presentation",
                "Provide ROI calculator or business case"
            ])
        
        # Risk indicators based on engagement patterns
        if touchpoints:
            recent_engagement = [tp.engagement_quality for tp in touchpoints[-3:]]
            if recent_engagement and statistics.mean(recent_engagement) < 0.5:
                recommendations['risk_indicators'].append("Declining engagement - risk of losing interest")
            
            # Long gaps between touchpoints
            if journey.longest_gap_hours and journey.longest_gap_hours > 168:  # 1 week
                recommendations['risk_indicators'].append("Long gap in communication - risk of going cold")
        
        # Opportunity identification
        sentiment_score = latest_touchpoint.get('sentiment_score', 0)
        if sentiment_score > 0.5:
            recommendations['opportunities'].append("Positive sentiment - good time for advancement")
        
        if latest_touchpoint.get('ai_confidence', 0) > 0.8:
            recommendations['opportunities'].append("High AI confidence - conversation going well")
        
        return recommendations
    
    async def _calculate_ai_contribution_score(self, deal: Deal, session: Session) -> float:
        """Calculate AI contribution to deal success"""
        
        if not deal.source_conversation_id:
            return 0.0
        
        # Get conversation analytics
        conversation = session.query(ConversationAnalytics).filter(
            ConversationAnalytics.conversation_id == deal.source_conversation_id
        ).first()
        
        if not conversation:
            return 0.0
        
        # Calculate AI contribution based on conversation metrics
        ai_contribution = 0.0
        
        # AI handled most of the conversation
        if conversation.human_intervention_count == 0:
            ai_contribution += 0.5
        else:
            # Partial AI contribution based on message ratio
            ai_ratio = conversation.ai_messages / max(conversation.total_messages, 1)
            ai_contribution += ai_ratio * 0.5
        
        # High conversation quality indicates good AI performance
        quality_contribution = conversation.conversation_quality_score * 0.3
        ai_contribution += quality_contribution
        
        # Resolution achieved by AI
        if conversation.resolution_achieved and conversation.human_intervention_count == 0:
            ai_contribution += 0.2
        
        return min(ai_contribution, 1.0)
    
    async def _calculate_avg_response_time(self, conversation_data: List[Dict[str, Any]]) -> float:
        """Calculate average response time from conversation data"""
        response_times = []
        
        for conv in conversation_data:
            if conv.get('avg_response_time'):
                response_times.append(conv['avg_response_time'])
        
        return statistics.mean(response_times) if response_times else 0.0
    
    async def _determine_conversation_length_preference(
        self,
        conversation_data: List[Dict[str, Any]]
    ) -> str:
        """Determine preferred conversation length based on patterns"""
        
        durations = []
        for conv in conversation_data:
            if conv.get('duration_minutes'):
                durations.append(conv['duration_minutes'])
        
        if not durations:
            return 'medium'
        
        avg_duration = statistics.mean(durations)
        
        if avg_duration < 5:
            return 'short'
        elif avg_duration < 15:
            return 'medium'
        else:
            return 'long'
    
    async def _generate_contact_ai_insights(
        self,
        contact: Contact,
        session: Session
    ) -> Dict[str, Any]:
        """Generate AI insights for contact profile"""
        
        insights = {}
        
        # Get recent conversations for analysis
        recent_conversations = session.query(ConversationAnalytics).filter(
            and_(
                ConversationAnalytics.user_id == contact.user_id,
                ConversationAnalytics.started_at >= datetime.utcnow() - timedelta(days=30)
            )
        ).all()
        
        if recent_conversations:
            # Communication pattern insights
            avg_quality = statistics.mean([c.conversation_quality_score for c in recent_conversations])
            insights['communication_patterns'] = {
                'avg_conversation_quality': avg_quality,
                'total_conversations': len(recent_conversations),
                'ai_success_rate': len([c for c in recent_conversations if c.resolution_achieved and c.human_intervention_count == 0]) / len(recent_conversations)
            }
            
            # Engagement insights
            engagement_trend = await self._analyze_engagement_trend(recent_conversations)
            insights['engagement_analysis'] = engagement_trend
        
        # Lead scoring insights
        latest_lead_score = session.query(LeadScore).filter(
            LeadScore.contact_id == contact.id
        ).order_by(desc(LeadScore.scored_at)).first()
        
        if latest_lead_score:
            insights['lead_scoring'] = {
                'current_score': latest_lead_score.overall_score,
                'grade': latest_lead_score.score_grade,
                'qualification_status': latest_lead_score.qualification_status,
                'conversion_probability': latest_lead_score.conversion_probability
            }
        
        # Journey insights
        journey = session.query(CustomerJourney).filter(
            CustomerJourney.contact_id == contact.id
        ).first()
        
        if journey:
            insights['journey_analysis'] = {
                'current_stage': journey.current_stage,
                'engagement_trend': journey.engagement_trend,
                'total_touchpoints': journey.total_touchpoints,
                'journey_duration_days': (datetime.utcnow() - journey.journey_started_at).days
            }
        
        return insights
    
    async def _analyze_engagement_trend(self, conversations: List[ConversationAnalytics]) -> Dict[str, Any]:
        """Analyze engagement trend over time"""
        
        if len(conversations) < 3:
            return {'trend': 'insufficient_data'}
        
        # Sort by date
        sorted_conversations = sorted(conversations, key=lambda x: x.started_at)
        
        # Analyze quality trend
        quality_scores = [c.conversation_quality_score for c in sorted_conversations]
        
        # Simple linear trend analysis
        x_values = list(range(len(quality_scores)))
        slope = np.polyfit(x_values, quality_scores, 1)[0]
        
        if slope > 0.05:
            trend = 'improving'
        elif slope < -0.05:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'current_quality': quality_scores[-1],
            'avg_quality': statistics.mean(quality_scores),
            'quality_volatility': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
        }

# Create global CRM service instance
kelly_crm_service = KellyCrmService()