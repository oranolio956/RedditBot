"""
Kelly CRM API Endpoints

FastAPI endpoints for customer relationship management including contact profiles,
lead scoring, deal management, pipeline tracking, and customer journey analysis.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from app.database.connection import get_session
from app.core.auth import get_current_user
from app.core.redis import redis_manager
from app.services.kelly_crm_service import kelly_crm_service
from app.models.kelly_crm import Contact, LeadScore, Deal, TouchPoint, CustomerJourney

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation

class ContactSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    search_fields: List[str] = ['name', 'email', 'company']
    filters: Dict[str, Any] = {}
    limit: int = Field(default=20, ge=1, le=100)
    include_inactive: bool = False

class ContactUpdateRequest(BaseModel):
    basic_info: Optional[Dict[str, Any]] = None
    business_context: Optional[Dict[str, Any]] = None
    communication_preferences: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class ContactEnrichmentRequest(BaseModel):
    user_id: str
    conversation_data: List[Dict[str, Any]]
    override_existing: bool = False
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class LeadScoringRequest(BaseModel):
    user_id: str
    force_recalculate: bool = False
    include_breakdown: bool = True
    include_recommendations: bool = True

class DealRequest(BaseModel):
    deal_name: str = Field(..., max_length=200)
    user_id: str
    account_id: str
    deal_value: float = Field(..., gt=0)
    currency: str = Field(default="USD", max_length=10)
    expected_close_date: str  # ISO format
    stage: str = Field(default="qualification")
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Optional[str] = None
    conversation_id: Optional[str] = None
    owner_id: str
    products_interested: Optional[List[str]] = None
    notes: Optional[str] = None

class DealUpdateRequest(BaseModel):
    deal_id: str
    updates: Dict[str, Any]
    notes: Optional[str] = None
    stage_change_reason: Optional[str] = None

class TouchPointRequest(BaseModel):
    user_id: str
    touchpoint_type: str = Field(..., regex="^(conversation|email|call|meeting|demo)$")
    channel: str
    direction: str = Field(..., regex="^(inbound|outbound)$")
    subject: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = Field(None, regex="^(positive|negative|neutral)$")
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    engagement_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    ai_handled: bool = False
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    conversation_id: Optional[str] = None
    timestamp: Optional[str] = None  # ISO format

@router.get("/contacts", summary="Search and list contacts")
async def get_contacts(
    query: Optional[str] = Query(None, description="Search query"),
    company: Optional[str] = Query(None, description="Company filter"),
    industry: Optional[str] = Query(None, description="Industry filter"),
    status: Optional[str] = Query(None, description="Status filter"),
    lead_score_min: Optional[float] = Query(None, ge=0, le=100, description="Minimum lead score"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    sort_by: str = Query("last_contact_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Search and list contacts with filtering, pagination, and sorting.
    
    Provides comprehensive contact search with advanced filters,
    lead scoring integration, and activity-based sorting.
    """
    try:
        async with get_session() as session:
            # Build query
            query_builder = session.query(Contact)
            
            # Apply filters
            if query:
                search_filter = or_(
                    Contact.first_name.ilike(f"%{query}%"),
                    Contact.last_name.ilike(f"%{query}%"),
                    Contact.display_name.ilike(f"%{query}%"),
                    Contact.email.ilike(f"%{query}%"),
                    Contact.company.ilike(f"%{query}%")
                )
                query_builder = query_builder.filter(search_filter)
            
            if company:
                query_builder = query_builder.filter(Contact.company.ilike(f"%{company}%"))
            
            if industry:
                query_builder = query_builder.filter(Contact.industry == industry)
            
            if status:
                query_builder = query_builder.filter(Contact.status == status)
            else:
                # Default to active contacts only
                query_builder = query_builder.filter(Contact.status == 'active')
            
            # Lead score filter
            if lead_score_min is not None:
                from sqlalchemy import exists
                query_builder = query_builder.filter(
                    exists().where(
                        and_(
                            LeadScore.contact_id == Contact.id,
                            LeadScore.overall_score >= lead_score_min
                        )
                    )
                )
            
            # Apply sorting
            if sort_by == "last_contact_at":
                sort_field = Contact.last_contact_at
            elif sort_by == "created_at":
                sort_field = Contact.created_at
            elif sort_by == "company":
                sort_field = Contact.company
            elif sort_by == "engagement_score":
                sort_field = Contact.engagement_score
            else:
                sort_field = Contact.last_contact_at
            
            if sort_order == "desc":
                query_builder = query_builder.order_by(sort_field.desc())
            else:
                query_builder = query_builder.order_by(sort_field.asc())
            
            # Get total count
            total_count = query_builder.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            contacts = query_builder.offset(offset).limit(page_size).all()
            
            # Format contacts for response
            formatted_contacts = []
            for contact in contacts:
                # Get latest lead score
                latest_score = session.query(LeadScore).filter(
                    LeadScore.contact_id == contact.id
                ).order_by(LeadScore.scored_at.desc()).first()
                
                contact_data = {
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
                    'engagement_score': contact.engagement_score,
                    'relationship_strength': contact.relationship_strength,
                    'last_contact_at': contact.last_contact_at.isoformat() if contact.last_contact_at else None,
                    'created_at': contact.created_at.isoformat(),
                    'lead_score': {
                        'overall_score': latest_score.overall_score if latest_score else None,
                        'grade': latest_score.score_grade if latest_score else None,
                        'qualification_status': latest_score.qualification_status if latest_score else None
                    } if latest_score else None
                }
                formatted_contacts.append(contact_data)
            
            # Calculate pagination info
            total_pages = (total_count + page_size - 1) // page_size
            has_next = page < total_pages
            has_prev = page > 1
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'contacts': formatted_contacts,
                        'pagination': {
                            'current_page': page,
                            'page_size': page_size,
                            'total_count': total_count,
                            'total_pages': total_pages,
                            'has_next': has_next,
                            'has_prev': has_prev
                        }
                    },
                    'metadata': {
                        'filters_applied': {
                            'query': query,
                            'company': company,
                            'industry': industry,
                            'status': status,
                            'lead_score_min': lead_score_min
                        },
                        'sort_by': sort_by,
                        'sort_order': sort_order,
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"Error getting contacts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get contacts: {str(e)}")

@router.get("/contacts/{contact_id}", summary="Get detailed contact profile")
async def get_contact_profile(
    contact_id: str,
    include_ai_insights: bool = Query(True, description="Include AI-generated insights"),
    include_journey: bool = Query(True, description="Include customer journey"),
    include_touchpoints: bool = Query(True, description="Include recent touchpoints"),
    include_deals: bool = Query(True, description="Include associated deals"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get detailed contact profile with AI insights and relationship data.
    
    Provides comprehensive contact information including AI-generated
    insights, customer journey analysis, and business context.
    """
    try:
        async with get_session() as session:
            # Get contact by ID or user_id
            if contact_id.startswith('user_'):
                user_id = contact_id
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
            else:
                contact = session.query(Contact).filter(Contact.id == contact_id).first()
            
            if not contact:
                raise HTTPException(status_code=404, detail="Contact not found")
            
            # Get contact profile with AI insights
            profile = await kelly_crm_service.get_contact_profile(
                user_id=contact.user_id,
                include_ai_insights=include_ai_insights
            )
            
            if not profile:
                raise HTTPException(status_code=404, detail="Contact profile not found")
            
            response_data = {
                'contact_profile': {
                    'basic_info': profile.basic_info,
                    'relationship_metrics': profile.relationship_metrics,
                    'communication_preferences': profile.communication_preferences,
                    'business_context': profile.business_context
                }
            }
            
            if include_ai_insights:
                response_data['ai_insights'] = profile.ai_insights
            
            # Include customer journey if requested
            if include_journey:
                journey = session.query(CustomerJourney).filter(
                    CustomerJourney.contact_id == contact.id
                ).first()
                
                if journey:
                    response_data['customer_journey'] = {
                        'current_stage': journey.current_stage,
                        'journey_status': journey.journey_status,
                        'total_touchpoints': journey.total_touchpoints,
                        'engagement_trend': journey.engagement_trend,
                        'journey_started_at': journey.journey_started_at.isoformat(),
                        'stage_entered_at': journey.stage_entered_at.isoformat(),
                        'next_best_actions': journey.next_best_actions
                    }
            
            # Include recent touchpoints if requested
            if include_touchpoints:
                touchpoints = session.query(TouchPoint).filter(
                    TouchPoint.contact_id == contact.id
                ).order_by(TouchPoint.occurred_at.desc()).limit(10).all()
                
                response_data['recent_touchpoints'] = [
                    {
                        'id': str(tp.id),
                        'type': tp.touchpoint_type,
                        'channel': tp.channel,
                        'direction': tp.direction,
                        'subject': tp.subject,
                        'sentiment': tp.sentiment,
                        'engagement_quality': tp.engagement_quality,
                        'ai_handled': tp.ai_handled,
                        'occurred_at': tp.occurred_at.isoformat()
                    }
                    for tp in touchpoints
                ]
            
            # Include associated deals if requested
            if include_deals:
                deals = session.query(Deal).filter(
                    Deal.contact_id == contact.id
                ).order_by(Deal.created_at.desc()).all()
                
                response_data['deals'] = [
                    {
                        'id': str(deal.id),
                        'name': deal.deal_name,
                        'value': deal.deal_value,
                        'currency': deal.currency,
                        'stage': deal.stage,
                        'probability': deal.probability,
                        'status': deal.status,
                        'expected_close_date': deal.expected_close_date.isoformat(),
                        'created_at': deal.created_at.isoformat()
                    }
                    for deal in deals
                ]
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': response_data,
                    'metadata': {
                        'contact_id': str(contact.id),
                        'user_id': contact.user_id,
                        'includes': {
                            'ai_insights': include_ai_insights,
                            'journey': include_journey,
                            'touchpoints': include_touchpoints,
                            'deals': include_deals
                        },
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contact profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get contact profile: {str(e)}")

@router.put("/contacts/{contact_id}", summary="Update contact information")
async def update_contact(
    contact_id: str,
    request: ContactUpdateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update contact information and preferences.
    
    Allows updating basic information, business context,
    communication preferences, notes, and tags.
    """
    try:
        async with get_session() as session:
            # Get contact
            if contact_id.startswith('user_'):
                user_id = contact_id
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
            else:
                contact = session.query(Contact).filter(Contact.id == contact_id).first()
            
            if not contact:
                raise HTTPException(status_code=404, detail="Contact not found")
            
            # Update basic info
            if request.basic_info:
                for field, value in request.basic_info.items():
                    if hasattr(contact, field) and value is not None:
                        setattr(contact, field, value)
            
            # Update business context
            if request.business_context:
                for field, value in request.business_context.items():
                    if hasattr(contact, field) and value is not None:
                        setattr(contact, field, value)
            
            # Update communication preferences
            if request.communication_preferences:
                for field, value in request.communication_preferences.items():
                    if hasattr(contact, field) and value is not None:
                        setattr(contact, field, value)
            
            # Update notes and tags
            if request.notes is not None:
                contact.notes = request.notes
            
            if request.tags is not None:
                contact.tags = request.tags
            
            # Update timestamp
            contact.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Trigger lead score recalculation
            await kelly_crm_service.calculate_lead_score(contact.user_id, force_recalculate=True)
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'contact_id': str(contact.id),
                        'user_id': contact.user_id,
                        'updated_fields': list(set(
                            list(request.basic_info.keys() if request.basic_info else []) +
                            list(request.business_context.keys() if request.business_context else []) +
                            list(request.communication_preferences.keys() if request.communication_preferences else []) +
                            (['notes'] if request.notes is not None else []) +
                            (['tags'] if request.tags is not None else [])
                        )),
                        'updated_at': contact.updated_at.isoformat(),
                        'lead_score_recalculated': True
                    },
                    'metadata': {
                        'updated_by': current_user['user_id'],
                        'update_timestamp': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating contact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update contact: {str(e)}")

@router.get("/contacts/{contact_id}/conversations", summary="Get contact's conversation history")
async def get_contact_conversations(
    contact_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum conversations to return"),
    include_quality_scores: bool = Query(True, description="Include quality scores"),
    include_ai_insights: bool = Query(False, description="Include AI insights"),
    days_back: int = Query(90, ge=1, le=365, description="Days to look back"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get conversation history for a specific contact.
    
    Provides detailed conversation analytics, quality scores,
    and AI insights for contact relationship analysis.
    """
    try:
        async with get_session() as session:
            # Get contact
            if contact_id.startswith('user_'):
                user_id = contact_id
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
            else:
                contact = session.query(Contact).filter(Contact.id == contact_id).first()
            
            if not contact:
                raise HTTPException(status_code=404, detail="Contact not found")
            
            # Get conversations
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            from app.models.kelly_analytics import ConversationAnalytics
            conversations = session.query(ConversationAnalytics).filter(
                and_(
                    ConversationAnalytics.user_id == contact.user_id,
                    ConversationAnalytics.started_at >= cutoff_date
                )
            ).order_by(ConversationAnalytics.started_at.desc()).limit(limit).all()
            
            # Format conversations
            formatted_conversations = []
            for conv in conversations:
                conv_data = {
                    'conversation_id': conv.conversation_id,
                    'started_at': conv.started_at.isoformat(),
                    'ended_at': conv.ended_at.isoformat() if conv.ended_at else None,
                    'duration_minutes': conv.duration_minutes,
                    'total_messages': conv.total_messages,
                    'ai_messages': conv.ai_messages,
                    'human_messages': conv.human_messages,
                    'human_intervention_count': conv.human_intervention_count,
                    'resolution_achieved': conv.resolution_achieved,
                    'lead_qualified': conv.lead_qualified,
                    'conversion_event': conv.conversion_event
                }
                
                # Include quality scores if requested
                if include_quality_scores:
                    from app.models.kelly_analytics import QualityScore
                    quality_score = session.query(QualityScore).filter(
                        QualityScore.conversation_id == conv.conversation_id
                    ).order_by(QualityScore.scored_at.desc()).first()
                    
                    if quality_score:
                        conv_data['quality_score'] = {
                            'overall_score': quality_score.overall_score,
                            'grade': quality_score.overall_grade,
                            'component_scores': {
                                'relevance': quality_score.relevance_score,
                                'helpfulness': quality_score.helpfulness_score,
                                'accuracy': quality_score.accuracy_score,
                                'empathy': quality_score.empathy_score,
                                'professionalism': quality_score.professionalism_score,
                                'efficiency': quality_score.efficiency_score
                            }
                        }
                
                # Include AI insights if requested
                if include_ai_insights:
                    from app.models.kelly_intelligence import ConversationInsight
                    insights = session.query(ConversationInsight).filter(
                        ConversationInsight.conversation_id == conv.conversation_id
                    ).order_by(ConversationInsight.generated_at.desc()).limit(3).all()
                    
                    conv_data['ai_insights'] = [
                        {
                            'type': insight.insight_type,
                            'title': insight.title,
                            'description': insight.description,
                            'priority': insight.priority,
                            'confidence_score': insight.confidence_score
                        }
                        for insight in insights
                    ]
                
                formatted_conversations.append(conv_data)
            
            # Calculate summary statistics
            total_conversations = len(conversations)
            avg_quality = sum(c.conversation_quality_score for c in conversations) / total_conversations if conversations else 0
            resolution_rate = len([c for c in conversations if c.resolution_achieved]) / total_conversations if conversations else 0
            ai_success_rate = len([c for c in conversations if c.resolution_achieved and c.human_intervention_count == 0]) / total_conversations if conversations else 0
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'conversations': formatted_conversations,
                        'summary': {
                            'total_conversations': total_conversations,
                            'avg_quality_score': round(avg_quality, 3),
                            'resolution_rate': round(resolution_rate, 3),
                            'ai_success_rate': round(ai_success_rate, 3),
                            'date_range': {
                                'start': cutoff_date.isoformat(),
                                'end': datetime.utcnow().isoformat()
                            }
                        }
                    },
                    'metadata': {
                        'contact_id': str(contact.id),
                        'user_id': contact.user_id,
                        'limit': limit,
                        'days_back': days_back,
                        'includes': {
                            'quality_scores': include_quality_scores,
                            'ai_insights': include_ai_insights
                        },
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contact conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@router.get("/lead-scoring", summary="Get lead scoring data")
async def get_lead_scoring_data(
    user_id: Optional[str] = Query(None, description="Specific user filter"),
    qualification_status: Optional[str] = Query(None, description="Qualification status filter"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum lead score"),
    max_score: Optional[float] = Query(None, ge=0, le=100, description="Maximum lead score"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    sort_by: str = Query("overall_score", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get lead scoring data with filtering and sorting.
    
    Provides comprehensive lead scoring information with AI-powered
    qualification status and conversion probability predictions.
    """
    try:
        async with get_session() as session:
            # Build query
            query_builder = session.query(LeadScore).join(Contact)
            
            # Apply filters
            if user_id:
                query_builder = query_builder.filter(Contact.user_id == user_id)
            
            if qualification_status:
                query_builder = query_builder.filter(LeadScore.qualification_status == qualification_status)
            
            if min_score is not None:
                query_builder = query_builder.filter(LeadScore.overall_score >= min_score)
            
            if max_score is not None:
                query_builder = query_builder.filter(LeadScore.overall_score <= max_score)
            
            # Apply sorting
            if sort_by == "overall_score":
                sort_field = LeadScore.overall_score
            elif sort_by == "conversion_probability":
                sort_field = LeadScore.conversion_probability
            elif sort_by == "scored_at":
                sort_field = LeadScore.scored_at
            else:
                sort_field = LeadScore.overall_score
            
            if sort_order == "desc":
                query_builder = query_builder.order_by(sort_field.desc())
            else:
                query_builder = query_builder.order_by(sort_field.asc())
            
            # Get lead scores
            lead_scores = query_builder.limit(limit).all()
            
            # Format lead scores
            formatted_scores = []
            for score in lead_scores:
                contact = session.query(Contact).filter(Contact.id == score.contact_id).first()
                
                score_data = {
                    'id': str(score.id),
                    'contact_info': {
                        'contact_id': str(contact.id),
                        'user_id': contact.user_id,
                        'name': f"{contact.first_name or ''} {contact.last_name or ''}".strip(),
                        'company': contact.company,
                        'email': contact.email
                    },
                    'scoring': {
                        'overall_score': score.overall_score,
                        'grade': score.score_grade,
                        'qualification_status': score.qualification_status,
                        'conversion_probability': score.conversion_probability,
                        'component_scores': {
                            'demographic': score.demographic_score,
                            'behavioral': score.behavioral_score,
                            'engagement': score.engagement_score,
                            'fit': score.fit_score,
                            'intent': score.intent_score,
                            'timing': score.timing_score
                        },
                        'bant_scores': {
                            'budget': score.budget_score,
                            'authority': score.authority_score,
                            'need': score.need_score,
                            'timeline': score.timeline_score
                        }
                    },
                    'metadata': {
                        'scored_at': score.scored_at.isoformat(),
                        'scoring_method': score.scoring_method,
                        'scoring_confidence': score.scoring_confidence
                    }
                }
                
                formatted_scores.append(score_data)
            
            # Calculate summary statistics
            total_count = len(formatted_scores)
            if total_count > 0:
                avg_score = sum(s['scoring']['overall_score'] for s in formatted_scores) / total_count
                qualified_count = len([s for s in formatted_scores if s['scoring']['qualification_status'] == 'qualified'])
                high_probability_count = len([s for s in formatted_scores if s['scoring']['conversion_probability'] and s['scoring']['conversion_probability'] > 0.7])
            else:
                avg_score = 0
                qualified_count = 0
                high_probability_count = 0
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'lead_scores': formatted_scores,
                        'summary': {
                            'total_count': total_count,
                            'avg_score': round(avg_score, 2),
                            'qualified_count': qualified_count,
                            'high_probability_count': high_probability_count,
                            'qualification_rate': round(qualified_count / total_count, 3) if total_count > 0 else 0
                        }
                    },
                    'metadata': {
                        'filters': {
                            'user_id': user_id,
                            'qualification_status': qualification_status,
                            'min_score': min_score,
                            'max_score': max_score
                        },
                        'sort_by': sort_by,
                        'sort_order': sort_order,
                        'limit': limit,
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"Error getting lead scoring data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get lead scoring data: {str(e)}")

@router.post("/contacts/{contact_id}/enrich", summary="Enrich contact profile from conversations")
async def enrich_contact_profile(
    contact_id: str,
    request: ContactEnrichmentRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Enrich contact profile using conversation data and AI analysis.
    
    Analyzes conversation patterns to extract personality traits,
    business context, communication preferences, and relationship insights.
    """
    try:
        # Validate contact exists
        async with get_session() as session:
            if contact_id.startswith('user_'):
                user_id = contact_id
                contact = session.query(Contact).filter(Contact.user_id == user_id).first()
            else:
                contact = session.query(Contact).filter(Contact.id == contact_id).first()
            
            if not contact:
                raise HTTPException(status_code=404, detail="Contact not found")
            
            # Validate user_id matches
            if request.user_id != contact.user_id:
                raise HTTPException(status_code=400, detail="User ID mismatch")
        
        # Perform enrichment
        enrichment_result = await kelly_crm_service.enrich_contact_from_conversations(
            user_id=request.user_id,
            conversation_data=request.conversation_data
        )
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': enrichment_result,
                'metadata': {
                    'contact_id': str(contact.id),
                    'user_id': request.user_id,
                    'conversations_analyzed': len(request.conversation_data),
                    'override_existing': request.override_existing,
                    'confidence_threshold': request.confidence_threshold,
                    'enriched_at': datetime.utcnow().isoformat(),
                    'enriched_by': current_user['user_id']
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enriching contact profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enrich contact: {str(e)}")

@router.post("/lead-scoring/{user_id}", summary="Calculate lead score")
async def calculate_lead_score(
    user_id: str,
    force_recalculate: bool = Query(False, description="Force recalculation"),
    include_breakdown: bool = Query(True, description="Include component breakdown"),
    include_recommendations: bool = Query(True, description="Include recommendations"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate AI-powered lead score for a contact.
    
    Performs comprehensive lead scoring analysis including BANT criteria,
    behavioral patterns, engagement metrics, and conversion probability.
    """
    try:
        # Calculate lead score
        scoring_result = await kelly_crm_service.calculate_lead_score(
            user_id=user_id,
            force_recalculate=force_recalculate
        )
        
        response_data = {
            'user_id': user_id,
            'overall_score': scoring_result.overall_score,
            'grade': scoring_result.grade,
            'qualification_status': scoring_result.qualification_status,
            'conversion_probability': scoring_result.conversion_probability,
            'calculated_at': datetime.utcnow().isoformat()
        }
        
        if include_breakdown:
            response_data['component_scores'] = scoring_result.component_scores
        
        if include_recommendations:
            response_data['recommendations'] = scoring_result.recommendations
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': response_data,
                'metadata': {
                    'force_recalculate': force_recalculate,
                    'scoring_method': 'ai_model',
                    'includes': {
                        'breakdown': include_breakdown,
                        'recommendations': include_recommendations
                    }
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating lead score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate lead score: {str(e)}")

@router.get("/deals", summary="Get deals and opportunities")
async def get_deals(
    contact_id: Optional[str] = Query(None, description="Contact filter"),
    stage: Optional[str] = Query(None, description="Deal stage filter"),
    status: Optional[str] = Query(None, description="Deal status filter"),
    owner_id: Optional[str] = Query(None, description="Deal owner filter"),
    min_value: Optional[float] = Query(None, ge=0, description="Minimum deal value"),
    max_value: Optional[float] = Query(None, ge=0, description="Maximum deal value"),
    ai_assisted: Optional[bool] = Query(None, description="AI-assisted deals filter"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get deals and opportunities with filtering and pagination.
    
    Provides comprehensive deal pipeline view with AI contribution
    tracking and revenue attribution analysis.
    """
    try:
        async with get_session() as session:
            # Build query
            query_builder = session.query(Deal).join(Contact)
            
            # Apply filters
            if contact_id:
                if contact_id.startswith('user_'):
                    query_builder = query_builder.filter(Contact.user_id == contact_id)
                else:
                    query_builder = query_builder.filter(Deal.contact_id == contact_id)
            
            if stage:
                query_builder = query_builder.filter(Deal.stage == stage)
            
            if status:
                query_builder = query_builder.filter(Deal.status == status)
            else:
                # Default to open deals
                query_builder = query_builder.filter(Deal.status == 'open')
            
            if owner_id:
                query_builder = query_builder.filter(Deal.owner_id == owner_id)
            
            if min_value is not None:
                query_builder = query_builder.filter(Deal.deal_value >= min_value)
            
            if max_value is not None:
                query_builder = query_builder.filter(Deal.deal_value <= max_value)
            
            if ai_assisted is not None:
                query_builder = query_builder.filter(Deal.ai_assisted == ai_assisted)
            
            # Apply sorting
            if sort_by == "deal_value":
                sort_field = Deal.deal_value
            elif sort_by == "probability":
                sort_field = Deal.probability
            elif sort_by == "expected_close_date":
                sort_field = Deal.expected_close_date
            elif sort_by == "created_at":
                sort_field = Deal.created_at
            else:
                sort_field = Deal.created_at
            
            if sort_order == "desc":
                query_builder = query_builder.order_by(sort_field.desc())
            else:
                query_builder = query_builder.order_by(sort_field.asc())
            
            # Get total count
            total_count = query_builder.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            deals = query_builder.offset(offset).limit(page_size).all()
            
            # Format deals
            formatted_deals = []
            for deal in deals:
                contact = session.query(Contact).filter(Contact.id == deal.contact_id).first()
                
                deal_data = {
                    'id': str(deal.id),
                    'name': deal.deal_name,
                    'contact_info': {
                        'contact_id': str(contact.id),
                        'user_id': contact.user_id,
                        'name': f"{contact.first_name or ''} {contact.last_name or ''}".strip(),
                        'company': contact.company,
                        'email': contact.email
                    },
                    'deal_details': {
                        'value': deal.deal_value,
                        'currency': deal.currency,
                        'stage': deal.stage,
                        'probability': deal.probability,
                        'status': deal.status,
                        'expected_close_date': deal.expected_close_date.isoformat(),
                        'owner_id': deal.owner_id
                    },
                    'ai_metrics': {
                        'ai_assisted': deal.ai_assisted,
                        'ai_contribution_score': deal.ai_contribution_score,
                        'source_conversation_id': deal.source_conversation_id
                    },
                    'timeline': {
                        'created_at': deal.created_at.isoformat(),
                        'updated_at': deal.updated_at.isoformat(),
                        'closed_at': deal.closed_at.isoformat() if deal.closed_at else None
                    }
                }
                
                formatted_deals.append(deal_data)
            
            # Calculate pipeline summary
            pipeline_value = sum(d['deal_details']['value'] * d['deal_details']['probability'] for d in formatted_deals)
            ai_assisted_deals = len([d for d in formatted_deals if d['ai_metrics']['ai_assisted']])
            avg_deal_size = sum(d['deal_details']['value'] for d in formatted_deals) / len(formatted_deals) if formatted_deals else 0
            
            # Calculate pagination info
            total_pages = (total_count + page_size - 1) // page_size
            has_next = page < total_pages
            has_prev = page > 1
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'deals': formatted_deals,
                        'pipeline_summary': {
                            'total_deals': len(formatted_deals),
                            'pipeline_value': round(pipeline_value, 2),
                            'avg_deal_size': round(avg_deal_size, 2),
                            'ai_assisted_count': ai_assisted_deals,
                            'ai_assisted_percentage': round(ai_assisted_deals / len(formatted_deals) * 100, 1) if formatted_deals else 0
                        },
                        'pagination': {
                            'current_page': page,
                            'page_size': page_size,
                            'total_count': total_count,
                            'total_pages': total_pages,
                            'has_next': has_next,
                            'has_prev': has_prev
                        }
                    },
                    'metadata': {
                        'filters': {
                            'contact_id': contact_id,
                            'stage': stage,
                            'status': status,
                            'owner_id': owner_id,
                            'value_range': [min_value, max_value],
                            'ai_assisted': ai_assisted
                        },
                        'sort_by': sort_by,
                        'sort_order': sort_order,
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"Error getting deals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get deals: {str(e)}")

@router.post("/deals", summary="Create new deal")
async def create_deal(
    request: DealRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a new deal in the sales pipeline.
    
    Creates deal with automatic AI contribution tracking and
    conversation attribution for revenue analysis.
    """
    try:
        # Validate expected_close_date format
        try:
            expected_close_date = datetime.fromisoformat(request.expected_close_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expected_close_date format. Use ISO format.")
        
        # Prepare deal data
        deal_data = {
            'name': request.deal_name,
            'user_id': request.user_id,
            'account_id': request.account_id,
            'value': request.deal_value,
            'currency': request.currency,
            'expected_close_date': request.expected_close_date,
            'stage': request.stage,
            'probability': request.probability,
            'source': request.source,
            'conversation_id': request.conversation_id,
            'owner_id': request.owner_id
        }
        
        # Create deal through CRM service
        deal_result = await kelly_crm_service.manage_deal_pipeline(
            deal_data=deal_data,
            action='create'
        )
        
        return JSONResponse(
            status_code=201,
            content={
                'success': True,
                'data': deal_result,
                'metadata': {
                    'created_by': current_user['user_id'],
                    'created_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating deal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create deal: {str(e)}")

@router.put("/deals/{deal_id}", summary="Update deal")
async def update_deal(
    deal_id: str,
    request: DealUpdateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update an existing deal in the pipeline.
    
    Supports stage progression, value updates, and
    probability adjustments with audit trail.
    """
    try:
        # Validate deal_id matches request
        if request.deal_id != deal_id:
            raise HTTPException(status_code=400, detail="Deal ID mismatch")
        
        # Prepare update data
        update_data = {
            'deal_id': deal_id,
            'updates': request.updates
        }
        
        # Update deal through CRM service
        update_result = await kelly_crm_service.manage_deal_pipeline(
            deal_data=update_data,
            action='update'
        )
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': update_result,
                'metadata': {
                    'updated_by': current_user['user_id'],
                    'updated_at': datetime.utcnow().isoformat(),
                    'notes': request.notes,
                    'stage_change_reason': request.stage_change_reason
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating deal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update deal: {str(e)}")

@router.get("/deals/{deal_id}/conversations", summary="Get deal-related conversations")
async def get_deal_conversations(
    deal_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get conversations linked to a specific deal.
    
    Provides conversation history that contributed to deal progression
    with quality metrics and AI contribution analysis.
    """
    try:
        async with get_session() as session:
            # Get deal
            deal = session.query(Deal).filter(Deal.id == deal_id).first()
            if not deal:
                raise HTTPException(status_code=404, detail="Deal not found")
            
            # Get conversations linked to this deal's contact
            from app.models.kelly_analytics import ConversationAnalytics
            contact = session.query(Contact).filter(Contact.id == deal.contact_id).first()
            
            conversations = session.query(ConversationAnalytics).filter(
                ConversationAnalytics.user_id == contact.user_id
            ).order_by(ConversationAnalytics.started_at.desc()).all()
            
            # Get touchpoints for this deal
            touchpoints = session.query(TouchPoint).filter(
                TouchPoint.deal_id == deal.id
            ).order_by(TouchPoint.occurred_at.desc()).all()
            
            # Format data
            conversation_data = []
            for conv in conversations:
                conversation_data.append({
                    'conversation_id': conv.conversation_id,
                    'started_at': conv.started_at.isoformat(),
                    'duration_minutes': conv.duration_minutes,
                    'quality_score': conv.conversation_quality_score,
                    'ai_confidence': conv.ai_confidence_avg,
                    'resolution_achieved': conv.resolution_achieved,
                    'lead_qualified': conv.lead_qualified,
                    'conversion_event': conv.conversion_event,
                    'revenue_attributed': conv.revenue_attributed
                })
            
            touchpoint_data = []
            for tp in touchpoints:
                touchpoint_data.append({
                    'id': str(tp.id),
                    'type': tp.touchpoint_type,
                    'channel': tp.channel,
                    'direction': tp.direction,
                    'subject': tp.subject,
                    'sentiment': tp.sentiment,
                    'engagement_quality': tp.engagement_quality,
                    'occurred_at': tp.occurred_at.isoformat()
                })
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'deal_info': {
                            'id': str(deal.id),
                            'name': deal.deal_name,
                            'value': deal.deal_value,
                            'stage': deal.stage,
                            'ai_contribution_score': deal.ai_contribution_score
                        },
                        'conversations': conversation_data,
                        'touchpoints': touchpoint_data,
                        'summary': {
                            'total_conversations': len(conversation_data),
                            'total_touchpoints': len(touchpoint_data),
                            'avg_quality_score': sum(c['quality_score'] for c in conversation_data) / len(conversation_data) if conversation_data else 0,
                            'conversion_events': len([c for c in conversation_data if c['conversion_event']])
                        }
                    },
                    'metadata': {
                        'deal_id': deal_id,
                        'contact_id': str(deal.contact_id),
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deal conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get deal conversations: {str(e)}")

@router.get("/conversion-funnel", summary="Get conversion funnel analytics")
async def get_conversion_funnel(
    account_id: Optional[str] = Query(None, description="Account filter"),
    time_period_days: int = Query(30, ge=1, le=365, description="Analysis period"),
    include_ai_metrics: bool = Query(True, description="Include AI performance metrics"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get conversion funnel analytics with stage progression metrics.
    
    Analyzes customer journey progression through sales stages
    with AI contribution tracking and conversion optimization insights.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
        
        async with get_session() as session:
            # Get customer journeys for analysis
            journeys_query = session.query(CustomerJourney).filter(
                CustomerJourney.journey_started_at >= cutoff_date
            )
            
            if account_id:
                # Filter by account through contact relationship
                journeys_query = journeys_query.join(Contact).filter(
                    Contact.user_id.like(f"{account_id}%")  # Simplified account filtering
                )
            
            journeys = journeys_query.all()
            
            # Calculate funnel metrics
            funnel_stages = ['awareness', 'consideration', 'decision', 'retention']
            funnel_data = {}
            
            for stage in funnel_stages:
                stage_count = len([j for j in journeys if j.current_stage == stage or 
                                 (stage == 'awareness' and j.awareness_entered_at) or
                                 (stage == 'consideration' and j.consideration_entered_at) or
                                 (stage == 'decision' and j.decision_entered_at) or
                                 (stage == 'retention' and j.retention_entered_at)])
                
                funnel_data[stage] = {
                    'count': stage_count,
                    'percentage': round(stage_count / len(journeys) * 100, 1) if journeys else 0
                }
            
            # Calculate conversion rates between stages
            conversion_rates = {}
            for i, stage in enumerate(funnel_stages[:-1]):
                current_stage_count = funnel_data[stage]['count']
                next_stage_count = funnel_data[funnel_stages[i + 1]]['count']
                
                conversion_rate = (next_stage_count / current_stage_count * 100) if current_stage_count > 0 else 0
                conversion_rates[f"{stage}_to_{funnel_stages[i + 1]}"] = round(conversion_rate, 1)
            
            # Get deals data for revenue analysis
            deals_query = session.query(Deal).filter(
                Deal.created_at >= cutoff_date
            )
            
            if account_id:
                deals_query = deals_query.filter(Deal.account_id == account_id)
            
            deals = deals_query.all()
            
            # Calculate deal conversion metrics
            total_deals = len(deals)
            won_deals = len([d for d in deals if d.status == 'won'])
            lost_deals = len([d for d in deals if d.status == 'lost'])
            open_deals = len([d for d in deals if d.status == 'open'])
            
            deal_conversion_rate = (won_deals / total_deals * 100) if total_deals > 0 else 0
            total_revenue = sum(d.actual_deal_value for d in deals if d.status == 'won' and d.actual_deal_value)
            
            # AI metrics if requested
            ai_metrics = {}
            if include_ai_metrics:
                ai_assisted_deals = len([d for d in deals if d.ai_assisted])
                ai_won_deals = len([d for d in deals if d.ai_assisted and d.status == 'won'])
                
                ai_metrics = {
                    'ai_assisted_deals': ai_assisted_deals,
                    'ai_assistance_rate': round(ai_assisted_deals / total_deals * 100, 1) if total_deals > 0 else 0,
                    'ai_conversion_rate': round(ai_won_deals / ai_assisted_deals * 100, 1) if ai_assisted_deals > 0 else 0,
                    'ai_revenue_contribution': sum(d.actual_deal_value * (d.ai_contribution_score or 0) 
                                                 for d in deals if d.status == 'won' and d.actual_deal_value)
                }
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'funnel_analysis': {
                            'stages': funnel_data,
                            'conversion_rates': conversion_rates,
                            'total_journeys': len(journeys)
                        },
                        'deal_analysis': {
                            'total_deals': total_deals,
                            'won_deals': won_deals,
                            'lost_deals': lost_deals,
                            'open_deals': open_deals,
                            'deal_conversion_rate': round(deal_conversion_rate, 1),
                            'total_revenue': round(total_revenue, 2)
                        },
                        'ai_metrics': ai_metrics,
                        'time_period': {
                            'start_date': cutoff_date.isoformat(),
                            'end_date': datetime.utcnow().isoformat(),
                            'days': time_period_days
                        }
                    },
                    'metadata': {
                        'account_id': account_id,
                        'includes_ai_metrics': include_ai_metrics,
                        'analysis_timestamp': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"Error getting conversion funnel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversion funnel: {str(e)}")

@router.get("/user/{user_id}/journey", summary="Get customer journey")
async def get_customer_journey(
    user_id: str,
    include_touchpoints: bool = Query(True, description="Include touchpoint details"),
    include_predictions: bool = Query(True, description="Include AI predictions"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get complete customer journey analysis for a user.
    
    Provides comprehensive journey mapping with stage progression,
    touchpoint analysis, and AI-powered next action recommendations.
    """
    try:
        async with get_session() as session:
            # Get contact
            contact = session.query(Contact).filter(Contact.user_id == user_id).first()
            if not contact:
                raise HTTPException(status_code=404, detail="Contact not found")
            
            # Get customer journey
            journey = session.query(CustomerJourney).filter(
                CustomerJourney.contact_id == contact.id
            ).first()
            
            if not journey:
                raise HTTPException(status_code=404, detail="Customer journey not found")
            
            # Build journey data
            journey_data = {
                'journey_id': str(journey.id),
                'contact_info': {
                    'contact_id': str(contact.id),
                    'user_id': contact.user_id,
                    'name': f"{contact.first_name or ''} {contact.last_name or ''}".strip(),
                    'company': contact.company
                },
                'journey_overview': {
                    'status': journey.journey_status,
                    'current_stage': journey.current_stage,
                    'previous_stage': journey.previous_stage,
                    'stage_entered_at': journey.stage_entered_at.isoformat(),
                    'journey_started_at': journey.journey_started_at.isoformat(),
                    'total_touchpoints': journey.total_touchpoints,
                    'engagement_trend': journey.engagement_trend
                },
                'stage_progression': {
                    'awareness': {
                        'entered_at': journey.awareness_entered_at.isoformat() if journey.awareness_entered_at else None,
                        'duration_hours': journey.awareness_duration_hours
                    },
                    'consideration': {
                        'entered_at': journey.consideration_entered_at.isoformat() if journey.consideration_entered_at else None,
                        'duration_hours': journey.consideration_duration_hours
                    },
                    'decision': {
                        'entered_at': journey.decision_entered_at.isoformat() if journey.decision_entered_at else None,
                        'duration_hours': journey.decision_duration_hours
                    },
                    'retention': {
                        'entered_at': journey.retention_entered_at.isoformat() if journey.retention_entered_at else None,
                        'duration_hours': journey.retention_duration_hours
                    }
                }
            }
            
            # Include touchpoints if requested
            if include_touchpoints:
                touchpoints = session.query(TouchPoint).filter(
                    TouchPoint.contact_id == contact.id
                ).order_by(TouchPoint.occurred_at.desc()).limit(20).all()
                
                journey_data['touchpoints'] = [
                    {
                        'id': str(tp.id),
                        'type': tp.touchpoint_type,
                        'channel': tp.channel,
                        'direction': tp.direction,
                        'subject': tp.subject,
                        'summary': tp.summary,
                        'sentiment': tp.sentiment,
                        'engagement_quality': tp.engagement_quality,
                        'journey_stage': tp.journey_stage,
                        'ai_handled': tp.ai_handled,
                        'occurred_at': tp.occurred_at.isoformat()
                    }
                    for tp in touchpoints
                ]
            
            # Include AI predictions if requested
            if include_predictions:
                journey_data['ai_insights'] = {
                    'next_best_actions': journey.next_best_actions or [],
                    'risk_indicators': journey.risk_indicators or [],
                    'success_indicators': journey.success_indicators or [],
                    'ai_recommendations': journey.ai_recommendations or {}
                }
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': journey_data,
                    'metadata': {
                        'user_id': user_id,
                        'includes': {
                            'touchpoints': include_touchpoints,
                            'predictions': include_predictions
                        },
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer journey: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get customer journey: {str(e)}")

@router.post("/user/{user_id}/touchpoint", summary="Track customer touchpoint")
async def track_customer_touchpoint(
    user_id: str,
    request: TouchPointRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Track a new customer touchpoint and update journey.
    
    Records customer interaction and automatically updates
    journey stage progression with AI analysis.
    """
    try:
        # Validate user_id matches request
        if request.user_id != user_id:
            raise HTTPException(status_code=400, detail="User ID mismatch")
        
        # Prepare touchpoint data
        touchpoint_data = {
            'type': request.touchpoint_type,
            'channel': request.channel,
            'direction': request.direction,
            'subject': request.subject,
            'summary': request.summary,
            'sentiment': request.sentiment,
            'sentiment_score': request.sentiment_score,
            'engagement_quality': request.engagement_quality,
            'ai_handled': request.ai_handled,
            'ai_confidence': request.ai_confidence,
            'conversation_id': request.conversation_id,
            'timestamp': request.timestamp or datetime.utcnow().isoformat()
        }
        
        # Track touchpoint through CRM service
        journey_analysis = await kelly_crm_service.track_customer_journey(
            user_id=user_id,
            touchpoint_data=touchpoint_data
        )
        
        return JSONResponse(
            status_code=201,
            content={
                'success': True,
                'data': {
                    'journey_analysis': journey_analysis.__dict__,
                    'touchpoint_recorded': True
                },
                'metadata': {
                    'user_id': user_id,
                    'touchpoint_type': request.touchpoint_type,
                    'recorded_by': current_user['user_id'],
                    'recorded_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking touchpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track touchpoint: {str(e)}")

# Helper functions

async def _has_account_access(user: Dict, account_id: str) -> bool:
    """Check if user has access to specific account"""
    # This would implement actual access control logic
    return True

async def _validate_contact_access(user: Dict, contact_id: str) -> bool:
    """Validate user access to specific contact"""
    # This would implement contact-level access control
    return True