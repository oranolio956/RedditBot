"""
Kelly Intelligence API Endpoints

FastAPI endpoints for AI-powered conversation intelligence, pattern recognition,
sentiment analysis, recommendations, and anomaly detection.
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

from app.database.connection import get_session
from app.core.auth import get_current_user
from app.core.redis import redis_manager
from app.services.kelly_intelligence_service import kelly_intelligence_service
from app.models.kelly_intelligence import (
    ConversationInsight, ConversationPattern, AiRecommendation,
    AnomalyDetection, TopicAnalysis
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation

class ConversationAnalysisRequest(BaseModel):
    conversation_id: str
    conversation_data: Dict[str, Any]
    include_recommendations: bool = True
    include_sentiment: bool = True
    include_patterns: bool = True
    include_topics: bool = True
    include_anomalies: bool = True

class SentimentAnalysisRequest(BaseModel):
    conversation_data: Dict[str, Any]
    detailed_emotions: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class PatternDiscoveryRequest(BaseModel):
    conversation_data: Dict[str, Any]
    pattern_types: List[str] = ['success', 'failure', 'escalation']
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)

class RecommendationRequest(BaseModel):
    conversation_id: str
    conversation_data: Dict[str, Any]
    focus_areas: List[str] = ['quality', 'efficiency', 'satisfaction']
    priority_filter: Optional[str] = Field(None, regex="^(low|medium|high|critical)$")

class AnomalyDetectionRequest(BaseModel):
    conversation_id: str
    conversation_data: Dict[str, Any]
    detection_sensitivity: float = Field(default=2.0, ge=1.0, le=5.0)
    include_explanations: bool = True

class TopicAnalysisRequest(BaseModel):
    conversation_id: str
    conversation_data: Dict[str, Any]
    extract_entities: bool = True
    business_context: bool = True

@router.post("/conversation/{conversation_id}/analysis", summary="Comprehensive conversation intelligence analysis")
async def analyze_conversation_intelligence(
    conversation_id: str,
    request: ConversationAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Perform comprehensive AI-powered conversation intelligence analysis.
    
    Provides sentiment analysis, pattern recognition, topic extraction,
    anomaly detection, and intelligent recommendations.
    """
    try:
        # Validate conversation data
        if not request.conversation_data.get('messages'):
            raise HTTPException(status_code=400, detail="Conversation data must include messages")
        
        # Validate conversation ID matches
        if request.conversation_id != conversation_id:
            raise HTTPException(status_code=400, detail="Conversation ID mismatch")
        
        # Check if user has access to this conversation
        account_id = request.conversation_data.get('account_id')
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to conversation")
        
        # Perform comprehensive analysis
        analysis_result = await kelly_intelligence_service.analyze_conversation_intelligence(
            conversation_id=conversation_id,
            conversation_data=request.conversation_data,
            include_recommendations=request.include_recommendations
        )
        
        # Filter results based on request preferences
        filtered_result = {
            'conversation_id': conversation_id,
            'analysis_timestamp': analysis_result['analysis_timestamp'],
            'confidence_score': analysis_result['confidence_score']
        }
        
        if request.include_sentiment and analysis_result.get('sentiment_analysis'):
            filtered_result['sentiment_analysis'] = analysis_result['sentiment_analysis']
        
        if request.include_patterns and analysis_result.get('pattern_matches'):
            filtered_result['pattern_matches'] = analysis_result['pattern_matches']
        
        if request.include_topics and analysis_result.get('topic_analysis'):
            filtered_result['topic_analysis'] = analysis_result['topic_analysis']
        
        if request.include_anomalies and analysis_result.get('anomalies'):
            filtered_result['anomalies'] = analysis_result['anomalies']
        
        if request.include_recommendations and analysis_result.get('recommendations'):
            filtered_result['recommendations'] = analysis_result['recommendations']
        
        filtered_result['insights'] = analysis_result.get('insights', [])
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': filtered_result,
                'metadata': {
                    'conversation_id': conversation_id,
                    'analysis_components': {
                        'sentiment': request.include_sentiment,
                        'patterns': request.include_patterns,
                        'topics': request.include_topics,
                        'anomalies': request.include_anomalies,
                        'recommendations': request.include_recommendations
                    },
                    'processed_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing conversation intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze conversation: {str(e)}")

@router.get("/patterns", summary="Get successful conversation patterns")
async def get_successful_patterns(
    account_id: Optional[str] = Query(None, description="Account ID filter"),
    pattern_type: Optional[str] = Query(None, description="Pattern type filter"),
    min_success_rate: float = Query(0.7, ge=0.0, le=1.0, description="Minimum success rate"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of patterns"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get successful conversation patterns for learning and replication.
    
    Returns patterns that have proven successful in similar conversations,
    with performance metrics and applicability guidance.
    """
    try:
        # Validate access
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        # Get successful patterns
        patterns = await kelly_intelligence_service.get_successful_patterns(
            account_id=account_id,
            pattern_type=pattern_type,
            min_success_rate=min_success_rate
        )
        
        # Limit results
        limited_patterns = patterns[:limit]
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'patterns': limited_patterns,
                    'total_found': len(patterns),
                    'returned_count': len(limited_patterns)
                },
                'metadata': {
                    'filters': {
                        'account_id': account_id,
                        'pattern_type': pattern_type,
                        'min_success_rate': min_success_rate
                    },
                    'retrieved_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting successful patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")

@router.post("/sentiment/{conversation_id}", summary="Analyze conversation sentiment")
async def analyze_sentiment(
    conversation_id: str,
    request: SentimentAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Perform detailed sentiment analysis on conversation content.
    
    Analyzes overall sentiment, emotional indicators, and provides
    confidence scores for sentiment classification.
    """
    try:
        # Validate conversation data
        if not request.conversation_data.get('messages'):
            raise HTTPException(status_code=400, detail="Conversation data must include messages")
        
        # Perform sentiment analysis
        sentiment_result = await kelly_intelligence_service.analyze_sentiment(
            conversation_data=request.conversation_data
        )
        
        # Filter by confidence threshold
        if sentiment_result.confidence < request.confidence_threshold:
            filtered_emotions = {
                emotion: score for emotion, score in sentiment_result.emotions.items()
                if score >= request.confidence_threshold
            }
        else:
            filtered_emotions = sentiment_result.emotions
        
        response_data = {
            'conversation_id': conversation_id,
            'sentiment': sentiment_result.sentiment,
            'confidence': sentiment_result.confidence,
            'score': sentiment_result.score,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        if request.detailed_emotions:
            response_data['emotions'] = filtered_emotions
            response_data['primary_emotion'] = max(filtered_emotions.items(), key=lambda x: x[1])[0] if filtered_emotions else None
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': response_data,
                'metadata': {
                    'confidence_threshold': request.confidence_threshold,
                    'detailed_emotions': request.detailed_emotions,
                    'emotions_detected': len(filtered_emotions) if request.detailed_emotions else 0
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")

@router.post("/recommendations", summary="Generate AI coaching recommendations")
async def generate_recommendations(
    request: RecommendationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate AI-powered coaching recommendations for conversation improvement.
    
    Provides specific, actionable suggestions for improving conversation
    quality, efficiency, and customer satisfaction.
    """
    try:
        # Validate conversation data
        if not request.conversation_data.get('messages'):
            raise HTTPException(status_code=400, detail="Conversation data must include messages")
        
        # Generate recommendations
        recommendations_result = await kelly_intelligence_service.generate_recommendations(
            conversation_id=request.conversation_id,
            conversation_data=request.conversation_data
        )
        
        # Filter by focus areas and priority
        filtered_suggestions = []
        for suggestion in recommendations_result.coaching_suggestions:
            # Filter by focus areas
            if not request.focus_areas or any(area in suggestion.get('category', '') for area in request.focus_areas):
                # Filter by priority
                if not request.priority_filter or suggestion.get('priority') == request.priority_filter:
                    filtered_suggestions.append(suggestion)
        
        response_data = {
            'conversation_id': request.conversation_id,
            'coaching_suggestions': filtered_suggestions,
            'pattern_recommendations': [p.__dict__ for p in recommendations_result.pattern_recommendations],
            'improvement_areas': recommendations_result.improvement_areas,
            'success_indicators': recommendations_result.success_indicators,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': response_data,
                'metadata': {
                    'focus_areas': request.focus_areas,
                    'priority_filter': request.priority_filter,
                    'total_suggestions': len(recommendations_result.coaching_suggestions),
                    'filtered_suggestions': len(filtered_suggestions),
                    'pattern_matches': len(recommendations_result.pattern_recommendations)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.post("/topics", summary="Analyze conversation topics")
async def analyze_topics(
    request: TopicAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze conversation topics and themes using AI.
    
    Identifies main topics, subtopics, business relevance,
    and extracts named entities from conversation content.
    """
    try:
        # Validate conversation data
        if not request.conversation_data.get('messages'):
            raise HTTPException(status_code=400, detail="Conversation data must include messages")
        
        # Analyze topics
        topics_result = await kelly_intelligence_service.analyze_topics(
            conversation_id=request.conversation_id,
            conversation_data=request.conversation_data
        )
        
        # Enhance topic data based on request options
        enhanced_topics = []
        for topic in topics_result:
            enhanced_topic = {
                'name': topic.get('name'),
                'category': topic.get('category'),
                'confidence': topic.get('confidence'),
                'sentiment': topic.get('sentiment'),
                'urgency': topic.get('urgency', 'normal')
            }
            
            if request.extract_entities:
                enhanced_topic['entities'] = topic.get('entities', [])
                enhanced_topic['keywords'] = topic.get('keywords', [])
            
            if request.business_context:
                enhanced_topic['business_relevance'] = topic.get('business_relevance', 0.5)
                enhanced_topic['product_relevance'] = topic.get('product_relevance', 0.5)
                enhanced_topic['sales_relevance'] = topic.get('sales_relevance', 0.5)
            
            enhanced_topics.append(enhanced_topic)
        
        # Sort by business relevance if requested
        if request.business_context:
            enhanced_topics.sort(key=lambda x: x.get('business_relevance', 0), reverse=True)
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'conversation_id': request.conversation_id,
                    'topics': enhanced_topics,
                    'topic_count': len(enhanced_topics),
                    'high_relevance_topics': len([t for t in enhanced_topics if t.get('business_relevance', 0) > 0.7]) if request.business_context else None,
                    'analyzed_at': datetime.utcnow().isoformat()
                },
                'metadata': {
                    'extract_entities': request.extract_entities,
                    'business_context': request.business_context,
                    'topics_stored': len(topics_result)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze topics: {str(e)}")

@router.post("/anomalies", summary="Detect conversation anomalies")
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Detect anomalies in conversation patterns and behavior.
    
    Identifies unusual patterns, performance deviations, and
    potential issues requiring attention or investigation.
    """
    try:
        # Validate conversation data
        if not request.conversation_data.get('messages'):
            raise HTTPException(status_code=400, detail="Conversation data must include messages")
        
        # Adjust detection sensitivity
        original_threshold = kelly_intelligence_service.anomaly_threshold
        kelly_intelligence_service.anomaly_threshold = request.detection_sensitivity
        
        try:
            # Detect anomalies
            anomalies_result = await kelly_intelligence_service.detect_anomalies(
                conversation_id=request.conversation_id,
                conversation_data=request.conversation_data
            )
        finally:
            # Restore original threshold
            kelly_intelligence_service.anomaly_threshold = original_threshold
        
        # Enhance anomaly data with explanations if requested
        enhanced_anomalies = []
        for anomaly in anomalies_result:
            enhanced_anomaly = {
                'metric': anomaly['metric'],
                'current_value': anomaly['current_value'],
                'expected_value': anomaly['expected_value'],
                'deviation_magnitude': anomaly['deviation_magnitude'],
                'severity': anomaly['severity'],
                'type': anomaly['type']
            }
            
            if request.include_explanations:
                # Generate human-readable explanation
                metric_name = anomaly['metric'].replace('_', ' ')
                if anomaly['current_value'] > anomaly['expected_value']:
                    direction = 'higher than'
                else:
                    direction = 'lower than'
                
                enhanced_anomaly['explanation'] = (
                    f"The {metric_name} ({anomaly['current_value']:.2f}) is "
                    f"significantly {direction} expected ({anomaly['expected_value']:.2f}), "
                    f"representing a {anomaly['deviation_magnitude']:.1f} standard deviation anomaly."
                )
                
                # Add potential causes and recommendations
                enhanced_anomaly['potential_causes'] = _generate_anomaly_causes(anomaly)
                enhanced_anomaly['recommendations'] = _generate_anomaly_recommendations(anomaly)
            
            enhanced_anomalies.append(enhanced_anomaly)
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'conversation_id': request.conversation_id,
                    'anomalies': enhanced_anomalies,
                    'anomaly_count': len(enhanced_anomalies),
                    'high_severity_count': len([a for a in enhanced_anomalies if a['severity'] == 'high']),
                    'detected_at': datetime.utcnow().isoformat()
                },
                'metadata': {
                    'detection_sensitivity': request.detection_sensitivity,
                    'include_explanations': request.include_explanations,
                    'threshold_used': request.detection_sensitivity
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")

@router.get("/insights", summary="Get AI-generated insights")
async def get_ai_insights(
    conversation_id: Optional[str] = Query(None, description="Specific conversation filter"),
    account_id: Optional[str] = Query(None, description="Account ID filter"),
    insight_type: Optional[str] = Query(None, description="Insight type filter"),
    priority: Optional[str] = Query(None, regex="^(low|medium|high|critical)$", description="Priority filter"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of insights"),
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get AI-generated insights from conversation intelligence analysis.
    
    Returns insights about patterns, opportunities, risks, and
    recommendations based on recent conversation analysis.
    """
    try:
        # Validate access
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        async with get_session() as session:
            # Build query for insights
            query = session.query(ConversationInsight).filter(
                ConversationInsight.generated_at >= cutoff_time
            )
            
            if conversation_id:
                query = query.filter(ConversationInsight.conversation_id == conversation_id)
            
            if account_id:
                query = query.filter(ConversationInsight.account_id == account_id)
            
            if insight_type:
                query = query.filter(ConversationInsight.insight_type == insight_type)
            
            if priority:
                query = query.filter(ConversationInsight.priority == priority)
            
            # Get insights ordered by priority and confidence
            insights = query.order_by(
                ConversationInsight.priority.desc(),
                ConversationInsight.confidence_score.desc(),
                ConversationInsight.generated_at.desc()
            ).limit(limit).all()
            
            # Format insights for response
            formatted_insights = []
            for insight in insights:
                formatted_insight = {
                    'id': str(insight.id),
                    'conversation_id': insight.conversation_id,
                    'type': insight.insight_type,
                    'category': insight.insight_category,
                    'priority': insight.priority,
                    'title': insight.title,
                    'description': insight.description,
                    'confidence_score': insight.confidence_score,
                    'impact_score': insight.impact_score,
                    'actionable': insight.actionable,
                    'status': insight.status,
                    'generated_at': insight.generated_at.isoformat(),
                    'recommended_actions': insight.recommended_actions,
                    'urgency_level': insight.urgency_level
                }
                formatted_insights.append(formatted_insight)
            
            # Get summary statistics
            total_insights = query.count()
            high_priority_count = len([i for i in formatted_insights if i['priority'] in ['high', 'critical']])
            actionable_count = len([i for i in formatted_insights if i['actionable']])
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'insights': formatted_insights,
                        'summary': {
                            'total_found': total_insights,
                            'returned_count': len(formatted_insights),
                            'high_priority_count': high_priority_count,
                            'actionable_count': actionable_count
                        }
                    },
                    'metadata': {
                        'filters': {
                            'conversation_id': conversation_id,
                            'account_id': account_id,
                            'insight_type': insight_type,
                            'priority': priority,
                            'hours_back': hours_back
                        },
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@router.get("/reports/intelligence", summary="Generate intelligence report")
async def generate_intelligence_report(
    account_id: Optional[str] = Query(None, description="Account ID filter"),
    report_type: str = Query("weekly", regex="^(daily|weekly|monthly|custom)$", description="Report type"),
    period_days: int = Query(7, ge=1, le=90, description="Report period in days"),
    include_patterns: bool = Query(True, description="Include pattern analysis"),
    include_anomalies: bool = Query(True, description="Include anomaly detection"),
    include_recommendations: bool = Query(True, description="Include recommendations"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate comprehensive intelligence report with insights and recommendations.
    
    Aggregates conversation intelligence data into executive summary
    with key insights, patterns, anomalies, and strategic recommendations.
    """
    try:
        # Validate access
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        start_date = datetime.utcnow() - timedelta(days=period_days)
        end_date = datetime.utcnow()
        
        async with get_session() as session:
            # Gather data for report
            report_data = {}
            
            # Get conversation insights summary
            insights_query = session.query(ConversationInsight).filter(
                ConversationInsight.generated_at >= start_date
            )
            if account_id:
                insights_query = insights_query.filter(ConversationInsight.account_id == account_id)
            
            insights = insights_query.all()
            
            report_data['insights_summary'] = {
                'total_insights': len(insights),
                'high_priority': len([i for i in insights if i.priority in ['high', 'critical']]),
                'actionable_insights': len([i for i in insights if i.actionable]),
                'categories': _summarize_insight_categories(insights)
            }
            
            # Get pattern analysis if requested
            if include_patterns:
                patterns = await kelly_intelligence_service.get_successful_patterns(
                    account_id=account_id,
                    min_success_rate=0.6
                )
                report_data['pattern_analysis'] = {
                    'successful_patterns': len([p for p in patterns if p['success_rate'] > 0.8]),
                    'emerging_patterns': len([p for p in patterns if 0.6 <= p['success_rate'] <= 0.8]),
                    'top_patterns': patterns[:5]  # Top 5 patterns
                }
            
            # Get anomaly summary if requested
            if include_anomalies:
                anomalies_query = session.query(AnomalyDetection).filter(
                    AnomalyDetection.detected_at >= start_date
                )
                if account_id:
                    anomalies_query = anomalies_query.filter(AnomalyDetection.account_id == account_id)
                
                anomalies = anomalies_query.all()
                report_data['anomaly_summary'] = {
                    'total_anomalies': len(anomalies),
                    'high_severity': len([a for a in anomalies if a.severity == 'high']),
                    'unresolved': len([a for a in anomalies if a.status == 'new']),
                    'types': _summarize_anomaly_types(anomalies)
                }
            
            # Get recommendations if requested
            if include_recommendations:
                recommendations_query = session.query(AiRecommendation).filter(
                    AiRecommendation.generated_at >= start_date
                )
                if account_id:
                    recommendations_query = recommendations_query.filter(AiRecommendation.account_id == account_id)
                
                recommendations = recommendations_query.all()
                report_data['recommendations_summary'] = {
                    'total_recommendations': len(recommendations),
                    'high_priority': len([r for r in recommendations if r.priority in ['high', 'critical']]),
                    'implemented': len([r for r in recommendations if r.status == 'implemented']),
                    'categories': _summarize_recommendation_categories(recommendations)
                }
            
            # Generate executive summary
            executive_summary = await _generate_executive_summary(report_data, period_days)
            
            # Generate key recommendations
            key_recommendations = await _generate_key_recommendations(report_data)
            
            # Create intelligence report
            from app.models.kelly_intelligence import IntelligenceReport
            intelligence_report = IntelligenceReport(
                report_name=f"{report_type.title()} Intelligence Report",
                report_type=report_type,
                report_category='intelligence',
                account_id=account_id,
                period_start=start_date,
                period_end=end_date,
                period_type=report_type,
                executive_summary=executive_summary,
                key_insights=report_data,
                recommendations=key_recommendations,
                generated_by='ai_system',
                generation_method='automated',
                detailed_data=report_data
            )
            
            session.add(intelligence_report)
            await session.commit()
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': {
                        'report_id': str(intelligence_report.id),
                        'report_name': intelligence_report.report_name,
                        'executive_summary': executive_summary,
                        'key_insights': report_data,
                        'recommendations': key_recommendations,
                        'period': {
                            'start': start_date.isoformat(),
                            'end': end_date.isoformat(),
                            'days': period_days
                        },
                        'generated_at': datetime.utcnow().isoformat()
                    },
                    'metadata': {
                        'report_type': report_type,
                        'account_id': account_id,
                        'includes': {
                            'patterns': include_patterns,
                            'anomalies': include_anomalies,
                            'recommendations': include_recommendations
                        }
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating intelligence report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

# Helper functions

async def _has_account_access(user: Dict, account_id: str) -> bool:
    """Check if user has access to specific account"""
    # This would implement actual access control logic
    return True

def _generate_anomaly_causes(anomaly: Dict[str, Any]) -> List[str]:
    """Generate potential causes for detected anomaly"""
    metric = anomaly['metric']
    causes = []
    
    if metric == 'avg_response_time':
        if anomaly['current_value'] > anomaly['expected_value']:
            causes.extend([
                "High system load or performance issues",
                "Complex queries requiring more AI processing time",
                "Network latency or connectivity issues",
                "AI model experiencing higher uncertainty"
            ])
        else:
            causes.extend([
                "Improved AI model performance",
                "Simpler queries or repetitive patterns",
                "System optimization improvements"
            ])
    elif metric == 'conversation_quality_score':
        if anomaly['current_value'] < anomaly['expected_value']:
            causes.extend([
                "AI model struggling with complex or unusual queries",
                "User communication style mismatch",
                "Technical issues affecting AI responses",
                "Need for model fine-tuning or training updates"
            ])
    
    return causes[:3]  # Return top 3 causes

def _generate_anomaly_recommendations(anomaly: Dict[str, Any]) -> List[str]:
    """Generate recommendations for addressing anomaly"""
    metric = anomaly['metric']
    recommendations = []
    
    if metric == 'avg_response_time':
        if anomaly['current_value'] > anomaly['expected_value']:
            recommendations.extend([
                "Monitor system performance and consider scaling resources",
                "Review and optimize AI model inference pipeline",
                "Implement response time alerts for early detection",
                "Consider human intervention for complex queries"
            ])
    elif metric == 'conversation_quality_score':
        if anomaly['current_value'] < anomaly['expected_value']:
            recommendations.extend([
                "Review conversation logs for pattern identification",
                "Consider additional AI training on similar scenarios",
                "Implement quality monitoring alerts",
                "Schedule human review of affected conversations"
            ])
    
    return recommendations[:3]  # Return top 3 recommendations

def _summarize_insight_categories(insights: List) -> Dict[str, int]:
    """Summarize insights by category"""
    categories = {}
    for insight in insights:
        category = insight.insight_category
        categories[category] = categories.get(category, 0) + 1
    return categories

def _summarize_anomaly_types(anomalies: List) -> Dict[str, int]:
    """Summarize anomalies by type"""
    types = {}
    for anomaly in anomalies:
        anomaly_type = anomaly.anomaly_type
        types[anomaly_type] = types.get(anomaly_type, 0) + 1
    return types

def _summarize_recommendation_categories(recommendations: List) -> Dict[str, int]:
    """Summarize recommendations by category"""
    categories = {}
    for rec in recommendations:
        category = rec.category
        categories[category] = categories.get(category, 0) + 1
    return categories

async def _generate_executive_summary(report_data: Dict, period_days: int) -> str:
    """Generate executive summary for intelligence report"""
    insights_count = report_data.get('insights_summary', {}).get('total_insights', 0)
    high_priority_insights = report_data.get('insights_summary', {}).get('high_priority', 0)
    
    summary = f"Over the past {period_days} days, the AI system generated {insights_count} insights "
    
    if high_priority_insights > 0:
        summary += f"with {high_priority_insights} requiring immediate attention. "
    else:
        summary += "with no critical issues detected. "
    
    if 'pattern_analysis' in report_data:
        successful_patterns = report_data['pattern_analysis'].get('successful_patterns', 0)
        summary += f"Analysis identified {successful_patterns} highly successful conversation patterns. "
    
    if 'anomaly_summary' in report_data:
        total_anomalies = report_data['anomaly_summary'].get('total_anomalies', 0)
        if total_anomalies > 0:
            summary += f"Detected {total_anomalies} anomalies requiring investigation. "
        else:
            summary += "No significant anomalies detected in conversation patterns. "
    
    summary += "System performance remains within expected parameters with opportunities for optimization identified."
    
    return summary

async def _generate_key_recommendations(report_data: Dict) -> List[str]:
    """Generate key strategic recommendations"""
    recommendations = []
    
    # Insights-based recommendations
    if report_data.get('insights_summary', {}).get('high_priority', 0) > 0:
        recommendations.append("Review and address high-priority insights to prevent potential issues")
    
    # Pattern-based recommendations
    if 'pattern_analysis' in report_data:
        successful_patterns = report_data['pattern_analysis'].get('successful_patterns', 0)
        if successful_patterns > 0:
            recommendations.append("Implement successful patterns across more conversations to improve outcomes")
    
    # Anomaly-based recommendations
    if 'anomaly_summary' in report_data:
        unresolved_anomalies = report_data['anomaly_summary'].get('unresolved', 0)
        if unresolved_anomalies > 0:
            recommendations.append("Investigate and resolve unresolved anomalies to maintain system performance")
    
    # General recommendations
    recommendations.extend([
        "Continue monitoring conversation quality and customer satisfaction metrics",
        "Expand AI training data based on successful interaction patterns",
        "Implement proactive alerts for early anomaly detection"
    ])
    
    return recommendations[:5]  # Return top 5 recommendations