"""
Kelly Analytics Engine

Real-time analytics engine for conversation quality scoring, performance benchmarking,
revenue attribution, and business intelligence generation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
import numpy as np
from sqlalchemy import func, and_, or_, desc, asc
from sqlalchemy.orm import Session

from app.database.connection import get_session
from app.models.kelly_analytics import (
    ConversationAnalytics, PerformanceMetric, RevenueAttribution, 
    QualityScore, TrendAnalysis, CustomAnalyticsQuery
)
from app.models.kelly_monitoring import SystemMetric, ActivityEvent
from app.models.kelly_crm import Contact, Deal, TouchPoint
from app.core.redis import redis_manager
from app.core.claude_ai import claude_client

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsResult:
    """Standardized analytics result structure"""
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    confidence: float = 1.0

@dataclass
class ConversationQualityMetrics:
    """Conversation quality assessment metrics"""
    overall_score: float
    component_scores: Dict[str, float]
    grade: str
    areas_for_improvement: List[str]
    strengths: List[str]

class KellyAnalyticsEngine:
    """
    Advanced analytics engine for Kelly AI system.
    
    Provides real-time analytics, quality scoring, performance benchmarking,
    and business intelligence capabilities.
    """
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache for analytics
        self.quality_weights = {
            'relevance': 0.25,
            'helpfulness': 0.20,
            'accuracy': 0.20,
            'empathy': 0.15,
            'professionalism': 0.10,
            'efficiency': 0.10
        }
        
    async def get_real_time_dashboard_metrics(
        self, 
        account_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get real-time dashboard metrics for Kelly AI system.
        
        Args:
            account_id: Optional account filter
            time_range_hours: Time range for metrics
            
        Returns:
            Comprehensive dashboard metrics
        """
        try:
            # Check cache first
            cache_key = f"dashboard_metrics:{account_id}:{time_range_hours}"
            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            async with get_session() as session:
                # Conversation metrics
                conv_query = session.query(ConversationAnalytics).filter(
                    ConversationAnalytics.started_at >= cutoff_time
                )
                if account_id:
                    conv_query = conv_query.filter(ConversationAnalytics.account_id == account_id)
                
                conversations = conv_query.all()
                
                # Performance metrics
                perf_query = session.query(PerformanceMetric).filter(
                    PerformanceMetric.period_start >= cutoff_time
                )
                if account_id:
                    perf_query = perf_query.filter(PerformanceMetric.account_id == account_id)
                
                performance_metrics = perf_query.all()
                
                # Quality scores
                quality_query = session.query(QualityScore).filter(
                    QualityScore.scored_at >= cutoff_time
                )
                if account_id:
                    quality_query = quality_query.filter(QualityScore.account_id == account_id)
                
                quality_scores = quality_query.all()
                
                # Revenue attribution
                revenue_query = session.query(RevenueAttribution).filter(
                    RevenueAttribution.conversion_occurred_at >= cutoff_time
                )
                if account_id:
                    revenue_query = revenue_query.filter(RevenueAttribution.account_id == account_id)
                
                revenue_data = revenue_query.all()
                
                # Build comprehensive metrics
                metrics = await self._build_dashboard_metrics(
                    conversations, performance_metrics, quality_scores, revenue_data
                )
                
                # Cache the result
                await redis_manager.set(
                    cache_key, 
                    json.dumps(metrics, default=str), 
                    expire=self.cache_ttl
                )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {str(e)}")
            raise
    
    async def _build_dashboard_metrics(
        self,
        conversations: List[ConversationAnalytics],
        performance_metrics: List[PerformanceMetric],
        quality_scores: List[QualityScore],
        revenue_data: List[RevenueAttribution]
    ) -> Dict[str, Any]:
        """Build comprehensive dashboard metrics from data"""
        
        # Conversation metrics
        total_conversations = len(conversations)
        active_conversations = len([c for c in conversations if c.ended_at is None])
        avg_quality = statistics.mean([c.conversation_quality_score for c in conversations]) if conversations else 0
        avg_duration = statistics.mean([c.duration_minutes for c in conversations if c.duration_minutes]) if conversations else 0
        
        # AI vs Human metrics
        ai_handled = len([c for c in conversations if c.human_intervention_count == 0])
        human_interventions = sum(c.human_intervention_count for c in conversations)
        ai_success_rate = len([c for c in conversations if c.resolution_achieved and c.human_intervention_count == 0]) / max(ai_handled, 1)
        
        # Quality distribution
        quality_distribution = {
            'excellent': len([q for q in quality_scores if q.overall_score >= 0.9]),
            'good': len([q for q in quality_scores if 0.7 <= q.overall_score < 0.9]),
            'average': len([q for q in quality_scores if 0.5 <= q.overall_score < 0.7]),
            'poor': len([q for q in quality_scores if q.overall_score < 0.5])
        }
        
        # Performance trends
        response_times = [p.avg_response_time_seconds for p in performance_metrics if p.avg_response_time_seconds]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Revenue metrics
        total_revenue = sum(r.revenue_amount for r in revenue_data)
        ai_attributed_revenue = sum(r.revenue_amount for r in revenue_data if r.ai_contribution and r.ai_contribution > 0.5)
        revenue_per_conversation = total_revenue / max(total_conversations, 1)
        
        # Anomalies and alerts
        anomalies = len([c for c in conversations if c.anomaly_detected])
        escalations = len([c for c in conversations if c.escalation_required])
        
        return {
            'overview': {
                'total_conversations': total_conversations,
                'active_conversations': active_conversations,
                'avg_quality_score': round(avg_quality, 3),
                'avg_duration_minutes': round(avg_duration, 2),
                'timestamp': datetime.utcnow().isoformat()
            },
            'ai_performance': {
                'ai_handled_conversations': ai_handled,
                'human_interventions': human_interventions,
                'ai_success_rate': round(ai_success_rate, 3),
                'intervention_rate': round(human_interventions / max(total_conversations, 1), 3),
                'avg_response_time_seconds': round(avg_response_time, 2)
            },
            'quality_metrics': {
                'distribution': quality_distribution,
                'avg_quality_score': round(avg_quality, 3),
                'quality_trend': await self._calculate_quality_trend(quality_scores)
            },
            'business_impact': {
                'total_revenue': round(total_revenue, 2),
                'ai_attributed_revenue': round(ai_attributed_revenue, 2),
                'revenue_per_conversation': round(revenue_per_conversation, 2),
                'conversion_rate': len([c for c in conversations if c.conversion_event]) / max(total_conversations, 1)
            },
            'alerts_and_issues': {
                'anomalies_detected': anomalies,
                'escalations_required': escalations,
                'quality_alerts': len([q for q in quality_scores if q.overall_score < 0.5]),
                'safety_alerts': len([q for q in quality_scores if q.safety_score < 0.8])
            },
            'trends': await self._generate_trend_indicators(conversations, quality_scores, revenue_data)
        }
    
    async def calculate_conversation_quality_score(
        self, 
        conversation_id: str,
        conversation_data: Dict[str, Any],
        ai_confidence: Optional[float] = None
    ) -> ConversationQualityMetrics:
        """
        Calculate comprehensive quality score for a conversation using AI analysis.
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_data: Conversation messages and metadata
            ai_confidence: AI confidence level if available
            
        Returns:
            Detailed quality metrics and assessment
        """
        try:
            # Use Claude AI to analyze conversation quality
            quality_prompt = f"""
            Analyze this conversation for quality and provide scores (0.0-1.0) for each dimension:
            
            Conversation: {json.dumps(conversation_data, indent=2)}
            
            Please evaluate:
            1. Relevance: How relevant are the responses to user queries?
            2. Helpfulness: How helpful are the responses in solving user problems?
            3. Accuracy: How accurate is the information provided?
            4. Empathy: How well does the AI understand and respond to emotions?
            5. Professionalism: How professional and appropriate is the communication?
            6. Efficiency: How efficiently are issues resolved?
            
            Also identify:
            - Key strengths of the conversation
            - Areas that need improvement
            - Overall conversation grade (A+, A, B+, B, C+, C, D, F)
            
            Return as JSON with scores and analysis.
            """
            
            claude_response = await claude_client.analyze_text(quality_prompt)
            quality_analysis = json.loads(claude_response)
            
            # Calculate weighted overall score
            component_scores = {
                'relevance': quality_analysis.get('relevance', 0.5),
                'helpfulness': quality_analysis.get('helpfulness', 0.5),
                'accuracy': quality_analysis.get('accuracy', 0.5),
                'empathy': quality_analysis.get('empathy', 0.5),
                'professionalism': quality_analysis.get('professionalism', 0.5),
                'efficiency': quality_analysis.get('efficiency', 0.5)
            }
            
            overall_score = sum(
                score * self.quality_weights.get(dimension, 0)
                for dimension, score in component_scores.items()
            )
            
            # Store quality score in database
            async with get_session() as session:
                quality_score = QualityScore(
                    conversation_id=conversation_id,
                    account_id=conversation_data.get('account_id'),
                    user_id=conversation_data.get('user_id'),
                    overall_score=overall_score,
                    overall_grade=quality_analysis.get('grade', 'B'),
                    **component_scores,
                    ai_confidence=ai_confidence,
                    scoring_method='ai_automatic',
                    scored_by='claude_ai',
                    conversation_timestamp=datetime.fromisoformat(conversation_data.get('timestamp', datetime.utcnow().isoformat())),
                    scoring_details=quality_analysis,
                    improvement_suggestions=quality_analysis.get('improvements', [])
                )
                session.add(quality_score)
                await session.commit()
            
            return ConversationQualityMetrics(
                overall_score=overall_score,
                component_scores=component_scores,
                grade=quality_analysis.get('grade', 'B'),
                areas_for_improvement=quality_analysis.get('improvements', []),
                strengths=quality_analysis.get('strengths', [])
            )
            
        except Exception as e:
            logger.error(f"Error calculating conversation quality: {str(e)}")
            # Return default scores if analysis fails
            return ConversationQualityMetrics(
                overall_score=0.5,
                component_scores={k: 0.5 for k in self.quality_weights.keys()},
                grade='C',
                areas_for_improvement=['Unable to analyze - system error'],
                strengths=[]
            )
    
    async def calculate_performance_metrics(
        self,
        account_id: str,
        period_type: str = 'day',
        include_ai_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for team and AI.
        
        Args:
            account_id: Account to analyze
            period_type: Analysis period (hour, day, week, month)
            include_ai_comparison: Whether to include AI vs human comparison
            
        Returns:
            Detailed performance metrics
        """
        try:
            # Define time period
            if period_type == 'hour':
                period_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                period_end = period_start + timedelta(hours=1)
            elif period_type == 'day':
                period_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                period_end = period_start + timedelta(days=1)
            elif period_type == 'week':
                period_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
                period_end = datetime.utcnow()
            else:  # month
                period_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                period_end = datetime.utcnow()
            
            async with get_session() as session:
                # Get conversations in period
                conversations = session.query(ConversationAnalytics).filter(
                    and_(
                        ConversationAnalytics.account_id == account_id,
                        ConversationAnalytics.started_at >= period_start,
                        ConversationAnalytics.started_at < period_end
                    )
                ).all()
                
                # Get quality scores
                quality_scores = session.query(QualityScore).filter(
                    and_(
                        QualityScore.account_id == account_id,
                        QualityScore.scored_at >= period_start,
                        QualityScore.scored_at < period_end
                    )
                ).all()
                
                # Calculate metrics
                total_conversations = len(conversations)
                if total_conversations == 0:
                    return self._empty_performance_metrics(period_start, period_end)
                
                # Response time metrics
                response_times = [c.avg_response_time_seconds for c in conversations if c.avg_response_time_seconds]
                avg_response_time = statistics.mean(response_times) if response_times else 0
                median_response_time = statistics.median(response_times) if response_times else 0
                p95_response_time = np.percentile(response_times, 95) if response_times else 0
                
                # Quality metrics
                avg_quality = statistics.mean([q.overall_score for q in quality_scores]) if quality_scores else 0
                resolution_rate = len([c for c in conversations if c.resolution_achieved]) / total_conversations
                escalation_rate = len([c for c in conversations if c.escalation_required]) / total_conversations
                
                # AI vs Human comparison
                ai_handled = [c for c in conversations if c.human_intervention_count == 0]
                human_handled = [c for c in conversations if c.human_intervention_count > 0]
                
                ai_success_rate = len([c for c in ai_handled if c.resolution_achieved]) / max(len(ai_handled), 1)
                human_success_rate = len([c for c in human_handled if c.resolution_achieved]) / max(len(human_handled), 1)
                
                # Business metrics
                leads_qualified = len([c for c in conversations if c.lead_qualified])
                conversions = len([c for c in conversations if c.conversion_event])
                total_revenue = sum(c.revenue_attributed for c in conversations if c.revenue_attributed) or 0
                
                # Calculate costs (simplified model)
                ai_cost = len(ai_handled) * 0.05  # $0.05 per AI conversation
                human_cost = sum(c.duration_minutes for c in human_handled if c.duration_minutes) * 0.5  # $0.50 per minute
                total_cost = ai_cost + human_cost
                
                metrics = {
                    'period': {
                        'type': period_type,
                        'start': period_start.isoformat(),
                        'end': period_end.isoformat()
                    },
                    'volume_metrics': {
                        'total_conversations': total_conversations,
                        'ai_handled': len(ai_handled),
                        'human_handled': len(human_handled),
                        'intervention_rate': len(human_handled) / total_conversations
                    },
                    'response_time_metrics': {
                        'avg_response_time_seconds': round(avg_response_time, 2),
                        'median_response_time_seconds': round(median_response_time, 2),
                        'p95_response_time_seconds': round(p95_response_time, 2)
                    },
                    'quality_metrics': {
                        'avg_quality_score': round(avg_quality, 3),
                        'resolution_rate': round(resolution_rate, 3),
                        'escalation_rate': round(escalation_rate, 3)
                    },
                    'ai_vs_human': {
                        'ai_success_rate': round(ai_success_rate, 3),
                        'human_success_rate': round(human_success_rate, 3),
                        'ai_efficiency_advantage': round(ai_success_rate - human_success_rate, 3)
                    } if include_ai_comparison else {},
                    'business_metrics': {
                        'leads_qualified': leads_qualified,
                        'conversions_achieved': conversions,
                        'conversion_rate': round(conversions / total_conversations, 3),
                        'revenue_generated': round(total_revenue, 2),
                        'revenue_per_conversation': round(total_revenue / total_conversations, 2)
                    },
                    'cost_metrics': {
                        'total_cost': round(total_cost, 2),
                        'cost_per_conversation': round(total_cost / total_conversations, 2),
                        'ai_cost': round(ai_cost, 2),
                        'human_cost': round(human_cost, 2),
                        'cost_savings_vs_all_human': round((total_conversations * 5.0) - total_cost, 2)  # Assuming $5 per human conversation
                    }
                }
                
                # Store performance metrics
                performance_metric = PerformanceMetric(
                    metric_type='team',
                    account_id=account_id,
                    period_type=period_type,
                    period_start=period_start,
                    period_end=period_end,
                    conversations_handled=total_conversations,
                    avg_response_time_seconds=avg_response_time,
                    median_response_time_seconds=median_response_time,
                    p95_response_time_seconds=p95_response_time,
                    avg_quality_score=avg_quality,
                    resolution_rate=resolution_rate,
                    escalation_rate=escalation_rate,
                    ai_handled_count=len(ai_handled),
                    human_handled_count=len(human_handled),
                    ai_success_rate=ai_success_rate,
                    human_success_rate=human_success_rate,
                    total_cost=total_cost,
                    cost_per_conversation=total_cost / total_conversations,
                    ai_cost=ai_cost,
                    human_cost=human_cost,
                    revenue_generated=total_revenue,
                    leads_qualified=leads_qualified,
                    conversions_achieved=conversions,
                    detailed_metrics=metrics
                )
                
                session.add(performance_metric)
                await session.commit()
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    async def calculate_revenue_attribution(
        self,
        conversation_id: str,
        deal_data: Dict[str, Any],
        attribution_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate revenue attribution for a conversation.
        
        Args:
            conversation_id: Conversation that led to revenue
            deal_data: Deal/transaction information
            attribution_confidence: Confidence in attribution (0.0-1.0)
            
        Returns:
            Revenue attribution details
        """
        try:
            async with get_session() as session:
                # Get conversation data
                conversation = session.query(ConversationAnalytics).filter(
                    ConversationAnalytics.conversation_id == conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Determine attribution type based on timing and interaction
                conversion_delay = datetime.utcnow() - conversation.started_at
                delay_hours = conversion_delay.total_seconds() / 3600
                
                if delay_hours <= 1:
                    attribution_type = 'direct'
                elif delay_hours <= 24:
                    attribution_type = 'assisted'
                else:
                    attribution_type = 'influenced'
                
                # Calculate AI vs human contribution
                if conversation.human_intervention_count == 0:
                    ai_contribution = 1.0
                    human_contribution = 0.0
                else:
                    # Estimate based on intervention ratio
                    total_messages = conversation.total_messages
                    ai_messages = conversation.ai_messages
                    ai_contribution = ai_messages / max(total_messages, 1)
                    human_contribution = 1.0 - ai_contribution
                
                # Create revenue attribution record
                revenue_attribution = RevenueAttribution(
                    conversation_id=conversation_id,
                    account_id=conversation.account_id,
                    user_id=conversation.user_id,
                    revenue_amount=deal_data.get('amount', 0),
                    currency=deal_data.get('currency', 'USD'),
                    attribution_type=attribution_type,
                    attribution_confidence=attribution_confidence,
                    touchpoint_type='conversation',
                    ai_contribution=ai_contribution,
                    human_contribution=human_contribution,
                    conversation_started_at=conversation.started_at,
                    conversion_occurred_at=datetime.utcnow(),
                    attribution_delay_hours=delay_hours,
                    deal_id=deal_data.get('deal_id'),
                    transaction_id=deal_data.get('transaction_id'),
                    journey_stage=deal_data.get('journey_stage', 'decision'),
                    attribution_factors={
                        'conversation_quality': conversation.conversation_quality_score,
                        'ai_confidence': conversation.ai_confidence_avg,
                        'resolution_achieved': conversation.resolution_achieved,
                        'customer_satisfaction': conversation.user_satisfaction_score
                    }
                )
                
                session.add(revenue_attribution)
                await session.commit()
                
                return {
                    'attribution_id': str(revenue_attribution.id),
                    'attribution_type': attribution_type,
                    'ai_contribution': ai_contribution,
                    'human_contribution': human_contribution,
                    'attribution_confidence': attribution_confidence,
                    'delay_hours': delay_hours,
                    'revenue_amount': deal_data.get('amount', 0)
                }
                
        except Exception as e:
            logger.error(f"Error calculating revenue attribution: {str(e)}")
            raise
    
    async def generate_trend_analysis(
        self,
        metric_name: str,
        account_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate trend analysis for a specific metric.
        
        Args:
            metric_name: Name of metric to analyze
            account_id: Optional account filter
            period_days: Analysis period in days
            
        Returns:
            Trend analysis with forecasting
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            
            async with get_session() as session:
                # Get historical data based on metric type
                if metric_name == 'conversation_quality':
                    query = session.query(QualityScore.scored_at, QualityScore.overall_score).filter(
                        QualityScore.scored_at >= cutoff_date
                    )
                    if account_id:
                        query = query.filter(QualityScore.account_id == account_id)
                    
                    data_points = [(row.scored_at, row.overall_score) for row in query.all()]
                    
                elif metric_name == 'response_time':
                    query = session.query(ConversationAnalytics.started_at, ConversationAnalytics.avg_response_time_seconds).filter(
                        and_(
                            ConversationAnalytics.started_at >= cutoff_date,
                            ConversationAnalytics.avg_response_time_seconds.isnot(None)
                        )
                    )
                    if account_id:
                        query = query.filter(ConversationAnalytics.account_id == account_id)
                    
                    data_points = [(row.started_at, row.avg_response_time_seconds) for row in query.all()]
                    
                else:
                    raise ValueError(f"Unsupported metric: {metric_name}")
                
                if len(data_points) < 3:
                    return {
                        'metric_name': metric_name,
                        'trend_direction': 'insufficient_data',
                        'message': 'Not enough data points for trend analysis'
                    }
                
                # Sort by timestamp
                data_points.sort(key=lambda x: x[0])
                
                # Extract values and calculate trend
                values = [point[1] for point in data_points]
                timestamps = [point[0] for point in data_points]
                
                # Calculate basic statistics
                current_value = values[-1]
                previous_value = values[-2] if len(values) > 1 else current_value
                change_amount = current_value - previous_value
                change_percentage = (change_amount / previous_value * 100) if previous_value != 0 else 0
                
                # Determine trend direction
                if abs(change_percentage) < 5:
                    trend_direction = 'stable'
                elif change_percentage > 0:
                    trend_direction = 'up'
                else:
                    trend_direction = 'down'
                
                # Calculate moving averages
                if len(values) >= 7:
                    moving_avg_7d = statistics.mean(values[-7:])
                else:
                    moving_avg_7d = statistics.mean(values)
                
                if len(values) >= 30:
                    moving_avg_30d = statistics.mean(values[-30:])
                else:
                    moving_avg_30d = statistics.mean(values)
                
                # Calculate volatility
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                mean_value = statistics.mean(values)
                volatility_score = std_dev / mean_value if mean_value != 0 else 0
                
                # Simple linear forecast
                if len(values) >= 3:
                    # Calculate trend slope
                    x_values = list(range(len(values)))
                    slope = np.polyfit(x_values, values, 1)[0]
                    forecast_next = current_value + slope
                    forecast_confidence = max(0.1, 1.0 - volatility_score)  # Lower confidence for volatile metrics
                else:
                    forecast_next = current_value
                    forecast_confidence = 0.5
                
                # Store trend analysis
                trend_analysis = TrendAnalysis(
                    trend_type='performance',
                    metric_name=metric_name,
                    account_id=account_id,
                    period_type='daily',
                    period_start=cutoff_date,
                    period_end=datetime.utcnow(),
                    current_value=current_value,
                    previous_value=previous_value,
                    change_amount=change_amount,
                    change_percentage=change_percentage,
                    trend_direction=trend_direction,
                    moving_average_7d=moving_avg_7d,
                    moving_average_30d=moving_avg_30d,
                    standard_deviation=std_dev,
                    volatility_score=volatility_score,
                    forecast_next_period=forecast_next,
                    forecast_confidence=forecast_confidence,
                    forecast_method='linear',
                    historical_data=[{'timestamp': ts.isoformat(), 'value': val} for ts, val in data_points[-30:]]
                )
                
                session.add(trend_analysis)
                await session.commit()
                
                return {
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'previous_value': previous_value,
                    'change_amount': change_amount,
                    'change_percentage': round(change_percentage, 2),
                    'trend_direction': trend_direction,
                    'moving_averages': {
                        '7_day': round(moving_avg_7d, 3),
                        '30_day': round(moving_avg_30d, 3)
                    },
                    'volatility_score': round(volatility_score, 3),
                    'forecast': {
                        'next_value': round(forecast_next, 3),
                        'confidence': round(forecast_confidence, 3),
                        'method': 'linear'
                    },
                    'data_points': len(data_points),
                    'analysis_period_days': period_days
                }
                
        except Exception as e:
            logger.error(f"Error generating trend analysis: {str(e)}")
            raise
    
    async def execute_custom_analytics_query(
        self,
        query_config: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Execute custom analytics query with caching and optimization.
        
        Args:
            query_config: Query configuration (filters, aggregations, etc.)
            user_id: User requesting the query
            
        Returns:
            Query results with metadata
        """
        try:
            # Generate query hash for caching
            query_hash = hash(json.dumps(query_config, sort_keys=True))
            cache_key = f"custom_query:{query_hash}"
            
            # Check cache first
            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                cached_data = json.loads(cached_result)
                if datetime.fromisoformat(cached_data['cache_expires_at']) > datetime.utcnow():
                    return cached_data['results']
            
            start_time = datetime.utcnow()
            
            async with get_session() as session:
                # Build query based on configuration
                filters = query_config.get('filters', {})
                aggregations = query_config.get('aggregations', {})
                time_range = query_config.get('time_range', {})
                
                # Start with base conversation analytics query
                base_query = session.query(ConversationAnalytics)
                
                # Apply time filters
                if time_range.get('start'):
                    base_query = base_query.filter(
                        ConversationAnalytics.started_at >= datetime.fromisoformat(time_range['start'])
                    )
                if time_range.get('end'):
                    base_query = base_query.filter(
                        ConversationAnalytics.started_at <= datetime.fromisoformat(time_range['end'])
                    )
                
                # Apply filters
                if filters.get('account_id'):
                    base_query = base_query.filter(ConversationAnalytics.account_id == filters['account_id'])
                if filters.get('quality_threshold'):
                    base_query = base_query.filter(ConversationAnalytics.conversation_quality_score >= filters['quality_threshold'])
                if filters.get('has_human_intervention') is not None:
                    if filters['has_human_intervention']:
                        base_query = base_query.filter(ConversationAnalytics.human_intervention_count > 0)
                    else:
                        base_query = base_query.filter(ConversationAnalytics.human_intervention_count == 0)
                
                # Execute query
                conversations = base_query.all()
                
                # Apply aggregations
                results = {}
                
                if 'count' in aggregations:
                    results['total_conversations'] = len(conversations)
                
                if 'avg_quality' in aggregations:
                    results['avg_quality_score'] = statistics.mean([c.conversation_quality_score for c in conversations]) if conversations else 0
                
                if 'avg_duration' in aggregations:
                    durations = [c.duration_minutes for c in conversations if c.duration_minutes]
                    results['avg_duration_minutes'] = statistics.mean(durations) if durations else 0
                
                if 'resolution_rate' in aggregations:
                    results['resolution_rate'] = len([c for c in conversations if c.resolution_achieved]) / max(len(conversations), 1)
                
                if 'ai_success_rate' in aggregations:
                    ai_handled = [c for c in conversations if c.human_intervention_count == 0]
                    ai_success = [c for c in ai_handled if c.resolution_achieved]
                    results['ai_success_rate'] = len(ai_success) / max(len(ai_handled), 1)
                
                if 'revenue_total' in aggregations:
                    results['total_revenue'] = sum(c.revenue_attributed for c in conversations if c.revenue_attributed) or 0
                
                # Apply grouping if requested
                grouping = query_config.get('grouping', [])
                if grouping:
                    grouped_results = {}
                    for group_field in grouping:
                        if group_field == 'account_id':
                            for conv in conversations:
                                if conv.account_id not in grouped_results:
                                    grouped_results[conv.account_id] = []
                                grouped_results[conv.account_id].append(conv)
                        # Add more grouping options as needed
                    
                    # Apply aggregations to each group
                    for group_key, group_conversations in grouped_results.items():
                        group_results = {}
                        if 'count' in aggregations:
                            group_results['count'] = len(group_conversations)
                        if 'avg_quality' in aggregations:
                            group_results['avg_quality'] = statistics.mean([c.conversation_quality_score for c in group_conversations])
                        results[f'group_{group_key}'] = group_results
                
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Store query in database
                custom_query = CustomAnalyticsQuery(
                    query_name=query_config.get('name', 'Unnamed Query'),
                    query_type=query_config.get('type', 'aggregation'),
                    created_by=user_id,
                    filters=filters,
                    aggregations=aggregations,
                    time_range=time_range,
                    execution_time_ms=int(execution_time),
                    rows_returned=len(conversations),
                    results=results,
                    results_hash=str(query_hash)
                )
                
                session.add(custom_query)
                await session.commit()
                
                # Cache results
                cache_data = {
                    'results': results,
                    'metadata': {
                        'execution_time_ms': int(execution_time),
                        'rows_processed': len(conversations),
                        'query_id': str(custom_query.id)
                    },
                    'cache_expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat()
                }
                
                await redis_manager.set(
                    cache_key,
                    json.dumps(cache_data, default=str),
                    expire=3600  # 1 hour
                )
                
                return {
                    'results': results,
                    'metadata': {
                        'execution_time_ms': int(execution_time),
                        'rows_processed': len(conversations),
                        'query_id': str(custom_query.id),
                        'cached': False
                    }
                }
                
        except Exception as e:
            logger.error(f"Error executing custom analytics query: {str(e)}")
            raise
    
    def _empty_performance_metrics(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Return empty performance metrics structure"""
        return {
            'period': {
                'start': period_start.isoformat(),
                'end': period_end.isoformat()
            },
            'volume_metrics': {
                'total_conversations': 0,
                'ai_handled': 0,
                'human_handled': 0,
                'intervention_rate': 0
            },
            'response_time_metrics': {
                'avg_response_time_seconds': 0,
                'median_response_time_seconds': 0,
                'p95_response_time_seconds': 0
            },
            'quality_metrics': {
                'avg_quality_score': 0,
                'resolution_rate': 0,
                'escalation_rate': 0
            },
            'business_metrics': {
                'leads_qualified': 0,
                'conversions_achieved': 0,
                'conversion_rate': 0,
                'revenue_generated': 0,
                'revenue_per_conversation': 0
            }
        }
    
    async def _calculate_quality_trend(self, quality_scores: List[QualityScore]) -> str:
        """Calculate quality trend direction"""
        if len(quality_scores) < 2:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_scores = sorted(quality_scores, key=lambda x: x.scored_at)
        
        # Compare recent vs older scores
        mid_point = len(sorted_scores) // 2
        older_avg = statistics.mean([s.overall_score for s in sorted_scores[:mid_point]])
        recent_avg = statistics.mean([s.overall_score for s in sorted_scores[mid_point:]])
        
        change_percentage = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
        
        if abs(change_percentage) < 5:
            return 'stable'
        elif change_percentage > 0:
            return 'improving'
        else:
            return 'declining'
    
    async def _generate_trend_indicators(
        self,
        conversations: List[ConversationAnalytics],
        quality_scores: List[QualityScore],
        revenue_data: List[RevenueAttribution]
    ) -> Dict[str, str]:
        """Generate trend indicators for dashboard"""
        
        trends = {}
        
        # Conversation volume trend
        if len(conversations) >= 2:
            # Simple trend based on recent vs older conversations
            mid_point = len(conversations) // 2
            older_count = len(conversations[:mid_point])
            recent_count = len(conversations[mid_point:])
            
            if recent_count > older_count * 1.1:
                trends['volume'] = 'increasing'
            elif recent_count < older_count * 0.9:
                trends['volume'] = 'decreasing'
            else:
                trends['volume'] = 'stable'
        else:
            trends['volume'] = 'insufficient_data'
        
        # Quality trend
        trends['quality'] = await self._calculate_quality_trend(quality_scores)
        
        # Revenue trend
        if len(revenue_data) >= 2:
            sorted_revenue = sorted(revenue_data, key=lambda x: x.conversion_occurred_at)
            mid_point = len(sorted_revenue) // 2
            older_revenue = sum(r.revenue_amount for r in sorted_revenue[:mid_point])
            recent_revenue = sum(r.revenue_amount for r in sorted_revenue[mid_point:])
            
            if recent_revenue > older_revenue * 1.1:
                trends['revenue'] = 'increasing'
            elif recent_revenue < older_revenue * 0.9:
                trends['revenue'] = 'decreasing'
            else:
                trends['revenue'] = 'stable'
        else:
            trends['revenue'] = 'insufficient_data'
        
        return trends

# Create global analytics engine instance
kelly_analytics_engine = KellyAnalyticsEngine()