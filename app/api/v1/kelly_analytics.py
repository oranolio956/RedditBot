"""
Kelly Analytics API Endpoints

FastAPI endpoints for comprehensive analytics including dashboard metrics,
conversation quality, performance analysis, and revenue attribution.
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
from app.services.kelly_analytics_engine import kelly_analytics_engine
from app.models.kelly_analytics import ConversationAnalytics, PerformanceMetric, TrendAnalysis

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation

class DashboardMetricsRequest(BaseModel):
    account_id: Optional[str] = None
    time_range_hours: int = Field(default=24, ge=1, le=8760)  # 1 hour to 1 year
    include_trends: bool = True
    include_ai_comparison: bool = True

class ConversationQualityRequest(BaseModel):
    conversation_id: str
    conversation_data: Dict[str, Any]
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    store_results: bool = True

class PerformanceMetricsRequest(BaseModel):
    account_id: str
    period_type: str = Field(default="day", regex="^(hour|day|week|month)$")
    include_ai_comparison: bool = True
    include_cost_analysis: bool = True

class RevenueAttributionRequest(BaseModel):
    conversation_id: str
    deal_data: Dict[str, Any]
    attribution_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class TrendAnalysisRequest(BaseModel):
    metric_name: str = Field(..., regex="^(conversation_quality|response_time|conversion_rate|satisfaction)$")
    account_id: Optional[str] = None
    period_days: int = Field(default=30, ge=7, le=365)
    include_forecast: bool = True

class CustomQueryRequest(BaseModel):
    query_name: str = Field(..., max_length=200)
    query_type: str = Field(..., regex="^(filter|aggregation|comparison|forecast)$")
    filters: Dict[str, Any] = {}
    aggregations: Dict[str, Any] = {}
    grouping: List[str] = []
    time_range: Dict[str, str]
    description: Optional[str] = None
    tags: List[str] = []

@router.get("/dashboard", summary="Get real-time dashboard metrics")
async def get_dashboard_metrics(
    account_id: Optional[str] = Query(None, description="Account ID filter"),
    time_range_hours: int = Query(24, ge=1, le=8760, description="Time range in hours"),
    include_trends: bool = Query(True, description="Include trend indicators"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get comprehensive real-time dashboard metrics for Kelly AI system.
    
    Provides overview of conversations, AI performance, quality metrics,
    business impact, and trend indicators.
    """
    try:
        # Validate access permissions
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        # Get dashboard metrics
        metrics = await kelly_analytics_engine.get_real_time_dashboard_metrics(
            account_id=account_id,
            time_range_hours=time_range_hours
        )
        
        # Add user context
        metrics['user_context'] = {
            'user_id': current_user['user_id'],
            'account_access': account_id or 'global',
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': metrics,
                'metadata': {
                    'time_range_hours': time_range_hours,
                    'account_filter': account_id,
                    'cache_status': 'hit' if 'cached' in str(metrics) else 'miss'
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard metrics: {str(e)}")

@router.post("/conversation-quality", summary="Calculate conversation quality score")
async def calculate_conversation_quality(
    request: ConversationQualityRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate comprehensive quality score for a conversation using AI analysis.
    
    Analyzes conversation content for relevance, helpfulness, accuracy,
    empathy, professionalism, and efficiency.
    """
    try:
        # Validate conversation data
        if not request.conversation_data.get('messages'):
            raise HTTPException(status_code=400, detail="Conversation data must include messages")
        
        # Calculate quality metrics
        quality_metrics = await kelly_analytics_engine.calculate_conversation_quality_score(
            conversation_id=request.conversation_id,
            conversation_data=request.conversation_data,
            ai_confidence=request.ai_confidence
        )
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'conversation_id': request.conversation_id,
                    'overall_score': quality_metrics.overall_score,
                    'grade': quality_metrics.grade,
                    'component_scores': quality_metrics.component_scores,
                    'areas_for_improvement': quality_metrics.areas_for_improvement,
                    'strengths': quality_metrics.strengths,
                    'ai_confidence': request.ai_confidence,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                },
                'metadata': {
                    'scoring_method': 'ai_analysis',
                    'model_used': 'claude_ai',
                    'stored': request.store_results
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating conversation quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate quality score: {str(e)}")

@router.get("/performance", summary="Get performance metrics")
async def get_performance_metrics(
    account_id: str = Query(..., description="Account ID to analyze"),
    period_type: str = Query("day", regex="^(hour|day|week|month)$", description="Analysis period"),
    include_ai_comparison: bool = Query(True, description="Include AI vs human comparison"),
    include_cost_analysis: bool = Query(True, description="Include cost metrics"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get comprehensive performance metrics for team and AI performance.
    
    Includes response times, quality metrics, business impact,
    and AI vs human performance comparison.
    """
    try:
        # Validate access
        if not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        # Calculate performance metrics
        metrics = await kelly_analytics_engine.calculate_performance_metrics(
            account_id=account_id,
            period_type=period_type,
            include_ai_comparison=include_ai_comparison
        )
        
        # Add cost analysis if requested
        if include_cost_analysis and 'cost_metrics' not in metrics:
            # Cost analysis is included by default in calculate_performance_metrics
            pass
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': metrics,
                'metadata': {
                    'account_id': account_id,
                    'period_type': period_type,
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'includes_ai_comparison': include_ai_comparison,
                    'includes_cost_analysis': include_cost_analysis
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/revenue-attribution", summary="Calculate revenue attribution")
async def calculate_revenue_attribution(
    request: RevenueAttributionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Calculate revenue attribution for conversations that led to deals.
    
    Tracks how conversations contribute to business outcomes and
    calculates AI vs human contribution to revenue generation.
    """
    try:
        # Validate deal data
        required_fields = ['amount', 'currency']
        missing_fields = [field for field in required_fields if field not in request.deal_data]
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required deal data fields: {missing_fields}"
            )
        
        # Calculate attribution
        attribution_result = await kelly_analytics_engine.calculate_revenue_attribution(
            conversation_id=request.conversation_id,
            deal_data=request.deal_data,
            attribution_confidence=request.attribution_confidence
        )
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': attribution_result,
                'metadata': {
                    'conversation_id': request.conversation_id,
                    'attribution_confidence': request.attribution_confidence,
                    'calculated_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating revenue attribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate attribution: {str(e)}")

@router.get("/trends", summary="Get trend analysis")
async def get_trend_analysis(
    metric_name: str = Query(..., regex="^(conversation_quality|response_time|conversion_rate|satisfaction)$"),
    account_id: Optional[str] = Query(None, description="Account ID filter"),
    period_days: int = Query(30, ge=7, le=365, description="Analysis period in days"),
    include_forecast: bool = Query(True, description="Include forecasting"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate trend analysis for specific metrics with forecasting.
    
    Analyzes historical performance data to identify trends,
    patterns, and provides forecasting for future periods.
    """
    try:
        # Validate access
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        # Generate trend analysis
        trend_analysis = await kelly_analytics_engine.generate_trend_analysis(
            metric_name=metric_name,
            account_id=account_id,
            period_days=period_days
        )
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': trend_analysis,
                'metadata': {
                    'metric_name': metric_name,
                    'account_id': account_id,
                    'period_days': period_days,
                    'includes_forecast': include_forecast,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating trend analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trend analysis: {str(e)}")

@router.post("/custom-query", summary="Execute custom analytics query")
async def execute_custom_query(
    request: CustomQueryRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Execute custom analytics query with flexible filtering and aggregation.
    
    Allows users to create custom analytics queries with specific filters,
    aggregations, and grouping options. Results are cached for performance.
    """
    try:
        # Validate time range
        try:
            start_time = datetime.fromisoformat(request.time_range['start'])
            end_time = datetime.fromisoformat(request.time_range['end'])
            if end_time <= start_time:
                raise HTTPException(status_code=400, detail="End time must be after start time")
        except (KeyError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid time range format")
        
        # Validate account access if account_id in filters
        if request.filters.get('account_id'):
            if not await _has_account_access(current_user, request.filters['account_id']):
                raise HTTPException(status_code=403, detail="Access denied to account")
        
        # Execute custom query
        query_config = {
            'name': request.query_name,
            'type': request.query_type,
            'filters': request.filters,
            'aggregations': request.aggregations,
            'grouping': request.grouping,
            'time_range': request.time_range,
            'description': request.description,
            'tags': request.tags
        }
        
        query_result = await kelly_analytics_engine.execute_custom_analytics_query(
            query_config=query_config,
            user_id=current_user['user_id']
        )
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': query_result['results'],
                'metadata': {
                    'query_name': request.query_name,
                    'query_type': request.query_type,
                    'execution_time_ms': query_result['metadata']['execution_time_ms'],
                    'rows_processed': query_result['metadata']['rows_processed'],
                    'query_id': query_result['metadata']['query_id'],
                    'cached': query_result['metadata'].get('cached', False),
                    'executed_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error executing custom query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute custom query: {str(e)}")

@router.get("/conversation/{conversation_id}/analytics", summary="Get conversation analytics")
async def get_conversation_analytics(
    conversation_id: str,
    include_quality_breakdown: bool = Query(True, description="Include quality score breakdown"),
    include_insights: bool = Query(True, description="Include AI insights"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get detailed analytics for a specific conversation.
    
    Provides comprehensive analysis including quality scores,
    AI insights, performance metrics, and business impact.
    """
    try:
        async with get_session() as session:
            # Get conversation analytics
            conversation = session.query(ConversationAnalytics).filter(
                ConversationAnalytics.conversation_id == conversation_id
            ).first()
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Check access permissions
            if not await _has_account_access(current_user, conversation.account_id):
                raise HTTPException(status_code=403, detail="Access denied to conversation")
            
            # Build response data
            analytics_data = {
                'conversation_id': conversation_id,
                'basic_metrics': {
                    'total_messages': conversation.total_messages,
                    'ai_messages': conversation.ai_messages,
                    'human_messages': conversation.human_messages,
                    'duration_minutes': conversation.duration_minutes,
                    'started_at': conversation.started_at.isoformat(),
                    'ended_at': conversation.ended_at.isoformat() if conversation.ended_at else None
                },
                'quality_metrics': {
                    'overall_score': conversation.conversation_quality_score,
                    'ai_confidence_avg': conversation.ai_confidence_avg,
                    'safety_score_avg': conversation.safety_score_avg
                },
                'business_impact': {
                    'lead_qualified': conversation.lead_qualified,
                    'conversion_event': conversation.conversion_event,
                    'revenue_attributed': conversation.revenue_attributed,
                    'resolution_achieved': conversation.resolution_achieved
                },
                'ai_performance': {
                    'human_intervention_count': conversation.human_intervention_count,
                    'escalation_required': conversation.escalation_required,
                    'anomaly_detected': conversation.anomaly_detected
                }
            }
            
            # Add quality breakdown if requested
            if include_quality_breakdown:
                from app.models.kelly_analytics import QualityScore
                quality_scores = session.query(QualityScore).filter(
                    QualityScore.conversation_id == conversation_id
                ).order_by(QualityScore.scored_at.desc()).all()
                
                if quality_scores:
                    latest_score = quality_scores[0]
                    analytics_data['quality_breakdown'] = {
                        'relevance_score': latest_score.relevance_score,
                        'helpfulness_score': latest_score.helpfulness_score,
                        'accuracy_score': latest_score.accuracy_score,
                        'empathy_score': latest_score.empathy_score,
                        'professionalism_score': latest_score.professionalism_score,
                        'efficiency_score': latest_score.efficiency_score,
                        'scored_at': latest_score.scored_at.isoformat(),
                        'scoring_method': latest_score.scoring_method
                    }
            
            # Add AI insights if requested
            if include_insights:
                from app.models.kelly_intelligence import ConversationInsight
                insights = session.query(ConversationInsight).filter(
                    ConversationInsight.conversation_id == conversation_id
                ).order_by(ConversationInsight.generated_at.desc()).limit(5).all()
                
                analytics_data['ai_insights'] = [
                    {
                        'insight_type': insight.insight_type,
                        'title': insight.title,
                        'description': insight.description,
                        'priority': insight.priority,
                        'confidence_score': insight.confidence_score,
                        'generated_at': insight.generated_at.isoformat()
                    }
                    for insight in insights
                ]
            
            return JSONResponse(
                status_code=200,
                content={
                    'success': True,
                    'data': analytics_data,
                    'metadata': {
                        'conversation_id': conversation_id,
                        'account_id': conversation.account_id,
                        'includes_quality_breakdown': include_quality_breakdown,
                        'includes_insights': include_insights,
                        'retrieved_at': datetime.utcnow().isoformat()
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation analytics: {str(e)}")

@router.get("/export/{format}", summary="Export analytics data")
async def export_analytics_data(
    format: str,
    query_id: Optional[str] = Query(None, description="Custom query ID to export"),
    account_id: Optional[str] = Query(None, description="Account ID filter"),
    time_range_days: int = Query(30, ge=1, le=365, description="Time range in days"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Export analytics data in various formats (PDF, CSV, JSON).
    
    Supports exporting dashboard metrics, custom query results,
    or filtered conversation analytics data.
    """
    try:
        if format not in ['pdf', 'csv', 'json']:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        # Validate access
        if account_id and not await _has_account_access(current_user, account_id):
            raise HTTPException(status_code=403, detail="Access denied to account")
        
        # Get data to export
        if query_id:
            # Export specific query results
            from app.models.kelly_analytics import CustomAnalyticsQuery
            async with get_session() as session:
                query = session.query(CustomAnalyticsQuery).filter(
                    CustomAnalyticsQuery.id == query_id
                ).first()
                
                if not query:
                    raise HTTPException(status_code=404, detail="Query not found")
                
                if query.created_by != current_user['user_id']:
                    raise HTTPException(status_code=403, detail="Access denied to query")
                
                export_data = query.results
                filename = f"query_{query_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        else:
            # Export dashboard metrics
            export_data = await kelly_analytics_engine.get_real_time_dashboard_metrics(
                account_id=account_id,
                time_range_hours=time_range_days * 24
            )
            filename = f"dashboard_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate export based on format
        if format == 'json':
            content = json.dumps(export_data, indent=2, default=str)
            media_type = 'application/json'
            filename += '.json'
        elif format == 'csv':
            # Convert to CSV format (simplified)
            import csv
            import io
            
            output = io.StringIO()
            if isinstance(export_data, dict) and 'overview' in export_data:
                # Dashboard metrics format
                writer = csv.writer(output)
                writer.writerow(['Metric', 'Value'])
                for key, value in export_data['overview'].items():
                    writer.writerow([key, value])
            else:
                # Generic dict format
                writer = csv.writer(output)
                writer.writerow(['Key', 'Value'])
                for key, value in export_data.items():
                    writer.writerow([key, str(value)])
            
            content = output.getvalue()
            media_type = 'text/csv'
            filename += '.csv'
        else:  # PDF format
            # For PDF generation, you would typically use a library like ReportLab
            # For now, return JSON format as placeholder
            content = json.dumps(export_data, indent=2, default=str)
            media_type = 'application/pdf'
            filename += '.pdf'
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'download_url': f"/api/v1/analytics/download/{filename}",
                    'format': format,
                    'size_bytes': len(content.encode('utf-8')),
                    'generated_at': datetime.utcnow().isoformat()
                },
                'metadata': {
                    'filename': filename,
                    'format': format,
                    'account_id': account_id,
                    'time_range_days': time_range_days,
                    'query_id': query_id
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting analytics data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")

@router.get("/reports/scheduled", summary="Get scheduled reports")
async def get_scheduled_reports(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get list of scheduled analytics reports for the user.
    
    Returns all scheduled reports with their configuration,
    next run times, and recent execution history.
    """
    try:
        # This would integrate with a scheduling system
        # For now, return placeholder data
        scheduled_reports = [
            {
                'id': 'weekly_dashboard',
                'name': 'Weekly Dashboard Report',
                'type': 'dashboard',
                'frequency': 'weekly',
                'next_run': (datetime.utcnow() + timedelta(days=2)).isoformat(),
                'last_run': (datetime.utcnow() - timedelta(days=5)).isoformat(),
                'recipients': [current_user['email']],
                'format': 'pdf',
                'active': True
            },
            {
                'id': 'monthly_performance',
                'name': 'Monthly Performance Analysis',
                'type': 'performance',
                'frequency': 'monthly',
                'next_run': (datetime.utcnow() + timedelta(days=15)).isoformat(),
                'last_run': (datetime.utcnow() - timedelta(days=15)).isoformat(),
                'recipients': [current_user['email']],
                'format': 'pdf',
                'active': True
            }
        ]
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': {
                    'scheduled_reports': scheduled_reports,
                    'total_count': len(scheduled_reports),
                    'active_count': len([r for r in scheduled_reports if r['active']])
                },
                'metadata': {
                    'user_id': current_user['user_id'],
                    'retrieved_at': datetime.utcnow().isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting scheduled reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled reports: {str(e)}")

@router.post("/reports/schedule", summary="Schedule analytics report")
async def schedule_analytics_report(
    report_config: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Schedule automated analytics report generation and delivery.
    
    Allows users to schedule regular reports with custom configurations,
    delivery settings, and recipient lists.
    """
    try:
        # Validate report configuration
        required_fields = ['name', 'type', 'frequency', 'recipients', 'format']
        missing_fields = [field for field in required_fields if field not in report_config]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Validate frequency
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
        if report_config['frequency'] not in valid_frequencies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid frequency. Must be one of: {valid_frequencies}"
            )
        
        # Validate format
        valid_formats = ['pdf', 'csv', 'json']
        if report_config['format'] not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {valid_formats}"
            )
        
        # Create scheduled report (this would integrate with a job scheduler)
        report_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate next run time based on frequency
        now = datetime.utcnow()
        if report_config['frequency'] == 'daily':
            next_run = now.replace(hour=8, minute=0, second=0) + timedelta(days=1)
        elif report_config['frequency'] == 'weekly':
            next_run = now.replace(hour=8, minute=0, second=0) + timedelta(days=7)
        elif report_config['frequency'] == 'monthly':
            next_run = now.replace(day=1, hour=8, minute=0, second=0) + timedelta(days=32)
            next_run = next_run.replace(day=1)
        else:  # quarterly
            next_run = now.replace(day=1, hour=8, minute=0, second=0) + timedelta(days=93)
        
        scheduled_report = {
            'id': report_id,
            'name': report_config['name'],
            'type': report_config['type'],
            'frequency': report_config['frequency'],
            'recipients': report_config['recipients'],
            'format': report_config['format'],
            'config': report_config,
            'created_by': current_user['user_id'],
            'created_at': now.isoformat(),
            'next_run': next_run.isoformat(),
            'active': True
        }
        
        # Store in database (placeholder - would use actual scheduling system)
        # await schedule_service.create_report(scheduled_report)
        
        return JSONResponse(
            status_code=201,
            content={
                'success': True,
                'data': {
                    'report_id': report_id,
                    'name': report_config['name'],
                    'frequency': report_config['frequency'],
                    'next_run': next_run.isoformat(),
                    'active': True
                },
                'metadata': {
                    'created_by': current_user['user_id'],
                    'created_at': now.isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling analytics report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule report: {str(e)}")

# Helper functions

async def _has_account_access(user: Dict, account_id: str) -> bool:
    """Check if user has access to specific account"""
    # This would implement actual access control logic
    # For now, return True for all authenticated users
    return True

async def _validate_query_permissions(user: Dict, query_config: Dict) -> bool:
    """Validate user permissions for query configuration"""
    # This would implement query-level access control
    # For now, return True for all authenticated users
    return True