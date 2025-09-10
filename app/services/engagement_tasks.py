"""
Engagement Tasks

Celery background tasks for proactive engagement processing.
Implements asynchronous pattern analysis, outreach scheduling, and campaign management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from celery import Celery
from celery.schedules import crontab
import structlog

from app.config.settings import get_settings
from app.database.connection import get_database_session
from app.models.user import User
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, ProactiveOutreach,
    EngagementMilestone, UserMilestoneProgress, OutreachStatus, OutreachType
)
from app.services.engagement_analyzer import EngagementAnalyzer
from app.services.behavioral_predictor import BehavioralPredictor
from app.services.proactive_outreach import ProactiveOutreachService

# Get settings
settings = get_settings()

# Configure Celery app
celery_app = Celery(
    'engagement_tasks',
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=['app.services.engagement_tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    # Pattern analysis every 30 minutes
    'analyze-user-patterns': {
        'task': 'app.services.engagement_tasks.analyze_user_patterns_batch',
        'schedule': crontab(minute='*/30'),
    },
    
    # Process outreaches every 5 minutes
    'process-scheduled-outreaches': {
        'task': 'app.services.engagement_tasks.process_scheduled_outreaches',
        'schedule': crontab(minute='*/5'),
    },
    
    # Find users needing engagement every hour
    'find-engagement-candidates': {
        'task': 'app.services.engagement_tasks.find_and_schedule_engagement',
        'schedule': crontab(minute=0),
    },
    
    # Update milestone progress daily
    'update-milestone-progress': {
        'task': 'app.services.engagement_tasks.update_milestone_progress_batch',
        'schedule': crontab(hour=2, minute=0),
    },
    
    # Train ML models weekly
    'train-behavioral-models': {
        'task': 'app.services.engagement_tasks.train_behavioral_models',
        'schedule': crontab(hour=3, minute=0, day_of_week=1),  # Monday at 3 AM
    },
    
    # Cleanup old data monthly
    'cleanup-engagement-data': {
        'task': 'app.services.engagement_tasks.cleanup_old_engagement_data',
        'schedule': crontab(hour=4, minute=0, day_of_month=1),
    },
}

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True, max_retries=3)
def analyze_user_patterns_single(self, user_id: str, telegram_id: int):
    """
    Analyze behavioral patterns for a single user.
    
    Args:
        user_id: User UUID string
        telegram_id: Telegram user ID
    """
    try:
        analyzer = EngagementAnalyzer()
        
        # Run async pattern analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            pattern = loop.run_until_complete(
                analyzer.update_user_behavior_patterns(user_id, telegram_id)
            )
            
            result = {
                'user_id': user_id,
                'telegram_id': telegram_id,
                'pattern_updated': pattern is not None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if pattern:
                result.update({
                    'churn_risk_score': pattern.churn_risk_score,
                    'needs_re_engagement': pattern.needs_re_engagement,
                    'total_interactions': pattern.total_interactions
                })
            
            logger.info("User pattern analysis completed", **result)
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(
            "Error analyzing user patterns",
            user_id=user_id,
            error=str(e),
            retry_count=self.request.retries
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'user_id': user_id,
            'error': str(e),
            'failed': True
        }


@celery_app.task(bind=True, max_retries=2)
def analyze_user_patterns_batch(self, batch_size: int = 100):
    """
    Batch analyze user patterns for users with recent activity.
    
    Args:
        batch_size: Number of users to process in this batch
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                _analyze_patterns_batch_async(batch_size)
            )
            
            logger.info(
                "Batch pattern analysis completed",
                users_processed=len(results),
                successful=len([r for r in results if not r.get('failed')]),
                failed=len([r for r in results if r.get('failed')])
            )
            
            return {
                'batch_size': batch_size,
                'users_processed': len(results),
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error in batch pattern analysis", error=str(e))
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=300)  # 5 minute retry
        
        return {'error': str(e), 'failed': True}


@celery_app.task(bind=True, max_retries=3)
def process_scheduled_outreaches(self, limit: int = 50):
    """
    Process scheduled outreaches that are ready to send.
    
    Args:
        limit: Maximum number of outreaches to process
    """
    try:
        outreach_service = ProactiveOutreachService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                outreach_service.process_scheduled_outreaches(limit)
            )
            
            logger.info(
                "Scheduled outreaches processed",
                total_processed=len(results),
                successful=len([r for r in results if r['status'] == 'sent']),
                failed=len([r for r in results if r['status'] == 'failed'])
            )
            
            return {
                'outreaches_processed': len(results),
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error processing scheduled outreaches", error=str(e))
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60)
        
        return {'error': str(e), 'failed': True}


@celery_app.task(bind=True, max_retries=2)
def find_and_schedule_engagement(self, max_users: int = 100):
    """
    Find users needing engagement and schedule appropriate outreaches.
    
    Args:
        max_users: Maximum number of users to process
    """
    try:
        analyzer = EngagementAnalyzer()
        outreach_service = ProactiveOutreachService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                _find_and_schedule_engagement_async(analyzer, outreach_service, max_users)
            )
            
            logger.info(
                "Engagement scheduling completed",
                users_analyzed=results['users_analyzed'],
                outreaches_scheduled=results['outreaches_scheduled']
            )
            
            return results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error finding and scheduling engagement", error=str(e))
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=300)
        
        return {'error': str(e), 'failed': True}


@celery_app.task(bind=True, max_retries=2)
def update_milestone_progress_batch(self, batch_size: int = 200):
    """
    Update milestone progress for all active users.
    
    Args:
        batch_size: Number of users to process in each batch
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                _update_milestone_progress_batch_async(batch_size)
            )
            
            logger.info(
                "Milestone progress update completed",
                users_processed=results['users_processed'],
                milestones_achieved=results['milestones_achieved']
            )
            
            return results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error updating milestone progress", error=str(e))
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=600)  # 10 minute retry
        
        return {'error': str(e), 'failed': True}


@celery_app.task(bind=True, max_retries=1)
def train_behavioral_models(self, min_samples: int = 500):
    """
    Train or retrain behavioral prediction models.
    
    Args:
        min_samples: Minimum samples required for training
    """
    try:
        predictor = BehavioralPredictor()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            training_results = loop.run_until_complete(
                predictor.train_models(min_samples)
            )
            
            logger.info(
                "Model training completed",
                models_trained=training_results.get('models_trained', []),
                training_samples=training_results.get('training_samples', 0)
            )
            
            return training_results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error training behavioral models", error=str(e))
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=1800)  # 30 minute retry
        
        return {'error': str(e), 'failed': True}


@celery_app.task(bind=True, max_retries=1)
def schedule_milestone_celebration(self, user_id: str, milestone_id: str, achievement_data: Dict[str, Any]):
    """
    Schedule a milestone celebration outreach.
    
    Args:
        user_id: User UUID string
        milestone_id: Milestone UUID string
        achievement_data: Data about the achievement
    """
    try:
        outreach_service = ProactiveOutreachService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            outreach = loop.run_until_complete(
                outreach_service.create_milestone_celebration(
                    user_id, milestone_id, achievement_data
                )
            )
            
            result = {
                'user_id': user_id,
                'milestone_id': milestone_id,
                'outreach_created': outreach is not None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if outreach:
                result['outreach_id'] = str(outreach.id)
                result['scheduled_for'] = outreach.scheduled_for.isoformat()
            
            logger.info("Milestone celebration scheduled", **result)
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(
            "Error scheduling milestone celebration",
            user_id=user_id,
            milestone_id=milestone_id,
            error=str(e)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60)
        
        return {'error': str(e), 'failed': True}


@celery_app.task(bind=True, max_retries=1)
def cleanup_old_engagement_data(self, days_to_keep: int = 365):
    """
    Clean up old engagement data to maintain performance.
    
    Args:
        days_to_keep: Number of days of engagement data to keep
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                _cleanup_old_data_async(days_to_keep)
            )
            
            logger.info(
                "Engagement data cleanup completed",
                records_deleted=results['records_deleted'],
                days_kept=days_to_keep
            )
            
            return results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error cleaning up engagement data", error=str(e))
        return {'error': str(e), 'failed': True}


# Async helper functions

async def _analyze_patterns_batch_async(batch_size: int) -> List[Dict[str, Any]]:
    """Async helper for batch pattern analysis."""
    try:
        async with get_database_session() as session:
            # Get users with recent activity who need pattern updates
            cutoff_date = datetime.utcnow() - timedelta(hours=24)
            
            # Find users who either:
            # 1. Have recent interactions but no pattern record
            # 2. Have patterns that are outdated
            users_query = """
                SELECT DISTINCT u.id, u.telegram_id 
                FROM users u
                LEFT JOIN user_behavior_patterns ubp ON u.id = ubp.user_id
                WHERE u.is_active = true 
                AND u.is_blocked = false
                AND (
                    (ubp.id IS NULL AND EXISTS (
                        SELECT 1 FROM user_engagements ue 
                        WHERE ue.user_id = u.id 
                        AND ue.interaction_timestamp >= :cutoff_date
                    ))
                    OR 
                    (ubp.last_pattern_analysis IS NULL OR ubp.last_pattern_analysis < :cutoff_date)
                )
                ORDER BY u.updated_at DESC
                LIMIT :batch_size
            """
            
            result = await session.execute(users_query, {
                'cutoff_date': cutoff_date,
                'batch_size': batch_size
            })
            users = result.fetchall()
            
            analyzer = EngagementAnalyzer()
            results = []
            
            for user in users:
                try:
                    pattern = await analyzer.update_user_behavior_patterns(
                        str(user.id), user.telegram_id
                    )
                    
                    results.append({
                        'user_id': str(user.id),
                        'telegram_id': user.telegram_id,
                        'pattern_updated': pattern is not None,
                        'churn_risk_score': pattern.churn_risk_score if pattern else None
                    })
                    
                except Exception as e:
                    logger.warning(
                        "Failed to analyze patterns for user",
                        user_id=str(user.id),
                        error=str(e)
                    )
                    results.append({
                        'user_id': str(user.id),
                        'error': str(e),
                        'failed': True
                    })
            
            return results
            
    except Exception as e:
        logger.error("Error in batch pattern analysis", error=str(e))
        raise


async def _find_and_schedule_engagement_async(
    analyzer: EngagementAnalyzer,
    outreach_service: ProactiveOutreachService,
    max_users: int
) -> Dict[str, Any]:
    """Async helper for finding and scheduling engagement."""
    try:
        # Find users needing engagement
        engagement_candidates = await analyzer.find_users_needing_engagement(max_users)
        
        scheduled_outreaches = []
        
        for candidate in engagement_candidates:
            user_id = candidate['user_id']
            recommendations = candidate['recommendations']
            
            # Schedule outreaches based on recommendations
            for rec in recommendations[:2]:  # Limit to top 2 recommendations per user
                try:
                    outreach_type = OutreachType(rec['type'])
                    priority = rec['priority']
                    
                    # Create context from recommendation
                    context = {
                        'trigger_event': rec['reason'],
                        'recommendation': rec,
                        **candidate['patterns']
                    }
                    
                    outreach = await outreach_service.schedule_proactive_outreach(
                        user_id=user_id,
                        outreach_type=outreach_type,
                        priority_score=priority,
                        context_data=context
                    )
                    
                    if outreach:
                        scheduled_outreaches.append({
                            'user_id': user_id,
                            'outreach_id': str(outreach.id),
                            'type': outreach_type.value,
                            'priority': priority
                        })
                
                except Exception as e:
                    logger.warning(
                        "Failed to schedule outreach",
                        user_id=user_id,
                        recommendation=rec,
                        error=str(e)
                    )
        
        return {
            'users_analyzed': len(engagement_candidates),
            'outreaches_scheduled': len(scheduled_outreaches),
            'scheduled_outreaches': scheduled_outreaches,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error finding and scheduling engagement", error=str(e))
        raise


async def _update_milestone_progress_batch_async(batch_size: int) -> Dict[str, Any]:
    """Async helper for batch milestone progress updates."""
    try:
        async with get_database_session() as session:
            # Get users with recent interactions for milestone updates
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            users_result = await session.execute(
                """
                SELECT DISTINCT u.id, u.telegram_id
                FROM users u
                WHERE u.is_active = true 
                AND EXISTS (
                    SELECT 1 FROM user_engagements ue 
                    WHERE ue.user_id = u.id 
                    AND ue.interaction_timestamp >= :cutoff_date
                )
                ORDER BY u.updated_at DESC
                LIMIT :batch_size
                """,
                {'cutoff_date': cutoff_date, 'batch_size': batch_size}
            )
            users = users_result.fetchall()
            
            analyzer = EngagementAnalyzer()
            outreach_service = ProactiveOutreachService()
            
            users_processed = 0
            milestones_achieved = 0
            celebrations_scheduled = 0
            
            for user in users:
                try:
                    # This would trigger milestone updates in the pattern analysis
                    await analyzer.update_user_behavior_patterns(
                        str(user.id), user.telegram_id
                    )
                    users_processed += 1
                    
                    # Check for newly achieved milestones
                    achieved_milestones = await _check_milestone_achievements(
                        session, str(user.id)
                    )
                    
                    milestones_achieved += len(achieved_milestones)
                    
                    # Schedule celebrations for new achievements
                    for milestone_data in achieved_milestones:
                        try:
                            outreach = await outreach_service.create_milestone_celebration(
                                str(user.id),
                                milestone_data['milestone_id'],
                                milestone_data
                            )
                            if outreach:
                                celebrations_scheduled += 1
                        except Exception as e:
                            logger.warning(
                                "Failed to schedule milestone celebration",
                                user_id=str(user.id),
                                milestone=milestone_data,
                                error=str(e)
                            )
                
                except Exception as e:
                    logger.warning(
                        "Failed to update milestones for user",
                        user_id=str(user.id),
                        error=str(e)
                    )
            
            return {
                'users_processed': users_processed,
                'milestones_achieved': milestones_achieved,
                'celebrations_scheduled': celebrations_scheduled,
                'timestamp': datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error("Error updating milestone progress", error=str(e))
        raise


async def _check_milestone_achievements(session, user_id: str) -> List[Dict[str, Any]]:
    """Check for newly achieved milestones."""
    try:
        # Get milestones achieved in the last 24 hours without celebrations
        recent_achievements = await session.execute(
            """
            SELECT ump.id, ump.milestone_id, em.milestone_name, 
                   em.display_name, em.description, ump.achieved_at
            FROM user_milestone_progress ump
            JOIN engagement_milestones em ON ump.milestone_id = em.id
            WHERE ump.user_id = :user_id
            AND ump.is_achieved = true
            AND ump.celebration_sent = false
            AND ump.achieved_at >= :cutoff_time
            """,
            {
                'user_id': user_id,
                'cutoff_time': datetime.utcnow() - timedelta(hours=24)
            }
        )
        
        achievements = []
        for row in recent_achievements.fetchall():
            achievements.append({
                'progress_id': str(row.id),
                'milestone_id': str(row.milestone_id),
                'name': row.milestone_name,
                'display_name': row.display_name,
                'description': row.description,
                'achieved_at': row.achieved_at.isoformat()
            })
        
        return achievements
        
    except Exception as e:
        logger.error("Error checking milestone achievements", error=str(e), user_id=user_id)
        return []


async def _cleanup_old_data_async(days_to_keep: int) -> Dict[str, Any]:
    """Async helper for cleaning up old engagement data."""
    try:
        async with get_database_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old user engagements (keep some for ML training)
            old_engagements = await session.execute(
                "DELETE FROM user_engagements WHERE created_at < :cutoff_date",
                {'cutoff_date': cutoff_date}
            )
            
            # Clean up completed/failed outreaches older than retention period
            old_outreaches = await session.execute(
                """DELETE FROM proactive_outreaches 
                   WHERE created_at < :cutoff_date 
                   AND status IN ('delivered', 'failed', 'cancelled')""",
                {'cutoff_date': cutoff_date}
            )
            
            await session.commit()
            
            return {
                'records_deleted': {
                    'engagements': old_engagements.rowcount,
                    'outreaches': old_outreaches.rowcount
                },
                'cutoff_date': cutoff_date.isoformat(),
                'days_kept': days_to_keep,
                'timestamp': datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error("Error cleaning up old data", error=str(e))
        raise


# Convenience function to update user patterns from other services
def update_user_patterns_task(user_id: str, telegram_id: int):
    """
    Convenience function to trigger user pattern analysis.
    
    Args:
        user_id: User UUID string
        telegram_id: Telegram user ID
    """
    analyze_user_patterns_single.delay(user_id, telegram_id)