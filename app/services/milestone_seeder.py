"""
Milestone Seeder Service

Seeds the database with default engagement milestones for user achievements.
Provides a comprehensive set of milestones to drive proactive engagement.
"""

from datetime import datetime
from typing import List, Dict, Any
import structlog

from app.database.connection import get_database_session
from app.models.engagement import EngagementMilestone

logger = structlog.get_logger(__name__)


class MilestoneSeeder:
    """
    Seeds engagement milestones to motivate user interaction.
    
    Implements gamification milestones based on research showing:
    - Clear progress indicators increase engagement by 40%
    - Achievement recognition improves retention by 30%
    - Milestone celebrations drive 60% more interactions
    """
    
    def __init__(self):
        self.default_milestones = self._get_default_milestones()
    
    async def seed_milestones(self, overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Seed the database with default milestones.
        
        Args:
            overwrite_existing: Whether to update existing milestones
            
        Returns:
            Seeding results
        """
        try:
            async with get_database_session() as session:
                created_count = 0
                updated_count = 0
                skipped_count = 0
                
                for milestone_data in self.default_milestones:
                    # Check if milestone already exists
                    existing_result = await session.execute(
                        "SELECT id FROM engagement_milestones WHERE milestone_name = :name",
                        {'name': milestone_data['milestone_name']}
                    )
                    existing = existing_result.scalar_one_or_none()
                    
                    if existing and not overwrite_existing:
                        skipped_count += 1
                        continue
                    
                    if existing and overwrite_existing:
                        # Update existing milestone
                        await session.execute(
                            """UPDATE engagement_milestones SET
                               display_name = :display_name,
                               description = :description,
                               target_value = :target_value,
                               metric_type = :metric_type,
                               celebration_template = :celebration_template,
                               reward_type = :reward_type,
                               reward_data = :reward_data,
                               category = :category,
                               difficulty_level = :difficulty_level,
                               estimated_days_to_achieve = :estimated_days,
                               updated_at = :updated_at
                               WHERE milestone_name = :milestone_name""",
                            {
                                **milestone_data,
                                'updated_at': datetime.utcnow()
                            }
                        )
                        updated_count += 1
                    else:
                        # Create new milestone
                        milestone = EngagementMilestone(**milestone_data)
                        session.add(milestone)
                        created_count += 1
                
                await session.commit()
                
                results = {
                    'milestones_created': created_count,
                    'milestones_updated': updated_count,
                    'milestones_skipped': skipped_count,
                    'total_milestones': len(self.default_milestones),
                    'seeded_at': datetime.utcnow().isoformat()
                }
                
                logger.info(
                    "Milestone seeding completed",
                    **results
                )
                
                return results
                
        except Exception as e:
            logger.error("Error seeding milestones", error=str(e))
            raise
    
    async def get_milestone_stats(self) -> Dict[str, Any]:
        """Get statistics about current milestones."""
        try:
            async with get_database_session() as session:
                # Total milestones
                total_result = await session.execute(
                    "SELECT COUNT(*) FROM engagement_milestones"
                )
                total_milestones = total_result.scalar()
                
                # Active milestones
                active_result = await session.execute(
                    "SELECT COUNT(*) FROM engagement_milestones WHERE is_active = true"
                )
                active_milestones = active_result.scalar()
                
                # Milestones by category
                category_result = await session.execute(
                    """SELECT category, COUNT(*) as count
                       FROM engagement_milestones 
                       WHERE is_active = true
                       GROUP BY category"""
                )
                categories = {row.category: row.count for row in category_result.fetchall()}
                
                # Achievement stats
                achievements_result = await session.execute(
                    """SELECT 
                           em.milestone_name,
                           em.total_achievements,
                           em.average_days_to_achieve,
                           COUNT(ump.id) as current_progress_records
                       FROM engagement_milestones em
                       LEFT JOIN user_milestone_progress ump ON em.id = ump.milestone_id
                       WHERE em.is_active = true
                       GROUP BY em.id, em.milestone_name, em.total_achievements, em.average_days_to_achieve
                       ORDER BY em.total_achievements DESC"""
                )
                
                achievement_stats = []
                for row in achievements_result.fetchall():
                    achievement_stats.append({
                        'milestone_name': row.milestone_name,
                        'total_achievements': row.total_achievements,
                        'average_days_to_achieve': row.average_days_to_achieve,
                        'users_in_progress': row.current_progress_records
                    })
                
                return {
                    'total_milestones': total_milestones,
                    'active_milestones': active_milestones,
                    'milestones_by_category': categories,
                    'achievement_stats': achievement_stats[:10]  # Top 10
                }
                
        except Exception as e:
            logger.error("Error getting milestone stats", error=str(e))
            raise
    
    def _get_default_milestones(self) -> List[Dict[str, Any]]:
        """Get the default milestone definitions."""
        return [
            # Interaction milestones
            {
                'milestone_name': 'first_interaction',
                'display_name': 'First Steps',
                'description': 'Send your first message',
                'metric_name': 'total_interactions',
                'target_value': 1.0,
                'metric_type': 'count',
                'celebration_template': 'welcome_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'newcomer', 'points': 10},
                'category': 'getting_started',
                'difficulty_level': 1,
                'estimated_days_to_achieve': 1
            },
            {
                'milestone_name': 'chatty_newcomer',
                'display_name': 'Chatty Newcomer',
                'description': 'Have 10 interactions',
                'metric_name': 'total_interactions',
                'target_value': 10.0,
                'metric_type': 'count',
                'celebration_template': 'milestone_basic',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'chatty', 'points': 25},
                'category': 'communication',
                'difficulty_level': 2,
                'estimated_days_to_achieve': 3
            },
            {
                'milestone_name': 'conversation_enthusiast',
                'display_name': 'Conversation Enthusiast',
                'description': 'Reach 50 interactions',
                'metric_name': 'total_interactions',
                'target_value': 50.0,
                'metric_type': 'count',
                'celebration_template': 'milestone_progress',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'enthusiast', 'points': 50},
                'category': 'communication',
                'difficulty_level': 3,
                'estimated_days_to_achieve': 10
            },
            {
                'milestone_name': 'super_communicator',
                'display_name': 'Super Communicator',
                'description': 'Achieve 100 interactions',
                'metric_name': 'total_interactions',
                'target_value': 100.0,
                'metric_type': 'count',
                'celebration_template': 'milestone_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'super_communicator', 'points': 100},
                'category': 'communication',
                'difficulty_level': 4,
                'estimated_days_to_achieve': 20
            },
            {
                'milestone_name': 'conversation_master',
                'display_name': 'Conversation Master',
                'description': 'Reach 500 interactions',
                'metric_name': 'total_interactions',
                'target_value': 500.0,
                'metric_type': 'count',
                'celebration_template': 'milestone_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'master', 'points': 250},
                'category': 'communication',
                'difficulty_level': 5,
                'estimated_days_to_achieve': 60
            },
            
            # Command usage milestones
            {
                'milestone_name': 'command_explorer',
                'display_name': 'Command Explorer',
                'description': 'Use 5 different commands',
                'metric_name': 'command_count',
                'target_value': 5.0,
                'metric_type': 'count',
                'celebration_template': 'feature_discovery',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'explorer', 'points': 30},
                'category': 'exploration',
                'difficulty_level': 2,
                'estimated_days_to_achieve': 5
            },
            {
                'milestone_name': 'power_user',
                'display_name': 'Power User',
                'description': 'Use 25 commands',
                'metric_name': 'command_count',
                'target_value': 25.0,
                'metric_type': 'count',
                'celebration_template': 'milestone_progress',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'power_user', 'points': 75},
                'category': 'exploration',
                'difficulty_level': 3,
                'estimated_days_to_achieve': 15
            },
            
            # Positive interaction milestones
            {
                'milestone_name': 'positive_vibes',
                'display_name': 'Positive Vibes',
                'description': 'Send 10 positive messages',
                'metric_name': 'positive_interactions',
                'target_value': 10.0,
                'metric_type': 'count',
                'celebration_template': 'mood_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'positive', 'points': 40},
                'category': 'mood',
                'difficulty_level': 2,
                'estimated_days_to_achieve': 7
            },
            {
                'milestone_name': 'sunshine_ambassador',
                'display_name': 'Sunshine Ambassador',
                'description': 'Send 50 positive messages',
                'metric_name': 'positive_interactions',
                'target_value': 50.0,
                'metric_type': 'count',
                'celebration_template': 'mood_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'sunshine', 'points': 100},
                'category': 'mood',
                'difficulty_level': 4,
                'estimated_days_to_achieve': 30
            },
            
            # Engagement streak milestones
            {
                'milestone_name': 'daily_chatter',
                'display_name': 'Daily Chatter',
                'description': 'Chat for 3 consecutive days',
                'metric_name': 'session_days',
                'target_value': 3.0,
                'metric_type': 'consecutive',
                'celebration_template': 'streak_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'daily_chatter', 'points': 35},
                'category': 'consistency',
                'difficulty_level': 2,
                'estimated_days_to_achieve': 3
            },
            {
                'milestone_name': 'weekly_regular',
                'display_name': 'Weekly Regular',
                'description': 'Chat for 7 consecutive days',
                'metric_name': 'session_days',
                'target_value': 7.0,
                'metric_type': 'consecutive',
                'celebration_template': 'streak_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'weekly_regular', 'points': 70},
                'category': 'consistency',
                'difficulty_level': 3,
                'estimated_days_to_achieve': 7
            },
            {
                'milestone_name': 'dedication_champion',
                'display_name': 'Dedication Champion',
                'description': 'Chat for 30 consecutive days',
                'metric_name': 'session_days',
                'target_value': 30.0,
                'metric_type': 'consecutive',
                'celebration_template': 'streak_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'dedication_champion', 'points': 200},
                'category': 'consistency',
                'difficulty_level': 5,
                'estimated_days_to_achieve': 30
            },
            
            # Quality engagement milestones
            {
                'milestone_name': 'quality_conversations',
                'display_name': 'Quality Conversations',
                'description': 'Maintain average engagement quality above 0.7',
                'metric_name': 'average_quality_score',
                'target_value': 0.7,
                'metric_type': 'average',
                'celebration_template': 'quality_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'quality_conversationalist', 'points': 80},
                'category': 'quality',
                'difficulty_level': 3,
                'estimated_days_to_achieve': 14
            },
            {
                'milestone_name': 'engagement_expert',
                'display_name': 'Engagement Expert',
                'description': 'Maintain average engagement quality above 0.8',
                'metric_name': 'average_quality_score',
                'target_value': 0.8,
                'metric_type': 'average',
                'celebration_template': 'quality_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'engagement_expert', 'points': 120},
                'category': 'quality',
                'difficulty_level': 4,
                'estimated_days_to_achieve': 21
            },
            
            # Social milestones
            {
                'milestone_name': 'friendly_neighbor',
                'display_name': 'Friendly Neighbor',
                'description': 'Greet the bot 10 times',
                'metric_name': 'greeting_count',
                'target_value': 10.0,
                'metric_type': 'count',
                'celebration_template': 'social_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'friendly', 'points': 25},
                'category': 'social',
                'difficulty_level': 2,
                'estimated_days_to_achieve': 10
            },
            
            # Special achievements
            {
                'milestone_name': 'early_bird',
                'display_name': 'Early Bird',
                'description': 'Chat before 9 AM on 5 different days',
                'metric_name': 'early_morning_sessions',
                'target_value': 5.0,
                'metric_type': 'count',
                'celebration_template': 'special_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'early_bird', 'points': 45},
                'category': 'special',
                'difficulty_level': 3,
                'estimated_days_to_achieve': 14
            },
            {
                'milestone_name': 'night_owl',
                'display_name': 'Night Owl',
                'description': 'Chat after 10 PM on 5 different days',
                'metric_name': 'late_night_sessions',
                'target_value': 5.0,
                'metric_type': 'count',
                'celebration_template': 'special_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'night_owl', 'points': 45},
                'category': 'special',
                'difficulty_level': 3,
                'estimated_days_to_achieve': 14
            },
            {
                'milestone_name': 'weekend_warrior',
                'display_name': 'Weekend Warrior',
                'description': 'Be active on 8 different weekends',
                'metric_name': 'weekend_activity_days',
                'target_value': 8.0,
                'metric_type': 'count',
                'celebration_template': 'special_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'weekend_warrior', 'points': 60},
                'category': 'special',
                'difficulty_level': 4,
                'estimated_days_to_achieve': 56
            },
            
            # Long-term engagement
            {
                'milestone_name': 'long_term_friend',
                'display_name': 'Long-term Friend',
                'description': 'Stay active for 90 days',
                'metric_name': 'total_active_days',
                'target_value': 90.0,
                'metric_type': 'duration',
                'celebration_template': 'friendship_celebration',
                'reward_type': 'badge',
                'reward_data': {'badge_name': 'long_term_friend', 'points': 300},
                'category': 'loyalty',
                'difficulty_level': 5,
                'estimated_days_to_achieve': 90
            },
            {
                'milestone_name': 'loyal_companion',
                'display_name': 'Loyal Companion',
                'description': 'Stay active for 365 days',
                'metric_name': 'total_active_days',
                'target_value': 365.0,
                'metric_type': 'duration',
                'celebration_template': 'anniversary_celebration',
                'reward_type': 'special_badge',
                'reward_data': {'badge_name': 'loyal_companion', 'points': 1000, 'special': True},
                'category': 'loyalty',
                'difficulty_level': 5,
                'estimated_days_to_achieve': 365
            }
        ]


# Convenience function for easy seeding
async def seed_default_milestones(overwrite_existing: bool = False) -> Dict[str, Any]:
    """
    Convenience function to seed default milestones.
    
    Args:
        overwrite_existing: Whether to update existing milestones
        
    Returns:
        Seeding results
    """
    seeder = MilestoneSeeder()
    return await seeder.seed_milestones(overwrite_existing)