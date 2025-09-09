"""
Database Initialization Script

Complete database setup including table creation, indexes, sample data,
and initial configuration for the AI conversation bot system.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any
import structlog

from sqlalchemy import text
from alembic.config import Config
from alembic import command

from app.config import settings
from app.database.connection import db_manager
from app.database.repositories import (
    user_repo, personality_repo, config_repo, feature_flag_repo,
    audit_repo, security_repo
)
from app.models import (
    User, PersonalityProfile, PersonalityTrait, RiskFactor,
    SystemConfiguration, FeatureFlag
)

logger = structlog.get_logger(__name__)


class DatabaseInitializer:
    """Database initialization and setup."""
    
    def __init__(self):
        self.sample_data_enabled = settings.environment == "development"
    
    async def initialize_database(self, drop_existing: bool = False) -> Dict[str, Any]:
        """Initialize database with tables, indexes, and sample data."""
        
        logger.info("Starting database initialization", 
                   drop_existing=drop_existing,
                   sample_data=self.sample_data_enabled)
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'steps': {},
            'success': False,
            'errors': []
        }
        
        try:
            # Step 1: Drop existing tables if requested
            if drop_existing:
                results['steps']['drop_tables'] = await self._drop_tables()
            
            # Step 2: Run migrations
            results['steps']['migrations'] = await self._run_migrations()
            
            # Step 3: Create initial system configuration
            results['steps']['system_config'] = await self._create_system_config()
            
            # Step 4: Create personality system data
            results['steps']['personality_system'] = await self._create_personality_system()
            
            # Step 5: Create risk assessment system
            results['steps']['risk_system'] = await self._create_risk_system()
            
            # Step 6: Create feature flags
            results['steps']['feature_flags'] = await self._create_feature_flags()
            
            # Step 7: Create sample data if enabled
            if self.sample_data_enabled:
                results['steps']['sample_data'] = await self._create_sample_data()
            
            # Step 8: Verify database integrity
            results['steps']['verification'] = await self._verify_database()
            
            results['success'] = all(step.get('success', False) for step in results['steps'].values())
            
            if results['success']:
                logger.info("Database initialization completed successfully")
            else:
                logger.error("Database initialization failed", results=results)
            
        except Exception as e:
            logger.error("Database initialization error", error=str(e))
            results['errors'].append(str(e))
            results['success'] = False
        
        return results
    
    async def _drop_tables(self) -> Dict[str, Any]:
        """Drop all existing tables."""
        try:
            logger.info("Dropping existing tables")
            
            async with db_manager.get_async_session() as session:
                # Drop tables in dependency order
                drop_queries = [
                    "DROP TABLE IF EXISTS performance_metrics CASCADE",
                    "DROP TABLE IF EXISTS security_events CASCADE", 
                    "DROP TABLE IF EXISTS audit_logs CASCADE",
                    "DROP TABLE IF EXISTS rate_limit_configs CASCADE",
                    "DROP TABLE IF EXISTS feature_flags CASCADE",
                    "DROP TABLE IF EXISTS system_configurations CASCADE",
                    "DROP TABLE IF EXISTS system_metrics CASCADE",
                    "DROP TABLE IF EXISTS conversation_analytics CASCADE",
                    "DROP TABLE IF EXISTS user_activities CASCADE",
                    "DROP TABLE IF EXISTS conversation_risks CASCADE",
                    "DROP TABLE IF EXISTS risk_assessments CASCADE",
                    "DROP TABLE IF EXISTS risk_factors CASCADE",
                    "DROP TABLE IF EXISTS user_personality_mappings CASCADE",
                    "DROP TABLE IF EXISTS personality_profiles CASCADE", 
                    "DROP TABLE IF EXISTS personality_traits CASCADE",
                    "DROP TABLE IF EXISTS messages CASCADE",
                    "DROP TABLE IF EXISTS conversations CASCADE",
                    "DROP TABLE IF EXISTS conversation_sessions CASCADE",
                    "DROP TABLE IF EXISTS users CASCADE",
                    
                    # Drop enums
                    "DROP TYPE IF EXISTS message_direction CASCADE",
                    "DROP TYPE IF EXISTS message_type CASCADE", 
                    "DROP TYPE IF EXISTS conversation_status CASCADE",
                    "DROP TYPE IF EXISTS session_status CASCADE",
                ]
                
                for query in drop_queries:
                    await session.execute(text(query))
                
                await session.commit()
            
            return {'success': True, 'message': 'Tables dropped successfully'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _run_migrations(self) -> Dict[str, Any]:
        """Run Alembic migrations."""
        try:
            logger.info("Running database migrations")
            
            # Configure Alembic
            alembic_cfg = Config("alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", settings.database.sync_url)
            
            # Run migrations to head
            command.upgrade(alembic_cfg, "head")
            
            return {'success': True, 'message': 'Migrations completed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _create_system_config(self) -> Dict[str, Any]:
        """Create initial system configuration."""
        try:
            logger.info("Creating system configuration")
            
            configs = [
                {
                    'key': 'bot.name',
                    'value': 'AI Conversation Bot',
                    'description': 'Bot display name'
                },
                {
                    'key': 'bot.version',
                    'value': '1.0.0',
                    'description': 'Bot version'
                },
                {
                    'key': 'conversation.max_history_length',
                    'value': 50,
                    'description': 'Maximum conversation history length'
                },
                {
                    'key': 'conversation.session_timeout_minutes',
                    'value': 30,
                    'description': 'Session timeout in minutes'
                },
                {
                    'key': 'ml.sentiment_analysis_enabled',
                    'value': True,
                    'description': 'Enable sentiment analysis'
                },
                {
                    'key': 'ml.personality_adaptation_enabled',
                    'value': True,
                    'description': 'Enable personality adaptation'
                },
                {
                    'key': 'risk_assessment.enabled',
                    'value': True,
                    'description': 'Enable risk assessment'
                },
                {
                    'key': 'risk_assessment.escalation_threshold',
                    'value': 0.7,
                    'description': 'Risk score threshold for escalation'
                },
                {
                    'key': 'rate_limiting.messages_per_minute',
                    'value': 20,
                    'description': 'Messages per minute limit'
                },
                {
                    'key': 'analytics.retention_days',
                    'value': 90,
                    'description': 'Analytics data retention days'
                },
            ]
            
            created_count = 0
            for config in configs:
                await config_repo.set_config_value(
                    key=config['key'],
                    value=config['value'],
                    change_reason=config['description']
                )
                created_count += 1
            
            return {
                'success': True,
                'message': f'Created {created_count} system configurations'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _create_personality_system(self) -> Dict[str, Any]:
        """Create personality traits and profiles."""
        try:
            logger.info("Creating personality system")
            
            # Create personality traits
            traits = [
                {
                    'name': 'openness',
                    'dimension': 'openness',
                    'description': 'Openness to new experiences and ideas',
                    'measurement_indicators': {
                        'keywords': ['creative', 'curious', 'explore', 'new', 'different'],
                        'question_types': ['open_ended', 'hypothetical'],
                        'response_patterns': ['elaborate', 'ask_questions']
                    },
                    'adaptation_rules': {
                        'high': {'response_style': 'creative', 'suggest_alternatives': True},
                        'low': {'response_style': 'structured', 'stick_to_facts': True}
                    }
                },
                {
                    'name': 'conscientiousness',
                    'dimension': 'conscientiousness', 
                    'description': 'Organization, reliability, and attention to detail',
                    'measurement_indicators': {
                        'keywords': ['plan', 'organize', 'schedule', 'detail', 'thorough'],
                        'message_patterns': ['structured', 'bullet_points', 'numbered_lists'],
                        'timing_patterns': ['regular', 'punctual']
                    },
                    'adaptation_rules': {
                        'high': {'provide_structure': True, 'detailed_responses': True},
                        'low': {'flexible_approach': True, 'summarized_responses': True}
                    }
                },
                {
                    'name': 'extraversion',
                    'dimension': 'extraversion',
                    'description': 'Social engagement and energy from interaction',
                    'measurement_indicators': {
                        'keywords': ['social', 'people', 'group', 'party', 'talk'],
                        'interaction_patterns': ['frequent', 'long_messages', 'emoji_usage'],
                        'response_speed': ['quick', 'immediate']
                    },
                    'adaptation_rules': {
                        'high': {'enthusiastic_tone': True, 'social_references': True},
                        'low': {'calm_tone': True, 'direct_approach': True}
                    }
                },
                {
                    'name': 'agreeableness',
                    'dimension': 'agreeableness',
                    'description': 'Cooperativeness and trust in others',
                    'measurement_indicators': {
                        'keywords': ['help', 'agree', 'support', 'understand', 'care'],
                        'communication_style': ['polite', 'supportive', 'diplomatic'],
                        'conflict_handling': ['avoid', 'compromise', 'empathize']
                    },
                    'adaptation_rules': {
                        'high': {'supportive_tone': True, 'validation': True},
                        'low': {'direct_feedback': True, 'honest_opinions': True}
                    }
                },
                {
                    'name': 'neuroticism',
                    'dimension': 'neuroticism',
                    'description': 'Emotional stability and stress management',
                    'measurement_indicators': {
                        'keywords': ['stress', 'worry', 'anxiety', 'problem', 'difficult'],
                        'emotional_expressions': ['frustrated', 'overwhelmed', 'concerned'],
                        'help_seeking': ['frequent', 'urgent', 'detailed_problems']
                    },
                    'adaptation_rules': {
                        'high': {'reassuring_tone': True, 'stress_reduction_tips': True},
                        'low': {'straightforward_approach': True, 'challenge_appropriately': True}
                    }
                },
                {
                    'name': 'humor',
                    'dimension': 'humor',
                    'description': 'Appreciation for and use of humor',
                    'measurement_indicators': {
                        'keywords': ['funny', 'joke', 'lol', 'haha', 'amusing'],
                        'emoji_usage': ['laughing', 'winking', 'playful'],
                        'response_to_humor': ['positive', 'reciprocal']
                    },
                    'adaptation_rules': {
                        'high': {'use_humor': True, 'playful_responses': True},
                        'low': {'serious_tone': True, 'professional_approach': True}
                    }
                },
                {
                    'name': 'empathy',
                    'dimension': 'empathy',
                    'description': 'Understanding and sharing others feelings',
                    'measurement_indicators': {
                        'keywords': ['feel', 'understand', 'sorry', 'empathize', 'relate'],
                        'emotional_sharing': ['personal_stories', 'feelings', 'experiences'],
                        'response_to_others': ['supportive', 'understanding']
                    },
                    'adaptation_rules': {
                        'high': {'emotional_validation': True, 'personal_touch': True},
                        'low': {'logical_approach': True, 'solution_focused': True}
                    }
                }
            ]
            
            created_traits = 0
            for trait_data in traits:
                trait = PersonalityTrait(**trait_data)
                async with db_manager.get_async_session() as session:
                    session.add(trait)
                    await session.commit()
                    created_traits += 1
            
            # Create personality profiles
            profiles = [
                {
                    'name': 'professional',
                    'display_name': 'Professional Assistant',
                    'description': 'Formal, efficient, and goal-oriented communication style',
                    'category': 'professional',
                    'trait_scores': {
                        'openness': 0.6,
                        'conscientiousness': 0.9,
                        'extraversion': 0.5,
                        'agreeableness': 0.7,
                        'neuroticism': 0.2,
                        'humor': 0.3,
                        'empathy': 0.6
                    },
                    'behavioral_patterns': {
                        'greeting_style': 'formal',
                        'response_length': 'concise',
                        'use_examples': True,
                        'provide_sources': True
                    },
                    'communication_style': {
                        'tone': 'professional',
                        'formality': 'high',
                        'structure': 'organized',
                        'emoji_usage': 'minimal'
                    },
                    'is_default': True
                },
                {
                    'name': 'friendly',
                    'display_name': 'Friendly Companion',
                    'description': 'Warm, casual, and emotionally supportive communication',
                    'category': 'casual',
                    'trait_scores': {
                        'openness': 0.8,
                        'conscientiousness': 0.6,
                        'extraversion': 0.8,
                        'agreeableness': 0.9,
                        'neuroticism': 0.3,
                        'humor': 0.8,
                        'empathy': 0.9
                    },
                    'behavioral_patterns': {
                        'greeting_style': 'casual',
                        'response_length': 'conversational',
                        'use_examples': True,
                        'share_experiences': True
                    },
                    'communication_style': {
                        'tone': 'warm',
                        'formality': 'low',
                        'structure': 'flexible',
                        'emoji_usage': 'moderate'
                    }
                },
                {
                    'name': 'analytical',
                    'display_name': 'Analytical Expert',
                    'description': 'Logical, detail-oriented, and evidence-based approach',
                    'category': 'analytical',
                    'trait_scores': {
                        'openness': 0.7,
                        'conscientiousness': 0.9,
                        'extraversion': 0.4,
                        'agreeableness': 0.5,
                        'neuroticism': 0.2,
                        'humor': 0.2,
                        'empathy': 0.4
                    },
                    'behavioral_patterns': {
                        'greeting_style': 'direct',
                        'response_length': 'detailed',
                        'use_data': True,
                        'provide_analysis': True
                    },
                    'communication_style': {
                        'tone': 'neutral',
                        'formality': 'medium',
                        'structure': 'logical',
                        'emoji_usage': 'none'
                    }
                },
                {
                    'name': 'creative',
                    'display_name': 'Creative Collaborator',
                    'description': 'Imaginative, expressive, and inspiration-focused',
                    'category': 'creative',
                    'trait_scores': {
                        'openness': 0.95,
                        'conscientiousness': 0.5,
                        'extraversion': 0.7,
                        'agreeableness': 0.8,
                        'neuroticism': 0.4,
                        'humor': 0.8,
                        'empathy': 0.7
                    },
                    'behavioral_patterns': {
                        'greeting_style': 'creative',
                        'response_length': 'expressive',
                        'use_metaphors': True,
                        'encourage_exploration': True
                    },
                    'communication_style': {
                        'tone': 'enthusiastic',
                        'formality': 'low',
                        'structure': 'flexible',
                        'emoji_usage': 'high'
                    }
                },
                {
                    'name': 'supportive',
                    'display_name': 'Supportive Counselor',
                    'description': 'Empathetic, patient, and emotionally intelligent',
                    'category': 'supportive',
                    'trait_scores': {
                        'openness': 0.7,
                        'conscientiousness': 0.7,
                        'extraversion': 0.5,
                        'agreeableness': 0.95,
                        'neuroticism': 0.2,
                        'humor': 0.5,
                        'empathy': 0.95
                    },
                    'behavioral_patterns': {
                        'greeting_style': 'warm',
                        'response_length': 'thoughtful',
                        'validate_emotions': True,
                        'offer_support': True
                    },
                    'communication_style': {
                        'tone': 'caring',
                        'formality': 'low',
                        'structure': 'adaptive',
                        'emoji_usage': 'supportive'
                    }
                }
            ]
            
            created_profiles = 0
            for profile_data in profiles:
                profile = PersonalityProfile(**profile_data)
                async with db_manager.get_async_session() as session:
                    session.add(profile)
                    await session.commit()
                    created_profiles += 1
            
            return {
                'success': True,
                'message': f'Created {created_traits} traits and {created_profiles} profiles'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _create_risk_system(self) -> Dict[str, Any]:
        """Create risk factors and assessment rules."""
        try:
            logger.info("Creating risk assessment system")
            
            risk_factors = [
                {
                    'name': 'inappropriate_content',
                    'category': 'content_safety',
                    'description': 'Detection of inappropriate or explicit content',
                    'base_risk_score': 0.7,
                    'detection_patterns': {
                        'keywords': ['explicit', 'inappropriate', 'offensive'],
                        'regex_patterns': [r'\b(adult|explicit)\s+content\b'],
                        'ml_models': ['content_classifier']
                    },
                    'escalation_rules': {
                        'score_threshold': 0.8,
                        'context_conditions': [
                            {'field': 'user_age', 'operator': 'lt', 'value': 18}
                        ]
                    }
                },
                {
                    'name': 'personal_information_sharing',
                    'category': 'privacy_violation',
                    'description': 'Detection of personal information sharing',
                    'base_risk_score': 0.5,
                    'detection_patterns': {
                        'patterns': ['phone_number', 'email', 'address', 'ssn'],
                        'regex_patterns': [
                            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
                        ]
                    },
                    'escalation_rules': {
                        'score_threshold': 0.6
                    }
                },
                {
                    'name': 'harassment_detection',
                    'category': 'harassment',
                    'description': 'Detection of harassment or bullying behavior',
                    'base_risk_score': 0.8,
                    'detection_patterns': {
                        'keywords': ['harassment', 'bullying', 'threat', 'intimidate'],
                        'sentiment_threshold': -0.7,
                        'repetitive_patterns': True
                    },
                    'escalation_rules': {
                        'score_threshold': 0.7
                    }
                },
                {
                    'name': 'spam_detection',
                    'category': 'spam',
                    'description': 'Detection of spam or repetitive unwanted content',
                    'base_risk_score': 0.3,
                    'detection_patterns': {
                        'repetitive_content': True,
                        'external_links': True,
                        'promotional_keywords': ['buy', 'sale', 'offer', 'deal', 'discount']
                    },
                    'escalation_rules': {
                        'score_threshold': 0.5,
                        'context_conditions': [
                            {'field': 'message_frequency', 'operator': 'gt', 'value': 10}
                        ]
                    }
                },
                {
                    'name': 'emotional_distress',
                    'category': 'emotional_distress',
                    'description': 'Detection of user emotional distress or crisis',
                    'base_risk_score': 0.6,
                    'detection_patterns': {
                        'keywords': ['depression', 'suicide', 'harm', 'crisis', 'help'],
                        'emotional_indicators': ['hopeless', 'worthless', 'desperate'],
                        'sentiment_threshold': -0.8
                    },
                    'escalation_rules': {
                        'score_threshold': 0.5  # Lower threshold for mental health
                    },
                    'mitigation_strategies': {
                        'immediate': ['supportive_response', 'crisis_resources'],
                        'followup': ['wellness_check', 'professional_referral']
                    }
                }
            ]
            
            created_factors = 0
            for factor_data in risk_factors:
                factor = RiskFactor(**factor_data)
                async with db_manager.get_async_session() as session:
                    session.add(factor)
                    await session.commit()
                    created_factors += 1
            
            return {
                'success': True,
                'message': f'Created {created_factors} risk factors'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _create_feature_flags(self) -> Dict[str, Any]:
        """Create initial feature flags."""
        try:
            logger.info("Creating feature flags")
            
            feature_flags = [
                {
                    'flag_key': 'personality_adaptation',
                    'name': 'Personality Adaptation',
                    'description': 'Enable AI personality adaptation based on user behavior',
                    'category': 'ml',
                    'is_enabled': True,
                    'strategy': 'all_users',
                    'rollout_percentage': 100.0
                },
                {
                    'flag_key': 'sentiment_analysis',
                    'name': 'Sentiment Analysis',
                    'description': 'Enable sentiment analysis of user messages',
                    'category': 'ml',
                    'is_enabled': True,
                    'strategy': 'all_users',
                    'rollout_percentage': 100.0
                },
                {
                    'flag_key': 'risk_assessment',
                    'name': 'Risk Assessment',
                    'description': 'Enable automated risk assessment of conversations',
                    'category': 'safety',
                    'is_enabled': True,
                    'strategy': 'all_users',
                    'rollout_percentage': 100.0
                },
                {
                    'flag_key': 'advanced_analytics',
                    'name': 'Advanced Analytics',
                    'description': 'Enable advanced conversation analytics and insights',
                    'category': 'analytics',
                    'is_enabled': True,
                    'strategy': 'percentage',
                    'rollout_percentage': 50.0
                },
                {
                    'flag_key': 'conversation_memory',
                    'name': 'Conversation Memory',
                    'description': 'Enable long-term conversation memory and context',
                    'category': 'conversation',
                    'is_enabled': False,
                    'strategy': 'gradual_rollout',
                    'rollout_percentage': 10.0
                },
                {
                    'flag_key': 'multilingual_support',
                    'name': 'Multilingual Support',
                    'description': 'Enable support for multiple languages',
                    'category': 'i18n',
                    'is_enabled': False,
                    'strategy': 'user_attributes',
                    'target_attributes': {'language': {'in': ['es', 'fr', 'de']}}
                }
            ]
            
            created_flags = 0
            for flag_data in feature_flags:
                flag = FeatureFlag(**flag_data)
                async with db_manager.get_async_session() as session:
                    session.add(flag)
                    await session.commit()
                    created_flags += 1
            
            return {
                'success': True,
                'message': f'Created {created_flags} feature flags'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _create_sample_data(self) -> Dict[str, Any]:
        """Create sample users and conversations for development."""
        try:
            logger.info("Creating sample data")
            
            # Create sample users
            sample_users = [
                {
                    'telegram_id': 123456789,
                    'username': 'testuser1',
                    'first_name': 'Alice',
                    'last_name': 'Johnson',
                    'language_code': 'en',
                    'message_count': 25,
                    'command_count': 5,
                    'preferences': {
                        'notifications': True,
                        'theme': 'light',
                        'language': 'en'
                    }
                },
                {
                    'telegram_id': 987654321,
                    'username': 'testuser2', 
                    'first_name': 'Bob',
                    'language_code': 'en',
                    'message_count': 15,
                    'command_count': 8,
                    'is_premium': True,
                    'preferences': {
                        'notifications': False,
                        'theme': 'dark',
                        'language': 'en'
                    }
                },
                {
                    'telegram_id': 555666777,
                    'username': 'devuser',
                    'first_name': 'Developer',
                    'last_name': 'Test',
                    'language_code': 'en',
                    'message_count': 50,
                    'command_count': 20,
                    'preferences': {
                        'notifications': True,
                        'theme': 'dark',
                        'language': 'en',
                        'developer_mode': True
                    }
                }
            ]
            
            created_users = 0
            for user_data in sample_users:
                user = await user_repo.create(user_data)
                if user:
                    created_users += 1
            
            # Create sample audit logs
            await audit_repo.log_event(
                event_type='system_event',
                action='database_initialization',
                description='Database initialization completed with sample data',
                event_data={'sample_users': created_users}
            )
            
            return {
                'success': True,
                'message': f'Created {created_users} sample users and related data'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _verify_database(self) -> Dict[str, Any]:
        """Verify database integrity and setup."""
        try:
            logger.info("Verifying database integrity")
            
            verification_results = {}
            
            # Check table existence and basic functionality
            repositories = {
                'users': user_repo,
                'personality_profiles': personality_repo,
                'system_configurations': config_repo,
                'feature_flags': feature_flag_repo
            }
            
            for table_name, repo in repositories.items():
                try:
                    # Test basic operations
                    health = await repo.health_check()
                    count = await repo.count_by_filters([])
                    
                    verification_results[table_name] = {
                        'healthy': health['status'] == 'healthy',
                        'record_count': count,
                        'response_time_ms': health['response_time_ms']
                    }
                    
                except Exception as e:
                    verification_results[table_name] = {
                        'healthy': False,
                        'error': str(e)
                    }
            
            # Check critical configurations exist
            critical_configs = [
                'bot.name',
                'conversation.session_timeout_minutes',
                'ml.personality_adaptation_enabled',
                'risk_assessment.enabled'
            ]
            
            missing_configs = []
            for config_key in critical_configs:
                value = await config_repo.get_config_value(config_key)
                if value is None:
                    missing_configs.append(config_key)
            
            verification_results['critical_configs'] = {
                'all_present': len(missing_configs) == 0,
                'missing': missing_configs
            }
            
            # Overall health
            all_tables_healthy = all(
                result.get('healthy', False) 
                for result in verification_results.values() 
                if isinstance(result, dict)
            )
            
            return {
                'success': all_tables_healthy and len(missing_configs) == 0,
                'verification_results': verification_results,
                'message': 'Database verification completed'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


async def main():
    """Main initialization function."""
    
    # Initialize database connection
    await db_manager.initialize()
    
    try:
        # Create initializer
        initializer = DatabaseInitializer()
        
        # Run initialization
        results = await initializer.initialize_database(drop_existing=False)
        
        if results['success']:
            logger.info("✅ Database initialization completed successfully")
            
            # Print summary
            print("\n=== Database Initialization Summary ===")
            for step, result in results['steps'].items():
                status = "✅" if result.get('success', False) else "❌"
                message = result.get('message', result.get('error', 'No details'))
                print(f"{status} {step}: {message}")
            
            print(f"\n🚀 Database is ready for AI conversation bot operations!")
        else:
            logger.error("❌ Database initialization failed")
            print("\n=== Initialization Errors ===")
            for step, result in results['steps'].items():
                if not result.get('success', False):
                    print(f"❌ {step}: {result.get('error', 'Unknown error')}")
            
            if results.get('errors'):
                print("\n=== Additional Errors ===")
                for error in results['errors']:
                    print(f"❌ {error}")
    
    finally:
        # Close database connection
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())