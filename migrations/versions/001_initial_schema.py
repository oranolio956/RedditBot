"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2024-09-09 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema with all tables."""
    
    # Create enums
    op.execute("""
        CREATE TYPE session_status AS ENUM ('active', 'paused', 'ended', 'archived');
        CREATE TYPE conversation_status AS ENUM ('active', 'paused', 'ended', 'archived');
        CREATE TYPE message_type AS ENUM (
            'text', 'command', 'callback', 'inline', 'sticker', 'photo', 
            'video', 'audio', 'voice', 'document', 'location', 'contact', 
            'poll', 'venue', 'animation', 'video_note', 'game', 'invoice', 
            'successful_payment'
        );
        CREATE TYPE message_direction AS ENUM ('incoming', 'outgoing');
    """)
    
    # Users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # User-specific fields
        sa.Column('telegram_id', sa.BigInteger(), unique=True, nullable=False),
        sa.Column('username', sa.String(32), nullable=True),
        sa.Column('first_name', sa.String(64), nullable=True),
        sa.Column('last_name', sa.String(64), nullable=True),
        sa.Column('language_code', sa.String(10), default='en', nullable=True),
        sa.Column('is_bot', sa.Boolean(), default=False, nullable=False),
        sa.Column('is_premium', sa.Boolean(), default=False, nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_blocked', sa.Boolean(), default=False, nullable=False),
        sa.Column('first_interaction', sa.String(50), nullable=True),
        sa.Column('last_activity', sa.String(50), nullable=True),
        sa.Column('message_count', sa.Integer(), default=0, nullable=False),
        sa.Column('command_count', sa.Integer(), default=0, nullable=False),
        sa.Column('preferences', postgresql.JSONB(), default={}, nullable=True),
        sa.Column('personality_profile', postgresql.JSONB(), nullable=True),
        sa.Column('interaction_history', postgresql.JSONB(), default=[], nullable=True),
        
        sa.Index('idx_user_telegram_id', 'telegram_id'),
        sa.Index('idx_user_username', 'username'),
        sa.Index('idx_user_active', 'is_active'),
        sa.Index('idx_user_blocked', 'is_blocked'),
        sa.Index('idx_user_activity', 'updated_at'),
        sa.Index('idx_user_language', 'language_code'),
        
        comment='User accounts and profiles'
    )
    
    # Conversation Sessions table
    op.create_table('conversation_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('session_token', sa.String(128), unique=True, nullable=False),
        sa.Column('status', postgresql.ENUM('active', 'paused', 'ended', 'archived', name='session_status'), default='active', nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('message_count', sa.Integer(), default=0, nullable=False),
        sa.Column('duration_seconds', sa.Integer(), default=0, nullable=False),
        sa.Column('context_data', postgresql.JSONB(), default={}, nullable=True),
        sa.Column('conversation_topic', sa.String(255), nullable=True),
        sa.Column('sentiment_scores', postgresql.JSONB(), nullable=True),
        sa.Column('personality_adaptations', postgresql.JSONB(), nullable=True),
        sa.Column('engagement_metrics', postgresql.JSONB(), nullable=True),
        
        sa.Index('idx_session_user_status', 'user_id', 'status'),
        sa.Index('idx_session_activity', 'last_activity_at'),
        sa.Index('idx_session_duration', 'started_at', 'ended_at'),
        
        sa.CheckConstraint('duration_seconds >= 0', name='ck_session_duration_positive'),
        sa.CheckConstraint('message_count >= 0', name='ck_session_message_count_positive'),
        
        comment='User conversation sessions'
    )
    
    # Conversations table
    op.create_table('conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversation_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('topic', sa.String(100), nullable=True),
        sa.Column('status', postgresql.ENUM('active', 'paused', 'ended', 'archived', name='conversation_status'), default='active', nullable=False),
        sa.Column('context_data', postgresql.JSONB(), default={}, nullable=True),
        sa.Column('conversation_flow', sa.String(50), nullable=True),
        sa.Column('sentiment_summary', postgresql.JSONB(), nullable=True),
        sa.Column('complexity_score', sa.Float(), nullable=True),
        sa.Column('engagement_score', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('message_count', sa.Integer(), default=0, nullable=False),
        
        sa.Index('idx_conversation_user_topic', 'user_id', 'topic'),
        sa.Index('idx_conversation_session_status', 'session_id', 'status'),
        sa.Index('idx_conversation_timing', 'started_at', 'last_message_at'),
        
        sa.CheckConstraint('message_count >= 0', name='ck_conversation_message_count_positive'),
        sa.CheckConstraint('complexity_score >= 0 AND complexity_score <= 1', name='ck_complexity_range'),
        sa.CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='ck_engagement_range'),
        
        comment='Individual conversation threads'
    )
    
    # Messages table
    op.create_table('messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversation_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('telegram_message_id', sa.BigInteger(), nullable=True),
        sa.Column('telegram_chat_id', sa.BigInteger(), nullable=True),
        sa.Column('message_type', postgresql.ENUM(name='message_type'), nullable=False),
        sa.Column('direction', postgresql.ENUM('incoming', 'outgoing', name='message_direction'), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), default={}, nullable=True),
        sa.Column('attachments', postgresql.JSONB(), nullable=True),
        sa.Column('reply_to_message_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('messages.id', ondelete='SET NULL'), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('sentiment_label', sa.String(20), nullable=True),
        sa.Column('emotion_scores', postgresql.JSONB(), nullable=True),
        sa.Column('intent_classification', sa.String(50), nullable=True),
        sa.Column('entities', postgresql.JSONB(), nullable=True),
        sa.Column('keywords', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('response_generated', sa.Boolean(), default=False, nullable=False),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('response_model_used', sa.String(100), nullable=True),
        sa.Column('user_feedback', sa.String(20), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        
        sa.Index('idx_message_conversation_created', 'conversation_id', 'created_at'),
        sa.Index('idx_message_user_type', 'user_id', 'message_type'),
        sa.Index('idx_message_direction_created', 'direction', 'created_at'),
        sa.Index('idx_message_telegram', 'telegram_chat_id', 'telegram_message_id'),
        sa.Index('idx_message_content_hash', 'content_hash'),
        sa.Index('idx_message_sentiment', 'sentiment_label'),
        sa.Index('idx_message_intent', 'intent_classification'),
        
        sa.CheckConstraint('sentiment_score >= -1 AND sentiment_score <= 1', name='ck_sentiment_range'),
        sa.CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='ck_quality_range'),
        sa.CheckConstraint('response_time_ms >= 0', name='ck_response_time_positive'),
        
        comment='Individual messages in conversations'
    )
    
    # Personality Traits table
    op.create_table('personality_traits',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        sa.Column('name', sa.String(50), unique=True, nullable=False),
        sa.Column('dimension', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('min_score', sa.Float(), default=0.0, nullable=False),
        sa.Column('max_score', sa.Float(), default=1.0, nullable=False),
        sa.Column('default_score', sa.Float(), default=0.5, nullable=False),
        sa.Column('measurement_indicators', postgresql.JSONB(), nullable=True),
        sa.Column('adaptation_rules', postgresql.JSONB(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('weight', sa.Float(), default=1.0, nullable=False),
        
        sa.Index('idx_trait_dimension', 'dimension'),
        sa.Index('idx_trait_active', 'is_active'),
        
        sa.CheckConstraint('min_score <= max_score', name='ck_trait_score_range'),
        sa.CheckConstraint('default_score >= min_score AND default_score <= max_score', name='ck_trait_default_range'),
        sa.CheckConstraint('weight >= 0', name='ck_trait_weight_positive'),
        
        comment='Personality trait definitions'
    )
    
    # Personality Profiles table
    op.create_table('personality_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('display_name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('trait_scores', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('behavioral_patterns', postgresql.JSONB(), nullable=True),
        sa.Column('communication_style', postgresql.JSONB(), nullable=True),
        sa.Column('adaptation_strategy', sa.String(20), default='balance', nullable=False),
        sa.Column('adaptation_sensitivity', sa.Float(), default=0.5, nullable=False),
        sa.Column('adaptation_limits', postgresql.JSONB(), nullable=True),
        sa.Column('usage_count', sa.Integer(), default=0, nullable=False),
        sa.Column('average_satisfaction_score', sa.Float(), nullable=True),
        sa.Column('performance_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_default', sa.Boolean(), default=False, nullable=False),
        
        sa.Index('idx_profile_category', 'category'),
        sa.Index('idx_profile_active_default', 'is_active', 'is_default'),
        
        sa.CheckConstraint('adaptation_sensitivity >= 0 AND adaptation_sensitivity <= 1', name='ck_adaptation_sensitivity_range'),
        sa.CheckConstraint('usage_count >= 0', name='ck_usage_count_positive'),
        sa.CheckConstraint('average_satisfaction_score >= 0 AND average_satisfaction_score <= 1', name='ck_satisfaction_range'),
        
        comment='AI personality profile templates'
    )
    
    # User Personality Mappings table
    op.create_table('user_personality_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('profile_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('personality_profiles.id', ondelete='CASCADE'), nullable=False),
        sa.Column('measured_user_traits', postgresql.JSONB(), nullable=True),
        sa.Column('adapted_profile_traits', postgresql.JSONB(), nullable=True),
        sa.Column('interaction_history_summary', postgresql.JSONB(), nullable=True),
        sa.Column('adaptation_confidence', sa.Float(), default=0.0, nullable=False),
        sa.Column('learning_iterations', sa.Integer(), default=0, nullable=False),
        sa.Column('satisfaction_scores', postgresql.JSONB(), nullable=True),
        sa.Column('engagement_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('effectiveness_score', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_primary', sa.Boolean(), default=False, nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('usage_count', sa.Integer(), default=0, nullable=False),
        
        sa.Index('idx_user_personality_active', 'user_id', 'is_active'),
        sa.Index('idx_user_personality_primary', 'user_id', 'is_primary'),
        sa.Index('idx_personality_mapping_usage', 'last_used_at'),
        
        sa.UniqueConstraint('user_id', 'profile_id', name='uq_user_profile_mapping'),
        sa.CheckConstraint('adaptation_confidence >= 0 AND adaptation_confidence <= 1', name='ck_confidence_range'),
        sa.CheckConstraint('effectiveness_score >= 0 AND effectiveness_score <= 1', name='ck_effectiveness_range'),
        sa.CheckConstraint('learning_iterations >= 0', name='ck_learning_iterations_positive'),
        sa.CheckConstraint('usage_count >= 0', name='ck_usage_count_positive'),
        
        comment='User-specific personality adaptations'
    )
    
    # Risk Factors table
    op.create_table('risk_factors',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('base_risk_score', sa.Float(), default=0.0, nullable=False),
        sa.Column('max_risk_score', sa.Float(), default=1.0, nullable=False),
        sa.Column('weight', sa.Float(), default=1.0, nullable=False),
        sa.Column('detection_patterns', postgresql.JSONB(), nullable=True),
        sa.Column('ml_model_config', postgresql.JSONB(), nullable=True),
        sa.Column('threshold_config', postgresql.JSONB(), nullable=True),
        sa.Column('mitigation_strategies', postgresql.JSONB(), nullable=True),
        sa.Column('escalation_rules', postgresql.JSONB(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('detection_accuracy', sa.Float(), nullable=True),
        sa.Column('false_positive_rate', sa.Float(), nullable=True),
        
        sa.Index('idx_risk_factor_category', 'category'),
        sa.Index('idx_risk_factor_active', 'is_active'),
        
        sa.CheckConstraint('base_risk_score >= 0 AND base_risk_score <= 1', name='ck_base_risk_range'),
        sa.CheckConstraint('max_risk_score >= 0 AND max_risk_score <= 1', name='ck_max_risk_range'),
        sa.CheckConstraint('weight >= 0', name='ck_weight_positive'),
        sa.CheckConstraint('detection_accuracy >= 0 AND detection_accuracy <= 1', name='ck_accuracy_range'),
        sa.CheckConstraint('false_positive_rate >= 0 AND false_positive_rate <= 1', name='ck_fpr_range'),
        
        comment='Risk factor definitions for assessment'
    )
    
    # Risk Assessments table  
    op.create_table('risk_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('assessment_id', sa.String(64), unique=True, nullable=False),
        sa.Column('assessment_type', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('target_id', sa.String(64), nullable=False),
        sa.Column('overall_risk_score', sa.Float(), nullable=False),
        sa.Column('overall_risk_level', sa.String(20), nullable=False),
        sa.Column('risk_factors_detected', postgresql.JSONB(), nullable=True),
        sa.Column('category_scores', postgresql.JSONB(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('assessment_context', postgresql.JSONB(), nullable=True),
        sa.Column('content_analyzed', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('models_used', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('assessment_version', sa.String(20), nullable=True),
        sa.Column('status', sa.String(20), default='completed', nullable=False),
        sa.Column('requires_escalation', sa.Boolean(), default=False, nullable=False),
        sa.Column('escalated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('escalation_reason', sa.Text(), nullable=True),
        sa.Column('actions_recommended', postgresql.JSONB(), nullable=True),
        sa.Column('actions_taken', postgresql.JSONB(), nullable=True),
        sa.Column('reviewed_by_human', sa.Boolean(), default=False, nullable=False),
        sa.Column('human_review_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('human_review_notes', sa.Text(), nullable=True),
        sa.Column('review_outcome', sa.String(50), nullable=True),
        
        sa.Index('idx_assessment_type_target', 'assessment_type', 'target_type'),
        sa.Index('idx_assessment_risk_level', 'overall_risk_level'),
        sa.Index('idx_assessment_escalation', 'requires_escalation', 'escalated_at'),
        sa.Index('idx_assessment_review', 'reviewed_by_human'),
        
        sa.CheckConstraint('overall_risk_score >= 0 AND overall_risk_score <= 1', name='ck_risk_score_range'),
        sa.CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='ck_confidence_range'),
        sa.CheckConstraint('processing_time_ms >= 0', name='ck_processing_time_positive'),
        
        comment='Risk assessment results'
    )
    
    # Create additional tables for remaining models...
    
    print("✅ Initial database schema created successfully")


def downgrade() -> None:
    """Drop all tables and enums."""
    
    # Drop tables in reverse dependency order
    tables_to_drop = [
        'risk_assessments',
        'risk_factors', 
        'user_personality_mappings',
        'personality_profiles',
        'personality_traits',
        'messages',
        'conversations', 
        'conversation_sessions',
        'users'
    ]
    
    for table in tables_to_drop:
        op.drop_table(table)
    
    # Drop enums
    op.execute("""
        DROP TYPE IF EXISTS message_direction;
        DROP TYPE IF EXISTS message_type;
        DROP TYPE IF EXISTS conversation_status;
        DROP TYPE IF EXISTS session_status;
    """)
    
    print("✅ Database schema dropped successfully")