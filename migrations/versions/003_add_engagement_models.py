"""Add proactive engagement models

Revision ID: 003_add_engagement_models
Revises: 002_add_stripe_payment_tables
Create Date: 2024-09-10 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_add_engagement_models'
down_revision = '002_add_stripe_payment_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Create engagement enums
    engagement_type_enum = postgresql.ENUM(
        'message', 'command', 'callback', 'voice_message', 'document', 
        'sticker', 'reaction', 'inline_query', 
        name='engagementtype'
    )
    sentiment_type_enum = postgresql.ENUM(
        'very_positive', 'positive', 'neutral', 'negative', 'very_negative', 'unknown',
        name='sentimenttype'
    )
    outreach_type_enum = postgresql.ENUM(
        'milestone_celebration', 're_engagement', 'personalized_checkin', 
        'feature_suggestion', 'mood_support', 'topic_follow_up', 'achievement_unlock',
        name='outreachtype'
    )
    outreach_status_enum = postgresql.ENUM(
        'scheduled', 'sent', 'delivered', 'read', 'responded', 'failed', 'cancelled',
        name='outreachstatus'
    )
    
    engagement_type_enum.create(op.get_bind())
    sentiment_type_enum.create(op.get_bind())
    outreach_type_enum.create(op.get_bind())
    outreach_status_enum.create(op.get_bind())
    
    # Create user_engagements table
    op.create_table('user_engagements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('telegram_id', sa.BigInteger(), nullable=False),
        sa.Column('engagement_type', engagement_type_enum, nullable=False),
        sa.Column('interaction_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('message_text', sa.Text(), nullable=True),
        sa.Column('command_name', sa.String(length=100), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('sentiment_type', sentiment_type_enum, nullable=True),
        sa.Column('response_time_seconds', sa.Integer(), nullable=True),
        sa.Column('message_length', sa.Integer(), nullable=True),
        sa.Column('contains_emoji', sa.Boolean(), nullable=False, default=False),
        sa.Column('contains_question', sa.Boolean(), nullable=False, default=False),
        sa.Column('detected_topics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('user_intent', sa.String(length=100), nullable=True),
        sa.Column('mood_indicators', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('engagement_quality_score', sa.Float(), nullable=True),
        sa.Column('is_meaningful_interaction', sa.Boolean(), nullable=False, default=True),
        sa.Column('conversation_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('previous_bot_message', sa.Text(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for user_engagements
    op.create_index('idx_engagement_user_timestamp', 'user_engagements', ['user_id', 'interaction_timestamp'])
    op.create_index('idx_engagement_telegram_timestamp', 'user_engagements', ['telegram_id', 'interaction_timestamp'])
    op.create_index('idx_engagement_type_timestamp', 'user_engagements', ['engagement_type', 'interaction_timestamp'])
    op.create_index('idx_engagement_sentiment', 'user_engagements', ['sentiment_type', 'sentiment_score'])
    op.create_index('idx_engagement_session', 'user_engagements', ['session_id', 'interaction_timestamp'])
    op.create_index('idx_engagement_command', 'user_engagements', ['command_name', 'interaction_timestamp'])
    op.create_index('idx_engagement_quality', 'user_engagements', ['engagement_quality_score', 'interaction_timestamp'])
    
    # Create user_behavior_patterns table
    op.create_table('user_behavior_patterns',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('telegram_id', sa.BigInteger(), nullable=False),
        sa.Column('total_interactions', sa.Integer(), nullable=False, default=0),
        sa.Column('daily_interaction_average', sa.Float(), nullable=True),
        sa.Column('most_active_hour', sa.Integer(), nullable=True),
        sa.Column('most_active_day', sa.Integer(), nullable=True),
        sa.Column('average_session_length_minutes', sa.Float(), nullable=True),
        sa.Column('average_sentiment_score', sa.Float(), nullable=True),
        sa.Column('dominant_sentiment', sentiment_type_enum, nullable=True),
        sa.Column('engagement_quality_trend', sa.Float(), nullable=True),
        sa.Column('response_time_average_seconds', sa.Float(), nullable=True),
        sa.Column('preferred_interaction_types', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('favorite_commands', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('topic_interests', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_highly_engaged', sa.Boolean(), nullable=False, default=False),
        sa.Column('shows_declining_engagement', sa.Boolean(), nullable=False, default=False),
        sa.Column('needs_re_engagement', sa.Boolean(), nullable=False, default=False),
        sa.Column('churn_risk_score', sa.Float(), nullable=True),
        sa.Column('optimal_outreach_hour', sa.Integer(), nullable=True),
        sa.Column('days_since_last_interaction', sa.Integer(), nullable=True),
        sa.Column('longest_absence_days', sa.Integer(), nullable=True),
        sa.Column('milestones_achieved', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('next_milestone_target', sa.String(length=100), nullable=True),
        sa.Column('milestone_progress_percent', sa.Float(), nullable=True),
        sa.Column('last_pattern_analysis', sa.DateTime(timezone=True), nullable=True),
        sa.Column('pattern_analysis_version', sa.String(length=20), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id'),
        sa.UniqueConstraint('telegram_id'),
    )
    
    # Create indexes for user_behavior_patterns
    op.create_index('idx_behavior_churn_risk', 'user_behavior_patterns', ['churn_risk_score', 'updated_at'])
    op.create_index('idx_behavior_engagement_trend', 'user_behavior_patterns', ['engagement_quality_trend', 'updated_at'])
    op.create_index('idx_behavior_needs_reengagement', 'user_behavior_patterns', ['needs_re_engagement', 'updated_at'])
    op.create_index('idx_behavior_last_interaction', 'user_behavior_patterns', ['days_since_last_interaction'])
    op.create_index('idx_behavior_analysis_date', 'user_behavior_patterns', ['last_pattern_analysis'])
    
    # Create engagement_milestones table
    op.create_table('engagement_milestones',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('milestone_name', sa.String(length=100), nullable=False),
        sa.Column('display_name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('target_value', sa.Float(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('celebration_template', sa.String(length=200), nullable=True),
        sa.Column('reward_type', sa.String(length=50), nullable=True),
        sa.Column('reward_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('difficulty_level', sa.Integer(), nullable=True),
        sa.Column('estimated_days_to_achieve', sa.Integer(), nullable=True),
        sa.Column('total_achievements', sa.Integer(), nullable=False, default=0),
        sa.Column('average_days_to_achieve', sa.Float(), nullable=True),
        sa.Column('last_achieved_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('milestone_name'),
    )
    
    # Create indexes for engagement_milestones
    op.create_index('idx_milestone_active', 'engagement_milestones', ['is_active', 'metric_name'])
    op.create_index('idx_milestone_category', 'engagement_milestones', ['category', 'difficulty_level'])
    op.create_index('idx_milestone_achievements', 'engagement_milestones', ['total_achievements', 'updated_at'])
    
    # Create user_milestone_progress table
    op.create_table('user_milestone_progress',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('milestone_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('telegram_id', sa.BigInteger(), nullable=False),
        sa.Column('current_value', sa.Float(), nullable=False, default=0.0),
        sa.Column('target_value', sa.Float(), nullable=False),
        sa.Column('progress_percentage', sa.Float(), nullable=True),
        sa.Column('is_achieved', sa.Boolean(), nullable=False, default=False),
        sa.Column('achieved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('days_to_achieve', sa.Integer(), nullable=True),
        sa.Column('celebration_sent', sa.Boolean(), nullable=False, default=False),
        sa.Column('celebration_sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_updated_value', sa.Float(), nullable=True),
        sa.Column('progress_velocity', sa.Float(), nullable=True),
        sa.Column('estimated_completion_date', sa.DateTime(timezone=True), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['milestone_id'], ['engagement_milestones.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for user_milestone_progress
    op.create_index('idx_milestone_progress_user', 'user_milestone_progress', ['user_id', 'is_achieved'])
    op.create_index('idx_milestone_progress_telegram', 'user_milestone_progress', ['telegram_id', 'is_achieved'])
    op.create_index('idx_milestone_progress_achieved', 'user_milestone_progress', ['is_achieved', 'achieved_at'])
    op.create_index('idx_milestone_progress_completion', 'user_milestone_progress', ['estimated_completion_date', 'progress_percentage'])
    op.create_index('idx_milestone_progress_celebration', 'user_milestone_progress', ['celebration_sent', 'is_achieved'])
    op.create_index('idx_milestone_progress_unique', 'user_milestone_progress', ['user_id', 'milestone_id'], unique=True)
    
    # Create proactive_outreaches table
    op.create_table('proactive_outreaches',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('telegram_id', sa.BigInteger(), nullable=False),
        sa.Column('outreach_type', outreach_type_enum, nullable=False),
        sa.Column('campaign_id', sa.String(length=100), nullable=True),
        sa.Column('priority_score', sa.Float(), nullable=True),
        sa.Column('scheduled_for', sa.DateTime(timezone=True), nullable=False),
        sa.Column('optimal_timing_used', sa.Boolean(), nullable=False, default=True),
        sa.Column('message_template', sa.String(length=200), nullable=True),
        sa.Column('message_content', sa.Text(), nullable=False),
        sa.Column('personalization_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('trigger_event', sa.String(length=100), nullable=True),
        sa.Column('trigger_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('behavioral_indicators', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('status', outreach_status_enum, nullable=False, default='scheduled'),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('user_responded', sa.Boolean(), nullable=False, default=False),
        sa.Column('response_time_minutes', sa.Integer(), nullable=True),
        sa.Column('response_sentiment', sentiment_type_enum, nullable=True),
        sa.Column('response_content', sa.Text(), nullable=True),
        sa.Column('engagement_improvement', sa.Boolean(), nullable=True),
        sa.Column('led_to_extended_session', sa.Boolean(), nullable=False, default=False),
        sa.Column('effectiveness_score', sa.Float(), nullable=True),
        sa.Column('follow_up_needed', sa.Boolean(), nullable=False, default=False),
        sa.Column('next_follow_up_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('max_retries', sa.Integer(), nullable=False, default=3),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for proactive_outreaches
    op.create_index('idx_outreach_user_scheduled', 'proactive_outreaches', ['user_id', 'scheduled_for'])
    op.create_index('idx_outreach_telegram_scheduled', 'proactive_outreaches', ['telegram_id', 'scheduled_for'])
    op.create_index('idx_outreach_status_scheduled', 'proactive_outreaches', ['status', 'scheduled_for'])
    op.create_index('idx_outreach_type_priority', 'proactive_outreaches', ['outreach_type', 'priority_score'])
    op.create_index('idx_outreach_campaign', 'proactive_outreaches', ['campaign_id', 'created_at'])
    op.create_index('idx_outreach_effectiveness', 'proactive_outreaches', ['effectiveness_score', 'outreach_type'])
    op.create_index('idx_outreach_followup', 'proactive_outreaches', ['follow_up_needed', 'next_follow_up_at'])


def downgrade():
    # Drop tables in reverse order
    op.drop_table('proactive_outreaches')
    op.drop_table('user_milestone_progress')
    op.drop_table('engagement_milestones')
    op.drop_table('user_behavior_patterns')
    op.drop_table('user_engagements')
    
    # Drop enums
    op.execute('DROP TYPE outreachstatus')
    op.execute('DROP TYPE outreachtype')
    op.execute('DROP TYPE sentimenttype')
    op.execute('DROP TYPE engagementtype')