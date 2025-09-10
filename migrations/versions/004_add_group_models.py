"""
Add group chat models

Revision ID: 004
Revises: 003
Create Date: 2024-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade():
    """Add group chat functionality tables."""
    
    # Create enums
    op.execute("CREATE TYPE group_type AS ENUM ('private_group', 'public_group', 'supergroup', 'channel')")
    op.execute("CREATE TYPE member_role AS ENUM ('member', 'admin', 'creator', 'restricted', 'left', 'banned')")
    op.execute("CREATE TYPE group_status AS ENUM ('active', 'paused', 'restricted', 'archived')")
    op.execute("CREATE TYPE message_frequency AS ENUM ('low', 'moderate', 'high', 'very_high')")
    
    # Create group_sessions table
    op.create_table(
        'group_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Group identification
        sa.Column('telegram_chat_id', sa.BigInteger(), nullable=False, unique=True),
        sa.Column('group_type', postgresql.ENUM('private_group', 'public_group', 'supergroup', 'channel', name='group_type'), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('username', sa.String(32), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        
        # Group status and settings
        sa.Column('status', postgresql.ENUM('active', 'paused', 'restricted', 'archived', name='group_status'), 
                 nullable=False, server_default='active'),
        sa.Column('is_bot_admin', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('bot_permissions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        
        # Activity tracking
        sa.Column('member_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('active_member_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_messages', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('bot_mentions', sa.Integer(), nullable=False, server_default='0'),
        
        # Engagement metrics
        sa.Column('conversation_topics', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=list),
        sa.Column('engagement_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('sentiment_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('message_frequency', postgresql.ENUM('low', 'moderate', 'high', 'very_high', name='message_frequency'), 
                 nullable=False, server_default='low'),
        
        # Settings
        sa.Column('group_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('moderation_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('language_preferences', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('timezone', sa.String(50), nullable=True),
        
        # Rate limiting
        sa.Column('rate_limit_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('spam_detection_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('recent_violations', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=list),
        
        # Activity timing
        sa.Column('first_interaction', sa.DateTime(), nullable=True),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.Column('last_message_at', sa.DateTime(), nullable=True),
        
        # Analytics data
        sa.Column('peak_activity_hours', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('member_interaction_patterns', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('conversation_flow_state', sa.String(100), nullable=True),
    )
    
    # Create group_members table
    op.create_table(
        'group_members',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Relationships
        sa.Column('group_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('group_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('telegram_user_id', sa.BigInteger(), nullable=False),
        
        # Member info
        sa.Column('role', postgresql.ENUM('member', 'admin', 'creator', 'restricted', 'left', 'banned', name='member_role'), 
                 nullable=False, server_default='member'),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        
        # Activity metrics
        sa.Column('message_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('mention_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_message_at', sa.DateTime(), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(), nullable=True),
        
        # Engagement
        sa.Column('interaction_frequency', postgresql.ENUM('low', 'moderate', 'high', 'very_high', name='member_frequency'), 
                 nullable=False, server_default='low'),
        sa.Column('engagement_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('influence_score', sa.Float(), nullable=False, server_default='0.0'),
        
        # Context and preferences
        sa.Column('member_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('notification_preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('conversation_topics', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=list),
        sa.Column('interaction_patterns', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('sentiment_profile', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Risk and moderation
        sa.Column('risk_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('violation_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_violation_at', sa.DateTime(), nullable=True),
        sa.Column('moderation_notes', sa.Text(), nullable=True),
        sa.Column('permissions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('restrictions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Timing
        sa.Column('joined_at', sa.DateTime(), nullable=True),
        sa.Column('left_at', sa.DateTime(), nullable=True),
        
        # Constraints
        sa.UniqueConstraint('group_id', 'user_id', name='uq_group_member'),
    )
    
    # Create group_conversations table
    op.create_table(
        'group_conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Relationships
        sa.Column('group_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('group_sessions.id', ondelete='CASCADE'), nullable=False),
        
        # Conversation metadata
        sa.Column('thread_id', sa.String(64), nullable=False),
        sa.Column('topic', sa.String(255), nullable=True),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        
        # Participation tracking
        sa.Column('participant_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('participant_ids', postgresql.ARRAY(sa.BigInteger), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('bot_interactions', sa.Integer(), nullable=False, server_default='0'),
        
        # Analytics
        sa.Column('engagement_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('sentiment_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('complexity_score', sa.Float(), nullable=True),
        sa.Column('toxicity_score', sa.Float(), nullable=False, server_default='0.0'),
        
        # Content analysis
        sa.Column('keywords', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('language_distribution', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Timing
        sa.Column('started_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('last_message_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=False, server_default='0'),
        
        # Context
        sa.Column('conversation_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=dict),
        sa.Column('moderator_actions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default=list),
    )
    
    # Create group_analytics table
    op.create_table(
        'group_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Relationships
        sa.Column('group_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('group_sessions.id', ondelete='CASCADE'), nullable=False),
        
        # Time period
        sa.Column('period_type', sa.String(20), nullable=False),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        
        # Activity metrics
        sa.Column('total_messages', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('unique_participants', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('bot_interactions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('conversation_threads', sa.Integer(), nullable=False, server_default='0'),
        
        # Engagement metrics
        sa.Column('average_engagement', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('peak_activity_hour', sa.Integer(), nullable=True),
        sa.Column('activity_distribution', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Content analysis
        sa.Column('sentiment_distribution', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('topic_distribution', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('language_usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Member analytics
        sa.Column('new_members', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('members_left', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('active_member_ratio', sa.Float(), nullable=False, server_default='0.0'),
        
        # Performance metrics
        sa.Column('response_time_avg', sa.Float(), nullable=True),
        sa.Column('violations_detected', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('moderation_actions', sa.Integer(), nullable=False, server_default='0'),
        
        # Growth metrics
        sa.Column('growth_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('trend_direction', sa.String(20), nullable=True),
        
        # Insights
        sa.Column('insights', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('anomalies', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    
    # Create indexes for group_sessions
    op.create_index('idx_group_chat_id', 'group_sessions', ['telegram_chat_id'])
    op.create_index('idx_group_status_activity', 'group_sessions', ['status', 'last_activity'])
    op.create_index('idx_group_type_members', 'group_sessions', ['group_type', 'member_count'])
    op.create_index('idx_group_engagement', 'group_sessions', ['engagement_score'])
    
    # Create indexes for group_members
    op.create_index('idx_member_group_role', 'group_members', ['group_id', 'role'])
    op.create_index('idx_member_activity', 'group_members', ['is_active', 'last_seen_at'])
    op.create_index('idx_member_engagement', 'group_members', ['engagement_score'])
    op.create_index('idx_member_risk', 'group_members', ['risk_score'])
    op.create_index('idx_member_telegram_user', 'group_members', ['telegram_user_id'])
    
    # Create indexes for group_conversations
    op.create_index('idx_group_conv_thread', 'group_conversations', ['group_id', 'thread_id'])
    op.create_index('idx_group_conv_topic', 'group_conversations', ['group_id', 'topic'])
    op.create_index('idx_group_conv_timing', 'group_conversations', ['started_at', 'last_message_at'])
    op.create_index('idx_group_conv_engagement', 'group_conversations', ['engagement_score'])
    op.create_index('idx_group_conv_toxicity', 'group_conversations', ['toxicity_score'])
    
    # Create indexes for group_analytics
    op.create_index('idx_analytics_group_period', 'group_analytics', ['group_id', 'period_type', 'period_start'])
    op.create_index('idx_analytics_engagement', 'group_analytics', ['average_engagement'])
    op.create_index('idx_analytics_growth', 'group_analytics', ['growth_rate'])
    
    # Add check constraints for group_sessions
    op.create_check_constraint('ck_group_member_count_positive', 'group_sessions', 'member_count >= 0')
    op.create_check_constraint('ck_group_active_members_positive', 'group_sessions', 'active_member_count >= 0')
    op.create_check_constraint('ck_group_active_members_valid', 'group_sessions', 'active_member_count <= member_count')
    op.create_check_constraint('ck_group_engagement_range', 'group_sessions', 'engagement_score >= 0 AND engagement_score <= 1')
    op.create_check_constraint('ck_group_total_messages_positive', 'group_sessions', 'total_messages >= 0')
    op.create_check_constraint('ck_group_bot_mentions_positive', 'group_sessions', 'bot_mentions >= 0')
    
    # Add check constraints for group_members
    op.create_check_constraint('ck_member_message_count_positive', 'group_members', 'message_count >= 0')
    op.create_check_constraint('ck_member_mention_count_positive', 'group_members', 'mention_count >= 0')
    op.create_check_constraint('ck_member_engagement_range', 'group_members', 'engagement_score >= 0 AND engagement_score <= 1')
    op.create_check_constraint('ck_member_influence_range', 'group_members', 'influence_score >= 0 AND influence_score <= 1')
    op.create_check_constraint('ck_member_risk_range', 'group_members', 'risk_score >= 0 AND risk_score <= 1')
    op.create_check_constraint('ck_member_violations_positive', 'group_members', 'violation_count >= 0')
    
    # Add check constraints for group_conversations
    op.create_check_constraint('ck_group_conv_participants_positive', 'group_conversations', 'participant_count >= 0')
    op.create_check_constraint('ck_group_conv_messages_positive', 'group_conversations', 'message_count >= 0')
    op.create_check_constraint('ck_group_conv_bot_interactions_positive', 'group_conversations', 'bot_interactions >= 0')
    op.create_check_constraint('ck_group_conv_engagement_range', 'group_conversations', 'engagement_score >= 0 AND engagement_score <= 1')
    op.create_check_constraint('ck_group_conv_complexity_range', 'group_conversations', 'complexity_score >= 0 AND complexity_score <= 1')
    op.create_check_constraint('ck_group_conv_toxicity_range', 'group_conversations', 'toxicity_score >= 0 AND toxicity_score <= 1')
    op.create_check_constraint('ck_group_conv_duration_positive', 'group_conversations', 'duration_seconds >= 0')
    
    # Add check constraints for group_analytics
    op.create_check_constraint('ck_analytics_total_messages_positive', 'group_analytics', 'total_messages >= 0')
    op.create_check_constraint('ck_analytics_participants_positive', 'group_analytics', 'unique_participants >= 0')
    op.create_check_constraint('ck_analytics_bot_interactions_positive', 'group_analytics', 'bot_interactions >= 0')
    op.create_check_constraint('ck_analytics_threads_positive', 'group_analytics', 'conversation_threads >= 0')
    op.create_check_constraint('ck_analytics_new_members_positive', 'group_analytics', 'new_members >= 0')
    op.create_check_constraint('ck_analytics_left_members_positive', 'group_analytics', 'members_left >= 0')
    op.create_check_constraint('ck_analytics_violations_positive', 'group_analytics', 'violations_detected >= 0')
    op.create_check_constraint('ck_analytics_mod_actions_positive', 'group_analytics', 'moderation_actions >= 0')
    op.create_check_constraint('ck_analytics_engagement_range', 'group_analytics', 'average_engagement >= 0 AND average_engagement <= 1')
    op.create_check_constraint('ck_analytics_ratio_range', 'group_analytics', 'active_member_ratio >= 0 AND active_member_ratio <= 1')
    op.create_check_constraint('ck_analytics_hour_range', 'group_analytics', 'peak_activity_hour >= 0 AND peak_activity_hour <= 23')


def downgrade():
    """Remove group chat functionality."""
    
    # Drop tables in reverse order
    op.drop_table('group_analytics')
    op.drop_table('group_conversations')
    op.drop_table('group_members')
    op.drop_table('group_sessions')
    
    # Drop enums
    op.execute("DROP TYPE IF EXISTS message_frequency")
    op.execute("DROP TYPE IF EXISTS group_status") 
    op.execute("DROP TYPE IF EXISTS member_role")
    op.execute("DROP TYPE IF EXISTS group_type")