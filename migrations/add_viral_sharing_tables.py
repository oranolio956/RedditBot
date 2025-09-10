"""Add viral sharing and referral tables

Revision ID: viral_sharing_001
Revises: Previous migration
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'viral_sharing_001'
down_revision = None  # Update this with your latest migration
branch_labels = None
depends_on = None


def upgrade():
    """Create viral sharing and referral tables."""
    
    # Create shareable_content table
    op.create_table(
        'shareable_content',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('content_type', sa.String(50), nullable=False, index=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('content_data', postgresql.JSONB(), nullable=False),
        sa.Column('image_url', sa.String(500), nullable=True),
        sa.Column('video_url', sa.String(500), nullable=True),
        sa.Column('viral_score', sa.Float(), default=0.0, nullable=False, index=True),
        sa.Column('hashtags', postgresql.JSON(), nullable=True),
        sa.Column('optimal_platforms', postgresql.JSON(), nullable=True),
        sa.Column('is_anonymized', sa.Boolean(), default=True, nullable=False),
        sa.Column('anonymization_level', sa.String(20), default='high', nullable=False),
        sa.Column('source_conversation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('source_user_anonymous_id', sa.String(50), nullable=True),
        sa.Column('view_count', sa.Integer(), default=0, nullable=False),
        sa.Column('share_count', sa.Integer(), default=0, nullable=False),
        sa.Column('like_count', sa.Integer(), default=0, nullable=False),
        sa.Column('comment_count', sa.Integer(), default=0, nullable=False),
        sa.Column('is_published', sa.Boolean(), default=False, nullable=False, index=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ai_enhancement_data', postgresql.JSONB(), nullable=True),
        
        # Indexes
        sa.Index('idx_content_viral_score', 'viral_score'),
        sa.Index('idx_content_type_published', 'content_type', 'is_published'),
        sa.Index('idx_content_published_viral', 'is_published', 'viral_score'),
        sa.Index('idx_content_expires', 'expires_at'),
        
        comment='Viral-optimized shareable content generated from bot interactions'
    )
    
    # Create content_shares table
    op.create_table(
        'content_shares',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sharer_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('sharer_anonymous_id', sa.String(50), nullable=True),
        sa.Column('platform', sa.String(20), nullable=False, index=True),
        sa.Column('share_format', sa.String(50), nullable=True),
        sa.Column('share_url', sa.String(500), nullable=True),
        sa.Column('share_text', sa.Text(), nullable=True),
        sa.Column('impressions', sa.Integer(), default=0, nullable=False),
        sa.Column('clicks', sa.Integer(), default=0, nullable=False),
        sa.Column('secondary_shares', sa.Integer(), default=0, nullable=False),
        sa.Column('referral_source', sa.String(100), nullable=True),
        sa.Column('campaign_id', sa.String(100), nullable=True),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['content_id'], ['shareable_content.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['sharer_user_id'], ['users.id'], ondelete='SET NULL'),
        
        # Indexes
        sa.Index('idx_share_platform_created', 'platform', 'created_at'),
        sa.Index('idx_share_content_platform', 'content_id', 'platform'),
        sa.Index('idx_share_performance', 'impressions', 'clicks', 'secondary_shares'),
        
        comment='Individual shares of viral content across social platforms'
    )
    
    # Create referral_programs table
    op.create_table(
        'referral_programs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('referrer_reward', postgresql.JSON(), nullable=False),
        sa.Column('referee_reward', postgresql.JSON(), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False, index=True),
        sa.Column('max_referrals_per_user', sa.Integer(), nullable=True),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=True),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('minimum_activity_level', sa.String(20), default='basic', nullable=False),
        sa.Column('total_referrals', sa.Integer(), default=0, nullable=False),
        sa.Column('successful_conversions', sa.Integer(), default=0, nullable=False),
        
        comment='Referral program configuration and tracking'
    )
    
    # Create user_referrals table
    op.create_table(
        'user_referrals',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('program_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('referrer_user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('referee_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('referee_contact', sa.String(200), nullable=True),
        sa.Column('referral_code', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('referral_method', sa.String(50), nullable=False),
        sa.Column('shared_content_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('first_click_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('signup_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('first_activity_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('conversion_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(20), default='pending', nullable=False, index=True),
        sa.Column('referrer_reward_given', sa.Boolean(), default=False, nullable=False),
        sa.Column('referee_reward_given', sa.Boolean(), default=False, nullable=False),
        sa.Column('click_count', sa.Integer(), default=0, nullable=False),
        sa.Column('attribution_data', postgresql.JSONB(), nullable=True),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['program_id'], ['referral_programs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['referrer_user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['referee_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['shared_content_id'], ['shareable_content.id'], ondelete='SET NULL'),
        
        # Indexes
        sa.Index('idx_referral_code', 'referral_code'),
        sa.Index('idx_referral_status_created', 'status', 'created_at'),
        sa.Index('idx_referral_program_status', 'program_id', 'status'),
        sa.Index('idx_referral_conversion_tracking', 'first_click_at', 'signup_at', 'conversion_at'),
        
        # Constraints
        sa.UniqueConstraint('referral_code', name='unique_referral_code'),
        
        comment='Individual user referrals and their conversion tracking'
    )
    
    # Create viral_metrics table
    op.create_table(
        'viral_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        sa.Column('date', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('period_type', sa.String(10), nullable=False, index=True),
        sa.Column('total_content_created', sa.Integer(), default=0, nullable=False),
        sa.Column('total_content_shared', sa.Integer(), default=0, nullable=False),
        sa.Column('total_views', sa.Integer(), default=0, nullable=False),
        sa.Column('total_shares', sa.Integer(), default=0, nullable=False),
        sa.Column('total_engagement', sa.Integer(), default=0, nullable=False),
        sa.Column('total_referrals_sent', sa.Integer(), default=0, nullable=False),
        sa.Column('total_referrals_converted', sa.Integer(), default=0, nullable=False),
        sa.Column('new_users_from_referrals', sa.Integer(), default=0, nullable=False),
        sa.Column('viral_coefficient', sa.Float(), default=0.0, nullable=False),
        sa.Column('average_shares_per_user', sa.Float(), default=0.0, nullable=False),
        sa.Column('top_performing_content_type', sa.String(50), nullable=True),
        sa.Column('top_performing_platform', sa.String(20), nullable=True),
        sa.Column('metrics_data', postgresql.JSONB(), nullable=True),
        
        # Indexes
        sa.Index('idx_metrics_date_period', 'date', 'period_type'),
        sa.Index('idx_metrics_viral_coefficient', 'viral_coefficient'),
        
        # Constraints
        sa.UniqueConstraint('date', 'period_type', name='unique_daily_metrics'),
        
        comment='Aggregate viral and growth metrics for analytics'
    )
    
    # Add foreign key constraint to shareable_content
    op.create_foreign_key(
        'fk_content_source_conversation',
        'shareable_content',
        'conversations',
        ['source_conversation_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade():
    """Drop viral sharing and referral tables."""
    
    # Drop foreign key constraints first
    op.drop_constraint('fk_content_source_conversation', 'shareable_content', type_='foreignkey')
    
    # Drop tables in reverse order of dependencies
    op.drop_table('viral_metrics')
    op.drop_table('user_referrals')
    op.drop_table('referral_programs')
    op.drop_table('content_shares')
    op.drop_table('shareable_content')