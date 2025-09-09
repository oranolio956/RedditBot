"""Add Stripe payment tables

Revision ID: 002_add_stripe_payment_tables
Revises: 001_initial_schema
Create Date: 2024-09-09 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_add_stripe_payment_tables'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create Stripe payment tables."""
    
    # Create Stripe customers table
    op.create_table('stripe_customers',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # User relationship
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  unique=True, nullable=False),
        
        # Stripe data
        sa.Column('stripe_customer_id', sa.String(255), unique=True, nullable=False),
        sa.Column('email', sa.String(255), nullable=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('phone', sa.String(50), nullable=True),
        
        # Customer settings
        sa.Column('default_payment_method_id', sa.String(255), nullable=True),
        sa.Column('invoice_prefix', sa.String(20), nullable=True),
        sa.Column('currency', sa.String(3), default='USD', nullable=False),
        sa.Column('timezone', sa.String(50), nullable=True),
        
        # Address information
        sa.Column('address_line1', sa.String(255), nullable=True),
        sa.Column('address_line2', sa.String(255), nullable=True),
        sa.Column('address_city', sa.String(100), nullable=True),
        sa.Column('address_state', sa.String(100), nullable=True),
        sa.Column('address_postal_code', sa.String(20), nullable=True),
        sa.Column('address_country', sa.String(2), nullable=True),
        
        # Customer portal
        sa.Column('portal_configuration_id', sa.String(255), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB(), default={}, nullable=True),
        
        # Indexes
        sa.Index('idx_stripe_customer_user_id', 'user_id'),
        sa.Index('idx_stripe_customer_stripe_id', 'stripe_customer_id'),
        sa.Index('idx_stripe_customer_email', 'email'),
        
        comment='Stripe customer records linked to users'
    )
    
    # Create Stripe subscriptions table
    op.create_table('stripe_subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Relationships
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  nullable=False),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('stripe_customers.id', ondelete='CASCADE'), nullable=False),
        
        # Stripe data
        sa.Column('stripe_subscription_id', sa.String(255), unique=True, nullable=False),
        sa.Column('stripe_product_id', sa.String(255), nullable=False),
        sa.Column('stripe_price_id', sa.String(255), nullable=False),
        
        # Subscription details
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('trial_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trial_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancel_at_period_end', sa.Boolean(), default=False, nullable=False),
        sa.Column('canceled_at', sa.DateTime(timezone=True), nullable=True),
        
        # Pricing
        sa.Column('currency', sa.String(3), default='USD', nullable=False),
        sa.Column('interval_type', sa.String(20), nullable=False),
        sa.Column('interval_count', sa.Integer(), default=1, nullable=False),
        sa.Column('unit_amount', sa.Integer(), nullable=False),
        
        # Service configuration
        sa.Column('tier_name', sa.String(50), nullable=False),
        sa.Column('features', postgresql.JSONB(), default=[], nullable=True),
        sa.Column('usage_limits', postgresql.JSONB(), default={}, nullable=True),
        
        # Billing
        sa.Column('collection_method', sa.String(20), default='charge_automatically', nullable=False),
        sa.Column('days_until_due', sa.Integer(), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB(), default={}, nullable=True),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('active', 'trialing', 'past_due', 'canceled', 'unpaid', 'incomplete', 'incomplete_expired')",
            name='ck_subscription_status'
        ),
        sa.CheckConstraint(
            "interval_type IN ('day', 'week', 'month', 'year')",
            name='ck_subscription_interval_type'
        ),
        sa.CheckConstraint('unit_amount > 0', name='ck_subscription_amount_positive'),
        sa.CheckConstraint('interval_count > 0', name='ck_subscription_interval_positive'),
        
        # Indexes
        sa.Index('idx_subscription_user_id', 'user_id'),
        sa.Index('idx_subscription_customer_id', 'customer_id'),
        sa.Index('idx_subscription_status', 'status'),
        sa.Index('idx_subscription_period', 'current_period_start', 'current_period_end'),
        sa.Index('idx_subscription_stripe_id', 'stripe_subscription_id'),
        sa.Index('idx_subscription_tier', 'tier_name'),
        
        comment='Stripe subscription records'
    )
    
    # Create Stripe payment methods table
    op.create_table('stripe_payment_methods',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Relationships
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  nullable=False),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('stripe_customers.id', ondelete='CASCADE'), nullable=False),
        
        # Stripe data
        sa.Column('stripe_payment_method_id', sa.String(255), unique=True, nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        
        # Card details (if applicable)
        sa.Column('card_brand', sa.String(20), nullable=True),
        sa.Column('card_last4', sa.String(4), nullable=True),
        sa.Column('card_exp_month', sa.Integer(), nullable=True),
        sa.Column('card_exp_year', sa.Integer(), nullable=True),
        sa.Column('card_country', sa.String(2), nullable=True),
        
        # Bank account details (if applicable)
        sa.Column('bank_account_bank_name', sa.String(255), nullable=True),
        sa.Column('bank_account_last4', sa.String(4), nullable=True),
        sa.Column('bank_account_account_type', sa.String(20), nullable=True),
        
        # Status
        sa.Column('is_default', sa.Boolean(), default=False, nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB(), default={}, nullable=True),
        
        # Constraints
        sa.CheckConstraint(
            "type IN ('card', 'us_bank_account', 'sepa_debit', 'ach_debit')",
            name='ck_payment_method_type'
        ),
        
        # Indexes
        sa.Index('idx_payment_method_user_id', 'user_id'),
        sa.Index('idx_payment_method_customer_id', 'customer_id'),
        sa.Index('idx_payment_method_default', 'customer_id', 'is_default'),
        sa.Index('idx_payment_method_stripe_id', 'stripe_payment_method_id'),
        
        comment='Stripe payment method records'
    )
    
    # Create Stripe invoices table
    op.create_table('stripe_invoices',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Relationships
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  nullable=False),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('stripe_customers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('subscription_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('stripe_subscriptions.id', ondelete='SET NULL'), nullable=True),
        
        # Stripe data
        sa.Column('stripe_invoice_id', sa.String(255), unique=True, nullable=False),
        sa.Column('stripe_charge_id', sa.String(255), nullable=True),
        sa.Column('stripe_payment_intent_id', sa.String(255), nullable=True),
        
        # Invoice details
        sa.Column('invoice_number', sa.String(100), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('currency', sa.String(3), default='USD', nullable=False),
        sa.Column('amount_total', sa.Integer(), nullable=False),
        sa.Column('amount_paid', sa.Integer(), default=0, nullable=False),
        sa.Column('amount_due', sa.Integer(), default=0, nullable=False),
        sa.Column('subtotal', sa.Integer(), nullable=False),
        sa.Column('tax_amount', sa.Integer(), default=0, nullable=False),
        
        # Dates
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finalized_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        
        # URLs
        sa.Column('invoice_pdf_url', sa.Text(), nullable=True),
        sa.Column('hosted_invoice_url', sa.Text(), nullable=True),
        
        # Payment attempt tracking
        sa.Column('attempt_count', sa.Integer(), default=0, nullable=False),
        sa.Column('next_payment_attempt', sa.DateTime(timezone=True), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB(), default={}, nullable=True),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('draft', 'open', 'paid', 'void', 'uncollectible')",
            name='ck_invoice_status'
        ),
        sa.CheckConstraint('amount_total >= 0', name='ck_invoice_total_positive'),
        sa.CheckConstraint('amount_paid >= 0', name='ck_invoice_paid_positive'),
        sa.CheckConstraint('amount_due >= 0', name='ck_invoice_due_positive'),
        sa.CheckConstraint('subtotal >= 0', name='ck_invoice_subtotal_positive'),
        sa.CheckConstraint('tax_amount >= 0', name='ck_invoice_tax_positive'),
        sa.CheckConstraint('attempt_count >= 0', name='ck_invoice_attempts_positive'),
        
        # Indexes
        sa.Index('idx_invoice_user_id', 'user_id'),
        sa.Index('idx_invoice_customer_id', 'customer_id'),
        sa.Index('idx_invoice_subscription_id', 'subscription_id'),
        sa.Index('idx_invoice_status', 'status'),
        sa.Index('idx_invoice_due_date', 'due_date'),
        sa.Index('idx_invoice_period', 'period_start', 'period_end'),
        sa.Index('idx_invoice_stripe_id', 'stripe_invoice_id'),
        
        comment='Stripe invoice records'
    )
    
    # Create payment attempts table
    op.create_table('payment_attempts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Relationships
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  nullable=False),
        sa.Column('subscription_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('stripe_subscriptions.id', ondelete='CASCADE'), nullable=True),
        sa.Column('invoice_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('stripe_invoices.id', ondelete='CASCADE'), nullable=True),
        
        # Attempt details
        sa.Column('attempt_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('failure_code', sa.String(100), nullable=True),
        sa.Column('failure_message', sa.Text(), nullable=True),
        
        # Payment method used
        sa.Column('payment_method_type', sa.String(50), nullable=True),
        sa.Column('payment_method_last4', sa.String(4), nullable=True),
        
        # Amounts
        sa.Column('amount_attempted', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(3), default='USD', nullable=False),
        
        # Stripe IDs
        sa.Column('stripe_payment_intent_id', sa.String(255), nullable=True),
        sa.Column('stripe_charge_id', sa.String(255), nullable=True),
        
        # Next attempt scheduling
        sa.Column('next_attempt_at', sa.DateTime(timezone=True), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB(), default={}, nullable=True),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed', 'pending', 'requires_action', 'canceled')",
            name='ck_payment_attempt_status'
        ),
        sa.CheckConstraint('attempt_number > 0', name='ck_attempt_number_positive'),
        sa.CheckConstraint('amount_attempted >= 0', name='ck_amount_attempted_positive'),
        
        # Indexes
        sa.Index('idx_attempt_user_status', 'user_id', 'status'),
        sa.Index('idx_attempt_subscription', 'subscription_id'),
        sa.Index('idx_attempt_invoice', 'invoice_id'),
        sa.Index('idx_attempt_next_attempt', 'next_attempt_at'),
        sa.Index('idx_attempt_created', 'created_at'),
        
        comment='Payment attempt tracking'
    )
    
    # Create webhook events table
    op.create_table('stripe_webhook_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        
        # Stripe webhook data
        sa.Column('stripe_event_id', sa.String(255), unique=True, nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('api_version', sa.String(20), nullable=True),
        
        # Processing status
        sa.Column('status', sa.String(50), default='pending', nullable=False),
        sa.Column('retry_count', sa.Integer(), default=0, nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Event data
        sa.Column('data', postgresql.JSONB(), nullable=False),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'skipped')",
            name='ck_webhook_event_status'
        ),
        sa.CheckConstraint('retry_count >= 0', name='ck_retry_count_positive'),
        
        # Indexes
        sa.Index('idx_webhook_event_type', 'event_type'),
        sa.Index('idx_webhook_status', 'status'),
        sa.Index('idx_webhook_created', 'created_at'),
        sa.Index('idx_webhook_stripe_id', 'stripe_event_id'),
        
        comment='Stripe webhook event processing log'
    )
    
    print("✅ Stripe payment tables created successfully")


def downgrade() -> None:
    """Drop Stripe payment tables."""
    
    # Drop tables in reverse dependency order
    tables_to_drop = [
        'stripe_webhook_events',
        'payment_attempts',
        'stripe_invoices',
        'stripe_payment_methods', 
        'stripe_subscriptions',
        'stripe_customers'
    ]
    
    for table in tables_to_drop:
        op.drop_table(table)
    
    print("✅ Stripe payment tables dropped successfully")