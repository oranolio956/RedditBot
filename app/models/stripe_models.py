"""
Stripe Payment Models

Database models for Stripe payment integration including customers,
subscriptions, invoices, payment methods, and webhook events.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Column, String, Integer, BigInteger, Boolean, Text, JSON, Index, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy import DateTime

from app.database.base import FullAuditModel


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration matching Stripe values."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"


class InvoiceStatus(str, Enum):
    """Invoice status enumeration matching Stripe values."""
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"


class PaymentAttemptStatus(str, Enum):
    """Payment attempt status enumeration."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PENDING = "pending"
    REQUIRES_ACTION = "requires_action"
    CANCELED = "canceled"


class WebhookEventStatus(str, Enum):
    """Webhook event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StripeCustomer(FullAuditModel):
    """
    Stripe customer model for storing customer information.
    
    Links Telegram users to Stripe customers for payment processing.
    """
    
    __tablename__ = "stripe_customers"
    
    # User relationship
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Reference to users table"
    )
    
    # Stripe data
    stripe_customer_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Stripe customer ID (cus_...)"
    )
    
    email = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Customer email address"
    )
    
    name = Column(
        String(255),
        nullable=True,
        comment="Customer full name"
    )
    
    phone = Column(
        String(50),
        nullable=True,
        comment="Customer phone number"
    )
    
    # Customer settings
    default_payment_method_id = Column(
        String(255),
        nullable=True,
        comment="Default payment method ID"
    )
    
    invoice_prefix = Column(
        String(20),
        nullable=True,
        comment="Custom invoice number prefix"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        comment="Customer's default currency"
    )
    
    timezone = Column(
        String(50),
        nullable=True,
        comment="Customer timezone for billing"
    )
    
    # Address information
    address_line1 = Column(
        String(255),
        nullable=True,
        comment="Address line 1"
    )
    
    address_line2 = Column(
        String(255),
        nullable=True,
        comment="Address line 2"
    )
    
    address_city = Column(
        String(100),
        nullable=True,
        comment="City"
    )
    
    address_state = Column(
        String(100),
        nullable=True,
        comment="State or province"
    )
    
    address_postal_code = Column(
        String(20),
        nullable=True,
        comment="Postal/ZIP code"
    )
    
    address_country = Column(
        String(2),
        nullable=True,
        comment="Country code (ISO 3166-1 alpha-2)"
    )
    
    # Customer portal
    portal_configuration_id = Column(
        String(255),
        nullable=True,
        comment="Custom portal configuration ID"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        default=dict,
        nullable=True,
        comment="Additional metadata from Stripe"
    )
    
    # Relationships
    user = relationship("User", back_populates="stripe_customer")
    subscriptions = relationship("StripeSubscription", back_populates="customer", cascade="all, delete-orphan")
    invoices = relationship("StripeInvoice", back_populates="customer", cascade="all, delete-orphan")
    payment_methods = relationship("StripePaymentMethod", back_populates="customer", cascade="all, delete-orphan")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_stripe_customer_user_id', 'user_id'),
        Index('idx_stripe_customer_stripe_id', 'stripe_customer_id'),
        Index('idx_stripe_customer_email', 'email'),
        {"comment": "Stripe customer records linked to users"}
    )
    
    def get_full_address(self) -> Optional[Dict[str, str]]:
        """Get complete address as dictionary."""
        if not self.address_line1:
            return None
            
        address = {
            "line1": self.address_line1,
            "city": self.address_city,
            "country": self.address_country
        }
        
        if self.address_line2:
            address["line2"] = self.address_line2
        if self.address_state:
            address["state"] = self.address_state
        if self.address_postal_code:
            address["postal_code"] = self.address_postal_code
            
        return address
    
    def __str__(self) -> str:
        return f"StripeCustomer({self.stripe_customer_id}, email={self.email})"


class StripeSubscription(FullAuditModel):
    """
    Stripe subscription model for managing recurring billing.
    
    Tracks subscription status, billing periods, and service configuration.
    """
    
    __tablename__ = "stripe_subscriptions"
    
    # Relationships
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to users table"
    )
    
    customer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("stripe_customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to stripe_customers table"
    )
    
    # Stripe data
    stripe_subscription_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Stripe subscription ID (sub_...)"
    )
    
    stripe_product_id = Column(
        String(255),
        nullable=False,
        comment="Stripe product ID"
    )
    
    stripe_price_id = Column(
        String(255),
        nullable=False,
        comment="Stripe price ID"
    )
    
    # Subscription details
    status = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Subscription status"
    )
    
    current_period_start = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Current billing period start"
    )
    
    current_period_end = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Current billing period end"
    )
    
    trial_start = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Trial period start"
    )
    
    trial_end = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Trial period end"
    )
    
    cancel_at_period_end = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether to cancel at period end"
    )
    
    canceled_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Cancellation timestamp"
    )
    
    # Pricing
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        comment="Subscription currency"
    )
    
    interval_type = Column(
        String(20),
        nullable=False,
        comment="Billing interval (day, week, month, year)"
    )
    
    interval_count = Column(
        Integer,
        default=1,
        nullable=False,
        comment="Number of intervals between billings"
    )
    
    unit_amount = Column(
        Integer,
        nullable=False,
        comment="Amount in cents"
    )
    
    # Service configuration
    tier_name = Column(
        String(50),
        nullable=False,
        comment="Subscription tier (basic, premium, enterprise)"
    )
    
    features = Column(
        JSONB,
        default=list,
        nullable=True,
        comment="List of included features"
    )
    
    usage_limits = Column(
        JSONB,
        default=dict,
        nullable=True,
        comment="Service usage limits"
    )
    
    # Billing
    collection_method = Column(
        String(20),
        default="charge_automatically",
        nullable=False,
        comment="How to collect payment"
    )
    
    days_until_due = Column(
        Integer,
        nullable=True,
        comment="Days until payment is due"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        default=dict,
        nullable=True,
        comment="Additional metadata from Stripe"
    )
    
    # Relationships
    user = relationship("User", back_populates="stripe_subscriptions")
    customer = relationship("StripeCustomer", back_populates="subscriptions")
    invoices = relationship("StripeInvoice", back_populates="subscription", cascade="all, delete-orphan")
    payment_attempts = relationship("PaymentAttempt", back_populates="subscription", cascade="all, delete-orphan")
    
    # Database constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'trialing', 'past_due', 'canceled', 'unpaid', 'incomplete', 'incomplete_expired')",
            name='ck_subscription_status'
        ),
        CheckConstraint(
            "interval_type IN ('day', 'week', 'month', 'year')",
            name='ck_subscription_interval_type'
        ),
        CheckConstraint('unit_amount > 0', name='ck_subscription_amount_positive'),
        CheckConstraint('interval_count > 0', name='ck_subscription_interval_positive'),
        Index('idx_subscription_user_id', 'user_id'),
        Index('idx_subscription_status', 'status'),
        Index('idx_subscription_period', 'current_period_start', 'current_period_end'),
        Index('idx_subscription_stripe_id', 'stripe_subscription_id'),
        Index('idx_subscription_tier', 'tier_name'),
        {"comment": "Stripe subscription records"}
    )
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
    
    @property
    def is_trial(self) -> bool:
        """Check if subscription is in trial period."""
        return self.status == SubscriptionStatus.TRIALING
    
    @property
    def days_until_renewal(self) -> int:
        """Calculate days until next renewal."""
        if not self.current_period_end:
            return 0
        delta = self.current_period_end.date() - datetime.utcnow().date()
        return max(0, delta.days)
    
    def get_amount_display(self) -> str:
        """Get formatted amount for display."""
        amount = self.unit_amount / 100
        interval_display = f"/{self.interval_type}"
        if self.interval_count > 1:
            interval_display = f"/{self.interval_count} {self.interval_type}s"
        return f"${amount:.2f}{interval_display}"
    
    def __str__(self) -> str:
        return f"StripeSubscription({self.stripe_subscription_id}, tier={self.tier_name}, status={self.status})"


class StripePaymentMethod(FullAuditModel):
    """
    Stripe payment method model for stored payment instruments.
    
    Tracks customer payment methods (cards, bank accounts, etc.).
    """
    
    __tablename__ = "stripe_payment_methods"
    
    # Relationships
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to users table"
    )
    
    customer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("stripe_customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to stripe_customers table"
    )
    
    # Stripe data
    stripe_payment_method_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Stripe payment method ID (pm_...)"
    )
    
    type = Column(
        String(50),
        nullable=False,
        comment="Payment method type (card, us_bank_account, etc.)"
    )
    
    # Card details (if applicable)
    card_brand = Column(
        String(20),
        nullable=True,
        comment="Card brand (visa, mastercard, etc.)"
    )
    
    card_last4 = Column(
        String(4),
        nullable=True,
        comment="Last 4 digits of card number"
    )
    
    card_exp_month = Column(
        Integer,
        nullable=True,
        comment="Card expiration month"
    )
    
    card_exp_year = Column(
        Integer,
        nullable=True,
        comment="Card expiration year"
    )
    
    card_country = Column(
        String(2),
        nullable=True,
        comment="Card issuing country"
    )
    
    # Bank account details (if applicable)
    bank_account_bank_name = Column(
        String(255),
        nullable=True,
        comment="Bank name"
    )
    
    bank_account_last4 = Column(
        String(4),
        nullable=True,
        comment="Last 4 digits of account number"
    )
    
    bank_account_account_type = Column(
        String(20),
        nullable=True,
        comment="Account type (checking, savings)"
    )
    
    # Status
    is_default = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether this is the default payment method"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether payment method is active"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        default=dict,
        nullable=True,
        comment="Additional metadata from Stripe"
    )
    
    # Relationships
    user = relationship("User")
    customer = relationship("StripeCustomer", back_populates="payment_methods")
    
    # Database constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "type IN ('card', 'us_bank_account', 'sepa_debit', 'ach_debit')",
            name='ck_payment_method_type'
        ),
        Index('idx_payment_method_user_id', 'user_id'),
        Index('idx_payment_method_customer_id', 'customer_id'),
        Index('idx_payment_method_default', 'customer_id', 'is_default'),
        Index('idx_payment_method_stripe_id', 'stripe_payment_method_id'),
        {"comment": "Stripe payment method records"}
    )
    
    def get_display_name(self) -> str:
        """Get user-friendly display name for payment method."""
        if self.type == "card":
            brand = self.card_brand.title() if self.card_brand else "Card"
            return f"{brand} •••• {self.card_last4}"
        elif self.type == "us_bank_account":
            bank = self.bank_account_bank_name or "Bank Account"
            return f"{bank} •••• {self.bank_account_last4}"
        else:
            return f"{self.type.title()} Payment Method"
    
    @property
    def is_card_expired(self) -> bool:
        """Check if card is expired (for card payment methods)."""
        if self.type != "card" or not self.card_exp_month or not self.card_exp_year:
            return False
        
        now = datetime.utcnow()
        return (self.card_exp_year < now.year or 
                (self.card_exp_year == now.year and self.card_exp_month < now.month))
    
    def __str__(self) -> str:
        return f"StripePaymentMethod({self.stripe_payment_method_id}, {self.get_display_name()})"


class StripeInvoice(FullAuditModel):
    """
    Stripe invoice model for billing records.
    
    Tracks invoices, payments, and billing history.
    """
    
    __tablename__ = "stripe_invoices"
    
    # Relationships
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to users table"
    )
    
    customer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("stripe_customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to stripe_customers table"
    )
    
    subscription_id = Column(
        UUID(as_uuid=True),
        ForeignKey("stripe_subscriptions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Reference to stripe_subscriptions table"
    )
    
    # Stripe data
    stripe_invoice_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Stripe invoice ID (in_...)"
    )
    
    stripe_charge_id = Column(
        String(255),
        nullable=True,
        comment="Stripe charge ID (ch_...)"
    )
    
    stripe_payment_intent_id = Column(
        String(255),
        nullable=True,
        comment="Stripe payment intent ID (pi_...)"
    )
    
    # Invoice details
    invoice_number = Column(
        String(100),
        nullable=True,
        comment="Human-readable invoice number"
    )
    
    status = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Invoice status"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        comment="Invoice currency"
    )
    
    amount_total = Column(
        Integer,
        nullable=False,
        comment="Total amount in cents"
    )
    
    amount_paid = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Amount paid in cents"
    )
    
    amount_due = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Amount due in cents"
    )
    
    subtotal = Column(
        Integer,
        nullable=False,
        comment="Subtotal before tax in cents"
    )
    
    tax_amount = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Tax amount in cents"
    )
    
    # Dates
    period_start = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Billing period start"
    )
    
    period_end = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Billing period end"
    )
    
    due_date = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Payment due date"
    )
    
    finalized_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When invoice was finalized"
    )
    
    paid_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When invoice was paid"
    )
    
    # URLs
    invoice_pdf_url = Column(
        Text,
        nullable=True,
        comment="URL to PDF version of invoice"
    )
    
    hosted_invoice_url = Column(
        Text,
        nullable=True,
        comment="URL to hosted invoice page"
    )
    
    # Payment attempt tracking
    attempt_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of payment attempts"
    )
    
    next_payment_attempt = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Next scheduled payment attempt"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        default=dict,
        nullable=True,
        comment="Additional metadata from Stripe"
    )
    
    # Relationships
    user = relationship("User")
    customer = relationship("StripeCustomer", back_populates="invoices")
    subscription = relationship("StripeSubscription", back_populates="invoices")
    payment_attempts = relationship("PaymentAttempt", back_populates="invoice", cascade="all, delete-orphan")
    
    # Database constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'open', 'paid', 'void', 'uncollectible')",
            name='ck_invoice_status'
        ),
        CheckConstraint('amount_total >= 0', name='ck_invoice_total_positive'),
        CheckConstraint('amount_paid >= 0', name='ck_invoice_paid_positive'),
        CheckConstraint('amount_due >= 0', name='ck_invoice_due_positive'),
        CheckConstraint('subtotal >= 0', name='ck_invoice_subtotal_positive'),
        CheckConstraint('tax_amount >= 0', name='ck_invoice_tax_positive'),
        CheckConstraint('attempt_count >= 0', name='ck_invoice_attempts_positive'),
        Index('idx_invoice_user_id', 'user_id'),
        Index('idx_invoice_status', 'status'),
        Index('idx_invoice_due_date', 'due_date'),
        Index('idx_invoice_period', 'period_start', 'period_end'),
        Index('idx_invoice_stripe_id', 'stripe_invoice_id'),
        {"comment": "Stripe invoice records"}
    )
    
    @property
    def is_paid(self) -> bool:
        """Check if invoice is fully paid."""
        return self.status == InvoiceStatus.PAID
    
    @property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        if not self.due_date or self.is_paid:
            return False
        return datetime.utcnow() > self.due_date
    
    def get_amount_display(self) -> str:
        """Get formatted total amount for display."""
        return f"${self.amount_total / 100:.2f}"
    
    def get_status_display(self) -> str:
        """Get user-friendly status display."""
        status_map = {
            InvoiceStatus.DRAFT: "Draft",
            InvoiceStatus.OPEN: "Unpaid",
            InvoiceStatus.PAID: "Paid",
            InvoiceStatus.VOID: "Void",
            InvoiceStatus.UNCOLLECTIBLE: "Uncollectible"
        }
        return status_map.get(self.status, self.status.title())
    
    def __str__(self) -> str:
        return f"StripeInvoice({self.stripe_invoice_id}, {self.get_amount_display()}, {self.status})"


class PaymentAttempt(FullAuditModel):
    """
    Payment attempt tracking for retry logic and analytics.
    
    Records all payment attempts including successes and failures.
    """
    
    __tablename__ = "payment_attempts"
    
    # Relationships
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to users table"
    )
    
    subscription_id = Column(
        UUID(as_uuid=True),
        ForeignKey("stripe_subscriptions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Reference to stripe_subscriptions table"
    )
    
    invoice_id = Column(
        UUID(as_uuid=True),
        ForeignKey("stripe_invoices.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Reference to stripe_invoices table"
    )
    
    # Attempt details
    attempt_number = Column(
        Integer,
        nullable=False,
        comment="Attempt number for this payment"
    )
    
    status = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Attempt status"
    )
    
    failure_code = Column(
        String(100),
        nullable=True,
        comment="Stripe failure code"
    )
    
    failure_message = Column(
        Text,
        nullable=True,
        comment="Human-readable failure message"
    )
    
    # Payment method used
    payment_method_type = Column(
        String(50),
        nullable=True,
        comment="Type of payment method used"
    )
    
    payment_method_last4 = Column(
        String(4),
        nullable=True,
        comment="Last 4 digits of payment method"
    )
    
    # Amounts
    amount_attempted = Column(
        Integer,
        nullable=False,
        comment="Amount attempted in cents"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        comment="Currency of attempt"
    )
    
    # Stripe IDs
    stripe_payment_intent_id = Column(
        String(255),
        nullable=True,
        comment="Stripe payment intent ID"
    )
    
    stripe_charge_id = Column(
        String(255),
        nullable=True,
        comment="Stripe charge ID (if successful)"
    )
    
    # Next attempt scheduling
    next_attempt_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When to attempt payment again"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        default=dict,
        nullable=True,
        comment="Additional attempt metadata"
    )
    
    # Relationships
    user = relationship("User")
    subscription = relationship("StripeSubscription", back_populates="payment_attempts")
    invoice = relationship("StripeInvoice", back_populates="payment_attempts")
    
    # Database constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('succeeded', 'failed', 'pending', 'requires_action', 'canceled')",
            name='ck_payment_attempt_status'
        ),
        CheckConstraint('attempt_number > 0', name='ck_attempt_number_positive'),
        CheckConstraint('amount_attempted >= 0', name='ck_amount_attempted_positive'),
        Index('idx_attempt_user_status', 'user_id', 'status'),
        Index('idx_attempt_subscription', 'subscription_id'),
        Index('idx_attempt_next_attempt', 'next_attempt_at'),
        Index('idx_attempt_created', 'created_at'),
        {"comment": "Payment attempt tracking"}
    )
    
    @property
    def is_successful(self) -> bool:
        """Check if payment attempt was successful."""
        return self.status == PaymentAttemptStatus.SUCCEEDED
    
    @property
    def is_failed(self) -> bool:
        """Check if payment attempt failed."""
        return self.status == PaymentAttemptStatus.FAILED
    
    def get_failure_display(self) -> Optional[str]:
        """Get user-friendly failure message."""
        if not self.is_failed:
            return None
        
        failure_messages = {
            "card_declined": "Your card was declined",
            "insufficient_funds": "Insufficient funds",
            "card_expired": "Your card has expired",
            "incorrect_cvc": "Incorrect security code",
            "processing_error": "Payment processing error",
            "generic_decline": "Payment was declined"
        }
        
        return failure_messages.get(self.failure_code, self.failure_message or "Payment failed")
    
    def __str__(self) -> str:
        return f"PaymentAttempt(#{self.attempt_number}, ${self.amount_attempted/100:.2f}, {self.status})"


class StripeWebhookEvent(FullAuditModel):
    """
    Stripe webhook event tracking for reliability and debugging.
    
    Ensures webhook events are processed exactly once and provides audit trail.
    """
    
    __tablename__ = "stripe_webhook_events"
    
    # Stripe webhook data
    stripe_event_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Stripe event ID (evt_...)"
    )
    
    event_type = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Stripe event type"
    )
    
    api_version = Column(
        String(20),
        nullable=True,
        comment="Stripe API version"
    )
    
    # Processing status
    status = Column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
        comment="Processing status"
    )
    
    retry_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of processing attempts"
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if processing failed"
    )
    
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When event was successfully processed"
    )
    
    # Event data
    data = Column(
        JSONB,
        nullable=False,
        comment="Full webhook event data"
    )
    
    # Database constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'skipped')",
            name='ck_webhook_event_status'
        ),
        CheckConstraint('retry_count >= 0', name='ck_retry_count_positive'),
        Index('idx_webhook_event_type', 'event_type'),
        Index('idx_webhook_status', 'status'),
        Index('idx_webhook_created', 'created_at'),
        Index('idx_webhook_stripe_id', 'stripe_event_id'),
        {"comment": "Stripe webhook event processing log"}
    )
    
    @property
    def is_processed(self) -> bool:
        """Check if event has been successfully processed."""
        return self.status == WebhookEventStatus.COMPLETED
    
    @property
    def needs_retry(self) -> bool:
        """Check if event needs to be retried."""
        return self.status == WebhookEventStatus.FAILED and self.retry_count < 5
    
    def __str__(self) -> str:
        return f"StripeWebhookEvent({self.stripe_event_id}, {self.event_type}, {self.status})"