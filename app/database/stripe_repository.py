"""
Stripe Database Repository

Repository pattern implementation for Stripe payment-related database operations.
Provides clean abstraction layer for all Stripe data access with proper error handling.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError, NoResultFound

from app.database.base import get_async_session
from app.models.user import User
from app.models.stripe_models import (
    StripeCustomer, StripeSubscription, StripePaymentMethod, 
    StripeInvoice, PaymentAttempt, StripeWebhookEvent,
    SubscriptionStatus, InvoiceStatus, PaymentAttemptStatus, WebhookEventStatus
)

logger = logging.getLogger(__name__)


class StripeRepository:
    """
    Repository for Stripe payment-related database operations.
    
    Provides high-level methods for managing customers, subscriptions,
    payments, and webhooks with proper error handling and logging.
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session or get_async_session()
    
    # Customer operations
    
    async def create_customer(
        self,
        user_id: UUID,
        stripe_customer_id: str,
        email: str,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        **kwargs
    ) -> StripeCustomer:
        """
        Create a new Stripe customer record.
        
        Args:
            user_id: User UUID
            stripe_customer_id: Stripe customer ID
            email: Customer email
            name: Customer name
            phone: Customer phone
            **kwargs: Additional customer fields
            
        Returns:
            Created StripeCustomer instance
        """
        try:
            customer_data = {
                "user_id": user_id,
                "stripe_customer_id": stripe_customer_id,
                "email": email,
                "name": name,
                "phone": phone,
                **kwargs
            }
            
            customer = StripeCustomer(**customer_data)
            self.session.add(customer)
            await self.session.commit()
            await self.session.refresh(customer)
            
            logger.info(f"Created customer record for Stripe customer {stripe_customer_id}")
            return customer
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating customer {stripe_customer_id}: {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating customer {stripe_customer_id}: {e}")
            raise
    
    async def get_customer_by_user_id(self, user_id: UUID) -> Optional[StripeCustomer]:
        """Get customer by user ID."""
        try:
            result = await self.session.execute(
                select(StripeCustomer)
                .where(and_(
                    StripeCustomer.user_id == user_id,
                    StripeCustomer.is_deleted == False
                ))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching customer for user {user_id}: {e}")
            raise
    
    async def get_customer_by_stripe_id(self, stripe_customer_id: str) -> Optional[StripeCustomer]:
        """Get customer by Stripe customer ID."""
        try:
            result = await self.session.execute(
                select(StripeCustomer)
                .where(and_(
                    StripeCustomer.stripe_customer_id == stripe_customer_id,
                    StripeCustomer.is_deleted == False
                ))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching customer {stripe_customer_id}: {e}")
            raise
    
    async def update_customer(
        self, 
        customer_id: UUID, 
        **update_data
    ) -> Optional[StripeCustomer]:
        """Update customer record."""
        try:
            await self.session.execute(
                update(StripeCustomer)
                .where(StripeCustomer.id == customer_id)
                .values(**update_data, updated_at=datetime.utcnow())
            )
            await self.session.commit()
            
            # Return updated customer
            return await self.get_customer_by_id(customer_id)
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating customer {customer_id}: {e}")
            raise
    
    async def get_customer_by_id(self, customer_id: UUID) -> Optional[StripeCustomer]:
        """Get customer by ID."""
        try:
            result = await self.session.execute(
                select(StripeCustomer)
                .where(and_(
                    StripeCustomer.id == customer_id,
                    StripeCustomer.is_deleted == False
                ))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching customer {customer_id}: {e}")
            raise
    
    # Subscription operations
    
    async def create_subscription(
        self,
        user_id: UUID,
        customer_id: UUID,
        stripe_subscription_id: str,
        **subscription_data
    ) -> StripeSubscription:
        """
        Create a new subscription record.
        
        Args:
            user_id: User UUID
            customer_id: Customer UUID
            stripe_subscription_id: Stripe subscription ID
            **subscription_data: Additional subscription fields
            
        Returns:
            Created StripeSubscription instance
        """
        try:
            data = {
                "user_id": user_id,
                "customer_id": customer_id,
                "stripe_subscription_id": stripe_subscription_id,
                **subscription_data
            }
            
            subscription = StripeSubscription(**data)
            self.session.add(subscription)
            await self.session.commit()
            await self.session.refresh(subscription)
            
            logger.info(f"Created subscription record for {stripe_subscription_id}")
            return subscription
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating subscription {stripe_subscription_id}: {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating subscription {stripe_subscription_id}: {e}")
            raise
    
    async def get_subscription_by_stripe_id(
        self, 
        stripe_subscription_id: str
    ) -> Optional[StripeSubscription]:
        """Get subscription by Stripe subscription ID."""
        try:
            result = await self.session.execute(
                select(StripeSubscription)
                .options(joinedload(StripeSubscription.customer))
                .options(joinedload(StripeSubscription.user))
                .where(and_(
                    StripeSubscription.stripe_subscription_id == stripe_subscription_id,
                    StripeSubscription.is_deleted == False
                ))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching subscription {stripe_subscription_id}: {e}")
            raise
    
    async def get_active_subscription_by_user_id(
        self, 
        user_id: UUID
    ) -> Optional[StripeSubscription]:
        """Get active subscription for user."""
        try:
            result = await self.session.execute(
                select(StripeSubscription)
                .options(joinedload(StripeSubscription.customer))
                .where(and_(
                    StripeSubscription.user_id == user_id,
                    StripeSubscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING,
                        SubscriptionStatus.PAST_DUE
                    ]),
                    StripeSubscription.is_deleted == False
                ))
                .order_by(StripeSubscription.created_at.desc())
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching active subscription for user {user_id}: {e}")
            raise
    
    async def get_subscriptions_by_user_id(
        self, 
        user_id: UUID,
        limit: int = 10
    ) -> List[StripeSubscription]:
        """Get all subscriptions for user."""
        try:
            result = await self.session.execute(
                select(StripeSubscription)
                .options(joinedload(StripeSubscription.customer))
                .where(and_(
                    StripeSubscription.user_id == user_id,
                    StripeSubscription.is_deleted == False
                ))
                .order_by(StripeSubscription.created_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching subscriptions for user {user_id}: {e}")
            raise
    
    async def update_subscription_status(
        self, 
        subscription_id: UUID, 
        status: str
    ) -> None:
        """Update subscription status."""
        try:
            await self.session.execute(
                update(StripeSubscription)
                .where(StripeSubscription.id == subscription_id)
                .values(status=status, updated_at=datetime.utcnow())
            )
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating subscription {subscription_id} status: {e}")
            raise
    
    async def update_subscription_from_stripe_data(
        self, 
        subscription_id: UUID, 
        stripe_data: Dict[str, Any]
    ) -> None:
        """Update subscription from Stripe webhook data."""
        try:
            update_data = {
                "status": stripe_data.get("status"),
                "current_period_start": datetime.fromtimestamp(
                    stripe_data.get("current_period_start")
                ) if stripe_data.get("current_period_start") else None,
                "current_period_end": datetime.fromtimestamp(
                    stripe_data.get("current_period_end")
                ) if stripe_data.get("current_period_end") else None,
                "cancel_at_period_end": stripe_data.get("cancel_at_period_end", False),
                "updated_at": datetime.utcnow()
            }
            
            # Handle optional fields
            if stripe_data.get("canceled_at"):
                update_data["canceled_at"] = datetime.fromtimestamp(stripe_data["canceled_at"])
            
            if stripe_data.get("trial_start"):
                update_data["trial_start"] = datetime.fromtimestamp(stripe_data["trial_start"])
                
            if stripe_data.get("trial_end"):
                update_data["trial_end"] = datetime.fromtimestamp(stripe_data["trial_end"])
            
            # Update metadata
            update_data["metadata"] = stripe_data
            
            await self.session.execute(
                update(StripeSubscription)
                .where(StripeSubscription.id == subscription_id)
                .values(**update_data)
            )
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating subscription {subscription_id} from Stripe data: {e}")
            raise
    
    async def update_subscription_metadata(
        self, 
        subscription_id: UUID, 
        metadata: Dict[str, Any]
    ) -> None:
        """Update subscription metadata."""
        try:
            subscription = await self.session.get(StripeSubscription, subscription_id)
            if subscription:
                current_metadata = subscription.metadata or {}
                current_metadata.update(metadata)
                subscription.metadata = current_metadata
                subscription.updated_at = datetime.utcnow()
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating subscription {subscription_id} metadata: {e}")
            raise
    
    # Invoice operations
    
    async def create_invoice(
        self,
        user_id: UUID,
        customer_id: UUID,
        stripe_invoice_id: str,
        **invoice_data
    ) -> StripeInvoice:
        """Create invoice record."""
        try:
            data = {
                "user_id": user_id,
                "customer_id": customer_id,
                "stripe_invoice_id": stripe_invoice_id,
                **invoice_data
            }
            
            invoice = StripeInvoice(**data)
            self.session.add(invoice)
            await self.session.commit()
            await self.session.refresh(invoice)
            
            return invoice
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating invoice {stripe_invoice_id}: {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating invoice {stripe_invoice_id}: {e}")
            raise
    
    async def get_invoice_by_stripe_id(self, stripe_invoice_id: str) -> Optional[StripeInvoice]:
        """Get invoice by Stripe invoice ID."""
        try:
            result = await self.session.execute(
                select(StripeInvoice)
                .where(StripeInvoice.stripe_invoice_id == stripe_invoice_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching invoice {stripe_invoice_id}: {e}")
            raise
    
    async def get_invoices_by_user_id(
        self, 
        user_id: UUID, 
        limit: int = 10
    ) -> List[StripeInvoice]:
        """Get invoices for user."""
        try:
            result = await self.session.execute(
                select(StripeInvoice)
                .where(StripeInvoice.user_id == user_id)
                .order_by(StripeInvoice.created_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching invoices for user {user_id}: {e}")
            raise
    
    async def get_latest_invoice_by_user_id(self, user_id: UUID) -> Optional[StripeInvoice]:
        """Get latest invoice for user."""
        try:
            result = await self.session.execute(
                select(StripeInvoice)
                .where(StripeInvoice.user_id == user_id)
                .order_by(StripeInvoice.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching latest invoice for user {user_id}: {e}")
            raise
    
    # Payment method operations
    
    async def create_payment_method_from_stripe_data(
        self,
        user_id: UUID,
        customer_id: UUID,
        stripe_pm_data: Dict[str, Any]
    ) -> StripePaymentMethod:
        """Create payment method from Stripe data."""
        try:
            pm_data = {
                "user_id": user_id,
                "customer_id": customer_id,
                "stripe_payment_method_id": stripe_pm_data["id"],
                "type": stripe_pm_data["type"]
            }
            
            # Handle card data
            if stripe_pm_data["type"] == "card" and "card" in stripe_pm_data:
                card = stripe_pm_data["card"]
                pm_data.update({
                    "card_brand": card.get("brand"),
                    "card_last4": card.get("last4"),
                    "card_exp_month": card.get("exp_month"),
                    "card_exp_year": card.get("exp_year"),
                    "card_country": card.get("country")
                })
            
            # Handle bank account data
            elif stripe_pm_data["type"] == "us_bank_account" and "us_bank_account" in stripe_pm_data:
                bank = stripe_pm_data["us_bank_account"]
                pm_data.update({
                    "bank_account_bank_name": bank.get("bank_name"),
                    "bank_account_last4": bank.get("last4"),
                    "bank_account_account_type": bank.get("account_type")
                })
            
            pm_data["metadata"] = stripe_pm_data
            
            payment_method = StripePaymentMethod(**pm_data)
            self.session.add(payment_method)
            await self.session.commit()
            await self.session.refresh(payment_method)
            
            return payment_method
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating payment method {stripe_pm_data['id']}: {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating payment method {stripe_pm_data['id']}: {e}")
            raise
    
    async def get_payment_methods_by_user_id(
        self, 
        user_id: UUID
    ) -> List[StripePaymentMethod]:
        """Get payment methods for user."""
        try:
            result = await self.session.execute(
                select(StripePaymentMethod)
                .where(and_(
                    StripePaymentMethod.user_id == user_id,
                    StripePaymentMethod.is_active == True
                ))
                .order_by(
                    StripePaymentMethod.is_default.desc(),
                    StripePaymentMethod.created_at.desc()
                )
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching payment methods for user {user_id}: {e}")
            raise
    
    async def get_payment_method_by_stripe_id(
        self, 
        stripe_payment_method_id: str
    ) -> Optional[StripePaymentMethod]:
        """Get payment method by Stripe ID."""
        try:
            result = await self.session.execute(
                select(StripePaymentMethod)
                .where(StripePaymentMethod.stripe_payment_method_id == stripe_payment_method_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching payment method {stripe_payment_method_id}: {e}")
            raise
    
    # Payment attempt operations
    
    async def create_payment_attempt(
        self,
        user_id: UUID,
        subscription_id: Optional[UUID] = None,
        invoice_id: Optional[UUID] = None,
        attempt_number: int = 1,
        status: str = "failed",
        amount_attempted: int = 0,
        **attempt_data
    ) -> PaymentAttempt:
        """Create payment attempt record."""
        try:
            data = {
                "user_id": user_id,
                "subscription_id": subscription_id,
                "invoice_id": invoice_id,
                "attempt_number": attempt_number,
                "status": status,
                "amount_attempted": amount_attempted,
                **attempt_data
            }
            
            attempt = PaymentAttempt(**data)
            self.session.add(attempt)
            await self.session.commit()
            await self.session.refresh(attempt)
            
            return attempt
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating payment attempt: {e}")
            raise
    
    async def get_payment_attempts_by_subscription_id(
        self, 
        subscription_id: UUID
    ) -> List[PaymentAttempt]:
        """Get payment attempts for subscription."""
        try:
            result = await self.session.execute(
                select(PaymentAttempt)
                .where(PaymentAttempt.subscription_id == subscription_id)
                .order_by(PaymentAttempt.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching payment attempts for subscription {subscription_id}: {e}")
            raise
    
    # Webhook operations
    
    async def create_webhook_event(self, event_data: Dict[str, Any]) -> StripeWebhookEvent:
        """Create webhook event record."""
        try:
            webhook_event = StripeWebhookEvent(
                stripe_event_id=event_data["id"],
                event_type=event_data["type"],
                api_version=event_data.get("api_version"),
                data=event_data,
                status="pending"
            )
            
            self.session.add(webhook_event)
            await self.session.commit()
            await self.session.refresh(webhook_event)
            
            return webhook_event
            
        except IntegrityError:
            # Event already exists - that's OK for idempotency
            await self.session.rollback()
            return await self.get_webhook_event_by_stripe_id(event_data["id"])
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating webhook event {event_data['id']}: {e}")
            raise
    
    async def upsert_webhook_event(self, event_data: Dict[str, Any]) -> StripeWebhookEvent:
        """Create or update webhook event record."""
        existing_event = await self.get_webhook_event_by_stripe_id(event_data["id"])
        
        if existing_event:
            # Update existing event data
            existing_event.data = event_data
            existing_event.api_version = event_data.get("api_version")
            existing_event.updated_at = datetime.utcnow()
            await self.session.commit()
            return existing_event
        else:
            # Create new event
            return await self.create_webhook_event(event_data)
    
    async def get_webhook_event_by_stripe_id(
        self, 
        stripe_event_id: str
    ) -> Optional[StripeWebhookEvent]:
        """Get webhook event by Stripe event ID."""
        try:
            result = await self.session.execute(
                select(StripeWebhookEvent)
                .where(StripeWebhookEvent.stripe_event_id == stripe_event_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching webhook event {stripe_event_id}: {e}")
            raise
    
    async def update_webhook_event_status(
        self,
        webhook_event_id: UUID,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update webhook event processing status."""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status == "completed":
                update_data["processed_at"] = datetime.utcnow()
            elif status == "failed" and error_message:
                update_data["error_message"] = error_message
                # Increment retry count
                webhook = await self.session.get(StripeWebhookEvent, webhook_event_id)
                if webhook:
                    update_data["retry_count"] = webhook.retry_count + 1
            
            await self.session.execute(
                update(StripeWebhookEvent)
                .where(StripeWebhookEvent.id == webhook_event_id)
                .values(**update_data)
            )
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating webhook event {webhook_event_id}: {e}")
            raise
    
    async def mark_webhook_processed(self, webhook_event_id: UUID) -> None:
        """Mark webhook event as processed."""
        await self.update_webhook_event_status(webhook_event_id, "completed")
    
    async def mark_webhook_failed(
        self, 
        webhook_event_id: UUID, 
        error_message: str
    ) -> None:
        """Mark webhook event as failed."""
        await self.update_webhook_event_status(webhook_event_id, "failed", error_message)
    
    # User operations (for updating preferences)
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            raise
    
    async def update_user_preferences(
        self, 
        user_id: UUID, 
        preferences: Dict[str, Any]
    ) -> None:
        """Update user preferences."""
        try:
            user = await self.session.get(User, user_id)
            if user:
                current_preferences = user.preferences or {}
                current_preferences.update(preferences)
                user.preferences = current_preferences
                user.updated_at = datetime.utcnow()
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating user {user_id} preferences: {e}")
            raise
    
    # Analytics and reporting
    
    async def get_subscription_analytics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get subscription analytics for date range."""
        try:
            # Total subscriptions created
            total_created = await self.session.execute(
                select(func.count(StripeSubscription.id))
                .where(and_(
                    StripeSubscription.created_at >= start_date,
                    StripeSubscription.created_at <= end_date
                ))
            )
            
            # Active subscriptions
            active_count = await self.session.execute(
                select(func.count(StripeSubscription.id))
                .where(StripeSubscription.status.in_([
                    SubscriptionStatus.ACTIVE,
                    SubscriptionStatus.TRIALING
                ]))
            )
            
            # Revenue (from paid invoices)
            revenue = await self.session.execute(
                select(func.coalesce(func.sum(StripeInvoice.amount_paid), 0))
                .where(and_(
                    StripeInvoice.paid_at >= start_date,
                    StripeInvoice.paid_at <= end_date,
                    StripeInvoice.status == InvoiceStatus.PAID
                ))
            )
            
            # Subscription by tier
            tier_distribution = await self.session.execute(
                select(
                    StripeSubscription.tier_name,
                    func.count(StripeSubscription.id).label('count')
                )
                .where(StripeSubscription.status.in_([
                    SubscriptionStatus.ACTIVE,
                    SubscriptionStatus.TRIALING
                ]))
                .group_by(StripeSubscription.tier_name)
            )
            
            return {
                "total_subscriptions_created": total_created.scalar(),
                "active_subscriptions": active_count.scalar(),
                "total_revenue_cents": revenue.scalar(),
                "tier_distribution": dict(tier_distribution.all())
            }
            
        except Exception as e:
            logger.error(f"Error fetching subscription analytics: {e}")
            raise
    
    async def get_payment_failure_rate(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Get payment failure rate for date range."""
        try:
            total_attempts = await self.session.execute(
                select(func.count(PaymentAttempt.id))
                .where(and_(
                    PaymentAttempt.created_at >= start_date,
                    PaymentAttempt.created_at <= end_date
                ))
            )
            
            failed_attempts = await self.session.execute(
                select(func.count(PaymentAttempt.id))
                .where(and_(
                    PaymentAttempt.created_at >= start_date,
                    PaymentAttempt.created_at <= end_date,
                    PaymentAttempt.status == PaymentAttemptStatus.FAILED
                ))
            )
            
            total = total_attempts.scalar() or 0
            failed = failed_attempts.scalar() or 0
            
            return (failed / total) if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating payment failure rate: {e}")
            raise
    
    # Cleanup operations
    
    async def cleanup_old_webhook_events(self, days_old: int = 30) -> int:
        """Clean up old webhook events."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            result = await self.session.execute(
                delete(StripeWebhookEvent)
                .where(and_(
                    StripeWebhookEvent.created_at < cutoff_date,
                    StripeWebhookEvent.status == WebhookEventStatus.COMPLETED
                ))
            )
            
            deleted_count = result.rowcount
            await self.session.commit()
            
            logger.info(f"Cleaned up {deleted_count} old webhook events")
            return deleted_count
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error cleaning up webhook events: {e}")
            raise