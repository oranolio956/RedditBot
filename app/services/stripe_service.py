"""
Stripe Payment Service

Production-ready payment processing with Stripe for subscription management.
Handles the $150/week retainer model with comprehensive payment features.

Features:
- Subscription management (create, update, cancel)
- Payment processing and retry logic
- Webhook handling for events
- Customer portal integration
- Tax and invoice management
- Payment method management
- Dunning and failed payment recovery
"""

import stripe
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import logging
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.config.settings import get_settings
from app.core.redis import get_redis_client
from app.models.payment import (
    StripeCustomer, Subscription, Payment, PaymentMethod,
    Invoice, WebhookEvent, SubscriptionStatus, PaymentStatus,
    SubscriptionPlan
)
from app.models.user import User

logger = structlog.get_logger(__name__)


class PriceIds:
    """Stripe price IDs for subscription plans."""
    BASIC_WEEKLY = "price_basic_weekly"  # $150/week
    PREMIUM_WEEKLY = "price_premium_weekly"  # $250/week
    ENTERPRISE_WEEKLY = "price_enterprise_weekly"  # $500/week


class StripeEvents:
    """Stripe webhook event types."""
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    CUSTOMER_DELETED = "customer.deleted"
    
    SUBSCRIPTION_CREATED = "customer.subscription.created"
    SUBSCRIPTION_UPDATED = "customer.subscription.updated"
    SUBSCRIPTION_DELETED = "customer.subscription.deleted"
    SUBSCRIPTION_TRIAL_ENDING = "customer.subscription.trial_will_end"
    
    PAYMENT_SUCCEEDED = "payment_intent.succeeded"
    PAYMENT_FAILED = "payment_intent.payment_failed"
    
    INVOICE_CREATED = "invoice.created"
    INVOICE_PAID = "invoice.paid"
    INVOICE_PAYMENT_FAILED = "invoice.payment_failed"
    INVOICE_UPCOMING = "invoice.upcoming"
    
    CHARGE_SUCCEEDED = "charge.succeeded"
    CHARGE_FAILED = "charge.failed"
    CHARGE_REFUNDED = "charge.refunded"


class StripePaymentService:
    """
    Comprehensive Stripe payment service for subscription management.
    
    Handles:
    - Customer and subscription lifecycle
    - Payment processing and recovery
    - Webhook event processing
    - Tax calculation and invoicing
    - Customer portal management
    - Analytics and reporting
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.settings = get_settings()
        self.redis = None
        
        # Initialize Stripe
        stripe.api_key = self.settings.stripe.secret_key
        stripe.api_version = "2023-10-16"
        
        # Subscription plan configurations
        self.plans = {
            SubscriptionPlan.BASIC: {
                'price_id': PriceIds.BASIC_WEEKLY,
                'amount': 15000,  # $150.00 in cents
                'interval': 'week',
                'trial_days': 7,
                'features': {
                    'conversations_per_day': 100,
                    'personality_adaptation': True,
                    'priority_support': False,
                    'custom_personalities': 1,
                    'analytics_access': 'basic'
                }
            },
            SubscriptionPlan.PREMIUM: {
                'price_id': PriceIds.PREMIUM_WEEKLY,
                'amount': 25000,  # $250.00 in cents
                'interval': 'week',
                'trial_days': 14,
                'features': {
                    'conversations_per_day': 500,
                    'personality_adaptation': True,
                    'priority_support': True,
                    'custom_personalities': 5,
                    'analytics_access': 'advanced'
                }
            },
            SubscriptionPlan.ENTERPRISE: {
                'price_id': PriceIds.ENTERPRISE_WEEKLY,
                'amount': 50000,  # $500.00 in cents
                'interval': 'week',
                'trial_days': 14,
                'features': {
                    'conversations_per_day': -1,  # Unlimited
                    'personality_adaptation': True,
                    'priority_support': True,
                    'custom_personalities': -1,  # Unlimited
                    'analytics_access': 'full',
                    'dedicated_support': True,
                    'sla': True
                }
            }
        }
        
        # Payment retry configuration
        self.retry_config = {
            'max_attempts': 3,
            'retry_delays': [1, 3, 7],  # Days between retries
            'dunning_emails': True,
            'service_suspension_after': 14  # Days
        }
        
    async def initialize(self):
        """Initialize the payment service."""
        try:
            logger.info("Initializing Stripe payment service...")
            
            # Initialize Redis client
            self.redis = await get_redis_client()
            
            # Verify Stripe connection
            await self._verify_stripe_connection()
            
            # Create products and prices if they don't exist
            await self._ensure_products_and_prices()
            
            logger.info("Stripe payment service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Stripe service", error=str(e))
            raise
    
    async def _verify_stripe_connection(self):
        """Verify Stripe API connection."""
        try:
            # Test API connection
            stripe.Account.retrieve()
            logger.info("Stripe API connection verified")
        except stripe.error.AuthenticationError:
            logger.error("Invalid Stripe API key")
            raise
        except Exception as e:
            logger.error("Failed to connect to Stripe", error=str(e))
            raise
    
    async def _ensure_products_and_prices(self):
        """Ensure required products and prices exist in Stripe."""
        try:
            # Create product if it doesn't exist
            products = stripe.Product.list(limit=100)
            product_exists = any(p.metadata.get('app') == 'reddit_bot' for p in products)
            
            if not product_exists:
                product = stripe.Product.create(
                    name="AI Conversation Bot Subscription",
                    description="Weekly subscription for AI-powered conversation services",
                    metadata={'app': 'reddit_bot'}
                )
                logger.info(f"Created Stripe product: {product.id}")
            else:
                product = next(p for p in products if p.metadata.get('app') == 'reddit_bot')
            
            # Create prices for each plan
            for plan, config in self.plans.items():
                try:
                    price = stripe.Price.retrieve(config['price_id'])
                except stripe.error.InvalidRequestError:
                    # Price doesn't exist, create it
                    price = stripe.Price.create(
                        product=product.id,
                        unit_amount=config['amount'],
                        currency='usd',
                        recurring={'interval': config['interval']},
                        metadata={'plan': plan.value}
                    )
                    logger.info(f"Created Stripe price for {plan.value}: {price.id}")
                    
                    # Update config with actual price ID
                    config['price_id'] = price.id
                    
        except Exception as e:
            logger.error("Error ensuring products and prices", error=str(e))
    
    async def create_customer(
        self, 
        user_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> StripeCustomer:
        """
        Create a Stripe customer for a user.
        
        Args:
            user_id: Internal user ID
            email: Customer email
            name: Customer name
            metadata: Additional metadata
            
        Returns:
            StripeCustomer model instance
        """
        try:
            # Check if customer already exists
            existing_customer = await self.db.execute(
                select(StripeCustomer).where(StripeCustomer.user_id == user_id)
            )
            existing_customer = existing_customer.scalar_one_or_none()
            
            if existing_customer:
                logger.info(f"Customer already exists for user {user_id}")
                return existing_customer
            
            # Create Stripe customer
            customer_metadata = {'user_id': user_id}
            if metadata:
                customer_metadata.update(metadata)
            
            stripe_customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=customer_metadata,
                tax_exempt='none',
                preferred_locales=['en']
            )
            
            # Save to database
            db_customer = StripeCustomer(
                user_id=user_id,
                stripe_customer_id=stripe_customer.id,
                email=email,
                name=name,
                currency='usd',
                metadata=customer_metadata
            )
            
            self.db.add(db_customer)
            await self.db.commit()
            
            logger.info(f"Created Stripe customer {stripe_customer.id} for user {user_id}")
            return db_customer
            
        except Exception as e:
            logger.error("Error creating customer", error=str(e))
            await self.db.rollback()
            raise
    
    async def create_subscription(
        self,
        user_id: str,
        plan: SubscriptionPlan,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None
    ) -> Tuple[Subscription, Dict[str, Any]]:
        """
        Create a subscription for a user.
        
        Args:
            user_id: Internal user ID
            plan: Subscription plan
            payment_method_id: Stripe payment method ID
            trial_days: Override default trial period
            
        Returns:
            Tuple of (Subscription model, setup intent for payment collection)
        """
        try:
            # Get or create customer
            customer = await self.db.execute(
                select(StripeCustomer).where(StripeCustomer.user_id == user_id)
            )
            customer = customer.scalar_one_or_none()
            
            if not customer:
                raise ValueError(f"No Stripe customer found for user {user_id}")
            
            plan_config = self.plans[plan]
            trial_period_days = trial_days or plan_config['trial_days']
            
            # Attach payment method if provided
            if payment_method_id:
                stripe.PaymentMethod.attach(
                    payment_method_id,
                    customer=customer.stripe_customer_id
                )
                
                # Set as default payment method
                stripe.Customer.modify(
                    customer.stripe_customer_id,
                    invoice_settings={'default_payment_method': payment_method_id}
                )
            
            # Create subscription
            subscription_params = {
                'customer': customer.stripe_customer_id,
                'items': [{'price': plan_config['price_id']}],
                'trial_period_days': trial_period_days,
                'metadata': {
                    'user_id': user_id,
                    'plan': plan.value
                },
                'payment_settings': {
                    'payment_method_types': ['card'],
                    'save_default_payment_method': 'on_subscription'
                },
                'trial_settings': {
                    'end_behavior': {
                        'missing_payment_method': 'pause'
                    }
                }
            }
            
            # Add tax behavior
            if self.settings.stripe.automatic_tax:
                subscription_params['automatic_tax'] = {'enabled': True}
            
            stripe_subscription = stripe.Subscription.create(**subscription_params)
            
            # Save to database
            db_subscription = Subscription(
                user_id=user_id,
                stripe_subscription_id=stripe_subscription.id,
                stripe_customer_id=customer.stripe_customer_id,
                plan=plan,
                status=SubscriptionStatus.TRIALING if trial_period_days > 0 else SubscriptionStatus.ACTIVE,
                current_period_start=datetime.fromtimestamp(stripe_subscription.current_period_start),
                current_period_end=datetime.fromtimestamp(stripe_subscription.current_period_end),
                trial_end=datetime.fromtimestamp(stripe_subscription.trial_end) if stripe_subscription.trial_end else None,
                cancel_at_period_end=False,
                metadata={'stripe_status': stripe_subscription.status}
            )
            
            self.db.add(db_subscription)
            await self.db.commit()
            
            # Create setup intent if no payment method
            setup_intent = None
            if not payment_method_id:
                setup_intent = stripe.SetupIntent.create(
                    customer=customer.stripe_customer_id,
                    usage='off_session',
                    metadata={'subscription_id': stripe_subscription.id}
                )
            
            logger.info(
                f"Created subscription {stripe_subscription.id} for user {user_id}",
                plan=plan.value,
                trial_days=trial_period_days
            )
            
            return db_subscription, {
                'subscription_id': stripe_subscription.id,
                'client_secret': setup_intent.client_secret if setup_intent else None,
                'trial_end': stripe_subscription.trial_end,
                'status': stripe_subscription.status
            }
            
        except Exception as e:
            logger.error("Error creating subscription", error=str(e))
            await self.db.rollback()
            raise
    
    async def update_subscription(
        self,
        subscription_id: str,
        new_plan: Optional[SubscriptionPlan] = None,
        cancel_at_period_end: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Subscription:
        """Update an existing subscription."""
        try:
            # Get subscription from database
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if not db_subscription:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            update_params = {}
            
            # Handle plan change
            if new_plan and new_plan != db_subscription.plan:
                plan_config = self.plans[new_plan]
                update_params['items'] = [{
                    'id': stripe.Subscription.retrieve(subscription_id)['items']['data'][0]['id'],
                    'price': plan_config['price_id']
                }]
                update_params['proration_behavior'] = 'create_prorations'
            
            # Handle cancellation
            if cancel_at_period_end is not None:
                update_params['cancel_at_period_end'] = cancel_at_period_end
            
            # Update metadata
            if metadata:
                update_params['metadata'] = metadata
            
            # Update in Stripe
            if update_params:
                stripe_subscription = stripe.Subscription.modify(
                    subscription_id,
                    **update_params
                )
                
                # Update database
                if new_plan:
                    db_subscription.plan = new_plan
                if cancel_at_period_end is not None:
                    db_subscription.cancel_at_period_end = cancel_at_period_end
                
                db_subscription.updated_at = datetime.utcnow()
                await self.db.commit()
                
                logger.info(f"Updated subscription {subscription_id}", changes=update_params)
            
            return db_subscription
            
        except Exception as e:
            logger.error("Error updating subscription", error=str(e))
            await self.db.rollback()
            raise
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediately: bool = False,
        reason: Optional[str] = None
    ) -> Subscription:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Stripe subscription ID
            immediately: Cancel immediately vs at period end
            reason: Cancellation reason for analytics
        """
        try:
            # Get subscription from database
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if not db_subscription:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            # Cancel in Stripe
            if immediately:
                stripe_subscription = stripe.Subscription.delete(subscription_id)
                db_subscription.status = SubscriptionStatus.CANCELED
                db_subscription.canceled_at = datetime.utcnow()
            else:
                stripe_subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                    metadata={'cancellation_reason': reason} if reason else None
                )
                db_subscription.cancel_at_period_end = True
            
            db_subscription.metadata = db_subscription.metadata or {}
            db_subscription.metadata['cancellation_reason'] = reason
            db_subscription.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            logger.info(
                f"Canceled subscription {subscription_id}",
                immediately=immediately,
                reason=reason
            )
            
            return db_subscription
            
        except Exception as e:
            logger.error("Error canceling subscription", error=str(e))
            await self.db.rollback()
            raise
    
    async def process_webhook(
        self,
        payload: bytes,
        signature: str
    ) -> Dict[str, Any]:
        """
        Process a Stripe webhook event.
        
        Args:
            payload: Raw webhook payload
            signature: Stripe signature header
            
        Returns:
            Processing result
        """
        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.settings.stripe.webhook_secret
            )
            
            # Check for duplicate processing
            existing_event = await self.db.execute(
                select(WebhookEvent).where(
                    WebhookEvent.stripe_event_id == event['id']
                )
            )
            if existing_event.scalar_one_or_none():
                logger.info(f"Webhook event {event['id']} already processed")
                return {'status': 'duplicate', 'event_id': event['id']}
            
            # Save webhook event
            webhook_event = WebhookEvent(
                stripe_event_id=event['id'],
                event_type=event['type'],
                data=event['data'],
                processed=False
            )
            self.db.add(webhook_event)
            
            # Process based on event type
            result = await self._process_webhook_event(event)
            
            # Mark as processed
            webhook_event.processed = True
            webhook_event.processed_at = datetime.utcnow()
            webhook_event.result = result
            
            await self.db.commit()
            
            logger.info(
                f"Processed webhook event {event['id']}",
                event_type=event['type'],
                result=result.get('status')
            )
            
            return result
            
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            raise
        except Exception as e:
            logger.error("Error processing webhook", error=str(e))
            await self.db.rollback()
            raise
    
    async def _process_webhook_event(self, event: Dict) -> Dict[str, Any]:
        """Process specific webhook event types."""
        event_type = event['type']
        data = event['data']['object']
        
        try:
            if event_type == StripeEvents.SUBSCRIPTION_CREATED:
                return await self._handle_subscription_created(data)
            
            elif event_type == StripeEvents.SUBSCRIPTION_UPDATED:
                return await self._handle_subscription_updated(data)
            
            elif event_type == StripeEvents.SUBSCRIPTION_DELETED:
                return await self._handle_subscription_deleted(data)
            
            elif event_type == StripeEvents.INVOICE_PAID:
                return await self._handle_invoice_paid(data)
            
            elif event_type == StripeEvents.INVOICE_PAYMENT_FAILED:
                return await self._handle_invoice_payment_failed(data)
            
            elif event_type == StripeEvents.SUBSCRIPTION_TRIAL_ENDING:
                return await self._handle_trial_ending(data)
            
            else:
                logger.debug(f"Unhandled webhook event type: {event_type}")
                return {'status': 'unhandled', 'event_type': event_type}
                
        except Exception as e:
            logger.error(f"Error handling webhook event {event_type}", error=str(e))
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_subscription_created(self, subscription_data: Dict) -> Dict[str, Any]:
        """Handle subscription created event."""
        try:
            # Update or create subscription in database
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_data['id']
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if not db_subscription:
                # Create new subscription record
                user_id = subscription_data['metadata'].get('user_id')
                if not user_id:
                    logger.warning(f"No user_id in subscription metadata: {subscription_data['id']}")
                    return {'status': 'error', 'error': 'missing_user_id'}
                
                plan_name = subscription_data['metadata'].get('plan', 'BASIC')
                plan = SubscriptionPlan[plan_name]
                
                db_subscription = Subscription(
                    user_id=user_id,
                    stripe_subscription_id=subscription_data['id'],
                    stripe_customer_id=subscription_data['customer'],
                    plan=plan,
                    status=SubscriptionStatus[subscription_data['status'].upper()],
                    current_period_start=datetime.fromtimestamp(subscription_data['current_period_start']),
                    current_period_end=datetime.fromtimestamp(subscription_data['current_period_end']),
                    trial_end=datetime.fromtimestamp(subscription_data['trial_end']) if subscription_data.get('trial_end') else None
                )
                self.db.add(db_subscription)
            
            await self.db.commit()
            return {'status': 'success', 'subscription_id': subscription_data['id']}
            
        except Exception as e:
            logger.error("Error handling subscription created", error=str(e))
            raise
    
    async def _handle_subscription_updated(self, subscription_data: Dict) -> Dict[str, Any]:
        """Handle subscription updated event."""
        try:
            # Update subscription in database
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_data['id']
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if db_subscription:
                db_subscription.status = SubscriptionStatus[subscription_data['status'].upper()]
                db_subscription.current_period_start = datetime.fromtimestamp(subscription_data['current_period_start'])
                db_subscription.current_period_end = datetime.fromtimestamp(subscription_data['current_period_end'])
                db_subscription.cancel_at_period_end = subscription_data.get('cancel_at_period_end', False)
                db_subscription.updated_at = datetime.utcnow()
                
                await self.db.commit()
                
                # Notify user of subscription changes
                await self._notify_subscription_change(db_subscription)
            
            return {'status': 'success', 'subscription_id': subscription_data['id']}
            
        except Exception as e:
            logger.error("Error handling subscription updated", error=str(e))
            raise
    
    async def _handle_subscription_deleted(self, subscription_data: Dict) -> Dict[str, Any]:
        """Handle subscription deleted event."""
        try:
            # Update subscription status
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_data['id']
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if db_subscription:
                db_subscription.status = SubscriptionStatus.CANCELED
                db_subscription.canceled_at = datetime.utcnow()
                db_subscription.updated_at = datetime.utcnow()
                
                await self.db.commit()
                
                # Notify user of cancellation
                await self._notify_subscription_canceled(db_subscription)
            
            return {'status': 'success', 'subscription_id': subscription_data['id']}
            
        except Exception as e:
            logger.error("Error handling subscription deleted", error=str(e))
            raise
    
    async def _handle_invoice_paid(self, invoice_data: Dict) -> Dict[str, Any]:
        """Handle successful invoice payment."""
        try:
            # Record payment
            payment = Payment(
                stripe_payment_intent_id=invoice_data.get('payment_intent'),
                stripe_customer_id=invoice_data['customer'],
                amount=Decimal(invoice_data['amount_paid']) / 100,
                currency=invoice_data['currency'],
                status=PaymentStatus.SUCCEEDED,
                description=f"Invoice {invoice_data['number']}",
                metadata={'invoice_id': invoice_data['id']}
            )
            self.db.add(payment)
            
            # Update subscription payment status
            if invoice_data.get('subscription'):
                await self._update_subscription_payment_status(
                    invoice_data['subscription'],
                    'paid'
                )
            
            await self.db.commit()
            return {'status': 'success', 'invoice_id': invoice_data['id']}
            
        except Exception as e:
            logger.error("Error handling invoice paid", error=str(e))
            raise
    
    async def _handle_invoice_payment_failed(self, invoice_data: Dict) -> Dict[str, Any]:
        """Handle failed invoice payment."""
        try:
            # Record failed payment
            payment = Payment(
                stripe_payment_intent_id=invoice_data.get('payment_intent'),
                stripe_customer_id=invoice_data['customer'],
                amount=Decimal(invoice_data['amount_due']) / 100,
                currency=invoice_data['currency'],
                status=PaymentStatus.FAILED,
                description=f"Failed invoice {invoice_data['number']}",
                failure_reason=invoice_data.get('last_payment_error', {}).get('message'),
                metadata={'invoice_id': invoice_data['id']}
            )
            self.db.add(payment)
            
            # Handle payment retry logic
            if invoice_data.get('subscription'):
                await self._handle_payment_failure(invoice_data['subscription'])
            
            await self.db.commit()
            return {'status': 'success', 'invoice_id': invoice_data['id']}
            
        except Exception as e:
            logger.error("Error handling invoice payment failed", error=str(e))
            raise
    
    async def _handle_trial_ending(self, subscription_data: Dict) -> Dict[str, Any]:
        """Handle trial ending notification."""
        try:
            # Send trial ending reminder
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_data['id']
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if db_subscription:
                await self._send_trial_ending_notification(db_subscription)
            
            return {'status': 'success', 'subscription_id': subscription_data['id']}
            
        except Exception as e:
            logger.error("Error handling trial ending", error=str(e))
            raise
    
    async def _handle_payment_failure(self, subscription_id: str):
        """Handle payment failure with retry logic."""
        try:
            # Get subscription
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if not db_subscription:
                return
            
            # Increment failure count
            db_subscription.metadata = db_subscription.metadata or {}
            failure_count = db_subscription.metadata.get('payment_failures', 0) + 1
            db_subscription.metadata['payment_failures'] = failure_count
            
            # Apply dunning logic
            if failure_count <= self.retry_config['max_attempts']:
                # Schedule retry
                retry_delay = self.retry_config['retry_delays'][min(failure_count - 1, len(self.retry_config['retry_delays']) - 1)]
                retry_date = datetime.utcnow() + timedelta(days=retry_delay)
                
                db_subscription.metadata['next_retry'] = retry_date.isoformat()
                
                # Send dunning email
                if self.retry_config['dunning_emails']:
                    await self._send_dunning_email(db_subscription, failure_count)
            else:
                # Max retries exceeded - suspend service
                if (datetime.utcnow() - db_subscription.current_period_start).days >= self.retry_config['service_suspension_after']:
                    db_subscription.status = SubscriptionStatus.PAST_DUE
                    await self._suspend_service(db_subscription)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error("Error handling payment failure", error=str(e))
    
    async def create_checkout_session(
        self,
        user_id: str,
        plan: SubscriptionPlan,
        success_url: str,
        cancel_url: str
    ) -> Dict[str, Any]:
        """Create a Stripe Checkout session for subscription."""
        try:
            # Get or create customer
            customer = await self.db.execute(
                select(StripeCustomer).where(StripeCustomer.user_id == user_id)
            )
            customer = customer.scalar_one_or_none()
            
            if not customer:
                raise ValueError(f"No Stripe customer found for user {user_id}")
            
            plan_config = self.plans[plan]
            
            # Create checkout session
            session = stripe.checkout.Session.create(
                customer=customer.stripe_customer_id,
                payment_method_types=['card'],
                line_items=[{
                    'price': plan_config['price_id'],
                    'quantity': 1
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                subscription_data={
                    'trial_period_days': plan_config['trial_days'],
                    'metadata': {
                        'user_id': user_id,
                        'plan': plan.value
                    }
                },
                metadata={
                    'user_id': user_id,
                    'plan': plan.value
                }
            )
            
            logger.info(f"Created checkout session {session.id} for user {user_id}")
            
            return {
                'session_id': session.id,
                'checkout_url': session.url
            }
            
        except Exception as e:
            logger.error("Error creating checkout session", error=str(e))
            raise
    
    async def create_customer_portal_session(
        self,
        user_id: str,
        return_url: str
    ) -> Dict[str, Any]:
        """Create a customer portal session for self-service."""
        try:
            # Get customer
            customer = await self.db.execute(
                select(StripeCustomer).where(StripeCustomer.user_id == user_id)
            )
            customer = customer.scalar_one_or_none()
            
            if not customer:
                raise ValueError(f"No Stripe customer found for user {user_id}")
            
            # Create portal session
            session = stripe.billing_portal.Session.create(
                customer=customer.stripe_customer_id,
                return_url=return_url
            )
            
            logger.info(f"Created portal session for user {user_id}")
            
            return {
                'session_id': session.id,
                'portal_url': session.url
            }
            
        except Exception as e:
            logger.error("Error creating portal session", error=str(e))
            raise
    
    async def get_subscription_status(self, user_id: str) -> Dict[str, Any]:
        """Get detailed subscription status for a user."""
        try:
            # Get active subscription
            subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.user_id == user_id,
                    Subscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING,
                        SubscriptionStatus.PAST_DUE
                    ])
                ).order_by(Subscription.created_at.desc())
            )
            subscription = subscription.scalar_one_or_none()
            
            if not subscription:
                return {
                    'has_subscription': False,
                    'can_use_service': False
                }
            
            # Get plan features
            plan_features = self.plans[subscription.plan]['features']
            
            # Calculate days until renewal/expiry
            days_until_renewal = (subscription.current_period_end - datetime.utcnow()).days
            
            return {
                'has_subscription': True,
                'subscription_id': subscription.stripe_subscription_id,
                'plan': subscription.plan.value,
                'status': subscription.status.value,
                'can_use_service': subscription.status in [
                    SubscriptionStatus.ACTIVE,
                    SubscriptionStatus.TRIALING
                ],
                'is_trialing': subscription.status == SubscriptionStatus.TRIALING,
                'trial_end': subscription.trial_end.isoformat() if subscription.trial_end else None,
                'current_period_end': subscription.current_period_end.isoformat(),
                'days_until_renewal': days_until_renewal,
                'cancel_at_period_end': subscription.cancel_at_period_end,
                'features': plan_features,
                'payment_failures': subscription.metadata.get('payment_failures', 0) if subscription.metadata else 0
            }
            
        except Exception as e:
            logger.error("Error getting subscription status", error=str(e))
            return {
                'has_subscription': False,
                'can_use_service': False,
                'error': str(e)
            }
    
    async def _update_subscription_payment_status(self, subscription_id: str, status: str):
        """Update subscription payment status."""
        try:
            db_subscription = await self.db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            db_subscription = db_subscription.scalar_one_or_none()
            
            if db_subscription:
                db_subscription.metadata = db_subscription.metadata or {}
                db_subscription.metadata['last_payment_status'] = status
                db_subscription.metadata['last_payment_date'] = datetime.utcnow().isoformat()
                
                # Reset failure count on successful payment
                if status == 'paid':
                    db_subscription.metadata['payment_failures'] = 0
                    if db_subscription.status == SubscriptionStatus.PAST_DUE:
                        db_subscription.status = SubscriptionStatus.ACTIVE
                
                await self.db.commit()
                
        except Exception as e:
            logger.error("Error updating subscription payment status", error=str(e))
    
    async def _notify_subscription_change(self, subscription: Subscription):
        """Send notification about subscription changes."""
        # Implementation would send notification via Telegram bot
        logger.info(f"Notifying user {subscription.user_id} of subscription change")
    
    async def _notify_subscription_canceled(self, subscription: Subscription):
        """Send notification about subscription cancellation."""
        # Implementation would send notification via Telegram bot
        logger.info(f"Notifying user {subscription.user_id} of subscription cancellation")
    
    async def _send_trial_ending_notification(self, subscription: Subscription):
        """Send trial ending reminder."""
        # Implementation would send notification via Telegram bot
        logger.info(f"Sending trial ending reminder to user {subscription.user_id}")
    
    async def _send_dunning_email(self, subscription: Subscription, attempt: int):
        """Send dunning email for failed payment."""
        # Implementation would send email via email service
        logger.info(f"Sending dunning email to user {subscription.user_id}, attempt {attempt}")
    
    async def _suspend_service(self, subscription: Subscription):
        """Suspend service access for overdue subscription."""
        # Implementation would update user access permissions
        logger.info(f"Suspending service for user {subscription.user_id}")
    
    async def get_revenue_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get revenue metrics and analytics."""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Get successful payments
            payments = await self.db.execute(
                select(Payment).where(
                    Payment.created_at >= since_date,
                    Payment.status == PaymentStatus.SUCCEEDED
                )
            )
            payments = payments.scalars().all()
            
            # Calculate metrics
            total_revenue = sum(p.amount for p in payments)
            payment_count = len(payments)
            average_payment = total_revenue / payment_count if payment_count > 0 else 0
            
            # Get active subscriptions by plan
            subscriptions = await self.db.execute(
                select(Subscription).where(
                    Subscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING
                    ])
                )
            )
            subscriptions = subscriptions.scalars().all()
            
            plan_distribution = {}
            for sub in subscriptions:
                plan_distribution[sub.plan.value] = plan_distribution.get(sub.plan.value, 0) + 1
            
            # Calculate MRR (Monthly Recurring Revenue)
            mrr = 0
            for sub in subscriptions:
                if sub.status == SubscriptionStatus.ACTIVE:
                    plan_config = self.plans[sub.plan]
                    # Convert weekly to monthly
                    monthly_amount = (plan_config['amount'] / 100) * 4.33
                    mrr += monthly_amount
            
            return {
                'period_days': days,
                'total_revenue': float(total_revenue),
                'payment_count': payment_count,
                'average_payment': float(average_payment),
                'active_subscriptions': len(subscriptions),
                'plan_distribution': plan_distribution,
                'monthly_recurring_revenue': mrr,
                'average_revenue_per_user': mrr / len(subscriptions) if subscriptions else 0
            }
            
        except Exception as e:
            logger.error("Error calculating revenue metrics", error=str(e))
            return {}


# Export main classes
__all__ = [
    'StripePaymentService',
    'PriceIds',
    'StripeEvents'
]