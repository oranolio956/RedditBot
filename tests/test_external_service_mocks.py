"""
External Service Mocking Tests

Comprehensive test suite for mocking external service integrations:
- OpenAI API integration and response handling
- Stripe payment processing and webhook handling
- Telegram Bot API operations and webhook processing
- Service resilience and error handling
- Rate limiting and retry mechanisms
- Circuit breaker patterns
- Response caching and optimization
- Performance testing with mocked services
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
import httpx
import stripe
from openai import AsyncOpenAI
from aiogram import Bot
from aiogram.types import Update, Message, User as TelegramUser

from app.services.llm_service import LLMService, LLMResponse, LLMError
from app.services.stripe_service import StripeService, StripeError
from app.services.telegram_service import TelegramService, TelegramError
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerError


class TestOpenAIMocking:
    """Test OpenAI API integration with comprehensive mocking."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        client = Mock(spec=AsyncOpenAI)
        client.chat = Mock()
        client.chat.completions = Mock()
        client.chat.completions.create = AsyncMock()
        return client
    
    @pytest.fixture
    def llm_service(self, mock_openai_client):
        """Create LLMService with mocked OpenAI client."""
        service = LLMService()
        service.client = mock_openai_client
        return service
    
    @pytest.mark.asyncio
    async def test_successful_chat_completion(self, llm_service, mock_openai_client):
        """Test successful chat completion with OpenAI."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a test response from OpenAI"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 30
        mock_response.model = "gpt-4"
        mock_response.id = "chatcmpl-test-123"
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test completion request
        prompt = "Hello, how can you help me today?"
        response = await llm_service.get_completion(prompt)
        
        # Verify request was made correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        
        assert call_args.kwargs["model"] is not None
        assert len(call_args.kwargs["messages"]) >= 1
        assert call_args.kwargs["messages"][-1]["content"] == prompt
        
        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from OpenAI"
        assert response.usage["total_tokens"] == 50
        assert response.model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self, llm_service, mock_openai_client):
        """Test OpenAI rate limit error handling."""
        from openai import RateLimitError
        
        # Mock rate limit error
        mock_openai_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body={"error": {"code": "rate_limit_exceeded"}}
        )
        
        # Should handle rate limit gracefully
        with pytest.raises(LLMError) as exc_info:
            await llm_service.get_completion("Test prompt")
        
        assert "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.error_type == "rate_limit"
    
    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self, llm_service, mock_openai_client):
        """Test OpenAI timeout handling with retries."""
        import openai
        
        # Mock timeout error
        mock_openai_client.chat.completions.create.side_effect = [
            openai.APITimeoutError("Request timed out"),
            openai.APITimeoutError("Request timed out"),
            Mock(choices=[Mock(message=Mock(content="Success after retries"))])
        ]
        
        # Should retry and eventually succeed
        response = await llm_service.get_completion("Test prompt", max_retries=3)
        
        # Verify retries were attempted
        assert mock_openai_client.chat.completions.create.call_count == 3
        assert response.content == "Success after retries"
    
    @pytest.mark.asyncio
    async def test_openai_invalid_api_key(self, llm_service, mock_openai_client):
        """Test OpenAI invalid API key handling."""
        from openai import AuthenticationError
        
        # Mock authentication error
        mock_openai_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=Mock(status_code=401),
            body={"error": {"code": "invalid_api_key"}}
        )
        
        # Should raise authentication error
        with pytest.raises(LLMError) as exc_info:
            await llm_service.get_completion("Test prompt")
        
        assert exc_info.value.error_type == "authentication"
    
    @pytest.mark.asyncio
    async def test_openai_content_filter(self, llm_service, mock_openai_client):
        """Test OpenAI content filter handling."""
        # Mock content filter response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "content_filter"
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Should handle content filter gracefully
        response = await llm_service.get_completion("Inappropriate prompt")
        
        assert response.content is None or "content filtered" in response.content.lower()
        assert response.finish_reason == "content_filter"
    
    @pytest.mark.asyncio
    async def test_openai_streaming_response(self, llm_service, mock_openai_client):
        """Test OpenAI streaming response handling."""
        # Mock streaming response
        async def mock_stream():
            chunks = [
                Mock(choices=[Mock(delta=Mock(content="Hello"))]),
                Mock(choices=[Mock(delta=Mock(content=" there"))]),
                Mock(choices=[Mock(delta=Mock(content="!"))])
            ]
            for chunk in chunks:
                yield chunk
        
        mock_openai_client.chat.completions.create.return_value = mock_stream()
        
        # Test streaming completion
        full_response = ""
        async for chunk in llm_service.get_streaming_completion("Test prompt"):
            full_response += chunk
        
        assert full_response == "Hello there!"
    
    @pytest.mark.asyncio
    async def test_openai_performance_monitoring(self, llm_service, mock_openai_client):
        """Test OpenAI API performance monitoring."""
        # Mock responses with varying delays
        async def slow_response():
            await asyncio.sleep(0.1)  # Simulate network delay
            return Mock(
                choices=[Mock(message=Mock(content="Slow response"))],
                usage=Mock(total_tokens=25)
            )
        
        mock_openai_client.chat.completions.create.side_effect = slow_response
        
        # Monitor response time
        start_time = time.time()
        response = await llm_service.get_completion("Test prompt")
        response_time = time.time() - start_time
        
        # Verify performance tracking
        assert response_time >= 0.1  # Should include simulated delay
        assert hasattr(llm_service, '_performance_metrics')
        assert llm_service._performance_metrics['last_response_time'] > 0


class TestStripeMocking:
    """Test Stripe payment integration with comprehensive mocking."""
    
    @pytest.fixture
    def mock_stripe(self):
        """Create mock Stripe client."""
        with patch('stripe.PaymentIntent') as mock_pi, \
             patch('stripe.Customer') as mock_customer, \
             patch('stripe.Webhook') as mock_webhook:
            
            yield {
                'payment_intent': mock_pi,
                'customer': mock_customer,
                'webhook': mock_webhook
            }
    
    @pytest.fixture
    def stripe_service(self, mock_stripe):
        """Create StripeService with mocked Stripe client."""
        service = StripeService()
        return service
    
    @pytest.mark.asyncio
    async def test_create_payment_intent_success(self, stripe_service, mock_stripe):
        """Test successful payment intent creation."""
        # Mock successful payment intent creation
        mock_pi = Mock()
        mock_pi.id = "pi_test_1234567890"
        mock_pi.client_secret = "pi_test_1234567890_secret_test"
        mock_pi.amount = 999
        mock_pi.currency = "usd"
        mock_pi.status = "requires_payment_method"
        
        mock_stripe['payment_intent'].create.return_value = mock_pi
        
        # Test payment intent creation
        result = await stripe_service.create_payment_intent(
            amount=999,
            currency="usd",
            customer_id="cus_test_customer",
            metadata={"user_id": "test-user-123"}
        )
        
        # Verify Stripe API call
        mock_stripe['payment_intent'].create.assert_called_once()
        call_args = mock_stripe['payment_intent'].create.call_args.kwargs
        
        assert call_args["amount"] == 999
        assert call_args["currency"] == "usd"
        assert call_args["customer"] == "cus_test_customer"
        assert call_args["metadata"]["user_id"] == "test-user-123"
        
        # Verify response
        assert result["payment_intent_id"] == "pi_test_1234567890"
        assert result["client_secret"] == "pi_test_1234567890_secret_test"
        assert result["amount"] == 999
        assert result["status"] == "requires_payment_method"
    
    @pytest.mark.asyncio
    async def test_confirm_payment_intent(self, stripe_service, mock_stripe):
        """Test payment intent confirmation."""
        # Mock payment intent confirmation
        mock_pi = Mock()
        mock_pi.id = "pi_test_confirm"
        mock_pi.status = "succeeded"
        mock_pi.amount_received = 999
        mock_pi.charges = Mock()
        mock_pi.charges.data = [Mock(id="ch_test_charge")]
        
        mock_stripe['payment_intent'].confirm.return_value = mock_pi
        
        # Test payment confirmation
        result = await stripe_service.confirm_payment_intent(
            payment_intent_id="pi_test_confirm",
            payment_method="pm_test_card"
        )
        
        # Verify confirmation call
        mock_stripe['payment_intent'].confirm.assert_called_once_with(
            "pi_test_confirm",
            payment_method="pm_test_card"
        )
        
        # Verify result
        assert result["status"] == "succeeded"
        assert result["amount_received"] == 999
    
    @pytest.mark.asyncio
    async def test_stripe_card_error(self, stripe_service, mock_stripe):
        """Test Stripe card error handling."""
        import stripe
        
        # Mock card error
        mock_stripe['payment_intent'].confirm.side_effect = stripe.error.CardError(
            message="Your card was declined.",
            param="payment_method",
            code="card_declined"
        )
        
        # Should handle card error gracefully
        with pytest.raises(StripeError) as exc_info:
            await stripe_service.confirm_payment_intent(
                "pi_test_decline",
                "pm_test_declined_card"
            )
        
        assert exc_info.value.error_type == "card_error"
        assert "declined" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_stripe_webhook_processing(self, stripe_service, mock_stripe):
        """Test Stripe webhook event processing."""
        # Mock webhook construction and processing
        mock_event = Mock()
        mock_event.type = "payment_intent.succeeded"
        mock_event.data = Mock()
        mock_event.data.object = Mock()
        mock_event.data.object.id = "pi_webhook_test"
        mock_event.data.object.metadata = {"user_id": "test-user-456"}
        
        mock_stripe['webhook'].construct_event.return_value = mock_event
        
        # Test webhook processing
        payload = json.dumps({"type": "payment_intent.succeeded"})
        signature = "test_signature"
        
        result = await stripe_service.process_webhook(payload, signature)
        
        # Verify webhook construction
        mock_stripe['webhook'].construct_event.assert_called_once_with(
            payload, signature, stripe_service.webhook_secret
        )
        
        # Verify processing result
        assert result["event_type"] == "payment_intent.succeeded"
        assert result["payment_intent_id"] == "pi_webhook_test"
        assert result["processed"] is True
    
    @pytest.mark.asyncio
    async def test_stripe_idempotency(self, stripe_service, mock_stripe):
        """Test Stripe idempotency key handling."""
        mock_pi = Mock()
        mock_pi.id = "pi_idempotent_test"
        mock_stripe['payment_intent'].create.return_value = mock_pi
        
        # Test with idempotency key
        idempotency_key = "test_idem_key_123"
        
        await stripe_service.create_payment_intent(
            amount=1000,
            currency="usd",
            idempotency_key=idempotency_key
        )
        
        # Verify idempotency key was included
        call_args = mock_stripe['payment_intent'].create.call_args.kwargs
        assert call_args.get("idempotency_key") == idempotency_key
    
    @pytest.mark.asyncio
    async def test_stripe_rate_limiting(self, stripe_service, mock_stripe):
        """Test Stripe rate limiting handling."""
        import stripe
        
        # Mock rate limit error followed by success
        mock_stripe['payment_intent'].create.side_effect = [
            stripe.error.RateLimitError("Too many requests"),
            Mock(id="pi_after_rate_limit", status="requires_payment_method")
        ]
        
        # Should retry and succeed
        result = await stripe_service.create_payment_intent(
            amount=500,
            currency="usd",
            max_retries=2
        )
        
        # Verify retry was attempted
        assert mock_stripe['payment_intent'].create.call_count == 2
        assert result["payment_intent_id"] == "pi_after_rate_limit"


class TestTelegramMocking:
    """Test Telegram Bot API integration with comprehensive mocking."""
    
    @pytest.fixture
    def mock_telegram_bot(self):
        """Create mock Telegram bot."""
        bot = Mock(spec=Bot)
        bot.get_me = AsyncMock()
        bot.send_message = AsyncMock()
        bot.send_photo = AsyncMock()
        bot.send_voice = AsyncMock()
        bot.edit_message_text = AsyncMock()
        bot.delete_message = AsyncMock()
        bot.get_file = AsyncMock()
        bot.download_file = AsyncMock()
        bot.set_webhook = AsyncMock()
        return bot
    
    @pytest.fixture
    def telegram_service(self, mock_telegram_bot):
        """Create TelegramService with mocked bot."""
        service = TelegramService()
        service.bot = mock_telegram_bot
        return service
    
    @pytest.mark.asyncio
    async def test_send_text_message(self, telegram_service, mock_telegram_bot):
        """Test sending text message via Telegram."""
        # Mock successful message send
        mock_message = Mock()
        mock_message.message_id = 12345
        mock_message.date = datetime.utcnow()
        mock_message.text = "Hello, this is a test message!"
        
        mock_telegram_bot.send_message.return_value = mock_message
        
        # Test message sending
        result = await telegram_service.send_message(
            chat_id=123456789,
            text="Hello, this is a test message!",
            parse_mode="HTML"
        )
        
        # Verify bot call
        mock_telegram_bot.send_message.assert_called_once()
        call_args = mock_telegram_bot.send_message.call_args.kwargs
        
        assert call_args["chat_id"] == 123456789
        assert call_args["text"] == "Hello, this is a test message!"
        assert call_args["parse_mode"] == "HTML"
        
        # Verify result
        assert result["message_id"] == 12345
        assert result["sent"] is True
    
    @pytest.mark.asyncio
    async def test_send_message_with_keyboard(self, telegram_service, mock_telegram_bot):
        """Test sending message with inline keyboard."""
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        
        mock_message = Mock()
        mock_message.message_id = 54321
        mock_telegram_bot.send_message.return_value = mock_message
        
        # Create inline keyboard
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Option 1", callback_data="opt1")],
            [InlineKeyboardButton(text="Option 2", callback_data="opt2")]
        ])
        
        # Test message with keyboard
        result = await telegram_service.send_message(
            chat_id=987654321,
            text="Choose an option:",
            reply_markup=keyboard
        )
        
        # Verify keyboard was included
        call_args = mock_telegram_bot.send_message.call_args.kwargs
        assert call_args["reply_markup"] is not None
        assert result["message_id"] == 54321
    
    @pytest.mark.asyncio
    async def test_handle_telegram_error(self, telegram_service, mock_telegram_bot):
        """Test Telegram API error handling."""
        from aiogram.exceptions import TelegramBadRequest
        
        # Mock Telegram error
        mock_telegram_bot.send_message.side_effect = TelegramBadRequest(
            method="sendMessage",
            message="Bad Request: chat not found"
        )
        
        # Should handle error gracefully
        with pytest.raises(TelegramError) as exc_info:
            await telegram_service.send_message(
                chat_id=999999999,
                text="This will fail"
            )
        
        assert exc_info.value.error_type == "bad_request"
        assert "chat not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_file_upload_handling(self, telegram_service, mock_telegram_bot):
        """Test file upload via Telegram."""
        # Mock file upload response
        mock_message = Mock()
        mock_message.message_id = 99999
        mock_message.voice = Mock()
        mock_message.voice.file_id = "voice_file_123"
        mock_message.voice.duration = 5
        
        mock_telegram_bot.send_voice.return_value = mock_message
        
        # Test voice file upload
        voice_data = b"fake voice file content"
        
        result = await telegram_service.send_voice(
            chat_id=111222333,
            voice=voice_data,
            caption="Voice message test"
        )
        
        # Verify upload call
        mock_telegram_bot.send_voice.assert_called_once()
        call_args = mock_telegram_bot.send_voice.call_args.kwargs
        
        assert call_args["chat_id"] == 111222333
        assert call_args["voice"] == voice_data
        assert call_args["caption"] == "Voice message test"
        
        # Verify result
        assert result["message_id"] == 99999
        assert result["file_id"] == "voice_file_123"
        assert result["duration"] == 5
    
    @pytest.mark.asyncio
    async def test_webhook_processing(self, telegram_service, mock_telegram_bot):
        """Test Telegram webhook update processing."""
        # Mock update object
        mock_update = Mock()
        mock_update.update_id = 123456
        mock_update.message = Mock()
        mock_update.message.message_id = 789
        mock_update.message.from_user = Mock()
        mock_update.message.from_user.id = 987654321
        mock_update.message.from_user.username = "testuser"
        mock_update.message.text = "/start"
        mock_update.message.chat = Mock()
        mock_update.message.chat.id = 987654321
        
        # Test webhook processing
        result = await telegram_service.process_update(mock_update)
        
        # Verify update processing
        assert result["update_id"] == 123456
        assert result["message_id"] == 789
        assert result["user_id"] == 987654321
        assert result["text"] == "/start"
        assert result["processed"] is True
    
    @pytest.mark.asyncio
    async def test_bot_rate_limiting(self, telegram_service, mock_telegram_bot):
        """Test Telegram bot rate limiting."""
        from aiogram.exceptions import TelegramRetryAfter
        
        # Mock rate limit error
        mock_telegram_bot.send_message.side_effect = [
            TelegramRetryAfter(method="sendMessage", message="Too Many Requests", retry_after=1),
            Mock(message_id=555555)  # Success after retry
        ]
        
        # Should handle rate limiting with retry
        result = await telegram_service.send_message(
            chat_id=444555666,
            text="Rate limited message",
            retry_on_rate_limit=True
        )
        
        # Verify retry was attempted
        assert mock_telegram_bot.send_message.call_count == 2
        assert result["message_id"] == 555555
    
    @pytest.mark.asyncio
    async def test_file_download(self, telegram_service, mock_telegram_bot):
        """Test file download from Telegram."""
        # Mock file info and download
        mock_file = Mock()
        mock_file.file_id = "download_test_123"
        mock_file.file_path = "voice/file123.ogg"
        mock_file.file_size = 12345
        
        mock_telegram_bot.get_file.return_value = mock_file
        
        async def mock_download(file_path, destination):
            # Simulate file download
            with open(destination, 'wb') as f:
                f.write(b"fake downloaded content")
        
        mock_telegram_bot.download_file.side_effect = mock_download
        
        # Test file download
        local_path = "/tmp/test_download.ogg"
        result = await telegram_service.download_file(
            file_id="download_test_123",
            destination=local_path
        )
        
        # Verify download process
        mock_telegram_bot.get_file.assert_called_once_with("download_test_123")
        mock_telegram_bot.download_file.assert_called_once_with(
            "voice/file123.ogg", local_path
        )
        
        assert result["downloaded"] is True
        assert result["file_size"] == 12345
        assert result["local_path"] == local_path


class TestServiceCircuitBreaker:
    """Test circuit breaker patterns for external service resilience."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=2,
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker opening after failures."""
        # Mock failing service
        failing_service = AsyncMock(side_effect=Exception("Service unavailable"))
        
        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_service)
        
        # Circuit should now be open
        assert circuit_breaker.state == "open"
        
        # Further calls should fail fast
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(failing_service)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery mechanism."""
        # Open the circuit
        failing_service = AsyncMock(side_effect=Exception("Service down"))
        
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_service)
        
        assert circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(2.1)
        
        # Create working service for recovery test
        working_service = AsyncMock(return_value="Service recovered")
        
        # First call should transition to half-open
        result = await circuit_breaker.call(working_service)
        
        assert result == "Service recovered"
        assert circuit_breaker.state == "closed"  # Should close after successful call
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_llm_service(self, circuit_breaker):
        """Test circuit breaker integration with LLM service."""
        # Mock LLM service with intermittent failures
        mock_llm = AsyncMock()
        
        # Simulate service degradation
        call_count = 0
        
        async def flaky_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("OpenAI API error")
            return LLMResponse(content="Success after recovery", model="gpt-4")
        
        mock_llm.side_effect = flaky_llm_call
        
        # First two calls should fail
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(mock_llm, "test prompt")
        
        # Third call should succeed and close circuit
        result = await circuit_breaker.call(mock_llm, "test prompt")
        assert result.content == "Success after recovery"
        assert circuit_breaker.state == "closed"


class TestServicePerformanceMonitoring:
    """Test performance monitoring and metrics collection for external services."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor."""
        class PerformanceMonitor:
            def __init__(self):
                self.metrics = {
                    'response_times': [],
                    'error_rates': [],
                    'success_count': 0,
                    'error_count': 0
                }
            
            async def track_call(self, service_func, *args, **kwargs):
                start_time = time.time()
                try:
                    result = await service_func(*args, **kwargs)
                    self.metrics['success_count'] += 1
                    return result
                except Exception as e:
                    self.metrics['error_count'] += 1
                    raise
                finally:
                    response_time = time.time() - start_time
                    self.metrics['response_times'].append(response_time)
            
            def get_avg_response_time(self):
                if not self.metrics['response_times']:
                    return 0
                return sum(self.metrics['response_times']) / len(self.metrics['response_times'])
            
            def get_error_rate(self):
                total_calls = self.metrics['success_count'] + self.metrics['error_count']
                if total_calls == 0:
                    return 0
                return self.metrics['error_count'] / total_calls
        
        return PerformanceMonitor()
    
    @pytest.mark.asyncio
    async def test_service_response_time_monitoring(self, performance_monitor):
        """Test response time monitoring for external services."""
        # Mock services with different response times
        async def fast_service():
            await asyncio.sleep(0.01)  # 10ms
            return "fast response"
        
        async def slow_service():
            await asyncio.sleep(0.1)   # 100ms
            return "slow response"
        
        # Track multiple calls
        await performance_monitor.track_call(fast_service)
        await performance_monitor.track_call(slow_service)
        await performance_monitor.track_call(fast_service)
        
        # Verify performance metrics
        avg_response_time = performance_monitor.get_avg_response_time()
        assert 0.01 < avg_response_time < 0.1  # Should be between fast and slow
        
        response_times = performance_monitor.metrics['response_times']
        assert len(response_times) == 3
        assert min(response_times) >= 0.01
        assert max(response_times) >= 0.1
    
    @pytest.mark.asyncio
    async def test_service_error_rate_monitoring(self, performance_monitor):
        """Test error rate monitoring for external services."""
        # Mock services with varying reliability
        call_count = 0
        
        async def unreliable_service():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every third call
                raise Exception("Service failure")
            return "success"
        
        # Make multiple calls
        results = []
        for _ in range(6):
            try:
                result = await performance_monitor.track_call(unreliable_service)
                results.append(result)
            except Exception:
                pass  # Expected failures
        
        # Verify error rate calculation
        error_rate = performance_monitor.get_error_rate()
        assert error_rate == 2/6  # 2 failures out of 6 calls
        
        assert performance_monitor.metrics['success_count'] == 4
        assert performance_monitor.metrics['error_count'] == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_service_monitoring(self, performance_monitor):
        """Test monitoring of concurrent service calls."""
        # Mock concurrent service calls
        async def concurrent_service(delay: float):
            await asyncio.sleep(delay)
            return f"completed after {delay}s"
        
        # Create concurrent tasks
        delays = [0.05, 0.1, 0.03, 0.08, 0.02]
        tasks = []
        
        for delay in delays:
            task = performance_monitor.track_call(concurrent_service, delay)
            tasks.append(task)
        
        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify concurrent execution
        assert total_time < 0.15  # Should be close to max delay (0.1s), not sum of delays
        assert len(results) == 5
        assert all("completed after" in str(result) for result in results)
        
        # Verify all calls were tracked
        assert len(performance_monitor.metrics['response_times']) == 5
        assert performance_monitor.metrics['success_count'] == 5


class TestServiceIntegrationResilience:
    """Test overall service integration resilience and recovery."""
    
    @pytest.mark.asyncio
    async def test_multi_service_failure_recovery(self):
        """Test recovery when multiple external services fail."""
        # Mock multiple services
        openai_service = AsyncMock()
        stripe_service = AsyncMock()
        telegram_service = AsyncMock()
        
        # Simulate cascading failures
        openai_service.side_effect = Exception("OpenAI down")
        stripe_service.side_effect = Exception("Stripe down")
        telegram_service.return_value = {"status": "success"}  # Only Telegram works
        
        # Test service orchestration with fallbacks
        services_status = {}
        
        try:
            await openai_service()
            services_status['openai'] = 'healthy'
        except Exception:
            services_status['openai'] = 'unhealthy'
        
        try:
            await stripe_service()
            services_status['stripe'] = 'healthy'
        except Exception:
            services_status['stripe'] = 'unhealthy'
        
        try:
            result = await telegram_service()
            services_status['telegram'] = 'healthy' if result['status'] == 'success' else 'unhealthy'
        except Exception:
            services_status['telegram'] = 'unhealthy'
        
        # Verify partial functionality
        assert services_status['openai'] == 'unhealthy'
        assert services_status['stripe'] == 'unhealthy'
        assert services_status['telegram'] == 'healthy'
        
        # System should still be partially operational
        healthy_services = [k for k, v in services_status.items() if v == 'healthy']
        assert len(healthy_services) > 0
    
    @pytest.mark.asyncio
    async def test_service_dependency_chain_resilience(self):
        """Test resilience when services depend on each other."""
        # Mock service chain: Telegram -> LLM -> Response
        telegram_input = {"message": "Hello bot"}
        
        # Mock LLM service that might fail
        llm_success = True
        
        async def mock_llm_process(message):
            if not llm_success:
                raise Exception("LLM service unavailable")
            return {"response": f"Processed: {message}"}
        
        async def mock_telegram_respond(response_data):
            return {"message_sent": True, "message_id": 12345}
        
        # Test normal operation
        try:
            llm_result = await mock_llm_process(telegram_input["message"])
            telegram_result = await mock_telegram_respond(llm_result)
            
            assert telegram_result["message_sent"] is True
            normal_operation_success = True
        except Exception:
            normal_operation_success = False
        
        assert normal_operation_success is True
        
        # Test with LLM failure and fallback
        llm_success = False
        
        try:
            await mock_llm_process(telegram_input["message"])
            fallback_needed = False
        except Exception:
            fallback_needed = True
        
        if fallback_needed:
            # Use fallback response
            fallback_response = {"response": "Service temporarily unavailable. Please try again later."}
            telegram_result = await mock_telegram_respond(fallback_response)
            
            assert telegram_result["message_sent"] is True
            fallback_success = True
        else:
            fallback_success = False
        
        assert fallback_success is True


if __name__ == "__main__":
    # Run external service mocking tests
    pytest.main([__file__, "-v", "--tb=short"])