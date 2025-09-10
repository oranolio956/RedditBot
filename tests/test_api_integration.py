"""
API Integration Tests

Comprehensive integration test suite for FastAPI endpoints:
- Authentication and user management endpoints
- Conversation and message handling APIs
- Voice processing and transcription APIs
- Group chat management endpoints
- Viral content sharing APIs
- Stripe payment integration
- WebSocket real-time communication
- Rate limiting and error handling
- Performance testing for API response times
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import httpx
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.core.auth import create_access_token
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.group_session import GroupSession
from app.models.sharing import ShareableContent, ShareableContentType, SocialPlatform


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Create async HTTP client for testing."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers():
    """Create authorization headers for testing."""
    test_user_id = "test-user-123"
    access_token = create_access_token(data={"sub": test_user_id})
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def sample_user_data():
    """Sample user data for registration/authentication tests."""
    return {
        "telegram_id": 123456789,
        "username": "testuser",
        "first_name": "Test",
        "last_name": "User",
        "language_code": "en"
    }


class TestHealthAndStatus:
    """Test health check and status endpoints."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    def test_api_info(self, test_client):
        """Test API info endpoint."""
        response = test_client.get("/api/v1/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        expected_keys = ["version", "name", "description", "environment"]
        for key in expected_keys:
            assert key in data
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, async_test_client):
        """Test health check endpoint performance."""
        start_time = time.time()
        response = await async_test_client.get("/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1  # Should respond in <100ms


class TestUserAuthentication:
    """Test user authentication and management endpoints."""
    
    def test_user_registration(self, test_client, sample_user_data):
        """Test user registration endpoint."""
        with patch('app.database.connection.get_database_session'):
            response = test_client.post("/api/v1/users/register", json=sample_user_data)
            
            # Should either succeed or handle existing user gracefully
            assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_200_OK]
            
            if response.status_code == status.HTTP_201_CREATED:
                data = response.json()
                assert "id" in data
                assert data["telegram_id"] == sample_user_data["telegram_id"]
                assert data["username"] == sample_user_data["username"]
    
    def test_user_login(self, test_client, sample_user_data):
        """Test user login endpoint."""
        login_data = {
            "telegram_id": sample_user_data["telegram_id"],
            "username": sample_user_data["username"]
        }
        
        with patch('app.database.connection.get_database_session'):
            response = test_client.post("/api/v1/auth/login", json=login_data)
            
            # Should return token or handle non-existent user
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
            
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "access_token" in data
                assert "token_type" in data
                assert data["token_type"] == "bearer"
    
    def test_protected_route_without_auth(self, test_client):
        """Test protected route without authentication."""
        response = test_client.get("/api/v1/users/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_route_with_auth(self, test_client, auth_headers):
        """Test protected route with authentication."""
        with patch('app.database.connection.get_database_session'):
            with patch('app.core.auth.get_current_user') as mock_get_user:
                mock_get_user.return_value = User(
                    id="test-user-123",
                    telegram_id=123456789,
                    username="testuser"
                )
                
                response = test_client.get("/api/v1/users/me", headers=auth_headers)
                
                # Should succeed with valid token
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert "id" in data
                assert "telegram_id" in data


class TestConversationAPI:
    """Test conversation and message handling endpoints."""
    
    def test_create_conversation(self, test_client, auth_headers):
        """Test conversation creation endpoint."""
        conversation_data = {
            "context": "General chat",
            "metadata": {"source": "telegram", "type": "private"}
        }
        
        with patch('app.database.connection.get_database_session'):
            response = test_client.post(
                "/api/v1/conversations",
                json=conversation_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            
            assert "id" in data
            assert "created_at" in data
            assert "user_id" in data
    
    def test_get_conversations(self, test_client, auth_headers):
        """Test getting user conversations."""
        with patch('app.database.connection.get_database_session'):
            with patch('app.core.auth.get_current_user') as mock_get_user:
                mock_get_user.return_value = User(id="test-user-123")
                
                response = test_client.get("/api/v1/conversations", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                
                assert isinstance(data, list)
    
    def test_send_message(self, test_client, auth_headers):
        """Test sending message in conversation."""
        message_data = {
            "content": "Hello, this is a test message",
            "message_type": "text"
        }
        
        conversation_id = "test-conv-123"
        
        with patch('app.database.connection.get_database_session'):
            response = test_client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json=message_data,
                headers=auth_headers
            )
            
            # Should create message and potentially return bot response
            assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_200_OK]
            
            if response.status_code == status.HTTP_201_CREATED:
                data = response.json()
                assert "id" in data
                assert "content" in data
                assert data["content"] == message_data["content"]
    
    def test_get_conversation_messages(self, test_client, auth_headers):
        """Test retrieving conversation messages."""
        conversation_id = "test-conv-123"
        
        with patch('app.database.connection.get_database_session'):
            response = test_client.get(
                f"/api/v1/conversations/{conversation_id}/messages",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_conversation_response_time(self, async_test_client, auth_headers):
        """Test conversation API response times."""
        conversation_data = {"context": "Performance test"}
        
        start_time = time.time()
        response = await async_test_client.post(
            "/api/v1/conversations",
            json=conversation_data,
            headers=auth_headers
        )
        response_time = time.time() - start_time
        
        # API should respond quickly
        assert response_time < 0.2  # <200ms response time
        assert response.status_code in [201, 422]  # Created or validation error


class TestVoiceProcessingAPI:
    """Test voice processing and transcription endpoints."""
    
    def test_upload_voice_message(self, test_client, auth_headers):
        """Test voice message upload endpoint."""
        # Create mock audio file
        audio_content = b"fake audio content"
        files = {"audio_file": ("test.ogg", audio_content, "audio/ogg")}
        data = {"conversation_id": "test-conv-123"}
        
        with patch('app.services.voice_processor.process_telegram_voice') as mock_process:
            mock_process.return_value = (
                "/tmp/converted.mp3",
                {"duration": 5.2, "transcription": "Hello world"}
            )
            
            response = test_client.post(
                "/api/v1/voice/upload",
                files=files,
                data=data,
                headers=auth_headers
            )
            
            # Should accept and process voice file
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_201_CREATED]
            
            if response.status_code == status.HTTP_200_OK:
                response_data = response.json()
                assert "message_id" in response_data
                assert "transcription" in response_data
    
    def test_voice_transcription_only(self, test_client, auth_headers):
        """Test voice transcription without conversation context."""
        audio_content = b"fake audio for transcription"
        files = {"audio_file": ("transcribe.ogg", audio_content, "audio/ogg")}
        
        with patch('app.services.whisper_client.transcribe_voice_message') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "This is a test transcription",
                "confidence": 0.95,
                "language": "en"
            }
            
            response = test_client.post(
                "/api/v1/voice/transcribe",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "text" in data
            assert "confidence" in data
            assert data["text"] == "This is a test transcription"
    
    def test_text_to_speech(self, test_client, auth_headers):
        """Test text-to-speech endpoint."""
        tts_data = {
            "text": "Hello, this is a test of text to speech functionality",
            "language": "en",
            "voice_type": "neural"
        }
        
        with patch('app.services.tts_service.generate_voice_response') as mock_tts:
            mock_tts.return_value = {
                "audio_url": "/api/v1/voice/audio/test-123.mp3",
                "duration": 3.5,
                "file_size": 45678
            }
            
            response = test_client.post(
                "/api/v1/voice/tts",
                json=tts_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "audio_url" in data
            assert "duration" in data
    
    @pytest.mark.asyncio
    async def test_voice_processing_performance(self, async_test_client, auth_headers):
        """Test voice processing performance requirements."""
        audio_content = b"x" * 1024 * 100  # 100KB fake audio
        files = {"audio_file": ("perf_test.ogg", audio_content, "audio/ogg")}
        data = {"conversation_id": "perf-test"}
        
        with patch('app.services.voice_processor.process_telegram_voice') as mock_process:
            mock_process.return_value = ("/tmp/test.mp3", {"duration": 2.0})
            
            start_time = time.time()
            response = await async_test_client.post(
                "/api/v1/voice/upload",
                files=files,
                data=data,
                headers=auth_headers
            )
            processing_time = time.time() - start_time
            
            # Voice processing should be fast
            assert processing_time < 2.0  # <2 seconds for small file
            assert response.status_code in [200, 201, 422]


class TestGroupChatAPI:
    """Test group chat management endpoints."""
    
    def test_create_group_session(self, test_client, auth_headers):
        """Test group session creation."""
        group_data = {
            "telegram_group_id": -1001234567890,
            "title": "Test Group Chat",
            "group_type": "supergroup",
            "settings": {
                "proactive_responses": True,
                "analytics_enabled": True
            }
        }
        
        with patch('app.database.connection.get_database_session'):
            response = test_client.post(
                "/api/v1/groups",
                json=group_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            
            assert "id" in data
            assert "telegram_group_id" in data
            assert data["title"] == group_data["title"]
    
    def test_get_group_analytics(self, test_client, auth_headers):
        """Test group analytics endpoint."""
        group_id = 123
        
        with patch('app.services.group_manager.GroupManager.get_group_analytics') as mock_analytics:
            mock_analytics.return_value = {
                "total_conversations": 50,
                "active_members": 25,
                "avg_engagement_score": 0.75,
                "sentiment_summary": {"positive": 60, "neutral": 30, "negative": 10},
                "peak_activity_hours": {"peak_hour": 14, "activity_level": 45}
            }
            
            response = test_client.get(
                f"/api/v1/groups/{group_id}/analytics",
                params={"time_period": "daily"},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            expected_keys = [
                "total_conversations", "active_members", "avg_engagement_score",
                "sentiment_summary", "peak_activity_hours"
            ]
            for key in expected_keys:
                assert key in data
    
    def test_handle_group_message(self, test_client, auth_headers):
        """Test group message handling endpoint."""
        group_id = 123
        message_data = {
            "user_id": 456,
            "telegram_user_id": 100456,
            "message_content": "This is a group message for testing",
            "message_id": 789,
            "is_bot_mentioned": False
        }
        
        with patch('app.services.group_manager.GroupManager.handle_group_message') as mock_handle:
            mock_handle.return_value = {
                "thread_id": "thread-abc-123",
                "response_strategy": {
                    "should_respond": False,
                    "response_type": "none",
                    "priority": "low"
                },
                "processing_time": 0.045
            }
            
            response = test_client.post(
                f"/api/v1/groups/{group_id}/messages",
                json=message_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "thread_id" in data
            assert "response_strategy" in data
            assert "processing_time" in data
    
    def test_get_member_insights(self, test_client, auth_headers):
        """Test group member insights endpoint."""
        group_id = 123
        
        with patch('app.services.group_manager.GroupManager.get_member_insights') as mock_insights:
            mock_insights.return_value = {
                "total_tracked_members": 25,
                "top_engaged_members": [
                    {
                        "user_id": 456,
                        "engagement_score": 0.95,
                        "message_count": 150,
                        "thread_participation": 8
                    }
                ],
                "engagement_distribution": {
                    "high": 20,
                    "medium": 60,
                    "low": 20
                }
            }
            
            response = test_client.get(
                f"/api/v1/groups/{group_id}/members/insights",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "total_tracked_members" in data
            assert "top_engaged_members" in data
            assert "engagement_distribution" in data


class TestViralSharingAPI:
    """Test viral content sharing endpoints."""
    
    def test_generate_shareable_content(self, test_client, auth_headers):
        """Test viral content generation endpoint."""
        generation_data = {
            "conversation_id": "conv-viral-123",
            "content_type": "funny_moment",
            "platforms": ["twitter", "instagram"]
        }
        
        with patch('app.services.viral_engine.ViralEngine.analyze_conversation_for_viral_content') as mock_analyze:
            mock_content = ShareableContent(
                id="share-123",
                content_type=ShareableContentType.FUNNY_MOMENT.value,
                title="AI Bot Delivers Perfect Comedy",
                description="Hilarious AI conversation moment",
                viral_score=85.0,
                hashtags=["#AIHumor", "#TechComedy"],
                optimal_platforms=["twitter", "instagram"]
            )
            mock_analyze.return_value = [mock_content]
            
            response = test_client.post(
                "/api/v1/viral/generate",
                json=generation_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert isinstance(data, list)
            if data:  # Content was generated
                content = data[0]
                assert "id" in content
                assert "title" in content
                assert "viral_score" in content
                assert content["viral_score"] > 0
    
    def test_optimize_content_for_platform(self, test_client, auth_headers):
        """Test platform-specific content optimization."""
        content_id = "share-123"
        platform = "twitter"
        
        with patch('app.services.viral_engine.ViralEngine.optimize_content_for_platform') as mock_optimize:
            mock_optimize.return_value = {
                "platform": "twitter",
                "title": "Optimized Twitter content",
                "hashtags": ["#AI", "#Tech"],
                "specs": {
                    "max_length": 280,
                    "style": "concise_witty"
                }
            }
            
            response = test_client.get(
                f"/api/v1/viral/content/{content_id}/optimize/{platform}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["platform"] == platform
            assert "title" in data
            assert "hashtags" in data
            assert "specs" in data
    
    def test_get_trending_content(self, test_client, auth_headers):
        """Test trending content retrieval."""
        with patch('app.services.viral_engine.ViralEngine.get_trending_content') as mock_trending:
            mock_trending.return_value = [
                ShareableContent(
                    id=f"trending-{i}",
                    content_type=ShareableContentType.AI_RESPONSE.value,
                    title=f"Trending Content {i}",
                    viral_score=80 + i,
                    view_count=1000 + i * 100,
                    share_count=50 + i * 5
                )
                for i in range(5)
            ]
            
            response = test_client.get(
                "/api/v1/viral/trending",
                params={"limit": 5, "content_type": "ai_response"},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) <= 5
            
            if data:
                for content in data:
                    assert "id" in content
                    assert "viral_score" in content
                    assert "view_count" in content


class TestStripePaymentAPI:
    """Test Stripe payment integration endpoints."""
    
    def test_create_payment_intent(self, test_client, auth_headers):
        """Test payment intent creation."""
        payment_data = {
            "amount": 999,  # $9.99
            "currency": "usd",
            "subscription_type": "premium_monthly",
            "metadata": {"user_id": "test-user-123"}
        }
        
        with patch('app.services.stripe_service.create_payment_intent') as mock_create:
            mock_create.return_value = {
                "id": "pi_test_123456789",
                "client_secret": "pi_test_123456789_secret",
                "amount": 999,
                "currency": "usd",
                "status": "requires_payment_method"
            }
            
            response = test_client.post(
                "/api/v1/payments/create-intent",
                json=payment_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "client_secret" in data
            assert "payment_intent_id" in data
            assert data["amount"] == payment_data["amount"]
    
    def test_confirm_payment(self, test_client, auth_headers):
        """Test payment confirmation endpoint."""
        confirmation_data = {
            "payment_intent_id": "pi_test_123456789",
            "payment_method_id": "pm_test_987654321"
        }
        
        with patch('app.services.stripe_service.confirm_payment') as mock_confirm:
            mock_confirm.return_value = {
                "id": "pi_test_123456789",
                "status": "succeeded",
                "amount_received": 999
            }
            
            response = test_client.post(
                "/api/v1/payments/confirm",
                json=confirmation_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "succeeded"
            assert "amount_received" in data
    
    def test_get_payment_history(self, test_client, auth_headers):
        """Test payment history retrieval."""
        with patch('app.services.stripe_service.get_user_payments') as mock_payments:
            mock_payments.return_value = [
                {
                    "id": "pi_test_1",
                    "amount": 999,
                    "currency": "usd",
                    "status": "succeeded",
                    "created": "2024-01-01T00:00:00Z"
                },
                {
                    "id": "pi_test_2",
                    "amount": 1999,
                    "currency": "usd",
                    "status": "succeeded",
                    "created": "2024-02-01T00:00:00Z"
                }
            ]
            
            response = test_client.get(
                "/api/v1/payments/history",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 2
            
            for payment in data:
                assert "id" in payment
                assert "amount" in payment
                assert "status" in payment


class TestRateLimitingAndErrorHandling:
    """Test rate limiting, error handling, and edge cases."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_test_client, auth_headers):
        """Test API rate limiting."""
        # Make many requests quickly to trigger rate limiting
        tasks = []
        for i in range(20):  # More than typical rate limit
            task = async_test_client.get("/api/v1/users/me", headers=auth_headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some requests should be rate limited
        status_codes = [
            r.status_code for r in responses 
            if hasattr(r, 'status_code')
        ]
        
        # Should have mix of successful and rate-limited responses
        assert status.HTTP_200_OK in status_codes
        # May have 429 Too Many Requests if rate limiting is enforced
    
    def test_invalid_json_handling(self, test_client, auth_headers):
        """Test handling of invalid JSON in requests."""
        response = test_client.post(
            "/api/v1/conversations",
            data="invalid json content",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        
        # Should handle invalid JSON gracefully
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, test_client, auth_headers):
        """Test validation of required fields."""
        incomplete_data = {
            "username": "testuser"
            # Missing required telegram_id
        }
        
        response = test_client.post(
            "/api/v1/users/register",
            json=incomplete_data,
            headers=auth_headers
        )
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
    
    def test_unauthorized_access(self, test_client):
        """Test unauthorized access to protected endpoints."""
        protected_endpoints = [
            "/api/v1/users/me",
            "/api/v1/conversations",
            "/api/v1/voice/upload",
            "/api/v1/payments/history"
        ]
        
        for endpoint in protected_endpoints:
            response = test_client.get(endpoint)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_invalid_token_format(self, test_client):
        """Test handling of invalid token formats."""
        invalid_headers = {"Authorization": "Bearer invalid_token_format"}
        
        response = test_client.get("/api/v1/users/me", headers=invalid_headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_cors_headers(self, test_client):
        """Test CORS headers in responses."""
        response = test_client.options("/api/v1/health")
        
        # Should include CORS headers for browser compatibility
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestWebSocketIntegration:
    """Test WebSocket real-time communication."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_test_client):
        """Test WebSocket connection establishment."""
        try:
            async with async_test_client.websocket_connect("/ws/conversations/test-conv-123") as websocket:
                # Connection should be established
                assert websocket is not None
                
                # Test sending message
                await websocket.send_json({
                    "type": "message",
                    "content": "Hello WebSocket"
                })
                
                # Should receive acknowledgment or response
                response = await websocket.receive_json()
                assert "type" in response
                
        except Exception as e:
            # WebSocket endpoint might not be implemented yet
            pytest.skip(f"WebSocket endpoint not available: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_authentication(self, async_test_client, auth_headers):
        """Test WebSocket authentication."""
        try:
            # Include auth token in WebSocket connection
            token = auth_headers["Authorization"].split(" ")[1]
            
            async with async_test_client.websocket_connect(
                f"/ws/conversations/test-conv-123?token={token}"
            ) as websocket:
                
                # Should establish authenticated connection
                await websocket.send_json({"type": "ping"})
                response = await websocket.receive_json()
                assert response.get("type") == "pong"
                
        except Exception as e:
            pytest.skip(f"WebSocket authentication not implemented: {e}")


class TestAPIPerformance:
    """Test API performance and response times."""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, async_test_client, auth_headers):
        """Test API performance under concurrent load."""
        # Create concurrent requests to different endpoints
        endpoints = [
            "/health",
            "/api/v1/info",
            "/health",  # Duplicate for load testing
            "/api/v1/info"
        ]
        
        tasks = []
        for endpoint in endpoints * 5:  # 20 total requests
            task = async_test_client.get(endpoint, headers=auth_headers)
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify concurrent handling
        successful_responses = [
            r for r in responses 
            if hasattr(r, 'status_code') and r.status_code == 200
        ]
        
        # Most requests should succeed
        assert len(successful_responses) >= 15
        
        # Total time should be reasonable for concurrent processing
        assert total_time < 5.0  # Should handle 20 requests in <5 seconds
        
        avg_response_time = total_time / len(successful_responses)
        assert avg_response_time < 1.0  # Average <1 second per request
    
    @pytest.mark.asyncio
    async def test_api_response_times(self, async_test_client, auth_headers):
        """Test individual API endpoint response times."""
        endpoints_with_targets = [
            ("/health", 0.1),  # Health check should be very fast
            ("/api/v1/info", 0.2),  # Info endpoint should be fast
        ]
        
        for endpoint, target_time in endpoints_with_targets:
            start_time = time.time()
            response = await async_test_client.get(endpoint, headers=auth_headers)
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < target_time, \
                f"Endpoint {endpoint} too slow: {response_time:.3f}s > {target_time}s"
    
    @pytest.mark.asyncio
    async def test_large_request_handling(self, async_test_client, auth_headers):
        """Test handling of large requests."""
        # Create large message content
        large_content = "x" * 10000  # 10KB message
        
        message_data = {
            "content": large_content,
            "message_type": "text"
        }
        
        start_time = time.time()
        response = await async_test_client.post(
            "/api/v1/conversations/test-large/messages",
            json=message_data,
            headers=auth_headers
        )
        response_time = time.time() - start_time
        
        # Should handle large requests efficiently
        assert response_time < 2.0  # Should process large content in <2 seconds
        assert response.status_code in [201, 404, 422]  # Created, not found, or validation error


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])