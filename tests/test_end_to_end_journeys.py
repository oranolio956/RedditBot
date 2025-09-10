"""
End-to-End Test Suite for Reddit Bot
Complete user journey testing with realistic scenarios and data flows.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import os

# Import test factories
from .factories import (
    UserFactory, ConversationFactory, MessageFactory, 
    GroupSessionFactory, ShareableContentFactory
)

# Mock imports for services that might not be available
try:
    from app.main import app
    from app.database import get_db
    from app.models import User, Conversation, Message, GroupSession
    from app.services.voice_processing import VoiceProcessor
    from app.services.engagement_analyzer import EngagementAnalyzer
    from app.services.group_manager import GroupManager
    from app.services.viral_engine import ViralEngine
except ImportError:
    # Create mock objects for testing environment
    app = MagicMock()
    get_db = MagicMock()
    User = MagicMock()
    Conversation = MagicMock()
    Message = MagicMock()
    GroupSession = MagicMock()
    VoiceProcessor = MagicMock()
    EngagementAnalyzer = MagicMock()
    GroupManager = MagicMock()
    ViralEngine = MagicMock()


class TestCompleteUserJourneys:
    """Test complete user journeys from registration to viral content sharing."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for testing."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    async def test_user_data(self):
        """Generate test user data for journey testing."""
        return {
            "username": "test_journey_user",
            "email": "journey@test.com",
            "password": "secure_password123",
            "preferences": {
                "voice_enabled": True,
                "engagement_level": "high",
                "viral_sharing": True
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_new_user_onboarding_journey(self, async_client, test_user_data):
        """Test complete new user onboarding from registration to first conversation."""
        journey_start = time.time()
        journey_data = {}
        
        # Step 1: User registration
        registration_response = await async_client.post("/api/v1/auth/register", json=test_user_data)
        assert registration_response.status_code == 201
        journey_data["user_id"] = registration_response.json()["user_id"]
        journey_data["access_token"] = registration_response.json()["access_token"]
        
        # Step 2: Email verification (mock)
        verification_response = await async_client.post(
            f"/api/v1/auth/verify-email/{journey_data['user_id']}"
        )
        assert verification_response.status_code == 200
        
        # Step 3: Profile completion
        profile_data = {
            "display_name": "Journey Tester",
            "bio": "Testing complete user journeys",
            "interests": ["technology", "ai", "automation"]
        }
        profile_response = await async_client.put(
            "/api/v1/users/profile",
            json=profile_data,
            headers={"Authorization": f"Bearer {journey_data['access_token']}"}
        )
        assert profile_response.status_code == 200
        
        # Step 4: First conversation initiation
        conversation_data = {
            "title": "My First AI Conversation",
            "initial_message": "Hello, I'm new here! Can you help me get started?",
            "preferences": {
                "voice_enabled": True,
                "response_style": "friendly"
            }
        }
        conversation_response = await async_client.post(
            "/api/v1/conversations",
            json=conversation_data,
            headers={"Authorization": f"Bearer {journey_data['access_token']}"}
        )
        assert conversation_response.status_code == 201
        journey_data["conversation_id"] = conversation_response.json()["conversation_id"]
        
        # Step 5: AI response generation and delivery
        await asyncio.sleep(2)  # Simulate AI processing time
        
        messages_response = await async_client.get(
            f"/api/v1/conversations/{journey_data['conversation_id']}/messages",
            headers={"Authorization": f"Bearer {journey_data['access_token']}"}
        )
        assert messages_response.status_code == 200
        messages = messages_response.json()["messages"]
        assert len(messages) >= 2  # User message + AI response
        
        journey_duration = time.time() - journey_start
        assert journey_duration < 30  # Complete onboarding should take < 30 seconds
        
        # Verify journey completion metrics
        assert "user_id" in journey_data
        assert "conversation_id" in journey_data
        assert len(messages) >= 2
    
    @pytest.mark.asyncio
    async def test_voice_conversation_complete_flow(self, async_client, test_user_data):
        """Test complete voice conversation flow from upload to response."""
        # Setup authenticated user
        auth_response = await async_client.post("/api/v1/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            # Simulate audio data (in real implementation, would be actual audio)
            temp_audio.write(b"fake_audio_data_for_testing")
            temp_audio_path = temp_audio.name
        
        try:
            # Step 1: Upload voice message
            with open(temp_audio_path, "rb") as audio_file:
                upload_response = await async_client.post(
                    "/api/v1/conversations/voice/upload",
                    files={"audio": ("test_voice.mp3", audio_file, "audio/mp3")},
                    headers=headers
                )
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            
            # Step 2: Voice processing and transcription
            process_response = await async_client.post(
                "/api/v1/conversations/voice/process",
                json={"upload_id": upload_data["upload_id"]},
                headers=headers
            )
            assert process_response.status_code == 200
            
            # Step 3: AI response generation
            await asyncio.sleep(3)  # Simulate processing time
            
            # Step 4: Voice synthesis for response
            synthesis_response = await async_client.post(
                "/api/v1/conversations/voice/synthesize",
                json={
                    "text": "Thank you for your voice message. I understand your request.",
                    "voice_settings": {"speed": 1.0, "pitch": 1.0}
                },
                headers=headers
            )
            assert synthesis_response.status_code == 200
            
            # Step 5: Verify complete voice conversation cycle
            conversation_history = await async_client.get(
                "/api/v1/conversations/recent",
                headers=headers
            )
            assert conversation_history.status_code == 200
            
        finally:
            # Cleanup
            os.unlink(temp_audio_path)
    
    @pytest.mark.asyncio
    async def test_group_conversation_lifecycle(self, async_client):
        """Test complete group conversation from creation to viral content generation."""
        # Create multiple test users for group testing
        users = []
        for i in range(5):
            user_data = {
                "username": f"group_user_{i}",
                "email": f"group_{i}@test.com",
                "password": "group_pass123"
            }
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            users.append({
                "id": response.json()["user_id"],
                "token": response.json()["access_token"]
            })
        
        # Step 1: Create group session
        group_creator = users[0]
        group_data = {
            "name": "Test Group Journey",
            "description": "Testing complete group conversation lifecycle",
            "max_participants": 10,
            "privacy_level": "private"
        }
        
        group_response = await async_client.post(
            "/api/v1/groups/create",
            json=group_data,
            headers={"Authorization": f"Bearer {group_creator['token']}"}
        )
        assert group_response.status_code == 201
        group_id = group_response.json()["group_id"]
        
        # Step 2: Add members to group
        for user in users[1:]:
            invite_response = await async_client.post(
                f"/api/v1/groups/{group_id}/invite",
                json={"user_id": user["id"]},
                headers={"Authorization": f"Bearer {group_creator['token']}"}
            )
            assert invite_response.status_code == 200
            
            # Accept invitation
            accept_response = await async_client.post(
                f"/api/v1/groups/{group_id}/join",
                headers={"Authorization": f"Bearer {user['token']}"}
            )
            assert accept_response.status_code == 200
        
        # Step 3: Generate group conversation with multiple participants
        conversation_messages = [
            "Welcome everyone to our test group!",
            "This is exciting! How does the AI handle multiple participants?",
            "I'm curious about the group dynamics analysis.",
            "Can we generate viral content from our conversation?",
            "Let's see how the engagement metrics work!"
        ]
        
        message_ids = []
        for i, message_text in enumerate(conversation_messages):
            user = users[i]
            message_response = await async_client.post(
                f"/api/v1/groups/{group_id}/messages",
                json={"content": message_text, "message_type": "text"},
                headers={"Authorization": f"Bearer {user['token']}"}
            )
            assert message_response.status_code == 201
            message_ids.append(message_response.json()["message_id"])
            
            # Wait for AI response
            await asyncio.sleep(1)
        
        # Step 4: Analyze group engagement
        engagement_response = await async_client.get(
            f"/api/v1/groups/{group_id}/analytics",
            headers={"Authorization": f"Bearer {group_creator['token']}"}
        )
        assert engagement_response.status_code == 200
        analytics = engagement_response.json()
        
        assert analytics["participant_count"] == 5
        assert analytics["message_count"] >= len(conversation_messages)
        assert "engagement_score" in analytics
        
        # Step 5: Generate viral content from group conversation
        viral_request = {
            "conversation_id": group_id,
            "content_type": "highlight_reel",
            "platforms": ["twitter", "linkedin", "reddit"],
            "tone": "engaging"
        }
        
        viral_response = await async_client.post(
            "/api/v1/viral/generate",
            json=viral_request,
            headers={"Authorization": f"Bearer {group_creator['token']}"}
        )
        assert viral_response.status_code == 200
        viral_content = viral_response.json()
        
        assert "generated_content" in viral_content
        assert len(viral_content["platform_variants"]) == 3
        assert viral_content["engagement_prediction"]["score"] > 0.5
    
    @pytest.mark.asyncio
    async def test_payment_and_premium_features_journey(self, async_client, test_user_data):
        """Test complete payment flow and premium feature access."""
        # Step 1: User login
        login_response = await async_client.post("/api/v1/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Step 2: Check current subscription status
        subscription_response = await async_client.get(
            "/api/v1/users/subscription",
            headers=headers
        )
        assert subscription_response.status_code == 200
        initial_plan = subscription_response.json()["plan_type"]
        
        # Step 3: Browse premium plans
        plans_response = await async_client.get("/api/v1/billing/plans")
        assert plans_response.status_code == 200
        available_plans = plans_response.json()["plans"]
        premium_plan = next(plan for plan in available_plans if plan["tier"] == "premium")
        
        # Step 4: Initiate payment process
        payment_intent_response = await async_client.post(
            "/api/v1/billing/create-payment-intent",
            json={
                "plan_id": premium_plan["id"],
                "payment_method": "card"
            },
            headers=headers
        )
        assert payment_intent_response.status_code == 200
        payment_data = payment_intent_response.json()
        
        # Step 5: Mock payment confirmation
        payment_confirmation = await async_client.post(
            "/api/v1/billing/confirm-payment",
            json={
                "payment_intent_id": payment_data["payment_intent_id"],
                "payment_method_id": "pm_test_card_visa"
            },
            headers=headers
        )
        assert payment_confirmation.status_code == 200
        
        # Step 6: Verify subscription upgrade
        await asyncio.sleep(2)  # Allow time for subscription processing
        
        updated_subscription = await async_client.get(
            "/api/v1/users/subscription",
            headers=headers
        )
        assert updated_subscription.status_code == 200
        assert updated_subscription.json()["plan_type"] == "premium"
        
        # Step 7: Test premium feature access
        premium_features_tests = [
            ("/api/v1/conversations/advanced-ai", {"model": "gpt-4", "temperature": 0.8}),
            ("/api/v1/voice/premium-voices", {"voice_id": "premium_voice_1"}),
            ("/api/v1/analytics/detailed", {"timeframe": "last_30_days"}),
            ("/api/v1/groups/unlimited", {"group_type": "unlimited_members"})
        ]
        
        for endpoint, test_data in premium_features_tests:
            feature_response = await async_client.post(
                endpoint,
                json=test_data,
                headers=headers
            )
            assert feature_response.status_code in [200, 201]  # Premium features should be accessible


class TestErrorRecoveryJourneys:
    """Test complete user journeys with error scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_network_interruption_recovery(self, async_client):
        """Test user journey recovery from network interruptions."""
        # Simulate user starting conversation
        user_data = {
            "username": "network_test_user",
            "email": "network@test.com", 
            "password": "network123"
        }
        
        auth_response = await async_client.post("/api/v1/auth/register", json=user_data)
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Start conversation normally
        conversation_response = await async_client.post(
            "/api/v1/conversations",
            json={"title": "Network Test", "initial_message": "Testing network recovery"},
            headers=headers
        )
        conversation_id = conversation_response.json()["conversation_id"]
        
        # Simulate network interruption during message sending
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Network timeout")
            
            # Attempt to send message during network issue
            with pytest.raises(asyncio.TimeoutError):
                await async_client.post(
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={"content": "This message should fail", "retry_on_failure": True},
                    headers=headers
                )
        
        # Test automatic retry mechanism
        retry_response = await async_client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "This message should succeed after recovery"},
            headers=headers
        )
        assert retry_response.status_code == 201
        
        # Verify conversation state was preserved
        messages_response = await async_client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            headers=headers
        )
        assert messages_response.status_code == 200
        messages = messages_response.json()["messages"]
        assert len(messages) >= 2  # Original + recovered message
    
    @pytest.mark.asyncio
    async def test_service_degradation_graceful_handling(self, async_client):
        """Test user journey during service degradation with graceful fallbacks."""
        user_data = {
            "username": "degradation_user",
            "email": "degraded@test.com",
            "password": "degraded123"
        }
        
        auth_response = await async_client.post("/api/v1/auth/register", json=user_data)
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Test with AI service temporarily unavailable
        with patch('app.services.ai_service.generate_response') as mock_ai:
            mock_ai.side_effect = Exception("AI service temporarily unavailable")
            
            # Should fallback to cached responses or basic functionality
            conversation_response = await async_client.post(
                "/api/v1/conversations",
                json={
                    "title": "Degraded Service Test",
                    "initial_message": "Testing service degradation",
                    "fallback_enabled": True
                },
                headers=headers
            )
            
            # Should succeed with degraded but functional service
            assert conversation_response.status_code in [200, 201, 202]  # 202 for degraded mode
            
            # Verify degraded mode indication
            if conversation_response.status_code == 202:
                response_data = conversation_response.json()
                assert response_data.get("service_mode") == "degraded"
                assert "fallback_message" in response_data


class TestPerformanceJourneys:
    """Test user journeys under various performance conditions."""
    
    @pytest.mark.asyncio
    async def test_high_load_user_journey(self, async_client):
        """Test user journey performance under high concurrent load."""
        async def simulate_user_journey(user_index):
            """Simulate a complete user journey."""
            journey_start = time.time()
            
            # User registration
            user_data = {
                "username": f"load_user_{user_index}",
                "email": f"load_{user_index}@test.com",
                "password": "load123"
            }
            
            auth_response = await async_client.post("/api/v1/auth/register", json=user_data)
            if auth_response.status_code != 201:
                return {"success": False, "error": "Registration failed"}
            
            access_token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Create conversation
            conversation_response = await async_client.post(
                "/api/v1/conversations",
                json={
                    "title": f"Load Test Conversation {user_index}",
                    "initial_message": "Testing under high load"
                },
                headers=headers
            )
            
            if conversation_response.status_code not in [200, 201]:
                return {"success": False, "error": "Conversation creation failed"}
            
            journey_duration = time.time() - journey_start
            return {
                "success": True,
                "duration": journey_duration,
                "user_index": user_index
            }
        
        # Run 50 concurrent user journeys
        concurrent_users = 50
        tasks = [simulate_user_journey(i) for i in range(concurrent_users)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_journeys = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_journeys = [r for r in results if not isinstance(r, dict) or not r.get("success")]
        
        success_rate = len(successful_journeys) / concurrent_users
        avg_duration = np.mean([r["duration"] for r in successful_journeys]) if successful_journeys else 0
        
        # Performance assertions
        assert success_rate >= 0.95  # 95% success rate under load
        assert avg_duration < 10.0   # Average journey under 10 seconds
        assert len(failed_journeys) < concurrent_users * 0.05  # Less than 5% failures
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_during_long_conversations(self, async_client):
        """Test memory efficiency during extended conversation sessions."""
        user_data = {
            "username": "memory_test_user",
            "email": "memory@test.com",
            "password": "memory123"
        }
        
        auth_response = await async_client.post("/api/v1/auth/register", json=user_data)
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Create conversation
        conversation_response = await async_client.post(
            "/api/v1/conversations",
            json={"title": "Memory Test", "initial_message": "Starting memory efficiency test"},
            headers=headers
        )
        conversation_id = conversation_response.json()["conversation_id"]
        
        # Monitor memory usage during extended conversation
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_readings = [initial_memory]
        
        # Send 100 messages to test memory efficiency
        for i in range(100):
            message_response = await async_client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json={"content": f"Extended conversation message {i} with some content to test memory"},
                headers=headers
            )
            assert message_response.status_code == 201
            
            # Sample memory every 10 messages
            if i % 10 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory efficiency assertions
        assert memory_growth < 100  # Less than 100MB growth for 100 messages
        
        # Check for memory leaks (memory should not continuously grow)
        memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        assert memory_trend < 1.0  # Less than 1MB growth per measurement


class TestAccessibilityJourneys:
    """Test user journeys with accessibility considerations."""
    
    @pytest.mark.asyncio
    async def test_voice_only_user_journey(self, async_client):
        """Test complete user journey for voice-only interactions."""
        # Voice-only user registration
        voice_user_data = {
            "username": "voice_only_user",
            "email": "voiceonly@test.com",
            "password": "voice123",
            "accessibility": {
                "voice_only": True,
                "screen_reader": False,
                "keyboard_only": False
            }
        }
        
        auth_response = await async_client.post("/api/v1/auth/register", json=voice_user_data)
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Voice-based navigation and interaction
        voice_navigation = await async_client.post(
            "/api/v1/accessibility/voice-commands",
            json={
                "command": "create new conversation",
                "voice_settings": {"guidance_level": "detailed"}
            },
            headers=headers
        )
        assert voice_navigation.status_code == 200
        
        # Voice-guided conversation creation
        conversation_response = await async_client.post(
            "/api/v1/conversations/voice-guided",
            json={
                "voice_instruction": "I want to start a new conversation about accessibility testing",
                "accessibility_mode": "voice_only"
            },
            headers=headers
        )
        assert conversation_response.status_code == 201
        
        conversation_id = conversation_response.json()["conversation_id"]
        
        # Verify voice-optimized responses
        messages_response = await async_client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"format": "voice_optimized"},
            headers=headers
        )
        assert messages_response.status_code == 200
        
        messages = messages_response.json()["messages"]
        for message in messages:
            assert "audio_url" in message  # Each message should have audio version
            assert "voice_metadata" in message  # Voice-specific metadata
    
    @pytest.mark.asyncio
    async def test_screen_reader_compatible_journey(self, async_client):
        """Test user journey with screen reader compatibility."""
        screen_reader_user_data = {
            "username": "screen_reader_user",
            "email": "screenreader@test.com",
            "password": "screen123",
            "accessibility": {
                "screen_reader": True,
                "high_contrast": True,
                "reduced_motion": True
            }
        }
        
        auth_response = await async_client.post("/api/v1/auth/register", json=screen_reader_user_data)
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Request accessible UI components
        ui_config_response = await async_client.get(
            "/api/v1/ui/accessibility-config",
            headers=headers
        )
        assert ui_config_response.status_code == 200
        
        ui_config = ui_config_response.json()
        assert ui_config["screen_reader_optimized"] == True
        assert ui_config["aria_labels_enabled"] == True
        assert ui_config["semantic_markup"] == True
        
        # Test keyboard navigation endpoints
        navigation_response = await async_client.get(
            "/api/v1/accessibility/keyboard-shortcuts",
            headers=headers
        )
        assert navigation_response.status_code == 200
        
        shortcuts = navigation_response.json()["shortcuts"]
        assert "new_conversation" in shortcuts
        assert "send_message" in shortcuts
        assert "navigate_menu" in shortcuts


# Performance and load testing utilities
class JourneyPerformanceMonitor:
    """Monitor performance metrics during user journey testing."""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "journey_durations": []
        }
    
    async def start_monitoring(self):
        """Start background performance monitoring."""
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop performance monitoring and return metrics."""
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        return {
            "avg_response_time": np.mean(self.metrics["response_times"]) if self.metrics["response_times"] else 0,
            "max_memory_usage": max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
            "avg_cpu_usage": np.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "total_journeys": len(self.metrics["journey_durations"])
        }
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                process = psutil.Process()
                self.metrics["memory_usage"].append(process.memory_info().rss / 1024 / 1024)  # MB
                self.metrics["cpu_usage"].append(process.cpu_percent())
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if individual readings fail
                await asyncio.sleep(1)


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
async def performance_monitor():
    """Session-wide performance monitoring."""
    monitor = JourneyPerformanceMonitor()
    await monitor.start_monitoring()
    yield monitor
    metrics = await monitor.stop_monitoring()
    
    # Log final performance metrics
    print(f"\n=== End-to-End Journey Performance Metrics ===")
    print(f"Average Response Time: {metrics['avg_response_time']:.2f}ms")
    print(f"Max Memory Usage: {metrics['max_memory_usage']:.2f}MB")
    print(f"Average CPU Usage: {metrics['avg_cpu_usage']:.2f}%")
    print(f"Total Journeys Completed: {metrics['total_journeys']}")


@pytest.fixture(autouse=True)
async def setup_test_environment():
    """Setup clean test environment for each journey test."""
    # Clear any existing test data
    # Reset service states
    # Initialize clean database state
    yield
    # Cleanup after test