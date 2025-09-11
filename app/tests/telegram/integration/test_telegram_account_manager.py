"""
Integration Tests for TelegramAccountManager

Tests for the main account manager service including:
- Account initialization and startup
- Message processing workflow
- Response generation and timing
- Safety integration and enforcement  
- AI service integration
- Database operations
- Error handling and recovery
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pyrogram.errors import FloodWait, UserDeactivated, ChatWriteForbidden

from app.services.telegram_account_manager import TelegramAccountManager, AccountHealth, EngagementOpportunity
from app.models.telegram_account import TelegramAccount, AccountStatus, SafetyLevel
from app.models.telegram_community import TelegramCommunity, EngagementStrategy
from app.models.telegram_conversation import TelegramConversation, MessageDirection


class TestTelegramAccountManagerInit:
    """Test account manager initialization"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization_success(
        self, 
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test successful manager initialization"""
        mock_database_repository.get_telegram_account.return_value = sample_telegram_account
        
        result = await telegram_account_manager.initialize()
        
        assert result is True
        assert telegram_account_manager.account is not None
        assert telegram_account_manager.account.phone_number == "+12345678901"
        
        # Verify AI services initialization
        telegram_account_manager.consciousness.initialize.assert_called_once()
        telegram_account_manager.memory.initialize.assert_called_once()
        telegram_account_manager.telepathy.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_manager_initialization_account_not_found(
        self,
        telegram_account_manager,
        mock_database_repository
    ):
        """Test initialization failure when account not found"""
        mock_database_repository.get_telegram_account.return_value = None
        
        result = await telegram_account_manager.initialize()
        
        assert result is False
        assert telegram_account_manager.account is None
    
    @pytest.mark.asyncio
    async def test_manager_start_success(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository,
        mock_pyrogram_client
    ):
        """Test successful manager startup"""
        telegram_account_manager.account = sample_telegram_account
        telegram_account_manager.client = mock_pyrogram_client
        
        result = await telegram_account_manager.start()
        
        assert result is True
        assert telegram_account_manager.is_running is True
        
        # Verify client started and account updated
        mock_pyrogram_client.start.assert_called_once()
        assert sample_telegram_account.status == AccountStatus.ACTIVE
        mock_database_repository.update_telegram_account.assert_called()
    
    @pytest.mark.asyncio
    async def test_manager_stop(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository,
        mock_pyrogram_client
    ):
        """Test manager shutdown"""
        telegram_account_manager.account = sample_telegram_account
        telegram_account_manager.client = mock_pyrogram_client
        telegram_account_manager.is_running = True
        
        await telegram_account_manager.stop()
        
        assert telegram_account_manager.is_running is False
        mock_pyrogram_client.stop.assert_called_once()
        assert sample_telegram_account.status == AccountStatus.INACTIVE
        mock_database_repository.update_telegram_account.assert_called()


class TestMessageProcessing:
    """Test message processing workflow"""
    
    @pytest.mark.asyncio
    async def test_incoming_message_processing(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_telegram_message,
        mock_database_repository,
        mock_consciousness_mirror,
        mock_emotion_detector
    ):
        """Test processing of incoming messages"""
        telegram_account_manager.account = sample_telegram_account
        
        # Mock conversation creation
        mock_conversation = Mock()
        mock_conversation.id = "conv123"
        mock_conversation.engagement_score = 75.0
        mock_conversation.conversation_context = {}
        mock_conversation.get_recent_topics.return_value = ["DeFi", "crypto"]
        mock_conversation.update_sentiment_history = Mock()
        
        with patch.object(telegram_account_manager, '_get_or_create_conversation', return_value=mock_conversation), \
             patch.object(telegram_account_manager, '_store_message'), \
             patch.object(telegram_account_manager, '_should_respond', return_value=True), \
             patch.object(telegram_account_manager, '_generate_and_send_response'):
            
            await telegram_account_manager._process_incoming_message(mock_telegram_message)
            
            # Verify emotion analysis was called
            mock_emotion_detector.analyze_text.assert_called_once_with(mock_telegram_message.text)
            
            # Verify sentiment update
            mock_conversation.update_sentiment_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mention_processing_high_priority(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_telegram_message,
        mock_database_repository
    ):
        """Test processing of direct mentions with high priority"""
        telegram_account_manager.account = sample_telegram_account
        
        mock_conversation = Mock()
        mock_conversation.id = "conv123"
        
        with patch.object(telegram_account_manager, '_get_or_create_conversation', return_value=mock_conversation), \
             patch.object(telegram_account_manager, '_store_message'), \
             patch.object(telegram_account_manager, '_analyze_message') as mock_analyze, \
             patch.object(telegram_account_manager, '_check_safety_limits', return_value=True), \
             patch.object(telegram_account_manager, '_generate_and_send_response'):
            
            await telegram_account_manager._process_mention(mock_telegram_message)
            
            # Verify high priority analysis
            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args
            assert call_args[1]['high_priority'] is True
    
    @pytest.mark.asyncio
    async def test_message_analysis_comprehensive(
        self,
        telegram_account_manager,
        sample_telegram_account,
        sample_telegram_conversation,
        mock_telegram_message,
        mock_emotion_detector,
        mock_consciousness_mirror,
        mock_memory_palace,
        mock_temporal_archaeology,
        mock_telepathy_engine
    ):
        """Test comprehensive message analysis"""
        telegram_account_manager.account = sample_telegram_account
        
        analysis = await telegram_account_manager._analyze_message(
            mock_telegram_message, 
            sample_telegram_conversation
        )
        
        # Verify all AI services were called
        mock_emotion_detector.analyze_text.assert_called_once()
        mock_consciousness_mirror.adapt_to_context.assert_called_once()
        mock_memory_palace.retrieve_relevant_memories.assert_called_once()
        mock_temporal_archaeology.analyze_interaction_timing.assert_called_once()
        mock_telepathy_engine.process_communication.assert_called_once()
        
        # Verify analysis structure
        assert "emotion" in analysis
        assert "adapted_personality" in analysis
        assert "memory_context" in analysis
        assert "temporal_insights" in analysis
        assert "telepathy_insights" in analysis
        assert analysis["high_priority"] is False


class TestResponseGeneration:
    """Test response generation and timing"""
    
    @pytest.mark.asyncio
    async def test_response_generation_workflow(
        self,
        telegram_account_manager,
        sample_telegram_account,
        sample_telegram_conversation,
        mock_telegram_message,
        mock_pyrogram_client,
        mock_consciousness_mirror,
        mock_memory_palace
    ):
        """Test complete response generation workflow"""
        telegram_account_manager.account = sample_telegram_account
        telegram_account_manager.client = mock_pyrogram_client
        
        # Mock response generation
        test_response = "This is a helpful response about DeFi protocols!"
        mock_consciousness_mirror.generate_response.return_value = test_response
        
        # Mock sent message
        mock_sent_message = Mock()
        mock_sent_message.id = 67890
        mock_sent_message.text = test_response
        mock_pyrogram_client.send_message.return_value = mock_sent_message
        
        analysis = {
            "emotion": {"primary_emotion": "positive", "intensity": 0.7},
            "temporal_insights": {"optimal_timing": True}
        }
        
        with patch.object(telegram_account_manager, '_calculate_typing_time', return_value=2.0), \
             patch.object(telegram_account_manager, '_calculate_response_delay', return_value=5.0), \
             patch.object(telegram_account_manager, '_store_message'), \
             patch.object(telegram_account_manager, '_update_response_metrics'), \
             patch.object(telegram_account_manager, '_get_community_context', return_value={}):
            
            await telegram_account_manager._generate_and_send_response(
                mock_telegram_message,
                sample_telegram_conversation,
                analysis
            )
            
            # Verify response was generated and sent
            mock_consciousness_mirror.generate_response.assert_called_once()
            mock_pyrogram_client.send_message.assert_called_once()
            
            # Verify memory storage
            mock_memory_palace.store_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_typing_time_calculation(self, telegram_account_manager):
        """Test realistic typing time calculation"""
        short_response = "Yes"
        long_response = "This is a much longer response that would take more time to type naturally"
        
        short_time = await telegram_account_manager._calculate_typing_time(short_response)
        long_time = await telegram_account_manager._calculate_typing_time(long_response)
        
        # Longer responses should take more time
        assert long_time > short_time
        
        # Should be within reasonable bounds (1-15 seconds)
        assert 1.0 <= short_time <= 15.0
        assert 1.0 <= long_time <= 15.0
    
    @pytest.mark.asyncio
    async def test_response_delay_calculation(self, telegram_account_manager):
        """Test response delay calculation based on analysis"""
        # High priority message
        high_priority_analysis = {
            "high_priority": True,
            "emotion": {"intensity": 0.8}
        }
        
        high_priority_delay = await telegram_account_manager._calculate_response_delay(high_priority_analysis)
        
        # Normal message
        normal_analysis = {
            "high_priority": False,
            "emotion": {"intensity": 0.4}
        }
        
        normal_delay = await telegram_account_manager._calculate_response_delay(normal_analysis)
        
        # High priority should have shorter delay
        assert high_priority_delay < normal_delay
        
        # Both should be at least 2 seconds
        assert high_priority_delay >= 2.0
        assert normal_delay >= 2.0


class TestSafetyIntegration:
    """Test safety system integration"""
    
    @pytest.mark.asyncio
    async def test_safety_limits_enforcement(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test safety limits prevent actions"""
        # Set up account at daily limits
        sample_telegram_account.messages_sent_today = 50
        sample_telegram_account.last_activity_reset = datetime.utcnow()
        telegram_account_manager.account = sample_telegram_account
        
        result = await telegram_account_manager._check_safety_limits()
        assert result is False
        
        # Reset limits
        sample_telegram_account.messages_sent_today = 25
        result = await telegram_account_manager._check_safety_limits()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_flood_wait_handling(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test flood wait error handling"""
        telegram_account_manager.account = sample_telegram_account
        flood_wait_error = FloodWait(value=30)
        
        with patch('asyncio.sleep') as mock_sleep:
            await telegram_account_manager._handle_flood_wait(flood_wait_error)
            
            # Verify wait time was respected
            mock_sleep.assert_called_once_with(30)
            
            # Verify risk score increased
            assert sample_telegram_account.risk_score > 0
            
            # Verify safety event was created
            mock_database_repository.create_safety_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_high_risk_account_restrictions(
        self,
        telegram_account_manager,
        sample_telegram_account
    ):
        """Test restrictions on high-risk accounts"""
        # Set high risk score
        sample_telegram_account.risk_score = 80.0
        telegram_account_manager.account = sample_telegram_account
        
        result = await telegram_account_manager._check_safety_limits()
        assert result is False  # Should be blocked due to high risk
    
    @pytest.mark.asyncio
    async def test_safety_event_recording(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test safety event recording"""
        telegram_account_manager.account = sample_telegram_account
        
        await telegram_account_manager._record_safety_event(
            "test_event",
            "Test safety event",
            {"test_data": "value"}
        )
        
        # Verify safety event was created
        mock_database_repository.create_safety_event.assert_called_once()
        call_args = mock_database_repository.create_safety_event.call_args[0][0]
        
        assert call_args.event_type == "test_event"
        assert call_args.description == "Test safety event"
        assert call_args.data["test_data"] == "value"


class TestHealthMonitoring:
    """Test account health monitoring"""
    
    @pytest.mark.asyncio
    async def test_health_check_execution(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test health check execution"""
        telegram_account_manager.account = sample_telegram_account
        
        # Mock recent safety events count
        mock_database_repository.count_recent_safety_events.return_value = 3
        
        health = await telegram_account_manager._perform_health_check()
        
        assert isinstance(health, AccountHealth)
        assert health.is_healthy == sample_telegram_account.is_healthy
        assert health.risk_score == sample_telegram_account.risk_score
        assert health.active_warnings == 3
        assert isinstance(health.daily_limits_status, dict)
        assert isinstance(health.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_daily_counter_reset(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test automatic daily counter reset"""
        # Set counters and old reset time
        sample_telegram_account.messages_sent_today = 25
        sample_telegram_account.groups_joined_today = 2
        sample_telegram_account.dms_sent_today = 3
        sample_telegram_account.last_activity_reset = datetime.utcnow() - timedelta(days=1)
        
        telegram_account_manager.account = sample_telegram_account
        
        await telegram_account_manager._perform_health_check()
        
        # Verify counters were reset
        assert sample_telegram_account.messages_sent_today == 0
        assert sample_telegram_account.groups_joined_today == 0
        assert sample_telegram_account.dms_sent_today == 0
        
        # Verify account was updated
        mock_database_repository.update_telegram_account.assert_called()
    
    @pytest.mark.asyncio
    async def test_health_recommendations(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test health recommendation generation"""
        # Set various health issues
        sample_telegram_account.risk_score = 60.0  # High risk
        sample_telegram_account.engagement_rate = 0.1  # Low engagement
        telegram_account_manager.account = sample_telegram_account
        
        # Mock high warning count
        mock_database_repository.count_recent_safety_events.return_value = 8
        
        health = await telegram_account_manager._perform_health_check()
        
        # Should have multiple recommendations
        assert len(health.recommendations) >= 2
        assert any("reduce activity" in rec.lower() for rec in health.recommendations)
        assert any("engagement" in rec.lower() for rec in health.recommendations)


class TestEngagementProcessor:
    """Test engagement opportunity processing"""
    
    @pytest.mark.asyncio
    async def test_engagement_opportunity_creation(
        self,
        telegram_account_manager
    ):
        """Test engagement opportunity creation"""
        # Mock finding opportunities
        mock_opportunities = [
            EngagementOpportunity(
                chat_id=-1001234567890,
                message_id=12345,
                opportunity_type="reply",
                confidence=0.8,
                context={"topic": "DeFi"},
                recommended_response="Great question about DeFi!"
            )
        ]
        
        with patch.object(telegram_account_manager, '_find_engagement_opportunities', return_value=mock_opportunities), \
             patch.object(telegram_account_manager, '_check_safety_limits', return_value=True), \
             patch.object(telegram_account_manager, '_process_engagement_opportunity') as mock_process:
            
            telegram_account_manager.is_running = True
            
            # Run one iteration of engagement processor
            await telegram_account_manager._engagement_processor()
            
            # Verify opportunity was processed
            mock_process.assert_called_once_with(mock_opportunities[0])


class TestAccountStatusReporting:
    """Test account status and metrics reporting"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_status_report(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_database_repository
    ):
        """Test comprehensive account status reporting"""
        telegram_account_manager.account = sample_telegram_account
        
        # Mock database responses
        mock_database_repository.count_active_communities.return_value = 5
        mock_database_repository.count_recent_conversations.return_value = 15
        mock_database_repository.count_recent_safety_events.return_value = 2
        
        status = await telegram_account_manager.get_account_status()
        
        # Verify status structure
        assert "account_id" in status
        assert "status" in status
        assert "health" in status
        assert "metrics" in status
        assert "safety" in status
        assert "daily_activity" in status
        
        # Verify health section
        health = status["health"]
        assert "is_healthy" in health
        assert "risk_score" in health
        assert "recommendations" in health
        
        # Verify metrics
        metrics = status["metrics"]
        assert metrics["active_communities"] == 5
        assert metrics["recent_conversations"] == 15
        
        # Verify safety info
        safety = status["safety"]
        assert safety["risk_score"] == sample_telegram_account.risk_score
        assert safety["spam_warnings"] == sample_telegram_account.spam_warnings


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_pyrogram_error_handling(
        self,
        telegram_account_manager,
        sample_telegram_account,
        mock_pyrogram_client
    ):
        """Test handling of various Pyrogram errors"""
        telegram_account_manager.account = sample_telegram_account
        telegram_account_manager.client = mock_pyrogram_client
        
        # Test different error scenarios
        errors_to_test = [
            FloodWait(value=60),
            UserDeactivated(),
            ChatWriteForbidden()
        ]
        
        for error in errors_to_test:
            mock_pyrogram_client.send_message.side_effect = error
            
            with patch.object(telegram_account_manager, '_handle_flood_wait') as mock_handle_flood, \
                 patch.object(telegram_account_manager, '_record_safety_event') as mock_record:
                
                # This would normally be called during response generation
                try:
                    await mock_pyrogram_client.send_message(chat_id=123, text="test")
                except type(error):
                    if isinstance(error, FloodWait):
                        await telegram_account_manager._handle_flood_wait(error)
                    else:
                        await telegram_account_manager._record_safety_event(
                            "pyrogram_error", 
                            str(error)
                        )
                
                # Verify appropriate handler was called
                if isinstance(error, FloodWait):
                    mock_handle_flood.assert_called_once()
                else:
                    mock_record.assert_called_once()
            
            # Reset mock
            mock_pyrogram_client.send_message.side_effect = None
    
    @pytest.mark.asyncio
    async def test_ai_service_failure_handling(
        self,
        telegram_account_manager,
        sample_telegram_account,
        sample_telegram_conversation,
        mock_telegram_message,
        mock_consciousness_mirror
    ):
        """Test handling of AI service failures"""
        telegram_account_manager.account = sample_telegram_account
        
        # Mock AI service failure
        mock_consciousness_mirror.generate_response.side_effect = Exception("AI service unavailable")
        
        analysis = {"emotion": {"primary_emotion": "neutral"}}
        
        with patch.object(telegram_account_manager, '_record_safety_event') as mock_record:
            # This should handle the error gracefully
            try:
                await telegram_account_manager._generate_and_send_response(
                    mock_telegram_message,
                    sample_telegram_conversation,
                    analysis
                )
            except Exception:
                pass  # Expected to handle gracefully
            
            # Should record the error
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_error_recovery(
        self,
        telegram_account_manager,
        mock_database_repository
    ):
        """Test recovery from database errors"""
        # Mock database failure
        mock_database_repository.get_telegram_account.side_effect = Exception("Database unavailable")
        
        result = await telegram_account_manager.initialize()
        
        # Should handle gracefully and return False
        assert result is False
        assert telegram_account_manager.account is None