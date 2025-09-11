"""
Performance Benchmarks and Timing Tests

Comprehensive performance testing for Telegram account management:
- Response time benchmarks
- Message processing throughput
- Memory usage optimization
- Concurrent operation performance
- Timing accuracy verification
- Load testing scenarios
"""

import pytest
import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import concurrent.futures
from typing import List, Dict, Any

from app.services.telegram_account_manager import TelegramAccountManager
from app.models.telegram_account import TelegramAccount, AccountStatus
from app.models.telegram_conversation import TelegramConversation


class TestResponseTimeBenchmarks:
    """Test response time benchmarks for critical operations"""
    
    @pytest.fixture
    def benchmark_account_manager(self):
        """Create optimized account manager for benchmarking"""
        # Mock all dependencies for maximum performance
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        # Setup fast mock responses
        services['database'].get_telegram_account = AsyncMock(return_value=None)
        services['database'].update_telegram_account = AsyncMock()
        services['consciousness'].generate_response = AsyncMock(return_value="Quick response")
        services['emotion_detector'].analyze_text = AsyncMock(return_value={
            "primary_emotion": "neutral", "confidence": 0.5
        })
        services['memory'].retrieve_relevant_memories = AsyncMock(return_value=[])
        services['temporal'].analyze_interaction_timing = AsyncMock(return_value={
            "optimal_timing": True
        })
        services['telepathy'].process_communication = AsyncMock(return_value={
            "confidence": 0.8
        })
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Benchmark",
            status=AccountStatus.ACTIVE
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="benchmark_session",
            **services
        )
        manager.account = account
        
        return manager, services
    
    @pytest.mark.asyncio
    async def test_message_analysis_benchmark(self, benchmark_account_manager):
        """Benchmark message analysis performance"""
        manager, services = benchmark_account_manager
        
        conversation = TelegramConversation(
            account_id=manager.account.id,
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Create mock message
        mock_message = Mock()
        mock_message.text = "What's the best DeFi protocol for yield farming?"
        mock_message.id = 12345
        mock_message.chat = Mock()
        mock_message.chat.id = -1001234567890
        
        # Benchmark single analysis
        start_time = time.perf_counter()
        analysis = await manager._analyze_message(mock_message, conversation)
        end_time = time.perf_counter()
        
        single_analysis_time = end_time - start_time
        
        # Should complete within target time
        assert single_analysis_time < 0.1  # 100ms target
        assert "emotion" in analysis
        assert "memory_context" in analysis
        
        # Benchmark multiple analyses
        num_analyses = 100
        start_time = time.perf_counter()
        
        for i in range(num_analyses):
            mock_message.text = f"Test message {i}"
            mock_message.id = i
            await manager._analyze_message(mock_message, conversation)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / num_analyses
        
        # Average should be well under target
        assert avg_time < 0.05  # 50ms average target
        
        print(f"Message analysis benchmark:")
        print(f"  Single analysis: {single_analysis_time:.4f}s")
        print(f"  Average of {num_analyses}: {avg_time:.4f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_response_generation_benchmark(self, benchmark_account_manager):
        """Benchmark response generation performance"""
        manager, services = benchmark_account_manager
        
        conversation = TelegramConversation(
            account_id=manager.account.id,
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        mock_message = Mock()
        mock_message.text = "Can you explain yield farming strategies?"
        mock_message.chat = Mock()
        mock_message.chat.id = -1001234567890
        mock_message.id = 1
        mock_message.from_user = Mock()
        mock_message.from_user.id = 987654321
        
        # Mock Pyrogram client
        mock_client = Mock()
        mock_client.send_message = AsyncMock(return_value=Mock())
        mock_client.send_chat_action = Mock()
        mock_client.send_chat_action.return_value.__aenter__ = AsyncMock()
        mock_client.send_chat_action.return_value.__aexit__ = AsyncMock()
        
        manager.client = mock_client
        
        analysis = {
            "emotion": {"primary_emotion": "curious", "intensity": 0.7},
            "temporal_insights": {"optimal_timing": True}
        }
        
        # Benchmark response generation
        start_time = time.perf_counter()
        
        with patch.object(manager, '_calculate_typing_time', return_value=1.0), \
             patch.object(manager, '_calculate_response_delay', return_value=2.0), \
             patch.object(manager, '_store_message'), \
             patch.object(manager, '_update_response_metrics'), \
             patch.object(manager, '_get_community_context', return_value={}):
            
            await manager._generate_and_send_response(
                mock_message, conversation, analysis
            )
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        
        # Should complete quickly (excluding simulated delays)
        assert response_time < 0.5  # 500ms target
        
        print(f"Response generation benchmark: {response_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_safety_check_benchmark(self, benchmark_account_manager):
        """Benchmark safety check performance"""
        manager, services = benchmark_account_manager
        
        # Setup database mock for safety checks
        services['database'].count_recent_safety_events = AsyncMock(return_value=0)
        
        # Benchmark single safety check
        start_time = time.perf_counter()
        result = await manager._check_safety_limits()
        end_time = time.perf_counter()
        
        single_check_time = end_time - start_time
        
        # Should be very fast
        assert single_check_time < 0.01  # 10ms target
        assert result is True  # Should pass with clean account
        
        # Benchmark multiple safety checks
        num_checks = 1000
        start_time = time.perf_counter()
        
        for _ in range(num_checks):
            await manager._check_safety_limits()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / num_checks
        
        # Should maintain performance under load
        assert avg_time < 0.005  # 5ms average target
        
        print(f"Safety check benchmark:")
        print(f"  Single check: {single_check_time:.6f}s")
        print(f"  Average of {num_checks}: {avg_time:.6f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_benchmark(self, benchmark_account_manager):
        """Benchmark concurrent message analysis"""
        manager, services = benchmark_account_manager
        
        conversation = TelegramConversation(
            account_id=manager.account.id,
            user_id=987654321,
            chat_id=-1001234567890
        )
        
        # Create multiple test messages
        test_messages = []
        for i in range(50):
            mock_message = Mock()
            mock_message.text = f"Test concurrent message {i} about DeFi and crypto trading"
            mock_message.id = i
            mock_message.chat = Mock()
            mock_message.chat.id = -1001234567890
            test_messages.append(mock_message)
        
        # Benchmark concurrent processing
        start_time = time.perf_counter()
        
        # Process messages concurrently
        tasks = []
        for message in test_messages:
            task = manager._analyze_message(message, conversation)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Verify all processed
        assert len(results) == 50
        
        # Should benefit from concurrency
        avg_time_per_message = total_time / 50
        assert avg_time_per_message < 0.1  # 100ms per message
        
        print(f"Concurrent analysis benchmark:")
        print(f"  {len(test_messages)} messages in {total_time:.2f}s")
        print(f"  Average per message: {avg_time_per_message:.4f}s")


class TestThroughputBenchmarks:
    """Test throughput and capacity benchmarks"""
    
    @pytest.mark.asyncio
    async def test_message_processing_throughput(self):
        """Test message processing throughput under load"""
        # Setup high-performance manager
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        # Ultra-fast mock responses
        services['consciousness'].generate_response = AsyncMock(return_value="Fast response")
        services['emotion_detector'].analyze_text = AsyncMock(return_value={
            "primary_emotion": "neutral", "confidence": 0.5
        })
        services['memory'].retrieve_relevant_memories = AsyncMock(return_value=[])
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Throughput",
            status=AccountStatus.ACTIVE,
            max_messages_per_day=1000  # High limit for testing
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash", 
            session_name="throughput_test",
            **services
        )
        manager.account = account
        
        # Test different message volumes
        message_volumes = [10, 50, 100, 200]
        
        for volume in message_volumes:
            messages = [f"Throughput test message {i}" for i in range(volume)]
            
            start_time = time.perf_counter()
            
            # Process messages as fast as possible
            for i, message_text in enumerate(messages):
                if await manager._check_safety_limits():
                    # Simulate minimal processing
                    mock_message = Mock()
                    mock_message.text = message_text
                    mock_message.id = i
                    
                    conversation = TelegramConversation(
                        account_id=account.id,
                        user_id=987654321,
                        chat_id=-1001234567890
                    )
                    
                    await manager._analyze_message(mock_message, conversation)
                    account.messages_sent_today += 1
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = volume / duration
            
            print(f"Throughput for {volume} messages: {throughput:.1f} msg/sec")
            
            # Should maintain reasonable throughput
            if volume <= 50:
                assert throughput >= 50  # At least 50 msg/sec for small batches
            else:
                assert throughput >= 20  # At least 20 msg/sec for large batches
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_capacity(self):
        """Test capacity for handling concurrent conversations"""
        # Test different numbers of concurrent conversations
        conversation_counts = [5, 10, 20, 30]
        
        for count in conversation_counts:
            services = {
                'database': Mock(),
                'consciousness': Mock(),
                'memory': Mock(),
                'emotion_detector': Mock(),
                'temporal': Mock(),
                'telepathy': Mock()
            }
            
            # Setup mocks
            services['consciousness'].generate_response = AsyncMock(return_value="Response")
            services['emotion_detector'].analyze_text = AsyncMock(return_value={
                "primary_emotion": "neutral", "confidence": 0.5
            })
            services['memory'].retrieve_relevant_memories = AsyncMock(return_value=[])
            
            account = TelegramAccount(
                phone_number="+12345678901",
                first_name="Capacity",
                status=AccountStatus.ACTIVE
            )
            
            manager = TelegramAccountManager(
                account_id=str(account.id),
                api_id=12345,
                api_hash="test_hash",
                session_name="capacity_test",
                **services
            )
            manager.account = account
            
            # Create conversations
            conversations = []
            for i in range(count):
                conv = TelegramConversation(
                    account_id=account.id,
                    user_id=987654321 + i,
                    chat_id=-1001234567890 - i
                )
                conversations.append(conv)
            
            # Process messages from all conversations concurrently
            start_time = time.perf_counter()
            
            async def process_conversation(conv, message_text):
                mock_message = Mock()
                mock_message.text = message_text
                mock_message.id = 1
                mock_message.chat = Mock()
                mock_message.chat.id = conv.chat_id
                
                return await manager._analyze_message(mock_message, conv)
            
            # Process all concurrently
            tasks = []
            for i, conv in enumerate(conversations):
                task = process_conversation(conv, f"Message from conversation {i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            print(f"Concurrent conversations: {count} processed in {duration:.2f}s")
            
            # Should handle all conversations
            assert len(results) == count
            
            # Should complete within reasonable time
            assert duration < 5.0  # 5 seconds max
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self):
        """Test performance under sustained load"""
        services = {
            'database': Mock(),
            'consciousness': Mock(), 
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        # Setup mocks
        services['consciousness'].generate_response = AsyncMock(return_value="Response")
        services['emotion_detector'].analyze_text = AsyncMock(return_value={
            "primary_emotion": "neutral", "confidence": 0.5
        })
        services['memory'].retrieve_relevant_memories = AsyncMock(return_value=[])
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Sustained",
            status=AccountStatus.ACTIVE,
            max_messages_per_day=2000
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="sustained_test", 
            **services
        )
        manager.account = account
        
        # Run sustained load for 30 seconds
        duration_seconds = 30
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        message_count = 0
        performance_samples = []
        
        while time.perf_counter() < end_time:
            batch_start = time.perf_counter()
            
            # Process batch of messages
            for i in range(10):  # 10 messages per batch
                if await manager._check_safety_limits():
                    mock_message = Mock()
                    mock_message.text = f"Sustained load message {message_count}"
                    mock_message.id = message_count
                    
                    conversation = TelegramConversation(
                        account_id=account.id,
                        user_id=987654321,
                        chat_id=-1001234567890
                    )
                    
                    await manager._analyze_message(mock_message, conversation)
                    message_count += 1
                    account.messages_sent_today += 1
            
            batch_end = time.perf_counter()
            batch_duration = batch_end - batch_start
            batch_throughput = 10 / batch_duration
            
            performance_samples.append(batch_throughput)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Analyze sustained performance
        avg_throughput = statistics.mean(performance_samples)
        min_throughput = min(performance_samples)
        max_throughput = max(performance_samples)
        
        print(f"Sustained load results ({duration_seconds}s):")
        print(f"  Total messages: {message_count}")
        print(f"  Average throughput: {avg_throughput:.1f} msg/sec")
        print(f"  Min throughput: {min_throughput:.1f} msg/sec")
        print(f"  Max throughput: {max_throughput:.1f} msg/sec")
        
        # Performance should remain stable
        assert avg_throughput >= 10  # At least 10 msg/sec sustained
        assert min_throughput >= avg_throughput * 0.5  # No more than 50% degradation


class TestMemoryUsageBenchmarks:
    """Test memory usage and optimization"""
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under various loads"""
        # Setup manager
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        services['consciousness'].generate_response = AsyncMock(return_value="Response")
        services['emotion_detector'].analyze_text = AsyncMock(return_value={
            "primary_emotion": "neutral", "confidence": 0.5
        })
        services['memory'].retrieve_relevant_memories = AsyncMock(return_value=[])
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Memory",
            status=AccountStatus.ACTIVE
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="memory_test",
            **services
        )
        manager.account = account
        
        # Measure baseline memory
        baseline_memory = self.get_memory_usage()
        
        # Process increasing numbers of messages
        message_counts = [100, 500, 1000, 2000]
        memory_usage = []
        
        for count in message_counts:
            # Process messages
            for i in range(count):
                mock_message = Mock()
                mock_message.text = f"Memory test message {i} with some content to use memory"
                mock_message.id = i
                
                conversation = TelegramConversation(
                    account_id=account.id,
                    user_id=987654321,
                    chat_id=-1001234567890
                )
                
                await manager._analyze_message(mock_message, conversation)
            
            # Measure memory after processing
            current_memory = self.get_memory_usage()
            memory_increase = current_memory - baseline_memory
            memory_usage.append(memory_increase)
            
            print(f"Memory usage after {count} messages: +{memory_increase:.1f}MB")
        
        # Memory growth should be reasonable
        # Should not grow linearly with message count (indicates memory leaks)
        max_memory_increase = max(memory_usage)
        assert max_memory_increase < 100  # Less than 100MB increase
        
        # Later increases should not be significantly larger (no major leaks)
        if len(memory_usage) >= 2:
            growth_rate = memory_usage[-1] / memory_usage[0]
            assert growth_rate < 5  # Less than 5x growth from first to last
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_effectiveness(self):
        """Test memory cleanup and garbage collection"""
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        services['consciousness'].generate_response = AsyncMock(return_value="Response")
        services['emotion_detector'].analyze_text = AsyncMock(return_value={
            "primary_emotion": "neutral", "confidence": 0.5
        })
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Cleanup",
            status=AccountStatus.ACTIVE
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="cleanup_test",
            **services
        )
        manager.account = account
        
        baseline_memory = self.get_memory_usage()
        
        # Create and process many objects
        for cycle in range(5):
            # Create temporary objects
            temp_conversations = []
            for i in range(200):
                conversation = TelegramConversation(
                    account_id=account.id,
                    user_id=987654321 + i,
                    chat_id=-1001234567890 - i
                )
                temp_conversations.append(conversation)
                
                mock_message = Mock()
                mock_message.text = f"Cleanup test message {i}"
                mock_message.id = i
                
                await manager._analyze_message(mock_message, conversation)
            
            # Clear references
            temp_conversations.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            current_memory = self.get_memory_usage()
            memory_increase = current_memory - baseline_memory
            
            print(f"Memory after cycle {cycle + 1}: +{memory_increase:.1f}MB")
        
        # Memory should not continuously grow
        final_memory = self.get_memory_usage()
        final_increase = final_memory - baseline_memory
        
        # Should not accumulate significant memory
        assert final_increase < 50  # Less than 50MB increase after cleanup


class TestTimingAccuracyBenchmarks:
    """Test timing accuracy and consistency"""
    
    @pytest.mark.asyncio
    async def test_typing_simulation_accuracy(self):
        """Test accuracy of typing simulation timing"""
        from app.services.telegram_account_manager import TelegramAccountManager
        
        # Create manager for timing tests
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(), 
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Timing",
            status=AccountStatus.ACTIVE
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="timing_test",
            **services
        )
        
        # Test typing times for different message lengths
        test_messages = [
            "Hi",  # 2 chars
            "Hello there!",  # 12 chars
            "This is a longer message to test typing speed calculation",  # 56 chars
            "This is a very long message that simulates a detailed response about DeFi protocols and yield farming strategies that might take longer to type naturally"  # 153 chars
        ]
        
        for message in test_messages:
            # Test multiple times for consistency
            times = []
            for _ in range(20):
                typing_time = await manager._calculate_typing_time(message)
                times.append(typing_time)
            
            avg_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            # Calculate expected time (15 chars/sec baseline)
            expected_time = len(message) / 15
            
            print(f"Message length {len(message)}: avg={avg_time:.2f}s, expected≈{expected_time:.2f}s, std={std_dev:.2f}s")
            
            # Should be reasonably close to expected (within 50% due to randomization)
            assert avg_time >= expected_time * 0.5
            assert avg_time <= expected_time * 2.0
            
            # Should have some variation (not completely uniform)
            assert std_dev > 0.1  # At least 0.1 second variation
    
    @pytest.mark.asyncio
    async def test_response_delay_accuracy(self):
        """Test accuracy of response delay calculations"""
        from app.services.telegram_account_manager import TelegramAccountManager
        
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Delay",
            status=AccountStatus.ACTIVE
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="delay_test",
            **services
        )
        
        # Test different analysis scenarios
        test_scenarios = [
            {
                "name": "high_priority",
                "analysis": {
                    "high_priority": True,
                    "emotion": {"intensity": 0.9}
                },
                "expected_range": (1, 10)  # Should be fast
            },
            {
                "name": "normal",
                "analysis": {
                    "high_priority": False,
                    "emotion": {"intensity": 0.5}
                },
                "expected_range": (5, 60)  # Normal timing
            },
            {
                "name": "complex",
                "analysis": {
                    "high_priority": False,
                    "emotion": {"intensity": 0.3},
                    "temporal_insights": {"optimal_timing": False}
                },
                "expected_range": (10, 300)  # Longer delay
            }
        ]
        
        for scenario in test_scenarios:
            delays = []
            for _ in range(50):
                delay = await manager._calculate_response_delay(scenario["analysis"])
                delays.append(delay)
            
            avg_delay = statistics.mean(delays)
            min_delay = min(delays)
            max_delay = max(delays)
            
            expected_min, expected_max = scenario["expected_range"]
            
            print(f"{scenario['name']}: avg={avg_delay:.1f}s, range={min_delay:.1f}-{max_delay:.1f}s")
            
            # Should be within expected range
            assert avg_delay >= expected_min
            assert avg_delay <= expected_max
            
            # All delays should be within reasonable bounds
            assert all(2 <= delay <= 300 for delay in delays)
    
    @pytest.mark.asyncio
    async def test_timing_consistency_under_load(self):
        """Test timing consistency under concurrent load"""
        services = {
            'database': Mock(),
            'consciousness': Mock(),
            'memory': Mock(),
            'emotion_detector': Mock(),
            'temporal': Mock(),
            'telepathy': Mock()
        }
        
        account = TelegramAccount(
            phone_number="+12345678901",
            first_name="Consistency",
            status=AccountStatus.ACTIVE
        )
        
        manager = TelegramAccountManager(
            account_id=str(account.id),
            api_id=12345,
            api_hash="test_hash",
            session_name="consistency_test",
            **services
        )
        
        # Test concurrent timing calculations
        async def calculate_timing():
            analysis = {"high_priority": False, "emotion": {"intensity": 0.5}}
            return await manager._calculate_response_delay(analysis)
        
        # Run many concurrent calculations
        num_concurrent = 100
        start_time = time.perf_counter()
        
        tasks = [calculate_timing() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        avg_result = statistics.mean(results)
        std_dev = statistics.stdev(results)
        
        print(f"Concurrent timing test ({num_concurrent} calculations):")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average delay: {avg_result:.1f}s")
        print(f"  Standard deviation: {std_dev:.1f}s")
        
        # Should complete quickly
        assert total_time < 1.0  # All calculations in under 1 second
        
        # Results should be reasonable and varied
        assert 5 <= avg_result <= 60  # Reasonable average
        assert std_dev > 2.0  # Good variation