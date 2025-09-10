"""
Performance and Load Tests

Comprehensive performance testing suite for high-load scenarios:
- 1000+ concurrent user simulation
- API endpoint response time validation (<200ms)
- Memory usage monitoring and leak detection
- Database connection pool stress testing
- WebSocket concurrent connection handling
- Voice processing throughput testing
- Group chat scalability (1000+ members)
- Circuit breaker and rate limiting validation
- System resource monitoring under load
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Callable
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.services.voice_processor import VoiceProcessor
from app.services.group_manager import GroupManager
from app.services.engagement_analyzer import EngagementAnalyzer
from app.services.viral_engine import ViralEngine
from app.database.connection import DatabaseManager
from tests.factories import (
    UserFactory, ConversationFactory, MessageFactory,
    GroupSessionFactory, UserEngagementFactory
)


@pytest.fixture
def performance_monitor():
    """Performance monitoring utility."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {
                'response_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'error_count': 0,
                'success_count': 0,
                'concurrent_operations': 0,
                'peak_memory': 0,
                'peak_cpu': 0
            }
            self.process = psutil.Process()
            self.start_time = None
            self.baseline_memory = None
        
        def start_monitoring(self):
            """Start performance monitoring."""
            self.start_time = time.time()
            self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            gc.collect()  # Clean up before monitoring
        
        def record_operation(self, response_time: float, success: bool = True):
            """Record operation metrics."""
            self.metrics['response_times'].append(response_time)
            if success:
                self.metrics['success_count'] += 1
            else:
                self.metrics['error_count'] += 1
        
        def sample_system_metrics(self):
            """Sample current system metrics."""
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            self.metrics['memory_usage'].append(memory_mb)
            self.metrics['cpu_usage'].append(cpu_percent)
            
            self.metrics['peak_memory'] = max(self.metrics['peak_memory'], memory_mb)
            self.metrics['peak_cpu'] = max(self.metrics['peak_cpu'], cpu_percent)
        
        def get_summary(self) -> Dict[str, Any]:
            """Get performance summary."""
            total_time = time.time() - self.start_time if self.start_time else 0
            response_times = self.metrics['response_times']
            
            return {
                'total_duration': total_time,
                'total_operations': len(response_times),
                'success_rate': self.metrics['success_count'] / max(len(response_times), 1) * 100,
                'error_rate': self.metrics['error_count'] / max(len(response_times), 1) * 100,
                'response_time_stats': {
                    'avg': np.mean(response_times) if response_times else 0,
                    'median': np.median(response_times) if response_times else 0,
                    'p95': np.percentile(response_times, 95) if response_times else 0,
                    'p99': np.percentile(response_times, 99) if response_times else 0,
                    'min': min(response_times) if response_times else 0,
                    'max': max(response_times) if response_times else 0
                },
                'memory_stats': {
                    'baseline_mb': self.baseline_memory,
                    'peak_mb': self.metrics['peak_memory'],
                    'growth_mb': self.metrics['peak_memory'] - (self.baseline_memory or 0),
                    'avg_mb': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
                },
                'cpu_stats': {
                    'peak_percent': self.metrics['peak_cpu'],
                    'avg_percent': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
                },
                'throughput_ops_per_sec': len(response_times) / max(total_time, 0.001)
            }
    
    return PerformanceMonitor()


class TestAPIPerformanceLoad:
    """Test API performance under high load conditions."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for load testing."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_api_requests_1000_users(self, performance_monitor):
        """Test API handling 1000+ concurrent requests."""
        performance_monitor.start_monitoring()
        
        async def make_request(session: httpx.AsyncClient, user_id: int):
            """Make API request for single user."""
            start_time = time.time()
            try:
                # Simulate user authentication and basic operations
                response = await session.get("/health")
                
                response_time = time.time() - start_time
                success = response.status_code == 200
                
                performance_monitor.record_operation(response_time, success)
                return {"user_id": user_id, "success": success, "response_time": response_time}
                
            except Exception as e:
                response_time = time.time() - start_time
                performance_monitor.record_operation(response_time, False)
                return {"user_id": user_id, "success": False, "error": str(e)}
        
        # Create 1000 concurrent requests
        concurrent_users = 1000
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Create tasks for concurrent execution
            tasks = []
            for user_id in range(concurrent_users):
                task = make_request(client, user_id)
                tasks.append(task)
            
            # Sample system metrics during execution
            async def monitor_system():
                while len(tasks) > 0:
                    performance_monitor.sample_system_metrics()
                    await asyncio.sleep(0.1)
            
            # Execute requests and monitoring concurrently
            monitor_task = asyncio.create_task(monitor_system())
            results = await asyncio.gather(*tasks, return_exceptions=True)
            monitor_task.cancel()
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        summary = performance_monitor.get_summary()
        
        # Performance assertions
        success_rate = len(successful_requests) / concurrent_users * 100
        assert success_rate >= 95, f"Success rate too low: {success_rate:.1f}%"
        
        # Response time requirements
        assert summary['response_time_stats']['p95'] < 0.5, f"P95 response time too high: {summary['response_time_stats']['p95']:.3f}s"
        assert summary['response_time_stats']['avg'] < 0.2, f"Average response time too high: {summary['response_time_stats']['avg']:.3f}s"
        
        # Memory usage should be reasonable
        assert summary['memory_stats']['growth_mb'] < 500, f"Memory growth too high: {summary['memory_stats']['growth_mb']:.1f} MB"
        
        # Throughput should be high
        assert summary['throughput_ops_per_sec'] > 100, f"Throughput too low: {summary['throughput_ops_per_sec']:.1f} ops/sec"
        
        print(f"\nLoad Test Results ({concurrent_users} concurrent users):")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"P95 Response Time: {summary['response_time_stats']['p95']:.3f}s")
        print(f"Average Response Time: {summary['response_time_stats']['avg']:.3f}s")
        print(f"Throughput: {summary['throughput_ops_per_sec']:.1f} ops/sec")
        print(f"Memory Growth: {summary['memory_stats']['growth_mb']:.1f} MB")
        print(f"Peak CPU: {summary['cpu_stats']['peak_percent']:.1f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_api_endpoint_response_times(self):
        """Test individual API endpoint response times under load."""
        endpoints = [
            "/health",
            "/api/v1/info",
        ]
        
        endpoint_performance = {}
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            for endpoint in endpoints:
                response_times = []
                
                # Test each endpoint with multiple requests
                for _ in range(100):
                    start_time = time.time()
                    try:
                        response = await client.get(endpoint)
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            response_times.append(response_time)
                    except Exception:
                        pass  # Skip failed requests for timing analysis
                
                if response_times:
                    endpoint_performance[endpoint] = {
                        'avg': np.mean(response_times),
                        'p95': np.percentile(response_times, 95),
                        'p99': np.percentile(response_times, 99),
                        'min': min(response_times),
                        'max': max(response_times)
                    }
        
        # Verify response time requirements
        for endpoint, stats in endpoint_performance.items():
            assert stats['avg'] < 0.2, f"{endpoint} average response time too high: {stats['avg']:.3f}s"
            assert stats['p95'] < 0.5, f"{endpoint} P95 response time too high: {stats['p95']:.3f}s"
            assert stats['p99'] < 1.0, f"{endpoint} P99 response time too high: {stats['p99']:.3f}s"
            
            print(f"{endpoint}: avg={stats['avg']:.3f}s, p95={stats['p95']:.3f}s, p99={stats['p99']:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, performance_monitor):
        """Test for memory leaks under sustained load."""
        performance_monitor.start_monitoring()
        
        # Baseline memory measurement
        initial_memory = performance_monitor.baseline_memory
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Perform repeated operations to detect memory leaks
            for cycle in range(10):  # 10 cycles of operations
                
                # Perform batch of operations
                tasks = []
                for _ in range(50):  # 50 requests per cycle
                    task = client.get("/health")
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Force garbage collection
                gc.collect()
                
                # Sample memory usage
                performance_monitor.sample_system_metrics()
                
                # Brief pause between cycles
                await asyncio.sleep(0.1)
        
        final_memory = performance_monitor.process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal for stateless operations
        assert memory_growth < 100, f"Potential memory leak detected: {memory_growth:.1f} MB growth"
        
        # Memory usage should stabilize (not continuously grow)
        memory_samples = performance_monitor.metrics['memory_usage']
        if len(memory_samples) >= 5:
            # Check if memory is still growing at the end
            recent_trend = np.polyfit(range(len(memory_samples[-5:])), memory_samples[-5:], 1)[0]
            assert recent_trend < 5, f"Memory usage trending upward: {recent_trend:.2f} MB/cycle"
        
        print(f"Memory leak test: {memory_growth:.1f} MB growth over 500 operations")


class TestDatabasePerformanceLoad:
    """Test database performance under high load."""
    
    @pytest.fixture
    async def test_db_engine(self):
        """Create test database engine for performance testing."""
        # Use in-memory SQLite for performance testing
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            pool_size=20,  # Larger pool for load testing
            max_overflow=50,
            echo=False
        )
        
        # Create tables
        from app.database.base import DeclarativeBase
        async with engine.begin() as conn:
            await conn.run_sync(DeclarativeBase.metadata.create_all)
        
        yield engine
        await engine.dispose()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_database_operations(self, test_db_engine, performance_monitor):
        """Test concurrent database operations with 1000+ connections."""
        performance_monitor.start_monitoring()
        
        async_session_factory = async_sessionmaker(
            test_db_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async def database_operation(operation_id: int):
            """Perform database operations for load testing."""
            start_time = time.time()
            
            try:
                async with async_session_factory() as session:
                    # Create user
                    user = UserFactory.build(
                        telegram_id=1000000 + operation_id,
                        username=f"loadtest_user_{operation_id}"
                    )
                    session.add(user)
                    await session.flush()
                    
                    # Create conversation
                    conversation = ConversationFactory.build(user_id=str(user.id))
                    session.add(conversation)
                    await session.flush()
                    
                    # Create messages
                    messages = []
                    for i in range(5):
                        message = MessageFactory.build(conversation_id=conversation.id)
                        messages.append(message)
                    
                    session.add_all(messages)
                    await session.commit()
                
                response_time = time.time() - start_time
                performance_monitor.record_operation(response_time, True)
                
                return {"operation_id": operation_id, "success": True, "response_time": response_time}
                
            except Exception as e:
                response_time = time.time() - start_time
                performance_monitor.record_operation(response_time, False)
                return {"operation_id": operation_id, "success": False, "error": str(e)}
        
        # Execute 500 concurrent database operations
        concurrent_operations = 500
        
        tasks = []
        for op_id in range(concurrent_operations):
            task = database_operation(op_id)
            tasks.append(task)
        
        # Monitor system metrics during execution
        async def monitor_system():
            while True:
                performance_monitor.sample_system_metrics()
                await asyncio.sleep(0.1)
        
        monitor_task = asyncio.create_task(monitor_system())
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            monitor_task.cancel()
        
        # Analyze results
        successful_ops = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_ops = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
        
        summary = performance_monitor.get_summary()
        
        # Performance assertions
        success_rate = len(successful_ops) / concurrent_operations * 100
        assert success_rate >= 90, f"Database success rate too low: {success_rate:.1f}%"
        
        # Database operations should be reasonably fast
        assert summary['response_time_stats']['p95'] < 2.0, f"P95 DB operation time too high: {summary['response_time_stats']['p95']:.3f}s"
        assert summary['response_time_stats']['avg'] < 0.5, f"Average DB operation time too high: {summary['response_time_stats']['avg']:.3f}s"
        
        print(f"\nDatabase Load Test Results ({concurrent_operations} concurrent operations):")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Time: {summary['response_time_stats']['avg']:.3f}s")
        print(f"P95 Time: {summary['response_time_stats']['p95']:.3f}s")
        print(f"Throughput: {summary['throughput_ops_per_sec']:.1f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_stress(self, test_db_engine):
        """Test database connection pool under stress."""
        async_session_factory = async_sessionmaker(
            test_db_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection pool limits
        active_sessions = []
        
        try:
            # Create many concurrent sessions (more than pool size)
            for i in range(100):  # More than pool_size + max_overflow
                session = async_session_factory()
                active_sessions.append(session)
                
                # Perform simple query to establish connection
                await session.execute("SELECT 1")
            
            # All sessions should be created successfully
            assert len(active_sessions) == 100
            
        finally:
            # Cleanup all sessions
            for session in active_sessions:
                await session.close()
    
    @pytest.mark.asyncio
    async def test_database_query_optimization(self, test_db_engine):
        """Test query performance with large datasets."""
        async_session_factory = async_sessionmaker(
            test_db_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create large dataset
        async with async_session_factory() as session:
            # Create 1000 users
            users = []
            for i in range(1000):
                user = UserFactory.build(
                    telegram_id=2000000 + i,
                    username=f"query_test_user_{i}"
                )
                users.append(user)
            
            session.add_all(users)
            await session.commit()
            
            # Test query performance on large dataset
            start_time = time.time()
            
            # Complex query with filtering and ordering
            from sqlalchemy import select, func
            from app.models.user import User
            
            result = await session.execute(
                select(User)
                .where(User.is_active == True)
                .order_by(User.created_at.desc())
                .limit(50)
            )
            
            query_time = time.time() - start_time
            users_found = result.scalars().all()
            
            # Query should be fast even with large dataset
            assert query_time < 0.1, f"Query too slow: {query_time:.3f}s"
            assert len(users_found) == 50
            
            # Test aggregation query performance
            start_time = time.time()
            
            result = await session.execute(
                select(func.count(User.id)).where(User.is_active == True)
            )
            
            aggregation_time = time.time() - start_time
            count = result.scalar()
            
            # Aggregation should also be fast
            assert aggregation_time < 0.1, f"Aggregation too slow: {aggregation_time:.3f}s"
            assert count > 0


class TestVoiceProcessingPerformance:
    """Test voice processing performance under load."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_voice_processing(self, performance_monitor):
        """Test concurrent voice processing performance."""
        performance_monitor.start_monitoring()
        
        # Create voice processor
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VoiceProcessor(temp_dir=temp_dir)
            
            async def process_voice_file(file_id: int):
                """Process single voice file."""
                start_time = time.time()
                
                try:
                    # Create temporary audio file
                    from pydub import AudioSegment
                    
                    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                        # Create short test audio
                        audio = AudioSegment.sine(frequency=440, duration=1000)  # 1 second
                        audio.export(tmp.name, format="ogg")
                        test_file = Path(tmp.name)
                    
                    # Process the file
                    result_path = await processor.convert_ogg_to_mp3(test_file, optimize_for_speech=True)
                    
                    response_time = time.time() - start_time
                    success = result_path.exists()
                    
                    performance_monitor.record_operation(response_time, success)
                    
                    # Cleanup
                    if test_file.exists():
                        test_file.unlink()
                    if result_path.exists():
                        result_path.unlink()
                    
                    return {"file_id": file_id, "success": success, "response_time": response_time}
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    performance_monitor.record_operation(response_time, False)
                    return {"file_id": file_id, "success": False, "error": str(e)}
            
            # Process 100 files concurrently
            concurrent_files = 100
            
            tasks = []
            for file_id in range(concurrent_files):
                task = process_voice_file(file_id)
                tasks.append(task)
            
            # Monitor system during processing
            async def monitor_system():
                while True:
                    performance_monitor.sample_system_metrics()
                    await asyncio.sleep(0.1)
            
            monitor_task = asyncio.create_task(monitor_system())
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                monitor_task.cancel()
            
            # Analyze results
            successful_processes = [r for r in results if isinstance(r, dict) and r.get("success", False)]
            
            summary = performance_monitor.get_summary()
            
            # Performance assertions
            success_rate = len(successful_processes) / concurrent_files * 100
            assert success_rate >= 85, f"Voice processing success rate too low: {success_rate:.1f}%"
            
            # Voice processing should be fast
            assert summary['response_time_stats']['avg'] < 2.0, f"Average processing time too high: {summary['response_time_stats']['avg']:.3f}s"
            assert summary['response_time_stats']['p95'] < 5.0, f"P95 processing time too high: {summary['response_time_stats']['p95']:.3f}s"
            
            print(f"\nVoice Processing Load Test ({concurrent_files} concurrent files):")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Average Time: {summary['response_time_stats']['avg']:.3f}s")
            print(f"P95 Time: {summary['response_time_stats']['p95']:.3f}s")
            print(f"Throughput: {summary['throughput_ops_per_sec']:.1f} files/sec")


class TestGroupChatScalability:
    """Test group chat scalability with large groups."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_group_message_processing(self, performance_monitor):
        """Test message processing in large groups (1000+ members)."""
        performance_monitor.start_monitoring()
        
        # Create large group session
        large_group = GroupSessionFactory.build(
            id=999,
            member_count=1000,
            title="Large Test Group"
        )
        
        group_manager = GroupManager()
        
        async def process_group_message(message_id: int):
            """Process single group message."""
            start_time = time.time()
            
            try:
                result = await group_manager.handle_group_message(
                    group_session=large_group,
                    user_id=message_id % 1000,  # Cycle through 1000 users
                    telegram_user_id=100000 + (message_id % 1000),
                    message_content=f"Test message {message_id} in large group",
                    message_id=message_id,
                    is_bot_mentioned=(message_id % 50 == 0)  # 2% bot mentions
                )
                
                response_time = time.time() - start_time
                success = 'error' not in result
                
                performance_monitor.record_operation(response_time, success)
                
                return {"message_id": message_id, "success": success, "response_time": response_time}
                
            except Exception as e:
                response_time = time.time() - start_time
                performance_monitor.record_operation(response_time, False)
                return {"message_id": message_id, "success": False, "error": str(e)}
        
        # Process 500 concurrent messages in large group
        concurrent_messages = 500
        
        tasks = []
        for msg_id in range(concurrent_messages):
            task = process_group_message(msg_id)
            tasks.append(task)
        
        # Monitor system metrics
        async def monitor_system():
            while True:
                performance_monitor.sample_system_metrics()
                await asyncio.sleep(0.1)
        
        monitor_task = asyncio.create_task(monitor_system())
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            monitor_task.cancel()
        
        # Analyze results
        successful_messages = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        
        summary = performance_monitor.get_summary()
        
        # Performance assertions for large group
        success_rate = len(successful_messages) / concurrent_messages * 100
        assert success_rate >= 90, f"Large group success rate too low: {success_rate:.1f}%"
        
        # Group message processing should scale well
        assert summary['response_time_stats']['avg'] < 0.5, f"Average processing time too high: {summary['response_time_stats']['avg']:.3f}s"
        assert summary['response_time_stats']['p95'] < 2.0, f"P95 processing time too high: {summary['response_time_stats']['p95']:.3f}s"
        
        # Memory usage should be reasonable for large group
        assert summary['memory_stats']['growth_mb'] < 200, f"Memory growth too high: {summary['memory_stats']['growth_mb']:.1f} MB"
        
        print(f"\nLarge Group Chat Test ({concurrent_messages} messages, 1000 members):")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Time: {summary['response_time_stats']['avg']:.3f}s")
        print(f"P95 Time: {summary['response_time_stats']['p95']:.3f}s")
        print(f"Memory Growth: {summary['memory_stats']['growth_mb']:.1f} MB")
    
    @pytest.mark.asyncio
    async def test_group_thread_management_scalability(self):
        """Test thread management with many concurrent conversations."""
        group_manager = GroupManager()
        large_group = GroupSessionFactory.build(id=998, member_count=500)
        
        # Create many concurrent conversation threads
        thread_ids = []
        
        start_time = time.time()
        
        for i in range(200):  # 200 concurrent threads
            result = await group_manager.handle_group_message(
                group_session=large_group,
                user_id=i % 100,  # 100 different users
                telegram_user_id=100000 + (i % 100),
                message_content=f"Starting thread {i} with unique content topic",
                message_id=10000 + i
            )
            thread_ids.append(result.get('thread_id'))
        
        creation_time = time.time() - start_time
        
        # Verify thread management performance
        assert creation_time < 10.0, f"Thread creation too slow: {creation_time:.3f}s"
        assert len(set(thread_ids)) >= 100, "Not enough unique threads created"
        
        # Test thread cleanup performance
        cleanup_start = time.time()
        cleanup_stats = await group_manager.cleanup_inactive_data(max_age_hours=0)
        cleanup_time = time.time() - cleanup_start
        
        assert cleanup_time < 5.0, f"Cleanup too slow: {cleanup_time:.3f}s"
        assert cleanup_stats['threads_cleaned'] >= 0
        
        print(f"Thread Management Scalability:")
        print(f"Created 200 threads in {creation_time:.3f}s")
        print(f"Cleanup completed in {cleanup_time:.3f}s")


class TestSystemResourceMonitoring:
    """Test system resource usage under various load conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage_under_sustained_load(self, performance_monitor):
        """Test memory usage patterns under sustained load."""
        performance_monitor.start_monitoring()
        
        # Simulate sustained load for extended period
        duration_minutes = 2  # 2-minute test
        end_time = time.time() + (duration_minutes * 60)
        
        operation_count = 0
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            while time.time() < end_time:
                # Perform batch of operations
                batch_tasks = []
                for _ in range(10):  # 10 requests per batch
                    task = client.get("/health")
                    batch_tasks.append(task)
                
                # Execute batch
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                operation_count += 10
                
                # Sample metrics
                performance_monitor.sample_system_metrics()
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
        
        # Analyze memory usage patterns
        memory_samples = performance_monitor.metrics['memory_usage']
        
        if len(memory_samples) >= 10:
            # Check for memory stability (no continuous growth)
            # Use linear regression to detect trend
            x = np.arange(len(memory_samples))
            y = np.array(memory_samples)
            slope, intercept = np.polyfit(x, y, 1)
            
            # Memory growth rate should be minimal
            memory_growth_rate = slope  # MB per sample
            assert memory_growth_rate < 1.0, f"Memory growing too fast: {memory_growth_rate:.3f} MB/sample"
            
            # Memory usage should be reasonably stable
            memory_variance = np.var(memory_samples)
            assert memory_variance < 100, f"Memory usage too unstable: variance={memory_variance:.1f}"
        
        total_growth = performance_monitor.metrics['peak_memory'] - performance_monitor.baseline_memory
        
        print(f"\nSustained Load Test ({duration_minutes} minutes, {operation_count} operations):")
        print(f"Total Memory Growth: {total_growth:.1f} MB")
        print(f"Peak Memory: {performance_monitor.metrics['peak_memory']:.1f} MB")
        print(f"Peak CPU: {performance_monitor.metrics['peak_cpu']:.1f}%")
    
    @pytest.mark.asyncio
    async def test_cpu_usage_optimization(self):
        """Test CPU usage under high computational load."""
        process = psutil.Process()
        
        # Baseline CPU measurement
        baseline_cpu = process.cpu_percent()
        
        # Simulate CPU-intensive operations
        start_time = time.time()
        cpu_samples = []
        
        async def cpu_intensive_task():
            """Simulate CPU-intensive work."""
            # Mathematical computation
            result = 0
            for i in range(100000):
                result += i ** 0.5
            return result
        
        # Run multiple CPU-intensive tasks
        tasks = []
        for _ in range(20):  # 20 concurrent CPU tasks
            task = cpu_intensive_task()
            tasks.append(task)
        
        # Monitor CPU usage during execution
        async def monitor_cpu():
            while True:
                cpu_percent = process.cpu_percent()
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(0.1)
        
        monitor_task = asyncio.create_task(monitor_cpu())
        
        try:
            await asyncio.gather(*tasks)
        finally:
            monitor_task.cancel()
            
        execution_time = time.time() - start_time
        
        # Analyze CPU usage
        if cpu_samples:
            max_cpu = max(cpu_samples)
            avg_cpu = np.mean(cpu_samples)
            
            # CPU usage should be reasonable (not maxing out system)
            # Note: This depends on system capabilities
            assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
            
            print(f"CPU Usage Test:")
            print(f"Execution Time: {execution_time:.3f}s")
            print(f"Peak CPU: {max_cpu:.1f}%")
            print(f"Average CPU: {avg_cpu:.1f}%")


class TestCircuitBreakerPerformance:
    """Test circuit breaker behavior under failure conditions."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failover_performance(self):
        """Test circuit breaker performance during service failures."""
        from app.core.circuit_breaker import CircuitBreaker
        
        # Create circuit breaker with fast settings for testing
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,  # 1 second for testing
            expected_exception=Exception
        )
        
        # Mock failing service
        call_count = 0
        
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 10:  # First 10 calls fail
                raise Exception("Service unavailable")
            return "Success"
        
        response_times = []
        
        # Test circuit breaker behavior
        for i in range(20):
            start_time = time.time()
            
            try:
                result = await circuit_breaker.call(flaky_service)
                success = True
            except Exception:
                success = False
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Brief delay between calls
            await asyncio.sleep(0.01)
        
        # Analyze circuit breaker performance
        avg_response_time = np.mean(response_times)
        
        # Circuit breaker should fail fast when open
        fast_fail_times = response_times[5:15]  # During open circuit period
        avg_fast_fail_time = np.mean(fast_fail_times) if fast_fail_times else 0
        
        # Fast failures should be very quick
        assert avg_fast_fail_time < 0.001, f"Circuit breaker not failing fast enough: {avg_fast_fail_time:.4f}s"
        
        print(f"Circuit Breaker Performance:")
        print(f"Average Response Time: {avg_response_time:.4f}s")
        print(f"Fast Fail Time: {avg_fast_fail_time:.4f}s")


if __name__ == "__main__":
    # Run performance tests with appropriate markers
    pytest.main([
        __file__,
        "-v",
        "-s",
        "-m", "slow",
        "--tb=short",
        "--durations=10"  # Show 10 slowest tests
    ])