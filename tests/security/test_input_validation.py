"""
Security tests for input validation and sanitization.
Tests boundary conditions, malformed data, and injection attempts.
"""

import pytest
import numpy as np
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.services.consciousness_mirror import ConsciousnessMirror, KeystrokeDynamics
from app.services.memory_palace import MemoryPalaceEngine, SpatialIndexer
from app.services.temporal_archaeology import LinguisticProfiler


class TestInputSanitization:
    """Test input sanitization for all user-provided data."""
    
    @pytest.mark.asyncio
    async def test_consciousness_mirror_message_sanitization(self):
        """Test message input sanitization in consciousness mirror."""
        mirror = ConsciousnessMirror("test_user")
        
        # Malicious message inputs
        malicious_inputs = [
            # SQL injection attempts
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            
            # XSS attempts  
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            
            # Command injection
            "; rm -rf / ;",
            "$(rm -rf /)",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            
            # Null bytes and control characters
            "test\x00message",
            "test\x1f\x7fmessage",
            
            # Extremely long input
            "A" * 100000,
            
            # Binary data
            b"\xff\xfe\xfd".decode('utf-8', errors='ignore'),
            
            # Unicode exploits
            "test\u202emessage",  # Right-to-left override
            "test\ufeffmessage",  # Zero-width no-break space
        ]
        
        for malicious_input in malicious_inputs:
            try:
                result = await mirror.process_message(malicious_input)
                
                # Verify output is sanitized
                result_str = str(result)
                
                # Should not contain dangerous patterns
                dangerous_patterns = [
                    "DROP TABLE", "rm -rf", "<script>", "javascript:",
                    "../", "..\\", "\x00", "\xff", "\xfe"
                ]
                
                for pattern in dangerous_patterns:
                    assert pattern not in result_str, \
                        f"Dangerous pattern '{pattern}' found in output"
                        
                # Output should be reasonable length
                assert len(result_str) < 10000, "Output suspiciously long"
                
            except ValueError as e:
                # Input validation rejection is acceptable
                assert "invalid" in str(e).lower() or "forbidden" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_keystroke_data_validation(self):
        """Test validation of keystroke dynamics data."""
        mirror = ConsciousnessMirror("test_user")
        
        # Malicious keystroke data
        malicious_keystroke_data = [
            # Negative timing values
            {
                "dwell_times": [-1, -10, -100],
                "flight_times": [-50, -200],
                "deletions": -5,
                "total_keys": -10
            },
            
            # Extremely large values (potential DoS)
            {
                "dwell_times": [float('inf'), 1e10, 1e20],
                "flight_times": [float('inf')],
                "deletions": 1e10,
                "total_keys": 1e10
            },
            
            # Non-numeric values
            {
                "dwell_times": ["invalid", None, {}],
                "flight_times": [True, False],
                "deletions": "many",
                "total_keys": []
            },
            
            # Empty/missing data
            {
                "dwell_times": [],
                "flight_times": None,
                "deletions": None,
                "total_keys": 0
            },
            
            # Inconsistent data
            {
                "dwell_times": [100, 200],
                "flight_times": [50],  # Mismatched lengths
                "deletions": 1000,     # More deletions than total keys
                "total_keys": 5
            }
        ]
        
        for keystroke_data in malicious_keystroke_data:
            try:
                result = await mirror.process_message(
                    "test message", 
                    keystroke_data=keystroke_data
                )
                
                # If processing succeeds, verify output is reasonable
                if 'thought_velocity' in result:
                    velocity = result['thought_velocity']
                    assert isinstance(velocity, (int, float)), "Invalid velocity type"
                    assert 0 <= velocity <= 100, f"Invalid velocity value: {velocity}"
                    assert not (np.isnan(velocity) or np.isinf(velocity)), "Invalid velocity"
                    
            except (ValueError, TypeError, KeyError) as e:
                # Input validation failure is expected for malicious data
                assert any(word in str(e).lower() for word in 
                          ["invalid", "malformed", "out of range", "type"])
    
    def test_spatial_bounds_validation(self):
        """Test validation of 3D spatial bounds in memory palace."""
        indexer = SpatialIndexer()
        
        # Invalid spatial bounds
        invalid_bounds = [
            # Wrong number of dimensions
            [1, 2, 3],  # Should be 6 values
            [1, 2, 3, 4, 5, 6, 7],  # Too many values
            [],  # Empty
            
            # Invalid values
            [float('inf'), 1, 2, 3, 4, 5],  # Infinity
            [float('nan'), 1, 2, 3, 4, 5],  # NaN
            [1, 2, 3, float('-inf'), 5, 6],  # Negative infinity
            
            # Non-numeric values
            ["a", "b", "c", "d", "e", "f"],
            [None, None, None, None, None, None],
            [True, False, True, False, True, False],
            
            # Inverted bounds (min > max)
            [5, 4, 3, 2, 1, 0],
            
            # Extremely large values (potential memory exhaustion)
            [1e20, 1e20, 1e20, 1e21, 1e21, 1e21],
        ]
        
        for i, bounds in enumerate(invalid_bounds):
            try:
                indexer.insert(f"item_{i}", bounds)
                
                # If insertion succeeds, verify it was sanitized
                if f"item_{i}" in indexer.items:
                    stored_bounds = indexer.items[f"item_{i}"]
                    
                    # Should be 6 numeric values
                    assert len(stored_bounds) == 6, "Bounds not properly validated"
                    
                    for value in stored_bounds:
                        assert isinstance(value, (int, float)), "Non-numeric value stored"
                        assert not (np.isnan(value) or np.isinf(value)), "Invalid value stored"
                        assert -1e6 <= value <= 1e6, "Value outside reasonable range"
                        
                # Should have reasonable bounds relationships
                assert stored_bounds[0] <= stored_bounds[3], "X bounds inverted"
                assert stored_bounds[1] <= stored_bounds[4], "Y bounds inverted" 
                assert stored_bounds[2] <= stored_bounds[5], "Z bounds inverted"
                    
            except (ValueError, TypeError, IndexError) as e:
                # Validation failure expected for invalid input
                pass
    
    @pytest.mark.asyncio
    async def test_message_content_limits(self):
        """Test message content size and complexity limits."""
        profiler = LinguisticProfiler()
        
        # Test various message sizes and complexities
        test_cases = [
            # Extremely long messages
            "A" * 1000000,  # 1MB of text
            
            # Deeply nested structures in JSON-like text
            '{"a":' * 10000 + '1' + '}' * 10000,
            
            # Repeated patterns that could cause regex issues
            "ab" * 50000,
            
            # Unicode stress test
            "🎉" * 10000,  # Emoji
            "测试" * 10000,  # Chinese characters
            "\u202e" * 1000,  # RTL override characters
            
            # Control characters
            "\n" * 10000,  # Many newlines
            "\t" * 10000,  # Many tabs
            "\x00" * 1000,  # Null bytes
            
            # Mixed dangerous content
            "<script>" + "a" * 100000 + "</script>",
        ]
        
        from app.models.message import Message
        
        for i, content in enumerate(test_cases):
            try:
                # Create message object
                message = Message(
                    id=i,
                    user_id="test_user",
                    content=content,
                    created_at=datetime.utcnow()
                )
                
                # Test linguistic analysis
                fingerprint = await profiler.create_fingerprint([message])
                
                # Verify analysis completed without issues
                assert fingerprint.confidence_score >= 0, "Invalid confidence score"
                assert fingerprint.vocabulary_size >= 0, "Invalid vocabulary size"
                assert fingerprint.avg_message_length >= 0, "Invalid message length"
                
                # Verify reasonable processing time (no DoS)
                import time
                start = time.time()
                
                patterns = profiler._extract_character_patterns(content)
                
                end = time.time()
                processing_time = end - start
                
                assert processing_time < 5.0, f"Processing took too long: {processing_time}s"
                
                # Verify output patterns are reasonable
                for key, value in patterns.items():
                    assert isinstance(value, (int, float)), f"Invalid pattern type for {key}"
                    assert not (np.isnan(value) or np.isinf(value)), f"Invalid pattern value for {key}"
                    
            except (ValueError, MemoryError, RecursionError) as e:
                # These errors are acceptable for malicious input
                assert any(word in str(e).lower() for word in 
                          ["too large", "invalid", "memory", "recursion", "limit"])


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        mirror = ConsciousnessMirror("test_user")
        
        # Empty message
        result = await mirror.process_message("")
        assert isinstance(result, dict), "Should handle empty message gracefully"
        
        # Whitespace only
        result = await mirror.process_message("   \n\t  ")
        assert isinstance(result, dict), "Should handle whitespace-only message"
        
        # Single character
        result = await mirror.process_message("a")
        assert isinstance(result, dict), "Should handle single character"
        
        # Unicode edge cases
        result = await mirror.process_message("\u0000")  # Null character
        assert isinstance(result, dict), "Should handle null character"
    
    def test_extreme_numeric_values(self):
        """Test handling of extreme numeric values."""
        dynamics = KeystrokeDynamics()
        
        # Test extreme values
        extreme_values = [
            0,
            float('-inf'),
            float('inf'), 
            float('nan'),
            1e-100,  # Very small
            1e100,   # Very large
            -1e100,  # Very large negative
        ]
        
        for value in extreme_values:
            try:
                dynamics.dwell_times = [value]
                dynamics.flight_times = [value] 
                dynamics.typing_speed = value
                dynamics.emotional_pressure = value
                
                # If assignment succeeds, values should be sanitized
                assert all(isinstance(v, (int, float)) for v in dynamics.dwell_times)
                assert all(not np.isnan(v) and not np.isinf(v) for v in dynamics.dwell_times)
                
            except (ValueError, OverflowError):
                # Rejection of extreme values is acceptable
                pass
    
    @pytest.mark.asyncio  
    async def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access protection."""
        import asyncio
        
        mirror = ConsciousnessMirror("test_user")
        
        async def process_message(i):
            return await mirror.process_message(f"Message {i}")
        
        # Send many messages concurrently
        tasks = [process_message(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no race conditions or corruption
        successful_results = [r for r in results if isinstance(r, dict)]
        
        assert len(successful_results) == 100, "Some messages failed to process"
        
        # Verify state consistency
        for result in successful_results:
            assert 'personality' in result, "Missing personality data"
            assert 'mirror_accuracy' in result, "Missing accuracy data"
            
            personality = result['personality']
            assert len(personality) == 5, "Wrong personality vector length"
            assert all(0 <= p <= 1 for p in personality), "Invalid personality values"


class TestMemoryExhaustion:
    """Test protection against memory exhaustion attacks."""
    
    def test_conversation_history_limits(self):
        """Test that conversation history has reasonable limits."""
        mirror = ConsciousnessMirror("test_user")
        
        # Check initial limits
        assert mirror.conversation_history.maxlen <= 10000, "History limit too high"
        assert mirror.keystroke_buffer.maxlen <= 1000, "Keystroke buffer too high"
        
        # Try to exceed limits
        initial_memory = mirror.conversation_history.maxlen
        
        # Add many items
        for i in range(initial_memory + 1000):
            mirror.conversation_history.append({
                'timestamp': datetime.utcnow(),
                'message': f'test {i}',
                'personality': [0.5] * 5,
                'features': {}
            })
        
        # Verify limit is enforced
        assert len(mirror.conversation_history) <= initial_memory, \
            "History limit not enforced"
    
    def test_spatial_index_memory_limits(self):
        """Test spatial index memory consumption limits."""
        indexer = SpatialIndexer()
        
        # Try to add many items
        item_count = 0
        try:
            for i in range(100000):  # Large number of items
                bounds = [i, i, i, i+1, i+1, i+1]
                indexer.insert(f"item_{i}", bounds)
                item_count += 1
                
                # Check memory usage periodically
                if i % 1000 == 0:
                    import sys
                    memory_usage = len(indexer.items) + (
                        sys.getsizeof(indexer.root) if indexer.root else 0
                    )
                    
                    # Should not grow without bounds
                    assert memory_usage < 100000000, "Memory usage too high"  # 100MB limit
                    
        except MemoryError:
            # Memory limit hit - this is acceptable protection
            pass
        
        # Should have processed reasonable number of items
        assert item_count > 1000, "Should handle reasonable number of items"
    
    @pytest.mark.asyncio
    async def test_linguistic_analysis_memory(self):
        """Test memory usage in linguistic analysis."""
        profiler = LinguisticProfiler()
        
        from app.models.message import Message
        
        # Create many large messages
        large_messages = []
        for i in range(100):
            # Each message is large but not unreasonable
            content = f"This is a large message number {i}. " * 1000  # ~30KB each
            message = Message(
                id=i,
                user_id="test_user", 
                content=content,
                created_at=datetime.utcnow()
            )
            large_messages.append(message)
        
        # Test analysis doesn't exhaust memory
        fingerprint = await profiler.create_fingerprint(large_messages)
        
        # Verify reasonable results
        assert fingerprint.vocabulary_size > 0, "Should extract vocabulary"
        assert fingerprint.confidence_score >= 0, "Should have valid confidence"
        
        # Check that analysis structures don't grow unbounded
        assert len(fingerprint.ngram_distribution.get('unigrams', {})) < 10000, \
            "Unigram dictionary too large"
        assert len(fingerprint.unique_phrases) < 100, \
            "Too many unique phrases stored"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])