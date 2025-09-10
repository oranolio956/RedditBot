"""
Security tests for buffer overflow protection in spatial algorithms and recursive functions.
Tests stack overflow, memory corruption, and algorithmic complexity attacks.
"""

import pytest
import sys
import threading
from unittest.mock import patch, Mock
import gc
import resource

from app.services.memory_palace import SpatialIndexer, MemoryRoom3D, NavigationEngine


class TestRecursionLimits:
    """Test protection against stack overflow through excessive recursion."""
    
    def test_spatial_indexer_recursion_limit(self):
        """Test spatial indexer handles deep recursion gracefully."""
        indexer = SpatialIndexer()
        
        # Create deeply nested tree structure
        # Insert items in a way that forces maximum tree depth
        overlapping_items = []
        
        for i in range(1000):
            # Each item slightly overlaps with the previous one
            # This forces a linear tree structure (worst case)
            x = i * 0.1
            bounds = [x, 0, 0, x + 0.05, 1, 1]
            overlapping_items.append((f"item_{i}", bounds))
        
        # Set recursion limit to reasonable value
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1000)  # Lower limit to test protection
        
        try:
            inserted_count = 0
            for item_id, bounds in overlapping_items:
                try:
                    indexer.insert(item_id, bounds)
                    inserted_count += 1
                except RecursionError:
                    # Recursion limit hit - this is acceptable protection
                    assert inserted_count > 100, "Should handle reasonable depth"
                    break
                except RuntimeError as e:
                    if "recursion" in str(e).lower():
                        assert inserted_count > 100, "Should handle reasonable depth"
                        break
                    else:
                        raise
                        
            # If no recursion error, verify tree is still functional
            if inserted_count == len(overlapping_items):
                # Test query still works
                results = indexer.query_range([0, 0, 0, 100, 1, 1])
                assert len(results) > 0, "Tree should still be queryable"
                
        finally:
            sys.setrecursionlimit(original_limit)
    
    def test_tree_splitting_recursion(self):
        """Test tree node splitting doesn't cause infinite recursion."""
        indexer = SpatialIndexer()
        
        # Create pathological case: many items at same location
        same_location_items = []
        for i in range(100):
            # All items at exactly the same position
            bounds = [0, 0, 0, 0.001, 0.001, 0.001]
            same_location_items.append((f"duplicate_{i}", bounds))
        
        # Set lower node capacity to force more splits
        indexer.node_capacity = 4
        
        try:
            for item_id, bounds in same_location_items:
                indexer.insert(item_id, bounds)
                
            # Verify tree is still functional
            results = indexer.query_range([-1, -1, -1, 1, 1, 1])
            assert len(results) == len(same_location_items), "All items should be found"
            
        except RecursionError:
            pytest.fail("Tree splitting caused stack overflow")
        except RuntimeError as e:
            if "recursion" in str(e).lower():
                pytest.fail("Tree splitting caused recursion error")
            else:
                raise
    
    def test_query_recursion_protection(self):
        """Test query operations don't cause stack overflow."""
        indexer = SpatialIndexer()
        
        # Create very unbalanced tree
        for i in range(500):
            bounds = [i, 0, 0, i+1, 1, 1]
            indexer.insert(f"linear_{i}", bounds)
        
        # Query large range that intersects all nodes
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)  # Very low limit
        
        try:
            results = indexer.query_range([-1, -1, -1, 1000, 2, 2])
            # Should complete without stack overflow
            assert isinstance(results, list), "Query should return results"
            
        except RecursionError:
            pytest.fail("Query caused stack overflow")
        finally:
            sys.setrecursionlimit(original_limit)


class TestMemoryBounds:
    """Test protection against memory exhaustion and buffer overflows."""
    
    def test_spatial_bounds_array_protection(self):
        """Test bounds arrays don't cause buffer overflows."""
        indexer = SpatialIndexer()
        
        # Test with malformed bounds that could cause buffer overflow
        malicious_bounds = [
            # Array with wrong size (could cause out-of-bounds access)
            [1, 2],  # Too short
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Too long
            
            # Extremely large values that could cause integer overflow
            [2**63-1, 2**63-1, 2**63-1, 2**63, 2**63, 2**63],
            [-2**63, -2**63, -2**63, 2**63-1, 2**63-1, 2**63-1],
        ]
        
        for i, bounds in enumerate(malicious_bounds):
            try:
                indexer.insert(f"malicious_{i}", bounds)
                
                # If insertion succeeds, verify bounds were sanitized
                if f"malicious_{i}" in indexer.items:
                    stored_bounds = indexer.items[f"malicious_{i}"]
                    
                    # Should be exactly 6 elements
                    assert len(stored_bounds) == 6, "Bounds array not properly sized"
                    
                    # Should be within reasonable range
                    for bound in stored_bounds:
                        assert isinstance(bound, (int, float)), "Invalid bound type"
                        assert -1e10 <= bound <= 1e10, "Bound outside safe range"
                        
            except (ValueError, TypeError, IndexError):
                # Input validation should catch malformed bounds
                pass
            except OverflowError:
                # Overflow protection is working
                pass
    
    def test_memory_palace_room_limits(self):
        """Test memory palace room creation has reasonable limits."""
        room = MemoryRoom3D("test_room", "library", (0, 0, 0))
        
        # Try to create excessive number of anchors
        original_anchors = len(room.anchors)
        
        # Attempt to add many anchors (potential memory exhaustion)
        for i in range(10000):
            try:
                room.anchors.append({
                    "id": f"anchor_{i}",
                    "position": (i, i, i),
                    "memory_id": None,
                    "association_strength": 0.0,
                    "visual_marker": f"marker_{i}"
                })
            except MemoryError:
                # Memory limit protection is working
                break
                
        # Should have reasonable limit
        assert len(room.anchors) <= 1000, "Too many anchors allowed"
        assert len(room.anchors) >= original_anchors, "Should allow some anchors"
    
    def test_navigation_history_bounds(self):
        """Test navigation history has bounded memory usage."""
        navigator = NavigationEngine()
        
        # Fill navigation history to test bounds
        for i in range(1000):
            navigator.navigation_history.append({
                "from": f"room_{i}",
                "to": f"room_{i+1}",
                "timestamp": f"time_{i}",
                "path": [(j, j, j) for j in range(100)]  # Large path
            })
        
        # History should be bounded
        assert len(navigator.navigation_history) <= 100, "History not bounded"
        
        # Memory usage should be reasonable
        history_size = len(str(navigator.navigation_history))
        assert history_size < 1000000, "Navigation history using too much memory"  # 1MB limit


class TestAlgorithmicComplexity:
    """Test protection against algorithmic complexity attacks."""
    
    def test_spatial_query_complexity(self):
        """Test spatial queries have reasonable time complexity."""
        indexer = SpatialIndexer()
        
        # Create large number of items
        items_count = 1000
        for i in range(items_count):
            bounds = [i % 10, i % 10, i % 10, (i % 10) + 1, (i % 10) + 1, (i % 10) + 1]
            indexer.insert(f"item_{i}", bounds)
        
        import time
        
        # Test various query patterns
        query_patterns = [
            # Small query (should be fast)
            [0, 0, 0, 1, 1, 1],
            
            # Large query (should still be reasonable)
            [-100, -100, -100, 100, 100, 100],
            
            # Point query
            [5, 5, 5, 5.001, 5.001, 5.001],
        ]
        
        for query_bounds in query_patterns:
            start_time = time.time()
            
            results = indexer.query_range(query_bounds)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            # Query should complete quickly
            assert query_time < 1.0, f"Query took too long: {query_time}s"
            
            # Results should be reasonable
            assert isinstance(results, list), "Query should return list"
            assert len(results) <= items_count, "Too many results"
    
    def test_nearest_neighbor_complexity(self):
        """Test nearest neighbor search has bounded complexity."""
        indexer = SpatialIndexer()
        
        # Add many items
        for i in range(500):
            bounds = [i, i, i, i+1, i+1, i+1]
            indexer.insert(f"item_{i}", bounds)
        
        import time
        
        # Test nearest neighbor queries
        test_points = [
            [0, 0, 0],
            [250, 250, 250], 
            [500, 500, 500],
            [-100, -100, -100],  # Outside all items
        ]
        
        for point in test_points:
            for k in [1, 5, 10, 50]:
                start_time = time.time()
                
                neighbors = indexer.nearest_neighbors(point, k)
                
                end_time = time.time()
                query_time = end_time - start_time
                
                # Should complete quickly
                assert query_time < 2.0, f"NN query took too long: {query_time}s"
                
                # Should return at most k results
                assert len(neighbors) <= k, "Too many neighbors returned"
                
                # Results should be sorted by distance
                distances = [dist for _, dist in neighbors]
                assert distances == sorted(distances), "Results not sorted by distance"
    
    def test_tree_balancing_performance(self):
        """Test tree doesn't degrade to linear search."""
        indexer = SpatialIndexer()
        
        # Insert items in problematic order (could create unbalanced tree)
        ordered_items = []
        for i in range(200):
            bounds = [i, 0, 0, i+0.1, 1, 1]
            ordered_items.append((f"ordered_{i}", bounds))
        
        # Time insertions
        import time
        start_time = time.time()
        
        for item_id, bounds in ordered_items:
            indexer.insert(item_id, bounds)
        
        insertion_time = time.time() - start_time
        
        # Insertions should not take too long (would indicate O(n^2) behavior)
        assert insertion_time < 5.0, f"Insertions too slow: {insertion_time}s"
        
        # Query performance should still be good
        start_time = time.time()
        
        results = indexer.query_range([50, 0, 0, 150, 1, 1])
        
        query_time = time.time() - start_time
        
        assert query_time < 0.1, f"Query too slow: {query_time}s"
        assert len(results) > 0, "Should find results"


class TestThreadSafety:
    """Test thread safety and race condition protection."""
    
    def test_concurrent_spatial_insertions(self):
        """Test spatial indexer handles concurrent insertions safely."""
        indexer = SpatialIndexer()
        errors = []
        
        def insert_items(thread_id, count):
            try:
                for i in range(count):
                    bounds = [thread_id + i*0.1, 0, 0, thread_id + i*0.1 + 0.05, 1, 1]
                    indexer.insert(f"thread_{thread_id}_item_{i}", bounds)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Create multiple threads inserting concurrently
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=insert_items, args=(thread_id, 100))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive(), f"Thread didn't complete in time"
        
        # Check for errors
        if errors:
            pytest.fail(f"Concurrent access errors: {errors}")
        
        # Verify final state is consistent
        total_items = len(indexer.items)
        assert total_items <= 500, "Too many items (duplicates?)"
        assert total_items >= 400, "Too few items (race condition?)"
        
        # Verify tree structure is still valid
        results = indexer.query_range([-1, -1, -1, 10, 2, 2])
        assert isinstance(results, list), "Tree structure corrupted"
    
    def test_memory_palace_concurrent_navigation(self):
        """Test memory palace navigation is thread-safe."""
        navigator = NavigationEngine()
        errors = []
        
        def navigate_rooms(thread_id):
            try:
                for i in range(10):
                    room = MemoryRoom3D(f"room_{thread_id}_{i}", "test", (i, 0, 0))
                    # Simulate navigation (this would normally be async)
                    navigator.current_position = room.position
                    navigator.current_room = room
                    
                    navigator.navigation_history.append({
                        "from": f"prev_{thread_id}_{i}",
                        "to": f"room_{thread_id}_{i}",
                        "timestamp": f"time_{thread_id}_{i}",
                        "path": [room.position]
                    })
            except Exception as e:
                errors.append(f"Navigation thread {thread_id}: {str(e)}")
        
        # Create multiple navigation threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=navigate_rooms, args=(thread_id,))
            threads.append(thread)
        
        # Start and wait for threads
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        # Check for errors
        if errors:
            pytest.fail(f"Navigation errors: {errors}")
        
        # Verify state consistency
        assert len(navigator.navigation_history) <= 100, "History size incorrect"


class TestResourceExhaustion:
    """Test protection against resource exhaustion attacks."""
    
    def test_memory_usage_limits(self):
        """Test memory usage stays within reasonable bounds."""
        indexer = SpatialIndexer()
        
        # Monitor memory usage
        initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Add many items
        for i in range(1000):
            bounds = [i % 100, i % 100, i % 100, (i % 100) + 1, (i % 100) + 1, (i % 100) + 1]
            indexer.insert(f"mem_test_{i}", bounds)
            
            # Check memory periodically
            if i % 100 == 0:
                current_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable (platform-dependent units)
                # On Linux: KB, On macOS: bytes
                max_increase = 100 * 1024 * 1024  # 100MB
                if sys.platform == 'linux':
                    max_increase = 100 * 1024  # 100MB in KB
                
                assert memory_increase < max_increase, f"Memory usage too high: {memory_increase}"
        
        # Force garbage collection
        gc.collect()
    
    def test_file_descriptor_limits(self):
        """Test operations don't leak file descriptors."""
        # This test would be more relevant if the code opened files or sockets
        # For now, just verify no obvious leaks in object creation
        
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many objects
        for i in range(100):
            indexer = SpatialIndexer()
            for j in range(10):
                bounds = [j, j, j, j+1, j+1, j+1]
                indexer.insert(f"fd_test_{i}_{j}", bounds)
            
            # Delete reference
            del indexer
        
        # Force cleanup
        gc.collect()
        
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Should not have excessive object growth
        assert object_increase < 1000, f"Too many objects created: {object_increase}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])