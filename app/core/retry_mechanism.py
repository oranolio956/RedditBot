"""
Retry Mechanism

Advanced retry logic with exponential backoff, jitter, and circuit breaker integration
for production-grade resilience.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional, Type, Union, List
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    FIBONACCI_BACKOFF = "fibonacci_backoff"

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_factor: float = 2.0
    exceptions: tuple = (Exception,)
    retry_on_result: Optional[Callable[[Any], bool]] = None

class RetryMechanism:
    """Production-grade retry mechanism with advanced features"""
    
    def __init__(self):
        self.retry_stats = {}
        self.fibonacci_cache = {0: 0, 1: 1}
    
    def with_retry(self, config: RetryConfig = None):
        """Decorator for adding retry logic to functions"""
        if config is None:
            config = RetryConfig()
        
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_with_retry_async(func, config, *args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._execute_with_retry_sync(func, config, *args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    async def _execute_with_retry_async(self, func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
        """Execute async function with retry logic"""
        func_name = f"{func.__module__}.{func.__name__}"
        attempt = 0
        last_exception = None
        start_time = time.time()
        
        # Initialize stats
        if func_name not in self.retry_stats:
            self.retry_stats[func_name] = {
                'total_calls': 0,
                'total_retries': 0,
                'success_rate': 1.0,
                'avg_attempts': 1.0,
                'last_failure': None
            }
        
        stats = self.retry_stats[func_name]
        stats['total_calls'] += 1
        
        for attempt in range(config.max_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{config.max_attempts} for {func_name}")
                
                result = await func(*args, **kwargs)
                
                # Check if result should trigger retry
                if config.retry_on_result and config.retry_on_result(result):
                    raise Exception(f"Result-based retry triggered for {func_name}")
                
                # Success - update stats
                execution_time = time.time() - start_time
                stats['avg_attempts'] = (stats['avg_attempts'] * (stats['total_calls'] - 1) + (attempt + 1)) / stats['total_calls']
                stats['success_rate'] = (stats['success_rate'] * (stats['total_calls'] - 1) + 1.0) / stats['total_calls']
                
                if attempt > 0:
                    logger.info(f"Function {func_name} succeeded on attempt {attempt + 1} after {execution_time:.2f}s")
                
                return result
                
            except config.exceptions as e:
                last_exception = e
                attempt_num = attempt + 1
                
                logger.warning(f"Attempt {attempt_num}/{config.max_attempts} failed for {func_name}: {str(e)}")
                
                # Don't wait after the last attempt
                if attempt_num < config.max_attempts:
                    delay = self._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before retry {attempt_num + 1}")
                    await asyncio.sleep(delay)
                
                stats['total_retries'] += 1
                stats['last_failure'] = datetime.now().isoformat()
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func_name}: {str(e)}")
                stats['success_rate'] = (stats['success_rate'] * (stats['total_calls'] - 1) + 0.0) / stats['total_calls']
                raise
        
        # All attempts failed
        execution_time = time.time() - start_time
        stats['avg_attempts'] = (stats['avg_attempts'] * (stats['total_calls'] - 1) + config.max_attempts) / stats['total_calls']
        stats['success_rate'] = (stats['success_rate'] * (stats['total_calls'] - 1) + 0.0) / stats['total_calls']
        
        logger.error(f"All {config.max_attempts} attempts failed for {func_name} after {execution_time:.2f}s")
        raise last_exception
    
    def _execute_with_retry_sync(self, func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
        """Execute sync function with retry logic"""
        func_name = f"{func.__module__}.{func.__name__}"
        attempt = 0
        last_exception = None
        start_time = time.time()
        
        # Initialize stats
        if func_name not in self.retry_stats:
            self.retry_stats[func_name] = {
                'total_calls': 0,
                'total_retries': 0,
                'success_rate': 1.0,
                'avg_attempts': 1.0,
                'last_failure': None
            }
        
        stats = self.retry_stats[func_name]
        stats['total_calls'] += 1
        
        for attempt in range(config.max_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{config.max_attempts} for {func_name}")
                
                result = func(*args, **kwargs)
                
                # Check if result should trigger retry
                if config.retry_on_result and config.retry_on_result(result):
                    raise Exception(f"Result-based retry triggered for {func_name}")
                
                # Success - update stats
                execution_time = time.time() - start_time
                stats['avg_attempts'] = (stats['avg_attempts'] * (stats['total_calls'] - 1) + (attempt + 1)) / stats['total_calls']
                stats['success_rate'] = (stats['success_rate'] * (stats['total_calls'] - 1) + 1.0) / stats['total_calls']
                
                if attempt > 0:
                    logger.info(f"Function {func_name} succeeded on attempt {attempt + 1} after {execution_time:.2f}s")
                
                return result
                
            except config.exceptions as e:
                last_exception = e
                attempt_num = attempt + 1
                
                logger.warning(f"Attempt {attempt_num}/{config.max_attempts} failed for {func_name}: {str(e)}")
                
                # Don't wait after the last attempt
                if attempt_num < config.max_attempts:
                    delay = self._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before retry {attempt_num + 1}")
                    time.sleep(delay)
                
                stats['total_retries'] += 1
                stats['last_failure'] = datetime.now().isoformat()
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func_name}: {str(e)}")
                stats['success_rate'] = (stats['success_rate'] * (stats['total_calls'] - 1) + 0.0) / stats['total_calls']
                raise
        
        # All attempts failed
        execution_time = time.time() - start_time
        stats['avg_attempts'] = (stats['avg_attempts'] * (stats['total_calls'] - 1) + config.max_attempts) / stats['total_calls']
        stats['success_rate'] = (stats['success_rate'] * (stats['total_calls'] - 1) + 0.0) / stats['total_calls']
        
        logger.error(f"All {config.max_attempts} attempts failed for {func_name} after {execution_time:.2f}s")
        raise last_exception
    
    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate delay before next retry attempt"""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_factor ** attempt)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib_value = self._fibonacci(attempt + 1)
            delay = config.base_delay * fib_value
        else:
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number with caching"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]
        
        if n <= 1:
            return n
        
        result = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        self.fibonacci_cache[n] = result
        return result
    
    def get_retry_stats(self) -> dict:
        """Get retry statistics for all functions"""
        return self.retry_stats.copy()
    
    def reset_stats(self, func_name: str = None):
        """Reset retry statistics"""
        if func_name:
            if func_name in self.retry_stats:
                del self.retry_stats[func_name]
        else:
            self.retry_stats.clear()
    
    def get_health_metrics(self) -> dict:
        """Get health metrics for monitoring"""
        if not self.retry_stats:
            return {
                'status': 'healthy',
                'total_functions': 0,
                'overall_success_rate': 1.0,
                'total_retries': 0
            }
        
        total_calls = sum(stats['total_calls'] for stats in self.retry_stats.values())
        total_retries = sum(stats['total_retries'] for stats in self.retry_stats.values())
        
        # Calculate weighted average success rate
        weighted_success_rate = sum(
            stats['success_rate'] * stats['total_calls'] 
            for stats in self.retry_stats.values()
        ) / max(1, total_calls)
        
        # Identify problematic functions
        problematic_functions = [
            func_name for func_name, stats in self.retry_stats.items()
            if stats['success_rate'] < 0.8
        ]
        
        return {
            'status': 'healthy' if weighted_success_rate > 0.95 else 'degraded' if weighted_success_rate > 0.8 else 'unhealthy',
            'total_functions': len(self.retry_stats),
            'overall_success_rate': weighted_success_rate,
            'total_calls': total_calls,
            'total_retries': total_retries,
            'retry_rate': total_retries / max(1, total_calls),
            'problematic_functions': problematic_functions,
            'last_updated': datetime.now().isoformat()
        }

# Global retry mechanism instance
retry_mechanism = RetryMechanism()

# Convenience decorators for common retry patterns
def retry_on_exception(max_attempts: int = 3, 
                      base_delay: float = 1.0,
                      exceptions: tuple = (Exception,),
                      strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
    """Convenience decorator for retry on exception"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        exceptions=exceptions,
        strategy=strategy
    )
    return retry_mechanism.with_retry(config)

def retry_database_operations(max_attempts: int = 3):
    """Specific retry decorator for database operations"""
    import psycopg2
    import redis
    
    database_exceptions = (
        psycopg2.Error,
        redis.RedisError,
        ConnectionError,
        TimeoutError
    )
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=0.5,
        max_delay=10.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exceptions=database_exceptions,
        jitter=True
    )
    return retry_mechanism.with_retry(config)

def retry_api_calls(max_attempts: int = 5):
    """Specific retry decorator for API calls"""
    import httpx
    
    api_exceptions = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.ReadError,
        ConnectionError
    )
    
    def should_retry_on_status(result):
        """Check if HTTP response should trigger retry"""
        if hasattr(result, 'status_code'):
            # Retry on 5xx errors and 429 (rate limit)
            return result.status_code >= 500 or result.status_code == 429
        return False
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exceptions=api_exceptions,
        retry_on_result=should_retry_on_status,
        jitter=True
    )
    return retry_mechanism.with_retry(config)

def retry_ml_operations(max_attempts: int = 2):
    """Specific retry decorator for ML operations"""
    ml_exceptions = (
        RuntimeError,  # CUDA/model errors
        MemoryError,   # Out of memory
        OSError,       # Model loading errors
        ValueError     # Invalid input shapes
    )
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=2.0,
        max_delay=20.0,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        exceptions=ml_exceptions,
        jitter=False  # ML operations need deterministic timing
    )
    return retry_mechanism.with_retry(config)

# Example usage patterns
if __name__ == "__main__":
    # Example 1: Basic retry
    @retry_on_exception(max_attempts=3)
    async def flaky_api_call():
        import random
        if random.random() < 0.7:
            raise ConnectionError("API temporarily unavailable")
        return "Success!"
    
    # Example 2: Database retry
    @retry_database_operations(max_attempts=5)
    async def database_query():
        # Simulate database operation
        pass
    
    # Example 3: Custom retry configuration
    custom_config = RetryConfig(
        max_attempts=10,
        base_delay=0.1,
        max_delay=5.0,
        strategy=RetryStrategy.FIBONACCI_BACKOFF,
        exceptions=(ConnectionError, TimeoutError),
        jitter=True
    )
    
    @retry_mechanism.with_retry(custom_config)
    async def custom_operation():
        # Your operation here
        pass
