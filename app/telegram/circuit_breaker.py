"""
Circuit Breaker Pattern Implementation

Provides resilient API calls with automatic failure detection,
recovery mechanisms, and adaptive behavior for Telegram Bot API.
"""

import asyncio
import time
import random
from typing import Callable, Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from functools import wraps
import math

import structlog
from aiogram.exceptions import (
    TelegramRetryAfter, 
    TelegramBadRequest, 
    TelegramServerError,
    TelegramNetworkError,
    TelegramForbiddenError
)

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureType(Enum):
    """Types of failures for different handling."""
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    name: str
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds to wait before trying again
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: int = 30  # Request timeout in seconds
    
    # Advanced settings
    exponential_backoff: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    jitter_enabled: bool = True
    adaptive_timeout: bool = True
    
    # Failure type specific settings
    rate_limit_multiplier: float = 2.0  # Extra wait for rate limits
    server_error_threshold: int = 3     # Lower threshold for server errors


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    
    # Timing stats
    average_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    
    # Failure breakdown
    failure_types: Dict[FailureType, int] = field(default_factory=dict)
    
    # State transitions
    state_transitions: List[Tuple[float, CircuitState]] = field(default_factory=list)
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    # Recovery metrics
    recovery_attempts: int = 0
    successful_recoveries: int = 0


class CircuitBreaker:
    """
    Individual circuit breaker implementation.
    
    Implements the circuit breaker pattern with:
    - Automatic failure detection
    - Exponential backoff
    - Adaptive timeouts
    - Jitter for avoiding thundering herd
    - Different handling for different error types
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
        # Adaptive behavior
        self.current_timeout = config.timeout
        self.backoff_multiplier = 1.0
        
        # Response time tracking for adaptive timeout
        self._response_times: List[float] = []
        self._max_response_history = 100
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exception: When function fails
        """
        async with self._lock:
            # Check if circuit should be opened
            if await self._should_reject_call():
                self.stats.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.config.name}' is open"
                )
            
            # Transition to half-open if testing recovery
            if self.state == CircuitState.OPEN:
                if await self._should_attempt_recovery():
                    await self._transition_to_half_open()
                else:
                    self.stats.rejected_requests += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.config.name}' is open"
                    )
        
        # Execute the function call
        return await self._execute_with_retry(func, *args, **kwargs)
    
    async def _should_reject_call(self) -> bool:
        """Check if call should be rejected."""
        if self.state == CircuitState.CLOSED:
            return False
        
        if self.state == CircuitState.OPEN:
            return not await self._should_attempt_recovery()
        
        # HALF_OPEN state allows limited calls
        return False
    
    async def _should_attempt_recovery(self) -> bool:
        """Check if recovery should be attempted."""
        if self.state != CircuitState.OPEN:
            return False
        
        now = time.time()
        recovery_timeout = self.config.recovery_timeout * self.backoff_multiplier
        
        return now - self.last_failure_time >= recovery_timeout
    
    async def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.stats.recovery_attempts += 1
        
        await self._record_state_transition()
        logger.info(f"Circuit breaker '{self.config.name}' transitioned to HALF_OPEN")
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                start_time = time.time()
                
                # Apply adaptive timeout
                timeout = self.current_timeout
                if attempt > 0:
                    timeout *= (2 ** attempt)  # Exponential increase
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
                
                # Record success
                response_time = time.time() - start_time
                await self._record_success(response_time)
                
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                failure_type = FailureType.TIMEOUT
                
                logger.warning(
                    f"Timeout in circuit breaker '{self.config.name}'",
                    attempt=attempt + 1,
                    timeout=timeout
                )
                
                await self._record_failure(failure_type)
                
                if attempt < self.config.max_retry_attempts - 1:
                    delay = await self._calculate_retry_delay(attempt, failure_type)
                    await asyncio.sleep(delay)
                
            except Exception as e:
                last_exception = e
                failure_type = await self._classify_failure(e)
                
                logger.warning(
                    f"Failure in circuit breaker '{self.config.name}'",
                    error=str(e),
                    error_type=type(e).__name__,
                    failure_type=failure_type.value,
                    attempt=attempt + 1
                )
                
                await self._record_failure(failure_type)
                
                # Don't retry certain error types
                if failure_type in [FailureType.BAD_REQUEST, FailureType.FORBIDDEN]:
                    break
                
                # Special handling for rate limits
                if failure_type == FailureType.RATE_LIMIT:
                    if hasattr(e, 'retry_after'):
                        delay = e.retry_after * self.config.rate_limit_multiplier
                        logger.info(f"Rate limited, waiting {delay} seconds")
                        await asyncio.sleep(delay)
                        continue
                
                if attempt < self.config.max_retry_attempts - 1:
                    delay = await self._calculate_retry_delay(attempt, failure_type)
                    await asyncio.sleep(delay)
        
        # All retries failed
        raise last_exception
    
    async def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type for appropriate handling."""
        if isinstance(exception, TelegramRetryAfter):
            return FailureType.RATE_LIMIT
        elif isinstance(exception, TelegramNetworkError):
            return FailureType.NETWORK_ERROR
        elif isinstance(exception, TelegramServerError):
            return FailureType.SERVER_ERROR
        elif isinstance(exception, TelegramBadRequest):
            return FailureType.BAD_REQUEST
        elif isinstance(exception, TelegramForbiddenError):
            return FailureType.FORBIDDEN
        elif isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        else:
            return FailureType.UNKNOWN
    
    async def _calculate_retry_delay(self, attempt: int, failure_type: FailureType) -> float:
        """Calculate delay before retry."""
        base_delay = self.config.retry_delay
        
        if self.config.exponential_backoff:
            delay = base_delay * (2 ** attempt)
        else:
            delay = base_delay
        
        # Adjust delay based on failure type
        if failure_type == FailureType.RATE_LIMIT:
            delay *= self.config.rate_limit_multiplier
        elif failure_type == FailureType.SERVER_ERROR:
            delay *= 1.5  # Longer delay for server errors
        
        # Add jitter if enabled
        if self.config.jitter_enabled:
            jitter = delay * 0.1 * random.random()
            delay += jitter
        
        return min(delay, 60.0)  # Cap at 60 seconds
    
    async def _record_success(self, response_time: float) -> None:
        """Record successful execution."""
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.last_success_time = time.time()
        
        # Update response time stats
        self._response_times.append(response_time)
        if len(self._response_times) > self._max_response_history:
            self._response_times.pop(0)
        
        self.stats.average_response_time = sum(self._response_times) / len(self._response_times)
        self.stats.max_response_time = max(self.stats.max_response_time, response_time)
        self.stats.min_response_time = min(self.stats.min_response_time, response_time)
        
        # Update adaptive timeout
        if self.config.adaptive_timeout:
            await self._update_adaptive_timeout()
        
        # Handle state transitions
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
            self.backoff_multiplier = 1.0
    
    async def _record_failure(self, failure_type: FailureType) -> None:
        """Record failed execution."""
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.stats.last_failure_time = time.time()
        self.last_failure_time = self.stats.last_failure_time
        
        # Update failure type stats
        if failure_type not in self.stats.failure_types:
            self.stats.failure_types[failure_type] = 0
        self.stats.failure_types[failure_type] += 1
        
        # Handle state transitions
        self.failure_count += 1
        
        # Check if circuit should open
        threshold = self.config.failure_threshold
        if failure_type == FailureType.SERVER_ERROR:
            threshold = self.config.server_error_threshold
        
        if self.failure_count >= threshold:
            if self.state == CircuitState.CLOSED:
                await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()
                # Increase backoff for repeated failures
                self.backoff_multiplier = min(16.0, self.backoff_multiplier * 2)
    
    async def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.backoff_multiplier = 1.0
        self.stats.successful_recoveries += 1
        
        await self._record_state_transition()
        logger.info(f"Circuit breaker '{self.config.name}' transitioned to CLOSED")
    
    async def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        
        await self._record_state_transition()
        logger.warning(f"Circuit breaker '{self.config.name}' transitioned to OPEN")
    
    async def _record_state_transition(self) -> None:
        """Record state transition for metrics."""
        self.stats.state_transitions.append((time.time(), self.state))
        
        # Keep only recent transitions
        if len(self.stats.state_transitions) > 100:
            self.stats.state_transitions = self.stats.state_transitions[-50:]
    
    async def _update_adaptive_timeout(self) -> None:
        """Update timeout based on response time patterns."""
        if len(self._response_times) < 10:
            return
        
        # Calculate 95th percentile response time
        sorted_times = sorted(self._response_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[p95_index]
        
        # Set timeout to 3x the 95th percentile, with bounds
        new_timeout = max(5.0, min(120.0, p95_time * 3))
        
        # Smooth the change
        self.current_timeout = (self.current_timeout * 0.7) + (new_timeout * 0.3)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests
        
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "backoff_multiplier": self.backoff_multiplier,
            "current_timeout": self.current_timeout,
            "success_rate": success_rate,
            "stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "average_response_time": self.stats.average_response_time,
                "max_response_time": self.stats.max_response_time,
                "min_response_time": self.stats.min_response_time if self.stats.min_response_time != float('inf') else 0,
                "failure_types": {ft.value: count for ft, count in self.stats.failure_types.items()},
                "recovery_attempts": self.stats.recovery_attempts,
                "successful_recoveries": self.stats.successful_recoveries,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time,
            }
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services/operations.
    
    Provides centralized configuration and monitoring of circuit breakers.
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig(
            name="default",
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=3,
            timeout=30,
            exponential_backoff=True,
            adaptive_timeout=True,
            jitter_enabled=True
        )
    
    async def initialize(self) -> None:
        """Initialize circuit breaker manager."""
        try:
            # Create default circuit breakers for common operations
            await self._create_default_breakers()
            
            logger.info("Circuit breaker manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize circuit breaker manager", error=str(e))
            raise
    
    async def _create_default_breakers(self) -> None:
        """Create default circuit breakers for Telegram operations."""
        # Main Telegram API breaker
        telegram_config = CircuitBreakerConfig(
            name="telegram_api",
            failure_threshold=5,
            recovery_timeout=30,
            success_threshold=2,
            timeout=20,
            exponential_backoff=True,
            adaptive_timeout=True,
            rate_limit_multiplier=1.5
        )
        self.breakers["telegram_api"] = CircuitBreaker(telegram_config)
        
        # Message sending specific breaker
        message_config = CircuitBreakerConfig(
            name="send_message",
            failure_threshold=3,
            recovery_timeout=15,
            success_threshold=2,
            timeout=10,
            server_error_threshold=2
        )
        self.breakers["send_message"] = CircuitBreaker(message_config)
        
        # File operations breaker
        file_config = CircuitBreakerConfig(
            name="file_operations",
            failure_threshold=3,
            recovery_timeout=45,
            success_threshold=3,
            timeout=60,  # File operations can take longer
            max_retry_attempts=2
        )
        self.breakers["file_operations"] = CircuitBreaker(file_config)
        
        # Webhook operations breaker
        webhook_config = CircuitBreakerConfig(
            name="webhook",
            failure_threshold=8,
            recovery_timeout=120,
            success_threshold=5,
            timeout=30
        )
        self.breakers["webhook"] = CircuitBreaker(webhook_config)
    
    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker by name."""
        if name not in self.breakers:
            config = CircuitBreakerConfig(name=name, **self.default_config.__dict__)
            config.name = name
            self.breakers[name] = CircuitBreaker(config)
            logger.info(f"Created new circuit breaker: {name}")
        
        return self.breakers[name]
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create circuit breaker with custom configuration."""
        config.name = name
        self.breakers[name] = CircuitBreaker(config)
        logger.info(f"Created custom circuit breaker: {name}")
        return self.breakers[name]
    
    async def execute(self, breaker_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function through named circuit breaker."""
        breaker = self.get_breaker(breaker_name)
        return await breaker.call(func, *args, **kwargs)
    
    def circuit_breaker(self, name: str):
        """Decorator for circuit breaker protection."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.execute(name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers."""
        stats = {}
        
        for name, breaker in self.breakers.items():
            stats[name] = await breaker.get_stats()
        
        # Global statistics
        total_requests = sum(breaker.stats.total_requests for breaker in self.breakers.values())
        total_failures = sum(breaker.stats.failed_requests for breaker in self.breakers.values())
        total_successes = sum(breaker.stats.successful_requests for breaker in self.breakers.values())
        
        global_success_rate = 0.0
        if total_requests > 0:
            global_success_rate = total_successes / total_requests
        
        stats["_global"] = {
            "total_breakers": len(self.breakers),
            "total_requests": total_requests,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "global_success_rate": global_success_rate,
            "open_breakers": sum(1 for b in self.breakers.values() if b.state == CircuitState.OPEN),
            "half_open_breakers": sum(1 for b in self.breakers.values() if b.state == CircuitState.HALF_OPEN),
        }
        
        return stats
    
    async def reset_breaker(self, name: str) -> bool:
        """Reset circuit breaker to closed state."""
        if name not in self.breakers:
            return False
        
        breaker = self.breakers[name]
        breaker.state = CircuitState.CLOSED
        breaker.failure_count = 0
        breaker.success_count = 0
        breaker.backoff_multiplier = 1.0
        
        logger.info(f"Reset circuit breaker: {name}")
        return True
    
    async def reset_all_breakers(self) -> int:
        """Reset all circuit breakers."""
        count = 0
        for name in self.breakers:
            if await self.reset_breaker(name):
                count += 1
        
        logger.info(f"Reset {count} circuit breakers")
        return count
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all circuit breakers."""
        health_status = {
            "healthy": True,
            "breakers": {}
        }
        
        open_breakers = 0
        for name, breaker in self.breakers.items():
            breaker_healthy = breaker.state != CircuitState.OPEN
            if not breaker_healthy:
                open_breakers += 1
            
            health_status["breakers"][name] = {
                "healthy": breaker_healthy,
                "state": breaker.state.value,
                "failure_rate": (
                    breaker.stats.failed_requests / max(1, breaker.stats.total_requests)
                ),
                "last_failure": breaker.stats.last_failure_time,
            }
        
        # Overall health is unhealthy if more than 50% of breakers are open
        if len(self.breakers) > 0:
            health_status["healthy"] = (open_breakers / len(self.breakers)) < 0.5
        
        health_status["summary"] = {
            "total_breakers": len(self.breakers),
            "open_breakers": open_breakers,
            "healthy_percentage": (
                ((len(self.breakers) - open_breakers) / max(1, len(self.breakers))) * 100
            ),
        }
        
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up circuit breaker resources."""
        try:
            self.breakers.clear()
            logger.info("Circuit breaker manager cleanup completed")
            
        except Exception as e:
            logger.error("Error during circuit breaker cleanup", error=str(e))