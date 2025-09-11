"""
Circuit Breaker System

Implements circuit breaker pattern for external API calls and AI services
to prevent cascading failures and provide graceful degradation.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure mode - requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3   # Successes needed to close from half-open
    timeout: int = 30           # Request timeout in seconds
    expected_exception: type = Exception

@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opens: int = 0
    current_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: list = field(default_factory=list)

class CircuitBreaker:
    """Circuit breaker implementation for a specific service"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_state_change = datetime.utcnow()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        async with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit should allow request
            if not await self._should_allow_request():
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
                
                await self._on_success()
                return result
                
            except asyncio.TimeoutError:
                await self._on_failure(f"Timeout after {self.config.timeout}s")
                raise CircuitBreakerTimeoutError(f"Circuit breaker {self.name} timeout")
                
            except self.config.expected_exception as e:
                await self._on_failure(str(e))
                raise
                
            except Exception as e:
                await self._on_failure(str(e))
                raise CircuitBreakerError(f"Circuit breaker {self.name} error: {str(e)}")
    
    async def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state"""
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if enough time has passed to try half-open
            if (datetime.utcnow() - self.last_state_change).total_seconds() >= self.config.recovery_timeout:
                await self._transition_to_half_open()
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            return True
            
        return False
    
    async def _on_success(self):
        """Handle successful request"""
        self.stats.successful_requests += 1
        self.stats.last_success_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            # Count successes in half-open state
            if self.stats.successful_requests >= self.config.success_threshold:
                await self._transition_to_closed()
        
        # Reset failure count on success
        self.stats.current_failures = 0
    
    async def _on_failure(self, error_message: str):
        """Handle failed request"""
        self.stats.failed_requests += 1
        self.stats.current_failures += 1
        self.stats.last_failure_time = datetime.utcnow()
        
        logger.warning(f"Circuit breaker {self.name} failure: {error_message}")
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.stats.current_failures >= self.config.failure_threshold):
            await self._transition_to_open()
            
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker {self.name} opening due to failures")
            self.state = CircuitState.OPEN
            self.stats.circuit_opens += 1
            self.last_state_change = datetime.utcnow()
            self.stats.state_changes.append({
                'from': self.state.value,
                'to': CircuitState.OPEN.value,
                'timestamp': self.last_state_change.isoformat(),
                'reason': f"{self.stats.current_failures} failures"
            })
    
    async def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        logger.info(f"Circuit breaker {self.name} transitioning to half-open")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.utcnow()
        self.stats.state_changes.append({
            'from': CircuitState.OPEN.value,
            'to': CircuitState.HALF_OPEN.value,
            'timestamp': self.last_state_change.isoformat(),
            'reason': f"Recovery timeout of {self.config.recovery_timeout}s elapsed"
        })
        
        # Reset counters for half-open testing
        self.stats.successful_requests = 0
        self.stats.failed_requests = 0
    
    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        logger.info(f"Circuit breaker {self.name} closing - service recovered")
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.utcnow()
        self.stats.state_changes.append({
            'from': CircuitState.HALF_OPEN.value,
            'to': CircuitState.CLOSED.value,
            'timestamp': self.last_state_change.isoformat(),
            'reason': f"{self.config.success_threshold} successful requests"
        })
        
        # Reset failure tracking
        self.stats.current_failures = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        success_rate = (
            self.stats.successful_requests / max(1, self.stats.total_requests)
        )
        
        return {
            'name': self.name,
            'state': self.state.value,
            'last_state_change': self.last_state_change.isoformat(),
            'stats': {
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'current_failures': self.stats.current_failures,
                'circuit_opens': self.stats.circuit_opens,
                'success_rate': success_rate,
                'last_success': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
                'last_failure': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            },
            'state_changes': self.stats.state_changes[-10:]  # Last 10 changes
        }

class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_configs: Dict[str, CircuitBreakerConfig] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize circuit breaker manager with default configurations"""
        if self.initialized:
            return
            
        logger.info("Initializing circuit breaker manager...")
        
        # Define default configurations for different service types
        self.default_configs = {
            'openai_api': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2,
                timeout=30
            ),
            'anthropic_api': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2,
                timeout=30
            ),
            'huggingface_api': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30,
                success_threshold=3,
                timeout=20
            ),
            'database': CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=10,
                success_threshold=5,
                timeout=5
            ),
            'redis': CircuitBreakerConfig(
                failure_threshold=8,
                recovery_timeout=15,
                success_threshold=3,
                timeout=3
            ),
            'external_api': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                success_threshold=3,
                timeout=15
            ),
            'ai_model': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                timeout=10
            )
        }
        
        # Create circuit breakers for critical services
        critical_services = [
            'openai_api', 'anthropic_api', 'database', 'redis'
        ]
        
        for service in critical_services:
            await self.get_circuit_breaker(service)
        
        self.initialized = True
        logger.info(f"Circuit breaker manager initialized with {len(self.circuit_breakers)} circuit breakers")
    
    async def get_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            # Use provided config or default for service type
            if config is None:
                # Try to find matching default config
                for service_type, default_config in self.default_configs.items():
                    if service_type in service_name.lower():
                        config = default_config
                        break
                else:
                    # Use generic external API config
                    config = self.default_configs.get('external_api', CircuitBreakerConfig())
            
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, config)
            logger.info(f"Created circuit breaker for service: {service_name}")
        
        return self.circuit_breakers[service_name]
    
    async def call_with_circuit_breaker(self, service_name: str, func: Callable[..., Awaitable[Any]], 
                                       *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = await self.get_circuit_breaker(service_name)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    # Convenience methods for specific services
    
    async def call_openai_api(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Call OpenAI API with circuit breaker"""
        return await self.call_with_circuit_breaker('openai_api', func, *args, **kwargs)
    
    async def call_anthropic_api(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Call Anthropic API with circuit breaker"""
        return await self.call_with_circuit_breaker('anthropic_api', func, *args, **kwargs)
    
    async def call_database(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Call database with circuit breaker"""
        return await self.call_with_circuit_breaker('database', func, *args, **kwargs)
    
    async def call_redis(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Call Redis with circuit breaker"""
        return await self.call_with_circuit_breaker('redis', func, *args, **kwargs)
    
    async def call_ai_model(self, model_name: str, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Call AI model with circuit breaker"""
        service_name = f"ai_model_{model_name}"
        return await self.call_with_circuit_breaker(service_name, func, *args, **kwargs)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {
            name: circuit_breaker.get_status()
            for name, circuit_breaker in self.circuit_breakers.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
        half_open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN)
        
        return {
            'total_circuit_breakers': total_breakers,
            'healthy_breakers': total_breakers - open_breakers - half_open_breakers,
            'open_breakers': open_breakers,
            'half_open_breakers': half_open_breakers,
            'overall_health': 'healthy' if open_breakers == 0 else 'degraded' if open_breakers < total_breakers else 'critical',
            'circuit_breaker_details': {
                name: {
                    'state': cb.state.value,
                    'success_rate': cb.stats.successful_requests / max(1, cb.stats.total_requests),
                    'last_failure': cb.stats.last_failure_time.isoformat() if cb.stats.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    async def reset_circuit_breaker(self, service_name: str):
        """Manually reset a circuit breaker to closed state"""
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            await circuit_breaker._transition_to_closed()
            logger.info(f"Manually reset circuit breaker: {service_name}")
    
    async def cleanup(self):
        """Clean up circuit breaker manager"""
        logger.info("Cleaning up circuit breaker manager...")
        self.circuit_breakers.clear()
        self.initialized = False

# Custom exceptions
class CircuitBreakerError(Exception):
    """Base circuit breaker exception"""
    pass

class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open - requests blocked"""
    pass

class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Circuit breaker timeout"""
    pass

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()