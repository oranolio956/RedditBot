"""
Main FastAPI Application

Entry point for the Telegram ML Bot application with FastAPI.
Provides REST API endpoints, webhook handling, and health monitoring.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.config import settings
from app.database.connection import db_manager
from app.database.manager import db_service
from app.core.redis import redis_manager
from app.core.auth import get_auth_manager
from app.api.v1 import router as api_v1_router
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware
from app.middleware.error_handling import ErrorHandlingMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware, RateLimitSecurityMiddleware
from app.middleware.input_validation import InputValidationMiddleware
from app.telegram.bot import get_bot, cleanup_bot
from app.services.ml_initialization import initialize_ml_models, cleanup_ml_models
from app.services.background_tasks import start_background_tasks
from app.websocket.manager import websocket_manager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.monitoring.log_format == "json"
        else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application,
    including database connections, Redis, and other resources.
    """
    # Startup
    logger.info("Starting Telegram ML Bot application")
    
    try:
        # Initialize database service (includes connection manager)
        await db_service.initialize()
        logger.info("Database service initialized")
        
        # Initialize Redis connections
        await redis_manager.initialize()
        logger.info("Redis connection initialized")
        
        # Initialize authentication manager
        auth_manager = await get_auth_manager()
        logger.info("Authentication manager initialized")
        
        # Initialize Telegram bot
        telegram_bot = await get_bot()
        logger.info("Telegram bot initialized")
        
        # Initialize ML models and AI engines
        await initialize_ml_models()
        logger.info("ML models and AI engines initialized")
        
        # Start background tasks
        await start_background_tasks()
        logger.info("Background tasks started")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Telegram ML Bot application")
    
    try:
        # Shutdown database service (includes health monitoring and backups)
        await db_service.shutdown()
        logger.info("Database service shutdown completed")
        
        # Cleanup Telegram bot
        await cleanup_bot()
        logger.info("Telegram bot cleanup completed")
        
        # Close Redis connections
        await redis_manager.close()
        logger.info("Redis connections closed")
        
        # Cleanup ML resources
        await cleanup_ml_models()
        logger.info("ML models cleaned up")
        
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error("Error during application shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance Telegram bot with ML capabilities and real-time analytics",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# Add middleware (order matters - first added is outermost)

# Error handling middleware (outermost)
app.add_middleware(ErrorHandlingMiddleware)

# Input validation middleware
if settings.security.enable_input_sanitization:
    app.add_middleware(
        InputValidationMiddleware,
        max_request_size=settings.security.max_request_size,
        max_json_size=settings.security.max_json_size,
        max_string_length=settings.security.max_message_length,
        enable_sql_injection_detection=True,
        enable_xss_detection=True,
        enable_command_injection_detection=True,
        enable_path_traversal_detection=True
    )

# Security headers middleware
if settings.security.enable_security_headers:
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts_max_age=settings.security.hsts_max_age,
        enable_hsts=settings.is_production
    )

# Security-focused rate limiting middleware
app.add_middleware(
    RateLimitSecurityMiddleware,
    suspicious_threshold=settings.security.rate_limit_per_ip_per_minute * 2,
    block_duration=settings.security.block_duration,
    enable_progressive_delays=True
)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting middleware
if settings.security.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure with your actual hosts
    )

# Include API routers
app.include_router(
    api_v1_router,
    prefix=settings.api_prefix,
    tags=["API v1"]
)

# Telegram webhook endpoint with signature verification
if settings.telegram.webhook_url:
    from app.core.auth import verify_telegram_webhook
    
    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request):
        """Handle Telegram webhook updates with signature verification."""
        try:
            # Verify webhook signature for security
            if settings.security.webhook_signature_required:
                is_valid = await verify_telegram_webhook(
                    request, 
                    require_signature=True
                )
                if not is_valid:
                    logger.warning(
                        "Invalid webhook signature",
                        ip=request.client.host if request.client else "unknown",
                        user_agent=request.headers.get("user-agent", "unknown")
                    )
                    raise HTTPException(
                        status_code=401, 
                        detail="Invalid webhook signature"
                    )
            
            # Rate limiting for webhook endpoint
            client_ip = request.client.host if request.client else "unknown"
            from app.core.auth import get_auth_manager
            auth_manager = await get_auth_manager()
            
            # Allow 60 webhook calls per minute per IP
            is_rate_limited = await auth_manager.rate_limit_check(
                f"webhook:{client_ip}", 
                limit=60, 
                window=60
            )
            
            if is_rate_limited:
                logger.warning(
                    "Webhook rate limit exceeded",
                    ip=client_ip
                )
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limit exceeded"
                )
            
            # Get bot instance
            from app.telegram.bot import telegram_bot
            if not telegram_bot:
                raise HTTPException(status_code=503, detail="Bot not initialized")
            
            # Process update through bot's webhook manager
            update_data = await request.json()
            
            # Log webhook received for monitoring
            logger.info(
                "Webhook received",
                update_id=update_data.get("update_id"),
                chat_id=update_data.get("message", {}).get("chat", {}).get("id"),
                ip=client_ip
            )
            
            # Process the update (placeholder - implement actual bot logic)
            # await telegram_bot.process_update(update_data)
            
            return {"status": "received", "timestamp": structlog.processors.TimeStamper(fmt="iso")._make_stamper()()}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Error processing webhook", 
                error=str(e),
                ip=request.client.host if request.client else "unknown"
            )
            raise HTTPException(status_code=500, detail="Webhook processing error")


# WebSocket endpoints for Kelly AI monitoring

@app.websocket("/ws/kelly/monitoring")
async def kelly_monitoring_websocket(websocket: WebSocket, user_id: Optional[int] = None):
    """
    WebSocket endpoint for Kelly AI real-time monitoring dashboard.
    
    Provides real-time updates for metrics, activities, alerts, and interventions.
    """
    try:
        # Connect to WebSocket manager
        connection_id = await websocket_manager.connect(websocket, user_id)
        
        # Start monitoring session
        session_config = {
            "user_id": user_id,
            "session_type": "monitoring",
            "alert_levels": ["medium", "high", "critical"],
            "metrics_interval": 15
        }
        
        session_id = await websocket_manager.start_real_time_monitoring_session(
            connection_id, session_config
        )
        
        logger.info(f"Kelly monitoring WebSocket connected: {connection_id} (user: {user_id})")
        
        try:
            # Keep connection alive and handle messages
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    message_type = message.get("type")
                    
                    if message_type == "subscribe":
                        # Subscribe to additional monitoring rooms
                        rooms = message.get("rooms", [])
                        for room in rooms:
                            await websocket_manager.join_monitoring_room(connection_id, room)
                    
                    elif message_type == "unsubscribe":
                        # Unsubscribe from monitoring rooms
                        rooms = message.get("rooms", [])
                        for room in rooms:
                            await websocket_manager.leave_monitoring_room(connection_id, room)
                    
                    elif message_type == "ping":
                        # Respond to ping
                        await websocket_manager.send_to_connection(
                            connection_id, 
                            "pong", 
                            {"timestamp": datetime.utcnow().isoformat()}
                        )
                    
                    elif message_type == "get_metrics":
                        # Send current metrics
                        from app.services.kelly_monitoring_service import kelly_monitoring_service
                        metrics = await kelly_monitoring_service.get_live_metrics()
                        await websocket_manager.send_to_connection(
                            connection_id,
                            "metrics_snapshot",
                            metrics
                        )
                    
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket_manager.send_to_connection(
                        connection_id,
                        "error", 
                        {"message": "Invalid JSON message"}
                    )
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket_manager.send_to_connection(
                        connection_id,
                        "error",
                        {"message": "Message processing error"}
                    )
        
        except WebSocketDisconnect:
            logger.info(f"Kelly monitoring WebSocket disconnected: {connection_id}")
        
        finally:
            # Clean up connection
            await websocket_manager.end_ai_session(session_id)
            await websocket_manager.disconnect(connection_id)
            
    except Exception as e:
        logger.error(f"Kelly monitoring WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

@app.websocket("/ws/kelly/conversation/{conversation_id}")
async def kelly_conversation_websocket(websocket: WebSocket, conversation_id: str, user_id: Optional[int] = None):
    """
    WebSocket endpoint for monitoring a specific Kelly AI conversation.
    
    Provides real-time updates for a specific conversation including
    interventions, AI confidence changes, and safety alerts.
    """
    try:
        # Connect to WebSocket manager
        connection_id = await websocket_manager.connect(websocket, user_id)
        
        # Subscribe to conversation-specific updates
        await websocket_manager.subscribe_to_topic(connection_id, f"conversation_{conversation_id}")
        await websocket_manager.subscribe_to_topic(connection_id, f"intervention_{conversation_id}")
        
        logger.info(f"Kelly conversation WebSocket connected: {connection_id} (conversation: {conversation_id})")
        
        try:
            # Send initial conversation status
            from app.core.redis import redis_manager
            conv_key = f"kelly:conversation_track:{conversation_id}"
            conversation_data = await redis_manager.get(conv_key)
            
            if conversation_data:
                await websocket_manager.send_to_connection(
                    connection_id,
                    "conversation_status",
                    json.loads(conversation_data)
                )
            
            # Keep connection alive and handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    message_type = message.get("type")
                    
                    if message_type == "get_status":
                        # Send current conversation status
                        conversation_data = await redis_manager.get(conv_key)
                        if conversation_data:
                            await websocket_manager.send_to_connection(
                                connection_id,
                                "conversation_status",
                                json.loads(conversation_data)
                            )
                    
                    elif message_type == "get_intervention_status":
                        # Send intervention status
                        intervention_key = f"kelly:intervention:{conversation_id}"
                        intervention_data = await redis_manager.get(intervention_key)
                        
                        status = "ai_active"
                        if intervention_data:
                            intervention = json.loads(intervention_data)
                            status = intervention.get("status", "unknown")
                        
                        await websocket_manager.send_to_connection(
                            connection_id,
                            "intervention_status",
                            {"status": status, "conversation_id": conversation_id}
                        )
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling conversation WebSocket message: {e}")
        
        except WebSocketDisconnect:
            logger.info(f"Kelly conversation WebSocket disconnected: {connection_id}")
        
        finally:
            await websocket_manager.disconnect(connection_id)
            
    except Exception as e:
        logger.error(f"Kelly conversation WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

@app.websocket("/ws/kelly/alerts")
async def kelly_alerts_websocket(websocket: WebSocket, user_id: Optional[int] = None):
    """
    WebSocket endpoint for real-time alert notifications.
    
    Provides immediate notifications for new alerts, alert status changes,
    and escalations in the Kelly AI system.
    """
    try:
        connection_id = await websocket_manager.connect(websocket, user_id)
        
        # Subscribe to alert updates
        await websocket_manager.subscribe_to_topic(connection_id, "alerts_monitoring")
        await websocket_manager.subscribe_to_topic(connection_id, "monitoring_dashboard")
        
        logger.info(f"Kelly alerts WebSocket connected: {connection_id} (user: {user_id})")
        
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    message_type = message.get("type")
                    
                    if message_type == "get_active_alerts":
                        # Send current active alerts
                        from app.core.redis import redis_manager
                        alert_keys = await redis_manager.keys("kelly:alert:active:*")
                        
                        alerts = []
                        for key in alert_keys[:20]:  # Limit to 20 most recent
                            alert_data = await redis_manager.get(key)
                            if alert_data:
                                alerts.append(json.loads(alert_data))
                        
                        await websocket_manager.send_to_connection(
                            connection_id,
                            "active_alerts",
                            {"alerts": alerts, "count": len(alerts)}
                        )
                    
                    elif message_type == "alert_filter":
                        # Update alert filtering preferences
                        filters = message.get("filters", {})
                        # Store user preferences (implementation depends on requirements)
                        await websocket_manager.send_to_connection(
                            connection_id,
                            "filter_updated",
                            {"filters": filters}
                        )
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling alerts WebSocket message: {e}")
        
        except WebSocketDisconnect:
            logger.info(f"Kelly alerts WebSocket disconnected: {connection_id}")
        
        finally:
            await websocket_manager.disconnect(connection_id)
            
    except Exception as e:
        logger.error(f"Kelly alerts WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

# Health check endpoints

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the health status of the application and its dependencies.
    """
    try:
        # Check database health using comprehensive health monitor
        db_health_status = await db_service.get_health_status()
        db_healthy = db_health_status['overall_status'] == 'healthy'
        
        # Check Redis health
        redis_healthy = await redis_manager.health_check()
        
        # Check Telegram bot health
        bot_healthy = True
        try:
            from app.telegram.bot import telegram_bot
            if telegram_bot:
                bot_status = await telegram_bot.get_status()
                bot_healthy = bot_status.get('health', {}).get('bot', True)
        except Exception:
            bot_healthy = False
        
        # Overall health status
        healthy = db_healthy and redis_healthy and bot_healthy
        
        status = {
            "status": "healthy" if healthy else "unhealthy",
            "timestamp": structlog.processors.TimeStamper(fmt="iso")._make_stamper()(),
            "version": settings.app_version,
            "environment": settings.environment,
            "checks": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy",
                "telegram_bot": "healthy" if bot_healthy else "unhealthy",
            }
        }
        
        return status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with system metrics.
    
    Provides comprehensive health information including performance metrics,
    database statistics, and repository health status.
    """
    try:
        import psutil
        
        # Get comprehensive database health status
        db_health_status = await db_service.get_health_status()
        
        # Basic health checks
        redis_healthy = await redis_manager.health_check()
        
        # System metrics
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        }
        
        # Database statistics
        db_stats = await db_service.get_statistics()
        
        # Redis info
        redis_info = await redis_manager.get_info()
        
        status = {
            "status": "healthy" if (db_health_status['overall_status'] == 'healthy' and redis_healthy) else "unhealthy",
            "timestamp": structlog.processors.TimeStamper(fmt="iso")._make_stamper()(),
            "version": settings.app_version,
            "environment": settings.environment,
            "system": system_info,
            "database": {
                "health_status": db_health_status,
                "statistics": {
                    "total_tables": db_stats.get('total_tables', 0),
                    "cache_hit_ratio": db_stats.get('cache_hit_ratio_percent', 0),
                    "total_live_tuples": db_stats.get('total_live_tuples', 0)
                }
            },
            "redis": {
                "healthy": redis_healthy,
                "info": redis_info,
            }
        }
        
        return status
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        return {"status": "error", "error": str(e)}


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for monitoring and alerting.
    """
    if not settings.monitoring.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    try:
        # Get Telegram bot metrics
        telegram_metrics = ""
        try:
            from app.telegram.bot import telegram_bot
            if telegram_bot and telegram_bot.metrics:
                telegram_metrics = telegram_bot.metrics.get_prometheus_metrics()
        except Exception:
            pass  # Continue without Telegram metrics
        
        # Get system metrics
        system_metrics = generate_latest()
        
        # Combine metrics
        combined_metrics = system_metrics.decode('utf-8') + "\n" + telegram_metrics
        
        return Response(
            content=combined_metrics,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Metrics unavailable")


# Enhanced health check endpoints

@app.get("/health/ai", tags=["Health"])
async def ai_health_check() -> Dict[str, Any]:
    """
    AI systems health check.
    
    Provides comprehensive health information for all AI components including
    ML models, engines, and processing capabilities.
    """
    try:
        from app.services.ml_initialization import model_manager
        
        # Get ML model health
        ml_health = await model_manager.health_check()
        
        # Check individual AI engines
        engine_health = {}
        engines = ['consciousness', 'emotional_intelligence', 'meta_reality', 
                  'transcendence', 'digital_telepathy', 'quantum_consciousness']
        
        for engine_name in engines:
            engine = model_manager.get_engine(engine_name)
            if engine and hasattr(engine, 'health_check'):
                try:
                    engine_health[engine_name] = await engine.health_check()
                except Exception as e:
                    engine_health[engine_name] = {'status': 'error', 'error': str(e)}
            else:
                engine_health[engine_name] = {'status': 'not_initialized'}
        
        # Overall AI health status
        healthy_engines = sum(1 for health in engine_health.values() 
                             if health.get('status') == 'healthy')
        total_engines = len(engines)
        
        overall_status = {
            "status": "healthy" if healthy_engines == total_engines else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "ml_models": ml_health,
            "ai_engines": engine_health,
            "engines_healthy": healthy_engines,
            "engines_total": total_engines,
            "health_ratio": healthy_engines / total_engines if total_engines > 0 else 0
        }
        
        return overall_status
        
    except Exception as e:
        logger.error(f"AI health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="AI health check failed")

@app.get("/health/database", tags=["Health"])
async def database_health_check() -> Dict[str, Any]:
    """
    Database health check with performance metrics.
    
    Provides detailed database health including connection pool status,
    query performance, and optimization recommendations.
    """
    try:
        from app.core.database_optimization import db_optimizer
        
        # Get database health from service
        db_health = await db_service.get_health_status()
        
        # Get performance stats
        performance_stats = db_optimizer.get_performance_stats()
        
        # Get pool optimization analysis
        pool_analysis = await db_optimizer.optimize_connection_pool()
        
        # Get query analysis
        query_analysis = await db_optimizer.analyze_query_patterns()
        
        return {
            "status": db_health.get('overall_status', 'unknown'),
            "timestamp": datetime.utcnow().isoformat(),
            "database_health": db_health,
            "performance_stats": performance_stats,
            "connection_pool": pool_analysis,
            "query_analysis": query_analysis
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database health check failed")

@app.get("/health/cache", tags=["Health"])
async def cache_health_check() -> Dict[str, Any]:
    """
    Cache systems health check.
    
    Provides health information for multi-tier caching including
    hit ratios, memory usage, and performance metrics.
    """
    try:
        from app.core.advanced_cache import cache_manager
        
        # Get cache statistics
        cache_stats = await cache_manager.get_cache_stats()
        
        # Get Redis health
        redis_healthy = await redis_manager.health_check()
        redis_info = await redis_manager.get_info()
        
        # Determine overall cache health
        cache_healthy = (
            cache_stats.get('hit_ratio', 0) > 0.5 and
            redis_healthy and
            cache_stats.get('l1_cache', {}).get('memory_utilization', 0) < 0.9
        )
        
        return {
            "status": "healthy" if cache_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "cache_stats": cache_stats,
            "redis_health": redis_healthy,
            "redis_info": redis_info,
            "recommendations": [
                "Consider increasing cache TTL" if cache_stats.get('hit_ratio', 0) < 0.7 else None,
                "Memory usage high" if cache_stats.get('l1_cache', {}).get('memory_utilization', 0) > 0.8 else None
            ]
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Cache health check failed")

@app.get("/health/websocket", tags=["Health"])
async def websocket_health_check() -> Dict[str, Any]:
    """
    WebSocket connections health check.
    
    Provides information about active WebSocket connections,
    AI sessions, and real-time communication health.
    """
    try:
        websocket_stats = websocket_manager.get_stats()
        
        # Check if WebSocket manager is healthy
        websocket_healthy = (
            websocket_stats.get('total_connections', 0) >= 0 and
            len(websocket_stats.get('connection_health', {})) == websocket_stats.get('total_connections', 0)
        )
        
        return {
            "status": "healthy" if websocket_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "websocket_stats": websocket_stats,
            "active_connections": websocket_stats.get('total_connections', 0),
            "active_ai_sessions": websocket_stats.get('active_ai_sessions', 0),
            "topics_active": websocket_stats.get('total_topics', 0)
        }
        
    except Exception as e:
        logger.error(f"WebSocket health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="WebSocket health check failed")

@app.get("/health/circuit-breakers", tags=["Health"])
async def circuit_breakers_health_check() -> Dict[str, Any]:
    """
    Circuit breakers health check.
    
    Provides status of all circuit breakers protecting external services
    and internal components from cascading failures.
    """
    try:
        from app.core.circuit_breaker import circuit_breaker_manager
        
        # Get circuit breaker health summary
        cb_health = circuit_breaker_manager.get_health_summary()
        
        # Get detailed status of all circuit breakers
        cb_status = circuit_breaker_manager.get_all_status()
        
        return {
            "status": cb_health.get('overall_health', 'unknown'),
            "timestamp": datetime.utcnow().isoformat(),
            "health_summary": cb_health,
            "circuit_breaker_details": cb_status,
            "recommendations": [
                f"Check {name} service" for name, details in cb_status.items()
                if details.get('state') == 'open'
            ]
        }
        
    except Exception as e:
        logger.error(f"Circuit breakers health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Circuit breakers health check failed")

@app.get("/health/background-tasks", tags=["Health"])
async def background_tasks_health_check() -> Dict[str, Any]:
    """
    Background tasks health check.
    
    Provides status of all background tasks including monitoring,
    cache management, and AI processing tasks.
    """
    try:
        from app.services.background_tasks import get_background_task_status
        
        task_status = get_background_task_status()
        
        # Determine if background tasks are healthy
        tasks_healthy = (
            task_status.get('running', False) and
            task_status.get('active_tasks', 0) > 0
        )
        
        return {
            "status": "healthy" if tasks_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "background_tasks": task_status
        }
        
    except Exception as e:
        logger.error(f"Background tasks health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Background tasks health check failed")

@app.get("/health/comprehensive", tags=["Health"])
async def comprehensive_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check of all system components.
    
    Aggregates health status from all subsystems and provides
    an overall system health assessment with recommendations.
    """
    try:
        # Gather health from all subsystems
        health_checks = {}
        
        try:
            health_checks['ai'] = await ai_health_check()
        except:
            health_checks['ai'] = {'status': 'error'}
            
        try:
            health_checks['database'] = await database_health_check()
        except:
            health_checks['database'] = {'status': 'error'}
            
        try:
            health_checks['cache'] = await cache_health_check()
        except:
            health_checks['cache'] = {'status': 'error'}
            
        try:
            health_checks['websocket'] = await websocket_health_check()
        except:
            health_checks['websocket'] = {'status': 'error'}
            
        try:
            health_checks['circuit_breakers'] = await circuit_breakers_health_check()
        except:
            health_checks['circuit_breakers'] = {'status': 'error'}
            
        try:
            health_checks['background_tasks'] = await background_tasks_health_check()
        except:
            health_checks['background_tasks'] = {'status': 'error'}
        
        # Calculate overall health
        healthy_systems = sum(
            1 for health in health_checks.values()
            if health.get('status') == 'healthy'
        )
        total_systems = len(health_checks)
        health_ratio = healthy_systems / total_systems
        
        # Determine overall status
        if health_ratio >= 0.9:
            overall_status = 'healthy'
        elif health_ratio >= 0.7:
            overall_status = 'degraded'
        else:
            overall_status = 'critical'
        
        # Generate recommendations
        recommendations = []
        for system_name, health in health_checks.items():
            if health.get('status') != 'healthy':
                recommendations.append(f"Check {system_name} system - status: {health.get('status')}")
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "health_ratio": health_ratio,
            "healthy_systems": healthy_systems,
            "total_systems": total_systems,
            "subsystem_health": health_checks,
            "recommendations": recommendations,
            "system_info": {
                "version": settings.app_version,
                "environment": settings.environment,
                "uptime": "calculated_uptime_would_go_here"
            }
        }
        
    except Exception as e:
        logger.error(f"Comprehensive health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Comprehensive health check failed")


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with basic application information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs" if settings.debug else "unavailable",
    }


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with error logging."""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error" if settings.is_production else str(exc),
            "status_code": 500,
            "path": request.url.path,
        }
    )


# WebSocket endpoints for real-time features

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time communication.
    
    Supports:
    - Consciousness mirroring real-time updates
    - Emotional intelligence live feedback  
    - Quantum consciousness state broadcasting
    - Digital telepathy network communication
    - Neural dream streaming
    - Meta-reality session updates
    """
    connection_id = None
    try:
        connection_id = await websocket_manager.connect(websocket)
        logger.info(f"WebSocket connection established: {connection_id}")
        
        while True:
            # Wait for messages from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                payload = message.get("data", {})
                
                # Handle different message types
                if message_type == "ping":
                    await websocket_manager.send_to_connection(
                        connection_id, "pong", {"timestamp": datetime.utcnow().isoformat()}
                    )
                    
                elif message_type == "subscribe":
                    topic = payload.get("topic")
                    if topic:
                        await websocket_manager.subscribe_to_topic(connection_id, topic)
                        
                elif message_type == "unsubscribe":
                    topic = payload.get("topic")
                    if topic:
                        await websocket_manager.unsubscribe_from_topic(connection_id, topic)
                        
                elif message_type == "start_consciousness_session":
                    session_config = payload.get("config", {})
                    session_id = await websocket_manager.start_consciousness_session(
                        connection_id, session_config
                    )
                    
                elif message_type == "start_emotional_session":
                    session_config = payload.get("config", {})
                    session_id = await websocket_manager.start_emotional_intelligence_session(
                        connection_id, session_config
                    )
                    
                elif message_type == "start_quantum_session":
                    session_config = payload.get("config", {})
                    session_id = await websocket_manager.start_quantum_consciousness_session(
                        connection_id, session_config
                    )
                    
                elif message_type == "start_telepathy_session":
                    session_config = payload.get("config", {})
                    session_id = await websocket_manager.start_digital_telepathy_session(
                        connection_id, session_config
                    )
                    
                else:
                    await websocket_manager.send_to_connection(
                        connection_id, "error", {"message": f"Unknown message type: {message_type}"}
                    )
                    
            except json.JSONDecodeError:
                await websocket_manager.send_to_connection(
                    connection_id, "error", {"message": "Invalid JSON format"}
                )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)

@app.websocket("/ws/auth")
async def authenticated_websocket_endpoint(websocket: WebSocket, token: str = None):
    """
    Authenticated WebSocket endpoint for user-specific real-time features.
    
    Requires authentication token for user-specific features like:
    - Personal consciousness sessions
    - Emotional intelligence profiles
    - Private telepathy networks
    """
    connection_id = None
    user_id = None
    
    try:
        # Validate authentication token
        if token:
            try:
                from app.core.auth import decode_token
                user_data = decode_token(token)
                user_id = user_data.get("user_id")
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {str(e)}")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        
        connection_id = await websocket_manager.connect(websocket, user_id)
        logger.info(f"Authenticated WebSocket connection established: {connection_id} (user: {user_id})")
        
        # Send user-specific welcome message
        await websocket_manager.send_to_connection(connection_id, "authenticated_welcome", {
            "user_id": user_id,
            "connection_id": connection_id,
            "features_available": [
                "personal_consciousness_sessions",
                "emotional_intelligence_profiles",
                "private_telepathy_networks",
                "quantum_consciousness_tracking",
                "meta_reality_experiences",
                "transcendence_protocols"
            ]
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                payload = message.get("data", {})
                
                # Handle authenticated-only features
                if message_type == "get_user_sessions":
                    # Get user's active AI sessions
                    user_sessions = {
                        session_id: session_data
                        for session_id, session_data in websocket_manager.ai_sessions.items()
                        if session_data.get("connection_id") == connection_id
                    }
                    
                    await websocket_manager.send_to_connection(
                        connection_id, "user_sessions", {"sessions": user_sessions}
                    )
                    
                elif message_type == "end_session":
                    session_id = payload.get("session_id")
                    if session_id:
                        await websocket_manager.end_ai_session(session_id)
                        
                # Handle all other message types like regular WebSocket
                else:
                    # Process through regular WebSocket logic
                    pass
                    
            except json.JSONDecodeError:
                await websocket_manager.send_to_connection(
                    connection_id, "error", {"message": "Invalid JSON format"}
                )
                
    except WebSocketDisconnect:
        logger.info(f"Authenticated WebSocket client disconnected: {connection_id}")
        
    except Exception as e:
        logger.error(f"Authenticated WebSocket error: {str(e)}")
        
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)

# Additional WebSocket endpoint for system monitoring
@app.websocket("/ws/admin")
async def admin_websocket_endpoint(websocket: WebSocket, admin_token: str = None):
    """
    Admin WebSocket endpoint for system monitoring and management.
    
    Provides real-time access to:
    - System health metrics
    - AI engine status
    - Circuit breaker states
    - Cache performance
    - Background task status
    """
    connection_id = None
    
    try:
        # Validate admin token
        if not admin_token or admin_token != settings.admin_token:
            await websocket.close(code=4003, reason="Admin access denied")
            return
        
        connection_id = await websocket_manager.connect(websocket)
        
        # Subscribe to admin topics
        await websocket_manager.subscribe_to_topic(connection_id, "admin_metrics")
        await websocket_manager.subscribe_to_topic(connection_id, "system_health")
        await websocket_manager.subscribe_to_topic(connection_id, "ai_status")
        
        logger.info(f"Admin WebSocket connection established: {connection_id}")
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "get_system_status":
                    from app.services.background_tasks import get_background_task_status
                    from app.core.circuit_breaker import circuit_breaker_manager
                    from app.core.advanced_cache import cache_manager
                    
                    system_status = {
                        "background_tasks": get_background_task_status(),
                        "circuit_breakers": circuit_breaker_manager.get_health_summary(),
                        "cache_stats": await cache_manager.get_cache_stats(),
                        "websocket_stats": websocket_manager.get_stats()
                    }
                    
                    await websocket_manager.send_to_connection(
                        connection_id, "system_status", system_status
                    )
                    
            except json.JSONDecodeError:
                await websocket_manager.send_to_connection(
                    connection_id, "error", {"message": "Invalid JSON format"}
                )
                
    except WebSocketDisconnect:
        logger.info(f"Admin WebSocket client disconnected: {connection_id}")
        
    except Exception as e:
        logger.error(f"Admin WebSocket error: {str(e)}")
        
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)


# Startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    logger.info("FastAPI application started successfully")


# Shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown tasks."""
    logger.info("FastAPI application shutting down")


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        loop="uvloop" if not settings.debug else "auto",
        log_level=settings.monitoring.log_level.lower(),
    )