"""
Main FastAPI Application

Entry point for the Telegram ML Bot application with FastAPI.
Provides REST API endpoints, webhook handling, and health monitoring.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends
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
from app.api.v1 import router as api_v1_router
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware
from app.middleware.error_handling import ErrorHandlingMiddleware
from app.telegram.bot import get_bot, cleanup_bot

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
        
        # Initialize Telegram bot
        telegram_bot = await get_bot()
        logger.info("Telegram bot initialized")
        
        # Initialize ML models (if needed)
        # await ml_service.initialize()
        
        # Start background tasks
        # await start_background_tasks()
        
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
        # await ml_service.cleanup()
        
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

# Telegram webhook endpoint
if settings.telegram.webhook_url:
    from app.telegram.webhook import create_webhook_route_handler
    
    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request):
        """Handle Telegram webhook updates."""
        try:
            # Get bot instance
            from app.telegram.bot import telegram_bot
            if not telegram_bot:
                raise HTTPException(status_code=503, detail="Bot not initialized")
            
            # Process update through bot's webhook manager
            update_data = await request.json()
            
            # This would process the update through aiogram
            # For now, return success response
            return {"status": "received"}
            
        except Exception as e:
            logger.error("Error processing webhook", error=str(e))
            raise HTTPException(status_code=500, detail="Webhook processing error")


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