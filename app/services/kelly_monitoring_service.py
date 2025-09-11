"""
Kelly AI Monitoring Service

Service for collecting, processing, and broadcasting real-time monitoring data
including metrics collection, activity tracking, and performance monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import statistics
import uuid

import structlog
from sqlalchemy import func, and_, or_

from app.core.redis import redis_manager
from app.database.manager import db_service
from app.models.kelly_monitoring import (
    SystemMetric, ActivityEvent, AlertInstance, InterventionLog,
    EmergencyAction, MonitoringSession, PerformanceBenchmark
)
from app.websocket.manager import websocket_manager

logger = structlog.get_logger()

class KellyMonitoringService:
    """Service for Kelly AI real-time monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics_buffer = defaultdict(deque)  # Buffer for real-time metrics
        self.performance_buffer = defaultdict(deque)  # Buffer for performance data
        self.activity_buffer = deque(maxlen=1000)  # Recent activities
        
        # Configuration
        self.metrics_retention_seconds = 300  # 5 minutes of real-time data
        self.buffer_flush_interval = 10  # Flush to DB every 10 seconds
        self.broadcast_interval = 5  # Broadcast updates every 5 seconds
        
        # State tracking
        self.is_running = False
        self.background_tasks = []
        
    async def initialize(self):
        """Initialize the monitoring service"""
        if self.is_running:
            return
            
        logger.info("Initializing Kelly monitoring service...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._metrics_broadcast_loop()),
            asyncio.create_task(self._buffer_flush_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._activity_aggregation_loop())
        ]
        
        self.is_running = True
        logger.info("Kelly monitoring service initialized")
    
    async def shutdown(self):
        """Shutdown the monitoring service"""
        if not self.is_running:
            return
            
        logger.info("Shutting down Kelly monitoring service...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Flush remaining data
        await self._flush_metrics_to_database()
        await self._flush_performance_to_database()
        
        self.is_running = False
        logger.info("Kelly monitoring service shutdown complete")
    
    # ===== METRICS COLLECTION =====
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        account_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value"""
        try:
            timestamp = datetime.utcnow()
            
            metric_data = {
                "metric_name": metric_name,
                "metric_type": metric_type,
                "value": value,
                "timestamp": timestamp,
                "account_id": account_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "metadata": metadata or {}
            }
            
            # Add to real-time buffer
            self.metrics_buffer[metric_name].append(metric_data)
            
            # Maintain buffer size
            if len(self.metrics_buffer[metric_name]) > 1000:
                self.metrics_buffer[metric_name].popleft()
            
            # Store in Redis for immediate access
            redis_key = f"kelly:metric:{metric_name}:latest"
            await redis_manager.setex(redis_key, 300, json.dumps({
                "value": value,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {}
            }))
            
            # Also store time series data in Redis
            ts_key = f"kelly:metric_timeseries:{metric_name}:{int(timestamp.timestamp() // 60)}"  # Per minute
            await redis_manager.setex(ts_key, 3600, value)  # Keep for 1 hour
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
    
    async def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics"""
        try:
            # Collect metrics from various sources
            metrics = {}
            
            # Active conversations
            conv_keys = await redis_manager.keys("kelly:conversation_track:*")
            metrics["conversations_active"] = len(conv_keys)
            
            # Messages processed today
            today = datetime.now().strftime("%Y-%m-%d")
            daily_keys = await redis_manager.keys(f"kelly:daily_stats:*:{today}")
            
            total_messages = 0
            total_conversations = 0
            ai_confidence_sum = 0.0
            safety_score_sum = 0.0
            message_count = 0
            
            for key in daily_keys:
                daily_data = await redis_manager.lrange(key, 0, -1)
                
                # Count unique conversations
                user_ids = set()
                for data_str in daily_data:
                    try:
                        data = json.loads(data_str)
                        user_ids.add(data.get("user_id"))
                        ai_confidence_sum += data.get("ai_confidence", 0.7)
                        safety_score_sum += data.get("safety_score", 1.0)
                        message_count += 1
                    except:
                        continue
                
                total_conversations += len(user_ids)
                total_messages += len(daily_data)
            
            metrics.update({
                "conversations_total_today": total_conversations,
                "messages_processed": total_messages,
                "ai_confidence_avg": ai_confidence_sum / max(message_count, 1),
                "safety_score_avg": safety_score_sum / max(message_count, 1)
            })
            
            # Interventions and alerts
            intervention_keys = await redis_manager.keys("kelly:intervention:*")
            alert_keys = await redis_manager.keys("kelly:alert:active:*")
            
            metrics.update({
                "human_interventions": len(intervention_keys),
                "alert_count": len(alert_keys)
            })
            
            # System performance
            system_load = await self._calculate_system_load()
            response_time = await self._get_average_response_time()
            
            metrics.update({
                "system_load": system_load,
                "response_time_avg": response_time
            })
            
            # Claude API metrics
            claude_requests = await redis_manager.get("kelly:claude:requests_today") or "0"
            claude_cost = await redis_manager.get("kelly:claude:cost_today") or "0.0"
            
            metrics.update({
                "claude_requests_today": int(claude_requests),
                "claude_cost_today": float(claude_cost),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting live metrics: {e}")
            return {}
    
    async def get_metric_history(
        self,
        metric_name: str,
        duration_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=duration_minutes)
            
            # Get from buffer first (most recent data)
            buffer_data = []
            if metric_name in self.metrics_buffer:
                for metric_data in self.metrics_buffer[metric_name]:
                    if metric_data["timestamp"] >= start_time:
                        buffer_data.append({
                            "timestamp": metric_data["timestamp"].isoformat(),
                            "value": metric_data["value"],
                            "metadata": metric_data.get("metadata", {})
                        })
            
            # Get from database for older data
            async with db_service.get_session() as session:
                db_metrics = await session.execute(
                    session.query(SystemMetric)
                    .filter(
                        and_(
                            SystemMetric.metric_name == metric_name,
                            SystemMetric.timestamp >= start_time,
                            SystemMetric.timestamp <= end_time
                        )
                    )
                    .order_by(SystemMetric.timestamp)
                )
                
                db_data = []
                for metric in db_metrics.scalars():
                    db_data.append({
                        "timestamp": metric.timestamp.isoformat(),
                        "value": metric.value,
                        "metadata": metric.metadata or {}
                    })
            
            # Combine and sort
            all_data = buffer_data + db_data
            all_data.sort(key=lambda x: x["timestamp"])
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error getting metric history for {metric_name}: {e}")
            return []
    
    # ===== ACTIVITY TRACKING =====
    
    async def log_activity(
        self,
        event_type: str,
        title: str,
        description: str = "",
        severity: str = "low",
        category: str = "system",
        account_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an activity event"""
        try:
            activity_data = {
                "id": str(uuid.uuid4()),
                "event_type": event_type,
                "title": title,
                "description": description,
                "severity": severity,
                "category": category,
                "account_id": account_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            # Add to buffer
            self.activity_buffer.append(activity_data)
            
            # Store in Redis for immediate access
            activity_key = f"kelly:activity:{account_id or 'system'}:{int(datetime.utcnow().timestamp())}"
            await redis_manager.setex(activity_key, 86400 * 7, json.dumps({
                **activity_data,
                "created_at": activity_data["created_at"].isoformat()
            }))
            
            # Broadcast to WebSocket subscribers
            await websocket_manager.broadcast_activity_update(activity_data)
            
        except Exception as e:
            logger.error(f"Error logging activity {event_type}: {e}")
    
    async def get_recent_activities(
        self,
        limit: int = 50,
        severity_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        account_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent activity events"""
        try:
            # Get from buffer (most recent)
            activities = list(self.activity_buffer)
            
            # Apply filters
            if severity_filter:
                activities = [a for a in activities if a["severity"] == severity_filter]
            
            if category_filter:
                activities = [a for a in activities if a["category"] == category_filter]
            
            if account_filter:
                activities = [a for a in activities if a["account_id"] == account_filter]
            
            # Sort by creation time (most recent first)
            activities.sort(key=lambda x: x["created_at"], reverse=True)
            
            # Convert timestamps to ISO format
            for activity in activities:
                activity["created_at"] = activity["created_at"].isoformat()
            
            return activities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent activities: {e}")
            return []
    
    # ===== PERFORMANCE MONITORING =====
    
    async def record_performance_metric(
        self,
        component: str,
        metric_type: str,
        value: float,
        unit: str = "ms",
        account_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric"""
        try:
            perf_data = {
                "component": component,
                "metric_type": metric_type,
                "value": value,
                "unit": unit,
                "account_id": account_id,
                "conversation_id": conversation_id,
                "measured_at": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            # Add to buffer
            buffer_key = f"{component}:{metric_type}"
            self.performance_buffer[buffer_key].append(perf_data)
            
            # Maintain buffer size
            if len(self.performance_buffer[buffer_key]) > 1000:
                self.performance_buffer[buffer_key].popleft()
            
        except Exception as e:
            logger.error(f"Error recording performance metric {component}:{metric_type}: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        try:
            summary = {}
            
            # Calculate statistics for each component/metric combination
            for buffer_key, data_points in self.performance_buffer.items():
                if not data_points:
                    continue
                
                component, metric_type = buffer_key.split(":", 1)
                
                values = [dp["value"] for dp in data_points]
                
                if component not in summary:
                    summary[component] = {}
                
                summary[component][metric_type] = {
                    "current": values[-1] if values else 0,
                    "avg": statistics.mean(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else (values[-1] if values else 0),
                    "sample_count": len(values)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    # ===== ALERT GENERATION =====
    
    async def check_and_generate_alerts(self):
        """Check metrics and generate alerts if thresholds are exceeded"""
        try:
            alerts_generated = []
            
            # Get current metrics
            metrics = await self.get_live_metrics()
            
            # Check alert conditions
            alert_conditions = [
                {
                    "metric": "system_load",
                    "threshold": 80.0,
                    "operator": ">",
                    "severity": "high",
                    "title": "High System Load",
                    "description": "System load exceeds 80%"
                },
                {
                    "metric": "response_time_avg",
                    "threshold": 2000.0,
                    "operator": ">",
                    "severity": "medium",
                    "title": "High Response Time",
                    "description": "Average response time exceeds 2 seconds"
                },
                {
                    "metric": "alert_count",
                    "threshold": 10,
                    "operator": ">",
                    "severity": "high",
                    "title": "High Alert Volume",
                    "description": "Too many active alerts"
                },
                {
                    "metric": "ai_confidence_avg",
                    "threshold": 0.4,
                    "operator": "<",
                    "severity": "medium",
                    "title": "Low AI Confidence",
                    "description": "Average AI confidence is low"
                },
                {
                    "metric": "safety_score_avg",
                    "threshold": 0.6,
                    "operator": "<",
                    "severity": "critical",
                    "title": "Low Safety Score",
                    "description": "Average safety score is critically low"
                }
            ]
            
            for condition in alert_conditions:
                metric_value = metrics.get(condition["metric"], 0)
                threshold = condition["threshold"]
                operator = condition["operator"]
                
                triggered = False
                if operator == ">" and metric_value > threshold:
                    triggered = True
                elif operator == "<" and metric_value < threshold:
                    triggered = True
                elif operator == "==" and metric_value == threshold:
                    triggered = True
                
                if triggered:
                    # Check if alert already exists (don't spam)
                    alert_key = f"kelly:alert_cooldown:{condition['metric']}"
                    if await redis_manager.get(alert_key):
                        continue  # Alert already generated recently
                    
                    # Generate alert
                    alert_id = await self._create_alert(
                        alert_type=f"metric_threshold_{condition['metric']}",
                        title=condition["title"],
                        description=f"{condition['description']}. Current value: {metric_value}, threshold: {threshold}",
                        severity=condition["severity"],
                        category="performance",
                        metadata={
                            "metric_name": condition["metric"],
                            "current_value": metric_value,
                            "threshold": threshold,
                            "operator": operator
                        }
                    )
                    
                    alerts_generated.append(alert_id)
                    
                    # Set cooldown to prevent spam
                    await redis_manager.setex(alert_key, 300, "1")  # 5 minute cooldown
            
            return alerts_generated
            
        except Exception as e:
            logger.error(f"Error checking and generating alerts: {e}")
            return []
    
    async def _create_alert(
        self,
        alert_type: str,
        title: str,
        description: str,
        severity: str,
        category: str,
        account_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert"""
        try:
            alert_id = str(uuid.uuid4())
            
            alert_data = {
                "id": alert_id,
                "alert_type": alert_type,
                "title": title,
                "description": description,
                "severity": severity,
                "category": category,
                "account_id": account_id,
                "conversation_id": conversation_id,
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "auto_generated": True,
                "requires_human_review": severity in ["high", "critical"],
                "metadata": metadata or {}
            }
            
            # Store in Redis for immediate access
            if account_id:
                alert_key = f"kelly:alert:active:{account_id}:{alert_id}"
            else:
                alert_key = f"kelly:alert:active:system:{alert_id}"
            
            await redis_manager.setex(alert_key, 86400 * 7, json.dumps(alert_data))
            
            # Broadcast alert
            await websocket_manager.broadcast_alert_notification(alert_data)
            
            # Log activity
            await self.log_activity(
                event_type="alert_generated",
                title=f"Alert Generated: {title}",
                description=description,
                severity=severity,
                category="alert",
                account_id=account_id,
                conversation_id=conversation_id,
                metadata={"alert_id": alert_id}
            )
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    # ===== BACKGROUND TASKS =====
    
    async def _metrics_collection_loop(self):
        """Background task to collect metrics"""
        while self.is_running:
            try:
                # Record system metrics
                await self._collect_system_metrics()
                
                # Check for alerts
                await self.check_and_generate_alerts()
                
                await asyncio.sleep(self.broadcast_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_broadcast_loop(self):
        """Background task to broadcast metrics"""
        while self.is_running:
            try:
                # Get current metrics
                metrics = await self.get_live_metrics()
                
                # Broadcast to WebSocket subscribers
                await websocket_manager.broadcast_metrics_update(metrics)
                
                await asyncio.sleep(self.broadcast_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics broadcast loop: {e}")
                await asyncio.sleep(10)
    
    async def _buffer_flush_loop(self):
        """Background task to flush buffers to database"""
        while self.is_running:
            try:
                await self._flush_metrics_to_database()
                await self._flush_activities_to_database()
                await self._flush_performance_to_database()
                
                await asyncio.sleep(self.buffer_flush_interval)
                
            except Exception as e:
                logger.error(f"Error in buffer flush loop: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitoring_loop(self):
        """Background task to monitor performance"""
        while self.is_running:
            try:
                # Monitor API response times
                await self._monitor_api_performance()
                
                # Monitor database performance  
                await self._monitor_database_performance()
                
                # Monitor Redis performance
                await self._monitor_redis_performance()
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _activity_aggregation_loop(self):
        """Background task to aggregate activity data"""
        while self.is_running:
            try:
                # Aggregate hourly activity statistics
                await self._aggregate_hourly_activities()
                
                # Clean up old activity data
                await self._cleanup_old_activities()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Error in activity aggregation loop: {e}")
                await asyncio.sleep(1800)
    
    # ===== HELPER METHODS =====
    
    async def _collect_system_metrics(self):
        """Collect various system metrics"""
        try:
            import psutil
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            await self.record_metric("system_cpu_percent", cpu_percent)
            await self.record_metric("system_memory_percent", memory_percent)
            
            # Redis info
            try:
                redis_info = await redis_manager.info()
                await self.record_metric("redis_memory_usage", redis_info.get("used_memory", 0))
                await self.record_metric("redis_connected_clients", redis_info.get("connected_clients", 0))
                await self.record_metric("redis_ops_per_sec", redis_info.get("instantaneous_ops_per_sec", 0))
            except:
                pass
            
            # WebSocket connections
            ws_stats = websocket_manager.get_stats()
            await self.record_metric("websocket_connections", ws_stats["total_connections"])
            await self.record_metric("websocket_messages_sent", ws_stats["messages_sent"])
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _calculate_system_load(self) -> float:
        """Calculate overall system load percentage"""
        try:
            # Get recent CPU, memory, and connection metrics
            conv_count = len(await redis_manager.keys("kelly:conversation_track:*"))
            alert_count = len(await redis_manager.keys("kelly:alert:active:*"))
            
            # Simple load calculation
            base_load = min(100.0, conv_count * 2.0)  # 2% per conversation
            alert_load = min(20.0, alert_count * 2.0)  # 2% per alert
            
            return min(100.0, base_load + alert_load)
            
        except:
            return 0.0
    
    async def _get_average_response_time(self) -> float:
        """Get average response time from recent data"""
        try:
            # Get from performance buffer
            response_times = []
            
            for buffer_key, data_points in self.performance_buffer.items():
                if "response_time" in buffer_key:
                    response_times.extend([dp["value"] for dp in data_points])
            
            if response_times:
                return statistics.mean(response_times[-20:])  # Last 20 measurements
            
            return 0.0
            
        except:
            return 0.0
    
    async def _flush_metrics_to_database(self):
        """Flush metrics buffer to database"""
        try:
            if not self.metrics_buffer:
                return
            
            async with db_service.get_session() as session:
                metrics_to_insert = []
                
                for metric_name, data_points in self.metrics_buffer.items():
                    # Take up to 100 points per metric per flush
                    points_to_process = []
                    for _ in range(min(100, len(data_points))):
                        if data_points:
                            points_to_process.append(data_points.popleft())
                    
                    for point in points_to_process:
                        metric = SystemMetric(
                            metric_name=point["metric_name"],
                            metric_type=point["metric_type"],
                            value=point["value"],
                            timestamp=point["timestamp"],
                            account_id=point.get("account_id"),
                            conversation_id=point.get("conversation_id"),
                            user_id=point.get("user_id"),
                            metadata=point.get("metadata")
                        )
                        metrics_to_insert.append(metric)
                
                if metrics_to_insert:
                    session.add_all(metrics_to_insert)
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Error flushing metrics to database: {e}")
    
    async def _flush_activities_to_database(self):
        """Flush activities buffer to database"""
        try:
            if not self.activity_buffer:
                return
            
            async with db_service.get_session() as session:
                activities_to_insert = []
                
                # Process up to 50 activities per flush
                for _ in range(min(50, len(self.activity_buffer))):
                    if self.activity_buffer:
                        activity_data = self.activity_buffer.popleft()
                        
                        activity = ActivityEvent(
                            event_type=activity_data["event_type"],
                            title=activity_data["title"],
                            description=activity_data["description"],
                            account_id=activity_data.get("account_id"),
                            conversation_id=activity_data.get("conversation_id"),
                            user_id=activity_data.get("user_id"),
                            severity=activity_data["severity"],
                            category=activity_data["category"],
                            created_at=activity_data["created_at"],
                            metadata=activity_data.get("metadata")
                        )
                        activities_to_insert.append(activity)
                
                if activities_to_insert:
                    session.add_all(activities_to_insert)
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Error flushing activities to database: {e}")
    
    async def _flush_performance_to_database(self):
        """Flush performance buffer to database"""
        try:
            if not self.performance_buffer:
                return
            
            async with db_service.get_session() as session:
                benchmarks_to_insert = []
                
                for buffer_key, data_points in self.performance_buffer.items():
                    component, metric_type = buffer_key.split(":", 1)
                    
                    # Process up to 20 points per metric per flush
                    points_to_process = []
                    for _ in range(min(20, len(data_points))):
                        if data_points:
                            points_to_process.append(data_points.popleft())
                    
                    if points_to_process:
                        # Calculate statistics for this batch
                        values = [p["value"] for p in points_to_process]
                        
                        benchmark = PerformanceBenchmark(
                            benchmark_type=metric_type,
                            component=component,
                            value=statistics.mean(values),
                            unit=points_to_process[0]["unit"],
                            min_value=min(values),
                            max_value=max(values),
                            avg_value=statistics.mean(values),
                            p95_value=statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                            sample_size=len(values),
                            measurement_period_seconds=self.buffer_flush_interval,
                            measured_at=points_to_process[-1]["measured_at"],
                            account_id=points_to_process[0].get("account_id"),
                            conversation_id=points_to_process[0].get("conversation_id"),
                            metadata=points_to_process[0].get("metadata")
                        )
                        benchmarks_to_insert.append(benchmark)
                
                if benchmarks_to_insert:
                    session.add_all(benchmarks_to_insert)
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Error flushing performance to database: {e}")
    
    async def _monitor_api_performance(self):
        """Monitor API endpoint performance"""
        try:
            # This would typically measure actual API response times
            # For now, we'll simulate based on system load
            
            load = await self._calculate_system_load()
            simulated_response_time = 100 + (load * 5)  # Base 100ms + 5ms per % load
            
            await self.record_performance_metric(
                component="api",
                metric_type="response_time", 
                value=simulated_response_time,
                unit="ms"
            )
            
        except Exception as e:
            logger.error(f"Error monitoring API performance: {e}")
    
    async def _monitor_database_performance(self):
        """Monitor database performance"""
        try:
            # Measure a simple query time
            start_time = datetime.utcnow()
            
            async with db_service.get_session() as session:
                await session.execute("SELECT 1")
            
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            await self.record_performance_metric(
                component="database",
                metric_type="query_time",
                value=query_time,
                unit="ms"
            )
            
        except Exception as e:
            logger.error(f"Error monitoring database performance: {e}")
    
    async def _monitor_redis_performance(self):
        """Monitor Redis performance"""
        try:
            # Measure Redis ping time
            start_time = datetime.utcnow()
            await redis_manager.ping()
            ping_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            await self.record_performance_metric(
                component="redis",
                metric_type="ping_time",
                value=ping_time,
                unit="ms"
            )
            
        except Exception as e:
            logger.error(f"Error monitoring Redis performance: {e}")
    
    async def _aggregate_hourly_activities(self):
        """Aggregate activity data by hour"""
        try:
            # This would create hourly summaries of activity data
            # Implementation depends on specific requirements
            pass
            
        except Exception as e:
            logger.error(f"Error aggregating hourly activities: {e}")
    
    async def _cleanup_old_activities(self):
        """Clean up old activity data"""
        try:
            # Remove Redis activity data older than 7 days
            cutoff_timestamp = int((datetime.utcnow() - timedelta(days=7)).timestamp())
            
            activity_keys = await redis_manager.keys("kelly:activity:*")
            for key in activity_keys:
                try:
                    key_timestamp = int(key.split(":")[-1])
                    if key_timestamp < cutoff_timestamp:
                        await redis_manager.delete(key)
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error cleaning up old activities: {e}")

# Global monitoring service instance
kelly_monitoring_service = KellyMonitoringService()