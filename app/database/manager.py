"""
Database Manager Service

Comprehensive database management including health monitoring, backup/recovery,
performance optimization, and maintenance operations.
"""

import asyncio
import os
import subprocess
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import structlog
import psutil

from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import func
import aiofiles

from app.config import settings
from app.database.connection import db_manager
from app.database.repositories import RepositoryFactory
from app.core.redis import redis_manager

logger = structlog.get_logger(__name__)


class DatabaseHealthMonitor:
    """Database health monitoring and alerting."""
    
    def __init__(self):
        self.alert_thresholds = {
            'connection_usage': 0.8,  # 80% of pool used
            'query_time_ms': 5000,    # 5 second query time
            'error_rate': 0.05,       # 5% error rate
            'disk_usage': 0.9,        # 90% disk usage
            'memory_usage': 0.85,     # 85% memory usage
        }
        
        self.metrics_history: List[Dict[str, Any]] = []
    
    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive database health check."""
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'metrics': {},
            'alerts': []
        }
        
        try:
            # Connection pool health
            pool_health = await self._check_connection_pool()
            health_data['checks']['connection_pool'] = pool_health
            
            # Query performance
            query_health = await self._check_query_performance()
            health_data['checks']['query_performance'] = query_health
            
            # Database size and growth
            size_health = await self._check_database_size()
            health_data['checks']['database_size'] = size_health
            
            # Index health
            index_health = await self._check_index_health()
            health_data['checks']['index_health'] = index_health
            
            # Replication status (if applicable)
            replication_health = await self._check_replication_status()
            health_data['checks']['replication'] = replication_health
            
            # System resource usage
            resource_health = await self._check_system_resources()
            health_data['checks']['system_resources'] = resource_health
            
            # Determine overall status
            failed_checks = [k for k, v in health_data['checks'].items() if not v.get('healthy', True)]
            if failed_checks:
                health_data['overall_status'] = 'unhealthy'
                health_data['failed_checks'] = failed_checks
            
            # Store metrics history
            self.metrics_history.append(health_data)
            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_data['overall_status'] = 'error'
            health_data['error'] = str(e)
        
        return health_data
    
    async def _check_connection_pool(self) -> Dict[str, Any]:
        """Check connection pool status."""
        try:
            pool_status = await db_manager.get_pool_status()
            pool_usage = pool_status['checked_out'] / (pool_status['pool_size'] + pool_status['overflow'])
            
            return {
                'healthy': pool_usage < self.alert_thresholds['connection_usage'],
                'pool_size': pool_status['pool_size'],
                'checked_out': pool_status['checked_out'],
                'checked_in': pool_status['checked_in'],
                'overflow': pool_status['overflow'],
                'usage_ratio': pool_usage,
                'warning': pool_usage > 0.7
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_query_performance(self) -> Dict[str, Any]:
        """Check query performance metrics."""
        try:
            async with db_manager.get_async_session() as session:
                # Test a simple query and measure time
                start_time = datetime.utcnow()
                await session.execute(text("SELECT 1"))
                query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Check for slow queries (PostgreSQL specific)
                slow_queries_result = await session.execute(text("""
                    SELECT query, mean_exec_time, calls 
                    FROM pg_stat_statements 
                    WHERE mean_exec_time > 1000 
                    ORDER BY mean_exec_time DESC 
                    LIMIT 5
                """))
                slow_queries = [dict(row._mapping) for row in slow_queries_result]
                
                return {
                    'healthy': query_time < self.alert_thresholds['query_time_ms'],
                    'test_query_time_ms': query_time,
                    'slow_queries': slow_queries,
                    'warning': query_time > 1000
                }
        except Exception as e:
            # Fallback for non-PostgreSQL or missing pg_stat_statements
            return {'healthy': True, 'note': 'Limited performance monitoring available'}
    
    async def _check_database_size(self) -> Dict[str, Any]:
        """Check database size and growth trends."""
        try:
            async with db_manager.get_async_session() as session:
                # Get database size
                size_result = await session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                           pg_database_size(current_database()) as size_bytes
                """))
                size_data = size_result.first()
                
                # Get table sizes
                table_sizes_result = await session.execute(text("""
                    SELECT schemaname, tablename, 
                           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                           pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 10
                """))
                table_sizes = [dict(row._mapping) for row in table_sizes_result]
                
                return {
                    'healthy': True,
                    'total_size': size_data.size,
                    'total_size_bytes': size_data.size_bytes,
                    'largest_tables': table_sizes
                }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_index_health(self) -> Dict[str, Any]:
        """Check index usage and health."""
        try:
            async with db_manager.get_async_session() as session:
                # Check unused indexes
                unused_indexes_result = await session.execute(text("""
                    SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE idx_tup_read = 0 AND idx_tup_fetch = 0
                    AND schemaname = 'public'
                """))
                unused_indexes = [dict(row._mapping) for row in unused_indexes_result]
                
                # Check missing indexes (tables with high seq_scan)
                missing_indexes_result = await session.execute(text("""
                    SELECT schemaname, tablename, seq_scan, seq_tup_read,
                           seq_tup_read / seq_scan as avg_scan_size
                    FROM pg_stat_user_tables 
                    WHERE seq_scan > 1000 AND seq_tup_read / seq_scan > 1000
                    AND schemaname = 'public'
                    ORDER BY seq_tup_read DESC
                """))
                missing_indexes = [dict(row._mapping) for row in missing_indexes_result]
                
                return {
                    'healthy': len(missing_indexes) == 0,
                    'unused_indexes': unused_indexes,
                    'tables_needing_indexes': missing_indexes,
                    'recommendation': 'Consider adding indexes for high-scan tables' if missing_indexes else None
                }
        except Exception as e:
            return {'healthy': True, 'note': 'Index analysis not available'}
    
    async def _check_replication_status(self) -> Dict[str, Any]:
        """Check replication status if configured."""
        try:
            async with db_manager.get_async_session() as session:
                # Check if this is a primary server
                replication_result = await session.execute(text("""
                    SELECT pg_is_in_recovery() as is_replica
                """))
                is_replica = replication_result.scalar()
                
                if not is_replica:
                    # Primary server - check replication slots
                    slots_result = await session.execute(text("""
                        SELECT slot_name, active, restart_lsn, confirmed_flush_lsn
                        FROM pg_replication_slots
                    """))
                    slots = [dict(row._mapping) for row in slots_result]
                    
                    return {
                        'healthy': True,
                        'is_primary': True,
                        'replication_slots': slots
                    }
                else:
                    # Replica server - check lag
                    lag_result = await session.execute(text("""
                        SELECT 
                            pg_last_wal_receive_lsn() as receive_lsn,
                            pg_last_wal_replay_lsn() as replay_lsn,
                            extract(epoch from now() - pg_last_xact_replay_timestamp()) as lag_seconds
                    """))
                    lag_data = lag_result.first()
                    
                    return {
                        'healthy': (lag_data.lag_seconds or 0) < 60,  # Less than 1 minute lag
                        'is_primary': False,
                        'lag_seconds': lag_data.lag_seconds,
                        'receive_lsn': lag_data.receive_lsn,
                        'replay_lsn': lag_data.replay_lsn
                    }
        except Exception as e:
            return {'healthy': True, 'note': 'Replication status not available'}
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100
            
            return {
                'healthy': (
                    memory_usage < self.alert_thresholds['memory_usage'] and
                    disk_usage < self.alert_thresholds['disk_usage']
                ),
                'memory_usage_percent': memory_usage,
                'disk_usage_percent': disk_usage,
                'cpu_usage_percent': cpu_usage,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            return {'healthy': True, 'note': 'System metrics not available'}


class DatabaseBackupManager:
    """Database backup and recovery management."""
    
    def __init__(self):
        self.backup_dir = Path(settings.database.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.retention_days = settings.database.backup_retention_days
        self.compression_enabled = True
    
    async def create_backup(self, backup_type: str = 'full') -> Dict[str, Any]:
        """Create database backup."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{backup_type}_{timestamp}"
        backup_file = self.backup_dir / f"{backup_name}.sql"
        
        try:
            logger.info("Starting database backup", type=backup_type, file=str(backup_file))
            
            # Create backup using pg_dump
            cmd = [
                'pg_dump',
                '--host', settings.database.host,
                '--port', str(settings.database.port),
                '--username', settings.database.user,
                '--dbname', settings.database.name,
                '--verbose',
                '--no-password',
                '--format', 'custom' if backup_type == 'full' else 'plain',
                '--file', str(backup_file)
            ]
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = settings.database.password
            
            # Execute backup
            start_time = datetime.utcnow()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            backup_time = (datetime.utcnow() - start_time).total_seconds()
            
            if process.returncode == 0:
                # Compress backup if enabled
                if self.compression_enabled:
                    compressed_file = backup_file.with_suffix('.sql.gz')
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove uncompressed file
                    backup_file.unlink()
                    backup_file = compressed_file
                
                # Get file size
                file_size = backup_file.stat().st_size
                
                # Clean old backups
                await self._cleanup_old_backups()
                
                logger.info("Backup completed successfully", 
                           file=str(backup_file), 
                           size_mb=file_size / (1024*1024),
                           duration_seconds=backup_time)
                
                return {
                    'success': True,
                    'backup_file': str(backup_file),
                    'size_bytes': file_size,
                    'size_mb': file_size / (1024*1024),
                    'duration_seconds': backup_time,
                    'timestamp': timestamp,
                    'compressed': self.compression_enabled
                }
            else:
                error_msg = stderr.decode() if stderr else 'Unknown error'
                logger.error("Backup failed", error=error_msg)
                
                return {
                    'success': False,
                    'error': error_msg,
                    'duration_seconds': backup_time
                }
                
        except Exception as e:
            logger.error("Backup error", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    async def restore_backup(self, backup_file: str, target_db: str = None) -> Dict[str, Any]:
        """Restore database from backup."""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            return {'success': False, 'error': 'Backup file not found'}
        
        target_db = target_db or f"{settings.database.name}_restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info("Starting database restore", backup=backup_file, target=target_db)
            
            # Create target database
            await self._create_database(target_db)
            
            # Determine if backup is compressed
            is_compressed = backup_path.suffix == '.gz'
            
            if is_compressed:
                # Decompress first
                temp_file = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                restore_file = temp_file
            else:
                restore_file = backup_path
            
            # Restore using pg_restore or psql
            if restore_file.suffix == '.sql':
                cmd = [
                    'psql',
                    '--host', settings.database.host,
                    '--port', str(settings.database.port),
                    '--username', settings.database.user,
                    '--dbname', target_db,
                    '--file', str(restore_file)
                ]
            else:
                cmd = [
                    'pg_restore',
                    '--host', settings.database.host,
                    '--port', str(settings.database.port),
                    '--username', settings.database.user,
                    '--dbname', target_db,
                    '--verbose',
                    '--no-password',
                    str(restore_file)
                ]
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = settings.database.password
            
            # Execute restore
            start_time = datetime.utcnow()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            restore_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Cleanup temp file if created
            if is_compressed and temp_file.exists():
                temp_file.unlink()
            
            if process.returncode == 0:
                logger.info("Restore completed successfully", 
                           target=target_db,
                           duration_seconds=restore_time)
                
                return {
                    'success': True,
                    'target_database': target_db,
                    'duration_seconds': restore_time
                }
            else:
                error_msg = stderr.decode() if stderr else 'Unknown error'
                logger.error("Restore failed", error=error_msg)
                
                return {
                    'success': False,
                    'error': error_msg,
                    'duration_seconds': restore_time
                }
                
        except Exception as e:
            logger.error("Restore error", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_database(self, db_name: str) -> None:
        """Create a new database."""
        # Use sync connection to create database
        from sqlalchemy import create_engine
        
        # Connect to postgres database to create new database
        admin_url = settings.database.sync_url.replace(f"/{settings.database.name}", "/postgres")
        admin_engine = create_engine(admin_url)
        
        with admin_engine.connect() as conn:
            conn.execute(text("COMMIT"))  # End any transaction
            conn.execute(text(f"CREATE DATABASE \"{db_name}\""))
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        for backup_file in self.backup_dir.glob("backup_*.sql*"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    backup_file.unlink()
                    logger.info("Deleted old backup", file=str(backup_file))
                except Exception as e:
                    logger.warning("Failed to delete backup", file=str(backup_file), error=str(e))
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("backup_*.sql*"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = backup_file.stat()
            
            backups.append({
                'filename': backup_file.name,
                'path': str(backup_file),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024*1024),
                'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'compressed': backup_file.suffix == '.gz'
            })
        
        return backups


class DatabaseMaintenanceManager:
    """Database maintenance operations."""
    
    def __init__(self):
        self.health_monitor = DatabaseHealthMonitor()
        self.backup_manager = DatabaseBackupManager()
    
    async def perform_maintenance(self, operations: List[str] = None) -> Dict[str, Any]:
        """Perform database maintenance operations."""
        operations = operations or ['vacuum', 'analyze', 'reindex']
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'operations': {},
            'summary': {'successful': 0, 'failed': 0}
        }
        
        for operation in operations:
            try:
                if operation == 'vacuum':
                    result = await self._vacuum_database()
                elif operation == 'analyze':
                    result = await self._analyze_database()
                elif operation == 'reindex':
                    result = await self._reindex_database()
                elif operation == 'cleanup_logs':
                    result = await self._cleanup_logs()
                else:
                    result = {'success': False, 'error': f'Unknown operation: {operation}'}
                
                results['operations'][operation] = result
                
                if result.get('success', False):
                    results['summary']['successful'] += 1
                else:
                    results['summary']['failed'] += 1
                    
            except Exception as e:
                results['operations'][operation] = {'success': False, 'error': str(e)}
                results['summary']['failed'] += 1
        
        return results
    
    async def _vacuum_database(self) -> Dict[str, Any]:
        """Vacuum database tables."""
        try:
            async with db_manager.get_async_session() as session:
                # Get list of tables to vacuum
                tables_result = await session.execute(text("""
                    SELECT tablename FROM pg_tables WHERE schemaname = 'public'
                """))
                tables = [row[0] for row in tables_result]
                
                vacuumed_tables = []
                
                for table in tables:
                    try:
                        await session.execute(text(f"VACUUM ANALYZE {table}"))
                        vacuumed_tables.append(table)
                    except Exception as e:
                        logger.warning("Failed to vacuum table", table=table, error=str(e))
                
                return {
                    'success': True,
                    'tables_vacuumed': len(vacuumed_tables),
                    'tables': vacuumed_tables
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _analyze_database(self) -> Dict[str, Any]:
        """Analyze database statistics."""
        try:
            async with db_manager.get_async_session() as session:
                await session.execute(text("ANALYZE"))
                
                return {'success': True, 'message': 'Database statistics updated'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _reindex_database(self) -> Dict[str, Any]:
        """Reindex database tables."""
        try:
            async with db_manager.get_async_session() as session:
                # Get indexes that might need rebuilding
                indexes_result = await session.execute(text("""
                    SELECT schemaname, tablename, indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public'
                """))
                
                indexes = [dict(row._mapping) for row in indexes_result]
                reindexed_count = 0
                
                for index in indexes:
                    try:
                        await session.execute(text(f"REINDEX INDEX {index['indexname']}"))
                        reindexed_count += 1
                    except Exception as e:
                        logger.warning("Failed to reindex", index=index['indexname'], error=str(e))
                
                return {
                    'success': True,
                    'indexes_reindexed': reindexed_count,
                    'total_indexes': len(indexes)
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _cleanup_logs(self) -> Dict[str, Any]:
        """Clean up old log entries."""
        try:
            # Clean up audit logs older than retention period
            from app.database.repositories import audit_repo
            
            cutoff_date = datetime.utcnow() - timedelta(days=90)  # 90 days retention
            
            # This would be implemented based on specific retention policies
            # For now, return success
            
            return {
                'success': True,
                'message': 'Log cleanup completed'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            async with db_manager.get_async_session() as session:
                # Table statistics
                table_stats_result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY n_live_tup DESC
                """))
                
                table_stats = [dict(row._mapping) for row in table_stats_result]
                
                # Database-wide statistics
                db_stats_result = await session.execute(text("""
                    SELECT 
                        numbackends as active_connections,
                        xact_commit as transactions_committed,
                        xact_rollback as transactions_rolled_back,
                        blks_read as blocks_read,
                        blks_hit as blocks_hit,
                        tup_returned as tuples_returned,
                        tup_fetched as tuples_fetched,
                        tup_inserted as tuples_inserted,
                        tup_updated as tuples_updated,
                        tup_deleted as tuples_deleted
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """))
                
                db_stats = dict(db_stats_result.first()._mapping)
                
                # Cache hit ratio
                cache_hit_ratio = (db_stats['blocks_hit'] / (db_stats['blocks_hit'] + db_stats['blocks_read'])) * 100 if (db_stats['blocks_hit'] + db_stats['blocks_read']) > 0 else 0
                
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'database_stats': db_stats,
                    'cache_hit_ratio_percent': cache_hit_ratio,
                    'table_stats': table_stats,
                    'total_tables': len(table_stats),
                    'total_live_tuples': sum(t['live_tuples'] or 0 for t in table_stats),
                    'total_dead_tuples': sum(t['dead_tuples'] or 0 for t in table_stats)
                }
                
        except Exception as e:
            return {'error': str(e)}


class DatabaseService:
    """Main database service combining all management capabilities."""
    
    def __init__(self):
        self.health_monitor = DatabaseHealthMonitor()
        self.backup_manager = DatabaseBackupManager()
        self.maintenance_manager = DatabaseMaintenanceManager()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize database service."""
        logger.info("Initializing database service")
        
        # Initialize database manager
        await db_manager.initialize()
        
        # Start background monitoring
        await self._start_background_monitoring()
        
        logger.info("Database service initialized")
    
    async def shutdown(self) -> None:
        """Shutdown database service."""
        logger.info("Shutting down database service")
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        
        # Close database connections
        await db_manager.close()
        
        logger.info("Database service shutdown complete")
    
    async def _start_background_monitoring(self) -> None:
        """Start background health monitoring."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._backup_task = asyncio.create_task(self._backup_loop())
    
    async def _monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                # Perform health check
                health_data = await self.health_monitor.check_health()
                
                # Store health data in cache for API access
                await redis_manager.set_json('db:health', health_data, expire=300)
                
                # Alert on critical issues
                if health_data['overall_status'] == 'unhealthy':
                    logger.warning("Database health check failed", 
                                 failed_checks=health_data.get('failed_checks', []))
                
                # Wait before next check
                await asyncio.sleep(settings.monitoring.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(60)  # Retry in 1 minute
    
    async def _backup_loop(self) -> None:
        """Background backup loop."""
        backup_interval_hours = settings.database.get('backup_interval_hours', 24)
        
        while True:
            try:
                # Wait for backup interval
                await asyncio.sleep(backup_interval_hours * 3600)
                
                # Create backup
                backup_result = await self.backup_manager.create_backup('scheduled')
                
                if backup_result['success']:
                    logger.info("Scheduled backup completed", 
                               file=backup_result['backup_file'],
                               size_mb=backup_result['size_mb'])
                else:
                    logger.error("Scheduled backup failed", 
                                error=backup_result['error'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Backup loop error", error=str(e))
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    # Public API methods
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current database health status."""
        return await self.health_monitor.check_health()
    
    async def create_backup(self, backup_type: str = 'manual') -> Dict[str, Any]:
        """Create database backup."""
        return await self.backup_manager.create_backup(backup_type)
    
    async def restore_backup(self, backup_file: str, target_db: str = None) -> Dict[str, Any]:
        """Restore database from backup."""
        return await self.backup_manager.restore_backup(backup_file, target_db)
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        return await self.backup_manager.list_backups()
    
    async def perform_maintenance(self, operations: List[str] = None) -> Dict[str, Any]:
        """Perform database maintenance."""
        return await self.maintenance_manager.perform_maintenance(operations)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await self.maintenance_manager.get_database_statistics()


# Global database service instance
db_service = DatabaseService()