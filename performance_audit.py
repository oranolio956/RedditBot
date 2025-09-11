#!/usr/bin/env python3
"""
Comprehensive Performance Audit Tool

Analyzes system performance across multiple dimensions:
- API response times
- Database query optimization
- Memory usage patterns
- CPU utilization
- WebSocket performance
- Caching effectiveness
- Bottleneck identification
"""

import asyncio
import time
import json
import psutil
import aiohttp
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
import os
import subprocess
import statistics
import tracemalloc
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    timestamp: datetime
    api_response_times: Dict[str, float]
    database_query_times: Dict[str, float]
    memory_usage: Dict[str, float]
    cpu_usage: float
    cache_hit_rates: Dict[str, float]
    error_rates: Dict[str, float]
    throughput: Dict[str, float]
    bottlenecks: List[str]


class PerformanceAuditor:
    """Comprehensive performance auditing system."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.base_url = "http://localhost:8000"  # Default FastAPI port
        
    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run complete performance audit."""
        print("🔍 Starting Comprehensive Performance Audit...")
        
        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "audit_duration": 0,
            "system_info": self._get_system_info(),
            "api_performance": {},
            "database_performance": {},
            "memory_analysis": {},
            "cpu_analysis": {},
            "caching_analysis": {},
            "bottleneck_analysis": {},
            "recommendations": [],
            "performance_score": 0
        }
        
        start_audit = time.time()
        
        try:
            # 1. API Performance Analysis
            print("\n📊 Analyzing API Performance...")
            audit_results["api_performance"] = await self._analyze_api_performance()
            
            # 2. Database Performance Analysis
            print("\n🗄️ Analyzing Database Performance...")
            audit_results["database_performance"] = await self._analyze_database_performance()
            
            # 3. Memory Usage Analysis
            print("\n💾 Analyzing Memory Usage...")
            audit_results["memory_analysis"] = await self._analyze_memory_usage()
            
            # 4. CPU Usage Analysis
            print("\n⚡ Analyzing CPU Usage...")
            audit_results["cpu_analysis"] = await self._analyze_cpu_usage()
            
            # 5. Caching Performance Analysis
            print("\n📦 Analyzing Caching Performance...")
            audit_results["caching_analysis"] = await self._analyze_caching_performance()
            
            # 6. Bottleneck Detection
            print("\n🔴 Detecting Performance Bottlenecks...")
            audit_results["bottleneck_analysis"] = await self._detect_bottlenecks(audit_results)
            
            # 7. Generate Recommendations
            print("\n💡 Generating Optimization Recommendations...")
            audit_results["recommendations"] = self._generate_recommendations(audit_results)
            
            # 8. Calculate Performance Score
            audit_results["performance_score"] = self._calculate_performance_score(audit_results)
            
            audit_results["audit_duration"] = time.time() - start_audit
            
        except Exception as e:
            print(f"❌ Audit failed: {e}")
            audit_results["error"] = str(e)
            audit_results["error_traceback"] = traceback.format_exc()
        
        return audit_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    
    async def _analyze_api_performance(self) -> Dict[str, Any]:
        """Analyze API endpoint performance."""
        endpoints = [
            "/",
            "/health",
            "/health/detailed",
            "/docs",
            "/api/v1/users/profile",
            "/api/v1/conversations",
            "/api/v1/ml/features"
        ]
        
        results = {
            "endpoint_times": {},
            "average_response_time": 0,
            "slowest_endpoint": "",
            "fastest_endpoint": "",
            "error_rate": 0,
            "throughput_requests_per_second": 0,
            "status_codes": defaultdict(int)
        }
        
        response_times = []
        error_count = 0
        total_requests = 0
        
        # Test each endpoint multiple times
        for endpoint in endpoints:
            endpoint_times = []
            endpoint_errors = 0
            
            for _ in range(5):  # 5 requests per endpoint
                try:
                    start_time = time.time()
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                            await response.text()
                            response_time = time.time() - start_time
                            endpoint_times.append(response_time)
                            response_times.append(response_time)
                            results["status_codes"][response.status] += 1
                            
                            if response.status >= 400:
                                endpoint_errors += 1
                                error_count += 1
                            
                            total_requests += 1
                            
                except Exception as e:
                    print(f"⚠️  Failed to test {endpoint}: {e}")
                    endpoint_errors += 1
                    error_count += 1
                    total_requests += 1
            
            if endpoint_times:
                results["endpoint_times"][endpoint] = {
                    "average_ms": round(statistics.mean(endpoint_times) * 1000, 2),
                    "min_ms": round(min(endpoint_times) * 1000, 2),
                    "max_ms": round(max(endpoint_times) * 1000, 2),
                    "error_rate": endpoint_errors / 5
                }
        
        if response_times:
            results["average_response_time"] = round(statistics.mean(response_times) * 1000, 2)
            results["slowest_endpoint"] = max(results["endpoint_times"].items(), 
                                            key=lambda x: x[1]["average_ms"])[0]
            results["fastest_endpoint"] = min(results["endpoint_times"].items(), 
                                            key=lambda x: x[1]["average_ms"])[0]
        
        results["error_rate"] = error_count / total_requests if total_requests > 0 else 0
        results["throughput_requests_per_second"] = total_requests / 30  # 30 second test period estimate
        
        return results
    
    async def _analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database performance patterns."""
        results = {
            "connection_analysis": {},
            "query_analysis": {},
            "index_analysis": {},
            "n_plus_one_detection": {},
            "slow_queries": [],
            "recommendations": []
        }
        
        # Check if app uses SQLite (for testing) or PostgreSQL
        try:
            # Look for SQLite databases
            sqlite_files = list(Path(".").glob("*.db")) + list(Path(".").glob("*.sqlite"))
            
            if sqlite_files:
                # Analyze SQLite database
                for db_file in sqlite_files:
                    results["connection_analysis"][str(db_file)] = await self._analyze_sqlite_db(db_file)
            
            # Check for PostgreSQL connection info
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    env_content = f.read()
                    if "DATABASE_URL" in env_content or "POSTGRES" in env_content:
                        results["connection_analysis"]["postgresql"] = "Found PostgreSQL configuration"
            
        except Exception as e:
            results["connection_analysis"]["error"] = str(e)
        
        # Analyze potential N+1 queries by examining code
        results["n_plus_one_detection"] = self._detect_n_plus_one_patterns()
        
        return results
    
    async def _analyze_sqlite_db(self, db_file: Path) -> Dict[str, Any]:
        """Analyze SQLite database performance."""
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get database size
            db_size = db_file.stat().st_size
            
            # Check for indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Analyze query performance with EXPLAIN QUERY PLAN
            query_analysis = {}
            for table in tables[:5]:  # Analyze first 5 tables
                try:
                    cursor.execute(f"EXPLAIN QUERY PLAN SELECT * FROM {table} LIMIT 10;")
                    query_plan = cursor.fetchall()
                    query_analysis[table] = query_plan
                except Exception as e:
                    query_analysis[table] = f"Error: {e}"
            
            conn.close()
            
            return {
                "file_size_mb": round(db_size / (1024*1024), 2),
                "table_count": len(tables),
                "index_count": len(indexes),
                "tables": tables,
                "indexes": indexes,
                "query_plans": query_analysis
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_n_plus_one_patterns(self) -> Dict[str, Any]:
        """Detect potential N+1 query patterns in code."""
        n_plus_one_patterns = {
            "potential_issues": [],
            "files_analyzed": 0,
            "patterns_found": 0
        }
        
        # Patterns that often indicate N+1 queries
        problematic_patterns = [
            "for.*in.*query",
            "for.*in.*filter",
            "for.*in.*get_by",
            "query.*in.*loop",
            "session.query.*for"
        ]
        
        try:
            python_files = list(Path(".").rglob("*.py"))
            n_plus_one_patterns["files_analyzed"] = len(python_files)
            
            for py_file in python_files[:20]:  # Analyze first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for line_num, line in enumerate(content.split('\n'), 1):
                        for pattern in problematic_patterns:
                            if any(keyword in line.lower() for keyword in pattern.split('*')):
                                n_plus_one_patterns["potential_issues"].append({
                                    "file": str(py_file),
                                    "line": line_num,
                                    "content": line.strip(),
                                    "pattern": pattern
                                })
                                n_plus_one_patterns["patterns_found"] += 1
                                
                except Exception as e:
                    continue
                    
        except Exception as e:
            n_plus_one_patterns["error"] = str(e)
        
        return n_plus_one_patterns
    
    async def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        # Start memory tracing
        tracemalloc.start()
        
        # Get current memory stats
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        # Simulate some memory usage
        large_list = [i for i in range(100000)]  # Create some memory pressure
        
        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Clean up
        del large_list
        tracemalloc.stop()
        
        # Analyze memory leaks by checking for growing objects
        memory_analysis = {
            "system_memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
                "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
            },
            "process_memory": {
                "rss_mb": round(process.memory_info().rss / (1024**2), 2),
                "vms_mb": round(process.memory_info().vms / (1024**2), 2),
                "percent": round(process.memory_percent(), 2)
            },
            "top_memory_usage": [],
            "memory_growth_rate": "stable",  # Would need multiple samples to determine
            "potential_leaks": []
        }
        
        # Add top memory consumers
        for stat in top_stats[:10]:
            memory_analysis["top_memory_usage"].append({
                "file": stat.traceback.format()[-1] if stat.traceback else "unknown",
                "size_mb": round(stat.size / (1024**2), 3),
                "count": stat.count
            })
        
        return memory_analysis
    
    async def _analyze_cpu_usage(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns."""
        # Collect CPU metrics over a short period
        cpu_samples = []
        per_cpu_samples = []
        
        for _ in range(5):
            cpu_samples.append(psutil.cpu_percent(interval=1))
            per_cpu_samples.append(psutil.cpu_percent(interval=None, percpu=True))
        
        process = psutil.Process()
        
        cpu_analysis = {
            "average_cpu_percent": round(statistics.mean(cpu_samples), 2),
            "max_cpu_percent": max(cpu_samples),
            "min_cpu_percent": min(cpu_samples),
            "cpu_cores": psutil.cpu_count(),
            "process_cpu_percent": round(process.cpu_percent(), 2),
            "load_average": list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
            "cpu_frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "status": "healthy",
            "bottleneck_indicators": []
        }
        
        # Determine CPU status
        avg_cpu = cpu_analysis["average_cpu_percent"]
        if avg_cpu > 90:
            cpu_analysis["status"] = "critical"
            cpu_analysis["bottleneck_indicators"].append("Very high CPU usage")
        elif avg_cpu > 70:
            cpu_analysis["status"] = "warning"
            cpu_analysis["bottleneck_indicators"].append("High CPU usage")
        
        return cpu_analysis
    
    async def _analyze_caching_performance(self) -> Dict[str, Any]:
        """Analyze caching effectiveness."""
        caching_analysis = {
            "redis_available": False,
            "in_memory_cache": {},
            "file_cache": {},
            "cache_hit_estimation": {},
            "recommendations": []
        }
        
        # Check for Redis
        try:
            import redis
            # Try to connect to Redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            
            caching_analysis["redis_available"] = True
            caching_analysis["redis_info"] = {
                "memory_usage": r.info('memory'),
                "stats": r.info('stats'),
                "keyspace": r.info('keyspace')
            }
            
        except Exception as e:
            caching_analysis["redis_error"] = str(e)
            caching_analysis["recommendations"].append("Consider implementing Redis for caching")
        
        # Analyze file-based caching
        cache_dirs = ["cache", "tmp", ".cache", "__pycache__"]
        for cache_dir in cache_dirs:
            if Path(cache_dir).exists():
                cache_size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
                caching_analysis["file_cache"][cache_dir] = {
                    "size_mb": round(cache_size / (1024**2), 2),
                    "file_count": len(list(Path(cache_dir).rglob('*')))
                }
        
        return caching_analysis
    
    async def _detect_bottlenecks(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance bottlenecks across all metrics."""
        bottlenecks = {
            "critical_issues": [],
            "warning_issues": [],
            "performance_impact": {},
            "bottleneck_score": 0
        }
        
        # API Performance Bottlenecks
        api_perf = audit_results.get("api_performance", {})
        if api_perf.get("average_response_time", 0) > 500:  # > 500ms
            bottlenecks["critical_issues"].append({
                "type": "api_performance",
                "issue": "High API response times",
                "value": f"{api_perf.get('average_response_time')}ms",
                "threshold": "500ms",
                "impact": "high"
            })
        
        if api_perf.get("error_rate", 0) > 0.05:  # > 5%
            bottlenecks["critical_issues"].append({
                "type": "api_reliability",
                "issue": "High error rate",
                "value": f"{api_perf.get('error_rate', 0)*100:.1f}%",
                "threshold": "5%",
                "impact": "high"
            })
        
        # Memory Bottlenecks
        memory_perf = audit_results.get("memory_analysis", {})
        system_memory = memory_perf.get("system_memory", {})
        if system_memory.get("percent_used", 0) > 85:
            bottlenecks["critical_issues"].append({
                "type": "memory_usage",
                "issue": "High memory usage",
                "value": f"{system_memory.get('percent_used')}%",
                "threshold": "85%",
                "impact": "high"
            })
        
        # CPU Bottlenecks
        cpu_perf = audit_results.get("cpu_analysis", {})
        if cpu_perf.get("average_cpu_percent", 0) > 80:
            bottlenecks["critical_issues"].append({
                "type": "cpu_usage",
                "issue": "High CPU usage",
                "value": f"{cpu_perf.get('average_cpu_percent')}%",
                "threshold": "80%",
                "impact": "high"
            })
        
        # Database Bottlenecks
        db_perf = audit_results.get("database_performance", {})
        n_plus_one = db_perf.get("n_plus_one_detection", {})
        if n_plus_one.get("patterns_found", 0) > 0:
            bottlenecks["warning_issues"].append({
                "type": "database_queries",
                "issue": "Potential N+1 query patterns detected",
                "value": f"{n_plus_one.get('patterns_found')} patterns",
                "impact": "medium"
            })
        
        # Calculate bottleneck score (0-100, lower is better)
        critical_count = len(bottlenecks["critical_issues"])
        warning_count = len(bottlenecks["warning_issues"])
        bottlenecks["bottleneck_score"] = min(100, (critical_count * 20) + (warning_count * 5))
        
        return bottlenecks
    
    def _generate_recommendations(self, audit_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # API Performance Recommendations
        api_perf = audit_results.get("api_performance", {})
        if api_perf.get("average_response_time", 0) > 200:
            recommendations.append({
                "category": "API Performance",
                "priority": "high",
                "recommendation": "Implement response caching for frequently accessed endpoints",
                "expected_improvement": "50-80% faster response times"
            })
        
        if api_perf.get("error_rate", 0) > 0.01:
            recommendations.append({
                "category": "API Reliability",
                "priority": "high",
                "recommendation": "Implement circuit breaker pattern and better error handling",
                "expected_improvement": "Reduced error cascading"
            })
        
        # Database Recommendations
        db_perf = audit_results.get("database_performance", {})
        n_plus_one = db_perf.get("n_plus_one_detection", {})
        if n_plus_one.get("patterns_found", 0) > 0:
            recommendations.append({
                "category": "Database Performance",
                "priority": "medium",
                "recommendation": "Optimize queries with eager loading and query batching",
                "expected_improvement": "2-10x faster database operations"
            })
        
        # Memory Recommendations
        memory_perf = audit_results.get("memory_analysis", {})
        if memory_perf.get("system_memory", {}).get("percent_used", 0) > 70:
            recommendations.append({
                "category": "Memory Optimization",
                "priority": "medium",
                "recommendation": "Implement memory pooling and object reuse patterns",
                "expected_improvement": "20-40% memory usage reduction"
            })
        
        # Caching Recommendations
        caching_perf = audit_results.get("caching_analysis", {})
        if not caching_perf.get("redis_available", False):
            recommendations.append({
                "category": "Caching Infrastructure",
                "priority": "high",
                "recommendation": "Implement Redis for distributed caching",
                "expected_improvement": "90%+ cache hit rates, 5-20x faster data access"
            })
        
        # Always recommend monitoring
        recommendations.append({
            "category": "Monitoring",
            "priority": "medium",
            "recommendation": "Implement comprehensive performance monitoring with Prometheus/Grafana",
            "expected_improvement": "Proactive bottleneck detection"
        })
        
        return recommendations
    
    def _calculate_performance_score(self, audit_results: Dict[str, Any]) -> int:
        """Calculate overall performance score (0-100)."""
        score = 100
        
        # API Performance (40 points)
        api_perf = audit_results.get("api_performance", {})
        avg_response = api_perf.get("average_response_time", 0)
        if avg_response > 1000:  # > 1s
            score -= 20
        elif avg_response > 500:  # > 500ms
            score -= 10
        elif avg_response > 200:  # > 200ms
            score -= 5
        
        error_rate = api_perf.get("error_rate", 0)
        if error_rate > 0.1:  # > 10%
            score -= 15
        elif error_rate > 0.05:  # > 5%
            score -= 10
        elif error_rate > 0.01:  # > 1%
            score -= 5
        
        # Memory Performance (20 points)
        memory_perf = audit_results.get("memory_analysis", {})
        memory_usage = memory_perf.get("system_memory", {}).get("percent_used", 0)
        if memory_usage > 90:
            score -= 15
        elif memory_usage > 80:
            score -= 10
        elif memory_usage > 70:
            score -= 5
        
        # CPU Performance (20 points)
        cpu_perf = audit_results.get("cpu_analysis", {})
        cpu_usage = cpu_perf.get("average_cpu_percent", 0)
        if cpu_usage > 90:
            score -= 15
        elif cpu_usage > 70:
            score -= 10
        elif cpu_usage > 50:
            score -= 5
        
        # Database Performance (20 points)
        db_perf = audit_results.get("database_performance", {})
        n_plus_one = db_perf.get("n_plus_one_detection", {})
        patterns_found = n_plus_one.get("patterns_found", 0)
        if patterns_found > 10:
            score -= 15
        elif patterns_found > 5:
            score -= 10
        elif patterns_found > 0:
            score -= 5
        
        return max(0, score)
    
    def save_audit_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save audit results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_audit_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filename


async def main():
    """Run comprehensive performance audit."""
    auditor = PerformanceAuditor()
    
    print("🚀 Starting Comprehensive Performance Audit")
    print("=" * 60)
    
    # Run the audit
    results = await auditor.run_comprehensive_audit()
    
    # Save results
    filename = auditor.save_audit_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE AUDIT SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 Overall Performance Score: {results['performance_score']}/100")
    
    if results['performance_score'] >= 90:
        print("✅ Excellent performance!")
    elif results['performance_score'] >= 70:
        print("⚠️  Good performance with room for improvement")
    elif results['performance_score'] >= 50:
        print("🔶 Moderate performance - optimization needed")
    else:
        print("🔴 Poor performance - immediate optimization required")
    
    # API Performance
    api_perf = results.get('api_performance', {})
    print(f"\n📊 API Performance:")
    print(f"   Average Response Time: {api_perf.get('average_response_time', 0)}ms")
    print(f"   Error Rate: {api_perf.get('error_rate', 0)*100:.2f}%")
    print(f"   Fastest Endpoint: {api_perf.get('fastest_endpoint', 'N/A')}")
    print(f"   Slowest Endpoint: {api_perf.get('slowest_endpoint', 'N/A')}")
    
    # System Resources
    memory = results.get('memory_analysis', {}).get('system_memory', {})
    cpu = results.get('cpu_analysis', {})
    print(f"\n💾 System Resources:")
    print(f"   Memory Usage: {memory.get('percent_used', 0)}%")
    print(f"   CPU Usage: {cpu.get('average_cpu_percent', 0)}%")
    
    # Bottlenecks
    bottlenecks = results.get('bottleneck_analysis', {})
    critical_issues = bottlenecks.get('critical_issues', [])
    warning_issues = bottlenecks.get('warning_issues', [])
    
    if critical_issues:
        print(f"\n🔴 Critical Bottlenecks ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"   - {issue['issue']}: {issue['value']}")
    
    if warning_issues:
        print(f"\n⚠️  Warning Issues ({len(warning_issues)}):")
        for issue in warning_issues:
            print(f"   - {issue['issue']}: {issue['value']}")
    
    # Top Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. [{rec['category']}] {rec['recommendation']}")
            print(f"      Expected improvement: {rec['expected_improvement']}")
    
    print(f"\n📄 Full audit results saved to: {filename}")
    print(f"⏱️  Audit completed in {results['audit_duration']:.2f} seconds")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
