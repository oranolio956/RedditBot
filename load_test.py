#!/usr/bin/env python3
"""
Load Testing Framework for Reddit/Telegram Bot

Implements comprehensive load testing scenarios to validate
performance under various conditions.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random


@dataclass
class LoadTestResult:
    """Load test result container."""
    scenario_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    errors: List[str]
    start_time: datetime
    end_time: datetime
    duration: float


class LoadTester:
    """Comprehensive load testing framework."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        
    async def run_single_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
        """Execute a single HTTP request and measure performance."""
        start_time = time.time()
        
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                await response.text()
                response_time = time.time() - start_time
                
                return {
                    'success': True,
                    'response_time': response_time,
                    'status_code': response.status,
                    'error': None
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'response_time': response_time,
                'status_code': 0,
                'error': str(e)
            }
    
    async def run_user_session(self, user_id: int, endpoints: List[str], requests_per_user: int) -> List[Dict[str, Any]]:
        """Simulate a single user session with multiple requests."""
        results = []
        
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for _ in range(requests_per_user):
                # Randomly select an endpoint
                endpoint = random.choice(endpoints)
                
                # Add some realistic delay between requests
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
                result = await self.run_single_request(session, endpoint)
                result['user_id'] = user_id
                result['endpoint'] = endpoint
                results.append(result)
        
        return results
    
    async def run_load_test_scenario(self, scenario: Dict[str, Any]) -> LoadTestResult:
        """Run a complete load test scenario."""
        print(f"\n🏁 Running scenario: {scenario['name']}")
        print(f"   Virtual users: {scenario['virtual_users']}")
        print(f"   Duration: {scenario['duration']}")
        print(f"   Endpoints: {scenario['endpoints']}")
        
        start_time = datetime.now()
        
        # Calculate requests per user based on duration
        duration_minutes = self._parse_duration(scenario['duration'])
        requests_per_user = max(1, int(duration_minutes * 10))  # ~10 requests per minute per user
        
        # Create tasks for all virtual users
        tasks = []
        for user_id in range(scenario['virtual_users']):
            task = self.run_user_session(
                user_id=user_id,
                endpoints=scenario['endpoints'],
                requests_per_user=requests_per_user
            )
            tasks.append(task)
        
        # Execute all user sessions concurrently
        print(f"   🚀 Starting {scenario['virtual_users']} virtual users...")
        
        try:
            # Add ramp-up period
            ramp_up_seconds = self._parse_duration(scenario.get('ramp_up', '30s'))
            batch_size = max(1, scenario['virtual_users'] // 10)  # Ramp up in 10 batches
            
            all_results = []
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for results in batch_results:
                    if isinstance(results, list):
                        all_results.extend(results)
                
                # Ramp-up delay
                if i + batch_size < len(tasks):
                    await asyncio.sleep(ramp_up_seconds / 10)
            
        except Exception as e:
            print(f"   ❌ Scenario failed: {e}")
            all_results = []
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Analyze results
        result = self._analyze_results(scenario['name'], all_results, start_time, end_time, duration)
        
        print(f"   ✅ Completed: {result.successful_requests}/{result.total_requests} requests")
        print(f"   📈 Avg response time: {result.average_response_time:.2f}ms")
        print(f"   📉 Throughput: {result.requests_per_second:.1f} req/s")
        
        return result
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to seconds."""
        if duration_str.endswith('s'):
            return float(duration_str[:-1])
        elif duration_str.endswith('m'):
            return float(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return float(duration_str[:-1]) * 3600
        else:
            return float(duration_str)
    
    def _analyze_results(self, scenario_name: str, results: List[Dict[str, Any]], start_time: datetime, end_time: datetime, duration: float) -> LoadTestResult:
        """Analyze load test results and generate metrics."""
        if not results:
            return LoadTestResult(
                scenario_name=scenario_name,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=1.0,
                errors=["No results collected"],
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
        
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        
        # Response time analysis
        response_times = [r['response_time'] * 1000 for r in results]  # Convert to ms
        successful_times = [r['response_time'] * 1000 for r in results if r['success']]
        
        if successful_times:
            avg_response_time = statistics.mean(successful_times)
            min_response_time = min(successful_times)
            max_response_time = max(successful_times)
            p95_response_time = self._calculate_percentile(successful_times, 95)
            p99_response_time = self._calculate_percentile(successful_times, 99)
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        
        # Throughput calculation
        requests_per_second = total_requests / duration if duration > 0 else 0
        
        # Error analysis
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        errors = [r['error'] for r in results if r['error']]
        unique_errors = list(set(errors))
        
        return LoadTestResult(
            scenario_name=scenario_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=unique_errors,
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def run_comprehensive_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load testing with multiple scenarios."""
        scenarios = [
            {
                'name': 'baseline_load',
                'description': 'Normal operation load test',
                'virtual_users': 10,
                'duration': '2m',
                'ramp_up': '30s',
                'endpoints': ['/', '/health', '/docs']
            },
            {
                'name': 'api_stress_test',
                'description': 'API endpoint stress test',
                'virtual_users': 25,
                'duration': '3m',
                'ramp_up': '45s',
                'endpoints': ['/health', '/health/detailed', '/']
            },
            {
                'name': 'spike_test',
                'description': 'Sudden traffic spike test',
                'virtual_users': 50,
                'duration': '1m',
                'ramp_up': '5s',
                'endpoints': ['/health', '/']
            }
        ]
        
        print("🎯 Starting Comprehensive Load Test")
        print("=" * 50)
        
        all_results = []
        
        for scenario in scenarios:
            try:
                result = await self.run_load_test_scenario(scenario)
                all_results.append(result)
                self.results.append(result)
                
                # Brief pause between scenarios
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"   ❌ Scenario {scenario['name']} failed: {e}")
                continue
        
        # Generate comprehensive report
        report = self._generate_load_test_report(all_results)
        
        return report
    
    def _generate_load_test_report(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Generate comprehensive load test report."""
        if not results:
            return {
                'status': 'failed',
                'error': 'No test results available',
                'timestamp': datetime.now().isoformat()
            }
        
        # Overall statistics
        total_requests = sum(r.total_requests for r in results)
        total_successful = sum(r.successful_requests for r in results)
        total_failed = sum(r.failed_requests for r in results)
        
        avg_response_times = [r.average_response_time for r in results if r.successful_requests > 0]
        overall_avg_response = statistics.mean(avg_response_times) if avg_response_times else 0
        
        max_throughput = max(r.requests_per_second for r in results)
        overall_error_rate = total_failed / total_requests if total_requests > 0 else 0
        
        # Performance assessment
        performance_grade = self._assess_performance(results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'scenarios_run': len(results),
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'failed_requests': total_failed,
                'overall_success_rate': (total_successful / total_requests * 100) if total_requests > 0 else 0,
                'overall_error_rate': overall_error_rate * 100,
                'average_response_time_ms': overall_avg_response,
                'max_throughput_rps': max_throughput
            },
            'scenario_results': [],
            'performance_assessment': performance_grade,
            'bottlenecks_identified': self._identify_bottlenecks(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Add detailed scenario results
        for result in results:
            scenario_data = {
                'name': result.scenario_name,
                'total_requests': result.total_requests,
                'success_rate': (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0,
                'average_response_time_ms': result.average_response_time,
                'p95_response_time_ms': result.p95_response_time,
                'p99_response_time_ms': result.p99_response_time,
                'throughput_rps': result.requests_per_second,
                'error_rate': result.error_rate * 100,
                'duration_seconds': result.duration,
                'unique_errors': len(result.errors)
            }
            report['scenario_results'].append(scenario_data)
        
        return report
    
    def _assess_performance(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Assess overall performance grade."""
        if not results:
            return {'grade': 'F', 'score': 0, 'status': 'failed'}
        
        score = 100
        
        # Response time assessment (40 points)
        avg_response_times = [r.average_response_time for r in results if r.successful_requests > 0]
        if avg_response_times:
            max_response_time = max(avg_response_times)
            if max_response_time > 2000:  # > 2s
                score -= 40
            elif max_response_time > 1000:  # > 1s
                score -= 25
            elif max_response_time > 500:  # > 500ms
                score -= 15
            elif max_response_time > 200:  # > 200ms
                score -= 5
        
        # Error rate assessment (30 points)
        max_error_rate = max(r.error_rate for r in results)
        if max_error_rate > 0.1:  # > 10%
            score -= 30
        elif max_error_rate > 0.05:  # > 5%
            score -= 20
        elif max_error_rate > 0.01:  # > 1%
            score -= 10
        
        # Throughput assessment (30 points)
        max_throughput = max(r.requests_per_second for r in results)
        if max_throughput < 10:
            score -= 30
        elif max_throughput < 50:
            score -= 20
        elif max_throughput < 100:
            score -= 10
        
        # Determine grade
        if score >= 90:
            grade = 'A'
            status = 'excellent'
        elif score >= 80:
            grade = 'B'
            status = 'good'
        elif score >= 70:
            grade = 'C'
            status = 'acceptable'
        elif score >= 60:
            grade = 'D'
            status = 'poor'
        else:
            grade = 'F'
            status = 'failing'
        
        return {
            'grade': grade,
            'score': max(0, score),
            'status': status,
            'max_response_time_ms': max(avg_response_times) if avg_response_times else 0,
            'max_error_rate_percent': max_error_rate * 100,
            'max_throughput_rps': max_throughput
        }
    
    def _identify_bottlenecks(self, results: List[LoadTestResult]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for result in results:
            if result.error_rate > 0.05:  # > 5% error rate
                bottlenecks.append(f"High error rate in {result.scenario_name}: {result.error_rate*100:.1f}%")
            
            if result.average_response_time > 1000:  # > 1s response time
                bottlenecks.append(f"Slow response times in {result.scenario_name}: {result.average_response_time:.0f}ms")
            
            if result.requests_per_second < 10:  # < 10 RPS
                bottlenecks.append(f"Low throughput in {result.scenario_name}: {result.requests_per_second:.1f} req/s")
        
        return bottlenecks
    
    def _generate_recommendations(self, results: List[LoadTestResult]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Check if application is running
        if any("Connection refused" in str(result.errors) for result in results):
            recommendations.append("Start the application server before running load tests")
        
        # Response time recommendations
        avg_response_times = [r.average_response_time for r in results if r.successful_requests > 0]
        if avg_response_times and max(avg_response_times) > 500:
            recommendations.append("Implement response caching to reduce response times")
            recommendations.append("Optimize database queries with indexing")
        
        # Error rate recommendations
        if any(r.error_rate > 0.01 for r in results):
            recommendations.append("Implement proper error handling and circuit breaker patterns")
            recommendations.append("Add health checks and graceful degradation")
        
        # Throughput recommendations
        if all(r.requests_per_second < 100 for r in results):
            recommendations.append("Consider horizontal scaling with load balancers")
            recommendations.append("Implement connection pooling and async processing")
        
        # General recommendations
        recommendations.extend([
            "Set up continuous performance monitoring",
            "Implement auto-scaling based on load metrics",
            "Add performance budgets to CI/CD pipeline"
        ])
        
        return recommendations
    
    def save_results(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save load test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename


async def main():
    """Run comprehensive load testing."""
    tester = LoadTester()
    
    print("💪 Load Testing Framework")
    print("=" * 30)
    
    # Run load tests
    report = await tester.run_comprehensive_load_test()
    
    # Save results
    filename = tester.save_results(report)
    
    # Display summary
    print("\n" + "=" * 50)
    print("📋 LOAD TEST SUMMARY")
    print("=" * 50)
    
    summary = report['test_summary']
    assessment = report['performance_assessment']
    
    print(f"\n🏆 Performance Grade: {assessment['grade']} ({assessment['score']}/100)")
    print(f"   Status: {assessment['status'].title()}")
    
    print(f"\n📈 Test Results:")
    print(f"   Scenarios: {summary['scenarios_run']}")
    print(f"   Total Requests: {summary['total_requests']}")
    print(f"   Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"   Average Response Time: {summary['average_response_time_ms']:.1f}ms")
    print(f"   Max Throughput: {summary['max_throughput_rps']:.1f} req/s")
    
    # Scenario breakdown
    print(f"\n📅 Scenario Results:")
    for scenario in report['scenario_results']:
        print(f"   {scenario['name']}:")
        print(f"     Success Rate: {scenario['success_rate']:.1f}%")
        print(f"     Avg Response: {scenario['average_response_time_ms']:.1f}ms")
        print(f"     Throughput: {scenario['throughput_rps']:.1f} req/s")
    
    # Bottlenecks
    bottlenecks = report['bottlenecks_identified']
    if bottlenecks:
        print(f"\n❌ Bottlenecks Identified ({len(bottlenecks)}):")
        for bottleneck in bottlenecks:
            print(f"   - {bottleneck}")
    
    # Recommendations
    recommendations = report['recommendations']
    print(f"\n💡 Recommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📄 Full results saved to: {filename}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
