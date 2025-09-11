#!/usr/bin/env python3
"""
Test script for Kelly AI monitoring system endpoints and WebSocket functionality.

This script tests all the new Kelly monitoring APIs and WebSocket endpoints
to ensure they are working correctly.
"""

import asyncio
import json
import logging
import aiohttp
import websockets
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KellyMonitoringTester:
    """Test suite for Kelly monitoring system"""
    
    def __init__(self, base_url: str = "http://localhost:8000", ws_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.ws_url = ws_url
        self.session = None
        self.test_results = []
    
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        logger.info("Test setup complete")
    
    async def cleanup(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
        logger.info("Test cleanup complete")
    
    async def test_health_check(self):
        """Test basic health check endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Health check passed: {data.get('status')}")
                    self.test_results.append(("health_check", True, "Health endpoint responding"))
                else:
                    self.test_results.append(("health_check", False, f"Status: {response.status}"))
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.test_results.append(("health_check", False, str(e)))
    
    async def test_monitoring_metrics_endpoint(self):
        """Test /kelly/monitoring/metrics endpoint"""
        try:
            headers = {"Authorization": "Bearer test-token"}  # Mock auth
            async with self.session.get(
                f"{self.base_url}/api/v1/kelly/monitoring/metrics",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = [
                        "conversations_active", "messages_processed", 
                        "ai_confidence_avg", "system_load"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        logger.info("Metrics endpoint test passed")
                        self.test_results.append(("metrics_endpoint", True, "All required fields present"))
                    else:
                        self.test_results.append(("metrics_endpoint", False, f"Missing fields: {missing_fields}"))
                else:
                    self.test_results.append(("metrics_endpoint", False, f"Status: {response.status}"))
                    
        except Exception as e:
            logger.error(f"Metrics endpoint test failed: {e}")
            self.test_results.append(("metrics_endpoint", False, str(e)))
    
    async def test_activity_feed_endpoint(self):
        """Test /kelly/monitoring/activity/feed endpoint"""
        try:
            headers = {"Authorization": "Bearer test-token"}
            async with self.session.get(
                f"{self.base_url}/api/v1/kelly/monitoring/activity/feed",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ["events", "total_count", "unread_count"]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        logger.info("Activity feed endpoint test passed")
                        self.test_results.append(("activity_feed", True, "All required fields present"))
                    else:
                        self.test_results.append(("activity_feed", False, f"Missing fields: {missing_fields}"))
                else:
                    self.test_results.append(("activity_feed", False, f"Status: {response.status}"))
                    
        except Exception as e:
            logger.error(f"Activity feed test failed: {e}")
            self.test_results.append(("activity_feed", False, str(e)))
    
    async def test_alerts_endpoint(self):
        """Test /kelly/alerts/active endpoint"""
        try:
            headers = {"Authorization": "Bearer test-token"}
            async with self.session.get(
                f"{self.base_url}/api/v1/kelly/alerts/active",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        logger.info(f"Alerts endpoint test passed - {len(data)} alerts found")
                        self.test_results.append(("alerts_endpoint", True, f"Found {len(data)} alerts"))
                    else:
                        self.test_results.append(("alerts_endpoint", False, "Expected list of alerts"))
                else:
                    self.test_results.append(("alerts_endpoint", False, f"Status: {response.status}"))
                    
        except Exception as e:
            logger.error(f"Alerts endpoint test failed: {e}")
            self.test_results.append(("alerts_endpoint", False, str(e)))
    
    async def test_intervention_status(self):
        """Test intervention status endpoint"""
        try:
            headers = {"Authorization": "Bearer test-token"}
            test_conversation_id = "test_account_test_user"
            
            async with self.session.get(
                f"{self.base_url}/api/v1/kelly/intervention/status/{test_conversation_id}",
                headers=headers
            ) as response:
                if response.status in [200, 404]:  # 404 is acceptable for non-existent conversation
                    if response.status == 200:
                        data = await response.json()
                        required_fields = ["conversation_id", "status", "can_release"]
                        
                        missing_fields = [field for field in required_fields if field not in data]
                        
                        if not missing_fields:
                            logger.info("Intervention status test passed")
                            self.test_results.append(("intervention_status", True, "Status endpoint working"))
                        else:
                            self.test_results.append(("intervention_status", False, f"Missing fields: {missing_fields}"))
                    else:
                        logger.info("Intervention status test passed (conversation not found)")
                        self.test_results.append(("intervention_status", True, "Correctly handles non-existent conversation"))
                else:
                    self.test_results.append(("intervention_status", False, f"Status: {response.status}"))
                    
        except Exception as e:
            logger.error(f"Intervention status test failed: {e}")
            self.test_results.append(("intervention_status", False, str(e)))
    
    async def test_emergency_status(self):
        """Test emergency status endpoint"""
        try:
            headers = {"Authorization": "Bearer test-token"}
            async with self.session.get(
                f"{self.base_url}/api/v1/kelly/emergency/status",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = [
                        "emergency_mode_active", "system_wide_stop_active",
                        "active_emergency_actions"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        logger.info("Emergency status test passed")
                        self.test_results.append(("emergency_status", True, "Status endpoint working"))
                    else:
                        self.test_results.append(("emergency_status", False, f"Missing fields: {missing_fields}"))
                else:
                    self.test_results.append(("emergency_status", False, f"Status: {response.status}"))
                    
        except Exception as e:
            logger.error(f"Emergency status test failed: {e}")
            self.test_results.append(("emergency_status", False, str(e)))
    
    async def test_websocket_monitoring(self):
        """Test WebSocket monitoring connection"""
        try:
            ws_uri = f"{self.ws_url}/ws/kelly/monitoring"
            
            async with websockets.connect(ws_uri) as websocket:
                # Test connection
                logger.info("WebSocket connection established")
                
                # Send a ping message
                await websocket.send(json.dumps({
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                message = json.loads(response)
                
                if message.get("type") == "pong":
                    logger.info("WebSocket ping/pong test passed")
                    self.test_results.append(("websocket_monitoring", True, "Ping/pong successful"))
                else:
                    logger.info(f"Received message: {message}")
                    self.test_results.append(("websocket_monitoring", True, "Connection established"))
                
        except asyncio.TimeoutError:
            logger.warning("WebSocket test timed out (this may be expected if endpoint doesn't exist yet)")
            self.test_results.append(("websocket_monitoring", False, "Connection timeout"))
        except Exception as e:
            logger.error(f"WebSocket test failed: {e}")
            self.test_results.append(("websocket_monitoring", False, str(e)))
    
    async def test_websocket_alerts(self):
        """Test WebSocket alerts connection"""
        try:
            ws_uri = f"{self.ws_url}/ws/kelly/alerts"
            
            async with websockets.connect(ws_uri) as websocket:
                logger.info("Alerts WebSocket connection established")
                
                # Request active alerts
                await websocket.send(json.dumps({
                    "type": "get_active_alerts",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                message = json.loads(response)
                
                if message.get("type") == "active_alerts":
                    logger.info("Alerts WebSocket test passed")
                    self.test_results.append(("websocket_alerts", True, "Alerts retrieval successful"))
                else:
                    logger.info(f"Received message: {message}")
                    self.test_results.append(("websocket_alerts", True, "Connection established"))
                
        except asyncio.TimeoutError:
            logger.warning("Alerts WebSocket test timed out")
            self.test_results.append(("websocket_alerts", False, "Connection timeout"))
        except Exception as e:
            logger.error(f"Alerts WebSocket test failed: {e}")
            self.test_results.append(("websocket_alerts", False, str(e)))
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("Starting Kelly monitoring system tests...")
        
        await self.setup()
        
        tests = [
            self.test_health_check,
            self.test_monitoring_metrics_endpoint,
            self.test_activity_feed_endpoint,
            self.test_alerts_endpoint,
            self.test_intervention_status,
            self.test_emergency_status,
            self.test_websocket_monitoring,
            self.test_websocket_alerts
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {e}")
                self.test_results.append((test.__name__, False, f"Exception: {e}"))
        
        await self.cleanup()
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print test results"""
        print("\n" + "="*80)
        print("KELLY MONITORING SYSTEM TEST RESULTS")
        print("="*80)
        
        passed = 0
        failed = 0
        
        for test_name, success, message in self.test_results:
            status = "PASS" if success else "FAIL"
            print(f"{test_name.ljust(25)}: {status.ljust(6)} - {message}")
            
            if success:
                passed += 1
            else:
                failed += 1
        
        print("-"*80)
        print(f"Total tests: {passed + failed}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(passed / (passed + failed) * 100):.1f}%" if (passed + failed) > 0 else "No tests run")
        print("="*80)

async def main():
    """Main test function"""
    # Check if custom URLs are provided
    import sys
    
    base_url = "http://localhost:8000"
    ws_url = "ws://localhost:8000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    if len(sys.argv) > 2:
        ws_url = sys.argv[2]
    
    print(f"Testing Kelly monitoring system at {base_url}")
    print(f"WebSocket testing at {ws_url}")
    
    tester = KellyMonitoringTester(base_url, ws_url)
    await tester.run_all_tests()

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import aiohttp
        import websockets
    except ImportError:
        print("Missing required packages. Install with:")
        print("pip install aiohttp websockets")
        exit(1)
    
    asyncio.run(main())