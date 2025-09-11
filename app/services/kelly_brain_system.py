"""
Kelly Brain System - Main Integration Service

Orchestrates all Kelly components including personality, userbot, DM detection,
conversation management, and safety monitoring for a unified Kelly brain experience.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
from pyrogram import types

from app.core.redis import redis_manager
from app.services.kelly_personality_service import kelly_personality_service
from app.services.kelly_telegram_userbot import kelly_userbot
from app.services.kelly_dm_detector import kelly_dm_detector
from app.services.kelly_conversation_manager import kelly_conversation_manager
from app.services.kelly_safety_monitor import kelly_safety_monitor

logger = structlog.get_logger()

class KellyBrainSystem:
    """
    Main orchestration service for the Kelly brain Telegram messaging system.
    Integrates all components for seamless operation.
    """
    
    def __init__(self):
        # Core components
        self.personality_service = kelly_personality_service
        self.userbot = kelly_userbot
        self.dm_detector = kelly_dm_detector
        self.conversation_manager = kelly_conversation_manager
        self.safety_monitor = kelly_safety_monitor
        
        # System status
        self.is_initialized = False
        self.system_health = {
            "overall_status": "initializing",
            "components": {},
            "last_health_check": None
        }
        
    async def initialize(self):
        """Initialize the complete Kelly brain system"""
        try:
            logger.info("Initializing Kelly Brain System...")
            
            # Initialize all components
            await self._initialize_components()
            
            # Setup message processing pipeline
            await self._setup_message_pipeline()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            # Start system coordination
            asyncio.create_task(self._system_coordination_loop())
            
            self.is_initialized = True
            self.system_health["overall_status"] = "running"
            
            logger.info("Kelly Brain System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly Brain System: {e}")
            self.system_health["overall_status"] = "error"
            raise

    async def _initialize_components(self):
        """Initialize all Kelly components"""
        components = [
            ("personality_service", self.personality_service),
            ("userbot", self.userbot),
            ("dm_detector", self.dm_detector),
            ("conversation_manager", self.conversation_manager),
            ("safety_monitor", self.safety_monitor)
        ]
        
        for component_name, component in components:
            try:
                await component.initialize()
                self.system_health["components"][component_name] = "running"
                logger.info(f"Initialized {component_name}")
            except Exception as e:
                self.system_health["components"][component_name] = "error"
                logger.error(f"Failed to initialize {component_name}: {e}")
                raise

    async def _setup_message_pipeline(self):
        """Setup the message processing pipeline"""
        try:
            # Register Kelly brain as message processor in userbot
            await self._register_message_processor()
            
            logger.info("Message processing pipeline configured")
            
        except Exception as e:
            logger.error(f"Failed to setup message pipeline: {e}")
            raise

    async def _register_message_processor(self):
        """Register Kelly brain as the main message processor"""
        # This would integrate with the userbot's message handling
        # The userbot would call process_incoming_message for each message
        pass

    async def process_telegram_message(
        self, 
        account_id: str, 
        message: types.Message
    ) -> Dict[str, Any]:
        """
        Main message processing function - called by userbot for each incoming message
        """
        try:
            if not self.is_initialized:
                return {"error": "Kelly brain system not initialized"}
            
            # Step 1: Safety assessment
            safety_assessment = await self.safety_monitor.assess_conversation_safety(
                account_id=account_id,
                user_id=str(message.from_user.id),
                message_text=message.text or "",
                conversation_history=await self._get_conversation_history(account_id, message.from_user.id)
            )
            
            # Step 2: Check if conversation should continue
            if safety_assessment.overall_threat_level.value in ["high", "critical"]:
                # Handle threat automatically
                await self._handle_safety_threat(account_id, message, safety_assessment)
                return {
                    "processed": True,
                    "action": "threat_handled",
                    "threat_level": safety_assessment.overall_threat_level.value,
                    "safety_assessment": safety_assessment.__dict__
                }
            
            # Step 3: Process through conversation manager
            conversation_result = await self.conversation_manager.process_incoming_message(
                account_id, message
            )
            
            # Step 4: Send response if generated
            if conversation_result.get("responded", False):
                response_data = conversation_result.get("response", {})
                
                # Apply anti-detection delays and send
                await self._send_kelly_response(
                    account_id, 
                    message, 
                    response_data
                )
            
            # Step 5: Log system activity
            await self._log_system_activity(account_id, message, conversation_result, safety_assessment)
            
            return {
                "processed": True,
                "conversation_result": conversation_result,
                "safety_assessment": safety_assessment.__dict__,
                "system_status": "operational"
            }
            
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            return {
                "processed": False,
                "error": str(e),
                "system_status": "error"
            }

    async def _get_conversation_history(self, account_id: str, user_id: int) -> List[str]:
        """Get conversation history for safety assessment"""
        try:
            history = await self.userbot.get_conversation_history(account_id, str(user_id), limit=20)
            return [msg.get("original_message", "") for msg in history if msg.get("original_message")]
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def _handle_safety_threat(
        self, 
        account_id: str, 
        message: types.Message, 
        safety_assessment
    ):
        """Handle detected safety threats"""
        try:
            user_id = str(message.from_user.id)
            
            # Auto-block if recommended
            if any(action.value == "auto_block" for action in safety_assessment.recommended_actions):
                await self._auto_block_user(account_id, user_id, safety_assessment)
            
            # Send warning if recommended
            if any(action.value == "warning_issued" for action in safety_assessment.recommended_actions):
                await self._send_safety_warning(account_id, message)
            
            # Escalate if required
            if safety_assessment.escalation_required:
                await self._escalate_threat(account_id, user_id, safety_assessment)
            
        except Exception as e:
            logger.error(f"Error handling safety threat: {e}")

    async def _auto_block_user(self, account_id: str, user_id: str, safety_assessment):
        """Auto-block user for safety violations"""
        try:
            # Block through personality service
            await self.personality_service._auto_block_user(user_id, [
                flag.category for flag in safety_assessment.red_flags
            ])
            
            logger.warning(f"Auto-blocked user {user_id} on account {account_id}")
            
        except Exception as e:
            logger.error(f"Error auto-blocking user: {e}")

    async def _send_safety_warning(self, account_id: str, message: types.Message):
        """Send safety warning message"""
        try:
            warning_messages = [
                "I'm not comfortable with that direction. Let's talk about something else?",
                "That doesn't seem appropriate. How about we change the topic?",
                "I'd prefer to keep our conversation respectful."
            ]
            
            import random
            warning_text = random.choice(warning_messages)
            
            # Send through userbot
            client = self.userbot.clients.get(account_id)
            if client:
                await client.send_message(
                    chat_id=message.chat.id,
                    text=warning_text
                )
            
        except Exception as e:
            logger.error(f"Error sending safety warning: {e}")

    async def _escalate_threat(self, account_id: str, user_id: str, safety_assessment):
        """Escalate threat to human review"""
        try:
            escalation_data = {
                "account_id": account_id,
                "user_id": user_id,
                "threat_level": safety_assessment.overall_threat_level.value,
                "red_flags": [flag.category.value for flag in safety_assessment.red_flags],
                "escalated_at": datetime.now().isoformat(),
                "requires_attention": True
            }
            
            # Store escalation for admin review
            escalation_key = f"kelly:escalation:{account_id}_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await redis_manager.setex(escalation_key, 86400 * 7, json.dumps(escalation_data))
            
            # Add to priority queue
            await redis_manager.lpush("kelly:escalation_queue", escalation_key)
            
            logger.critical(f"Escalated threat from user {user_id} on account {account_id}")
            
        except Exception as e:
            logger.error(f"Error escalating threat: {e}")

    async def _send_kelly_response(
        self, 
        account_id: str, 
        original_message: types.Message, 
        response_data: Dict[str, Any]
    ):
        """Send Kelly's response with proper timing and anti-detection"""
        try:
            client = self.userbot.clients.get(account_id)
            if not client:
                logger.error(f"No client found for account {account_id}")
                return
            
            response_text = response_data.get("text", "")
            typing_delay = response_data.get("typing_delay", 5.0)
            
            # Apply typing simulation
            await self._simulate_typing_with_kelly_patterns(
                client, original_message.chat.id, typing_delay
            )
            
            # Send the response
            await client.send_message(
                chat_id=original_message.chat.id,
                text=response_text,
                reply_to_message_id=original_message.id if random.random() < 0.3 else None
            )
            
            logger.info(f"Kelly response sent from account {account_id}")
            
        except Exception as e:
            logger.error(f"Error sending Kelly response: {e}")

    async def _simulate_typing_with_kelly_patterns(
        self, 
        client, 
        chat_id: int, 
        duration: float
    ):
        """Simulate typing with Kelly's natural patterns"""
        try:
            import random
            
            # Break typing into natural chunks
            chunks = max(1, int(duration / 5))  # 5-second max chunks
            
            for i in range(chunks):
                # Start typing
                await client.send_chat_action(chat_id, "typing")
                
                # Type for a portion of the duration
                chunk_duration = duration / chunks
                variation = random.uniform(0.8, 1.2)
                actual_duration = chunk_duration * variation
                
                await asyncio.sleep(min(5.0, actual_duration))
                
                # Small pause between chunks (natural behavior)
                if i < chunks - 1:
                    await asyncio.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            logger.error(f"Error simulating typing: {e}")

    async def _log_system_activity(
        self, 
        account_id: str, 
        message: types.Message, 
        conversation_result: Dict[str, Any],
        safety_assessment
    ):
        """Log comprehensive system activity"""
        try:
            activity_data = {
                "timestamp": datetime.now().isoformat(),
                "account_id": account_id,
                "user_id": str(message.from_user.id),
                "message_length": len(message.text or ""),
                "conversation_processed": conversation_result.get("processed", False),
                "response_generated": conversation_result.get("responded", False),
                "threat_level": safety_assessment.overall_threat_level.value,
                "red_flags_count": len(safety_assessment.red_flags),
                "ai_features_used": conversation_result.get("response", {}).get("ai_features_used", []),
                "response_method": conversation_result.get("response", {}).get("method", "none")
            }
            
            # Store in daily activity log
            today = datetime.now().strftime("%Y-%m-%d")
            activity_key = f"kelly:activity_log:{account_id}:{today}"
            await redis_manager.lpush(activity_key, json.dumps(activity_data))
            await redis_manager.ltrim(activity_key, 0, 999)  # Keep last 1000 activities
            await redis_manager.expire(activity_key, 86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Error logging system activity: {e}")

    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)

    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "components": {},
                "metrics": {}
            }
            
            # Check component health
            components = [
                ("personality_service", self.personality_service),
                ("userbot", self.userbot),
                ("dm_detector", self.dm_detector),
                ("conversation_manager", self.conversation_manager),
                ("safety_monitor", self.safety_monitor)
            ]
            
            for component_name, component in components:
                try:
                    # Basic health check - ensure component is accessible
                    if hasattr(component, 'health_check'):
                        component_health = await component.health_check()
                    else:
                        component_health = "running"  # Assume healthy if no explicit check
                    
                    health_status["components"][component_name] = component_health
                    
                except Exception as e:
                    health_status["components"][component_name] = f"error: {str(e)}"
                    health_status["overall_status"] = "degraded"
            
            # Get system metrics
            health_status["metrics"] = await self._get_system_metrics()
            
            # Store health status
            self.system_health = health_status
            await redis_manager.setex("kelly:system_health", 300, json.dumps(health_status))
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            self.system_health["overall_status"] = "error"

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Get basic metrics
            total_accounts = len(self.userbot.account_configs)
            active_accounts = sum(1 for config in self.userbot.account_configs.values() if config.enabled)
            
            # Get today's activity counts
            today = datetime.now().strftime("%Y-%m-%d")
            total_messages_today = 0
            total_conversations_today = 0
            
            for account_id in self.userbot.account_configs.keys():
                activity_key = f"kelly:activity_log:{account_id}:{today}"
                activities = await redis_manager.llen(activity_key)
                total_messages_today += activities
                
                # Count unique conversations
                activity_data = await redis_manager.lrange(activity_key, 0, -1)
                unique_users = set()
                for activity in activity_data:
                    data = json.loads(activity)
                    unique_users.add(data.get("user_id"))
                total_conversations_today += len(unique_users)
            
            return {
                "total_accounts": total_accounts,
                "active_accounts": active_accounts,
                "messages_processed_today": total_messages_today,
                "conversations_today": total_conversations_today,
                "system_uptime": "calculated_uptime",  # Would calculate from initialization time
                "memory_usage": "memory_stats",  # Would get actual memory stats
                "redis_connection": "connected" if await redis_manager.ping() else "disconnected"
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}

    async def _system_coordination_loop(self):
        """System coordination and optimization"""
        while True:
            try:
                # Optimize system performance
                await self._optimize_system_performance()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Update AI models
                await self._update_ai_models()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in system coordination: {e}")
                await asyncio.sleep(3600)

    async def _optimize_system_performance(self):
        """Optimize system performance"""
        try:
            # Could implement:
            # - Memory cleanup
            # - Cache optimization
            # - Connection pool management
            # - Resource allocation adjustments
            pass
            
        except Exception as e:
            logger.error(f"Error optimizing system performance: {e}")

    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        try:
            # Clean up old activity logs (older than 7 days)
            cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Could implement cleanup of:
            # - Old conversation histories
            # - Expired threat assessments
            # - Old activity logs
            # - Cached AI analysis results
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    async def _update_ai_models(self):
        """Update AI models based on recent data"""
        try:
            # Could implement:
            # - Model retraining based on recent conversations
            # - Personality adaptation learning
            # - Safety pattern updates
            # - Response quality improvements
            pass
            
        except Exception as e:
            logger.error(f"Error updating AI models: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "initialized": self.is_initialized,
            "health": self.system_health,
            "components_status": {
                "personality_service": "running" if self.personality_service else "not_loaded",
                "userbot": "running" if self.userbot else "not_loaded",
                "dm_detector": "running" if self.dm_detector else "not_loaded",
                "conversation_manager": "running" if self.conversation_manager else "not_loaded",
                "safety_monitor": "running" if self.safety_monitor else "not_loaded"
            }
        }

    async def shutdown(self):
        """Gracefully shutdown the Kelly brain system"""
        try:
            logger.info("Shutting down Kelly Brain System...")
            
            # Stop all accounts
            await self.userbot.stop_all_accounts()
            
            # Save final state
            await self._save_final_state()
            
            self.is_initialized = False
            self.system_health["overall_status"] = "shutdown"
            
            logger.info("Kelly Brain System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Kelly brain shutdown: {e}")

    async def _save_final_state(self):
        """Save final system state before shutdown"""
        try:
            final_state = {
                "shutdown_timestamp": datetime.now().isoformat(),
                "final_health": self.system_health,
                "accounts_status": {
                    account_id: config.enabled 
                    for account_id, config in self.userbot.account_configs.items()
                }
            }
            
            await redis_manager.setex("kelly:final_state", 86400, json.dumps(final_state))
            
        except Exception as e:
            logger.error(f"Error saving final state: {e}")

# Global instance
kelly_brain_system = KellyBrainSystem()