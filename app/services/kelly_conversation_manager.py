"""
Kelly Conversation Manager

Advanced conversation management system integrating all revolutionary AI features
for natural, intelligent, and safe Kelly-based conversations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from pyrogram import types

from app.core.redis import redis_manager
from app.services.kelly_personality_service import kelly_personality_service, KellyPersonalityConfig, ConversationStage
from app.services.kelly_telegram_userbot import kelly_userbot
from app.services.kelly_dm_detector import kelly_dm_detector, DMAnalysis, MessagePriority

# Revolutionary AI Features Integration
from app.services.consciousness_mirror import ConsciousnessMirror
from app.services.memory_palace_service import MemoryPalaceService
from app.services.emotional_intelligence_service import EmotionalIntelligenceService
from app.services.temporal_archaeology import TemporalArchaeology
from app.services.digital_telepathy_engine import DigitalTelepathyEngine
from app.services.quantum_consciousness_service import QuantumConsciousnessService
from app.services.synesthesia_engine import SynesthesiaEngine
from app.services.neural_dreams_service import NeuralDreamsService

logger = structlog.get_logger()

class ConversationStatus(Enum):
    """Status of conversations"""
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ARCHIVED = "archived"

@dataclass
class ConversationSession:
    """Active conversation session data"""
    account_id: str
    user_id: str
    conversation_id: str
    status: ConversationStatus
    stage: ConversationStage
    last_message_time: datetime
    message_count: int
    kelly_config: KellyPersonalityConfig
    ai_insights: Dict[str, Any]
    safety_metrics: Dict[str, float]
    engagement_metrics: Dict[str, float]

class KellyConversationManager:
    """Advanced conversation management with full AI integration"""
    
    def __init__(self):
        # Core services
        self.personality_service = kelly_personality_service
        self.userbot = kelly_userbot
        self.dm_detector = kelly_dm_detector
        
        # Revolutionary AI Features
        self.consciousness_mirror = ConsciousnessMirror()
        self.memory_palace = MemoryPalaceService()
        self.emotional_intelligence = EmotionalIntelligenceService()
        self.temporal_archaeology = TemporalArchaeology()
        self.digital_telepathy = DigitalTelepathyEngine()
        self.quantum_consciousness = QuantumConsciousnessService()
        self.synesthesia_engine = SynesthesiaEngine()
        self.neural_dreams = NeuralDreamsService()
        
        # Active conversation sessions
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Conversation queues by priority
        self.message_queues = {
            MessagePriority.URGENT: [],
            MessagePriority.HIGH: [],
            MessagePriority.NORMAL: [],
            MessagePriority.LOW: []
        }
        
        # AI feature settings
        self.ai_features_enabled = {
            "consciousness_mirroring": True,
            "memory_palace": True,
            "emotional_intelligence": True,
            "temporal_archaeology": True,
            "digital_telepathy": True,
            "quantum_consciousness": True,
            "synesthesia": True,
            "neural_dreams": True
        }
        
    async def initialize(self):
        """Initialize the conversation management system"""
        try:
            # Initialize all AI services
            await self.consciousness_mirror.initialize()
            await self.memory_palace.initialize()
            await self.emotional_intelligence.initialize()
            await self.temporal_archaeology.initialize()
            await self.digital_telepathy.initialize()
            await self.quantum_consciousness.initialize()
            await self.synesthesia_engine.initialize()
            await self.neural_dreams.initialize()
            
            # Initialize core services
            await self.personality_service.initialize()
            await self.userbot.initialize()
            await self.dm_detector.initialize()
            
            # Load active conversation sessions
            await self._load_active_sessions()
            
            # Start conversation processing loop
            asyncio.create_task(self._conversation_processing_loop())
            
            logger.info("Kelly conversation manager initialized with full AI integration")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly conversation manager: {e}")
            raise

    async def _load_active_sessions(self):
        """Load active conversation sessions from Redis"""
        try:
            keys = await redis_manager.scan_iter(match="kelly:session:*")
            async for key in keys:
                data = await redis_manager.get(key)
                if data:
                    session_data = json.loads(data)
                    session_id = key.split(":")[-1]
                    
                    # Reconstruct session object
                    session = ConversationSession(
                        account_id=session_data["account_id"],
                        user_id=session_data["user_id"],
                        conversation_id=session_data["conversation_id"],
                        status=ConversationStatus(session_data["status"]),
                        stage=ConversationStage(session_data["stage"]),
                        last_message_time=datetime.fromisoformat(session_data["last_message_time"]),
                        message_count=session_data["message_count"],
                        kelly_config=KellyPersonalityConfig(**session_data["kelly_config"]),
                        ai_insights=session_data.get("ai_insights", {}),
                        safety_metrics=session_data.get("safety_metrics", {}),
                        engagement_metrics=session_data.get("engagement_metrics", {})
                    )
                    
                    self.active_sessions[session_id] = session
                    
        except Exception as e:
            logger.error(f"Error loading active sessions: {e}")

    async def process_incoming_message(
        self, 
        account_id: str, 
        message: types.Message
    ) -> Dict[str, Any]:
        """Process incoming message with full AI analysis and response generation"""
        try:
            # Step 1: DM Detection and Analysis
            dm_analysis = await self.dm_detector.analyze_dm(account_id, message)
            
            # Step 2: Revolutionary AI Analysis
            ai_insights = await self._comprehensive_ai_analysis(message, dm_analysis)
            
            # Step 3: Decision Making with Quantum Consciousness
            response_decision = await self._quantum_decision_making(
                account_id, message, dm_analysis, ai_insights
            )
            
            # Step 4: Generate Kelly Response if appropriate
            if response_decision["should_respond"]:
                response_data = await self._generate_kelly_response(
                    account_id, message, dm_analysis, ai_insights
                )
                
                # Step 5: Memory Palace Storage
                await self._store_conversation_memory(
                    account_id, message, dm_analysis, ai_insights, response_data
                )
                
                return {
                    "processed": True,
                    "responded": True,
                    "analysis": dm_analysis.__dict__,
                    "ai_insights": ai_insights,
                    "response": response_data,
                    "decision": response_decision
                }
            else:
                return {
                    "processed": True,
                    "responded": False,
                    "analysis": dm_analysis.__dict__,
                    "ai_insights": ai_insights,
                    "decision": response_decision
                }
                
        except Exception as e:
            logger.error(f"Error processing incoming message: {e}")
            return {"processed": False, "error": str(e)}

    async def _comprehensive_ai_analysis(
        self, 
        message: types.Message, 
        dm_analysis: DMAnalysis
    ) -> Dict[str, Any]:
        """Perform comprehensive AI analysis using all revolutionary features"""
        ai_insights = {}
        
        try:
            user_id = str(message.from_user.id)
            text = message.text or ""
            
            # Consciousness Mirroring - Deep personality reflection
            if self.ai_features_enabled["consciousness_mirroring"]:
                consciousness_data = await self.consciousness_mirror.analyze_personality(user_id, text)
                ai_insights["consciousness_mirror"] = consciousness_data
                
                # Advanced personality mirroring for Kelly's response style
                mirrored_traits = await self.consciousness_mirror.mirror_personality_traits(
                    consciousness_data, dm_analysis.personality_traits
                )
                ai_insights["personality_mirroring"] = mirrored_traits
            
            # Emotional Intelligence - Deep emotional understanding
            if self.ai_features_enabled["emotional_intelligence"]:
                emotional_state = await self.emotional_intelligence.analyze_emotional_state(
                    text, {"user_id": user_id, "conversation_context": dm_analysis.__dict__}
                )
                ai_insights["emotional_intelligence"] = emotional_state
                
                # Emotional response suggestions
                emotional_response = await self.emotional_intelligence.suggest_emotional_response(
                    emotional_state, dm_analysis.suggested_response_tone
                )
                ai_insights["emotional_response_strategy"] = emotional_response
            
            # Temporal Archaeology - Pattern analysis across time
            if self.ai_features_enabled["temporal_archaeology"]:
                temporal_patterns = await self.temporal_archaeology.analyze_conversation_patterns(
                    user_id, [text]
                )
                ai_insights["temporal_archaeology"] = temporal_patterns
                
                # Predict conversation trajectory
                trajectory = await self.temporal_archaeology.predict_conversation_trajectory(
                    user_id, temporal_patterns
                )
                ai_insights["conversation_trajectory"] = trajectory
            
            # Digital Telepathy - Predictive response optimization
            if self.ai_features_enabled["digital_telepathy"]:
                telepathy_insights = await self.digital_telepathy.predict_optimal_response(
                    text, dm_analysis.personality_traits
                )
                ai_insights["digital_telepathy"] = telepathy_insights
                
                # Mind reading simulation for better understanding
                mental_state = await self.digital_telepathy.simulate_mental_state(
                    user_id, text, dm_analysis.emotional_state
                )
                ai_insights["mental_state_simulation"] = mental_state
            
            # Synesthesia Engine - Multi-sensory conversation understanding
            if self.ai_features_enabled["synesthesia"]:
                synesthetic_analysis = await self.synesthesia_engine.analyze_text_synesthesia(
                    text, {"emotional_context": dm_analysis.emotional_state}
                )
                ai_insights["synesthesia"] = synesthetic_analysis
                
                # Color-emotion mapping for response tone
                emotional_colors = await self.synesthesia_engine.map_emotions_to_colors(
                    dm_analysis.emotional_state
                )
                ai_insights["emotional_color_mapping"] = emotional_colors
            
            # Neural Dreams - Subconscious pattern recognition
            if self.ai_features_enabled["neural_dreams"]:
                dream_analysis = await self.neural_dreams.analyze_conversation_dreams(
                    user_id, text, dm_analysis.conversation_intent
                )
                ai_insights["neural_dreams"] = dream_analysis
                
                # Dream-inspired response creativity
                creative_suggestions = await self.neural_dreams.generate_creative_responses(
                    text, dream_analysis
                )
                ai_insights["dream_creativity"] = creative_suggestions
            
            # Memory Palace - Contextual memory integration
            if self.ai_features_enabled["memory_palace"]:
                memory_context = await self.memory_palace.retrieve_conversation_context(
                    user_id, text, limit=10
                )
                ai_insights["memory_palace"] = memory_context
                
                # Contextual relationship building
                relationship_insights = await self.memory_palace.analyze_relationship_development(
                    user_id, memory_context
                )
                ai_insights["relationship_development"] = relationship_insights
            
            return ai_insights
            
        except Exception as e:
            logger.error(f"Error in comprehensive AI analysis: {e}")
            return ai_insights

    async def _quantum_decision_making(
        self,
        account_id: str,
        message: types.Message,
        dm_analysis: DMAnalysis,
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use Quantum Consciousness for advanced decision making"""
        try:
            if not self.ai_features_enabled["quantum_consciousness"]:
                # Fallback to basic decision
                should_respond, reason = await self.dm_detector.should_respond_to_message(
                    account_id, dm_analysis
                )
                return {"should_respond": should_respond, "reason": reason, "method": "basic"}
            
            # Quantum decision context
            decision_context = {
                "message_analysis": dm_analysis.__dict__,
                "ai_insights": ai_insights,
                "account_id": account_id,
                "user_id": str(message.from_user.id),
                "conversation_stage": dm_analysis.priority.value,
                "safety_score": dm_analysis.safety_score,
                "engagement_potential": dm_analysis.engagement_potential
            }
            
            # Quantum consciousness processing
            quantum_decision = await self.quantum_consciousness.process_decision_context(
                decision_context
            )
            
            # Enhanced decision with quantum insights
            quantum_response_probability = quantum_decision.get("response_probability", 0.5)
            quantum_safety_assessment = quantum_decision.get("safety_assessment", 0.8)
            quantum_engagement_prediction = quantum_decision.get("engagement_prediction", 0.5)
            
            # Multi-dimensional decision matrix
            decision_score = (
                quantum_response_probability * 0.4 +
                quantum_safety_assessment * 0.3 +
                quantum_engagement_prediction * 0.2 +
                dm_analysis.engagement_potential * 0.1
            )
            
            should_respond = decision_score > 0.6 and dm_analysis.safety_score > 0.3
            
            return {
                "should_respond": should_respond,
                "reason": f"Quantum decision score: {decision_score:.2f}",
                "method": "quantum_consciousness",
                "quantum_insights": quantum_decision,
                "decision_score": decision_score
            }
            
        except Exception as e:
            logger.error(f"Error in quantum decision making: {e}")
            # Fallback to basic decision
            should_respond, reason = await self.dm_detector.should_respond_to_message(
                account_id, dm_analysis
            )
            return {"should_respond": should_respond, "reason": reason, "method": "fallback"}

    async def _generate_kelly_response(
        self,
        account_id: str,
        message: types.Message,
        dm_analysis: DMAnalysis,
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Kelly response using integrated AI features"""
        try:
            user_id = str(message.from_user.id)
            conversation_id = f"{account_id}_{user_id}"
            
            # Get Kelly configuration for this account
            kelly_config = await self.personality_service.get_kelly_config(account_id)
            
            # Enhanced response generation with AI insights
            if ai_insights and self.ai_features_enabled["consciousness_mirroring"]:
                # Use consciousness mirroring for personality-matched response
                mirrored_response = await self._generate_consciousness_mirrored_response(
                    user_id, message.text, ai_insights, kelly_config
                )
                
                if mirrored_response:
                    # Apply additional AI enhancements
                    enhanced_response = await self._enhance_response_with_ai(
                        mirrored_response, ai_insights, dm_analysis
                    )
                    
                    # Calculate natural typing delay
                    typing_delay = await self._calculate_ai_enhanced_typing_delay(
                        enhanced_response, ai_insights, kelly_config
                    )
                    
                    return {
                        "text": enhanced_response,
                        "typing_delay": typing_delay,
                        "method": "ai_enhanced",
                        "ai_features_used": list(ai_insights.keys()),
                        "enhancement_details": await self._get_enhancement_details(ai_insights)
                    }
            
            # Fallback to standard Kelly response
            response, metadata = await self.personality_service.generate_kelly_response(
                user_id, conversation_id, message.text, kelly_config
            )
            
            return {
                "text": response,
                "typing_delay": metadata.get("typing_delay", 5.0),
                "method": "standard_kelly",
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error generating Kelly response: {e}")
            return {
                "text": "I'm having trouble responding right now. Can you try again? 😊",
                "typing_delay": 3.0,
                "method": "error_fallback",
                "error": str(e)
            }

    async def _generate_consciousness_mirrored_response(
        self,
        user_id: str,
        message_text: str,
        ai_insights: Dict[str, Any],
        kelly_config: KellyPersonalityConfig
    ) -> Optional[str]:
        """Generate response using consciousness mirroring"""
        try:
            consciousness_data = ai_insights.get("consciousness_mirror", {})
            personality_mirroring = ai_insights.get("personality_mirroring", {})
            
            if not consciousness_data:
                return None
            
            # Generate personality-matched response
            mirrored_response = await self.consciousness_mirror.generate_personality_response(
                consciousness_data, ai_insights.get("emotional_intelligence", {})
            )
            
            # Apply Kelly's personality traits to the mirrored response
            kelly_adjusted_response = await self._apply_kelly_personality_to_response(
                mirrored_response, kelly_config, personality_mirroring
            )
            
            return kelly_adjusted_response
            
        except Exception as e:
            logger.error(f"Error generating consciousness mirrored response: {e}")
            return None

    async def _apply_kelly_personality_to_response(
        self,
        response: str,
        kelly_config: KellyPersonalityConfig,
        personality_mirroring: Dict[str, Any]
    ) -> str:
        """Apply Kelly's personality traits to a generated response"""
        try:
            # Adjust warmth level
            if kelly_config.warmth_level > 0.7:
                if not any(emoji in response for emoji in ["😊", "💕", "🌟", "💫", "😘"]):
                    response += " 😊"
            
            # Adjust playfulness
            if kelly_config.playfulness > 0.6:
                playful_additions = ["hehe", "lol", "aww"]
                if len(response.split()) > 5 and "?" not in response:
                    response += f" {playful_additions[hash(response) % len(playful_additions)]}"
            
            # Apply mirroring adjustments
            if personality_mirroring:
                energy_level = personality_mirroring.get("energy_level", 0.5)
                if energy_level > 0.7:
                    response = response.replace(".", "!")
                elif energy_level < 0.3:
                    response = response.replace("!", ".")
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying Kelly personality to response: {e}")
            return response

    async def _enhance_response_with_ai(
        self,
        response: str,
        ai_insights: Dict[str, Any],
        dm_analysis: DMAnalysis
    ) -> str:
        """Enhance response using various AI features"""
        try:
            enhanced_response = response
            
            # Digital Telepathy enhancement
            if "digital_telepathy" in ai_insights:
                telepathy_insights = ai_insights["digital_telepathy"]
                if telepathy_insights.get("response_optimization"):
                    enhanced_response = await self.digital_telepathy.optimize_response(
                        enhanced_response, str(dm_analysis.user_id)
                    )
            
            # Emotional Intelligence enhancement
            if "emotional_response_strategy" in ai_insights:
                emotional_strategy = ai_insights["emotional_response_strategy"]
                enhanced_response = await self._apply_emotional_strategy(
                    enhanced_response, emotional_strategy
                )
            
            # Synesthesia enhancement for multi-sensory appeal
            if "synesthesia" in ai_insights:
                synesthetic_data = ai_insights["synesthesia"]
                enhanced_response = await self._apply_synesthetic_enhancement(
                    enhanced_response, synesthetic_data
                )
            
            # Neural Dreams creative enhancement
            if "dream_creativity" in ai_insights:
                dream_suggestions = ai_insights["dream_creativity"]
                enhanced_response = await self._apply_creative_enhancement(
                    enhanced_response, dream_suggestions
                )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response with AI: {e}")
            return response

    async def _apply_emotional_strategy(self, response: str, emotional_strategy: Dict) -> str:
        """Apply emotional response strategy"""
        try:
            strategy_type = emotional_strategy.get("strategy_type", "neutral")
            
            if strategy_type == "empathetic":
                if "understand" not in response.lower():
                    response = "I understand... " + response
            elif strategy_type == "enthusiastic":
                if "!" not in response:
                    response = response.replace(".", "!")
            elif strategy_type == "supportive":
                if not any(word in response.lower() for word in ["here", "support", "you"]):
                    response += " I'm here for you 💕"
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying emotional strategy: {e}")
            return response

    async def _apply_synesthetic_enhancement(self, response: str, synesthetic_data: Dict) -> str:
        """Apply synesthetic enhancements to response"""
        try:
            # Color-emotion mapping for emoji selection
            emotional_colors = synesthetic_data.get("emotional_colors", {})
            
            if "warm" in emotional_colors:
                warm_emojis = ["🌟", "☀️", "🔥", "💛"]
                if not any(emoji in response for emoji in warm_emojis):
                    response += f" {warm_emojis[hash(response) % len(warm_emojis)]}"
            elif "cool" in emotional_colors:
                cool_emojis = ["💙", "🌊", "❄️", "🌙"]
                if not any(emoji in response for emoji in cool_emojis):
                    response += f" {cool_emojis[hash(response) % len(cool_emojis)]}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying synesthetic enhancement: {e}")
            return response

    async def _apply_creative_enhancement(self, response: str, dream_suggestions: Dict) -> str:
        """Apply creative enhancements from neural dreams"""
        try:
            creative_elements = dream_suggestions.get("creative_elements", [])
            
            for element in creative_elements[:1]:  # Apply one creative element
                if element.get("type") == "metaphor" and len(response.split()) > 5:
                    metaphor = element.get("content", "")
                    if metaphor and len(metaphor) < 30:
                        response += f" {metaphor}"
                elif element.get("type") == "wordplay":
                    # Add subtle wordplay if appropriate
                    wordplay = element.get("content", "")
                    if wordplay and len(wordplay) < 20:
                        response = response.replace(".", f" {wordplay}.")
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying creative enhancement: {e}")
            return response

    async def _calculate_ai_enhanced_typing_delay(
        self,
        response: str,
        ai_insights: Dict[str, Any],
        kelly_config: KellyPersonalityConfig
    ) -> float:
        """Calculate typing delay enhanced with AI insights"""
        try:
            # Base calculation from Kelly config
            words = len(response.split())
            base_time = words / kelly_config.typing_speed_base * 60
            
            # AI enhancements
            emotional_state = ai_insights.get("emotional_intelligence", {})
            if emotional_state:
                excitement = emotional_state.get("excitement", 0.5)
                if excitement > 0.7:
                    base_time *= 0.8  # Type faster when excited
                elif excitement < 0.3:
                    base_time *= 1.3  # Type slower when calm
            
            # Consciousness mirroring adjustment
            consciousness_data = ai_insights.get("consciousness_mirror", {})
            if consciousness_data:
                energy_level = consciousness_data.get("energy_level", 0.5)
                base_time *= (2 - energy_level)  # Higher energy = faster typing
            
            # Add natural variation
            import random
            variation = random.uniform(0.8, 1.2)
            total_delay = base_time * variation
            
            # Ensure within reasonable bounds
            return max(2.0, min(300.0, total_delay))
            
        except Exception as e:
            logger.error(f"Error calculating AI enhanced typing delay: {e}")
            # Fallback to simple calculation
            return max(3.0, len(response.split()) * 0.5)

    async def _get_enhancement_details(self, ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Get details about AI enhancements applied"""
        details = {}
        
        if "consciousness_mirror" in ai_insights:
            details["personality_matching"] = "Applied consciousness mirroring for personality alignment"
        
        if "emotional_intelligence" in ai_insights:
            details["emotional_adaptation"] = "Adapted response to emotional state"
        
        if "digital_telepathy" in ai_insights:
            details["response_optimization"] = "Optimized response using predictive insights"
        
        if "synesthesia" in ai_insights:
            details["multi_sensory"] = "Enhanced with synesthetic elements"
        
        if "neural_dreams" in ai_insights:
            details["creative_enhancement"] = "Added creative elements from dream analysis"
        
        return details

    async def _store_conversation_memory(
        self,
        account_id: str,
        message: types.Message,
        dm_analysis: DMAnalysis,
        ai_insights: Dict[str, Any],
        response_data: Dict[str, Any]
    ):
        """Store comprehensive conversation memory in Memory Palace"""
        try:
            memory_data = {
                "account_id": account_id,
                "user_id": str(message.from_user.id),
                "timestamp": datetime.now().isoformat(),
                "original_message": message.text,
                "kelly_response": response_data.get("text", ""),
                "conversation_stage": dm_analysis.priority.value,
                "safety_score": dm_analysis.safety_score,
                "engagement_potential": dm_analysis.engagement_potential,
                "ai_insights": ai_insights,
                "response_method": response_data.get("method", "unknown"),
                "enhancement_details": response_data.get("enhancement_details", {})
            }
            
            await self.memory_palace.store_conversation_memory(
                str(message.from_user.id),
                message.text,
                memory_data,
                dm_analysis.conversation_intent
            )
            
        except Exception as e:
            logger.error(f"Error storing conversation memory: {e}")

    async def _conversation_processing_loop(self):
        """Main conversation processing loop"""
        while True:
            try:
                # Process message queues by priority
                for priority in [MessagePriority.URGENT, MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW]:
                    queue = self.message_queues[priority]
                    if queue:
                        message_data = queue.pop(0)
                        await self._process_queued_message(message_data)
                
                # Cleanup old sessions
                await self._cleanup_old_sessions()
                
                # Sleep before next iteration
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in conversation processing loop: {e}")
                await asyncio.sleep(5.0)

    async def _process_queued_message(self, message_data: Dict[str, Any]):
        """Process a message from the queue"""
        try:
            # Implementation for processing queued messages
            # This would handle delayed responses, follow-ups, etc.
            pass
            
        except Exception as e:
            logger.error(f"Error processing queued message: {e}")

    async def _cleanup_old_sessions(self):
        """Clean up old conversation sessions"""
        try:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, session in self.active_sessions.items():
                # Archive sessions older than 24 hours with no activity
                if (current_time - session.last_message_time).total_seconds() > 86400:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                await self._archive_session(session_id)
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")

    async def _archive_session(self, session_id: str):
        """Archive a conversation session"""
        try:
            session = self.active_sessions.get(session_id)
            if session:
                session.status = ConversationStatus.ARCHIVED
                
                # Save to archive
                archive_key = f"kelly:archive:{session_id}"
                archive_data = json.dumps({
                    "session_data": session.__dict__,
                    "archived_at": datetime.now().isoformat()
                }, default=str)
                await redis_manager.setex(archive_key, 86400 * 30, archive_data)  # 30 days
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                # Remove from Redis active sessions
                active_key = f"kelly:session:{session_id}"
                await redis_manager.delete(active_key)
                
        except Exception as e:
            logger.error(f"Error archiving session: {e}")

    async def get_conversation_insights(self, account_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation insights using all AI features"""
        try:
            conversation_id = f"{account_id}_{user_id}"
            
            # Get conversation memory from Memory Palace
            memory_context = await self.memory_palace.retrieve_conversation_context(
                user_id, "", limit=50
            )
            
            # Get relationship development insights
            relationship_insights = await self.memory_palace.analyze_relationship_development(
                user_id, memory_context
            )
            
            # Get conversation statistics
            conv_stats = await self.personality_service.get_conversation_stats(conversation_id)
            
            # Get AI-powered insights
            ai_insights = {}
            
            if memory_context:
                # Temporal archaeology analysis
                temporal_patterns = await self.temporal_archaeology.analyze_conversation_patterns(
                    user_id, [msg.get("content", "") for msg in memory_context[-10:]]
                )
                ai_insights["temporal_patterns"] = temporal_patterns
                
                # Consciousness mirroring insights
                personality_evolution = await self.consciousness_mirror.analyze_personality_evolution(
                    user_id, memory_context
                )
                ai_insights["personality_evolution"] = personality_evolution
            
            return {
                "conversation_id": conversation_id,
                "memory_context": memory_context[-10:],  # Last 10 memories
                "relationship_insights": relationship_insights,
                "conversation_stats": conv_stats,
                "ai_insights": ai_insights,
                "insights_generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation insights: {e}")
            return {"error": str(e)}

    async def toggle_ai_feature(self, feature_name: str, enabled: bool) -> bool:
        """Toggle AI feature on/off"""
        if feature_name in self.ai_features_enabled:
            self.ai_features_enabled[feature_name] = enabled
            
            # Save to Redis
            await redis_manager.setex(
                "kelly:ai_features_config",
                86400 * 30,
                json.dumps(self.ai_features_enabled)
            )
            
            return True
        return False

    async def get_ai_features_status(self) -> Dict[str, bool]:
        """Get status of all AI features"""
        return self.ai_features_enabled.copy()

# Global instance
kelly_conversation_manager = KellyConversationManager()