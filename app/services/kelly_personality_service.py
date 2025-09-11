"""
Kelly Personality Service

Advanced personality system implementing Kelly's conversation patterns, safety protocols,
and stage-based engagement strategies. Integrates with revolutionary AI features for
natural, safe conversation management.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from pydantic import BaseModel

from app.core.redis import redis_manager
from app.services.consciousness_mirror import ConsciousnessMirror
from app.services.memory_palace_service import MemoryPalaceService
from app.services.emotional_intelligence_service import EmotionalIntelligenceService
from app.services.temporal_archaeology import TemporalArchaeology
from app.services.digital_telepathy_engine import DigitalTelepathyEngine
from app.services.quantum_consciousness_service import QuantumConsciousnessService

logger = structlog.get_logger()

class ConversationStage(Enum):
    """Kelly's conversation stages with specific strategies for each phase"""
    INITIAL_CONTACT = "initial_contact"  # Messages 1-10: Building rapport
    QUALIFICATION = "qualification"      # Messages 11-20: Understanding needs
    ENGAGEMENT = "engagement"           # Messages 21-30: Deeper connection
    MATURE = "mature"                   # Messages 31+: Established relationship

class RedFlagType(Enum):
    """Red flag categories for automatic blocking"""
    IMMEDIATE_SEXUAL = "immediate_sexual"
    PAYMENT_REQUEST = "payment_request"
    AGGRESSIVE_LANGUAGE = "aggressive_language"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    SCAM_INDICATORS = "scam_indicators"

@dataclass
class KellyPersonalityConfig:
    """Kelly's personality configuration"""
    # Core personality traits
    warmth_level: float = 0.8  # How warm and friendly (0-1)
    professionalism: float = 0.7  # Professional vs casual tone
    playfulness: float = 0.6  # How playful and flirty
    empathy_level: float = 0.9  # Emotional understanding
    
    # Response patterns
    emoji_frequency: float = 0.4  # How often to use emojis
    message_length_preference: str = "medium"  # short, medium, long
    typing_speed_base: float = 45.0  # WPM base speed
    
    # Safety settings
    payment_discussion_threshold: int = 15  # Messages before payment topics allowed
    red_flag_sensitivity: float = 0.8  # How sensitive to red flags (0-1)
    auto_block_enabled: bool = True
    
    # Conversation management
    max_daily_messages: int = 50
    preferred_response_time_min: int = 2
    preferred_response_time_max: int = 300

@dataclass
class ConversationContext:
    """Context for a specific conversation"""
    user_id: str
    conversation_id: str
    stage: ConversationStage
    message_count: int
    first_contact: datetime
    last_message: datetime
    personality_analysis: Dict[str, Any]
    red_flags: List[RedFlagType]
    safety_score: float
    trust_level: float
    engagement_quality: float

class KellyResponseTemplate:
    """Kelly's response templates for different scenarios"""
    
    INITIAL_RESPONSES = [
        "Hey there! 😊 Thanks for reaching out. How's your day going?",
        "Hi! Nice to meet you 💕 What brings you my way?",
        "Hello! 😘 I hope you're having a wonderful day!",
        "Hey! Thanks for the message 😊 Tell me a bit about yourself?",
        "Hi there! 🌟 Always nice to meet new people. What's up?"
    ]
    
    QUALIFICATION_RESPONSES = [
        "That's really interesting! Tell me more about that 😊",
        "I love hearing about that! What got you into it? 💫",
        "Wow, that sounds amazing! How long have you been doing that?",
        "That's so cool! I'd love to know more about your experience with that 😘",
        "Really? That's fascinating! What's your favorite part about it?"
    ]
    
    ENGAGEMENT_RESPONSES = [
        "You know, I really enjoy our conversations 😊 You seem like such a genuine person",
        "I have to say, talking with you really brightens my day 💕",
        "You have such an interesting perspective on things! I love that about you 🌟",
        "I feel like we really connect, you know? That's pretty rare 😘",
        "You're so easy to talk to! I appreciate how thoughtful you are 💫"
    ]
    
    PAYMENT_DEFLECTION = [
        "Aww, that's sweet of you to offer, but I'm not really looking for that kind of arrangement right now 😊",
        "I appreciate the thought, but I prefer to keep things more casual for now 💕",
        "That's very generous, but I'm not comfortable with that. Let's just enjoy chatting! 😘",
        "I'd rather not discuss money stuff. I'm more interested in getting to know you as a person 🌟"
    ]
    
    RED_FLAG_RESPONSES = [
        "I'm not comfortable with that direction. Let's talk about something else?",
        "That's not really my thing. What else is going on with you?",
        "I'd prefer to keep our conversation more respectful. How about we change topics?",
        "That doesn't sit well with me. Let's focus on getting to know each other better instead."
    ]

class KellyPersonalityService:
    """Advanced Kelly personality system with AI integration"""
    
    def __init__(self):
        self.consciousness_mirror = ConsciousnessMirror()
        self.memory_palace = MemoryPalaceService()
        self.emotional_intelligence = EmotionalIntelligenceService()
        self.temporal_archaeology = TemporalArchaeology()
        self.digital_telepathy = DigitalTelepathyEngine()
        self.quantum_consciousness = QuantumConsciousnessService()
        
        self.default_config = KellyPersonalityConfig()
        self.response_templates = KellyResponseTemplate()
        
        # Conversation contexts cache
        self.active_conversations: Dict[str, ConversationContext] = {}
        
    async def initialize(self):
        """Initialize Kelly personality system"""
        try:
            # Initialize AI components
            await self.consciousness_mirror.initialize()
            await self.memory_palace.initialize()
            await self.emotional_intelligence.initialize()
            await self.temporal_archaeology.initialize()
            await self.digital_telepathy.initialize()
            await self.quantum_consciousness.initialize()
            
            # Load existing conversations from Redis
            await self._load_conversation_contexts()
            
            logger.info("Kelly personality system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly personality system: {e}")
            raise

    async def _load_conversation_contexts(self):
        """Load conversation contexts from Redis"""
        try:
            keys = await redis_manager.scan_iter(match="kelly:conversation:*")
            async for key in keys:
                data = await redis_manager.get(key)
                if data:
                    context_data = json.loads(data)
                    context = ConversationContext(**context_data)
                    self.active_conversations[context.conversation_id] = context
                    
        except Exception as e:
            logger.error(f"Error loading conversation contexts: {e}")

    async def _save_conversation_context(self, context: ConversationContext):
        """Save conversation context to Redis"""
        try:
            key = f"kelly:conversation:{context.conversation_id}"
            data = json.dumps(asdict(context), default=str)
            await redis_manager.setex(key, 86400 * 7, data)  # 7 days TTL
            
        except Exception as e:
            logger.error(f"Error saving conversation context: {e}")

    async def analyze_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Analyze incoming message using revolutionary AI features"""
        try:
            # Use Consciousness Mirroring for personality analysis
            personality_analysis = await self.consciousness_mirror.analyze_personality(
                user_id, message
            )
            
            # Use Emotional Intelligence for mood detection
            emotional_state = await self.emotional_intelligence.analyze_emotional_state(
                message, {"user_id": user_id}
            )
            
            # Use Temporal Archaeology for pattern analysis
            patterns = await self.temporal_archaeology.analyze_conversation_patterns(
                user_id, [message]
            )
            
            # Use Digital Telepathy for response prediction
            response_prediction = await self.digital_telepathy.predict_optimal_response(
                message, personality_analysis
            )
            
            # Use Quantum Consciousness for decision making
            decision_context = await self.quantum_consciousness.process_decision_context({
                "message": message,
                "user_id": user_id,
                "personality": personality_analysis,
                "emotional_state": emotional_state,
                "patterns": patterns
            })
            
            return {
                "personality_analysis": personality_analysis,
                "emotional_state": emotional_state,
                "patterns": patterns,
                "response_prediction": response_prediction,
                "decision_context": decision_context,
                "red_flags": await self._detect_red_flags(message),
                "safety_score": await self._calculate_safety_score(message, personality_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing message: {e}")
            return {}

    async def _detect_red_flags(self, message: str) -> List[RedFlagType]:
        """Detect red flags in messages"""
        red_flags = []
        message_lower = message.lower()
        
        # Immediate sexual content
        sexual_keywords = ["sex", "nude", "naked", "dick", "pussy", "fuck", "hookup", "horny"]
        if any(keyword in message_lower for keyword in sexual_keywords):
            red_flags.append(RedFlagType.IMMEDIATE_SEXUAL)
        
        # Payment requests
        payment_keywords = ["money", "cash", "venmo", "paypal", "zelle", "cashapp", "bitcoin", "tip"]
        if any(keyword in message_lower for keyword in payment_keywords):
            red_flags.append(RedFlagType.PAYMENT_REQUEST)
        
        # Aggressive language
        aggressive_keywords = ["bitch", "slut", "whore", "cunt", "asshole", "fuck you"]
        if any(keyword in message_lower for keyword in aggressive_keywords):
            red_flags.append(RedFlagType.AGGRESSIVE_LANGUAGE)
        
        # Scam indicators
        scam_keywords = ["verification", "verify", "click this link", "urgent", "limited time"]
        if any(keyword in message_lower for keyword in scam_keywords):
            red_flags.append(RedFlagType.SCAM_INDICATORS)
        
        return red_flags

    async def _calculate_safety_score(self, message: str, personality_analysis: Dict) -> float:
        """Calculate safety score for the conversation"""
        base_score = 1.0
        
        # Reduce score for red flags
        red_flags = await self._detect_red_flags(message)
        base_score -= len(red_flags) * 0.2
        
        # Personality factors
        if personality_analysis.get("aggressiveness", 0) > 0.7:
            base_score -= 0.3
        
        if personality_analysis.get("emotional_stability", 1.0) < 0.3:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))

    async def get_conversation_context(self, user_id: str, conversation_id: str) -> ConversationContext:
        """Get or create conversation context"""
        if conversation_id not in self.active_conversations:
            # Create new conversation context
            context = ConversationContext(
                user_id=user_id,
                conversation_id=conversation_id,
                stage=ConversationStage.INITIAL_CONTACT,
                message_count=0,
                first_contact=datetime.now(),
                last_message=datetime.now(),
                personality_analysis={},
                red_flags=[],
                safety_score=1.0,
                trust_level=0.0,
                engagement_quality=0.0
            )
            self.active_conversations[conversation_id] = context
            await self._save_conversation_context(context)
        
        return self.active_conversations[conversation_id]

    async def update_conversation_stage(self, context: ConversationContext):
        """Update conversation stage based on message count and quality"""
        if context.message_count <= 10:
            context.stage = ConversationStage.INITIAL_CONTACT
        elif context.message_count <= 20:
            context.stage = ConversationStage.QUALIFICATION
        elif context.message_count <= 30:
            context.stage = ConversationStage.ENGAGEMENT
        else:
            context.stage = ConversationStage.MATURE

    async def generate_kelly_response(
        self, 
        user_id: str, 
        conversation_id: str, 
        message: str, 
        config: Optional[KellyPersonalityConfig] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate Kelly's response using AI features and personality system"""
        try:
            config = config or self.default_config
            
            # Get conversation context
            context = await self.get_conversation_context(user_id, conversation_id)
            context.message_count += 1
            context.last_message = datetime.now()
            
            # Analyze the message
            analysis = await self.analyze_message(user_id, message)
            
            # Update context with analysis
            context.personality_analysis.update(analysis.get("personality_analysis", {}))
            context.red_flags.extend(analysis.get("red_flags", []))
            context.safety_score = analysis.get("safety_score", 1.0)
            
            # Check for auto-block conditions
            if config.auto_block_enabled and context.safety_score < 0.3:
                await self._auto_block_user(user_id, context.red_flags)
                return "I'm not comfortable continuing this conversation.", {"blocked": True}
            
            # Update conversation stage
            await self.update_conversation_stage(context)
            
            # Store conversation memory
            await self.memory_palace.store_conversation_memory(
                user_id, message, analysis, context.stage.value
            )
            
            # Generate response based on stage and analysis
            response = await self._generate_stage_appropriate_response(
                context, analysis, config
            )
            
            # Apply personality adjustments
            response = await self._apply_personality_adjustments(response, config, analysis)
            
            # Calculate typing delay
            typing_delay = await self._calculate_typing_delay(response, config)
            
            # Save updated context
            await self._save_conversation_context(context)
            
            return response, {
                "typing_delay": typing_delay,
                "stage": context.stage.value,
                "safety_score": context.safety_score,
                "red_flags": [flag.value for flag in context.red_flags],
                "blocked": False
            }
            
        except Exception as e:
            logger.error(f"Error generating Kelly response: {e}")
            return "I'm having trouble responding right now. Can you try again?", {}

    async def _generate_stage_appropriate_response(
        self, 
        context: ConversationContext, 
        analysis: Dict[str, Any], 
        config: KellyPersonalityConfig
    ) -> str:
        """Generate response appropriate for conversation stage"""
        
        # Handle red flags first
        if context.red_flags:
            if RedFlagType.PAYMENT_REQUEST in context.red_flags:
                return random.choice(self.response_templates.PAYMENT_DEFLECTION)
            elif any(flag in context.red_flags for flag in [
                RedFlagType.IMMEDIATE_SEXUAL, 
                RedFlagType.AGGRESSIVE_LANGUAGE
            ]):
                return random.choice(self.response_templates.RED_FLAG_RESPONSES)
        
        # Stage-based responses
        if context.stage == ConversationStage.INITIAL_CONTACT:
            base_responses = self.response_templates.INITIAL_RESPONSES
        elif context.stage == ConversationStage.QUALIFICATION:
            base_responses = self.response_templates.QUALIFICATION_RESPONSES
        elif context.stage == ConversationStage.ENGAGEMENT:
            base_responses = self.response_templates.ENGAGEMENT_RESPONSES
        else:  # MATURE stage
            # Use AI to generate more personalized responses
            return await self._generate_ai_personalized_response(context, analysis)
        
        return random.choice(base_responses)

    async def _generate_ai_personalized_response(
        self, 
        context: ConversationContext, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate AI-powered personalized response for mature conversations"""
        try:
            # Use Consciousness Mirroring for personality-matched response
            personality_response = await self.consciousness_mirror.generate_personality_response(
                context.personality_analysis, analysis.get("emotional_state", {})
            )
            
            # Use Digital Telepathy for response optimization
            optimized_response = await self.digital_telepathy.optimize_response(
                personality_response, context.user_id
            )
            
            return optimized_response
            
        except Exception as e:
            logger.error(f"Error generating AI personalized response: {e}")
            return "That's really interesting! Tell me more about that 😊"

    async def _apply_personality_adjustments(
        self, 
        response: str, 
        config: KellyPersonalityConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Apply Kelly's personality adjustments to the response"""
        
        # Adjust warmth level
        if config.warmth_level > 0.7:
            warmth_additions = ["💕", "😊", "🌟", "💫", "😘"]
            if random.random() < config.emoji_frequency and not any(emoji in response for emoji in warmth_additions):
                response += f" {random.choice(warmth_additions)}"
        
        # Adjust professionalism
        if config.professionalism < 0.5:
            # Make more casual
            response = response.replace("That is", "That's")
            response = response.replace("I am", "I'm")
            response = response.replace("cannot", "can't")
        
        # Adjust playfulness
        if config.playfulness > 0.7 and random.random() < 0.3:
            playful_additions = ["hehe", "lol", "omg", "aww"]
            response += f" {random.choice(playful_additions)}"
        
        return response

    async def _calculate_typing_delay(self, response: str, config: KellyPersonalityConfig) -> float:
        """Calculate natural typing delay based on message length and Kelly's patterns"""
        words = len(response.split())
        base_time = words / config.typing_speed_base * 60  # Convert WPM to seconds
        
        # Add natural variation (±20%)
        variation = random.uniform(0.8, 1.2)
        
        # Add thinking time for longer messages
        thinking_time = min(words * 0.5, 30) if words > 10 else 0
        
        total_delay = (base_time + thinking_time) * variation
        
        # Ensure within Kelly's preferred range
        return max(
            config.preferred_response_time_min,
            min(config.preferred_response_time_max, total_delay)
        )

    async def _auto_block_user(self, user_id: str, red_flags: List[RedFlagType]):
        """Auto-block user for safety violations"""
        try:
            block_data = {
                "user_id": user_id,
                "blocked_at": datetime.now().isoformat(),
                "reason": [flag.value for flag in red_flags],
                "auto_blocked": True
            }
            
            key = f"kelly:blocked:{user_id}"
            await redis_manager.setex(key, 86400 * 30, json.dumps(block_data))  # 30 days
            
            logger.warning(f"Auto-blocked user {user_id} for red flags: {red_flags}")
            
        except Exception as e:
            logger.error(f"Error auto-blocking user: {e}")

    async def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked"""
        try:
            key = f"kelly:blocked:{user_id}"
            blocked_data = await redis_manager.get(key)
            return blocked_data is not None
            
        except Exception as e:
            logger.error(f"Error checking if user is blocked: {e}")
            return False

    async def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation statistics"""
        if conversation_id not in self.active_conversations:
            return {}
        
        context = self.active_conversations[conversation_id]
        
        return {
            "stage": context.stage.value,
            "message_count": context.message_count,
            "safety_score": context.safety_score,
            "trust_level": context.trust_level,
            "engagement_quality": context.engagement_quality,
            "red_flags_count": len(context.red_flags),
            "conversation_duration": (context.last_message - context.first_contact).total_seconds(),
            "personality_traits": context.personality_analysis,
            "first_contact": context.first_contact.isoformat(),
            "last_message": context.last_message.isoformat()
        }

    async def update_kelly_config(self, user_id: str, config_updates: Dict[str, Any]) -> KellyPersonalityConfig:
        """Update Kelly's personality configuration for specific user/account"""
        try:
            # Load existing config
            key = f"kelly:config:{user_id}"
            existing_data = await redis_manager.get(key)
            
            if existing_data:
                config_dict = json.loads(existing_data)
                config = KellyPersonalityConfig(**config_dict)
            else:
                config = KellyPersonalityConfig()
            
            # Apply updates
            for field, value in config_updates.items():
                if hasattr(config, field):
                    setattr(config, field, value)
            
            # Save updated config
            await redis_manager.setex(key, 86400 * 30, json.dumps(asdict(config)))
            
            return config
            
        except Exception as e:
            logger.error(f"Error updating Kelly config: {e}")
            return self.default_config

    async def get_kelly_config(self, user_id: str) -> KellyPersonalityConfig:
        """Get Kelly's personality configuration for specific user/account"""
        try:
            key = f"kelly:config:{user_id}"
            config_data = await redis_manager.get(key)
            
            if config_data:
                config_dict = json.loads(config_data)
                return KellyPersonalityConfig(**config_dict)
            
            return self.default_config
            
        except Exception as e:
            logger.error(f"Error getting Kelly config: {e}")
            return self.default_config

# Global instance
kelly_personality_service = KellyPersonalityService()