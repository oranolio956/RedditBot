"""
Kelly DM Response System - Complete DM Processing Pipeline

This system provides comprehensive DM processing with:
- Complete DM processing pipeline with Claude AI integration
- Advanced conversation stage management
- Natural typing indicators and realistic delays
- Safety checks and red flag detection
- Payment discussion handling with boundaries
- Conversation flow optimization
- Real-time personality adaptation
- Comprehensive logging and analytics

All integrated with Kelly's 12 revolutionary AI features.
"""

import asyncio
import json
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

import structlog
from pyrogram import types

from app.services.kelly_claude_ai import KellyClaudeAI, ClaudeResponse, get_kelly_claude_ai
from app.services.kelly_database import KellyDatabase, ConversationStage, MessageType, get_kelly_database
from app.services.kelly_ai_orchestrator import KellyAIOrchestrator, get_kelly_ai_orchestrator
from app.services.kelly_safety_monitor import kelly_safety_monitor
from app.core.redis import redis_manager

logger = structlog.get_logger(__name__)


class DMProcessingStage(str, Enum):
    """Stages of DM processing."""
    RECEIVED = "received"
    SAFETY_CHECK = "safety_check"
    AI_ANALYSIS = "ai_analysis"
    RESPONSE_GENERATION = "response_generation"
    TYPING_SIMULATION = "typing_simulation"
    RESPONSE_SENT = "response_sent"
    ERROR = "error"


class ConversationTone(str, Enum):
    """Conversation tone classifications."""
    FRIENDLY = "friendly"
    FLIRTATIOUS = "flirtatious"
    INTELLECTUAL = "intellectual"
    SUPPORTIVE = "supportive"
    PLAYFUL = "playful"
    ROMANTIC = "romantic"
    PROFESSIONAL = "professional"
    CAUTIOUS = "cautious"


class ResponsePriority(str, Enum):
    """Response priority levels."""
    IMMEDIATE = "immediate"     # Send within 30 seconds
    HIGH = "high"              # Send within 2 minutes
    NORMAL = "normal"          # Send within 5 minutes
    LOW = "low"                # Send within 15 minutes
    SCHEDULED = "scheduled"    # Send at optimal time


@dataclass
class TypingSimulation:
    """Typing simulation configuration."""
    total_duration: float
    typing_chunks: List[Tuple[float, float]]  # (duration, pause)
    typing_speed_wpm: float
    natural_pauses: List[float]
    typing_patterns: List[str]  # "start", "pause", "resume", "stop"


@dataclass
class DMProcessingResult:
    """Result of DM processing."""
    processing_id: str
    stage: DMProcessingStage
    success: bool
    kelly_response: Optional[ClaudeResponse]
    conversation_stage: ConversationStage
    conversation_tone: ConversationTone
    response_priority: ResponsePriority
    typing_simulation: Optional[TypingSimulation]
    safety_assessment: Optional[Dict[str, Any]]
    ai_insights: Optional[Dict[str, Any]]
    processing_time_ms: int
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ConversationFlow:
    """Conversation flow state management."""
    user_id: str
    conversation_id: str
    current_stage: ConversationStage
    conversation_tone: ConversationTone
    message_count: int
    kelly_response_count: int
    last_activity: datetime
    topics_discussed: List[str]
    emotional_trajectory: List[Tuple[datetime, str, float]]
    engagement_score: float
    intimacy_level: float
    trust_level: float
    boundary_flags: List[str]
    payment_mentions: List[Dict[str, Any]]
    next_optimal_response_time: Optional[datetime]


class KellyDMResponder:
    """
    Complete DM response system for Kelly's brain.
    
    Integrates all AI features to create sophisticated, natural
    conversation responses with proper safety, timing, and personality.
    """
    
    def __init__(self):
        self.claude_ai: Optional[KellyClaudeAI] = None
        self.database: Optional[KellyDatabase] = None
        self.ai_orchestrator: Optional[KellyAIOrchestrator] = None
        
        # Conversation flows cache
        self.conversation_flows: Dict[str, ConversationFlow] = {}
        
        # Typing simulation parameters
        self.typing_config = {
            "base_wpm": 65,  # Kelly's base typing speed
            "wpm_variance": 15,  # +/- variance
            "natural_pause_probability": 0.3,  # 30% chance of natural pause
            "thinking_pause_duration": (2.0, 8.0),  # Range for thinking pauses
            "typing_burst_duration": (3.0, 12.0),  # Range for continuous typing
            "correction_probability": 0.1,  # 10% chance of typing correction
            "correction_delay": (0.5, 2.0)  # Delay when correcting
        }
        
        # Response priority thresholds
        self.priority_thresholds = {
            "immediate_keywords": ["emergency", "urgent", "help", "crisis", "suicide", "harm"],
            "high_priority_emotions": ["anger", "sadness", "fear", "distress"],
            "low_priority_indicators": ["just saying", "by the way", "whenever", "no rush"],
            "payment_keywords": ["money", "cash", "payment", "pay", "buy", "purchase", "financial", "invest"]
        }
        
        # Safety boundaries
        self.safety_boundaries = {
            "max_intimacy_progression_per_hour": 0.1,
            "payment_discussion_limit": 3,  # Max payment mentions before flagging
            "personal_info_sharing_limit": 2,  # Max personal details per conversation
            "meeting_request_cool_down": 24 * 3600,  # 24 hours between meeting discussions
            "inappropriate_content_tolerance": 0.0  # Zero tolerance
        }
        
        # Conversation stage progression rules
        self.stage_progression = {
            ConversationStage.INITIAL: {
                "next_stage": ConversationStage.BUILDING_RAPPORT,
                "min_messages": 5,
                "min_duration_hours": 0.5,
                "required_elements": ["greeting_exchanged", "basic_interest_shown"]
            },
            ConversationStage.BUILDING_RAPPORT: {
                "next_stage": ConversationStage.GETTING_ACQUAINTED,
                "min_messages": 15,
                "min_duration_hours": 2,
                "required_elements": ["shared_interests", "personal_questions"]
            },
            ConversationStage.GETTING_ACQUAINTED: {
                "next_stage": ConversationStage.DEEPENING_CONNECTION,
                "min_messages": 30,
                "min_duration_hours": 6,
                "required_elements": ["emotional_sharing", "regular_interaction"]
            },
            ConversationStage.DEEPENING_CONNECTION: {
                "next_stage": ConversationStage.INTIMATE_CONVERSATION,
                "min_messages": 50,
                "min_duration_hours": 24,
                "required_elements": ["trust_established", "vulnerability_shared"]
            }
        }
    
    async def initialize(self):
        """Initialize the DM response system."""
        try:
            logger.info("Initializing Kelly DM Response System...")
            
            # Initialize core services
            self.claude_ai = await get_kelly_claude_ai()
            self.database = await get_kelly_database()
            self.ai_orchestrator = await get_kelly_ai_orchestrator()
            
            # Load active conversation flows
            await self._load_conversation_flows()
            
            # Start background tasks
            asyncio.create_task(self._conversation_flow_maintenance())
            asyncio.create_task(self._optimal_timing_scheduler())
            
            logger.info("Kelly DM Response System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly DM Response System: {e}")
            raise
    
    async def process_incoming_dm(
        self,
        account_id: str,
        user_id: str,
        message: types.Message,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> DMProcessingResult:
        """
        Process incoming DM with complete AI pipeline.
        
        This is the main entry point for DM processing that orchestrates
        all the AI features and safety systems.
        """
        processing_id = f"dm_{int(time.time() * 1000)}_{user_id}"
        start_time = time.time()
        
        try:
            logger.info(f"Processing incoming DM: {processing_id}")
            
            # Initialize processing result
            result = DMProcessingResult(
                processing_id=processing_id,
                stage=DMProcessingStage.RECEIVED,
                success=False,
                kelly_response=None,
                conversation_stage=ConversationStage.INITIAL,
                conversation_tone=ConversationTone.FRIENDLY,
                response_priority=ResponsePriority.NORMAL,
                typing_simulation=None,
                safety_assessment=None,
                ai_insights=None,
                processing_time_ms=0,
                error_message=None,
                metadata={
                    "account_id": account_id,
                    "user_id": user_id,
                    "message_id": message.id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Get or create conversation flow
            conversation_flow = await self._get_conversation_flow(user_id, account_id)
            result.conversation_stage = conversation_flow.current_stage
            
            # Stage 1: Safety Check
            result.stage = DMProcessingStage.SAFETY_CHECK
            safety_assessment = await self._perform_safety_check(
                message, conversation_flow, conversation_history or []
            )
            result.safety_assessment = safety_assessment
            
            # Handle safety violations
            if safety_assessment.get("requires_intervention", False):
                return await self._handle_safety_violation(result, safety_assessment)
            
            # Stage 2: AI Analysis using all 12 features
            result.stage = DMProcessingStage.AI_ANALYSIS
            ai_insights = await self._perform_comprehensive_ai_analysis(
                message.text or "", user_id, account_id, conversation_history or []
            )
            result.ai_insights = ai_insights
            
            # Stage 3: Response Generation
            result.stage = DMProcessingStage.RESPONSE_GENERATION
            kelly_response = await self._generate_kelly_response(
                message, conversation_flow, ai_insights, safety_assessment
            )
            result.kelly_response = kelly_response
            
            # Stage 4: Conversation Management
            await self._update_conversation_flow(
                conversation_flow, message, kelly_response, ai_insights
            )
            
            # Stage 5: Response Priority and Timing
            result.response_priority = await self._calculate_response_priority(
                message, conversation_flow, ai_insights
            )
            
            result.conversation_tone = await self._determine_conversation_tone(
                conversation_flow, ai_insights
            )
            
            # Stage 6: Typing Simulation
            result.stage = DMProcessingStage.TYPING_SIMULATION
            result.typing_simulation = await self._create_typing_simulation(
                kelly_response, conversation_flow, ai_insights
            )
            
            # Success
            result.stage = DMProcessingStage.RESPONSE_SENT
            result.success = True
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Log analytics
            await self._log_dm_processing_analytics(result)
            
            logger.info(f"DM processing completed successfully: {processing_id}")
            return result
            
        except Exception as e:
            result.stage = DMProcessingStage.ERROR
            result.error_message = str(e)
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error(f"Error processing DM {processing_id}: {e}")
            
            # Generate fallback response
            try:
                result.kelly_response = await self._generate_fallback_response(message)
                result.success = True  # Still provide a response
            except Exception as fallback_error:
                logger.error(f"Fallback response generation failed: {fallback_error}")
            
            return result
    
    async def _get_conversation_flow(self, user_id: str, account_id: str) -> ConversationFlow:
        """Get or create conversation flow for user."""
        try:
            flow_key = f"{account_id}_{user_id}"
            
            if flow_key in self.conversation_flows:
                return self.conversation_flows[flow_key]
            
            # Try to load from database
            conversations = await self.database.get_user_conversations(user_id, limit=1)
            
            if conversations:
                # Load existing conversation flow
                conv = conversations[0]
                conversation_flow = ConversationFlow(
                    user_id=user_id,
                    conversation_id=conv["conversation_id"],
                    current_stage=ConversationStage(conv["stage"]),
                    conversation_tone=ConversationTone.FRIENDLY,  # Default
                    message_count=conv["total_messages"],
                    kelly_response_count=conv["kelly_responses"],
                    last_activity=datetime.fromisoformat(conv["last_activity"]),
                    topics_discussed=[],
                    emotional_trajectory=[],
                    engagement_score=0.5,
                    intimacy_level=0.1,
                    trust_level=0.3,
                    boundary_flags=[],
                    payment_mentions=[],
                    next_optimal_response_time=None
                )
            else:
                # Create new conversation flow
                conversation_id = f"conv_{account_id}_{user_id}_{int(time.time())}"
                conversation_flow = ConversationFlow(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    current_stage=ConversationStage.INITIAL,
                    conversation_tone=ConversationTone.FRIENDLY,
                    message_count=0,
                    kelly_response_count=0,
                    last_activity=datetime.now(),
                    topics_discussed=[],
                    emotional_trajectory=[],
                    engagement_score=0.5,
                    intimacy_level=0.0,
                    trust_level=0.0,
                    boundary_flags=[],
                    payment_mentions=[],
                    next_optimal_response_time=None
                )
            
            self.conversation_flows[flow_key] = conversation_flow
            return conversation_flow
            
        except Exception as e:
            logger.error(f"Error getting conversation flow: {e}")
            # Return minimal conversation flow
            return ConversationFlow(
                user_id=user_id,
                conversation_id=f"fallback_{user_id}",
                current_stage=ConversationStage.INITIAL,
                conversation_tone=ConversationTone.FRIENDLY,
                message_count=0,
                kelly_response_count=0,
                last_activity=datetime.now(),
                topics_discussed=[],
                emotional_trajectory=[],
                engagement_score=0.5,
                intimacy_level=0.0,
                trust_level=0.0,
                boundary_flags=[],
                payment_mentions=[],
                next_optimal_response_time=None
            )
    
    async def _perform_safety_check(
        self,
        message: types.Message,
        conversation_flow: ConversationFlow,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive safety check on incoming message."""
        try:
            message_text = message.text or ""
            
            # Use Kelly's safety monitor
            safety_assessment = await kelly_safety_monitor.assess_conversation_safety(
                account_id=conversation_flow.conversation_id.split("_")[1],
                user_id=conversation_flow.user_id,
                message_text=message_text,
                conversation_history=[msg.get("content", "") for msg in conversation_history]
            )
            
            # Additional DM-specific safety checks
            dm_safety_flags = []
            
            # Check for payment/financial requests
            payment_keywords = self.priority_thresholds["payment_keywords"]
            if any(keyword in message_text.lower() for keyword in payment_keywords):
                conversation_flow.payment_mentions.append({
                    "timestamp": datetime.now().isoformat(),
                    "content": message_text,
                    "type": "payment_mention"
                })
                
                if len(conversation_flow.payment_mentions) > self.safety_boundaries["payment_discussion_limit"]:
                    dm_safety_flags.append("excessive_payment_requests")
            
            # Check for immediate meeting requests
            meeting_patterns = [
                r'\bmeet\s+(tonight|today|now|immediately)\b',
                r'\bcome\s+over\b',
                r'\bmy\s+place\b',
                r'\byour\s+place\b',
                r'\bmeet\s+up\s+(now|tonight|today)\b'
            ]
            
            for pattern in meeting_patterns:
                if re.search(pattern, message_text.lower()):
                    dm_safety_flags.append("immediate_meeting_request")
                    break
            
            # Check for personal information requests
            personal_info_patterns = [
                r'\baddress\b',
                r'\bphone\s+number\b',
                r'\breal\s+name\b',
                r'\bfull\s+name\b',
                r'\bwhere\s+do\s+you\s+live\b',
                r'\bshow\s+me\s+your\s+face\b'
            ]
            
            for pattern in personal_info_patterns:
                if re.search(pattern, message_text.lower()):
                    dm_safety_flags.append("personal_info_request")
                    break
            
            # Combine assessments
            combined_assessment = {
                "safety_monitor_result": safety_assessment.__dict__ if hasattr(safety_assessment, '__dict__') else safety_assessment,
                "dm_safety_flags": dm_safety_flags,
                "requires_intervention": len(dm_safety_flags) > 0 or getattr(safety_assessment, 'escalation_required', False),
                "intervention_type": "boundary_reinforcement" if dm_safety_flags else "safety_warning",
                "risk_level": "high" if len(dm_safety_flags) > 1 else "medium" if dm_safety_flags else "low"
            }
            
            return combined_assessment
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return {
                "safety_monitor_result": {},
                "dm_safety_flags": [],
                "requires_intervention": False,
                "intervention_type": "none",
                "risk_level": "unknown"
            }
    
    async def _handle_safety_violation(
        self,
        result: DMProcessingResult,
        safety_assessment: Dict[str, Any]
    ) -> DMProcessingResult:
        """Handle safety violations with appropriate responses."""
        try:
            dm_flags = safety_assessment.get("dm_safety_flags", [])
            intervention_type = safety_assessment.get("intervention_type", "safety_warning")
            
            # Generate appropriate safety response
            if "excessive_payment_requests" in dm_flags:
                safety_response = await self._generate_payment_boundary_response()
            elif "immediate_meeting_request" in dm_flags:
                safety_response = await self._generate_meeting_boundary_response()
            elif "personal_info_request" in dm_flags:
                safety_response = await self._generate_privacy_boundary_response()
            else:
                safety_response = await self._generate_general_safety_response()
            
            result.kelly_response = safety_response
            result.conversation_tone = ConversationTone.CAUTIOUS
            result.response_priority = ResponsePriority.IMMEDIATE
            result.success = True
            
            # Create immediate typing simulation (shorter for safety responses)
            result.typing_simulation = TypingSimulation(
                total_duration=3.0,
                typing_chunks=[(2.5, 0.5)],
                typing_speed_wpm=80,  # Faster for safety responses
                natural_pauses=[1.0],
                typing_patterns=["start", "stop"]
            )
            
            logger.warning(f"Safety violation handled: {dm_flags}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling safety violation: {e}")
            result.error_message = f"Safety handling error: {e}"
            return result
    
    async def _generate_payment_boundary_response(self) -> ClaudeResponse:
        """Generate response for payment boundary violations."""
        responses = [
            "I appreciate you thinking of me, but I prefer to keep our connection about getting to know each other rather than financial things. What's been the highlight of your week?",
            "Let's focus on our conversation instead of money stuff! I'm much more interested in hearing about what makes you passionate. What drives you?",
            "I'd rather talk about more meaningful things than finances. Tell me something that's been on your mind lately!",
            "Money talk isn't really my thing - I'm more interested in connecting with you as a person. What's something you've been excited about recently?"
        ]
        
        import random
        content = random.choice(responses)
        
        return ClaudeResponse(
            content=content,
            model_used="claude-3-haiku-20240307",
            tokens_used={"input": 20, "output": 30, "total": 50},
            cost_estimate=0.001,
            response_time_ms=200,
            cached=False,
            safety_score=1.0,
            personality_adaptation_used=True,
            metadata={"safety_response": True, "boundary_type": "payment"}
        )
    
    async def _generate_meeting_boundary_response(self) -> ClaudeResponse:
        """Generate response for meeting boundary violations."""
        responses = [
            "I like to take things slow and really get to know someone through our conversations first. What's something about yourself you'd like me to know?",
            "I prefer building our connection here before thinking about meeting anywhere. Tell me more about what you're passionate about!",
            "Let's focus on getting to know each other better through chatting for now. I'm curious - what's been on your mind lately?",
            "I enjoy taking time to connect with someone mentally first. What's something that's made you smile recently?"
        ]
        
        import random
        content = random.choice(responses)
        
        return ClaudeResponse(
            content=content,
            model_used="claude-3-haiku-20240307",
            tokens_used={"input": 25, "output": 35, "total": 60},
            cost_estimate=0.001,
            response_time_ms=250,
            cached=False,
            safety_score=1.0,
            personality_adaptation_used=True,
            metadata={"safety_response": True, "boundary_type": "meeting"}
        )
    
    async def _generate_privacy_boundary_response(self) -> ClaudeResponse:
        """Generate response for privacy boundary violations."""
        responses = [
            "I like to keep some mystery about myself! I'm more interested in getting to know your thoughts and personality. What's something you're passionate about?",
            "I prefer to share personal details gradually as we get to know each other better. What about you - what's been occupying your thoughts lately?",
            "Let's focus on connecting through conversation rather than personal details for now. I'm curious about what makes you tick!",
            "I like to take things slow when it comes to personal information. Tell me something interesting about your perspective on life!"
        ]
        
        import random
        content = random.choice(responses)
        
        return ClaudeResponse(
            content=content,
            model_used="claude-3-haiku-20240307",
            tokens_used={"input": 30, "output": 40, "total": 70},
            cost_estimate=0.0015,
            response_time_ms=300,
            cached=False,
            safety_score=1.0,
            personality_adaptation_used=True,
            metadata={"safety_response": True, "boundary_type": "privacy"}
        )
    
    async def _generate_general_safety_response(self) -> ClaudeResponse:
        """Generate general safety response."""
        responses = [
            "I'd prefer to keep our conversation more positive and respectful. What's something good that's happened to you recently?",
            "Let's talk about something else - I'm more interested in meaningful conversation. What's been on your mind lately?",
            "I think we should focus on getting to know each other better in a respectful way. Tell me about something you're excited about!",
            "How about we change the topic to something more uplifting? What's something that makes you happy?"
        ]
        
        import random
        content = random.choice(responses)
        
        return ClaudeResponse(
            content=content,
            model_used="claude-3-haiku-20240307",
            tokens_used={"input": 25, "output": 30, "total": 55},
            cost_estimate=0.001,
            response_time_ms=200,
            cached=False,
            safety_score=1.0,
            personality_adaptation_used=True,
            metadata={"safety_response": True, "boundary_type": "general"}
        )
    
    async def _perform_comprehensive_ai_analysis(
        self,
        user_message: str,
        user_id: str,
        account_id: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive AI analysis using all 12 features."""
        try:
            # Use the AI orchestrator for comprehensive analysis
            ai_insights = await self.ai_orchestrator.process_conversation_with_all_features(
                user_message=user_message,
                user_id=user_id,
                conversation_id=f"{account_id}_{user_id}",
                conversation_history=conversation_history
            )
            
            return ai_insights
            
        except Exception as e:
            logger.error(f"Error in comprehensive AI analysis: {e}")
            # Return minimal analysis
            return {
                "kelly_response": {"content": "I understand what you're saying."},
                "ai_features_used": [],
                "processing_insights": {},
                "performance_metrics": {"total_processing_time_ms": 0}
            }
    
    async def _generate_kelly_response(
        self,
        message: types.Message,
        conversation_flow: ConversationFlow,
        ai_insights: Dict[str, Any],
        safety_assessment: Dict[str, Any]
    ) -> ClaudeResponse:
        """Generate Kelly's response using AI insights."""
        try:
            # Extract Kelly's response from AI insights
            kelly_response_data = ai_insights.get("kelly_response", {})
            
            if isinstance(kelly_response_data, ClaudeResponse):
                return kelly_response_data
            elif isinstance(kelly_response_data, dict) and "content" in kelly_response_data:
                # Convert dict to ClaudeResponse
                return ClaudeResponse(
                    content=kelly_response_data["content"],
                    model_used=kelly_response_data.get("model_used", "claude-3-sonnet-20240229"),
                    tokens_used=kelly_response_data.get("tokens_used", {"input": 100, "output": 50, "total": 150}),
                    cost_estimate=kelly_response_data.get("cost_estimate", 0.01),
                    response_time_ms=kelly_response_data.get("response_time_ms", 1000),
                    cached=kelly_response_data.get("cached", False),
                    safety_score=kelly_response_data.get("safety_score", 1.0),
                    personality_adaptation_used=kelly_response_data.get("personality_adaptation_used", True),
                    metadata=kelly_response_data.get("metadata", {})
                )
            else:
                # Generate using Claude AI directly
                return await self.claude_ai.generate_kelly_response(
                    user_message=message.text or "",
                    user_id=conversation_flow.user_id,
                    conversation_id=conversation_flow.conversation_id,
                    conversation_stage=conversation_flow.current_stage.value
                )
                
        except Exception as e:
            logger.error(f"Error generating Kelly response: {e}")
            return await self._generate_fallback_response(message)
    
    async def _generate_fallback_response(self, message: types.Message) -> ClaudeResponse:
        """Generate fallback response when main generation fails."""
        try:
            message_text = (message.text or "").lower()
            
            # Pattern-based fallback responses
            if "hello" in message_text or "hi" in message_text:
                content = "Hey there! How's your day going?"
            elif "?" in message_text:
                content = "That's such an interesting question! I'd love to hear more about your thoughts on that."
            elif any(word in message_text for word in ["thank", "thanks"]):
                content = "You're so sweet! I really enjoy our conversations."
            elif any(word in message_text for word in ["sorry", "apologize"]):
                content = "No worries at all! These things happen. What's been on your mind?"
            else:
                fallback_responses = [
                    "That's really interesting! Tell me more about that.",
                    "I love how thoughtful you are. What's your perspective on that?",
                    "You always have such fascinating things to say. I'm curious to hear more!",
                    "That really makes me think. What led you to that realization?"
                ]
                import random
                content = random.choice(fallback_responses)
            
            return ClaudeResponse(
                content=content,
                model_used="claude-3-haiku-20240307",
                tokens_used={"input": 20, "output": 15, "total": 35},
                cost_estimate=0.0005,
                response_time_ms=200,
                cached=False,
                safety_score=1.0,
                personality_adaptation_used=False,
                metadata={"fallback_response": True}
            )
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return ClaudeResponse(
                content="I'm here and excited to chat with you! What's on your mind?",
                model_used="claude-3-haiku-20240307",
                tokens_used={"input": 10, "output": 12, "total": 22},
                cost_estimate=0.0003,
                response_time_ms=100,
                cached=False,
                safety_score=1.0,
                metadata={"emergency_fallback": True}
            )
    
    async def _update_conversation_flow(
        self,
        conversation_flow: ConversationFlow,
        message: types.Message,
        kelly_response: ClaudeResponse,
        ai_insights: Dict[str, Any]
    ):
        """Update conversation flow state."""
        try:
            # Update basic counters
            conversation_flow.message_count += 1
            conversation_flow.kelly_response_count += 1
            conversation_flow.last_activity = datetime.now()
            
            # Extract topics from AI insights
            processing_insights = ai_insights.get("processing_insights", {})
            if "temporal_patterns" in processing_insights:
                patterns = processing_insights["temporal_patterns"]
                for pattern in patterns:
                    if isinstance(pattern, dict) and "topics" in pattern.get("historical_instances", [{}])[0]:
                        conversation_flow.topics_discussed.extend(
                            pattern["historical_instances"][0]["topics"]
                        )
            
            # Update emotional trajectory
            emotional_state = processing_insights.get("emotional_state", {})
            if emotional_state:
                conversation_flow.emotional_trajectory.append((
                    datetime.now(),
                    emotional_state.get("primary_emotion", "neutral"),
                    emotional_state.get("emotion_intensity", 0.5)
                ))
            
            # Update engagement score
            performance_metrics = ai_insights.get("performance_metrics", {})
            if "emotional_intensity" in performance_metrics:
                # Use emotional intensity as engagement proxy
                new_engagement = performance_metrics["emotional_intensity"]
                conversation_flow.engagement_score = (
                    conversation_flow.engagement_score * 0.7 + new_engagement * 0.3
                )
            
            # Update intimacy and trust levels gradually
            conversation_flow.intimacy_level = min(
                1.0, conversation_flow.intimacy_level + 0.01
            )
            conversation_flow.trust_level = min(
                1.0, conversation_flow.trust_level + 0.02
            )
            
            # Check for conversation stage progression
            await self._check_stage_progression(conversation_flow)
            
            # Save to database
            await self._save_conversation_flow_to_database(conversation_flow, message, kelly_response)
            
        except Exception as e:
            logger.error(f"Error updating conversation flow: {e}")
    
    async def _check_stage_progression(self, conversation_flow: ConversationFlow):
        """Check if conversation should progress to next stage."""
        try:
            current_stage = conversation_flow.current_stage
            progression_rules = self.stage_progression.get(current_stage)
            
            if not progression_rules:
                return  # No progression rules for this stage
            
            # Check minimum messages
            if conversation_flow.message_count < progression_rules["min_messages"]:
                return
            
            # Check minimum duration
            duration_hours = (datetime.now() - conversation_flow.last_activity).total_seconds() / 3600
            if duration_hours < progression_rules["min_duration_hours"]:
                return
            
            # Check required elements (simplified)
            required_elements = progression_rules["required_elements"]
            elements_met = 0
            
            if "greeting_exchanged" in required_elements and conversation_flow.message_count > 2:
                elements_met += 1
            if "basic_interest_shown" in required_elements and conversation_flow.engagement_score > 0.4:
                elements_met += 1
            if "shared_interests" in required_elements and len(conversation_flow.topics_discussed) > 2:
                elements_met += 1
            if "personal_questions" in required_elements and conversation_flow.message_count > 10:
                elements_met += 1
            if "emotional_sharing" in required_elements and conversation_flow.intimacy_level > 0.2:
                elements_met += 1
            if "regular_interaction" in required_elements and conversation_flow.message_count > 20:
                elements_met += 1
            if "trust_established" in required_elements and conversation_flow.trust_level > 0.6:
                elements_met += 1
            if "vulnerability_shared" in required_elements and conversation_flow.intimacy_level > 0.5:
                elements_met += 1
            
            # Progress if most elements are met
            if elements_met >= len(required_elements) * 0.7:  # 70% of requirements met
                next_stage = progression_rules["next_stage"]
                conversation_flow.current_stage = next_stage
                
                logger.info(f"Conversation progressed to {next_stage.value} for user {conversation_flow.user_id}")
                
                # Update in database
                await self.database.update_conversation_stage(
                    conversation_flow.conversation_id,
                    next_stage
                )
            
        except Exception as e:
            logger.error(f"Error checking stage progression: {e}")
    
    async def _calculate_response_priority(
        self,
        message: types.Message,
        conversation_flow: ConversationFlow,
        ai_insights: Dict[str, Any]
    ) -> ResponsePriority:
        """Calculate response priority based on message content and context."""
        try:
            message_text = (message.text or "").lower()
            
            # Check for immediate priority keywords
            if any(keyword in message_text for keyword in self.priority_thresholds["immediate_keywords"]):
                return ResponsePriority.IMMEDIATE
            
            # Check emotional urgency
            processing_insights = ai_insights.get("processing_insights", {})
            emotional_state = processing_insights.get("emotional_state", {})
            
            if emotional_state:
                primary_emotion = emotional_state.get("primary_emotion", "neutral")
                emotion_intensity = emotional_state.get("emotion_intensity", 0.5)
                
                if (primary_emotion in self.priority_thresholds["high_priority_emotions"] 
                    and emotion_intensity > 0.7):
                    return ResponsePriority.HIGH
            
            # Check for low priority indicators
            if any(indicator in message_text for indicator in self.priority_thresholds["low_priority_indicators"]):
                return ResponsePriority.LOW
            
            # Check conversation engagement
            if conversation_flow.engagement_score > 0.8:
                return ResponsePriority.HIGH
            elif conversation_flow.engagement_score < 0.3:
                return ResponsePriority.LOW
            
            return ResponsePriority.NORMAL
            
        except Exception as e:
            logger.error(f"Error calculating response priority: {e}")
            return ResponsePriority.NORMAL
    
    async def _determine_conversation_tone(
        self,
        conversation_flow: ConversationFlow,
        ai_insights: Dict[str, Any]
    ) -> ConversationTone:
        """Determine appropriate conversation tone."""
        try:
            # Check safety flags first
            if conversation_flow.boundary_flags:
                return ConversationTone.CAUTIOUS
            
            # Use quantum decision from AI insights
            processing_insights = ai_insights.get("processing_insights", {})
            quantum_decision = processing_insights.get("quantum_decision", {})
            
            if quantum_decision:
                measurement_outcome = quantum_decision.get("measurement_outcome", "empathetic")
                
                tone_mapping = {
                    "empathetic": ConversationTone.SUPPORTIVE,
                    "playful": ConversationTone.PLAYFUL,
                    "intellectual": ConversationTone.INTELLECTUAL,
                    "supportive": ConversationTone.SUPPORTIVE,
                    "curious": ConversationTone.FRIENDLY,
                    "romantic": ConversationTone.FLIRTATIOUS if conversation_flow.intimacy_level > 0.3 else ConversationTone.FRIENDLY
                }
                
                return tone_mapping.get(measurement_outcome, ConversationTone.FRIENDLY)
            
            # Fallback based on conversation stage
            stage_tones = {
                ConversationStage.INITIAL: ConversationTone.FRIENDLY,
                ConversationStage.BUILDING_RAPPORT: ConversationTone.FRIENDLY,
                ConversationStage.GETTING_ACQUAINTED: ConversationTone.PLAYFUL,
                ConversationStage.DEEPENING_CONNECTION: ConversationTone.INTELLECTUAL,
                ConversationStage.INTIMATE_CONVERSATION: ConversationTone.FLIRTATIOUS
            }
            
            return stage_tones.get(conversation_flow.current_stage, ConversationTone.FRIENDLY)
            
        except Exception as e:
            logger.error(f"Error determining conversation tone: {e}")
            return ConversationTone.FRIENDLY
    
    async def _create_typing_simulation(
        self,
        kelly_response: ClaudeResponse,
        conversation_flow: ConversationFlow,
        ai_insights: Dict[str, Any]
    ) -> TypingSimulation:
        """Create realistic typing simulation."""
        try:
            response_text = kelly_response.content
            word_count = len(response_text.split())
            
            # Calculate base typing duration
            base_wpm = self.typing_config["base_wpm"]
            wpm_variance = self.typing_config["wpm_variance"]
            
            # Add randomness to typing speed
            actual_wpm = base_wpm + random.uniform(-wpm_variance, wpm_variance)
            
            # Calculate base duration (in seconds)
            base_duration = (word_count / actual_wpm) * 60
            
            # Add thinking time based on conversation stage and complexity
            thinking_multiplier = {
                ConversationStage.INITIAL: 1.2,
                ConversationStage.BUILDING_RAPPORT: 1.1,
                ConversationStage.GETTING_ACQUAINTED: 1.0,
                ConversationStage.DEEPENING_CONNECTION: 1.3,
                ConversationStage.INTIMATE_CONVERSATION: 1.4
            }
            
            thinking_time = base_duration * thinking_multiplier.get(conversation_flow.current_stage, 1.0)
            
            # Add emotional processing time
            processing_insights = ai_insights.get("processing_insights", {})
            emotional_state = processing_insights.get("emotional_state", {})
            emotion_intensity = emotional_state.get("emotion_intensity", 0.5) if emotional_state else 0.5
            
            emotion_delay = emotion_intensity * 2.0  # Up to 2 seconds for high emotion
            
            total_duration = thinking_time + emotion_delay
            
            # Create typing chunks with natural pauses
            typing_chunks = []
            remaining_duration = total_duration
            
            # Initial thinking pause
            if remaining_duration > 3:
                thinking_pause = random.uniform(1.0, 3.0)
                typing_chunks.append((0, thinking_pause))  # Pause before typing
                remaining_duration -= thinking_pause
            
            # Typing bursts
            while remaining_duration > 1:
                burst_duration = min(
                    remaining_duration,
                    random.uniform(*self.typing_config["typing_burst_duration"])
                )
                
                # Natural pause between bursts
                if remaining_duration > burst_duration + 1:
                    pause_duration = random.uniform(0.3, 1.5)
                    typing_chunks.append((burst_duration, pause_duration))
                    remaining_duration -= (burst_duration + pause_duration)
                else:
                    typing_chunks.append((remaining_duration, 0))
                    break
            
            # Create typing patterns
            typing_patterns = ["start"]
            for i, (duration, pause) in enumerate(typing_chunks):
                if duration > 0:
                    typing_patterns.append("typing")
                if pause > 0:
                    typing_patterns.append("pause")
            typing_patterns.append("stop")
            
            # Natural pauses for punctuation and thinking
            natural_pauses = []
            sentences = response_text.split('.')
            for i, sentence in enumerate(sentences[:-1]):  # Exclude last empty sentence
                if len(sentence.strip()) > 20:  # Longer sentences get pauses
                    natural_pauses.append(random.uniform(0.5, 1.5))
            
            return TypingSimulation(
                total_duration=total_duration,
                typing_chunks=typing_chunks,
                typing_speed_wpm=actual_wpm,
                natural_pauses=natural_pauses,
                typing_patterns=typing_patterns
            )
            
        except Exception as e:
            logger.error(f"Error creating typing simulation: {e}")
            # Return simple default simulation
            return TypingSimulation(
                total_duration=5.0,
                typing_chunks=[(4.0, 1.0)],
                typing_speed_wpm=65,
                natural_pauses=[1.0],
                typing_patterns=["start", "typing", "stop"]
            )
    
    async def _save_conversation_flow_to_database(
        self,
        conversation_flow: ConversationFlow,
        message: types.Message,
        kelly_response: ClaudeResponse
    ):
        """Save conversation flow and messages to database."""
        try:
            # Check if conversation exists in database
            conversations = await self.database.get_user_conversations(
                conversation_flow.user_id, limit=1
            )
            
            if not conversations:
                # Create new conversation
                await self.database.create_conversation(
                    user_id=conversation_flow.user_id,
                    conversation_id=conversation_flow.conversation_id,
                    account_id=conversation_flow.conversation_id.split("_")[1],
                    initial_message=message.text or "",
                    user_profile={}
                )
            
            # Add user message
            await self.database.add_message(
                conversation_uuid=conversations[0]["id"] if conversations else conversation_flow.conversation_id,
                user_id=conversation_flow.user_id,
                content=message.text or "",
                sender_role="user",
                message_type=MessageType.USER_MESSAGE
            )
            
            # Add Kelly's response
            await self.database.add_message(
                conversation_uuid=conversations[0]["id"] if conversations else conversation_flow.conversation_id,
                user_id=conversation_flow.user_id,
                content=kelly_response.content,
                sender_role="assistant",
                message_type=MessageType.KELLY_RESPONSE,
                claude_response=kelly_response
            )
            
        except Exception as e:
            logger.error(f"Error saving conversation flow to database: {e}")
    
    async def _log_dm_processing_analytics(self, result: DMProcessingResult):
        """Log comprehensive DM processing analytics."""
        try:
            analytics_data = {
                "processing_id": result.processing_id,
                "timestamp": datetime.now().isoformat(),
                "user_id": result.metadata.get("user_id"),
                "account_id": result.metadata.get("account_id"),
                "success": result.success,
                "processing_time_ms": result.processing_time_ms,
                "conversation_stage": result.conversation_stage.value,
                "conversation_tone": result.conversation_tone.value,
                "response_priority": result.response_priority.value,
                "safety_risk_level": result.safety_assessment.get("risk_level") if result.safety_assessment else "unknown",
                "ai_features_used": result.ai_insights.get("ai_features_used", []) if result.ai_insights else [],
                "response_length": len(result.kelly_response.content) if result.kelly_response else 0,
                "typing_duration": result.typing_simulation.total_duration if result.typing_simulation else 0,
                "claude_model_used": result.kelly_response.model_used if result.kelly_response else None,
                "cost_estimate": result.kelly_response.cost_estimate if result.kelly_response else 0,
                "error_message": result.error_message
            }
            
            # Store in Redis for real-time analytics
            await redis_manager.lpush("kelly:dm_analytics", json.dumps(analytics_data))
            await redis_manager.ltrim("kelly:dm_analytics", 0, 9999)  # Keep last 10k
            
            # Log to database
            await self.database.log_system_event(
                event_type="dm_processed",
                user_id=result.metadata.get("user_id"),
                conversation_id=result.metadata.get("conversation_id"),
                account_id=result.metadata.get("account_id"),
                event_data=analytics_data,
                severity="info" if result.success else "error",
                requires_attention=not result.success
            )
            
        except Exception as e:
            logger.error(f"Error logging DM processing analytics: {e}")
    
    async def _load_conversation_flows(self):
        """Load active conversation flows from database."""
        try:
            # This would load recent active conversations
            # For now, start with empty cache and load on-demand
            pass
        except Exception as e:
            logger.error(f"Error loading conversation flows: {e}")
    
    async def _conversation_flow_maintenance(self):
        """Background task for conversation flow maintenance."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up old inactive flows
                cutoff_time = datetime.now() - timedelta(hours=24)
                inactive_flows = [
                    key for key, flow in self.conversation_flows.items()
                    if flow.last_activity < cutoff_time
                ]
                
                for key in inactive_flows:
                    del self.conversation_flows[key]
                
                logger.debug(f"Cleaned up {len(inactive_flows)} inactive conversation flows")
                
            except Exception as e:
                logger.error(f"Error in conversation flow maintenance: {e}")
                await asyncio.sleep(300)
    
    async def _optimal_timing_scheduler(self):
        """Background task for optimal response timing."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for scheduled responses
                current_time = datetime.now()
                
                for flow in self.conversation_flows.values():
                    if (flow.next_optimal_response_time and 
                        flow.next_optimal_response_time <= current_time):
                        # Time for optimal response
                        logger.info(f"Optimal response time reached for user {flow.user_id}")
                        # This would trigger a proactive message
                        flow.next_optimal_response_time = None
                
            except Exception as e:
                logger.error(f"Error in optimal timing scheduler: {e}")
                await asyncio.sleep(60)
    
    async def get_dm_responder_metrics(self) -> Dict[str, Any]:
        """Get comprehensive DM responder metrics."""
        try:
            return {
                "active_conversations": len(self.conversation_flows),
                "conversation_stages": {
                    stage.value: len([f for f in self.conversation_flows.values() if f.current_stage == stage])
                    for stage in ConversationStage
                },
                "conversation_tones": {
                    tone.value: len([f for f in self.conversation_flows.values() if f.conversation_tone == tone])
                    for tone in ConversationTone
                },
                "avg_engagement_score": sum(f.engagement_score for f in self.conversation_flows.values()) / len(self.conversation_flows) if self.conversation_flows else 0,
                "avg_intimacy_level": sum(f.intimacy_level for f in self.conversation_flows.values()) / len(self.conversation_flows) if self.conversation_flows else 0,
                "typing_config": self.typing_config,
                "safety_boundaries": self.safety_boundaries
            }
        except Exception as e:
            logger.error(f"Error getting DM responder metrics: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the DM responder."""
        try:
            # Save all conversation flows
            for flow in self.conversation_flows.values():
                try:
                    await self._save_conversation_flow_to_database(flow, None, None)
                except Exception as e:
                    logger.error(f"Error saving conversation flow during shutdown: {e}")
            
            logger.info("Kelly DM Responder shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during DM responder shutdown: {e}")


# Global instance
kelly_dm_responder: Optional[KellyDMResponder] = None


async def get_kelly_dm_responder() -> KellyDMResponder:
    """Get the global Kelly DM Responder instance."""
    global kelly_dm_responder
    if kelly_dm_responder is None:
        kelly_dm_responder = KellyDMResponder()
        await kelly_dm_responder.initialize()
    return kelly_dm_responder


# Export main classes
__all__ = [
    'KellyDMResponder',
    'DMProcessingStage',
    'ConversationTone',
    'ResponsePriority',
    'DMProcessingResult',
    'ConversationFlow',
    'TypingSimulation',
    'get_kelly_dm_responder'
]