"""
Kelly DM Detection and Filtering System

Advanced DM detection with anti-detection measures, intelligent filtering,
and human-like behavior simulation for safe Telegram operations.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from pyrogram import types, Client
from pyrogram.enums import ChatType, MessageMediaType

from app.core.redis import redis_manager
from app.services.consciousness_mirror import ConsciousnessMirror
from app.services.temporal_archaeology import TemporalArchaeology
from app.services.emotional_intelligence_service import EmotionalIntelligenceService

logger = structlog.get_logger()

class MessagePriority(Enum):
    """Message priority levels for response ordering"""
    URGENT = "urgent"           # Immediate response needed
    HIGH = "high"              # High priority conversation
    NORMAL = "normal"          # Regular conversation
    LOW = "low"                # Low priority or filtered
    IGNORE = "ignore"          # Should be ignored completely

class ConversationQuality(Enum):
    """Quality assessment of conversation"""
    EXCELLENT = "excellent"    # High-value conversation
    GOOD = "good"             # Quality conversation
    AVERAGE = "average"       # Regular conversation
    POOR = "poor"             # Low-quality conversation
    TOXIC = "toxic"           # Should be avoided/blocked

class DetectionFilter(Enum):
    """Types of detection filters"""
    SPAM_FILTER = "spam_filter"
    BOT_FILTER = "bot_filter"
    SCAM_FILTER = "scam_filter"
    QUALITY_FILTER = "quality_filter"
    PRIORITY_FILTER = "priority_filter"

@dataclass
class DMAnalysis:
    """Analysis result for a DM"""
    message_id: int
    user_id: int
    username: str
    chat_id: int
    message_text: str
    timestamp: datetime
    
    # Detection results
    is_dm: bool
    is_spam: bool
    is_bot: bool
    is_scam: bool
    
    # Quality metrics
    priority: MessagePriority
    quality: ConversationQuality
    engagement_potential: float  # 0-1 score
    response_urgency: float     # 0-1 score
    
    # Safety metrics
    safety_score: float         # 0-1 score
    trust_indicators: List[str]
    risk_indicators: List[str]
    
    # AI analysis
    personality_traits: Dict[str, float]
    emotional_state: Dict[str, float]
    conversation_intent: str
    suggested_response_tone: str

@dataclass
class AntiDetectionMetrics:
    """Metrics for anti-detection behavior"""
    account_id: str
    last_activity: datetime
    message_count_24h: int
    response_rate_24h: float
    average_response_time: float
    typing_patterns: Dict[str, float]
    activity_patterns: List[datetime]
    risk_level: float  # 0-1 scale

class KellyDMDetector:
    """Advanced DM detection and filtering system with anti-detection"""
    
    def __init__(self):
        self.consciousness_mirror = ConsciousnessMirror()
        self.temporal_archaeology = TemporalArchaeology()
        self.emotional_intelligence = EmotionalIntelligenceService()
        
        # Detection patterns and rules
        self.spam_patterns = self._load_spam_patterns()
        self.bot_indicators = self._load_bot_indicators()
        self.scam_patterns = self._load_scam_patterns()
        
        # Quality assessment criteria
        self.quality_metrics = self._load_quality_metrics()
        
        # Anti-detection profiles
        self.detection_profiles: Dict[str, AntiDetectionMetrics] = {}
        
    async def initialize(self):
        """Initialize the DM detection system"""
        try:
            await self.consciousness_mirror.initialize()
            await self.temporal_archaeology.initialize()
            await self.emotional_intelligence.initialize()
            
            # Load existing detection profiles
            await self._load_detection_profiles()
            
            logger.info("Kelly DM detection system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DM detection system: {e}")
            raise

    def _load_spam_patterns(self) -> List[Dict[str, Any]]:
        """Load spam detection patterns"""
        return [
            {
                "pattern": r"(?i)(free|money|bitcoin|crypto|investment|earn)",
                "weight": 0.7,
                "type": "financial_spam"
            },
            {
                "pattern": r"(?i)(click here|visit|website|link|\.com|\.net)",
                "weight": 0.6,
                "type": "link_spam"
            },
            {
                "pattern": r"(?i)(congratulations|winner|prize|lottery|selected)",
                "weight": 0.8,
                "type": "lottery_spam"
            },
            {
                "pattern": r"(?i)(urgent|limited time|act now|expires)",
                "weight": 0.5,
                "type": "urgency_spam"
            },
            {
                "pattern": r"[A-Z]{5,}",  # Excessive caps
                "weight": 0.4,
                "type": "caps_spam"
            }
        ]

    def _load_bot_indicators(self) -> List[Dict[str, Any]]:
        """Load bot detection indicators"""
        return [
            {
                "indicator": "response_time_too_fast",
                "threshold": 1.0,  # Less than 1 second
                "weight": 0.8
            },
            {
                "indicator": "identical_messages",
                "threshold": 0.95,  # 95% similarity
                "weight": 0.9
            },
            {
                "indicator": "no_typing_indicator",
                "threshold": 0.8,  # 80% of messages without typing
                "weight": 0.6
            },
            {
                "indicator": "perfect_grammar",
                "threshold": 0.95,  # Too perfect grammar
                "weight": 0.4
            },
            {
                "indicator": "fixed_response_patterns",
                "threshold": 0.7,
                "weight": 0.7
            }
        ]

    def _load_scam_patterns(self) -> List[Dict[str, Any]]:
        """Load scam detection patterns"""
        return [
            {
                "pattern": r"(?i)(verify|verification|confirm|security)",
                "weight": 0.6,
                "type": "verification_scam"
            },
            {
                "pattern": r"(?i)(account|suspended|locked|blocked)",
                "weight": 0.7,
                "type": "account_scam"
            },
            {
                "pattern": r"(?i)(send|transfer|payment|bank|card)",
                "weight": 0.8,
                "type": "payment_scam"
            },
            {
                "pattern": r"(?i)(emergency|help|hospital|accident)",
                "weight": 0.5,
                "type": "emergency_scam"
            }
        ]

    def _load_quality_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Load conversation quality assessment metrics"""
        return {
            "excellent": {
                "indicators": ["thoughtful_questions", "personal_sharing", "emotional_depth"],
                "min_message_length": 50,
                "engagement_threshold": 0.8
            },
            "good": {
                "indicators": ["active_listening", "relevant_responses", "genuine_interest"],
                "min_message_length": 25,
                "engagement_threshold": 0.6
            },
            "average": {
                "indicators": ["basic_conversation", "some_engagement"],
                "min_message_length": 10,
                "engagement_threshold": 0.4
            },
            "poor": {
                "indicators": ["short_responses", "low_engagement", "generic_messages"],
                "min_message_length": 5,
                "engagement_threshold": 0.2
            }
        }

    async def _load_detection_profiles(self):
        """Load anti-detection profiles from Redis"""
        try:
            keys = await redis_manager.scan_iter(match="kelly:detection_profile:*")
            async for key in keys:
                data = await redis_manager.get(key)
                if data:
                    profile_data = json.loads(data)
                    account_id = key.split(":")[-1]
                    
                    # Convert datetime strings back to datetime objects
                    if "last_activity" in profile_data:
                        profile_data["last_activity"] = datetime.fromisoformat(profile_data["last_activity"])
                    if "activity_patterns" in profile_data:
                        profile_data["activity_patterns"] = [
                            datetime.fromisoformat(dt) for dt in profile_data["activity_patterns"]
                        ]
                    
                    profile = AntiDetectionMetrics(**profile_data)
                    self.detection_profiles[account_id] = profile
                    
        except Exception as e:
            logger.error(f"Error loading detection profiles: {e}")

    async def analyze_dm(self, account_id: str, message: types.Message) -> DMAnalysis:
        """Comprehensive DM analysis with AI integration"""
        try:
            # Basic message info
            user = message.from_user
            text = message.text or ""
            
            # Determine if it's a DM
            is_dm = message.chat.type == ChatType.PRIVATE
            
            # Run detection filters
            is_spam = await self._detect_spam(text, user)
            is_bot = await self._detect_bot_behavior(account_id, message)
            is_scam = await self._detect_scam(text, user)
            
            # AI-powered analysis
            personality_analysis = await self.consciousness_mirror.analyze_personality(
                str(user.id), text
            )
            
            emotional_analysis = await self.emotional_intelligence.analyze_emotional_state(
                text, {"user_id": str(user.id)}
            )
            
            conversation_patterns = await self.temporal_archaeology.analyze_conversation_patterns(
                str(user.id), [text]
            )
            
            # Calculate quality and priority
            quality = await self._assess_conversation_quality(text, personality_analysis)
            priority = await self._calculate_message_priority(
                text, user, is_spam, is_bot, is_scam, quality
            )
            
            # Calculate engagement metrics
            engagement_potential = await self._calculate_engagement_potential(
                text, personality_analysis, emotional_analysis
            )
            
            response_urgency = await self._calculate_response_urgency(
                text, emotional_analysis, conversation_patterns
            )
            
            # Safety assessment
            safety_score = await self._calculate_safety_score(
                text, user, is_spam, is_bot, is_scam
            )
            
            trust_indicators = await self._identify_trust_indicators(text, user)
            risk_indicators = await self._identify_risk_indicators(text, user)
            
            # Determine conversation intent and suggested response tone
            conversation_intent = await self._determine_conversation_intent(
                text, personality_analysis, emotional_analysis
            )
            
            suggested_tone = await self._suggest_response_tone(
                personality_analysis, emotional_analysis, quality
            )
            
            # Create analysis result
            analysis = DMAnalysis(
                message_id=message.id,
                user_id=user.id,
                username=user.username or "unknown",
                chat_id=message.chat.id,
                message_text=text,
                timestamp=datetime.now(),
                
                is_dm=is_dm,
                is_spam=is_spam,
                is_bot=is_bot,
                is_scam=is_scam,
                
                priority=priority,
                quality=quality,
                engagement_potential=engagement_potential,
                response_urgency=response_urgency,
                
                safety_score=safety_score,
                trust_indicators=trust_indicators,
                risk_indicators=risk_indicators,
                
                personality_traits=personality_analysis,
                emotional_state=emotional_analysis,
                conversation_intent=conversation_intent,
                suggested_response_tone=suggested_tone
            )
            
            # Update anti-detection metrics
            await self._update_detection_metrics(account_id, analysis)
            
            # Store analysis for learning
            await self._store_analysis(account_id, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing DM: {e}")
            # Return basic analysis on error
            return DMAnalysis(
                message_id=message.id,
                user_id=message.from_user.id,
                username=message.from_user.username or "unknown",
                chat_id=message.chat.id,
                message_text=message.text or "",
                timestamp=datetime.now(),
                
                is_dm=message.chat.type == ChatType.PRIVATE,
                is_spam=False,
                is_bot=False,
                is_scam=False,
                
                priority=MessagePriority.NORMAL,
                quality=ConversationQuality.AVERAGE,
                engagement_potential=0.5,
                response_urgency=0.5,
                
                safety_score=0.8,
                trust_indicators=[],
                risk_indicators=[],
                
                personality_traits={},
                emotional_state={},
                conversation_intent="unknown",
                suggested_response_tone="neutral"
            )

    async def _detect_spam(self, text: str, user: types.User) -> bool:
        """Detect spam messages using pattern matching"""
        import re
        
        spam_score = 0.0
        
        for pattern_info in self.spam_patterns:
            if re.search(pattern_info["pattern"], text):
                spam_score += pattern_info["weight"]
        
        # Additional heuristics
        if len(text) > 500:  # Very long messages
            spam_score += 0.3
        
        if text.count("!") > 5:  # Excessive exclamation marks
            spam_score += 0.2
        
        if user and not user.username:  # No username (suspicious)
            spam_score += 0.1
        
        return spam_score > 0.7

    async def _detect_bot_behavior(self, account_id: str, message: types.Message) -> bool:
        """Detect bot-like behavior patterns"""
        try:
            user_id = str(message.from_user.id)
            
            # Check response timing patterns
            conversation_key = f"kelly:conversation_timing:{account_id}_{user_id}"
            timing_data = await redis_manager.lrange(conversation_key, 0, 9)  # Last 10 messages
            
            if len(timing_data) >= 3:
                # Analyze response timing consistency (bots are often too consistent)
                times = [json.loads(data)["response_time"] for data in timing_data]
                
                # Calculate variance in response times
                if len(times) > 1:
                    avg_time = sum(times) / len(times)
                    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
                    
                    # Low variance indicates bot behavior
                    if variance < 0.5 and avg_time < 2.0:
                        return True
            
            # Check for repetitive patterns
            text = message.text or ""
            if len(text) > 0:
                # Store message for pattern analysis
                message_key = f"kelly:user_messages:{user_id}"
                await redis_manager.lpush(message_key, text)
                await redis_manager.ltrim(message_key, 0, 19)  # Keep last 20 messages
                await redis_manager.expire(message_key, 86400 * 7)  # 7 days
                
                # Check for identical or very similar messages
                recent_messages = await redis_manager.lrange(message_key, 0, 9)
                if len(recent_messages) >= 3:
                    similarities = []
                    for msg in recent_messages[1:]:
                        similarity = self._calculate_text_similarity(text, msg)
                        similarities.append(similarity)
                    
                    avg_similarity = sum(similarities) / len(similarities)
                    if avg_similarity > 0.85:  # Very similar messages
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting bot behavior: {e}")
            return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    async def _detect_scam(self, text: str, user: types.User) -> bool:
        """Detect scam messages"""
        import re
        
        scam_score = 0.0
        
        for pattern_info in self.scam_patterns:
            if re.search(pattern_info["pattern"], text):
                scam_score += pattern_info["weight"]
        
        # Additional scam indicators
        if user and user.is_fake:
            scam_score += 0.5
        
        # Check for urgency + money combination
        urgency_words = ["urgent", "immediately", "now", "quickly"]
        money_words = ["money", "payment", "transfer", "send"]
        
        has_urgency = any(word in text.lower() for word in urgency_words)
        has_money = any(word in text.lower() for word in money_words)
        
        if has_urgency and has_money:
            scam_score += 0.6
        
        return scam_score > 0.6

    async def _assess_conversation_quality(self, text: str, personality_analysis: Dict) -> ConversationQuality:
        """Assess the quality of the conversation"""
        
        # Basic quality indicators
        message_length = len(text)
        word_count = len(text.split())
        
        # Personality-based quality assessment
        engagement_level = personality_analysis.get("engagement_level", 0.5)
        emotional_depth = personality_analysis.get("emotional_depth", 0.5)
        thoughtfulness = personality_analysis.get("thoughtfulness", 0.5)
        
        quality_score = (engagement_level + emotional_depth + thoughtfulness) / 3
        
        # Adjust based on message characteristics
        if message_length >= 100 and word_count >= 15:
            quality_score += 0.2
        elif message_length < 20:
            quality_score -= 0.2
        
        # Check for quality indicators
        quality_words = ["because", "think", "feel", "believe", "experience", "understand"]
        quality_indicators = sum(1 for word in quality_words if word in text.lower())
        quality_score += quality_indicators * 0.05
        
        # Determine quality level
        if quality_score >= 0.8:
            return ConversationQuality.EXCELLENT
        elif quality_score >= 0.6:
            return ConversationQuality.GOOD
        elif quality_score >= 0.4:
            return ConversationQuality.AVERAGE
        else:
            return ConversationQuality.POOR

    async def _calculate_message_priority(
        self,
        text: str,
        user: types.User,
        is_spam: bool,
        is_bot: bool,
        is_scam: bool,
        quality: ConversationQuality
    ) -> MessagePriority:
        """Calculate message priority for response ordering"""
        
        if is_spam or is_bot or is_scam:
            return MessagePriority.IGNORE
        
        # Check for urgent indicators
        urgent_words = ["urgent", "emergency", "help", "important", "asap"]
        has_urgent = any(word in text.lower() for word in urgent_words)
        
        if has_urgent:
            return MessagePriority.URGENT
        
        # Quality-based priority
        if quality in [ConversationQuality.EXCELLENT, ConversationQuality.GOOD]:
            return MessagePriority.HIGH
        elif quality == ConversationQuality.AVERAGE:
            return MessagePriority.NORMAL
        else:
            return MessagePriority.LOW

    async def _calculate_engagement_potential(
        self,
        text: str,
        personality_analysis: Dict,
        emotional_analysis: Dict
    ) -> float:
        """Calculate engagement potential score (0-1)"""
        
        base_score = 0.5
        
        # Personality factors
        if personality_analysis:
            openness = personality_analysis.get("openness", 0.5)
            extraversion = personality_analysis.get("extraversion", 0.5)
            agreeableness = personality_analysis.get("agreeableness", 0.5)
            
            personality_score = (openness + extraversion + agreeableness) / 3
            base_score = (base_score + personality_score) / 2
        
        # Emotional factors
        if emotional_analysis:
            positive_emotion = emotional_analysis.get("joy", 0) + emotional_analysis.get("excitement", 0)
            negative_emotion = emotional_analysis.get("anger", 0) + emotional_analysis.get("sadness", 0)
            
            emotional_score = max(0, min(1, 0.5 + positive_emotion - negative_emotion))
            base_score = (base_score + emotional_score) / 2
        
        # Message content factors
        question_count = text.count("?")
        if question_count > 0:
            base_score += 0.1
        
        # Check for conversation starters
        starter_phrases = ["tell me about", "what do you think", "how do you feel", "what's your"]
        if any(phrase in text.lower() for phrase in starter_phrases):
            base_score += 0.2
        
        return max(0.0, min(1.0, base_score))

    async def _calculate_response_urgency(
        self,
        text: str,
        emotional_analysis: Dict,
        conversation_patterns: Dict
    ) -> float:
        """Calculate response urgency score (0-1)"""
        
        base_urgency = 0.3
        
        # Emotional urgency
        if emotional_analysis:
            negative_emotions = emotional_analysis.get("anger", 0) + emotional_analysis.get("sadness", 0)
            excitement = emotional_analysis.get("excitement", 0)
            
            if negative_emotions > 0.7:
                base_urgency += 0.4
            elif excitement > 0.7:
                base_urgency += 0.3
        
        # Text-based urgency indicators
        urgent_words = ["urgent", "important", "quickly", "asap", "emergency"]
        urgent_count = sum(1 for word in urgent_words if word in text.lower())
        base_urgency += urgent_count * 0.1
        
        # Question urgency
        if "?" in text:
            base_urgency += 0.2
        
        # Conversation patterns
        if conversation_patterns:
            response_expectation = conversation_patterns.get("response_expectation", 0.5)
            base_urgency = (base_urgency + response_expectation) / 2
        
        return max(0.0, min(1.0, base_urgency))

    async def _calculate_safety_score(
        self,
        text: str,
        user: types.User,
        is_spam: bool,
        is_bot: bool,
        is_scam: bool
    ) -> float:
        """Calculate safety score (0-1, 1 being safest)"""
        
        base_score = 1.0
        
        # Major safety issues
        if is_spam:
            base_score -= 0.4
        if is_bot:
            base_score -= 0.3
        if is_scam:
            base_score -= 0.5
        
        # User factors
        if user:
            if user.is_fake:
                base_score -= 0.3
            if not user.username:
                base_score -= 0.1
        
        # Content safety
        unsafe_words = ["sex", "nude", "money", "bitcoin", "investment", "drugs"]
        unsafe_count = sum(1 for word in unsafe_words if word in text.lower())
        base_score -= unsafe_count * 0.1
        
        return max(0.0, min(1.0, base_score))

    async def _identify_trust_indicators(self, text: str, user: types.User) -> List[str]:
        """Identify trust indicators in the message"""
        indicators = []
        
        # User indicators
        if user:
            if user.username:
                indicators.append("has_username")
            if user.is_verified:
                indicators.append("verified_account")
            if not user.is_fake:
                indicators.append("authentic_account")
        
        # Message indicators
        if len(text) > 50:
            indicators.append("detailed_message")
        
        if "?" in text:
            indicators.append("asks_questions")
        
        personal_words = ["i", "me", "my", "myself"]
        if any(word in text.lower() for word in personal_words):
            indicators.append("personal_sharing")
        
        return indicators

    async def _identify_risk_indicators(self, text: str, user: types.User) -> List[str]:
        """Identify risk indicators in the message"""
        indicators = []
        
        # User risks
        if user:
            if user.is_fake:
                indicators.append("fake_account")
            if not user.username:
                indicators.append("no_username")
        
        # Content risks
        money_words = ["money", "payment", "send", "transfer", "bitcoin", "cash"]
        if any(word in text.lower() for word in money_words):
            indicators.append("mentions_money")
        
        link_indicators = ["http", "www", ".com", ".net", "click"]
        if any(indicator in text.lower() for indicator in link_indicators):
            indicators.append("contains_links")
        
        sexual_words = ["sex", "nude", "naked", "hookup"]
        if any(word in text.lower() for word in sexual_words):
            indicators.append("sexual_content")
        
        return indicators

    async def _determine_conversation_intent(
        self,
        text: str,
        personality_analysis: Dict,
        emotional_analysis: Dict
    ) -> str:
        """Determine the intent behind the conversation"""
        
        # Check for specific intents
        if any(word in text.lower() for word in ["buy", "sell", "money", "payment"]):
            return "commercial"
        
        if any(word in text.lower() for word in ["date", "meet", "coffee", "dinner"]):
            return "romantic"
        
        if any(word in text.lower() for word in ["help", "support", "advice"]):
            return "support_seeking"
        
        if "?" in text:
            return "information_seeking"
        
        # Based on emotional state
        if emotional_analysis:
            if emotional_analysis.get("excitement", 0) > 0.7:
                return "enthusiasm"
            elif emotional_analysis.get("sadness", 0) > 0.7:
                return "emotional_support"
        
        # Default to social
        return "social_conversation"

    async def _suggest_response_tone(
        self,
        personality_analysis: Dict,
        emotional_analysis: Dict,
        quality: ConversationQuality
    ) -> str:
        """Suggest appropriate response tone"""
        
        # High-quality conversations deserve warmer tones
        if quality in [ConversationQuality.EXCELLENT, ConversationQuality.GOOD]:
            base_tone = "warm_engaging"
        else:
            base_tone = "friendly_neutral"
        
        # Adjust based on emotional state
        if emotional_analysis:
            if emotional_analysis.get("sadness", 0) > 0.6:
                return "empathetic_supportive"
            elif emotional_analysis.get("excitement", 0) > 0.7:
                return "enthusiastic_matching"
            elif emotional_analysis.get("anger", 0) > 0.6:
                return "calm_de_escalating"
        
        # Adjust based on personality
        if personality_analysis:
            if personality_analysis.get("extraversion", 0.5) > 0.7:
                return "energetic_playful"
            elif personality_analysis.get("openness", 0.5) > 0.7:
                return "curious_thoughtful"
        
        return base_tone

    async def _update_detection_metrics(self, account_id: str, analysis: DMAnalysis):
        """Update anti-detection metrics for the account"""
        try:
            # Get or create profile
            if account_id not in self.detection_profiles:
                self.detection_profiles[account_id] = AntiDetectionMetrics(
                    account_id=account_id,
                    last_activity=datetime.now(),
                    message_count_24h=0,
                    response_rate_24h=0.0,
                    average_response_time=30.0,
                    typing_patterns={},
                    activity_patterns=[],
                    risk_level=0.0
                )
            
            profile = self.detection_profiles[account_id]
            
            # Update basic metrics
            profile.last_activity = datetime.now()
            profile.message_count_24h += 1
            
            # Update activity patterns (keep last 100 activities)
            profile.activity_patterns.append(datetime.now())
            if len(profile.activity_patterns) > 100:
                profile.activity_patterns = profile.activity_patterns[-100:]
            
            # Calculate risk level based on analysis
            risk_factors = []
            if analysis.is_spam:
                risk_factors.append("engaging_with_spam")
            if analysis.safety_score < 0.5:
                risk_factors.append("low_safety_conversations")
            if analysis.response_urgency > 0.8:
                risk_factors.append("too_eager_responses")
            
            profile.risk_level = len(risk_factors) * 0.2
            
            # Save profile
            await self._save_detection_profile(profile)
            
        except Exception as e:
            logger.error(f"Error updating detection metrics: {e}")

    async def _save_detection_profile(self, profile: AntiDetectionMetrics):
        """Save detection profile to Redis"""
        try:
            # Convert datetime objects to strings for JSON serialization
            profile_data = {
                "account_id": profile.account_id,
                "last_activity": profile.last_activity.isoformat(),
                "message_count_24h": profile.message_count_24h,
                "response_rate_24h": profile.response_rate_24h,
                "average_response_time": profile.average_response_time,
                "typing_patterns": profile.typing_patterns,
                "activity_patterns": [dt.isoformat() for dt in profile.activity_patterns],
                "risk_level": profile.risk_level
            }
            
            key = f"kelly:detection_profile:{profile.account_id}"
            await redis_manager.setex(key, 86400 * 7, json.dumps(profile_data))
            
        except Exception as e:
            logger.error(f"Error saving detection profile: {e}")

    async def _store_analysis(self, account_id: str, analysis: DMAnalysis):
        """Store analysis results for learning and monitoring"""
        try:
            analysis_data = {
                "account_id": account_id,
                "message_id": analysis.message_id,
                "user_id": analysis.user_id,
                "username": analysis.username,
                "timestamp": analysis.timestamp.isoformat(),
                "is_dm": analysis.is_dm,
                "is_spam": analysis.is_spam,
                "is_bot": analysis.is_bot,
                "is_scam": analysis.is_scam,
                "priority": analysis.priority.value,
                "quality": analysis.quality.value,
                "engagement_potential": analysis.engagement_potential,
                "safety_score": analysis.safety_score,
                "conversation_intent": analysis.conversation_intent
            }
            
            # Store in daily analysis log
            today = datetime.now().strftime("%Y-%m-%d")
            key = f"kelly:analysis_log:{account_id}:{today}"
            await redis_manager.lpush(key, json.dumps(analysis_data))
            await redis_manager.ltrim(key, 0, 999)  # Keep last 1000 analyses
            await redis_manager.expire(key, 86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")

    async def get_detection_stats(self, account_id: str) -> Dict[str, Any]:
        """Get detection statistics for an account"""
        try:
            profile = self.detection_profiles.get(account_id)
            if not profile:
                return {"error": "Profile not found"}
            
            # Get today's analysis data
            today = datetime.now().strftime("%Y-%m-%d")
            key = f"kelly:analysis_log:{account_id}:{today}"
            analyses = await redis_manager.lrange(key, 0, -1)
            
            # Calculate stats
            total_messages = len(analyses)
            spam_count = sum(1 for analysis in analyses if json.loads(analysis).get("is_spam", False))
            bot_count = sum(1 for analysis in analyses if json.loads(analysis).get("is_bot", False))
            scam_count = sum(1 for analysis in analyses if json.loads(analysis).get("is_scam", False))
            
            return {
                "account_id": account_id,
                "last_activity": profile.last_activity.isoformat(),
                "message_count_24h": profile.message_count_24h,
                "risk_level": profile.risk_level,
                "total_messages_today": total_messages,
                "spam_detected": spam_count,
                "bots_detected": bot_count,
                "scams_detected": scam_count,
                "filter_efficiency": {
                    "spam_rate": spam_count / total_messages if total_messages > 0 else 0,
                    "bot_rate": bot_count / total_messages if total_messages > 0 else 0,
                    "scam_rate": scam_count / total_messages if total_messages > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting detection stats: {e}")
            return {"error": str(e)}

    async def should_respond_to_message(self, account_id: str, analysis: DMAnalysis) -> Tuple[bool, str]:
        """Determine if Kelly should respond to a message based on analysis"""
        
        # Never respond to filtered content
        if analysis.priority == MessagePriority.IGNORE:
            return False, "Message filtered (spam/bot/scam)"
        
        # Safety check
        if analysis.safety_score < 0.3:
            return False, "Safety score too low"
        
        # Quality check
        if analysis.quality == ConversationQuality.TOXIC:
            return False, "Toxic conversation quality"
        
        # Check anti-detection limits
        profile = self.detection_profiles.get(account_id)
        if profile and profile.risk_level > 0.8:
            return False, "Account risk level too high"
        
        # Priority-based response decision
        if analysis.priority == MessagePriority.URGENT:
            return True, "Urgent message"
        elif analysis.priority == MessagePriority.HIGH:
            return True, "High priority message"
        elif analysis.priority == MessagePriority.NORMAL:
            # Use engagement potential to decide
            return analysis.engagement_potential > 0.4, f"Normal priority (engagement: {analysis.engagement_potential:.2f})"
        else:  # LOW priority
            # Only respond to high-engagement low-priority messages
            return analysis.engagement_potential > 0.7, f"Low priority (engagement: {analysis.engagement_potential:.2f})"

# Global instance
kelly_dm_detector = KellyDMDetector()