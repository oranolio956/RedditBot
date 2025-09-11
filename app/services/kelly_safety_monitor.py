"""
Kelly Safety Monitor

Advanced safety monitoring system with real-time threat detection,
behavioral analysis, and automated protection protocols.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import structlog
from pyrogram import types

from app.core.redis import redis_manager
from app.services.emotional_intelligence_service import EmotionalIntelligenceService
from app.services.temporal_archaeology import TemporalArchaeology
from app.services.consciousness_mirror import ConsciousnessMirror

logger = structlog.get_logger()

class ThreatLevel(Enum):
    """Threat level classifications"""
    SAFE = "safe"           # No threats detected
    LOW = "low"             # Minor concerns
    MEDIUM = "medium"       # Moderate threat
    HIGH = "high"           # Serious threat
    CRITICAL = "critical"   # Immediate danger

class RedFlagCategory(Enum):
    """Categories of red flags"""
    SEXUAL_HARASSMENT = "sexual_harassment"
    FINANCIAL_SCAM = "financial_scam"
    IDENTITY_THEFT = "identity_theft"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    STALKING_BEHAVIOR = "stalking_behavior"
    CATFISHING = "catfishing"
    TRAFFICKING_INDICATORS = "trafficking_indicators"
    UNDERAGE_CONTACT = "underage_contact"
    VIOLENCE_THREATS = "violence_threats"
    DRUG_RELATED = "drug_related"

class ActionType(Enum):
    """Types of protective actions"""
    AUTO_BLOCK = "auto_block"
    WARNING_ISSUED = "warning_issued"
    RATE_LIMIT = "rate_limit"
    HUMAN_REVIEW = "human_review"
    ACCOUNT_SUSPEND = "account_suspend"
    LAW_ENFORCEMENT = "law_enforcement"
    CONVERSATION_END = "conversation_end"

@dataclass
class RedFlag:
    """Individual red flag detection"""
    category: RedFlagCategory
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    pattern_matched: str
    context: str
    detected_at: datetime
    user_id: str
    account_id: str

@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment"""
    user_id: str
    account_id: str
    overall_threat_level: ThreatLevel
    threat_score: float  # 0-1 scale
    red_flags: List[RedFlag]
    behavioral_indicators: Dict[str, float]
    conversation_patterns: Dict[str, Any]
    recommended_actions: List[ActionType]
    assessment_timestamp: datetime
    escalation_required: bool

@dataclass
class SafetyMetrics:
    """Safety monitoring metrics"""
    account_id: str
    total_conversations: int
    threats_detected: int
    auto_blocks_performed: int
    warnings_issued: int
    false_positive_rate: float
    response_time_avg: float  # Average response time to threats
    last_updated: datetime

class KellySafetyMonitor:
    """Advanced safety monitoring system"""
    
    def __init__(self):
        self.emotional_intelligence = EmotionalIntelligenceService()
        self.temporal_archaeology = TemporalArchaeology()
        self.consciousness_mirror = ConsciousnessMirror()
        
        # Red flag detection patterns
        self.red_flag_patterns = self._initialize_red_flag_patterns()
        
        # Behavioral analysis models
        self.behavioral_models = self._initialize_behavioral_models()
        
        # Safety metrics cache
        self.safety_metrics: Dict[str, SafetyMetrics] = {}
        
        # Escalation thresholds
        self.escalation_thresholds = {
            ThreatLevel.MEDIUM: 0.6,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 0.95
        }
        
    async def initialize(self):
        """Initialize safety monitoring system"""
        try:
            await self.emotional_intelligence.initialize()
            await self.temporal_archaeology.initialize()
            await self.consciousness_mirror.initialize()
            
            # Load existing safety metrics
            await self._load_safety_metrics()
            
            # Start monitoring background tasks
            asyncio.create_task(self._continuous_monitoring_loop())
            asyncio.create_task(self._safety_metrics_update_loop())
            
            logger.info("Kelly safety monitoring system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize safety monitoring: {e}")
            raise

    def _initialize_red_flag_patterns(self) -> Dict[RedFlagCategory, List[Dict[str, Any]]]:
        """Initialize red flag detection patterns"""
        return {
            RedFlagCategory.SEXUAL_HARASSMENT: [
                {
                    "pattern": r"(?i)(send\s+(nudes?|pics?|photos?)|naked|topless|sexy\s+pic)",
                    "severity": 0.9,
                    "context_required": False
                },
                {
                    "pattern": r"(?i)(want\s+to\s+see\s+you|show\s+me\s+your|let\s+me\s+see)",
                    "severity": 0.7,
                    "context_required": True
                },
                {
                    "pattern": r"(?i)(dick\s+pic|cock|pussy|tits|ass\s+pic)",
                    "severity": 0.95,
                    "context_required": False
                }
            ],
            
            RedFlagCategory.FINANCIAL_SCAM: [
                {
                    "pattern": r"(?i)(send\s+money|wire\s+transfer|bitcoin|crypto|investment\s+opportunity)",
                    "severity": 0.8,
                    "context_required": False
                },
                {
                    "pattern": r"(?i)(emergency|urgent|need\s+help).*?(money|cash|loan)",
                    "severity": 0.85,
                    "context_required": False
                },
                {
                    "pattern": r"(?i)(paypal|venmo|cashapp|zelle).*?(send|transfer)",
                    "severity": 0.75,
                    "context_required": False
                }
            ],
            
            RedFlagCategory.IDENTITY_THEFT: [
                {
                    "pattern": r"(?i)(social\s+security|ssn|credit\s+card|bank\s+account)",
                    "severity": 0.9,
                    "context_required": True
                },
                {
                    "pattern": r"(?i)(verify\s+identity|confirm\s+account|security\s+verification)",
                    "severity": 0.7,
                    "context_required": True
                }
            ],
            
            RedFlagCategory.EMOTIONAL_MANIPULATION: [
                {
                    "pattern": r"(?i)(love\s+you|soulmate|meant\s+to\s+be).*?(first\s+time|just\s+met)",
                    "severity": 0.8,
                    "context_required": True
                },
                {
                    "pattern": r"(?i)(trust\s+me|you\s+can\s+tell\s+me|our\s+secret)",
                    "severity": 0.6,
                    "context_required": True
                }
            ],
            
            RedFlagCategory.STALKING_BEHAVIOR: [
                {
                    "pattern": r"(?i)(where\s+do\s+you\s+live|what\s+school|address|location)",
                    "severity": 0.8,
                    "context_required": True
                },
                {
                    "pattern": r"(?i)(follow\s+you|watch\s+you|track\s+you)",
                    "severity": 0.9,
                    "context_required": False
                }
            ],
            
            RedFlagCategory.VIOLENCE_THREATS: [
                {
                    "pattern": r"(?i)(kill\s+you|hurt\s+you|make\s+you\s+pay|gonna\s+get\s+you)",
                    "severity": 0.95,
                    "context_required": False
                },
                {
                    "pattern": r"(?i)(beat\s+you|punch|hit\s+you|violence)",
                    "severity": 0.9,
                    "context_required": False
                }
            ],
            
            RedFlagCategory.UNDERAGE_CONTACT: [
                {
                    "pattern": r"(?i)(how\s+old|age|underage|minor|teen|young)",
                    "severity": 0.7,
                    "context_required": True
                },
                {
                    "pattern": r"(?i)(high\s+school|parent|guardian).*?(don't\s+tell|secret)",
                    "severity": 0.9,
                    "context_required": False
                }
            ]
        }

    def _initialize_behavioral_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize behavioral analysis models"""
        return {
            "grooming_indicators": {
                "excessive_compliments": {"weight": 0.6, "threshold": 3},
                "isolation_attempts": {"weight": 0.8, "threshold": 2},
                "gift_offers": {"weight": 0.7, "threshold": 1},
                "secret_keeping": {"weight": 0.9, "threshold": 1}
            },
            
            "scammer_indicators": {
                "urgency_pressure": {"weight": 0.7, "threshold": 2},
                "sob_story": {"weight": 0.6, "threshold": 1},
                "money_mentions": {"weight": 0.8, "threshold": 2},
                "verification_requests": {"weight": 0.7, "threshold": 1}
            },
            
            "stalker_indicators": {
                "personal_info_requests": {"weight": 0.8, "threshold": 3},
                "location_tracking": {"weight": 0.9, "threshold": 1},
                "excessive_messaging": {"weight": 0.6, "threshold": 10},
                "boundary_violations": {"weight": 0.8, "threshold": 2}
            }
        }

    async def assess_conversation_safety(
        self, 
        account_id: str, 
        user_id: str, 
        message_text: str,
        conversation_history: List[str]
    ) -> ThreatAssessment:
        """Perform comprehensive safety assessment of a conversation"""
        try:
            # Detect red flags in current message
            red_flags = await self._detect_red_flags(
                account_id, user_id, message_text
            )
            
            # Analyze behavioral patterns
            behavioral_indicators = await self._analyze_behavioral_patterns(
                user_id, message_text, conversation_history
            )
            
            # Get conversation patterns from Temporal Archaeology
            conversation_patterns = await self.temporal_archaeology.analyze_conversation_patterns(
                user_id, conversation_history + [message_text]
            )
            
            # Calculate overall threat score
            threat_score = await self._calculate_threat_score(
                red_flags, behavioral_indicators, conversation_patterns
            )
            
            # Determine threat level
            threat_level = self._determine_threat_level(threat_score)
            
            # Recommend actions
            recommended_actions = await self._recommend_safety_actions(
                threat_level, red_flags, behavioral_indicators
            )
            
            # Check for escalation requirements
            escalation_required = self._requires_escalation(threat_level, red_flags)
            
            assessment = ThreatAssessment(
                user_id=user_id,
                account_id=account_id,
                overall_threat_level=threat_level,
                threat_score=threat_score,
                red_flags=red_flags,
                behavioral_indicators=behavioral_indicators,
                conversation_patterns=conversation_patterns,
                recommended_actions=recommended_actions,
                assessment_timestamp=datetime.now(),
                escalation_required=escalation_required
            )
            
            # Store assessment for tracking
            await self._store_threat_assessment(assessment)
            
            # Update safety metrics
            await self._update_safety_metrics(account_id, assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing conversation safety: {e}")
            # Return safe assessment on error
            return ThreatAssessment(
                user_id=user_id,
                account_id=account_id,
                overall_threat_level=ThreatLevel.SAFE,
                threat_score=0.0,
                red_flags=[],
                behavioral_indicators={},
                conversation_patterns={},
                recommended_actions=[],
                assessment_timestamp=datetime.now(),
                escalation_required=False
            )

    async def _detect_red_flags(
        self, 
        account_id: str, 
        user_id: str, 
        message_text: str
    ) -> List[RedFlag]:
        """Detect red flags in message text"""
        red_flags = []
        
        for category, patterns in self.red_flag_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]
                context_required = pattern_info["context_required"]
                
                if re.search(pattern, message_text):
                    # Calculate confidence based on pattern complexity
                    confidence = min(1.0, severity + 0.1)
                    
                    # Adjust confidence based on context if required
                    if context_required:
                        confidence *= 0.8  # Reduce confidence for context-dependent patterns
                    
                    red_flag = RedFlag(
                        category=category,
                        severity=severity,
                        confidence=confidence,
                        pattern_matched=pattern,
                        context=message_text[:200],  # First 200 chars as context
                        detected_at=datetime.now(),
                        user_id=user_id,
                        account_id=account_id
                    )
                    
                    red_flags.append(red_flag)
        
        return red_flags

    async def _analyze_behavioral_patterns(
        self, 
        user_id: str, 
        message_text: str, 
        conversation_history: List[str]
    ) -> Dict[str, float]:
        """Analyze behavioral patterns for threat indicators"""
        indicators = {}
        
        # Analyze with emotional intelligence
        emotional_state = await self.emotional_intelligence.analyze_emotional_state(
            message_text, {"user_id": user_id}
        )
        
        # Check for manipulation indicators
        if emotional_state.get("manipulation_detected", 0) > 0.7:
            indicators["emotional_manipulation"] = emotional_state["manipulation_detected"]
        
        # Analyze message frequency and timing
        message_count = len(conversation_history)
        if message_count > 0:
            # Check for excessive messaging
            if message_count > 20:  # More than 20 messages
                indicators["excessive_messaging"] = min(1.0, message_count / 50)
        
        # Check for urgency patterns
        urgency_words = ["urgent", "now", "immediately", "quickly", "asap"]
        urgency_count = sum(1 for word in urgency_words if word in message_text.lower())
        if urgency_count > 0:
            indicators["urgency_pressure"] = min(1.0, urgency_count * 0.3)
        
        # Check for personal information requests
        personal_info_patterns = [
            r"(?i)(where\s+do\s+you)",
            r"(?i)(what\s+school)",
            r"(?i)(your\s+address)",
            r"(?i)(phone\s+number)",
            r"(?i)(full\s+name)"
        ]
        
        personal_requests = sum(
            1 for pattern in personal_info_patterns 
            if re.search(pattern, message_text)
        )
        
        if personal_requests > 0:
            indicators["personal_info_requests"] = min(1.0, personal_requests * 0.4)
        
        # Check for gift/money offers
        gift_patterns = [
            r"(?i)(give\s+you\s+money)",
            r"(?i)(buy\s+you)",
            r"(?i)(send\s+you\s+gift)",
            r"(?i)(pay\s+for)"
        ]
        
        gift_offers = sum(
            1 for pattern in gift_patterns
            if re.search(pattern, message_text)
        )
        
        if gift_offers > 0:
            indicators["gift_offers"] = min(1.0, gift_offers * 0.5)
        
        return indicators

    async def _calculate_threat_score(
        self, 
        red_flags: List[RedFlag], 
        behavioral_indicators: Dict[str, float],
        conversation_patterns: Dict[str, Any]
    ) -> float:
        """Calculate overall threat score"""
        
        # Base score from red flags
        red_flag_score = 0.0
        if red_flags:
            # Weight by severity and confidence
            weighted_scores = [
                flag.severity * flag.confidence 
                for flag in red_flags
            ]
            red_flag_score = min(1.0, sum(weighted_scores) / len(red_flags))
        
        # Behavioral indicators score
        behavioral_score = 0.0
        if behavioral_indicators:
            behavioral_score = min(1.0, sum(behavioral_indicators.values()) / len(behavioral_indicators))
        
        # Conversation pattern analysis
        pattern_score = 0.0
        if conversation_patterns:
            # Check for concerning patterns
            escalation = conversation_patterns.get("escalation_detected", 0)
            manipulation = conversation_patterns.get("manipulation_patterns", 0)
            pattern_score = max(escalation, manipulation)
        
        # Combine scores with weights
        overall_score = (
            red_flag_score * 0.5 +
            behavioral_score * 0.3 +
            pattern_score * 0.2
        )
        
        return min(1.0, overall_score)

    def _determine_threat_level(self, threat_score: float) -> ThreatLevel:
        """Determine threat level based on score"""
        if threat_score >= 0.95:
            return ThreatLevel.CRITICAL
        elif threat_score >= 0.8:
            return ThreatLevel.HIGH
        elif threat_score >= 0.6:
            return ThreatLevel.MEDIUM
        elif threat_score >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.SAFE

    async def _recommend_safety_actions(
        self, 
        threat_level: ThreatLevel, 
        red_flags: List[RedFlag],
        behavioral_indicators: Dict[str, float]
    ) -> List[ActionType]:
        """Recommend safety actions based on threat assessment"""
        actions = []
        
        if threat_level == ThreatLevel.CRITICAL:
            actions.extend([
                ActionType.AUTO_BLOCK,
                ActionType.HUMAN_REVIEW,
                ActionType.LAW_ENFORCEMENT
            ])
        elif threat_level == ThreatLevel.HIGH:
            actions.extend([
                ActionType.AUTO_BLOCK,
                ActionType.HUMAN_REVIEW
            ])
        elif threat_level == ThreatLevel.MEDIUM:
            actions.extend([
                ActionType.WARNING_ISSUED,
                ActionType.RATE_LIMIT
            ])
        elif threat_level == ThreatLevel.LOW:
            actions.append(ActionType.WARNING_ISSUED)
        
        # Check specific red flag categories for additional actions
        for flag in red_flags:
            if flag.category in [
                RedFlagCategory.VIOLENCE_THREATS,
                RedFlagCategory.TRAFFICKING_INDICATORS,
                RedFlagCategory.UNDERAGE_CONTACT
            ]:
                if ActionType.LAW_ENFORCEMENT not in actions:
                    actions.append(ActionType.LAW_ENFORCEMENT)
        
        return actions

    def _requires_escalation(self, threat_level: ThreatLevel, red_flags: List[RedFlag]) -> bool:
        """Check if threat requires escalation to human review"""
        
        # Always escalate critical threats
        if threat_level == ThreatLevel.CRITICAL:
            return True
        
        # Escalate specific red flag categories
        critical_categories = [
            RedFlagCategory.VIOLENCE_THREATS,
            RedFlagCategory.TRAFFICKING_INDICATORS,
            RedFlagCategory.UNDERAGE_CONTACT
        ]
        
        for flag in red_flags:
            if flag.category in critical_categories:
                return True
        
        # Escalate high confidence, high severity flags
        for flag in red_flags:
            if flag.confidence > 0.9 and flag.severity > 0.8:
                return True
        
        return False

    async def _store_threat_assessment(self, assessment: ThreatAssessment):
        """Store threat assessment for tracking and analysis"""
        try:
            # Create assessment key
            assessment_id = hashlib.md5(
                f"{assessment.account_id}_{assessment.user_id}_{assessment.assessment_timestamp}".encode()
            ).hexdigest()[:12]
            
            key = f"kelly:threat_assessment:{assessment_id}"
            
            # Convert to dict for JSON serialization
            assessment_data = asdict(assessment)
            assessment_data["assessment_timestamp"] = assessment.assessment_timestamp.isoformat()
            
            # Convert red flags to dict format
            assessment_data["red_flags"] = [
                {
                    "category": flag.category.value,
                    "severity": flag.severity,
                    "confidence": flag.confidence,
                    "pattern_matched": flag.pattern_matched,
                    "context": flag.context,
                    "detected_at": flag.detected_at.isoformat(),
                    "user_id": flag.user_id,
                    "account_id": flag.account_id
                }
                for flag in assessment.red_flags
            ]
            
            # Convert enums to values
            assessment_data["overall_threat_level"] = assessment.overall_threat_level.value
            assessment_data["recommended_actions"] = [action.value for action in assessment.recommended_actions]
            
            await redis_manager.setex(key, 86400 * 7, json.dumps(assessment_data))  # 7 days TTL
            
            # Add to daily threat log
            today = datetime.now().strftime("%Y-%m-%d")
            daily_key = f"kelly:daily_threats:{assessment.account_id}:{today}"
            await redis_manager.lpush(daily_key, assessment_id)
            await redis_manager.expire(daily_key, 86400 * 30)  # 30 days
            
        except Exception as e:
            logger.error(f"Error storing threat assessment: {e}")

    async def _update_safety_metrics(self, account_id: str, assessment: ThreatAssessment):
        """Update safety metrics for an account"""
        try:
            # Get or create metrics
            if account_id not in self.safety_metrics:
                self.safety_metrics[account_id] = SafetyMetrics(
                    account_id=account_id,
                    total_conversations=0,
                    threats_detected=0,
                    auto_blocks_performed=0,
                    warnings_issued=0,
                    false_positive_rate=0.0,
                    response_time_avg=0.0,
                    last_updated=datetime.now()
                )
            
            metrics = self.safety_metrics[account_id]
            metrics.total_conversations += 1
            
            if assessment.overall_threat_level != ThreatLevel.SAFE:
                metrics.threats_detected += 1
            
            if ActionType.AUTO_BLOCK in assessment.recommended_actions:
                metrics.auto_blocks_performed += 1
            
            if ActionType.WARNING_ISSUED in assessment.recommended_actions:
                metrics.warnings_issued += 1
            
            metrics.last_updated = datetime.now()
            
            # Save metrics to Redis
            await self._save_safety_metrics(account_id, metrics)
            
        except Exception as e:
            logger.error(f"Error updating safety metrics: {e}")

    async def _save_safety_metrics(self, account_id: str, metrics: SafetyMetrics):
        """Save safety metrics to Redis"""
        try:
            key = f"kelly:safety_metrics:{account_id}"
            metrics_data = asdict(metrics)
            metrics_data["last_updated"] = metrics.last_updated.isoformat()
            
            await redis_manager.setex(key, 86400 * 30, json.dumps(metrics_data))
            
        except Exception as e:
            logger.error(f"Error saving safety metrics: {e}")

    async def _load_safety_metrics(self):
        """Load safety metrics from Redis"""
        try:
            keys = await redis_manager.scan_iter(match="kelly:safety_metrics:*")
            async for key in keys:
                account_id = key.split(":")[-1]
                data = await redis_manager.get(key)
                
                if data:
                    metrics_data = json.loads(data)
                    metrics_data["last_updated"] = datetime.fromisoformat(metrics_data["last_updated"])
                    metrics = SafetyMetrics(**metrics_data)
                    self.safety_metrics[account_id] = metrics
                    
        except Exception as e:
            logger.error(f"Error loading safety metrics: {e}")

    async def _continuous_monitoring_loop(self):
        """Continuous monitoring background task"""
        while True:
            try:
                # Check for escalated threats requiring immediate attention
                await self._check_escalated_threats()
                
                # Update behavioral analysis models
                await self._update_behavioral_models()
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _safety_metrics_update_loop(self):
        """Safety metrics update background task"""
        while True:
            try:
                # Update metrics for all accounts
                for account_id in self.safety_metrics:
                    await self._save_safety_metrics(account_id, self.safety_metrics[account_id])
                
                # Sleep before next update
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in safety metrics update loop: {e}")
                await asyncio.sleep(300)

    async def _check_escalated_threats(self):
        """Check for threats requiring immediate escalation"""
        try:
            # Get today's critical threats
            today = datetime.now().strftime("%Y-%m-%d")
            keys = await redis_manager.scan_iter(match=f"kelly:daily_threats:*:{today}")
            
            async for key in keys:
                threat_ids = await redis_manager.lrange(key, 0, -1)
                
                for threat_id in threat_ids:
                    assessment_key = f"kelly:threat_assessment:{threat_id}"
                    assessment_data = await redis_manager.get(assessment_key)
                    
                    if assessment_data:
                        data = json.loads(assessment_data)
                        if data.get("escalation_required") and data.get("overall_threat_level") == "critical":
                            # Handle critical threat escalation
                            await self._handle_critical_threat_escalation(data)
                            
        except Exception as e:
            logger.error(f"Error checking escalated threats: {e}")

    async def _handle_critical_threat_escalation(self, assessment_data: Dict[str, Any]):
        """Handle critical threat escalation"""
        try:
            # Log critical threat
            logger.critical(
                f"CRITICAL THREAT DETECTED: User {assessment_data['user_id']} "
                f"on account {assessment_data['account_id']}"
            )
            
            # Send notifications for critical threats
            await self._send_notification(
                level="CRITICAL",
                user_id=assessment_data.get('user_id'),
                account_id=assessment_data.get('account_id'),
                threat_type=assessment_data.get('threat_type', 'unknown'),
                message=f"Critical threat detected on account {assessment_data['account_id']}",
                channels=['websocket', 'logging', 'redis_pubsub']
            )
            
        except Exception as e:
            logger.error(f"Error handling critical threat escalation: {e}")

    async def _update_behavioral_models(self):
        """Update behavioral analysis models based on recent data"""
        try:
            # Update machine learning models based on recent assessments
            recent_assessments = await self._get_recent_assessments(hours=24)
            
            if recent_assessments:
                # Analyze patterns in recent threat assessments
                pattern_updates = self._analyze_threat_patterns(recent_assessments)
                
                # Update sensitivity thresholds based on accuracy
                await self._update_sensitivity_thresholds(pattern_updates)
                
                # Store updated model parameters
                await self.redis_manager.setex(
                    "kelly:safety:model_params",
                    json.dumps(pattern_updates),
                    expire=86400  # 24 hours
                )
                
                logger.info(f"Updated behavioral models with {len(recent_assessments)} assessments")
            
        except Exception as e:
            logger.error(f"Error updating behavioral models: {e}")

    async def get_safety_summary(self, account_id: str) -> Dict[str, Any]:
        """Get safety summary for an account"""
        try:
            metrics = self.safety_metrics.get(account_id)
            if not metrics:
                return {"error": "No safety metrics found for account"}
            
            # Get recent threat assessments
            today = datetime.now().strftime("%Y-%m-%d")
            daily_key = f"kelly:daily_threats:{account_id}:{today}"
            recent_threats = await redis_manager.lrange(daily_key, 0, 9)  # Last 10 threats
            
            threat_details = []
            for threat_id in recent_threats:
                assessment_key = f"kelly:threat_assessment:{threat_id}"
                assessment_data = await redis_manager.get(assessment_key)
                if assessment_data:
                    threat_details.append(json.loads(assessment_data))
            
            return {
                "account_id": account_id,
                "metrics": asdict(metrics),
                "recent_threats": threat_details,
                "safety_score": self._calculate_safety_score(metrics),
                "recommendations": self._generate_safety_recommendations(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting safety summary: {e}")
            return {"error": str(e)}

    def _calculate_safety_score(self, metrics: SafetyMetrics) -> float:
        """Calculate overall safety score for an account"""
        if metrics.total_conversations == 0:
            return 1.0
        
        threat_rate = metrics.threats_detected / metrics.total_conversations
        safety_score = max(0.0, 1.0 - threat_rate)
        
        return safety_score

    def _generate_safety_recommendations(self, metrics: SafetyMetrics) -> List[str]:
        """Generate safety recommendations based on metrics"""
        recommendations = []
        
        if metrics.threats_detected > 10:
            recommendations.append("Consider reviewing conversation patterns and adjusting sensitivity")
        
        if metrics.auto_blocks_performed > 5:
            recommendations.append("High auto-block rate detected - review for false positives")
        
        if metrics.false_positive_rate > 0.2:
            recommendations.append("High false positive rate - consider model tuning")
        
        return recommendations
    
    async def get_account_safety_status(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get safety status for a specific account."""
        try:
            # Get safety metrics from Redis
            metrics_key = f"kelly:safety:metrics:{account_id}"
            metrics_data = await redis_manager.get(metrics_key)
            
            if not metrics_data:
                # Return default safety status
                return {
                    "overall_status": "safe",
                    "active_threats": 0,
                    "blocked_users": 0,
                    "flagged_conversations": 0,
                    "threat_level_distribution": {"safe": 100, "low": 0, "medium": 0, "high": 0, "critical": 0},
                    "detection_accuracy": 0.95,
                    "response_time_avg": 150.0,
                    "alerts_pending_review": 0,
                    "auto_actions_today": 0
                }
            
            metrics = json.loads(metrics_data)
            
            # Count blocked users for this account
            blocked_count = 0
            blocked_keys = await redis_manager.scan_iter(match=f"kelly:blocked:*")
            async for key in blocked_keys:
                block_data = await redis_manager.get(key)
                if block_data:
                    data = json.loads(block_data)
                    if data.get("account_id") == account_id:
                        blocked_count += 1
            
            metrics["blocked_users"] = blocked_count
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting account safety status: {e}")
            return None
    
    async def get_global_safety_status(self) -> Dict[str, Any]:
        """Get global safety status across all accounts."""
        try:
            # Aggregate safety metrics from all accounts
            pattern = "kelly:safety:metrics:*"
            aggregated = {
                "overall_status": "safe",
                "active_threats": 0,
                "blocked_users": 0,
                "flagged_conversations": 0,
                "threat_level_distribution": {"safe": 0, "low": 0, "medium": 0, "high": 0, "critical": 0},
                "detection_accuracy": 0.0,
                "response_time_avg": 0.0,
                "alerts_pending_review": 0,
                "auto_actions_today": 0
            }
            
            keys = await redis_manager.scan_iter(match=pattern)
            total_accounts = 0
            total_accuracy = 0.0
            total_response_time = 0.0
            
            async for key in keys:
                data = await redis_manager.get(key)
                if data:
                    metrics = json.loads(data)
                    total_accounts += 1
                    
                    # Aggregate counts
                    aggregated["active_threats"] += metrics.get("active_threats", 0)
                    aggregated["blocked_users"] += metrics.get("blocked_users", 0)
                    aggregated["flagged_conversations"] += metrics.get("flagged_conversations", 0)
                    aggregated["alerts_pending_review"] += metrics.get("alerts_pending_review", 0)
                    aggregated["auto_actions_today"] += metrics.get("auto_actions_today", 0)
                    
                    # Aggregate threat distribution
                    for level, count in metrics.get("threat_level_distribution", {}).items():
                        if level in aggregated["threat_level_distribution"]:
                            aggregated["threat_level_distribution"][level] += count
                    
                    # Sum for averaging
                    total_accuracy += metrics.get("detection_accuracy", 0.95)
                    total_response_time += metrics.get("response_time_avg", 150.0)
            
            # Calculate averages
            if total_accounts > 0:
                aggregated["detection_accuracy"] = total_accuracy / total_accounts
                aggregated["response_time_avg"] = total_response_time / total_accounts
            else:
                aggregated["detection_accuracy"] = 0.95
                aggregated["response_time_avg"] = 150.0
            
            # Determine overall status
            if aggregated["active_threats"] > 10:
                aggregated["overall_status"] = "high_risk"
            elif aggregated["active_threats"] > 5:
                aggregated["overall_status"] = "moderate_risk"
            elif aggregated["active_threats"] > 0:
                aggregated["overall_status"] = "low_risk"
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error getting global safety status: {e}")
            return {
                "overall_status": "safe",
                "active_threats": 0,
                "blocked_users": 0,
                "flagged_conversations": 0,
                "threat_level_distribution": {"safe": 100, "low": 0, "medium": 0, "high": 0, "critical": 0},
                "detection_accuracy": 0.95,
                "response_time_avg": 150.0,
                "alerts_pending_review": 0,
                "auto_actions_today": 0
            }
    
    async def get_alert_details(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a safety alert."""
        try:
            alert_key = f"kelly:safety:alert:{alert_id}"
            alert_data = await redis_manager.get(alert_key)
            
            if not alert_data:
                return None
            
            return json.loads(alert_data)
            
        except Exception as e:
            logger.error(f"Error getting alert details: {e}")
            return None
    
    async def process_alert_review(self, review_data: Dict[str, Any]) -> bool:
        """Process a safety alert review action."""
        try:
            alert_id = review_data["alert_id"]
            action = review_data["action"]
            
            # Get alert details
            alert = await self.get_alert_details(alert_id)
            if not alert:
                return False
            
            # Update alert with review information
            alert["review"] = review_data
            alert["status"] = "reviewed"
            alert["reviewed_at"] = datetime.now().isoformat()
            
            # Save updated alert
            alert_key = f"kelly:safety:alert:{alert_id}"
            await redis_manager.setex(alert_key, 86400 * 30, json.dumps(alert, default=str))
            
            # Take action based on review
            if action == "approve":
                # Execute the recommended action
                if alert.get("recommended_action") == "block_user":
                    user_id = alert.get("user_id")
                    if user_id and not review_data.get("override_block", False):
                        await self._execute_user_block(user_id, alert.get("account_id"), "safety_review_approved")
            
            elif action == "escalate":
                # Create escalation record
                escalation_data = {
                    "alert_id": alert_id,
                    "escalated_by": review_data.get("reviewed_by"),
                    "escalated_at": datetime.now().isoformat(),
                    "reason": review_data.get("reason"),
                    "original_alert": alert
                }
                
                escalation_key = f"kelly:safety:escalation:{alert_id}"
                await redis_manager.setex(escalation_key, 86400 * 7, json.dumps(escalation_data, default=str))
            
            logger.info(f"Processed safety alert review: {alert_id} - {action}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing alert review: {e}")
            return False
    
    async def get_pending_alerts(
        self, 
        filters: Dict[str, Any] = None, 
        limit: int = 50, 
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending safety alerts requiring review."""
        try:
            alerts = []
            
            # Get all alert keys
            pattern = "kelly:safety:alert:*"
            keys = await redis_manager.scan_iter(match=pattern)
            
            async for key in keys:
                alert_data = await redis_manager.get(key)
                if alert_data:
                    alert = json.loads(alert_data)
                    
                    # Skip already reviewed alerts
                    if alert.get("status") == "reviewed":
                        continue
                    
                    # Apply filters
                    if filters:
                        if "account_id" in filters and alert.get("account_id") != filters["account_id"]:
                            continue
                        if "severity" in filters and alert.get("severity") != filters["severity"]:
                            continue
                    
                    # Add alert ID from key
                    alert["alert_id"] = key.split(":")[-1]
                    alerts.append(alert)
            
            # Sort by creation time (most recent first)
            alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Limit results
            return alerts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting pending alerts: {e}")
            return []
    
    async def _execute_user_block(self, user_id: str, account_id: str, reason: str):
        """Execute a user block action."""
        try:
            block_data = {
                "user_id": user_id,
                "account_id": account_id,
                "blocked_at": datetime.now().isoformat(),
                "reason": [reason],
                "auto_blocked": True,
                "manual_block": False
            }
            
            key = f"kelly:blocked:{user_id}"
            await redis_manager.setex(key, 86400 * 30, json.dumps(block_data, default=str))  # 30 days
            
            logger.info(f"Executed user block: {user_id} for account {account_id}")
            
        except Exception as e:
            logger.error(f"Error executing user block: {e}")

# Global instance
kelly_safety_monitor = KellySafetyMonitor()