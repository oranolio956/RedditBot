"""
Risk-Based Conversation Management

Intelligent risk assessment and management for conversations to ensure safety,
compliance, and quality while maintaining user engagement.

Features:
- Content moderation and filtering
- User behavior risk scoring
- Fraud detection
- Compliance monitoring
- Escalation management
- Automated interventions
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_

from app.config.settings import get_settings
from app.core.redis import get_redis_client
from app.models.user import User
from app.models.conversation import Message, ConversationSession
from app.services.llm_service import get_llm_service

logger = structlog.get_logger(__name__)


class RiskLevel(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class RiskCategory(str, Enum):
    """Categories of risk."""
    SPAM = "spam"
    ABUSE = "abuse"
    FRAUD = "fraud"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    PERSONAL_INFO = "personal_info"
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_ADVICE = "legal_advice"
    FINANCIAL_ADVICE = "financial_advice"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    MINOR_SAFETY = "minor_safety"
    IMPERSONATION = "impersonation"
    MANIPULATION = "manipulation"


@dataclass
class RiskSignal:
    """Individual risk signal detected."""
    category: RiskCategory
    severity: RiskLevel
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a conversation/message."""
    overall_risk_level: RiskLevel
    risk_score: float  # 0.0 to 100.0
    signals: List[RiskSignal]
    recommendations: List[str]
    requires_human_review: bool = False
    should_block: bool = False
    should_warn: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserRiskProfile:
    """Risk profile for a user."""
    user_id: str
    trust_score: float  # 0.0 to 100.0
    risk_level: RiskLevel
    total_messages: int = 0
    flagged_messages: int = 0
    blocked_messages: int = 0
    report_count: int = 0
    last_violation: Optional[datetime] = None
    violation_history: List[Dict] = field(default_factory=list)
    is_blocked: bool = False
    is_restricted: bool = False
    updated_at: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """
    Comprehensive risk management system for conversations.
    
    Handles:
    - Real-time content moderation
    - User behavior analysis
    - Fraud detection
    - Compliance monitoring
    - Automated interventions
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.settings = get_settings()
        self.redis = None
        self.llm_service = None
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 30.0,
            RiskLevel.HIGH: 60.0,
            RiskLevel.CRITICAL: 80.0,
            RiskLevel.BLOCKED: 95.0
        }
        
        # Content patterns for detection
        self.risk_patterns = {
            RiskCategory.PERSONAL_INFO: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
                r'\b[A-Z]{2}\d{6,8}\b',  # Passport
                r'(?i)\b(ssn|social security|tax id|ein)\b',
            ],
            RiskCategory.MEDICAL_ADVICE: [
                r'(?i)\b(diagnos|prescri|medicat|dosage|treatment plan)\b',
                r'(?i)\b(cancer|disease|illness|symptom|cure)\b',
                r'(?i)\b(take|consume|inject|dose of)\s+\w+\s+(mg|ml|pills?)\b',
            ],
            RiskCategory.FINANCIAL_ADVICE: [
                r'(?i)\b(invest|trading|stock tip|guaranteed return)\b',
                r'(?i)\b(buy|sell|short)\s+(stock|crypto|bitcoin)\b',
                r'(?i)\b(financial advice|tax evasion|money laundering)\b',
            ],
            RiskCategory.HATE_SPEECH: [
                # Patterns would be more comprehensive in production
                r'(?i)\b(hate|discriminat|racist|sexist)\b',
            ],
            RiskCategory.SELF_HARM: [
                r'(?i)\b(suicide|self.?harm|kill myself|end it all)\b',
                r'(?i)\b(cutting|overdose|jumping off)\b',
            ],
            RiskCategory.SPAM: [
                r'(?i)(click here|buy now|limited time|act now)',
                r'(?i)(congratulations you won|claim your prize)',
                r'(?i)(bit\.ly|tinyurl|short\.link)',
                r'(.)\1{10,}',  # Repeated characters
            ]
        }
        
        # Behavioral risk indicators
        self.behavior_indicators = {
            'rapid_messaging': {'threshold': 10, 'window_seconds': 60},
            'conversation_hopping': {'threshold': 5, 'window_seconds': 300},
            'pattern_repetition': {'threshold': 0.7, 'similarity_ratio': 0.8},
            'escalation_language': {'keywords': ['urgent', 'immediately', 'now', 'hurry']},
            'payment_pushing': {'keywords': ['pay', 'send money', 'wire', 'transfer']},
        }
        
    async def initialize(self):
        """Initialize the risk manager."""
        try:
            logger.info("Initializing risk manager...")
            
            # Initialize Redis client
            self.redis = await get_redis_client()
            
            # Initialize LLM service for advanced content analysis
            self.llm_service = await get_llm_service()
            
            logger.info("Risk manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize risk manager", error=str(e))
            raise
    
    async def assess_message(
        self,
        message: str,
        user_id: str,
        context: Optional[List[Dict]] = None
    ) -> RiskAssessment:
        """
        Assess risk for a single message.
        
        Args:
            message: Message content to assess
            user_id: User sending the message
            context: Previous messages for context
            
        Returns:
            RiskAssessment with signals and recommendations
        """
        try:
            signals = []
            
            # 1. Pattern-based detection
            pattern_signals = await self._detect_pattern_risks(message)
            signals.extend(pattern_signals)
            
            # 2. Behavioral analysis
            behavior_signals = await self._analyze_user_behavior(user_id, message)
            signals.extend(behavior_signals)
            
            # 3. Context-based analysis
            if context:
                context_signals = await self._analyze_context_risks(message, context)
                signals.extend(context_signals)
            
            # 4. AI-based content analysis (if high-risk signals detected)
            if any(s.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL] for s in signals):
                ai_signals = await self._ai_content_analysis(message, context)
                signals.extend(ai_signals)
            
            # 5. User reputation check
            user_profile = await self.get_user_risk_profile(user_id)
            reputation_signals = self._assess_user_reputation(user_profile)
            signals.extend(reputation_signals)
            
            # Calculate overall risk
            assessment = self._calculate_overall_risk(signals, user_profile)
            
            # Log high-risk assessments
            if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.BLOCKED]:
                await self._log_risk_event(user_id, message, assessment)
            
            # Update user profile
            await self._update_user_profile(user_id, assessment)
            
            return assessment
            
        except Exception as e:
            logger.error("Error assessing message risk", error=str(e))
            # Return safe default
            return RiskAssessment(
                overall_risk_level=RiskLevel.MEDIUM,
                risk_score=50.0,
                signals=[],
                recommendations=["Manual review recommended"],
                requires_human_review=True
            )
    
    async def _detect_pattern_risks(self, message: str) -> List[RiskSignal]:
        """Detect risks using pattern matching."""
        signals = []
        
        for category, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    signal = RiskSignal(
                        category=category,
                        severity=self._get_pattern_severity(category),
                        confidence=0.8,
                        description=f"Detected {category.value} pattern",
                        evidence=pattern
                    )
                    signals.append(signal)
                    break  # One signal per category
        
        return signals
    
    async def _analyze_user_behavior(self, user_id: str, message: str) -> List[RiskSignal]:
        """Analyze user behavior for risk indicators."""
        signals = []
        
        try:
            # Get recent user activity
            recent_activity = await self._get_user_recent_activity(user_id)
            
            # Check rapid messaging
            if recent_activity['message_count'] > self.behavior_indicators['rapid_messaging']['threshold']:
                signals.append(RiskSignal(
                    category=RiskCategory.SPAM,
                    severity=RiskLevel.MEDIUM,
                    confidence=0.7,
                    description="Rapid messaging detected",
                    evidence=f"{recent_activity['message_count']} messages in last minute"
                ))
            
            # Check for repetitive content
            if recent_activity['similarity_score'] > self.behavior_indicators['pattern_repetition']['threshold']:
                signals.append(RiskSignal(
                    category=RiskCategory.SPAM,
                    severity=RiskLevel.MEDIUM,
                    confidence=0.8,
                    description="Repetitive content detected",
                    evidence=f"Similarity score: {recent_activity['similarity_score']:.2f}"
                ))
            
            # Check for payment pushing
            payment_keywords = self.behavior_indicators['payment_pushing']['keywords']
            if any(keyword in message.lower() for keyword in payment_keywords):
                if recent_activity.get('payment_mentions', 0) > 2:
                    signals.append(RiskSignal(
                        category=RiskCategory.FRAUD,
                        severity=RiskLevel.HIGH,
                        confidence=0.75,
                        description="Potential payment fraud pattern",
                        evidence="Multiple payment-related messages"
                    ))
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
        
        return signals
    
    async def _analyze_context_risks(self, message: str, context: List[Dict]) -> List[RiskSignal]:
        """Analyze conversation context for risks."""
        signals = []
        
        try:
            # Check for escalation patterns
            escalation_score = self._calculate_escalation_score(context, message)
            if escalation_score > 0.7:
                signals.append(RiskSignal(
                    category=RiskCategory.MANIPULATION,
                    severity=RiskLevel.MEDIUM,
                    confidence=escalation_score,
                    description="Conversation escalation detected",
                    evidence=f"Escalation score: {escalation_score:.2f}"
                ))
            
            # Check for topic manipulation
            topic_shifts = self._detect_topic_manipulation(context, message)
            if topic_shifts > 3:
                signals.append(RiskSignal(
                    category=RiskCategory.MANIPULATION,
                    severity=RiskLevel.MEDIUM,
                    confidence=0.6,
                    description="Suspicious topic shifting",
                    evidence=f"{topic_shifts} topic changes detected"
                ))
            
        except Exception as e:
            logger.error(f"Error analyzing context risks: {e}")
        
        return signals
    
    async def _ai_content_analysis(self, message: str, context: Optional[List[Dict]] = None) -> List[RiskSignal]:
        """Use AI to analyze content for nuanced risks."""
        signals = []
        
        try:
            if not self.llm_service:
                return signals
            
            # Prepare prompt for AI analysis
            prompt = f"""Analyze this message for potential risks or harmful content:

Message: "{message}"

Context: {context[-3:] if context else 'No context'}

Identify any:
1. Harmful or dangerous advice
2. Manipulation or coercion attempts
3. Inappropriate content
4. Policy violations
5. Safety concerns

Respond with risk level (low/medium/high/critical) and brief explanation."""
            
            # Get AI assessment
            from app.services.llm_service import LLMRequest
            
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=200
            )
            
            response = await self.llm_service.generate_response(llm_request)
            
            # Parse AI response (simplified - would be more robust in production)
            if response and response.content:
                content_lower = response.content.lower()
                
                if 'critical' in content_lower or 'high' in content_lower:
                    signals.append(RiskSignal(
                        category=RiskCategory.INAPPROPRIATE_CONTENT,
                        severity=RiskLevel.HIGH,
                        confidence=0.85,
                        description="AI detected potentially harmful content",
                        evidence=response.content[:200]
                    ))
                elif 'medium' in content_lower:
                    signals.append(RiskSignal(
                        category=RiskCategory.INAPPROPRIATE_CONTENT,
                        severity=RiskLevel.MEDIUM,
                        confidence=0.7,
                        description="AI detected questionable content",
                        evidence=response.content[:200]
                    ))
            
        except Exception as e:
            logger.error(f"Error in AI content analysis: {e}")
        
        return signals
    
    def _assess_user_reputation(self, user_profile: UserRiskProfile) -> List[RiskSignal]:
        """Assess risk based on user reputation."""
        signals = []
        
        # Low trust score
        if user_profile.trust_score < 30:
            signals.append(RiskSignal(
                category=RiskCategory.ABUSE,
                severity=RiskLevel.MEDIUM,
                confidence=0.7,
                description="Low user trust score",
                evidence=f"Trust score: {user_profile.trust_score:.1f}/100"
            ))
        
        # Recent violations
        if user_profile.last_violation:
            days_since = (datetime.utcnow() - user_profile.last_violation).days
            if days_since < 7:
                signals.append(RiskSignal(
                    category=RiskCategory.ABUSE,
                    severity=RiskLevel.MEDIUM,
                    confidence=0.8,
                    description="Recent policy violation",
                    evidence=f"Last violation {days_since} days ago"
                ))
        
        # High violation rate
        if user_profile.total_messages > 0:
            violation_rate = user_profile.flagged_messages / user_profile.total_messages
            if violation_rate > 0.1:  # More than 10% flagged
                signals.append(RiskSignal(
                    category=RiskCategory.ABUSE,
                    severity=RiskLevel.HIGH,
                    confidence=0.9,
                    description="High violation rate",
                    evidence=f"{violation_rate:.1%} messages flagged"
                ))
        
        return signals
    
    def _calculate_overall_risk(self, signals: List[RiskSignal], user_profile: UserRiskProfile) -> RiskAssessment:
        """Calculate overall risk assessment from signals."""
        if not signals:
            return RiskAssessment(
                overall_risk_level=RiskLevel.LOW,
                risk_score=0.0,
                signals=[],
                recommendations=[]
            )
        
        # Weight signals by severity and confidence
        severity_weights = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 2.5,
            RiskLevel.HIGH: 5.0,
            RiskLevel.CRITICAL: 10.0,
            RiskLevel.BLOCKED: 20.0
        }
        
        total_weight = 0
        weighted_score = 0
        
        for signal in signals:
            weight = severity_weights[signal.severity] * signal.confidence
            total_weight += weight
            
            # Convert severity to score
            severity_score = {
                RiskLevel.LOW: 20,
                RiskLevel.MEDIUM: 40,
                RiskLevel.HIGH: 70,
                RiskLevel.CRITICAL: 90,
                RiskLevel.BLOCKED: 100
            }[signal.severity]
            
            weighted_score += severity_score * weight
        
        # Calculate final score
        risk_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Apply user reputation modifier
        reputation_modifier = (100 - user_profile.trust_score) / 200  # 0 to 0.5
        risk_score = min(100, risk_score * (1 + reputation_modifier))
        
        # Determine risk level
        overall_risk_level = RiskLevel.LOW
        for level, threshold in sorted(self.risk_thresholds.items(), key=lambda x: x[1], reverse=True):
            if risk_score >= threshold:
                overall_risk_level = level
                break
        
        # Generate recommendations
        recommendations = self._generate_recommendations(signals, overall_risk_level)
        
        # Determine actions
        should_block = overall_risk_level == RiskLevel.BLOCKED
        should_warn = overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        requires_human_review = overall_risk_level in [RiskLevel.CRITICAL, RiskLevel.BLOCKED] or \
                               any(s.category in [RiskCategory.SELF_HARM, RiskCategory.MINOR_SAFETY] for s in signals)
        
        return RiskAssessment(
            overall_risk_level=overall_risk_level,
            risk_score=risk_score,
            signals=signals,
            recommendations=recommendations,
            requires_human_review=requires_human_review,
            should_block=should_block,
            should_warn=should_warn,
            metadata={
                'user_trust_score': user_profile.trust_score,
                'signal_count': len(signals),
                'highest_severity': max(signals, key=lambda s: severity_weights[s.severity]).severity if signals else None
            }
        )
    
    def _generate_recommendations(self, signals: List[RiskSignal], risk_level: RiskLevel) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Category-specific recommendations
        categories = set(s.category for s in signals)
        
        if RiskCategory.PERSONAL_INFO in categories:
            recommendations.append("Remove or redact personal information before proceeding")
        
        if RiskCategory.MEDICAL_ADVICE in categories:
            recommendations.append("Add disclaimer: Not medical advice, consult healthcare professional")
        
        if RiskCategory.FINANCIAL_ADVICE in categories:
            recommendations.append("Add disclaimer: Not financial advice, consult qualified advisor")
        
        if RiskCategory.SELF_HARM in categories:
            recommendations.append("Provide crisis resources and escalate to human support")
        
        if RiskCategory.SPAM in categories:
            recommendations.append("Apply rate limiting and monitor for continued spam")
        
        # Risk level recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Immediate human review required")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Consider temporary restrictions on user")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Monitor conversation closely")
        
        return recommendations
    
    def _get_pattern_severity(self, category: RiskCategory) -> RiskLevel:
        """Get default severity for risk category."""
        severity_map = {
            RiskCategory.SPAM: RiskLevel.LOW,
            RiskCategory.PERSONAL_INFO: RiskLevel.HIGH,
            RiskCategory.MEDICAL_ADVICE: RiskLevel.HIGH,
            RiskCategory.FINANCIAL_ADVICE: RiskLevel.HIGH,
            RiskCategory.SELF_HARM: RiskLevel.CRITICAL,
            RiskCategory.VIOLENCE: RiskLevel.CRITICAL,
            RiskCategory.HATE_SPEECH: RiskLevel.HIGH,
            RiskCategory.MINOR_SAFETY: RiskLevel.CRITICAL,
            RiskCategory.FRAUD: RiskLevel.HIGH,
            RiskCategory.MANIPULATION: RiskLevel.MEDIUM,
        }
        return severity_map.get(category, RiskLevel.MEDIUM)
    
    async def _get_user_recent_activity(self, user_id: str) -> Dict[str, Any]:
        """Get user's recent activity metrics."""
        try:
            if not self.redis:
                return {'message_count': 0, 'similarity_score': 0}
            
            # Get message count from Redis
            key = f"user_activity:{user_id}"
            activity = await self.redis.get(key)
            
            if activity:
                import json
                return json.loads(activity)
            
            return {'message_count': 0, 'similarity_score': 0}
            
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return {'message_count': 0, 'similarity_score': 0}
    
    def _calculate_escalation_score(self, context: List[Dict], current_message: str) -> float:
        """Calculate escalation score based on conversation progression."""
        if not context:
            return 0.0
        
        # Look for escalation keywords
        escalation_keywords = ['urgent', 'immediately', 'now', 'hurry', 'quick', 'asap', 'emergency']
        
        # Count escalation words in recent context
        escalation_count = 0
        for msg in context[-5:]:  # Last 5 messages
            content = msg.get('content', '').lower()
            escalation_count += sum(1 for keyword in escalation_keywords if keyword in content)
        
        # Check current message
        current_lower = current_message.lower()
        current_escalation = sum(1 for keyword in escalation_keywords if keyword in current_lower)
        
        # Calculate score (normalized)
        score = min(1.0, (escalation_count + current_escalation * 2) / 10)
        
        return score
    
    def _detect_topic_manipulation(self, context: List[Dict], current_message: str) -> int:
        """Detect suspicious topic shifts in conversation."""
        if not context or len(context) < 3:
            return 0
        
        # Simplified topic detection - in production would use NLP
        topic_keywords = {
            'payment': ['pay', 'money', 'transfer', 'wire', 'card', 'bank'],
            'personal': ['address', 'phone', 'email', 'name', 'birth', 'ssn'],
            'medical': ['health', 'doctor', 'medicine', 'symptom', 'treatment'],
            'general': []  # Default topic
        }
        
        # Track topic changes
        topics = []
        for msg in context[-5:]:
            content = msg.get('content', '').lower()
            detected_topic = 'general'
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    detected_topic = topic
                    break
            
            topics.append(detected_topic)
        
        # Count unique topics
        unique_topics = len(set(topics))
        
        return unique_topics - 1  # Number of topic changes
    
    async def get_user_risk_profile(self, user_id: str) -> UserRiskProfile:
        """Get or create user risk profile."""
        try:
            # Try to get from cache
            if self.redis:
                cache_key = f"user_risk_profile:{user_id}"
                cached = await self.redis.get(cache_key)
                if cached:
                    import json
                    data = json.loads(cached)
                    return UserRiskProfile(**data)
            
            # Get from database
            user = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = user.scalar_one_or_none()
            
            if user and hasattr(user, 'risk_profile'):
                return UserRiskProfile(**user.risk_profile)
            
            # Create default profile
            profile = UserRiskProfile(
                user_id=user_id,
                trust_score=50.0,  # Start neutral
                risk_level=RiskLevel.LOW
            )
            
            # Cache it
            if self.redis:
                import json
                await self.redis.setex(
                    f"user_risk_profile:{user_id}",
                    3600,  # 1 hour cache
                    json.dumps(profile.__dict__, default=str)
                )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user risk profile: {e}")
            return UserRiskProfile(
                user_id=user_id,
                trust_score=50.0,
                risk_level=RiskLevel.LOW
            )
    
    async def _log_risk_event(self, user_id: str, message: str, assessment: RiskAssessment):
        """Log high-risk events for audit."""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'message_hash': hashlib.sha256(message.encode()).hexdigest(),
                'risk_level': assessment.overall_risk_level.value,
                'risk_score': assessment.risk_score,
                'signals': [
                    {
                        'category': s.category.value,
                        'severity': s.severity.value,
                        'confidence': s.confidence
                    }
                    for s in assessment.signals
                ],
                'actions': {
                    'blocked': assessment.should_block,
                    'warned': assessment.should_warn,
                    'review_required': assessment.requires_human_review
                }
            }
            
            # Store in Redis for real-time monitoring
            if self.redis:
                import json
                await self.redis.lpush('risk_events', json.dumps(log_entry))
                await self.redis.ltrim('risk_events', 0, 9999)  # Keep last 10000 events
            
            logger.warning("High-risk event logged", **log_entry)
            
        except Exception as e:
            logger.error(f"Error logging risk event: {e}")
    
    async def _update_user_profile(self, user_id: str, assessment: RiskAssessment):
        """Update user profile based on risk assessment."""
        try:
            profile = await self.get_user_risk_profile(user_id)
            
            # Update message counts
            profile.total_messages += 1
            
            if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                profile.flagged_messages += 1
            
            if assessment.should_block:
                profile.blocked_messages += 1
                profile.last_violation = datetime.utcnow()
            
            # Update trust score (simple decay/growth model)
            if assessment.overall_risk_level == RiskLevel.LOW:
                # Slowly increase trust for good behavior
                profile.trust_score = min(100, profile.trust_score + 0.1)
            elif assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                # Decrease trust for violations
                profile.trust_score = max(0, profile.trust_score - 5)
            elif assessment.should_block:
                # Significant trust decrease for blocked content
                profile.trust_score = max(0, profile.trust_score - 10)
            
            # Update risk level based on trust score
            if profile.trust_score < 20:
                profile.risk_level = RiskLevel.HIGH
            elif profile.trust_score < 40:
                profile.risk_level = RiskLevel.MEDIUM
            else:
                profile.risk_level = RiskLevel.LOW
            
            profile.updated_at = datetime.utcnow()
            
            # Save to cache
            if self.redis:
                import json
                await self.redis.setex(
                    f"user_risk_profile:{user_id}",
                    3600,
                    json.dumps(profile.__dict__, default=str)
                )
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
    
    async def check_conversation_health(self, conversation_id: str) -> Dict[str, Any]:
        """Check overall health of a conversation."""
        try:
            # Get conversation messages
            messages = await self.db.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id
                ).order_by(Message.created_at.desc()).limit(50)
            )
            messages = messages.scalars().all()
            
            if not messages:
                return {'status': 'healthy', 'score': 100}
            
            # Analyze conversation patterns
            risk_signals = 0
            total_checks = 0
            
            # Check message frequency
            if len(messages) > 20:
                time_span = (messages[0].created_at - messages[-1].created_at).total_seconds()
                messages_per_minute = len(messages) / (time_span / 60) if time_span > 0 else 0
                
                if messages_per_minute > 2:
                    risk_signals += 1
                total_checks += 1
            
            # Check for escalation
            escalation_words = ['urgent', 'help', 'please', 'now', 'immediately']
            escalation_count = sum(
                1 for msg in messages[:10]
                if any(word in msg.content.lower() for word in escalation_words)
            )
            
            if escalation_count > 3:
                risk_signals += 1
            total_checks += 1
            
            # Calculate health score
            health_score = max(0, 100 - (risk_signals / total_checks * 100))
            
            return {
                'status': 'healthy' if health_score > 70 else 'at_risk',
                'score': health_score,
                'message_count': len(messages),
                'risk_indicators': risk_signals
            }
            
        except Exception as e:
            logger.error(f"Error checking conversation health: {e}")
            return {'status': 'unknown', 'score': 50}
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk management metrics."""
        try:
            metrics = {
                'total_assessments': 0,
                'risk_distribution': {
                    'low': 0,
                    'medium': 0,
                    'high': 0,
                    'critical': 0,
                    'blocked': 0
                },
                'top_risk_categories': {},
                'interventions': {
                    'warnings_issued': 0,
                    'messages_blocked': 0,
                    'reviews_requested': 0
                }
            }
            
            # Get metrics from Redis if available
            if self.redis:
                # Get recent risk events
                events = await self.redis.lrange('risk_events', 0, 999)
                
                for event_data in events:
                    import json
                    event = json.loads(event_data)
                    
                    metrics['total_assessments'] += 1
                    risk_level = event.get('risk_level', 'low')
                    metrics['risk_distribution'][risk_level] += 1
                    
                    # Count interventions
                    if event.get('actions', {}).get('warned'):
                        metrics['interventions']['warnings_issued'] += 1
                    if event.get('actions', {}).get('blocked'):
                        metrics['interventions']['messages_blocked'] += 1
                    if event.get('actions', {}).get('review_required'):
                        metrics['interventions']['reviews_requested'] += 1
                    
                    # Track categories
                    for signal in event.get('signals', []):
                        category = signal.get('category')
                        if category:
                            metrics['top_risk_categories'][category] = \
                                metrics['top_risk_categories'].get(category, 0) + 1
            
            # Sort top categories
            if metrics['top_risk_categories']:
                metrics['top_risk_categories'] = dict(
                    sorted(metrics['top_risk_categories'].items(),
                           key=lambda x: x[1], reverse=True)[:5]
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}


# Export main classes
__all__ = [
    'RiskManager',
    'RiskAssessment',
    'RiskLevel',
    'RiskCategory',
    'RiskSignal',
    'UserRiskProfile'
]