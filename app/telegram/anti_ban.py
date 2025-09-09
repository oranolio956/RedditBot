"""
Anti-Ban System for Telegram Bots

Advanced anti-detection measures including natural typing patterns,
behavior randomization, timing variations, and usage pattern analysis.
"""

import asyncio
import time
import random
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

import structlog
import numpy as np
from scipy import stats

from app.config import settings
from .rate_limiter import AdvancedRateLimiter
from .metrics import TelegramMetrics

logger = structlog.get_logger(__name__)


class BehaviorPattern(Enum):
    """Different behavior patterns for anti-detection."""
    HUMAN_LIKE = "human_like"
    CASUAL_USER = "casual_user"
    POWER_USER = "power_user"
    BUSINESS_HOURS = "business_hours"
    NIGHT_OWL = "night_owl"
    WEEKEND_WARRIOR = "weekend_warrior"


class RiskLevel(Enum):
    """Risk levels for ban detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UserBehaviorProfile:
    """User behavior profile for personalized anti-ban measures."""
    user_id: int
    first_interaction: float
    total_messages: int = 0
    average_response_time: float = 0.0
    typical_active_hours: List[int] = field(default_factory=list)
    message_length_distribution: List[int] = field(default_factory=list)
    command_usage_pattern: Dict[str, int] = field(default_factory=dict)
    interaction_frequency: float = 0.0
    last_seen: float = 0.0
    risk_score: float = 0.0
    pattern_type: BehaviorPattern = BehaviorPattern.HUMAN_LIKE
    
    # Anti-detection metrics
    typing_speed_variation: float = 0.2
    pause_patterns: List[float] = field(default_factory=list)
    burst_activity_count: int = 0
    suspicious_timing_count: int = 0


@dataclass
class TypingPattern:
    """Natural typing pattern simulation."""
    base_speed: float = 150  # Characters per second
    variation_range: Tuple[float, float] = (0.7, 1.5)  # Speed variation multiplier
    pause_probability: float = 0.15  # Probability of natural pause
    pause_duration_range: Tuple[float, float] = (0.3, 2.0)  # Pause duration
    thinking_pause_probability: float = 0.05  # Longer thinking pauses
    thinking_pause_range: Tuple[float, float] = (2.0, 8.0)
    
    # Error simulation (realistic typing)
    typo_probability: float = 0.02
    correction_delay_range: Tuple[float, float] = (0.5, 1.5)


@dataclass
class AntiDetectionMetrics:
    """Metrics for anti-detection effectiveness."""
    total_interactions: int = 0
    pattern_variations_applied: int = 0
    suspicious_activity_prevented: int = 0
    risk_escalations: int = 0
    successful_interventions: int = 0
    false_positive_rate: float = 0.0
    detection_confidence: float = 0.0


class AntiBanManager:
    """
    Comprehensive anti-ban system for Telegram bots.
    
    Features:
    - Natural typing simulation
    - Behavior pattern randomization
    - Risk assessment and mitigation
    - User profiling for personalized patterns
    - Activity pattern analysis
    - Suspicious behavior detection
    - Adaptive timing adjustments
    """
    
    def __init__(self, rate_limiter: AdvancedRateLimiter, metrics: TelegramMetrics):
        self.rate_limiter = rate_limiter
        self.metrics = metrics
        
        # User behavior tracking
        self.user_profiles: Dict[int, UserBehaviorProfile] = {}
        self.global_patterns: Dict[str, Any] = {}
        
        # Anti-detection settings
        self.typing_pattern = TypingPattern()
        self.behavior_patterns: Dict[BehaviorPattern, Dict[str, Any]] = {}
        
        # Risk assessment
        self.risk_thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.95
        }
        
        # Metrics tracking
        self.anti_detection_metrics = AntiDetectionMetrics()
        
        # Pattern analysis
        self._activity_windows: Dict[str, List[float]] = {}
        self._suspicious_patterns: Set[str] = set()
        
        # Randomization seeds for consistency
        self._daily_seed = None
        self._last_seed_update = None
    
    async def initialize(self) -> None:
        """Initialize anti-ban system."""
        try:
            # Setup behavior patterns
            await self._setup_behavior_patterns()
            
            # Initialize pattern analysis
            await self._initialize_pattern_analysis()
            
            # Load existing user profiles
            await self._load_user_profiles()
            
            # Setup daily randomization seed
            await self._update_daily_seed()
            
            logger.info("Anti-ban system initialized")
            
        except Exception as e:
            logger.error("Failed to initialize anti-ban system", error=str(e))
            raise
    
    async def _setup_behavior_patterns(self) -> None:
        """Setup different behavior patterns for randomization."""
        self.behavior_patterns = {
            BehaviorPattern.HUMAN_LIKE: {
                "response_delay_range": (1.0, 5.0),
                "typing_speed_multiplier": (0.8, 1.2),
                "pause_frequency": 0.15,
                "burst_probability": 0.05,
                "active_hours": list(range(7, 23)),  # 7 AM to 11 PM
                "message_length_preference": "medium",
            },
            
            BehaviorPattern.CASUAL_USER: {
                "response_delay_range": (2.0, 10.0),
                "typing_speed_multiplier": (0.6, 1.0),
                "pause_frequency": 0.25,
                "burst_probability": 0.02,
                "active_hours": list(range(9, 22)),  # 9 AM to 10 PM
                "message_length_preference": "short",
            },
            
            BehaviorPattern.POWER_USER: {
                "response_delay_range": (0.5, 3.0),
                "typing_speed_multiplier": (1.0, 1.5),
                "pause_frequency": 0.08,
                "burst_probability": 0.12,
                "active_hours": list(range(6, 24)),  # 6 AM to midnight
                "message_length_preference": "long",
            },
            
            BehaviorPattern.BUSINESS_HOURS: {
                "response_delay_range": (0.5, 2.0),
                "typing_speed_multiplier": (1.2, 1.4),
                "pause_frequency": 0.05,
                "burst_probability": 0.08,
                "active_hours": list(range(8, 18)),  # 8 AM to 6 PM
                "message_length_preference": "medium",
            },
            
            BehaviorPattern.NIGHT_OWL: {
                "response_delay_range": (3.0, 8.0),
                "typing_speed_multiplier": (0.7, 1.1),
                "pause_frequency": 0.20,
                "burst_probability": 0.03,
                "active_hours": list(range(18, 24)) + list(range(0, 6)),
                "message_length_preference": "long",
            },
            
            BehaviorPattern.WEEKEND_WARRIOR: {
                "response_delay_range": (1.0, 6.0),
                "typing_speed_multiplier": (0.8, 1.3),
                "pause_frequency": 0.18,
                "burst_probability": 0.15,
                "active_hours": list(range(10, 24)),
                "message_length_preference": "variable",
            },
        }
    
    async def _initialize_pattern_analysis(self) -> None:
        """Initialize pattern analysis components."""
        # Activity windows for different time scales
        self._activity_windows = {
            "hourly": [],
            "daily": [],
            "weekly": [],
        }
        
        # Suspicious pattern detection
        self._suspicious_patterns = {
            "rapid_fire_messages",
            "identical_timing",
            "non_human_response_times",
            "repetitive_patterns",
            "unusual_activity_spikes",
        }
    
    async def _load_user_profiles(self) -> None:
        """Load existing user profiles from storage."""
        try:
            # In production, this would load from Redis/database
            # For now, initialize empty profiles
            self.user_profiles = {}
            logger.info("User profiles initialized")
            
        except Exception as e:
            logger.error("Failed to load user profiles", error=str(e))
    
    async def _update_daily_seed(self) -> None:
        """Update daily randomization seed for consistent patterns."""
        today = datetime.now().date()
        
        if self._last_seed_update != today:
            # Create deterministic but unpredictable seed based on date
            seed_string = f"{today}_{settings.telegram.bot_token[-8:]}"
            self._daily_seed = hash(seed_string) % (2**32)
            self._last_seed_update = today
            
            # Use the seed for numpy random state
            np.random.seed(self._daily_seed)
            
            logger.info(f"Updated daily randomization seed for {today}")
    
    async def get_user_profile(self, user_id: int) -> UserBehaviorProfile:
        """Get or create user behavior profile."""
        if user_id not in self.user_profiles:
            # Create new profile with randomized pattern
            pattern = await self._assign_behavior_pattern(user_id)
            
            self.user_profiles[user_id] = UserBehaviorProfile(
                user_id=user_id,
                first_interaction=time.time(),
                pattern_type=pattern
            )
            
            logger.info(f"Created new user profile for {user_id} with pattern {pattern.value}")
        
        return self.user_profiles[user_id]
    
    async def _assign_behavior_pattern(self, user_id: int) -> BehaviorPattern:
        """Assign behavior pattern based on user characteristics."""
        # Use deterministic randomization based on user ID and daily seed
        user_seed = (user_id + self._daily_seed) % (2**32)
        random.seed(user_seed)
        
        # Weight patterns based on realism
        pattern_weights = {
            BehaviorPattern.HUMAN_LIKE: 0.4,
            BehaviorPattern.CASUAL_USER: 0.25,
            BehaviorPattern.POWER_USER: 0.15,
            BehaviorPattern.BUSINESS_HOURS: 0.1,
            BehaviorPattern.NIGHT_OWL: 0.08,
            BehaviorPattern.WEEKEND_WARRIOR: 0.02,
        }
        
        patterns = list(pattern_weights.keys())
        weights = list(pattern_weights.values())
        
        return random.choices(patterns, weights=weights)[0]
    
    async def calculate_typing_delay(
        self,
        text: str,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate natural typing delay with anti-detection measures.
        
        Now integrates with advanced typing simulator for enhanced realism.
        """
        try:
            # Try to use advanced typing simulator if available
            try:
                from app.services.typing_integration import typing_integration
                
                if typing_integration and typing_integration.enable_advanced_simulation:
                    # Determine risk level from context
                    risk_level = None
                    if context:
                        risk_str = context.get('risk_level', 'low')
                        from .anti_ban import RiskLevel
                        risk_level = RiskLevel(risk_str) if risk_str in ['low', 'medium', 'high', 'critical'] else RiskLevel.LOW
                    
                    # Get enhanced delay calculation
                    enhanced_delay = await typing_integration.calculate_typing_delay_enhanced(
                        text=text,
                        user_id=user_id,
                        context=context,
                        risk_level=risk_level
                    )
                    
                    if enhanced_delay > 0:
                        return enhanced_delay
            
            except Exception as e:
                logger.debug("Advanced typing simulation unavailable, using fallback", error=str(e))
            
            # Original calculation as fallback
            profile = await self.get_user_profile(user_id)
            pattern_config = self.behavior_patterns[profile.pattern_type]
            
            # Base typing calculation
            char_count = len(text)
            base_delay = char_count / self.typing_pattern.base_speed
            
            # Apply user-specific speed variation
            speed_multiplier = random.uniform(*pattern_config["typing_speed_multiplier"])
            delay = base_delay * speed_multiplier
            
            # Add natural pauses
            if random.random() < pattern_config["pause_frequency"]:
                if random.random() < self.typing_pattern.thinking_pause_probability:
                    # Thinking pause
                    pause = random.uniform(*self.typing_pattern.thinking_pause_range)
                else:
                    # Normal pause
                    pause = random.uniform(*self.typing_pattern.pause_duration_range)
                
                delay += pause
                profile.pause_patterns.append(pause)
            
            # Context-based adjustments
            if context:
                if context.get("is_command"):
                    # Commands are typed faster
                    delay *= 0.7
                elif context.get("is_long_response"):
                    # Longer responses have more variation
                    delay *= random.uniform(0.8, 1.4)
                elif context.get("is_error_correction"):
                    # Error corrections take longer
                    delay += random.uniform(*self.typing_pattern.correction_delay_range)
            
            # Apply anti-detection randomization
            delay = await self._apply_anti_detection_timing(delay, profile)
            
            # Update profile
            profile.typing_speed_variation = np.std(profile.pause_patterns[-10:]) if len(profile.pause_patterns) >= 10 else 0.2
            
            # Ensure reasonable bounds
            return max(0.3, min(delay, 15.0))  # 0.3s to 15s
            
        except Exception as e:
            logger.error("Error calculating typing delay", error=str(e))
            return random.uniform(1.0, 3.0)  # Fallback
    
    async def _apply_anti_detection_timing(
        self,
        base_delay: float,
        profile: UserBehaviorProfile
    ) -> float:
        """Apply anti-detection timing variations."""
        # Avoid too-regular patterns
        variation = profile.typing_speed_variation
        if variation < 0.1:  # Too consistent, add variation
            multiplier = random.uniform(0.7, 1.6)
            profile.typing_speed_variation += 0.05
        else:
            multiplier = random.gauss(1.0, variation)
            multiplier = max(0.5, min(multiplier, 2.0))
        
        # Apply time-of-day effects
        current_hour = datetime.now().hour
        pattern_config = self.behavior_patterns[profile.pattern_type]
        
        if current_hour in pattern_config["active_hours"]:
            # Active hours - normal speed
            time_multiplier = 1.0
        else:
            # Off hours - slower, more variation
            time_multiplier = random.uniform(1.2, 2.0)
        
        return base_delay * multiplier * time_multiplier
    
    async def assess_risk_level(
        self,
        user_id: int,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[RiskLevel, float, Dict[str, Any]]:
        """Assess risk level for a user action."""
        try:
            profile = await self.get_user_profile(user_id)
            risk_factors = {}
            total_risk = 0.0
            
            # Analyze timing patterns
            timing_risk, timing_factors = await self._analyze_timing_risk(profile, action)
            risk_factors.update(timing_factors)
            total_risk += timing_risk * 0.3
            
            # Analyze frequency patterns
            frequency_risk, frequency_factors = await self._analyze_frequency_risk(profile, action)
            risk_factors.update(frequency_factors)
            total_risk += frequency_risk * 0.25
            
            # Analyze behavior consistency
            behavior_risk, behavior_factors = await self._analyze_behavior_consistency(profile)
            risk_factors.update(behavior_factors)
            total_risk += behavior_risk * 0.2
            
            # Analyze interaction patterns
            interaction_risk, interaction_factors = await self._analyze_interaction_patterns(profile, context)
            risk_factors.update(interaction_factors)
            total_risk += interaction_risk * 0.15
            
            # Global pattern analysis
            global_risk, global_factors = await self._analyze_global_patterns(user_id, action)
            risk_factors.update(global_factors)
            total_risk += global_risk * 0.1
            
            # Determine risk level
            risk_level = RiskLevel.LOW
            for level, threshold in reversed(self.risk_thresholds.items()):
                if total_risk >= threshold:
                    risk_level = level
                    break
            
            # Update profile risk score
            profile.risk_score = total_risk
            
            # Log high-risk activities
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(
                    f"High risk activity detected",
                    user_id=user_id,
                    action=action,
                    risk_level=risk_level.value,
                    risk_score=total_risk,
                    factors=risk_factors
                )
                
                self.anti_detection_metrics.risk_escalations += 1
            
            return risk_level, total_risk, risk_factors
            
        except Exception as e:
            logger.error("Risk assessment failed", error=str(e))
            return RiskLevel.LOW, 0.0, {}
    
    async def _analyze_timing_risk(
        self,
        profile: UserBehaviorProfile,
        action: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze timing-based risk factors."""
        risk = 0.0
        factors = {}
        
        now = time.time()
        
        # Check for too-rapid responses
        if profile.last_seen > 0:
            time_since_last = now - profile.last_seen
            if time_since_last < 0.5:  # Less than 500ms
                risk += 0.4
                factors["rapid_response"] = time_since_last
                profile.suspicious_timing_count += 1
        
        # Check for non-human response patterns
        if len(profile.pause_patterns) >= 5:
            pause_std = np.std(profile.pause_patterns[-5:])
            if pause_std < 0.1:  # Too consistent
                risk += 0.3
                factors["consistent_timing"] = pause_std
        
        # Check time of day patterns
        current_hour = datetime.now().hour
        pattern_config = self.behavior_patterns[profile.pattern_type]
        if current_hour not in pattern_config["active_hours"]:
            # Activity outside typical hours
            risk += 0.1
            factors["off_hours_activity"] = current_hour
        
        profile.last_seen = now
        return risk, factors
    
    async def _analyze_frequency_risk(
        self,
        profile: UserBehaviorProfile,
        action: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze frequency-based risk factors."""
        risk = 0.0
        factors = {}
        
        # Analyze recent activity bursts
        recent_window = 300  # 5 minutes
        now = time.time()
        
        # Count recent interactions
        recent_count = sum(
            1 for timestamp in getattr(profile, '_recent_activities', [])
            if now - timestamp < recent_window
        )
        
        if recent_count > 20:  # More than 20 actions in 5 minutes
            risk += 0.5
            factors["activity_burst"] = recent_count
            profile.burst_activity_count += 1
        
        # Check command frequency
        if action in profile.command_usage_pattern:
            command_count = profile.command_usage_pattern[action]
            total_commands = sum(profile.command_usage_pattern.values())
            
            if total_commands > 0:
                command_ratio = command_count / total_commands
                if command_ratio > 0.6:  # More than 60% of one command
                    risk += 0.3
                    factors["command_repetition"] = command_ratio
        
        return risk, factors
    
    async def _analyze_behavior_consistency(
        self,
        profile: UserBehaviorProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze behavior consistency risk factors."""
        risk = 0.0
        factors = {}
        
        # Check if behavior matches assigned pattern
        expected_pattern = self.behavior_patterns[profile.pattern_type]
        
        # Analyze response time consistency
        if profile.average_response_time > 0:
            expected_range = expected_pattern["response_delay_range"]
            if not (expected_range[0] <= profile.average_response_time <= expected_range[1] * 2):
                risk += 0.2
                factors["response_time_mismatch"] = profile.average_response_time
        
        # Check typing speed variations
        if profile.typing_speed_variation < 0.05:  # Too mechanical
            risk += 0.25
            factors["mechanical_typing"] = profile.typing_speed_variation
        
        return risk, factors
    
    async def _analyze_interaction_patterns(
        self,
        profile: UserBehaviorProfile,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze interaction pattern risk factors."""
        risk = 0.0
        factors = {}
        
        if not context:
            return risk, factors
        
        # Check for bot-like interaction patterns
        if context.get("immediate_command_response"):
            risk += 0.15
            factors["immediate_command"] = True
        
        if context.get("identical_message_timing"):
            risk += 0.3
            factors["identical_timing"] = True
        
        # Check message complexity vs response time
        message_length = context.get("message_length", 0)
        response_time = context.get("response_time", 0)
        
        if message_length > 100 and response_time < 2.0:
            # Long message, fast response - suspicious
            risk += 0.2
            factors["fast_complex_response"] = {
                "length": message_length,
                "time": response_time
            }
        
        return risk, factors
    
    async def _analyze_global_patterns(
        self,
        user_id: int,
        action: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze global pattern risk factors."""
        risk = 0.0
        factors = {}
        
        # Check for globally suspicious patterns
        now = time.time()
        hour_key = f"hour_{int(now // 3600)}"
        
        if hour_key not in self._activity_windows:
            self._activity_windows[hour_key] = []
        
        self._activity_windows[hour_key].append((user_id, action, now))
        
        # Analyze recent global activity
        recent_activities = [
            activity for activities in self._activity_windows.values()
            for activity in activities
            if now - activity[2] < 3600  # Last hour
        ]
        
        # Check for coordinated activity (multiple users doing same thing)
        action_counts = {}
        for _, act, _ in recent_activities:
            action_counts[act] = action_counts.get(act, 0) + 1
        
        if action_counts.get(action, 0) > 50:  # More than 50 of same action in hour
            risk += 0.1
            factors["high_global_frequency"] = action_counts[action]
        
        return risk, factors
    
    async def apply_risk_mitigation(
        self,
        user_id: int,
        risk_level: RiskLevel,
        risk_score: float,
        action: str
    ) -> Dict[str, Any]:
        """Apply risk mitigation measures."""
        mitigation_actions = {}
        
        try:
            profile = await self.get_user_profile(user_id)
            
            if risk_level == RiskLevel.CRITICAL:
                # Critical risk - aggressive mitigation
                mitigation_actions["force_delay"] = random.uniform(10.0, 30.0)
                mitigation_actions["require_captcha"] = True
                mitigation_actions["limit_interactions"] = True
                mitigation_actions["enhanced_monitoring"] = True
                
                # Reset some behavioral patterns
                profile.typing_speed_variation += 0.1
                profile.suspicious_timing_count = 0
                
                logger.warning(f"Applied critical risk mitigation for user {user_id}")
                self.anti_detection_metrics.successful_interventions += 1
                
            elif risk_level == RiskLevel.HIGH:
                # High risk - moderate mitigation
                mitigation_actions["force_delay"] = random.uniform(3.0, 8.0)
                mitigation_actions["add_typing_variation"] = True
                mitigation_actions["enhanced_monitoring"] = True
                
                # Adjust behavior patterns
                if profile.typing_speed_variation < 0.15:
                    profile.typing_speed_variation = 0.15
                
                logger.info(f"Applied high risk mitigation for user {user_id}")
                self.anti_detection_metrics.successful_interventions += 1
                
            elif risk_level == RiskLevel.MEDIUM:
                # Medium risk - light mitigation
                mitigation_actions["force_delay"] = random.uniform(1.0, 3.0)
                mitigation_actions["add_typing_variation"] = True
                
                # Slight behavior adjustment
                profile.typing_speed_variation = max(0.1, profile.typing_speed_variation)
                
            # Always apply pattern randomization for any risk above LOW
            if risk_level != RiskLevel.LOW:
                mitigation_actions["randomize_patterns"] = True
                self.anti_detection_metrics.pattern_variations_applied += 1
            
            return mitigation_actions
            
        except Exception as e:
            logger.error("Risk mitigation failed", error=str(e))
            return {"force_delay": 1.0}  # Minimal fallback
    
    async def update_patterns(self) -> None:
        """Update behavioral patterns based on learning."""
        try:
            # Update daily seed
            await self._update_daily_seed()
            
            # Analyze effectiveness
            await self._analyze_detection_effectiveness()
            
            # Update global patterns
            await self._update_global_patterns()
            
            logger.info("Anti-ban patterns updated")
            
        except Exception as e:
            logger.error("Pattern update failed", error=str(e))
    
    async def _analyze_detection_effectiveness(self) -> None:
        """Analyze effectiveness of anti-detection measures."""
        try:
            # Calculate metrics
            total_interactions = self.anti_detection_metrics.total_interactions
            if total_interactions > 0:
                risk_rate = self.anti_detection_metrics.risk_escalations / total_interactions
                intervention_rate = self.anti_detection_metrics.successful_interventions / total_interactions
                
                # Update confidence based on success
                self.anti_detection_metrics.detection_confidence = min(1.0, 
                    0.5 + (intervention_rate * 0.5) - (risk_rate * 0.3)
                )
            
            # Adjust thresholds if needed
            if self.anti_detection_metrics.risk_escalations > 100:
                # Too many escalations, lower thresholds
                for level in self.risk_thresholds:
                    self.risk_thresholds[level] *= 0.95
            
        except Exception as e:
            logger.error("Detection effectiveness analysis failed", error=str(e))
    
    async def _update_global_patterns(self) -> None:
        """Update global activity patterns."""
        try:
            # Clean old activity data
            now = time.time()
            cutoff = now - 86400  # 24 hours
            
            for window_key in list(self._activity_windows.keys()):
                self._activity_windows[window_key] = [
                    activity for activity in self._activity_windows[window_key]
                    if activity[2] > cutoff
                ]
                
                # Remove empty windows
                if not self._activity_windows[window_key]:
                    del self._activity_windows[window_key]
            
        except Exception as e:
            logger.error("Global pattern update failed", error=str(e))
    
    async def cleanup_old_data(self) -> None:
        """Clean up old anti-ban data."""
        try:
            now = time.time()
            cutoff = now - 86400 * 7  # 7 days
            
            # Clean user profiles
            expired_users = []
            for user_id, profile in self.user_profiles.items():
                if profile.last_seen < cutoff:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.user_profiles[user_id]
            
            # Reset metrics periodically
            if now % 86400 < 3600:  # Once per day
                self.anti_detection_metrics = AntiDetectionMetrics()
            
            if expired_users:
                logger.info(f"Cleaned {len(expired_users)} expired user profiles")
            
        except Exception as e:
            logger.error("Anti-ban data cleanup failed", error=str(e))
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get anti-ban system metrics."""
        try:
            return {
                "total_users": len(self.user_profiles),
                "behavior_patterns": len(self.behavior_patterns),
                "risk_thresholds": {level.value: threshold for level, threshold in self.risk_thresholds.items()},
                "detection_metrics": {
                    "total_interactions": self.anti_detection_metrics.total_interactions,
                    "pattern_variations_applied": self.anti_detection_metrics.pattern_variations_applied,
                    "suspicious_activity_prevented": self.anti_detection_metrics.suspicious_activity_prevented,
                    "risk_escalations": self.anti_detection_metrics.risk_escalations,
                    "successful_interventions": self.anti_detection_metrics.successful_interventions,
                    "detection_confidence": self.anti_detection_metrics.detection_confidence,
                },
                "pattern_distribution": {
                    pattern.value: sum(1 for p in self.user_profiles.values() if p.pattern_type == pattern)
                    for pattern in BehaviorPattern
                },
                "activity_windows": len(self._activity_windows),
                "daily_seed": self._daily_seed,
                "last_seed_update": str(self._last_seed_update) if self._last_seed_update else None,
            }
            
        except Exception as e:
            logger.error("Failed to get anti-ban metrics", error=str(e))
            return {}
    
    async def cleanup(self) -> None:
        """Clean up anti-ban resources."""
        try:
            self.user_profiles.clear()
            self.global_patterns.clear()
            self._activity_windows.clear()
            self._suspicious_patterns.clear()
            
            logger.info("Anti-ban system cleanup completed")
            
        except Exception as e:
            logger.error("Error during anti-ban cleanup", error=str(e))