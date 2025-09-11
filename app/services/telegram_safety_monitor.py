"""
Telegram Safety Monitor
Advanced safety and anti-detection system for Telegram account management.
Provides real-time monitoring, risk assessment, and automated safety measures.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from pyrogram.errors import (
    FloodWait, UserDeactivated, UserNotMutualContact,
    ChatWriteForbidden, SlowmodeWait, MessageNotModified,
    PeerFlood, SpamWait, UserBannedInChannel
)

from app.models.telegram_account import TelegramAccount, AccountStatus, SafetyLevel, AccountSafetyEvent
from app.models.telegram_community import TelegramCommunity, CommunityStatus
from app.services.risk_manager import RiskManager
from app.database.repositories import DatabaseRepository


class ThreatLevel(str, Enum):
    """Threat level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyAction(str, Enum):
    """Safety action types"""
    CONTINUE = "continue"
    SLOW_DOWN = "slow_down"
    PAUSE = "pause"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyAlert:
    """Safety alert data structure"""
    threat_level: ThreatLevel
    alert_type: str
    description: str
    recommended_action: SafetyAction
    data: Dict[str, Any]
    timestamp: datetime


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    overall_risk_score: float
    risk_factors: List[Dict[str, Any]]
    threat_level: ThreatLevel
    recommended_actions: List[str]
    safe_to_continue: bool
    cooldown_period: Optional[int]  # seconds


@dataclass
class AccountHealth:
    """Real-time account health status"""
    is_healthy: bool
    health_score: float
    active_warnings: int
    recent_errors: List[str]
    daily_activity: Dict[str, int]
    recommendations: List[str]


class TelegramSafetyMonitor:
    """
    Advanced safety monitoring system that provides real-time protection
    against account bans, spam detection, and rate limiting.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        database: DatabaseRepository
    ):
        self.risk_manager = risk_manager
        self.database = database
        
        self.logger = logging.getLogger(__name__)
        
        # Safety thresholds and limits
        self.safety_thresholds = {
            "max_messages_per_hour": 20,
            "max_messages_per_day": 50,
            "max_groups_joined_per_day": 2,
            "max_dms_per_day": 5,
            "max_flood_waits_per_hour": 3,
            "max_consecutive_errors": 5,
            "min_delay_between_messages": 30,  # seconds
            "max_risk_score": 70.0,
            "critical_risk_score": 85.0
        }
        
        # Error pattern analysis
        self.error_patterns = {
            "flood_wait": {
                "severity": "high",
                "risk_increase": 15.0,
                "cooldown_multiplier": 2.0
            },
            "spam_wait": {
                "severity": "critical",
                "risk_increase": 25.0,
                "cooldown_multiplier": 3.0
            },
            "peer_flood": {
                "severity": "critical",
                "risk_increase": 30.0,
                "cooldown_multiplier": 4.0
            },
            "user_banned": {
                "severity": "critical",
                "risk_increase": 50.0,
                "cooldown_multiplier": 0  # Immediate stop
            }
        }
        
        # Natural behavior patterns for anti-detection
        self.behavior_patterns = {
            "typing_speeds": {
                "min_chars_per_second": 8,
                "max_chars_per_second": 25,
                "average": 15
            },
            "response_delays": {
                "min_seconds": 2,
                "max_seconds": 300,
                "typical_range": (10, 60)
            },
            "activity_patterns": {
                "peak_hours": [9, 12, 15, 18, 20],
                "low_hours": [1, 2, 3, 4, 5, 6],
                "weekend_multiplier": 0.7
            }
        }
        
        # Active monitoring state
        self.monitoring_active = False
        self.alert_queue = asyncio.Queue()
        self.last_activity_check = datetime.utcnow()
        self.error_history = []
    
    async def initialize(self, account_id: str):
        """Initialize safety monitor for specific account"""
        self.account_id = account_id
        self.monitoring_active = True
        
        # Start background monitoring tasks
        asyncio.create_task(self._continuous_monitoring())
        asyncio.create_task(self._alert_processor())
        asyncio.create_task(self._health_checker())
        
        self.logger.info(f"Safety monitor initialized for account {account_id}")
    
    async def check_safety_before_action(
        self,
        action_type: str,
        target_chat_id: Optional[int] = None,
        message_content: Optional[str] = None
    ) -> Tuple[bool, Optional[SafetyAlert]]:
        """
        Check if action is safe to perform before execution
        """
        
        try:
            account = await self.database.get_telegram_account(self.account_id)
            if not account:
                return False, SafetyAlert(
                    threat_level=ThreatLevel.CRITICAL,
                    alert_type="account_not_found",
                    description="Account not found in database",
                    recommended_action=SafetyAction.EMERGENCY_STOP,
                    data={},
                    timestamp=datetime.utcnow()
                )
            
            # Perform comprehensive risk assessment
            risk_assessment = await self._assess_current_risk(account, action_type)
            
            # Check if account is healthy
            health = await self._check_account_health(account)
            
            # Check specific action safety
            action_safety = await self._check_action_safety(
                account, action_type, target_chat_id, message_content
            )
            
            # Determine overall safety
            is_safe = (
                risk_assessment.safe_to_continue and
                health.is_healthy and
                action_safety["is_safe"]
            )
            
            # Generate alert if not safe
            alert = None
            if not is_safe:
                alert = SafetyAlert(
                    threat_level=risk_assessment.threat_level,
                    alert_type="safety_check_failed",
                    description=f"Action '{action_type}' blocked due to safety concerns",
                    recommended_action=self._determine_safety_action(risk_assessment),
                    data={
                        "risk_score": risk_assessment.overall_risk_score,
                        "health_score": health.health_score,
                        "action_type": action_type,
                        "specific_issues": action_safety.get("issues", [])
                    },
                    timestamp=datetime.utcnow()
                )
                
                # Queue alert for processing
                await self.alert_queue.put(alert)
            
            return is_safe, alert
            
        except Exception as e:
            self.logger.error(f"Error in safety check: {e}")
            return False, SafetyAlert(
                threat_level=ThreatLevel.HIGH,
                alert_type="safety_check_error",
                description=f"Safety check failed: {str(e)}",
                recommended_action=SafetyAction.PAUSE,
                data={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    async def handle_telegram_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> SafetyAction:
        """
        Handle Telegram API errors and determine appropriate response
        """
        
        error_type = type(error).__name__
        error_message = str(error)
        
        self.logger.warning(f"Handling Telegram error: {error_type} - {error_message}")
        
        # Record error in history
        error_record = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "timestamp": datetime.utcnow()
        }
        self.error_history.append(error_record)
        
        # Keep only recent errors
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.error_history = [
            e for e in self.error_history 
            if e["timestamp"] > cutoff_time
        ]
        
        # Analyze error pattern and determine response
        if isinstance(error, FloodWait):
            return await self._handle_flood_wait(error, context)
        elif isinstance(error, (SpamWait, PeerFlood)):
            return await self._handle_spam_related_error(error, context)
        elif isinstance(error, UserBannedInChannel):
            return await self._handle_ban_error(error, context)
        elif isinstance(error, (UserDeactivated, UserNotMutualContact)):
            return await self._handle_user_error(error, context)
        else:
            return await self._handle_general_error(error, context)
    
    async def get_optimal_timing(
        self,
        action_type: str,
        community_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get optimal timing for action based on safety and behavior patterns
        """
        
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        
        # Base delay calculation
        if action_type == "send_message":
            base_delay = random.uniform(10, 60)  # 10-60 seconds
        elif action_type == "join_group":
            base_delay = random.uniform(300, 1800)  # 5-30 minutes
        elif action_type == "send_dm":
            base_delay = random.uniform(120, 600)  # 2-10 minutes
        else:
            base_delay = random.uniform(30, 120)  # 30-120 seconds
        
        # Apply behavior pattern adjustments
        if current_hour in self.behavior_patterns["activity_patterns"]["low_hours"]:
            base_delay *= 2.0  # Slower during low activity hours
        elif current_hour in self.behavior_patterns["activity_patterns"]["peak_hours"]:
            base_delay *= 0.8  # Slightly faster during peak hours
        
        # Weekend adjustment
        if current_time.weekday() >= 5:  # Saturday or Sunday
            base_delay *= self.behavior_patterns["activity_patterns"]["weekend_multiplier"]
        
        # Community-specific adjustments
        if community_id:
            community = await self.database.get_telegram_community(community_id)
            if community and community.peak_activity_hours:
                current_hour_str = str(current_hour).zfill(2)
                if current_hour_str in community.peak_activity_hours:
                    base_delay *= 0.7  # Faster during community peak hours
        
        # Safety adjustments based on recent activity
        account = await self.database.get_telegram_account(self.account_id)
        if account:
            if account.risk_score > 50:
                base_delay *= 1.5  # Slower when risk is elevated
            if account.messages_sent_today > account.max_messages_per_day * 0.8:
                base_delay *= 2.0  # Much slower when approaching limits
        
        # Calculate typing time for messages
        typing_time = 0
        if action_type == "send_message" and "message_length" in locals():
            chars_per_second = random.uniform(
                self.behavior_patterns["typing_speeds"]["min_chars_per_second"],
                self.behavior_patterns["typing_speeds"]["max_chars_per_second"]
            )
            typing_time = message_length / chars_per_second
            typing_time += random.uniform(0.5, 2.0)  # Thinking pauses
        
        return {
            "recommended_delay": max(30, int(base_delay)),  # Minimum 30 seconds
            "typing_time": max(1, int(typing_time)),
            "optimal_window_start": current_time + timedelta(seconds=base_delay),
            "optimal_window_end": current_time + timedelta(seconds=base_delay * 1.5),
            "safety_score": await self._calculate_timing_safety_score(base_delay)
        }
    
    async def monitor_account_reputation(self, account_id: str) -> Dict[str, Any]:
        """
        Monitor account reputation across communities
        """
        
        account = await self.database.get_telegram_account(account_id)
        if not account:
            return {"error": "Account not found"}
        
        communities = await self.database.get_account_communities(account_id)
        
        reputation_data = {
            "overall_reputation": 0.0,
            "community_scores": [],
            "trend": "stable",
            "risk_factors": [],
            "recommendations": []
        }
        
        total_reputation = 0.0
        total_weight = 0.0
        
        for community in communities:
            if community.status == CommunityStatus.ACTIVE:
                weight = community.engagement_score / 100.0
                total_reputation += community.reputation_score * weight
                total_weight += weight
                
                reputation_data["community_scores"].append({
                    "community": community.title,
                    "score": community.reputation_score,
                    "trend": community.reputation_trend,
                    "warnings": community.warning_count
                })
                
                # Check for risk factors
                if community.warning_count > 2:
                    reputation_data["risk_factors"].append(
                        f"High warnings in {community.title}"
                    )
                
                if community.reputation_score < 30:
                    reputation_data["risk_factors"].append(
                        f"Low reputation in {community.title}"
                    )
        
        if total_weight > 0:
            reputation_data["overall_reputation"] = total_reputation / total_weight
        
        # Determine trend
        improving_count = sum(1 for c in communities if c.reputation_trend == "improving")
        declining_count = sum(1 for c in communities if c.reputation_trend == "declining")
        
        if improving_count > declining_count:
            reputation_data["trend"] = "improving"
        elif declining_count > improving_count:
            reputation_data["trend"] = "declining"
        
        # Generate recommendations
        if reputation_data["overall_reputation"] < 50:
            reputation_data["recommendations"].append("Focus on helpful, positive interactions")
        
        if len(reputation_data["risk_factors"]) > 2:
            reputation_data["recommendations"].append("Review and improve community engagement strategies")
        
        return reputation_data
    
    async def generate_safety_report(self, account_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive safety report
        """
        
        account = await self.database.get_telegram_account(account_id)
        if not account:
            return {"error": "Account not found"}
        
        # Get recent safety events
        recent_events = await self.database.get_recent_safety_events(
            account_id, hours=24
        )
        
        # Calculate metrics
        health = await self._check_account_health(account)
        risk_assessment = await self._assess_current_risk(account, "general")
        reputation = await self.monitor_account_reputation(account_id)
        
        # Analyze error patterns
        error_analysis = await self._analyze_error_patterns()
        
        # Generate recommendations
        recommendations = await self._generate_safety_recommendations(
            account, health, risk_assessment, error_analysis
        )
        
        report = {
            "account_id": account_id,
            "report_timestamp": datetime.utcnow().isoformat(),
            "health_status": {
                "is_healthy": health.is_healthy,
                "health_score": health.health_score,
                "active_warnings": health.active_warnings
            },
            "risk_assessment": {
                "overall_risk_score": risk_assessment.overall_risk_score,
                "threat_level": risk_assessment.threat_level,
                "safe_to_continue": risk_assessment.safe_to_continue
            },
            "reputation_status": reputation,
            "recent_activity": {
                "messages_sent_today": account.messages_sent_today,
                "groups_joined_today": account.groups_joined_today,
                "dms_sent_today": account.dms_sent_today,
                "last_activity": account.session_last_used.isoformat() if account.session_last_used else None
            },
            "safety_events": [
                {
                    "type": event.event_type,
                    "severity": event.severity,
                    "description": event.description,
                    "timestamp": event.created_at.isoformat()
                }
                for event in recent_events[-10:]  # Last 10 events
            ],
            "error_patterns": error_analysis,
            "recommendations": recommendations,
            "compliance_status": {
                "gdpr_compliant": account.gdpr_consent_given,
                "ai_disclosed": account.is_ai_disclosed,
                "privacy_policy_accepted": account.privacy_policy_accepted
            }
        }
        
        return report
    
    # Private methods for safety monitoring
    async def _assess_current_risk(
        self,
        account: TelegramAccount,
        action_type: str
    ) -> RiskAssessment:
        """Assess current risk level for account"""
        
        risk_factors = []
        base_risk = account.risk_score
        
        # Recent error frequency
        recent_errors = [e for e in self.error_history if e["timestamp"] > datetime.utcnow() - timedelta(hours=1)]
        if len(recent_errors) > 3:
            risk_factors.append({
                "factor": "high_error_frequency",
                "impact": 15.0,
                "description": f"{len(recent_errors)} errors in last hour"
            })
            base_risk += 15.0
        
        # Daily activity limits
        activity_ratio = account.messages_sent_today / account.max_messages_per_day
        if activity_ratio > 0.8:
            risk_factors.append({
                "factor": "approaching_daily_limit",
                "impact": 10.0,
                "description": f"Used {activity_ratio:.1%} of daily message limit"
            })
            base_risk += 10.0
        
        # Recent flood waits
        recent_flood_waits = await self.database.count_recent_safety_events(
            self.account_id, "flood_wait", hours=1
        )
        if recent_flood_waits > 1:
            risk_factors.append({
                "factor": "recent_flood_waits",
                "impact": 20.0,
                "description": f"{recent_flood_waits} flood waits in last hour"
            })
            base_risk += 20.0
        
        # Account warnings
        if account.spam_warnings > 2:
            risk_factors.append({
                "factor": "multiple_spam_warnings",
                "impact": 25.0,
                "description": f"{account.spam_warnings} spam warnings"
            })
            base_risk += 25.0
        
        # Determine threat level
        if base_risk < 30:
            threat_level = ThreatLevel.LOW
        elif base_risk < 60:
            threat_level = ThreatLevel.MEDIUM
        elif base_risk < 85:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.CRITICAL
        
        # Determine if safe to continue
        safe_to_continue = (
            base_risk < self.safety_thresholds["max_risk_score"] and
            account.status in [AccountStatus.ACTIVE, AccountStatus.WARMING_UP] and
            not account.daily_limits_reached
        )
        
        # Generate recommendations
        recommendations = []
        if base_risk > 50:
            recommendations.append("Reduce activity frequency")
        if recent_errors:
            recommendations.append("Wait for error conditions to clear")
        if activity_ratio > 0.7:
            recommendations.append("Approach daily limits with caution")
        
        # Calculate cooldown period if needed
        cooldown_period = None
        if not safe_to_continue:
            if threat_level == ThreatLevel.CRITICAL:
                cooldown_period = 3600  # 1 hour
            elif threat_level == ThreatLevel.HIGH:
                cooldown_period = 1800  # 30 minutes
            else:
                cooldown_period = 600   # 10 minutes
        
        return RiskAssessment(
            overall_risk_score=min(100.0, base_risk),
            risk_factors=risk_factors,
            threat_level=threat_level,
            recommended_actions=recommendations,
            safe_to_continue=safe_to_continue,
            cooldown_period=cooldown_period
        )
    
    async def _check_account_health(self, account: TelegramAccount) -> AccountHealth:
        """Check current account health status"""
        
        # Calculate health score
        health_score = 100.0
        
        # Deduct for risk factors
        health_score -= account.risk_score * 0.5
        health_score -= account.spam_warnings * 10
        health_score -= account.flood_wait_count * 5
        
        # Deduct for activity ratio
        if account.max_messages_per_day > 0:
            activity_ratio = account.messages_sent_today / account.max_messages_per_day
            if activity_ratio > 0.8:
                health_score -= (activity_ratio - 0.8) * 50
        
        # Count active warnings
        active_warnings = await self.database.count_recent_safety_events(
            self.account_id, None, hours=24
        )
        
        # Recent errors
        recent_errors = [e["error_type"] for e in self.error_history[-5:]]
        
        # Generate recommendations
        recommendations = []
        if health_score < 70:
            recommendations.append("Account health is declining - review recent activity")
        if active_warnings > 3:
            recommendations.append("Multiple recent warnings - exercise extra caution")
        if account.risk_score > 60:
            recommendations.append("High risk score - consider reducing activity")
        
        is_healthy = (
            health_score >= 70 and
            account.status in [AccountStatus.ACTIVE, AccountStatus.WARMING_UP] and
            active_warnings < 5
        )
        
        return AccountHealth(
            is_healthy=is_healthy,
            health_score=max(0.0, health_score),
            active_warnings=active_warnings,
            recent_errors=recent_errors,
            daily_activity={
                "messages": account.messages_sent_today,
                "groups": account.groups_joined_today,
                "dms": account.dms_sent_today
            },
            recommendations=recommendations
        )
    
    async def _check_action_safety(
        self,
        account: TelegramAccount,
        action_type: str,
        target_chat_id: Optional[int],
        message_content: Optional[str]
    ) -> Dict[str, Any]:
        """Check safety of specific action"""
        
        issues = []
        is_safe = True
        
        # Check message content safety
        if message_content and action_type == "send_message":
            content_safety = await self._analyze_message_safety(message_content)
            if not content_safety["is_safe"]:
                issues.extend(content_safety["issues"])
                is_safe = False
        
        # Check timing safety
        timing_safety = await self._check_timing_safety(action_type)
        if not timing_safety["is_safe"]:
            issues.extend(timing_safety["issues"])
            is_safe = False
        
        # Check community-specific safety
        if target_chat_id:
            community_safety = await self._check_community_safety(target_chat_id)
            if not community_safety["is_safe"]:
                issues.extend(community_safety["issues"])
                is_safe = False
        
        return {
            "is_safe": is_safe,
            "issues": issues,
            "confidence": 0.9 if is_safe else 0.3
        }
    
    # Error handling methods
    async def _handle_flood_wait(self, error: FloodWait, context: Dict[str, Any]) -> SafetyAction:
        """Handle FloodWait error"""
        
        wait_time = error.value
        
        # Record safety event
        await self._record_safety_event(
            "flood_wait",
            f"Flood wait: {wait_time} seconds",
            {"wait_time": wait_time, "context": context}
        )
        
        # Update account risk
        account = await self.database.get_telegram_account(self.account_id)
        if account:
            risk_increase = min(20.0, wait_time / 10.0)
            account.increment_risk_score(risk_increase, f"FloodWait: {wait_time}s")
            account.flood_wait_count += 1
            await self.database.update_telegram_account(account)
        
        # Determine action based on wait time
        if wait_time > 300:  # 5 minutes
            return SafetyAction.EMERGENCY_STOP
        elif wait_time > 60:  # 1 minute
            return SafetyAction.STOP
        else:
            return SafetyAction.PAUSE
    
    async def _handle_spam_related_error(self, error: Exception, context: Dict[str, Any]) -> SafetyAction:
        """Handle spam-related errors"""
        
        error_type = type(error).__name__
        
        await self._record_safety_event(
            "spam_error",
            f"Spam-related error: {error_type}",
            {"error_type": error_type, "context": context}
        )
        
        # Update account
        account = await self.database.get_telegram_account(self.account_id)
        if account:
            account.spam_warnings += 1
            account.increment_risk_score(30.0, f"Spam error: {error_type}")
            await self.database.update_telegram_account(account)
        
        return SafetyAction.EMERGENCY_STOP
    
    async def _handle_ban_error(self, error: UserBannedInChannel, context: Dict[str, Any]) -> SafetyAction:
        """Handle ban errors"""
        
        await self._record_safety_event(
            "user_banned",
            f"User banned in channel: {error}",
            {"context": context}
        )
        
        # Update account and community
        account = await self.database.get_telegram_account(self.account_id)
        if account:
            account.increment_risk_score(50.0, "User banned in channel")
            await self.database.update_telegram_account(account)
        
        # Update community status if applicable
        if "chat_id" in context:
            community = await self.database.get_community_by_chat_id(
                self.account_id, context["chat_id"]
            )
            if community:
                community.status = CommunityStatus.BANNED
                await self.database.update_telegram_community(community)
        
        return SafetyAction.EMERGENCY_STOP
    
    async def _handle_user_error(self, error: Exception, context: Dict[str, Any]) -> SafetyAction:
        """Handle user-related errors"""
        
        error_type = type(error).__name__
        
        await self._record_safety_event(
            "user_error",
            f"User error: {error_type}",
            {"error_type": error_type, "context": context}
        )
        
        return SafetyAction.CONTINUE  # Usually safe to continue with other operations
    
    async def _handle_general_error(self, error: Exception, context: Dict[str, Any]) -> SafetyAction:
        """Handle general errors"""
        
        error_type = type(error).__name__
        
        await self._record_safety_event(
            "general_error",
            f"General error: {error_type}",
            {"error_type": error_type, "error_message": str(error), "context": context}
        )
        
        # Count consecutive errors
        consecutive_errors = len([
            e for e in self.error_history[-5:] 
            if (datetime.utcnow() - e["timestamp"]).total_seconds() < 300
        ])
        
        if consecutive_errors >= self.safety_thresholds["max_consecutive_errors"]:
            return SafetyAction.PAUSE
        else:
            return SafetyAction.SLOW_DOWN
    
    # Background monitoring tasks
    async def _continuous_monitoring(self):
        """Continuous background monitoring"""
        
        while self.monitoring_active:
            try:
                # Perform periodic safety checks
                await self._periodic_safety_check()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processor(self):
        """Process safety alerts"""
        
        while self.monitoring_active:
            try:
                # Wait for alerts
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=60)
                
                # Process alert
                await self._process_safety_alert(alert)
                
            except asyncio.TimeoutError:
                continue  # No alerts to process
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
    
    async def _health_checker(self):
        """Periodic health checks"""
        
        while self.monitoring_active:
            try:
                account = await self.database.get_telegram_account(self.account_id)
                if account:
                    health = await self._check_account_health(account)
                    
                    if not health.is_healthy:
                        alert = SafetyAlert(
                            threat_level=ThreatLevel.MEDIUM,
                            alert_type="health_degraded",
                            description=f"Account health degraded: {health.health_score:.1f}",
                            recommended_action=SafetyAction.SLOW_DOWN,
                            data={"health": health.__dict__},
                            timestamp=datetime.utcnow()
                        )
                        await self.alert_queue.put(alert)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(300)
    
    # Helper methods
    async def _record_safety_event(
        self,
        event_type: str,
        description: str,
        data: Dict[str, Any]
    ):
        """Record safety event in database"""
        
        event = AccountSafetyEvent(
            account_id=self.account_id,
            event_type=event_type,
            severity="medium",
            description=description,
            data=data
        )
        
        await self.database.create_safety_event(event)
    
    def _determine_safety_action(self, risk_assessment: RiskAssessment) -> SafetyAction:
        """Determine appropriate safety action based on risk"""
        
        if risk_assessment.threat_level == ThreatLevel.CRITICAL:
            return SafetyAction.EMERGENCY_STOP
        elif risk_assessment.threat_level == ThreatLevel.HIGH:
            return SafetyAction.STOP
        elif risk_assessment.threat_level == ThreatLevel.MEDIUM:
            return SafetyAction.PAUSE
        else:
            return SafetyAction.SLOW_DOWN
    
    async def _calculate_timing_safety_score(self, delay: float) -> float:
        """Calculate safety score for timing"""
        
        # Longer delays are generally safer
        if delay > 300:  # 5 minutes
            return 0.9
        elif delay > 120:  # 2 minutes
            return 0.8
        elif delay > 60:   # 1 minute
            return 0.7
        elif delay > 30:   # 30 seconds
            return 0.6
        else:
            return 0.4  # Too fast, risky
    
    # Additional helper methods would continue here...
    # (Truncating for brevity but including the essential safety monitoring framework)