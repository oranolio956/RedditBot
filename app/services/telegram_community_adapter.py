"""
Telegram Community Adapter
Advanced system for adapting AI behavior to different Telegram communities.
Provides community-specific personality, engagement strategies, and content optimization.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

from app.models.telegram_community import (
    TelegramCommunity, CommunityType, EngagementStrategy, 
    CommunityStatus, CommunityEngagementEvent, CommunityInsight
)
from app.models.telegram_conversation import TelegramConversation
from app.services.consciousness_mirror import ConsciousnessMirror
from app.services.memory_palace import MemoryPalace
from app.services.behavioral_predictor import BehavioralPredictor
from app.services.engagement_analyzer import EngagementAnalyzer
from app.database.repositories import DatabaseRepository


@dataclass
class CommunityProfile:
    """Comprehensive community behavioral profile"""
    activity_patterns: Dict[str, Any]
    communication_style: Dict[str, Any]
    content_preferences: Dict[str, Any]
    member_dynamics: Dict[str, Any]
    moderation_patterns: Dict[str, Any]
    optimal_engagement_times: List[str]


@dataclass
class AdaptationStrategy:
    """Specific adaptation strategy for a community"""
    personality_adjustments: Dict[str, float]
    communication_guidelines: List[str]
    content_suggestions: List[str]
    timing_recommendations: Dict[str, Any]
    risk_factors: List[str]
    success_metrics: Dict[str, float]


class TelegramCommunityAdapter:
    """
    Advanced community adaptation system that learns and adapts to different
    Telegram community dynamics for optimal engagement and authenticity.
    """
    
    def __init__(
        self,
        consciousness_mirror: ConsciousnessMirror,
        memory_palace: MemoryPalace,
        behavioral_predictor: BehavioralPredictor,
        engagement_analyzer: EngagementAnalyzer,
        database: DatabaseRepository
    ):
        self.consciousness = consciousness_mirror
        self.memory = memory_palace
        self.behavioral_predictor = behavioral_predictor
        self.engagement_analyzer = engagement_analyzer
        self.database = database
        
        self.logger = logging.getLogger(__name__)
        
        # Community analysis patterns
        self.analysis_patterns = {
            "activity_indicators": [
                "message_frequency", "response_speed", "member_participation",
                "peak_hours", "content_types", "engagement_levels"
            ],
            "communication_styles": [
                "formality_level", "humor_acceptance", "technical_depth",
                "emoji_usage", "slang_prevalence", "discussion_length"
            ],
            "content_preferences": [
                "topic_diversity", "media_types", "link_sharing",
                "question_frequency", "opinion_sharing", "fact_checking"
            ]
        }
        
        # Engagement strategy templates
        self.strategy_templates = {
            EngagementStrategy.LURKER: {
                "message_frequency": 0.1,  # Very low
                "response_probability": 0.05,
                "proactive_engagement": False,
                "observation_focus": True
            },
            EngagementStrategy.PARTICIPANT: {
                "message_frequency": 0.3,
                "response_probability": 0.25,
                "proactive_engagement": True,
                "helpful_responses": True
            },
            EngagementStrategy.CONTRIBUTOR: {
                "message_frequency": 0.6,
                "response_probability": 0.4,
                "knowledge_sharing": True,
                "question_asking": True
            },
            EngagementStrategy.LEADER: {
                "message_frequency": 0.8,
                "response_probability": 0.6,
                "initiative_taking": True,
                "guidance_providing": True
            }
        }
    
    async def initialize(self, account_id: str):
        """Initialize community adapter for specific account"""
        self.account_id = account_id
        await self.consciousness.initialize(account_id)
        await self.memory.initialize(f"communities_{account_id}")
        
        self.logger.info(f"Community adapter initialized for account {account_id}")
    
    async def analyze_community(self, community_id: str) -> CommunityProfile:
        """
        Perform comprehensive analysis of community behavior and dynamics
        """
        
        community = await self.database.get_telegram_community(community_id)
        if not community:
            raise ValueError(f"Community {community_id} not found")
        
        # Gather community data
        recent_conversations = await self.database.get_community_conversations(
            community_id, days=30
        )
        engagement_events = await self.database.get_community_engagement_events(
            community_id, days=30
        )
        
        # Analyze activity patterns
        activity_patterns = await self._analyze_activity_patterns(
            recent_conversations, engagement_events
        )
        
        # Analyze communication style
        communication_style = await self._analyze_communication_style(
            recent_conversations
        )
        
        # Analyze content preferences
        content_preferences = await self._analyze_content_preferences(
            recent_conversations, engagement_events
        )
        
        # Analyze member dynamics
        member_dynamics = await self._analyze_member_dynamics(
            community, recent_conversations
        )
        
        # Analyze moderation patterns
        moderation_patterns = await self._analyze_moderation_patterns(
            community, engagement_events
        )
        
        # Determine optimal engagement times
        optimal_times = await self._determine_optimal_engagement_times(
            activity_patterns
        )
        
        profile = CommunityProfile(
            activity_patterns=activity_patterns,
            communication_style=communication_style,
            content_preferences=content_preferences,
            member_dynamics=member_dynamics,
            moderation_patterns=moderation_patterns,
            optimal_engagement_times=optimal_times
        )
        
        # Store profile insights
        await self._store_community_profile(community, profile)
        
        return profile
    
    async def adapt_personality_for_community(
        self,
        community_id: str,
        base_personality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt base personality for specific community context
        """
        
        community = await self.database.get_telegram_community(community_id)
        if not community:
            return base_personality
        
        # Get community profile
        profile = await self.analyze_community(community_id)
        
        # Start with base personality
        adapted_personality = base_personality.copy()
        
        # Apply community-specific adaptations
        adaptations = await self._calculate_personality_adaptations(
            community, profile
        )
        
        for trait, adjustment in adaptations.items():
            if trait in adapted_personality:
                current_value = adapted_personality[trait]
                adapted_personality[trait] = max(0.0, min(1.0, current_value + adjustment))
        
        # Store adapted personality in community record
        community.community_personality = adapted_personality
        await self.database.update_telegram_community(community)
        
        return adapted_personality
    
    async def get_engagement_strategy(
        self,
        community_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AdaptationStrategy:
        """
        Get specific engagement strategy for community
        """
        
        community = await self.database.get_telegram_community(community_id)
        if not community:
            raise ValueError(f"Community {community_id} not found")
        
        # Get community profile
        profile = await self.analyze_community(community_id)
        
        # Get base strategy template
        base_strategy = self.strategy_templates.get(
            community.engagement_strategy, 
            self.strategy_templates[EngagementStrategy.PARTICIPANT]
        )
        
        # Calculate personality adjustments
        personality_adjustments = await self._calculate_personality_adaptations(
            community, profile
        )
        
        # Generate communication guidelines
        communication_guidelines = await self._generate_communication_guidelines(
            community, profile
        )
        
        # Generate content suggestions
        content_suggestions = await self._generate_content_suggestions(
            community, profile, context
        )
        
        # Generate timing recommendations
        timing_recommendations = await self._generate_timing_recommendations(
            profile
        )
        
        # Identify risk factors
        risk_factors = await self._identify_risk_factors(community, profile)
        
        # Calculate success metrics
        success_metrics = await self._calculate_success_metrics(community)
        
        strategy = AdaptationStrategy(
            personality_adjustments=personality_adjustments,
            communication_guidelines=communication_guidelines,
            content_suggestions=content_suggestions,
            timing_recommendations=timing_recommendations,
            risk_factors=risk_factors,
            success_metrics=success_metrics
        )
        
        # Store strategy insights
        await self._store_strategy_insights(community, strategy)
        
        return strategy
    
    async def optimize_message_for_community(
        self,
        message: str,
        community_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Optimize message content for specific community
        """
        
        community = await self.database.get_telegram_community(community_id)
        if not community:
            return message
        
        profile = await self.analyze_community(community_id)
        
        # Apply community-specific optimizations
        optimized_message = message
        
        # Formality adjustments
        if profile.communication_style.get("formality_level", "medium") == "high":
            optimized_message = await self._increase_formality(optimized_message)
        elif profile.communication_style.get("formality_level", "medium") == "low":
            optimized_message = await self._decrease_formality(optimized_message)
        
        # Emoji adjustments
        emoji_usage = profile.communication_style.get("emoji_usage", 0.5)
        if emoji_usage > 0.7 and "😊" not in optimized_message and "👍" not in optimized_message:
            optimized_message = await self._add_appropriate_emoji(optimized_message)
        elif emoji_usage < 0.3:
            optimized_message = await self._remove_emoji(optimized_message)
        
        # Length adjustments
        avg_length = profile.communication_style.get("average_message_length", 100)
        if len(optimized_message) > avg_length * 1.5:
            optimized_message = await self._shorten_message(optimized_message, avg_length)
        
        # Technical depth adjustments
        tech_level = profile.communication_style.get("technical_depth", 0.5)
        if tech_level < 0.3:
            optimized_message = await self._simplify_technical_language(optimized_message)
        
        return optimized_message
    
    async def predict_engagement_success(
        self,
        message: str,
        community_id: str,
        timing: datetime
    ) -> float:
        """
        Predict likelihood of successful engagement for message
        """
        
        community = await self.database.get_telegram_community(community_id)
        if not community:
            return 0.5
        
        profile = await self.analyze_community(community_id)
        
        # Factors affecting engagement success
        factors = {
            "timing": 0.0,
            "content": 0.0,
            "length": 0.0,
            "style": 0.0,
            "relevance": 0.0
        }
        
        # Timing factor
        optimal_times = profile.optimal_engagement_times
        current_hour = timing.strftime("%H")
        if current_hour in optimal_times:
            factors["timing"] = 0.8
        else:
            factors["timing"] = 0.3
        
        # Content factor
        preferred_topics = profile.content_preferences.get("popular_topics", [])
        if any(topic.lower() in message.lower() for topic in preferred_topics):
            factors["content"] = 0.7
        else:
            factors["content"] = 0.4
        
        # Length factor
        optimal_length = profile.communication_style.get("average_message_length", 100)
        length_ratio = len(message) / optimal_length
        if 0.5 <= length_ratio <= 1.5:
            factors["length"] = 0.8
        else:
            factors["length"] = 0.4
        
        # Style factor
        formality_match = await self._check_formality_match(message, profile)
        factors["style"] = formality_match
        
        # Relevance factor
        recent_topics = await self._get_recent_community_topics(community_id)
        if any(topic.lower() in message.lower() for topic in recent_topics):
            factors["relevance"] = 0.9
        else:
            factors["relevance"] = 0.5
        
        # Calculate weighted success probability
        weights = {
            "timing": 0.25,
            "content": 0.25,
            "length": 0.15,
            "style": 0.15,
            "relevance": 0.20
        }
        
        success_probability = sum(factors[factor] * weights[factor] for factor in factors)
        
        return min(1.0, max(0.0, success_probability))
    
    async def generate_community_insights(self, community_id: str) -> List[CommunityInsight]:
        """
        Generate AI insights about community optimization opportunities
        """
        
        community = await self.database.get_telegram_community(community_id)
        if not community:
            return []
        
        profile = await self.analyze_community(community_id)
        strategy = await self.get_engagement_strategy(community_id)
        
        insights = []
        
        # Timing optimization insight
        if profile.activity_patterns.get("peak_hours"):
            peak_hours = profile.activity_patterns["peak_hours"]
            insight = CommunityInsight(
                community_id=community.id,
                insight_type="timing",
                title="Optimal Engagement Timing",
                description=f"Community is most active during {', '.join(peak_hours)}. Scheduling messages during these times could increase engagement by 40-60%.",
                confidence_score=0.85,
                impact_potential="high",
                implementation_difficulty="easy",
                recommended_actions=[
                    f"Schedule important messages during {peak_hours[0]} hour",
                    "Monitor engagement rates during different time periods",
                    "Adjust automated responses to peak activity times"
                ],
                expected_outcomes=[
                    "Increased message visibility",
                    "Higher response rates",
                    "Better community integration"
                ]
            )
            insights.append(insight)
        
        # Content optimization insight
        popular_topics = profile.content_preferences.get("popular_topics", [])
        if popular_topics:
            insight = CommunityInsight(
                community_id=community.id,
                insight_type="content",
                title="Content Strategy Optimization",
                description=f"Community shows high engagement with topics: {', '.join(popular_topics[:3])}. Focusing on these areas could improve engagement scores.",
                confidence_score=0.75,
                impact_potential="medium",
                implementation_difficulty="medium",
                recommended_actions=[
                    f"Prepare content related to {popular_topics[0]}",
                    "Monitor trending topics in community",
                    "Share relevant insights on popular subjects"
                ],
                expected_outcomes=[
                    "Increased content relevance",
                    "Higher engagement rates",
                    "Stronger community positioning"
                ]
            )
            insights.append(insight)
        
        # Personality adaptation insight
        if strategy.personality_adjustments:
            major_adjustments = {k: v for k, v in strategy.personality_adjustments.items() if abs(v) > 0.2}
            if major_adjustments:
                insight = CommunityInsight(
                    community_id=community.id,
                    insight_type="behavioral",
                    title="Personality Adaptation Opportunity",
                    description=f"Community dynamics suggest personality adjustments: {', '.join(major_adjustments.keys())}. This could improve relationship building.",
                    confidence_score=0.70,
                    impact_potential="medium",
                    implementation_difficulty="easy",
                    recommended_actions=[
                        "Apply suggested personality adaptations",
                        "Monitor community response to changes",
                        "Fine-tune based on feedback"
                    ],
                    expected_outcomes=[
                        "Better community fit",
                        "Improved relationship building",
                        "More authentic interactions"
                    ]
                )
                insights.append(insight)
        
        # Risk mitigation insight
        if strategy.risk_factors:
            high_risk_factors = [rf for rf in strategy.risk_factors if "high" in rf.lower()]
            if high_risk_factors:
                insight = CommunityInsight(
                    community_id=community.id,
                    insight_type="safety",
                    title="Risk Mitigation Strategy",
                    description=f"Identified risk factors: {', '.join(high_risk_factors)}. Implementing mitigation strategies is recommended.",
                    confidence_score=0.90,
                    impact_potential="high",
                    implementation_difficulty="easy",
                    recommended_actions=[
                        "Implement additional safety checks",
                        "Reduce activity frequency temporarily",
                        "Monitor community guidelines more closely"
                    ],
                    expected_outcomes=[
                        "Reduced account risk",
                        "Better compliance",
                        "Safer long-term engagement"
                    ]
                )
                insights.append(insight)
        
        # Store insights
        for insight in insights:
            await self.database.create_community_insight(insight)
        
        return insights
    
    # Private analysis methods
    async def _analyze_activity_patterns(
        self,
        conversations: List[TelegramConversation],
        events: List[CommunityEngagementEvent]
    ) -> Dict[str, Any]:
        """Analyze community activity patterns"""
        
        if not conversations and not events:
            return {"analysis_incomplete": True}
        
        # Message frequency analysis
        message_counts = defaultdict(int)
        hour_counts = defaultdict(int)
        
        for conv in conversations:
            if conv.last_message_date:
                hour = conv.last_message_date.hour
                hour_counts[hour] += 1
                message_counts[conv.last_message_date.date()] += conv.message_count
        
        # Peak hours
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_hour_list = [str(hour) for hour, count in peak_hours]
        
        # Activity level
        total_messages = sum(message_counts.values())
        avg_daily_messages = total_messages / max(1, len(message_counts))
        
        if avg_daily_messages > 100:
            activity_level = "very_high"
        elif avg_daily_messages > 50:
            activity_level = "high"
        elif avg_daily_messages > 20:
            activity_level = "medium"
        else:
            activity_level = "low"
        
        return {
            "peak_hours": peak_hour_list,
            "activity_level": activity_level,
            "avg_daily_messages": avg_daily_messages,
            "total_conversations": len(conversations),
            "engagement_events": len(events)
        }
    
    async def _analyze_communication_style(
        self,
        conversations: List[TelegramConversation]
    ) -> Dict[str, Any]:
        """Analyze community communication style"""
        
        if not conversations:
            return {"analysis_incomplete": True}
        
        # Get sample messages from conversations
        all_messages = []
        for conv in conversations[:10]:  # Sample from recent conversations
            messages = await self.database.get_conversation_messages(conv.id, limit=20)
            all_messages.extend(messages)
        
        if not all_messages:
            return {"analysis_incomplete": True}
        
        # Analyze message characteristics
        total_length = sum(len(msg.content or "") for msg in all_messages)
        avg_length = total_length / len(all_messages) if all_messages else 0
        
        # Count emoji usage
        emoji_count = sum(len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', msg.content or "")) for msg in all_messages)
        emoji_frequency = emoji_count / len(all_messages) if all_messages else 0
        
        # Formality indicators
        formal_indicators = ["please", "thank you", "regards", "sincerely"]
        informal_indicators = ["hey", "lol", "omg", "btw", "thx"]
        
        formal_count = sum(sum(indicator in (msg.content or "").lower() for indicator in formal_indicators) for msg in all_messages)
        informal_count = sum(sum(indicator in (msg.content or "").lower() for indicator in informal_indicators) for msg in all_messages)
        
        if formal_count > informal_count:
            formality_level = "high"
        elif informal_count > formal_count * 2:
            formality_level = "low"
        else:
            formality_level = "medium"
        
        # Question frequency
        question_count = sum(1 for msg in all_messages if msg.content and "?" in msg.content)
        question_frequency = question_count / len(all_messages) if all_messages else 0
        
        return {
            "average_message_length": avg_length,
            "emoji_usage": min(1.0, emoji_frequency),
            "formality_level": formality_level,
            "question_frequency": question_frequency,
            "sample_size": len(all_messages)
        }
    
    async def _analyze_content_preferences(
        self,
        conversations: List[TelegramConversation],
        events: List[CommunityEngagementEvent]
    ) -> Dict[str, Any]:
        """Analyze community content preferences"""
        
        # Extract topics from conversations
        all_topics = []
        for conv in conversations:
            if conv.topics_discussed:
                all_topics.extend(conv.topics_discussed)
        
        # Count topic frequency
        topic_counts = defaultdict(int)
        for topic in all_topics:
            if isinstance(topic, str):
                topic_counts[topic.lower()] += 1
            elif isinstance(topic, dict) and "topic" in topic:
                topic_counts[topic["topic"].lower()] += 1
        
        # Get most popular topics
        popular_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        popular_topic_list = [topic for topic, count in popular_topics]
        
        # Analyze content types from events
        content_types = defaultdict(int)
        for event in events:
            event_type = event.event_type
            if "photo" in event_type:
                content_types["images"] += 1
            elif "video" in event_type:
                content_types["videos"] += 1
            elif "link" in event_type:
                content_types["links"] += 1
            else:
                content_types["text"] += 1
        
        return {
            "popular_topics": popular_topic_list,
            "topic_diversity": len(topic_counts),
            "content_types": dict(content_types),
            "total_topics_mentioned": len(all_topics)
        }
    
    async def _analyze_member_dynamics(
        self,
        community: TelegramCommunity,
        conversations: List[TelegramConversation]
    ) -> Dict[str, Any]:
        """Analyze member interaction dynamics"""
        
        # Analyze participation patterns
        active_participants = set()
        for conv in conversations:
            if conv.participants:
                active_participants.update(conv.participants)
        
        # Calculate engagement distribution
        engagement_scores = []
        for conv in conversations:
            if conv.engagement_score > 0:
                engagement_scores.append(conv.engagement_score)
        
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        
        return {
            "active_participants": len(active_participants),
            "avg_engagement_score": avg_engagement,
            "total_conversations": len(conversations),
            "community_size": community.member_count or 0
        }
    
    async def _analyze_moderation_patterns(
        self,
        community: TelegramCommunity,
        events: List[CommunityEngagementEvent]
    ) -> Dict[str, Any]:
        """Analyze moderation patterns and restrictions"""
        
        # Count moderation events
        moderation_events = [e for e in events if "warning" in e.event_type or "restriction" in e.event_type]
        
        moderation_frequency = len(moderation_events) / max(1, len(events))
        
        if moderation_frequency > 0.1:
            strictness = "strict"
        elif moderation_frequency > 0.05:
            strictness = "moderate"
        else:
            strictness = "lenient"
        
        return {
            "moderation_frequency": moderation_frequency,
            "strictness": strictness,
            "total_moderation_events": len(moderation_events),
            "community_warnings": community.warning_count
        }
    
    async def _determine_optimal_engagement_times(
        self,
        activity_patterns: Dict[str, Any]
    ) -> List[str]:
        """Determine optimal times for engagement"""
        
        peak_hours = activity_patterns.get("peak_hours", [])
        
        # If we have peak hours, return them
        if peak_hours:
            return peak_hours
        
        # Default engagement times based on general patterns
        return ["09", "12", "18", "20"]  # 9 AM, 12 PM, 6 PM, 8 PM
    
    async def _calculate_personality_adaptations(
        self,
        community: TelegramCommunity,
        profile: CommunityProfile
    ) -> Dict[str, float]:
        """Calculate personality trait adjustments for community"""
        
        adaptations = {}
        
        # Formality adaptations
        formality_level = profile.communication_style.get("formality_level", "medium")
        if formality_level == "high":
            adaptations["formality"] = 0.3
            adaptations["professionalism"] = 0.2
        elif formality_level == "low":
            adaptations["casualness"] = 0.3
            adaptations["humor"] = 0.2
        
        # Engagement strategy adaptations
        if community.engagement_strategy == EngagementStrategy.LEADER:
            adaptations["confidence"] = 0.3
            adaptations["assertiveness"] = 0.2
        elif community.engagement_strategy == EngagementStrategy.LURKER:
            adaptations["observation_focus"] = 0.4
            adaptations["response_frequency"] = -0.5
        
        # Activity level adaptations
        activity_level = profile.activity_patterns.get("activity_level", "medium")
        if activity_level == "very_high":
            adaptations["energy"] = 0.2
            adaptations["responsiveness"] = 0.3
        elif activity_level == "low":
            adaptations["patience"] = 0.2
            adaptations["thoughtfulness"] = 0.3
        
        return adaptations
    
    # Additional helper methods would continue here...
    # (For brevity, including key methods but truncating some implementation details)
    
    async def _generate_communication_guidelines(
        self,
        community: TelegramCommunity,
        profile: CommunityProfile
    ) -> List[str]:
        """Generate communication guidelines"""
        guidelines = []
        
        formality = profile.communication_style.get("formality_level", "medium")
        if formality == "high":
            guidelines.append("Use formal language and proper grammar")
            guidelines.append("Avoid slang and casual expressions")
        else:
            guidelines.append("Use casual, friendly language")
            guidelines.append("Emoji usage is encouraged")
        
        return guidelines
    
    async def _generate_content_suggestions(
        self,
        community: TelegramCommunity,
        profile: CommunityProfile,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate content suggestions"""
        suggestions = []
        
        popular_topics = profile.content_preferences.get("popular_topics", [])
        for topic in popular_topics[:3]:
            suggestions.append(f"Share insights about {topic}")
        
        return suggestions
    
    async def _generate_timing_recommendations(
        self,
        profile: CommunityProfile
    ) -> Dict[str, Any]:
        """Generate timing recommendations"""
        return {
            "optimal_hours": profile.optimal_engagement_times,
            "avoid_hours": ["02", "03", "04", "05"],  # Late night hours
            "response_window": "2-30 minutes"
        }
    
    async def _identify_risk_factors(
        self,
        community: TelegramCommunity,
        profile: CommunityProfile
    ) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if community.warning_count > 2:
            risks.append("High warning count - exercise caution")
        
        if profile.moderation_patterns.get("strictness") == "strict":
            risks.append("Strict moderation - follow guidelines carefully")
        
        return risks
    
    async def _calculate_success_metrics(
        self,
        community: TelegramCommunity
    ) -> Dict[str, float]:
        """Calculate success metrics for community"""
        return {
            "current_engagement_score": community.engagement_score,
            "target_engagement_score": community.engagement_score * 1.2,
            "current_reputation": community.reputation_score,
            "target_reputation": min(100.0, community.reputation_score * 1.1)
        }
    
    # Storage methods
    async def _store_community_profile(
        self,
        community: TelegramCommunity,
        profile: CommunityProfile
    ):
        """Store community profile insights"""
        await self.memory.store_memory(
            content=f"Community profile for {community.title}",
            memory_type="community_analysis",
            importance=0.8,
            context={
                "community_id": str(community.id),
                "profile": profile.__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _store_strategy_insights(
        self,
        community: TelegramCommunity,
        strategy: AdaptationStrategy
    ):
        """Store strategy insights"""
        await self.memory.store_memory(
            content=f"Engagement strategy for {community.title}",
            memory_type="engagement_strategy",
            importance=0.9,
            context={
                "community_id": str(community.id),
                "strategy": strategy.__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
        )