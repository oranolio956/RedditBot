"""
Viral Engine Service

Smart sharing triggers and viral content generation system.
Creates engaging, shareable content from bot interactions
with AI-powered optimization for maximum viral potential.
"""

import json
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import desc, func
import structlog

from app.models.sharing import (
    ShareableContent, ContentShare, ShareableContentType, SocialPlatform,
    ViralMetrics
)
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.llm_service import LLMService

logger = structlog.get_logger(__name__)


@dataclass
class ViralTrigger:
    """Configuration for viral content triggers."""
    content_type: ShareableContentType
    trigger_conditions: Dict[str, Any]
    min_viral_score: float
    platforms: List[SocialPlatform]
    generation_prompt: str


class ViralEngine:
    """
    AI-powered viral content generation and optimization engine.
    
    Analyzes conversations in real-time to identify shareable moments,
    generates optimized content for different social platforms,
    and tracks viral performance to improve future content.
    """
    
    def __init__(self, db: Session, llm_service: LLMService):
        self.db = db
        self.llm_service = llm_service
        
        # Viral triggers configuration
        self.triggers = self._setup_viral_triggers()
        
        # Trending hashtags cache (updated hourly)
        self._trending_hashtags = {}
        self._hashtags_updated_at = None
    
    def _setup_viral_triggers(self) -> List[ViralTrigger]:
        """Configure triggers for different types of viral content."""
        return [
            ViralTrigger(
                content_type=ShareableContentType.FUNNY_MOMENT,
                trigger_conditions={
                    "sentiment_spike": 0.8,  # Very positive sentiment
                    "humor_indicators": ["😂", "lol", "haha", "funny", "hilarious"],
                    "min_message_length": 20,
                    "max_message_length": 280
                },
                min_viral_score=75.0,
                platforms=[SocialPlatform.TWITTER, SocialPlatform.INSTAGRAM, SocialPlatform.TIKTOK],
                generation_prompt="Create a funny, shareable moment from this conversation that would make people laugh and want to share it. Focus on wit, timing, and universal relatability."
            ),
            
            ViralTrigger(
                content_type=ShareableContentType.PERSONALITY_INSIGHT,
                trigger_conditions={
                    "personality_reveal": True,
                    "insight_depth": 0.7,
                    "relatability_score": 0.6
                },
                min_viral_score=70.0,
                platforms=[SocialPlatform.INSTAGRAM, SocialPlatform.LINKEDIN, SocialPlatform.TWITTER],
                generation_prompt="Transform this personality insight into a shareable card that helps people understand themselves better. Make it inspiring and actionable."
            ),
            
            ViralTrigger(
                content_type=ShareableContentType.WISDOM_QUOTE,
                trigger_conditions={
                    "wisdom_indicators": ["learned", "realize", "understand", "insight", "truth"],
                    "emotional_impact": 0.7,
                    "universal_appeal": 0.6
                },
                min_viral_score=65.0,
                platforms=[SocialPlatform.INSTAGRAM, SocialPlatform.LINKEDIN, SocialPlatform.FACEBOOK],
                generation_prompt="Extract a powerful, quotable insight that would inspire and resonate with a wide audience. Focus on universal truths and life lessons."
            ),
            
            ViralTrigger(
                content_type=ShareableContentType.AI_RESPONSE,
                trigger_conditions={
                    "ai_cleverness": 0.8,
                    "surprising_insight": True,
                    "conversation_flow": "breakthrough_moment"
                },
                min_viral_score=80.0,
                platforms=[SocialPlatform.TWITTER, SocialPlatform.REDDIT, SocialPlatform.TIKTOK],
                generation_prompt="Highlight this AI's clever or surprisingly insightful response in a way that showcases artificial intelligence capabilities while being entertaining."
            ),
            
            ViralTrigger(
                content_type=ShareableContentType.EDUCATIONAL_SUMMARY,
                trigger_conditions={
                    "educational_value": 0.7,
                    "topic_complexity": "medium",
                    "learning_moment": True
                },
                min_viral_score=60.0,
                platforms=[SocialPlatform.LINKEDIN, SocialPlatform.TWITTER, SocialPlatform.REDDIT],
                generation_prompt="Create an educational post that teaches something valuable in an engaging, easy-to-understand way. Include actionable takeaways."
            )
        ]
    
    async def analyze_conversation_for_viral_content(
        self, 
        conversation: Conversation,
        real_time: bool = True
    ) -> List[ShareableContent]:
        """
        Analyze a conversation for viral content opportunities.
        
        Args:
            conversation: Conversation to analyze
            real_time: Whether this is real-time analysis
            
        Returns:
            List of generated shareable content
        """
        try:
            logger.info(
                "analyzing_conversation_viral_content",
                conversation_id=conversation.id,
                message_count=len(conversation.messages)
            )
            
            generated_content = []
            
            # Get conversation context
            context = await self._extract_conversation_context(conversation)
            
            # Check each viral trigger
            for trigger in self.triggers:
                if await self._check_trigger_conditions(context, trigger):
                    content = await self._generate_viral_content(
                        conversation, context, trigger
                    )
                    if content and content.viral_score >= trigger.min_viral_score:
                        generated_content.append(content)
                        logger.info(
                            "viral_content_generated",
                            content_type=trigger.content_type,
                            viral_score=content.viral_score
                        )
            
            # Save and return generated content
            for content in generated_content:
                self.db.add(content)
            
            if generated_content:
                self.db.commit()
            
            return generated_content
            
        except Exception as e:
            logger.error("viral_analysis_failed", error=str(e))
            self.db.rollback()
            return []
    
    async def _extract_conversation_context(
        self, 
        conversation: Conversation
    ) -> Dict[str, Any]:
        """Extract rich context from conversation for viral analysis."""
        
        # Get recent messages
        recent_messages = conversation.messages[-10:] if conversation.messages else []
        
        # Analyze sentiment trajectory
        sentiment_scores = []
        for msg in recent_messages:
            if hasattr(msg, 'analysis_data') and msg.analysis_data:
                sentiment = msg.analysis_data.get('sentiment', {}).get('score', 0.5)
                sentiment_scores.append(sentiment)
        
        # Calculate conversation flow metrics
        sentiment_spike = max(sentiment_scores) - min(sentiment_scores) if sentiment_scores else 0
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        # Extract key phrases and topics
        all_text = " ".join([msg.content for msg in recent_messages if msg.content])
        humor_indicators = self._count_humor_indicators(all_text)
        wisdom_indicators = self._count_wisdom_indicators(all_text)
        personality_markers = self._extract_personality_markers(all_text)
        
        # Check for AI breakthrough moments
        ai_cleverness = self._assess_ai_cleverness(recent_messages)
        
        return {
            "message_count": len(recent_messages),
            "total_length": len(all_text),
            "sentiment_spike": sentiment_spike,
            "avg_sentiment": avg_sentiment,
            "humor_indicators": humor_indicators,
            "wisdom_indicators": wisdom_indicators,
            "personality_markers": personality_markers,
            "ai_cleverness": ai_cleverness,
            "recent_messages": recent_messages,
            "full_text": all_text,
            "conversation_age": (datetime.utcnow() - conversation.created_at).total_seconds() / 3600
        }
    
    def _count_humor_indicators(self, text: str) -> int:
        """Count humor indicators in text."""
        indicators = ["😂", "🤣", "😄", "😆", "lol", "haha", "funny", "hilarious", "joke", "laugh"]
        count = 0
        text_lower = text.lower()
        for indicator in indicators:
            count += text_lower.count(indicator)
        return count
    
    def _count_wisdom_indicators(self, text: str) -> int:
        """Count wisdom/insight indicators in text."""
        indicators = [
            "learned", "realize", "understand", "insight", "truth", "wisdom",
            "discovery", "breakthrough", "epiphany", "clarity", "perspective"
        ]
        count = 0
        text_lower = text.lower()
        for indicator in indicators:
            count += text_lower.count(indicator)
        return count
    
    def _extract_personality_markers(self, text: str) -> List[str]:
        """Extract personality-related markers from text."""
        personality_patterns = {
            "introversion": ["quiet", "alone", "solitude", "introspective", "private"],
            "extraversion": ["social", "party", "people", "outgoing", "energetic"],
            "openness": ["creative", "curious", "explore", "imagine", "artistic"],
            "conscientiousness": ["organized", "planned", "disciplined", "goal", "achieve"],
            "neuroticism": ["worry", "stress", "anxious", "emotional", "sensitive"],
            "agreeableness": ["kind", "helpful", "caring", "empathy", "compassionate"]
        }
        
        found_markers = []
        text_lower = text.lower()
        
        for trait, markers in personality_patterns.items():
            for marker in markers:
                if marker in text_lower:
                    found_markers.append(f"{trait}:{marker}")
        
        return found_markers
    
    def _assess_ai_cleverness(self, messages: List[Message]) -> float:
        """Assess how clever or insightful the AI responses were."""
        ai_messages = [msg for msg in messages if msg.is_from_bot]
        if not ai_messages:
            return 0.0
        
        cleverness_score = 0.0
        
        for msg in ai_messages:
            # Check for sophisticated language
            if any(word in msg.content.lower() for word in [
                "paradox", "nuance", "perspective", "fascinating", "intriguing",
                "consider", "alternatively", "however", "furthermore", "moreover"
            ]):
                cleverness_score += 0.2
            
            # Check for creative analogies or metaphors
            if any(phrase in msg.content.lower() for phrase in [
                "like", "similar to", "imagine", "picture this", "think of it as"
            ]):
                cleverness_score += 0.1
            
            # Check for humor or wit
            if any(indicator in msg.content.lower() for indicator in [
                "ironically", "funny thing", "plot twist", "surprise", "unexpected"
            ]):
                cleverness_score += 0.3
        
        return min(cleverness_score, 1.0)
    
    async def _check_trigger_conditions(
        self, 
        context: Dict[str, Any], 
        trigger: ViralTrigger
    ) -> bool:
        """Check if conversation context meets viral trigger conditions."""
        
        conditions = trigger.trigger_conditions
        
        # Check sentiment spike condition
        if "sentiment_spike" in conditions:
            if context["sentiment_spike"] < conditions["sentiment_spike"]:
                return False
        
        # Check humor indicators
        if "humor_indicators" in conditions:
            if context["humor_indicators"] < len(conditions["humor_indicators"]):
                return False
        
        # Check message length constraints
        if "min_message_length" in conditions:
            if context["total_length"] < conditions["min_message_length"]:
                return False
        
        if "max_message_length" in conditions:
            if context["total_length"] > conditions["max_message_length"]:
                return False
        
        # Check AI cleverness
        if "ai_cleverness" in conditions:
            if context["ai_cleverness"] < conditions["ai_cleverness"]:
                return False
        
        # Check personality markers
        if "personality_reveal" in conditions and conditions["personality_reveal"]:
            if len(context["personality_markers"]) < 2:
                return False
        
        # Check wisdom indicators
        if "wisdom_indicators" in conditions:
            if context["wisdom_indicators"] < len(conditions["wisdom_indicators"]):
                return False
        
        return True
    
    async def _generate_viral_content(
        self,
        conversation: Conversation,
        context: Dict[str, Any],
        trigger: ViralTrigger
    ) -> Optional[ShareableContent]:
        """Generate optimized viral content using AI."""
        
        try:
            # Prepare AI generation prompt
            generation_context = {
                "conversation_excerpt": context["full_text"][-500:],  # Last 500 chars
                "content_type": trigger.content_type,
                "target_platforms": trigger.platforms,
                "viral_elements": self._identify_viral_elements(context),
                "trending_hashtags": await self._get_trending_hashtags(trigger.platforms)
            }
            
            # Generate content using LLM
            prompt = f"""
            {trigger.generation_prompt}
            
            Context: {generation_context['conversation_excerpt']}
            Content Type: {trigger.content_type}
            Target Platforms: {', '.join([p.value for p in trigger.platforms])}
            
            Create viral content that:
            1. Is highly shareable and engaging
            2. Maintains anonymity (no personal details)
            3. Appeals to the target platforms
            4. Includes relevant trending hashtags
            5. Has a strong hook in the first few words
            
            Return JSON with:
            - title: Catchy, shareable title
            - description: Platform-optimized description
            - content_data: Structured content for different formats
            - hashtags: Relevant trending hashtags
            - viral_elements: What makes this shareable
            """
            
            response = await self.llm_service.generate_completion(
                prompt=prompt,
                max_tokens=800,
                temperature=0.8
            )
            
            # Parse AI response
            try:
                content_json = json.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing if AI doesn't return valid JSON
                content_json = self._parse_ai_response_fallback(response)
            
            # Calculate viral score
            viral_score = self._calculate_viral_score(context, content_json, trigger)
            
            # Create shareable content
            content = ShareableContent(
                content_type=trigger.content_type.value,
                title=content_json.get("title", "")[:200],
                description=content_json.get("description", "")[:500],
                content_data=content_json.get("content_data", {}),
                viral_score=viral_score,
                hashtags=content_json.get("hashtags", []),
                optimal_platforms=[p.value for p in trigger.platforms],
                is_anonymized=True,
                anonymization_level="high",
                source_conversation_id=conversation.id,
                source_user_anonymous_id=self._generate_anonymous_id(conversation.user_id),
                ai_enhancement_data={
                    "generation_context": generation_context,
                    "trigger_type": trigger.content_type.value,
                    "viral_elements": content_json.get("viral_elements", [])
                }
            )
            
            return content
            
        except Exception as e:
            logger.error("viral_content_generation_failed", error=str(e))
            return None
    
    def _identify_viral_elements(self, context: Dict[str, Any]) -> List[str]:
        """Identify elements that make content viral."""
        elements = []
        
        if context["sentiment_spike"] > 0.6:
            elements.append("emotional_impact")
        
        if context["humor_indicators"] > 2:
            elements.append("humor")
        
        if context["ai_cleverness"] > 0.7:
            elements.append("ai_insight")
        
        if len(context["personality_markers"]) > 2:
            elements.append("relatability")
        
        if context["wisdom_indicators"] > 1:
            elements.append("wisdom")
        
        return elements
    
    async def _get_trending_hashtags(
        self, 
        platforms: List[SocialPlatform]
    ) -> List[str]:
        """Get trending hashtags for specified platforms."""
        
        # Check cache freshness
        if (self._hashtags_updated_at and 
            datetime.utcnow() - self._hashtags_updated_at < timedelta(hours=1)):
            return self._trending_hashtags.get("combined", [])
        
        # This would integrate with social media APIs in production
        # For now, return curated trending hashtags
        trending_by_platform = {
            SocialPlatform.TWITTER: [
                "#AI", "#Psychology", "#Mindfulness", "#SelfImprovement",
                "#MentalHealth", "#Wisdom", "#Growth", "#Insight"
            ],
            SocialPlatform.INSTAGRAM: [
                "#SelfCare", "#Mindset", "#PersonalGrowth", "#Motivation",
                "#Wellness", "#Inspiration", "#LifeLessons", "#Positivity"
            ],
            SocialPlatform.TIKTOK: [
                "#Psychology", "#MentalHealthTok", "#SelfImprovement",
                "#LifeHacks", "#Mindfulness", "#Therapy", "#Growth"
            ],
            SocialPlatform.LINKEDIN: [
                "#PersonalDevelopment", "#Leadership", "#Mindset",
                "#ProfessionalGrowth", "#SelfImprovement", "#Wellness"
            ]
        }
        
        # Combine hashtags for requested platforms
        combined_hashtags = set()
        for platform in platforms:
            combined_hashtags.update(trending_by_platform.get(platform, []))
        
        self._trending_hashtags["combined"] = list(combined_hashtags)
        self._hashtags_updated_at = datetime.utcnow()
        
        return list(combined_hashtags)
    
    def _parse_ai_response_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when AI doesn't return valid JSON."""
        lines = response.split('\n')
        
        content_data = {
            "title": "",
            "description": "",
            "content_data": {"text": response[:500]},
            "hashtags": [],
            "viral_elements": ["ai_generated"]
        }
        
        # Try to extract title (first line)
        if lines:
            content_data["title"] = lines[0][:200]
        
        # Try to extract hashtags
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, response)
        content_data["hashtags"] = hashtags[:10]  # Limit to 10 hashtags
        
        return content_data
    
    def _calculate_viral_score(
        self,
        context: Dict[str, Any],
        content_json: Dict[str, Any],
        trigger: ViralTrigger
    ) -> float:
        """Calculate viral potential score for generated content."""
        
        base_score = 50.0
        
        # Content quality factors
        if len(content_json.get("title", "")) > 10:
            base_score += 10
        
        if len(content_json.get("hashtags", [])) >= 3:
            base_score += 10
        
        # Context-based factors
        base_score += context["sentiment_spike"] * 20
        base_score += context["ai_cleverness"] * 15
        base_score += min(context["humor_indicators"] * 5, 15)
        base_score += min(len(context["personality_markers"]) * 3, 12)
        
        # Viral elements bonus
        viral_elements = content_json.get("viral_elements", [])
        base_score += len(viral_elements) * 5
        
        # Platform optimization
        if len(trigger.platforms) >= 3:
            base_score += 8  # Multi-platform content has higher viral potential
        
        return min(base_score, 100.0)
    
    def _generate_anonymous_id(self, user_id: str) -> str:
        """Generate anonymized user identifier for analytics."""
        import hashlib
        return hashlib.sha256(f"anon_{user_id}".encode()).hexdigest()[:16]
    
    async def get_trending_content(
        self, 
        limit: int = 10,
        content_type: Optional[ShareableContentType] = None
    ) -> List[ShareableContent]:
        """Get currently trending viral content."""
        
        query = self.db.query(ShareableContent)\
            .filter(ShareableContent.is_published == True)
        
        if content_type:
            query = query.filter(ShareableContent.content_type == content_type.value)
        
        # Calculate trending score (views + shares * 10 + viral_score)
        trending_content = query\
            .order_by(desc(
                ShareableContent.view_count + 
                ShareableContent.share_count * 10 + 
                ShareableContent.viral_score
            ))\
            .limit(limit)\
            .all()
        
        return trending_content
    
    async def optimize_content_for_platform(
        self,
        content: ShareableContent,
        platform: SocialPlatform
    ) -> Dict[str, Any]:
        """Optimize content for specific social platform."""
        
        platform_specs = {
            SocialPlatform.TWITTER: {
                "max_length": 280,
                "image_ratio": "16:9",
                "hashtag_limit": 2,
                "style": "concise_witty"
            },
            SocialPlatform.INSTAGRAM: {
                "max_length": 2200,
                "image_ratio": "1:1",
                "hashtag_limit": 30,
                "style": "visual_storytelling"
            },
            SocialPlatform.TIKTOK: {
                "max_length": 150,
                "video_length": 60,
                "hashtag_limit": 5,
                "style": "trendy_engaging"
            },
            SocialPlatform.LINKEDIN: {
                "max_length": 3000,
                "image_ratio": "4:3",
                "hashtag_limit": 5,
                "style": "professional_insightful"
            }
        }
        
        specs = platform_specs.get(platform, platform_specs[SocialPlatform.TWITTER])
        
        # Optimize content based on platform specs
        optimized = {
            "platform": platform.value,
            "title": content.title[:specs["max_length"]],
            "description": content.description,
            "hashtags": (content.hashtags or [])[:specs["hashtag_limit"]],
            "specs": specs,
            "share_url": f"/share/{content.id}?platform={platform.value}"
        }
        
        return optimized
    
    async def track_content_performance(
        self,
        content_id: str,
        platform: SocialPlatform,
        metrics: Dict[str, int]
    ):
        """Track viral content performance across platforms."""
        
        content = self.db.query(ShareableContent).filter(
            ShareableContent.id == content_id
        ).first()
        
        if not content:
            return
        
        # Update aggregate metrics
        content.view_count += metrics.get("views", 0)
        content.share_count += metrics.get("shares", 0)
        content.like_count += metrics.get("likes", 0)
        content.comment_count += metrics.get("comments", 0)
        
        self.db.commit()
        
        logger.info(
            "content_performance_tracked",
            content_id=content_id,
            platform=platform.value,
            total_shares=content.share_count
        )