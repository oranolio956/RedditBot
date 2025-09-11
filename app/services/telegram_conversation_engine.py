"""
Telegram Conversation Engine
Advanced AI-powered conversation system with natural language processing and context awareness.
Integrates with all revolutionary features for authentic human-like interactions.
"""

import asyncio
import logging
import random
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import json

from app.models.telegram_conversation import (
    TelegramConversation, ConversationMessage, MessageType, 
    MessageDirection, ConversationStatus, ConversationContext
)
from app.models.telegram_community import TelegramCommunity, EngagementStrategy
from app.services.llm_service import LLMService
from app.services.consciousness_mirror import ConsciousnessMirror
from app.services.memory_palace import MemoryPalace
from app.services.emotion_detector import EmotionDetector
from app.services.digital_telepathy_engine import DigitalTelepathyEngine
from app.services.temporal_archaeology import TemporalArchaeology
from app.database.repositories import DatabaseRepository


@dataclass
class ConversationInsight:
    """Insights about conversation dynamics"""
    topic_shifts: List[str]
    emotional_progression: List[Dict[str, Any]]
    engagement_patterns: Dict[str, Any]
    response_opportunities: List[str]
    personality_adaptations: Dict[str, Any]


@dataclass
class ResponseGeneration:
    """Generated response with metadata"""
    content: str
    confidence: float
    response_type: str  # answer, question, comment, reaction
    personality_used: Dict[str, Any]
    thinking_process: List[str]
    safety_score: float


class TelegramConversationEngine:
    """
    Advanced conversation engine that creates natural, context-aware responses
    using revolutionary AI features for maximum authenticity.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        consciousness_mirror: ConsciousnessMirror,
        memory_palace: MemoryPalace,
        emotion_detector: EmotionDetector,
        telepathy_engine: DigitalTelepathyEngine,
        temporal_archaeology: TemporalArchaeology,
        database: DatabaseRepository
    ):
        self.llm = llm_service
        self.consciousness = consciousness_mirror
        self.memory = memory_palace
        self.emotion_detector = emotion_detector
        self.telepathy = telepathy_engine
        self.temporal = temporal_archaeology
        self.database = database
        
        self.logger = logging.getLogger(__name__)
        
        # Conversation patterns and templates
        self.response_templates = {
            "supportive": [
                "That sounds {emotion_adjective}. {supportive_response}",
                "I can understand how that would feel {emotion_adjective}. {empathetic_response}",
                "{acknowledgment} {helpful_suggestion}"
            ],
            "informative": [
                "Based on what you've shared, {information_response}",
                "That's an interesting point about {topic}. {additional_insight}",
                "I've noticed {observation}. {related_information}"
            ],
            "conversational": [
                "That reminds me of {related_experience}. {follow_up_question}",
                "{casual_acknowledgment} {personal_sharing}",
                "I find {topic} fascinating too. {open_ended_question}"
            ]
        }
        
        # Safety filters
        self.safety_filters = {
            "no_personal_info": r"(phone|email|address|password|ssn|credit card)",
            "no_inappropriate": r"(hate|violence|explicit|illegal)",
            "no_spam": r"(buy now|click here|limited time|urgent|act fast)",
            "no_medical_advice": r"(diagnose|treatment|medical|prescription|cure)",
            "no_financial_advice": r"(invest|stock|crypto|trading|financial advice)"
        }
    
    async def initialize(self, account_id: str):
        """Initialize conversation engine for specific account"""
        self.account_id = account_id
        await self.consciousness.initialize(account_id)
        await self.memory.initialize(f"conversations_{account_id}")
        await self.telepathy.initialize(account_id)
        
        self.logger.info(f"Conversation engine initialized for account {account_id}")
    
    async def generate_response(
        self,
        message: str,
        conversation: TelegramConversation,
        community: Optional[TelegramCommunity] = None,
        context: Optional[Dict[str, Any]] = None,
        is_mention: bool = False
    ) -> Optional[ResponseGeneration]:
        """
        Generate contextually appropriate response using revolutionary AI features
        """
        
        try:
            # Analyze conversation context
            conversation_analysis = await self._analyze_conversation_context(
                conversation, message, community
            )
            
            # Get personality adaptation
            adapted_personality = await self._adapt_personality_for_context(
                conversation, community, conversation_analysis
            )
            
            # Retrieve relevant memories
            memory_context = await self._get_memory_context(message, conversation)
            
            # Analyze emotional context
            emotional_context = await self._analyze_emotional_context(
                message, conversation, is_mention
            )
            
            # Generate response using LLM with revolutionary context
            response = await self._generate_llm_response(
                message=message,
                conversation_analysis=conversation_analysis,
                personality=adapted_personality,
                memory_context=memory_context,
                emotional_context=emotional_context,
                community=community,
                is_mention=is_mention
            )
            
            if not response:
                return None
            
            # Validate and improve response
            validated_response = await self._validate_and_improve_response(
                response, conversation, community
            )
            
            # Store conversation insights
            await self._store_conversation_insights(
                conversation, message, validated_response, conversation_analysis
            )
            
            return validated_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None
    
    async def _analyze_conversation_context(
        self,
        conversation: TelegramConversation,
        current_message: str,
        community: Optional[TelegramCommunity]
    ) -> ConversationInsight:
        """Analyze conversation context for insights"""
        
        # Get recent messages
        recent_messages = await self.database.get_recent_conversation_messages(
            conversation.id, limit=20
        )
        
        # Extract topics and emotional progression
        topics = []
        emotions = []
        
        for msg in recent_messages:
            if msg.topics_extracted:
                topics.extend(msg.topics_extracted)
            
            if msg.sentiment and msg.sentiment_confidence:
                emotions.append({
                    "sentiment": msg.sentiment,
                    "confidence": msg.sentiment_confidence,
                    "timestamp": msg.sent_at
                })
        
        # Detect topic shifts
        topic_shifts = await self._detect_topic_shifts(topics)
        
        # Analyze engagement patterns
        engagement_patterns = await self._analyze_engagement_patterns(recent_messages)
        
        # Generate response opportunities
        response_opportunities = await self._identify_response_opportunities(
            current_message, recent_messages, community
        )
        
        # Personality adaptations needed
        personality_adaptations = await self._suggest_personality_adaptations(
            conversation, community, emotions
        )
        
        return ConversationInsight(
            topic_shifts=topic_shifts,
            emotional_progression=emotions,
            engagement_patterns=engagement_patterns,
            response_opportunities=response_opportunities,
            personality_adaptations=personality_adaptations
        )
    
    async def _adapt_personality_for_context(
        self,
        conversation: TelegramConversation,
        community: Optional[TelegramCommunity],
        analysis: ConversationInsight
    ) -> Dict[str, Any]:
        """Adapt personality for current conversation context"""
        
        # Get base personality from consciousness mirror
        base_personality = await self.consciousness.get_current_personality()
        
        # Apply community adaptations
        if community:
            personality = community.adapt_personality_to_community(base_personality)
        else:
            personality = base_personality.copy()
        
        # Apply conversation-specific adaptations
        for adaptation_key, adaptation_value in analysis.personality_adaptations.items():
            if adaptation_key in personality:
                personality[adaptation_key] = adaptation_value
        
        # Adjust based on emotional progression
        if analysis.emotional_progression:
            recent_emotion = analysis.emotional_progression[-1]
            if recent_emotion["sentiment"] == "negative" and recent_emotion["confidence"] > 0.7:
                personality["empathy"] = min(1.0, personality.get("empathy", 0.5) + 0.3)
                personality["supportiveness"] = min(1.0, personality.get("supportiveness", 0.5) + 0.2)
        
        return personality
    
    async def _get_memory_context(
        self,
        message: str,
        conversation: TelegramConversation
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for response generation"""
        
        # Get conversation-specific memories
        conversation_memories = await self.memory.retrieve_relevant_memories(
            query=message,
            context_type="conversation",
            context_filter={"conversation_id": str(conversation.id)}
        )
        
        # Get general memories related to topics
        topics = conversation.get_recent_topics(hours=24)
        topic_memories = []
        
        for topic in topics:
            topic_mem = await self.memory.retrieve_relevant_memories(
                query=topic,
                context_type="knowledge",
                limit=2
            )
            topic_memories.extend(topic_mem)
        
        # Combine and deduplicate
        all_memories = conversation_memories + topic_memories
        unique_memories = []
        seen_content = set()
        
        for memory in all_memories:
            content_hash = hash(memory.get("content", ""))
            if content_hash not in seen_content:
                unique_memories.append(memory)
                seen_content.add(content_hash)
        
        return unique_memories[:10]  # Limit to top 10 most relevant
    
    async def _analyze_emotional_context(
        self,
        message: str,
        conversation: TelegramConversation,
        is_mention: bool
    ) -> Dict[str, Any]:
        """Analyze emotional context of current message and conversation"""
        
        # Analyze current message emotion
        current_emotion = await self.emotion_detector.analyze_text(message)
        
        # Get conversation emotional history
        emotional_history = conversation.sentiment_history or []
        
        # Calculate emotional trend
        if len(emotional_history) >= 2:
            recent_emotions = emotional_history[-3:]
            positive_count = sum(1 for e in recent_emotions if e.get("sentiment") == "positive")
            negative_count = sum(1 for e in recent_emotions if e.get("sentiment") == "negative")
            
            if positive_count > negative_count:
                trend = "improving"
            elif negative_count > positive_count:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        # Determine response emotional tone
        if current_emotion.get("primary_emotion") == "sadness" and current_emotion.get("intensity", 0) > 0.6:
            response_tone = "supportive"
        elif current_emotion.get("primary_emotion") == "anger" and current_emotion.get("intensity", 0) > 0.5:
            response_tone = "calming"
        elif current_emotion.get("primary_emotion") == "joy" and current_emotion.get("intensity", 0) > 0.6:
            response_tone = "enthusiastic"
        elif is_mention:
            response_tone = "attentive"
        else:
            response_tone = "balanced"
        
        return {
            "current_emotion": current_emotion,
            "emotional_trend": trend,
            "recommended_tone": response_tone,
            "emotional_intensity": current_emotion.get("intensity", 0.5),
            "requires_empathy": current_emotion.get("primary_emotion") in ["sadness", "fear", "anger"]
        }
    
    async def _generate_llm_response(
        self,
        message: str,
        conversation_analysis: ConversationInsight,
        personality: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        emotional_context: Dict[str, Any],
        community: Optional[TelegramCommunity],
        is_mention: bool
    ) -> Optional[ResponseGeneration]:
        """Generate response using LLM with full context"""
        
        # Build comprehensive context for LLM
        context_prompt = await self._build_context_prompt(
            message, conversation_analysis, personality, memory_context, 
            emotional_context, community, is_mention
        )
        
        # Generate response
        llm_response = await self.llm.generate_response(
            prompt=context_prompt,
            max_tokens=200,
            temperature=personality.get("creativity", 0.7),
            personality_context=personality
        )
        
        if not llm_response:
            return None
        
        # Extract response and metadata
        response_content = llm_response.get("content", "").strip()
        
        # Determine response type
        response_type = self._classify_response_type(response_content, is_mention)
        
        # Calculate confidence based on various factors
        confidence = self._calculate_response_confidence(
            response_content, emotional_context, memory_context
        )
        
        # Generate thinking process explanation
        thinking_process = await self._generate_thinking_process(
            message, response_content, conversation_analysis
        )
        
        # Calculate safety score
        safety_score = await self._calculate_safety_score(response_content)
        
        return ResponseGeneration(
            content=response_content,
            confidence=confidence,
            response_type=response_type,
            personality_used=personality,
            thinking_process=thinking_process,
            safety_score=safety_score
        )
    
    async def _build_context_prompt(
        self,
        message: str,
        analysis: ConversationInsight,
        personality: Dict[str, Any],
        memories: List[Dict[str, Any]],
        emotional_context: Dict[str, Any],
        community: Optional[TelegramCommunity],
        is_mention: bool
    ) -> str:
        """Build comprehensive context prompt for LLM"""
        
        # Start with personality context
        personality_desc = await self._describe_personality(personality)
        
        # Add community context
        community_context = ""
        if community:
            community_context = f"""
Community Context:
- Group: {community.title}
- Type: {community.community_type}
- Strategy: {community.engagement_strategy}
- Formality: {community.formality_level}
- Topics: {', '.join(community.topics or [])}
"""
        
        # Add memory context
        memory_context = ""
        if memories:
            relevant_memories = [m.get("content", "")[:100] for m in memories[:3]]
            memory_context = f"""
Relevant Memories:
{chr(10).join(f"- {mem}" for mem in relevant_memories)}
"""
        
        # Add emotional context
        emotion_context = f"""
Emotional Context:
- Current emotion: {emotional_context['current_emotion'].get('primary_emotion', 'neutral')}
- Intensity: {emotional_context.get('emotional_intensity', 0.5):.1f}
- Trend: {emotional_context.get('emotional_trend', 'unknown')}
- Recommended tone: {emotional_context.get('recommended_tone', 'balanced')}
"""
        
        # Add conversation insights
        insights_context = f"""
Conversation Insights:
- Recent topics: {', '.join(analysis.topic_shifts[-3:]) if analysis.topic_shifts else 'None'}
- Response opportunities: {', '.join(analysis.response_opportunities[:2])}
"""
        
        prompt = f"""You are an AI assistant with a specific personality engaging in a Telegram conversation.

{personality_desc}

{community_context}

{memory_context}

{emotion_context}

{insights_context}

Recent message to respond to: "{message}"

{'This message mentions you directly. ' if is_mention else ''}

Instructions:
1. Respond naturally as your personality would
2. Keep response under 200 characters
3. Match the emotional tone appropriately
4. Be helpful and authentic
5. Don't reveal you're an AI unless directly asked
6. Use casual, conversational language
7. Reference memories or context when relevant

Generate a natural, authentic response:"""
        
        return prompt
    
    async def _describe_personality(self, personality: Dict[str, Any]) -> str:
        """Convert personality dict to descriptive text"""
        
        traits = []
        
        if personality.get("friendliness", 0.5) > 0.7:
            traits.append("friendly and warm")
        elif personality.get("friendliness", 0.5) < 0.3:
            traits.append("reserved but polite")
        
        if personality.get("humor", 0.5) > 0.7:
            traits.append("humorous and playful")
        
        if personality.get("empathy", 0.5) > 0.7:
            traits.append("highly empathetic")
        
        if personality.get("curiosity", 0.5) > 0.7:
            traits.append("curious and inquisitive")
        
        if personality.get("supportiveness", 0.5) > 0.7:
            traits.append("supportive and encouraging")
        
        formality = personality.get("formality", 0.5)
        if formality > 0.7:
            traits.append("formal in communication")
        elif formality < 0.3:
            traits.append("casual and relaxed")
        
        if not traits:
            traits = ["balanced and adaptable"]
        
        return f"Personality: You are {', '.join(traits)}."
    
    def _classify_response_type(self, response: str, is_mention: bool) -> str:
        """Classify the type of response generated"""
        
        if is_mention:
            return "mention_reply"
        
        if "?" in response:
            return "question"
        
        if any(word in response.lower() for word in ["help", "suggest", "try", "should", "could"]):
            return "helpful"
        
        if any(word in response.lower() for word in ["understand", "feel", "sorry", "support"]):
            return "supportive"
        
        if any(word in response.lower() for word in ["interesting", "cool", "wow", "amazing"]):
            return "enthusiastic"
        
        return "conversational"
    
    def _calculate_response_confidence(
        self,
        response: str,
        emotional_context: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for generated response"""
        
        confidence = 0.5  # Base confidence
        
        # Length appropriateness
        if 10 <= len(response) <= 150:
            confidence += 0.2
        
        # Emotional appropriateness
        if emotional_context.get("requires_empathy", False):
            empathy_words = ["understand", "feel", "sorry", "support", "here for"]
            if any(word in response.lower() for word in empathy_words):
                confidence += 0.15
        
        # Memory relevance
        if memories:
            confidence += min(0.15, len(memories) * 0.03)
        
        # Safety check
        if not any(re.search(pattern, response.lower()) for pattern in self.safety_filters.values()):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _generate_thinking_process(
        self,
        message: str,
        response: str,
        analysis: ConversationInsight
    ) -> List[str]:
        """Generate explanation of thinking process for transparency"""
        
        process = []
        
        # Message analysis
        process.append(f"Analyzed incoming message: '{message[:50]}...'")
        
        # Context consideration
        if analysis.topic_shifts:
            process.append(f"Considered recent topics: {', '.join(analysis.topic_shifts[-2:])}")
        
        # Emotional response
        process.append(f"Matched emotional tone for appropriate response")
        
        # Response generation
        process.append(f"Generated response: '{response[:50]}...'")
        
        # Safety validation
        process.append("Validated response for safety and appropriateness")
        
        return process
    
    async def _calculate_safety_score(self, response: str) -> float:
        """Calculate safety score for response"""
        
        safety_score = 1.0
        
        # Check against safety filters
        for filter_name, pattern in self.safety_filters.items():
            if re.search(pattern, response.lower()):
                safety_score -= 0.2
                self.logger.warning(f"Safety filter triggered: {filter_name}")
        
        # Length check
        if len(response) > 300:
            safety_score -= 0.1
        
        # Spam-like patterns
        if response.count("!") > 3 or response.count("?") > 2:
            safety_score -= 0.1
        
        return max(0.0, safety_score)
    
    async def _validate_and_improve_response(
        self,
        response: ResponseGeneration,
        conversation: TelegramConversation,
        community: Optional[TelegramCommunity]
    ) -> ResponseGeneration:
        """Validate and improve generated response"""
        
        # Safety validation
        if response.safety_score < 0.7:
            # Generate safer alternative
            safe_alternatives = [
                "I appreciate you sharing that with me.",
                "That's an interesting perspective.",
                "Thanks for letting me know about that.",
                "I understand what you're saying."
            ]
            response.content = random.choice(safe_alternatives)
            response.safety_score = 0.9
            response.confidence *= 0.7
        
        # Length validation
        if len(response.content) > 200:
            # Truncate while preserving meaning
            sentences = response.content.split('.')
            truncated = sentences[0]
            if len(truncated) < 180 and len(sentences) > 1:
                truncated += ". " + sentences[1]
            response.content = truncated.strip()
        
        # Community appropriateness
        if community and community.formality_level == "formal":
            # Make response more formal
            response.content = response.content.replace("can't", "cannot")
            response.content = response.content.replace("won't", "will not")
            response.content = response.content.replace("didn't", "did not")
        
        return response
    
    async def _store_conversation_insights(
        self,
        conversation: TelegramConversation,
        message: str,
        response: ResponseGeneration,
        analysis: ConversationInsight
    ):
        """Store insights from conversation for future learning"""
        
        # Store in memory palace
        await self.memory.store_memory(
            content=f"Conversation: '{message}' -> '{response.content}'",
            memory_type="conversation_pattern",
            importance=response.confidence,
            context={
                "conversation_id": str(conversation.id),
                "response_type": response.response_type,
                "confidence": response.confidence,
                "topics": analysis.topic_shifts,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update conversation context
        conversation.add_context("last_response_confidence", response.confidence)
        conversation.add_context("last_response_type", response.response_type)
        
        if analysis.topic_shifts:
            conversation.add_context("recent_topics", analysis.topic_shifts[-5:])
    
    # Helper methods for conversation analysis
    async def _detect_topic_shifts(self, topics: List[str]) -> List[str]:
        """Detect topic shifts in conversation"""
        if len(topics) < 2:
            return topics
        
        # Simple topic shift detection - group similar topics
        unique_topics = []
        for topic in topics:
            if not any(topic.lower() in existing.lower() or existing.lower() in topic.lower() 
                      for existing in unique_topics):
                unique_topics.append(topic)
        
        return unique_topics[-5:]  # Return last 5 unique topics
    
    async def _analyze_engagement_patterns(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """Analyze engagement patterns in conversation"""
        
        if not messages:
            return {}
        
        our_messages = [m for m in messages if m.direction == MessageDirection.OUTGOING]
        their_messages = [m for m in messages if m.direction == MessageDirection.INCOMING]
        
        patterns = {
            "message_ratio": len(our_messages) / len(messages) if messages else 0,
            "avg_response_time": sum(m.response_time for m in messages if m.response_time) / len(messages) if messages else 0,
            "question_frequency": sum(1 for m in messages if m.content and "?" in m.content) / len(messages) if messages else 0,
            "engagement_level": "high" if len(their_messages) > len(our_messages) else "medium"
        }
        
        return patterns
    
    async def _identify_response_opportunities(
        self,
        message: str,
        recent_messages: List[ConversationMessage],
        community: Optional[TelegramCommunity]
    ) -> List[str]:
        """Identify specific response opportunities"""
        
        opportunities = []
        
        # Question in message
        if "?" in message:
            opportunities.append("answer_question")
        
        # Emotional content
        emotion_keywords = {
            "support": ["sad", "upset", "worried", "anxious", "frustrated"],
            "celebrate": ["happy", "excited", "great", "awesome", "congratulations"],
            "clarify": ["confused", "unclear", "don't understand"]
        }
        
        for opp_type, keywords in emotion_keywords.items():
            if any(keyword in message.lower() for keyword in keywords):
                opportunities.append(opp_type)
        
        # Topic expertise
        if community and community.topics:
            for topic in community.topics:
                if topic.lower() in message.lower():
                    opportunities.append(f"share_knowledge_{topic}")
        
        # Conversation continuation
        if not recent_messages or len(recent_messages) < 3:
            opportunities.append("build_rapport")
        
        return opportunities[:3]  # Limit to top 3
    
    async def _suggest_personality_adaptations(
        self,
        conversation: TelegramConversation,
        community: Optional[TelegramCommunity],
        emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Suggest personality adaptations for current context"""
        
        adaptations = {}
        
        # Community-based adaptations
        if community:
            if community.engagement_strategy == EngagementStrategy.LEADER:
                adaptations["confidence"] = 0.8
                adaptations["assertiveness"] = 0.7
            elif community.engagement_strategy == EngagementStrategy.PARTICIPANT:
                adaptations["collaborativeness"] = 0.8
        
        # Emotion-based adaptations
        if emotions:
            recent_negative = sum(1 for e in emotions[-3:] if e.get("sentiment") == "negative")
            if recent_negative >= 2:
                adaptations["empathy"] = 0.9
                adaptations["supportiveness"] = 0.8
        
        return adaptations