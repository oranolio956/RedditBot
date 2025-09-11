"""
Kelly Intelligence Service

AI-powered conversation intelligence, pattern recognition, sentiment analysis,
recommendation generation, and anomaly detection.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import statistics
import numpy as np
from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session

from app.database.connection import get_session
from app.models.kelly_intelligence import (
    ConversationInsight, ConversationPattern, AiRecommendation,
    AnomalyDetection, TopicAnalysis, IntelligenceReport
)
from app.models.kelly_analytics import ConversationAnalytics, QualityScore
from app.models.kelly_monitoring import AlertInstance
from app.core.redis import redis_manager
from app.core.claude_ai import claude_client

logger = logging.getLogger(__name__)

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    sentiment: str  # positive, negative, neutral
    confidence: float
    score: float  # -1.0 to 1.0
    emotions: Dict[str, float]

@dataclass
class PatternMatch:
    """Pattern matching result"""
    pattern_id: str
    pattern_name: str
    confidence: float
    relevance_score: float
    match_details: Dict[str, Any]

@dataclass
class ConversationRecommendations:
    """Comprehensive conversation recommendations"""
    coaching_suggestions: List[Dict[str, Any]]
    pattern_recommendations: List[PatternMatch]
    improvement_areas: List[str]
    success_indicators: List[str]

class KellyIntelligenceService:
    """
    Advanced AI-powered intelligence service for conversation analysis.
    
    Provides sentiment analysis, pattern recognition, anomaly detection,
    and intelligent recommendations for improving conversations.
    """
    
    def __init__(self):
        self.sentiment_cache_ttl = 3600  # 1 hour
        self.pattern_cache_ttl = 7200   # 2 hours
        self.anomaly_threshold = 2.0    # Standard deviations for anomaly detection
        
        # Emotion keywords for sentiment enhancement
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'thrilled'],
            'anger': ['angry', 'frustrated', 'annoyed', 'irritated', 'mad'],
            'fear': ['worried', 'scared', 'anxious', 'nervous', 'concerned'],
            'sadness': ['sad', 'disappointed', 'upset', 'discouraged', 'down'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished'],
            'trust': ['confident', 'sure', 'certain', 'comfortable', 'assured']
        }
    
    async def analyze_conversation_intelligence(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any],
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive conversation intelligence analysis.
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_data: Full conversation with messages and metadata
            include_recommendations: Whether to generate recommendations
            
        Returns:
            Complete intelligence analysis results
        """
        try:
            # Parallel analysis tasks
            tasks = [
                self.analyze_sentiment(conversation_data),
                self.detect_conversation_patterns(conversation_data),
                self.analyze_topics(conversation_id, conversation_data),
                self.detect_anomalies(conversation_id, conversation_data)
            ]
            
            if include_recommendations:
                tasks.append(self.generate_recommendations(conversation_id, conversation_data))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            sentiment_analysis = results[0] if not isinstance(results[0], Exception) else None
            pattern_matches = results[1] if not isinstance(results[1], Exception) else []
            topic_analysis = results[2] if not isinstance(results[2], Exception) else []
            anomalies = results[3] if not isinstance(results[3], Exception) else []
            recommendations = results[4] if include_recommendations and not isinstance(results[4], Exception) else None
            
            # Generate insights based on analysis
            insights = await self._generate_conversation_insights(
                conversation_id, conversation_data, sentiment_analysis,
                pattern_matches, topic_analysis, anomalies
            )
            
            # Compile comprehensive analysis
            analysis_result = {
                'conversation_id': conversation_id,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'sentiment_analysis': sentiment_analysis.__dict__ if sentiment_analysis else None,
                'pattern_matches': [p.__dict__ for p in pattern_matches],
                'topic_analysis': topic_analysis,
                'anomalies': anomalies,
                'insights': insights,
                'recommendations': recommendations.__dict__ if recommendations else None,
                'confidence_score': self._calculate_overall_confidence(
                    sentiment_analysis, pattern_matches, topic_analysis
                )
            }
            
            # Cache the analysis
            cache_key = f"conversation_intelligence:{conversation_id}"
            await redis_manager.set(
                cache_key,
                json.dumps(analysis_result, default=str),
                expire=self.sentiment_cache_ttl
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in conversation intelligence analysis: {str(e)}")
            raise
    
    async def analyze_sentiment(self, conversation_data: Dict[str, Any]) -> SentimentAnalysis:
        """
        Advanced sentiment analysis using Claude AI and keyword detection.
        
        Args:
            conversation_data: Conversation messages and metadata
            
        Returns:
            Detailed sentiment analysis
        """
        try:
            messages = conversation_data.get('messages', [])
            if not messages:
                return SentimentAnalysis(
                    sentiment='neutral',
                    confidence=0.5,
                    score=0.0,
                    emotions={}
                )
            
            # Combine all user messages for analysis
            user_messages = [
                msg['content'] for msg in messages 
                if msg.get('role') == 'user' and msg.get('content')
            ]
            
            if not user_messages:
                return SentimentAnalysis(
                    sentiment='neutral',
                    confidence=0.5,
                    score=0.0,
                    emotions={}
                )
            
            conversation_text = ' '.join(user_messages)
            
            # Use Claude AI for sentiment analysis
            sentiment_prompt = f"""
            Analyze the sentiment and emotions in this conversation:
            
            {conversation_text}
            
            Provide:
            1. Overall sentiment (positive, negative, neutral)
            2. Sentiment score (-1.0 to 1.0)
            3. Confidence level (0.0 to 1.0)
            4. Detected emotions with confidence scores
            5. Key emotional indicators
            
            Return as JSON format.
            """
            
            claude_response = await claude_client.analyze_text(sentiment_prompt)
            sentiment_data = json.loads(claude_response)
            
            # Enhance with keyword-based emotion detection
            emotion_scores = self._detect_emotions_by_keywords(conversation_text)
            
            # Combine Claude analysis with keyword detection
            combined_emotions = sentiment_data.get('emotions', {})
            for emotion, score in emotion_scores.items():
                if emotion in combined_emotions:
                    combined_emotions[emotion] = max(combined_emotions[emotion], score)
                else:
                    combined_emotions[emotion] = score
            
            return SentimentAnalysis(
                sentiment=sentiment_data.get('sentiment', 'neutral'),
                confidence=sentiment_data.get('confidence', 0.5),
                score=sentiment_data.get('score', 0.0),
                emotions=combined_emotions
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Return default sentiment if analysis fails
            return SentimentAnalysis(
                sentiment='neutral',
                confidence=0.3,
                score=0.0,
                emotions={}
            )
    
    async def detect_conversation_patterns(
        self, 
        conversation_data: Dict[str, Any]
    ) -> List[PatternMatch]:
        """
        Detect successful conversation patterns using AI and stored patterns.
        
        Args:
            conversation_data: Conversation messages and metadata
            
        Returns:
            List of matched patterns with confidence scores
        """
        try:
            # Get stored conversation patterns
            async with get_session() as session:
                stored_patterns = session.query(ConversationPattern).filter(
                    ConversationPattern.status == 'active'
                ).all()
            
            if not stored_patterns:
                return []
            
            messages = conversation_data.get('messages', [])
            conversation_flow = self._extract_conversation_flow(messages)
            
            pattern_matches = []
            
            # Check each stored pattern
            for pattern in stored_patterns:
                match_confidence = await self._calculate_pattern_match(
                    conversation_flow, pattern.pattern_signature
                )
                
                if match_confidence > 0.3:  # Threshold for pattern match
                    relevance_score = await self._calculate_pattern_relevance(
                        conversation_data, pattern
                    )
                    
                    pattern_match = PatternMatch(
                        pattern_id=str(pattern.id),
                        pattern_name=pattern.pattern_name,
                        confidence=match_confidence,
                        relevance_score=relevance_score,
                        match_details={
                            'pattern_type': pattern.pattern_type,
                            'success_rate': pattern.success_rate,
                            'applicable_scenarios': pattern.applicable_scenarios
                        }
                    )
                    pattern_matches.append(pattern_match)
            
            # Sort by relevance and confidence
            pattern_matches.sort(
                key=lambda x: x.relevance_score * x.confidence,
                reverse=True
            )
            
            return pattern_matches[:5]  # Return top 5 matches
            
        except Exception as e:
            logger.error(f"Error detecting conversation patterns: {str(e)}")
            return []
    
    async def analyze_topics(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze conversation topics and themes using AI.
        
        Args:
            conversation_id: Conversation identifier
            conversation_data: Conversation messages and metadata
            
        Returns:
            List of detected topics with analysis
        """
        try:
            messages = conversation_data.get('messages', [])
            if not messages:
                return []
            
            # Combine all messages for topic analysis
            full_conversation = ' '.join([
                msg.get('content', '') for msg in messages 
                if msg.get('content')
            ])
            
            # Use Claude AI for topic analysis
            topic_prompt = f"""
            Analyze the topics and themes in this conversation:
            
            {full_conversation}
            
            Identify:
            1. Main topics discussed (with confidence scores)
            2. Subtopics or related themes
            3. Business relevance of each topic (0.0 to 1.0)
            4. Emotional context of topic discussions
            5. Any product or service mentions
            6. Customer needs or pain points expressed
            
            Return as JSON with detailed topic analysis.
            """
            
            claude_response = await claude_client.analyze_text(topic_prompt)
            topic_data = json.loads(claude_response)
            
            # Store topic analysis in database
            topics_to_store = []
            
            for topic_info in topic_data.get('topics', []):
                topic_analysis = TopicAnalysis(
                    conversation_id=conversation_id,
                    account_id=conversation_data.get('account_id'),
                    user_id=conversation_data.get('user_id'),
                    topic_name=topic_info.get('name'),
                    topic_category=topic_info.get('category', 'general'),
                    subtopics=topic_info.get('subtopics', []),
                    detection_confidence=topic_info.get('confidence', 0.5),
                    detection_method='ai_analysis',
                    model_used='claude_ai',
                    sentiment=topic_info.get('sentiment'),
                    sentiment_score=topic_info.get('sentiment_score'),
                    urgency_level=topic_info.get('urgency', 'normal'),
                    mentioned_entities=topic_info.get('entities', []),
                    keywords=topic_info.get('keywords', []),
                    business_relevance=topic_info.get('business_relevance', 0.5),
                    product_relevance=topic_info.get('product_relevance', 0.5),
                    first_mentioned_at=datetime.utcnow(),
                    last_mentioned_at=datetime.utcnow(),
                    conversation_timestamp=datetime.fromisoformat(
                        conversation_data.get('timestamp', datetime.utcnow().isoformat())
                    ),
                    analysis_details=topic_info
                )
                topics_to_store.append(topic_analysis)
            
            # Save to database
            async with get_session() as session:
                for topic in topics_to_store:
                    session.add(topic)
                await session.commit()
            
            return topic_data.get('topics', [])
            
        except Exception as e:
            logger.error(f"Error analyzing topics: {str(e)}")
            return []
    
    async def detect_anomalies(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in conversation patterns and behavior.
        
        Args:
            conversation_id: Conversation identifier
            conversation_data: Conversation messages and metadata
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            
            # Get historical data for comparison
            account_id = conversation_data.get('account_id')
            if not account_id:
                return anomalies
            
            async with get_session() as session:
                # Get recent conversations for baseline
                recent_conversations = session.query(ConversationAnalytics).filter(
                    and_(
                        ConversationAnalytics.account_id == account_id,
                        ConversationAnalytics.started_at >= datetime.utcnow() - timedelta(days=30)
                    )
                ).all()
            
            if len(recent_conversations) < 5:  # Need minimum data for anomaly detection
                return anomalies
            
            # Extract current conversation metrics
            current_metrics = {
                'message_count': len(conversation_data.get('messages', [])),
                'avg_response_time': conversation_data.get('avg_response_time', 0),
                'quality_score': conversation_data.get('quality_score', 0.5),
                'duration_minutes': conversation_data.get('duration_minutes', 0)
            }
            
            # Calculate baselines from historical data
            baselines = {
                'message_count': [c.total_messages for c in recent_conversations],
                'avg_response_time': [c.avg_response_time_seconds for c in recent_conversations if c.avg_response_time_seconds],
                'quality_score': [c.conversation_quality_score for c in recent_conversations],
                'duration_minutes': [c.duration_minutes for c in recent_conversations if c.duration_minutes]
            }
            
            # Detect anomalies for each metric
            for metric_name, current_value in current_metrics.items():
                if metric_name not in baselines or not baselines[metric_name]:
                    continue
                
                baseline_values = baselines[metric_name]
                mean_baseline = statistics.mean(baseline_values)
                std_baseline = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
                
                if std_baseline == 0:
                    continue
                
                # Calculate z-score
                z_score = abs(current_value - mean_baseline) / std_baseline
                
                if z_score > self.anomaly_threshold:
                    # Determine anomaly type and severity
                    anomaly_type = 'performance' if metric_name in ['avg_response_time', 'duration_minutes'] else 'behavior'
                    severity = 'high' if z_score > 3.0 else 'medium'
                    
                    anomaly_details = {
                        'metric': metric_name,
                        'current_value': current_value,
                        'expected_value': mean_baseline,
                        'deviation_magnitude': z_score,
                        'severity': severity,
                        'type': anomaly_type
                    }
                    
                    # Store anomaly in database
                    anomaly_detection = AnomalyDetection(
                        conversation_id=conversation_id,
                        account_id=account_id,
                        user_id=conversation_data.get('user_id'),
                        anomaly_type=anomaly_type,
                        anomaly_category='negative' if z_score > 0 else 'positive',
                        severity=severity,
                        title=f"Anomalous {metric_name.replace('_', ' ')}",
                        description=f"Detected unusual {metric_name}: {current_value:.2f} (expected: {mean_baseline:.2f})",
                        detection_method='statistical',
                        confidence_score=min(z_score / 5.0, 1.0),  # Normalize confidence
                        anomaly_score=z_score,
                        expected_value=mean_baseline,
                        actual_value=current_value,
                        deviation_magnitude=z_score,
                        statistical_significance=z_score,
                        occurrence_time=datetime.fromisoformat(
                            conversation_data.get('timestamp', datetime.utcnow().isoformat())
                        ),
                        detection_data=anomaly_details
                    )
                    
                    async with get_session() as session:
                        session.add(anomaly_detection)
                        await session.commit()
                    
                    anomalies.append(anomaly_details)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    async def generate_recommendations(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any]
    ) -> ConversationRecommendations:
        """
        Generate AI coaching recommendations for conversation improvement.
        
        Args:
            conversation_id: Conversation identifier
            conversation_data: Conversation messages and metadata
            
        Returns:
            Comprehensive recommendations for improvement
        """
        try:
            # Analyze conversation for recommendation opportunities
            messages = conversation_data.get('messages', [])
            quality_score = conversation_data.get('quality_score', 0.5)
            
            # Use Claude AI to generate specific recommendations
            recommendation_prompt = f"""
            Analyze this conversation and provide specific coaching recommendations:
            
            Conversation Quality Score: {quality_score}
            Messages: {json.dumps(messages, indent=2)}
            
            Provide recommendations for:
            1. Response improvement (specific suggestions)
            2. Conversation flow optimization
            3. Empathy and tone enhancements
            4. Efficiency improvements
            5. Customer satisfaction tactics
            
            For each recommendation:
            - Specific actionable advice
            - Expected impact (low/medium/high)
            - Implementation difficulty (easy/medium/hard)
            - Priority level (low/medium/high/critical)
            
            Return as structured JSON.
            """
            
            claude_response = await claude_client.analyze_text(recommendation_prompt)
            recommendations_data = json.loads(claude_response)
            
            # Store recommendations in database
            coaching_suggestions = []
            
            for rec in recommendations_data.get('recommendations', []):
                ai_recommendation = AiRecommendation(
                    conversation_id=conversation_id,
                    user_id=conversation_data.get('user_id'),
                    account_id=conversation_data.get('account_id'),
                    recommendation_type='coaching',
                    category=rec.get('category', 'general'),
                    title=rec.get('title'),
                    description=rec.get('description'),
                    specific_suggestion=rec.get('suggestion'),
                    example_implementation=rec.get('example'),
                    based_on_analysis=rec.get('analysis', {}),
                    confidence_score=rec.get('confidence', 0.7),
                    ai_model_used='claude_ai',
                    expected_improvement=rec.get('expected_impact_percentage'),
                    impact_area=rec.get('impact_area', 'quality'),
                    priority=rec.get('priority', 'medium'),
                    urgency=rec.get('urgency', 'normal'),
                    difficulty_level=rec.get('difficulty', 'medium'),
                    trigger_event='conversation_analysis',
                    supporting_data=rec
                )
                
                coaching_suggestions.append({
                    'title': rec.get('title'),
                    'description': rec.get('description'),
                    'suggestion': rec.get('suggestion'),
                    'priority': rec.get('priority', 'medium'),
                    'impact': rec.get('expected_impact', 'medium'),
                    'difficulty': rec.get('difficulty', 'medium')
                })
                
                async with get_session() as session:
                    session.add(ai_recommendation)
                    await session.commit()
            
            # Get pattern recommendations
            pattern_matches = await self.detect_conversation_patterns(conversation_data)
            
            return ConversationRecommendations(
                coaching_suggestions=coaching_suggestions,
                pattern_recommendations=pattern_matches,
                improvement_areas=recommendations_data.get('improvement_areas', []),
                success_indicators=recommendations_data.get('success_indicators', [])
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ConversationRecommendations(
                coaching_suggestions=[],
                pattern_recommendations=[],
                improvement_areas=[],
                success_indicators=[]
            )
    
    async def get_successful_patterns(
        self,
        account_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        min_success_rate: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get successful conversation patterns for learning and replication.
        
        Args:
            account_id: Optional account filter
            pattern_type: Optional pattern type filter
            min_success_rate: Minimum success rate for patterns
            
        Returns:
            List of successful patterns with details
        """
        try:
            async with get_session() as session:
                query = session.query(ConversationPattern).filter(
                    and_(
                        ConversationPattern.status == 'active',
                        ConversationPattern.success_rate >= min_success_rate
                    )
                )
                
                if pattern_type:
                    query = query.filter(ConversationPattern.pattern_type == pattern_type)
                
                patterns = query.order_by(desc(ConversationPattern.success_rate)).all()
                
                pattern_list = []
                for pattern in patterns:
                    pattern_data = {
                        'id': str(pattern.id),
                        'name': pattern.pattern_name,
                        'type': pattern.pattern_type,
                        'category': pattern.pattern_category,
                        'description': pattern.description,
                        'success_rate': pattern.success_rate,
                        'sample_size': pattern.sample_size,
                        'applicable_scenarios': pattern.applicable_scenarios,
                        'performance_metrics': {
                            'avg_quality': pattern.avg_conversation_quality,
                            'avg_satisfaction': pattern.avg_customer_satisfaction,
                            'avg_resolution_time': pattern.avg_resolution_time_minutes,
                            'conversion_rate': pattern.conversion_rate
                        },
                        'usage_stats': {
                            'times_recommended': pattern.times_recommended,
                            'times_used': pattern.times_used,
                            'success_when_used': pattern.success_when_used
                        }
                    }
                    pattern_list.append(pattern_data)
                
                return pattern_list
                
        except Exception as e:
            logger.error(f"Error getting successful patterns: {str(e)}")
            return []
    
    def _detect_emotions_by_keywords(self, text: str) -> Dict[str, float]:
        """Detect emotions using keyword matching"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Normalize by text length and keyword count
                score = min(matches / len(keywords), 1.0)
                emotion_scores[emotion] = score
        
        return emotion_scores
    
    def _extract_conversation_flow(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract conversation flow patterns"""
        if not messages:
            return {}
        
        # Analyze message patterns
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        ai_messages = [msg for msg in messages if msg.get('role') == 'assistant']
        
        return {
            'total_exchanges': len(user_messages),
            'avg_user_message_length': statistics.mean([len(msg.get('content', '')) for msg in user_messages]) if user_messages else 0,
            'avg_ai_message_length': statistics.mean([len(msg.get('content', '')) for msg in ai_messages]) if ai_messages else 0,
            'conversation_initiator': messages[0].get('role') if messages else 'unknown',
            'message_ratio': len(ai_messages) / max(len(user_messages), 1),
            'conversation_length': len(messages)
        }
    
    async def _calculate_pattern_match(
        self,
        conversation_flow: Dict[str, Any],
        pattern_signature: Dict[str, Any]
    ) -> float:
        """Calculate how well a conversation matches a stored pattern"""
        if not pattern_signature:
            return 0.0
        
        match_score = 0.0
        total_weights = 0.0
        
        # Compare key metrics with weights
        metrics_to_compare = {
            'total_exchanges': 0.3,
            'message_ratio': 0.2,
            'avg_user_message_length': 0.2,
            'conversation_length': 0.3
        }
        
        for metric, weight in metrics_to_compare.items():
            if metric in conversation_flow and metric in pattern_signature:
                current_value = conversation_flow[metric]
                expected_value = pattern_signature[metric]
                
                if expected_value != 0:
                    # Calculate similarity (closer to 1.0 = better match)
                    difference_ratio = abs(current_value - expected_value) / expected_value
                    similarity = max(0.0, 1.0 - difference_ratio)
                    match_score += similarity * weight
                    total_weights += weight
        
        return match_score / max(total_weights, 1.0)
    
    async def _calculate_pattern_relevance(
        self,
        conversation_data: Dict[str, Any],
        pattern: 'ConversationPattern'
    ) -> float:
        """Calculate relevance of a pattern to current conversation context"""
        relevance_score = 0.5  # Base relevance
        
        # Check scenario applicability
        if pattern.applicable_scenarios:
            # This would involve more complex matching logic
            # For now, return base relevance
            pass
        
        # Adjust based on conversation quality and context
        quality_score = conversation_data.get('quality_score', 0.5)
        if quality_score < 0.7:  # Low quality conversations might benefit more from patterns
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    async def _generate_conversation_insights(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any],
        sentiment_analysis: Optional[SentimentAnalysis],
        pattern_matches: List[PatternMatch],
        topic_analysis: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate insights based on comprehensive analysis"""
        insights = []
        
        # Sentiment insights
        if sentiment_analysis and sentiment_analysis.confidence > 0.7:
            if sentiment_analysis.sentiment == 'negative' and sentiment_analysis.score < -0.5:
                insights.append({
                    'type': 'sentiment_alert',
                    'priority': 'high',
                    'title': 'Negative Customer Sentiment Detected',
                    'description': f"Customer sentiment is strongly negative (score: {sentiment_analysis.score:.2f})",
                    'recommended_actions': ['Follow up with customer', 'Review conversation quality', 'Consider escalation']
                })
        
        # Pattern insights
        if pattern_matches:
            best_pattern = pattern_matches[0]
            if best_pattern.confidence > 0.8:
                insights.append({
                    'type': 'pattern_match',
                    'priority': 'medium',
                    'title': f'Strong Pattern Match: {best_pattern.pattern_name}',
                    'description': f"Conversation closely matches successful pattern with {best_pattern.confidence:.1%} confidence",
                    'recommended_actions': ['Continue following pattern', 'Apply pattern lessons']
                })
        
        # Topic insights
        high_relevance_topics = [t for t in topic_analysis if t.get('business_relevance', 0) > 0.8]
        if high_relevance_topics:
            insights.append({
                'type': 'business_opportunity',
                'priority': 'high',
                'title': 'High Business Relevance Topics Detected',
                'description': f"Found {len(high_relevance_topics)} topics with high business relevance",
                'recommended_actions': ['Follow up on business opportunities', 'Schedule sales call']
            })
        
        # Anomaly insights
        high_severity_anomalies = [a for a in anomalies if a.get('severity') == 'high']
        if high_severity_anomalies:
            insights.append({
                'type': 'anomaly_alert',
                'priority': 'critical',
                'title': f'{len(high_severity_anomalies)} High Severity Anomalies Detected',
                'description': 'Unusual patterns detected that require attention',
                'recommended_actions': ['Investigate anomalies', 'Review conversation handling']
            })
        
        # Store insights in database
        for insight in insights:
            conversation_insight = ConversationInsight(
                conversation_id=conversation_id,
                account_id=conversation_data.get('account_id'),
                user_id=conversation_data.get('user_id'),
                insight_type=insight['type'],
                insight_category='intelligence',
                priority=insight['priority'],
                title=insight['title'],
                description=insight['description'],
                confidence_score=0.8,  # Default confidence for generated insights
                ai_model_used='kelly_intelligence',
                impact_score=0.7,
                actionable=True,
                recommended_actions=insight['recommended_actions']
            )
            
            async with get_session() as session:
                session.add(conversation_insight)
                await session.commit()
        
        return insights
    
    def _calculate_overall_confidence(
        self,
        sentiment_analysis: Optional[SentimentAnalysis],
        pattern_matches: List[PatternMatch],
        topic_analysis: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence in analysis results"""
        confidence_scores = []
        
        if sentiment_analysis:
            confidence_scores.append(sentiment_analysis.confidence)
        
        if pattern_matches:
            avg_pattern_confidence = statistics.mean([p.confidence for p in pattern_matches])
            confidence_scores.append(avg_pattern_confidence)
        
        if topic_analysis:
            topic_confidences = [t.get('confidence', 0.5) for t in topic_analysis]
            if topic_confidences:
                avg_topic_confidence = statistics.mean(topic_confidences)
                confidence_scores.append(avg_topic_confidence)
        
        return statistics.mean(confidence_scores) if confidence_scores else 0.5

# Create global intelligence service instance
kelly_intelligence_service = KellyIntelligenceService()