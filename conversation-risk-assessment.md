# Conversation Quality Risk Assessment Systems
## ML-Driven Real-Time Decision Framework

### Executive Summary
This document outlines comprehensive risk assessment systems for conversation quality, focusing on time-waster detection, engagement scoring, and behavioral pattern analysis. The system uses ML models for real-time decision-making in sales automation and customer service platforms.

## 1. Time-Waster Detection Algorithms

### 1.1 Multi-Signal Detection Framework
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from typing import Dict, List, Tuple
import pandas as pd

class TimeWasterDetector:
    def __init__(self):
        # Ensemble of models for robust detection
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        # Feature weights for different signals
        self.feature_weights = {
            'message_complexity': 0.15,
            'response_latency': 0.20,
            'question_to_answer_ratio': 0.25,
            'commitment_indicators': 0.20,
            'temporal_patterns': 0.20
        }
    
    def extract_time_waster_features(self, conversation_history: List[Dict]) -> Dict:
        """Extract features indicating time-wasting behavior"""
        
        features = {}
        
        # 1. Message Complexity Analysis
        features.update(self._analyze_message_complexity(conversation_history))
        
        # 2. Response Pattern Analysis
        features.update(self._analyze_response_patterns(conversation_history))
        
        # 3. Commitment Signal Analysis
        features.update(self._analyze_commitment_signals(conversation_history))
        
        # 4. Temporal Behavior Analysis
        features.update(self._analyze_temporal_patterns(conversation_history))
        
        return features
    
    def _analyze_message_complexity(self, conversation_history: List[Dict]) -> Dict:
        """Analyze message complexity patterns"""
        messages = [msg['text'] for msg in conversation_history if msg['sender'] == 'user']
        
        if not messages:
            return {'avg_message_length': 0, 'complexity_variance': 0}
        
        # Message length analysis
        lengths = [len(msg.split()) for msg in messages]
        
        # Lexical diversity (unique words / total words)
        all_words = ' '.join(messages).lower().split()
        lexical_diversity = len(set(all_words)) / len(all_words) if all_words else 0
        
        # Complexity indicators
        avg_length = np.mean(lengths)
        length_variance = np.var(lengths)
        
        # Question density (high questions = time waster)
        question_count = sum(1 for msg in messages if '?' in msg)
        question_density = question_count / len(messages)
        
        return {
            'avg_message_length': avg_length,
            'message_length_variance': length_variance,
            'lexical_diversity': lexical_diversity,
            'question_density': question_density
        }
    
    def _analyze_response_patterns(self, conversation_history: List[Dict]) -> Dict:
        """Analyze response timing and patterns"""
        user_messages = [msg for msg in conversation_history if msg['sender'] == 'user']
        bot_messages = [msg for msg in conversation_history if msg['sender'] == 'bot']
        
        if len(user_messages) < 2:
            return {'avg_response_time': 0, 'response_consistency': 0}
        
        # Response timing analysis
        response_times = []
        for i in range(1, len(user_messages)):
            time_diff = user_messages[i]['timestamp'] - user_messages[i-1]['timestamp']
            response_times.append(time_diff)
        
        avg_response_time = np.mean(response_times) if response_times else 0
        response_variance = np.var(response_times) if response_times else 0
        
        # Engagement degradation (responses getting shorter over time)
        if len(user_messages) >= 5:
            early_lengths = [len(msg['text']) for msg in user_messages[:3]]
            late_lengths = [len(msg['text']) for msg in user_messages[-3:]]
            engagement_degradation = np.mean(early_lengths) - np.mean(late_lengths)
        else:
            engagement_degradation = 0
        
        return {
            'avg_response_time': avg_response_time,
            'response_time_variance': response_variance,
            'engagement_degradation': max(0, engagement_degradation) / 100  # Normalize
        }
    
    def _analyze_commitment_signals(self, conversation_history: List[Dict]) -> Dict:
        """Analyze signals of genuine interest vs time-wasting"""
        user_messages = [msg['text'].lower() for msg in conversation_history 
                        if msg['sender'] == 'user']
        
        # Positive commitment indicators
        positive_signals = [
            'interested', 'when can', 'how much', 'pricing', 'schedule', 
            'meeting', 'demo', 'trial', 'purchase', 'budget', 'timeline',
            'decision', 'next step', 'sign up', 'contract', 'agreement'
        ]
        
        # Time-waster indicators
        negative_signals = [
            'maybe later', 'just browsing', 'just curious', 'not ready',
            'thinking about it', 'might be interested', 'possibly',
            'just looking', 'information only', 'no rush', 'no urgency'
        ]
        
        # Vague language patterns
        vague_patterns = [
            'might', 'could', 'perhaps', 'possibly', 'maybe', 'probably',
            'kind of', 'sort of', 'i guess', 'not sure'
        ]
        
        all_text = ' '.join(user_messages)
        
        positive_score = sum(1 for signal in positive_signals if signal in all_text)
        negative_score = sum(1 for signal in negative_signals if signal in all_text)
        vague_score = sum(1 for pattern in vague_patterns if pattern in all_text)
        
        total_messages = len(user_messages)
        
        return {
            'commitment_positive_ratio': positive_score / total_messages if total_messages > 0 else 0,
            'commitment_negative_ratio': negative_score / total_messages if total_messages > 0 else 0,
            'vague_language_ratio': vague_score / total_messages if total_messages > 0 else 0,
            'commitment_net_score': (positive_score - negative_score - vague_score) / total_messages if total_messages > 0 else 0
        }
    
    def _analyze_temporal_patterns(self, conversation_history: List[Dict]) -> Dict:
        """Analyze temporal behavior patterns"""
        if len(conversation_history) < 3:
            return {'session_length': 0, 'message_frequency': 0}
        
        timestamps = [msg['timestamp'] for msg in conversation_history]
        session_length = max(timestamps) - min(timestamps)
        
        # Message frequency (messages per minute)
        message_frequency = len(conversation_history) / (session_length / 60) if session_length > 0 else 0
        
        # Peak activity periods vs typical time-waster patterns
        user_messages = [msg for msg in conversation_history if msg['sender'] == 'user']
        if len(user_messages) >= 5:
            # Calculate acceleration/deceleration in conversation
            early_frequency = len(user_messages[:len(user_messages)//2])
            late_frequency = len(user_messages[len(user_messages)//2:])
            conversation_acceleration = late_frequency - early_frequency
        else:
            conversation_acceleration = 0
        
        return {
            'session_length': min(session_length / 3600, 24),  # Cap at 24 hours, convert to hours
            'message_frequency': min(message_frequency, 10),  # Cap at 10 messages/minute
            'conversation_acceleration': conversation_acceleration
        }
    
    def predict_time_waster_probability(self, conversation_history: List[Dict]) -> Dict:
        """Predict probability that user is a time waster"""
        features = self.extract_time_waster_features(conversation_history)
        
        # Convert to feature vector
        feature_vector = np.array([
            features.get('avg_message_length', 0),
            features.get('message_length_variance', 0),
            features.get('lexical_diversity', 0),
            features.get('question_density', 0),
            features.get('avg_response_time', 0),
            features.get('response_time_variance', 0),
            features.get('engagement_degradation', 0),
            features.get('commitment_positive_ratio', 0),
            features.get('commitment_negative_ratio', 0),
            features.get('vague_language_ratio', 0),
            features.get('commitment_net_score', 0),
            features.get('session_length', 0),
            features.get('message_frequency', 0),
            features.get('conversation_acceleration', 0)
        ]).reshape(1, -1)
        
        # Ensemble prediction
        predictions = {}
        for name, model in self.models.items():
            try:
                prob = model.predict_proba(feature_vector)[0][1]  # Probability of time waster
                predictions[name] = prob
            except:
                predictions[name] = 0.5  # Default neutral probability
        
        # Weighted ensemble
        ensemble_probability = np.mean(list(predictions.values()))
        
        # Risk categorization
        if ensemble_probability > 0.8:
            risk_level = 'HIGH'
        elif ensemble_probability > 0.6:
            risk_level = 'MEDIUM'
        elif ensemble_probability > 0.4:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'time_waster_probability': ensemble_probability,
            'risk_level': risk_level,
            'individual_predictions': predictions,
            'key_features': features,
            'confidence': self._calculate_confidence(predictions)
        }
    
    def _calculate_confidence(self, predictions: Dict) -> float:
        """Calculate confidence in ensemble prediction"""
        values = list(predictions.values())
        return 1.0 - np.std(values)  # Lower std = higher confidence
```

### 1.2 Real-Time Time-Waster Scoring
```python
class RealTimeTimeWasterScoring:
    def __init__(self):
        self.detector = TimeWasterDetector()
        self.score_history = {}  # Track scores over time
        
    async def update_time_waster_score(self, chat_id: int, new_message: Dict) -> Dict:
        """Update time waster score in real-time"""
        
        # Get conversation history
        conversation_history = await self.get_conversation_history(chat_id)
        conversation_history.append(new_message)
        
        # Calculate current score
        current_score = self.detector.predict_time_waster_probability(conversation_history)
        
        # Update score history
        if chat_id not in self.score_history:
            self.score_history[chat_id] = []
        
        self.score_history[chat_id].append({
            'timestamp': new_message['timestamp'],
            'score': current_score['time_waster_probability'],
            'risk_level': current_score['risk_level']
        })
        
        # Calculate trend
        score_trend = self._calculate_score_trend(chat_id)
        
        return {
            **current_score,
            'score_trend': score_trend,
            'recommendation': self._generate_recommendation(current_score, score_trend)
        }
    
    def _calculate_score_trend(self, chat_id: int) -> str:
        """Calculate whether time waster score is trending up or down"""
        if chat_id not in self.score_history or len(self.score_history[chat_id]) < 3:
            return 'INSUFFICIENT_DATA'
        
        recent_scores = [entry['score'] for entry in self.score_history[chat_id][-5:]]
        
        if len(recent_scores) >= 3:
            early_avg = np.mean(recent_scores[:2])
            late_avg = np.mean(recent_scores[-2:])
            
            if late_avg > early_avg + 0.1:
                return 'INCREASING'
            elif early_avg > late_avg + 0.1:
                return 'DECREASING' 
            else:
                return 'STABLE'
        
        return 'STABLE'
    
    def _generate_recommendation(self, current_score: Dict, trend: str) -> Dict:
        """Generate actionable recommendations based on score and trend"""
        probability = current_score['time_waster_probability']
        risk_level = current_score['risk_level']
        
        if risk_level == 'HIGH':
            if trend == 'INCREASING':
                return {
                    'action': 'TERMINATE',
                    'priority': 'IMMEDIATE',
                    'message': 'High probability time waster with increasing risk - recommend immediate termination',
                    'suggested_response': 'Thank you for your interest. Feel free to contact us when you\'re ready to move forward.'
                }
            else:
                return {
                    'action': 'QUALIFY_HARD',
                    'priority': 'HIGH',
                    'message': 'High probability time waster - implement hard qualification',
                    'suggested_response': 'To best assist you, can you share your specific timeline and budget for this project?'
                }
        
        elif risk_level == 'MEDIUM':
            return {
                'action': 'QUALIFY_SOFT',
                'priority': 'MEDIUM',
                'message': 'Medium probability time waster - implement soft qualification',
                'suggested_response': 'What specific outcomes are you looking to achieve with this solution?'
            }
        
        else:
            return {
                'action': 'CONTINUE',
                'priority': 'LOW',
                'message': 'Low time waster probability - continue normal conversation',
                'suggested_response': None
            }
```

## 2. Engagement Scoring Metrics

### 2.1 Multi-Dimensional Engagement Model
```python
class EngagementScorer:
    def __init__(self):
        # Engagement dimension weights
        self.dimension_weights = {
            'attention': 0.25,      # How focused is the user
            'interest': 0.25,       # Level of genuine interest
            'interaction': 0.20,    # Quality of interactions
            'progression': 0.15,    # Movement towards goal
            'emotional': 0.15       # Emotional engagement
        }
        
        # Decay factors for temporal relevance
        self.temporal_decay = {
            'immediate': 1.0,       # Last 5 messages
            'recent': 0.8,          # Last 10 messages  
            'older': 0.6            # Older messages
        }
    
    def calculate_engagement_score(self, conversation_history: List[Dict]) -> Dict:
        """Calculate multi-dimensional engagement score"""
        
        if not conversation_history:
            return self._create_zero_score()
        
        # Calculate individual dimension scores
        attention_score = self._calculate_attention_score(conversation_history)
        interest_score = self._calculate_interest_score(conversation_history)
        interaction_score = self._calculate_interaction_score(conversation_history)
        progression_score = self._calculate_progression_score(conversation_history)
        emotional_score = self._calculate_emotional_score(conversation_history)
        
        # Apply temporal weighting
        weighted_scores = self._apply_temporal_weighting(
            conversation_history,
            {
                'attention': attention_score,
                'interest': interest_score, 
                'interaction': interaction_score,
                'progression': progression_score,
                'emotional': emotional_score
            }
        )
        
        # Calculate overall engagement score
        overall_score = sum(
            weighted_scores[dim] * weight 
            for dim, weight in self.dimension_weights.items()
        )
        
        # Determine engagement level
        engagement_level = self._determine_engagement_level(overall_score)
        
        return {
            'overall_score': overall_score,
            'engagement_level': engagement_level,
            'dimension_scores': weighted_scores,
            'score_breakdown': {
                'attention': attention_score,
                'interest': interest_score,
                'interaction': interaction_score,
                'progression': progression_score,
                'emotional': emotional_score
            },
            'insights': self._generate_engagement_insights(weighted_scores)
        }
    
    def _calculate_attention_score(self, conversation_history: List[Dict]) -> float:
        """Calculate attention/focus score"""
        user_messages = [msg for msg in conversation_history if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        # Response time consistency (quick, consistent responses = high attention)
        response_times = []
        for i in range(1, len(user_messages)):
            time_diff = user_messages[i]['timestamp'] - user_messages[i-1]['timestamp']
            response_times.append(time_diff)
        
        if response_times:
            avg_response_time = np.mean(response_times)
            response_consistency = 1.0 / (1.0 + np.std(response_times) / 60)  # Normalize by minute
            
            # Ideal response time is 30 seconds to 5 minutes
            time_score = 1.0 if 30 <= avg_response_time <= 300 else 0.5
            
            attention_score = (response_consistency + time_score) / 2
        else:
            attention_score = 0.5
        
        # Message length consistency (shows sustained attention)
        message_lengths = [len(msg['text'].split()) for msg in user_messages]
        if len(message_lengths) > 1:
            length_consistency = 1.0 / (1.0 + np.std(message_lengths) / 10)  # Normalize
            attention_score = (attention_score + length_consistency) / 2
        
        return min(1.0, attention_score)
    
    def _calculate_interest_score(self, conversation_history: List[Dict]) -> float:
        """Calculate genuine interest score"""
        user_messages = [msg['text'].lower() for msg in conversation_history 
                        if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        all_text = ' '.join(user_messages)
        
        # High interest indicators
        high_interest_signals = [
            'tell me more', 'how does', 'what about', 'can you explain',
            'i want to know', 'interested in', 'sounds good', 'that\'s great',
            'perfect', 'exactly', 'love it', 'impressive', 'amazing'
        ]
        
        # Medium interest indicators
        medium_interest_signals = [
            'interesting', 'good', 'nice', 'okay', 'i see', 'makes sense',
            'understood', 'got it', 'thanks', 'helpful'
        ]
        
        # Low interest indicators
        low_interest_signals = [
            'maybe', 'not sure', 'hmm', 'okay', 'fine', 'whatever',
            'i guess', 'possibly', 'might be'
        ]
        
        high_count = sum(1 for signal in high_interest_signals if signal in all_text)
        medium_count = sum(1 for signal in medium_interest_signals if signal in all_text)
        low_count = sum(1 for signal in low_interest_signals if signal in all_text)
        
        total_messages = len(user_messages)
        
        # Weighted interest score
        interest_score = (high_count * 1.0 + medium_count * 0.6 + low_count * 0.2) / total_messages
        
        return min(1.0, interest_score)
    
    def _calculate_interaction_score(self, conversation_history: List[Dict]) -> float:
        """Calculate quality of interactions score"""
        user_messages = [msg for msg in conversation_history if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        # Question quality (specific questions = higher score)
        questions = [msg['text'] for msg in user_messages if '?' in msg['text']]
        
        # Specific question indicators
        specific_question_patterns = [
            'how much', 'when', 'where', 'who', 'what time', 'which',
            'can you', 'do you', 'will you', 'would you', 'could you'
        ]
        
        specific_questions = 0
        for question in questions:
            question_lower = question.lower()
            if any(pattern in question_lower for pattern in specific_question_patterns):
                specific_questions += 1
        
        question_quality = specific_questions / len(user_messages) if user_messages else 0
        
        # Response depth (longer responses = more engagement)
        avg_message_length = np.mean([len(msg['text'].split()) for msg in user_messages])
        depth_score = min(1.0, avg_message_length / 20)  # Normalize to 20 words
        
        # Follow-up patterns (building on previous messages)
        followup_indicators = ['also', 'and', 'additionally', 'furthermore', 'what else', 'another']
        followup_count = sum(1 for msg in user_messages 
                           if any(indicator in msg['text'].lower() for indicator in followup_indicators))
        followup_score = followup_count / len(user_messages) if user_messages else 0
        
        interaction_score = (question_quality + depth_score + followup_score) / 3
        
        return min(1.0, interaction_score)
    
    def _calculate_progression_score(self, conversation_history: List[Dict]) -> float:
        """Calculate progression towards goals score"""
        user_messages = [msg['text'].lower() for msg in conversation_history 
                        if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        # Progressive commitment indicators (ordered by sales funnel stage)
        progression_stages = {
            'awareness': ['learn about', 'tell me about', 'what is', 'how does'],
            'interest': ['interested in', 'sounds good', 'tell me more', 'benefits'],
            'consideration': ['pricing', 'cost', 'plans', 'options', 'features', 'compare'],
            'intent': ['buy', 'purchase', 'sign up', 'get started', 'trial', 'demo'],
            'decision': ['ready to', 'let\'s do it', 'yes', 'agree', 'contract', 'when can we']
        }
        
        stage_weights = {'awareness': 0.2, 'interest': 0.4, 'consideration': 0.6, 'intent': 0.8, 'decision': 1.0}
        
        all_text = ' '.join(user_messages)
        max_stage_score = 0
        
        for stage, indicators in progression_stages.items():
            if any(indicator in all_text for indicator in indicators):
                max_stage_score = max(max_stage_score, stage_weights[stage])
        
        # Progression velocity (advancement through stages)
        if len(user_messages) >= 5:
            early_text = ' '.join(user_messages[:len(user_messages)//2])
            late_text = ' '.join(user_messages[len(user_messages)//2:])
            
            early_stage = max([stage_weights[stage] for stage, indicators in progression_stages.items() 
                              if any(indicator in early_text for indicator in indicators)] or [0])
            late_stage = max([stage_weights[stage] for stage, indicators in progression_stages.items() 
                             if any(indicator in late_text for indicator in indicators)] or [0])
            
            progression_velocity = max(0, late_stage - early_stage)
        else:
            progression_velocity = 0
        
        progression_score = (max_stage_score + progression_velocity) / 2
        
        return min(1.0, progression_score)
    
    def _calculate_emotional_score(self, conversation_history: List[Dict]) -> float:
        """Calculate emotional engagement score"""
        user_messages = [msg['text'].lower() for msg in conversation_history 
                        if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        all_text = ' '.join(user_messages)
        
        # Positive emotional indicators
        positive_emotions = [
            'excited', 'great', 'perfect', 'amazing', 'love', 'fantastic',
            'excellent', 'wonderful', 'awesome', 'brilliant', 'impressed',
            'thrilled', 'delighted', '!', ':)', 'thank you'
        ]
        
        # Strong engagement indicators
        strong_emotions = [
            'wow', 'incredible', 'unbelievable', 'outstanding', 'phenomenal',
            '!!!', 'absolutely', 'definitely', 'certainly', 'exactly'
        ]
        
        # Count emotional expressions
        positive_count = sum(1 for emotion in positive_emotions if emotion in all_text)
        strong_count = sum(1 for emotion in strong_emotions if emotion in all_text)
        
        # Exclamation and emoji usage (indicators of emotional engagement)
        exclamation_count = all_text.count('!')
        
        total_emotional_signals = positive_count + (strong_count * 1.5) + (exclamation_count * 0.5)
        total_messages = len(user_messages)
        
        emotional_score = total_emotional_signals / total_messages if total_messages > 0 else 0
        
        return min(1.0, emotional_score)
    
    def _apply_temporal_weighting(self, conversation_history: List[Dict], scores: Dict) -> Dict:
        """Apply temporal weighting to give more importance to recent engagement"""
        if len(conversation_history) <= 5:
            return scores  # No temporal weighting for short conversations
        
        total_messages = len(conversation_history)
        
        # Recent messages get higher weight
        recent_weight = 0.5
        older_weight = 0.5
        
        # Calculate weighted average (simple implementation)
        # In practice, you'd apply this per dimension based on message recency
        weighted_scores = {}
        for dimension, score in scores.items():
            # Apply slight boost to recent performance
            weighted_scores[dimension] = score * (1.0 + recent_weight * 0.1)
        
        return weighted_scores
    
    def _determine_engagement_level(self, overall_score: float) -> str:
        """Determine engagement level category"""
        if overall_score >= 0.8:
            return 'VERY_HIGH'
        elif overall_score >= 0.6:
            return 'HIGH'
        elif overall_score >= 0.4:
            return 'MEDIUM'
        elif overall_score >= 0.2:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _generate_engagement_insights(self, scores: Dict) -> List[str]:
        """Generate actionable insights based on engagement scores"""
        insights = []
        
        # Identify strongest and weakest dimensions
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_scores[0]
        weakest = sorted_scores[-1]
        
        insights.append(f"Strongest engagement: {strongest[0]} ({strongest[1]:.2f})")
        insights.append(f"Area for improvement: {weakest[0]} ({weakest[1]:.2f})")
        
        # Specific recommendations
        if scores['attention'] < 0.3:
            insights.append("Low attention - consider shorter, more engaging responses")
        
        if scores['interest'] > 0.7:
            insights.append("High interest detected - good time to advance conversation")
        
        if scores['progression'] < 0.3:
            insights.append("Low progression - focus on qualifying and advancing")
        
        return insights
    
    def _create_zero_score(self) -> Dict:
        """Create zero engagement score for empty conversations"""
        return {
            'overall_score': 0.0,
            'engagement_level': 'NONE',
            'dimension_scores': {dim: 0.0 for dim in self.dimension_weights.keys()},
            'score_breakdown': {dim: 0.0 for dim in self.dimension_weights.keys()},
            'insights': ['No conversation data available']
        }
```

### 2.2 Real-Time Engagement Monitoring
```python
class RealTimeEngagementMonitor:
    def __init__(self):
        self.scorer = EngagementScorer()
        self.engagement_history = {}
        self.alert_thresholds = {
            'drop_alert': 0.3,      # Alert if engagement drops by 30%
            'low_engagement': 0.2,  # Alert if overall engagement < 20%
            'high_engagement': 0.8  # Opportunity alert if engagement > 80%
        }
    
    async def monitor_engagement(self, chat_id: int, conversation_history: List[Dict]) -> Dict:
        """Monitor engagement in real-time and generate alerts"""
        
        current_engagement = self.scorer.calculate_engagement_score(conversation_history)
        
        # Track engagement history
        if chat_id not in self.engagement_history:
            self.engagement_history[chat_id] = []
        
        self.engagement_history[chat_id].append({
            'timestamp': conversation_history[-1]['timestamp'] if conversation_history else time.time(),
            'score': current_engagement['overall_score'],
            'level': current_engagement['engagement_level']
        })
        
        # Generate alerts
        alerts = self._check_engagement_alerts(chat_id, current_engagement)
        
        # Calculate trends
        trend_analysis = self._analyze_engagement_trend(chat_id)
        
        return {
            **current_engagement,
            'alerts': alerts,
            'trend_analysis': trend_analysis,
            'recommendations': self._generate_engagement_recommendations(
                current_engagement, trend_analysis, alerts
            )
        }
    
    def _check_engagement_alerts(self, chat_id: int, current_engagement: Dict) -> List[Dict]:
        """Check for engagement-based alerts"""
        alerts = []
        current_score = current_engagement['overall_score']
        
        # Check for engagement drop
        if (chat_id in self.engagement_history and 
            len(self.engagement_history[chat_id]) > 3):
            
            recent_scores = [entry['score'] for entry in self.engagement_history[chat_id][-3:]]
            avg_recent = np.mean(recent_scores)
            
            if len(self.engagement_history[chat_id]) > 6:
                older_scores = [entry['score'] for entry in self.engagement_history[chat_id][-6:-3]]
                avg_older = np.mean(older_scores)
                
                if avg_older - avg_recent > self.alert_thresholds['drop_alert']:
                    alerts.append({
                        'type': 'ENGAGEMENT_DROP',
                        'severity': 'HIGH',
                        'message': f'Engagement dropped by {(avg_older - avg_recent):.1%}',
                        'action_required': True
                    })
        
        # Check for low engagement
        if current_score < self.alert_thresholds['low_engagement']:
            alerts.append({
                'type': 'LOW_ENGAGEMENT',
                'severity': 'MEDIUM',
                'message': f'Very low engagement detected ({current_score:.1%})',
                'action_required': True
            })
        
        # Check for high engagement (opportunity)
        if current_score > self.alert_thresholds['high_engagement']:
            alerts.append({
                'type': 'HIGH_ENGAGEMENT_OPPORTUNITY',
                'severity': 'LOW',
                'message': f'High engagement detected ({current_score:.1%}) - opportunity to advance',
                'action_required': False
            })
        
        return alerts
    
    def _analyze_engagement_trend(self, chat_id: int) -> Dict:
        """Analyze engagement trends over time"""
        if (chat_id not in self.engagement_history or 
            len(self.engagement_history[chat_id]) < 3):
            return {'trend': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        scores = [entry['score'] for entry in self.engagement_history[chat_id]]
        
        # Simple linear trend analysis
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        
        # Determine trend direction
        if slope > 0.05:
            trend = 'INCREASING'
        elif slope < -0.05:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'
        
        # Calculate trend confidence based on R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((scores - y_pred) ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, r_squared)
        
        return {
            'trend': trend,
            'slope': slope,
            'confidence': confidence,
            'prediction': self._predict_next_engagement(scores, slope, intercept)
        }
    
    def _predict_next_engagement(self, scores: List[float], slope: float, intercept: float) -> float:
        """Predict next engagement score based on trend"""
        next_x = len(scores)
        predicted_score = slope * next_x + intercept
        return max(0, min(1, predicted_score))  # Clamp between 0 and 1
    
    def _generate_engagement_recommendations(self, engagement: Dict, trend: Dict, alerts: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations based on engagement analysis"""
        recommendations = []
        
        current_score = engagement['overall_score']
        engagement_level = engagement['engagement_level']
        
        # High-priority alerts
        high_priority_alerts = [alert for alert in alerts if alert['severity'] == 'HIGH']
        if high_priority_alerts:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'IMMEDIATE_INTERVENTION',
                'message': 'Engagement dropping rapidly - implement recovery strategy',
                'tactics': ['Ask engaging question', 'Change topic', 'Provide value proposition']
            })
        
        # Trend-based recommendations
        if trend['trend'] == 'DECREASING' and trend['confidence'] > 0.6:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'ENGAGEMENT_RECOVERY',
                'message': 'Engagement trending downward - focus on re-engagement',
                'tactics': ['Summarize value', 'Ask direct question', 'Introduce new benefit']
            })
        
        elif trend['trend'] == 'INCREASING' and current_score > 0.6:
            recommendations.append({
                'priority': 'LOW',
                'action': 'CAPITALIZE_MOMENTUM',
                'message': 'Strong engagement momentum - good time to advance',
                'tactics': ['Ask for commitment', 'Schedule next step', 'Present call-to-action']
            })
        
        # Level-based recommendations
        if engagement_level == 'VERY_LOW':
            recommendations.append({
                'priority': 'HIGH',
                'action': 'QUALIFICATION_CHECK',
                'message': 'Very low engagement - verify prospect qualification',
                'tactics': ['Confirm interest', 'Check decision-making authority', 'Verify timeline']
            })
        
        elif engagement_level == 'VERY_HIGH':
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'ADVANCE_CONVERSATION',
                'message': 'Excellent engagement - advance to next stage',
                'tactics': ['Present specific solution', 'Discuss implementation', 'Address objections']
            })
        
        return recommendations
```

## 3. Conversation Quality Indicators

### 3.1 Multi-Factor Quality Assessment
```python
class ConversationQualityIndicators:
    def __init__(self):
        self.quality_dimensions = {
            'relevance': 0.25,      # How relevant are responses
            'coherence': 0.20,      # Logical flow and consistency
            'depth': 0.20,          # Depth of discussion
            'clarity': 0.15,        # Clarity of communication
            'goal_alignment': 0.20  # Alignment with conversation goals
        }
        
        # NLP models for quality assessment
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
        self.coherence_model = self._initialize_coherence_model()
        
    def assess_conversation_quality(self, conversation_history: List[Dict], goals: List[str] = None) -> Dict:
        """Comprehensive conversation quality assessment"""
        
        if not conversation_history:
            return self._create_zero_quality_score()
        
        # Calculate individual quality dimensions
        relevance_score = self._calculate_relevance_score(conversation_history, goals)
        coherence_score = self._calculate_coherence_score(conversation_history)
        depth_score = self._calculate_depth_score(conversation_history)
        clarity_score = self._calculate_clarity_score(conversation_history)
        goal_alignment_score = self._calculate_goal_alignment_score(conversation_history, goals)
        
        # Calculate weighted overall quality score
        quality_scores = {
            'relevance': relevance_score,
            'coherence': coherence_score,
            'depth': depth_score,
            'clarity': clarity_score,
            'goal_alignment': goal_alignment_score
        }
        
        overall_quality = sum(
            quality_scores[dimension] * weight
            for dimension, weight in self.quality_dimensions.items()
        )
        
        # Quality level categorization
        quality_level = self._determine_quality_level(overall_quality)
        
        # Generate quality insights
        insights = self._generate_quality_insights(quality_scores)
        
        return {
            'overall_quality_score': overall_quality,
            'quality_level': quality_level,
            'dimension_scores': quality_scores,
            'quality_insights': insights,
            'improvement_suggestions': self._generate_improvement_suggestions(quality_scores),
            'quality_trend': self._analyze_quality_trend(conversation_history)
        }
    
    def _calculate_relevance_score(self, conversation_history: List[Dict], goals: List[str] = None) -> float:
        """Calculate how relevant the conversation is to stated goals"""
        if not goals:
            # If no specific goals, assess general business conversation relevance
            goals = ['product inquiry', 'service request', 'support', 'sales', 'information']
        
        user_messages = [msg['text'].lower() for msg in conversation_history 
                        if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        all_text = ' '.join(user_messages)
        
        # Create relevance keywords for different goal types
        goal_keywords = {
            'product inquiry': ['product', 'features', 'specifications', 'how does', 'what is', 'tell me about'],
            'service request': ['help', 'support', 'service', 'assistance', 'problem', 'issue'],
            'support': ['error', 'bug', 'not working', 'problem', 'fix', 'troubleshoot'],
            'sales': ['price', 'cost', 'buy', 'purchase', 'demo', 'trial', 'pricing'],
            'information': ['information', 'details', 'learn', 'understand', 'explain']
        }
        
        relevance_scores = []
        for goal in goals:
            goal_lower = goal.lower()
            if goal_lower in goal_keywords:
                keywords = goal_keywords[goal_lower]
                keyword_matches = sum(1 for keyword in keywords if keyword in all_text)
                goal_relevance = keyword_matches / len(keywords)
                relevance_scores.append(goal_relevance)
            else:
                # Custom goal - use simple keyword matching
                relevance_scores.append(0.5 if goal_lower in all_text else 0.1)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _calculate_coherence_score(self, conversation_history: List[Dict]) -> float:
        """Calculate logical flow and consistency of conversation"""
        if len(conversation_history) < 3:
            return 0.5  # Insufficient data
        
        # Topic consistency analysis
        user_messages = [msg['text'] for msg in conversation_history if msg['sender'] == 'user']
        bot_messages = [msg['text'] for msg in conversation_history if msg['sender'] == 'bot']
        
        # Calculate topic drift (simplified)
        topic_consistency = self._calculate_topic_consistency(user_messages)
        
        # Response relevance (how well bot responses relate to user messages)
        response_relevance = self._calculate_response_relevance(user_messages, bot_messages)
        
        # Conversation flow (logical progression)
        flow_score = self._calculate_conversation_flow(conversation_history)
        
        coherence_score = (topic_consistency + response_relevance + flow_score) / 3
        
        return min(1.0, coherence_score)
    
    def _calculate_topic_consistency(self, user_messages: List[str]) -> float:
        """Calculate how consistently user stays on topic"""
        if len(user_messages) < 3:
            return 0.5
        
        # Simple approach: keyword overlap between messages
        all_keywords = []
        message_keywords = []
        
        for message in user_messages:
            # Extract keywords (simplified - remove stop words, get important words)
            words = [word.lower() for word in message.split() 
                    if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'they']]
            message_keywords.append(set(words))
            all_keywords.extend(words)
        
        if not all_keywords:
            return 0.5
        
        # Calculate average keyword overlap between consecutive messages
        overlaps = []
        for i in range(1, len(message_keywords)):
            if message_keywords[i] and message_keywords[i-1]:
                overlap = len(message_keywords[i] & message_keywords[i-1])
                union = len(message_keywords[i] | message_keywords[i-1])
                if union > 0:
                    overlaps.append(overlap / union)
        
        return np.mean(overlaps) if overlaps else 0.5
    
    def _calculate_response_relevance(self, user_messages: List[str], bot_messages: List[str]) -> float:
        """Calculate how relevant bot responses are to user messages"""
        if not user_messages or not bot_messages:
            return 0.5
        
        # Simplified relevance scoring based on keyword overlap
        relevance_scores = []
        
        min_length = min(len(user_messages), len(bot_messages))
        
        for i in range(min_length):
            user_words = set(word.lower() for word in user_messages[i].split() if len(word) > 3)
            bot_words = set(word.lower() for word in bot_messages[i].split() if len(word) > 3)
            
            if user_words and bot_words:
                overlap = len(user_words & bot_words)
                union = len(user_words | bot_words)
                if union > 0:
                    relevance_scores.append(overlap / union)
        
        return np.mean(relevance_scores) if relevance_scores else 0.5
    
    def _calculate_conversation_flow(self, conversation_history: List[Dict]) -> float:
        """Calculate logical flow of conversation"""
        # Simplified flow analysis based on message types and patterns
        
        # Check for natural conversation patterns
        patterns = {
            'question_answer': 0,
            'statement_acknowledgment': 0,
            'request_response': 0
        }
        
        for i in range(len(conversation_history) - 1):
            current_msg = conversation_history[i]['text'].lower()
            next_msg = conversation_history[i + 1]['text'].lower()
            
            # Question followed by answer
            if '?' in current_msg and current_msg != next_msg:
                patterns['question_answer'] += 1
            
            # Request followed by response
            request_indicators = ['can you', 'please', 'could you', 'would you']
            if any(indicator in current_msg for indicator in request_indicators):
                patterns['request_response'] += 1
        
        total_exchanges = max(1, len(conversation_history) - 1)
        total_patterns = sum(patterns.values())
        
        flow_score = total_patterns / total_exchanges
        
        return min(1.0, flow_score)
    
    def _calculate_depth_score(self, conversation_history: List[Dict]) -> float:
        """Calculate depth and substance of conversation"""
        user_messages = [msg['text'] for msg in conversation_history if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        # Message complexity indicators
        avg_message_length = np.mean([len(msg.split()) for msg in user_messages])
        
        # Technical/specific vocabulary usage
        technical_indicators = [
            'specifically', 'particularly', 'detailed', 'in-depth', 'comprehensive',
            'analysis', 'implementation', 'integration', 'configuration', 'optimization',
            'requirements', 'specifications', 'methodology', 'approach', 'strategy'
        ]
        
        all_text = ' '.join(user_messages).lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in all_text)
        technical_density = technical_count / len(user_messages)
        
        # Question depth (specific vs. generic questions)
        questions = [msg for msg in user_messages if '?' in msg]
        specific_question_indicators = [
            'how exactly', 'what specific', 'which particular', 'when precisely',
            'why specifically', 'how much', 'how many', 'what type of'
        ]
        
        specific_questions = sum(1 for question in questions 
                               if any(indicator in question.lower() for indicator in specific_question_indicators))
        
        question_depth = specific_questions / len(questions) if questions else 0
        
        # Combine depth indicators
        length_score = min(1.0, avg_message_length / 15)  # Normalize to 15 words average
        depth_score = (length_score + technical_density + question_depth) / 3
        
        return min(1.0, depth_score)
    
    def _calculate_clarity_score(self, conversation_history: List[Dict]) -> float:
        """Calculate clarity of communication"""
        user_messages = [msg['text'] for msg in conversation_history if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        clarity_scores = []
        
        for message in user_messages:
            # Grammar and structure indicators (simplified)
            words = message.split()
            
            # Sentence structure (presence of proper punctuation)
            has_punctuation = any(punct in message for punct in '.!?')
            punctuation_score = 1.0 if has_punctuation else 0.5
            
            # Spelling approximation (very simplified)
            # In practice, you'd use a spell checker
            potential_typos = sum(1 for word in words if len(word) > 10)  # Very long words might be typos
            typo_score = 1.0 - min(0.5, potential_typos / len(words))
            
            # Clarity words usage
            clarity_words = ['clearly', 'specifically', 'exactly', 'precisely', 'namely']
            vague_words = ['maybe', 'probably', 'possibly', 'kind of', 'sort of', 'i guess']
            
            clarity_count = sum(1 for word in clarity_words if word in message.lower())
            vague_count = sum(1 for word in vague_words if word in message.lower())
            
            clarity_word_score = (clarity_count - vague_count) / len(words) + 0.5
            clarity_word_score = max(0, min(1, clarity_word_score))
            
            message_clarity = (punctuation_score + typo_score + clarity_word_score) / 3
            clarity_scores.append(message_clarity)
        
        return np.mean(clarity_scores)
    
    def _calculate_goal_alignment_score(self, conversation_history: List[Dict], goals: List[str] = None) -> float:
        """Calculate how well conversation aligns with stated goals"""
        if not goals:
            return 0.5  # Neutral score when no goals specified
        
        user_messages = [msg['text'].lower() for msg in conversation_history if msg['sender'] == 'user']
        
        if not user_messages:
            return 0.0
        
        all_text = ' '.join(user_messages)
        
        # Goal progression indicators
        goal_progression = {
            'information_gathering': ['what', 'how', 'when', 'where', 'why', 'tell me', 'explain'],
            'problem_solving': ['problem', 'issue', 'fix', 'solve', 'help', 'trouble', 'error'],
            'decision_making': ['decide', 'choose', 'option', 'compare', 'versus', 'better', 'prefer'],
            'transaction': ['buy', 'purchase', 'order', 'price', 'cost', 'pay', 'subscribe'],
            'support': ['support', 'assistance', 'help', 'guide', 'instruction', 'tutorial']
        }
        
        alignment_scores = []
        
        for goal in goals:
            goal_lower = goal.lower()
            
            # Direct goal mention
            direct_mention = 1.0 if goal_lower in all_text else 0.0
            
            # Related indicator presence
            indicator_score = 0.0
            for category, indicators in goal_progression.items():
                if any(category_word in goal_lower for category_word in category.split('_')):
                    matches = sum(1 for indicator in indicators if indicator in all_text)
                    indicator_score = matches / len(indicators)
                    break
            
            goal_score = (direct_mention + indicator_score) / 2
            alignment_scores.append(goal_score)
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    def _determine_quality_level(self, overall_quality: float) -> str:
        """Determine quality level category"""
        if overall_quality >= 0.85:
            return 'EXCELLENT'
        elif overall_quality >= 0.7:
            return 'GOOD'
        elif overall_quality >= 0.5:
            return 'AVERAGE'
        elif overall_quality >= 0.3:
            return 'POOR'
        else:
            return 'VERY_POOR'
    
    def _generate_quality_insights(self, quality_scores: Dict) -> List[str]:
        """Generate insights about conversation quality"""
        insights = []
        
        # Identify strengths and weaknesses
        sorted_scores = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        best_dimension = sorted_scores[0]
        worst_dimension = sorted_scores[-1]
        
        insights.append(f"Strongest quality aspect: {best_dimension[0]} ({best_dimension[1]:.2f})")
        insights.append(f"Needs improvement: {worst_dimension[0]} ({worst_dimension[1]:.2f})")
        
        # Specific quality insights
        if quality_scores['coherence'] < 0.4:
            insights.append("Conversation lacks coherent flow - responses seem disconnected")
        
        if quality_scores['depth'] > 0.7:
            insights.append("High-quality, in-depth conversation with substantial content")
        
        if quality_scores['clarity'] < 0.4:
            insights.append("Communication clarity issues - messages are unclear or confusing")
        
        if quality_scores['goal_alignment'] > 0.7:
            insights.append("Excellent goal alignment - conversation staying focused on objectives")
        
        return insights
    
    def _generate_improvement_suggestions(self, quality_scores: Dict) -> List[Dict]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        for dimension, score in quality_scores.items():
            if score < 0.5:  # Areas needing improvement
                suggestion = self._get_dimension_improvement_suggestion(dimension, score)
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _get_dimension_improvement_suggestion(self, dimension: str, score: float) -> Dict:
        """Get specific improvement suggestion for a quality dimension"""
        suggestions = {
            'relevance': {
                'issue': 'Low conversation relevance',
                'suggestion': 'Steer conversation toward stated goals and relevant topics',
                'tactics': ['Ask focused questions', 'Redirect off-topic discussions', 'Reinforce key objectives']
            },
            'coherence': {
                'issue': 'Poor conversation flow',
                'suggestion': 'Improve logical connection between messages',
                'tactics': ['Reference previous messages', 'Use transitional phrases', 'Maintain topic consistency']
            },
            'depth': {
                'issue': 'Shallow conversation',
                'suggestion': 'Encourage more detailed and substantive responses',
                'tactics': ['Ask follow-up questions', 'Request specifics', 'Dive deeper into topics']
            },
            'clarity': {
                'issue': 'Unclear communication',
                'suggestion': 'Improve message clarity and comprehension',
                'tactics': ['Use simpler language', 'Ask for clarification', 'Summarize key points']
            },
            'goal_alignment': {
                'issue': 'Poor goal alignment',
                'suggestion': 'Better align conversation with stated objectives',
                'tactics': ['Regularly check goals', 'Redirect to objectives', 'Measure progress']
            }
        }
        
        return suggestions.get(dimension)
    
    def _analyze_quality_trend(self, conversation_history: List[Dict]) -> Dict:
        """Analyze how conversation quality changes over time"""
        if len(conversation_history) < 6:
            return {'trend': 'INSUFFICIENT_DATA'}
        
        # Split conversation into segments for trend analysis
        segment_size = max(3, len(conversation_history) // 3)
        segments = []
        
        for i in range(0, len(conversation_history), segment_size):
            segment = conversation_history[i:i + segment_size]
            if len(segment) >= 3:  # Minimum for quality assessment
                segments.append(segment)
        
        if len(segments) < 2:
            return {'trend': 'INSUFFICIENT_DATA'}
        
        # Calculate quality for each segment
        segment_qualities = []
        for segment in segments:
            quality = self.assess_conversation_quality(segment)
            segment_qualities.append(quality['overall_quality_score'])
        
        # Determine trend
        if len(segment_qualities) >= 3:
            early_avg = np.mean(segment_qualities[:len(segment_qualities)//2])
            late_avg = np.mean(segment_qualities[len(segment_qualities)//2:])
            
            if late_avg > early_avg + 0.1:
                trend = 'IMPROVING'
            elif early_avg > late_avg + 0.1:
                trend = 'DECLINING'
            else:
                trend = 'STABLE'
        else:
            trend = 'STABLE'
        
        return {
            'trend': trend,
            'segment_scores': segment_qualities,
            'quality_variance': np.var(segment_qualities),
            'overall_direction': late_avg - early_avg if len(segment_qualities) >= 2 else 0
        }
    
    def _create_zero_quality_score(self) -> Dict:
        """Create zero quality score for empty conversations"""
        return {
            'overall_quality_score': 0.0,
            'quality_level': 'NONE',
            'dimension_scores': {dim: 0.0 for dim in self.quality_dimensions.keys()},
            'quality_insights': ['No conversation data available'],
            'improvement_suggestions': [],
            'quality_trend': {'trend': 'NO_DATA'}
        }
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis model (placeholder)"""
        # In production, you'd initialize a real sentiment analysis model
        return None
    
    def _initialize_coherence_model(self):
        """Initialize coherence analysis model (placeholder)"""
        # In production, you'd initialize a real coherence model
        return None
```

## 4. Intent Classification Systems

### 4.1 Multi-Class Intent Recognition
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

class IntentClassificationSystem:
    def __init__(self):
        # Intent categories with confidence thresholds
        self.intent_categories = {
            'information_seeking': {
                'threshold': 0.7,
                'priority': 'MEDIUM',
                'expected_actions': ['provide_information', 'send_resources']
            },
            'product_inquiry': {
                'threshold': 0.8,
                'priority': 'HIGH',
                'expected_actions': ['product_demo', 'feature_explanation', 'pricing']
            },
            'support_request': {
                'threshold': 0.75,
                'priority': 'HIGH',
                'expected_actions': ['technical_support', 'troubleshooting', 'escalation']
            },
            'purchase_intent': {
                'threshold': 0.85,
                'priority': 'VERY_HIGH',
                'expected_actions': ['sales_process', 'pricing', 'contract_discussion']
            },
            'complaint': {
                'threshold': 0.7,
                'priority': 'HIGH',
                'expected_actions': ['acknowledge_issue', 'escalate', 'resolve']
            },
            'casual_conversation': {
                'threshold': 0.6,
                'priority': 'LOW',
                'expected_actions': ['engage_naturally', 'guide_to_business']
            },
            'qualification_needed': {
                'threshold': 0.7,
                'priority': 'MEDIUM',
                'expected_actions': ['qualify_lead', 'gather_requirements']
            },
            'objection': {
                'threshold': 0.75,
                'priority': 'HIGH',
                'expected_actions': ['address_objection', 'provide_evidence']
            },
            'scheduling': {
                'threshold': 0.8,
                'priority': 'HIGH',
                'expected_actions': ['schedule_meeting', 'send_calendar']
            },
            'time_wasting': {
                'threshold': 0.7,
                'priority': 'LOW',
                'expected_actions': ['polite_disengage', 'qualify_further']
            }
        }
        
        # Initialize models
        self.models = {}
        self.vectorizers = {}
        self.is_trained = False
        
        # Training data templates (in production, use real training data)
        self.training_data = self._generate_training_data()
        
    def _generate_training_data(self) -> Dict[str, List[str]]:
        """Generate training data for intent classification"""
        return {
            'information_seeking': [
                "tell me about your product",
                "what services do you offer",
                "how does this work",
                "can you explain more",
                "i want to learn about",
                "what are the details",
                "give me information on"
            ],
            'product_inquiry': [
                "what features does it have",
                "show me the product",
                "what can your product do",
                "product specifications",
                "how does your product compare",
                "what makes your product different",
                "product demonstration"
            ],
            'support_request': [
                "i need help with",
                "having trouble with",
                "something is not working",
                "error message",
                "technical issue",
                "support needed",
                "problem with setup"
            ],
            'purchase_intent': [
                "how much does it cost",
                "what are your prices",
                "i want to buy",
                "ready to purchase",
                "pricing information",
                "where do i sign up",
                "let's move forward"
            ],
            'complaint': [
                "unhappy with service",
                "this doesn't work",
                "disappointed with",
                "poor experience",
                "not satisfied",
                "want to cancel",
                "issue with billing"
            ],
            'casual_conversation': [
                "how are you",
                "nice to meet you",
                "how's your day",
                "thanks for reaching out",
                "appreciate your time",
                "good morning",
                "have a great day"
            ],
            'qualification_needed': [
                "just looking around",
                "exploring options",
                "not sure what i need",
                "what do you recommend",
                "depends on price",
                "need to think about it",
                "comparing solutions"
            ],
            'objection': [
                "too expensive",
                "not in budget",
                "need to discuss with team",
                "not the right time",
                "concerned about",
                "what if it doesn't work",
                "heard bad things about"
            ],
            'scheduling': [
                "can we schedule a call",
                "when are you available",
                "let's set up a meeting",
                "book a demo",
                "calendar availability",
                "meeting request",
                "schedule appointment"
            ],
            'time_wasting': [
                "just browsing",
                "not really interested",
                "maybe sometime later",
                "just curious",
                "no budget right now",
                "still thinking about it",
                "not ready to decide"
            ]
        }
    
    def train_models(self):
        """Train intent classification models"""
        # Prepare training data
        texts = []
        labels = []
        
        for intent, examples in self.training_data.items():
            texts.extend(examples)
            labels.extend([intent] * len(examples))
        
        # Create feature vectors
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        
        X = vectorizer.fit_transform(texts)
        
        # Train multiple models
        models_to_train = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'svm': SVC(probability=True, kernel='linear', C=1.0),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        for model_name, model in models_to_train.items():
            model.fit(X, labels)
            self.models[model_name] = model
        
        self.vectorizers['tfidf'] = vectorizer
        self.is_trained = True
    
    def classify_intent(self, message: str, conversation_context: List[Dict] = None) -> Dict:
        """Classify intent of a message with confidence scores"""
        if not self.is_trained:
            self.train_models()
        
        # Preprocess message
        processed_message = self._preprocess_message(message)
        
        # Extract features
        message_features = self.vectorizers['tfidf'].transform([processed_message])
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(message_features)[0]
            pred_proba = model.predict_proba(message_features)[0]
            
            predictions[model_name] = pred
            
            # Get probability for each class
            classes = model.classes_
            probabilities[model_name] = dict(zip(classes, pred_proba))
        
        # Ensemble prediction (majority vote with confidence weighting)
        intent_scores = {}
        for intent in self.intent_categories.keys():
            scores = []
            for model_name in self.models.keys():
                if intent in probabilities[model_name]:
                    scores.append(probabilities[model_name][intent])
                else:
                    scores.append(0.0)
            intent_scores[intent] = np.mean(scores)
        
        # Find best intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        predicted_intent = best_intent[0]
        confidence = best_intent[1]
        
        # Apply context adjustments
        if conversation_context:
            predicted_intent, confidence = self._apply_context_adjustment(
                predicted_intent, confidence, conversation_context
            )
        
        # Check confidence threshold
        threshold = self.intent_categories[predicted_intent]['threshold']
        is_confident = confidence >= threshold
        
        return {
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'is_confident_prediction': is_confident,
            'alternative_intents': sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[1:4],
            'all_scores': intent_scores,
            'individual_predictions': predictions,
            'priority': self.intent_categories[predicted_intent]['priority'],
            'expected_actions': self.intent_categories[predicted_intent]['expected_actions'],
            'context_adjusted': conversation_context is not None
        }
    
    def _preprocess_message(self, message: str) -> str:
        """Preprocess message for intent classification"""
        # Convert to lowercase
        message = message.lower()
        
        # Remove special characters but keep spaces
        message = re.sub(r'[^\w\s]', ' ', message)
        
        # Remove extra whitespace
        message = ' '.join(message.split())
        
        return message
    
    def _apply_context_adjustment(self, predicted_intent: str, confidence: float, 
                                 conversation_context: List[Dict]) -> Tuple[str, float]:
        """Apply conversation context to adjust intent prediction"""
        if len(conversation_context) < 2:
            return predicted_intent, confidence
        
        # Analyze recent conversation context (last 3 messages)
        recent_messages = conversation_context[-3:]
        context_intents = []
        
        for msg in recent_messages:
            if msg.get('sender') == 'user':
                context_classification = self.classify_intent(msg['text'])
                context_intents.append(context_classification['predicted_intent'])
        
        # Context-based adjustments
        context_adjustments = {
            # If user was asking about pricing, purchase intent more likely
            'product_inquiry': {
                'purchase_intent': 0.15,
                'information_seeking': 0.1
            },
            # If user had support issues, complaints more likely
            'support_request': {
                'complaint': 0.2,
                'support_request': 0.1
            },
            # If conversation was casual, less likely to be high-intent
            'casual_conversation': {
                'purchase_intent': -0.1,
                'product_inquiry': -0.05
            }
        }
        
        # Apply adjustments based on context
        adjusted_confidence = confidence
        for context_intent in context_intents:
            if context_intent in context_adjustments:
                if predicted_intent in context_adjustments[context_intent]:
                    adjustment = context_adjustments[context_intent][predicted_intent]
                    adjusted_confidence += adjustment
        
        # Clamp confidence between 0 and 1
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return predicted_intent, adjusted_confidence
    
    def analyze_intent_progression(self, conversation_history: List[Dict]) -> Dict:
        """Analyze how user intent evolves throughout conversation"""
        if len(conversation_history) < 3:
            return {'status': 'INSUFFICIENT_DATA'}
        
        user_messages = [msg for msg in conversation_history if msg.get('sender') == 'user']
        
        if len(user_messages) < 2:
            return {'status': 'INSUFFICIENT_USER_MESSAGES'}
        
        # Classify intent for each user message
        intent_progression = []
        for i, msg in enumerate(user_messages):
            classification = self.classify_intent(msg['text'], conversation_history[:i*2+1])
            intent_progression.append({
                'message_index': i,
                'intent': classification['predicted_intent'],
                'confidence': classification['confidence'],
                'priority': classification['priority']
            })
        
        # Analyze progression patterns
        intent_sequence = [item['intent'] for item in intent_progression]
        priority_sequence = [item['priority'] for item in intent_progression]
        
        # Calculate intent consistency
        intent_changes = sum(1 for i in range(1, len(intent_sequence)) 
                           if intent_sequence[i] != intent_sequence[i-1])
        intent_consistency = 1.0 - (intent_changes / max(1, len(intent_sequence) - 1))
        
        # Determine progression direction
        priority_values = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'VERY_HIGH': 4}
        priority_numbers = [priority_values.get(p, 1) for p in priority_sequence]
        
        if len(priority_numbers) >= 2:
            early_avg = np.mean(priority_numbers[:len(priority_numbers)//2])
            late_avg = np.mean(priority_numbers[len(priority_numbers)//2:])
            
            if late_avg > early_avg + 0.5:
                progression_direction = 'ESCALATING'
            elif early_avg > late_avg + 0.5:
                progression_direction = 'DE_ESCALATING'
            else:
                progression_direction = 'STABLE'
        else:
            progression_direction = 'STABLE'
        
        # Final intent assessment
        latest_intent = intent_progression[-1]
        
        return {
            'status': 'ANALYZED',
            'intent_progression': intent_progression,
            'final_intent': latest_intent,
            'intent_consistency': intent_consistency,
            'progression_direction': progression_direction,
            'recommendation': self._generate_intent_recommendation(
                latest_intent, progression_direction, intent_consistency
            )
        }
    
    def _generate_intent_recommendation(self, latest_intent: Dict, 
                                      progression_direction: str, 
                                      consistency: float) -> Dict:
        """Generate recommendations based on intent analysis"""
        intent_type = latest_intent['intent']
        confidence = latest_intent['confidence']
        priority = latest_intent['priority']
        
        if priority == 'VERY_HIGH':
            if progression_direction == 'ESCALATING':
                return {
                    'action': 'IMMEDIATE_ESCALATION',
                    'message': 'High-priority intent with escalating pattern - immediate action required',
                    'tactics': ['Direct response to intent', 'Escalate to specialist', 'Expedite process']
                }
            else:
                return {
                    'action': 'HIGH_PRIORITY_RESPONSE',
                    'message': 'Very high priority intent detected',
                    'tactics': ['Prioritize response', 'Address directly', 'Follow up quickly']
                }
        
        elif consistency < 0.3:
            return {
                'action': 'CLARIFY_INTENT',
                'message': 'Low intent consistency - user seems uncertain',
                'tactics': ['Ask clarifying questions', 'Provide options', 'Guide conversation']
            }
        
        elif progression_direction == 'DE_ESCALATING':
            return {
                'action': 'RE_ENGAGE',
                'message': 'Intent priority decreasing - risk of losing engagement',
                'tactics': ['Provide compelling value', 'Ask engaging questions', 'Create urgency']
            }
        
        else:
            return {
                'action': 'CONTINUE_CURRENT_APPROACH',
                'message': 'Intent analysis looks positive - maintain current approach',
                'tactics': ['Continue current conversation flow', 'Monitor for changes']
            }
```

### 4.2 Real-Time Intent Monitoring
```python
class RealTimeIntentMonitor:
    def __init__(self):
        self.classifier = IntentClassificationSystem()
        self.intent_history = {}
        self.alert_thresholds = {
            'intent_change_frequency': 0.5,  # Alert if intent changes >50% of time
            'confidence_drop': 0.3,          # Alert if confidence drops by 30%
            'high_priority_intent': 0.8      # Alert threshold for high-priority intents
        }
        
    async def monitor_intent_realtime(self, chat_id: int, message: Dict, 
                                     conversation_history: List[Dict]) -> Dict:
        """Monitor intent changes in real-time"""
        
        # Classify current message intent
        current_classification = self.classifier.classify_intent(
            message['text'], conversation_history
        )
        
        # Update intent history
        if chat_id not in self.intent_history:
            self.intent_history[chat_id] = []
        
        self.intent_history[chat_id].append({
            'timestamp': message.get('timestamp', time.time()),
            'intent': current_classification['predicted_intent'],
            'confidence': current_classification['confidence'],
            'priority': current_classification['priority']
        })
        
        # Analyze intent patterns
        pattern_analysis = self._analyze_intent_patterns(chat_id)
        
        # Generate alerts
        alerts = self._check_intent_alerts(chat_id, current_classification)
        
        # Generate recommendations
        recommendations = self._generate_realtime_recommendations(
            current_classification, pattern_analysis, alerts
        )
        
        return {
            'current_intent': current_classification,
            'pattern_analysis': pattern_analysis,
            'alerts': alerts,
            'recommendations': recommendations,
            'intent_history': self.intent_history[chat_id][-5:]  # Last 5 intents
        }
    
    def _analyze_intent_patterns(self, chat_id: int) -> Dict:
        """Analyze patterns in intent progression"""
        if chat_id not in self.intent_history or len(self.intent_history[chat_id]) < 3:
            return {'status': 'INSUFFICIENT_DATA'}
        
        history = self.intent_history[chat_id]
        
        # Intent volatility (how often intent changes)
        intent_changes = 0
        for i in range(1, len(history)):
            if history[i]['intent'] != history[i-1]['intent']:
                intent_changes += 1
        
        volatility = intent_changes / max(1, len(history) - 1)
        
        # Confidence trend
        confidences = [entry['confidence'] for entry in history]
        if len(confidences) >= 3:
            recent_confidence = np.mean(confidences[-2:])
            older_confidence = np.mean(confidences[:-2])
            confidence_trend = recent_confidence - older_confidence
        else:
            confidence_trend = 0
        
        # Priority progression
        priority_values = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'VERY_HIGH': 4}
        priorities = [priority_values.get(entry['priority'], 1) for entry in history]
        
        if len(priorities) >= 3:
            priority_trend = np.polyfit(range(len(priorities)), priorities, 1)[0]
        else:
            priority_trend = 0
        
        # Most common intent
        recent_intents = [entry['intent'] for entry in history[-3:]]
        from collections import Counter
        intent_counts = Counter(recent_intents)
        dominant_intent = intent_counts.most_common(1)[0][0] if intent_counts else None
        
        return {
            'status': 'ANALYZED',
            'volatility': volatility,
            'confidence_trend': confidence_trend,
            'priority_trend': priority_trend,
            'dominant_intent': dominant_intent,
            'pattern_stability': 1.0 - volatility,  # Inverse of volatility
            'overall_direction': self._determine_overall_direction(priority_trend, confidence_trend)
        }
    
    def _determine_overall_direction(self, priority_trend: float, confidence_trend: float) -> str:
        """Determine overall direction of intent progression"""
        if priority_trend > 0.3 and confidence_trend > 0.1:
            return 'POSITIVE_ESCALATION'
        elif priority_trend < -0.3 and confidence_trend < -0.1:
            return 'NEGATIVE_ESCALATION'
        elif abs(priority_trend) < 0.2 and abs(confidence_trend) < 0.1:
            return 'STABLE'
        elif priority_trend > 0.2:
            return 'PRIORITY_INCREASING'
        elif priority_trend < -0.2:
            return 'PRIORITY_DECREASING'
        else:
            return 'MIXED_SIGNALS'
    
    def _check_intent_alerts(self, chat_id: int, current_classification: Dict) -> List[Dict]:
        """Check for intent-based alerts"""
        alerts = []
        
        if chat_id not in self.intent_history:
            return alerts
        
        history = self.intent_history[chat_id]
        current_confidence = current_classification['confidence']
        current_priority = current_classification['priority']
        
        # High-priority intent alert
        if current_priority == 'VERY_HIGH' and current_confidence > self.alert_thresholds['high_priority_intent']:
            alerts.append({
                'type': 'HIGH_PRIORITY_INTENT',
                'severity': 'HIGH',
                'message': f"Very high priority intent detected: {current_classification['predicted_intent']}",
                'confidence': current_confidence,
                'action_required': True
            })
        
        # Confidence drop alert
        if len(history) >= 3:
            recent_confidences = [entry['confidence'] for entry in history[-3:]]
            avg_recent = np.mean(recent_confidences)
            
            if len(history) >= 6:
                older_confidences = [entry['confidence'] for entry in history[-6:-3]]
                avg_older = np.mean(older_confidences)
                
                confidence_drop = avg_older - avg_recent
                if confidence_drop > self.alert_thresholds['confidence_drop']:
                    alerts.append({
                        'type': 'CONFIDENCE_DROP',
                        'severity': 'MEDIUM',
                        'message': f"Intent confidence dropped by {confidence_drop:.1%}",
                        'drop_amount': confidence_drop,
                        'action_required': True
                    })
        
        # Intent volatility alert
        if len(history) >= 5:
            recent_intents = [entry['intent'] for entry in history[-5:]]
            unique_intents = len(set(recent_intents))
            volatility = unique_intents / len(recent_intents)
            
            if volatility > self.alert_thresholds['intent_change_frequency']:
                alerts.append({
                    'type': 'HIGH_INTENT_VOLATILITY',
                    'severity': 'MEDIUM',
                    'message': f"High intent volatility detected ({volatility:.1%})",
                    'volatility': volatility,
                    'action_required': True
                })
        
        return alerts
    
    def _generate_realtime_recommendations(self, current_classification: Dict, 
                                         pattern_analysis: Dict, alerts: List[Dict]) -> List[Dict]:
        """Generate real-time recommendations based on intent monitoring"""
        recommendations = []
        
        current_intent = current_classification['predicted_intent']
        current_confidence = current_classification['confidence']
        current_priority = current_classification['priority']
        
        # High-priority alerts require immediate action
        high_priority_alerts = [alert for alert in alerts if alert['severity'] == 'HIGH']
        if high_priority_alerts:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'action': 'ADDRESS_HIGH_PRIORITY_INTENT',
                'message': 'Immediate attention required for high-priority intent',
                'specific_actions': current_classification['expected_actions']
            })
        
        # Pattern-based recommendations
        if pattern_analysis.get('status') == 'ANALYZED':
            overall_direction = pattern_analysis.get('overall_direction')
            
            if overall_direction == 'POSITIVE_ESCALATION':
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'CAPITALIZE_MOMENTUM',
                    'message': 'Positive intent escalation - capitalize on momentum',
                    'specific_actions': ['Advance conversation', 'Present solutions', 'Ask for commitment']
                })
            
            elif overall_direction == 'NEGATIVE_ESCALATION':
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'DAMAGE_CONTROL',
                    'message': 'Negative intent escalation - implement damage control',
                    'specific_actions': ['Address concerns', 'Provide reassurance', 'Offer alternatives']
                })
            
            elif pattern_analysis.get('volatility', 0) > 0.5:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'STABILIZE_CONVERSATION',
                    'message': 'High intent volatility - focus on stabilizing conversation',
                    'specific_actions': ['Clarify objectives', 'Provide structure', 'Guide conversation']
                })
        
        # Intent-specific recommendations
        intent_specific_recs = self._get_intent_specific_recommendations(current_intent, current_confidence)
        if intent_specific_recs:
            recommendations.extend(intent_specific_recs)
        
        return recommendations
    
    def _get_intent_specific_recommendations(self, intent: str, confidence: float) -> List[Dict]:
        """Get recommendations specific to detected intent"""
        intent_recommendations = {
            'purchase_intent': [
                {
                    'priority': 'HIGH',
                    'action': 'ACCELERATE_SALES_PROCESS',
                    'message': 'Purchase intent detected - accelerate sales process',
                    'specific_actions': ['Present pricing', 'Discuss terms', 'Schedule contract review']
                }
            ],
            'objection': [
                {
                    'priority': 'HIGH',
                    'action': 'ADDRESS_OBJECTION',
                    'message': 'Objection detected - address concerns directly',
                    'specific_actions': ['Listen actively', 'Acknowledge concern', 'Provide evidence/alternatives']
                }
            ],
            'complaint': [
                {
                    'priority': 'HIGH',
                    'action': 'SERVICE_RECOVERY',
                    'message': 'Complaint detected - implement service recovery',
                    'specific_actions': ['Apologize if appropriate', 'Escalate to specialist', 'Offer resolution']
                }
            ],
            'support_request': [
                {
                    'priority': 'MEDIUM',
                    'action': 'PROVIDE_SUPPORT',
                    'message': 'Support request identified - provide assistance',
                    'specific_actions': ['Gather details', 'Provide solution', 'Follow up']
                }
            ],
            'time_wasting': [
                {
                    'priority': 'LOW',
                    'action': 'QUALIFY_OR_DISENGAGE',
                    'message': 'Time-wasting behavior detected - qualify or disengage',
                    'specific_actions': ['Ask qualifying questions', 'Set boundaries', 'Polite disengagement']
                }
            ]
        }
        
        recommendations = intent_recommendations.get(intent, [])
        
        # Adjust based on confidence level
        for rec in recommendations:
            if confidence < 0.7:
                rec['message'] += f" (Low confidence: {confidence:.1%})"
                rec['specific_actions'].insert(0, 'Verify intent through clarifying questions')
        
        return recommendations
```

This comprehensive risk assessment system provides:

1. **Time-Waster Detection**: Multi-signal ML algorithms analyzing message patterns, commitment indicators, and temporal behaviors
2. **Engagement Scoring**: Multi-dimensional scoring across attention, interest, interaction, progression, and emotional engagement
3. **Conversation Quality**: Assessment of relevance, coherence, depth, clarity, and goal alignment
4. **Intent Classification**: Real-time classification of user intents with confidence scoring and context awareness

The system integrates seamlessly with the existing Telegram bot architecture and provides actionable insights for optimizing conversation outcomes and resource allocation.