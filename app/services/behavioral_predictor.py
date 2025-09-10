"""
Behavioral Predictor Service

ML-based prediction service for user behavior analysis, churn risk assessment,
mood detection, and personalized recommendation generation.
"""

import asyncio
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

from sqlalchemy import select, and_, func, desc
import structlog

from app.database.connection import get_database_session
from app.models.user import User
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, EngagementType, SentimentType,
    OutreachType, ProactiveOutreach, OutreachStatus
)

logger = structlog.get_logger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    prediction: Union[float, int, str]
    confidence: float
    features_used: List[str]
    model_version: str
    timestamp: datetime


@dataclass
class UserFeatures:
    """Container for user feature data."""
    user_id: str
    telegram_id: int
    
    # Activity features
    total_interactions: int = 0
    daily_avg_interactions: float = 0.0
    most_active_hour: int = 12
    session_length_avg: float = 0.0
    
    # Engagement features  
    avg_sentiment: float = 0.0
    engagement_trend: float = 0.0
    avg_response_time: float = 60.0
    quality_score_avg: float = 0.5
    
    # Behavioral features
    days_since_last: int = 0
    longest_absence: int = 0
    interaction_variety: float = 0.0
    question_ratio: float = 0.0
    
    # Preference features
    command_usage_rate: float = 0.0
    voice_usage_rate: float = 0.0
    emoji_usage_rate: float = 0.0
    
    # Temporal features
    weekday_activity: float = 0.5
    weekend_activity: float = 0.5
    evening_activity: float = 0.5
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.total_interactions,
            self.daily_avg_interactions,
            self.most_active_hour,
            self.session_length_avg,
            self.avg_sentiment,
            self.engagement_trend,
            self.avg_response_time,
            self.quality_score_avg,
            self.days_since_last,
            self.longest_absence,
            self.interaction_variety,
            self.question_ratio,
            self.command_usage_rate,
            self.voice_usage_rate,
            self.emoji_usage_rate,
            self.weekday_activity,
            self.weekend_activity,
            self.evening_activity
        ])


class BehavioralPredictor:
    """
    ML-based behavioral prediction service.
    
    Implements predictive models for:
    - Churn risk assessment
    - Mood detection and trends
    - Optimal engagement timing
    - Next best action recommendations
    """
    
    def __init__(self, model_dir: str = "models/behavioral"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model instances
        self.churn_model: Optional[RandomForestClassifier] = None
        self.mood_model: Optional[GradientBoostingRegressor] = None
        self.timing_model: Optional[LogisticRegression] = None
        self.action_model: Optional[RandomForestClassifier] = None
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        
        # Model metadata
        self.model_version = "1.0.0"
        self.last_training = None
        self.feature_names = [
            'total_interactions', 'daily_avg_interactions', 'most_active_hour',
            'session_length_avg', 'avg_sentiment', 'engagement_trend',
            'avg_response_time', 'quality_score_avg', 'days_since_last',
            'longest_absence', 'interaction_variety', 'question_ratio',
            'command_usage_rate', 'voice_usage_rate', 'emoji_usage_rate',
            'weekday_activity', 'weekend_activity', 'evening_activity'
        ]
        
        # Load existing models
        asyncio.create_task(self._load_models())
    
    async def predict_churn_risk(self, user_id: str) -> PredictionResult:
        """
        Predict churn risk for a user.
        
        Args:
            user_id: User UUID
            
        Returns:
            PredictionResult with churn probability (0-1)
        """
        try:
            features = await self._extract_user_features(user_id)
            if not features:
                return PredictionResult(
                    prediction=0.5,
                    confidence=0.0,
                    features_used=[],
                    model_version=self.model_version,
                    timestamp=datetime.utcnow()
                )
            
            # Use model if available, otherwise fallback to heuristic
            if self.churn_model:
                X = self.feature_scaler.transform([features.to_array()])
                churn_prob = self.churn_model.predict_proba(X)[0][1]  # Probability of churn
                confidence = max(self.churn_model.predict_proba(X)[0])
            else:
                churn_prob, confidence = self._heuristic_churn_risk(features)
            
            logger.info(
                "Churn risk predicted",
                user_id=user_id,
                churn_probability=churn_prob,
                confidence=confidence,
                model_used=self.churn_model is not None
            )
            
            return PredictionResult(
                prediction=float(churn_prob),
                confidence=float(confidence),
                features_used=self.feature_names,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error predicting churn risk", error=str(e), user_id=user_id)
            raise
    
    async def predict_mood_trend(self, user_id: str, days_ahead: int = 7) -> PredictionResult:
        """
        Predict user mood trend for the next period.
        
        Args:
            user_id: User UUID
            days_ahead: Days to predict ahead
            
        Returns:
            PredictionResult with mood trend prediction (-1 to 1)
        """
        try:
            features = await self._extract_user_features(user_id)
            if not features:
                return PredictionResult(
                    prediction=0.0,
                    confidence=0.0,
                    features_used=[],
                    model_version=self.model_version,
                    timestamp=datetime.utcnow()
                )
            
            # Use model if available, otherwise fallback to trend analysis
            if self.mood_model:
                X = self.feature_scaler.transform([features.to_array()])
                mood_trend = self.mood_model.predict(X)[0]
                confidence = 0.7  # Model-based confidence
            else:
                mood_trend, confidence = self._heuristic_mood_trend(features)
            
            logger.info(
                "Mood trend predicted",
                user_id=user_id,
                mood_trend=mood_trend,
                confidence=confidence,
                days_ahead=days_ahead
            )
            
            return PredictionResult(
                prediction=float(mood_trend),
                confidence=float(confidence),
                features_used=self.feature_names,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error predicting mood trend", error=str(e), user_id=user_id)
            raise
    
    async def predict_optimal_timing(self, user_id: str) -> PredictionResult:
        """
        Predict optimal timing for user outreach.
        
        Args:
            user_id: User UUID
            
        Returns:
            PredictionResult with optimal hour (0-23)
        """
        try:
            features = await self._extract_user_features(user_id)
            if not features:
                return PredictionResult(
                    prediction=14,  # Default to 2 PM
                    confidence=0.0,
                    features_used=[],
                    model_version=self.model_version,
                    timestamp=datetime.utcnow()
                )
            
            # Use existing most active hour as baseline
            optimal_hour = features.most_active_hour
            confidence = 0.6
            
            # Enhance with model if available
            if self.timing_model:
                X = self.feature_scaler.transform([features.to_array()])
                # Predict best time category (morning, afternoon, evening)
                time_category = self.timing_model.predict(X)[0]
                confidence = max(self.timing_model.predict_proba(X)[0])
                
                # Map to specific hours
                time_mapping = {
                    0: 10,  # Morning
                    1: 14,  # Afternoon
                    2: 19   # Evening
                }
                optimal_hour = time_mapping.get(time_category, optimal_hour)
            
            logger.info(
                "Optimal timing predicted",
                user_id=user_id,
                optimal_hour=optimal_hour,
                confidence=confidence
            )
            
            return PredictionResult(
                prediction=int(optimal_hour),
                confidence=float(confidence),
                features_used=self.feature_names,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error predicting optimal timing", error=str(e), user_id=user_id)
            raise
    
    async def recommend_next_action(self, user_id: str) -> PredictionResult:
        """
        Recommend next best action for user engagement.
        
        Args:
            user_id: User UUID
            
        Returns:
            PredictionResult with recommended action type
        """
        try:
            features = await self._extract_user_features(user_id)
            if not features:
                return PredictionResult(
                    prediction="personalized_checkin",
                    confidence=0.0,
                    features_used=[],
                    model_version=self.model_version,
                    timestamp=datetime.utcnow()
                )
            
            # Use model if available, otherwise heuristic rules
            if self.action_model:
                X = self.feature_scaler.transform([features.to_array()])
                action_idx = self.action_model.predict(X)[0]
                confidence = max(self.action_model.predict_proba(X)[0])
                
                action_types = [
                    "personalized_checkin", "re_engagement", "milestone_celebration",
                    "feature_suggestion", "mood_support", "topic_follow_up"
                ]
                recommended_action = action_types[action_idx]
            else:
                recommended_action, confidence = self._heuristic_action_recommendation(features)
            
            logger.info(
                "Next action recommended",
                user_id=user_id,
                action=recommended_action,
                confidence=confidence
            )
            
            return PredictionResult(
                prediction=recommended_action,
                confidence=float(confidence),
                features_used=self.feature_names,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error recommending next action", error=str(e), user_id=user_id)
            raise
    
    async def detect_interest_topics(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Detect user's topics of interest based on interaction patterns.
        
        Args:
            user_id: User UUID
            limit: Maximum number of topics to return
            
        Returns:
            List of topics with relevance scores
        """
        try:
            async with get_database_session() as session:
                # Get recent interactions with topic data
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                interactions_result = await session.execute(
                    select(UserEngagement)
                    .where(
                        and_(
                            UserEngagement.user_id == user_id,
                            UserEngagement.interaction_timestamp >= cutoff_date,
                            UserEngagement.detected_topics.isnot(None)
                        )
                    )
                    .order_by(desc(UserEngagement.interaction_timestamp))
                )
                interactions = interactions_result.scalars().all()
                
                if not interactions:
                    return []
                
                # Aggregate topic mentions with weights
                topic_scores = {}
                
                for interaction in interactions:
                    if not interaction.detected_topics:
                        continue
                    
                    # Weight by recency and engagement quality
                    days_ago = (datetime.utcnow() - interaction.interaction_timestamp).days
                    recency_weight = max(0.1, 1 - (days_ago / 30))
                    quality_weight = interaction.engagement_quality_score or 0.5
                    
                    for topic in interaction.detected_topics:
                        if topic not in topic_scores:
                            topic_scores[topic] = {
                                'mentions': 0,
                                'weighted_score': 0.0,
                                'latest_mention': interaction.interaction_timestamp
                            }
                        
                        topic_scores[topic]['mentions'] += 1
                        topic_scores[topic]['weighted_score'] += recency_weight * quality_weight
                        
                        if interaction.interaction_timestamp > topic_scores[topic]['latest_mention']:
                            topic_scores[topic]['latest_mention'] = interaction.interaction_timestamp
                
                # Rank topics by weighted score
                ranked_topics = []
                for topic, data in topic_scores.items():
                    relevance_score = data['weighted_score'] / len(interactions)  # Normalize by total interactions
                    
                    ranked_topics.append({
                        'topic': topic,
                        'relevance_score': round(relevance_score, 3),
                        'mention_count': data['mentions'],
                        'latest_mention': data['latest_mention'].isoformat(),
                        'days_since_mention': (datetime.utcnow() - data['latest_mention']).days
                    })
                
                # Sort by relevance and return top topics
                ranked_topics.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                logger.info(
                    "Interest topics detected",
                    user_id=user_id,
                    total_topics=len(ranked_topics),
                    top_topics=[t['topic'] for t in ranked_topics[:3]]
                )
                
                return ranked_topics[:limit]
                
        except Exception as e:
            logger.error("Error detecting interest topics", error=str(e), user_id=user_id)
            return []
    
    async def train_models(self, min_samples: int = 100) -> Dict[str, Any]:
        """
        Train or retrain ML models with current data.
        
        Args:
            min_samples: Minimum number of samples required for training
            
        Returns:
            Training results and model metrics
        """
        try:
            logger.info("Starting model training", min_samples=min_samples)
            
            # Extract training data
            training_data = await self._extract_training_data()
            
            if len(training_data['features']) < min_samples:
                logger.warning(
                    "Insufficient training data",
                    available_samples=len(training_data['features']),
                    required_samples=min_samples
                )
                return {"error": "Insufficient training data"}
            
            features = np.array(training_data['features'])
            
            # Fit feature scaler
            self.feature_scaler.fit(features)
            features_scaled = self.feature_scaler.transform(features)
            
            results = {}
            
            # Train churn prediction model
            if 'churn_labels' in training_data and len(training_data['churn_labels']) >= min_samples:
                results['churn_model'] = await self._train_churn_model(
                    features_scaled, 
                    training_data['churn_labels']
                )
            
            # Train mood prediction model
            if 'mood_scores' in training_data and len(training_data['mood_scores']) >= min_samples:
                results['mood_model'] = await self._train_mood_model(
                    features_scaled,
                    training_data['mood_scores']
                )
            
            # Train timing model
            if 'timing_labels' in training_data and len(training_data['timing_labels']) >= min_samples:
                results['timing_model'] = await self._train_timing_model(
                    features_scaled,
                    training_data['timing_labels']
                )
            
            # Train action recommendation model
            if 'action_labels' in training_data and len(training_data['action_labels']) >= min_samples:
                results['action_model'] = await self._train_action_model(
                    features_scaled,
                    training_data['action_labels']
                )
            
            # Save models
            await self._save_models()
            
            self.last_training = datetime.utcnow()
            
            logger.info(
                "Model training completed",
                models_trained=list(results.keys()),
                training_samples=len(features),
                timestamp=self.last_training.isoformat()
            )
            
            return {
                'training_completed': self.last_training.isoformat(),
                'models_trained': list(results.keys()),
                'training_samples': len(features),
                'model_version': self.model_version,
                'results': results
            }
            
        except Exception as e:
            logger.error("Error training models", error=str(e))
            raise
    
    # Private helper methods
    
    async def _extract_user_features(self, user_id: str) -> Optional[UserFeatures]:
        """Extract features for a specific user."""
        try:
            async with get_database_session() as session:
                # Get user behavior pattern
                pattern_result = await session.execute(
                    select(UserBehaviorPattern).where(UserBehaviorPattern.user_id == user_id)
                )
                pattern = pattern_result.scalar_one_or_none()
                
                if not pattern:
                    return None
                
                # Get recent interactions for additional features
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                interactions_result = await session.execute(
                    select(UserEngagement)
                    .where(
                        and_(
                            UserEngagement.user_id == user_id,
                            UserEngagement.interaction_timestamp >= cutoff_date
                        )
                    )
                )
                interactions = interactions_result.scalars().all()
                
                # Calculate additional features
                interaction_variety = len(set(i.engagement_type for i in interactions)) / len(EngagementType) if interactions else 0
                question_ratio = len([i for i in interactions if i.contains_question]) / max(1, len(interactions))
                command_ratio = len([i for i in interactions if i.engagement_type == EngagementType.COMMAND]) / max(1, len(interactions))
                voice_ratio = len([i for i in interactions if i.engagement_type == EngagementType.VOICE_MESSAGE]) / max(1, len(interactions))
                emoji_ratio = len([i for i in interactions if i.contains_emoji]) / max(1, len(interactions))
                
                # Calculate temporal activity patterns
                weekday_interactions = [i for i in interactions if i.interaction_timestamp.weekday() < 5]
                weekend_interactions = [i for i in interactions if i.interaction_timestamp.weekday() >= 5]
                evening_interactions = [i for i in interactions if i.interaction_timestamp.hour >= 18]
                
                weekday_activity = len(weekday_interactions) / max(1, len(interactions))
                weekend_activity = len(weekend_interactions) / max(1, len(interactions))
                evening_activity = len(evening_interactions) / max(1, len(interactions))
                
                return UserFeatures(
                    user_id=user_id,
                    telegram_id=pattern.telegram_id,
                    total_interactions=pattern.total_interactions or 0,
                    daily_avg_interactions=pattern.daily_interaction_average or 0,
                    most_active_hour=pattern.most_active_hour or 12,
                    session_length_avg=pattern.average_session_length_minutes or 0,
                    avg_sentiment=pattern.average_sentiment_score or 0,
                    engagement_trend=pattern.engagement_quality_trend or 0,
                    avg_response_time=pattern.response_time_average_seconds or 60,
                    quality_score_avg=0.5,  # Would need to calculate from interactions
                    days_since_last=pattern.days_since_last_interaction or 0,
                    longest_absence=pattern.longest_absence_days or 0,
                    interaction_variety=interaction_variety,
                    question_ratio=question_ratio,
                    command_usage_rate=command_ratio,
                    voice_usage_rate=voice_ratio,
                    emoji_usage_rate=emoji_ratio,
                    weekday_activity=weekday_activity,
                    weekend_activity=weekend_activity,
                    evening_activity=evening_activity
                )
                
        except Exception as e:
            logger.error("Error extracting user features", error=str(e), user_id=user_id)
            return None
    
    def _heuristic_churn_risk(self, features: UserFeatures) -> Tuple[float, float]:
        """Calculate churn risk using heuristic rules."""
        risk = 0.0
        
        # Days since last interaction (40% weight)
        if features.days_since_last > 14:
            risk += 0.4
        elif features.days_since_last > 7:
            risk += 0.2
        elif features.days_since_last > 3:
            risk += 0.1
        
        # Engagement trend (30% weight)
        if features.engagement_trend < -0.3:
            risk += 0.3
        elif features.engagement_trend < -0.1:
            risk += 0.15
        
        # Sentiment (20% weight)
        if features.avg_sentiment < -0.3:
            risk += 0.2
        elif features.avg_sentiment < 0:
            risk += 0.1
        
        # Interaction frequency (10% weight)
        if features.daily_avg_interactions < 0.5:
            risk += 0.1
        elif features.daily_avg_interactions < 1:
            risk += 0.05
        
        confidence = 0.6  # Heuristic confidence
        return min(1.0, risk), confidence
    
    def _heuristic_mood_trend(self, features: UserFeatures) -> Tuple[float, float]:
        """Predict mood trend using heuristic rules."""
        # Base on recent sentiment and engagement trends
        mood_trend = (features.avg_sentiment + features.engagement_trend) / 2
        
        # Adjust based on activity patterns
        if features.daily_avg_interactions > 2:
            mood_trend += 0.1  # High activity suggests positive mood
        elif features.daily_avg_interactions < 0.5:
            mood_trend -= 0.1  # Low activity might suggest negative mood
        
        confidence = 0.5  # Moderate confidence for heuristic
        return np.clip(mood_trend, -1, 1), confidence
    
    def _heuristic_action_recommendation(self, features: UserFeatures) -> Tuple[str, float]:
        """Recommend action using heuristic rules."""
        # High churn risk - re-engagement
        churn_risk, _ = self._heuristic_churn_risk(features)
        if churn_risk > 0.7:
            return "re_engagement", 0.8
        
        # Declining engagement - mood support
        if features.engagement_trend < -0.2:
            return "mood_support", 0.7
        
        # Long absence - personalized check-in
        if features.days_since_last > 5:
            return "personalized_checkin", 0.6
        
        # Low activity but good sentiment - feature suggestion
        if features.daily_avg_interactions < 1 and features.avg_sentiment > 0.2:
            return "feature_suggestion", 0.5
        
        # Regular activity - topic follow-up
        if features.daily_avg_interactions >= 1:
            return "topic_follow_up", 0.4
        
        # Default
        return "personalized_checkin", 0.3
    
    async def _extract_training_data(self) -> Dict[str, List]:
        """Extract training data from database."""
        training_data = {
            'features': [],
            'churn_labels': [],
            'mood_scores': [],
            'timing_labels': [],
            'action_labels': []
        }
        
        try:
            async with get_database_session() as session:
                # Get users with behavior patterns
                patterns_result = await session.execute(
                    select(UserBehaviorPattern)
                    .where(UserBehaviorPattern.last_pattern_analysis.isnot(None))
                    .limit(1000)  # Limit for performance
                )
                patterns = patterns_result.scalars().all()
                
                for pattern in patterns:
                    try:
                        features = await self._extract_user_features(str(pattern.user_id))
                        if not features:
                            continue
                        
                        training_data['features'].append(features.to_array())
                        
                        # Churn labels (1 if high risk, 0 if low risk)
                        churn_label = 1 if pattern.churn_risk_score > 0.6 else 0
                        training_data['churn_labels'].append(churn_label)
                        
                        # Mood scores (sentiment as continuous target)
                        mood_score = pattern.average_sentiment_score or 0
                        training_data['mood_scores'].append(mood_score)
                        
                        # Timing labels (0=morning, 1=afternoon, 2=evening)
                        if pattern.most_active_hour:
                            if pattern.most_active_hour < 12:
                                timing_label = 0
                            elif pattern.most_active_hour < 18:
                                timing_label = 1
                            else:
                                timing_label = 2
                        else:
                            timing_label = 1  # Default afternoon
                        training_data['timing_labels'].append(timing_label)
                        
                        # Action labels (simplified to main categories)
                        if pattern.churn_risk_score > 0.7:
                            action_label = 1  # re_engagement
                        elif pattern.shows_declining_engagement:
                            action_label = 4  # mood_support
                        elif pattern.days_since_last_interaction > 5:
                            action_label = 0  # personalized_checkin
                        else:
                            action_label = 5  # topic_follow_up
                        training_data['action_labels'].append(action_label)
                        
                    except Exception as e:
                        logger.warning("Error processing pattern for training", error=str(e), pattern_id=str(pattern.id))
                        continue
                
                logger.info(
                    "Training data extracted",
                    total_samples=len(training_data['features'])
                )
                
                return training_data
                
        except Exception as e:
            logger.error("Error extracting training data", error=str(e))
            return training_data
    
    async def _train_churn_model(self, X: np.ndarray, y: List[int]) -> Dict[str, Any]:
        """Train churn prediction model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.churn_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.churn_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.churn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'feature_importance': self.churn_model.feature_importances_.tolist()
            }
            
        except Exception as e:
            logger.error("Error training churn model", error=str(e))
            return {'error': str(e)}
    
    async def _train_mood_model(self, X: np.ndarray, y: List[float]) -> Dict[str, Any]:
        """Train mood prediction model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.mood_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.mood_model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.mood_model.score(X_train, y_train)
            test_score = self.mood_model.score(X_test, y_test)
            
            return {
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': self.mood_model.feature_importances_.tolist()
            }
            
        except Exception as e:
            logger.error("Error training mood model", error=str(e))
            return {'error': str(e)}
    
    async def _train_timing_model(self, X: np.ndarray, y: List[int]) -> Dict[str, Any]:
        """Train optimal timing model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.timing_model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42,
                max_iter=1000
            )
            
            self.timing_model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, self.timing_model.predict(X_test))
            
            return {
                'accuracy': accuracy,
                'classes': self.timing_model.classes_.tolist()
            }
            
        except Exception as e:
            logger.error("Error training timing model", error=str(e))
            return {'error': str(e)}
    
    async def _train_action_model(self, X: np.ndarray, y: List[int]) -> Dict[str, Any]:
        """Train action recommendation model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.action_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                class_weight='balanced'
            )
            
            self.action_model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, self.action_model.predict(X_test))
            
            return {
                'accuracy': accuracy,
                'feature_importance': self.action_model.feature_importances_.tolist(),
                'classes': self.action_model.classes_.tolist()
            }
            
        except Exception as e:
            logger.error("Error training action model", error=str(e))
            return {'error': str(e)}
    
    async def _load_models(self) -> None:
        """Load existing models from disk."""
        try:
            model_files = {
                'churn_model': self.model_dir / "churn_model.joblib",
                'mood_model': self.model_dir / "mood_model.joblib", 
                'timing_model': self.model_dir / "timing_model.joblib",
                'action_model': self.model_dir / "action_model.joblib",
                'scaler': self.model_dir / "feature_scaler.joblib"
            }
            
            for model_name, file_path in model_files.items():
                if file_path.exists():
                    if model_name == 'scaler':
                        self.feature_scaler = joblib.load(file_path)
                    else:
                        setattr(self, model_name, joblib.load(file_path))
                    logger.info(f"Loaded {model_name} from {file_path}")
                    
            # Load metadata
            metadata_file = self.model_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get('version', self.model_version)
                    if metadata.get('last_training'):
                        self.last_training = datetime.fromisoformat(metadata['last_training'])
                        
        except Exception as e:
            logger.warning("Error loading models", error=str(e))
    
    async def _save_models(self) -> None:
        """Save models to disk."""
        try:
            model_mapping = {
                'churn_model': self.churn_model,
                'mood_model': self.mood_model,
                'timing_model': self.timing_model,
                'action_model': self.action_model
            }
            
            for model_name, model in model_mapping.items():
                if model is not None:
                    file_path = self.model_dir / f"{model_name}.joblib"
                    joblib.dump(model, file_path)
                    logger.info(f"Saved {model_name} to {file_path}")
            
            # Save feature scaler
            scaler_path = self.model_dir / "feature_scaler.joblib"
            joblib.dump(self.feature_scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'last_training': self.last_training.isoformat() if self.last_training else None,
                'feature_names': self.feature_names
            }
            
            metadata_file = self.model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error("Error saving models", error=str(e))