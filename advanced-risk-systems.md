# Advanced Risk Assessment Systems
## Behavioral Patterns, Red Flags, and Optimization Frameworks

## 5. Behavioral Pattern Analysis

### 5.1 User Behavior Profiling System
```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple, Optional

class BehavioralPatternAnalyzer:
    def __init__(self):
        # Behavior categories and their weights
        self.behavior_dimensions = {
            'temporal_patterns': 0.20,     # When and how often they engage
            'communication_style': 0.25,  # How they communicate
            'engagement_depth': 0.20,     # Level of engagement
            'progression_velocity': 0.15, # Speed of moving through funnel
            'response_patterns': 0.20     # Response timing and consistency
        }
        
        # Behavioral clusters (learned from data)
        self.behavior_clusters = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Known behavioral patterns
        self.pattern_templates = {
            'high_value_prospect': {
                'characteristics': {
                    'avg_session_duration': (15, 45),  # minutes
                    'response_time_consistency': (0.7, 1.0),
                    'message_length_avg': (10, 30),    # words
                    'question_quality_score': (0.6, 1.0),
                    'engagement_progression': (0.5, 1.0)
                },
                'conversion_probability': 0.75,
                'priority': 'VERY_HIGH'
            },
            'analytical_evaluator': {
                'characteristics': {
                    'avg_session_duration': (20, 60),
                    'technical_language_ratio': (0.3, 0.8),
                    'question_depth_score': (0.7, 1.0),
                    'comparison_mentions': (2, 10),
                    'detail_requests': (3, 15)
                },
                'conversion_probability': 0.65,
                'priority': 'HIGH'
            },
            'price_shopper': {
                'characteristics': {
                    'price_mentions': (3, 20),
                    'discount_inquiries': (1, 5),
                    'comparison_frequency': (0.4, 0.9),
                    'urgency_indicators': (0.1, 0.4),
                    'commitment_signals': (0.1, 0.3)
                },
                'conversion_probability': 0.35,
                'priority': 'MEDIUM'
            },
            'tire_kicker': {
                'characteristics': {
                    'session_frequency': (0.1, 0.3),   # Low frequency
                    'commitment_avoidance': (0.6, 1.0),
                    'vague_responses': (0.5, 0.9),
                    'timeline_indefinite': (0.7, 1.0),
                    'budget_evasion': (0.6, 1.0)
                },
                'conversion_probability': 0.15,
                'priority': 'LOW'
            },
            'urgent_buyer': {
                'characteristics': {
                    'urgency_language': (0.6, 1.0),
                    'quick_responses': (0.8, 1.0),
                    'decision_indicators': (0.5, 1.0),
                    'timeline_specific': (0.7, 1.0),
                    'authority_confirmed': (0.6, 1.0)
                },
                'conversion_probability': 0.85,
                'priority': 'VERY_HIGH'
            },
            'social_validator': {
                'characteristics': {
                    'social_proof_requests': (2, 8),
                    'testimonial_interest': (0.4, 0.8),
                    'peer_comparison': (0.3, 0.7),
                    'case_study_engagement': (0.5, 1.0),
                    'authority_seeking': (0.4, 0.8)
                },
                'conversion_probability': 0.55,
                'priority': 'MEDIUM'
            }
        }
    
    def extract_behavioral_features(self, conversation_history: List[Dict], 
                                   user_metadata: Dict = None) -> Dict:
        """Extract comprehensive behavioral features from conversation history"""
        
        if not conversation_history:
            return self._create_zero_behavior_profile()
        
        user_messages = [msg for msg in conversation_history if msg.get('sender') == 'user']
        
        if not user_messages:
            return self._create_zero_behavior_profile()
        
        features = {}
        
        # 1. Temporal Pattern Analysis
        features.update(self._analyze_temporal_patterns(user_messages))
        
        # 2. Communication Style Analysis
        features.update(self._analyze_communication_style(user_messages))
        
        # 3. Engagement Depth Analysis
        features.update(self._analyze_engagement_depth(user_messages))
        
        # 4. Progression Velocity Analysis
        features.update(self._analyze_progression_velocity(conversation_history))
        
        # 5. Response Pattern Analysis
        features.update(self._analyze_response_patterns(user_messages))
        
        return features
    
    def _analyze_temporal_patterns(self, user_messages: List[Dict]) -> Dict:
        """Analyze when and how often user engages"""
        if not user_messages:
            return {}
        
        timestamps = [msg.get('timestamp', time.time()) for msg in user_messages]
        
        # Session duration
        session_duration = (max(timestamps) - min(timestamps)) / 60  # minutes
        
        # Message frequency
        if session_duration > 0:
            message_frequency = len(user_messages) / session_duration
        else:
            message_frequency = len(user_messages)
        
        # Response intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            intervals.append(interval)
        
        avg_response_interval = np.mean(intervals) if intervals else 0
        response_consistency = 1.0 / (1.0 + np.std(intervals)) if intervals and np.std(intervals) > 0 else 1.0
        
        # Time of day pattern (if timestamp includes hour)
        hours = []
        for ts in timestamps:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
                hours.append(dt.hour)
        
        # Business hours engagement (9 AM - 5 PM)
        business_hours_ratio = sum(1 for h in hours if 9 <= h <= 17) / len(hours) if hours else 0.5
        
        return {
            'session_duration': session_duration,
            'message_frequency': message_frequency,
            'avg_response_interval': avg_response_interval,
            'response_consistency': response_consistency,
            'business_hours_ratio': business_hours_ratio,
            'total_messages': len(user_messages)
        }
    
    def _analyze_communication_style(self, user_messages: List[Dict]) -> Dict:
        """Analyze how user communicates"""
        all_text = ' '.join([msg.get('text', '') for msg in user_messages]).lower()
        total_words = len(all_text.split())
        
        if total_words == 0:
            return {}
        
        # Message length analysis
        message_lengths = [len(msg.get('text', '').split()) for msg in user_messages]
        avg_message_length = np.mean(message_lengths)
        message_length_variance = np.var(message_lengths)
        
        # Language formality
        formal_indicators = ['please', 'thank you', 'appreciate', 'kindly', 'sincerely', 'regards']
        casual_indicators = ['hey', 'yeah', 'ok', 'cool', 'awesome', 'lol', 'btw']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in all_text)
        casual_count = sum(1 for indicator in casual_indicators if indicator in all_text)
        
        formality_score = (formal_count - casual_count) / len(user_messages) + 0.5
        formality_score = max(0, min(1, formality_score))
        
        # Technical language usage
        technical_terms = [
            'api', 'integration', 'implementation', 'infrastructure', 'scalability',
            'optimization', 'configuration', 'architecture', 'deployment', 'analytics',
            'metrics', 'dashboard', 'automation', 'workflow', 'database', 'security'
        ]
        
        technical_count = sum(1 for term in technical_terms if term in all_text)
        technical_language_ratio = technical_count / total_words if total_words > 0 else 0
        
        # Question patterns
        questions = [msg.get('text', '') for msg in user_messages if '?' in msg.get('text', '')]
        question_ratio = len(questions) / len(user_messages) if user_messages else 0
        
        # Specific vs. generic questions
        specific_question_indicators = [
            'how much', 'when exactly', 'what specific', 'which particular', 'how many'
        ]
        specific_questions = sum(1 for q in questions 
                               if any(indicator in q.lower() for indicator in specific_question_indicators))
        question_specificity = specific_questions / len(questions) if questions else 0
        
        return {
            'avg_message_length': avg_message_length,
            'message_length_variance': message_length_variance,
            'formality_score': formality_score,
            'technical_language_ratio': technical_language_ratio,
            'question_ratio': question_ratio,
            'question_specificity': question_specificity
        }
    
    def _analyze_engagement_depth(self, user_messages: List[Dict]) -> Dict:
        """Analyze depth of user engagement"""
        all_text = ' '.join([msg.get('text', '') for msg in user_messages]).lower()
        
        # Deep engagement indicators
        deep_engagement_signals = [
            'tell me more', 'explain in detail', 'how does this work', 'what are the benefits',
            'show me example', 'can you elaborate', 'dive deeper', 'comprehensive'
        ]
        
        # Surface engagement indicators
        surface_signals = [
            'ok', 'fine', 'sure', 'maybe', 'i see', 'got it'
        ]
        
        deep_count = sum(1 for signal in deep_engagement_signals if signal in all_text)
        surface_count = sum(1 for signal in surface_signals if signal in all_text)
        
        engagement_depth_score = (deep_count - surface_count * 0.5) / len(user_messages)
        engagement_depth_score = max(0, engagement_depth_score)
        
        # Follow-up question patterns
        followup_indicators = ['also', 'additionally', 'what about', 'and', 'another question']
        followup_count = sum(1 for msg in user_messages 
                           if any(indicator in msg.get('text', '').lower() for indicator in followup_indicators))
        followup_ratio = followup_count / len(user_messages)
        
        # Detail requests
        detail_request_indicators = [
            'details', 'specifications', 'more information', 'breakdown', 'step by step'
        ]
        detail_requests = sum(1 for indicator in detail_request_indicators if indicator in all_text)
        detail_request_ratio = detail_requests / len(user_messages)
        
        return {
            'engagement_depth_score': engagement_depth_score,
            'followup_ratio': followup_ratio,
            'detail_request_ratio': detail_request_ratio
        }
    
    def _analyze_progression_velocity(self, conversation_history: List[Dict]) -> Dict:
        """Analyze speed of moving through conversation funnel"""
        user_messages = [msg for msg in conversation_history if msg.get('sender') == 'user']
        
        if len(user_messages) < 3:
            return {'progression_velocity': 0.0}
        
        # Sales funnel stage indicators
        funnel_stages = {
            'awareness': ['what is', 'tell me about', 'learn about', 'understand'],
            'interest': ['interested', 'sounds good', 'like that', 'appealing'],
            'consideration': ['compare', 'versus', 'options', 'alternatives', 'evaluate'],
            'intent': ['price', 'cost', 'buy', 'purchase', 'trial', 'demo'],
            'decision': ['ready', 'let\'s do it', 'sign up', 'move forward', 'agree']
        }
        
        stage_values = {'awareness': 1, 'interest': 2, 'consideration': 3, 'intent': 4, 'decision': 5}
        
        # Calculate stage progression over time
        message_stages = []
        for msg in user_messages:
            text = msg.get('text', '').lower()
            max_stage = 0
            
            for stage, indicators in funnel_stages.items():
                if any(indicator in text for indicator in indicators):
                    max_stage = max(max_stage, stage_values[stage])
            
            message_stages.append(max_stage if max_stage > 0 else 1)  # Default to awareness
        
        # Calculate velocity (stage progression rate)
        if len(message_stages) > 1:
            progression_velocity = (max(message_stages) - min(message_stages)) / len(message_stages)
        else:
            progression_velocity = 0.0
        
        # Commitment escalation
        commitment_indicators = [
            'definitely', 'absolutely', 'certainly', 'for sure', 'without doubt'
        ]
        
        all_text = ' '.join([msg.get('text', '') for msg in user_messages]).lower()
        commitment_count = sum(1 for indicator in commitment_indicators if indicator in all_text)
        commitment_escalation = commitment_count / len(user_messages)
        
        return {
            'progression_velocity': progression_velocity,
            'max_funnel_stage': max(message_stages),
            'commitment_escalation': commitment_escalation,
            'stage_consistency': 1.0 - np.std(message_stages) / 5.0 if len(message_stages) > 1 else 1.0
        }
    
    def _analyze_response_patterns(self, user_messages: List[Dict]) -> Dict:
        """Analyze user response patterns and consistency"""
        if len(user_messages) < 2:
            return {'response_pattern_score': 0.5}
        
        # Response timing analysis
        timestamps = [msg.get('timestamp', time.time()) for msg in user_messages]
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        if not intervals:
            return {'response_pattern_score': 0.5}
        
        # Response time consistency
        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)
        consistency = 1.0 / (1.0 + interval_std / 60) if interval_std > 0 else 1.0  # Normalize by minute
        
        # Quick response indicator (responses < 2 minutes)
        quick_responses = sum(1 for interval in intervals if interval < 120)
        quick_response_ratio = quick_responses / len(intervals)
        
        # Response length consistency
        message_lengths = [len(msg.get('text', '').split()) for msg in user_messages]
        length_consistency = 1.0 / (1.0 + np.std(message_lengths) / 10) if np.std(message_lengths) > 0 else 1.0
        
        response_pattern_score = (consistency + quick_response_ratio + length_consistency) / 3
        
        return {
            'response_pattern_score': response_pattern_score,
            'avg_response_interval': avg_interval,
            'response_consistency': consistency,
            'quick_response_ratio': quick_response_ratio,
            'length_consistency': length_consistency
        }
    
    def classify_behavioral_pattern(self, behavioral_features: Dict) -> Dict:
        """Classify user into behavioral pattern categories"""
        
        pattern_scores = {}
        
        for pattern_name, pattern_config in self.pattern_templates.items():
            score = self._calculate_pattern_match_score(behavioral_features, pattern_config['characteristics'])
            pattern_scores[pattern_name] = {
                'match_score': score,
                'conversion_probability': pattern_config['conversion_probability'],
                'priority': pattern_config['priority']
            }
        
        # Find best matching pattern
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1]['match_score'])
        
        return {
            'primary_pattern': best_pattern[0],
            'match_confidence': best_pattern[1]['match_score'],
            'conversion_probability': best_pattern[1]['conversion_probability'],
            'priority': best_pattern[1]['priority'],
            'all_pattern_scores': pattern_scores,
            'behavioral_insights': self._generate_behavioral_insights(best_pattern[0], behavioral_features)
        }
    
    def _calculate_pattern_match_score(self, features: Dict, pattern_characteristics: Dict) -> float:
        """Calculate how well features match a behavioral pattern"""
        matches = 0
        total_characteristics = 0
        
        for characteristic, expected_range in pattern_characteristics.items():
            if characteristic in features:
                total_characteristics += 1
                value = features[characteristic]
                
                # Check if value falls within expected range
                if isinstance(expected_range, tuple) and len(expected_range) == 2:
                    min_val, max_val = expected_range
                    if min_val <= value <= max_val:
                        matches += 1
                    else:
                        # Partial credit for close matches
                        range_size = max_val - min_val
                        if value < min_val:
                            distance = min_val - value
                        else:
                            distance = value - max_val
                        
                        partial_credit = max(0, 1 - (distance / range_size))
                        matches += partial_credit
                elif isinstance(expected_range, (int, float)):
                    # Exact value matching with tolerance
                    tolerance = abs(expected_range * 0.2)  # 20% tolerance
                    if abs(value - expected_range) <= tolerance:
                        matches += 1
        
        return matches / total_characteristics if total_characteristics > 0 else 0.0
    
    def _generate_behavioral_insights(self, pattern_name: str, features: Dict) -> List[str]:
        """Generate insights about user behavioral pattern"""
        insights = []
        
        pattern_insights = {
            'high_value_prospect': [
                "User shows strong engagement and commitment signals",
                "Likely to convert with proper nurturing",
                "Focus on value demonstration and next steps"
            ],
            'analytical_evaluator': [
                "User conducts thorough evaluation process",
                "Provide detailed technical information and comparisons",
                "Be patient with longer decision timeline"
            ],
            'price_shopper': [
                "Primary focus on price and discounts",
                "Emphasize value proposition and ROI",
                "Consider offering limited-time incentives"
            ],
            'tire_kicker': [
                "Shows browsing behavior without clear intent",
                "Requires strong qualification to determine viability",
                "Consider setting boundaries on time investment"
            ],
            'urgent_buyer': [
                "High sense of urgency and quick decision-making",
                "Accelerate sales process and remove friction",
                "Prioritize immediate response and availability"
            ],
            'social_validator': [
                "Seeks external validation and social proof",
                "Provide testimonials, case studies, and references",
                "Highlight popularity and peer adoption"
            ]
        }
        
        base_insights = pattern_insights.get(pattern_name, ["Standard behavioral pattern detected"])
        insights.extend(base_insights)
        
        # Add feature-specific insights
        if features.get('technical_language_ratio', 0) > 0.3:
            insights.append("High technical language usage - user likely has technical background")
        
        if features.get('response_consistency', 0) > 0.8:
            insights.append("Very consistent response patterns - indicates high engagement")
        
        if features.get('progression_velocity', 0) > 0.5:
            insights.append("Fast progression through sales funnel - ready for advancement")
        
        return insights
    
    def track_behavioral_evolution(self, chat_id: int, current_features: Dict, 
                                 historical_features: List[Dict]) -> Dict:
        """Track how user behavior evolves over time"""
        if not historical_features or len(historical_features) < 2:
            return {'status': 'INSUFFICIENT_HISTORY'}
        
        # Compare current features with historical averages
        feature_evolution = {}
        
        for feature_name in current_features.keys():
            if feature_name in historical_features[-1]:
                historical_values = [f.get(feature_name, 0) for f in historical_features[-3:]]
                historical_avg = np.mean(historical_values)
                current_value = current_features[feature_name]
                
                change = current_value - historical_avg
                change_percentage = (change / historical_avg * 100) if historical_avg != 0 else 0
                
                feature_evolution[feature_name] = {
                    'current_value': current_value,
                    'historical_avg': historical_avg,
                    'absolute_change': change,
                    'percentage_change': change_percentage,
                    'trend': 'INCREASING' if change > 0.1 else 'DECREASING' if change < -0.1 else 'STABLE'
                }
        
        # Identify significant behavioral shifts
        significant_changes = [
            feature for feature, data in feature_evolution.items() 
            if abs(data['percentage_change']) > 25  # 25% change threshold
        ]
        
        return {
            'status': 'ANALYZED',
            'feature_evolution': feature_evolution,
            'significant_changes': significant_changes,
            'overall_trend': self._determine_overall_behavioral_trend(feature_evolution),
            'behavioral_shift_risk': len(significant_changes) / len(feature_evolution) if feature_evolution else 0
        }
    
    def _determine_overall_behavioral_trend(self, feature_evolution: Dict) -> str:
        """Determine overall behavioral trend direction"""
        if not feature_evolution:
            return 'NO_DATA'
        
        positive_changes = sum(1 for data in feature_evolution.values() if data['trend'] == 'INCREASING')
        negative_changes = sum(1 for data in feature_evolution.values() if data['trend'] == 'DECREASING')
        
        if positive_changes > negative_changes * 1.5:
            return 'IMPROVING'
        elif negative_changes > positive_changes * 1.5:
            return 'DECLINING'
        else:
            return 'MIXED'
    
    def _create_zero_behavior_profile(self) -> Dict:
        """Create zero behavioral profile for empty data"""
        return {
            'session_duration': 0,
            'message_frequency': 0,
            'avg_response_interval': 0,
            'response_consistency': 0,
            'business_hours_ratio': 0.5,
            'total_messages': 0,
            'avg_message_length': 0,
            'formality_score': 0.5,
            'technical_language_ratio': 0,
            'question_ratio': 0,
            'engagement_depth_score': 0,
            'progression_velocity': 0,
            'response_pattern_score': 0.5
        }
```

### 5.2 Advanced Behavioral Clustering
```python
class BehavioralClustering:
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.cluster_profiles = {}
        self.is_trained = False
        
    def train_behavioral_clusters(self, behavioral_data: List[Dict]) -> Dict:
        """Train clustering models on behavioral data"""
        if len(behavioral_data) < 10:
            return {'status': 'INSUFFICIENT_DATA', 'required_minimum': 10}
        
        # Convert to DataFrame
        df = pd.DataFrame(behavioral_data)
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(df)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Determine optimal number of clusters using elbow method
        optimal_k = self._find_optimal_clusters(X_pca)
        
        # Train K-Means
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = self.kmeans_model.fit_predict(X_pca)
        
        # Train DBSCAN for outlier detection
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = self.dbscan_model.fit_predict(X_pca)
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(df, kmeans_labels, optimal_k)
        
        self.is_trained = True
        
        return {
            'status': 'TRAINED',
            'optimal_clusters': optimal_k,
            'cluster_analysis': cluster_analysis,
            'outliers_detected': sum(1 for label in dbscan_labels if label == -1),
            'feature_importance': self._calculate_feature_importance(df, kmeans_labels)
        }
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        k_range = range(2, min(max_k + 1, len(X) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diff2 = np.diff(diffs)
            elbow_idx = np.argmax(diff2) + 2  # +2 because of double diff
            return k_range[min(elbow_idx, len(k_range) - 1)]
        else:
            return k_range[0]  # Default to minimum
    
    def _analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> Dict:
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate cluster characteristics
            cluster_profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'feature_means': cluster_data.mean().to_dict(),
                'feature_stds': cluster_data.std().to_dict(),
                'behavioral_type': self._classify_cluster_behavior(cluster_data.mean().to_dict())
            }
            
            cluster_analysis[f'cluster_{cluster_id}'] = cluster_profile
        
        return cluster_analysis
    
    def _classify_cluster_behavior(self, cluster_means: Dict) -> str:
        """Classify cluster behavior type based on mean features"""
        # High engagement + high progression = Premium prospects
        if (cluster_means.get('engagement_depth_score', 0) > 0.6 and 
            cluster_means.get('progression_velocity', 0) > 0.4):
            return 'premium_prospects'
        
        # High technical language + detailed questions = Technical evaluators
        elif (cluster_means.get('technical_language_ratio', 0) > 0.3 and
              cluster_means.get('question_specificity', 0) > 0.5):
            return 'technical_evaluators'
        
        # Low engagement + inconsistent responses = Tire kickers
        elif (cluster_means.get('engagement_depth_score', 0) < 0.3 and
              cluster_means.get('response_consistency', 0) < 0.4):
            return 'tire_kickers'
        
        # Quick responses + urgency = Urgent buyers
        elif (cluster_means.get('quick_response_ratio', 0) > 0.7 and
              cluster_means.get('progression_velocity', 0) > 0.5):
            return 'urgent_buyers'
        
        # High formality + detailed requests = Corporate prospects
        elif (cluster_means.get('formality_score', 0) > 0.7 and
              cluster_means.get('detail_request_ratio', 0) > 0.4):
            return 'corporate_prospects'
        
        else:
            return 'general_prospects'
    
    def _calculate_feature_importance(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Calculate which features are most important for clustering"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train RF classifier to predict cluster labels
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(df, labels)
        
        # Get feature importances
        feature_importance = dict(zip(df.columns, rf.feature_importances_))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def predict_behavioral_cluster(self, behavioral_features: Dict) -> Dict:
        """Predict which behavioral cluster a user belongs to"""
        if not self.is_trained:
            return {'status': 'NOT_TRAINED'}
        
        # Convert to DataFrame
        df = pd.DataFrame([behavioral_features])
        df = df.fillna(0)  # Fill missing values
        
        # Standardize and apply PCA
        X_scaled = self.scaler.transform(df)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict cluster
        cluster_id = self.kmeans_model.predict(X_pca)[0]
        
        # Calculate distance to cluster center (confidence)
        center = self.kmeans_model.cluster_centers_[cluster_id]
        distance = np.linalg.norm(X_pca[0] - center)
        confidence = max(0, 1 - distance / 2)  # Normalize distance
        
        # Check for outliers
        outlier_label = self.dbscan_model.fit_predict(X_pca)[0]
        is_outlier = outlier_label == -1
        
        return {
            'status': 'PREDICTED',
            'cluster_id': cluster_id,
            'cluster_name': f'cluster_{cluster_id}',
            'confidence': confidence,
            'is_outlier': is_outlier,
            'behavioral_type': self.cluster_profiles.get(f'cluster_{cluster_id}', {}).get('behavioral_type', 'unknown'),
            'distance_to_center': distance
        }
```

## 6. Red Flag Detection Systems

### 6.1 Multi-Layer Red Flag Detection
```python
class RedFlagDetectionSystem:
    def __init__(self):
        # Red flag categories with severity levels
        self.red_flag_categories = {
            'fraud_indicators': {
                'severity': 'CRITICAL',
                'threshold': 0.8,
                'immediate_action': True
            },
            'spam_behavior': {
                'severity': 'HIGH',
                'threshold': 0.7,
                'immediate_action': True
            },
            'abusive_language': {
                'severity': 'HIGH',
                'threshold': 0.6,
                'immediate_action': True
            },
            'time_wasting_patterns': {
                'severity': 'MEDIUM',
                'threshold': 0.6,
                'immediate_action': False
            },
            'competitor_intelligence': {
                'severity': 'MEDIUM',
                'threshold': 0.5,
                'immediate_action': False
            },
            'technical_probing': {
                'severity': 'LOW',
                'threshold': 0.7,
                'immediate_action': False
            },
            'price_fishing': {
                'severity': 'LOW',
                'threshold': 0.5,
                'immediate_action': False
            }
        }
        
        # Pattern libraries for each red flag type
        self.detection_patterns = self._initialize_detection_patterns()
        
    def _initialize_detection_patterns(self) -> Dict:
        """Initialize red flag detection patterns"""
        return {
            'fraud_indicators': {
                'payment_fraud': [
                    'stolen card', 'chargeback', 'dispute payment', 'not authorized',
                    'fraudulent', 'unauthorized transaction'
                ],
                'identity_fraud': [
                    'not my real name', 'fake information', 'temporary email',
                    'disposable phone', 'someone else\'s details'
                ],
                'account_takeover': [
                    'account compromised', 'didn\'t create account', 'hacked account',
                    'unauthorized access'
                ]
            },
            'spam_behavior': {
                'message_spam': [
                    r'(.)\1{10,}',  # Repeated characters
                    r'[A-Z]{20,}',  # All caps
                    r'!!!{3,}',     # Multiple exclamations
                    r'\b(URGENT|IMMEDIATE|ACT NOW|LIMITED TIME)\b'
                ],
                'link_spam': [
                    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                    'click here', 'visit link', 'check out'
                ],
                'contact_spam': [
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
                    'whatsapp me', 'telegram me', 'signal me'
                ]
            },
            'abusive_language': {
                'profanity': [
                    # This would contain profanity patterns - using placeholders
                    'profanity_pattern_1', 'profanity_pattern_2'
                ],
                'threats': [
                    'i will sue', 'legal action', 'report you', 'shut you down',
                    'destroy your business', 'ruin your reputation'
                ],
                'harassment': [
                    'keep bothering', 'stop messaging', 'harassment', 'stalking'
                ]
            },
            'time_wasting_patterns': {
                'indefinite_timeline': [
                    'maybe later', 'not sure when', 'sometime in future',
                    'no rush', 'no hurry', 'whenever'
                ],
                'budget_evasion': [
                    'no budget', 'depends on price', 'need to see cost first',
                    'budget is tight', 'money is issue'
                ],
                'decision_avoidance': [
                    'need to think', 'discuss with team', 'consult with others',
                    'not ready to decide', 'still considering'
                ]
            },
            'competitor_intelligence': {
                'competitor_mentions': [
                    'competitor_name_1', 'competitor_name_2',  # Replace with actual competitors
                    'what about [competitor]', 'compared to [competitor]'
                ],
                'feature_probing': [
                    'what features do you have', 'full feature list',
                    'all capabilities', 'complete functionality'
                ],
                'pricing_probing': [
                    'your pricing model', 'how do you price', 'pricing strategy',
                    'discount structure', 'enterprise pricing'
                ]
            },
            'technical_probing': {
                'security_probing': [
                    'security vulnerabilities', 'penetration testing', 'exploit',
                    'security flaws', 'backdoor', 'breach'
                ],
                'infrastructure_probing': [
                    'server architecture', 'database structure', 'cloud provider',
                    'infrastructure details', 'technical stack'
                ]
            },
            'price_fishing': {
                'price_focus': [
                    'cheapest option', 'lowest price', 'best deal', 'discount available',
                    'price match', 'negotiate price'
                ],
                'comparison_shopping': [
                    'checking other options', 'shopping around', 'comparing prices',
                    'other vendors', 'alternative solutions'
                ]
            }
        }
    
    def analyze_red_flags(self, conversation_history: List[Dict], 
                         user_metadata: Dict = None) -> Dict:
        """Comprehensive red flag analysis"""
        
        if not conversation_history:
            return {'status': 'NO_DATA', 'red_flags': []}
        
        user_messages = [msg for msg in conversation_history if msg.get('sender') == 'user']
        all_text = ' '.join([msg.get('text', '') for msg in user_messages]).lower()
        
        detected_red_flags = []
        category_scores = {}
        
        # Analyze each red flag category
        for category, config in self.red_flag_categories.items():
            category_score, category_flags = self._analyze_category(
                category, all_text, user_messages, config
            )
            
            category_scores[category] = category_score
            
            if category_score >= config['threshold']:
                detected_red_flags.extend(category_flags)
        
        # Calculate overall red flag risk
        overall_risk = self._calculate_overall_risk(category_scores)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk)
        
        # Generate recommendations
        recommendations = self._generate_red_flag_recommendations(detected_red_flags, risk_level)
        
        return {
            'status': 'ANALYZED',
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'detected_red_flags': detected_red_flags,
            'category_scores': category_scores,
            'recommendations': recommendations,
            'immediate_action_required': any(flag['immediate_action'] for flag in detected_red_flags)
        }
    
    def _analyze_category(self, category: str, all_text: str, 
                         user_messages: List[Dict], config: Dict) -> Tuple[float, List[Dict]]:
        """Analyze specific red flag category"""
        
        if category not in self.detection_patterns:
            return 0.0, []
        
        category_patterns = self.detection_patterns[category]
        detected_flags = []
        total_score = 0.0
        pattern_count = 0
        
        # Analyze each pattern group in category
        for pattern_group, patterns in category_patterns.items():
            pattern_count += len(patterns)
            group_score = 0.0
            
            for pattern in patterns:
                if isinstance(pattern, str):
                    # Simple string matching
                    if pattern.lower() in all_text:
                        group_score += 1.0
                        detected_flags.append({
                            'category': category,
                            'pattern_group': pattern_group,
                            'matched_pattern': pattern,
                            'severity': config['severity'],
                            'immediate_action': config['immediate_action'],
                            'context': self._extract_context(all_text, pattern)
                        })
                else:
                    # Regex pattern matching
                    import re
                    matches = re.findall(pattern, all_text, re.IGNORECASE)
                    if matches:
                        group_score += len(matches) * 0.5  # Multiple matches
                        detected_flags.append({
                            'category': category,
                            'pattern_group': pattern_group,
                            'matched_pattern': f'regex: {pattern}',
                            'matches': matches,
                            'severity': config['severity'],
                            'immediate_action': config['immediate_action'],
                            'context': f'Found {len(matches)} matches'
                        })
            
            total_score += min(1.0, group_score)  # Cap each group at 1.0
        
        # Normalize score by number of pattern groups
        category_score = total_score / len(category_patterns) if category_patterns else 0.0
        
        # Apply additional contextual analysis
        contextual_score = self._apply_contextual_analysis(category, user_messages)
        final_score = min(1.0, category_score + contextual_score * 0.2)  # 20% boost from context
        
        return final_score, detected_flags
    
    def _apply_contextual_analysis(self, category: str, user_messages: List[Dict]) -> float:
        """Apply contextual analysis to improve detection accuracy"""
        
        if not user_messages:
            return 0.0
        
        contextual_indicators = {
            'spam_behavior': self._analyze_spam_context,
            'time_wasting_patterns': self._analyze_time_wasting_context,
            'abusive_language': self._analyze_abuse_context,
            'competitor_intelligence': self._analyze_competitor_context
        }
        
        analyzer = contextual_indicators.get(category)
        if analyzer:
            return analyzer(user_messages)
        
        return 0.0
    
    def _analyze_spam_context(self, user_messages: List[Dict]) -> float:
        """Analyze spam behavior context"""
        spam_indicators = 0.0
        
        # Message repetition
        message_texts = [msg.get('text', '') for msg in user_messages]
        unique_messages = len(set(message_texts))
        if len(message_texts) > 3 and unique_messages / len(message_texts) < 0.5:
            spam_indicators += 0.3
        
        # Rapid message sending
        timestamps = [msg.get('timestamp', 0) for msg in user_messages]
        if len(timestamps) > 5:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_interval = np.mean(intervals)
            if avg_interval < 10:  # Less than 10 seconds between messages
                spam_indicators += 0.4
        
        # Message length analysis
        avg_length = np.mean([len(msg.get('text', '')) for msg in user_messages])
        if avg_length < 10 or avg_length > 1000:  # Very short or very long messages
            spam_indicators += 0.2
        
        return min(1.0, spam_indicators)
    
    def _analyze_time_wasting_context(self, user_messages: List[Dict]) -> float:
        """Analyze time-wasting behavior context"""
        time_wasting_score = 0.0
        
        all_text = ' '.join([msg.get('text', '') for msg in user_messages]).lower()
        
        # Lack of commitment progression
        commitment_words = ['yes', 'definitely', 'sure', 'absolutely', 'agree']
        tentative_words = ['maybe', 'possibly', 'might', 'could', 'perhaps']
        
        commitment_count = sum(1 for word in commitment_words if word in all_text)
        tentative_count = sum(1 for word in tentative_words if word in all_text)
        
        if tentative_count > commitment_count * 2:
            time_wasting_score += 0.3
        
        # Session length without progression
        if len(user_messages) > 10:
            # Long conversation without clear intent
            intent_indicators = ['buy', 'purchase', 'price', 'demo', 'trial', 'meeting']
            intent_mentions = sum(1 for indicator in intent_indicators if indicator in all_text)
            
            if intent_mentions == 0:
                time_wasting_score += 0.4
        
        return min(1.0, time_wasting_score)
    
    def _analyze_abuse_context(self, user_messages: List[Dict]) -> float:
        """Analyze abusive behavior context"""
        abuse_score = 0.0
        
        # Escalating negative sentiment
        message_sentiments = []
        for msg in user_messages[-5:]:  # Last 5 messages
            # Simplified sentiment analysis
            text = msg.get('text', '').lower()
            negative_words = ['angry', 'frustrated', 'terrible', 'awful', 'hate', 'worst']
            positive_words = ['good', 'great', 'fine', 'okay', 'thanks', 'appreciate']
            
            neg_count = sum(1 for word in negative_words if word in text)
            pos_count = sum(1 for word in positive_words if word in text)
            
            sentiment = pos_count - neg_count
            message_sentiments.append(sentiment)
        
        if len(message_sentiments) >= 3:
            # Check for increasingly negative trend
            if all(s < 0 for s in message_sentiments[-3:]):
                abuse_score += 0.5
        
        return min(1.0, abuse_score)
    
    def _analyze_competitor_context(self, user_messages: List[Dict]) -> float:
        """Analyze competitor intelligence gathering context"""
        intel_score = 0.0
        
        all_text = ' '.join([msg.get('text', '') for msg in user_messages]).lower()
        
        # Detailed technical questions without buying intent
        technical_questions = all_text.count('?')
        buying_indicators = ['buy', 'purchase', 'price', 'cost', 'budget']
        buying_mentions = sum(1 for indicator in buying_indicators if indicator in all_text)
        
        if technical_questions > 5 and buying_mentions == 0:
            intel_score += 0.4
        
        # Specific feature probing
        feature_words = ['feature', 'capability', 'function', 'specification', 'detail']
        feature_mentions = sum(1 for word in feature_words if word in all_text)
        
        if feature_mentions > len(user_messages) * 0.5:  # More than 50% of messages
            intel_score += 0.3
        
        return min(1.0, intel_score)
    
    def _extract_context(self, text: str, pattern: str, context_length: int = 50) -> str:
        """Extract context around matched pattern"""
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        
        index = text_lower.find(pattern_lower)
        if index == -1:
            return ""
        
        start = max(0, index - context_length)
        end = min(len(text), index + len(pattern) + context_length)
        
        return f"...{text[start:end]}..."
    
    def _calculate_overall_risk(self, category_scores: Dict) -> float:
        """Calculate overall red flag risk score"""
        if not category_scores:
            return 0.0
        
        # Weight categories by severity
        severity_weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.6,
            'LOW': 0.4
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            severity = self.red_flag_categories[category]['severity']
            weight = severity_weights[severity]
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_risk_level(self, overall_risk: float) -> str:
        """Determine risk level based on overall score"""
        if overall_risk >= 0.8:
            return 'CRITICAL'
        elif overall_risk >= 0.6:
            return 'HIGH'
        elif overall_risk >= 0.4:
            return 'MEDIUM'
        elif overall_risk >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _generate_red_flag_recommendations(self, detected_flags: List[Dict], risk_level: str) -> List[Dict]:
        """Generate recommendations based on detected red flags"""
        recommendations = []
        
        # Immediate action flags
        immediate_flags = [flag for flag in detected_flags if flag['immediate_action']]
        if immediate_flags:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'action': 'TERMINATE_CONVERSATION',
                'message': 'Critical red flags detected - terminate conversation immediately',
                'affected_categories': list(set(flag['category'] for flag in immediate_flags))
            })
        
        # Category-specific recommendations
        category_groups = {}
        for flag in detected_flags:
            category = flag['category']
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(flag)
        
        for category, flags in category_groups.items():
            recommendation = self._get_category_recommendation(category, flags, risk_level)
            if recommendation:
                recommendations.append(recommendation)
        
        return recommendations
    
    def _get_category_recommendation(self, category: str, flags: List[Dict], risk_level: str) -> Optional[Dict]:
        """Get specific recommendation for red flag category"""
        
        category_recommendations = {
            'fraud_indicators': {
                'action': 'SECURITY_ESCALATION',
                'message': 'Potential fraud detected - escalate to security team',
                'priority': 'IMMEDIATE'
            },
            'spam_behavior': {
                'action': 'IMPLEMENT_RATE_LIMITING',
                'message': 'Spam behavior detected - implement rate limiting',
                'priority': 'HIGH'
            },
            'abusive_language': {
                'action': 'ESCALATE_TO_HUMAN',
                'message': 'Abusive language detected - escalate to human agent',
                'priority': 'HIGH'
            },
            'time_wasting_patterns': {
                'action': 'QUALIFY_OR_DISENGAGE',
                'message': 'Time-wasting patterns detected - implement qualification',
                'priority': 'MEDIUM'
            },
            'competitor_intelligence': {
                'action': 'LIMIT_INFORMATION_SHARING',
                'message': 'Potential competitor intelligence gathering - limit information',
                'priority': 'MEDIUM'
            }
        }
        
        base_recommendation = category_recommendations.get(category)
        if not base_recommendation:
            return None
        
        return {
            **base_recommendation,
            'detected_patterns': [flag['matched_pattern'] for flag in flags],
            'flag_count': len(flags)
        }
```

## 7. Conversion Probability Scoring

### 7.1 ML-Based Conversion Prediction
```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

class ConversionProbabilityScorer:
    def __init__(self):
        # Model ensemble for robust predictions
        self.models = {
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # Feature importance weights
        self.feature_weights = {
            'engagement_score': 0.25,
            'intent_strength': 0.20,
            'behavioral_pattern': 0.15,
            'progression_velocity': 0.15,
            'time_investment': 0.10,
            'qualification_score': 0.10,
            'temporal_factors': 0.05
        }
    
    def extract_conversion_features(self, conversation_data: Dict) -> Dict:
        """Extract features for conversion probability prediction"""
        
        features = {}
        
        # 1. Engagement Features
        engagement = conversation_data.get('engagement_analysis', {})
        features.update({
            'overall_engagement_score': engagement.get('overall_score', 0),
            'attention_score': engagement.get('dimension_scores', {}).get('attention', 0),
            'interest_score': engagement.get('dimension_scores', {}).get('interest', 0),
            'progression_score': engagement.get('dimension_scores', {}).get('progression', 0),
            'emotional_score': engagement.get('dimension_scores', {}).get('emotional', 0)
        })
        
        # 2. Intent Features
        intent = conversation_data.get('intent_analysis', {})
        features.update({
            'purchase_intent_probability': intent.get('purchase_intent_probability', 0),
            'intent_consistency': intent.get('intent_consistency', 0),
            'intent_progression': intent.get('progression_direction_score', 0),
            'high_priority_intent_ratio': intent.get('high_priority_intent_ratio', 0)
        })
        
        # 3. Behavioral Features
        behavior = conversation_data.get('behavioral_analysis', {})
        features.update({
            'behavioral_pattern_match': behavior.get('primary_pattern_confidence', 0),
            'response_consistency': behavior.get('response_consistency', 0),
            'commitment_signals': behavior.get('commitment_score', 0),
            'technical_engagement': behavior.get('technical_language_ratio', 0)
        })
        
        # 4. Conversation Quality Features
        quality = conversation_data.get('quality_analysis', {})
        features.update({
            'conversation_quality': quality.get('overall_quality_score', 0),
            'goal_alignment': quality.get('dimension_scores', {}).get('goal_alignment', 0),
            'coherence_score': quality.get('dimension_scores', {}).get('coherence', 0)
        })
        
        # 5. Temporal Features
        temporal = conversation_data.get('temporal_analysis', {})
        features.update({
            'session_duration': temporal.get('session_duration', 0),
            'message_frequency': temporal.get('message_frequency', 0),
            'response_speed': temporal.get('avg_response_speed', 0),
            'conversation_recency': temporal.get('recency_score', 0)
        })
        
        # 6. Risk Features (inverted)
        risk = conversation_data.get('risk_analysis', {})
        features.update({
            'time_waster_probability': 1.0 - risk.get('time_waster_probability', 0),
            'red_flag_score': 1.0 - risk.get('overall_risk_score', 0),
            'spam_likelihood': 1.0 - risk.get('spam_probability', 0)
        })
        
        # 7. Derived Features
        features.update(self._calculate_derived_features(features))
        
        return features
    
    def _calculate_derived_features(self, base_features: Dict) -> Dict:
        """Calculate derived features from base features"""
        derived = {}
        
        # Engagement momentum (engagement × intent)
        derived['engagement_momentum'] = (
            base_features.get('overall_engagement_score', 0) * 
            base_features.get('purchase_intent_probability', 0)
        )
        
        # Quality-weighted engagement
        derived['quality_weighted_engagement'] = (
            base_features.get('overall_engagement_score', 0) * 
            base_features.get('conversation_quality', 0)
        )
        
        # Behavioral consistency score
        derived['behavioral_consistency'] = (
            base_features.get('behavioral_pattern_match', 0) * 
            base_features.get('response_consistency', 0)
        )
        
        # Progression efficiency (progression / time)
        session_duration = base_features.get('session_duration', 1)
        progression = base_features.get('progression_score', 0)
        derived['progression_efficiency'] = progression / max(1, session_duration / 60)  # per hour
        
        # Intent-behavior alignment
        derived['intent_behavior_alignment'] = min(
            base_features.get('purchase_intent_probability', 0),
            base_features.get('behavioral_pattern_match', 0)
        )
        
        return derived
    
    def train_conversion_models(self, training_data: List[Dict], conversion_labels: List[int]) -> Dict:
        """Train conversion probability models"""
        
        if len(training_data) < 50:
            return {'status': 'INSUFFICIENT_DATA', 'required_minimum': 50}
        
        # Extract features
        feature_matrix = []
        for data in training_data:
            features = self.extract_conversion_features(data)
            feature_vector = list(features.values())
            feature_matrix.append(feature_vector)
        
        X = np.array(feature_matrix)
        y = np.array(conversion_labels)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        model_performance = {}
        for model_name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
            
            model_performance[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': auc_score,
                'feature_importance': self._get_feature_importance(model, model_name)
            }
        
        self.is_trained = True
        
        return {
            'status': 'TRAINED',
            'model_performance': model_performance,
            'best_model': max(model_performance.items(), key=lambda x: x[1]['cv_mean'])[0],
            'training_samples': len(training_data),
            'feature_count': X.shape[1]
        }
    
    def _get_feature_importance(self, model, model_name: str) -> Dict:
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            return {'type': 'tree_based', 'importances': model.feature_importances_.tolist()}
        elif hasattr(model, 'coef_') and model_name == 'logistic_regression':
            return {'type': 'linear', 'coefficients': model.coef_[0].tolist()}
        else:
            return {'type': 'not_available', 'importances': []}
    
    def predict_conversion_probability(self, conversation_data: Dict) -> Dict:
        """Predict conversion probability for a conversation"""
        
        if not self.is_trained:
            return {'status': 'NOT_TRAINED'}
        
        # Extract features
        features = self.extract_conversion_features(conversation_data)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Handle missing values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # Get predictions from all models
        model_predictions = {}
        model_probabilities = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(feature_vector_scaled)[0]
            model_predictions[model_name] = pred
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(feature_vector_scaled)[0][1]
                model_probabilities[model_name] = prob
            else:
                model_probabilities[model_name] = float(pred)
        
        # Ensemble prediction (weighted average)
        ensemble_probability = np.mean(list(model_probabilities.values()))
        
        # Calculate prediction confidence
        prob_std = np.std(list(model_probabilities.values()))
        confidence = max(0, 1 - prob_std * 2)  # Higher std = lower confidence
        
        # Determine conversion likelihood category
        likelihood = self._categorize_conversion_likelihood(ensemble_probability)
        
        # Generate insights
        insights = self._generate_conversion_insights(features, ensemble_probability)
        
        return {
            'status': 'PREDICTED',
            'conversion_probability': ensemble_probability,
            'confidence': confidence,
            'likelihood_category': likelihood,
            'model_predictions': model_predictions,
            'model_probabilities': model_probabilities,
            'key_features': self._identify_key_features(features),
            'insights': insights,
            'recommendations': self._generate_conversion_recommendations(ensemble_probability, features)
        }
    
    def _categorize_conversion_likelihood(self, probability: float) -> str:
        """Categorize conversion probability into likelihood levels"""
        if probability >= 0.8:
            return 'VERY_HIGH'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _identify_key_features(self, features: Dict, top_n: int = 5) -> List[Dict]:
        """Identify the most important features for this prediction"""
        # Sort features by value (higher values generally indicate stronger signals)
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        
        key_features = []
        for feature_name, value in sorted_features[:top_n]:
            importance = self._get_feature_business_impact(feature_name)
            key_features.append({
                'feature': feature_name,
                'value': value,
                'business_impact': importance
            })
        
        return key_features
    
    def _get_feature_business_impact(self, feature_name: str) -> str:
        """Get business impact description for feature"""
        impact_mapping = {
            'overall_engagement_score': 'User engagement level directly correlates with conversion',
            'purchase_intent_probability': 'Strong purchase intent is the best conversion predictor',
            'behavioral_pattern_match': 'Matching high-value behavioral patterns increases conversion',
            'progression_score': 'Forward progression through sales funnel indicates readiness',
            'commitment_signals': 'Commitment language strongly predicts conversion',
            'quality_weighted_engagement': 'High-quality engaged conversations convert better',
            'engagement_momentum': 'Combined engagement and intent creates conversion momentum'
        }
        
        return impact_mapping.get(feature_name, 'Contributing factor to conversion probability')
    
    def _generate_conversion_insights(self, features: Dict, probability: float) -> List[str]:
        """Generate insights about conversion probability"""
        insights = []
        
        # High-level insight
        if probability >= 0.7:
            insights.append(f"High conversion probability ({probability:.1%}) - strong signals detected")
        elif probability >= 0.4:
            insights.append(f"Moderate conversion probability ({probability:.1%}) - mixed signals")
        else:
            insights.append(f"Low conversion probability ({probability:.1%}) - weak signals")
        
        # Feature-specific insights
        if features.get('overall_engagement_score', 0) > 0.7:
            insights.append("Strong engagement detected - user is actively participating")
        
        if features.get('purchase_intent_probability', 0) > 0.6:
            insights.append("Clear purchase intent signals - ready for sales advancement")
        
        if features.get('behavioral_pattern_match', 0) > 0.6:
            insights.append("User matches high-value behavioral patterns")
        
        if features.get('time_waster_probability', 0) < 0.3:
            insights.append("Low time-waster probability - genuine prospect")
        
        if features.get('progression_efficiency', 0) > 0.5:
            insights.append("Efficient progression through conversation - decisive user")
        
        return insights
    
    def _generate_conversion_recommendations(self, probability: float, features: Dict) -> List[Dict]:
        """Generate recommendations based on conversion probability"""
        recommendations = []
        
        if probability >= 0.7:
            # High probability - accelerate
            recommendations.append({
                'priority': 'HIGH',
                'action': 'ACCELERATE_CONVERSION',
                'message': 'High conversion probability - focus on closing',
                'tactics': ['Present pricing', 'Schedule demo', 'Address final objections', 'Create urgency']
            })
        
        elif probability >= 0.4:
            # Medium probability - nurture
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'NURTURE_PROSPECT',
                'message': 'Medium conversion probability - continue building value',
                'tactics': ['Provide case studies', 'Address concerns', 'Build relationship', 'Educate on benefits']
            })
        
        else:
            # Low probability - qualify or disengage
            recommendations.append({
                'priority': 'LOW',
                'action': 'QUALIFY_OR_DISENGAGE',
                'message': 'Low conversion probability - qualify further or consider disengaging',
                'tactics': ['Verify budget', 'Confirm timeline', 'Check decision authority', 'Set expectations']
            })
        
        # Feature-specific recommendations
        if features.get('overall_engagement_score', 0) < 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'IMPROVE_ENGAGEMENT',
                'message': 'Low engagement detected - focus on re-engaging prospect',
                'tactics': ['Ask engaging questions', 'Provide valuable content', 'Change conversation approach']
            })
        
        if features.get('progression_score', 0) < 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'ADVANCE_CONVERSATION',
                'message': 'Stalled conversation progression - advance to next stage',
                'tactics': ['Ask qualifying questions', 'Present next steps', 'Schedule follow-up']
            })
        
        return recommendations
```

This comprehensive advanced risk assessment system provides:

1. **Behavioral Pattern Analysis**: ML-driven clustering and pattern recognition to identify user archetypes and predict behavior
2. **Red Flag Detection**: Multi-layer detection system for fraud, spam, abuse, time-wasting, and competitive intelligence gathering
3. **Conversion Probability Scoring**: Ensemble ML models predicting conversion likelihood with actionable insights

The systems integrate seamlessly with existing conversation monitoring and provide real-time risk assessment and optimization recommendations for sales automation and customer service platforms.

Next, I'll complete the remaining systems (Resource Allocation Optimization, Human Intervention Triggers, and A/B Testing) in a follow-up file.