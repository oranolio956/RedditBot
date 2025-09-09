# Production ML Pipeline for Personality Management
## Continuous Learning and Deployment Architecture

## 6. Vector Embeddings for Personality Traits

### 6.1 Multi-Modal Embedding System

```python
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import faiss
from typing import Dict, List, Tuple, Optional
import pickle
import asyncio
from dataclasses import dataclass

@dataclass
class PersonalityVector:
    """Structured personality vector representation"""
    user_id: str
    conversation_id: str
    timestamp: int
    text_embedding: np.ndarray
    trait_embedding: np.ndarray
    context_embedding: np.ndarray
    combined_embedding: np.ndarray
    metadata: Dict

class MultiModalPersonalityEmbedder:
    """Multi-modal embedding system for personality representations"""
    
    def __init__(self, 
                 text_model: str = "all-MiniLM-L6-v2",
                 trait_dim: int = 128,
                 context_dim: int = 256):
        
        # Text embedding model
        self.text_encoder = SentenceTransformer(text_model)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        # Personality trait encoder
        self.trait_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 10 personality traits
            nn.ReLU(),
            nn.Linear(64, trait_dim),
            nn.Tanh()
        )
        
        # Context encoder for conversation metadata
        self.context_encoder = nn.Sequential(
            nn.Linear(50, 128),  # Context features
            nn.ReLU(),
            nn.Linear(128, context_dim),
            nn.Tanh()
        )
        
        # Fusion network
        total_dim = self.text_dim + trait_dim + context_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        
        # Dimension parameters
        self.trait_dim = trait_dim
        self.context_dim = context_dim
        self.final_dim = 256
        
        # FAISS index for efficient similarity search
        self.index = None
        self.personality_database = []
        
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text using sentence transformer"""
        return self.text_encoder.encode(texts, convert_to_numpy=True)
    
    def encode_traits(self, trait_vectors: np.ndarray) -> np.ndarray:
        """Encode personality traits"""
        with torch.no_grad():
            trait_tensor = torch.FloatTensor(trait_vectors)
            encoded = self.trait_encoder(trait_tensor)
            return encoded.numpy()
    
    def encode_context(self, context_features: np.ndarray) -> np.ndarray:
        """Encode conversation context"""
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context_features)
            encoded = self.context_encoder(context_tensor)
            return encoded.numpy()
    
    def create_combined_embedding(self, 
                                text: str,
                                personality_traits: Dict[str, float],
                                context_features: Dict) -> PersonalityVector:
        """Create combined multi-modal embedding"""
        
        # Text embedding
        text_emb = self.encode_text([text])[0]
        
        # Trait embedding
        trait_vector = np.array([
            personality_traits.get('extraversion', 0.5),
            personality_traits.get('agreeableness', 0.5),
            personality_traits.get('conscientiousness', 0.5),
            personality_traits.get('neuroticism', 0.5),
            personality_traits.get('openness', 0.5),
            personality_traits.get('humor', 0.5),
            personality_traits.get('empathy', 0.5),
            personality_traits.get('formality', 0.5),
            personality_traits.get('enthusiasm', 0.5),
            personality_traits.get('supportiveness', 0.5)
        ]).reshape(1, -1)
        trait_emb = self.encode_traits(trait_vector)[0]
        
        # Context embedding
        context_vector = self.extract_context_features(context_features).reshape(1, -1)
        context_emb = self.encode_context(context_vector)[0]
        
        # Combine embeddings
        combined_input = np.concatenate([text_emb, trait_emb, context_emb])
        
        with torch.no_grad():
            combined_tensor = torch.FloatTensor(combined_input).unsqueeze(0)
            final_emb = self.fusion_network(combined_tensor)[0].numpy()
        
        return PersonalityVector(
            user_id=context_features.get('user_id', ''),
            conversation_id=context_features.get('conversation_id', ''),
            timestamp=context_features.get('timestamp', 0),
            text_embedding=text_emb,
            trait_embedding=trait_emb,
            context_embedding=context_emb,
            combined_embedding=final_emb,
            metadata=context_features
        )
    
    def extract_context_features(self, context: Dict) -> np.ndarray:
        """Extract numerical features from conversation context"""
        features = [
            context.get('message_length', 0) / 1000.0,  # Normalized
            context.get('response_time', 0) / 60.0,
            context.get('sentiment_score', 0),
            context.get('emoji_count', 0) / 10.0,
            context.get('question_count', 0) / 5.0,
            context.get('conversation_turn', 0) / 20.0,
            context.get('time_of_day', 12) / 24.0,
            context.get('day_of_week', 3) / 7.0,
            context.get('engagement_score', 0.5),
            context.get('topic_coherence', 0.5),
        ]
        
        # Pad to 50 features with zeros
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def build_faiss_index(self, personality_vectors: List[PersonalityVector]):
        """Build FAISS index for efficient similarity search"""
        # Extract embeddings
        embeddings = np.array([pv.combined_embedding for pv in personality_vectors])
        
        # Build index
        self.index = faiss.IndexFlatIP(self.final_dim)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Store personality database
        self.personality_database = personality_vectors
        
    def find_similar_personalities(self, 
                                 query_vector: PersonalityVector,
                                 k: int = 5) -> List[Tuple[PersonalityVector, float]]:
        """Find k most similar personality vectors"""
        if self.index is None:
            return []
        
        # Normalize query
        query_emb = query_vector.combined_embedding.reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, k)
        
        # Return results with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.personality_database):
                results.append((self.personality_database[idx], float(score)))
        
        return results
    
    async def update_embedding_async(self, personality_vector: PersonalityVector):
        """Asynchronously update the embedding index"""
        # Add to database
        self.personality_database.append(personality_vector)
        
        # Add to index
        if self.index is not None:
            emb = personality_vector.combined_embedding.reshape(1, -1)
            faiss.normalize_L2(emb)
            self.index.add(emb)

class PersonalityClusterAnalyzer:
    """Analyze personality patterns using clustering"""
    
    def __init__(self, embedder: MultiModalPersonalityEmbedder):
        self.embedder = embedder
        self.kmeans = None
        self.pca = None
        self.cluster_profiles = {}
        
    def analyze_personality_clusters(self, 
                                   personality_vectors: List[PersonalityVector],
                                   n_clusters: int = 10) -> Dict:
        """Analyze personality clusters in the data"""
        
        # Extract embeddings
        embeddings = np.array([pv.combined_embedding for pv in personality_vectors])
        
        # Dimensionality reduction for visualization
        self.pca = PCA(n_components=50)
        reduced_embeddings = self.pca.fit_transform(embeddings)
        
        # Clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(reduced_embeddings)
        
        # Analyze each cluster
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_vectors = [pv for i, pv in enumerate(personality_vectors) if cluster_mask[i]]
            
            # Compute cluster statistics
            analysis = self.analyze_cluster(cluster_vectors, cluster_id)
            cluster_analysis[cluster_id] = analysis
            
        return {
            'cluster_analysis': cluster_analysis,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'total_vectors': len(personality_vectors)
        }
    
    def analyze_cluster(self, cluster_vectors: List[PersonalityVector], cluster_id: int) -> Dict:
        """Analyze a specific personality cluster"""
        if not cluster_vectors:
            return {}
        
        # Extract trait values
        trait_values = {
            'extraversion': [], 'agreeableness': [], 'conscientiousness': [],
            'neuroticism': [], 'openness': [], 'humor': [], 'empathy': [],
            'formality': [], 'enthusiasm': [], 'supportiveness': []
        }
        
        engagement_scores = []
        response_times = []
        message_lengths = []
        
        for pv in cluster_vectors:
            # Extract traits from metadata or embedding
            for trait in trait_values.keys():
                trait_values[trait].append(pv.metadata.get(trait, 0.5))
            
            engagement_scores.append(pv.metadata.get('engagement_score', 0.5))
            response_times.append(pv.metadata.get('response_time', 0))
            message_lengths.append(pv.metadata.get('message_length', 0))
        
        # Compute statistics
        analysis = {
            'cluster_id': cluster_id,
            'size': len(cluster_vectors),
            'trait_means': {trait: np.mean(values) for trait, values in trait_values.items()},
            'trait_stds': {trait: np.std(values) for trait, values in trait_values.items()},
            'avg_engagement': np.mean(engagement_scores),
            'avg_response_time': np.mean(response_times),
            'avg_message_length': np.mean(message_lengths),
            'dominant_traits': self.find_dominant_traits(trait_values),
            'personality_archetype': self.classify_archetype(trait_values)
        }
        
        return analysis
    
    def find_dominant_traits(self, trait_values: Dict[str, List[float]]) -> List[str]:
        """Find the most dominant personality traits in a cluster"""
        trait_means = {trait: np.mean(values) for trait, values in trait_values.items()}
        
        # Sort traits by mean value
        sorted_traits = sorted(trait_means.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3 traits
        return [trait for trait, _ in sorted_traits[:3]]
    
    def classify_archetype(self, trait_values: Dict[str, List[float]]) -> str:
        """Classify personality archetype based on dominant traits"""
        means = {trait: np.mean(values) for trait, values in trait_values.items()}
        
        # Define archetype rules
        if means['extraversion'] > 0.7 and means['enthusiasm'] > 0.7:
            return "Energetic Extrovert"
        elif means['empathy'] > 0.8 and means['supportiveness'] > 0.8:
            return "Supportive Helper"
        elif means['humor'] > 0.7 and means['openness'] > 0.6:
            return "Humorous Creative"
        elif means['conscientiousness'] > 0.8 and means['formality'] > 0.6:
            return "Professional Formal"
        elif means['agreeableness'] > 0.8 and means['neuroticism'] < 0.3:
            return "Calm Agreeable"
        else:
            return "Balanced Personality"
    
    def recommend_personality_adjustments(self, 
                                        user_vector: PersonalityVector,
                                        target_engagement: float = 0.8) -> Dict[str, float]:
        """Recommend personality adjustments to improve engagement"""
        if self.kmeans is None:
            return {}
        
        # Find user's cluster
        user_emb = self.pca.transform([user_vector.combined_embedding])
        user_cluster = self.kmeans.predict(user_emb)[0]
        
        # Get high-engagement users from same cluster
        cluster_vectors = [
            pv for pv in self.embedder.personality_database
            if self.kmeans.predict(self.pca.transform([pv.combined_embedding]))[0] == user_cluster
            and pv.metadata.get('engagement_score', 0) > target_engagement
        ]
        
        if not cluster_vectors:
            return {}
        
        # Calculate average traits of high-engagement users
        target_traits = {}
        trait_names = ['extraversion', 'agreeableness', 'conscientiousness',
                      'neuroticism', 'openness', 'humor', 'empathy',
                      'formality', 'enthusiasm', 'supportiveness']
        
        for trait in trait_names:
            values = [pv.metadata.get(trait, 0.5) for pv in cluster_vectors]
            target_traits[trait] = np.mean(values)
        
        # Calculate adjustments
        current_traits = {
            trait: user_vector.metadata.get(trait, 0.5) 
            for trait in trait_names
        }
        
        adjustments = {
            trait: target_traits[trait] - current_traits[trait]
            for trait in trait_names
        }
        
        # Limit adjustment magnitude
        max_adjustment = 0.2
        for trait in adjustments:
            adjustments[trait] = max(-max_adjustment, 
                                   min(max_adjustment, adjustments[trait]))
        
        return adjustments

# Semantic Personality Search
class SemanticPersonalitySearch:
    """Semantic search for personality patterns"""
    
    def __init__(self, embedder: MultiModalPersonalityEmbedder):
        self.embedder = embedder
        self.trait_descriptions = {
            'extraversion': 'outgoing energetic social talkative',
            'agreeableness': 'cooperative trusting helpful kind',
            'conscientiousness': 'organized responsible reliable disciplined',
            'neuroticism': 'anxious worried stressed emotional',
            'openness': 'creative imaginative curious artistic',
            'humor': 'funny witty playful jokes laughing',
            'empathy': 'understanding caring compassionate supportive',
            'formality': 'professional proper respectful polite',
            'enthusiasm': 'excited passionate energetic motivated',
            'supportiveness': 'encouraging helpful caring nurturing'
        }
    
    def search_by_description(self, 
                            description: str,
                            top_k: int = 10) -> List[Tuple[PersonalityVector, float]]:
        """Search for personalities matching a text description"""
        
        # Encode the search query
        query_embedding = self.embedder.encode_text([description])[0]
        
        # Search through personality database
        similarities = []
        for pv in self.embedder.personality_database:
            # Compare with text embedding of the personality vector
            similarity = np.dot(query_embedding, pv.text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(pv.text_embedding)
            )
            similarities.append((pv, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def search_by_traits(self, 
                        target_traits: Dict[str, float],
                        top_k: int = 10) -> List[Tuple[PersonalityVector, float]]:
        """Search for personalities with similar traits"""
        
        # Create target trait vector
        target_vector = np.array([
            target_traits.get(trait, 0.5) 
            for trait in ['extraversion', 'agreeableness', 'conscientiousness',
                         'neuroticism', 'openness', 'humor', 'empathy',
                         'formality', 'enthusiasm', 'supportiveness']
        ])
        
        # Search through personality database
        similarities = []
        for pv in self.embedder.personality_database:
            # Extract traits from metadata
            pv_traits = np.array([
                pv.metadata.get(trait, 0.5)
                for trait in ['extraversion', 'agreeableness', 'conscientiousness',
                             'neuroticism', 'openness', 'humor', 'empathy',
                             'formality', 'enthusiasm', 'supportiveness']
            ])
            
            # Compute cosine similarity
            similarity = np.dot(target_vector, pv_traits) / (
                np.linalg.norm(target_vector) * np.linalg.norm(pv_traits)
            )
            similarities.append((pv, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_complementary_personalities(self, 
                                       base_personality: PersonalityVector,
                                       top_k: int = 5) -> List[Tuple[PersonalityVector, float]]:
        """Find personalities that complement the base personality"""
        
        # Define complementary trait mapping
        complementary_traits = {
            'extraversion': 'introversion',  # Opposite
            'agreeableness': 'agreeableness',  # Similar
            'conscientiousness': 'conscientiousness',  # Similar
            'neuroticism': 'stability',  # Opposite
            'openness': 'openness',  # Similar
            'humor': 'humor',  # Similar
            'empathy': 'empathy',  # Similar
            'formality': 'casualness',  # Opposite
            'enthusiasm': 'calm',  # Opposite
            'supportiveness': 'supportiveness'  # Similar
        }
        
        # Calculate complementary trait targets
        base_traits = {
            trait: base_personality.metadata.get(trait, 0.5)
            for trait in complementary_traits.keys()
        }
        
        target_traits = {}
        for trait, complement in complementary_traits.items():
            if complement.endswith('ness') or complement == trait:
                # Want similar trait
                target_traits[trait] = base_traits[trait]
            else:
                # Want opposite trait
                target_traits[trait] = 1.0 - base_traits[trait]
        
        # Search for complementary personalities
        return self.search_by_traits(target_traits, top_k)

# Real-time Embedding Updates
class RealTimeEmbeddingManager:
    """Manage real-time updates to personality embeddings"""
    
    def __init__(self, embedder: MultiModalPersonalityEmbedder, redis_client):
        self.embedder = embedder
        self.redis = redis_client
        self.update_queue = asyncio.Queue()
        self.batch_size = 32
        self.update_interval = 60  # seconds
        
    async def start_update_worker(self):
        """Start background worker for processing embedding updates"""
        while True:
            try:
                # Collect batch of updates
                updates = []
                for _ in range(self.batch_size):
                    try:
                        update = await asyncio.wait_for(
                            self.update_queue.get(), 
                            timeout=self.update_interval
                        )
                        updates.append(update)
                    except asyncio.TimeoutError:
                        break
                
                if updates:
                    await self.process_embedding_updates(updates)
                
            except Exception as e:
                print(f"Error in embedding update worker: {e}")
                await asyncio.sleep(5)
    
    async def queue_embedding_update(self, 
                                   user_id: str,
                                   conversation_data: Dict):
        """Queue an embedding update"""
        update_data = {
            'user_id': user_id,
            'conversation_data': conversation_data,
            'timestamp': int(time.time())
        }
        await self.update_queue.put(update_data)
    
    async def process_embedding_updates(self, updates: List[Dict]):
        """Process a batch of embedding updates"""
        new_vectors = []
        
        for update in updates:
            try:
                # Create personality vector
                personality_vector = self.embedder.create_combined_embedding(
                    text=update['conversation_data']['message'],
                    personality_traits=update['conversation_data']['personality_traits'],
                    context_features=update['conversation_data']
                )
                
                new_vectors.append(personality_vector)
                
            except Exception as e:
                print(f"Error processing embedding update: {e}")
        
        # Batch update FAISS index
        if new_vectors:
            await self.batch_update_index(new_vectors)
    
    async def batch_update_index(self, personality_vectors: List[PersonalityVector]):
        """Update FAISS index with new vectors"""
        # Extract embeddings
        embeddings = np.array([pv.combined_embedding for pv in personality_vectors])
        
        # Normalize and add to index
        faiss.normalize_L2(embeddings)
        self.embedder.index.add(embeddings)
        
        # Update database
        self.embedder.personality_database.extend(personality_vectors)
        
        # Cache vectors in Redis for persistence
        for pv in personality_vectors:
            await self.cache_personality_vector(pv)
    
    async def cache_personality_vector(self, pv: PersonalityVector):
        """Cache personality vector in Redis"""
        cache_key = f"personality_vector:{pv.user_id}:{pv.timestamp}"
        
        # Serialize vector
        vector_data = {
            'user_id': pv.user_id,
            'conversation_id': pv.conversation_id,
            'timestamp': pv.timestamp,
            'combined_embedding': pv.combined_embedding.tolist(),
            'metadata': pv.metadata
        }
        
        await self.redis.setex(
            cache_key, 
            86400,  # 24 hour TTL
            pickle.dumps(vector_data)
        )
    
    async def load_cached_vectors(self) -> List[PersonalityVector]:
        """Load cached personality vectors from Redis"""
        pattern = "personality_vector:*"
        keys = await self.redis.keys(pattern)
        
        cached_vectors = []
        for key in keys:
            try:
                data = await self.redis.get(key)
                if data:
                    vector_data = pickle.loads(data)
                    
                    # Reconstruct PersonalityVector
                    pv = PersonalityVector(
                        user_id=vector_data['user_id'],
                        conversation_id=vector_data['conversation_id'],
                        timestamp=vector_data['timestamp'],
                        text_embedding=np.zeros(384),  # Placeholder
                        trait_embedding=np.zeros(128),  # Placeholder
                        context_embedding=np.zeros(256),  # Placeholder
                        combined_embedding=np.array(vector_data['combined_embedding']),
                        metadata=vector_data['metadata']
                    )
                    
                    cached_vectors.append(pv)
                    
            except Exception as e:
                print(f"Error loading cached vector {key}: {e}")
        
        return cached_vectors
```

## 7. Continuous Learning from Conversation Feedback

### 7.1 Online Learning System

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import asyncio
from collections import deque
import time
import json
from dataclasses import dataclass, asdict

@dataclass
class FeedbackRecord:
    """Structured feedback record"""
    user_id: str
    conversation_id: str
    timestamp: int
    personality_before: Dict[str, float]
    personality_after: Dict[str, float]
    response_text: str
    feedback_type: str  # 'explicit', 'implicit', 'behavioral'
    feedback_score: float  # -1 to 1
    context_features: Dict
    model_version: str

class OnlinePersonalityLearner:
    """Online learning system for continuous personality adaptation"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 learning_rate: float = 0.001,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.base_model = base_model
        self.optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
        
        # Experience replay for stability
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Online learning parameters
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Feedback processing
        self.feedback_buffer = asyncio.Queue(maxsize=1000)
        self.last_update = time.time()
        self.update_frequency = 300  # 5 minutes
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adaptation_metrics = {}
        
    async def process_feedback(self, feedback: FeedbackRecord):
        """Process new feedback for online learning"""
        try:
            # Add to feedback buffer
            await self.feedback_buffer.put(feedback)
            
            # Store in memory for batch learning
            self.memory.append(feedback)
            
            # Immediate adaptation for high-impact feedback
            if abs(feedback.feedback_score) > 0.8:  # Strong feedback
                await self.immediate_adaptation(feedback)
                
        except Exception as e:
            print(f"Error processing feedback: {e}")
    
    async def immediate_adaptation(self, feedback: FeedbackRecord):
        """Immediate model adaptation for strong feedback"""
        self.base_model.train()
        
        # Convert feedback to training sample
        features = self.extract_features(feedback)
        target = self.create_adaptation_target(feedback)
        
        # Single gradient step
        self.optimizer.zero_grad()
        
        prediction = self.base_model(features)
        loss = nn.MSELoss()(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Log adaptation
        self.log_adaptation(feedback, loss.item())
    
    async def batch_learning_update(self):
        """Periodic batch learning from accumulated feedback"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch_feedback = list(self.memory)[-self.batch_size:]
        
        self.base_model.train()
        total_loss = 0
        
        for feedback in batch_feedback:
            features = self.extract_features(feedback)
            target = self.create_adaptation_target(feedback)
            
            prediction = self.base_model(features)
            loss = nn.MSELoss()(prediction, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(batch_feedback)
        print(f"Batch learning update: avg_loss = {avg_loss:.4f}")
        
        # Update performance tracking
        self.performance_history.append({
            'timestamp': time.time(),
            'avg_loss': avg_loss,
            'batch_size': len(batch_feedback)
        })
    
    def extract_features(self, feedback: FeedbackRecord) -> torch.Tensor:
        """Extract features from feedback record"""
        features = []
        
        # Personality features (before)
        trait_names = ['extraversion', 'agreeableness', 'conscientiousness',
                      'neuroticism', 'openness', 'humor', 'empathy',
                      'formality', 'enthusiasm', 'supportiveness']
        
        for trait in trait_names:
            features.append(feedback.personality_before.get(trait, 0.5))
        
        # Context features
        context = feedback.context_features
        features.extend([
            context.get('message_length', 0) / 1000.0,
            context.get('response_time', 0) / 60.0,
            context.get('sentiment_score', 0),
            context.get('engagement_score', 0.5),
            context.get('conversation_turn', 0) / 20.0
        ])
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def create_adaptation_target(self, feedback: FeedbackRecord) -> torch.Tensor:
        """Create target for adaptation based on feedback"""
        # Start with current personality
        target_traits = feedback.personality_before.copy()
        
        # Adjust based on feedback score
        adjustment_factor = feedback.feedback_score * 0.1  # Small adjustments
        
        # Determine which traits to adjust based on feedback type
        if feedback.feedback_type == 'engagement':
            # Boost engagement-related traits
            target_traits['enthusiasm'] += adjustment_factor
            target_traits['humor'] += adjustment_factor * 0.5
            
        elif feedback.feedback_type == 'helpfulness':
            # Boost helpfulness-related traits
            target_traits['supportiveness'] += adjustment_factor
            target_traits['empathy'] += adjustment_factor
            
        elif feedback.feedback_type == 'appropriateness':
            # Adjust formality and empathy
            target_traits['formality'] += adjustment_factor
            target_traits['neuroticism'] -= adjustment_factor * 0.5
        
        # Clamp values between 0 and 1
        for trait in target_traits:
            target_traits[trait] = max(0.0, min(1.0, target_traits[trait]))
        
        # Convert to tensor
        trait_names = ['extraversion', 'agreeableness', 'conscientiousness',
                      'neuroticism', 'openness', 'humor', 'empathy',
                      'formality', 'enthusiasm', 'supportiveness']
        
        target_values = [target_traits[trait] for trait in trait_names]
        return torch.FloatTensor(target_values).unsqueeze(0)
    
    def log_adaptation(self, feedback: FeedbackRecord, loss: float):
        """Log adaptation event"""
        adaptation_log = {
            'timestamp': time.time(),
            'user_id': feedback.user_id,
            'feedback_type': feedback.feedback_type,
            'feedback_score': feedback.feedback_score,
            'loss': loss,
            'personality_change': self.calculate_personality_change(feedback)
        }
        
        # Store in adaptation metrics
        if feedback.feedback_type not in self.adaptation_metrics:
            self.adaptation_metrics[feedback.feedback_type] = []
        
        self.adaptation_metrics[feedback.feedback_type].append(adaptation_log)
        
        # Keep only recent adaptations
        if len(self.adaptation_metrics[feedback.feedback_type]) > 100:
            self.adaptation_metrics[feedback.feedback_type] = \
                self.adaptation_metrics[feedback.feedback_type][-100:]
    
    def calculate_personality_change(self, feedback: FeedbackRecord) -> Dict[str, float]:
        """Calculate personality change magnitude"""
        changes = {}
        
        for trait in feedback.personality_before.keys():
            before = feedback.personality_before[trait]
            after = feedback.personality_after.get(trait, before)
            changes[trait] = after - before
        
        return changes
    
    async def start_continuous_learning(self):
        """Start continuous learning background process"""
        while True:
            try:
                # Process feedback batch
                if time.time() - self.last_update > self.update_frequency:
                    await self.batch_learning_update()
                    self.last_update = time.time()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in continuous learning: {e}")
                await asyncio.sleep(5)
    
    def get_adaptation_statistics(self) -> Dict:
        """Get statistics about model adaptations"""
        stats = {
            'total_feedback': len(self.memory),
            'adaptation_types': {},
            'recent_performance': []
        }
        
        # Adaptation type statistics
        for feedback_type, adaptations in self.adaptation_metrics.items():
            recent_adaptations = [a for a in adaptations 
                                if time.time() - a['timestamp'] < 3600]  # Last hour
            
            if recent_adaptations:
                stats['adaptation_types'][feedback_type] = {
                    'count': len(recent_adaptations),
                    'avg_score': np.mean([a['feedback_score'] for a in recent_adaptations]),
                    'avg_loss': np.mean([a['loss'] for a in recent_adaptations])
                }
        
        # Recent performance
        recent_performance = [p for p in self.performance_history 
                            if time.time() - p['timestamp'] < 3600]
        
        if recent_performance:
            stats['recent_performance'] = {
                'avg_loss': np.mean([p['avg_loss'] for p in recent_performance]),
                'updates_count': len(recent_performance)
            }
        
        return stats

class FeedbackCollector:
    """Collect and process user feedback"""
    
    def __init__(self, online_learner: OnlinePersonalityLearner):
        self.online_learner = online_learner
        self.implicit_feedback_extractors = {
            'response_time': self.extract_response_time_feedback,
            'engagement': self.extract_engagement_feedback,
            'conversation_length': self.extract_conversation_length_feedback,
            'sentiment': self.extract_sentiment_feedback
        }
    
    async def collect_explicit_feedback(self, 
                                      user_id: str,
                                      conversation_id: str,
                                      rating: int,  # 1-5 scale
                                      comment: Optional[str] = None):
        """Collect explicit user feedback"""
        
        # Convert rating to feedback score (-1 to 1)
        feedback_score = (rating - 3) / 2.0
        
        feedback = FeedbackRecord(
            user_id=user_id,
            conversation_id=conversation_id,
            timestamp=int(time.time()),
            personality_before={},  # To be filled from conversation context
            personality_after={},   # To be filled from adaptation
            response_text=comment or "",
            feedback_type='explicit',
            feedback_score=feedback_score,
            context_features={'rating': rating, 'comment': comment},
            model_version='v1.0'
        )
        
        await self.online_learner.process_feedback(feedback)
    
    async def collect_implicit_feedback(self, conversation_data: Dict):
        """Extract and collect implicit feedback from conversation"""
        
        for feedback_type, extractor in self.implicit_feedback_extractors.items():
            try:
                feedback = await extractor(conversation_data)
                if feedback:
                    await self.online_learner.process_feedback(feedback)
            except Exception as e:
                print(f"Error extracting {feedback_type} feedback: {e}")
    
    async def extract_response_time_feedback(self, conversation_data: Dict) -> Optional[FeedbackRecord]:
        """Extract feedback from user response time"""
        response_time = conversation_data.get('response_time', 0)
        
        if response_time == 0:
            return None
        
        # Fast response (< 30s) = positive, slow (> 300s) = negative
        if response_time < 30:
            feedback_score = 0.3
        elif response_time > 300:
            feedback_score = -0.3
        else:
            # Neutral feedback for moderate response times
            return None
        
        return FeedbackRecord(
            user_id=conversation_data['user_id'],
            conversation_id=conversation_data['conversation_id'],
            timestamp=int(time.time()),
            personality_before=conversation_data.get('personality_before', {}),
            personality_after=conversation_data.get('personality_after', {}),
            response_text="",
            feedback_type='response_time',
            feedback_score=feedback_score,
            context_features={'response_time': response_time},
            model_version='v1.0'
        )
    
    async def extract_engagement_feedback(self, conversation_data: Dict) -> Optional[FeedbackRecord]:
        """Extract feedback from engagement metrics"""
        engagement_indicators = {
            'emoji_count': conversation_data.get('emoji_count', 0),
            'question_count': conversation_data.get('question_count', 0),
            'message_length': conversation_data.get('message_length', 0),
            'exclamation_count': conversation_data.get('exclamation_count', 0)
        }
        
        # Calculate engagement score
        engagement_score = 0
        engagement_score += min(engagement_indicators['emoji_count'] * 0.1, 0.3)
        engagement_score += min(engagement_indicators['question_count'] * 0.2, 0.4)
        engagement_score += min(engagement_indicators['message_length'] / 100, 0.2)
        engagement_score += min(engagement_indicators['exclamation_count'] * 0.1, 0.1)
        
        # Convert to feedback score (-0.5 to 0.5 range for implicit feedback)
        feedback_score = min(max(engagement_score - 0.3, -0.5), 0.5)
        
        if abs(feedback_score) < 0.1:  # Not significant enough
            return None
        
        return FeedbackRecord(
            user_id=conversation_data['user_id'],
            conversation_id=conversation_data['conversation_id'],
            timestamp=int(time.time()),
            personality_before=conversation_data.get('personality_before', {}),
            personality_after=conversation_data.get('personality_after', {}),
            response_text="",
            feedback_type='engagement',
            feedback_score=feedback_score,
            context_features=engagement_indicators,
            model_version='v1.0'
        )
    
    async def extract_conversation_length_feedback(self, conversation_data: Dict) -> Optional[FeedbackRecord]:
        """Extract feedback from conversation length"""
        conversation_turns = conversation_data.get('conversation_turns', 0)
        
        if conversation_turns < 3:
            return None  # Too short to be meaningful
        
        # Long conversations (>10 turns) = positive, short (3-5 turns) = slightly negative
        if conversation_turns >= 10:
            feedback_score = 0.4
        elif conversation_turns <= 5:
            feedback_score = -0.2
        else:
            feedback_score = 0.1  # Neutral positive for moderate length
        
        return FeedbackRecord(
            user_id=conversation_data['user_id'],
            conversation_id=conversation_data['conversation_id'],
            timestamp=int(time.time()),
            personality_before=conversation_data.get('personality_before', {}),
            personality_after=conversation_data.get('personality_after', {}),
            response_text="",
            feedback_type='conversation_length',
            feedback_score=feedback_score,
            context_features={'conversation_turns': conversation_turns},
            model_version='v1.0'
        )
    
    async def extract_sentiment_feedback(self, conversation_data: Dict) -> Optional[FeedbackRecord]:
        """Extract feedback from sentiment analysis"""
        sentiment_score = conversation_data.get('sentiment_score', 0)
        
        if abs(sentiment_score) < 0.2:  # Neutral sentiment
            return None
        
        # Use sentiment as direct feedback (scaled down for implicit feedback)
        feedback_score = sentiment_score * 0.3
        
        return FeedbackRecord(
            user_id=conversation_data['user_id'],
            conversation_id=conversation_data['conversation_id'],
            timestamp=int(time.time()),
            personality_before=conversation_data.get('personality_before', {}),
            personality_after=conversation_data.get('personality_after', {}),
            response_text="",
            feedback_type='sentiment',
            feedback_score=feedback_score,
            context_features={'sentiment_score': sentiment_score},
            model_version='v1.0'
        )

# Feedback Integration with Existing System
class ContinuousLearningIntegration:
    """Integration layer for continuous learning with the main personality system"""
    
    def __init__(self, 
                 personality_service,
                 online_learner: OnlinePersonalityLearner,
                 feedback_collector: FeedbackCollector):
        
        self.personality_service = personality_service
        self.online_learner = online_learner
        self.feedback_collector = feedback_collector
        
        # Learning configuration
        self.learning_enabled = True
        self.minimum_confidence = 0.6  # Minimum confidence to apply adaptations
        self.max_adaptation_rate = 0.1  # Maximum change per adaptation
    
    async def enhance_conversation_response(self, 
                                          user_id: str,
                                          message: str,
                                          current_personality: Dict[str, float]) -> Dict:
        """Enhance conversation response with continuous learning"""
        
        # Get base response from personality service
        base_response = await self.personality_service.generate_response(
            user_id, message, current_personality
        )
        
        # Apply online learning adaptations if enabled
        if self.learning_enabled:
            adapted_personality = await self.apply_learned_adaptations(
                user_id, current_personality, message
            )
            
            if adapted_personality != current_personality:
                # Generate response with adapted personality
                enhanced_response = await self.personality_service.generate_response(
                    user_id, message, adapted_personality
                )
                
                return {
                    'response': enhanced_response['response'],
                    'personality_before': current_personality,
                    'personality_after': adapted_personality,
                    'adaptation_applied': True,
                    'confidence': enhanced_response.get('confidence', 0.8)
                }
        
        return {
            'response': base_response['response'],
            'personality_before': current_personality,
            'personality_after': current_personality,
            'adaptation_applied': False,
            'confidence': base_response.get('confidence', 0.8)
        }
    
    async def apply_learned_adaptations(self, 
                                      user_id: str,
                                      current_personality: Dict[str, float],
                                      context: str) -> Dict[str, float]:
        """Apply learned personality adaptations"""
        
        try:
            # Extract features for the current context
            features = self.extract_context_features(user_id, context, current_personality)
            
            # Get adaptation suggestions from online learner
            with torch.no_grad():
                suggested_traits = self.online_learner.base_model(features)
                suggested_traits = suggested_traits.squeeze().numpy()
            
            # Apply adaptations with confidence thresholding
            adapted_personality = current_personality.copy()
            
            trait_names = ['extraversion', 'agreeableness', 'conscientiousness',
                          'neuroticism', 'openness', 'humor', 'empathy',
                          'formality', 'enthusiasm', 'supportiveness']
            
            for i, trait in enumerate(trait_names):
                if i < len(suggested_traits):
                    suggested_value = float(suggested_traits[i])
                    current_value = current_personality.get(trait, 0.5)
                    
                    # Calculate adaptation with rate limiting
                    adaptation = suggested_value - current_value
                    adaptation = max(-self.max_adaptation_rate, 
                                   min(self.max_adaptation_rate, adaptation))
                    
                    adapted_personality[trait] = max(0.0, min(1.0, 
                                                            current_value + adaptation))
            
            return adapted_personality
            
        except Exception as e:
            print(f"Error applying learned adaptations: {e}")
            return current_personality
    
    def extract_context_features(self, 
                               user_id: str,
                               context: str,
                               personality: Dict[str, float]) -> torch.Tensor:
        """Extract features for adaptation prediction"""
        
        features = []
        
        # Personality features
        trait_names = ['extraversion', 'agreeableness', 'conscientiousness',
                      'neuroticism', 'openness', 'humor', 'empathy',
                      'formality', 'enthusiasm', 'supportiveness']
        
        for trait in trait_names:
            features.append(personality.get(trait, 0.5))
        
        # Context features
        features.extend([
            len(context) / 1000.0,  # Message length
            context.count('?') / 5.0,  # Question count
            context.count('!') / 5.0,  # Exclamation count
            len([c for c in context if c in '😀😃😄😁😊😋']) / 10.0,  # Emoji count
            0.5  # Placeholder for time-based features
        ])
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    async def process_post_conversation_feedback(self, conversation_data: Dict):
        """Process feedback after conversation completion"""
        
        try:
            # Collect implicit feedback
            await self.feedback_collector.collect_implicit_feedback(conversation_data)
            
            # If explicit feedback available, process it
            if 'user_rating' in conversation_data:
                await self.feedback_collector.collect_explicit_feedback(
                    user_id=conversation_data['user_id'],
                    conversation_id=conversation_data['conversation_id'],
                    rating=conversation_data['user_rating'],
                    comment=conversation_data.get('user_comment')
                )
            
        except Exception as e:
            print(f"Error processing post-conversation feedback: {e}")
    
    async def get_learning_status(self) -> Dict:
        """Get current learning system status"""
        
        stats = self.online_learner.get_adaptation_statistics()
        
        return {
            'learning_enabled': self.learning_enabled,
            'total_feedback_records': stats['total_feedback'],
            'adaptation_types': stats['adaptation_types'],
            'recent_performance': stats.get('recent_performance', {}),
            'model_parameters': {
                'learning_rate': self.online_learner.learning_rate,
                'batch_size': self.online_learner.batch_size,
                'memory_size': len(self.online_learner.memory)
            }
        }
    
    def enable_learning(self):
        """Enable continuous learning"""
        self.learning_enabled = True
    
    def disable_learning(self):
        """Disable continuous learning"""
        self.learning_enabled = False
```

This production ML pipeline provides:

**Continuous Learning Features:**
- **Online Learning**: Real-time model adaptation from user feedback
- **Feedback Collection**: Both explicit ratings and implicit behavioral signals
- **Experience Replay**: Stable learning from accumulated conversation data
- **Adaptation Rate Limiting**: Prevents excessive personality changes

**Vector Embedding System:**
- **Multi-modal Embeddings**: Text, traits, and context combined
- **FAISS Integration**: Efficient similarity search for personality matching
- **Real-time Updates**: Streaming personality vector updates
- **Semantic Search**: Find personalities by description or trait similarity

**Integration Points:**
- **Redis Caching**: Fast access to personality vectors
- **Async Processing**: Non-blocking feedback collection
- **Batch Updates**: Efficient bulk processing
- **Performance Monitoring**: Learning effectiveness tracking

The system maintains a balance between responsiveness to user feedback and stability, ensuring personalities evolve naturally while avoiding erratic behavior changes.