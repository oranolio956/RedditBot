"""
ML Models and AI Engines Initialization Service

Handles initialization and cleanup of all ML models, transformers,
and AI engines used throughout the application.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModel, AutoTokenizer, pipeline,
    BertModel, BertTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np

from app.config import settings
from app.core.performance_utils import performance_monitor

logger = logging.getLogger(__name__)

class MLModelManager:
    """Centralized ML model management"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.engines: Dict[str, Any] = {}
        self.initialized = False
        
    async def initialize_transformers(self):
        """Initialize transformer models for AI features"""
        try:
            logger.info("Initializing transformer models...")
            
            # BERT for semantic understanding
            with performance_monitor("bert_initialization"):
                self.models['bert'] = BertModel.from_pretrained('bert-base-uncased')
                self.tokenizers['bert'] = BertTokenizer.from_pretrained('bert-base-uncased')
                logger.info("BERT model initialized")
            
            # Sentence transformers for semantic similarity
            with performance_monitor("sentence_transformer_initialization"):
                self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer initialized")
            
            # GPT-2 for text generation (consciousness, dreams)
            with performance_monitor("gpt2_initialization"):
                self.models['gpt2'] = GPT2LMHeadModel.from_pretrained('gpt2')
                self.tokenizers['gpt2'] = GPT2Tokenizer.from_pretrained('gpt2')
                self.tokenizers['gpt2'].pad_token = self.tokenizers['gpt2'].eos_token
                logger.info("GPT-2 model initialized")
            
            # Emotion classification pipeline
            with performance_monitor("emotion_pipeline_initialization"):
                self.pipelines['emotion'] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                logger.info("Emotion classification pipeline initialized")
            
            # Sentiment analysis pipeline
            with performance_monitor("sentiment_pipeline_initialization"):
                self.pipelines['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("Sentiment analysis pipeline initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize transformer models: {str(e)}")
            raise
    
    async def initialize_ai_engines(self):
        """Initialize AI engines for revolutionary features"""
        try:
            logger.info("Initializing AI engines...")
            
            # Import engines (these create singleton instances)
            from app.services.consciousness_engine import create_consciousness_engine
            from app.services.memory_palace_engine import create_memory_palace_engine
            from app.services.emotional_intelligence_engine import create_emotional_intelligence_engine
            from app.services.synesthesia_engine import create_synesthesia_engine
            from app.services.meta_reality_engine import create_meta_reality_engine
            from app.services.transcendence_engine import create_transcendence_engine
            from app.services.digital_telepathy_engine import DigitalTelepathyEngine
            from app.services.personality_engine import create_personality_engine
            from app.services.quantum_consciousness_engine import create_quantum_consciousness_engine
            
            # Initialize engines
            with performance_monitor("consciousness_engine_init"):
                self.engines['consciousness'] = create_consciousness_engine()
                logger.info("Consciousness engine initialized")
            
            with performance_monitor("memory_palace_engine_init"):
                self.engines['memory_palace'] = create_memory_palace_engine()
                logger.info("Memory palace engine initialized")
            
            with performance_monitor("emotional_intelligence_engine_init"):
                self.engines['emotional_intelligence'] = create_emotional_intelligence_engine()
                logger.info("Emotional intelligence engine initialized")
            
            with performance_monitor("synesthesia_engine_init"):
                self.engines['synesthesia'] = create_synesthesia_engine()
                logger.info("Synesthesia engine initialized")
            
            with performance_monitor("meta_reality_engine_init"):
                self.engines['meta_reality'] = create_meta_reality_engine()
                logger.info("Meta reality engine initialized")
            
            with performance_monitor("transcendence_engine_init"):
                self.engines['transcendence'] = create_transcendence_engine()
                logger.info("Transcendence engine initialized")
            
            with performance_monitor("personality_engine_init"):
                self.engines['personality'] = create_personality_engine()
                logger.info("Personality engine initialized")
            
            with performance_monitor("quantum_consciousness_engine_init"):
                self.engines['quantum_consciousness'] = create_quantum_consciousness_engine()
                logger.info("Quantum consciousness engine initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI engines: {str(e)}")
            raise
    
    async def initialize_circuit_breakers(self):
        """Initialize circuit breakers for external API calls"""
        try:
            from app.core.circuit_breaker import CircuitBreakerManager
            
            self.engines['circuit_breaker'] = CircuitBreakerManager()
            await self.engines['circuit_breaker'].initialize()
            logger.info("Circuit breakers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize circuit breakers: {str(e)}")
            raise
    
    async def initialize_caching(self):
        """Initialize advanced caching systems"""
        try:
            from app.core.advanced_cache import AdvancedCacheManager
            
            self.engines['cache'] = AdvancedCacheManager()
            await self.engines['cache'].initialize()
            logger.info("Advanced caching initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize caching: {str(e)}")
            raise
    
    async def initialize_websocket_manager(self):
        """Initialize WebSocket connection manager"""
        try:
            from app.websocket.manager import WebSocketManager
            
            self.engines['websocket'] = WebSocketManager()
            await self.engines['websocket'].initialize()
            logger.info("WebSocket manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {str(e)}")
            raise
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name"""
        return self.models.get(model_name)
    
    def get_tokenizer(self, tokenizer_name: str) -> Optional[Any]:
        """Get a loaded tokenizer by name"""
        return self.tokenizers.get(tokenizer_name)
    
    def get_pipeline(self, pipeline_name: str) -> Optional[Any]:
        """Get a loaded pipeline by name"""
        return self.pipelines.get(pipeline_name)
    
    def get_engine(self, engine_name: str) -> Optional[Any]:
        """Get an initialized engine by name"""
        return self.engines.get(engine_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all ML components"""
        health_status = {
            'models_loaded': len(self.models),
            'tokenizers_loaded': len(self.tokenizers),
            'pipelines_loaded': len(self.pipelines),
            'engines_initialized': len(self.engines),
            'initialized': self.initialized,
            'components': {}
        }
        
        # Test each component
        try:
            # Test BERT
            if 'bert' in self.models:
                test_input = self.tokenizers['bert']("test", return_tensors="pt")
                with torch.no_grad():
                    _ = self.models['bert'](**test_input)
                health_status['components']['bert'] = 'healthy'
            
            # Test sentence transformer
            if 'sentence_transformer' in self.models:
                _ = self.models['sentence_transformer'].encode("test")
                health_status['components']['sentence_transformer'] = 'healthy'
            
            # Test pipelines
            for pipeline_name, pipeline in self.pipelines.items():
                try:
                    _ = pipeline("test")
                    health_status['components'][pipeline_name] = 'healthy'
                except Exception as e:
                    health_status['components'][pipeline_name] = f'error: {str(e)}'
            
            # Test engines
            for engine_name, engine in self.engines.items():
                if hasattr(engine, 'health_check'):
                    try:
                        engine_health = await engine.health_check()
                        health_status['components'][engine_name] = engine_health
                    except Exception as e:
                        health_status['components'][engine_name] = f'error: {str(e)}'
                else:
                    health_status['components'][engine_name] = 'no health check available'
                    
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_status['error'] = str(e)
        
        return health_status
    
    async def cleanup(self):
        """Cleanup all models and engines"""
        try:
            logger.info("Cleaning up ML models and engines...")
            
            # Cleanup engines that have cleanup methods
            for engine_name, engine in self.engines.items():
                if hasattr(engine, 'cleanup'):
                    try:
                        await engine.cleanup()
                        logger.info(f"{engine_name} engine cleaned up")
                    except Exception as e:
                        logger.error(f"Failed to cleanup {engine_name} engine: {str(e)}")
            
            # Clear model references
            self.models.clear()
            self.tokenizers.clear()
            self.pipelines.clear()
            self.engines.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.initialized = False
            logger.info("ML cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup ML models: {str(e)}")
            raise

# Global model manager instance
model_manager = MLModelManager()

async def initialize_ml_models():
    """Initialize all ML models and AI engines"""
    global model_manager
    
    if model_manager.initialized:
        logger.info("ML models already initialized")
        return
    
    try:
        logger.info("Starting ML models initialization...")
        
        # Initialize components in order
        await model_manager.initialize_transformers()
        await model_manager.initialize_ai_engines()
        await model_manager.initialize_circuit_breakers()
        await model_manager.initialize_caching()
        await model_manager.initialize_websocket_manager()
        
        model_manager.initialized = True
        logger.info("All ML models and AI engines initialized successfully")
        
        # Run health check
        health_status = await model_manager.health_check()
        logger.info(f"ML system health: {health_status}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {str(e)}")
        # Attempt cleanup on failure
        try:
            await model_manager.cleanup()
        except:
            pass
        raise

async def cleanup_ml_models():
    """Cleanup all ML models and AI engines"""
    global model_manager
    
    try:
        await model_manager.cleanup()
    except Exception as e:
        logger.error(f"Failed to cleanup ML models: {str(e)}")
        raise

def get_model_manager() -> MLModelManager:
    """Get the global model manager instance"""
    return model_manager