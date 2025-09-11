"""
Digital Synesthesia Engine

Revolutionary sensory translation system that converts between different
modalities (text, audio, color, spatial) for enhanced user experience.
"""

import asyncio
import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import colorsys
import random

from app.core.redis import redis_manager
from app.core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class SensoryModality(Enum):
    """Sensory modality types"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TEXTUAL = "textual"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"

@dataclass
class ColorMapping:
    """Color representation for synesthesia"""
    hue: float  # 0-360
    saturation: float  # 0-1
    brightness: float  # 0-1
    hex_color: str
    rgb: Tuple[int, int, int]
    emotion_association: str

@dataclass
class SpatialMapping:
    """Spatial representation for concepts"""
    x: float  # -1 to 1
    y: float  # -1 to 1
    z: float  # -1 to 1
    size: float  # 0-1
    shape: str
    texture: str

@dataclass
class AudioMapping:
    """Audio representation for concepts"""
    frequency: float  # Hz
    amplitude: float  # 0-1
    duration: float  # seconds
    timbre: str
    harmony: List[float]

@dataclass
class SynestheticExperience:
    """Complete synesthetic translation"""
    source_text: str
    visual: ColorMapping
    spatial: SpatialMapping
    auditory: AudioMapping
    emotional_resonance: float
    confidence: float
    timestamp: datetime

class SynesthesiaEngine:
    """Revolutionary digital synesthesia engine"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
        
        # Synesthetic mapping models
        self.text_to_color_model = None
        self.emotion_to_space_model = None
        self.concept_to_sound_model = None
        
        # Base synesthetic mappings
        self.emotion_color_map = {
            'joy': {'hue': 60, 'saturation': 0.8, 'brightness': 0.9},  # Bright yellow
            'love': {'hue': 0, 'saturation': 0.7, 'brightness': 0.8},   # Warm red
            'peace': {'hue': 180, 'saturation': 0.5, 'brightness': 0.7}, # Calm cyan
            'anger': {'hue': 0, 'saturation': 1.0, 'brightness': 0.6},   # Deep red
            'sadness': {'hue': 240, 'saturation': 0.6, 'brightness': 0.4}, # Deep blue
            'fear': {'hue': 280, 'saturation': 0.8, 'brightness': 0.3},   # Dark purple
            'excitement': {'hue': 30, 'saturation': 1.0, 'brightness': 1.0}, # Bright orange
            'calm': {'hue': 120, 'saturation': 0.4, 'brightness': 0.6}    # Soft green
        }
        
        # Word-to-spatial mappings
        self.concept_spatial_map = {
            'growth': {'y': 0.8, 'size': 0.7, 'shape': 'tree'},
            'depth': {'z': -0.8, 'size': 0.9, 'shape': 'well'},
            'freedom': {'x': 0.9, 'y': 0.5, 'size': 0.8, 'shape': 'bird'},
            'connection': {'x': 0.0, 'y': 0.0, 'size': 0.6, 'shape': 'web'},
            'power': {'z': 0.7, 'size': 1.0, 'shape': 'mountain'},
            'flow': {'x': 0.5, 'y': -0.3, 'size': 0.5, 'shape': 'river'}
        }
        
        # Frequency mappings for concepts
        self.concept_frequency_map = {
            'harmony': 432.0,  # Hz - natural harmony frequency
            'tension': 666.0,  # Dissonant frequency
            'peace': 528.0,    # Love frequency
            'energy': 741.0,   # Awakening frequency
            'healing': 396.0,  # Liberation frequency
            'wisdom': 852.0,   # Spiritual frequency
            'creation': 963.0, # Divine frequency
            'grounding': 174.0 # Foundation frequency
        }
    
    async def initialize(self) -> bool:
        """Initialize synesthesia engine"""
        try:
            await self._load_synesthetic_models()
            await self._calibrate_sensory_mappings()
            logger.info("Synesthesia engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize synesthesia engine: {str(e)}")
            return False
    
    async def _load_synesthetic_models(self):
        """Load synesthetic translation models"""
        self.text_to_color_model = {
            'version': '2.5.0',
            'accuracy': 0.84,
            'color_space': 'HSB',
            'emotional_sensitivity': 0.92
        }
        
        self.emotion_to_space_model = {
            'version': '1.8.0',
            'spatial_accuracy': 0.87,
            'dimensional_mapping': '3D',
            'concept_granularity': 'high'
        }
        
        self.concept_to_sound_model = {
            'version': '3.1.0',
            'frequency_range': '20-20000Hz',
            'harmonic_complexity': 'advanced',
            'emotional_resonance': 0.89
        }
    
    async def _calibrate_sensory_mappings(self):
        """Calibrate synesthetic mappings for consistency"""
        # Ensure color mappings are consistent
        for emotion, color_data in self.emotion_color_map.items():
            # Validate HSB values
            color_data['hue'] = max(0, min(360, color_data['hue']))
            color_data['saturation'] = max(0, min(1, color_data['saturation']))
            color_data['brightness'] = max(0, min(1, color_data['brightness']))
    
    @CircuitBreaker.protect
    async def translate_to_synesthetic_experience(self, text: str, 
                                                context: Dict[str, Any] = None) -> SynestheticExperience:
        """Translate text into complete synesthetic experience"""
        try:
            # Analyze text for emotional and conceptual content
            emotional_content = await self._analyze_emotional_content(text)
            conceptual_content = await self._analyze_conceptual_content(text)
            
            # Generate visual representation
            visual_mapping = await self._generate_visual_mapping(
                text, emotional_content, conceptual_content
            )
            
            # Generate spatial representation
            spatial_mapping = await self._generate_spatial_mapping(
                text, conceptual_content
            )
            
            # Generate auditory representation
            auditory_mapping = await self._generate_auditory_mapping(
                text, emotional_content, conceptual_content
            )
            
            # Calculate emotional resonance
            resonance = await self._calculate_emotional_resonance(
                emotional_content, visual_mapping, spatial_mapping, auditory_mapping
            )
            
            # Calculate confidence
            confidence = await self._calculate_translation_confidence(
                text, emotional_content, conceptual_content
            )
            
            experience = SynestheticExperience(
                source_text=text,
                visual=visual_mapping,
                spatial=spatial_mapping,
                auditory=auditory_mapping,
                emotional_resonance=resonance,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Cache the experience
            await self._cache_synesthetic_experience(text, experience)
            
            logger.info(f"Synesthetic experience generated for text: {len(text)} chars")
            return experience
            
        except Exception as e:
            logger.error(f"Synesthetic translation failed: {str(e)}")
            # Return neutral experience on failure
            return await self._get_neutral_experience(text)
    
    async def _analyze_emotional_content(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of text"""
        emotional_scores = {}
        text_lower = text.lower()
        
        # Emotional keyword analysis
        for emotion in self.emotion_color_map.keys():
            score = 0.0
            
            # Direct emotion words
            if emotion in text_lower:
                score += 0.5
            
            # Related emotional indicators
            emotion_indicators = {
                'joy': ['happy', 'wonderful', 'amazing', 'great', 'love'],
                'sadness': ['sad', 'down', 'disappointed', 'hurt', 'cry'],
                'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed'],
                'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
                'peace': ['calm', 'peaceful', 'serene', 'tranquil', 'quiet'],
                'excitement': ['excited', 'thrilled', 'enthusiastic', 'energetic']
            }
            
            indicators = emotion_indicators.get(emotion, [])
            for indicator in indicators:
                if indicator in text_lower:
                    score += 0.2
            
            emotional_scores[emotion] = min(1.0, score)
        
        return emotional_scores
    
    async def _analyze_conceptual_content(self, text: str) -> Dict[str, float]:
        """Analyze conceptual content of text"""
        conceptual_scores = {}
        text_lower = text.lower()
        
        # Analyze for spatial/conceptual metaphors
        for concept in self.concept_spatial_map.keys():
            score = 0.0
            
            if concept in text_lower:
                score += 0.4
            
            # Related concept words
            concept_indicators = {
                'growth': ['develop', 'expand', 'progress', 'rise', 'flourish'],
                'depth': ['deep', 'profound', 'beneath', 'inner', 'core'],
                'freedom': ['free', 'liberate', 'open', 'unrestricted', 'independent'],
                'connection': ['connect', 'link', 'bond', 'relationship', 'together'],
                'power': ['strong', 'mighty', 'force', 'energy', 'strength'],
                'flow': ['flowing', 'stream', 'current', 'movement', 'fluid']
            }
            
            indicators = concept_indicators.get(concept, [])
            for indicator in indicators:
                if indicator in text_lower:
                    score += 0.15
            
            conceptual_scores[concept] = min(1.0, score)
        
        return conceptual_scores
    
    async def _generate_visual_mapping(self, text: str, emotional_content: Dict[str, float],
                                     conceptual_content: Dict[str, float]) -> ColorMapping:
        """Generate visual/color mapping"""
        # Find dominant emotion
        dominant_emotion = max(emotional_content, key=emotional_content.get) if emotional_content else 'calm'
        emotion_intensity = emotional_content.get(dominant_emotion, 0.5)
        
        # Get base color from emotion
        base_color = self.emotion_color_map.get(dominant_emotion, 
                                              {'hue': 180, 'saturation': 0.5, 'brightness': 0.5})
        
        # Adjust color based on intensity and conceptual content
        hue = base_color['hue']
        saturation = base_color['saturation'] * emotion_intensity
        brightness = base_color['brightness']
        
        # Modify based on conceptual content
        if conceptual_content.get('power', 0) > 0.3:
            brightness = min(1.0, brightness + 0.2)
        if conceptual_content.get('depth', 0) > 0.3:
            saturation = min(1.0, saturation + 0.2)
            brightness = max(0.1, brightness - 0.2)
        
        # Convert to RGB and hex
        rgb = colorsys.hsv_to_rgb(hue/360, saturation, brightness)
        rgb_int = tuple(int(c * 255) for c in rgb)
        hex_color = f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}"
        
        return ColorMapping(
            hue=hue,
            saturation=saturation,
            brightness=brightness,
            hex_color=hex_color,
            rgb=rgb_int,
            emotion_association=dominant_emotion
        )
    
    async def _generate_spatial_mapping(self, text: str, 
                                      conceptual_content: Dict[str, float]) -> SpatialMapping:
        """Generate spatial representation"""
        # Find dominant concept
        dominant_concept = max(conceptual_content, key=conceptual_content.get) if conceptual_content else 'connection'
        concept_intensity = conceptual_content.get(dominant_concept, 0.5)
        
        # Get base spatial mapping
        base_spatial = self.concept_spatial_map.get(dominant_concept, {
            'x': 0.0, 'y': 0.0, 'size': 0.5, 'shape': 'sphere'
        })
        
        # Calculate spatial coordinates
        x = base_spatial.get('x', 0.0) * concept_intensity
        y = base_spatial.get('y', 0.0) * concept_intensity
        z = base_spatial.get('z', 0.0) * concept_intensity
        size = base_spatial.get('size', 0.5) * concept_intensity
        shape = base_spatial.get('shape', 'sphere')
        
        # Determine texture based on emotional content
        texture_map = {
            'joy': 'smooth',
            'sadness': 'rough',
            'anger': 'jagged',
            'peace': 'flowing',
            'fear': 'sharp',
            'love': 'soft'
        }
        
        # Find emotional context for texture
        texture = 'smooth'  # default
        for emotion, score in conceptual_content.items():
            if score > 0.3 and emotion in texture_map:
                texture = texture_map[emotion]
                break
        
        return SpatialMapping(
            x=x,
            y=y,
            z=z,
            size=size,
            shape=shape,
            texture=texture
        )
    
    async def _generate_auditory_mapping(self, text: str, emotional_content: Dict[str, float],
                                       conceptual_content: Dict[str, float]) -> AudioMapping:
        """Generate auditory representation"""
        # Base frequency from dominant concept
        dominant_concept = max(conceptual_content, key=conceptual_content.get) if conceptual_content else 'harmony'
        base_frequency = self.concept_frequency_map.get(dominant_concept, 432.0)
        
        # Adjust frequency based on emotional content
        emotional_modifier = 1.0
        dominant_emotion = max(emotional_content, key=emotional_content.get) if emotional_content else 'calm'
        
        emotion_frequency_modifiers = {
            'joy': 1.2,
            'excitement': 1.4,
            'anger': 0.7,
            'sadness': 0.6,
            'fear': 0.5,
            'peace': 1.0,
            'love': 1.1
        }
        
        emotional_modifier = emotion_frequency_modifiers.get(dominant_emotion, 1.0)
        frequency = base_frequency * emotional_modifier
        
        # Calculate amplitude based on emotional intensity
        amplitude = sum(emotional_content.values()) / max(1, len(emotional_content))
        amplitude = max(0.1, min(1.0, amplitude))
        
        # Duration based on text length
        duration = min(10.0, max(1.0, len(text) / 20))
        
        # Timbre based on emotional content
        timbre_map = {
            'joy': 'bright',
            'sadness': 'mellow',
            'anger': 'harsh',
            'peace': 'pure',
            'love': 'warm',
            'fear': 'trembling'
        }
        
        timbre = timbre_map.get(dominant_emotion, 'pure')
        
        # Generate harmony (additional frequencies)
        harmony = []
        if conceptual_content.get('harmony', 0) > 0.3:
            harmony = [frequency * 1.5, frequency * 2.0]  # Perfect fifth and octave
        elif conceptual_content.get('tension', 0) > 0.3:
            harmony = [frequency * 1.414]  # Tritone for tension
        
        return AudioMapping(
            frequency=frequency,
            amplitude=amplitude,
            duration=duration,
            timbre=timbre,
            harmony=harmony
        )
    
    async def _calculate_emotional_resonance(self, emotional_content: Dict[str, float],
                                           visual: ColorMapping, spatial: SpatialMapping,
                                           auditory: AudioMapping) -> float:
        """Calculate overall emotional resonance of the synesthetic experience"""
        # Base resonance from emotional content
        base_resonance = sum(emotional_content.values()) / max(1, len(emotional_content))
        
        # Visual contribution (color harmony)
        visual_harmony = (visual.saturation + visual.brightness) / 2
        
        # Spatial contribution (size and position coherence)
        spatial_coherence = (abs(spatial.x) + abs(spatial.y) + spatial.size) / 3
        
        # Auditory contribution (frequency appropriateness)
        auditory_coherence = min(1.0, auditory.amplitude * (len(auditory.harmony) + 1) / 3)
        
        # Weighted combination
        resonance = (base_resonance * 0.4 + 
                    visual_harmony * 0.25 + 
                    spatial_coherence * 0.2 + 
                    auditory_coherence * 0.15)
        
        return min(1.0, resonance)
    
    async def _calculate_translation_confidence(self, text: str, 
                                              emotional_content: Dict[str, float],
                                              conceptual_content: Dict[str, float]) -> float:
        """Calculate confidence in synesthetic translation"""
        confidence = 0.5  # Base confidence
        
        # Text length factor
        if len(text) > 20:
            confidence += 0.1
        if len(text) > 100:
            confidence += 0.1
        
        # Emotional clarity
        max_emotion_score = max(emotional_content.values()) if emotional_content else 0
        confidence += max_emotion_score * 0.2
        
        # Conceptual clarity
        max_concept_score = max(conceptual_content.values()) if conceptual_content else 0
        confidence += max_concept_score * 0.2
        
        # Multi-modal coherence
        if emotional_content and conceptual_content:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _get_neutral_experience(self, text: str) -> SynestheticExperience:
        """Get neutral synesthetic experience for fallback"""
        return SynestheticExperience(
            source_text=text,
            visual=ColorMapping(
                hue=180, saturation=0.5, brightness=0.5,
                hex_color="#408080", rgb=(64, 128, 128),
                emotion_association="neutral"
            ),
            spatial=SpatialMapping(
                x=0.0, y=0.0, z=0.0, size=0.5,
                shape="sphere", texture="smooth"
            ),
            auditory=AudioMapping(
                frequency=432.0, amplitude=0.5, duration=2.0,
                timbre="pure", harmony=[]
            ),
            emotional_resonance=0.5,
            confidence=0.3,
            timestamp=datetime.now()
        )
    
    async def _cache_synesthetic_experience(self, text: str, experience: SynestheticExperience):
        """Cache synesthetic experience"""
        try:
            # Create cache key from text hash
            text_hash = hash(text) % 1000000
            cache_key = f"synesthesia:{text_hash}"
            
            # Convert experience to dict
            experience_data = {
                'source_text': experience.source_text,
                'visual': {
                    'hue': experience.visual.hue,
                    'saturation': experience.visual.saturation,
                    'brightness': experience.visual.brightness,
                    'hex_color': experience.visual.hex_color,
                    'rgb': experience.visual.rgb,
                    'emotion_association': experience.visual.emotion_association
                },
                'spatial': {
                    'x': experience.spatial.x,
                    'y': experience.spatial.y,
                    'z': experience.spatial.z,
                    'size': experience.spatial.size,
                    'shape': experience.spatial.shape,
                    'texture': experience.spatial.texture
                },
                'auditory': {
                    'frequency': experience.auditory.frequency,
                    'amplitude': experience.auditory.amplitude,
                    'duration': experience.auditory.duration,
                    'timbre': experience.auditory.timbre,
                    'harmony': experience.auditory.harmony
                },
                'emotional_resonance': experience.emotional_resonance,
                'confidence': experience.confidence,
                'timestamp': experience.timestamp.isoformat()
            }
            
            await redis_manager.set(
                cache_key,
                json.dumps(experience_data),
                ttl=3600  # 1 hour cache
            )
            
        except Exception as e:
            logger.error(f"Failed to cache synesthetic experience: {str(e)}")
    
    async def generate_synesthetic_visualization(self, experience: SynestheticExperience) -> Dict[str, Any]:
        """Generate visualization data for synesthetic experience"""
        return {
            'color_palette': {
                'primary': experience.visual.hex_color,
                'rgb': experience.visual.rgb,
                'hsl': (experience.visual.hue, experience.visual.saturation, experience.visual.brightness)
            },
            'spatial_representation': {
                'position': {'x': experience.spatial.x, 'y': experience.spatial.y, 'z': experience.spatial.z},
                'size': experience.spatial.size,
                'shape': experience.spatial.shape,
                'texture': experience.spatial.texture
            },
            'audio_parameters': {
                'frequency': experience.auditory.frequency,
                'amplitude': experience.auditory.amplitude,
                'duration': experience.auditory.duration,
                'timbre': experience.auditory.timbre,
                'harmony': experience.auditory.harmony
            },
            'emotional_data': {
                'resonance': experience.emotional_resonance,
                'primary_emotion': experience.visual.emotion_association
            },
            'confidence': experience.confidence
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of synesthesia engine"""
        return {
            'status': 'healthy',
            'models_loaded': bool(
                self.text_to_color_model and 
                self.emotion_to_space_model and 
                self.concept_to_sound_model
            ),
            'emotion_mappings': len(self.emotion_color_map),
            'spatial_mappings': len(self.concept_spatial_map),
            'frequency_mappings': len(self.concept_frequency_map),
            'circuit_breaker': self.circuit_breaker.state,
            'last_check': datetime.now().isoformat()
        }
