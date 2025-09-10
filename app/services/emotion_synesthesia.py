"""
Emotion Synesthesia Service

Advanced emotion-to-synesthetic experience translation implementing:
- Emotion-to-color mapping based on psychological research
- Emotional texture generation for haptic feedback
- Mood-based spatial environments
- Real-time emotion detection and visualization
- Multi-modal emotional synesthetic experiences
"""

import asyncio
import numpy as np
import colorsys
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from enum import Enum

# ML and emotion detection libraries
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from scipy import signal
from scipy.interpolate import interp1d

from app.core.config import get_settings
from app.models.synesthesia import SynestheticProfile
from app.services.synesthetic_engine import SynestheticStimulus, ModalityType

logger = logging.getLogger(__name__)
settings = get_settings()


class EmotionType(Enum):
    """Core emotion types based on psychological research"""
    JOY = "joy"
    SADNESS = "sadness"  
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    CONTEMPT = "contempt"
    SHAME = "shame"
    GUILT = "guilt"
    PRIDE = "pride"
    ENVY = "envy"
    GRATITUDE = "gratitude"
    HOPE = "hope"
    DESPAIR = "despair"
    EXCITEMENT = "excitement"
    CALM = "calm"
    ANXIETY = "anxiety"


class EmotionalIntensity(Enum):
    """Emotional intensity levels"""
    SUBTLE = "subtle"           # 0.1-0.3
    MODERATE = "moderate"       # 0.3-0.6
    STRONG = "strong"          # 0.6-0.8
    INTENSE = "intense"        # 0.8-1.0


@dataclass
class EmotionalState:
    """Complete emotional state representation"""
    primary_emotion: EmotionType
    intensity: float
    secondary_emotions: List[Tuple[EmotionType, float]]
    valence: float              # Positive/negative (-1 to 1)
    arousal: float              # Energy level (0 to 1)
    dominance: float            # Control/submission (-1 to 1)
    temporal_dynamics: Dict[str, float]


@dataclass
class EmotionalColorPalette:
    """Emotion-based color scheme"""
    primary_color: Dict[str, Any]
    secondary_colors: List[Dict[str, Any]]
    accent_colors: List[Dict[str, Any]]
    temperature: str            # warm, cool, neutral
    saturation_profile: str     # vivid, muted, mixed
    brightness_profile: str     # bright, dark, balanced


@dataclass
class EmotionalSpatialEnvironment:
    """3D emotional environment characteristics"""
    spatial_metaphor: str
    environmental_elements: List[Dict[str, Any]]
    lighting_scheme: Dict[str, Any]
    particle_systems: List[Dict[str, Any]]
    atmospheric_effects: Dict[str, Any]


class EmotionColorMapper:
    """Maps emotions to authentic color experiences based on research"""
    
    def __init__(self):
        # Research-based emotion-color associations
        # Based on studies by Kaya & Epps (2004), Palmer & Schloss (2010)
        self.emotion_color_mappings = {
            EmotionType.JOY: {
                'hue_range': (50, 70),      # Yellow-orange
                'saturation': (0.7, 0.9),
                'brightness': (0.8, 1.0),
                'temperature': 'warm',
                'associated_colors': ['#FFD700', '#FFA500', '#FF6347']
            },
            EmotionType.SADNESS: {
                'hue_range': (200, 240),    # Blue
                'saturation': (0.5, 0.8),
                'brightness': (0.2, 0.5),
                'temperature': 'cool',
                'associated_colors': ['#4682B4', '#87CEEB', '#191970']
            },
            EmotionType.ANGER: {
                'hue_range': (0, 20),       # Red
                'saturation': (0.8, 1.0),
                'brightness': (0.6, 0.9),
                'temperature': 'warm',
                'associated_colors': ['#DC143C', '#FF0000', '#8B0000']
            },
            EmotionType.FEAR: {
                'hue_range': (270, 300),    # Purple-violet
                'saturation': (0.4, 0.7),
                'brightness': (0.1, 0.4),
                'temperature': 'cool',
                'associated_colors': ['#4B0082', '#9400D3', '#8B008B']
            },
            EmotionType.SURPRISE: {
                'hue_range': (30, 50),      # Orange-yellow
                'saturation': (0.8, 1.0),
                'brightness': (0.9, 1.0),
                'temperature': 'warm',
                'associated_colors': ['#FFA500', '#FF8C00', '#FFD700']
            },
            EmotionType.DISGUST: {
                'hue_range': (80, 120),     # Green
                'saturation': (0.3, 0.6),
                'brightness': (0.2, 0.5),
                'temperature': 'cool',
                'associated_colors': ['#556B2F', '#808000', '#6B8E23']
            },
            EmotionType.LOVE: {
                'hue_range': (300, 340),    # Pink-magenta
                'saturation': (0.6, 0.9),
                'brightness': (0.7, 0.9),
                'temperature': 'warm',
                'associated_colors': ['#FF69B4', '#FF1493', '#DC143C']
            },
            EmotionType.CALM: {
                'hue_range': (180, 220),    # Cyan-blue
                'saturation': (0.3, 0.6),
                'brightness': (0.6, 0.8),
                'temperature': 'cool',
                'associated_colors': ['#87CEEB', '#B0E0E6', '#ADD8E6']
            },
            EmotionType.EXCITEMENT: {
                'hue_range': (10, 30),      # Red-orange
                'saturation': (0.8, 1.0),
                'brightness': (0.8, 1.0),
                'temperature': 'warm',
                'associated_colors': ['#FF4500', '#FF6347', '#FF7F50']
            },
            EmotionType.ANXIETY: {
                'hue_range': (40, 80),      # Yellow-green
                'saturation': (0.6, 0.9),
                'brightness': (0.4, 0.7),
                'temperature': 'neutral',
                'associated_colors': ['#ADFF2F', '#9ACD32', '#8FBC8F']
            }
        }
        
        # Intensity effect multipliers
        self.intensity_effects = {
            EmotionalIntensity.SUBTLE: {'sat_mult': 0.5, 'bright_mult': 0.7},
            EmotionalIntensity.MODERATE: {'sat_mult': 0.8, 'bright_mult': 0.9},
            EmotionalIntensity.STRONG: {'sat_mult': 1.0, 'bright_mult': 1.0},
            EmotionalIntensity.INTENSE: {'sat_mult': 1.2, 'bright_mult': 1.1}
        }
        
        # Color mixing rules for complex emotions
        self.mixing_weights = {
            'primary': 0.7,
            'secondary': 0.3,
            'blend_factor': 0.5
        }
    
    def map_emotion_to_color_palette(self, emotional_state: EmotionalState) -> EmotionalColorPalette:
        """Generate comprehensive color palette from emotional state"""
        
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Get base color mapping
        color_config = self.emotion_color_mappings.get(primary_emotion, 
                                                     self.emotion_color_mappings[EmotionType.CALM])
        
        # Calculate primary color
        primary_color = self._generate_primary_color(color_config, intensity)
        
        # Generate secondary colors from secondary emotions
        secondary_colors = []
        for sec_emotion, sec_intensity in emotional_state.secondary_emotions[:3]:  # Max 3 secondary
            sec_config = self.emotion_color_mappings.get(sec_emotion, color_config)
            sec_color = self._generate_primary_color(sec_config, sec_intensity)
            secondary_colors.append(sec_color)
        
        # Generate accent colors
        accent_colors = self._generate_accent_colors(primary_color, color_config)
        
        # Determine overall characteristics
        temperature = color_config['temperature']
        saturation_profile = self._determine_saturation_profile(emotional_state)
        brightness_profile = self._determine_brightness_profile(emotional_state)
        
        return EmotionalColorPalette(
            primary_color=primary_color,
            secondary_colors=secondary_colors,
            accent_colors=accent_colors,
            temperature=temperature,
            saturation_profile=saturation_profile,
            brightness_profile=brightness_profile
        )
    
    def _generate_primary_color(self, color_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """Generate primary color from emotion config and intensity"""
        
        # Get hue range
        hue_min, hue_max = color_config['hue_range']
        hue = np.random.uniform(hue_min, hue_max)
        
        # Get saturation and brightness ranges
        sat_min, sat_max = color_config['saturation']
        bright_min, bright_max = color_config['brightness']
        
        # Apply intensity effects
        intensity_category = self._classify_intensity(intensity)
        intensity_effects = self.intensity_effects[intensity_category]
        
        # Calculate final values
        saturation = np.random.uniform(sat_min, sat_max) * intensity_effects['sat_mult']
        brightness = np.random.uniform(bright_min, bright_max) * intensity_effects['bright_mult']
        
        # Clamp values
        saturation = max(0.0, min(1.0, saturation))
        brightness = max(0.0, min(1.0, brightness))
        
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        return {
            'hue': hue,
            'saturation': saturation,
            'brightness': brightness,
            'rgb': {
                'r': int(rgb[0] * 255),
                'g': int(rgb[1] * 255),
                'b': int(rgb[2] * 255)
            },
            'hex': hex_color,
            'intensity_factor': intensity,
            'emotion_source': color_config.get('emotion', 'unknown')
        }
    
    def _generate_accent_colors(self, primary_color: Dict[str, Any], 
                              color_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate harmonious accent colors"""
        
        base_hue = primary_color['hue']
        accents = []
        
        # Complementary color
        comp_hue = (base_hue + 180) % 360
        comp_color = {
            'hue': comp_hue,
            'saturation': primary_color['saturation'] * 0.8,
            'brightness': primary_color['brightness'] * 0.9
        }
        rgb = colorsys.hsv_to_rgb(comp_hue / 360.0, comp_color['saturation'], comp_color['brightness'])
        comp_color.update({
            'rgb': {'r': int(rgb[0] * 255), 'g': int(rgb[1] * 255), 'b': int(rgb[2] * 255)},
            'hex': f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}",
            'type': 'complementary'
        })
        accents.append(comp_color)
        
        # Triadic colors
        for offset in [120, 240]:
            tri_hue = (base_hue + offset) % 360
            tri_color = {
                'hue': tri_hue,
                'saturation': primary_color['saturation'] * 0.7,
                'brightness': primary_color['brightness'] * 0.8
            }
            rgb = colorsys.hsv_to_rgb(tri_hue / 360.0, tri_color['saturation'], tri_color['brightness'])
            tri_color.update({
                'rgb': {'r': int(rgb[0] * 255), 'g': int(rgb[1] * 255), 'b': int(rgb[2] * 255)},
                'hex': f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}",
                'type': 'triadic'
            })
            accents.append(tri_color)
        
        return accents
    
    def _classify_intensity(self, intensity: float) -> EmotionalIntensity:
        """Classify intensity level"""
        
        if intensity < 0.3:
            return EmotionalIntensity.SUBTLE
        elif intensity < 0.6:
            return EmotionalIntensity.MODERATE
        elif intensity < 0.8:
            return EmotionalIntensity.STRONG
        else:
            return EmotionalIntensity.INTENSE
    
    def _determine_saturation_profile(self, emotional_state: EmotionalState) -> str:
        """Determine overall saturation characteristics"""
        
        arousal = emotional_state.arousal
        intensity = emotional_state.intensity
        
        if arousal > 0.7 and intensity > 0.6:
            return 'vivid'
        elif arousal < 0.3 or intensity < 0.3:
            return 'muted'
        else:
            return 'mixed'
    
    def _determine_brightness_profile(self, emotional_state: EmotionalState) -> str:
        """Determine overall brightness characteristics"""
        
        valence = emotional_state.valence
        arousal = emotional_state.arousal
        
        if valence > 0.3 and arousal > 0.5:
            return 'bright'
        elif valence < -0.3 or arousal < 0.3:
            return 'dark'
        else:
            return 'balanced'


class EmotionSpatialMapper:
    """Maps emotions to spatial environments and layouts"""
    
    def __init__(self):
        # Emotional spatial metaphors
        self.spatial_metaphors = {
            EmotionType.JOY: {
                'metaphor': 'ascending_spiral',
                'direction': 'upward',
                'expansion': 'outward',
                'movement': 'light_floating'
            },
            EmotionType.SADNESS: {
                'metaphor': 'descending_valley',
                'direction': 'downward',
                'expansion': 'inward',
                'movement': 'heavy_sinking'
            },
            EmotionType.ANGER: {
                'metaphor': 'explosive_burst',
                'direction': 'outward',
                'expansion': 'aggressive',
                'movement': 'sharp_thrusts'
            },
            EmotionType.FEAR: {
                'metaphor': 'constricting_maze',
                'direction': 'backward',
                'expansion': 'contracting',
                'movement': 'trembling_retreat'
            },
            EmotionType.LOVE: {
                'metaphor': 'embracing_cocoon',
                'direction': 'enveloping',
                'expansion': 'gentle_growth',
                'movement': 'warm_flow'
            },
            EmotionType.CALM: {
                'metaphor': 'still_lake',
                'direction': 'centered',
                'expansion': 'stable',
                'movement': 'gentle_ripples'
            }
        }
        
        # Environmental element libraries
        self.environmental_elements = {
            'particles': {
                'joy': {'type': 'sparks', 'behavior': 'rising', 'color': 'golden'},
                'sadness': {'type': 'droplets', 'behavior': 'falling', 'color': 'blue'},
                'anger': {'type': 'embers', 'behavior': 'explosive', 'color': 'red'},
                'fear': {'type': 'shadows', 'behavior': 'fleeting', 'color': 'dark'},
                'love': {'type': 'petals', 'behavior': 'floating', 'color': 'pink'},
                'calm': {'type': 'motes', 'behavior': 'drifting', 'color': 'soft_blue'}
            },
            'geometry': {
                'joy': {'shapes': ['spheres', 'spirals'], 'form': 'organic_curves'},
                'sadness': {'shapes': ['teardrops', 'downward_curves'], 'form': 'flowing'},
                'anger': {'shapes': ['spikes', 'jagged_edges'], 'form': 'angular_sharp'},
                'fear': {'shapes': ['irregular_forms', 'broken_patterns'], 'form': 'fragmented'},
                'love': {'shapes': ['hearts', 'embracing_curves'], 'form': 'encompassing'},
                'calm': {'shapes': ['smooth_planes', 'gentle_waves'], 'form': 'harmonious'}
            }
        }
    
    def create_emotional_environment(self, emotional_state: EmotionalState,
                                   space_dimensions: Tuple[float, float, float] = (10.0, 10.0, 10.0)) -> EmotionalSpatialEnvironment:
        """Create complete 3D emotional environment"""
        
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Get spatial metaphor
        metaphor_config = self.spatial_metaphors.get(primary_emotion, 
                                                   self.spatial_metaphors[EmotionType.CALM])
        
        # Create environmental elements
        elements = self._create_environmental_elements(emotional_state, space_dimensions)
        
        # Generate lighting scheme
        lighting = self._create_emotional_lighting(emotional_state)
        
        # Create particle systems
        particles = self._create_particle_systems(emotional_state)
        
        # Generate atmospheric effects
        atmosphere = self._create_atmospheric_effects(emotional_state)
        
        return EmotionalSpatialEnvironment(
            spatial_metaphor=metaphor_config['metaphor'],
            environmental_elements=elements,
            lighting_scheme=lighting,
            particle_systems=particles,
            atmospheric_effects=atmosphere
        )
    
    def _create_environmental_elements(self, emotional_state: EmotionalState,
                                     dimensions: Tuple[float, float, float]) -> List[Dict[str, Any]]:
        """Create geometric and organic elements for the environment"""
        
        elements = []
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Get element config
        particle_config = self.environmental_elements['particles'].get(primary_emotion.value, 
                                                                      self.environmental_elements['particles']['calm'])
        geometry_config = self.environmental_elements['geometry'].get(primary_emotion.value,
                                                                     self.environmental_elements['geometry']['calm'])
        
        # Create central focus element
        central_element = {
            'type': 'central_focus',
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'scale': {'x': intensity * 2.0, 'y': intensity * 2.0, 'z': intensity * 2.0},
            'shape': geometry_config['shapes'][0],
            'form_style': geometry_config['form'],
            'emission_intensity': intensity,
            'animation': self._get_emotion_animation(primary_emotion),
            'color_influence': 0.8  # How much it affects surrounding colors
        }
        elements.append(central_element)
        
        # Create surrounding elements based on secondary emotions
        for i, (sec_emotion, sec_intensity) in enumerate(emotional_state.secondary_emotions[:2]):
            angle = i * np.pi  # 180 degrees apart
            distance = 3.0 + (sec_intensity * 2.0)
            
            x = distance * np.cos(angle)
            z = distance * np.sin(angle)
            y = sec_intensity * 1.5  # Vary height by intensity
            
            sec_config = self.environmental_elements['geometry'].get(sec_emotion.value,
                                                                   geometry_config)
            
            secondary_element = {
                'type': 'secondary_resonator',
                'position': {'x': x, 'y': y, 'z': z},
                'scale': {'x': sec_intensity, 'y': sec_intensity, 'z': sec_intensity},
                'shape': sec_config['shapes'][0],
                'form_style': sec_config['form'],
                'emission_intensity': sec_intensity * 0.7,
                'animation': self._get_emotion_animation(sec_emotion),
                'color_influence': 0.4,
                'connection_to_center': True
            }
            elements.append(secondary_element)
        
        return elements
    
    def _create_emotional_lighting(self, emotional_state: EmotionalState) -> Dict[str, Any]:
        """Create lighting scheme based on emotional state"""
        
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        valence = emotional_state.valence
        arousal = emotional_state.arousal
        
        # Base lighting configuration
        lighting_configs = {
            EmotionType.JOY: {'type': 'warm_radiant', 'temperature': 3200, 'intensity': 0.9},
            EmotionType.SADNESS: {'type': 'cool_diffuse', 'temperature': 6500, 'intensity': 0.3},
            EmotionType.ANGER: {'type': 'harsh_directional', 'temperature': 2800, 'intensity': 1.0},
            EmotionType.FEAR: {'type': 'flickering_sparse', 'temperature': 4000, 'intensity': 0.2},
            EmotionType.LOVE: {'type': 'soft_ambient', 'temperature': 3000, 'intensity': 0.7},
            EmotionType.CALM: {'type': 'balanced_even', 'temperature': 5000, 'intensity': 0.6}
        }
        
        base_config = lighting_configs.get(primary_emotion, lighting_configs[EmotionType.CALM])
        
        # Adjust based on emotional dimensions
        adjusted_intensity = base_config['intensity'] * intensity * (0.5 + arousal * 0.5)
        adjusted_temperature = base_config['temperature'] + (valence * 1000)  # Warm = positive
        
        return {
            'lighting_type': base_config['type'],
            'color_temperature': int(adjusted_temperature),
            'intensity': adjusted_intensity,
            'shadow_softness': 0.8 if arousal < 0.5 else 0.3,
            'dynamic_changes': arousal > 0.6,
            'ambient_contribution': 0.3 + (1.0 - arousal) * 0.4,
            'directional_lights': [
                {
                    'direction': {'x': 0.0, 'y': -1.0, 'z': 0.3},
                    'intensity': adjusted_intensity,
                    'color_temp': adjusted_temperature,
                    'shadow_quality': 'high' if intensity > 0.7 else 'medium'
                }
            ]
        }
    
    def _create_particle_systems(self, emotional_state: EmotionalState) -> List[Dict[str, Any]]:
        """Create particle effects for emotional visualization"""
        
        systems = []
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        arousal = emotional_state.arousal
        
        # Get particle configuration
        particle_config = self.environmental_elements['particles'].get(primary_emotion.value, 
                                                                      self.environmental_elements['particles']['calm'])
        
        # Primary particle system
        primary_system = {
            'id': 'primary_emotion_particles',
            'particle_type': particle_config['type'],
            'behavior': particle_config['behavior'],
            'color_theme': particle_config['color'],
            'emission_rate': int(50 + intensity * 200),  # 50-250 particles/sec
            'particle_life': 2.0 + intensity * 3.0,     # 2-5 second lifetime
            'velocity_range': {
                'min': arousal * 0.5,
                'max': arousal * 2.0
            },
            'size_range': {
                'min': intensity * 0.1,
                'max': intensity * 0.5
            },
            'opacity_curve': 'fade_in_out',
            'physics_enabled': True,
            'affected_by_wind': arousal > 0.3,
            'gravitational_influence': -1.0 if primary_emotion in [EmotionType.SADNESS, EmotionType.FEAR] else 0.0
        }
        systems.append(primary_system)
        
        # Secondary systems for complex emotions
        if len(emotional_state.secondary_emotions) > 0:
            sec_emotion, sec_intensity = emotional_state.secondary_emotions[0]
            sec_config = self.environmental_elements['particles'].get(sec_emotion.value, particle_config)
            
            secondary_system = {
                'id': 'secondary_emotion_particles',
                'particle_type': sec_config['type'],
                'behavior': sec_config['behavior'],
                'color_theme': sec_config['color'],
                'emission_rate': int(20 + sec_intensity * 80),
                'particle_life': 1.5 + sec_intensity * 2.0,
                'velocity_range': {
                    'min': sec_intensity * 0.3,
                    'max': sec_intensity * 1.5
                },
                'size_range': {
                    'min': sec_intensity * 0.05,
                    'max': sec_intensity * 0.3
                },
                'opacity_curve': 'fade_in_out',
                'blend_mode': 'additive',
                'interaction_with_primary': True
            }
            systems.append(secondary_system)
        
        return systems
    
    def _create_atmospheric_effects(self, emotional_state: EmotionalState) -> Dict[str, Any]:
        """Create atmospheric effects like fog, wind, temperature"""
        
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        arousal = emotional_state.arousal
        valence = emotional_state.valence
        
        # Fog/mist effects
        fog_density = 0.0
        fog_color = {'r': 128, 'g': 128, 'b': 128}
        
        if primary_emotion in [EmotionType.SADNESS, EmotionType.FEAR]:
            fog_density = intensity * 0.6
            fog_color = {'r': 100, 'g': 120, 'b': 140}  # Cool fog
        elif primary_emotion == EmotionType.LOVE:
            fog_density = intensity * 0.3
            fog_color = {'r': 200, 'g': 150, 'b': 150}  # Warm fog
        
        # Wind effects
        wind_strength = arousal * intensity
        wind_direction = {
            'x': np.random.uniform(-1.0, 1.0),
            'y': 0.0,  # Horizontal wind
            'z': np.random.uniform(-1.0, 1.0)
        }
        
        # Temperature feeling (visual representation)
        temperature_hint = 'neutral'
        if valence > 0.3:
            temperature_hint = 'warm'
        elif valence < -0.3:
            temperature_hint = 'cool'
        
        return {
            'fog': {
                'enabled': fog_density > 0.05,
                'density': fog_density,
                'color': fog_color,
                'height_falloff': 0.1,
                'animation_speed': arousal * 0.5
            },
            'wind': {
                'enabled': wind_strength > 0.2,
                'strength': wind_strength,
                'direction': wind_direction,
                'turbulence': arousal * 0.3,
                'affects_particles': True,
                'sound_enabled': wind_strength > 0.5
            },
            'temperature': {
                'hint': temperature_hint,
                'visual_warmth': max(-1.0, min(1.0, valence)),
                'heat_shimmer': valence > 0.6 and arousal > 0.7,
                'frost_effect': valence < -0.6 and arousal < 0.3
            },
            'ambient_sound': {
                'enabled': True,
                'volume': intensity * 0.4,
                'tone': 'pleasant' if valence > 0 else 'somber',
                'complexity': arousal * 0.6
            }
        }
    
    def _get_emotion_animation(self, emotion: EmotionType) -> Dict[str, Any]:
        """Get animation pattern for specific emotion"""
        
        animations = {
            EmotionType.JOY: {
                'type': 'bouncing_expansion',
                'speed': 1.5,
                'amplitude': 0.3,
                'pattern': 'sine_wave'
            },
            EmotionType.SADNESS: {
                'type': 'slow_contraction',
                'speed': 0.3,
                'amplitude': 0.1,
                'pattern': 'dampened_oscillation'
            },
            EmotionType.ANGER: {
                'type': 'rapid_pulsing',
                'speed': 3.0,
                'amplitude': 0.4,
                'pattern': 'sharp_spikes'
            },
            EmotionType.FEAR: {
                'type': 'trembling',
                'speed': 2.0,
                'amplitude': 0.15,
                'pattern': 'irregular_jitter'
            },
            EmotionType.LOVE: {
                'type': 'gentle_breathing',
                'speed': 0.8,
                'amplitude': 0.2,
                'pattern': 'smooth_sine'
            },
            EmotionType.CALM: {
                'type': 'steady_rotation',
                'speed': 0.5,
                'amplitude': 0.05,
                'pattern': 'constant_flow'
            }
        }
        
        return animations.get(emotion, animations[EmotionType.CALM])


class EmotionDetector:
    """Detects emotions from various input modalities"""
    
    def __init__(self):
        # Initialize emotion detection models
        try:
            self.text_emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
            self.text_emotion_model = None
        
        # Emotion mapping from model outputs to our enum
        self.emotion_mapping = {
            'joy': EmotionType.JOY,
            'sadness': EmotionType.SADNESS,
            'anger': EmotionType.ANGER,
            'fear': EmotionType.FEAR,
            'surprise': EmotionType.SURPRISE,
            'disgust': EmotionType.DISGUST,
            'love': EmotionType.LOVE
        }
    
    async def detect_emotion_from_text(self, text: str) -> EmotionalState:
        """Detect emotional state from text content"""
        
        try:
            if self.text_emotion_model:
                # Get emotion predictions
                predictions = self.text_emotion_model(text)
                
                # Sort by confidence
                predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
                
                # Extract primary emotion
                primary_pred = predictions[0]
                primary_emotion = self.emotion_mapping.get(
                    primary_pred['label'].lower(), 
                    EmotionType.CALM
                )
                primary_intensity = primary_pred['score']
                
                # Extract secondary emotions
                secondary_emotions = []
                for pred in predictions[1:4]:  # Top 3 secondary emotions
                    if pred['score'] > 0.1:  # Only significant predictions
                        sec_emotion = self.emotion_mapping.get(pred['label'].lower())
                        if sec_emotion and sec_emotion != primary_emotion:
                            secondary_emotions.append((sec_emotion, pred['score']))
                
                # Calculate emotional dimensions
                valence = self._calculate_valence(primary_emotion, primary_intensity, secondary_emotions)
                arousal = self._calculate_arousal(primary_emotion, primary_intensity)
                dominance = self._calculate_dominance(primary_emotion, primary_intensity)
                
                return EmotionalState(
                    primary_emotion=primary_emotion,
                    intensity=primary_intensity,
                    secondary_emotions=secondary_emotions,
                    valence=valence,
                    arousal=arousal,
                    dominance=dominance,
                    temporal_dynamics={'trend': 'stable', 'volatility': 0.1}
                )
                
            else:
                # Fallback: simple keyword-based detection
                return await self._fallback_text_emotion_detection(text)
                
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return EmotionalState(
                primary_emotion=EmotionType.CALM,
                intensity=0.5,
                secondary_emotions=[],
                valence=0.0,
                arousal=0.3,
                dominance=0.0,
                temporal_dynamics={'trend': 'stable', 'volatility': 0.0}
            )
    
    async def detect_emotion_from_audio_features(self, audio_features: Dict[str, Any]) -> EmotionalState:
        """Detect emotion from audio spectral features"""
        
        # Extract relevant features for emotion detection
        spectral_centroid = np.mean(audio_features.get('spectral_centroid', [1000]))
        tempo = audio_features.get('tempo', 120)
        loudness = np.mean(audio_features.get('rms_energy', [0.1]))
        
        # Simple heuristic emotion detection from audio
        # High tempo + high energy = excitement/joy
        # Low tempo + low energy = sadness/calm
        # High spectral centroid = brightness/joy
        # Low spectral centroid = darkness/sadness
        
        arousal = min(1.0, (tempo / 180.0 + loudness * 2.0) / 2.0)
        brightness = min(1.0, spectral_centroid / 4000.0)
        valence = (brightness * 2.0 - 1.0)  # -1 to 1
        
        # Map to primary emotion
        if arousal > 0.7 and valence > 0.3:
            primary_emotion = EmotionType.JOY
        elif arousal < 0.3 and valence < -0.3:
            primary_emotion = EmotionType.SADNESS
        elif arousal > 0.6 and valence < -0.2:
            primary_emotion = EmotionType.ANGER
        elif arousal > 0.8 and abs(valence) < 0.2:
            primary_emotion = EmotionType.EXCITEMENT
        else:
            primary_emotion = EmotionType.CALM
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=arousal,
            secondary_emotions=[],
            valence=valence,
            arousal=arousal,
            dominance=0.0,
            temporal_dynamics={'trend': 'stable', 'volatility': 0.2}
        )
    
    async def _fallback_text_emotion_detection(self, text: str) -> EmotionalState:
        """Simple keyword-based emotion detection as fallback"""
        
        text_lower = text.lower()
        
        # Emotion keywords
        emotion_keywords = {
            EmotionType.JOY: ['happy', 'joy', 'glad', 'excited', 'cheerful', 'delighted'],
            EmotionType.SADNESS: ['sad', 'unhappy', 'depressed', 'down', 'blue', 'melancholy'],
            EmotionType.ANGER: ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
            EmotionType.FEAR: ['scared', 'afraid', 'fearful', 'anxious', 'worried', 'nervous'],
            EmotionType.SURPRISE: ['surprised', 'amazed', 'astonished', 'shocked', 'stunned'],
            EmotionType.LOVE: ['love', 'adore', 'cherish', 'affection', 'romance', 'heart']
        }
        
        # Count keyword matches
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
        
        if emotion_scores:
            primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            intensity = min(1.0, emotion_scores[primary_emotion] * 2.0)
        else:
            primary_emotion = EmotionType.CALM
            intensity = 0.5
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            secondary_emotions=[],
            valence=0.0,
            arousal=intensity,
            dominance=0.0,
            temporal_dynamics={'trend': 'stable', 'volatility': 0.1}
        )
    
    def _calculate_valence(self, primary_emotion: EmotionType, intensity: float,
                          secondary_emotions: List[Tuple[EmotionType, float]]) -> float:
        """Calculate emotional valence (positive/negative)"""
        
        # Valence scores for emotions
        valence_scores = {
            EmotionType.JOY: 0.8,
            EmotionType.LOVE: 0.9,
            EmotionType.EXCITEMENT: 0.7,
            EmotionType.SURPRISE: 0.3,
            EmotionType.CALM: 0.1,
            EmotionType.SADNESS: -0.7,
            EmotionType.ANGER: -0.6,
            EmotionType.FEAR: -0.8,
            EmotionType.DISGUST: -0.9,
            EmotionType.ANXIETY: -0.5
        }
        
        primary_valence = valence_scores.get(primary_emotion, 0.0) * intensity
        
        # Adjust with secondary emotions
        for sec_emotion, sec_intensity in secondary_emotions:
            sec_valence = valence_scores.get(sec_emotion, 0.0) * sec_intensity * 0.3
            primary_valence += sec_valence
        
        return max(-1.0, min(1.0, primary_valence))
    
    def _calculate_arousal(self, primary_emotion: EmotionType, intensity: float) -> float:
        """Calculate emotional arousal (energy level)"""
        
        arousal_scores = {
            EmotionType.JOY: 0.8,
            EmotionType.ANGER: 0.9,
            EmotionType.FEAR: 0.8,
            EmotionType.EXCITEMENT: 1.0,
            EmotionType.SURPRISE: 0.9,
            EmotionType.SADNESS: 0.2,
            EmotionType.CALM: 0.1,
            EmotionType.LOVE: 0.5,
            EmotionType.DISGUST: 0.6,
            EmotionType.ANXIETY: 0.7
        }
        
        base_arousal = arousal_scores.get(primary_emotion, 0.5)
        return base_arousal * intensity
    
    def _calculate_dominance(self, primary_emotion: EmotionType, intensity: float) -> float:
        """Calculate emotional dominance (control/submission)"""
        
        dominance_scores = {
            EmotionType.ANGER: 0.7,
            EmotionType.JOY: 0.4,
            EmotionType.EXCITEMENT: 0.5,
            EmotionType.LOVE: 0.3,
            EmotionType.CALM: 0.2,
            EmotionType.SADNESS: -0.4,
            EmotionType.FEAR: -0.8,
            EmotionType.ANXIETY: -0.6,
            EmotionType.DISGUST: 0.1,
            EmotionType.SURPRISE: -0.2
        }
        
        base_dominance = dominance_scores.get(primary_emotion, 0.0)
        return base_dominance * intensity


class EmotionSynesthesiaEngine:
    """Main engine for emotion-to-synesthetic experiences"""
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.color_mapper = EmotionColorMapper()
        self.spatial_mapper = EmotionSpatialMapper()
        
        # Performance tracking
        self.translation_count = 0
        self.average_processing_time = 0.0
        
        logger.info("Emotion Synesthesia Engine initialized")
    
    async def create_synesthetic_experience_from_text(self, text: str,
                                                    include_spatial: bool = True,
                                                    user_profile: Optional[SynestheticProfile] = None) -> Dict[str, Any]:
        """Create complete synesthetic experience from text emotion analysis"""
        
        start_time = datetime.now()
        
        try:
            # Detect emotional state
            emotional_state = await self.emotion_detector.detect_emotion_from_text(text)
            
            # Generate color palette
            color_palette = self.color_mapper.map_emotion_to_color_palette(emotional_state)
            
            # Create spatial environment if requested
            spatial_environment = None
            if include_spatial:
                spatial_environment = self.spatial_mapper.create_emotional_environment(emotional_state)
            
            # Create synesthetic stimulus objects
            visual_stimulus = SynestheticStimulus(
                modality=ModalityType.VISUAL,
                data={
                    'color_palette': {
                        'primary': color_palette.primary_color,
                        'secondary': color_palette.secondary_colors,
                        'accents': color_palette.accent_colors,
                        'temperature': color_palette.temperature,
                        'saturation_profile': color_palette.saturation_profile,
                        'brightness_profile': color_palette.brightness_profile
                    },
                    'emotional_state': {
                        'primary_emotion': emotional_state.primary_emotion.value,
                        'intensity': emotional_state.intensity,
                        'valence': emotional_state.valence,
                        'arousal': emotional_state.arousal,
                        'dominance': emotional_state.dominance
                    }
                },
                metadata={'source': 'emotion_analysis', 'input_text': text},
                timestamp=datetime.now(),
                confidence=emotional_state.intensity
            )
            
            spatial_stimulus = None
            if spatial_environment:
                spatial_stimulus = SynestheticStimulus(
                    modality=ModalityType.SPATIAL,
                    data={
                        'spatial_metaphor': spatial_environment.spatial_metaphor,
                        'environmental_elements': spatial_environment.environmental_elements,
                        'lighting_scheme': spatial_environment.lighting_scheme,
                        'particle_systems': spatial_environment.particle_systems,
                        'atmospheric_effects': spatial_environment.atmospheric_effects
                    },
                    metadata={'source': 'emotion_spatial_mapping'},
                    timestamp=datetime.now(),
                    confidence=emotional_state.intensity * 0.9
                )
            
            # Apply user personalization
            if user_profile:
                visual_stimulus = await self._apply_user_color_preferences(visual_stimulus, user_profile)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            result = {
                'emotional_state': {
                    'primary_emotion': emotional_state.primary_emotion.value,
                    'intensity': emotional_state.intensity,
                    'secondary_emotions': [(e.value, i) for e, i in emotional_state.secondary_emotions],
                    'valence': emotional_state.valence,
                    'arousal': emotional_state.arousal,
                    'dominance': emotional_state.dominance
                },
                'visual_synesthesia': visual_stimulus.data,
                'spatial_synesthesia': spatial_stimulus.data if spatial_stimulus else None,
                'processing_metadata': {
                    'processing_time_ms': processing_time,
                    'input_text': text,
                    'confidence': emotional_state.intensity,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Emotion synesthesia experience created in {processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion synesthesia creation failed: {e}")
            raise
    
    async def create_synesthetic_experience_from_audio(self, audio_features: Dict[str, Any],
                                                     include_spatial: bool = True) -> Dict[str, Any]:
        """Create synesthetic experience from audio emotion analysis"""
        
        start_time = datetime.now()
        
        try:
            # Detect emotional state from audio
            emotional_state = await self.emotion_detector.detect_emotion_from_audio_features(audio_features)
            
            # Generate visual experience
            color_palette = self.color_mapper.map_emotion_to_color_palette(emotional_state)
            
            # Create spatial environment
            spatial_environment = None
            if include_spatial:
                spatial_environment = self.spatial_mapper.create_emotional_environment(emotional_state)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            return {
                'emotional_state': {
                    'primary_emotion': emotional_state.primary_emotion.value,
                    'intensity': emotional_state.intensity,
                    'valence': emotional_state.valence,
                    'arousal': emotional_state.arousal,
                    'dominance': emotional_state.dominance
                },
                'visual_synesthesia': {
                    'color_palette': {
                        'primary': color_palette.primary_color,
                        'secondary': color_palette.secondary_colors,
                        'accents': color_palette.accent_colors
                    }
                },
                'spatial_synesthesia': {
                    'environment': spatial_environment.__dict__ if spatial_environment else None
                },
                'processing_metadata': {
                    'processing_time_ms': processing_time,
                    'confidence': emotional_state.intensity,
                    'source': 'audio_analysis'
                }
            }
            
        except Exception as e:
            logger.error(f"Audio emotion synesthesia creation failed: {e}")
            raise
    
    async def _apply_user_color_preferences(self, visual_stimulus: SynestheticStimulus,
                                          profile: SynestheticProfile) -> SynestheticStimulus:
        """Apply user's color preferences to emotional visualization"""
        
        if not hasattr(profile, 'color_intensity'):
            return visual_stimulus
        
        # Apply color intensity preference
        intensity_factor = profile.color_intensity
        
        # Adjust primary color
        primary = visual_stimulus.data['color_palette']['primary']
        primary['saturation'] = min(1.0, primary['saturation'] * intensity_factor)
        primary['brightness'] = min(1.0, primary['brightness'] * intensity_factor)
        
        # Recalculate RGB
        rgb = colorsys.hsv_to_rgb(
            primary['hue'] / 360.0,
            primary['saturation'],
            primary['brightness']
        )
        primary['rgb'] = {'r': int(rgb[0] * 255), 'g': int(rgb[1] * 255), 'b': int(rgb[2] * 255)}
        primary['hex'] = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        # Adjust secondary colors
        for secondary in visual_stimulus.data['color_palette']['secondary']:
            secondary['saturation'] = min(1.0, secondary['saturation'] * intensity_factor)
            secondary['brightness'] = min(1.0, secondary['brightness'] * intensity_factor)
            
            rgb = colorsys.hsv_to_rgb(
                secondary['hue'] / 360.0,
                secondary['saturation'],
                secondary['brightness']
            )
            secondary['rgb'] = {'r': int(rgb[0] * 255), 'g': int(rgb[1] * 255), 'b': int(rgb[2] * 255)}
            secondary['hex'] = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        return visual_stimulus
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking"""
        
        self.translation_count += 1
        self.average_processing_time = (
            (self.average_processing_time * (self.translation_count - 1) + processing_time) /
            self.translation_count
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        
        return {
            'total_translations': self.translation_count,
            'average_processing_time_ms': round(self.average_processing_time, 2),
            'target_latency_ms': 120.0,
            'performance_status': 'optimal' if self.average_processing_time < 120 else 'degraded'
        }


# Global engine instance  
emotion_synesthesia_engine = EmotionSynesthesiaEngine()


async def get_emotion_synesthesia_engine() -> EmotionSynesthesiaEngine:
    """Dependency injection for emotion synesthesia engine"""
    return emotion_synesthesia_engine