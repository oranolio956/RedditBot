"""
Haptic Synthesizer - Advanced Tactile Feedback Generation

Implementation of cutting-edge haptic synesthesia research:
- Text-to-tactile generation with fingernail sensors + AudioLDM
- Real-time haptic pattern synthesis for VR/AR
- Cross-modal haptic feedback (audio -> touch, emotion -> texture)
- Multi-device haptic compatibility (Ultraleap, Tanvas, generic vibration)
"""

import asyncio
import numpy as np
import librosa
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import signal
from scipy.interpolate import interp1d

from app.core.config import get_settings
from app.models.synesthesia import SynestheticProfile, HapticPattern

logger = logging.getLogger(__name__)
settings = get_settings()


class HapticDeviceType(Enum):
    """Supported haptic device types"""
    ULTRALEAP = "ultraleap"              # Ultrasound haptics (mid-air)
    TANVAS = "tanvas"                    # Surface haptics (touchscreen)
    GENERIC_VIBRATION = "vibration"      # Standard vibration motors
    PNEUMATIC = "pneumatic"              # Air pressure haptics
    ELECTROTACTILE = "electrotactile"    # Electrical stimulation
    THERMAL = "thermal"                  # Temperature feedback


class HapticPrimitiveType(Enum):
    """Basic haptic sensations"""
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    TEXTURE = "texture"
    TEMPERATURE = "temperature"
    IMPACT = "impact"
    FLOW = "flow"
    SHAPE = "shape"


@dataclass
class HapticSensation:
    """Single haptic sensation with all parameters"""
    primitive_type: HapticPrimitiveType
    intensity: float                     # 0.0 to 1.0
    frequency_hz: Optional[float]        # For vibration/oscillation
    duration_ms: float
    spatial_position: Dict[str, float]   # x, y, z coordinates
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    waveform: str = "sine"              # sine, square, sawtooth, noise


@dataclass
class HapticSequence:
    """Temporal sequence of haptic sensations"""
    sensations: List[HapticSensation]
    total_duration_ms: float
    loop_count: int = 1
    synchronization_points: List[float] = None  # Time points for sync


class AudioToHapticTranslator:
    """Converts audio signals to authentic haptic experiences"""
    
    def __init__(self):
        self.sample_rate = 22050
        
        # Frequency ranges for different haptic sensations
        self.frequency_mapping = {
            # Bass frequencies -> pressure/impact sensations
            (20, 80): {'type': HapticPrimitiveType.IMPACT, 'base_intensity': 0.8},
            (80, 200): {'type': HapticPrimitiveType.PRESSURE, 'base_intensity': 0.7},
            
            # Mid frequencies -> vibration
            (200, 2000): {'type': HapticPrimitiveType.VIBRATION, 'base_intensity': 0.6},
            
            # High frequencies -> texture/fine detail
            (2000, 8000): {'type': HapticPrimitiveType.TEXTURE, 'base_intensity': 0.5},
            (8000, 20000): {'type': HapticPrimitiveType.FLOW, 'base_intensity': 0.3},
        }
        
        # Haptic device capabilities
        self.device_capabilities = {
            HapticDeviceType.ULTRALEAP: {
                'max_frequency': 1000.0,
                'spatial_resolution': {'x': 0.1, 'y': 0.1, 'z': 0.1},  # cm
                'intensity_levels': 256,
                'supports_spatial': True,
                'supports_texture': True
            },
            HapticDeviceType.TANVAS: {
                'max_frequency': 8000.0,
                'spatial_resolution': {'x': 0.01, 'y': 0.01, 'z': 0.0},
                'intensity_levels': 1024,
                'supports_spatial': True,
                'supports_texture': True
            },
            HapticDeviceType.GENERIC_VIBRATION: {
                'max_frequency': 500.0,
                'spatial_resolution': {'x': 1.0, 'y': 1.0, 'z': 0.0},
                'intensity_levels': 64,
                'supports_spatial': False,
                'supports_texture': False
            }
        }
    
    async def translate_audio_to_haptic(self, audio_data: np.ndarray,
                                      target_device: HapticDeviceType = HapticDeviceType.GENERIC_VIBRATION,
                                      user_profile: Optional[SynestheticProfile] = None) -> HapticSequence:
        """
        Convert audio waveform to haptic sensations
        
        Args:
            audio_data: Audio waveform as numpy array
            target_device: Target haptic device type
            user_profile: User's haptic preferences
            
        Returns:
            HapticSequence ready for device rendering
        """
        
        try:
            # Extract audio features
            features = await self._extract_haptic_features(audio_data)
            
            # Map frequency components to haptic sensations
            sensations = await self._map_frequencies_to_haptic(features, target_device)
            
            # Add temporal dynamics
            temporal_sensations = await self._add_temporal_dynamics(sensations, features)
            
            # Optimize for target device
            optimized_sensations = await self._optimize_for_device(temporal_sensations, target_device)
            
            # Apply user personalization
            if user_profile:
                optimized_sensations = await self._apply_haptic_personalization(
                    optimized_sensations, user_profile
                )
            
            # Calculate total duration
            total_duration = max([s.duration_ms + s.fade_out_ms for s in optimized_sensations]) if optimized_sensations else 0
            
            # Create synchronization points from beat tracking
            sync_points = await self._extract_synchronization_points(audio_data)
            
            sequence = HapticSequence(
                sensations=optimized_sensations,
                total_duration_ms=total_duration,
                synchronization_points=sync_points
            )
            
            logger.info(f"Generated haptic sequence: {len(optimized_sensations)} sensations, "
                       f"{total_duration:.1f}ms duration")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Audio-to-haptic translation failed: {e}")
            raise
    
    async def _extract_haptic_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract audio features relevant for haptic translation"""
        
        # Spectral analysis
        fft = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        magnitudes = np.abs(fft)
        
        # Only positive frequencies
        pos_mask = frequencies > 0
        pos_frequencies = frequencies[pos_mask]
        pos_magnitudes = magnitudes[pos_mask]
        
        # Temporal features
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=self.sample_rate)
        onset_times = librosa.onset.frames_to_time(onset_frames, sr=self.sample_rate)
        
        # Dynamic range
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        
        # Beat tracking
        tempo, beat_times = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        
        return {
            'frequencies': pos_frequencies,
            'magnitudes': pos_magnitudes,
            'onset_times': onset_times,
            'rms_energy': rms_energy,
            'tempo': tempo,
            'beat_times': beat_times,
            'total_duration': len(audio_data) / self.sample_rate
        }
    
    async def _map_frequencies_to_haptic(self, features: Dict[str, Any], 
                                       device_type: HapticDeviceType) -> List[HapticSensation]:
        """Map frequency components to haptic sensations"""
        
        sensations = []
        frequencies = features['frequencies']
        magnitudes = features['magnitudes']
        
        # Group frequencies by haptic mapping ranges
        for freq_range, haptic_config in self.frequency_mapping.items():
            min_freq, max_freq = freq_range
            
            # Find frequencies in this range
            in_range_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            range_frequencies = frequencies[in_range_mask]
            range_magnitudes = magnitudes[in_range_mask]
            
            if len(range_frequencies) > 0:
                # Calculate aggregate properties
                dominant_freq = range_frequencies[np.argmax(range_magnitudes)]
                total_energy = np.sum(range_magnitudes)
                
                # Create haptic sensation
                sensation = await self._create_sensation_from_frequency(
                    dominant_freq, total_energy, haptic_config, device_type
                )
                
                if sensation:
                    sensations.append(sensation)
        
        return sensations
    
    async def _create_sensation_from_frequency(self, frequency: float, 
                                             energy: float,
                                             haptic_config: Dict[str, Any],
                                             device_type: HapticDeviceType) -> Optional[HapticSensation]:
        """Create a haptic sensation from frequency and energy"""
        
        device_caps = self.device_capabilities[device_type]
        
        # Calculate intensity (normalize energy to 0-1 range)
        intensity = min(1.0, energy * haptic_config['base_intensity'] / 1000.0)
        
        if intensity < 0.05:  # Skip very weak sensations
            return None
        
        # Map frequency to haptic frequency (within device limits)
        haptic_frequency = min(frequency, device_caps['max_frequency'])
        
        # Calculate duration based on energy (more energy = longer sensation)
        duration_ms = 200 + (energy * 500)  # 200-700ms range
        
        # Spatial positioning (if device supports it)
        if device_caps['supports_spatial']:
            # Map frequency to spatial position (low freq = left, high freq = right)
            x_pos = (frequency - 20) / (20000 - 20)  # Normalize to 0-1
            spatial_pos = {'x': x_pos, 'y': 0.5, 'z': 0.0}
        else:
            spatial_pos = {'x': 0.5, 'y': 0.5, 'z': 0.0}  # Center position
        
        return HapticSensation(
            primitive_type=haptic_config['type'],
            intensity=intensity,
            frequency_hz=haptic_frequency if haptic_config['type'] == HapticPrimitiveType.VIBRATION else None,
            duration_ms=duration_ms,
            spatial_position=spatial_pos,
            fade_in_ms=50.0,
            fade_out_ms=100.0,
            waveform=self._select_waveform(haptic_config['type'])
        )
    
    async def _add_temporal_dynamics(self, sensations: List[HapticSensation], 
                                   features: Dict[str, Any]) -> List[HapticSensation]:
        """Add temporal dynamics based on audio rhythm and onsets"""
        
        temporal_sensations = []
        onset_times = features['onset_times']
        beat_times = features['beat_times']
        
        # Create onset-triggered sensations
        for onset_time in onset_times:
            # Find nearest beat
            if len(beat_times) > 0:
                nearest_beat_idx = np.argmin(np.abs(beat_times - onset_time))
                is_on_beat = np.abs(beat_times[nearest_beat_idx] - onset_time) < 0.1
            else:
                is_on_beat = False
            
            # Create impact sensation at onset
            impact_sensation = HapticSensation(
                primitive_type=HapticPrimitiveType.IMPACT,
                intensity=0.8 if is_on_beat else 0.5,
                duration_ms=100.0,
                spatial_position={'x': 0.5, 'y': 0.5, 'z': 0.0},
                fade_in_ms=0.0,
                fade_out_ms=50.0
            )
            
            temporal_sensations.append(impact_sensation)
        
        # Combine with frequency-based sensations
        all_sensations = sensations + temporal_sensations
        
        # Sort by start time (assume sensations start at time 0 for frequency-based)
        return sorted(all_sensations, key=lambda s: 0.0)  # Would need start_time field for proper sorting
    
    async def _optimize_for_device(self, sensations: List[HapticSensation], 
                                 device_type: HapticDeviceType) -> List[HapticSensation]:
        """Optimize haptic sensations for specific device capabilities"""
        
        device_caps = self.device_capabilities[device_type]
        optimized = []
        
        for sensation in sensations:
            # Quantize intensity to device levels
            quantized_intensity = round(sensation.intensity * device_caps['intensity_levels']) / device_caps['intensity_levels']
            
            # Clamp frequency to device limits
            if sensation.frequency_hz:
                clamped_frequency = min(sensation.frequency_hz, device_caps['max_frequency'])
            else:
                clamped_frequency = None
            
            # Adjust spatial resolution
            spatial_res = device_caps['spatial_resolution']
            quantized_position = {
                'x': round(sensation.spatial_position['x'] / spatial_res['x']) * spatial_res['x'],
                'y': round(sensation.spatial_position['y'] / spatial_res['y']) * spatial_res['y'],
                'z': round(sensation.spatial_position['z'] / spatial_res['z']) * spatial_res['z'] if spatial_res['z'] > 0 else 0.0
            }
            
            # Remove unsupported primitive types
            if not device_caps.get('supports_texture', True) and sensation.primitive_type == HapticPrimitiveType.TEXTURE:
                # Convert texture to vibration for devices that don't support it
                primitive_type = HapticPrimitiveType.VIBRATION
                if not clamped_frequency:
                    clamped_frequency = 200.0  # Default texture frequency
            else:
                primitive_type = sensation.primitive_type
            
            optimized_sensation = HapticSensation(
                primitive_type=primitive_type,
                intensity=quantized_intensity,
                frequency_hz=clamped_frequency,
                duration_ms=sensation.duration_ms,
                spatial_position=quantized_position,
                fade_in_ms=sensation.fade_in_ms,
                fade_out_ms=sensation.fade_out_ms,
                waveform=sensation.waveform
            )
            
            optimized.append(optimized_sensation)
        
        return optimized
    
    async def _apply_haptic_personalization(self, sensations: List[HapticSensation],
                                          profile: SynestheticProfile) -> List[HapticSensation]:
        """Apply user's haptic preferences"""
        
        if not hasattr(profile, 'texture_sensitivity'):
            return sensations
        
        sensitivity = profile.texture_sensitivity
        personalized = []
        
        for sensation in sensations:
            # Scale intensity based on user preference
            scaled_intensity = min(1.0, sensation.intensity * sensitivity)
            
            personalized_sensation = HapticSensation(
                primitive_type=sensation.primitive_type,
                intensity=scaled_intensity,
                frequency_hz=sensation.frequency_hz,
                duration_ms=sensation.duration_ms,
                spatial_position=sensation.spatial_position,
                fade_in_ms=sensation.fade_in_ms,
                fade_out_ms=sensation.fade_out_ms,
                waveform=sensation.waveform
            )
            
            personalized.append(personalized_sensation)
        
        return personalized
    
    async def _extract_synchronization_points(self, audio_data: np.ndarray) -> List[float]:
        """Extract key synchronization points from audio"""
        
        # Beat tracking
        tempo, beat_times = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        
        # Strong onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio_data, 
            sr=self.sample_rate,
            threshold=0.7,  # Only strong onsets
            pre_max=0.03,
            post_max=0.00,
            pre_avg=0.10,
            post_avg=0.10,
            delta=0.07,
            wait=0.03
        )
        onset_times = librosa.onset.frames_to_time(onset_frames, sr=self.sample_rate)
        
        # Combine and sort
        all_sync_points = np.concatenate([beat_times, onset_times])
        unique_sync_points = np.unique(all_sync_points)
        
        return unique_sync_points.tolist()
    
    def _select_waveform(self, primitive_type: HapticPrimitiveType) -> str:
        """Select appropriate waveform for haptic primitive"""
        
        waveform_map = {
            HapticPrimitiveType.VIBRATION: "sine",
            HapticPrimitiveType.PRESSURE: "square",
            HapticPrimitiveType.TEXTURE: "noise",
            HapticPrimitiveType.IMPACT: "impulse",
            HapticPrimitiveType.FLOW: "sawtooth",
            HapticPrimitiveType.SHAPE: "sine"
        }
        
        return waveform_map.get(primitive_type, "sine")


class TextToHapticMapper:
    """Maps text content to haptic textures and sensations"""
    
    def __init__(self):
        # Text-to-texture mappings based on lexical-gustatory research
        self.texture_mappings = {
            # Emotional textures
            'joy': {'type': HapticPrimitiveType.VIBRATION, 'freq': 15.0, 'intensity': 0.8, 'pattern': 'smooth_wave'},
            'sadness': {'type': HapticPrimitiveType.PRESSURE, 'freq': 3.0, 'intensity': 0.4, 'pattern': 'slow_pulses'},
            'anger': {'type': HapticPrimitiveType.TEXTURE, 'freq': 25.0, 'intensity': 1.0, 'pattern': 'sharp_spikes'},
            'fear': {'type': HapticPrimitiveType.VIBRATION, 'freq': 8.0, 'intensity': 0.6, 'pattern': 'trembling'},
            'surprise': {'type': HapticPrimitiveType.IMPACT, 'freq': 40.0, 'intensity': 0.9, 'pattern': 'sudden_burst'},
            'disgust': {'type': HapticPrimitiveType.TEXTURE, 'freq': 2.0, 'intensity': 0.3, 'pattern': 'rough_rejection'},
            
            # Consonant textures
            'hard_consonants': {'type': HapticPrimitiveType.TEXTURE, 'freq': 20.0, 'intensity': 0.7, 'pattern': 'sharp'},
            'soft_consonants': {'type': HapticPrimitiveType.FLOW, 'freq': 8.0, 'intensity': 0.5, 'pattern': 'smooth'},
            'liquid_consonants': {'type': HapticPrimitiveType.FLOW, 'freq': 12.0, 'intensity': 0.6, 'pattern': 'flowing'},
            
            # Vowel textures
            'open_vowels': {'type': HapticPrimitiveType.PRESSURE, 'freq': 5.0, 'intensity': 0.8, 'pattern': 'expanding'},
            'closed_vowels': {'type': HapticPrimitiveType.VIBRATION, 'freq': 15.0, 'intensity': 0.6, 'pattern': 'focused'},
        }
        
        # Word length effects
        self.length_effects = {
            'short': {'duration_multiplier': 0.5, 'intensity_boost': 0.2},
            'medium': {'duration_multiplier': 1.0, 'intensity_boost': 0.0},
            'long': {'duration_multiplier': 2.0, 'intensity_boost': -0.1}
        }
    
    async def translate_text_to_haptic(self, text: str, 
                                     target_device: HapticDeviceType = HapticDeviceType.GENERIC_VIBRATION) -> HapticSequence:
        """Convert text to haptic sensation sequence"""
        
        words = text.split()
        sensations = []
        current_time = 0.0
        
        for i, word in enumerate(words):
            if word.isalpha():
                # Analyze word characteristics
                word_textures = await self._analyze_word_texture(word)
                
                # Create haptic sensations for this word
                word_sensations = await self._create_word_haptics(
                    word_textures, current_time, target_device
                )
                
                sensations.extend(word_sensations)
                
                # Update timing (500ms per word + word-specific duration)
                word_duration = self._calculate_word_duration(word)
                current_time += word_duration + 100  # 100ms gap between words
        
        return HapticSequence(
            sensations=sensations,
            total_duration_ms=current_time
        )
    
    async def _analyze_word_texture(self, word: str) -> Dict[str, Any]:
        """Analyze phonetic and semantic texture of word"""
        
        word_lower = word.lower()
        
        # Count phonetic categories
        hard_consonants = len([c for c in word_lower if c in 'kgtpbdqx'])
        soft_consonants = len([c for c in word_lower if c in 'flmrwy'])
        liquid_consonants = len([c for c in word_lower if c in 'lrwy'])
        open_vowels = len([c for c in word_lower if c in 'aeiou'])
        closed_vowels = len([c for c in word_lower if c in 'iu'])
        
        # Determine dominant texture category
        texture_scores = {
            'hard_consonants': hard_consonants,
            'soft_consonants': soft_consonants,
            'liquid_consonants': liquid_consonants,
            'open_vowels': open_vowels,
            'closed_vowels': closed_vowels
        }
        
        dominant_texture = max(texture_scores.keys(), key=lambda k: texture_scores[k])
        
        # Determine word length category
        if len(word) <= 3:
            length_category = 'short'
        elif len(word) <= 7:
            length_category = 'medium'
        else:
            length_category = 'long'
        
        return {
            'word': word,
            'dominant_texture': dominant_texture,
            'length_category': length_category,
            'texture_scores': texture_scores,
            'total_length': len(word)
        }
    
    async def _create_word_haptics(self, word_texture: Dict[str, Any], 
                                 start_time_ms: float,
                                 target_device: HapticDeviceType) -> List[HapticSensation]:
        """Create haptic sensations for a single word"""
        
        sensations = []
        dominant_texture = word_texture['dominant_texture']
        length_category = word_texture['length_category']
        
        # Get base texture mapping
        texture_config = self.texture_mappings.get(dominant_texture, 
                                                  self.texture_mappings['soft_consonants'])
        
        # Get length effects
        length_effects = self.length_effects[length_category]
        
        # Calculate sensation parameters
        base_duration = 300.0 * length_effects['duration_multiplier']  # Base 300ms
        intensity = min(1.0, texture_config['intensity'] + length_effects['intensity_boost'])
        
        # Create primary sensation
        primary_sensation = HapticSensation(
            primitive_type=texture_config['type'],
            intensity=intensity,
            frequency_hz=texture_config['freq'],
            duration_ms=base_duration,
            spatial_position={'x': 0.5, 'y': 0.5, 'z': 0.0},
            fade_in_ms=50.0,
            fade_out_ms=100.0,
            waveform=self._pattern_to_waveform(texture_config['pattern'])
        )
        
        sensations.append(primary_sensation)
        
        # Add secondary sensations for complex words
        if length_category == 'long':
            # Add texture variation for long words
            secondary_sensation = HapticSensation(
                primitive_type=HapticPrimitiveType.TEXTURE,
                intensity=intensity * 0.6,
                frequency_hz=texture_config['freq'] * 0.7,
                duration_ms=base_duration * 0.8,
                spatial_position={'x': 0.3, 'y': 0.7, 'z': 0.0},
                fade_in_ms=100.0,
                fade_out_ms=150.0,
                waveform="noise"
            )
            
            sensations.append(secondary_sensation)
        
        return sensations
    
    def _calculate_word_duration(self, word: str) -> float:
        """Calculate haptic duration for a word based on its characteristics"""
        
        base_duration = 300.0  # 300ms base
        
        # Add time for each syllable (approximated by vowel clusters)
        vowel_clusters = len([c for c in word.lower() if c in 'aeiou'])
        syllable_time = vowel_clusters * 150.0  # 150ms per syllable
        
        # Add complexity time for consonant clusters
        consonants = len([c for c in word.lower() if c not in 'aeiou '])
        complexity_time = consonants * 20.0  # 20ms per consonant
        
        return base_duration + syllable_time + complexity_time
    
    def _pattern_to_waveform(self, pattern: str) -> str:
        """Convert texture pattern to waveform type"""
        
        pattern_waveforms = {
            'smooth_wave': 'sine',
            'slow_pulses': 'square',
            'sharp_spikes': 'sawtooth',
            'trembling': 'noise',
            'sudden_burst': 'impulse',
            'rough_rejection': 'noise',
            'sharp': 'square',
            'smooth': 'sine',
            'flowing': 'sawtooth',
            'expanding': 'sine',
            'focused': 'square'
        }
        
        return pattern_waveforms.get(pattern, 'sine')


class EmotionToHapticMapper:
    """Maps emotional states to haptic sensations"""
    
    def __init__(self):
        # Research-based emotion-haptic mappings
        self.emotion_haptic_map = {
            'joy': {
                'sensations': [
                    {'type': HapticPrimitiveType.VIBRATION, 'freq': 15.0, 'intensity': 0.8, 'duration': 500},
                    {'type': HapticPrimitiveType.FLOW, 'freq': 8.0, 'intensity': 0.6, 'duration': 800}
                ],
                'spatial_pattern': 'expanding_circles',
                'rhythm': 'upward_pulses'
            },
            'sadness': {
                'sensations': [
                    {'type': HapticPrimitiveType.PRESSURE, 'freq': 3.0, 'intensity': 0.5, 'duration': 1000}
                ],
                'spatial_pattern': 'downward_flow',
                'rhythm': 'slow_descent'
            },
            'anger': {
                'sensations': [
                    {'type': HapticPrimitiveType.TEXTURE, 'freq': 25.0, 'intensity': 1.0, 'duration': 300},
                    {'type': HapticPrimitiveType.IMPACT, 'freq': 50.0, 'intensity': 0.9, 'duration': 100}
                ],
                'spatial_pattern': 'sharp_bursts',
                'rhythm': 'aggressive_staccato'
            },
            'fear': {
                'sensations': [
                    {'type': HapticPrimitiveType.VIBRATION, 'freq': 12.0, 'intensity': 0.7, 'duration': 200}
                ],
                'spatial_pattern': 'trembling',
                'rhythm': 'irregular_rapid'
            }
        }
    
    async def emotion_to_haptic(self, emotion: str, intensity: float = 1.0,
                              duration_ms: float = 2000.0) -> HapticSequence:
        """Convert emotion to haptic sensation sequence"""
        
        if emotion not in self.emotion_haptic_map:
            emotion = 'joy'  # Default fallback
        
        emotion_config = self.emotion_haptic_map[emotion]
        sensations = []
        
        # Create base sensations
        for sensation_config in emotion_config['sensations']:
            sensation = HapticSensation(
                primitive_type=sensation_config['type'],
                intensity=sensation_config['intensity'] * intensity,
                frequency_hz=sensation_config['freq'],
                duration_ms=sensation_config['duration'],
                spatial_position={'x': 0.5, 'y': 0.5, 'z': 0.0},
                fade_in_ms=100.0,
                fade_out_ms=200.0
            )
            sensations.append(sensation)
        
        # Add spatial pattern variations
        spatial_sensations = await self._create_spatial_pattern(
            emotion_config['spatial_pattern'], intensity, duration_ms
        )
        sensations.extend(spatial_sensations)
        
        return HapticSequence(
            sensations=sensations,
            total_duration_ms=duration_ms,
            loop_count=1
        )
    
    async def _create_spatial_pattern(self, pattern_type: str, 
                                    intensity: float,
                                    duration_ms: float) -> List[HapticSensation]:
        """Create spatial haptic patterns"""
        
        sensations = []
        
        if pattern_type == 'expanding_circles':
            # Create concentric expanding sensations
            for i in range(3):
                radius = 0.2 + (i * 0.3)  # Expanding from center
                angle = (i * np.pi * 2 / 3)  # 120 degrees apart
                
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                
                sensation = HapticSensation(
                    primitive_type=HapticPrimitiveType.VIBRATION,
                    intensity=intensity * (1.0 - i * 0.2),  # Fade with distance
                    frequency_hz=10.0 + i * 5.0,
                    duration_ms=duration_ms / 2,
                    spatial_position={'x': x, 'y': y, 'z': 0.0},
                    fade_in_ms=i * 200.0,  # Staggered start
                    fade_out_ms=300.0
                )
                sensations.append(sensation)
        
        elif pattern_type == 'downward_flow':
            # Create flowing downward sensation
            for i in range(5):
                y_pos = 0.9 - (i * 0.2)  # Top to bottom
                
                sensation = HapticSensation(
                    primitive_type=HapticPrimitiveType.FLOW,
                    intensity=intensity * 0.6,
                    frequency_hz=5.0,
                    duration_ms=400.0,
                    spatial_position={'x': 0.5, 'y': y_pos, 'z': 0.0},
                    fade_in_ms=i * 100.0,  # Sequential activation
                    fade_out_ms=200.0
                )
                sensations.append(sensation)
        
        elif pattern_type == 'sharp_bursts':
            # Create sharp, random bursts
            for i in range(8):
                x_pos = np.random.uniform(0.2, 0.8)
                y_pos = np.random.uniform(0.2, 0.8)
                
                sensation = HapticSensation(
                    primitive_type=HapticPrimitiveType.IMPACT,
                    intensity=intensity * np.random.uniform(0.7, 1.0),
                    duration_ms=50.0,
                    spatial_position={'x': x_pos, 'y': y_pos, 'z': 0.0},
                    fade_in_ms=0.0,
                    fade_out_ms=30.0
                )
                sensations.append(sensation)
        
        elif pattern_type == 'trembling':
            # Create trembling pattern
            for i in range(20):
                # Small random offsets around center
                x_offset = np.random.uniform(-0.1, 0.1)
                y_offset = np.random.uniform(-0.1, 0.1)
                
                sensation = HapticSensation(
                    primitive_type=HapticPrimitiveType.VIBRATION,
                    intensity=intensity * 0.5 * np.random.uniform(0.8, 1.0),
                    frequency_hz=15.0 + np.random.uniform(-3.0, 3.0),
                    duration_ms=100.0,
                    spatial_position={
                        'x': 0.5 + x_offset, 
                        'y': 0.5 + y_offset, 
                        'z': 0.0
                    },
                    fade_in_ms=10.0,
                    fade_out_ms=20.0
                )
                sensations.append(sensation)
        
        return sensations


class HapticSynthesizer:
    """Main haptic synthesizer coordinating all haptic generation"""
    
    def __init__(self):
        self.audio_translator = AudioToHapticTranslator()
        self.text_mapper = TextToHapticMapper()
        self.emotion_mapper = EmotionToHapticMapper()
        
        # Performance tracking
        self.synthesis_count = 0
        self.average_latency = 0.0
        
        logger.info("Haptic Synthesizer initialized")
    
    async def synthesize_haptic_from_audio(self, audio_data: np.ndarray,
                                         device_type: HapticDeviceType = HapticDeviceType.GENERIC_VIBRATION,
                                         user_profile: Optional[SynestheticProfile] = None) -> HapticSequence:
        """Generate haptic sequence from audio input"""
        
        start_time = datetime.now()
        
        try:
            sequence = await self.audio_translator.translate_audio_to_haptic(
                audio_data, device_type, user_profile
            )
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Audio-to-haptic synthesis completed in {processing_time:.1f}ms")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Audio haptic synthesis failed: {e}")
            raise
    
    async def synthesize_haptic_from_text(self, text: str,
                                        device_type: HapticDeviceType = HapticDeviceType.GENERIC_VIBRATION,
                                        user_profile: Optional[SynestheticProfile] = None) -> HapticSequence:
        """Generate haptic sequence from text input"""
        
        start_time = datetime.now()
        
        try:
            sequence = await self.text_mapper.translate_text_to_haptic(text, device_type)
            
            # Apply user personalization if available
            if user_profile and hasattr(user_profile, 'texture_sensitivity'):
                # Scale intensities based on user preference
                for sensation in sequence.sensations:
                    sensation.intensity *= user_profile.texture_sensitivity
                    sensation.intensity = min(1.0, sensation.intensity)
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Text-to-haptic synthesis completed in {processing_time:.1f}ms")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Text haptic synthesis failed: {e}")
            raise
    
    async def synthesize_haptic_from_emotion(self, emotion: str, intensity: float = 1.0,
                                           duration_ms: float = 2000.0) -> HapticSequence:
        """Generate haptic sequence from emotional state"""
        
        start_time = datetime.now()
        
        try:
            sequence = await self.emotion_mapper.emotion_to_haptic(emotion, intensity, duration_ms)
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Emotion-to-haptic synthesis completed in {processing_time:.1f}ms")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Emotion haptic synthesis failed: {e}")
            raise
    
    def render_haptic_for_device(self, sequence: HapticSequence, 
                               device_type: HapticDeviceType) -> Dict[str, Any]:
        """Render haptic sequence for specific device output format"""
        
        device_caps = self.audio_translator.device_capabilities[device_type]
        
        # Convert to device-specific format
        if device_type == HapticDeviceType.GENERIC_VIBRATION:
            return self._render_for_vibration_motor(sequence)
        elif device_type == HapticDeviceType.ULTRALEAP:
            return self._render_for_ultraleap(sequence)
        elif device_type == HapticDeviceType.TANVAS:
            return self._render_for_tanvas(sequence)
        else:
            return self._render_generic_format(sequence)
    
    def _render_for_vibration_motor(self, sequence: HapticSequence) -> Dict[str, Any]:
        """Render for standard vibration motor (mobile/gamepad)"""
        
        vibration_patterns = []
        
        for sensation in sequence.sensations:
            # Convert to simple vibration pattern
            pattern = {
                'intensity': int(sensation.intensity * 255),  # 0-255 range
                'duration_ms': int(sensation.duration_ms),
                'delay_ms': int(sensation.fade_in_ms)
            }
            vibration_patterns.append(pattern)
        
        return {
            'device_type': 'vibration_motor',
            'patterns': vibration_patterns,
            'total_duration_ms': sequence.total_duration_ms
        }
    
    def _render_for_ultraleap(self, sequence: HapticSequence) -> Dict[str, Any]:
        """Render for Ultraleap ultrasound haptics"""
        
        control_points = []
        
        for sensation in sequence.sensations:
            # Convert to ultrasound control point
            point = {
                'position': {
                    'x': sensation.spatial_position['x'] * 200,  # Convert to mm
                    'y': sensation.spatial_position['y'] * 200,
                    'z': sensation.spatial_position['z'] * 200
                },
                'amplitude': sensation.intensity,
                'frequency': sensation.frequency_hz or 200.0,
                'duration_ms': sensation.duration_ms,
                'modulation': sensation.waveform
            }
            control_points.append(point)
        
        return {
            'device_type': 'ultraleap',
            'control_points': control_points,
            'total_duration_ms': sequence.total_duration_ms,
            'coordinate_system': 'device_space_mm'
        }
    
    def _render_for_tanvas(self, sequence: HapticSequence) -> Dict[str, Any]:
        """Render for Tanvas surface haptics"""
        
        surface_effects = []
        
        for sensation in sequence.sensations:
            # Convert to surface haptic effect
            effect = {
                'type': sensation.primitive_type.value,
                'position': {
                    'x': sensation.spatial_position['x'],  # 0-1 normalized
                    'y': sensation.spatial_position['y']
                },
                'intensity': sensation.intensity,
                'frequency': min(8000.0, sensation.frequency_hz or 1000.0),  # Tanvas limit
                'duration_ms': sensation.duration_ms,
                'texture_pattern': sensation.waveform
            }
            surface_effects.append(effect)
        
        return {
            'device_type': 'tanvas_surface',
            'effects': surface_effects,
            'total_duration_ms': sequence.total_duration_ms,
            'coordinate_system': 'normalized'
        }
    
    def _render_generic_format(self, sequence: HapticSequence) -> Dict[str, Any]:
        """Render in generic haptic format"""
        
        return {
            'device_type': 'generic',
            'sensations': [
                {
                    'type': s.primitive_type.value,
                    'intensity': s.intensity,
                    'frequency_hz': s.frequency_hz,
                    'duration_ms': s.duration_ms,
                    'position': s.spatial_position,
                    'fade_in_ms': s.fade_in_ms,
                    'fade_out_ms': s.fade_out_ms,
                    'waveform': s.waveform
                }
                for s in sequence.sensations
            ],
            'total_duration_ms': sequence.total_duration_ms,
            'synchronization_points': sequence.synchronization_points or []
        }
    
    def _update_performance_metrics(self, processing_time: float):
        """Update synthesis performance metrics"""
        
        self.synthesis_count += 1
        self.average_latency = (
            (self.average_latency * (self.synthesis_count - 1) + processing_time) /
            self.synthesis_count
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get synthesizer performance statistics"""
        
        return {
            'total_syntheses': self.synthesis_count,
            'average_latency_ms': round(self.average_latency, 2),
            'target_latency_ms': 100.0,  # Target for haptic synthesis
            'performance_status': 'optimal' if self.average_latency < 100 else 'degraded'
        }


# Global synthesizer instance
haptic_synthesizer = HapticSynthesizer()


async def get_haptic_synthesizer() -> HapticSynthesizer:
    """Dependency injection for haptic synthesizer"""
    return haptic_synthesizer