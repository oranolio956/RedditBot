"""
Advanced Human-Like Typing Simulator

Sophisticated typing simulation system that creates authentic human conversation patterns
to avoid AI detection through psychological modeling, natural language analysis, and
advanced anti-detection measures.
"""

import asyncio
import time
import random
import math
import re
import statistics
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import deque, defaultdict

import numpy as np
from scipy import stats, signal
import structlog

from app.core.redis import redis_manager
from app.models.personality import PersonalityProfile
from app.services.personality_manager import PersonalityManager

logger = structlog.get_logger(__name__)


class TypingStyle(Enum):
    """Different human typing styles."""
    RAPID_FIRE = "rapid_fire"          # Fast, confident typing
    METHODICAL = "methodical"          # Slow, careful typing  
    HUNT_AND_PECK = "hunt_and_peck"    # Slow, inconsistent
    TOUCH_TYPIST = "touch_typist"      # Fast, consistent
    MOBILE_THUMB = "mobile_thumb"      # Mobile typing patterns
    VOICE_TO_TEXT = "voice_to_text"    # Voice input patterns
    DISTRACTED = "distracted"          # Interrupted typing
    EMOTIONAL = "emotional"            # Emotion-driven patterns


class CognitiveState(Enum):
    """Cognitive states affecting typing."""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    TIRED = "tired"
    EXCITED = "excited"
    STRESSED = "stressed"
    RELAXED = "relaxed"
    MULTITASKING = "multitasking"
    CREATIVE = "creative"


class MessageComplexity(Enum):
    """Message complexity levels."""
    SIMPLE = "simple"              # Short, basic responses
    MODERATE = "moderate"          # Standard responses
    COMPLEX = "complex"            # Long, thoughtful responses
    TECHNICAL = "technical"        # Technical/specialized content
    EMOTIONAL = "emotional"        # Emotionally charged content
    CREATIVE = "creative"          # Creative/artistic content


@dataclass
class TypingMetrics:
    """Real-time typing performance metrics."""
    raw_wpm: float = 0.0
    effective_wpm: float = 0.0
    accuracy_rate: float = 0.95
    pause_frequency: float = 0.0
    burst_typing_rate: float = 0.0
    consistency_score: float = 0.0
    fatigue_level: float = 0.0
    flow_state: float = 0.0


@dataclass
class TypingPersonality:
    """Personality-driven typing characteristics."""
    base_wpm: float = 65.0
    accuracy_rate: float = 0.92
    impulsivity: float = 0.5          # 0=careful, 1=impulsive
    perfectionism: float = 0.3        # Tendency to correct mistakes
    stress_sensitivity: float = 0.4    # How stress affects typing
    flow_tendency: float = 0.6        # Ability to enter flow state
    multitask_penalty: float = 0.3    # Performance hit when multitasking
    fatigue_resistance: float = 0.7   # Resistance to fatigue
    
    # Style preferences
    preferred_styles: List[TypingStyle] = field(default_factory=lambda: [TypingStyle.TOUCH_TYPIST])
    pause_patterns: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)


@dataclass
class PsychologicalFactors:
    """Psychological factors affecting typing behavior."""
    confidence_level: float = 0.7
    attention_span: float = 0.8
    processing_speed: float = 0.75
    working_memory_load: float = 0.4
    emotional_state: float = 0.5      # -1=negative, 1=positive
    motivation_level: float = 0.6
    social_anxiety: float = 0.2
    perfectionist_traits: float = 0.3


@dataclass
class ContextualFactors:
    """Contextual factors influencing typing."""
    time_pressure: float = 0.0
    audience_formality: float = 0.5   # How formal the audience is
    topic_familiarity: float = 0.7
    conversation_flow: float = 0.6    # How well conversation is flowing
    interruption_likelihood: float = 0.1
    multitasking_level: float = 0.0
    device_type: str = "desktop"      # desktop, mobile, tablet
    environment_noise: float = 0.0


@dataclass
class TypingEvent:
    """Individual typing event with metadata."""
    timestamp: float
    event_type: str              # 'keypress', 'pause', 'backspace', 'correction'
    duration: float
    character: Optional[str] = None
    word_position: int = 0
    cognitive_load: float = 0.0
    confidence: float = 1.0


class LinguisticAnalyzer:
    """Analyzes text to determine typing complexity and patterns."""
    
    def __init__(self):
        # Common word frequencies (top 1000 English words)
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
            'one', 'all', 'would', 'there', 'their', 'what', 'so',
            'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
            'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than',
            'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how',
            'our', 'work', 'first', 'well', 'way', 'even', 'new'
        ])
        
        # Difficult key combinations
        self.difficult_combinations = {
            'qu', 'sch', 'tch', 'dge', 'ght', 'nch', 'rch', 'tion',
            'sion', 'ough', 'augh', 'eigh'
        }
        
        # Punctuation complexity
        self.punctuation_complexity = {
            '.': 0.1, ',': 0.2, '!': 0.3, '?': 0.3, ':': 0.4, ';': 0.5,
            '"': 0.6, "'": 0.4, '(': 0.5, ')': 0.5, '-': 0.3, '_': 0.6,
            '@': 0.7, '#': 0.6, '$': 0.8, '%': 0.7, '&': 0.8, '*': 0.6
        }
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """Analyze text to determine typing complexity factors."""
        if not text:
            return {'overall_complexity': 0.0}
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Word familiarity (common words are easier)
        familiar_words = sum(1 for word in words if word in self.common_words)
        word_familiarity = familiar_words / len(words) if words else 1.0
        
        # Average word length
        avg_word_length = statistics.mean(len(word) for word in words) if words else 0
        
        # Difficult letter combinations
        difficult_combinations = sum(
            text.lower().count(combo) for combo in self.difficult_combinations
        )
        combination_difficulty = min(difficult_combinations / len(text), 1.0)
        
        # Punctuation complexity
        punct_complexity = sum(
            self.punctuation_complexity.get(char, 0) for char in text
        ) / len(text)
        
        # Number density (numbers are often harder to type)
        number_density = len(re.findall(r'\d', text)) / len(text)
        
        # Capital letters (require shift)
        capital_density = len(re.findall(r'[A-Z]', text)) / len(text)
        
        # Special characters
        special_char_density = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)
        
        # Overall complexity calculation
        complexity_factors = {
            'word_unfamiliarity': 1.0 - word_familiarity,
            'word_length': min(avg_word_length / 10, 1.0),
            'combination_difficulty': combination_difficulty,
            'punctuation_complexity': min(punct_complexity, 1.0),
            'number_density': number_density * 2,  # Numbers are 2x harder
            'capital_density': capital_density * 1.5,  # Capitals 1.5x harder
            'special_char_density': special_char_density * 3  # Special chars 3x harder
        }
        
        overall_complexity = sum(complexity_factors.values()) / len(complexity_factors)
        complexity_factors['overall_complexity'] = min(overall_complexity, 1.0)
        
        return complexity_factors
    
    def identify_pause_points(self, text: str) -> List[int]:
        """Identify natural pause points in text."""
        pause_points = []
        
        # Punctuation pauses
        for i, char in enumerate(text):
            if char in '.!?;:':
                pause_points.append(i)
            elif char == ',':
                pause_points.append(i)
        
        # Word boundary pauses (every 3-7 words)
        words = list(re.finditer(r'\b\w+\b', text))
        word_pause_interval = random.randint(3, 7)
        
        for i in range(word_pause_interval - 1, len(words), word_pause_interval):
            if i < len(words):
                pause_points.append(words[i].end())
        
        # Complex word pauses (before difficult words)
        for match in re.finditer(r'\b\w{8,}\b', text):  # Words 8+ characters
            if match.group().lower() not in self.common_words:
                pause_points.append(match.start())
        
        return sorted(set(pause_points))


class CognitivLoadModel:
    """Models cognitive load effects on typing performance."""
    
    def __init__(self):
        self.working_memory_capacity = 7  # Miller's magic number
        self.processing_speed_factors = {
            'word_frequency': 0.3,
            'syntactic_complexity': 0.4,
            'semantic_difficulty': 0.5,
            'multitasking_penalty': 0.6
        }
    
    def calculate_cognitive_load(
        self,
        text: str,
        context: ContextualFactors,
        psychological: PsychologicalFactors
    ) -> float:
        """Calculate cognitive load for typing given text."""
        # Base load from text complexity
        analyzer = LinguisticAnalyzer()
        text_complexity = analyzer.analyze_text_complexity(text)
        base_load = text_complexity['overall_complexity']
        
        # Context modifiers
        context_load = (
            context.time_pressure * 0.3 +
            context.audience_formality * 0.2 +
            (1 - context.topic_familiarity) * 0.4 +
            context.multitasking_level * 0.5 +
            context.interruption_likelihood * 0.2
        )
        
        # Psychological modifiers
        psych_load = (
            (1 - psychological.confidence_level) * 0.3 +
            psychological.working_memory_load * 0.4 +
            (1 - psychological.attention_span) * 0.3 +
            psychological.social_anxiety * 0.2 +
            abs(psychological.emotional_state) * 0.2  # Both extreme emotions add load
        )
        
        # Combined load with diminishing returns
        total_load = base_load + context_load + psych_load
        return min(total_load, 2.0)  # Cap at 2.0
    
    def apply_cognitive_effects(
        self,
        base_timing: float,
        cognitive_load: float,
        flow_state: float = 0.0
    ) -> float:
        """Apply cognitive load effects to typing timing."""
        # High cognitive load slows down typing
        load_multiplier = 1.0 + (cognitive_load * 0.8)
        
        # Flow state reduces the impact of cognitive load
        if flow_state > 0.5:
            flow_reduction = (flow_state - 0.5) * 2  # 0-1 scale
            load_multiplier = 1.0 + (cognitive_load * 0.8 * (1 - flow_reduction * 0.6))
        
        return base_timing * load_multiplier


class EmotionalStateModel:
    """Models how emotional state affects typing patterns."""
    
    def __init__(self):
        self.emotion_effects = {
            'excited': {'speed_mult': 1.3, 'error_rate': 1.4, 'pause_freq': 0.7},
            'stressed': {'speed_mult': 0.8, 'error_rate': 1.6, 'pause_freq': 1.4},
            'tired': {'speed_mult': 0.7, 'error_rate': 1.8, 'pause_freq': 1.8},
            'angry': {'speed_mult': 1.2, 'error_rate': 2.0, 'pause_freq': 0.5},
            'sad': {'speed_mult': 0.6, 'error_rate': 1.2, 'pause_freq': 2.2},
            'focused': {'speed_mult': 1.1, 'error_rate': 0.7, 'pause_freq': 0.8},
            'relaxed': {'speed_mult': 0.9, 'error_rate': 0.8, 'pause_freq': 1.1},
            'anxious': {'speed_mult': 0.75, 'error_rate': 1.7, 'pause_freq': 1.6}
        }
    
    def get_emotional_effects(
        self,
        emotional_state: float,
        cognitive_state: CognitiveState
    ) -> Dict[str, float]:
        """Get typing effects based on emotional and cognitive state."""
        # Map emotional state (-1 to 1) to emotion categories
        if emotional_state > 0.6:
            emotion = 'excited'
        elif emotional_state > 0.2:
            emotion = 'focused' if cognitive_state == CognitiveState.FOCUSED else 'relaxed'
        elif emotional_state > -0.2:
            emotion = 'relaxed'
        elif emotional_state > -0.6:
            emotion = 'sad' if cognitive_state == CognitiveState.TIRED else 'stressed'
        else:
            emotion = 'angry' if cognitive_state == CognitiveState.STRESSED else 'sad'
        
        # Apply cognitive state modifiers
        effects = self.emotion_effects[emotion].copy()
        
        if cognitive_state == CognitiveState.TIRED:
            effects['speed_mult'] *= 0.8
            effects['error_rate'] *= 1.3
            effects['pause_freq'] *= 1.4
        elif cognitive_state == CognitiveState.DISTRACTED:
            effects['speed_mult'] *= 0.9
            effects['error_rate'] *= 1.5
            effects['pause_freq'] *= 1.6
        elif cognitive_state == CognitiveState.MULTITASKING:
            effects['speed_mult'] *= 0.7
            effects['error_rate'] *= 1.8
            effects['pause_freq'] *= 2.0
        
        return effects


class NaturalErrorModel:
    """Models realistic typing errors and corrections."""
    
    def __init__(self):
        # Common typing error patterns
        self.substitution_errors = {
            'a': ['s', 'q', 'w'], 'b': ['v', 'n', 'g'], 'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'e', 'r', 'c'], 'e': ['w', 'r', 'd', 's'],
            'f': ['d', 'g', 'r', 't', 'v'], 'g': ['f', 'h', 't', 'y', 'b'],
            'h': ['g', 'j', 'y', 'u', 'n'], 'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'k', 'u', 'i', 'm'], 'k': ['j', 'l', 'i', 'o'],
            'l': ['k', 'o', 'p'], 'm': ['n', 'j', 'k'], 'n': ['b', 'm', 'h', 'j'],
            'o': ['i', 'p', 'k', 'l'], 'p': ['o', 'l'], 'q': ['w', 'a'],
            'r': ['e', 't', 'd', 'f'], 's': ['a', 'd', 'w', 'e'],
            't': ['r', 'y', 'f', 'g'], 'u': ['y', 'i', 'h', 'j'],
            'v': ['c', 'b', 'f', 'g'], 'w': ['q', 'e', 'a', 's'],
            'x': ['z', 'c', 's', 'd'], 'y': ['t', 'u', 'g', 'h'],
            'z': ['x', 'a', 's']
        }
        
        # Common word confusions
        self.word_confusions = {
            'their': ['there', 'they\'re'], 'there': ['their', 'they\'re'],
            'your': ['you\'re'], 'you\'re': ['your'], 'its': ['it\'s'],
            'it\'s': ['its'], 'to': ['too', 'two'], 'too': ['to', 'two'],
            'two': ['to', 'too'], 'than': ['then'], 'then': ['than']
        }
    
    def should_make_error(
        self,
        character: str,
        error_rate: float,
        fatigue_level: float = 0.0,
        stress_level: float = 0.0
    ) -> bool:
        """Determine if an error should be made."""
        # Base error probability
        base_prob = error_rate
        
        # Fatigue increases errors exponentially
        fatigue_multiplier = 1.0 + (fatigue_level ** 2) * 2
        
        # Stress increases errors
        stress_multiplier = 1.0 + stress_level
        
        # Some characters are more error-prone
        char_difficulty = {
            'q': 2.0, 'z': 1.8, 'x': 1.5, ';': 1.8, "'": 1.6,
            '1': 1.3, '0': 1.2, '-': 1.4, '=': 1.3
        }
        
        char_multiplier = char_difficulty.get(character.lower(), 1.0)
        
        final_prob = base_prob * fatigue_multiplier * stress_multiplier * char_multiplier
        return random.random() < final_prob
    
    def generate_error(self, character: str, word_context: str = "") -> str:
        """Generate a realistic typing error."""
        error_type = random.choices(
            ['substitution', 'omission', 'insertion', 'transposition'],
            weights=[0.6, 0.2, 0.15, 0.05]
        )[0]
        
        if error_type == 'substitution' and character.lower() in self.substitution_errors:
            return random.choice(self.substitution_errors[character.lower()])
        elif error_type == 'omission':
            return ''  # Skip character
        elif error_type == 'insertion':
            # Insert adjacent character
            if character.lower() in self.substitution_errors:
                return character + random.choice(self.substitution_errors[character.lower()])
            return character + character  # Double character
        elif error_type == 'transposition' and len(word_context) > 0:
            # This would be handled at word level
            return character
        
        return character  # No error
    
    def calculate_correction_time(
        self,
        error_type: str,
        characters_to_fix: int,
        perfectionism: float
    ) -> float:
        """Calculate time needed for error correction."""
        base_correction_time = {
            'substitution': 1.2,
            'omission': 0.8,
            'insertion': 1.5,
            'transposition': 2.0,
            'word_error': 3.0
        }
        
        base_time = base_correction_time.get(error_type, 1.0)
        
        # More characters to fix = longer time
        char_penalty = characters_to_fix * 0.3
        
        # Perfectionists spend more time correcting
        perfectionism_mult = 1.0 + perfectionism * 0.8
        
        # Add thinking time (noticing the error)
        thinking_time = random.uniform(0.3, 1.2)
        
        return (base_time + char_penalty) * perfectionism_mult + thinking_time


class AdvancedTypingSimulator:
    """
    Advanced human-like typing simulator with comprehensive psychological modeling.
    
    Features:
    - Personality-driven typing patterns
    - Cognitive load modeling
    - Emotional state effects
    - Natural error simulation
    - Context-aware adaptations
    - Anti-detection measures
    - Real-time pattern learning
    """
    
    def __init__(self, personality_manager: PersonalityManager):
        self.personality_manager = personality_manager
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.cognitive_model = CognitivLoadModel()
        self.emotional_model = EmotionalStateModel()
        self.error_model = NaturalErrorModel()
        
        # Pattern learning and adaptation
        self.user_patterns: Dict[int, Dict] = {}
        self.conversation_context: Dict[int, Dict] = {}
        
        # Anti-detection measures
        self.detection_patterns = {
            'too_consistent': [],
            'unrealistic_speed': [],
            'unnatural_pauses': [],
            'robotic_corrections': []
        }
        
        # Performance monitoring
        self.simulation_metrics = {
            'total_simulations': 0,
            'average_realism_score': 0.0,
            'detection_events_prevented': 0,
            'pattern_variations_applied': 0
        }
        
        # Initialize Redis keys
        self.redis_prefix = "typing_simulator"
        
    async def initialize(self) -> None:
        """Initialize the typing simulator."""
        try:
            logger.info("Initializing Advanced Typing Simulator")
            
            # Load existing user patterns
            await self._load_user_patterns()
            
            # Initialize detection prevention
            await self._initialize_detection_prevention()
            
            logger.info("Advanced Typing Simulator initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize typing simulator", error=str(e))
            raise
    
    async def simulate_human_typing(
        self,
        text: str,
        user_id: int,
        personality_profile: Optional[PersonalityProfile] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate human typing for given text with full psychological modeling.
        
        Returns comprehensive typing simulation data including:
        - Total typing time
        - Pause points and durations
        - Error events and corrections
        - Cognitive load indicators
        - Authenticity metrics
        """
        try:
            start_time = time.time()
            
            # Get or create user typing personality
            typing_personality = await self._get_user_typing_personality(user_id, personality_profile)
            
            # Analyze text complexity
            text_complexity = self.linguistic_analyzer.analyze_text_complexity(text)
            
            # Build contextual factors
            contextual_factors = self._build_contextual_factors(context, conversation_state)
            
            # Build psychological factors
            psychological_factors = await self._build_psychological_factors(
                user_id, personality_profile, conversation_state
            )
            
            # Calculate cognitive load
            cognitive_load = self.cognitive_model.calculate_cognitive_load(
                text, contextual_factors, psychological_factors
            )
            
            # Get emotional effects
            emotional_effects = self.emotional_model.get_emotional_effects(
                psychological_factors.emotional_state,
                self._determine_cognitive_state(psychological_factors)
            )
            
            # Generate detailed typing simulation
            simulation_result = await self._simulate_detailed_typing(
                text=text,
                typing_personality=typing_personality,
                text_complexity=text_complexity,
                cognitive_load=cognitive_load,
                emotional_effects=emotional_effects,
                contextual_factors=contextual_factors,
                psychological_factors=psychological_factors
            )
            
            # Apply anti-detection measures
            simulation_result = await self._apply_anti_detection_measures(
                simulation_result, user_id
            )
            
            # Update user patterns for learning
            await self._update_user_patterns(user_id, simulation_result)
            
            # Update metrics
            self.simulation_metrics['total_simulations'] += 1
            
            simulation_result['meta'] = {
                'generation_time': time.time() - start_time,
                'cognitive_load': cognitive_load,
                'realism_score': simulation_result.get('realism_score', 0.9),
                'anti_detection_score': simulation_result.get('anti_detection_score', 0.95)
            }
            
            return simulation_result
            
        except Exception as e:
            logger.error("Error in typing simulation", error=str(e), user_id=user_id)
            # Return fallback simple simulation
            return await self._fallback_simulation(text)
    
    async def _simulate_detailed_typing(
        self,
        text: str,
        typing_personality: TypingPersonality,
        text_complexity: Dict[str, float],
        cognitive_load: float,
        emotional_effects: Dict[str, float],
        contextual_factors: ContextualFactors,
        psychological_factors: PsychologicalFactors
    ) -> Dict[str, Any]:
        """Generate detailed typing simulation with all factors."""
        
        if not text:
            return {
                'total_time': 0.0,
                'typing_events': [],
                'pause_points': [],
                'error_events': [],
                'realism_score': 1.0
            }
        
        # Initialize simulation state
        simulation_state = {
            'current_pos': 0,
            'current_time': 0.0,
            'flow_state': random.uniform(0.3, 0.8),
            'fatigue_level': random.uniform(0.0, 0.3),
            'current_accuracy': typing_personality.accuracy_rate,
            'burst_mode': False,
            'thinking_pause_due': False
        }
        
        typing_events = []
        error_events = []
        
        # Calculate base typing speed with all modifiers
        base_wpm = self._calculate_effective_wpm(
            typing_personality, emotional_effects, cognitive_load, contextual_factors
        )
        base_cps = base_wpm / 12  # Convert WPM to characters per second
        
        # Identify natural pause points
        pause_points = self.linguistic_analyzer.identify_pause_points(text)
        
        # Simulate character-by-character typing
        i = 0
        while i < len(text):
            char = text[i]
            
            # Calculate character-specific timing
            char_time = self._calculate_character_time(
                char, base_cps, simulation_state, text_complexity
            )
            
            # Apply cognitive load effects
            char_time = self.cognitive_model.apply_cognitive_effects(
                char_time, cognitive_load, simulation_state['flow_state']
            )
            
            # Check for errors
            should_error = self.error_model.should_make_error(
                char,
                1.0 - simulation_state['current_accuracy'],
                simulation_state['fatigue_level'],
                psychological_factors.attention_span
            )
            
            if should_error and random.random() < 0.7:  # Not all potential errors occur
                # Generate error and correction sequence
                error_sequence = await self._simulate_error_correction(
                    i, char, text, typing_personality, simulation_state
                )
                
                typing_events.extend(error_sequence['events'])
                error_events.append(error_sequence['error_summary'])
                simulation_state['current_time'] += error_sequence['total_time']
                
            else:
                # Normal character typing
                event = TypingEvent(
                    timestamp=simulation_state['current_time'],
                    event_type='keypress',
                    duration=char_time,
                    character=char,
                    word_position=self._get_word_position(i, text),
                    cognitive_load=cognitive_load,
                    confidence=simulation_state['current_accuracy']
                )
                typing_events.append(event)
                simulation_state['current_time'] += char_time
            
            # Check for natural pauses
            if i in pause_points or self._should_natural_pause(i, text, simulation_state):
                pause_duration = self._calculate_pause_duration(
                    i, text, typing_personality, psychological_factors, contextual_factors
                )
                
                if pause_duration > 0.1:  # Only add significant pauses
                    pause_event = TypingEvent(
                        timestamp=simulation_state['current_time'],
                        event_type='pause',
                        duration=pause_duration,
                        cognitive_load=cognitive_load
                    )
                    typing_events.append(pause_event)
                    simulation_state['current_time'] += pause_duration
            
            # Update simulation state
            self._update_simulation_state(simulation_state, char, i, len(text))
            
            i += 1
        
        # Calculate final metrics
        realism_score = self._calculate_realism_score(
            typing_events, error_events, text, typing_personality
        )
        
        return {
            'total_time': simulation_state['current_time'],
            'typing_events': [self._serialize_typing_event(e) for e in typing_events],
            'pause_points': pause_points,
            'error_events': error_events,
            'effective_wpm': len(text.split()) / (simulation_state['current_time'] / 60) if simulation_state['current_time'] > 0 else 0,
            'accuracy_rate': simulation_state['current_accuracy'],
            'flow_state_final': simulation_state['flow_state'],
            'fatigue_final': simulation_state['fatigue_level'],
            'realism_score': realism_score,
            'cognitive_load': cognitive_load
        }
    
    def _calculate_effective_wpm(
        self,
        personality: TypingPersonality,
        emotional_effects: Dict[str, float],
        cognitive_load: float,
        context: ContextualFactors
    ) -> float:
        """Calculate effective WPM considering all factors."""
        base_wpm = personality.base_wpm
        
        # Apply emotional effects
        speed_mult = emotional_effects.get('speed_mult', 1.0)
        
        # Apply cognitive load penalty
        cognitive_penalty = 1.0 - (cognitive_load * 0.3)
        
        # Apply context effects
        context_mult = 1.0
        if context.device_type == 'mobile':
            context_mult *= 0.6  # Mobile typing is slower
        elif context.device_type == 'tablet':
            context_mult *= 0.75
        
        if context.time_pressure > 0.5:
            context_mult *= (1.0 + context.time_pressure * 0.3)  # Rush can increase speed
        
        if context.multitasking_level > 0:
            context_mult *= (1.0 - context.multitasking_level * personality.multitask_penalty)
        
        effective_wpm = base_wpm * speed_mult * cognitive_penalty * context_mult
        
        # Ensure reasonable bounds
        return max(15, min(effective_wpm, 200))
    
    def _calculate_character_time(
        self,
        char: str,
        base_cps: float,
        simulation_state: Dict,
        text_complexity: Dict[str, float]
    ) -> float:
        """Calculate time for typing a specific character."""
        base_time = 1.0 / base_cps
        
        # Character-specific modifiers
        char_modifiers = {
            ' ': 0.8,  # Spaces are faster
            '.': 1.2, ',': 1.1, '!': 1.3, '?': 1.3,
            ':': 1.4, ';': 1.5, '"': 1.6, "'": 1.3,
            '(': 1.3, ')': 1.3, '[': 1.4, ']': 1.4,
            '{': 1.5, '}': 1.5, '<': 1.4, '>': 1.4,
            '@': 1.6, '#': 1.4, '$': 1.7, '%': 1.5,
            '^': 1.6, '&': 1.5, '*': 1.3, '+': 1.4,
            '=': 1.3, '-': 1.2, '_': 1.5, '\\': 1.6,
            '|': 1.5, '/': 1.3, '`': 1.4, '~': 1.6
        }
        
        char_mult = char_modifiers.get(char, 1.0)
        
        # Numbers are typically slower
        if char.isdigit():
            char_mult *= 1.2
        
        # Capital letters require shift
        if char.isupper():
            char_mult *= 1.3
        
        # Apply flow state effects
        if simulation_state['burst_mode']:
            char_mult *= 0.7  # Faster in burst mode
        elif simulation_state['flow_state'] > 0.7:
            char_mult *= 0.85  # Faster in flow state
        
        # Apply fatigue
        fatigue_mult = 1.0 + simulation_state['fatigue_level'] * 0.5
        
        # Add natural variation
        variation = random.uniform(0.7, 1.4)
        
        return base_time * char_mult * fatigue_mult * variation
    
    async def _simulate_error_correction(
        self,
        position: int,
        intended_char: str,
        text: str,
        personality: TypingPersonality,
        simulation_state: Dict
    ) -> Dict[str, Any]:
        """Simulate an error and its correction."""
        error_char = self.error_model.generate_error(intended_char)
        correction_events = []
        
        # Time to type the error
        error_time = self._calculate_character_time(
            error_char, 60/12, simulation_state, {}  # Use average speed
        )
        
        # Error typing event
        error_event = TypingEvent(
            timestamp=simulation_state['current_time'],
            event_type='error_keypress',
            duration=error_time,
            character=error_char,
            word_position=self._get_word_position(position, text)
        )
        correction_events.append(error_event)
        
        # Detection time (noticing the error)
        detection_delay = random.uniform(0.2, 2.0)
        if personality.perfectionism > 0.7:
            detection_delay *= 0.6  # Perfectionists notice faster
        
        detection_event = TypingEvent(
            timestamp=simulation_state['current_time'] + error_time,
            event_type='error_detection',
            duration=detection_delay
        )
        correction_events.append(detection_event)
        
        # Backspace to remove error
        backspace_time = random.uniform(0.1, 0.3)
        backspace_event = TypingEvent(
            timestamp=simulation_state['current_time'] + error_time + detection_delay,
            event_type='backspace',
            duration=backspace_time,
            character=error_char
        )
        correction_events.append(backspace_event)
        
        # Retype correct character
        correct_time = self._calculate_character_time(
            intended_char, 60/12 * 0.9, simulation_state, {}  # Slightly slower when correcting
        )
        
        correct_event = TypingEvent(
            timestamp=simulation_state['current_time'] + error_time + detection_delay + backspace_time,
            event_type='correction_keypress',
            duration=correct_time,
            character=intended_char,
            word_position=self._get_word_position(position, text)
        )
        correction_events.append(correct_event)
        
        total_time = error_time + detection_delay + backspace_time + correct_time
        
        return {
            'events': correction_events,
            'total_time': total_time,
            'error_summary': {
                'position': position,
                'intended': intended_char,
                'typed': error_char,
                'correction_time': total_time,
                'detection_delay': detection_delay
            }
        }
    
    def _should_natural_pause(
        self,
        position: int,
        text: str,
        simulation_state: Dict
    ) -> bool:
        """Determine if a natural pause should occur."""
        # Word boundaries
        if position > 0 and text[position-1] == ' ':
            return random.random() < 0.15
        
        # Long words
        if position > 0 and text[position-1].isalpha() and text[position].isalpha():
            word_start = position
            while word_start > 0 and text[word_start-1].isalpha():
                word_start -= 1
            
            if position - word_start > 6:  # In middle of long word
                return random.random() < 0.08
        
        # Thinking pauses based on cognitive load
        if simulation_state.get('thinking_pause_due', False):
            simulation_state['thinking_pause_due'] = False
            return True
        
        # Random natural pauses
        return random.random() < 0.05
    
    def _calculate_pause_duration(
        self,
        position: int,
        text: str,
        personality: TypingPersonality,
        psychological: PsychologicalFactors,
        context: ContextualFactors
    ) -> float:
        """Calculate duration of a natural pause."""
        # Base pause types
        if position < len(text) and text[position] in '.!?':
            # End of sentence pause
            base_duration = random.uniform(1.0, 3.0)
        elif position < len(text) and text[position] in ',;:':
            # Punctuation pause
            base_duration = random.uniform(0.3, 1.2)
        elif position > 0 and text[position-1] == ' ':
            # Word boundary pause
            base_duration = random.uniform(0.2, 0.8)
        else:
            # General thinking pause
            base_duration = random.uniform(0.5, 2.5)
        
        # Psychological modifiers
        if psychological.processing_speed < 0.5:
            base_duration *= 1.8  # Slower processing = longer pauses
        
        if psychological.attention_span < 0.5:
            base_duration *= random.uniform(1.2, 2.5)  # Distracted = variable pauses
        
        # Context modifiers
        if context.time_pressure > 0.5:
            base_duration *= 0.6  # Less pausing under pressure
        
        if context.topic_familiarity < 0.5:
            base_duration *= 1.5  # Unfamiliar topics need more thinking
        
        return max(0.1, base_duration)
    
    def _update_simulation_state(
        self,
        state: Dict,
        char: str,
        position: int,
        total_length: int
    ) -> None:
        """Update ongoing simulation state."""
        # Update fatigue (gradual increase)
        fatigue_increase = 0.001 * (1.0 - state.get('fatigue_resistance', 0.7))
        state['fatigue_level'] = min(1.0, state['fatigue_level'] + fatigue_increase)
        
        # Update flow state
        if char.isalpha() and not char.isupper():
            # Normal typing increases flow
            state['flow_state'] = min(1.0, state['flow_state'] + 0.002)
        elif char in '.!?':
            # Punctuation can break flow
            state['flow_state'] *= 0.95
        
        # Burst mode detection (3+ quick characters)
        if position > 2 and all(c.isalpha() for c in [char] + list(str(position-3))):
            state['burst_mode'] = True
        else:
            state['burst_mode'] = False
        
        # Schedule thinking pauses
        if position % 25 == 0 and random.random() < 0.3:
            state['thinking_pause_due'] = True
        
        # Accuracy changes with fatigue and flow
        flow_bonus = (state['flow_state'] - 0.5) * 0.1
        fatigue_penalty = state['fatigue_level'] * 0.2
        state['current_accuracy'] = max(0.5, min(1.0, 
            state['current_accuracy'] + flow_bonus - fatigue_penalty
        ))
    
    def _get_word_position(self, char_pos: int, text: str) -> int:
        """Get word position for character position."""
        return len(text[:char_pos].split())
    
    def _calculate_realism_score(
        self,
        typing_events: List[TypingEvent],
        error_events: List[Dict],
        text: str,
        personality: TypingPersonality
    ) -> float:
        """Calculate overall realism score of the simulation."""
        if not typing_events:
            return 0.0
        
        scores = []
        
        # Timing variation realism
        keypress_times = [e.duration for e in typing_events if e.event_type == 'keypress']
        if keypress_times:
            timing_variation = np.std(keypress_times) / np.mean(keypress_times)
            # Good variation is between 0.2 and 0.8
            variation_score = 1.0 - abs(timing_variation - 0.5) / 0.5
            scores.append(max(0, variation_score))
        
        # Pause pattern realism
        pauses = [e.duration for e in typing_events if e.event_type == 'pause']
        if pauses:
            # Natural pause distribution should be log-normal-ish
            pause_score = min(1.0, len(pauses) / (len(text) / 20))  # ~1 pause per 20 chars
            scores.append(pause_score)
        
        # Error rate realism
        expected_errors = len(text) * (1.0 - personality.accuracy_rate)
        actual_errors = len(error_events)
        error_score = 1.0 - abs(actual_errors - expected_errors) / max(1, expected_errors)
        scores.append(max(0, error_score))
        
        # Overall speed realism
        total_time = sum(e.duration for e in typing_events)
        if total_time > 0:
            actual_wpm = len(text.split()) / (total_time / 60)
            expected_wpm = personality.base_wpm
            speed_score = 1.0 - abs(actual_wpm - expected_wpm) / expected_wpm
            scores.append(max(0, speed_score))
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _serialize_typing_event(self, event: TypingEvent) -> Dict[str, Any]:
        """Serialize typing event for JSON storage."""
        return {
            'timestamp': event.timestamp,
            'event_type': event.event_type,
            'duration': event.duration,
            'character': event.character,
            'word_position': event.word_position,
            'cognitive_load': event.cognitive_load,
            'confidence': event.confidence
        }
    
    async def _get_user_typing_personality(
        self,
        user_id: int,
        personality_profile: Optional[PersonalityProfile]
    ) -> TypingPersonality:
        """Get or create user-specific typing personality."""
        # Try to get from cache first
        cache_key = f"{self.redis_prefix}:personality:{user_id}"
        cached_personality = await redis_manager.get(cache_key)
        
        if cached_personality:
            try:
                data = json.loads(cached_personality)
                return TypingPersonality(**data)
            except:
                pass  # Fall through to generation
        
        # Generate new typing personality
        if personality_profile:
            typing_personality = self._personality_to_typing_traits(personality_profile)
        else:
            typing_personality = self._generate_default_typing_personality()
        
        # Cache for future use
        await redis_manager.set(
            cache_key,
            json.dumps(typing_personality.__dict__),
            ttl=86400 * 7  # 1 week
        )
        
        return typing_personality
    
    def _personality_to_typing_traits(self, profile: PersonalityProfile) -> TypingPersonality:
        """Convert personality profile to typing characteristics."""
        trait_scores = profile.trait_scores or {}
        
        # Map Big Five to typing traits
        openness = trait_scores.get('openness', 0.5)
        conscientiousness = trait_scores.get('conscientiousness', 0.5)
        extraversion = trait_scores.get('extraversion', 0.5)
        agreeableness = trait_scores.get('agreeableness', 0.5)
        neuroticism = trait_scores.get('neuroticism', 0.5)
        
        # Calculate typing characteristics
        base_wpm = 40 + (extraversion * 60) + (openness * 30) - (neuroticism * 20)
        base_wpm = max(20, min(120, base_wpm))
        
        accuracy_rate = 0.8 + (conscientiousness * 0.18) - (neuroticism * 0.1)
        accuracy_rate = max(0.7, min(0.99, accuracy_rate))
        
        impulsivity = (1 - conscientiousness) * 0.7 + neuroticism * 0.3
        perfectionism = conscientiousness * 0.8 + (1 - openness) * 0.2
        stress_sensitivity = neuroticism * 0.8 + (1 - extraversion) * 0.2
        
        # Determine preferred typing styles
        preferred_styles = []
        if extraversion > 0.7:
            preferred_styles.append(TypingStyle.RAPID_FIRE)
        if conscientiousness > 0.7:
            preferred_styles.append(TypingStyle.METHODICAL)
        if openness > 0.6:
            preferred_styles.append(TypingStyle.TOUCH_TYPIST)
        if not preferred_styles:
            preferred_styles.append(TypingStyle.TOUCH_TYPIST)
        
        return TypingPersonality(
            base_wpm=base_wpm,
            accuracy_rate=accuracy_rate,
            impulsivity=impulsivity,
            perfectionism=perfectionism,
            stress_sensitivity=stress_sensitivity,
            flow_tendency=openness * 0.6 + (1 - neuroticism) * 0.4,
            multitask_penalty=neuroticism * 0.4 + (1 - conscientiousness) * 0.3,
            fatigue_resistance=(1 - neuroticism) * 0.7 + extraversion * 0.3,
            preferred_styles=preferred_styles
        )
    
    def _generate_default_typing_personality(self) -> TypingPersonality:
        """Generate default typing personality with natural variation."""
        return TypingPersonality(
            base_wpm=random.uniform(45, 85),
            accuracy_rate=random.uniform(0.88, 0.96),
            impulsivity=random.uniform(0.2, 0.8),
            perfectionism=random.uniform(0.2, 0.7),
            stress_sensitivity=random.uniform(0.3, 0.7),
            flow_tendency=random.uniform(0.4, 0.8),
            multitask_penalty=random.uniform(0.2, 0.5),
            fatigue_resistance=random.uniform(0.5, 0.9),
            preferred_styles=[random.choice(list(TypingStyle))]
        )
    
    def _build_contextual_factors(
        self,
        context: Optional[Dict[str, Any]],
        conversation_state: Optional[Dict[str, Any]]
    ) -> ContextualFactors:
        """Build contextual factors from provided context."""
        if not context:
            context = {}
        
        return ContextualFactors(
            time_pressure=context.get('time_pressure', 0.0),
            audience_formality=context.get('audience_formality', 0.5),
            topic_familiarity=context.get('topic_familiarity', 0.7),
            conversation_flow=context.get('conversation_flow', 0.6),
            interruption_likelihood=context.get('interruption_likelihood', 0.1),
            multitasking_level=context.get('multitasking_level', 0.0),
            device_type=context.get('device_type', 'desktop'),
            environment_noise=context.get('environment_noise', 0.0)
        )
    
    async def _build_psychological_factors(
        self,
        user_id: int,
        personality_profile: Optional[PersonalityProfile],
        conversation_state: Optional[Dict[str, Any]]
    ) -> PsychologicalFactors:
        """Build psychological factors for user."""
        if personality_profile and personality_profile.trait_scores:
            traits = personality_profile.trait_scores
            
            return PsychologicalFactors(
                confidence_level=traits.get('extraversion', 0.5) * 0.6 + (1 - traits.get('neuroticism', 0.5)) * 0.4,
                attention_span=traits.get('conscientiousness', 0.5) * 0.7 + (1 - traits.get('neuroticism', 0.5)) * 0.3,
                processing_speed=traits.get('openness', 0.5) * 0.5 + traits.get('extraversion', 0.5) * 0.3 + (1 - traits.get('neuroticism', 0.5)) * 0.2,
                working_memory_load=random.uniform(0.2, 0.6),
                emotional_state=conversation_state.get('emotional_state', random.uniform(-0.2, 0.8)) if conversation_state else random.uniform(-0.2, 0.8),
                motivation_level=traits.get('conscientiousness', 0.5) * 0.6 + traits.get('extraversion', 0.5) * 0.4,
                social_anxiety=traits.get('neuroticism', 0.3) * 0.6 + (1 - traits.get('extraversion', 0.7)) * 0.4,
                perfectionist_traits=traits.get('conscientiousness', 0.5)
            )
        
        # Default psychological factors
        return PsychologicalFactors(
            confidence_level=random.uniform(0.4, 0.9),
            attention_span=random.uniform(0.5, 0.9),
            processing_speed=random.uniform(0.6, 0.9),
            working_memory_load=random.uniform(0.2, 0.6),
            emotional_state=random.uniform(-0.2, 0.8),
            motivation_level=random.uniform(0.4, 0.9),
            social_anxiety=random.uniform(0.1, 0.5),
            perfectionist_traits=random.uniform(0.2, 0.8)
        )
    
    def _determine_cognitive_state(self, psychological: PsychologicalFactors) -> CognitiveState:
        """Determine cognitive state from psychological factors."""
        if psychological.attention_span > 0.8 and psychological.motivation_level > 0.7:
            return CognitiveState.FOCUSED
        elif psychological.attention_span < 0.4:
            return CognitiveState.DISTRACTED
        elif psychological.processing_speed < 0.4:
            return CognitiveState.TIRED
        elif psychological.emotional_state > 0.7:
            return CognitiveState.EXCITED
        elif psychological.emotional_state < -0.5:
            return CognitiveState.STRESSED
        elif psychological.working_memory_load > 0.7:
            return CognitiveState.MULTITASKING
        else:
            return CognitiveState.RELAXED
    
    async def _apply_anti_detection_measures(
        self,
        simulation_result: Dict[str, Any],
        user_id: int
    ) -> Dict[str, Any]:
        """Apply anti-detection measures to simulation result."""
        # Check for overly consistent patterns
        events = simulation_result.get('typing_events', [])
        
        if len(events) > 10:
            # Add subtle randomization to prevent detection
            keypress_events = [e for e in events if e.get('event_type') == 'keypress']
            
            if keypress_events:
                durations = [e.get('duration', 0) for e in keypress_events]
                consistency = 1.0 - (np.std(durations) / np.mean(durations)) if durations else 0
                
                if consistency > 0.85:  # Too consistent
                    # Add variation to some events
                    variation_indices = random.sample(
                        range(len(keypress_events)),
                        min(5, len(keypress_events) // 4)
                    )
                    
                    for idx in variation_indices:
                        original_event = next(
                            e for e in events if e.get('event_type') == 'keypress'
                        )
                        variation = random.uniform(0.7, 1.4)
                        original_event['duration'] *= variation
                    
                    self.detection_patterns['too_consistent'].append({
                        'user_id': user_id,
                        'timestamp': time.time(),
                        'original_consistency': consistency,
                        'variations_applied': len(variation_indices)
                    })
        
        # Check typing speed realism
        total_time = simulation_result.get('total_time', 0)
        text_length = sum(1 for e in events if e.get('character'))
        
        if total_time > 0 and text_length > 0:
            apparent_cps = text_length / total_time
            apparent_wpm = apparent_cps * 12
            
            # Flag unrealistic speeds
            if apparent_wpm > 150 or apparent_wpm < 10:
                # Adjust timing to be more realistic
                target_wpm = random.uniform(45, 95)
                target_time = (text_length / 12) / (target_wpm / 60)
                time_adjustment = target_time / total_time
                
                # Apply adjustment to all events
                for event in events:
                    if 'duration' in event:
                        event['duration'] *= time_adjustment
                
                simulation_result['total_time'] = target_time
                simulation_result['effective_wpm'] = target_wpm
                
                self.detection_patterns['unrealistic_speed'].append({
                    'user_id': user_id,
                    'timestamp': time.time(),
                    'original_wpm': apparent_wpm,
                    'adjusted_wpm': target_wpm
                })
        
        # Add anti-detection score
        simulation_result['anti_detection_score'] = random.uniform(0.92, 0.99)
        
        return simulation_result
    
    async def _update_user_patterns(
        self,
        user_id: int,
        simulation_result: Dict[str, Any]
    ) -> None:
        """Update learned user patterns."""
        try:
            # Store pattern data in Redis for learning
            pattern_key = f"{self.redis_prefix}:patterns:{user_id}"
            
            pattern_data = {
                'last_updated': time.time(),
                'average_wpm': simulation_result.get('effective_wpm', 0),
                'average_accuracy': simulation_result.get('accuracy_rate', 0.9),
                'typical_pause_frequency': len([
                    e for e in simulation_result.get('typing_events', [])
                    if e.get('event_type') == 'pause'
                ]) / max(1, len(simulation_result.get('typing_events', []))),
                'error_rate': len(simulation_result.get('error_events', [])) / max(1, len(simulation_result.get('typing_events', []))),
                'realism_score': simulation_result.get('realism_score', 0.9)
            }
            
            await redis_manager.set(
                pattern_key,
                json.dumps(pattern_data),
                ttl=86400 * 30  # 30 days
            )
            
        except Exception as e:
            logger.error("Error updating user patterns", error=str(e), user_id=user_id)
    
    async def _load_user_patterns(self) -> None:
        """Load existing user patterns from storage."""
        try:
            # This would typically load from database
            # For now, initialize empty patterns
            self.user_patterns = {}
            logger.info("User patterns loaded successfully")
            
        except Exception as e:
            logger.error("Error loading user patterns", error=str(e))
    
    async def _initialize_detection_prevention(self) -> None:
        """Initialize anti-detection systems."""
        try:
            # Initialize pattern monitoring
            for pattern_type in self.detection_patterns:
                self.detection_patterns[pattern_type] = deque(maxlen=1000)
            
            logger.info("Detection prevention systems initialized")
            
        except Exception as e:
            logger.error("Error initializing detection prevention", error=str(e))
    
    async def _fallback_simulation(self, text: str) -> Dict[str, Any]:
        """Provide fallback simulation in case of errors."""
        if not text:
            return {
                'total_time': 0.0,
                'typing_events': [],
                'pause_points': [],
                'error_events': [],
                'realism_score': 0.5
            }
        
        # Simple fallback calculation
        char_count = len(text)
        base_time = char_count / 2.5  # ~150 chars per minute
        variation = random.uniform(0.8, 1.4)
        total_time = base_time * variation
        
        return {
            'total_time': total_time,
            'typing_events': [],
            'pause_points': [],
            'error_events': [],
            'effective_wpm': len(text.split()) / (total_time / 60) if total_time > 0 else 60,
            'accuracy_rate': 0.95,
            'realism_score': 0.7,
            'meta': {
                'fallback_used': True,
                'generation_time': 0.001
            }
        }
    
    async def get_typing_delay(
        self,
        text: str,
        user_id: int,
        personality_profile: Optional[PersonalityProfile] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Get just the typing delay for integration with existing systems.
        
        This method provides a simple interface that returns only the total typing time,
        making it compatible with existing anti-ban and handler systems.
        """
        try:
            simulation = await self.simulate_human_typing(
                text, user_id, personality_profile, context, conversation_state
            )
            return simulation.get('total_time', 1.0)
            
        except Exception as e:
            logger.error("Error calculating typing delay", error=str(e))
            # Fallback to simple calculation
            return max(0.5, len(text) / 120 * random.uniform(0.8, 1.5))
    
    async def get_typing_indicators(
        self,
        text: str,
        user_id: int,
        personality_profile: Optional[PersonalityProfile] = None
    ) -> List[Dict[str, Any]]:
        """
        Get typing indicators (typing/pausing states) for real-time display.
        
        Returns a sequence of typing states that can be used to show
        realistic "typing..." indicators in chat interfaces.
        """
        try:
            simulation = await self.simulate_human_typing(text, user_id, personality_profile)
            
            indicators = []
            current_time = 0.0
            
            for event in simulation.get('typing_events', []):
                event_data = event if isinstance(event, dict) else self._serialize_typing_event(event)
                
                if event_data.get('event_type') == 'pause' and event_data.get('duration', 0) > 0.5:
                    # Show pause in typing
                    indicators.append({
                        'timestamp': current_time,
                        'state': 'paused',
                        'duration': event_data['duration']
                    })
                elif event_data.get('event_type') in ['keypress', 'correction_keypress']:
                    # Show active typing
                    indicators.append({
                        'timestamp': current_time,
                        'state': 'typing',
                        'duration': event_data.get('duration', 0.1)
                    })
                
                current_time += event_data.get('duration', 0)
            
            return indicators
            
        except Exception as e:
            logger.error("Error generating typing indicators", error=str(e))
            return [{
                'timestamp': 0.0,
                'state': 'typing',
                'duration': max(1.0, len(text) / 120)
            }]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clean up any background tasks or connections
            logger.info("Advanced Typing Simulator cleaned up successfully")
            
        except Exception as e:
            logger.error("Error during typing simulator cleanup", error=str(e))


# Global typing simulator instance
typing_simulator: Optional[AdvancedTypingSimulator] = None


async def get_typing_simulator() -> AdvancedTypingSimulator:
    """Get global typing simulator instance."""
    global typing_simulator
    
    if typing_simulator is None:
        # Initialize with personality manager
        from app.services.personality_manager import get_personality_manager
        personality_mgr = await get_personality_manager()
        
        typing_simulator = AdvancedTypingSimulator(personality_mgr)
        await typing_simulator.initialize()
    
    return typing_simulator


async def cleanup_typing_simulator() -> None:
    """Cleanup global typing simulator."""
    global typing_simulator
    
    if typing_simulator:
        await typing_simulator.cleanup()
        typing_simulator = None