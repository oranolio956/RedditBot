"""
Text Synesthesia Service

Advanced text-to-synesthetic experience translation implementing:
- Lexical-gustatory synesthesia (words -> tastes/textures)
- Grapheme-color synesthesia (letters -> colors)  
- Text-to-spatial mapping (semantic landscapes)
- Emotional text visualization
- Multi-modal text experience generation
"""

import asyncio
import numpy as np
import colorsys
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

# NLP and ML libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

from app.core.config import get_settings
from app.models.synesthesia import SynestheticProfile

logger = logging.getLogger(__name__)
settings = get_settings()


class TextualSynesthesiaType(Enum):
    """Types of text-based synesthetic experiences"""
    GRAPHEME_COLOR = "grapheme_color"          # Letters -> Colors
    LEXICAL_GUSTATORY = "lexical_gustatory"    # Words -> Tastes/Textures
    SEMANTIC_SPATIAL = "semantic_spatial"      # Meaning -> 3D Space
    EMOTIONAL_VISUAL = "emotional_visual"      # Emotion -> Visual patterns
    PHONEME_COLOR = "phoneme_color"           # Sounds -> Colors
    WORD_PERSONALITY = "word_personality"      # Words -> Character traits


@dataclass
class GraphemeColorMapping:
    """Color mappings for individual letters and characters"""
    character: str
    hue: float
    saturation: float
    brightness: float
    hex_color: str
    personality_traits: List[str]


@dataclass
class WordTasteProfile:
    """Taste and texture profile for individual words"""
    word: str
    primary_taste: str  # sweet, sour, bitter, salty, umami
    intensity: float
    texture: str       # smooth, rough, sharp, soft, fizzy
    temperature: str   # hot, cold, warm, cool, neutral
    complexity: float  # simple to complex taste
    emotional_valence: float  # positive to negative


@dataclass
class SemanticSpatialLayout:
    """3D spatial layout of text semantic meaning"""
    concepts: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    dimensions: Dict[str, float]
    coordinate_system: str


class GraphemeColorSynesthete:
    """Implements authentic grapheme-color synesthesia"""
    
    def __init__(self):
        # Research-based letter-color mappings (most common patterns)
        # Based on studies of actual synesthetes
        self.standard_mappings = {
            'A': {'hue': 0, 'sat': 0.8, 'bright': 0.9},      # Red
            'B': {'hue': 240, 'sat': 0.7, 'bright': 0.6},    # Blue
            'C': {'hue': 60, 'sat': 0.8, 'bright': 0.8},     # Yellow
            'D': {'hue': 120, 'sat': 0.6, 'bright': 0.5},    # Dark Green
            'E': {'hue': 30, 'sat': 0.9, 'bright': 0.9},     # Orange
            'F': {'hue': 300, 'sat': 0.7, 'bright': 0.7},    # Magenta
            'G': {'hue': 90, 'sat': 0.8, 'bright': 0.7},     # Green
            'H': {'hue': 20, 'sat': 0.6, 'bright': 0.8},     # Brown
            'I': {'hue': 0, 'sat': 0.0, 'bright': 1.0},      # White
            'J': {'hue': 45, 'sat': 0.9, 'bright': 0.6},     # Dark Yellow
            'K': {'hue': 0, 'sat': 0.0, 'bright': 0.3},      # Dark Gray
            'L': {'hue': 200, 'sat': 0.7, 'bright': 0.8},    # Light Blue
            'M': {'hue': 320, 'sat': 0.8, 'bright': 0.6},    # Pink
            'N': {'hue': 30, 'sat': 0.5, 'bright': 0.4},     # Brown
            'O': {'hue': 0, 'sat': 0.0, 'bright': 1.0},      # White/Clear
            'P': {'hue': 300, 'sat': 0.9, 'bright': 0.8},    # Purple
            'Q': {'hue': 240, 'sat': 0.6, 'bright': 0.3},    # Dark Blue
            'R': {'hue': 0, 'sat': 1.0, 'bright': 0.7},      # Red
            'S': {'hue': 60, 'sat': 0.9, 'bright': 0.9},     # Bright Yellow
            'T': {'hue': 180, 'sat': 0.7, 'bright': 0.6},    # Teal
            'U': {'hue': 270, 'sat': 0.8, 'bright': 0.8},    # Violet
            'V': {'hue': 120, 'sat': 0.9, 'bright': 0.5},    # Dark Green
            'W': {'hue': 0, 'sat': 0.0, 'bright': 0.9},      # Light Gray
            'X': {'hue': 0, 'sat': 0.0, 'bright': 0.2},      # Black
            'Y': {'hue': 50, 'sat': 1.0, 'bright': 1.0},     # Bright Yellow
            'Z': {'hue': 270, 'sat': 0.9, 'bright': 0.4},    # Dark Purple
        }
        
        # Number colors
        self.number_mappings = {
            '0': {'hue': 0, 'sat': 0.0, 'bright': 1.0},      # White
            '1': {'hue': 0, 'sat': 0.0, 'bright': 0.2},      # Black
            '2': {'hue': 120, 'sat': 0.8, 'bright': 0.7},    # Green
            '3': {'hue': 0, 'sat': 0.9, 'bright': 0.8},      # Red
            '4': {'hue': 240, 'sat': 0.8, 'bright': 0.6},    # Blue
            '5': {'hue': 60, 'sat': 0.9, 'bright': 0.9},     # Yellow
            '6': {'hue': 30, 'sat': 0.8, 'bright': 0.7},     # Orange
            '7': {'hue': 300, 'sat': 0.7, 'bright': 0.8},    # Purple
            '8': {'hue': 180, 'sat': 0.6, 'bright': 0.5},    # Dark Teal
            '9': {'hue': 270, 'sat': 0.8, 'bright': 0.7},    # Violet
        }
        
        # Character personality traits (synesthetic associations)
        self.character_personalities = {
            'A': ['confident', 'strong', 'leader'],
            'B': ['gentle', 'calm', 'reliable'],
            'C': ['cheerful', 'bright', 'energetic'],
            'D': ['steady', 'grounded', 'dependable'],
            'E': ['vibrant', 'excitable', 'warm'],
            'F': ['mysterious', 'elegant', 'sophisticated'],
            'G': ['natural', 'growing', 'fresh'],
            'H': ['sturdy', 'traditional', 'solid'],
            'I': ['pure', 'simple', 'clear'],
            'J': ['quirky', 'unique', 'individual'],
            'K': ['sharp', 'angular', 'decisive'],
            'L': ['flowing', 'liquid', 'smooth'],
            'M': ['nurturing', 'maternal', 'caring'],
            'N': ['grounding', 'earthy', 'stable'],
            'O': ['complete', 'whole', 'embracing'],
            'P': ['playful', 'bouncy', 'energetic'],
            'Q': ['regal', 'unique', 'distinguished'],
            'R': ['passionate', 'intense', 'rolling'],
            'S': ['sinuous', 'flowing', 'smooth'],
            'T': ['structured', 'balanced', 'tall'],
            'U': ['curved', 'embracing', 'containing'],
            'V': ['pointed', 'focused', 'directed'],
            'W': ['wide', 'expansive', 'welcoming'],
            'X': ['crossing', 'mysterious', 'unknown'],
            'Y': ['reaching', 'questioning', 'bright'],
            'Z': ['zigzag', 'dynamic', 'energetic']
        }
    
    def get_character_color(self, char: str, 
                           position: int = 0,
                           context: str = "") -> GraphemeColorMapping:
        """Get color mapping for a single character"""
        
        char_upper = char.upper()
        
        # Get base mapping
        if char_upper in self.standard_mappings:
            mapping = self.standard_mappings[char_upper]
        elif char in self.number_mappings:
            mapping = self.number_mappings[char]
        else:
            # Default mapping for other characters
            mapping = {'hue': (ord(char) * 137) % 360, 'sat': 0.7, 'bright': 0.7}
        
        # Apply positional variation (letters change slightly based on position)
        position_factor = (position % 10) * 0.02
        hue = (mapping['hue'] + position_factor * 360) % 360
        saturation = max(0, min(1, mapping['sat'] + (position % 3 - 1) * 0.1))
        brightness = max(0, min(1, mapping['bright'] + (position % 5 - 2) * 0.05))
        
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        return GraphemeColorMapping(
            character=char,
            hue=hue,
            saturation=saturation,
            brightness=brightness,
            hex_color=hex_color,
            personality_traits=self.character_personalities.get(char_upper, ['neutral'])
        )
    
    def colorize_text(self, text: str) -> List[GraphemeColorMapping]:
        """Convert entire text to sequence of colored characters"""
        
        colorized = []
        
        for i, char in enumerate(text):
            if char.isalpha() or char.isdigit():
                color_mapping = self.get_character_color(char, i, text)
                colorized.append(color_mapping)
            else:
                # Special characters get neutral colors
                colorized.append(GraphemeColorMapping(
                    character=char,
                    hue=0,
                    saturation=0,
                    brightness=0.5,
                    hex_color="#808080",
                    personality_traits=['punctuation']
                ))
        
        return colorized


class LexicalGustatoryMapper:
    """Maps words to taste and texture sensations"""
    
    def __init__(self):
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon')
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Research-based taste mappings for common word patterns
        self.taste_patterns = {
            # Phonetic patterns -> tastes
            'sharp_consonants': {'taste': 'sour', 'texture': 'sharp', 'intensity': 0.8},
            'soft_consonants': {'taste': 'sweet', 'texture': 'smooth', 'intensity': 0.6},
            'liquid_sounds': {'taste': 'umami', 'texture': 'flowing', 'intensity': 0.5},
            'explosive_sounds': {'taste': 'bitter', 'texture': 'rough', 'intensity': 0.9},
            'nasal_sounds': {'taste': 'salty', 'texture': 'grainy', 'intensity': 0.7},
        }
        
        # Semantic category mappings
        self.semantic_tastes = {
            'positive_emotion': {'taste': 'sweet', 'temperature': 'warm'},
            'negative_emotion': {'taste': 'bitter', 'temperature': 'cold'},
            'action_verbs': {'taste': 'salty', 'temperature': 'hot'},
            'abstract_concepts': {'taste': 'umami', 'temperature': 'neutral'},
            'sensory_words': {'taste': 'sour', 'temperature': 'cool'},
        }
        
        # Word length and complexity effects
        self.length_effects = {
            'short': {'intensity': 1.0, 'complexity': 0.2},
            'medium': {'intensity': 0.7, 'complexity': 0.6},
            'long': {'intensity': 0.5, 'complexity': 1.0},
        }
    
    async def get_word_taste(self, word: str, context: str = "") -> WordTasteProfile:
        """Generate taste profile for a single word"""
        
        word_lower = word.lower()
        
        # Analyze phonetic characteristics
        phonetic_category = self._classify_phonetic_pattern(word_lower)
        base_taste = self.taste_patterns.get(phonetic_category, 
                                           {'taste': 'umami', 'texture': 'smooth', 'intensity': 0.5})
        
        # Analyze semantic category
        semantic_category = await self._classify_semantic_category(word_lower, context)
        semantic_taste = self.semantic_tastes.get(semantic_category, 
                                                {'taste': 'umami', 'temperature': 'neutral'})
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(word)
        emotional_valence = sentiment['compound']
        
        # Determine word length category
        if len(word) <= 3:
            length_cat = 'short'
        elif len(word) <= 7:
            length_cat = 'medium'  
        else:
            length_cat = 'long'
        
        length_effect = self.length_effects[length_cat]
        
        # Combine all factors
        primary_taste = base_taste['taste']
        if abs(emotional_valence) > 0.5:
            primary_taste = semantic_taste['taste']
        
        # Calculate final intensity
        base_intensity = base_taste['intensity']
        emotion_intensity = abs(emotional_valence)
        final_intensity = (base_intensity + emotion_intensity) / 2 * length_effect['intensity']
        
        return WordTasteProfile(
            word=word,
            primary_taste=primary_taste,
            intensity=final_intensity,
            texture=base_taste['texture'],
            temperature=semantic_taste['temperature'],
            complexity=length_effect['complexity'],
            emotional_valence=emotional_valence
        )
    
    def _classify_phonetic_pattern(self, word: str) -> str:
        """Classify word's phonetic pattern for taste mapping"""
        
        # Count different types of sounds
        sharp_consonants = len(re.findall(r'[kgtpbdqxzs]', word))
        soft_consonants = len(re.findall(r'[flmrwny]', word))
        liquid_sounds = len(re.findall(r'[lrwy]', word))
        explosive_sounds = len(re.findall(r'[kgptbd]', word))
        nasal_sounds = len(re.findall(r'[mn]', word))
        
        # Determine dominant pattern
        patterns = {
            'sharp_consonants': sharp_consonants,
            'soft_consonants': soft_consonants, 
            'liquid_sounds': liquid_sounds,
            'explosive_sounds': explosive_sounds,
            'nasal_sounds': nasal_sounds
        }
        
        return max(patterns.keys(), key=lambda k: patterns[k])
    
    async def _classify_semantic_category(self, word: str, context: str) -> str:
        """Classify word's semantic category"""
        
        # Simple heuristic classification (could be enhanced with NLP models)
        positive_words = ['happy', 'joy', 'love', 'beautiful', 'amazing', 'wonderful']
        negative_words = ['sad', 'angry', 'hate', 'ugly', 'terrible', 'awful']
        action_words = ['run', 'jump', 'dance', 'sing', 'play', 'work']
        sensory_words = ['bright', 'loud', 'soft', 'smooth', 'rough', 'hot', 'cold']
        
        if any(pos_word in word for pos_word in positive_words):
            return 'positive_emotion'
        elif any(neg_word in word for neg_word in negative_words):
            return 'negative_emotion'
        elif any(action_word in word for action_word in action_words):
            return 'action_verbs'
        elif any(sensory_word in word for sensory_word in sensory_words):
            return 'sensory_words'
        else:
            return 'abstract_concepts'


class SemanticSpatialMapper:
    """Maps text semantic meaning to 3D spatial layouts"""
    
    def __init__(self):
        # Initialize sentence transformer for semantic embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Spatial metaphor mappings
        self.spatial_metaphors = {
            'time': {'axis': 'x', 'direction': 'forward'},
            'importance': {'axis': 'y', 'direction': 'up'},
            'emotion': {'axis': 'z', 'direction': 'out'},
            'complexity': {'axis': 'size', 'direction': 'bigger'},
            'certainty': {'axis': 'opacity', 'direction': 'solid'}
        }
    
    async def create_semantic_landscape(self, text: str) -> SemanticSpatialLayout:
        """Convert text to 3D semantic landscape"""
        
        # Split into sentences and extract concepts
        sentences = text.split('.')
        concepts = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Get semantic embedding
                embedding = self.sentence_model.encode([sentence.strip()])[0]
                
                # Analyze semantic properties
                concept = await self._analyze_concept_properties(sentence.strip(), embedding, i)
                concepts.append(concept)
        
        # Create relationships between concepts
        relationships = self._create_concept_relationships(concepts)
        
        # Calculate spatial dimensions
        dimensions = self._calculate_spatial_dimensions(concepts)
        
        return SemanticSpatialLayout(
            concepts=concepts,
            relationships=relationships,
            dimensions=dimensions,
            coordinate_system="semantic_space"
        )
    
    async def _analyze_concept_properties(self, text: str, embedding: np.ndarray, 
                                        position: int) -> Dict[str, Any]:
        """Analyze semantic properties of a concept"""
        
        # Extract key properties
        word_count = len(text.split())
        sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
        
        # Map to spatial coordinates
        # X-axis: temporal progression (sentence order)
        x = position * 2.0
        
        # Y-axis: emotional valence (-1 to 1 -> -5 to 5)
        y = sentiment['compound'] * 5.0
        
        # Z-axis: complexity (word count and embedding magnitude)
        complexity = word_count / 20.0 + np.linalg.norm(embedding) / 10.0
        z = complexity * 3.0
        
        return {
            'id': f"concept_{position}",
            'text': text,
            'position': {'x': x, 'y': y, 'z': z},
            'embedding': embedding.tolist()[:10],  # First 10 dimensions for storage
            'properties': {
                'word_count': word_count,
                'sentiment': sentiment['compound'],
                'complexity': complexity,
                'size': max(0.5, word_count / 10.0),
                'opacity': min(1.0, abs(sentiment['compound']) + 0.3)
            }
        }
    
    def _create_concept_relationships(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create relationships between concepts based on semantic similarity"""
        
        relationships = []
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                # Calculate similarity using embeddings
                emb1 = np.array(concept1['embedding'])
                emb2 = np.array(concept2['embedding'])
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # Create relationship if similarity is significant
                if similarity > 0.5:
                    relationships.append({
                        'source': concept1['id'],
                        'target': concept2['id'],
                        'strength': float(similarity),
                        'type': 'semantic_similarity',
                        'visual_style': 'flowing_line' if similarity > 0.8 else 'dashed_line'
                    })
        
        return relationships
    
    def _calculate_spatial_dimensions(self, concepts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate bounding dimensions of the semantic space"""
        
        if not concepts:
            return {'width': 10.0, 'height': 10.0, 'depth': 10.0}
        
        positions = [c['position'] for c in concepts]
        
        x_coords = [p['x'] for p in positions]
        y_coords = [p['y'] for p in positions] 
        z_coords = [p['z'] for p in positions]
        
        return {
            'width': max(x_coords) - min(x_coords) + 4.0,   # Add padding
            'height': max(y_coords) - min(y_coords) + 4.0,
            'depth': max(z_coords) - min(z_coords) + 4.0,
            'center_x': (max(x_coords) + min(x_coords)) / 2.0,
            'center_y': (max(y_coords) + min(y_coords)) / 2.0,
            'center_z': (max(z_coords) + min(z_coords)) / 2.0
        }


class TextSynesthesiaEngine:
    """Main engine for text-to-synesthetic experiences"""
    
    def __init__(self):
        self.grapheme_synesthete = GraphemeColorSynesthete()
        self.taste_mapper = LexicalGustatoryMapper()
        self.spatial_mapper = SemanticSpatialMapper()
        
        # Performance tracking
        self.translation_count = 0
        self.average_processing_time = 0.0
        
        logger.info("Text Synesthesia Engine initialized")
    
    async def translate_text_to_synesthetic(self, text: str,
                                          synesthesia_types: List[TextualSynesthesiaType],
                                          user_profile: Optional[SynestheticProfile] = None) -> Dict[str, Any]:
        """
        Main text-to-synesthetic translation function
        
        Args:
            text: Input text to translate
            synesthesia_types: List of desired synesthetic modalities
            user_profile: Optional user personalization
            
        Returns:
            Dictionary with all requested synesthetic translations
        """
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Grapheme-color synesthesia
            if TextualSynesthesiaType.GRAPHEME_COLOR in synesthesia_types:
                results['grapheme_colors'] = await self._translate_grapheme_color(text)
            
            # Lexical-gustatory synesthesia
            if TextualSynesthesiaType.LEXICAL_GUSTATORY in synesthesia_types:
                results['word_tastes'] = await self._translate_lexical_gustatory(text)
            
            # Semantic spatial mapping
            if TextualSynesthesiaType.SEMANTIC_SPATIAL in synesthesia_types:
                results['semantic_space'] = await self._translate_semantic_spatial(text)
            
            # Emotional visual patterns
            if TextualSynesthesiaType.EMOTIONAL_VISUAL in synesthesia_types:
                results['emotional_visuals'] = await self._translate_emotional_visual(text)
            
            # Phoneme color mapping
            if TextualSynesthesiaType.PHONEME_COLOR in synesthesia_types:
                results['phoneme_colors'] = await self._translate_phoneme_color(text)
            
            # Word personality mapping
            if TextualSynesthesiaType.WORD_PERSONALITY in synesthesia_types:
                results['word_personalities'] = await self._translate_word_personality(text)
            
            # Apply user personalization if available
            if user_profile:
                results = await self._apply_user_personalization(results, user_profile)
            
            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            # Add metadata
            results['metadata'] = {
                'input_text': text,
                'text_length': len(text),
                'word_count': len(text.split()),
                'processing_time_ms': processing_time,
                'synesthesia_types': [st.value for st in synesthesia_types],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Text synesthesia translation completed in {processing_time:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Text synesthesia translation failed: {e}")
            raise
    
    async def _translate_grapheme_color(self, text: str) -> Dict[str, Any]:
        """Translate text to grapheme-color synesthesia"""
        
        colored_chars = self.grapheme_synesthete.colorize_text(text)
        
        # Group into words
        words = []
        current_word = []
        
        for char_mapping in colored_chars:
            if char_mapping.character.isalpha() or char_mapping.character.isdigit():
                current_word.append(char_mapping)
            else:
                if current_word:
                    words.append(current_word)
                    current_word = []
                if not char_mapping.character.isspace():
                    words.append([char_mapping])  # Punctuation as separate word
        
        if current_word:
            words.append(current_word)
        
        # Create color palette from all characters
        all_colors = [char.hex_color for char in colored_chars if char.character.isalpha()]
        
        return {
            'colored_characters': [
                {
                    'character': cm.character,
                    'color': cm.hex_color,
                    'hue': cm.hue,
                    'saturation': cm.saturation,
                    'brightness': cm.brightness,
                    'personality': cm.personality_traits
                }
                for cm in colored_chars
            ],
            'colored_words': [
                {
                    'word': ''.join(char.character for char in word),
                    'colors': [char.hex_color for char in word],
                    'dominant_color': max(word, key=lambda x: x.saturation * x.brightness).hex_color
                }
                for word in words if len(word) > 0 and word[0].character.isalpha()
            ],
            'color_palette': list(set(all_colors)),
            'total_unique_colors': len(set(all_colors))
        }
    
    async def _translate_lexical_gustatory(self, text: str) -> Dict[str, Any]:
        """Translate text to taste and texture experiences"""
        
        words = text.split()
        taste_profiles = []
        
        for word in words:
            if word.isalpha():  # Only process alphabetic words
                taste_profile = await self.taste_mapper.get_word_taste(word, text)
                taste_profiles.append({
                    'word': taste_profile.word,
                    'taste': taste_profile.primary_taste,
                    'intensity': taste_profile.intensity,
                    'texture': taste_profile.texture,
                    'temperature': taste_profile.temperature,
                    'complexity': taste_profile.complexity,
                    'emotional_valence': taste_profile.emotional_valence
                })
        
        # Analyze overall taste composition
        taste_distribution = {}
        for profile in taste_profiles:
            taste = profile['taste']
            taste_distribution[taste] = taste_distribution.get(taste, 0) + 1
        
        return {
            'word_tastes': taste_profiles,
            'taste_distribution': taste_distribution,
            'overall_flavor_profile': self._calculate_overall_flavor(taste_profiles),
            'complexity_score': np.mean([p['complexity'] for p in taste_profiles]) if taste_profiles else 0.0,
            'emotional_taste_score': np.mean([p['emotional_valence'] for p in taste_profiles]) if taste_profiles else 0.0
        }
    
    async def _translate_semantic_spatial(self, text: str) -> Dict[str, Any]:
        """Translate text to 3D semantic spatial layout"""
        
        spatial_layout = await self.spatial_mapper.create_semantic_landscape(text)
        
        return {
            'concepts': spatial_layout.concepts,
            'relationships': spatial_layout.relationships,
            'dimensions': spatial_layout.dimensions,
            'coordinate_system': spatial_layout.coordinate_system,
            'visualization_hints': {
                'recommended_camera_position': {
                    'x': spatial_layout.dimensions.get('center_x', 0),
                    'y': spatial_layout.dimensions.get('center_y', 0) + 10,
                    'z': spatial_layout.dimensions.get('center_z', 0) + 15
                },
                'lighting': 'ambient_with_concept_highlights',
                'background': 'semantic_gradient'
            }
        }
    
    async def _translate_emotional_visual(self, text: str) -> Dict[str, Any]:
        """Translate emotional content to visual patterns"""
        
        # Analyze emotions in text
        sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
        
        # Map emotions to visual elements
        emotional_visuals = {
            'primary_emotion': self._classify_primary_emotion(sentiment),
            'emotion_intensity': abs(sentiment['compound']),
            'color_scheme': self._emotion_to_color_scheme(sentiment),
            'visual_patterns': self._emotion_to_visual_patterns(sentiment),
            'animation_style': self._emotion_to_animation(sentiment),
            'particle_effects': self._emotion_to_particles(sentiment)
        }
        
        return emotional_visuals
    
    async def _translate_phoneme_color(self, text: str) -> Dict[str, Any]:
        """Map phonetic sounds to colors"""
        
        # Simplified phoneme analysis (would benefit from phonetic transcription)
        phoneme_colors = []
        
        for word in text.split():
            if word.isalpha():
                # Analyze phonetic characteristics
                vowels = len(re.findall(r'[aeiouAEIOU]', word))
                consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', word))
                
                # Map to colors based on sound characteristics
                vowel_hue = (vowels * 60) % 360  # Vowels spread across hue wheel
                consonant_saturation = min(1.0, consonants / 10.0)
                
                rgb = colorsys.hsv_to_rgb(vowel_hue / 360.0, consonant_saturation, 0.8)
                hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                
                phoneme_colors.append({
                    'word': word,
                    'phonetic_hue': vowel_hue,
                    'phonetic_saturation': consonant_saturation,
                    'color': hex_color,
                    'vowel_count': vowels,
                    'consonant_count': consonants
                })
        
        return {
            'phoneme_colors': phoneme_colors,
            'phonetic_color_palette': [pc['color'] for pc in phoneme_colors],
            'vowel_distribution': [pc['vowel_count'] for pc in phoneme_colors],
            'consonant_distribution': [pc['consonant_count'] for pc in phoneme_colors]
        }
    
    async def _translate_word_personality(self, text: str) -> Dict[str, Any]:
        """Assign personality traits to words"""
        
        word_personalities = []
        
        for word in text.split():
            if word.isalpha():
                # Analyze word characteristics
                length_trait = 'brief' if len(word) <= 4 else 'expansive' if len(word) > 8 else 'balanced'
                
                # Phonetic personality
                if re.search(r'[kg]', word.lower()):
                    phonetic_trait = 'strong'
                elif re.search(r'[flmr]', word.lower()):
                    phonetic_trait = 'flowing'
                else:
                    phonetic_trait = 'neutral'
                
                # Semantic personality (simplified)
                if word.lower() in ['love', 'joy', 'happy', 'bright']:
                    semantic_trait = 'warm'
                elif word.lower() in ['dark', 'cold', 'sad', 'angry']:
                    semantic_trait = 'cool'
                else:
                    semantic_trait = 'balanced'
                
                word_personalities.append({
                    'word': word,
                    'length_trait': length_trait,
                    'phonetic_trait': phonetic_trait,
                    'semantic_trait': semantic_trait,
                    'overall_personality': f"{phonetic_trait}_{semantic_trait}_{length_trait}",
                    'character_archetype': self._determine_character_archetype(
                        length_trait, phonetic_trait, semantic_trait)
                })
        
        return {
            'word_personalities': word_personalities,
            'personality_distribution': self._analyze_personality_distribution(word_personalities)
        }
    
    def _calculate_overall_flavor(self, taste_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall flavor profile of the text"""
        
        if not taste_profiles:
            return {'dominant_taste': 'neutral', 'complexity': 0.0}
        
        # Weight by intensity
        weighted_tastes = {}
        total_weight = 0
        
        for profile in taste_profiles:
            taste = profile['taste']
            intensity = profile['intensity']
            weighted_tastes[taste] = weighted_tastes.get(taste, 0) + intensity
            total_weight += intensity
        
        # Normalize
        if total_weight > 0:
            for taste in weighted_tastes:
                weighted_tastes[taste] /= total_weight
        
        dominant_taste = max(weighted_tastes.keys(), key=lambda k: weighted_tastes[k])
        avg_complexity = np.mean([p['complexity'] for p in taste_profiles])
        
        return {
            'dominant_taste': dominant_taste,
            'taste_weights': weighted_tastes,
            'complexity': avg_complexity,
            'balance': 1.0 - max(weighted_tastes.values()) if weighted_tastes else 0.0
        }
    
    def _classify_primary_emotion(self, sentiment: Dict[str, float]) -> str:
        """Classify primary emotion from sentiment scores"""
        
        if sentiment['compound'] > 0.1:
            if sentiment['pos'] > 0.6:
                return 'joy'
            else:
                return 'contentment'
        elif sentiment['compound'] < -0.1:
            if sentiment['neg'] > 0.6:
                return 'anger'
            else:
                return 'sadness'
        else:
            return 'neutral'
    
    def _emotion_to_color_scheme(self, sentiment: Dict[str, float]) -> Dict[str, Any]:
        """Convert emotions to color schemes"""
        
        emotion = self._classify_primary_emotion(sentiment)
        intensity = abs(sentiment['compound'])
        
        color_schemes = {
            'joy': {'primary': '#FFD700', 'secondary': '#FFA500', 'accent': '#FF6347'},
            'sadness': {'primary': '#4682B4', 'secondary': '#87CEEB', 'accent': '#191970'},
            'anger': {'primary': '#DC143C', 'secondary': '#FF6347', 'accent': '#8B0000'},
            'contentment': {'primary': '#90EE90', 'secondary': '#98FB98', 'accent': '#00FF7F'},
            'neutral': {'primary': '#C0C0C0', 'secondary': '#D3D3D3', 'accent': '#A9A9A9'}
        }
        
        base_scheme = color_schemes.get(emotion, color_schemes['neutral'])
        
        return {
            'emotion': emotion,
            'colors': base_scheme,
            'intensity_factor': intensity,
            'saturation_multiplier': 0.5 + intensity * 0.5
        }
    
    def _emotion_to_visual_patterns(self, sentiment: Dict[str, float]) -> Dict[str, Any]:
        """Convert emotions to visual movement patterns"""
        
        emotion = self._classify_primary_emotion(sentiment)
        intensity = abs(sentiment['compound'])
        
        patterns = {
            'joy': 'rising_spirals',
            'sadness': 'falling_drops',
            'anger': 'sharp_zigzags',
            'contentment': 'gentle_waves',
            'neutral': 'steady_flow'
        }
        
        return {
            'pattern_type': patterns.get(emotion, 'steady_flow'),
            'movement_speed': intensity * 2.0,
            'amplitude': intensity * 1.5,
            'complexity': intensity
        }
    
    def _emotion_to_animation(self, sentiment: Dict[str, float]) -> str:
        """Map emotions to animation styles"""
        
        emotion = self._classify_primary_emotion(sentiment)
        
        animations = {
            'joy': 'bouncing_expansion',
            'sadness': 'slow_contraction', 
            'anger': 'rapid_pulsing',
            'contentment': 'gentle_breathing',
            'neutral': 'steady_rotation'
        }
        
        return animations.get(emotion, 'steady_rotation')
    
    def _emotion_to_particles(self, sentiment: Dict[str, float]) -> Dict[str, Any]:
        """Map emotions to particle effects"""
        
        emotion = self._classify_primary_emotion(sentiment)
        intensity = abs(sentiment['compound'])
        
        particles = {
            'joy': {'type': 'sparks', 'count': 50, 'behavior': 'upward_burst'},
            'sadness': {'type': 'droplets', 'count': 20, 'behavior': 'downward_drift'},
            'anger': {'type': 'embers', 'count': 100, 'behavior': 'chaotic_swirl'},
            'contentment': {'type': 'petals', 'count': 30, 'behavior': 'gentle_float'},
            'neutral': {'type': 'dust', 'count': 15, 'behavior': 'slow_drift'}
        }
        
        base_particles = particles.get(emotion, particles['neutral'])
        base_particles['intensity'] = intensity
        base_particles['particle_count'] = int(base_particles['count'] * (0.5 + intensity))
        
        return base_particles
    
    def _determine_character_archetype(self, length_trait: str, 
                                     phonetic_trait: str, 
                                     semantic_trait: str) -> str:
        """Determine character archetype based on word traits"""
        
        # Simplified archetype mapping
        archetypes = {
            ('brief', 'strong', 'cool'): 'warrior',
            ('brief', 'flowing', 'warm'): 'child',
            ('expansive', 'strong', 'cool'): 'sage',
            ('expansive', 'flowing', 'warm'): 'nurturer',
            ('balanced', 'neutral', 'balanced'): 'everyman'
        }
        
        return archetypes.get((length_trait, phonetic_trait, semantic_trait), 'everyman')
    
    def _analyze_personality_distribution(self, word_personalities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of personality traits in text"""
        
        trait_counts = {}
        
        for wp in word_personalities:
            archetype = wp['character_archetype']
            trait_counts[archetype] = trait_counts.get(archetype, 0) + 1
        
        total_words = len(word_personalities)
        
        return {
            'archetype_counts': trait_counts,
            'archetype_percentages': {k: v/total_words for k, v in trait_counts.items()},
            'dominant_archetype': max(trait_counts.keys(), key=lambda k: trait_counts[k]) if trait_counts else 'everyman',
            'personality_diversity': len(trait_counts) / max(1, total_words)
        }
    
    async def _apply_user_personalization(self, results: Dict[str, Any],
                                         profile: SynestheticProfile) -> Dict[str, Any]:
        """Apply user's personal synesthetic preferences"""
        
        # Apply color intensity preferences
        if 'grapheme_colors' in results and hasattr(profile, 'color_intensity'):
            intensity_factor = profile.color_intensity
            
            for char in results['grapheme_colors']['colored_characters']:
                char['saturation'] *= intensity_factor
                char['brightness'] *= intensity_factor
                
                # Recalculate hex color
                rgb = colorsys.hsv_to_rgb(
                    char['hue'] / 360.0,
                    min(1.0, char['saturation']),
                    min(1.0, char['brightness'])
                )
                char['color'] = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        # Apply texture sensitivity to taste mappings
        if 'word_tastes' in results and hasattr(profile, 'texture_sensitivity'):
            sensitivity = profile.texture_sensitivity
            
            for taste in results['word_tastes']['word_tastes']:
                taste['intensity'] *= sensitivity
        
        return results
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics"""
        
        self.translation_count += 1
        self.average_processing_time = (
            (self.average_processing_time * (self.translation_count - 1) + processing_time) /
            self.translation_count
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        return {
            'total_translations': self.translation_count,
            'average_processing_time_ms': round(self.average_processing_time, 2),
            'target_latency_ms': 150.0,  # Target for text processing
            'performance_status': 'optimal' if self.average_processing_time < 150 else 'degraded'
        }


# Global engine instance
text_synesthesia_engine = TextSynesthesiaEngine()


async def get_text_synesthesia_engine() -> TextSynesthesiaEngine:
    """Dependency injection for text synesthesia engine"""
    return text_synesthesia_engine