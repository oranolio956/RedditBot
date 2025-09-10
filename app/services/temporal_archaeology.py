"""
Temporal Conversation Archaeology Service

Recovers lost conversations from behavioral patterns using linguistic fingerprinting,
probabilistic generation, and context reconstruction algorithms.
Based on 2024 research in digital archaeology and memory recovery.
"""

import asyncio
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from uuid import UUID, uuid4
import logging
from collections import defaultdict, Counter
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, text
from sqlalchemy.orm import selectinload
import redis.asyncio as redis

from app.models.temporal_archaeology import (
    ConversationFragment, ReconstructedMessage, TemporalPattern,
    LinguisticFingerprint, GhostConversation, ArchaeologySession
)
from app.models.message import Message
from app.models.user import User
from app.core.config import settings
from app.services.llm_service import LLMService
from app.core.security_utils import (
    EncryptionService, InputSanitizer, PrivacyProtector, 
    RateLimiter, MLSecurityValidator, ConsentManager
)
from app.core.performance_utils import MultiLevelCache, ModelOptimizer

logger = logging.getLogger(__name__)


class LinguisticProfiler:
    """
    Creates unique linguistic fingerprints from user communication patterns.
    Based on 2024 research in fuzzy fingerprinting and behavioral authentication.
    """
    
    def __init__(self):
        self.n_gram_range = (1, 3)
        self.pos_patterns = []
        self.lexical_features = {}
        self.stylistic_markers = {}
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.n_gram_range,
            max_features=1000,
            min_df=2
        )
        
    async def create_fingerprint(self, messages: List[Message]) -> LinguisticFingerprint:
        """Create comprehensive linguistic fingerprint from message history."""
        if not messages:
            return self._empty_fingerprint()
            
        # Extract text corpus
        corpus = [msg.content for msg in messages if msg.content]
        
        # N-gram analysis
        ngram_features = self._extract_ngrams(corpus)
        
        # Lexical diversity metrics
        lexical_metrics = self._calculate_lexical_diversity(corpus)
        
        # Stylistic features
        style_features = self._extract_stylistic_features(corpus)
        
        # Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(messages)
        
        # Emotional signatures
        emotional_profile = self._extract_emotional_patterns(corpus)
        
        # Create fingerprint
        fingerprint = LinguisticFingerprint(
            user_id=messages[0].user_id if messages else None,
            ngram_distribution=ngram_features,
            lexical_diversity=lexical_metrics,
            stylistic_features=style_features,
            temporal_signatures=temporal_patterns,
            emotional_patterns=emotional_profile,
            vocabulary_size=len(set(' '.join(corpus).split())),
            avg_message_length=np.mean([len(msg.content.split()) for msg in messages]),
            unique_phrases=self._extract_unique_phrases(corpus),
            confidence_score=self._calculate_confidence(messages)
        )
        
        return fingerprint
        
    def _extract_ngrams(self, corpus: List[str]) -> Dict:
        """Extract n-gram distributions as behavioral markers."""
        all_text = ' '.join(corpus)
        words = all_text.lower().split()
        
        ngram_freq = {}
        
        # Unigrams
        ngram_freq['unigrams'] = dict(Counter(words).most_common(100))
        
        # Bigrams
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        ngram_freq['bigrams'] = dict(Counter(bigrams).most_common(50))
        
        # Trigrams
        trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
        ngram_freq['trigrams'] = dict(Counter(trigrams).most_common(30))
        
        # Character-level patterns
        char_patterns = self._extract_character_patterns(all_text)
        ngram_freq['char_patterns'] = char_patterns
        
        return ngram_freq
        
    def _calculate_lexical_diversity(self, corpus: List[str]) -> Dict:
        """Calculate various lexical diversity metrics."""
        all_words = []
        for text in corpus:
            all_words.extend(text.lower().split())
            
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        metrics = {
            'ttr': unique_words / total_words if total_words > 0 else 0,  # Type-Token Ratio
            'yules_k': self._calculate_yules_k(all_words),
            'simpsons_d': self._calculate_simpsons_d(all_words),
            'hapax_legomena': sum(1 for w, c in Counter(all_words).items() if c == 1),
            'dis_legomena': sum(1 for w, c in Counter(all_words).items() if c == 2)
        }
        
        return metrics
        
    def _extract_stylistic_features(self, corpus: List[str]) -> Dict:
        """Extract writing style features."""
        features = {
            'punctuation_usage': {},
            'capitalization_patterns': {},
            'sentence_structure': {},
            'emoji_usage': {},
            'abbreviations': []
        }
        
        for text in corpus:
            # Punctuation patterns
            for punct in '.,!?;:':
                features['punctuation_usage'][punct] = text.count(punct)
                
            # Capitalization
            features['capitalization_patterns']['all_caps'] = len(re.findall(r'\b[A-Z]+\b', text))
            features['capitalization_patterns']['title_case'] = len(re.findall(r'\b[A-Z][a-z]+\b', text))
            
            # Sentence length distribution
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                features['sentence_structure']['avg_length'] = np.mean([len(s.split()) for s in sentences if s])
                features['sentence_structure']['std_length'] = np.std([len(s.split()) for s in sentences if s])
                
            # Emoji detection
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags
                "]+", 
                flags=re.UNICODE
            )
            features['emoji_usage']['count'] = len(emoji_pattern.findall(text))
            
            # Common abbreviations
            abbrev_patterns = ['lol', 'btw', 'omg', 'tbh', 'imo', 'fyi', 'asap']
            for abbrev in abbrev_patterns:
                if abbrev in text.lower():
                    features['abbreviations'].append(abbrev)
                    
        return features
        
    def _analyze_temporal_patterns(self, messages: List[Message]) -> Dict:
        """Analyze temporal communication patterns."""
        if len(messages) < 2:
            return {}
            
        patterns = {
            'response_times': [],
            'active_hours': defaultdict(int),
            'message_burst_patterns': [],
            'conversation_gaps': []
        }
        
        for i in range(1, len(messages)):
            time_diff = (messages[i].created_at - messages[i-1].created_at).total_seconds()
            patterns['response_times'].append(time_diff)
            
            # Active hours
            hour = messages[i].created_at.hour
            patterns['active_hours'][hour] += 1
            
            # Detect bursts (rapid messages)
            if time_diff < 60:  # Messages within a minute
                patterns['message_burst_patterns'].append(time_diff)
            elif time_diff > 3600:  # Gaps over an hour
                patterns['conversation_gaps'].append(time_diff)
                
        # Statistical summaries
        if patterns['response_times']:
            patterns['avg_response_time'] = np.mean(patterns['response_times'])
            patterns['response_time_std'] = np.std(patterns['response_times'])
            
        return dict(patterns)
        
    def _extract_emotional_patterns(self, corpus: List[str]) -> Dict:
        """Extract emotional expression patterns."""
        emotional_markers = {
            'positive': ['happy', 'love', 'great', 'awesome', 'good', 'nice', 'wonderful'],
            'negative': ['sad', 'hate', 'bad', 'terrible', 'awful', 'horrible', 'angry'],
            'excited': ['!!!', 'wow', 'amazing', 'omg', 'excited'],
            'uncertain': ['maybe', 'perhaps', 'might', 'possibly', 'probably', 'guess']
        }
        
        patterns = {category: 0 for category in emotional_markers}
        
        for text in corpus:
            text_lower = text.lower()
            for category, markers in emotional_markers.items():
                for marker in markers:
                    patterns[category] += text_lower.count(marker)
                    
        # Normalize by corpus size
        total_words = sum(len(text.split()) for text in corpus)
        if total_words > 0:
            patterns = {k: v/total_words for k, v in patterns.items()}
            
        return patterns
        
    def _extract_unique_phrases(self, corpus: List[str]) -> List[str]:
        """Extract frequently used unique phrases."""
        all_text = ' '.join(corpus).lower()
        
        # Common phrases this user uses
        phrase_patterns = [
            r'you know what',
            r'to be honest',
            r'in my opinion',
            r'at the end of the day',
            r'the thing is',
            r'i mean',
            r'kind of',
            r'sort of'
        ]
        
        unique_phrases = []
        for pattern in phrase_patterns:
            if re.search(pattern, all_text):
                unique_phrases.append(pattern)
                
        return unique_phrases
        
    def _extract_character_patterns(self, text: str) -> Dict:
        """Extract character-level patterns."""
        patterns = {
            'double_letters': len(re.findall(r'(.)\1', text)),
            'triple_letters': len(re.findall(r'(.)\1\1', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
        }
        return patterns
        
    def _calculate_yules_k(self, words: List[str]) -> float:
        """Calculate Yule's K measure of lexical diversity."""
        freq_dist = Counter(words)
        m1 = len(words)
        m2 = sum([freq ** 2 for freq in freq_dist.values()])
        
        if m1 == m2:
            return 0
            
        return 10000 * (m2 - m1) / (m1 ** 2)
        
    def _calculate_simpsons_d(self, words: List[str]) -> float:
        """Calculate Simpson's D index of diversity."""
        freq_dist = Counter(words)
        N = len(words)
        
        if N <= 1:
            return 0
            
        numerator = sum([n * (n - 1) for n in freq_dist.values()])
        denominator = N * (N - 1)
        
        if denominator == 0:
            return 0
            
        return 1 - (numerator / denominator)
        
    def _calculate_confidence(self, messages: List[Message]) -> float:
        """Calculate confidence score for the fingerprint."""
        if not messages:
            return 0.0
            
        # Factors affecting confidence
        message_count = len(messages)
        time_span = (messages[-1].created_at - messages[0].created_at).days if len(messages) > 1 else 0
        avg_length = np.mean([len(msg.content.split()) for msg in messages])
        
        # Calculate confidence (0-1 scale)
        confidence = min(1.0, (
            min(message_count / 100, 1.0) * 0.4 +  # More messages = higher confidence
            min(time_span / 30, 1.0) * 0.3 +       # Longer time span = higher confidence
            min(avg_length / 50, 1.0) * 0.3        # Longer messages = higher confidence
        ))
        
        return confidence
        
    def _empty_fingerprint(self) -> LinguisticFingerprint:
        """Return empty fingerprint structure."""
        return LinguisticFingerprint(
            ngram_distribution={},
            lexical_diversity={},
            stylistic_features={},
            temporal_signatures={},
            emotional_patterns={},
            vocabulary_size=0,
            avg_message_length=0,
            unique_phrases=[],
            confidence_score=0.0
        )


class TemporalReconstructor:
    """
    Reconstructs missing conversations using temporal patterns and context.
    Implements probabilistic generation based on behavioral archaeology.
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.context_window = 10  # Messages before/after gap
        self.min_confidence_threshold = 0.3
        
    async def reconstruct_gap(
        self,
        before_messages: List[Message],
        after_messages: List[Message],
        fingerprint: LinguisticFingerprint,
        time_gap: timedelta
    ) -> List[ReconstructedMessage]:
        """Reconstruct messages in a conversation gap."""
        
        # Analyze context
        context_analysis = await self._analyze_context(before_messages, after_messages)
        
        # Estimate number of missing messages
        estimated_count = self._estimate_message_count(time_gap, fingerprint)
        
        # Generate reconstructions
        reconstructed = []
        for i in range(estimated_count):
            # Calculate temporal position
            time_offset = time_gap * (i + 1) / (estimated_count + 1)
            estimated_time = before_messages[-1].created_at + time_offset if before_messages else datetime.utcnow()
            
            # Generate message content
            content = await self._generate_message(
                context_analysis,
                fingerprint,
                position=i,
                total=estimated_count
            )
            
            # Calculate confidence
            confidence = self._calculate_reconstruction_confidence(
                content,
                context_analysis,
                fingerprint
            )
            
            if confidence >= self.min_confidence_threshold:
                reconstructed_msg = ReconstructedMessage(
                    user_id=fingerprint.user_id,
                    content=content,
                    estimated_timestamp=estimated_time,
                    confidence_score=confidence,
                    reconstruction_method="temporal_pattern_analysis",
                    context_before=[msg.id for msg in before_messages[-3:]],
                    context_after=[msg.id for msg in after_messages[:3]],
                    linguistic_match_score=self._calculate_linguistic_match(content, fingerprint),
                    evidence_markers=self._extract_evidence_markers(content, context_analysis)
                )
                reconstructed.append(reconstructed_msg)
                
        return reconstructed
        
    async def _analyze_context(
        self,
        before: List[Message],
        after: List[Message]
    ) -> Dict:
        """Analyze conversation context around gap."""
        
        analysis = {
            'topics_before': [],
            'topics_after': [],
            'emotional_trajectory': [],
            'conversation_flow': '',
            'likely_transition': ''
        }
        
        # Extract topics
        if before:
            before_text = ' '.join([msg.content for msg in before[-5:]])
            analysis['topics_before'] = await self._extract_topics(before_text)
            
        if after:
            after_text = ' '.join([msg.content for msg in after[:5]])
            analysis['topics_after'] = await self._extract_topics(after_text)
            
        # Analyze emotional trajectory
        analysis['emotional_trajectory'] = self._analyze_emotional_flow(before, after)
        
        # Determine conversation flow type
        analysis['conversation_flow'] = self._classify_conversation_flow(before, after)
        
        # Predict likely transition
        analysis['likely_transition'] = await self._predict_transition(
            analysis['topics_before'],
            analysis['topics_after']
        )
        
        return analysis
        
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        prompt = f"""
        Extract the main topics from this conversation excerpt.
        Return as a JSON list of topics (max 5).
        
        Text: {text}
        """
        
        response = await self.llm_service.generate_response(prompt, temperature=0.3)
        
        try:
            topics = json.loads(response)
            return topics if isinstance(topics, list) else []
        except:
            return []
            
    def _analyze_emotional_flow(
        self,
        before: List[Message],
        after: List[Message]
    ) -> List[float]:
        """Analyze emotional trajectory across gap."""
        trajectory = []
        
        # Simple sentiment scoring (-1 to 1)
        positive_words = set(['good', 'great', 'happy', 'love', 'awesome', 'nice'])
        negative_words = set(['bad', 'sad', 'hate', 'terrible', 'awful', 'angry'])
        
        for messages in [before[-3:] if before else [], after[:3] if after else []]:
            for msg in messages:
                words = set(msg.content.lower().split())
                pos_count = len(words & positive_words)
                neg_count = len(words & negative_words)
                
                if pos_count + neg_count > 0:
                    sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    sentiment = 0.0
                    
                trajectory.append(sentiment)
                
        return trajectory
        
    def _classify_conversation_flow(
        self,
        before: List[Message],
        after: List[Message]
    ) -> str:
        """Classify the type of conversation flow."""
        if not before and not after:
            return "isolated"
        elif not before:
            return "conversation_start"
        elif not after:
            return "conversation_end"
            
        # Analyze message patterns
        before_lengths = [len(msg.content.split()) for msg in before[-3:]]
        after_lengths = [len(msg.content.split()) for msg in after[:3]]
        
        avg_before = np.mean(before_lengths) if before_lengths else 0
        avg_after = np.mean(after_lengths) if after_lengths else 0
        
        if avg_after > avg_before * 1.5:
            return "escalating"
        elif avg_after < avg_before * 0.5:
            return "de-escalating"
        else:
            return "steady"
            
    async def _predict_transition(
        self,
        topics_before: List[str],
        topics_after: List[str]
    ) -> str:
        """Predict likely topic transition."""
        if not topics_before or not topics_after:
            return "unknown"
            
        # Check for topic continuity
        common_topics = set(topics_before) & set(topics_after)
        if common_topics:
            return f"continuation of {list(common_topics)[0]}"
            
        # Predict transition type
        prompt = f"""
        Given these conversation topics:
        Before gap: {topics_before}
        After gap: {topics_after}
        
        What's the most likely transition? (one short phrase)
        """
        
        response = await self.llm_service.generate_response(prompt, temperature=0.5)
        return response.strip()
        
    def _estimate_message_count(
        self,
        time_gap: timedelta,
        fingerprint: LinguisticFingerprint
    ) -> int:
        """Estimate number of missing messages based on patterns."""
        if not fingerprint.temporal_signatures:
            return min(3, max(1, int(time_gap.total_seconds() / 3600)))  # Default: 1 per hour
            
        # Use average response time from fingerprint
        avg_response = fingerprint.temporal_signatures.get('avg_response_time', 1800)
        
        # Estimate based on typical conversation density
        estimated = int(time_gap.total_seconds() / avg_response)
        
        # Apply reasonable bounds
        return min(10, max(1, estimated))
        
    async def _generate_message(
        self,
        context: Dict,
        fingerprint: LinguisticFingerprint,
        position: int,
        total: int
    ) -> str:
        """Generate reconstructed message content."""
        
        # Build generation prompt
        style_desc = self._describe_writing_style(fingerprint)
        
        prompt = f"""
        Reconstruct a missing message based on these patterns:
        
        Writing style: {style_desc}
        Topics before: {context['topics_before']}
        Topics after: {context['topics_after']}
        Conversation flow: {context['conversation_flow']}
        Likely transition: {context['likely_transition']}
        Message position: {position + 1} of {total}
        
        Generate a single message that would naturally fit in this gap.
        Match the user's typical vocabulary and style.
        """
        
        # Generate with style matching
        message = await self.llm_service.generate_response(prompt, temperature=0.7)
        
        # Apply linguistic patterns
        message = self._apply_linguistic_patterns(message, fingerprint)
        
        return message
        
    def _describe_writing_style(self, fingerprint: LinguisticFingerprint) -> str:
        """Create description of user's writing style."""
        style_elements = []
        
        if fingerprint.avg_message_length:
            if fingerprint.avg_message_length < 10:
                style_elements.append("very brief")
            elif fingerprint.avg_message_length > 30:
                style_elements.append("detailed")
                
        if fingerprint.stylistic_features:
            punct = fingerprint.stylistic_features.get('punctuation_usage', {})
            if punct.get('!', 0) > 2:
                style_elements.append("enthusiastic")
            if punct.get('?', 0) > 3:
                style_elements.append("questioning")
                
        if fingerprint.unique_phrases:
            style_elements.append(f"uses phrases like '{fingerprint.unique_phrases[0]}'")
            
        return ', '.join(style_elements) if style_elements else "conversational"
        
    def _apply_linguistic_patterns(
        self,
        message: str,
        fingerprint: LinguisticFingerprint
    ) -> str:
        """Apply user's linguistic patterns to generated message."""
        
        # Apply capitalization patterns
        if fingerprint.stylistic_features:
            caps = fingerprint.stylistic_features.get('capitalization_patterns', {})
            if caps.get('all_caps', 0) > 5:
                # User tends to use caps for emphasis
                important_words = ['not', 'very', 'really', 'never', 'always']
                for word in important_words:
                    if word in message.lower():
                        message = message.replace(word, word.upper())
                        
        # Apply common abbreviations
        if fingerprint.stylistic_features.get('abbreviations'):
            replacements = {
                'by the way': 'btw',
                'to be honest': 'tbh',
                'in my opinion': 'imo',
                'for your information': 'fyi'
            }
            for full, abbrev in replacements.items():
                if full in message.lower() and abbrev in fingerprint.stylistic_features['abbreviations']:
                    message = re.sub(full, abbrev, message, flags=re.IGNORECASE)
                    
        return message
        
    def _calculate_reconstruction_confidence(
        self,
        content: str,
        context: Dict,
        fingerprint: LinguisticFingerprint
    ) -> float:
        """Calculate confidence score for reconstruction."""
        
        confidence_factors = []
        
        # Fingerprint confidence
        confidence_factors.append(fingerprint.confidence_score * 0.3)
        
        # Context coherence
        if context['topics_before'] and context['topics_after']:
            # Check if content bridges topics
            content_lower = content.lower()
            topic_matches = sum(1 for topic in context['topics_before'] + context['topics_after'] 
                              if topic.lower() in content_lower)
            confidence_factors.append(min(topic_matches / 3, 1.0) * 0.3)
            
        # Linguistic match
        ling_match = self._calculate_linguistic_match(content, fingerprint)
        confidence_factors.append(ling_match * 0.2)
        
        # Style consistency
        style_match = self._calculate_style_match(content, fingerprint)
        confidence_factors.append(style_match * 0.2)
        
        return sum(confidence_factors)
        
    def _calculate_linguistic_match(
        self,
        content: str,
        fingerprint: LinguisticFingerprint
    ) -> float:
        """Calculate how well content matches linguistic fingerprint."""
        if not fingerprint.ngram_distribution:
            return 0.5
            
        matches = 0
        total = 0
        
        content_words = content.lower().split()
        
        # Check unigram matches
        if 'unigrams' in fingerprint.ngram_distribution:
            common_words = list(fingerprint.ngram_distribution['unigrams'].keys())[:20]
            for word in content_words:
                total += 1
                if word in common_words:
                    matches += 1
                    
        # Check bigram patterns
        if 'bigrams' in fingerprint.ngram_distribution and len(content_words) > 1:
            common_bigrams = list(fingerprint.ngram_distribution['bigrams'].keys())[:10]
            for i in range(len(content_words) - 1):
                bigram = (content_words[i], content_words[i+1])
                if bigram in common_bigrams:
                    matches += 2  # Weight bigrams higher
                    
        return matches / max(total, 1)
        
    def _calculate_style_match(
        self,
        content: str,
        fingerprint: LinguisticFingerprint
    ) -> float:
        """Calculate style consistency match."""
        if not fingerprint.stylistic_features:
            return 0.5
            
        match_score = 0.0
        factors = 0
        
        # Message length consistency
        content_length = len(content.split())
        if fingerprint.avg_message_length > 0:
            length_diff = abs(content_length - fingerprint.avg_message_length)
            length_match = max(0, 1 - (length_diff / fingerprint.avg_message_length))
            match_score += length_match
            factors += 1
            
        # Punctuation usage
        if 'punctuation_usage' in fingerprint.stylistic_features:
            expected_punct = fingerprint.stylistic_features['punctuation_usage']
            for punct, expected_count in expected_punct.items():
                if expected_count > 0:
                    actual_count = content.count(punct)
                    punct_match = 1 - abs(actual_count - expected_count) / max(expected_count, 1)
                    match_score += max(0, punct_match)
                    factors += 1
                    
        return match_score / max(factors, 1)
        
    def _extract_evidence_markers(
        self,
        content: str,
        context: Dict
    ) -> List[str]:
        """Extract evidence markers supporting reconstruction."""
        markers = []
        
        # Topic continuity
        for topic in context['topics_before'] + context['topics_after']:
            if topic.lower() in content.lower():
                markers.append(f"mentions '{topic}'")
                
        # Emotional consistency
        if context['emotional_trajectory']:
            avg_emotion = np.mean(context['emotional_trajectory'])
            if avg_emotion > 0.3 and any(word in content.lower() for word in ['good', 'great', 'happy']):
                markers.append("positive sentiment match")
            elif avg_emotion < -0.3 and any(word in content.lower() for word in ['bad', 'sad', 'disappointed']):
                markers.append("negative sentiment match")
                
        # Conversation flow match
        if context['conversation_flow'] == 'escalating' and len(content) > 100:
            markers.append("escalating pattern match")
        elif context['conversation_flow'] == 'de-escalating' and len(content) < 50:
            markers.append("de-escalating pattern match")
            
        return markers


class PatternArchaeologist:
    """
    Identifies and extracts conversation patterns for reconstruction.
    Uses digital archaeology techniques to uncover hidden communication traces.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pattern_cache = {}
        self.min_pattern_frequency = 3
        
    async def discover_patterns(
        self,
        user_id: UUID,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[TemporalPattern]:
        """Discover temporal patterns in user's conversation history."""
        
        # Get message history
        query = select(Message).where(Message.user_id == user_id)
        if time_range:
            query = query.where(
                and_(
                    Message.created_at >= time_range[0],
                    Message.created_at <= time_range[1]
                )
            )
        query = query.order_by(Message.created_at)
        
        result = await self.db.execute(query)
        messages = result.scalars().all()
        
        if len(messages) < 10:
            return []
            
        # Extract various pattern types
        patterns = []
        
        # Temporal patterns
        temporal = await self._extract_temporal_patterns(messages)
        patterns.extend(temporal)
        
        # Conversational patterns
        conversational = await self._extract_conversational_patterns(messages)
        patterns.extend(conversational)
        
        # Behavioral patterns
        behavioral = await self._extract_behavioral_patterns(messages)
        patterns.extend(behavioral)
        
        # Linguistic evolution patterns
        evolution = await self._extract_evolution_patterns(messages)
        patterns.extend(evolution)
        
        return patterns
        
    async def _extract_temporal_patterns(self, messages: List[Message]) -> List[TemporalPattern]:
        """Extract time-based communication patterns."""
        patterns = []
        
        # Hour of day patterns
        hour_distribution = defaultdict(int)
        for msg in messages:
            hour_distribution[msg.created_at.hour] += 1
            
        # Find peak hours
        peak_hours = sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        pattern = TemporalPattern(
            user_id=messages[0].user_id,
            pattern_type="peak_hours",
            pattern_signature={
                "peak_hours": [h for h, _ in peak_hours],
                "distribution": dict(hour_distribution)
            },
            frequency=max(hour_distribution.values()),
            confidence=0.8,
            discovered_at=datetime.utcnow()
        )
        patterns.append(pattern)
        
        # Day of week patterns
        day_distribution = defaultdict(int)
        for msg in messages:
            day_distribution[msg.created_at.weekday()] += 1
            
        pattern = TemporalPattern(
            user_id=messages[0].user_id,
            pattern_type="weekly_rhythm",
            pattern_signature={
                "active_days": sorted(day_distribution.items(), key=lambda x: x[1], reverse=True),
                "weekend_ratio": (day_distribution[5] + day_distribution[6]) / sum(day_distribution.values())
            },
            frequency=len(messages),
            confidence=0.7,
            discovered_at=datetime.utcnow()
        )
        patterns.append(pattern)
        
        # Response time patterns
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].user_id == messages[i-1].user_id:
                continue  # Same user, not a response
            time_diff = (messages[i].created_at - messages[i-1].created_at).total_seconds()
            if time_diff < 3600:  # Within an hour
                response_times.append(time_diff)
                
        if response_times:
            pattern = TemporalPattern(
                user_id=messages[0].user_id,
                pattern_type="response_timing",
                pattern_signature={
                    "avg_response_time": np.mean(response_times),
                    "median_response_time": np.median(response_times),
                    "std_response_time": np.std(response_times)
                },
                frequency=len(response_times),
                confidence=0.9 if len(response_times) > 20 else 0.6,
                discovered_at=datetime.utcnow()
            )
            patterns.append(pattern)
            
        return patterns
        
    async def _extract_conversational_patterns(self, messages: List[Message]) -> List[TemporalPattern]:
        """Extract conversation structure patterns."""
        patterns = []
        
        # Message length patterns
        lengths = [len(msg.content.split()) for msg in messages]
        
        pattern = TemporalPattern(
            user_id=messages[0].user_id,
            pattern_type="message_length",
            pattern_signature={
                "avg_length": np.mean(lengths),
                "std_length": np.std(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "typical_range": [np.percentile(lengths, 25), np.percentile(lengths, 75)]
            },
            frequency=len(lengths),
            confidence=0.85,
            discovered_at=datetime.utcnow()
        )
        patterns.append(pattern)
        
        # Conversation starter patterns
        starters = defaultdict(int)
        for msg in messages:
            first_words = ' '.join(msg.content.split()[:3]).lower()
            starters[first_words] += 1
            
        common_starters = [s for s, c in starters.items() if c >= self.min_pattern_frequency]
        
        if common_starters:
            pattern = TemporalPattern(
                user_id=messages[0].user_id,
                pattern_type="conversation_starters",
                pattern_signature={
                    "common_starters": common_starters[:10],
                    "frequencies": {s: starters[s] for s in common_starters[:10]}
                },
                frequency=sum(starters[s] for s in common_starters),
                confidence=0.7,
                discovered_at=datetime.utcnow()
            )
            patterns.append(pattern)
            
        return patterns
        
    async def _extract_behavioral_patterns(self, messages: List[Message]) -> List[TemporalPattern]:
        """Extract behavioral communication patterns."""
        patterns = []
        
        # Editing patterns (if messages have edit history)
        edit_count = sum(1 for msg in messages if msg.edited_at)
        
        if edit_count > 0:
            pattern = TemporalPattern(
                user_id=messages[0].user_id,
                pattern_type="editing_behavior",
                pattern_signature={
                    "edit_rate": edit_count / len(messages),
                    "total_edits": edit_count
                },
                frequency=edit_count,
                confidence=0.6,
                discovered_at=datetime.utcnow()
            )
            patterns.append(pattern)
            
        # Question asking patterns
        question_count = sum(1 for msg in messages if '?' in msg.content)
        
        pattern = TemporalPattern(
            user_id=messages[0].user_id,
            pattern_type="questioning_style",
            pattern_signature={
                "question_rate": question_count / len(messages),
                "total_questions": question_count,
                "questions_per_conversation": question_count / max(len(messages) / 10, 1)
            },
            frequency=question_count,
            confidence=0.75,
            discovered_at=datetime.utcnow()
        )
        patterns.append(pattern)
        
        # Emoji usage patterns
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        
        emoji_messages = sum(1 for msg in messages if emoji_pattern.search(msg.content))
        
        if emoji_messages > 0:
            pattern = TemporalPattern(
                user_id=messages[0].user_id,
                pattern_type="emoji_usage",
                pattern_signature={
                    "emoji_rate": emoji_messages / len(messages),
                    "total_emoji_messages": emoji_messages
                },
                frequency=emoji_messages,
                confidence=0.8,
                discovered_at=datetime.utcnow()
            )
            patterns.append(pattern)
            
        return patterns
        
    async def _extract_evolution_patterns(self, messages: List[Message]) -> List[TemporalPattern]:
        """Extract patterns showing linguistic evolution over time."""
        patterns = []
        
        if len(messages) < 20:
            return patterns
            
        # Split messages into time periods
        total_span = (messages[-1].created_at - messages[0].created_at).days
        if total_span < 7:
            return patterns
            
        # Analyze first quarter vs last quarter
        quarter_size = len(messages) // 4
        early_messages = messages[:quarter_size]
        recent_messages = messages[-quarter_size:]
        
        # Vocabulary evolution
        early_vocab = set(' '.join(msg.content for msg in early_messages).lower().split())
        recent_vocab = set(' '.join(msg.content for msg in recent_messages).lower().split())
        
        new_words = recent_vocab - early_vocab
        abandoned_words = early_vocab - recent_vocab
        
        pattern = TemporalPattern(
            user_id=messages[0].user_id,
            pattern_type="vocabulary_evolution",
            pattern_signature={
                "new_vocabulary": list(new_words)[:20],
                "abandoned_vocabulary": list(abandoned_words)[:20],
                "vocabulary_growth_rate": len(new_words) / len(early_vocab) if early_vocab else 0,
                "stability_score": len(early_vocab & recent_vocab) / len(early_vocab | recent_vocab) if early_vocab | recent_vocab else 0
            },
            frequency=len(new_words) + len(abandoned_words),
            confidence=0.65,
            discovered_at=datetime.utcnow()
        )
        patterns.append(pattern)
        
        # Style evolution
        early_avg_length = np.mean([len(msg.content.split()) for msg in early_messages])
        recent_avg_length = np.mean([len(msg.content.split()) for msg in recent_messages])
        
        pattern = TemporalPattern(
            user_id=messages[0].user_id,
            pattern_type="style_evolution",
            pattern_signature={
                "early_avg_length": early_avg_length,
                "recent_avg_length": recent_avg_length,
                "length_change_ratio": recent_avg_length / early_avg_length if early_avg_length > 0 else 1,
                "trend": "expanding" if recent_avg_length > early_avg_length * 1.2 else "contracting" if recent_avg_length < early_avg_length * 0.8 else "stable"
            },
            frequency=len(messages),
            confidence=0.7,
            discovered_at=datetime.utcnow()
        )
        patterns.append(pattern)
        
        return patterns


class ArchaeologyEngine:
    """
    Main engine for temporal conversation archaeology.
    Coordinates reconstruction of lost conversations.
    """
    
    def __init__(self, db: AsyncSession, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
        self.llm_service = LLMService()
        self.profiler = LinguisticProfiler()
        self.reconstructor = TemporalReconstructor(self.llm_service)
        self.archaeologist = PatternArchaeologist(db)
        
    async def analyze_conversation_gaps(
        self,
        user_id: UUID,
        threshold: timedelta = timedelta(hours=1)
    ) -> List[Dict]:
        """Identify significant gaps in conversation history."""
        
        # Get all messages
        result = await self.db.execute(
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.created_at)
        )
        messages = result.scalars().all()
        
        if len(messages) < 2:
            return []
            
        gaps = []
        for i in range(1, len(messages)):
            time_diff = messages[i].created_at - messages[i-1].created_at
            
            if time_diff > threshold:
                gaps.append({
                    'start': messages[i-1].created_at,
                    'end': messages[i].created_at,
                    'duration': time_diff,
                    'before_message': messages[i-1],
                    'after_message': messages[i],
                    'gap_index': i
                })
                
        return gaps
        
    async def reconstruct_conversation(
        self,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
        context_messages: Optional[List[Message]] = None
    ) -> GhostConversation:
        """Reconstruct a lost conversation within time range."""
        
        # Create or get linguistic fingerprint
        if not context_messages:
            # Get historical messages for fingerprinting
            result = await self.db.execute(
                select(Message)
                .where(
                    and_(
                        Message.user_id == user_id,
                        or_(
                            Message.created_at < start_time,
                            Message.created_at > end_time
                        )
                    )
                )
                .order_by(Message.created_at)
                .limit(100)
            )
            context_messages = result.scalars().all()
            
        fingerprint = await self.profiler.create_fingerprint(context_messages)
        
        # Save fingerprint
        self.db.add(fingerprint)
        
        # Get messages around the gap
        result = await self.db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    Message.created_at < start_time
                )
            )
            .order_by(Message.created_at.desc())
            .limit(10)
        )
        before_messages = list(reversed(result.scalars().all()))
        
        result = await self.db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    Message.created_at > end_time
                )
            )
            .order_by(Message.created_at)
            .limit(10)
        )
        after_messages = result.scalars().all()
        
        # Reconstruct messages
        reconstructed = await self.reconstructor.reconstruct_gap(
            before_messages,
            after_messages,
            fingerprint,
            end_time - start_time
        )
        
        # Save reconstructed messages
        for msg in reconstructed:
            self.db.add(msg)
            
        # Create ghost conversation
        ghost_conversation = GhostConversation(
            user_id=user_id,
            time_range_start=start_time,
            time_range_end=end_time,
            reconstructed_messages=[msg.id for msg in reconstructed],
            confidence_score=np.mean([msg.confidence_score for msg in reconstructed]) if reconstructed else 0,
            reconstruction_method="temporal_archaeology",
            fingerprint_id=fingerprint.id,
            evidence_summary=self._create_evidence_summary(reconstructed, before_messages, after_messages)
        )
        
        self.db.add(ghost_conversation)
        await self.db.commit()
        
        return ghost_conversation
        
    def _create_evidence_summary(
        self,
        reconstructed: List[ReconstructedMessage],
        before: List[Message],
        after: List[Message]
    ) -> Dict:
        """Create summary of reconstruction evidence."""
        
        summary = {
            'messages_reconstructed': len(reconstructed),
            'avg_confidence': np.mean([msg.confidence_score for msg in reconstructed]) if reconstructed else 0,
            'methods_used': list(set(msg.reconstruction_method for msg in reconstructed)),
            'context_messages': {
                'before': len(before),
                'after': len(after)
            },
            'evidence_markers': []
        }
        
        # Collect all evidence markers
        for msg in reconstructed:
            if msg.evidence_markers:
                summary['evidence_markers'].extend(msg.evidence_markers)
                
        summary['evidence_markers'] = list(set(summary['evidence_markers']))[:10]
        
        return summary
        
    async def create_archaeology_session(
        self,
        user_id: UUID,
        target_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ArchaeologySession:
        """Create a new archaeology session for exploring lost conversations."""
        
        # Discover patterns
        patterns = await self.archaeologist.discover_patterns(user_id, target_period)
        
        # Identify gaps
        gaps = await self.analyze_conversation_gaps(user_id)
        
        # Create session
        session = ArchaeologySession(
            user_id=user_id,
            patterns_discovered=[p.id for p in patterns],
            gaps_identified=len(gaps),
            total_messages_analyzed=await self._count_messages(user_id, target_period),
            time_range_analyzed={
                'start': target_period[0].isoformat() if target_period else None,
                'end': target_period[1].isoformat() if target_period else None
            },
            session_status="active"
        )
        
        self.db.add(session)
        
        # Save patterns
        for pattern in patterns:
            self.db.add(pattern)
            
        await self.db.commit()
        await self.db.refresh(session)
        
        # Cache session data
        await self._cache_session(session, patterns, gaps)
        
        return session
        
    async def _count_messages(
        self,
        user_id: UUID,
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> int:
        """Count messages in time range."""
        query = select(func.count(Message.id)).where(Message.user_id == user_id)
        
        if time_range:
            query = query.where(
                and_(
                    Message.created_at >= time_range[0],
                    Message.created_at <= time_range[1]
                )
            )
            
        result = await self.db.execute(query)
        return result.scalar() or 0
        
    async def _cache_session(
        self,
        session: ArchaeologySession,
        patterns: List[TemporalPattern],
        gaps: List[Dict]
    ):
        """Cache archaeology session data."""
        
        cache_data = {
            'session_id': str(session.id),
            'patterns': [
                {
                    'type': p.pattern_type,
                    'signature': p.pattern_signature,
                    'confidence': p.confidence
                }
                for p in patterns[:20]  # Limit to top 20
            ],
            'gaps': [
                {
                    'start': g['start'].isoformat(),
                    'end': g['end'].isoformat(),
                    'duration': g['duration'].total_seconds()
                }
                for g in gaps[:50]  # Limit to top 50
            ]
        }
        
        key = f"archaeology:session:{session.id}"
        await self.redis.setex(
            key,
            7200,  # 2 hour cache
            json.dumps(cache_data, default=str)
        )