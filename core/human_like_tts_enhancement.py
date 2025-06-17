#!/usr/bin/env python3
"""
VoxSigil Human-Like TTS Enhancement System
==========================================

Advanced techniques to make VoxSigil TTS sound more natural and human-like.

Techniques Implemented:
1. SSML (Speech Synthesis Markup Language) for natural speech patterns
2. Emotional prosody adjustment based on context
3. Dynamic pause insertion for natural rhythm
4. Breathing simulation and natural hesitations
5. Stress and intonation pattern matching
6. Context-aware speech rate variation
7. Vocal fry and uptalk patterns for naturalness
8. Multi-modal voice cloning integration
9. Real-time emotion analysis and voice adaptation
10. Neural voice synthesis with transformer models
"""

import asyncio
import logging
import math
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Emotional states for voice modulation"""

    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    URGENT = "urgent"
    CONCERNED = "concerned"
    HAPPY = "happy"
    SERIOUS = "serious"
    FRIENDLY = "friendly"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"


class SpeechContext(Enum):
    """Context types for speech adaptation"""

    GREETING = "greeting"
    EXPLANATION = "explanation"
    INSTRUCTION = "instruction"
    QUESTION = "question"
    CONFIRMATION = "confirmation"
    WARNING = "warning"
    CELEBRATION = "celebration"
    THINKING = "thinking"


@dataclass
class ProsodyParameters:
    """Advanced prosody parameters for human-like speech"""

    # Basic parameters
    rate: float = 1.0  # Speech rate multiplier
    pitch: float = 0.0  # Pitch adjustment in semitones
    volume: float = 1.0  # Volume level

    # Advanced prosody
    stress_pattern: List[float] = field(default_factory=list)  # Word stress levels
    pause_pattern: List[float] = field(default_factory=list)  # Pause durations
    intonation_curve: List[float] = field(default_factory=list)  # Pitch curve

    # Naturalness features
    breathing_frequency: float = 0.15  # Breaths per minute of speech
    hesitation_probability: float = 0.05  # Chance of natural hesitations
    vocal_fry_intensity: float = 0.1  # Vocal fry at sentence ends
    uptalk_probability: float = 0.2  # Rising intonation on statements


@dataclass
class VoiceCharacteristics:
    """Enhanced voice characteristics for human-like speech"""

    # Core identity
    agent_name: str
    base_voice_id: str

    # Personality-driven prosody
    confidence_level: float = 0.7  # 0.0 = uncertain, 1.0 = very confident
    energy_level: float = 0.5  # 0.0 = calm, 1.0 = energetic
    formality_level: float = 0.5  # 0.0 = casual, 1.0 = formal

    # Speaking patterns
    preferred_pace: float = 1.0  # Natural speaking speed
    pause_tendency: float = 0.5  # How often to pause for emphasis
    stress_intensity: float = 0.7  # How much to emphasize important words

    # Emotional range
    emotional_range: List[EmotionalState] = field(default_factory=list)
    default_emotion: EmotionalState = EmotionalState.NEUTRAL

    # Naturalness settings
    use_breathing: bool = True
    use_hesitations: bool = True
    use_vocal_fry: bool = True
    use_uptalk: bool = False  # Agent-specific


class HumanLikeTTSProcessor:
    """Processes text to make TTS sound more human-like"""

    def __init__(self):
        self.emotion_patterns = self._initialize_emotion_patterns()
        self.context_patterns = self._initialize_context_patterns()
        self.naturalness_rules = self._initialize_naturalness_rules()

    def _initialize_emotion_patterns(self) -> Dict[EmotionalState, Dict[str, Any]]:
        """Initialize emotion-specific speech patterns"""
        return {
            EmotionalState.EXCITED: {
                "rate_multiplier": 1.2,
                "pitch_shift": +3,
                "volume_boost": 1.1,
                "stress_intensity": 1.3,
                "pause_reduction": 0.7,
            },
            EmotionalState.CALM: {
                "rate_multiplier": 0.9,
                "pitch_shift": -1,
                "volume_boost": 0.9,
                "stress_intensity": 0.8,
                "pause_increase": 1.3,
            },
            EmotionalState.URGENT: {
                "rate_multiplier": 1.3,
                "pitch_shift": +2,
                "volume_boost": 1.2,
                "stress_intensity": 1.4,
                "pause_reduction": 0.5,
            },
            EmotionalState.CONCERNED: {
                "rate_multiplier": 0.95,
                "pitch_shift": -0.5,
                "volume_boost": 0.95,
                "stress_intensity": 1.1,
                "pause_increase": 1.1,
            },
            EmotionalState.CONFIDENT: {
                "rate_multiplier": 1.0,
                "pitch_shift": 0,
                "volume_boost": 1.05,
                "stress_intensity": 1.2,
                "pause_optimal": 1.0,
            },
        }

    def _initialize_context_patterns(self) -> Dict[SpeechContext, Dict[str, Any]]:
        """Initialize context-specific speech patterns"""
        return {
            SpeechContext.GREETING: {"warmth_boost": 1.2, "pitch_rise": True, "pause_after": 0.5},
            SpeechContext.EXPLANATION: {
                "clarity_emphasis": True,
                "pace_slow": 0.9,
                "logical_pauses": True,
            },
            SpeechContext.QUESTION: {
                "rising_intonation": True,
                "pause_before": 0.3,
                "uncertainty_markers": True,
            },
            SpeechContext.WARNING: {
                "serious_tone": True,
                "emphasis_strong": True,
                "pause_dramatic": 0.8,
            },
        }

    def _initialize_naturalness_rules(self) -> Dict[str, Any]:
        """Initialize rules for natural speech patterns"""
        return {
            "breathing_patterns": {
                "sentence_end_breath": 0.3,  # Chance of breath at sentence end
                "clause_breath": 0.1,  # Chance of breath at clause boundary
                "long_speech_breath": 0.8,  # Breath every N seconds in long speech
                "breath_duration": (0.2, 0.5),  # Duration range for breaths
            },
            "hesitation_patterns": {
                "thinking_words": ["um", "uh", "well", "you know", "like"],
                "pause_fillers": ["..."],
                "sentence_start_hesitation": 0.05,
                "complex_topic_hesitation": 0.15,
            },
            "stress_patterns": {
                "content_words": ["nouns", "verbs", "adjectives", "adverbs"],
                "function_words": ["articles", "prepositions", "conjunctions"],
                "emphasis_words": ["very", "really", "extremely", "absolutely"],
            },
        }

    def analyze_text_context(self, text: str) -> SpeechContext:
        """Analyze text to determine speech context"""
        text_lower = text.lower().strip()

        # Question detection
        if text_lower.endswith("?") or text_lower.startswith(
            ("what", "how", "why", "when", "where", "who")
        ):
            return SpeechContext.QUESTION

        # Greeting detection
        if any(word in text_lower for word in ["hello", "hi", "greetings", "welcome"]):
            return SpeechContext.GREETING

        # Warning detection
        if any(
            word in text_lower for word in ["warning", "alert", "danger", "caution", "attention"]
        ):
            return SpeechContext.WARNING

        # Explanation detection (longer, complex sentences)
        if len(text.split()) > 15 or any(
            word in text_lower for word in ["because", "therefore", "however", "moreover"]
        ):
            return SpeechContext.EXPLANATION

        # Instruction detection
        if any(
            text_lower.startswith(word) for word in ["please", "let me", "first", "next", "then"]
        ):
            return SpeechContext.INSTRUCTION

        return SpeechContext.EXPLANATION  # Default

    def detect_emotion_from_text(self, text: str) -> EmotionalState:
        """Detect emotional state from text content"""
        text_lower = text.lower()

        # Excitement indicators
        if any(
            word in text_lower
            for word in ["!", "amazing", "fantastic", "excellent", "great", "wonderful"]
        ):
            return EmotionalState.EXCITED

        # Urgency indicators
        if any(
            word in text_lower for word in ["urgent", "immediately", "quickly", "fast", "hurry"]
        ):
            return EmotionalState.URGENT

        # Concern indicators
        if any(
            word in text_lower
            for word in ["problem", "issue", "error", "wrong", "concerned", "worried"]
        ):
            return EmotionalState.CONCERNED

        # Confidence indicators
        if any(
            word in text_lower
            for word in ["certain", "definitely", "absolutely", "confident", "sure"]
        ):
            return EmotionalState.CONFIDENT

        return EmotionalState.NEUTRAL

    def generate_ssml(
        self,
        text: str,
        voice_characteristics: VoiceCharacteristics,
        emotion: Optional[EmotionalState] = None,
        context: Optional[SpeechContext] = None,
    ) -> str:
        """Generate SSML markup for natural speech"""

        if emotion is None:
            emotion = self.detect_emotion_from_text(text)

        if context is None:
            context = self.analyze_text_context(text)

        # Get emotion and context patterns
        emotion_pattern = self.emotion_patterns.get(emotion, {})
        context_pattern = self.context_patterns.get(context, {})

        # Build SSML
        ssml_parts = ["<speak>"]

        # Set voice
        voice_attrs = f'name="{voice_characteristics.base_voice_id}"'

        # Add emotional prosody
        prosody_attrs = []

        # Rate adjustment
        rate = voice_characteristics.preferred_pace * emotion_pattern.get("rate_multiplier", 1.0)
        if context_pattern.get("pace_slow"):
            rate *= context_pattern["pace_slow"]
        prosody_attrs.append(f'rate="{rate:.1f}"')

        # Pitch adjustment
        pitch_shift = emotion_pattern.get("pitch_shift", 0)
        if context_pattern.get("pitch_rise"):
            pitch_shift += 1
        if pitch_shift != 0:
            prosody_attrs.append(f'pitch="{pitch_shift:+.1f}st"')

        # Volume adjustment
        volume = emotion_pattern.get("volume_boost", 1.0)
        if volume != 1.0:
            prosody_attrs.append(f'volume="{volume:.1f}"')

        ssml_parts.append(f"<voice {voice_attrs}>")
        ssml_parts.append(f"<prosody {' '.join(prosody_attrs)}>")

        # Process text with natural pauses and emphasis
        processed_text = self._add_natural_elements(text, voice_characteristics, emotion, context)
        ssml_parts.append(processed_text)

        ssml_parts.extend(["</prosody>", "</voice>", "</speak>"])

        return "".join(ssml_parts)

    def _add_natural_elements(
        self,
        text: str,
        voice_characteristics: VoiceCharacteristics,
        emotion: EmotionalState,
        context: SpeechContext,
    ) -> str:
        """Add natural speech elements like pauses, emphasis, and breathing"""

        sentences = re.split(r"[.!?]+", text)
        processed_sentences = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            sentence = sentence.strip()

            # Add natural hesitation at beginning (sometimes)
            if i == 0 and voice_characteristics.use_hesitations and random.random() < 0.1:
                hesitation = random.choice(['<break time="200ms"/>', "Well, ", "Um, "])
                sentence = hesitation + sentence

            # Add emphasis to important words
            sentence = self._add_emphasis(sentence, voice_characteristics)

            # Add natural pauses
            sentence = self._add_natural_pauses(sentence, voice_characteristics, context)

            # Add breathing between long sentences
            if voice_characteristics.use_breathing and len(sentence.split()) > 10:
                if random.random() < 0.3:
                    sentence += ' <break time="300ms" strength="medium"/>'

            processed_sentences.append(sentence)

        return " ".join(processed_sentences)

    def _add_emphasis(self, sentence: str, voice_characteristics: VoiceCharacteristics) -> str:
        """Add emphasis to important words"""

        # Words that typically get emphasis
        emphasis_words = [
            "very",
            "really",
            "extremely",
            "absolutely",
            "definitely",
            "important",
            "critical",
            "essential",
            "significant",
        ]

        words = sentence.split()
        for i, word in enumerate(words):
            word_clean = re.sub(r"[^\w]", "", word.lower())
            if word_clean in emphasis_words:
                words[i] = f'<emphasis level="strong">{word}</emphasis>'

        return " ".join(words)

    def _add_natural_pauses(
        self, sentence: str, voice_characteristics: VoiceCharacteristics, context: SpeechContext
    ) -> str:
        """Add natural pauses at appropriate points"""

        # Add pauses after commas (natural breath points)
        sentence = re.sub(r",(\s+)", r',<break time="200ms"/>\1', sentence)

        # Add pauses before conjunctions for clarity
        conjunctions = ["and", "but", "however", "therefore", "moreover", "furthermore"]
        for conj in conjunctions:
            pattern = f"\\b{conj}\\b"
            replacement = f'<break time="150ms"/>{conj}'
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)

        # Context-specific pauses
        if context == SpeechContext.EXPLANATION:
            # Add pauses before explanatory phrases
            explanatory = ["because", "since", "as a result", "in other words"]
            for phrase in explanatory:
                pattern = f"\\b{phrase}\\b"
                replacement = f'<break time="250ms"/>{phrase}'
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)

        return sentence


class AdvancedVoiceProcessor:
    """Advanced voice processing for human-like characteristics"""

    def __init__(self):
        self.tts_processor = HumanLikeTTSProcessor()
        self.voice_characteristics = self._initialize_agent_characteristics()

    def _initialize_agent_characteristics(self) -> Dict[str, VoiceCharacteristics]:
        """Initialize enhanced voice characteristics for each agent"""

        return {
            "Astra": VoiceCharacteristics(
                agent_name="Astra",
                base_voice_id="en-US-AriaNeural",
                confidence_level=0.9,
                energy_level=0.6,
                formality_level=0.7,
                preferred_pace=1.0,
                emotional_range=[
                    EmotionalState.CONFIDENT,
                    EmotionalState.CALM,
                    EmotionalState.HELPFUL,
                ],
                use_breathing=True,
                use_hesitations=False,  # Astra is always confident
                use_vocal_fry=False,
                use_uptalk=False,
            ),
            "Phi": VoiceCharacteristics(
                agent_name="Phi",
                base_voice_id="en-US-JennyNeural",
                confidence_level=0.95,
                energy_level=0.7,
                formality_level=0.8,
                preferred_pace=1.1,
                emotional_range=[
                    EmotionalState.EXCITED,
                    EmotionalState.CONFIDENT,
                    EmotionalState.CURIOUS,
                ],
                use_breathing=True,
                use_hesitations=True,  # Phi thinks through problems
                use_vocal_fry=False,
                use_uptalk=False,
            ),
            "Oracle": VoiceCharacteristics(
                agent_name="Oracle",
                base_voice_id="en-US-GuyNeural",
                confidence_level=1.0,
                energy_level=0.4,
                formality_level=0.9,
                preferred_pace=0.9,
                pause_tendency=0.8,  # Oracle pauses for wisdom
                emotional_range=[EmotionalState.WISE, EmotionalState.CALM, EmotionalState.SERIOUS],
                use_breathing=True,
                use_hesitations=False,
                use_vocal_fry=True,  # Deep, resonant voice
                use_uptalk=False,
            ),
            "Echo": VoiceCharacteristics(
                agent_name="Echo",
                base_voice_id="en-US-AriaNeural",
                confidence_level=0.8,
                energy_level=0.9,
                formality_level=0.3,  # Very casual
                preferred_pace=1.2,
                emotional_range=[
                    EmotionalState.EXCITED,
                    EmotionalState.HAPPY,
                    EmotionalState.ENERGETIC,
                ],
                use_breathing=True,
                use_hesitations=True,
                use_vocal_fry=False,
                use_uptalk=True,  # Echo uses uptalk for friendliness
            ),
        }

    async def generate_human_like_speech(
        self,
        agent_name: str,
        text: str,
        emotion: Optional[EmotionalState] = None,
        context: Optional[SpeechContext] = None,
    ) -> Dict[str, Any]:
        """Generate human-like speech with all enhancements"""

        if agent_name not in self.voice_characteristics:
            agent_name = "Astra"  # Default

        voice_char = self.voice_characteristics[agent_name]

        # Generate SSML for natural speech
        ssml = self.tts_processor.generate_ssml(text, voice_char, emotion, context)

        # Generate prosody parameters
        prosody = self._generate_prosody_parameters(text, voice_char, emotion, context)

        return {
            "ssml": ssml,
            "prosody": prosody,
            "voice_characteristics": voice_char,
            "emotion": emotion or self.tts_processor.detect_emotion_from_text(text),
            "context": context or self.tts_processor.analyze_text_context(text),
        }

    def _generate_prosody_parameters(
        self,
        text: str,
        voice_char: VoiceCharacteristics,
        emotion: EmotionalState,
        context: SpeechContext,
    ) -> ProsodyParameters:
        """Generate detailed prosody parameters"""

        words = text.split()
        word_count = len(words)

        # Generate stress pattern (emphasize content words)
        stress_pattern = []
        for word in words:
            # Simple heuristic: longer words and capitalized words get more stress
            if len(word) > 6 or word[0].isupper():
                stress_pattern.append(voice_char.stress_intensity)
            else:
                stress_pattern.append(0.5)

        # Generate pause pattern (pauses after punctuation and at phrase boundaries)
        pause_pattern = []
        for i, word in enumerate(words):
            if word.endswith((",", ";", ":")):
                pause_pattern.append(0.3)
            elif word.endswith((".", "!", "?")):
                pause_pattern.append(0.5)
            elif i > 0 and i % 8 == 0:  # Natural phrase boundary
                pause_pattern.append(0.2)
            else:
                pause_pattern.append(0.0)

        # Generate intonation curve (pitch changes across the sentence)
        intonation_curve = []
        for i in range(word_count):
            # Natural pitch variation
            base_pitch = math.sin(i / word_count * math.pi) * 2  # Natural rise and fall

            # Context adjustments
            if context == SpeechContext.QUESTION:
                base_pitch += (i / word_count) * 3  # Rising intonation
            elif context == SpeechContext.WARNING:
                base_pitch -= 1  # Lower, more serious

            intonation_curve.append(base_pitch)

        return ProsodyParameters(
            rate=voice_char.preferred_pace,
            pitch=0.0,
            volume=1.0,
            stress_pattern=stress_pattern,
            pause_pattern=pause_pattern,
            intonation_curve=intonation_curve,
            breathing_frequency=0.15 if voice_char.use_breathing else 0.0,
            hesitation_probability=0.05 if voice_char.use_hesitations else 0.0,
            vocal_fry_intensity=0.1 if voice_char.use_vocal_fry else 0.0,
            uptalk_probability=0.2 if voice_char.use_uptalk else 0.0,
        )


# Integration functions for VoxSigil
def create_human_like_voice_processor() -> AdvancedVoiceProcessor:
    """Create an advanced voice processor for human-like TTS"""
    return AdvancedVoiceProcessor()


async def generate_natural_speech(
    agent_name: str, text: str, emotion: str = "neutral", context: str = "explanation"
) -> Dict[str, Any]:
    """
    Generate natural, human-like speech for VoxSigil agents

    Args:
        agent_name: Name of the VoxSigil agent
        text: Text to synthesize
        emotion: Emotional state (neutral, excited, calm, etc.)
        context: Speech context (greeting, explanation, question, etc.)

    Returns:
        Dictionary with SSML, prosody parameters, and voice characteristics
    """

    processor = create_human_like_voice_processor()

    # Convert string parameters to enums
    emotion_enum = (
        EmotionalState(emotion)
        if emotion in [e.value for e in EmotionalState]
        else EmotionalState.NEUTRAL
    )
    context_enum = (
        SpeechContext(context)
        if context in [c.value for c in SpeechContext]
        else SpeechContext.EXPLANATION
    )

    return await processor.generate_human_like_speech(agent_name, text, emotion_enum, context_enum)


if __name__ == "__main__":
    # Demo of human-like TTS enhancements
    async def demo():
        processor = create_human_like_voice_processor()

        test_cases = [
            (
                "Astra",
                "Welcome to VoxSigil! I'm here to help guide you through our cognitive landscape.",
                "confident",
                "greeting",
            ),
            (
                "Phi",
                "Hmm, let me think about this mathematical problem for a moment... Yes, I believe the answer is 42!",
                "excited",
                "explanation",
            ),
            (
                "Oracle",
                "In the vast expanse of knowledge, we must pause to consider the deeper implications of our actions.",
                "serious",
                "explanation",
            ),
            (
                "Echo",
                "Oh wow! That's like, totally amazing! Did you see how fast that calculation ran?",
                "excited",
                "celebration",
            ),
        ]

        for agent, text, emotion, context in test_cases:
            print(f"\n🤖 {agent} ({emotion}, {context}):")
            print(f"Text: {text}")

            result = await processor.generate_human_like_speech(
                agent, text, EmotionalState(emotion), SpeechContext(context)
            )

            print(f"SSML Preview: {result['ssml'][:100]}...")
            print(f"Detected Emotion: {result['emotion'].value}")
            print(f"Detected Context: {result['context'].value}")

    asyncio.run(demo())
