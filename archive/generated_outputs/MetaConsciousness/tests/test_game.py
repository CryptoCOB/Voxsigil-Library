"""
Tests for the Game-Semantic Compression (GSC) framework.
"""

import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MetaConsciousness.frameworks.game_compression.dialogue_functor import DialogueStrategyFunctor

class TestDialogueStrategyFunctor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.functor = DialogueStrategyFunctor(
            key_phrase_weight=1.5,
            question_weight=1.2,
            contradiction_weight=1.4,
            sentiment_change_weight=1.3,
            min_significance=0.7,
            preserve_opening=True,
            preserve_closing=True
        )
        
        # Create test dialogue samples
        self.dialogue_samples = [
            self._create_test_dialogue("question_answer", 10),
            self._create_test_dialogue("agreement_disagreement", 10),
            self._create_test_dialogue("multi_speaker", 15),
            self._create_test_dialogue("narrative", 20)
        ]
    
    def _create_test_dialogue(self, pattern: str = "question_answer", length: int = 10) -> List[Tuple[str, str]]:
        """Create a test dialogue with the specified pattern."""
        if pattern == "question_answer":
            dialogue = []
            for i in range(length // 2):
                dialogue.append(("Alice", f"What do you think about topic {i+1}?"))
                dialogue.append(("Bob", f"I think topic {i+1} is interesting because of reason {i+1}."))
            return dialogue
            
        elif pattern == "agreement_disagreement":
            dialogue = []
            for i in range(length // 2):
                if i % 2 == 0:
                    dialogue.append(("Alice", f"I believe statement {i+1} is correct."))
                    dialogue.append(("Bob", f"I agree with you about statement {i+1}."))
                else:
                    dialogue.append(("Alice", f"I believe statement {i+1} is correct."))
                    dialogue.append(("Bob", f"I disagree with statement {i+1} because of reason {i+1}."))
            return dialogue
            
        elif pattern == "multi_speaker":
            speakers = ["Alice", "Bob", "Charlie", "David"]
            dialogue = []
            for i in range(length):
                speaker = speakers[i % len(speakers)]
                if i % 7 == 0:
                    dialogue.append((speaker, f"What does everyone think about topic {i//7 + 1}?"))
                else:
                    dialogue.append((speaker, f"I have opinion {i} about the current topic."))
            return dialogue
            
        elif pattern == "narrative":
            dialogue = []
            speakers = ["Narrator", "Character1", "Character2"]
            
            # Create a simple narrative with key plot points
            dialogue.append((speakers[0], "Once upon a time in a distant land..."))
            dialogue.append((speakers[0], "There lived two friends who were inseparable."))
            dialogue.append((speakers[1], "I've always valued our friendship."))
            dialogue.append((speakers[2], "As have I. We've been through so much together."))
            dialogue.append((speakers[0], "But one day, something unexpected happened."))
            dialogue.append((speakers[0], "A mysterious stranger arrived in town."))
            dialogue.append((speakers[2], "Who do you think that could be?"))
            dialogue.append((speakers[1], "I'm not sure, but they seem familiar somehow."))
            dialogue.append((speakers[0], "The stranger approached them with an urgent message."))
            dialogue.append((speakers[1], "What news do you bring?"))
            dialogue.append((speakers[0], "\"I bring news of great importance,\" the stranger said."))
            dialogue.append((speakers[0], "\"A darkness is coming that threatens everything.\""))
            dialogue.append((speakers[2], "What kind of darkness? What do you mean?"))
            dialogue.append((speakers[0], "The stranger explained about an ancient prophecy."))
            dialogue.append((speakers[1], "I don't believe in prophecies."))
            dialogue.append((speakers[2], "But what if it's true? We should prepare."))
            dialogue.append((speakers[1], "You're right. Better safe than sorry."))
            dialogue.append((speakers[0], "And so they began their preparations."))
            dialogue.append((speakers[0], "Little did they know, the prophecy was indeed real."))
            dialogue.append((speakers[0], "And their journey was just beginning..."))
            
            # Trim or extend to match requested length
            if len(dialogue) > length:
                dialogue = dialogue[:length]
            while len(dialogue) < length:
                dialogue.append((speakers[0], f"Additional scene {len(dialogue) + 1}..."))
                
            return dialogue
            
        else:
            # Default: random dialogue
            speakers = ["Speaker1", "Speaker2"]
            utterances = [
                "Hello there!", "How are you?", "I'm fine, thanks.", 
                "What's new?", "Not much.", "That's interesting.",
                "I disagree.", "I agree.", "Tell me more.",
                "Let's change the subject.", "What do you think?", "I'm not sure."
            ]
            
            dialogue = []
            for i in range(length):
                speaker = speakers[i % 2]
                utterance = utterances[i % len(utterances)]
                dialogue.append((speaker, utterance))
                
            return dialogue
    
    def test_can_process(self):
        """Test the can_process method."""
        # Should accept lists of (speaker, utterance) pairs
        self.assertTrue(self.functor.can_process([("Alice", "Hello"), ("Bob", "Hi")]))
        
        # Should reject non-dialogue data
        self.assertFalse(self.functor.can_process("not a dialogue"))
        self.assertFalse(self.functor.can_process([1, 2, 3]))
        self.assertFalse(self.functor.can_process(np.zeros((10, 10))))
        self.assertFalse(self.functor.can_process([(1, 2), (3, 4)]))  # Path, not dialogue
    
    def test_compression_decompression(self):
        """Test compression and decompression of dialogues."""
        for i, dialogue in enumerate(self.dialogue_samples):
            # Add test name to output
            pattern_names = ["question_answer", "agreement_disagreement", "multi_speaker", "narrative"]
            pattern = pattern_names[i] if i < len(pattern_names) else f"pattern_{i}"
            
            # Compress
            compressed, metadata = self.functor.compress(dialogue)
            
            # Check that metadata has expected fields
            self.assertIn("original_length", metadata)
            self.assertIn("compressed_length", metadata)
            self.assertIn("compression_ratio", metadata)
            self.assertIn("speakers", metadata)
            
            # Verify compression ratio is meaningful (< 1.0)
            self.assertLess(metadata["compression_ratio"], 1.0, 
                          f"Compression ratio should be < 1.0 for {pattern}")
            
            # Decompress with neutral tone
            hints = {"tone": "neutral"}
            decompressed = self.functor.decompress(compressed, metadata, hints)
            
            # Verify decompressed length matches original
            self.assertEqual(len(dialogue), len(decompressed))
            
            # Save dialogue comparison
            self._save_dialogue_comparison(dialogue, decompressed, compressed, metadata, 
                                         f"test_game_{pattern}.txt", pattern)
            
            # Verify that key turns are preserved exactly
            key_indices = compressed["key_indices"]
            key_dialogue = compressed["key_dialogue"]
            
            preservation_count = 0
            for idx, (speaker, utterance) in zip(key_indices, key_dialogue):
                if idx < len(dialogue) and dialogue[idx] == (speaker, utterance):
                    preservation_count += 1
                    
            # All key turns should be preserved exactly
            self.assertEqual(preservation_count, len(key_indices), 
                           "All key turns should be preserved exactly")
            
            print(f"Dialogue {pattern}: {len(dialogue)} turns compressed to {len(key_indices)} key turns, Ratio: {metadata['compression_ratio']:.3f}")
    
    def test_tone_variations(self):
        """Test that tone hints affect the decompressed output."""
        # Use the narrative dialogue as it has emotional content
        dialogue = self._create_test_dialogue("narrative", 20)
        
        # Compress
        compressed, metadata = self.functor.compress(dialogue)
        
        # Decompress with different tones
        tones = ["neutral", "positive", "negative"]
        decompressions = {}
        
        for tone in tones:
            hints = {"tone": tone}
            decompressions[tone] = self.functor.decompress(compressed, metadata, hints)
            
            # Verify length matches original
            self.assertEqual(len(dialogue), len(decompressions[tone]))
            
            # Save comparison
            self._save_dialogue_comparison(dialogue, decompressions[tone], compressed, metadata,
                                         f"test_game_tone_{tone}.txt", f"narrative_{tone}")
        
        # Verify tones produce different outputs in non-key turns
        key_indices = set(compressed["key_indices"])
        
        # Compare neutral vs positive
        differences_pos = 0
        differences_neg = 0
        
        for i in range(len(dialogue)):
            if i not in key_indices:
                # Non-key turns should be generated differently based on tone
                if decompressions["neutral"][i] != decompressions["positive"][i]:
                    differences_pos += 1
                if decompressions["neutral"][i] != decompressions["negative"][i]:
                    differences_neg += 1
        
        # We expect at least some turns to be different
        self.assertGreater(differences_pos, 0, "Positive tone should produce some different output")
        self.assertGreater(differences_neg, 0, "Negative tone should produce some different output")
        
        print(f"Tone variations: {differences_pos} differences with positive tone, {differences_neg} with negative tone")
    
    def test_dialogue_structure(self):
        """Test that the dialogue structure is properly identified and used."""
        # Test with multi-speaker dialogue
        dialogue = self._create_test_dialogue("multi_speaker", 15)
        
        # Compress
        compressed, metadata = self.functor.compress(dialogue)
        
        # Examine structure
        structure = compressed["structure"]
        
        # Verify structure contains expected fields
        self.assertIn("speakers", structure)
        self.assertIn("num_speakers", structure)
        self.assertIn("total_turns", structure)
        self.assertIn("turn_distribution", structure)
        self.assertIn("patterns", structure)
        
        # Verify speakers are correctly identified
        expected_speakers = set(speaker for speaker, _ in dialogue)
        self.assertEqual(set(structure["speakers"]), expected_speakers)
        
        # Verify turn distribution matches actual distribution
        expected_distribution = {}
        for speaker, _ in dialogue:
            expected_distribution[speaker] = expected_distribution.get(speaker, 0) + 1
            
        for speaker, count in expected_distribution.items():
            self.assertEqual(structure["turn_distribution"][speaker], count)
        
        # Verify total turns matches original dialogue length
        self.assertEqual(structure["total_turns"], len(dialogue))
    
    def _save_dialogue_comparison(self, original, decompressed, compressed, metadata, filename, pattern):
        """Save a text file comparing original and decompressed dialogues."""
        key_indices = compressed["key_indices"]
        
        with open(filename, "w") as f:
            f.write(f"=== TEST DIALOGUE: {pattern.upper()} ===\n\n")
            
            f.write("=== ORIGINAL DIALOGUE ===\n\n")
            for j, (speaker, utterance) in enumerate(original):
                key_marker = " *" if j in key_indices else ""
                f.write(f"[{j}]{key_marker} {speaker}: {utterance}\n")
            
            f.write("\n\n=== RECONSTRUCTED DIALOGUE ===\n\n")
            for j, (speaker, utterance) in enumerate(decompressed):
                key_marker = " *" if j in key_indices else ""
                f.write(f"[{j}]{key_marker} {speaker}: {utterance}\n")
            
            f.write(f"\n=== METADATA ===\n")
            f.write(f"Compression ratio: {metadata['compression_ratio']:.3f}\n")
            f.write(f"Key turns: {len(key_indices)} out of {len(original)}\n")
            f.write(f"Speakers: {', '.join(metadata['speakers'])}\n")
            f.write(f"Patterns detected: {metadata['num_patterns']}\n")

if __name__ == "__main__":
    unittest.main()
