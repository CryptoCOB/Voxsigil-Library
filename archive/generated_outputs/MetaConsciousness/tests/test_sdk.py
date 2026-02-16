"""
Test suite for the Metaconsciousness SDK.

This suite tests all three frameworks and the MetaconsciousAgent across
different data types to verify compression, decompression, and agent
selection logic.
"""

import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MetaConsciousness.core.functors.partial_functor import PartialFunctor
from MetaConsciousness.frameworks.sheaf_compression.image_patch_functor import ImagePatchSheafFunctor
from MetaConsciousness.frameworks.homotopy_compression.path_functor import PathHomotopyFunctor
from MetaConsciousness.frameworks.game_compression.dialogue_functor import DialogueStrategyFunctor
from MetaConsciousness.agent.metaconscious_agent import MetaconsciousAgent

class TestSheafCompression(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.functor = ImagePatchSheafFunctor(
            patch_size=8,
            overlap=2,
            entropy_threshold=0.2,
            use_edges=True
        )
        
        # Create test image samples
        self.image_samples = [
            self._create_test_image(64, 64, pattern="gradient"),
            self._create_test_image(64, 64, pattern="checkerboard"),
            self._create_test_image(64, 64, pattern="noise")
        ]
    
    def _create_test_image(self, height, width, pattern="gradient"):
        """Create a test image with the specified pattern."""
        if pattern == "gradient":
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            X, Y = np.meshgrid(x, y)
            return (X + Y) * 128
        elif pattern == "checkerboard":
            x = np.arange(width)
            y = np.arange(height)
            X, Y = np.meshgrid(x, y)
            return ((X + Y) % 2) * 255
        elif pattern == "noise":
            return np.random.randint(0, 256, (height, width))
    
    def test_can_process(self):
        """Test can_process method."""
        # Should accept numpy arrays
        self.assertTrue(self.functor.can_process(np.zeros((10, 10))))
        self.assertTrue(self.functor.can_process(np.zeros((10, 10, 3))))
        
        # Should reject non-image data
        self.assertFalse(self.functor.can_process("not an image"))
        self.assertFalse(self.functor.can_process([1, 2, 3]))
    
    def test_compression_decompression(self):
        """Test compression and decompression of images."""
        for i, image in enumerate(self.image_samples):
            # Compress
            compressed, metadata = self.functor.compress(image)
            
            # Verify metadata
            self.assertIn("original_shape", metadata)
            self.assertIn("num_patches", metadata)
            self.assertIn("compression_ratio", metadata)
            
            # Decompress
            decompressed = self.functor.decompress(compressed, metadata)
            
            # Verify decompressed shape matches original
            self.assertEqual(image.shape, decompressed.shape)
            
            # Verify some level of similarity (won't be perfect due to lossy compression)
            similarity = np.corrcoef(image.flatten(), decompressed.flatten())[0, 1]
            self.assertGreater(similarity, 0.5, f"Image {i} similarity too low: {similarity}")
            
            # Generate visualization for debugging
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title("Original")
            plt.subplot(1, 2, 2)
            plt.imshow(decompressed, cmap='gray')
            plt.title(f"Reconstructed (Ratio: {metadata['compression_ratio']:.2f})")
            plt.tight_layout()
            plt.savefig(f"test_image_{i}_comparison.png")
            plt.close()

class TestHomotopyCompression(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.functor = PathHomotopyFunctor(
            epsilon=2.0,
            min_angle=15.0,
            interpolation="cubic",
            simplification="rdp",
            closed_path=False
        )
        
        # Create test path samples
        self.path_samples = [
            self._create_test_path(100, pattern="circle"),
            self._create_test_path(100, pattern="zigzag"),
            self._create_test_path(100, pattern="spiral")
        ]
    
    def _create_test_path(self, num_points, pattern="circle"):
        """Create a test path with the specified pattern."""
        if pattern == "circle":
            t = np.linspace(0, 2*np.pi, num_points)
            x = 50 + 40 * np.cos(t)
            y = 50 + 40 * np.sin(t)
            return list(zip(x, y))
        elif pattern == "zigzag":
            x = np.linspace(0, 100, num_points)
            y = 50 + 40 * np.sin(np.linspace(0, 10*np.pi, num_points))
            return list(zip(x, y))
        elif pattern == "spiral":
            t = np.linspace(0, 10*np.pi, num_points)
            r = np.linspace(0, 40, num_points)
            x = 50 + r * np.cos(t)
            y = 50 + r * np.sin(t)
            return list(zip(x, y))
    
    def test_can_process(self):
        """Test can_process method."""
        # Should accept lists of 2D points
        self.assertTrue(self.functor.can_process([(0, 0), (1, 1)]))
        self.assertTrue(self.functor.can_process(np.array([[0, 0], [1, 1]])))
        
        # Should reject non-path data
        self.assertFalse(self.functor.can_process("not a path"))
        self.assertFalse(self.functor.can_process([1, 2, 3]))
    
    def test_compression_decompression(self):
        """Test compression and decompression of paths."""
        for i, path in enumerate(self.path_samples):
            # Compress
            compressed, metadata = self.functor.compress(path)
            
            # Verify metadata
            self.assertIn("original_length", metadata)
            self.assertIn("compressed_length", metadata)
            self.assertIn("compression_ratio", metadata)
            
            # Decompress with same point count for comparison
            hints = {"density": len(path)}
            decompressed = self.functor.decompress(compressed, metadata, hints)
            
            # Verify decompressed length matches original
            self.assertEqual(len(path), len(decompressed))
            
            # Generate visualization for debugging
            plt.figure(figsize=(10, 5))
            path_array = np.array(path)
            decompressed_array = np.array(decompressed)
            critical_points = np.array(compressed["critical_points"])
            
            plt.subplot(1, 2, 1)
            plt.plot(path_array[:, 0], path_array[:, 1], 'b-', label='Original')
            plt.scatter(path_array[:, 0], path_array[:, 1], c='blue', s=10, alpha=0.3)
            plt.title(f"Original ({len(path)} points)")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(decompressed_array[:, 0], decompressed_array[:, 1], 'r-', label='Reconstructed')
            plt.scatter(critical_points[:, 0], critical_points[:, 1], c='green', s=50, label='Critical Points')
            plt.title(f"Reconstructed ({len(critical_points)} critical points)")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"test_path_{i}_comparison.png")
            plt.close()
            
            # Verify compression ratio
            print(f"Path {i} compression ratio: {metadata['compression_ratio']:.2f}")
            self.assertLess(metadata['compression_ratio'], 1.0, "Compression ratio should be < 1.0")

class TestGameCompression(unittest.TestCase):
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
            self._create_test_dialogue("multi_speaker", 15)
        ]
    
    def _create_test_dialogue(self, pattern="question_answer", length=10):
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
    
    def test_can_process(self):
        """Test can_process method."""
        # Should accept lists of (speaker, utterance) pairs
        self.assertTrue(self.functor.can_process([("Alice", "Hello"), ("Bob", "Hi")]))
        
        # Should reject non-dialogue data
        self.assertFalse(self.functor.can_process("not a dialogue"))
        self.assertFalse(self.functor.can_process([1, 2, 3]))
    
    def test_compression_decompression(self):
        """Test compression and decompression of dialogues."""
        for i, dialogue in enumerate(self.dialogue_samples):
            # Compress
            compressed, metadata = self.functor.compress(dialogue)
            
            # Verify metadata
            self.assertIn("original_length", metadata)
            self.assertIn("compressed_length", metadata)
            self.assertIn("compression_ratio", metadata)
            self.assertIn("speakers", metadata)
            
            # Decompress
            decompressed = self.functor.decompress(compressed, metadata, hints={"tone": "neutral"})
            
            # Verify decompressed length matches original
            self.assertEqual(len(dialogue), len(decompressed))
            
            # Count key turns preserved exactly
            key_indices = compressed["key_indices"]
            key_dialogue = compressed["key_dialogue"]
            
            preservation_count = 0
            for idx, (speaker, utterance) in zip(key_indices, key_dialogue):
                if idx < len(dialogue) and dialogue[idx] == (speaker, utterance):
                    preservation_count += 1
            
            # All key turns should be preserved exactly
            self.assertEqual(preservation_count, len(key_indices), 
                            "All key turns should be preserved exactly")
            
            # Verify compression ratio
            print(f"Dialogue {i} compression ratio: {metadata['compression_ratio']:.2f}")
            
            # Save dialogue comparison
            with open(f"test_dialogue_{i}_comparison.txt", "w") as f:
                f.write("=== ORIGINAL DIALOGUE ===\n\n")
                for j, (speaker, utterance) in enumerate(dialogue):
                    key_marker = " *" if j in key_indices else ""
                    f.write(f"[{j}]{key_marker} {speaker}: {utterance}\n")
                
                f.write("\n\n=== RECONSTRUCTED DIALOGUE ===\n\n")
                for j, (speaker, utterance) in enumerate(decompressed):
                    key_marker = " *" if j in key_indices else ""
                    f.write(f"[{j}]{key_marker} {speaker}: {utterance}\n")
                
                f.write(f"\nCompression ratio: {metadata['compression_ratio']:.2f}\n")
                f.write(f"Key turns: {len(key_indices)} out of {len(dialogue)}\n")

class TestMetaconsciousAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MetaconsciousAgent(config={
            "verbose": True,
            "version": "0.1.0",
            "learning_rate": 0.2,
            "visualization_dir": "test_visualizations"
        })
        
        # Create test samples of different types
        self.test_image = np.random.randint(0, 256, (64, 64))
        
        t = np.linspace(0, 2*np.pi, 100)
        x = 50 + 40 * np.cos(t)
        y = 50 + 40 * np.sin(t)
        self.test_path = list(zip(x, y))
        
        self.test_dialogue = [
            ("Alice", "Hello, what do you think about the weather?"),
            ("Bob", "It's quite nice today."),
            ("Alice", "I agree, it's very pleasant."),
            ("Bob", "Do you think it will rain tomorrow?"),
            ("Alice", "I don't think so, the forecast said it would be sunny."),
            ("Bob", "That's great! We could go for a picnic."),
            ("Alice", "That's a good idea. Let's plan on that."),
            ("Bob", "What time would work for you?"),
            ("Alice", "How about noon?"),
            ("Bob", "Noon works for me.")
        ]
    
    def test_agent_selection(self):
        """Test that the agent selects the appropriate functor for each data type."""
        # Image compression
        compressed_image, metadata = self.agent.compress(self.test_image)
        self.assertIn(metadata["framework"], ["sheaf_compression"])
        
        # Path compression
        compressed_path, metadata = self.agent.compress(self.test_path)
        self.assertIn(metadata["framework"], ["homotopy_compression"])
        
        # Dialogue compression
        compressed_dialogue, metadata = self.agent.compress(self.test_dialogue)
        self.assertIn(metadata["framework"], ["game_compression"])
    
    def test_end_to_end_pipeline(self):
        """Test the full compression and decompression pipeline with all data types."""
        # Image pipeline
        compressed_image, metadata = self.agent.compress(self.test_image)
        decompressed_image = self.agent.decompress(compressed_image, metadata)
        self.assertEqual(self.test_image.shape, decompressed_image.shape)
        
        # Path pipeline
        compressed_path, metadata = self.agent.compress(self.test_path)
        decompressed_path = self.agent.decompress(compressed_path, metadata, hints={"density": len(self.test_path)})
        self.assertEqual(len(self.test_path), len(decompressed_path))
        
        # Dialogue pipeline
        compressed_dialogue, metadata = self.agent.compress(self.test_dialogue)
        decompressed_dialogue = self.agent.decompress(compressed_dialogue, metadata)
        self.assertEqual(len(self.test_dialogue), len(decompressed_dialogue))
    
    def test_context_adaptation(self):
        """Test that the agent adapts based on context."""
        # With speed priority
        compressed_image_speed, metadata_speed = self.agent.compress(
            self.test_image, context={"prioritize_speed": True}
        )
        
        # With accuracy priority
        compressed_image_accuracy, metadata_accuracy = self.agent.compress(
            self.test_image, context={"prioritize_accuracy": True}
        )
        
        # Check if explanation mentions the priority
        speed_explanation = self.agent.explain_choice()
        self.assertIn("speed", speed_explanation["explanation"].lower())
        
        # Reset context
        self.agent.context = {}
        
        # With reconstruction quality priority
        compressed_image_quality, metadata_quality = self.agent.compress(
            self.test_image, context={"reconstruction_quality": True}
        )
        
        quality_explanation = self.agent.explain_choice()
        self.assertIn("accuracy", quality_explanation["explanation"].lower())
    
    def test_visualization(self):
        """Test visualization capabilities."""
        # Image visualization
        compressed_image, metadata = self.agent.compress(self.test_image)
        decompressed_image = self.agent.decompress(compressed_image, metadata)
        viz_path = self.agent.visualize(self.test_image, compressed_image, decompressed_image, metadata)
        self.assertTrue(os.path.exists(viz_path))
        
        # Path visualization
        compressed_path, metadata = self.agent.compress(self.test_path)
        decompressed_path = self.agent.decompress(compressed_path, metadata, hints={"density": len(self.test_path)})
        viz_path = self.agent.visualize(self.test_path, compressed_path, decompressed_path, metadata)
        self.assertTrue(os.path.exists(viz_path))
        
        # Dialogue visualization
        compressed_dialogue, metadata = self.agent.compress(self.test_dialogue)
        decompressed_dialogue = self.agent.decompress(compressed_dialogue, metadata)
        viz_path = self.agent.visualize(self.test_dialogue, compressed_dialogue, decompressed_dialogue, metadata)
        self.assertTrue(os.path.exists(viz_path))
    
    def test_training(self):
        """Test that the agent can learn from training."""
        # Create training data
        training_images = [np.random.randint(0, 256, (32, 32)) for _ in range(3)]
        
        # Train on images
        results = self.agent.train(training_images, iterations=2)
        
        # Verify training happened
        self.assertGreater(results["samples_processed"], 0)
        
        # Check that framework statistics improved
        self.assertGreater(
            results["learning"]["after"]["framework"].get("sheaf_compression", 0),
            results["learning"]["before"]["framework"].get("sheaf_compression", 0)
        )

if __name__ == "__main__":
    unittest.main()
