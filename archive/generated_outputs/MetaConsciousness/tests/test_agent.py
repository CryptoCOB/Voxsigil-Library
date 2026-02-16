"""
Tests for the MetaconsciousAgent.
"""

import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
import time
from typing import List, Tuple, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import concrete functors first to ensure registration
from MetaConsciousness.frameworks.sheaf_compression.image_patch_functor import ImagePatchSheafFunctor
from MetaConsciousness.frameworks.homotopy_compression.path_functor import PathHomotopyFunctor
from MetaConsciousness.frameworks.game_compression.dialogue_functor import DialogueStrategyFunctor

# Now import the agent
from MetaConsciousness.agent.metaconscious_agent import MetaconsciousAgent
from MetaConsciousness.core.registry import list_functors, list_frameworks

class TestMetaconsciousAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a test directory for visualizations
        self.test_viz_dir = "test_visualizations"
        os.makedirs(self.test_viz_dir, exist_ok=True)
        
        # Initialize the agent with test configuration
        self.agent = MetaconsciousAgent(config={
            "verbose": True,
            "version": "0.1.0",
            "learning_rate": 0.2,
            "visualization_dir": self.test_viz_dir,
            "cache_size": 10
        })
        
        # Create test samples of different data types
        self.test_image = self._create_test_image(64, 64, "gradient")
        self.test_path = self._create_test_path(100, "circle")
        self.test_dialogue = self._create_test_dialogue(10, "mixed")
    
    def _create_test_image(self, width: int, height: int, pattern: str = "gradient") -> np.ndarray:
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
            return ((X + Y) % 16 < 8) * 255
        else:
            return np.random.randint(0, 255, (height, width))
    
    def _create_test_path(self, num_points: int, pattern: str = "circle") -> List[Tuple[float, float]]:
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
        else:
            x = np.random.uniform(0, 100, num_points)
            y = np.random.uniform(0, 100, num_points)
            return list(zip(x, y))
    
    def _create_test_dialogue(self, length: int = 10, pattern: str = "mixed") -> List[Tuple[str, str]]:
        """Create a test dialogue with the specified pattern."""
        if pattern == "mixed":
            dialogue = [
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
            # Trim or extend to match requested length
            if len(dialogue) > length:
                dialogue = dialogue[:length]
            while len(dialogue) < length:
                dialogue.append(("Alice" if len(dialogue) % 2 == 0 else "Bob", f"Additional comment {len(dialogue) + 1}."))
            return dialogue
        else:
            # Simple alternating dialogue
            dialogue = []
            for i in range(length):
                speaker = "Alice" if i % 2 == 0 else "Bob"
                dialogue.append((speaker, f"Statement {i+1}"))
            return dialogue
    
    def test_agent_initialization(self):
        """Test agent initialization and configuration."""
        # Check that the agent initialized correctly
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.config["version"], "0.1.0")
        self.assertEqual(self.agent.learning_rate, 0.2)
        self.assertEqual(self.agent.cache_size, 10)
        
        # Check that frameworks and functors are loaded
        self.assertGreater(len(self.agent.frameworks), 0)
        self.assertGreater(len(self.agent.available_functors), 0)
        
        # Print frameworks and functors for debugging
        print(f"Loaded frameworks: {self.agent.frameworks}")
        print(f"Loaded functors: {self.agent.available_functors}")
    
    def test_functor_selection(self):
        """Test that the agent selects the appropriate functor for each data type."""
        # Image compression
        compressed_image, metadata = self.agent.compress(self.test_image)
        self.assertIn(metadata["framework"], ["sheaf_compression"])
        print(f"Selected for image: {metadata['functor']} from {metadata['framework']}")
        
        # Path compression
        compressed_path, metadata = self.agent.compress(self.test_path)
        self.assertIn(metadata["framework"], ["homotopy_compression"])
        print(f"Selected for path: {metadata['functor']} from {metadata['framework']}")
        
        # Dialogue compression
        compressed_dialogue, metadata = self.agent.compress(self.test_dialogue)
        self.assertIn(metadata["framework"], ["game_compression"])
        print(f"Selected for dialogue: {metadata['functor']} from {metadata['framework']}")
    
    def test_end_to_end_pipeline(self):
        """Test the full compression and decompression pipeline with all data types."""
        # Image pipeline
        compressed_image, metadata = self.agent.compress(self.test_image)
        decompressed_image = self.agent.decompress(compressed_image, metadata)
        self.assertEqual(self.test_image.shape, decompressed_image.shape)
        confidence = self.agent.calculate_reconstruction_confidence(self.test_image, decompressed_image, metadata)
        print(f"Image reconstruction confidence: {confidence:.3f}")
        
        # Path pipeline
        compressed_path, metadata = self.agent.compress(self.test_path)
        decompressed_path = self.agent.decompress(compressed_path, metadata, hints={"density": len(self.test_path)})
        self.assertEqual(len(self.test_path), len(decompressed_path))
        confidence = self.agent.calculate_reconstruction_confidence(self.test_path, decompressed_path, metadata)
        print(f"Path reconstruction confidence: {confidence:.3f}")
        
        # Dialogue pipeline
        compressed_dialogue, metadata = self.agent.compress(self.test_dialogue)
        decompressed_dialogue = self.agent.decompress(compressed_dialogue, metadata)
        self.assertEqual(len(self.test_dialogue), len(decompressed_dialogue))
        confidence = self.agent.calculate_reconstruction_confidence(self.test_dialogue, decompressed_dialogue, metadata)
        print(f"Dialogue reconstruction confidence: {confidence:.3f}")
    
    def test_caching(self):
        """Test that the caching system works."""
        # Compress the same image twice
        start = time.time()
        compressed1, metadata1 = self.agent.compress(self.test_image)
        first_time = time.time() - start
        
        start = time.time()
        compressed2, metadata2 = self.agent.compress(self.test_image)  # Should be cached
        second_time = time.time() - start
        
        # Second compression should be faster due to caching
        print(f"First compression: {first_time:.6f}s, Second compression: {second_time:.6f}s")
        self.assertLess(second_time, first_time * 0.5, "Cached compression should be significantly faster")
        
        # Results should be identical
        self.assertEqual(metadata1["functor"], metadata2["functor"])
        self.assertEqual(metadata1["compression_ratio"], metadata2["compression_ratio"])
    
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
        
        print("Speed context explanation: " + speed_explanation["explanation"])
        print("Quality context explanation: " + quality_explanation["explanation"])
    
    def test_explain_choice(self):
        """Test the explain_choice method."""
        # Compress something
        compressed, metadata = self.agent.compress(self.test_image)
        
        # Get explanation
        explanation_simple = self.agent.explain_choice(detailed=False)
        explanation_detailed = self.agent.explain_choice(detailed=True)
        
        # Check that explanations have expected fields
        self.assertIn("explanation", explanation_simple)
        self.assertIn("selected", explanation_simple)
        
        self.assertIn("explanation", explanation_detailed)
        self.assertIn("selected", explanation_detailed)
        self.assertIn("all_candidates", explanation_detailed)
        
        # Check that detailed explanation includes all candidates
        self.assertGreater(len(explanation_detailed["all_candidates"]), 0)
        
        # Print explanation
        print("Simple explanation: " + explanation_simple["explanation"])
        print("Detailed explanation first candidate score components:")
        for k, v in explanation_detailed["all_candidates"][0]["components"].items():
            print(f"  {k}: {v}")
    
    def test_performance_stats(self):
        """Test that performance statistics are tracked."""
        # Compress three different data types
        self.agent.compress(self.test_image)
        self.agent.compress(self.test_path)
        self.agent.compress(self.test_dialogue)
        
        # Get performance stats
        stats = self.agent.get_performance_stats()
        
        # Check that stats have expected sections
        self.assertIn("strategy_success", stats)
        self.assertIn("domain_success", stats)
        self.assertIn("functor_stats", stats)
        self.assertIn("framework_stats", stats)
        
        # Verify that all frameworks have stats
        for framework in self.agent.frameworks:
            self.assertIn(framework, stats["framework_stats"])
        
        # Print some stats for debugging
        print("\nFramework success rates:")
        for framework, success in stats["strategy_success"].items():
            print(f"  {framework}: {success:.3f}")
            
        print("\nDomain success rates:")
        for domain, success in list(stats["domain_success"].items())[:5]:  # Show first 5
            print(f"  {domain}: {success:.3f}")
    
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
        
        # Also check for text version of dialogue visualization
        text_viz_path = viz_path.replace('.png', '.txt')
        self.assertTrue(os.path.exists(text_viz_path))
    
    def test_training(self):
        """Test that the agent can learn from training."""
        # Create training data
        training_images = [self._create_test_image(32, 32, pattern) 
                          for pattern in ["gradient", "checkerboard", "noise"]]
        
        # Train on images
        results = self.agent.train(training_images, iterations=2)
        
        # Verify training happened
        self.assertGreater(results["samples_processed"], 0)
        
        # Check learning deltas
        print("\nLearning deltas for frameworks:")
        for framework, delta in results["learning"]["delta"]["framework"].items():
            print(f"  {framework}: {delta:.3f}")
        
        # Verify that at least one framework improved
        self.assertTrue(any(delta > 0 for delta in results["learning"]["delta"]["framework"].values()),
                       "At least one framework should improve during training")
        
        # Check specific framework improvement
        sheaf_delta = results["learning"]["delta"]["framework"].get("sheaf_compression", 0)
        print(f"Sheaf compression framework delta: {sheaf_delta:.5f}")
    
    def test_reconstruction_confidence(self):
        """Test the reconstruction confidence calculation."""
        # Test with all three data types
        data_types = [
            ("image", self.test_image),
            ("path", self.test_path),
            ("dialogue", self.test_dialogue)
        ]
        
        for name, data in data_types:
            # Compress and decompress
            compressed, metadata = self.agent.compress(data)
            decompressed = self.agent.decompress(compressed, metadata)
            
            # Calculate confidence
            confidence = self.agent.calculate_reconstruction_confidence(data, decompressed, metadata)
            
            # Confidence should be between 0 and 1
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            print(f"{name.capitalize()} reconstruction confidence: {confidence:.3f}")
            
            # Corrupt the decompressed data slightly and verify confidence decreases
            if name == "image":
                # Add noise to image
                noisy = decompressed.copy()
                noisy += np.random.normal(0, 20, noisy.shape).astype(noisy.dtype)
                noisy_confidence = self.agent.calculate_reconstruction_confidence(data, noisy, metadata)
                self.assertLess(noisy_confidence, confidence, "Confidence should decrease with noise")
                
            elif name == "path":
                # Perturb path points
                perturbed = [
                    (x + np.random.normal(0, 5), y + np.random.normal(0, 5))
                    for x, y in decompressed
                ]
                perturbed_confidence = self.agent.calculate_reconstruction_confidence(data, perturbed, metadata)
                self.assertLess(perturbed_confidence, confidence, "Confidence should decrease with perturbation")
                
            elif name == "dialogue":
                # Change some utterances
                modified = decompressed.copy()
                for i in range(len(modified)):
                    if i % 3 == 0:  # Modify every third utterance
                        speaker, utterance = modified[i]
                        modified[i] = (speaker, utterance + " [MODIFIED]")
                
                modified_confidence = self.agent.calculate_reconstruction_confidence(data, modified, metadata)
                self.assertLess(modified_confidence, confidence, "Confidence should decrease with modifications")
            
            print(f"{name.capitalize()} corrupted confidence: {noisy_confidence if name == 'image' else perturbed_confidence if name == 'path' else modified_confidence:.3f}")

if __name__ == "__main__":
    unittest.main()
