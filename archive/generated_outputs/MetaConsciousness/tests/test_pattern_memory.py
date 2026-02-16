"""
Tests for PatternMemory functionality.
"""
import unittest
import os
import tempfile
import json
import shutil
from unittest.mock import patch

# Fix the path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MetaConsciousness.memory.pattern_memory import PatternMemory
from MetaConsciousness.utils import log_event, get_events, clear_events

class TestPatternMemory(unittest.TestCase):
    """Test cases for PatternMemory."""

    def setUp(self) -> None:
    """Set up test environment."""
        # Clear event log before creating the object
        clear_events()

        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_pattern_profile.json")

        # Create the pattern memory with test config
        self.pattern_memory = PatternMemory({
            "memory_dir": self.test_dir,
            "pattern_profile_filename": "test_pattern_profile.json"
        })

    def tearDown(self) -> None:
    """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self) -> None:
    """Test initialization of PatternMemory."""
        self.assertEqual(self.pattern_memory.file_path, self.test_file)
        self.assertEqual(self.pattern_memory.patterns, {})
        self.assertFalse(self.pattern_memory.modified)

        # Check log events
        events = get_events()
        init_events = [e for e in events if "PatternMemory initialized" in e["message"]]
        # Debugging output to see what's in the events
        if not init_events:
            print("No initialization events found. All events:", events)
        self.assertTrue(len(init_events) > 0, "No initialization event was logged")

    def test_update_pattern(self) -> None:
    """Test updating pattern data."""
        # Update a pattern
        self.pattern_memory.update_pattern("checkerboard", {
            "vigilance": 0.7,
            "success": 0.8,
            "correlation": 0.9,
            "confidence": 0.85
        })

        # Check that the pattern was created and updated
        self.assertIn("checkerboard", self.pattern_memory.patterns)
        self.assertEqual(self.pattern_memory.patterns["checkerboard"]["encounters"], 1)
        self.assertEqual(self.pattern_memory.patterns["checkerboard"]["avg_vigilance"], 0.7)

        # Update the same pattern again
        self.pattern_memory.update_pattern("checkerboard", {
            "vigilance": 0.8,
            "success": 0.9,
            "correlation": 0.95,
            "confidence": 0.9
        })

        # Check that the pattern was updated with exponential moving average
        self.assertEqual(self.pattern_memory.patterns["checkerboard"]["encounters"], 2)
        # avg_vigilance should be between 0.7 and 0.8, closer to 0.8 due to EMA
        avg_vigilance = self.pattern_memory.patterns["checkerboard"]["avg_vigilance"]
        self.assertTrue(0.7 < avg_vigilance < 0.8)

        # Check outcomes
        self.assertEqual(len(self.pattern_memory.patterns["checkerboard"]["outcomes"]), 2)

        # Check modified flag
        self.assertTrue(self.pattern_memory.modified)

    def test_get_pattern_stats(self) -> None:
    """Test getting pattern statistics."""
        # Add a pattern
        self.pattern_memory.update_pattern("gradient", {
            "vigilance": 0.6,
            "success": 0.7,
            "correlation": 0.8,
            "confidence": 0.75
        })

        # Get stats
        stats = self.pattern_memory.get_pattern_stats("gradient")

        # Check stats
        self.assertEqual(stats["encounters"], 1)
        self.assertEqual(stats["avg_vigilance"], 0.6)
        self.assertEqual(stats["success_rate"], 0.7)

        # Check non-existent pattern
        stats = self.pattern_memory.get_pattern_stats("non_existent")
        self.assertEqual(stats["encounters"], 0)
        self.assertEqual(stats["avg_vigilance"], 0.5)

    def test_get_recommended_vigilance(self) -> None:
    """Test getting recommended vigilance."""
        # Add a pattern with multiple outcomes
        self.pattern_memory.update_pattern("checkerboard", {
            "vigilance": 0.7,
            "success": 0.8,
            "correlation": 0.9,
            "confidence": 0.85
        })

        self.pattern_memory.update_pattern("checkerboard", {
            "vigilance": 0.8,
            "success": 0.9,
            "correlation": 0.95,
            "confidence": 0.9
        })

        # Get recommended vigilance
        vigilance = self.pattern_memory.get_recommended_vigilance("checkerboard")

        # Should be biased toward the more successful outcome (0.8)
        self.assertTrue(0.7 <= vigilance <= 0.8)

        # Test default for non-existent pattern
        vigilance = self.pattern_memory.get_recommended_vigilance("non_existent", default=0.4)
        self.assertEqual(vigilance, 0.4)

    def test_save_and_load(self) -> None:
    """Test saving and loading pattern profile."""
        # Add some patterns
        self.pattern_memory.update_pattern("checkerboard", {
            "vigilance": 0.7,
            "success": 0.8
        })

        self.pattern_memory.update_pattern("gradient", {
            "vigilance": 0.6,
            "success": 0.7
        })

        # Save the profile
        self.pattern_memory.save()

        # Check that the file exists
        self.assertTrue(os.path.exists(self.test_file))

        # Create a new pattern memory and load the file
        new_memory = PatternMemory({
            "memory_dir": self.test_dir,
            "pattern_profile_filename": "test_pattern_profile.json"
        })

        # Check that patterns were loaded
        self.assertIn("checkerboard", new_memory.patterns)
        self.assertIn("gradient", new_memory.patterns)
        self.assertEqual(new_memory.patterns["checkerboard"]["encounters"], 1)
        self.assertEqual(new_memory.patterns["gradient"]["encounters"], 1)

if __name__ == "__main__":
    unittest.main()
