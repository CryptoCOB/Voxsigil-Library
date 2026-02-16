"""
Tests for StrategyEvolution functionality.
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

from MetaConsciousness.omega3.strategy_evolution import StrategyEvolution
from MetaConsciousness.utils import log_event, get_events, clear_events

class TestStrategyEvolution(unittest.TestCase):
    """Test cases for StrategyEvolution."""

    def setUp(self) -> None:
    """Set up test environment."""
        # Clear event log before creating the object
        clear_events()

        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_strategy_cache.json")

        # Create the strategy evolution with test config
        self.strategy_evolution = StrategyEvolution({
            "cache_dir": self.test_dir,
            "strategy_cache_filename": "test_strategy_cache.json",
            "context_factors": ["risk_level", "pattern_type"]
        })

    def tearDown(self) -> None:
    """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self) -> None:
    """Test initialization of StrategyEvolution."""
        self.assertEqual(self.strategy_evolution.file_path, self.test_file)
        self.assertEqual(self.strategy_evolution.strategies, {})
        self.assertFalse(self.strategy_evolution.modified)
        self.assertEqual(self.strategy_evolution.context_factors, ["risk_level", "pattern_type"])

        # Check log events - now the initialization should be captured
        events = get_events()
        init_events = [e for e in events if "StrategyEvolution initialized" in e["message"]]
        # Debugging output to see what's in the events
        if not init_events:
            print("No initialization events found. All events:", events)
        self.assertTrue(len(init_events) > 0, "No initialization event was logged")

    def test_context_key_generation(self) -> None:
    """Test generation of context keys."""
        # Test basic context
        context = {
            "risk_level": "high",
            "pattern_type": "checkerboard"
        }
        key = self.strategy_evolution._make_context_key(context)
        self.assertEqual(key, "pattern_type=checkerboard;risk_level=high")

        # Test with extra keys (should be ignored)
        context = {
            "risk_level": "high",
            "pattern_type": "checkerboard",
            "other_key": "value"
        }
        key = self.strategy_evolution._make_context_key(context)
        self.assertEqual(key, "pattern_type=checkerboard;risk_level=high")

        # Test with missing keys
        context = {
            "risk_level": "high"
        }
        key = self.strategy_evolution._make_context_key(context)
        self.assertEqual(key, "risk_level=high")

    def test_update_strategy(self) -> None:
    """Test updating strategy success rates."""
        # Update a strategy
        context = {
            "risk_level": "high",
            "pattern_type": "checkerboard"
        }

        self.strategy_evolution.update_strategy("increase_vigilance", context, 0.8)

        # Check that the strategy was created and updated
        context_key = self.strategy_evolution._make_context_key(context)
        self.assertIn(context_key, self.strategy_evolution.strategies)
        self.assertIn("increase_vigilance", self.strategy_evolution.strategies[context_key])

        # Check success rate (EMA with alpha=0.3)
        strategy_data = self.strategy_evolution.strategies[context_key]["increase_vigilance"]
        expected_rate = 0.5 * 0.7 + 0.8 * 0.3  # (1-alpha)*old + alpha*new
        self.assertAlmostEqual(strategy_data["success_rate"], expected_rate, places=5)

        # Update again
        self.strategy_evolution.update_strategy("increase_vigilance", context, 0.9)

        # Check updated success rate
        strategy_data = self.strategy_evolution.strategies[context_key]["increase_vigilance"]
        expected_rate = expected_rate * 0.7 + 0.9 * 0.3  # (1-alpha)*old + alpha*new
        self.assertAlmostEqual(strategy_data["success_rate"], expected_rate, places=5)

        # Check count
        self.assertEqual(strategy_data["count"], 2)

        # Check modified flag
        self.assertTrue(self.strategy_evolution.modified)

    def test_get_recommended_strategy(self) -> None:
    """Test getting recommended strategy."""
        # Add multiple strategies with different success rates
        context = {
            "risk_level": "high",
            "pattern_type": "checkerboard"
        }

        # Update strategies with different success rates
        self.strategy_evolution.update_strategy("increase_vigilance", context, 0.7)
        self.strategy_evolution.update_strategy("retry_with_alt_functor", context, 0.9)
        self.strategy_evolution.update_strategy("increase_and_retry", context, 0.6)

        # Get recommended strategy
        strategy, confidence = self.strategy_evolution.get_recommended_strategy(context)

        # Should recommend the strategy with highest success rate
        self.assertEqual(strategy, "retry_with_alt_functor")
        self.assertAlmostEqual(confidence, 0.5 * 0.7 + 0.9 * 0.3, places=5)  # EMA calculation

        # Test with available strategies filter
        strategy, confidence = self.strategy_evolution.get_recommended_strategy(
            context,
            available_strategies=["increase_vigilance", "increase_and_retry"]
        )

        # Should recommend the best available strategy
        self.assertEqual(strategy, "increase_vigilance")

        # Test with non-existent context
        strategy, confidence = self.strategy_evolution.get_recommended_strategy({
            "risk_level": "low",
            "pattern_type": "other"
        })

        # Should return default strategy with medium confidence
        self.assertEqual(strategy, "maintain_vigilance")
        self.assertEqual(confidence, 0.5)

    def test_get_strategy_scores(self) -> None:
    """Test getting strategy scores for a context."""
        # Add multiple strategies
        context = {
            "risk_level": "medium",
            "pattern_type": "gradient"
        }

        self.strategy_evolution.update_strategy("increase_vigilance", context, 0.7)
        self.strategy_evolution.update_strategy("decrease_vigilance", context, 0.8)

        # Get scores
        scores = self.strategy_evolution.get_strategy_scores(context)

        # Check scores
        self.assertIn("increase_vigilance", scores)
        self.assertIn("decrease_vigilance", scores)
        self.assertGreater(scores["decrease_vigilance"], scores["increase_vigilance"])

        # Check for non-existent context
        scores = self.strategy_evolution.get_strategy_scores({
            "risk_level": "low",
            "pattern_type": "other"
        })

        self.assertEqual(scores, {})

    def test_save_and_load(self) -> None:
    """Test saving and loading strategy cache."""
        # Add some strategies
        self.strategy_evolution.update_strategy("increase_vigilance", {
            "risk_level": "high",
            "pattern_type": "checkerboard"
        }, 0.7)

        self.strategy_evolution.update_strategy("decrease_vigilance", {
            "risk_level": "low",
            "pattern_type": "gradient"
        }, 0.8)

        # Save the cache
        self.strategy_evolution.save()

        # Check that the file exists
        self.assertTrue(os.path.exists(self.test_file))

        # Create a new strategy evolution and load the file
        new_evolution = StrategyEvolution({
            "cache_dir": self.test_dir,
            "strategy_cache_filename": "test_strategy_cache.json",
            "context_factors": ["risk_level", "pattern_type"]
        })

        # Check that strategies were loaded
        self.assertEqual(len(new_evolution.strategies), 2)

        # Check that specific strategies exist
        checkerboard_key = "pattern_type=checkerboard;risk_level=high"
        gradient_key = "pattern_type=gradient;risk_level=low"

        self.assertIn(checkerboard_key, new_evolution.strategies)
        self.assertIn(gradient_key, new_evolution.strategies)

        self.assertIn("increase_vigilance", new_evolution.strategies[checkerboard_key])
        self.assertIn("decrease_vigilance", new_evolution.strategies[gradient_key])

if __name__ == "__main__":
    unittest.main()
