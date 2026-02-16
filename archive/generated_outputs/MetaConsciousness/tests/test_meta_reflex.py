"""
Tests for MetaReflexLayer functionality.
"""
import unittest
import numpy as np
import os
import sys
import time
from unittest.mock import patch, MagicMock

# Fix the path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MetaConsciousness.core.meta_reflex import (
    MetaReflexLayer, FallbackStrategy, InputRiskLevel
)
from MetaConsciousness.art.art_controller import ARTController

# Import Omega3 directly from the package (which is an alias for Omega3Agent)
from MetaConsciousness.omega3 import Omega3
from MetaConsciousness.core.meta_decision_router import MetaDecisionRouter
from MetaConsciousness.utils import log_event, get_events, clear_events

class TestMetaReflexLayer(unittest.TestCase):
    """Test cases for MetaReflexLayer."""

    def setUp(self) -> None:
    """Set up test environment."""
        # Clear event log before creating the object
        clear_events()

        # Create ART controller
        self.art_controller = ARTController(
            vigilance=0.5,
            learning_rate=0.1,
            input_dim=64*64  # For 64x64 images
        )

        # Create Omega-3 agent - use Omega3 which is the alias for Omega3Agent
        self.omega3 = Omega3(config={
            "pattern_memory_enabled": False,  # Disable for tests
            "strategy_evolution_enabled": False  # Disable for tests
        })

        # Create decision router
        self.router = MetaDecisionRouter(
            art_controller=self.art_controller,
            omega3=self.omega3,
            config={"use_omega3": True}
        )

        # Create meta reflex layer
        self.meta_reflex = MetaReflexLayer(
            art_controller=self.art_controller,
            omega3_agent=self.omega3,
            decision_router=self.router,
            config={
                "reflex_enabled": True,
                "vigilance_boost": 0.2,
                "checkerboard_blur_kernel": 3,
                "fallback_strategy": "hybrid"
            }
        )

    def test_initialization(self) -> None:
    """Test initialization of MetaReflexLayer."""
        self.assertEqual(self.meta_reflex.art_controller, self.art_controller)
        self.assertEqual(self.meta_reflex.omega3_agent, self.omega3)
        self.assertEqual(self.meta_reflex.decision_router, self.router)

        # Check config parsing
        self.assertTrue(self.meta_reflex.reflex_enabled)
        self.assertEqual(self.meta_reflex.vigilance_boost, 0.2)
        self.assertEqual(self.meta_reflex.fallback_strategy, FallbackStrategy.HYBRID)

        # Check log events
        events = get_events()
        init_events = [e for e in events if "MetaReflexLayer initialized" in e["message"]]
        self.assertTrue(len(init_events) > 0, "No initialization event was logged")

        # Check initialization from config only
        config_meta_reflex = MetaReflexLayer(config={
            "reflex_enabled": False,
            "vigilance_boost": 0.3,
            "fallback_strategy": "retry"
        })

        self.assertFalse(config_meta_reflex.reflex_enabled)
        self.assertEqual(config_meta_reflex.vigilance_boost, 0.3)
        self.assertEqual(config_meta_reflex.fallback_strategy, FallbackStrategy.RETRY)

    def test_process_input_without_reflex(self) -> None:
    """Test processing input with reflex disabled."""
        # Create reflex layer with reflex disabled
        meta_reflex = MetaReflexLayer(
            art_controller=self.art_controller,
            omega3_agent=self.omega3,
            config={"reflex_enabled": False}
        )

        # Create test input
        test_input = np.random.rand(64, 64)

        # Process input
        processed, metadata = meta_reflex.process_input(test_input, "image")

        # Check that input was not modified
        self.assertTrue(np.array_equal(processed, test_input))

        # Check that reflex was not applied
        self.assertFalse(metadata["reflex_applied"])

        # Check that risk level was assessed
        self.assertIn("risk_level", metadata)
        self.assertIn("pattern_metrics", metadata)

    def test_process_checkerboard_pattern(self) -> None:
    """Test processing a checkerboard pattern."""
        # Create checkerboard pattern
        checkerboard = np.zeros((64, 64), dtype=np.uint8)
        tile_size = 8
        for i in range(0, 64, tile_size):
            for j in range(0, 64, tile_size):
                if (i // tile_size + j // tile_size) % 2 == 0:
                    checkerboard[i:i+tile_size, j:j+tile_size] = 255

        # Process input
        processed, metadata = self.meta_reflex.process_input(checkerboard, "image")

        # Check that risk level is high or critical
        self.assertIn(metadata["risk_level"], ["high", "critical"])

        # Check that checkerboard was detected
        self.assertTrue(metadata["pattern_metrics"]["checkerboard_detected"])
        self.assertGreater(metadata["pattern_metrics"]["checkerboard_score"], 0.7)

        # Check that reflex was applied
        self.assertTrue(metadata["reflex_applied"])
        self.assertTrue(metadata["vigilance_adjusted"])

        # Vigilance should be increased for checkerboard pattern
        self.assertGreater(metadata["new_vigilance"], metadata["original_vigilance"])

    def test_process_gradient_pattern(self) -> None:
    """Test processing a gradient pattern."""
        # Create gradient pattern
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        xx, yy = np.meshgrid(x, y)
        gradient = (xx * 255).astype(np.uint8)

        # Process input
        processed, metadata = self.meta_reflex.process_input(gradient, "image")

        # Check that gradient was detected
        self.assertTrue(metadata["pattern_metrics"]["gradient_detected"])
        self.assertGreater(metadata["pattern_metrics"]["gradient_score"], 0.6)

        # Check that reflex was applied
        self.assertTrue(metadata["reflex_applied"])

    def test_fallback_strategies(self) -> None:
    """Test different fallback strategies."""
        # Create checkerboard pattern (likely to trigger fallback)
        checkerboard = np.zeros((64, 64), dtype=np.uint8)
        tile_size = 8
        for i in range(0, 64, tile_size):
            for j in range(0, 64, tile_size):
                if (i // tile_size + j // tile_size) % 2 == 0:
                    checkerboard[i:i+tile_size, j:j+tile_size] = 255

        # Patch the decision router to always return retry=True
        original_update_vigilance = self.router.update_vigilance

        def patched_update_vigilance(*args, **kwargs):
            decision = original_update_vigilance(*args, **kwargs)
            decision["retry"] = True
            return decision

        self.router.update_vigilance = patched_update_vigilance

        # Test each fallback strategy
        for strategy in FallbackStrategy:
            meta_reflex = MetaReflexLayer(
                art_controller=self.art_controller,
                omega3_agent=self.omega3,
                decision_router=self.router,
                config={"fallback_strategy": strategy.value}
            )

            # Process input
            _, metadata = meta_reflex.process_input(checkerboard, "image")

            # Check that fallback was applied
            self.assertTrue(metadata["fallback_applied"])
            self.assertEqual(metadata["fallback_strategy"], strategy.value)

        # Restore original method
        self.router.update_vigilance = original_update_vigilance

    def test_report_outcome(self) -> None:
    """Test reporting outcome."""
        # Process input to get a decision
        test_input = np.random.rand(64, 64)
        processed, metadata = self.meta_reflex.process_input(test_input, "image")

        # Get decision ID
        decision_id = metadata["decision_id"]

        # Mock the router's report_outcome method
        original_report_outcome = self.router.report_outcome
        report_called = [False]

        def mock_report_outcome(d_id, outcome):
            report_called[0] = True
            self.assertEqual(d_id, decision_id)
            self.assertIn("success", outcome)
            original_report_outcome(d_id, outcome)

        self.router.report_outcome = mock_report_outcome

        # Report outcome
        outcome = {"success": 0.8, "correlation": 0.9, "confidence": 0.85}
        self.meta_reflex.report_outcome(decision_id, outcome)

        # Check that report was called
        self.assertTrue(report_called[0])

        # Restore original method
        self.router.report_outcome = original_report_outcome

    def test_get_statistics(self) -> None:
    """Test getting statistics."""
        # Process a few inputs to generate stats
        for _ in range(3):
            test_input = np.random.rand(64, 64)
            self.meta_reflex.process_input(test_input, "image")

        # Get statistics
        stats = self.meta_reflex.get_statistics()

        # Check statistics
        self.assertEqual(stats["processed_inputs"], 3)
        self.assertIn("high_risk_inputs", stats)
        self.assertIn("reflex_activations", stats)
        self.assertIn("fallback_counts", stats)

        # Check fallback counts
        for strategy in ["none", "retry", "decompose", "abstract", "hybrid"]:
            self.assertIn(strategy, stats["fallback_counts"])

if __name__ == "__main__":
    unittest.main()
