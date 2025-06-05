#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for art_controller.py."""

import os
import sys
import tempfile
import unittest

import numpy as np  # For array comparison in tests

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import ART components
from .art_controller import ARTController
from .art_logger import get_art_logger
from .art_manager import ARTManager


class TestARTController(unittest.TestCase):
    """Test cases for the ARTController class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test-specific logger
        self.logger = get_art_logger("test_art_controller")
        self.controller = ARTController(logger_instance=self.logger)
        print("Setting up TestARTController tests...")

    def tearDown(self):
        """Tear down test fixtures."""
        print("Tearing down TestARTController tests...")

    def test_initialization(self):
        """Test basic initialization of the ARTController."""
        self.assertIsNotNone(self.controller, "Controller should not be None")
        self.assertEqual(
            self.controller.vigilance, 0.5, "Default vigilance should be 0.5"
        )
        self.assertEqual(
            self.controller.learning_rate, 0.1, "Default learning rate should be 0.1"
        )
        print("Running test_initialization...")

    def test_category_creation(self):
        """Test the creation of new categories."""
        mock_input = [0.1, 0.2, 0.3]
        result = self.controller.train(mock_input)

        self.assertIsNotNone(result, "Result should not be None for new input")
        category_id = result.get("category_id")
        self.assertIsNotNone(category_id, "Category ID should be returned")
        self.assertIn(
            category_id,
            self.controller.categories,
            "Category should be tracked in categories dict",
        )
        self.assertEqual(
            len(self.controller.categories),
            1,
            "Should have 1 category after first input",
        )
        print("Running test_category_creation...")

    def test_resonance(self):
        """Test the resonance mechanism."""
        # Create first category
        mock_input1 = [0.1, 0.2, 0.3]
        result1 = self.controller.train(mock_input1)
        category1_id = result1.get("category_id")

        # Process similar input
        mock_input2 = [0.11, 0.21, 0.31]  # Similar to first input
        result2 = self.controller.train(mock_input2)
        category2_id = result2.get("category_id")

        # The similar input should resonate with the existing category
        self.assertEqual(
            category1_id,
            category2_id,
            "Similar input should resonate with existing category",
        )
        self.assertEqual(
            len(self.controller.categories),
            1,
            "Similar inputs should use the same category",
        )
        print("Running test_resonance...")

    def test_vigilance(self):
        """Test the vigilance parameter effect."""
        # Lower vigilance to allow more dissimilar inputs to resonate
        self.controller.set_vigilance(0.5)

        # Create first category
        mock_input1 = [0.1, 0.2, 0.3]
        result1 = self.controller.train(mock_input1)
        category1_id = result1.get("category_id")

        # Process moderately different input
        mock_input2 = [0.3, 0.4, 0.5]  # Moderately different
        result2 = self.controller.train(mock_input2)
        category2_id = result2.get("category_id")

        # With low vigilance, these should resonate
        self.assertEqual(
            category1_id,
            category2_id,
            "With low vigilance, somewhat different inputs should resonate",
        )

        # Now increase vigilance to be more selective
        self.controller.set_vigilance(0.9)

        # Process another moderately different input
        mock_input3 = [0.5, 0.6, 0.7]
        result3 = self.controller.train(mock_input3)
        category3_id = result3.get("category_id")

        # With high vigilance, these should create a new category
        self.assertNotEqual(
            category1_id,
            category3_id,
            "With high vigilance, different inputs should create new categories",
        )

    def test_learning(self):
        """Test the learning mechanism."""
        # Create first category
        mock_input1 = [0.1, 0.2, 0.3]
        result1 = self.controller.train(mock_input1)
        category_id = result1.get("category_id")

        # Get the original weights
        original_weights = self.controller.categories[category_id]["weights"].copy()

        # Process similar input to trigger learning
        mock_input2 = [0.15, 0.25, 0.35]  # Similar but slightly different
        self.controller.train(mock_input2)

        # Weights should be updated through learning
        new_weights = self.controller.categories[category_id]["weights"]
        self.assertFalse(
            np.array_equal(original_weights, new_weights),
            "Weights should be updated through learning",
        )

    def test_save_load_state(self):
        """Test saving and loading the controller state."""
        # Create some categories
        self.controller.train([0.1, 0.2, 0.3])
        self.controller.train([0.7, 0.8, 0.9])

        # Save state to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            save_path = temp_file.name

        self.controller.save_state(save_path)
        self.assertTrue(os.path.exists(save_path), "State file should exist after save")

        # Create a new controller and load state
        new_controller = ARTController(logger_instance=self.logger)
        new_controller.load_state(save_path)

        # Verify the new controller has the same categories
        self.assertEqual(
            len(self.controller.categories),
            len(new_controller.categories),
            "Loaded controller should have the same number of categories",
        )

        # Clean up
        os.unlink(save_path)


class TestARTManager(unittest.TestCase):
    """Test cases for the ARTManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test-specific logger
        self.logger = get_art_logger("test_art_manager")
        self.manager = ARTManager(logger_instance=self.logger)
        print("Setting up TestARTManager tests...")

    def tearDown(self):
        """Tear down test fixtures."""
        print("Tearing down TestARTManager tests...")

    def test_initialization(self):
        """Test basic initialization of ARTManager."""
        self.assertIsNotNone(self.manager, "ARTManager should not be None")
        self.assertIsNotNone(
            self.manager.controller, "ARTManager should have a controller"
        )
        self.assertIsNotNone(
            self.manager.pattern_analysis, "ARTManager should have pattern analysis"
        )
        self.assertIsNotNone(
            self.manager.duplication_checker,
            "ARTManager should have duplication checker",
        )
        print("Running test_initialization...")

    def test_analyze_input(self):
        """Test input analysis with ARTManager."""
        # Test with a simple text input
        result = self.manager.analyze_input("This is a test input for pattern analysis")

        self.assertIsInstance(result, dict, "Analysis result should be a dictionary")
        self.assertIn(
            "category", result, "Analysis should include category information"
        )
        self.assertIn(
            "is_novel_category", result, "Analysis should indicate if category is novel"
        )

        # First input should create a new category
        self.assertTrue(
            result["is_novel_category"], "First input should create a novel category"
        )

        # Test with another input to see if it creates a different category
        result2 = self.manager.analyze_input(
            "This is a completely different pattern for testing"
        )
        first_category_id = result["category"]["id"]
        second_category_id = result2["category"]["id"]

        # Different inputs should generally create different categories
        self.assertNotEqual(
            first_category_id,
            second_category_id,
            "Different inputs should create different categories",
        )

    def test_train_on_batch(self):
        """Test batch training with ARTManager."""
        # Create a batch of test data
        batch = [
            "Training example one",
            "Training example two",
            "Training example three",
            (
                "Query example",
                "Response example",
            ),  # Tuple format for query-response pairs
        ]

        result = self.manager.train_on_batch(batch)

        self.assertIsInstance(result, dict, "Training result should be a dictionary")
        self.assertIn("status", result, "Result should include status")
        self.assertIn(
            "items_processed", result, "Result should include items processed"
        )
        self.assertEqual(
            result["items_processed"], len(batch), "All items should be processed"
        )

    def test_log_pattern_trace(self):
        """Test pattern trace logging."""
        # First analyze an input to create a category
        input_text = "This is a test for pattern tracing"
        analysis = self.manager.analyze_input(input_text)
        category_id = analysis["category"]["id"]

        # Now log a trace for this pattern
        trace_result = self.manager.log_pattern_trace(
            pattern_id=category_id,
            source_data=input_text,
            context={"test": True},
            metadata={"purpose": "unit testing"},
        )

        self.assertIsInstance(trace_result, dict, "Trace result should be a dictionary")
        self.assertIn("pattern_id", trace_result, "Trace should include pattern ID")
        self.assertEqual(
            trace_result["pattern_id"],
            category_id,
            "Trace should reference correct pattern",
        )

    def test_passthrough_methods(self):
        """Test passthrough methods to ARTController."""
        # Test status
        status = self.manager.status()
        self.assertIsInstance(status, dict, "Status should be a dictionary")

        # Test vigilance setting
        self.manager.set_vigilance(0.6)
        self.assertEqual(
            self.manager.controller.vigilance, 0.6, "Vigilance should be updated"
        )

        # Test learning rate setting
        self.manager.set_learning_rate(0.3)
        self.assertEqual(
            self.manager.controller.learning_rate,
            0.3,
            "Learning rate should be updated",
        )


if __name__ == "__main__":
    unittest.main()
