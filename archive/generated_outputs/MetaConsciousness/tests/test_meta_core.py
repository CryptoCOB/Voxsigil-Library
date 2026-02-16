"""
Unit tests for MetaConsciousness core functionality.
"""
import sys
import os
import unittest
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path to import MetaConsciousness
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MetaConsciousness.core.meta_core import MetaConsciousness
from MetaConsciousness.utils.config import load_config
from MetaConsciousness.memory import EpisodicMemory
from MetaConsciousness.memory.thought_engine import ThoughtBranch, ChainStep

class TestMetaConsciousness(unittest.TestCase):
    
    def setUp(self):
        # Load default config
        self.config = load_config()
        # Initialize MetaConsciousness
        self.meta = MetaConsciousness(self.config)
    
    def test_initialization(self):
        """Test that MetaConsciousness initializes correctly."""
        self.assertEqual(self.meta.awareness, 0.5)
        self.assertEqual(self.meta.regulation, 0.0)
        self.assertEqual(len(self.meta.awareness_history), 0)
        self.assertEqual(len(self.meta.regulation_history), 0)
        self.assertEqual(len(self.meta.performance_history), 0)
    
    def test_monitor(self):
        """Test that monitoring updates awareness."""
        # Monitor high performance
        awareness = self.meta.monitor(np.array([0.9, 0.8, 0.9]))
        self.assertAlmostEqual(awareness, 0.8666, places=3)
        self.assertAlmostEqual(self.meta.awareness, 0.8666, places=3)
        self.assertEqual(len(self.meta.awareness_history), 1)
        self.assertEqual(len(self.meta.performance_history), 1)
        
        # Monitor low performance
        awareness = self.meta.monitor(np.array([0.2, 0.3, 0.1]))
        self.assertAlmostEqual(awareness, 0.2, places=3)
        self.assertAlmostEqual(self.meta.awareness, 0.2, places=3)
        self.assertEqual(len(self.meta.awareness_history), 2)
        self.assertEqual(len(self.meta.performance_history), 2)
    
    def test_regulate(self):
        """Test that regulation adjusts based on awareness."""
        # Set awareness to low value
        self.meta.awareness = 0.3
        regulation = self.meta.regulate()
        
        # Regulation should increase
        self.assertGreater(regulation, 0.0)
        self.assertEqual(len(self.meta.regulation_history), 1)
        
        # Set awareness to high value
        self.meta.awareness = 0.8
        prev_regulation = regulation
        regulation = self.meta.regulate()
        
        # Regulation should decrease
        self.assertLess(regulation, prev_regulation)
        self.assertEqual(len(self.meta.regulation_history), 2)
    
    def test_adaptive_regulation(self):
        """Test combined monitoring and regulation."""
        regulation = self.meta.adaptive_regulation(np.array([0.2, 0.2, 0.2]))
        
        # With low performance, regulation should increase
        self.assertGreater(regulation, 0.0)
        self.assertEqual(len(self.meta.awareness_history), 1)
        self.assertEqual(len(self.meta.regulation_history), 1)
        
        prev_regulation = regulation
        regulation = self.meta.adaptive_regulation(np.array([0.9, 0.9, 0.9]))
        
        # With high performance, regulation should decrease
        self.assertLess(regulation, prev_regulation)
        self.assertEqual(len(self.meta.awareness_history), 2)
        self.assertEqual(len(self.meta.regulation_history), 2)
    
    def test_evaluate(self):
        """Test that evaluate returns correct structure."""
        # Monitor and regulate to create history
        self.meta.adaptive_regulation(np.array([0.5, 0.5, 0.5]))
        
        # Get evaluation
        state = self.meta.evaluate()
        
        # Check structure
        self.assertIn('awareness', state)
        self.assertIn('regulation', state)
        self.assertIn('history_length', state)
        self.assertIn('timestamp', state)
        self.assertIn('explanation', state)
        
    def test_output_modulation(self):
        """Test that output modulation works correctly."""
        # Set awareness and regulation
        self.meta.awareness = 0.8
        self.meta.regulation = 0.2
        
        # Create test input
        input_tensor = torch.ones((1, self.config['input_dim']))
        
        # Get modulated output
        output = self.meta.generate_final_output(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.config['output_dim']))
        
        # Check that output values are positive (ReLU)
        self.assertTrue((output >= 0).all())
        
        # Now try with lower awareness
        self.meta.awareness = 0.2
        lower_output = self.meta.generate_final_output(input_tensor)
        
        # Lower awareness should lead to generally lower output values
        self.assertTrue(lower_output.sum() < output.sum())
    
    def test_state_dict(self):
        """Test state saving and loading."""
        # Add some history
        self.meta.adaptive_regulation(np.array([0.7, 0.8, 0.9]))
        self.meta.adaptive_regulation(np.array([0.4, 0.5, 0.6]))
        
        # Get state dict
        state = self.meta.state_dict()
        
        # Check keys
        self.assertIn('weights', state)
        self.assertIn('awareness', state)
        self.assertIn('regulation', state)
        self.assertIn('awareness_history', state)
        
        # Create new instance and load state
        new_meta = MetaConsciousness(self.config)
        new_meta.load_state_dict(state)
        
        # Check that state was loaded correctly
        self.assertEqual(new_meta.awareness, self.meta.awareness)
        self.assertEqual(new_meta.regulation, self.meta.regulation)
        self.assertEqual(len(new_meta.awareness_history), len(self.meta.awareness_history))

class TestEpisodicMemory(unittest.TestCase):
    
    def setUp(self):
        # Initialize memory
        self.memory = EpisodicMemory(capacity=5, decay_rate=0.2)
    
    def test_add_retrieve(self):
        """Test adding and retrieving memories."""
        # Add memories
        self.memory.add("Memory 1", importance=0.8)
        self.memory.add("Memory 2", importance=0.3)
        self.memory.add("Memory 3", importance=0.6)
        
        # Retrieve most important
        important = self.memory.retrieve_most_important(2)
        self.assertEqual(len(important), 2)
        self.assertEqual(important[0][0], "Memory 1")  # Most important
        
        # Retrieve recent
        recent = self.memory.retrieve_recent(2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0][0], "Memory 3")  # Most recent
    
    def test_capacity(self):
        """Test that capacity is respected."""
        # Add more memories than capacity
        for i in range(10):
            self.memory.add(f"Memory {i}", importance=i/10)
        
        # Should only keep the most important ones
        self.assertEqual(self.memory.count(), 5)  # Capacity is 5
        
        # Most important should be high-numbered memories
        important = self.memory.retrieve_most_important(1)
        self.assertEqual(important[0][0], "Memory 9")

class TestThoughtDynamics(unittest.TestCase):
    """Tests for thought dynamics functionality."""
    
    def setUp(self):
        # Load default config
        self.config = load_config()
        # Initialize MetaConsciousness
        self.meta = MetaConsciousness(self.config)
    
    def test_chain_of_thought(self):
        """Test Chain of Thought functionality."""
        # Set to CoT mode
        self.meta.switch_strategy("cot")
        self.assertEqual(self.meta.thinking_mode, "cot")
        
        # Add thought steps
        step1_id = self.meta.add_thought_step("First step in reasoning", confidence=0.8)
        step2_id = self.meta.add_thought_step("Second step in reasoning", confidence=0.7)
        
        # Check steps were added
        self.assertEqual(len(self.meta.thought_engine.chain), 2)
        
        # Check step content
        step1 = self.meta.thought_engine.get_step(step1_id)
        self.assertEqual(step1.reasoning, "First step in reasoning")
        self.assertEqual(step1.confidence, 0.8)
        
        # Finalize decision
        decision = self.meta.finalize_decision("Final decision based on chain of thought")
        
        # Check decision
        self.assertEqual(decision["explanation"], "Final decision based on chain of thought")
        self.assertEqual(len(decision["chain_steps"]), 2)
    
    def test_tree_of_thought(self):
        """Test Tree of Thought functionality."""
        # Set to ToT mode
        self.meta.switch_strategy("tot")
        self.assertEqual(self.meta.thinking_mode, "tot")
        
        # Add thought branches
        branch1_id = self.meta.add_thought_branch(
            ["Branch 1 step 1", "Branch 1 step 2"],
            confidence=0.7,
            evaluation_score=0.6
        )
        
        branch2_id = self.meta.add_thought_branch(
            ["Branch 2 step 1", "Branch 2 step 2", "Branch 2 step 3"],
            confidence=0.8,
            evaluation_score=0.9
        )
        
        # Check branches were added
        self.assertEqual(len(self.meta.thought_engine.branches), 2)
        
        # Check branch content
        branch1 = self.meta.thought_engine.get_branch(branch1_id)
        self.assertEqual(len(branch1.steps), 2)
        self.assertEqual(branch1.confidence, 0.7)
        self.assertEqual(branch1.evaluation_score, 0.6)
        
        # Check best branch is branch2 (higher evaluation score)
        best_branch = self.meta.thought_engine.get_best_branch()
        self.assertEqual(best_branch.branch_id, branch2_id)
        
        # Finalize decision
        decision = self.meta.finalize_decision("Final decision based on tree of thought")
        
        # Check decision
        self.assertEqual(decision["explanation"], "Final decision based on tree of thought")
        self.assertEqual(decision["chosen_branch_id"], branch2_id)
    
    def test_backtracking(self):
        """Test backtracking functionality."""
        # Add a branch and backtrack
        self.meta.switch_strategy("tot")
        branch_id = self.meta.add_thought_branch(
            ["Step 1", "Step 2", "Step 3"],
            confidence=0.7
        )
        
        # Verify branch exists
        self.assertIn(branch_id, self.meta.thought_engine.branches)
        
        # Backtrack
        result = self.meta.backtrack_thought(branch_id, "Testing backtracking")
        
        # Verify backtracking worked
        self.assertTrue(result)
        self.assertEqual(len(self.meta.thought_engine.backtrack_history), 1)
        self.assertEqual(self.meta.thought_engine.backtrack_history[0]["from_id"], branch_id)
        self.assertEqual(self.meta.thought_engine.backtrack_history[0]["reason"], "Testing backtracking")
    
    def test_strategy_switching(self):
        """Test strategy switching functionality."""
        # Start with CoT
        self.meta.switch_strategy("cot")
        self.assertEqual(self.meta.thinking_mode, "cot")
        
        # Switch to ToT
        result = self.meta.switch_strategy("tot")
        self.assertTrue(result)
        self.assertEqual(self.meta.thinking_mode, "tot")
        
        # Try invalid strategy
        result = self.meta.switch_strategy("invalid")
        self.assertFalse(result)
        self.assertEqual(self.meta.thinking_mode, "tot")  # Unchanged
    
    def test_auto_strategy_selection(self):
        """Test automatic strategy selection."""
        # Set awareness to neutral
        self.meta.awareness = 0.5
        
        # Simple task, high time pressure should select CoT
        strategy = self.meta.auto_select_strategy(0.3, 0.8)
        self.assertEqual(strategy, "cot")
        
        # Complex task, low time pressure should select ToT
        strategy = self.meta.auto_select_strategy(0.8, 0.2)
        self.assertEqual(strategy, "tot")
    
    def test_regulate_branch_exploration(self):
        """Test regulation of branch exploration."""
        # Set specific awareness and regulation
        self.meta.awareness = 0.8
        self.meta.regulation = 0.3
        
        # Get regulated exploration parameters
        params = self.meta.regulate_branch_exploration()
        
        # Check that parameters were adjusted based on awareness and regulation
        self.assertGreater(params["depth"], 1)
        self.assertGreater(params["width"], 1)
        self.assertEqual(params["confidence"], 0.8)

if __name__ == '__main__':
    unittest.main()
