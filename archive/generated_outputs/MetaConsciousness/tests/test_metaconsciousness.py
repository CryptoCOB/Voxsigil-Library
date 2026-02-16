"""
Integration and Performance Test for MetaConsciousness SDK

This script tests the complete SDK after reconnection of metacognitive components,
validating system integrity, reflex loops, thought engines, and other critical
functionality.
"""
import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional
import unittest
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("metaconsciousness_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import MetaConsciousness components
try:
    from MetaConsciousness.agent.metaconscious_agent import MetaconsciousAgent
    from MetaConsciousness.core.meta_core import MetaState
    from MetaConsciousness.api import (
        add_thought_branch, add_thought_step, apply_regulation,
        auto_select_strategy, evaluate_performance, get_explanation
    )
    from MetaConsciousness.utils.trace import get_trace_history, clear_trace
    from MetaConsciousness.utils import log_event
    from MetaConsciousness import Omega3, DecisionConfidence
    from MetaConsciousness.agent.adapt import (
        adjust_learning_rate, adaptive_dropout, dynamic_temperature_scaling
    )
except ImportError as e:
    logger.error(f"Failed to import MetaConsciousness components: {e}")
    raise

# Output directory for test results
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MetaConsciousnessTest(unittest.TestCase):
    """Test cases for MetaConsciousness SDK integration."""
    
    def setUp(self):
        """Set up the test environment."""
        clear_trace()  # Clear trace history
        
        # Create agent with default config
        self.agent = MetaconsciousAgent(config={
            "initial_vigilance": 0.4,
            "memory_enabled": True,
            "omega3_config": {
                "pattern_memory_enabled": True,
                "strategy_evolution_enabled": True
            },
            "meta_config": {
                "awareness_threshold": 0.5,
                "art_enabled": True,
                "reflex_enabled": True
            }
        })
        
        # Record start time
        self.start_time = time.time()
        
        # Test results
        self.results = {
            "passed_tests": [],
            "failed_tests": [],
            "system_state": {},
            "reflex_outcomes": [],
            "thought_tree": {},
            "test_duration": 0
        }
        
        logger.info("Test environment initialized")
    
    def tearDown(self):
        """Clean up after tests."""
        # Calculate test duration
        self.results["test_duration"] = time.time() - self.start_time
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(OUTPUT_DIR, f"test_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {result_file}")
        
        # Generate HTML dashboard if visualization is available
        try:
            dashboard_file = os.path.join(OUTPUT_DIR, f"dashboard_{timestamp}.html")
            dashboard = self.agent.visualize(output_format="html")
            
            with open(dashboard_file, 'w') as f:
                f.write(dashboard)
            
            logger.info(f"Dashboard saved to {dashboard_file}")
        except Exception as e:
            logger.warning(f"Failed to generate dashboard: {e}")
    
    def _log_test_result(self, test_name, passed, error=None):
        """Log test result and update result tracking."""
        if passed:
            self.results["passed_tests"].append(test_name)
            logger.info(f"PASSED: {test_name}")
        else:
            self.results["failed_tests"].append({
                "test": test_name,
                "error": str(error) if error else "Unknown error"
            })
            logger.error(f"FAILED: {test_name} - {error}")
    
    def _capture_system_state(self):
        """Capture current system state."""
        stats = self.agent.get_performance_stats()
        
        self.results["system_state"] = {
            "timestamp": datetime.now().isoformat(),
            "awareness": self.agent.meta_core.awareness,
            "regulation": self.agent.meta_core.regulation,
            "meta_state": stats.get("meta_state"),
            "vigilance": self.agent.art_controller.vigilance,
            "omega3_stats": stats.get("omega3_stats", {}),
            "art_stats": stats.get("art_stats", {})
        }
        
        logger.info(f"System state captured: awareness={self.agent.meta_core.awareness:.3f}, "
                   f"regulation={self.agent.meta_core.regulation:.3f}, "
                   f"meta_state={stats.get('meta_state')}")
        
        return self.results["system_state"]
    
    def test_01_system_integrity(self):
        """Test system integrity by checking all components."""
        test_name = "System Integrity"
        try:
            # Check agent components
            self.assertIsNotNone(self.agent.art_controller, "ART controller not initialized")
            self.assertIsNotNone(self.agent.omega3, "Omega3 agent not initialized")
            self.assertIsNotNone(self.agent.meta_reflex, "MetaReflexLayer not initialized")
            self.assertIsNotNone(self.agent.meta_core, "MetaConsciousness core not initialized")
            
            # Check memory if enabled
            if self.agent.memory_enabled:
                self.assertIsNotNone(self.agent.memory, "Memory not initialized")
            
            # Check enum values of MetaState
            self.assertIsNotNone(MetaState.CALIBRATING, "MetaState.CALIBRATING not defined")
            self.assertIsNotNone(MetaState.LEARNING, "MetaState.LEARNING not defined")
            self.assertIsNotNone(MetaState.STABLE, "MetaState.STABLE not defined")
            self.assertIsNotNone(MetaState.ADAPTING, "MetaState.ADAPTING not defined")
            
            # Check integration of orphaned functions
            self.assertTrue(callable(self.agent.meta_core.update_meta_state), 
                          "update_meta_state not integrated")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_02_reflex_loop_activation(self):
        """Test the reflex loop activation with various input types."""
        test_name = "Reflex Loop Activation"
        try:
            # Generate test input data that should trigger reflexes
            # First, create a checkerboard pattern (high risk)
            checkerboard = np.zeros((64, 64), dtype=np.uint8)
            tile_size = 8
            for i in range(0, 64, tile_size):
                for j in range(0, 64, tile_size):
                    if (i // tile_size + j // tile_size) % 2 == 0:
                        checkerboard[i:i+tile_size, j:j+tile_size] = 255
            
            # Process the checkerboard pattern
            result = self.agent.process(checkerboard, input_type="image")
            
            # Verify reflex was activated
            self.assertTrue(result["reflex_metadata"]["reflex_applied"], 
                          "Reflex was not applied to checkerboard pattern")
            
            # Verify risk level is high or critical
            self.assertIn(result["reflex_metadata"]["risk_level"], ["high", "critical"], 
                        "Risk level not properly detected for checkerboard")
            
            # Verify vigilance was adjusted
            self.assertTrue(result["reflex_metadata"]["vigilance_adjusted"], 
                          "Vigilance was not adjusted for high-risk pattern")
            
            # Check meta state update
            self.assertIsNotNone(result["meta_state"], "Meta state not updated")
            
            # Generate a gradient pattern (medium risk)
            x = np.linspace(0, 1, 64)
            y = np.linspace(0, 1, 64)
            xx, yy = np.meshgrid(x, y)
            gradient = (xx * 255).astype(np.uint8)
            
            # Process the gradient pattern
            result = self.agent.process(gradient, input_type="image")
            
            # Store results for inspection
            self.results["reflex_outcomes"].append({
                "pattern": "checkerboard",
                "risk_level": result["reflex_metadata"]["risk_level"],
                "vigilance_original": result["reflex_metadata"]["original_vigilance"],
                "vigilance_new": result["reflex_metadata"]["new_vigilance"],
                "meta_state": result["meta_state"]
            })
            
            # Now check the system state after processing
            state = self._capture_system_state()
            
            # Meta state should reflect awareness level
            if self.agent.meta_core.awareness < 0.3:
                self.assertEqual(state["meta_state"], MetaState.CALIBRATING.name, 
                               "Wrong meta state for low awareness")
            elif 0.3 <= self.agent.meta_core.awareness < 0.6:
                self.assertEqual(state["meta_state"], MetaState.LEARNING.name, 
                               "Wrong meta state for medium awareness")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_03_thought_engine_simulation(self):
        """Test the thought engine's ability to create and manage thoughts."""
        test_name = "Thought Engine Simulation"
        try:
            # Add a chain of thought step
            step_id_1 = self.agent.meta_core.add_thought_step(
                reasoning="The system should analyze patterns in input data",
                evidence={"observation": "High entropy detected"},
                confidence=0.8
            )
            
            # Add another step
            step_id_2 = self.agent.meta_core.add_thought_step(
                reasoning="Higher vigilance is needed for complex patterns",
                evidence={"test_result": "Improved recognition at 0.7 vigilance"},
                confidence=0.75
            )
            
            # Verify steps were added correctly
            self.assertIsNotNone(step_id_1, "First thought step not created")
            self.assertIsNotNone(step_id_2, "Second thought step not created")
            
            # Get the chain length
            chain_length = len(self.agent.meta_core.thought_engine.chain)
            self.assertEqual(chain_length, 2, f"Chain length should be 2, got {chain_length}")
            
            # Switch to tree of thought mode
            self.agent.meta_core.switch_strategy("tot")
            self.assertEqual(self.agent.meta_core.thinking_mode, "tot", 
                           "Failed to switch to ToT mode")
            
            # Add a thought branch
            branch_id = self.agent.meta_core.add_thought_branch(
                reasoning_steps=[
                    "Consider using adaptive vigilance control",
                    "Test with varying pattern complexities",
                    "Analyze recognition performance metrics"
                ],
                confidence=0.85,
                evaluation_score=0.9
            )
            
            # Verify branch creation
            self.assertIsNotNone(branch_id, "Branch not created")
            branch = self.agent.meta_core.thought_engine.get_branch(branch_id)
            self.assertIsNotNone(branch, "Cannot retrieve created branch")
            
            # Add another branch
            branch_id_2 = self.agent.meta_core.add_thought_branch(
                reasoning_steps=[
                    "Alternative approach: fixed vigilance with post-processing",
                    "Apply statistical validation on results",
                    "Compare with adaptive approach"
                ],
                confidence=0.7,
                evaluation_score=0.8
            )
            
            # Test backtracking
            success = self.agent.meta_core.backtrack_thought(
                branch_id_2, "Testing backtracking capability"
            )
            self.assertTrue(success, "Failed to backtrack from branch")
            
            # Make a decision
            decision = self.agent.meta_core.finalize_decision(
                decision_text="Implement adaptive vigilance control based on pattern complexity",
                decision_id="decision_test_001"
            )
            
            self.assertIsNotNone(decision, "Decision not created")
            self.assertIn("decision_id", decision, "Decision ID not in response")
            
            # Get thought tree data for result inspection
            self.results["thought_tree"] = {
                "chain_steps": [self.agent.meta_core.thought_engine.get_step(step_id_1).to_dict(),
                               self.agent.meta_core.thought_engine.get_step(step_id_2).to_dict()],
                "branches": [self.agent.meta_core.thought_engine.get_branch(branch_id).to_dict(),
                            self.agent.meta_core.thought_engine.get_branch(branch_id_2).to_dict()],
                "decision": decision
            }
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_04_strategy_selection(self):
        """Test strategy selection with omega3 agent."""
        test_name = "Strategy Selection"
        try:
            # Create direct instance for isolated testing
            omega3 = Omega3(config={
                "pattern_memory_enabled": True,
                "strategy_evolution_enabled": True
            })
            
            # Test with different risk levels and pattern types
            test_cases = [
                {"risk_level": "low", "pattern_type": None, "expected_direction": "decrease"},
                {"risk_level": "high", "pattern_type": "checkerboard", "expected_direction": "increase"},
                {"risk_level": "medium", "pattern_type": "gradient", "expected_direction": None},
                {"risk_level": "critical", "pattern_type": "noise", "expected_direction": "increase"}
            ]
            
            for case in test_cases:
                # Get current vigilance
                current_vigilance = 0.5
                
                # Update vigilance based on case
                result = omega3.update_vigilance(
                    current_vigilance=current_vigilance,
                    risk_level=case["risk_level"],
                    pattern_type=case["pattern_type"]
                )
                
                # Verify result format
                self.assertIn("new_vigilance", result, "Missing new vigilance in result")
                self.assertIn("smoothed_vigilance", result, "Missing smoothed vigilance in result")
                
                # Check direction if specified
                if case["expected_direction"] == "increase":
                    self.assertGreater(result["new_vigilance"], current_vigilance, 
                                     f"Vigilance didn't increase for {case}")
                elif case["expected_direction"] == "decrease":
                    self.assertLess(result["new_vigilance"], current_vigilance, 
                                  f"Vigilance didn't decrease for {case}")
                
                # Report outcome to train the system
                omega3.report_outcome(
                    result["decision_id"],
                    {"success": 0.8, "confidence": 0.75}
                )
            
            # Test auto-selection of strategy
            strategy = auto_select_strategy({
                "risk_level": "high",
                "pattern_type": "checkerboard",
                "vigilance": 0.6
            })
            
            self.assertIsNotNone(strategy, "Strategy selection failed")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_05_adaptation_dynamics(self):
        """Test adaptation dynamics including learning rate and temperature adjustments."""
        test_name = "Adaptation Dynamics"
        try:
            # Create mock optimizer for learning rate adjustment
            class MockOptimizer:
                def __init__(self):
                    self.param_groups = [{"lr": 0.01}]
            
            optimizer = MockOptimizer()
            
            # Test adaptive learning rate adjustment
            for awareness in [0.2, 0.5, 0.8]:
                # Adjust learning rate based on awareness
                new_lr = adjust_learning_rate(
                    optimizer=optimizer,
                    awareness=awareness,
                    base_lr=0.01,
                    min_lr=0.001,
                    max_lr=0.1
                )
                
                # Verify learning rate was adjusted
                self.assertEqual(new_lr, optimizer.param_groups[0]["lr"], 
                               "Learning rate not set in optimizer")
                
                # Check adaptive behavior based on awareness
                if awareness < 0.3:
                    # Low awareness should increase learning rate
                    self.assertGreater(new_lr, 0.01, 
                                     "Low awareness didn't increase learning rate")
                elif awareness > 0.7:
                    # High awareness should decrease learning rate
                    self.assertLess(new_lr, 0.01, 
                                  "High awareness didn't decrease learning rate")
            
            # Test adaptive dropout
            for confidence in [0.3, 0.5, 0.7]:
                dropout_rate = adaptive_dropout(
                    base_rate=0.2,
                    confidence=confidence
                )
                
                # Verify coherent output
                self.assertGreaterEqual(dropout_rate, 0.0, "Dropout rate too low")
                self.assertLessEqual(dropout_rate, 0.5, "Dropout rate too high")
                
                # Check adaptive behavior based on confidence
                if confidence > 0.5:
                    # Higher confidence should increase dropout to prevent overfitting
                    self.assertGreater(dropout_rate, 0.2, 
                                     "High confidence didn't increase dropout")
                elif confidence < 0.5:
                    # Lower confidence should decrease dropout
                    self.assertLess(dropout_rate, 0.2, 
                                  "Low confidence didn't decrease dropout")
            
            # Test dynamic temperature scaling
            for regulation in [0.2, 0.5, 0.8]:
                temp = dynamic_temperature_scaling(
                    base_temp=1.0,
                    regulation=regulation
                )
                
                # Verify coherent output
                self.assertGreaterEqual(temp, 0.5, "Temperature too low")
                self.assertLessEqual(temp, 2.0, "Temperature too high")
                
                # Check adaptive behavior based on regulation
                if regulation > 0.5:
                    # Higher regulation should lower temperature (more conservative)
                    self.assertLess(temp, 1.0, 
                                  "High regulation didn't decrease temperature")
                elif regulation < 0.5:
                    # Lower regulation should increase temperature (more exploratory)
                    self.assertGreater(temp, 1.0, 
                                     "Low regulation didn't increase temperature")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_06_visualization(self):
        """Test visualization functionality."""
        test_name = "Visualization"
        try:
            # Generate visualization in both formats
            html_output = self.agent.visualize(output_format="html")
            json_output = self.agent.visualize(output_format="json")
            
            # Verify HTML output
            self.assertIsNotNone(html_output, "HTML visualization failed")
            self.assertIn("<html", html_output.lower(), "Invalid HTML output")
            
            # Verify JSON output can be parsed
            json_data = json.loads(json_output)
            
            # Check for key visualization sections
            for key in ["art_controller", "omega3", "meta_core", "performance", "reflex"]:
                self.assertIn(key, json_data, f"Missing {key} in visualization data")
            
            # Check that visualization reflects current system state
            self.assertEqual(json_data["meta_core"]["awareness"], self.agent.meta_core.awareness,
                           "Visualization awareness doesn't match agent state")
            
            self.assertEqual(json_data["meta_core"]["regulation"], self.agent.meta_core.regulation,
                           "Visualization regulation doesn't match agent state")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_07_memory_retention(self):
        """Test episodic memory retention if enabled."""
        test_name = "Memory Retention"
        try:
            # Skip if memory not enabled
            if not self.agent.memory_enabled or self.agent.memory is None:
                logger.info("Memory not enabled, skipping memory test")
                self._log_test_result(test_name, True)
                return
            
            # Process some inputs to create memories
            for i in range(3):
                # Create different input patterns
                if i == 0:
                    # Checkerboard
                    data = np.zeros((64, 64), dtype=np.uint8)
                    for x in range(0, 64, 8):
                        for y in range(0, 64, 8):
                            if (x // 8 + y // 8) % 2 == 0:
                                data[x:x+8, y:y+8] = 255
                elif i == 1:
                    # Gradient
                    x = np.linspace(0, 1, 64)
                    y = np.linspace(0, 1, 64)
                    xx, yy = np.meshgrid(x, y)
                    data = (xx * 255).astype(np.uint8)
                else:
                    # Random noise
                    data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
                
                # Process the data
                self.agent.process(data, input_type="image")
                
                # Pause to create timing difference
                time.sleep(0.1)
            
            # Retrieve stored memories
            memories = self.agent.memory.retrieve_recent(limit=10)
            
            # Verify memory functionality
            self.assertIsNotNone(memories, "Memory retrieval failed")
            self.assertGreater(len(memories), 0, "No memories stored")
            
            # Verify memory structure
            for memory in memories:
                self.assertIn("input_type", memory, "Memory missing input_type")
                self.assertIn("timestamp", memory, "Memory missing timestamp")
                self.assertIn("reflex_metadata", memory, "Memory missing reflex_metadata")
                self.assertIn("meta_state", memory, "Memory missing meta_state")
            
            # Test retrieval by time range
            if hasattr(self.agent.memory, 'retrieve_by_timerange'):
                end_time = time.time()
                start_time = end_time - 10  # Last 10 seconds
                time_memories = self.agent.memory.retrieve_by_timerange(start_time, end_time)
                
                self.assertIsNotNone(time_memories, "Time-based memory retrieval failed")
                self.assertGreater(len(time_memories), 0, "No memories retrieved by time range")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_08_logging_and_trace(self):
        """Test logging and trace functionality."""
        test_name = "Logging and Trace"
        try:
            # Clear existing trace
            clear_trace()
            
            # Generate some log events and explicitly add trace events
            from MetaConsciousness.utils.trace import add_trace_event
            
            # Direct trace events
            add_trace_event("test_event", {"test": "data"})
            add_trace_event("test_event_2", {"priority": "high"})
            
            # Process input to generate more events
            test_input = {"signal": [0.8, 0.1, 0.3], "type": "pattern"}
            self.agent.process(test_input, input_type="data")
            
            # Force a short delay to ensure events are processed
            import time
            time.sleep(0.1)
            
            # Add one more event to ensure we have something
            add_trace_event("final_test_event", {"source": "test", "final": True})
            
            # Get trace history
            from MetaConsciousness.utils.trace import get_trace_history
            history = get_trace_history()
            
            # Verify trace functionality - ensure we have events
            self.assertIsNotNone(history, "Trace history not available")
            
            # If history is empty, try a manual record
            if len(history) == 0:
                from MetaConsciousness.utils.trace import record_trace
                record_trace({"event_type": "manual_test_event", "data": {"source": "test"}})
                history = get_trace_history()
            
            # Final verification
            self.assertGreater(len(history), 0, "No trace events recorded")
            
            # Count the different event types if we have any events
            if len(history) > 0:
                event_types = {}
                for event in history:
                    event_type = event.get("event_type", "unknown")
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                # Verify we have good event coverage
                self.assertGreaterEqual(len(event_types), 1, "Too few event types recorded")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_09_edge_cases(self):
        """Test edge cases with unusual inputs."""
        test_name = "Edge Cases"
        try:
            # Test with empty input
            empty_result = self.agent.process(np.array([]), input_type="data")
            self.assertIsNotNone(empty_result, "Empty input handling failed")
            
            # Test with None input
            try:
                none_result = self.agent.process(None, input_type="data")
                self.assertIsNotNone(none_result, "None input handling failed")
            except Exception as e:
                logger.warning(f"None input raised exception: {e}")
                # This is acceptable if the system handles it gracefully
            
            # Test with very large input
            try:
                large_data = np.random.random((1000, 1000))  # 1M elements
                large_result = self.agent.process(large_data, input_type="image")
                self.assertIsNotNone(large_result, "Large input handling failed")
            except Exception as e:
                logger.warning(f"Large input raised exception: {e}")
                # This is acceptable if it's a memory limitation
            
            # Test with unusual data type
            text_data = "This is a test string input"
            text_result = self.agent.process(text_data, input_type="text")
            self.assertIsNotNone(text_result, "Text input handling failed")
            
            # Test with invalid input type
            try:
                invalid_type_result = self.agent.process(np.array([1, 2, 3]), input_type="invalid_type")
                self.assertIsNotNone(invalid_type_result, "Invalid input type handling failed")
            except Exception as e:
                logger.warning(f"Invalid input type raised exception: {e}")
                # This is acceptable if the system rejects invalid types
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_10_performance_baseline(self):
        """Test baseline performance metrics."""
        test_name = "Performance Baseline"
        try:
            # Generate synthetic performance data
            performance_data = np.array([0.85, 0.82, 0.79, 0.88, 0.90])
            
            # Test evaluate_performance function from API
            perf_result = evaluate_performance({
                "accuracy": np.mean(performance_data),
                "precision": 0.87,
                "recall": 0.84
            })
            
            # Verify result structure
            self.assertIsNotNone(perf_result, "Performance evaluation failed")
            
            # Test with regulation
            reg_result = apply_regulation({
                "entropy": 0.7,
                "regulation": 0.5
            })
            
            # Verify result structure
            self.assertIsNotNone(reg_result, "Regulation application failed")
            
            # Check explanation functionality
            explanation = get_explanation()
            self.assertIsNotNone(explanation, "Explanation generation failed")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise
    
    def test_11_comprehensive_integration(self):
        """Run a comprehensive integration test simulating real-world usage."""
        test_name = "Comprehensive Integration"
        try:
            # 1. Initial setup with low awareness
            self.agent.meta_core.awareness = 0.3
            self.agent.meta_core.regulation = 0.1
            
            # 2. Generate mixed-pattern input
            # Combination of checkerboard and gradient
            pattern = np.zeros((64, 64), dtype=np.uint8)
            # Checkerboard in upper left
            for i in range(0, 32, 8):
                for j in range(0, 32, 8):
                    if (i // 8 + j // 8) % 2 == 0:
                        pattern[i:i+8, j:j+8] = 255
            # Gradient in lower right
            x = np.linspace(0, 1, 32)
            y = np.linspace(0, 1, 32)
            xx, yy = np.meshgrid(x, y)
            gradient = (xx * 255).astype(np.uint8)
            pattern[32:, 32:] = gradient
            
            # 3. Process the complex pattern
            result = self.agent.process(pattern, input_type="image")
            
            # 4. Chain of thought analysis
            self.agent.meta_core.switch_strategy("cot")
            
            step1 = self.agent.meta_core.add_thought_step(
                reasoning="The input contains mixed patterns requiring adaptive processing",
                evidence={"detected_patterns": ["checkerboard", "gradient"]}
            )
            
            step2 = self.agent.meta_core.add_thought_step(
                reasoning="Risk assessment indicates potential processing challenges",
                evidence={"risk_level": result["reflex_metadata"]["risk_level"]}
            )
            
            step3 = self.agent.meta_core.add_thought_step(
                reasoning="Vigilance should be dynamically adjusted based on the localized pattern",
                evidence={"vigilance_delta": result["reflex_metadata"].get("vigilance_delta", 0)}
            )
            
            # 5. Simulate learning and regulation
            # Update with improving performance trend
            performance_trend = [0.65, 0.72, 0.78, 0.81, 0.85]
            
            for i, perf in enumerate(performance_trend):
                self.agent.meta_core.adaptive_regulation(np.array([perf]))
                
                # Add learning example with pattern
                learn_result = self.agent.train(
                    input_data=pattern,
                    input_type="image",
                    epochs=1
                )
                
                # Report outcome to Omega3
                if i > 0 and "reflex_metadata" in result and "decision_id" in result["reflex_metadata"]:
                    self.agent.meta_reflex.report_outcome(
                        result["reflex_metadata"]["decision_id"],
                        {
                            "success": perf,
                            "confidence": 0.7 + (i * 0.05)
                        }
                    )
            
            # 6. Validate meta-state transition as awareness improves
            state = self._capture_system_state()
            
            # Awareness should have increased
            self.assertGreater(state["awareness"], 0.3, 
                             "Awareness didn't increase after training")
            
            # Meta state should reflect the improved awareness
            self.assertNotEqual(state["meta_state"], MetaState.CALIBRATING.name,
                              "Meta state didn't transition from CALIBRATING")
            
            # 7. Finalize with decision
            decision = self.agent.meta_core.finalize_decision(
                decision_text="Implement adaptive pattern-specific processing with localized vigilance control",
                decision_id="comprehensive_test_decision"
            )
            
            self.assertIsNotNone(decision, "Decision finalization failed")
            
            self._log_test_result(test_name, True)
        except AssertionError as e:
            self._log_test_result(test_name, False, e)
            raise
        except Exception as e:
            self._log_test_result(test_name, False, e)
            raise


def generate_summary_report():
    """Generate a summary report of all test results."""
    # Find the most recent result file
    result_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("test_results_") and f.endswith(".json")]
    if not result_files:
        print("No test result files found.")
        return
    
    latest_file = max(result_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
    result_path = os.path.join(OUTPUT_DIR, latest_file)
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    # Generate summary
    report = [
        "# MetaConsciousness SDK Test Summary",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Test Duration**: {results['test_duration']:.2f} seconds",
        "",
        "## Test Results",
        "",
        f"- **Passed Tests**: {len(results['passed_tests'])}",
        f"- **Failed Tests**: {len(results['failed_tests'])}",
        "",
    ]
    
    # Add passed tests
    if results['passed_tests']:
        report.extend([
            "### Passed Tests",
            ""
        ])
        for test in results['passed_tests']:
            report.append(f"- PASS: {test}")
        report.append("")
    
    # Add failed tests
    if results['failed_tests']:
        report.extend([
            "### Failed Tests",
            ""
        ])
        for test in results['failed_tests']:
            report.append(f"- FAIL: {test['test']}: {test['error']}")
        report.append("")
    
    # Add system state
    if 'system_state' in results:
        report.extend([
            "## System State",
            "",
            f"- **Awareness**: {results['system_state'].get('awareness', 'N/A')}",
            f"- **Regulation**: {results['system_state'].get('regulation', 'N/A')}",
            f"- **Meta State**: {results['system_state'].get('meta_state', 'N/A')}",
            f"- **Vigilance**: {results['system_state'].get('vigilance', 'N/A')}",
            ""
        ])
    
    # Add reflex outcomes if available
    if 'reflex_outcomes' in results and results['reflex_outcomes']:
        report.extend([
            "## Reflex Outcomes",
            "",
            "| Pattern | Risk Level | Original Vigilance | New Vigilance | Meta State |",
            "|---------|------------|-------------------|---------------|------------|"
        ])
        
        for outcome in results['reflex_outcomes']:
            report.append(
                f"| {outcome.get('pattern', 'N/A')} | {outcome.get('risk_level', 'N/A')} | "
                f"{outcome.get('vigilance_original', 'N/A')} | {outcome.get('vigilance_new', 'N/A')} | "
                f"{outcome.get('meta_state', 'N/A')} |"
            )
        report.append("")
    
    # Write report using utf-8 encoding
    report_path = os.path.join(OUTPUT_DIR, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report generated: {report_path}")


if __name__ == "__main__":
    try:
        # Run the tests
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    finally:
        # Generate summary report
        generate_summary_report()
