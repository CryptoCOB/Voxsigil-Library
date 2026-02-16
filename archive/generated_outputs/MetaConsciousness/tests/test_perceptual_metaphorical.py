"""
Unit tests for the perceptual and metaphorical layer components.
"""
import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path to import MetaConsciousness
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MetaConsciousness.core.metaphor_core import (
    identify_metaphors, generate_analogy, metaphorical_reasoning_chain,
    blend_concepts, is_metaphorical
)
from MetaConsciousness.omega3.awareness_dynamics import (
    AwarenessDynamics, AwarenessState, AttentionMode
)
from MetaConsciousness.core.perceptual_mapping import (
    cross_modal_mapping, simulate_embodied_experience, create_sensory_analogy,
    sensory_intensity_projection
)
from MetaConsciousness.omega3.narrative_consciousness import (
    build_internal_narrative, annotate_reasoning_with_qualia,
    simulate_reflective_echo, generate_reflective_questions
)

class TestMetaphorCore(unittest.TestCase):
    """Tests for the metaphor and analogy functionality."""
    
    def test_identify_metaphors(self):
        """Test metaphor identification in text."""
        text = "The river of consciousness flows through the landscape of memory."
        metaphors = identify_metaphors(text)
        self.assertGreater(len(metaphors), 0)
        self.assertEqual(metaphors[0].source_domain, "river")
        self.assertEqual(metaphors[0].target_domain, "consciousness")
    
    def test_generate_analogy(self):
        """Test generation of analogies for a concept."""
        analogy = generate_analogy("learning", "journey")
        self.assertEqual(analogy.source_domain, "journey")
        self.assertEqual(analogy.target_domain, "learning")
        self.assertIsNotNone(analogy.text)
    
    def test_metaphorical_reasoning(self):
        """Test metaphorical reasoning between concepts."""
        chain = metaphorical_reasoning_chain("creativity", "problem_solving")
        self.assertIn("steps", chain)
        self.assertGreaterEqual(len(chain["steps"]), 3)
        self.assertIn("bridge_domain", chain)
    
    def test_is_metaphorical(self):
        """Test detection of metaphorical language."""
        literal_text = "The data shows a 5% increase in performance metrics."
        metaphorical_text = "The sea of data revealed islands of insight."
        
        is_literal, literal_score = is_metaphorical(literal_text)
        is_metaphor, metaphor_score = is_metaphorical(metaphorical_text)
        
        # Literal text should have low metaphoricality
        self.assertLess(literal_score, 0.5)
        
        # Metaphorical text should have higher metaphoricality
        self.assertGreater(metaphor_score, literal_score)

class TestAwarenessDynamics(unittest.TestCase):
    """Tests for the awareness dynamics functionality."""
    
    def setUp(self):
        self.awareness = AwarenessDynamics()
    
    def test_update_state(self):
        """Test updating awareness state."""
        # Test with high focus conditions
        state = self.awareness.update_state(
            task_difficulty=0.6,
            current_skill=0.6,  # Perfect match = good flow
            external_inputs=[0.4, 0.5, 0.4],  # Low variance
            processing_load=0.3  # Low load
        )
        
        # Should be in flow or focusing state
        self.assertIn(state, [AwarenessState.FLOW, AwarenessState.FOCUSING])
        
        # Test with scattered conditions
        state = self.awareness.update_state(
            task_difficulty=0.9,
            current_skill=0.3,  # Mismatch = poor flow
            external_inputs=[0.2, 0.8, 0.3, 0.9],  # High variance
            processing_load=0.8  # High load
        )
        
        # Should be in scattered or overload state
        self.assertIn(state, [AwarenessState.SCATTERED, AwarenessState.OVERLOAD])
    
    def test_calculate_internal_noise(self):
        """Test calculation of internal noise."""
        # Uniform values = low noise
        low_noise = self.awareness.calculate_internal_noise([0.5, 0.5, 0.5, 0.5])
        
        # Varied values = high noise
        high_noise = self.awareness.calculate_internal_noise([0.1, 0.9, 0.2, 0.8])
        
        # Check that varied values produce higher noise
        self.assertGreater(high_noise, low_noise)
    
    def test_detect_flow_state(self):
        """Test detection of flow state."""
        # Perfect match and good attention = flow
        in_flow, metrics = self.awareness.detect_flow_state(
            task_difficulty=0.7,
            skill_level=0.7,
            attention_span=10.0
        )
        
        # Should report flow with high score
        self.assertGreater(metrics["flow_score"], 0.6)
        
        # Mismatch = not flow
        in_flow, metrics = self.awareness.detect_flow_state(
            task_difficulty=0.9,
            skill_level=0.3,
            attention_span=2.0
        )
        
        # Should report not flow with low score
        self.assertLess(metrics["flow_score"], 0.5)
        self.assertFalse(in_flow)

class TestPerceptualMapping(unittest.TestCase):
    """Tests for the perceptual mapping functionality."""
    
    def test_cross_modal_mapping(self):
        """Test mapping between perceptual modalities."""
        # Map from visual brightness to auditory volume
        auditory_feature = cross_modal_mapping("visual", "bright", "auditory")
        
        # Bright should map to high or loud in auditory domain
        self.assertIsNotNone(auditory_feature)
    
    def test_simulate_embodied_experience(self):
        """Test generation of embodied experiences."""
        profile = simulate_embodied_experience("The bright, warm sunshine filled the room.")
        
        # Should detect visual and thermal qualities
        self.assertIn("brightness", profile.visual)
        self.assertIn("temperature", profile.tactile)
        
        # Description should be generated
        self.assertIsNotNone(profile.description)
    
    def test_sensory_intensity_projection(self):
        """Test projecting concepts to sensory intensities."""
        # Test emotional concept
        anger_projections = sensory_intensity_projection("anger")
        
        # Anger should have high values for temperature, arousal
        self.assertGreater(anger_projections["tactile"]["temperature"], 0.7)
        self.assertGreater(anger_projections["emotional"]["arousal"], 0.7)
        
        # Test cognitive concept
        intelligence_projections = sensory_intensity_projection("intelligence")
        
        # Intelligence should have high values for clarity
        self.assertGreater(intelligence_projections["visual"]["clarity"], 0.7)

class TestNarrativeConsciousness(unittest.TestCase):
    """Tests for the narrative consciousness functionality."""
    
    def test_build_internal_narrative(self):
        """Test building narratives from decision traces."""
        # Create a simple decision trace
        trace = {
            "decision": {
                "decision_id": "test_decision",
                "explanation": "to use approach A instead of B",
                "chain_steps": ["step1", "step2"]
            },
            "steps": {
                "step1": {
                    "reasoning": "A is faster than B",
                    "confidence": 0.7
                },
                "step2": {
                    "reasoning": "A requires less resources than B",
                    "confidence": 0.8
                }
            }
        }
        
        # Build narrative
        narrative = build_internal_narrative(trace, style="analytical")
        
        # Narrative should include reasoning from steps
        self.assertIn("faster", narrative["text"])
        self.assertIn("resources", narrative["text"])
        
        # Test different style
        narrative2 = build_internal_narrative(trace, style="introspective")
        
        # Should produce different narratives
        self.assertNotEqual(narrative["text"], narrative2["text"])
    
    def test_annotate_reasoning_with_qualia(self):
        """Test annotation of thoughts with qualia."""
        # Test confident reasoning
        confident_thought = {
            "reasoning": "This is definitely the correct approach. It's clearly the best option.",
            "confidence": 0.9
        }
        
        annotation = annotate_reasoning_with_qualia("thought1", "step", confident_thought)
        
        # Should detect confident qualia
        self.assertIn("confident", annotation["top_qualia"])
        
        # Test uncertain reasoning
        uncertain_thought = {
            "reasoning": "I'm not sure about this. Maybe it could work, but I'm confused about some aspects.",
            "confidence": 0.3
        }
        
        annotation = annotate_reasoning_with_qualia("thought2", "step", uncertain_thought)
        
        # Should detect doubtful and confused qualia
        self.assertIn("doubtful", annotation["top_qualia"])
        self.assertIn("confused", annotation["top_qualia"])
    
    def test_simulate_reflective_echo(self):
        """Test generation of reflective echoes."""
        # Test basic reflection
        reflection = simulate_reflective_echo("This approach seems promising", depth=1)
        
        # Should produce at least one reflection
        self.assertGreaterEqual(len(reflection), 1)
        
        # Test deeper reflection
        deep_reflection = simulate_reflective_echo(
            "This approach seems promising", 
            depth=3, 
            self_awareness=0.9
        )
        
        # Should produce multiple reflections
        self.assertEqual(len(deep_reflection), 3)
    
    def test_generate_reflective_questions(self):
        """Test generation of reflective questions."""
        questions = generate_reflective_questions(
            "The model shows increasing performance over time",
            curiosity_level=0.7
        )
        
        # Should generate multiple questions
        self.assertGreaterEqual(len(questions), 3)
        
        # Each item should be a question
        for q in questions:
            self.assertIn("?", q)

if __name__ == "__main__":
    unittest.main()
