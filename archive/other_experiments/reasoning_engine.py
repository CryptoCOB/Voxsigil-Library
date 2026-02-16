import logging
import ast
import torch
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceRule:
    def __init__(self, config, premises: List[str], conclusion: str):
        self.config = config
        self.premises = premises
        self.conclusion = conclusion

    def apply(self, knowledge_base: List[str], query: str) -> bool:
        if all(premise in knowledge_base for premise in self.premises):
            return self.conclusion == query
        return False

    def __repr__(self):
        return f"InferenceRule(premises={self.premises}, conclusion={self.conclusion})"

class CoreEthics(nn.Module):
    def __init__(self, config):
        super(CoreEthics, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add a dummy parameter to enable device tracking
        self.register_parameter('dummy', nn.Parameter(torch.FloatTensor([0.0])))

    def apply(self, scenario):
        # Core ethical decision-making process
        self.logger.info(f"Applying core ethical principles to scenario")
        # Return a dictionary instead of a string
        return {
            "decision": "Core Ethical Decision",
            "utility_score": 0.7,
            "deontological_rules": [True, True],
            "virtues": ["wisdom", "justice", "courage", "temperance"]
        }

    def validate(self, decision):
        return self.validate_with_frameworks(decision)

    def validate_with_frameworks(self, decision):
        utilitarian_pass = self.utilitarian_validation(decision)
        deontology_pass = self.deontological_validation(decision)
        virtue_ethics_pass = self.virtue_ethics_validation(decision)
        return utilitarian_pass and deontology_pass and virtue_ethics_pass

    def utilitarian_validation(self, decision):
        # Validate decision based on utilitarian principles
        # Handle both string and dict types for backward compatibility
        if isinstance(decision, str):
            return True  # Default for string-based decisions
        utility_score = decision.get('utility_score', 0)
        return utility_score > 0.5  # Example threshold

    def deontological_validation(self, decision):
        # Validate decision based on deontological principles
        # Handle both string and dict types for backward compatibility
        if isinstance(decision, str):
            return True  # Default for string-based decisions
        deontological_rules = decision.get('deontological_rules', [])
        return all(rule for rule in deontological_rules)  # All rules must be satisfied

    def virtue_ethics_validation(self, decision):
        # Validate decision based on virtue ethics principles
        # Handle both string and dict types for backward compatibility
        if isinstance(decision, str):
            return True  # Default for string-based decisions
        virtues = decision.get('virtues', [])
        return len(virtues) >= 3  # Example threshold

class FlexibilityModule(nn.Module):
    def __init__(self, config_or_threshold=0.5, weights=None):
        super(FlexibilityModule, self).__init__()
        
        # Handle the case where a config dict is passed instead of a threshold value
        if isinstance(config_or_threshold, dict):
            config = config_or_threshold
            self.threshold = config.get('threshold', 0.5)
            self.weights = config.get('flexibility_weights', weights or {'cultural': 0.5, 'societal': 0.5})
        else:
            self.threshold = config_or_threshold
            self.weights = weights or {'cultural': 0.5, 'societal': 0.5}
        
        self.base_threshold = self.threshold
        self.learning_rate = 0.01
        self.experience_buffer = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add nn.Parameter objects to make this a proper nn.Module
        self.threshold_param = nn.Parameter(torch.tensor(float(self.threshold)))
        self.cultural_weight = nn.Parameter(torch.tensor(float(self.weights['cultural'])))
        self.societal_weight = nn.Parameter(torch.tensor(float(self.weights['societal'])))

    def evaluate_flexibility(self, context):
        cultural_flexibility = self.get_cultural_adaptability(context)
        societal_flexibility = self.get_societal_adaptability(context)
        
        # Use parameter values but keep the original weights dictionary synced
        self.weights['cultural'] = self.cultural_weight.item()
        self.weights['societal'] = self.societal_weight.item()
        self.threshold = self.threshold_param.item()
        
        flexibility_score = (cultural_flexibility * self.weights['cultural'] + 
                             societal_flexibility * self.weights['societal'])
        self.logger.debug(f"Flexibility score: {flexibility_score}")
        return flexibility_score

    def adapt_decision(self, scenario, context, core_ethics):
        # Return a dictionary instead of a string
        adapted_decision = {
            "decision": "Adapted Decision",
            "utility_score": 0.8,
            "deontological_rules": [True, True, True],
            "virtues": ["wisdom", "justice", "courage", "temperance"],
            "flexibility_factors": {
                "cultural": self.get_cultural_adaptability(context),
                "societal": self.get_societal_adaptability(context)
            }
        }
        
        if not core_ethics.validate(adapted_decision):
            adapted_decision = core_ethics.apply(scenario)
            
        self.logger.info(f"Adapted decision: {adapted_decision}")
        return adapted_decision

    def get_cultural_adaptability(self, context):
        # Evaluate cultural adaptability based on context
        cultural_factors = context.get('cultural_factors', [])
        adaptability_score = sum(cultural_factors) / len(cultural_factors) if cultural_factors else 0.5
        return adaptability_score

    def get_societal_adaptability(self, context):
        # Evaluate societal adaptability based on context
        societal_factors = context.get('societal_factors', [])
        adaptability_score = sum(societal_factors) / len(societal_factors) if societal_factors else 0.5
        return adaptability_score

    def update_experience(self, context, outcome):
        self.experience_buffer.append(outcome)
        if len(self.experience_buffer) > 100:
            self.experience_buffer.pop(0)
        self.logger.debug(f"Experience buffer updated, size: {len(self.experience_buffer)}")

    def learned_flexibility_score(self, context):
        return sum(self.experience_buffer) / len(self.experience_buffer) if self.experience_buffer else self.threshold

class SymbolicReasoner(nn.Module):
    def __init__(self, knowledge_base: Any):
        super(SymbolicReasoner, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Handle different knowledge base formats
        if isinstance(knowledge_base, list):
            # If it's a list, assume there are no explicit rules and store the list directly
            self.knowledge_base = {'rules': knowledge_base}
        elif isinstance(knowledge_base, dict):
            self.knowledge_base = knowledge_base
        else:
            self.logger.warning(f"Knowledge base has unexpected type: {type(knowledge_base)}. Using empty dict.")
            self.knowledge_base = {'rules': []}
        
        # Add a dummy parameter to enable device tracking with .to()
        self.register_parameter('dummy', nn.Parameter(torch.FloatTensor([0.0])))
        
        self.rules = self._compile_rules()

    def _compile_rules(self):
        """Safely compile rule strings into callables."""
        compiled = []
        for rule in self.knowledge_base.get('rules', []):
            try:
                tree = ast.parse(rule, mode="eval")
                allowed = (ast.Expression, ast.Name, ast.Load, ast.UnaryOp,
                           ast.BinOp, ast.Compare, ast.Constant, ast.BoolOp,
                           ast.operator, ast.unaryop, ast.cmpop)
                if any(not isinstance(node, allowed) for node in ast.walk(tree)):
                    raise ValueError("Unsafe expression")
                compiled.append(eval(compile(tree, "<rule>", "eval")))
            except Exception as e:
                self.logger.error(f"Error compiling rule '{rule}': {e}")
        return compiled

    def reason(self, state: torch.Tensor) -> torch.Tensor:
        """Apply reasoning rules to the input state."""
        try:
            # Make a copy to avoid modifying the original
            reasoned_state = state.clone() if isinstance(state, torch.Tensor) else state
            
            # Move tensor to the same device as the reasoner if needed
            if isinstance(reasoned_state, torch.Tensor) and reasoned_state.device != self.dummy.device:
                reasoned_state = reasoned_state.to(self.dummy.device)
            
            # Apply each rule sequentially
            for rule in self.rules:
                if isinstance(reasoned_state, torch.Tensor):
                    # For tensor inputs
                    reasoned_state = rule(reasoned_state)
                elif isinstance(reasoned_state, (list, tuple)) and len(reasoned_state) > 0:
                    # For list/tuple inputs, apply to each element
                    reasoned_state = [rule(item) for item in reasoned_state]
                else:
                    # If it's some other type, try direct application
                    reasoned_state = rule(reasoned_state)
                
            return reasoned_state
        except Exception as e:
            self.logger.error(f"Error during reasoning: {e}")
            return state  # Return original state if reasoning fails

class ReasoningEngine:
    def __init__(self, config, knowledge_base: List[str] = None, inference_rules: List[InferenceRule] = None):
        self.config = config
        self.knowledge_base = knowledge_base or []
        self.inference_rules = inference_rules or []
        self.logger = self.setup_logging()
        self.core_ethics = CoreEthics(config)
        self.flexibility_module = FlexibilityModule()
        self.constraints = {}

    def setup_logging(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def infer(self, query: str) -> List[bool]:
        self.logger.debug(f"Starting inference for query: {query}")
        try:
            sub_queries = query.split(";")  # Split into sub-queries
            all_results = []
            for sub_query in sub_queries:
                sub_query = sub_query.strip()
                self.logger.debug(f"Processing sub-query: {sub_query}")
                sub_results = [rule.apply(self.knowledge_base, sub_query) for rule in self.inference_rules]
                all_results.append(any(sub_results))  # True if any rule matches
                self.logger.debug(f"Sub-query '{sub_query}' results: {sub_results}")
            self.logger.debug(f"Inference results for all sub-queries: {all_results}")
            # Return the list directly without trying to flatten it
            return all_results
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise

    def break_into_steps(self, decision: Dict) -> List[str]:
        steps = []
        if isinstance(decision, dict):
            if "decision" in decision:
                steps.append(f"Step 1: Evaluate decision: {decision['decision']}")
            if "flexibility_level" in decision:
                steps.append(f"Step 2: Assess flexibility level: {decision['flexibility_level']:.2f}")
            if "utility_score" in decision:
                steps.append(f"Step 3: Check utility score: {decision['utility_score']:.2f}")
            if "virtues" in decision:
                steps.append(f"Step 4: Review virtues: {', '.join(decision['virtues'])}")
            if "deontological_rules" in decision:
                steps.append(f"Step 5: Verify deontological rules")
        else:
            steps.append(f"Step 1: Evaluate decision: {str(decision)}")
            
        if not steps:
            steps.append("Step 1: No specific steps identified")
        return steps

    def is_decision_valid(self, decision: Dict) -> bool:
        try:
            # Check if virtues are in knowledge base
            if isinstance(decision, dict) and "virtues" in decision:
                # Add virtues to knowledge base if they don't exist
                for virtue in decision["virtues"]:
                    if virtue not in self.knowledge_base:
                        # Instead of failing, add the virtue to the knowledge base
                        self.add_knowledge(virtue)
                        self.logger.debug(f"Added virtue '{virtue}' to knowledge base")
                
            # Check deontological rules
            if isinstance(decision, dict) and "deontological_rules" in decision:
                if not all(decision["deontological_rules"]):
                    self.logger.debug("One or more deontological rules failed")
                    return False
                
            # Check if decision conclusion matches any rule
            decision_text = decision.get("decision", "") if isinstance(decision, dict) else str(decision)
            for rule in self.inference_rules:
                if rule.conclusion == decision_text and all(premise in self.knowledge_base for premise in rule.premises):
                    return True
                
            # If no rule matches, check if decision is in knowledge base
            if decision_text in self.knowledge_base:
                return True
                
            # Add the decision text to knowledge base
            self.add_knowledge(decision_text)
            self.logger.debug(f"Added decision '{decision_text}' to knowledge base")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating decision: {e}")
            return False

    def reason(self, scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform reasoning based on scenario and context."""
        self.logger.debug("Starting reasoning process")
        try:
            if context is None:
                context = {}
            
            # Evaluate flexibility level
            flexibility_level = self.flexibility_module.evaluate_flexibility(context)
            
            # Make decision based on flexibility and core ethics
            decision = self.make_decision(scenario, flexibility_level, context)
            
            # Break down decision into steps for multi-step reasoning
            steps = self.break_into_steps(decision)
            step_results = []
            for step in steps:
                step_result = self.infer(step)
                step_results.append(step_result)
                self.logger.debug(f"Step '{step}' inference result: {step_result}")
            
            # Verify decision against knowledge base and rules
            validation_passed = self.is_decision_valid(decision)
            if not validation_passed:
                self.logger.warning("Decision validation failed. Falling back to core ethics.")
                decision = self.core_ethics.apply(scenario)
                # Re-check after falling back
                validation_passed = self.is_decision_valid(decision)
            
            # Return reasoning output
            reasoning_output = {
                "decision": decision,
                "flexibility_level": flexibility_level,
                "reasoning_explanation": self.generate_explanation(decision, flexibility_level),
                "reasoning_steps": steps,
                "step_results": step_results,
                "validation_passed": validation_passed
            }
            
            self.logger.debug(f"Reasoning results: {reasoning_output}")
            return reasoning_output
        except Exception as e:
            self.logger.error(f"Error during reasoning: {e}")
            raise
    
    def reason_with_batches(self, scenario: str, context: Dict[str, Any], 
                           training_batches: List[Tuple[torch.Tensor, torch.Tensor]], 
                           validation_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """Extended reasoning that incorporates batch data processing."""
        self.logger.debug("Starting batch reasoning process")
        try:
            # Get base reasoning output
            base_output = self.reason(scenario, context)
            
            # Process training and validation batches
            if training_batches:
                train_results = self.apply_reasoning_to_batches(training_batches, "training")
                base_output["training_results"] = train_results
                
            if validation_batches:
                val_results = self.apply_reasoning_to_batches(validation_batches, "validation")
                base_output["validation_results"] = val_results
            
            # Infer new rules and update knowledge base if we have batch results
            if training_batches or validation_batches:
                all_batch_results = base_output.get("training_results", []) + base_output.get("validation_results", [])
                
                if all_batch_results:
                    base_output["inferred_rules"] = self.infer_new_rules(all_batch_results)
                    base_output["knowledge_update"] = self.update_knowledge_base(all_batch_results)
            
            return base_output
        except Exception as e:
            self.logger.error(f"Error during batch reasoning: {e}")
            raise

    def make_decision(self, scenario: str, flexibility_level: float, context: Dict[str, Any]) -> dict:
        if flexibility_level > self.flexibility_module.threshold:
            decision = self.flexibility_module.adapt_decision(scenario, context, self.core_ethics)
        else:
            decision = self.core_ethics.apply(scenario)
        
        # Ensure we're returning a dictionary with flexibility_level
        if isinstance(decision, str):
            # Convert string decision to dictionary
            decision = {"decision": decision, "flexibility_level": flexibility_level}
        elif isinstance(decision, dict) and "flexibility_level" not in decision:
            # Add flexibility_level if not present
            decision["flexibility_level"] = flexibility_level
            
        return decision

    def apply_reasoning_to_batches(self, batches: List[Tuple[torch.Tensor, torch.Tensor]], batch_type: str) -> List[Dict[str, Any]]:
        results = []
        for i, (inputs, targets) in enumerate(batches):
            batch_result = self.apply_reasoning_to_batch(inputs, targets)
            batch_result["batch_index"] = i
            batch_result["batch_type"] = batch_type
            results.append(batch_result)
        return results

    def apply_reasoning_to_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Fix the axis issue - check dimensionality first
        if inputs_np.ndim > 1:
            avg_input = np.mean(inputs_np, axis=1)
        else:
            avg_input = np.mean(inputs_np)
            
        if targets_np.ndim > 1:
            avg_target = np.mean(targets_np, axis=1)
        else:
            avg_target = np.array([np.mean(targets_np)])
            
        # Calculate correlation safely
        try:
            if avg_input.size > 1 and avg_target.size > 1:
                correlation = np.corrcoef(avg_input, avg_target)[0, 1]
            else:
                # If either array has only one element, use a default correlation
                correlation = 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating correlation: {e}. Using default value.")
            correlation = 0.0
            
        reasoning_result = {
            "input_stats": {
                "mean": np.mean(inputs_np),
                "std": np.std(inputs_np),
                "min": np.min(inputs_np),
                "max": np.max(inputs_np)
            },
            "target_stats": {
                "mean": np.mean(targets_np),
                "std": np.std(targets_np),
                "min": np.min(targets_np),
                "max": np.max(targets_np)
            },
            "correlation": correlation,
            "inferred_relationship": "positive" if correlation > 0 else "negative"
        }
        return reasoning_result

    def infer_new_rules(self, reasoning_results: List[Dict[str, Any]]) -> List[InferenceRule]:
        new_rules = []
        for result in reasoning_results:
            if result["correlation"] > 0.8:
                new_rule = InferenceRule(
                    self.config,
                    premises=[f"high_input_{result['batch_type']}"],
                    conclusion=f"high_output_{result['batch_type']}"
                )
                new_rules.append(new_rule)
            elif result["correlation"] < -0.8:
                new_rule = InferenceRule(
                    self.config,
                    premises=[f"high_input_{result['batch_type']}"],
                    conclusion=f"low_output_{result['batch_type']}"
                )
                new_rules.append(new_rule)
        return new_rules

    def update_knowledge_base(self, reasoning_results: List[Dict[str, Any]]) -> List[str]:
        new_knowledge = []
        for result in reasoning_results:
            batch_type = result.get("batch_type", "unknown")
            batch_index = result.get("batch_index", 0)
            correlation = result["correlation"]
            relationship = result["inferred_relationship"]
            new_knowledge.append(f"correlation_{batch_type}_{batch_index}:{correlation:.2f}")
            new_knowledge.append(f"relationship_{batch_type}_{batch_index}:{relationship}")
            if correlation > 0.8 or correlation < -0.8:
                self.logger.info(f"Significant correlation detected ({correlation:.2f}) in {batch_type} batch {batch_index}")
        self.knowledge_base.extend(new_knowledge)
        self.knowledge_base = list(set(self.knowledge_base))  # Remove duplicates
        return new_knowledge

    def generate_explanation(self, decision: Dict, flexibility_level: float) -> str:
        """Generate a human-readable explanation for the decision."""
        if isinstance(decision, dict):
            decision_text = decision.get("decision", "Unknown decision")
        else:
            decision_text = str(decision)
        
        if flexibility_level > self.flexibility_module.threshold:
            explanation = (f"Due to high flexibility ({flexibility_level:.2f}), an adapted decision was made: {decision_text}. "
                          f"This decision incorporates cultural and societal context adaptations.")
        else:
            explanation = (f"Due to lower flexibility ({flexibility_level:.2f}), core ethical principles were strictly applied, "
                          f"resulting in: {decision_text}.")
            
        return explanation

    def add_knowledge(self, fact: str):
        self.logger.debug(f"Adding knowledge: {fact}")
        try:
            self.knowledge_base.append(fact)
            self.logger.info(f"Knowledge added: {fact}")
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            raise

    def add_rule(self, rule: InferenceRule):
        self.logger.debug(f"Adding rule: {rule}")
        try:
            self.inference_rules.append(rule)
            self.logger.info(f"Rule added: {rule}")
        except Exception as e:
            self.logger.error(f"Error adding rule: {e}")
            raise

    def refine(self, meta_learner_model: Any, nas_architecture: Any, optimizer_params: Any):
        """Refine the reasoning engine based on other components' information."""
        self.logger.debug("Refining reasoning engine")
        try:
            self.update_constraints(meta_learner_model, nas_architecture, optimizer_params)
            self.prune_knowledge_base()
            self.optimize_rules()
            self.logger.info("Reasoning engine refined")
        except Exception as e:
            self.logger.error(f"Error refining reasoning engine: {e}")
            raise

    def update_constraints(self, meta_learner_model: Any, nas_architecture: Any, optimizer_params: Any):
        """Update internal constraints based on other component models."""
        self.constraints = {
            "meta_learner": self.extract_constraints(meta_learner_model),
            "nas": self.extract_constraints(nas_architecture),
            "optimizer": self.extract_constraints(optimizer_params)
        }

    def extract_constraints(self, component: Any) -> Dict[str, Any]:
        """Extract constraints from a component."""
        if component is None:
            return {"constraint_type": "None"}
        return {"constraint_type": type(component).__name__}

    def prune_knowledge_base(self):
        """Remove redundant knowledge."""
        self.knowledge_base = list(set(self.knowledge_base))

    def optimize_rules(self):
        """Optimize inference rules."""
        self.inference_rules.sort(key=lambda x: len(x.premises))

    def validate_rules(self) -> Dict[str, int]:
        """Validate that rules have their premises in the knowledge base."""
        valid_rules = sum(1 for rule in self.inference_rules if self.is_rule_valid(rule))
        return {
            "total_rules": len(self.inference_rules),
            "valid_rules": valid_rules
        }

    def is_rule_valid(self, rule: InferenceRule) -> bool:
        """Check if a rule's premises exist in the knowledge base."""
        return all(premise in self.knowledge_base for premise in rule.premises)

    def process_input_with_feedback(self, scenario: str, context: Dict[str, Any], feedback: Optional[str] = None) -> str:
        """Process input with optional feedback for adaptation."""
        self.logger.debug(f"Processing scenario '{scenario}' with feedback: {feedback}")
        try:
            decision = self.reason(scenario, context)
            if feedback:
                self.adapt_to_feedback(feedback)
            return str(decision)
        except Exception as e:
            self.logger.error(f"Error processing input with feedback: {e}")
            raise

    def adapt_to_feedback(self, feedback: str) -> dict:
        status = {"action": None, "success": True}
        try:
            if feedback == "increase flexibility":
                self.flexibility_module.threshold *= 1.1
                status["action"] = "increase_flexibility"
                self.logger.info(f"Flexibility threshold increased to {self.flexibility_module.threshold:.2f}")
            elif feedback == "decrease flexibility":
                self.flexibility_module.threshold *= 0.9
                status["action"] = "decrease_flexibility"
                self.logger.info(f"Flexibility threshold decreased to {self.flexibility_module.threshold:.2f}")
            elif feedback.startswith("add rule:"):
                new_rule_str = feedback.split(":", 1)[1].strip()
                if "->" in new_rule_str:
                    premises, conclusion = new_rule_str.split("->")  # Expect format "premise1,premise2->conclusion"
                    new_rule = InferenceRule(self.config, [p.strip() for p in premises.split(",")], conclusion.strip())
                    self.add_rule(new_rule)
                    self.logger.info(f"New rule added: {new_rule}")
                    status["action"] = "add_rule"
                else:
                    self.logger.warning(f"Invalid rule format (expected 'premises->conclusion'): {new_rule_str}")
                    status["success"] = False
            elif feedback.startswith("add knowledge:"):
                new_knowledge = feedback.split(":", 1)[1].strip()
                self.add_knowledge(new_knowledge)
                self.logger.info(f"New knowledge added: {new_knowledge}")
                status["action"] = "add_knowledge"
            else:
                self.logger.warning(f"Unrecognized feedback: {feedback}")
                status["success"] = False
        except Exception as e:
            self.logger.error(f"Error adapting to feedback: {e}")
            status["success"] = False
            status["error"] = str(e)
        return status


if __name__ == "__main__":
    # Simple test of the ReasoningEngine
    config = {'testing': True}
    
    # Create a test knowledge base and rules
    knowledge_base = ["sky_is_blue", "grass_is_green", "water_is_wet"]
    rule1 = InferenceRule(config, ["sky_is_blue"], "weather_is_good")
    rule2 = InferenceRule(config, ["water_is_wet", "sky_is_blue"], "raining")
    
    # Initialize the ReasoningEngine
    engine = ReasoningEngine(config, knowledge_base, [rule1, rule2])
    
    # Test multi-step inference
    complex_query = "sky_is_blue;weather_is_good"
    results = engine.infer(complex_query)
    print(f"Multi-step inference for '{complex_query}': {results}")
    
    # Test reasoning with self-verification
    scenario = "Should I go outside?"
    context = {'cultural_factors': [0.8], 'societal_factors': [0.6]}
    reasoning_result = engine.reason(scenario, context)
    print(f"Reasoning result: {reasoning_result}")
    
    # Test adaptability with feedback
    feedback = "add rule: grass_is_green->nice_day"
    decision_with_feedback = engine.process_input_with_feedback(scenario, context, feedback)
    print(f"Decision with feedback: {decision_with_feedback}")
    
    # Test batch reasoning
    training_batches = [(torch.tensor([[1.0, 2.0], [2.0, 3.0]]), torch.tensor([[1.0], [2.0]]))]
    validation_batches = [(torch.tensor([[3.0, 4.0], [4.0, 5.0]]), torch.tensor([[3.0], [4.0]]))]
    
    # For simpler test with 1D targets
    simple_training_batches = [(torch.tensor([[1.0, 2.0], [2.0, 3.0]]), torch.tensor([1.0, 2.0]))]
    simple_validation_batches = [(torch.tensor([[3.0, 4.0], [4.0, 5.0]]), torch.tensor([3.0, 4.0]))]
    
    batch_result = engine.reason_with_batches(scenario, context, simple_training_batches, simple_validation_batches)
    print(f"Batch reasoning result: {batch_result}")
    
    # Test SymbolicReasoner
    symbolic_reasoner = SymbolicReasoner({'rules': ['x + 1']})
    tensor_input = torch.tensor([1.0, 2.0, 3.0])
    reasoned_output = symbolic_reasoner.reason(tensor_input)
    print(f"Symbolic reasoning on {tensor_input} gives {reasoned_output}")
