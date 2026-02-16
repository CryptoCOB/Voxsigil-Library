import logging
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from nebula.core.memory_router import EchoMemory, RAG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaConsciousness:
    def __init__(self, config):
        self.config = config
        self.meta_state = {"awareness": 0.5, "regulation": 0.5}
        self.performance_history = []
        self.regulation_history = []
        self.model = LinearRegression()
        
        # Add default values for output_dim and input_dim
        output_dim = config.get('output_dim', 10)
        input_dim = config.get('input_dim', 10)
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim))

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.echo_memory = EchoMemory()
        self.rag = RAG()

    def ingest_cat_results(self, results: Dict[str, Any]):
        """Update meta state using outputs from the CATEngine."""
        if not isinstance(results, dict):
            return
        if "awareness" in results:
            self.meta_state["awareness"] = float(results["awareness"])
        if "regulation" in results:
            self.meta_state["regulation"] = float(results["regulation"])
        self.logger.info("MetaConsciousness ingested CAT results: %s", results)

    def to(self, device):
        self.weights = self.weights.to(device)
        return self
        
    async def handle_error(self, e):
        # Define error handling logic
        self.logger.error(f"Error in MetaConsciousness: {e}")

    def monitor(self, performance: np.ndarray):
        awareness = np.mean(performance)
        self.meta_state["awareness"] = awareness
        self.performance_history.append(awareness)
        self.logger.info(f"Performance monitored. Awareness: {awareness}")
        self.predict_regulation()

    def regulate(self):
        if self.meta_state["awareness"] < 0.4:
            self.meta_state["regulation"] = min(self.meta_state["regulation"] + 0.1, 1.0)
            self.logger.info("Regulation increased due to low awareness.")
        elif self.meta_state["awareness"] > 0.6:
            self.meta_state["regulation"] = max(self.meta_state["regulation"] - 0.1, 0.0)
            self.logger.info("Regulation decreased due to high awareness.")
        self.regulation_history.append(self.meta_state["regulation"])

    def evaluate(self) -> dict:
        self.logger.info(f"Meta state evaluated. Awareness: {self.meta_state['awareness']}, Regulation: {self.meta_state['regulation']}")
        return self.meta_state

    def predict_regulation(self):
        if len(self.performance_history) > 10:
            X = np.array(range(len(self.performance_history))).reshape(-1, 1)
            y = np.array(self.performance_history)
            self.model.fit(X, y)
            predicted_awareness = self.model.predict([[len(self.performance_history) + 1]])[0]
            self.logger.info(f"Predicted awareness: {predicted_awareness}")
            if predicted_awareness < 0.4:
                self.meta_state["regulation"] = min(self.meta_state["regulation"] + 0.05, 1.0)
                self.logger.info("Regulation predicted to increase.")
            elif predicted_awareness > 0.6:
                self.meta_state["regulation"] = max(self.meta_state["regulation"] - 0.05, 0.0)
                self.logger.info("Regulation predicted to decrease.")

    def visualize_meta_state(self):
        if not self.performance_history:
            self.logger.warning("No performance history to visualize.")
            return
            
        smoothed_awareness = gaussian_filter1d(self.performance_history, sigma=2)
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance_history, label="Awareness")
        plt.plot(smoothed_awareness, label="Smoothed Awareness", linestyle='--')
        plt.axhline(y=0.4, color='r', linestyle='--', label="Low Awareness Threshold")
        plt.axhline(y=0.6, color='g', linestyle='--', label="High Awareness Threshold")
        plt.title("Meta State Awareness Over Time")
        plt.xlabel("Time")
        plt.ylabel("Awareness")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_regulation(self):
        if not self.regulation_history:
            self.logger.warning("No regulation history to visualize.")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.regulation_history, label="Regulation", color='orange')
        plt.title("Regulation Over Time")
        plt.xlabel("Time")
        plt.ylabel("Regulation")
        plt.legend()
        plt.grid(True)
        plt.show()

    def log_epoch_summary(self, epoch: int):
        self.logger.info(f"End of epoch {epoch + 1}:")
        self.logger.info(f"Current meta state: {self.meta_state}")

    def adaptive_regulation(self, performance: np.ndarray):
        self.monitor(performance)
        self.regulate()
        self.log_epoch_summary(len(self.performance_history) - 1)

    def integrate_external_data(self, external_data: dict):
        for key, value in external_data.items():
            if key in self.meta_state:
                self.meta_state[key] = value
        self.logger.info(f"External data integrated: {external_data}")
        
    def state_dict(self):
        return {
            'meta_state': self.meta_state,
            'performance_history': self.performance_history,
            'regulation_history': self.regulation_history,
            'model_state': self.model.get_params(),
            'weights': self.weights.detach().cpu().numpy()
        }
    
    def load_state_dict(self, state):
        self.meta_state = state['meta_state']
        self.performance_history = state['performance_history']
        self.regulation_history = state['regulation_history']
        self.model.set_params(**state['model_state'])
        self.weights = nn.Parameter(torch.tensor(state['weights']))

    def generate_final_output(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Generate the final output based on the current meta-state, model weights, and input data.

        Args:
            input_data (torch.Tensor): Input data that the meta-model processes.

        Returns:
            torch.Tensor: The generated output.
        """
        try:
            # Adjust the input based on the meta-state awareness and regulation levels
            adjusted_input = input_data * self.meta_state["awareness"] * (1 + self.meta_state["regulation"])
            
            # Compute the output by applying the model weights
            output = torch.matmul(self.weights, adjusted_input)
            
            # Apply activation if needed (e.g., ReLU for non-linearity)
            final_output = torch.relu(output)
            
            self.logger.info(f"Generated final output with awareness: {self.meta_state['awareness']} and regulation: {self.meta_state['regulation']}")
            return final_output

        except Exception as e:
            self.logger.error(f"Error in generate_final_output: {e}")
            return torch.zeros_like(input_data)  # Return a zero tensor in case of error

    async def integrate(self, *args) -> torch.Tensor:
        """
        Integrate various cognitive inputs to produce a unified meta-cognitive output.

        Args:
            *args: Variable length argument list containing tensors for:
                - refined_meta_output (torch.Tensor)
                - art_categories (Dict[str, Any])
                - quantum_features (torch.Tensor)
                - tot_analysis (torch.Tensor)
                - reasoning_output (torch.Tensor)
                - neuro_symbolic_output (torch.Tensor)

        Returns:
            torch.Tensor: The integrated meta-cognitive output.
        """
        try:
            # Step 1: Unpack the arguments
            refined_meta_output, art_categories, quantum_features, tot_analysis, reasoning_output, neuro_symbolic_output = args

            # Step 2: Combine inputs through weighted summation or concatenation
            combined_input = torch.cat([
                refined_meta_output, 
                quantum_features, 
                tot_analysis, 
                reasoning_output, 
                neuro_symbolic_output
            ], dim=-1)
            
            # Step 3: Apply a transformation (e.g., weighted sum or linear layer)
            combined_output = torch.matmul(self.weights, combined_input.T).T
            
            # Step 4: Incorporate ART categories into the final output
            art_weights = torch.tensor([art_categories.get(key, 0.0) for key in art_categories], dtype=torch.float32)
            if art_weights.shape[0] == combined_output.shape[0]:
                final_output = combined_output * art_weights.unsqueeze(-1)
            else:
                final_output = combined_output
            
            # Optional: Apply a non-linear activation
            final_output = torch.relu(final_output)

            self.logger.info("Integration of meta-cognitive inputs completed successfully.")
            return final_output

        except Exception as e:
            await self.handle_error(e)
            return torch.zeros_like(refined_meta_output)  # Return a zero tensor in case of error
    
    def run_initial_scan(self):
        """
        Run an initial diagnostic scan of the MetaConsciousness module.
        Returns basic information about the module's state and meta-awareness.
        """
        self.logger.info("Running initial scan of MetaConsciousness")
        try:
            scan_data = {
                "module_type": self.__class__.__name__,
                "awareness": self.meta_state["awareness"],
                "regulation": self.meta_state["regulation"],
                "performance_history_length": len(self.performance_history),
                "regulation_history_length": len(self.regulation_history),
                "weights_shape": list(self.weights.shape),
                "status": "operational"
            }
            return scan_data
        except Exception as e:
            self.logger.error(f"Error during initial scan: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def deep_scan(self):
        """
        Perform a more thorough scan of the MetaConsciousness module.
        Analyzes meta-awareness, regulation patterns, and weight distributions.
        """
        self.logger.info("Performing deep scan of MetaConsciousness")
        try:
            scan_data = self.run_initial_scan()
            
            # Add awareness pattern analysis
            if len(self.performance_history) > 0:
                scan_data["awareness_mean"] = np.mean(self.performance_history)
                scan_data["awareness_std"] = np.std(self.performance_history)
                scan_data["awareness_min"] = np.min(self.performance_history)
                scan_data["awareness_max"] = np.max(self.performance_history)
                scan_data["awareness_trend"] = "increasing" if len(self.performance_history) > 1 and self.performance_history[-1] > self.performance_history[0] else "decreasing"
            
            # Add regulation pattern analysis
            if len(self.regulation_history) > 0:
                scan_data["regulation_mean"] = np.mean(self.regulation_history)
                scan_data["regulation_std"] = np.std(self.regulation_history)
                scan_data["regulation_changes"] = sum(1 for i in range(1, len(self.regulation_history)) if self.regulation_history[i] != self.regulation_history[i-1])
            
            # Analyze weight distribution
            scan_data["weights_mean"] = float(self.weights.mean().item())
            scan_data["weights_std"] = float(self.weights.std().item())
            
            return scan_data
        except Exception as e:
            self.logger.error(f"Error during deep scan: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "basic_scan": self.run_initial_scan()
            }

    async def evaluate_proposals(self, nas_proposal, evo_proposal):
        """
        Evaluate the proposals from NAS and Evolutionary Optimizer.
        """
        try:
            self.logger.info("Evaluating proposals from NAS and Evolutionary Optimizer.")

            # Convert the proposals to appropriate tensors and move them to GPU if available
            device = self.weights.device
            architecture_tensor = self._convert_to_tensor(nas_proposal).to(device)
            parameters_tensor = self._convert_to_tensor(evo_proposal).to(device)

            # Evaluate architecture adaptability and performance
            architecture_score = await self._evaluate_architecture(architecture_tensor)

            # Evaluate parameter efficiency and overall compatibility
            parameter_score = await self._evaluate_parameters(parameters_tensor)

            # Final evaluation combining both architecture and parameter scores
            final_score = self._combine_scores(architecture_score, parameter_score)
            self.logger.info(f"Final evaluation score: {final_score}")

            # Decide on the best configuration
            decision = "architecture" if architecture_score > parameter_score else "parameters"
            self.logger.info(f"Selected proposal: {decision}")

            return {
                "architecture_score": architecture_score,
                "parameter_score": parameter_score,
                "final_score": final_score,
                "selected_proposal": decision
            }

        except Exception as e:
            self.logger.error(f"Error during proposal evaluation: {e}")
            raise

    def _convert_to_tensor(self, proposal):
        """Convert the proposal dictionary into a tensor format."""
        return torch.tensor([value for value in proposal.values()], dtype=torch.float32)

    async def _evaluate_architecture(self, architecture_tensor):
        """Evaluate the architecture based on its adaptability, performance, and efficiency."""
        try:
            # Use a simplified calculation for evaluation
            adaptability_score = torch.sum(architecture_tensor).item()
            performance_score = torch.mean(architecture_tensor).item()
            architecture_score = (adaptability_score * 0.6) + (performance_score * 0.4)
            return architecture_score
        except Exception as e:
            self.logger.error(f"Error during architecture evaluation: {e}")
            return 0.0

    async def _evaluate_parameters(self, parameters_tensor):
        """Evaluate the proposed parameters based on efficiency and compatibility."""
        try:
            efficiency_score = torch.min(parameters_tensor).item()
            compatibility_score = torch.std(parameters_tensor).item()
            parameter_score = (efficiency_score * 0.5) + (compatibility_score * 0.5)
            return parameter_score
        except Exception as e:
            self.logger.error(f"Error during parameter evaluation: {e}")
            return 0.0

    def _combine_scores(self, architecture_score, parameter_score):
        """Combine the architecture and parameter scores to make a final decision."""
        return (architecture_score + parameter_score) / 2.0
    
    def visualize_all(self):
        """Encapsulate all visualization methods into one call."""
        try:
            self.visualize_meta_state()
            self.visualize_regulation()
            self.logger.info("All visualizations completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during visualizations: {e}")

if __name__ == "__main__":
    # Simple test of MetaConsciousness
    config = {
        'output_dim': 10,
        'input_dim': 5
    }
    mc = MetaConsciousness(config)
    print(f"Initial meta state: {mc.meta_state}")
    
    # Test monitoring and regulation
    performance = np.array([0.3, 0.35, 0.4])  # Low performance
    mc.monitor(performance)
    mc.regulate()
    print(f"After low performance - Awareness: {mc.meta_state['awareness']}, Regulation: {mc.meta_state['regulation']}")
    
    # Test with higher performance
    performance = np.array([0.7, 0.75, 0.8])  # High performance
    mc.monitor(performance)
    mc.regulate()
    print(f"After high performance - Awareness: {mc.meta_state['awareness']}, Regulation: {mc.meta_state['regulation']}")
    
    # Test generating output
    test_input = torch.randn(5)  # Input matches the input_dim
    output = mc.generate_final_output(test_input)
    print(f"Output shape: {output.shape}")
