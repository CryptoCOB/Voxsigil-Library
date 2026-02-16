# --- START OF REFINED holistic_perception.py ---

import random
import numpy as np
import torch
import torch.nn as nn
import logging
from collections import deque
from typing import Dict, Optional, Any, List, Union, Tuple
import torch.nn.functional as F
import asyncio
import time  # For status timestamp

# Set up logging
# logger = logging.getLogger(__name__) # Use standard __name__
# Assume logger is configured by the main application entry point
# For standalone testing, uncomment basicConfig in __main__
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Stubs for Missing Dependencies (Test Block Only) ---
try:
    # Attempt relative import first
    from .shared_util import Hidden_LSTM

    logger.info("Successfully imported Hidden_LSTM from .shared_util")
except (ImportError, ModuleNotFoundError):
    try:
        # Fallback to absolute import
        from shared_util import Hidden_LSTM

        logger.info("Successfully imported Hidden_LSTM from shared_util")
    except (ImportError, ModuleNotFoundError):
        logger.warning("shared_util.Hidden_LSTM not found. Using DUMMY for testing.")

        # Dummy LSTM for testing purposes
        class Hidden_LSTM(nn.Module):
            def __init__(self, config: Dict):
                super().__init__()
                self.logger = logging.getLogger(f"{__name__}.DummyHidden_LSTM")
                self.logger.warning("Using DUMMY Hidden_LSTM class.")
                self.input_dim = config.get("LSTM_INPUT_DIM", 1024)
                self.hidden_dim = config.get("LSTM_HIDDEN_DIM", 512)
                self.output_dim = config.get("LSTM_OUTPUT_DIM", 256)
                # Dummy layer to allow processing
                self.dummy_linear = nn.Linear(self.input_dim, self.output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Simple passthrough for the dummy
                self.logger.debug(f"Dummy LSTM processing tensor of shape: {x.shape}")
                # Assume x shape is (batch, seq, feature), take last sequence item
                last_seq_item = x[:, -1, :]
                return self.dummy_linear(last_seq_item)


# --- Utilities ---


# Encapsulated Utility-4: NaN Check Utility
def check_for_nans(tensor: Optional[torch.Tensor], name: str):
    """Logs an error if NaNs are detected in a tensor."""
    if tensor is not None and torch.isnan(tensor).any():
        logger.error(f"NaN detected in tensor '{name}'!")
        return True
    # logger.debug(f"No NaN detected in tensor '{name}'.") # Can be too verbose
    return False


# --- Core Class ---


class HolisticPerception(nn.Module):
    """
    Integrates and fuses information from multiple sensory modalities.

    Handles buffering of recent inputs and applies a configurable fusion
    method (e.g., concatenate, sum, average, attention, neural) to produce
    a unified holistic representation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Input configuration
        self.input_dims = config.get(
            "input_dims",
            {  # Use get with default dict
                "visual": 128,
                "auditory": 128,
                "tactile": 64,
                "olfactory": 64,
                "video": 256,
                "proprioceptive": 32,  # Example added
            },
        )
        self.modalities = list(self.input_dims.keys())

        # Fusion configuration
        self.fusion_method = config.get("fusion_method", "attention")
        self.intermediate_fusion_dim = config.get(
            "intermediate_fusion_dim", 256
        )  # Dim before final fusion
        self.output_dim = config.get("holistic_output_dim", 512)
        self.activation_fn_name = config.get("activation", "relu").lower()

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger.info(f"HolisticPerception initializing on device: {self.device}")

        # Input projection layers (Project each modality to intermediate dim)
        self.projection_layers = nn.ModuleDict(
            {
                modality: nn.Linear(dim, self.intermediate_fusion_dim)
                for modality, dim in self.input_dims.items()
            }
        ).to(self.device)

        # Fusion mechanism layers
        self.fusion_layer = None  # Initialize specific fusion layer later
        self._setup_fusion_layer()  # Call helper to setup fusion

        # Buffers for recent inputs per modality
        self.buffer_size = config.get("buffer_size", 10)
        if self.buffer_size <= 0:
            self.logger.warning("Buffer size is <= 0. Buffering disabled.")
        self.input_buffers = {
            modality: deque(maxlen=self.buffer_size) if self.buffer_size > 0 else []
            for modality in self.modalities
        }

        # Feature-1: Temporal Fusion LSTM (Optional)
        self.use_temporal_fusion = config.get("use_temporal_fusion", False)
        if self.use_temporal_fusion:
            lstm_input_dim = self.intermediate_fusion_dim  # LSTM input = projected modality dim
            lstm_hidden_dim = config.get("temporal_lstm_hidden_dim", 256)
            # Allow configurable LSTM output dim; default to intermediate_fusion_dim to remain compatible.
            lstm_output_dim = config.get(
            "temporal_lstm_output_dim", self.intermediate_fusion_dim
            )

            # We need one LSTM per modality to process its temporal sequence
            self.temporal_lsms = nn.ModuleDict(
            {
                modality: nn.LSTM(
                lstm_input_dim,
                lstm_hidden_dim,
                batch_first=True,
                num_layers=1,
                ).to(self.device)
                for modality in self.modalities
            }
            )

            # Linear to map LSTM hidden state to configured LSTM output dim
            self.temporal_outputs = nn.ModuleDict(
            {
                modality: nn.Linear(lstm_hidden_dim, lstm_output_dim).to(
                self.device
                )
                for modality in self.modalities
            }
            )

            # If the LSTM output dim doesn't match the expected intermediate_fusion_dim,
            # add a small projection layer to align it back so the fusion pipeline remains unchanged.
            self._temporal_align_required = lstm_output_dim != self.intermediate_fusion_dim
            if self._temporal_align_required:
                self.temporal_output_align = nn.ModuleDict(
                    {
                        modality: nn.Linear(lstm_output_dim, self.intermediate_fusion_dim).to(
                            self.device
                        )
                for modality in self.modalities
                }
            )
            else:
                self.temporal_output_align = None

            self.logger.info(
            f"Temporal fusion (LSTM per modality) enabled. LSTM hidden dim: {lstm_hidden_dim}, LSTM output dim: {lstm_output_dim}"
            )

        # Add missing modality_buffers attribute
        self.modality_buffers = {}  # Dictionary to store data for each modality

        self.logger.info(
            f"Initialized HolisticPerception. Fusion: '{self.fusion_method}'. Output Dim: {self.output_dim}. Modalities: {self.modalities}"
        )

    def _setup_fusion_layer(self):
        """Initializes the specific fusion layer based on self.fusion_method."""
        num_modalities = len(self.modalities)
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
        }
        activation_fn = activation_map.get(self.activation_fn_name, nn.ReLU())
        self.logger.info(f"Using activation: {self.activation_fn_name}")

        if self.fusion_method == "concatenate":
            # Output layer takes concatenated intermediate representations
            self.fusion_layer = nn.Sequential(
                nn.Linear(
                    self.intermediate_fusion_dim * num_modalities, self.output_dim
                ),
                activation_fn,
            ).to(self.device)
        elif self.fusion_method in ["sum", "average"]:
            # No specific layer needed here, fusion happens via torch.sum/mean
            # But add final projection + activation
            self.fusion_layer = nn.Sequential(
                nn.Linear(
                    self.intermediate_fusion_dim, self.output_dim
                ),  # Project sum/avg to output
                activation_fn,
            ).to(self.device)
        elif self.fusion_method == "attention":
            num_heads = self.config.get("num_heads", 4)
            # Ensure intermediate_dim is divisible by num_heads, adjust if necessary
            if self.intermediate_fusion_dim % num_heads != 0:
                self.intermediate_fusion_dim = (
                    self.intermediate_fusion_dim // num_heads
                ) * num_heads
                self.logger.warning(
                    f"Adjusted intermediate_fusion_dim to {self.intermediate_fusion_dim} to be divisible by num_heads ({num_heads})."
                )
                # Need to re-initialize projection layers if intermediate_fusion_dim changed
                self.projection_layers = nn.ModuleDict(
                    {
                        modality: nn.Linear(dim, self.intermediate_fusion_dim)
                        for modality, dim in self.input_dims.items()
                    }
                ).to(self.device)

            self.attention = nn.MultiheadAttention(
                embed_dim=self.intermediate_fusion_dim,
                num_heads=num_heads,
                batch_first=True,  # Expect (batch, seq, feature) -> (batch, num_modalities, intermediate_dim)
            ).to(self.device)
            # Add final layer after attention pooling
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.intermediate_fusion_dim, self.output_dim), activation_fn
            ).to(self.device)
        elif self.fusion_method == "neural":
            # Initialize a simple sequential neural model for fusion
            # Input: concatenated intermediate representations
            self.fusion_layer = nn.Sequential(
                nn.Linear(
                    self.intermediate_fusion_dim * num_modalities,
                    self.intermediate_fusion_dim * 2,
                ),
                nn.ReLU(),
                nn.Linear(self.intermediate_fusion_dim * 2, self.output_dim),
                activation_fn,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        self.logger.info(
            f"Fusion layer setup complete for method '{self.fusion_method}'."
        )

    def to(self, device: Union[str, torch.device]):
        """Moves the module and its submodules to the specified device."""
        self.device = torch.device(device)
        super().to(self.device)  # Move the main module parameters
        # Explicitly move ModuleDicts and potential standalone layers
        self.projection_layers.to(self.device)
        if self.fusion_layer:
            self.fusion_layer.to(self.device)
        if hasattr(self, "attention"):
            self.attention.to(self.device)
        if hasattr(self, "temporal_lsms"):
            self.temporal_lsms.to(self.device)
        if hasattr(self, "temporal_outputs"):
            self.temporal_outputs.to(self.device)
        self.logger.info(f"HolisticPerception moved to device: {self.device}")
        return self

    # Encapsulated Utility-1: Input Validation Helper
    def _validate_input_dict(self, inputs: Dict[str, Any]) -> bool:
        """Validates the input dictionary structure and basic tensor types."""
        if not isinstance(inputs, dict):
            self.logger.error(f"Input must be a dictionary, received {type(inputs)}.")
            return False
        # Check for at least one known modality
        if not any(key in self.input_dims for key in inputs):
            self.logger.error(
                f"Input dictionary contains no known modalities: {list(inputs.keys())}"
            )
            return False
        # Check types (basic tensor check or if numpy array)
        for key, value in inputs.items():
            if key in self.input_dims:  # Only validate known modalities
                if not isinstance(value, (torch.Tensor, np.ndarray)):
                    try:
                        # Attempt conversion to catch fundamentally non-numeric data early
                        _ = torch.tensor(value, dtype=torch.float32)
                    except (TypeError, ValueError) as e:
                        self.logger.error(
                            f"Input for modality '{key}' is not a Tensor/ndarray or convertible: type={type(value)}, error={e}"
                        )
                        return False
        return True

    # Encapsulated Utility-6: Safe Device Mover for Inputs
    def _safe_to_device(
        self, tensor: Union[torch.Tensor, np.ndarray], modality_key: str
    ) -> Optional[torch.Tensor]:
        """Converts input to float tensor and moves to the correct device."""
        try:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            # Ensure float type and move to device
            if not isinstance(
                tensor, torch.Tensor
            ):  # Check again after potential numpy conversion
                raise TypeError("Input is not a valid tensor type.")
            return tensor.float().to(self.device)
        except (TypeError, RuntimeError, ValueError) as e:
            self.logger.error(
                f"Failed to convert/move tensor for modality '{modality_key}' to device {self.device}: {e}"
            )
            return None

    def buffer_input(self, inputs: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """Adds new input data for each modality to its respective buffer."""
        if self.buffer_size <= 0:
            return  # Buffering disabled

        timestamp = time.time()
        added_count = 0
        for key, value in inputs.items():
            if key in self.input_buffers:
                # Store tuple of (timestamp, data) if needed, or just data
                self.input_buffers[key].append({"ts": timestamp, "data": value})
                added_count += 1
            else:
                self.logger.warning(
                    f"Received input for unrecognized modality '{key}'. Ignoring."
                )
        self.logger.debug(f"Buffered inputs for {added_count} modalities.")

    # Encapsulated Utility-2: Buffer Aggregation Helper
    def _get_buffered_data(
        self, aggregation: str = "latest", num_items: int = 1
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves data from buffers based on the aggregation strategy.

        Args:
            aggregation: How to aggregate buffer ('latest', 'average', 'sequence').
            num_items: Number of items to retrieve for 'latest' or 'sequence'.

        Returns:
            Dictionary of tensors for each modality, ready for fusion, or None if no data.
        """
        if self.buffer_size <= 0:
            self.logger.warning("Buffering disabled, cannot retrieve buffered data.")
            return None

        aggregated_data = {}
        min_required_items = 1 if aggregation != "sequence" else max(1, num_items)

        for modality, buffer in self.input_buffers.items():
            if len(buffer) >= min_required_items:
                # Retrieve raw data (numpy or tensor) from buffer entries
                if aggregation == "latest":
                    # Get the 'data' field from the latest `num_items` entries
                    raw_items = [
                        buffer[-i]["data"]
                        for i in range(1, min(num_items, len(buffer)) + 1)
                    ]
                    # Average the latest items
                    processed_tensor = torch.mean(
                        torch.stack(
                            [
                                self._safe_to_device(d, modality)
                                for d in raw_items
                                if d is not None
                            ]
                        ),
                        dim=0,
                    )
                elif aggregation == "average":
                    # Average all items currently in the buffer
                    raw_items = [entry["data"] for entry in buffer]
                    processed_tensor = torch.mean(
                        torch.stack(
                            [
                                self._safe_to_device(d, modality)
                                for d in raw_items
                                if d is not None
                            ]
                        ),
                        dim=0,
                    )
                elif aggregation == "sequence":
                    # Get the latest `num_items` as a sequence
                    raw_items = [
                        buffer[-i]["data"]
                        for i in range(1, min(num_items, len(buffer)) + 1)
                    ][::-1]  # Get latest N, reverse to chronological
                    # Stack tensors along a new sequence dimension (dim 0 assumed for LSTM/Transformer)
                    safe_tensors = [
                        self._safe_to_device(d, modality)
                        for d in raw_items
                        if d is not None
                    ]
                    if safe_tensors:
                        processed_tensor = torch.stack(safe_tensors, dim=0)
                    else:
                        processed_tensor = None  # No valid tensors in sequence
                else:
                    self.logger.warning(
                        f"Unknown aggregation method: {aggregation}. Defaulting to 'latest'."
                    )
                    raw_items = [buffer[-1]["data"]]  # Get latest item
                    processed_tensor = self._safe_to_device(
                        raw_items[0], modality
                    )  # Process latest item

                if processed_tensor is not None and not check_for_nans(
                    processed_tensor, f"{modality}_{aggregation}"
                ):
                    aggregated_data[modality] = processed_tensor
            # else:
            #     self.logger.debug(f"Not enough data in buffer for modality '{modality}' (required {min_required_items}, have {len(buffer)}) for aggregation '{aggregation}'.")

        if not aggregated_data:
            self.logger.warning(
                f"Could not retrieve sufficient buffered data for aggregation '{aggregation}'."
            )
            return None

        return aggregated_data

    async def get_fused_representation(self, aggregation="weighted"):
        """
        Get the fused representation of all modalities.
        """
        modality_tensors = {}

        # Process each modality's data
        for modality in self.modalities:
            data = await self._get_modality_data(modality, aggregation)
            if data is None or not isinstance(data, torch.Tensor):
                continue

            # Move tensor to device
            data_tensor = data.to(self.device)

            # Add batch dimension if needed
            if data_tensor.dim() == 1:
                data_tensor = data_tensor.unsqueeze(0)

            # Get expected input dimension for this modality's projection layer
            expected_input_dim = self.projection_layers[modality].weight.shape[1]

            # Check for shape mismatch
            if data_tensor.shape[-1] != expected_input_dim:
                self.logger.warning(
                    f"Shape mismatch for {modality}: {data_tensor.shape[-1]} vs {expected_input_dim}"
                )

                # Handle specific case for 'echo' modality (1x4 tensor -> 1x128)
                if modality == "echo":
                    # Create a properly sized zero tensor
                    reshaped_tensor = torch.zeros(
                        data_tensor.shape[0],
                        expected_input_dim,
                        device=self.device,
                        dtype=data_tensor.dtype,
                    )

                    # Copy available data (first few dimensions)
                    feat_size = min(data_tensor.shape[-1], expected_input_dim)
                    reshaped_tensor[:, :feat_size] = data_tensor[:, :feat_size]

                    # Use the reshaped tensor
                    data_tensor = reshaped_tensor
                    self.logger.info(
                        f"Reshaped {modality} tensor from {data.shape} to {data_tensor.shape}"
                    )
                else:
                    # General case: pad or truncate as needed
                    if data_tensor.shape[-1] < expected_input_dim:
                        # Pad
                        padding_size = expected_input_dim - data_tensor.shape[-1]
                        data_tensor = F.pad(data_tensor, (0, padding_size))
                    else:
                        # Truncate
                        data_tensor = data_tensor[:, :expected_input_dim]

            # Now project the correctly shaped tensor
            try:
                projected = self.projection_layers[modality](data_tensor)
                modality_tensors[modality] = projected
            except Exception as e:
                self.logger.error(f"Failed to project {modality} tensor: {e}")
                continue

        # If no modality data is available, return a zero tensor
        if not modality_tensors:
            self.logger.warning("No modality data available for fusion")
            return torch.zeros(1, self.fusion_dim, device=self.device)

        # Combine all modality representations (with proper error handling)
        try:
            # Stack all tensors and take mean along modality dimension
            stacked_tensors = torch.stack(list(modality_tensors.values()), dim=1)
            fused = torch.mean(stacked_tensors, dim=1)
            self.logger.debug(f"Fused representation: {fused.shape}")
            return fused
        except Exception as e:
            self.logger.error(f"Error during modality fusion: {e}")
            # Return first available modality tensor as fallback
            return next(iter(modality_tensors.values()))

    async def _fuse_modalities(
        self,
        processed_inputs: Dict[str, torch.Tensor],
        modality_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[torch.Tensor]:
        """Applies the configured cross-modal fusion method."""
        if not processed_inputs:
            return None

        # Feature-2: Apply Modality Weights
        input_tensors: List[torch.Tensor] = []
        if modality_weights:
            # Normalize weights to sum to 1 (approx) across available modalities
            total_weight = sum(
                modality_weights.get(mod, 0.0) for mod in processed_inputs.keys()
            )
            if total_weight <= 1e-6:
                self.logger.warning(
                    "Modality weights sum to zero or less. Applying equal weighting."
                )
                modality_weights = None  # Fallback to equal weight
            else:
                # Apply normalized weights
                for modality, tensor in processed_inputs.items():
                    weight = modality_weights.get(modality, 0.0) / total_weight
                    input_tensors.append(
                        tensor * weight * len(processed_inputs)
                    )  # Scale by weight & num modalities
                self.logger.debug(
                    f"Applied modality weights: { {k: f'{modality_weights.get(k, 0) / total_weight:.2f}' for k in processed_inputs} }"
                )

        else:  # Default: Use all inputs directly
            input_tensors = list(processed_inputs.values())

        if not input_tensors:
            self.logger.error(
                "No valid input tensors available for fusion after weighting."
            )
            return None

        # Apply fusion method
        fused = None
        num_inputs = len(input_tensors)

        try:
            if self.fusion_method == "concatenate":
                # Ensure all tensors have batch dimension if needed (assume single item fusion here)
                # Concatenate along feature dimension
                if num_inputs > 0:
                    concatenated = torch.cat(input_tensors, dim=-1)
                    if not check_for_nans(concatenated, "fusion_concat"):
                        fused = self.fusion_layer(
                            concatenated
                        )  # Pass through final projection/activation
            elif self.fusion_method in ["sum", "average"]:
                stacked = torch.stack(input_tensors, dim=0)
                fused_intermediate = (
                    torch.sum(stacked, dim=0)
                    if self.fusion_method == "sum"
                    else torch.mean(stacked, dim=0)
                )
                if not check_for_nans(
                    fused_intermediate, f"fusion_{self.fusion_method}"
                ):
                    fused = self.fusion_layer(
                        fused_intermediate
                    )  # Pass through final projection/activation
            elif self.fusion_method == "attention":
                # Input for MHA: (batch, seq_len, embed_dim) -> (1, num_modalities, intermediate_dim)
                # First ensure each tensor has batch dimension
                batched_tensors = [
                    t.unsqueeze(0) if t.dim() == 1 else t for t in input_tensors
                ]

                # Log tensor shapes for debugging
                self.logger.debug(
                    f"Attention input tensor shapes: {[t.shape for t in batched_tensors]}"
                )

                # Instead of stacking, reshape each tensor and then concatenate along sequence dimension
                # Ensure each tensor is of shape [1, 1, feature_dim]
                reshaped_tensors = []
                for t in batched_tensors:
                    if t.dim() == 1:  # [feature]
                        reshaped = t.unsqueeze(0).unsqueeze(0)  # [1, 1, feature]
                    elif t.dim() == 2:  # [batch, feature]
                        reshaped = t.unsqueeze(1)  # [batch, 1, feature]
                    else:
                        reshaped = t  # Already has 3 dimensions
                    reshaped_tensors.append(reshaped)

                # Concatenate along sequence dimension (dim=1)
                attn_input = torch.cat(
                    reshaped_tensors, dim=1
                )  # Shape: [1, num_modalities, feature_dim]

                self.logger.debug(
                    f"Attention input shape: {attn_input.shape}, expected feature dim: {self.intermediate_fusion_dim}"
                )

                if not check_for_nans(attn_input, "fusion_attn_input"):
                    attn_output, _ = self.attention(
                        attn_input, attn_input, attn_input
                    )  # Self-attention
                    # Pool the attention output (e.g., mean pool over modalities)
                    pooled_output = attn_output.mean(
                        dim=1
                    )  # Shape: [Batch=1, FeatureDim]
                    if not check_for_nans(pooled_output, "fusion_attn_pooled"):
                        fused = self.fusion_layer(
                            pooled_output
                        )  # Pass pooled output through final layer
            elif self.fusion_method == "neural":
                # Similar to concat, feed concatenated tensors to the neural fusion model
                if num_inputs > 0:
                    concatenated = torch.cat(input_tensors, dim=-1)
                    if not check_for_nans(concatenated, "fusion_neural_input"):
                        fused = self.fusion_layer(concatenated)
            else:
                self.logger.error(
                    f"Fusion logic not implemented for method: {self.fusion_method}"
                )
                return None

            if fused is not None and check_for_nans(
                fused, f"final_fused_{self.fusion_method}"
            ):
                return None  # Return None if final result has NaNs
            return fused

        except Exception as e:
            self.logger.error(
                f"Error during fusion method '{self.fusion_method}': {e}", exc_info=True
            )
            return None

    async def _get_modality_data(self, modality: str, aggregation: str = "latest"):
        """
        Get data for a specific modality with the specified aggregation strategy.

        Parameters:
            modality (str): The modality to retrieve (e.g., 'vision', 'text', 'echo')
            aggregation (str): How to aggregate multiple data points:
                - 'latest': Return only the most recent data
                - 'weighted': Return weighted average of recent data, emphasizing recent items
                - 'mean': Return simple mean of all recent data

        Returns:
            torch.Tensor or None: The modality data tensor or None if no data is available
        """
        if modality not in self.modality_buffers or not self.modality_buffers[modality]:
            self.logger.debug(f"No data available for modality: {modality}")
            return None

        buffer = self.modality_buffers[modality]

        if aggregation == "latest":
            # Return the most recent data point
            return buffer[-1]

        elif aggregation == "weighted":
            # Apply exponential weighting to emphasize recent data
            weights = torch.tensor(
                [0.7**i for i in range(len(buffer) - 1, -1, -1)], device=self.device
            )
            # Normalize weights to sum to 1
            weights = weights / weights.sum()

            # Stack tensors and apply weights along first dimension
            stacked = torch.stack(buffer)
            weighted_sum = (
                stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))
            ).sum(dim=0)
            return weighted_sum

        elif aggregation == "mean":
            # Simple average of all data in the buffer
            return torch.stack(buffer).mean(dim=0)

        else:
            self.logger.warning(
                f"Unknown aggregation method: {aggregation}, using 'latest'"
            )
            return buffer[-1]

    # --- Deprecated / Alternative Methods ---
    # integrate_inputs is replaced by buffer_input + get_fused_representation
    # async def integrate_inputs(self, inputs: Dict[str, np.ndarray]) -> np.ndarray: # Deprecated

    # weighted_fusion provides a numpy-based alternative, keep separate for now
    async def weighted_fusion(self, weights: Dict[str, float]) -> Optional[np.ndarray]:
        """Applies weighted average fusion directly on NUMPY arrays in buffers."""
        self.logger.warning(
            "weighted_fusion operates on NumPy arrays in buffers, may conflict with Tensor pipeline."
        )
        weighted_inputs = []
        total_weight = sum(
            w
            for k, w in weights.items()
            if k in self.input_buffers and len(self.input_buffers[k]) > 0
        )
        if total_weight <= 1e-6:
            self.logger.warning("Weights sum to zero in weighted_fusion.")
            return None

        for key, buffer in self.input_buffers.items():
            if len(buffer) > 0 and key in weights:
                # Assumes buffer contains numpy arrays
                try:
                    # Get only the 'data' part from buffer entries
                    buffer_data = [entry["data"] for entry in buffer]
                    mean_val = np.mean(buffer_data, axis=0)
                    weighted_inputs.append((weights[key] / total_weight) * mean_val)
                except Exception as e:
                    self.logger.error(
                        f"Error processing buffer for {key} in weighted_fusion: {e}"
                    )
                    continue  # Skip modality on error

        if not weighted_inputs:
            self.logger.warning("No valid inputs for weighted fusion.")
            return None
        # Sum the weighted means
        fused_result = np.sum(weighted_inputs, axis=0)
        return fused_result

    # --- Configuration & Model Management ---
    # Feature-3: Configurable Activation Function handled in _setup_fusion_layer

    def set_fusion_method(self, method: str):
        """Sets a new fusion method and re-initializes relevant layers."""
        if method not in ["concatenate", "sum", "average", "attention", "neural"]:
            raise ValueError(f"Unsupported fusion method: {method}")
        self.fusion_method = method
        self.logger.info(f"Fusion method changed to: '{method}'")
        # Re-initialize the fusion layers based on the new method
        self._setup_fusion_layer()

    def save_model(self, path: str):
        """Saves the entire HolisticPerception model state."""
        try:
            # Save projection layers, fusion layer, attention, LSTMs etc.
            torch.save(self.state_dict(), path)
            self.logger.info(f"HolisticPerception model saved to {path}")
        except Exception as e:
            self.logger.error(
                f"Failed to save HolisticPerception model to {path}: {e}", exc_info=True
            )

    def load_model(self, path: str):
        """Loads the entire HolisticPerception model state."""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)  # Ensure loaded model is on the correct device
            self.logger.info(f"HolisticPerception model loaded from {path}")
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {path}")
        except Exception as e:
            self.logger.error(
                f"Failed to load HolisticPerception model from {path}: {e}",
                exc_info=True,
            )

    # Encapsulated Utility-5: Get Component Status
    def get_status(self) -> Dict:
        """Returns the current status and configuration of the module."""
        buffer_status = {mod: len(buf) for mod, buf in self.input_buffers.items()}
        status = {
            "timestamp": time.time(),
            "fusion_method": self.fusion_method,
            "output_dim": self.output_dim,
            "intermediate_fusion_dim": self.intermediate_fusion_dim,
            "modalities_configured": self.modalities,
            "buffer_size": self.buffer_size,
            "buffer_status": buffer_status,
            "device": str(self.device),
            "temporal_fusion_enabled": self.use_temporal_fusion,
            "activation_function": self.activation_fn_name,
        }
        return status

    @property
    def fusion_dim(self) -> int:
        """Returns the fusion output dimension (alias for output_dim)."""
        return self.output_dim


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("--- Running HolisticPerception Example ---")

    # Example config
    config = {
        "input_dims": {  # Use actual key from constructor
            "visual": 128,
            "auditory": 64,
            "tactile": 32,
            "proprioceptive": 16,
            "video": 256,
        },
        "holistic_output_dim": 512,
        "intermediate_fusion_dim": 256,  # Add intermediate dim
        "fusion_method": "attention",  # Try attention
        "num_heads": 8,  # For attention
        "activation": "gelu",  # Try GELU
        "buffer_size": 5,
        "use_temporal_fusion": False,  # Test without temporal first
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Initialize holistic perception
    hpm = HolisticPerception(config)
    hpm.to(config["device"])  # Move to device

    # --- Simulate Buffering Inputs ---
    logger.info("\n--- Simulating Input Buffering ---")
    num_steps = 7
    for i in range(num_steps):
        inputs = {
            modality: np.random.rand(dim).astype(np.float32)
            * (i + 1)  # Simulate changing data
            for modality, dim in hpm.input_dims.items()
        }
        # Add some noise: randomly skip a modality sometimes
        if random.random() < 0.2 and len(inputs) > 1:
            del_key = random.choice(list(inputs.keys()))
            del inputs[del_key]
            logger.info(f"Step {i + 1}: Skipped modality '{del_key}'")

        hpm.buffer_input(inputs)
        logger.info(
            f"Step {i + 1}: Buffered {len(inputs)} modalities. Status: { {k: len(b) for k, b in hpm.input_buffers.items()} }"
        )
        time.sleep(0.05)  # Simulate time passing

    # --- Test Fusion ---
    logger.info("\n--- Testing Fusion (Aggregation: 'latest') ---")

    async def run_fusion():
        # Get fused representation using latest buffered item(s)
        fused_latest = await hpm.get_fused_representation(
            aggregation="latest", num_items=1
        )
        if fused_latest is not None:
            logger.info(f"Fused output ('latest', 1 item) shape: {fused_latest.shape}")
            check_for_nans(fused_latest, "Fused Output ('latest', 1)")

        # Get fused representation using average of buffer
        fused_average = await hpm.get_fused_representation(aggregation="average")
        if fused_average is not None:
            logger.info(f"Fused output ('average') shape: {fused_average.shape}")
            check_for_nans(fused_average, "Fused Output ('average')")

        # Test with modality weights
        weights = {
            "visual": 0.6,
            "auditory": 0.3,
            "video": 0.8,
            "tactile": 0.1,
        }  # Example weights
        logger.info(
            f"\n--- Testing Fusion (Aggregation: 'latest' with Weights: {weights}) ---"
        )
        fused_weighted = await hpm.get_fused_representation(
            aggregation="latest", modality_weights=weights
        )
        if fused_weighted is not None:
            logger.info(
                f"Fused output ('latest', weighted) shape: {fused_weighted.shape}"
            )
            check_for_nans(fused_weighted, "Fused Output ('latest', weighted)")

        # --- Test Status ---
        logger.info("\n--- Testing Status ---")
        status = hpm.get_status()
        logger.info(f"Module Status: {status}")

    asyncio.run(run_fusion())

    # --- Example with LSTM (Conceptual) ---
    # This part shows how an LSTM *could* integrate, but needs Hidden_LSTM to be real
    # and a decision on how its output feeds into HPM (e.g., as another modality)
    # logger.info("\n--- Conceptual LSTM Integration ---")
    # try:
    #     # Use HPM output dim as input to LSTM? Or intermediate dim?
    #     lstm_input_dim = config['holistic_output_dim']
    #     lstm_config = {
    #         'LSTM_INPUT_DIM': lstm_input_dim,
    #         'LSTM_HIDDEN_DIM': 256,
    #         'LSTM_OUTPUT_DIM': 128,
    #     }
    #     # Need the actual Hidden_LSTM class available
    #     lstm_model = Hidden_LSTM(lstm_config).to(hpm.device)

    #     # Assume we have a sequence of fused outputs from HPM
    #     # fused_sequence = torch.stack([... previous fused_outputs ...], dim=1) # Shape: (batch, seq_len, feature)
    #     fused_sequence = torch.rand(1, 5, lstm_input_dim).to(hpm.device) # Dummy sequence

    #     lstm_processed_output = lstm_model(fused_sequence)
    #     logger.info(f"Conceptual LSTM output shape: {lstm_processed_output.shape}")

    # except NameError:
    #      logger.warning("Skipping LSTM conceptual test because Hidden_LSTM stub is in use.")
    # except Exception as e:
    #      logger.error(f"Error in conceptual LSTM test: {e}")

    logger.info("--- HolisticPerception Example Finished ---")


# --- END OF REFINED holistic_perception.py ---
