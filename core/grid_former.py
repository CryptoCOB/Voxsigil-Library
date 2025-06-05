#!/usr/bin/env python
"""
grid_former.py - GRID-Former Model Architecture

Implements a specialized Transformer-based architecture (GRID-Former:
Grid Representation and Inference with Deep Transformers) for ARC tasks.
Designed for direct neural network training on grid pattern recognition.
"""

import os
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GRID-Former")

# Constants
MAX_GRID_SIZE = 30  # Maximum dimension for ARC grids
NUM_COLORS = 10  # ARC uses 0-9 colors
HIDDEN_DIM = 256  # Hidden dimension for transformer
NUM_HEADS = 8  # Number of attention heads
NUM_LAYERS = 6  # Number of transformer layers
DROPOUT = 0.1  # Dropout rate for regularization


class GridPositionEncoding(nn.Module):
    """
    Position encoding module specialized for 2D grids.
    Provides spatial awareness for the transformer model.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_grid_size: int = MAX_GRID_SIZE,
        dropout: float = DROPOUT,
    
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size

        # Create position encoding tensors
        position_encoding = torch.zeros(max_grid_size, max_grid_size, hidden_dim)

        # Calculate position encodings for each cell in the grid
        for i in range(max_grid_size):
            for j in range(max_grid_size):
                for k in range(0, hidden_dim, 2):
                    theta = (i * 10000 ** (-k / hidden_dim)) + (
                        j * 10000 ** (-(k + 1) / hidden_dim)
                    )
                    position_encoding[i, j, k] = math.sin(theta)
                    if k + 1 < hidden_dim:
                        position_encoding[i, j, k + 1] = math.cos(theta)
        
        # Register buffer for position encoding (not a parameter to be learned)
        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, H*W, hidden_dim)
            grid_shape: Tuple (height, width) of the original grid

        Returns:
            Tensor with positional encoding added
        """
        batch_size = x.size(0)
        h, w = grid_shape

        # Extract relevant position encodings for this grid size
        # Using reshape instead of view to handle non-contiguous tensor
        pos = self.position_encoding[:h, :w, :].reshape(1, h * w, self.hidden_dim)
        pos = pos.repeat(batch_size, 1, 1)

        return self.dropout(x + pos)


class GridEmbedding(nn.Module):
    """
    Embedding layer for grid color values.
    Converts integer color values (0-9) to dense vector representations.
    """

    def __init__(self, hidden_dim: int, num_colors: int = NUM_COLORS):
        super().__init__()
        self.color_embedding = nn.Embedding(num_colors, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors  # Store num_colors as a class member

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Embed grid color values into hidden dimension space.

        Args:
            grid: Tensor of shape (batch_size, height, width) with integer color values

        Returns:
            Tensor of shape (batch_size, height*width, hidden_dim) with embeddings
        """
        batch_size, h, w = grid.shape

        # Flatten spatial dimensions for embedding
        flat_grid = grid.reshape(batch_size, h * w)

        # Ensure indices are within valid range (0 to num_colors-1)
        flat_grid = torch.clamp(flat_grid, min=0, max=self.num_colors - 1)

        # Replace invalid indices (-1 for padding) with 0, we'll mask them later
        flat_grid = torch.where(flat_grid < 0, torch.zeros_like(flat_grid), flat_grid)

        # Create embeddings
        embeddings = self.color_embedding(flat_grid)

        # Apply layer normalization
        embeddings = self.norm(embeddings)

        return embeddings


class GridPatternRecognitionLayer(nn.Module):
    """
    Specialized layer for detecting common ARC grid patterns.
    Focuses on operations like symmetry detection, rotations, and more.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Base pattern recognition components
        self.symmetry_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.transform_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Combine different pattern recognitions
        self.pattern_integration = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Apply pattern recognition on the input tensor.

        Args:
            x: Input tensor of shape (batch_size, H*W, hidden_dim)
            grid_shape: Tuple (height, width) of the original grid

        Returns:
            Tensor with pattern recognition features added
        """
        # Run detectors
        symmetry_features = self.symmetry_detector(x)
        transform_features = self.transform_detector(x)

        # Concatenate features
        combined_features = torch.cat([symmetry_features, transform_features], dim=2)

        # Integrate patterns
        pattern_output = self.pattern_integration(combined_features)

        # Residual connection
        output = x + pattern_output

        return self.norm(output)


class GridMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism specialized for grid data.
    Enables the model to attend to different parts of the grid simultaneously.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = NUM_HEADS, dropout: float = DROPOUT
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            "Hidden dimension must be divisible by number of heads"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention on the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Self-attention output tensor of same shape
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        # Project to queries, keys, values
        q = (
            self.query_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout with proper tensor shape check
        if attention_weights.requires_grad:
            attention_weights = self.dropout(attention_weights)
        else:
            # Handle non-differentiable tensors without dropout
            pass

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.hidden_dim)
        )

        # Final projection
        output = self.output_proj(attention_output)
        output = self.dropout(output)

        # Residual connection and normalization
        output = self.norm(output + residual)

        return output


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer layers
    """

    def __init__(self, hidden_dim: int, ff_dim: int = None, dropout: float = DROPOUT):
        super().__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 4

        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation on input tensor.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        residual = x
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        # Residual connection and normalization
        output = self.norm(x + residual)

        return output


class TransformerLayer(nn.Module):
    """
    Single transformer layer with multi-head attention and feed-forward network
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = NUM_HEADS, dropout: float = DROPOUT
    ):
        super().__init__()
        self.attention = GridMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer layer to input tensor.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class GridTransformerEncoder(nn.Module):
    """
    Transformer encoder specialized for grid data.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer encoder to input tensor.

        Args:
            x: Input tensor

        Returns:
            Encoded tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x


class GridTransformationDecoder(nn.Module):
    """
    Decoder for generating output grids from encoded representations.
    """

    def __init__(self, hidden_dim: int, num_colors: int = NUM_COLORS):
        super().__init__()

        # Sequence of linear layers to decode grid
        self.decode_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Decode the encoded representation into an output grid.

        Args:
            x: Encoded tensor of shape (batch_size, H*W, hidden_dim)
            output_shape: Desired output grid shape (height, width)

        Returns:
            Logits for output grid colors of shape (batch_size, height, width, num_colors)
        """
        batch_size = x.size(0)
        h, w = output_shape

        # Apply decoding layers (features still in sequence form)
        output = self.decode_layers(x)

        # Get color dimension
        num_colors = output.size(-1)

        # Need to convert from sequence format to 2D grid
        # Rather than reshape (which needs exact dimensions), we'll resize
        # First, use a simple approach to map the sequence to a temporary 2D grid
        seq_len = x.size(1)
        temp_h = int(seq_len**0.5)  # Find a reasonable square-ish shape
        temp_w = (seq_len + temp_h - 1) // temp_h  # Ceiling division

        # Pad or truncate the sequence to fit temp_h * temp_w
        padded_len = temp_h * temp_w
        if seq_len < padded_len:
            # Pad with zeros if needed
            padding = torch.zeros(
                batch_size, padded_len - seq_len, num_colors, device=output.device
            )
            output = torch.cat([output, padding], dim=1)
        elif seq_len > padded_len:
            # Truncate if needed (should rarely happen due to ceiling division)
            output = output[:, :padded_len]

        # Reshape to temporary grid
        temp_grid = output.reshape(batch_size, temp_h, temp_w, num_colors)

        # Convert to format needed for interpolation (B, C, H, W)
        temp_grid = temp_grid.permute(0, 3, 1, 2)

        # Use interpolation to get to target size
        target_grid = F.interpolate(
            temp_grid, size=(h, w), mode="bilinear", align_corners=False
        )

        # Convert back to expected output format (B, H, W, C)
        return target_grid.permute(0, 2, 3, 1)


class GRID_Former(nn.Module):
    """
    GRID-Former: Grid Representation and Inference with Deep Transformers.

    Complete model architecture for ARC grid pattern recognition tasks.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        num_colors: int = NUM_COLORS,
        max_grid_size: int = MAX_GRID_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.dropout = dropout

        # Grid Encoding Module
        self.grid_embedding = GridEmbedding(hidden_dim, num_colors)
        self.position_encoding = GridPositionEncoding(
            hidden_dim, max_grid_size, dropout
        )

        # Pattern Recognition Modules
        self.pattern_recognition = GridPatternRecognitionLayer(hidden_dim)

        # Transformer Encoder
        self.transformer = GridTransformerEncoder(
            hidden_dim, num_layers, num_heads, dropout
        )

        # Grid Transformation Decoder
        self.decoder = GridTransformationDecoder(hidden_dim, num_colors)

    def forward(
        self, input_grid: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Process input grid to predict output grid.

        Args:
            input_grid: Input tensor of shape (batch_size, height, width) with integer color values
            target_shape: Optional target shape for output grid, defaults to input shape if None

        Returns:
            Logits for output grid colors of shape (batch_size, output_height, output_width, num_colors)
        """
        if target_shape is None:
            target_shape = (input_grid.size(1), input_grid.size(2))

        batch_size, h, w = input_grid.shape

        # Step 1: Embed grid colors
        x = self.grid_embedding(input_grid)

        # Step 2: Add position encoding
        x = self.position_encoding(x, (h, w))

        # Step 3: Apply pattern recognition
        x = self.pattern_recognition(x, (h, w))

        # Step 4: Process through transformer
        x = self.transformer(x)

        # Step 5: Decode to output grid
        output = self.decoder(x, target_shape)

        return output

    def predict_grid_transformation(
        self, input_grid: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Predict output grid based on input grid.

        Args:
            input_grid: Input grid tensor
            target_shape: Optional target shape for output grid

        Returns:
            Predicted output grid with integer color values
        """
        self.eval()
        with torch.no_grad():
            # Get logits
            logits = self(input_grid, target_shape)

            # Convert to predictions (argmax for each position)
            predictions = torch.argmax(logits, dim=-1)

        return predictions

    def save_to_file(self, path: str) -> None:
        """
        Save model weights to file.

        Args:
            path: Path to save the model
        """
        # Check if path has a directory component that needs to be created
        directory = os.path.dirname(path)
        if directory:  # Only try to create directories if there is a directory path
            os.makedirs(directory, exist_ok=True)

        # Save model state with all parameters
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "metadata": {
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "max_grid_size": self.max_grid_size,
                    "num_colors": self.num_colors,
                    "dropout": self.dropout,
                },
            },
            path,
        )

        logger.info(f"Saved GRID-Former model to {path}")

    @classmethod
    def load_from_file(cls, path: str, device: str = "cpu") -> "GRID_Former":
        """
        Load model from file.

        Args:
            path: Path to load the model from
            device: Device to load the model to

        Returns:
            Loaded GRID-Former model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)

        # Extract parameters from metadata if available
        metadata = checkpoint.get("metadata", {})

        if metadata:
            # Create model with parameters from metadata
            model = cls(
                hidden_dim=metadata.get("hidden_dim", HIDDEN_DIM),
                max_grid_size=metadata.get("max_grid_size", MAX_GRID_SIZE),
                num_colors=metadata.get("num_colors", NUM_COLORS),
                num_heads=metadata.get("num_heads", NUM_HEADS),
                num_layers=metadata.get("num_layers", NUM_LAYERS),
                dropout=metadata.get("dropout", DROPOUT),
            )
            logger.info(f"Created model with parameters from metadata: {metadata}")
        else:
            # Fallback to old method
            model = cls(
                hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
                max_grid_size=checkpoint.get("max_grid_size", MAX_GRID_SIZE),
                num_colors=checkpoint.get("num_colors", NUM_COLORS),
            )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Set device
        model = model.to(device)

        logger.info(f"Loaded GRID-Former model from {path}")

        return model


# Unit testing function for development
def test_grid_former():
    """
    Test the GRID-Former model with random input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create random input grid
    batch_size = 2
    h, w = 5, 7
    input_grid = torch.randint(0, 10, (batch_size, h, w)).to(device)
    target_shape = (8, 8)  # Different output shape

    # Create model
    model = GRID_Former().to(device)

    # Forward pass
    output = model(input_grid, target_shape)

    # Check output shape
    expected_shape = (batch_size, target_shape[0], target_shape[1], NUM_COLORS)
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )

    print(f"Input shape: {input_grid.shape}")
    print(f"Output shape: {output.shape}")
    print("All tests passed!")

    # Test prediction
    pred = model.predict_grid_transformation(input_grid)
    print(f"Prediction shape: {pred.shape}")

    try:
        # Test save/load with absolute path to ensure it works
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            temp_path = tmp.name

        print(f"Saving model to temporary file: {temp_path}")
        model.save_to_file(temp_path)
        print("Model saved successfully")

        print("Loading model from file")
        loaded_model = GRID_Former.load_from_file(temp_path)
        print("Model loaded successfully")

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Removed temporary file: {temp_path}")
    except Exception as e:
        print(f"Error during save/load test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_grid_former()
