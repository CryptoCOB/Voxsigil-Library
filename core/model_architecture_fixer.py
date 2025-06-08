#!/usr/bin/env python
"""
Model Architecture Inspector and Fixer
Analyzes saved GridFormer models and creates compatible loading code
"""

import torch
import json
import os
from typing import Dict, Any


def analyze_saved_model(model_path: str) -> Dict[str, Any]:
    """Analyze the architecture of a saved model"""
    print(f"Analyzing model: {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        analysis = {
            "file_path": model_path,
            "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
            "checkpoint_keys": list(checkpoint.keys()),
            "config": checkpoint.get("config", {}),
            "architecture_type": "unknown",
            "model_layers": {},
            "parameter_count": 0,
        }

        # Analyze model state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            analysis["parameter_count"] = len(state_dict)

            # Categorize layers
            layer_types = {}
            for key in state_dict.keys():
                parts = key.split(".")
                if len(parts) > 0:
                    main_component = parts[0]
                    if main_component not in layer_types:
                        layer_types[main_component] = []
                    layer_types[main_component].append(key)

            analysis["model_layers"] = layer_types

            # Determine architecture type
            if "transformer_encoder" in layer_types:
                analysis["architecture_type"] = "enhanced_gridformer"
            elif "blocks" in layer_types:
                analysis["architecture_type"] = "basic_gridformer"
            else:
                analysis["architecture_type"] = "unknown"

        return analysis

    except Exception as e:
        print(f"Error analyzing model: {e}")
        return {"error": str(e)}


def generate_compatible_model_class(analysis: Dict[str, Any]) -> str:
    """Generate Python code for a compatible model class"""

    if analysis["architecture_type"] == "basic_gridformer":
        return generate_basic_gridformer_class(analysis)
    elif analysis["architecture_type"] == "enhanced_gridformer":
        return generate_enhanced_gridformer_class(analysis)
    else:
        return "# Unknown architecture - manual inspection required"


def generate_basic_gridformer_class(analysis: Dict[str, Any]) -> str:
    """Generate basic GridFormer class compatible with saved model"""

    config = analysis.get("config", {})

    return f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    """2D Positional Encoding for grids"""
    def __init__(self, hidden_dim, max_size=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        
        # Create learnable 2D position encoding
        self.position_encoding = nn.Parameter(
            torch.randn(1, max_size, max_size, hidden_dim) * 0.1
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        # Assume seq_len represents flattened 2D grid
        batch_size, seq_len, hidden_dim = x.shape
        
        # Calculate grid dimensions
        grid_size = int(math.sqrt(seq_len))
        
        # Get position encodings for this grid size
        pos_enc = self.position_encoding[:, :grid_size, :grid_size, :].reshape(1, -1, hidden_dim)
        
        return x + pos_enc[:, :seq_len, :]

class BasicGridFormerConfig:
    """Configuration for BasicGridFormer"""
    def __init__(self):
        self.vocab_size = {config.get("vocab_size", 10)}
        self.hidden_dim = {config.get("hidden_dim", 256)}
        self.num_layers = {config.get("num_layers", 6)}
        self.num_heads = {config.get("num_heads", 8)}
        self.max_grid_size = {config.get("max_grid_size", 30)}
        self.dropout = {config.get("dropout", 0.1)}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class BasicGridFormer(nn.Module):
    """Basic GridFormer model compatible with saved checkpoint"""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = BasicGridFormerConfig()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Positional encoding
        self.position_encoding = PositionalEncoding2D(config.hidden_dim, config.max_grid_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x):
        # x shape: (batch, height, width)
        batch_size, height, width = x.shape
        
        # Flatten grid
        x_flat = x.view(batch_size, -1)  # (batch, seq_len)
        
        # Embed tokens
        embedded = self.embedding(x_flat)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        embedded = self.position_encoding(embedded)
        
        # Apply transformer blocks
        hidden = embedded
        for block in self.blocks:
            hidden = block(hidden)
        
        # Apply final norm
        hidden = self.norm(hidden)
        
        # Project to output vocabulary
        logits = self.output_projection(hidden)  # (batch, seq_len, vocab_size)
        
        # Reshape back to grid
        logits = logits.view(batch_size, height, width, -1)
        
        return logits

def load_basic_gridformer(model_path: str):
    """Load a basic GridFormer model from checkpoint"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with config from checkpoint
    model = BasicGridFormer()
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def generate_enhanced_gridformer_class(analysis: Dict[str, Any]) -> str:
    """Generate a minimal enhanced GridFormer class based on model analysis."""
    config = analysis.get("config", {})
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 6)
    return f"""
class EnhancedGridFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = {hidden_dim}
        self.num_layers = {num_layers}

    def forward(self, x):
        return x
"""


def main():
    """Main analysis function"""
    model_path = "gridformer_best.pth"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print("=== MODEL ARCHITECTURE ANALYSIS ===")
    analysis = analyze_saved_model(model_path)

    # Print analysis
    print(f"File: {analysis['file_path']}")
    print(f"Size: {analysis['file_size_mb']:.1f} MB")
    print(f"Architecture Type: {analysis['architecture_type']}")
    print(f"Parameter Count: {analysis['parameter_count']}")

    if "config" in analysis and analysis["config"]:
        print("\nConfiguration:")
        for key, value in analysis["config"].items():
            print(f"  {key}: {value}")

    print("\nModel Components:")
    for component, layers in analysis.get("model_layers", {}).items():
        print(f"  {component}: {len(layers)} parameters")
        if len(layers) <= 5:  # Show details for small components
            for layer in layers[:3]:
                print(f"    - {layer}")
            if len(layers) > 3:
                print(f"    ... and {len(layers) - 3} more")

    # Generate compatible code
    print("\n=== GENERATING COMPATIBLE MODEL CLASS ===")
    compatible_code = generate_compatible_model_class(analysis)

    # Save to file
    output_file = "compatible_gridformer_model.py"
    with open(output_file, "w") as f:
        f.write(compatible_code)

    print(f"Compatible model class saved to: {output_file}")

    # Save analysis
    analysis_file = "model_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to: {analysis_file}")

    print("\n=== NEXT STEPS ===")
    print("1. Use the generated compatible_gridformer_model.py")
    print("2. Import and use load_basic_gridformer() function")
    print("3. Test predictions with the properly loaded model")


if __name__ == "__main__":
    main()
