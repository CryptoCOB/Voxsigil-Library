#!/usr/bin/env python
"""
Model Architecture Inspector and Fixer - HOLO-1.5 Enhanced
Analyzes saved GridFormer models and creates compatible loading code
"""

import torch
import json
import os
from typing import Dict, Any, Optional, Tuple

# HOLO-1.5 imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole


@vanta_core_module(
    name="model_architecture_processor",
    subsystem="optimization",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="Model architecture analysis and compatibility processing for GridFormer models",
    capabilities=["model_analysis", "architecture_inspection", "compatibility_generation", "checkpoint_processing", "code_generation"],
    cognitive_load=3.0,
    symbolic_depth=2,
    collaboration_patterns=["model_introspection", "architecture_adaptation", "compatibility_bridge"]
)
class ModelArchitectureProcessor(BaseCore):
    """
    Model Architecture Processor with HOLO-1.5 integration.
    
    Provides comprehensive model analysis, architecture inspection,
    and compatibility code generation for GridFormer models.
    """

    def __init__(self, vanta_core: Any, config: Dict[str, Any]):
        """
        Initialize the ModelArchitectureProcessor.
        
        Args:
            vanta_core: VantaCore instance for HOLO-1.5 compliance
            config: Configuration dictionary for BaseCore
        """
        # Initialize BaseCore
        super().__init__(vanta_core, config)
        
        # Processing state
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.generated_code_cache: Dict[str, str] = {}
        
    async def initialize(self) -> bool:
        """Initialize the ModelArchitectureProcessor for BaseCore compliance."""
        try:
            # Register with VantaCore for model analysis events
            if hasattr(self.vanta_core, 'register_component'):
                self.vanta_core.register_component(
                    "model_architecture_processor", 
                    self, 
                    {"type": "model_analysis_service"}
                )
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error initializing ModelArchitectureProcessor: {e}")
            return False

    def analyze_saved_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze the architecture of a saved model"""
        print(f"Analyzing model: {model_path}")

        # Check cache first
        if model_path in self.analysis_cache:
            return self.analysis_cache[model_path]

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

            # Cache the analysis
            self.analysis_cache[model_path] = analysis
            return analysis

        except Exception as e:
            print(f"Error analyzing model: {e}")
            return {"error": str(e)}

    def generate_compatible_model_class(self, analysis: Dict[str, Any]) -> str:
        """Generate Python code for a compatible model class"""
        
        analysis_key = str(hash(str(analysis)))
        if analysis_key in self.generated_code_cache:
            return self.generated_code_cache[analysis_key]
            
        if analysis["architecture_type"] == "basic_gridformer":
            code = self._generate_basic_gridformer_class(analysis)
        elif analysis["architecture_type"] == "enhanced_gridformer":
            code = self._generate_enhanced_gridformer_class(analysis)
        else:
            code = "# Unknown architecture - manual inspection required"
            
        # Cache the generated code
        self.generated_code_cache[analysis_key] = code
        return code

    def _generate_basic_gridformer_class(self, analysis: Dict[str, Any]) -> str:
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
'''

    def _generate_enhanced_gridformer_class(self, analysis: Dict[str, Any]) -> str:
        """Generate a minimal enhanced GridFormer class based on model analysis."""
        config = analysis.get("config", {})
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 6)

        template = f"""
import torch
import torch.nn as nn

class EnhancedGridFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = {hidden_dim}
        self.num_layers = {num_layers}
        
        # Add basic components
        self.embedding = nn.Embedding(10, {hidden_dim})  # Vocab size 10 for ARC
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model={hidden_dim},
                nhead=8,
                batch_first=True
            ),
            num_layers={num_layers}
        )
        self.output_projection = nn.Linear({hidden_dim}, 10)

    def forward(self, x):
        # Basic forward pass
        batch_size, height, width = x.shape
        x_flat = x.view(batch_size, -1)
        embedded = self.embedding(x_flat)
        encoded = self.transformer_encoder(embedded)
        logits = self.output_projection(encoded)
        return logits.view(batch_size, height, width, -1)

def load_enhanced_gridformer(model_path: str):
    \"\"\"Load an enhanced GridFormer model from checkpoint\"\"\"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EnhancedGridFormer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device
"""
        return template

    def load_compatible_model(self, model_path: str) -> Tuple[Optional[torch.nn.Module], Optional[torch.device]]:
        """Load a model using the generated compatible code"""
        try:
            analysis = self.analyze_saved_model(model_path)
            if "error" in analysis:
                return None, None
                
            compatible_code = self.generate_compatible_model_class(analysis)
            
            # Execute the generated code in a safe namespace
            namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "math": __import__("math")}
            exec(compatible_code, namespace)
            
            # Load the model based on architecture type
            if analysis["architecture_type"] == "basic_gridformer":
                return namespace["load_basic_gridformer"](model_path)
            elif analysis["architecture_type"] == "enhanced_gridformer":
                return namespace["load_enhanced_gridformer"](model_path)
            else:
                print(f"Unknown architecture type: {analysis['architecture_type']}")
                return None, None
                
        except Exception as e:
            print(f"Error loading compatible model: {e}")
            return None, None

    def save_analysis_report(self, model_path: str, output_dir: str = "./") -> str:
        """Generate and save a comprehensive analysis report"""
        analysis = self.analyze_saved_model(model_path)
        
        # Generate compatible code
        compatible_code = self.generate_compatible_model_class(analysis)
        
        # Save analysis as JSON
        analysis_file = os.path.join(output_dir, "model_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save compatible code
        code_file = os.path.join(output_dir, "compatible_gridformer_model.py")
        with open(code_file, "w") as f:
            f.write(compatible_code)
            
        # Generate summary report
        report_file = os.path.join(output_dir, "analysis_report.md")
        with open(report_file, "w") as f:
            f.write(self._generate_analysis_report(analysis))
            
        return report_file

    def _generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a markdown analysis report"""
        report = f"""# Model Architecture Analysis Report

## Model Information
- **File**: {analysis.get('file_path', 'Unknown')}
- **Size**: {analysis.get('file_size_mb', 0):.1f} MB
- **Architecture Type**: {analysis.get('architecture_type', 'Unknown')}
- **Parameter Count**: {analysis.get('parameter_count', 0)}

## Configuration
"""
        
        config = analysis.get("config", {})
        if config:
            for key, value in config.items():
                report += f"- **{key}**: {value}\n"
        else:
            report += "No configuration found in checkpoint.\n"
            
        report += "\n## Model Components\n"
        
        layers = analysis.get("model_layers", {})
        for component, layer_list in layers.items():
            report += f"- **{component}**: {len(layer_list)} parameters\n"
            if len(layer_list) <= 5:
                for layer in layer_list[:3]:
                    report += f"  - {layer}\n"
                if len(layer_list) > 3:
                    report += f"  - ... and {len(layer_list) - 3} more\n"
                    
        report += "\n## Next Steps\n"
        report += "1. Use the generated `compatible_gridformer_model.py`\n"
        report += "2. Import and use the appropriate load function\n"
        report += "3. Test predictions with the properly loaded model\n"
        
        return report


# Legacy function wrappers for backward compatibility
def analyze_saved_model(model_path: str) -> Dict[str, Any]:
    """Legacy wrapper for analyze_saved_model"""
    processor = ModelArchitectureProcessor(None, {})
    return processor.analyze_saved_model(model_path)


def generate_compatible_model_class(analysis: Dict[str, Any]) -> str:
    """Legacy wrapper for generate_compatible_model_class"""
    processor = ModelArchitectureProcessor(None, {})
    return processor.generate_compatible_model_class(analysis)


def main():
    """Main analysis function"""
    processor = ModelArchitectureProcessor(None, {})
    model_path = "gridformer_best.pth"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print("=== MODEL ARCHITECTURE ANALYSIS ===")
    
    # Generate comprehensive report
    report_file = processor.save_analysis_report(model_path)
    print(f"Analysis report saved to: {report_file}")


if __name__ == "__main__":
    main()
