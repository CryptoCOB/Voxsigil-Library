#!/usr/bin/env python
"""
Model Architecture Inspector and Fixer - HOLO-1.5 Enhanced
Analyzes saved GridFormer models and creates compatible loading code
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

# HOLO-1.5 imports
from .base import BaseCore, CognitiveMeshRole, vanta_core_module


@vanta_core_module(
    name="model_architecture_processor",
    subsystem="optimization",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="Model architecture analysis and compatibility processing for GridFormer models",
    capabilities=[
        "model_analysis",
        "architecture_inspection",
        "compatibility_generation",
        "checkpoint_processing",
        "code_generation",
    ],
    cognitive_load=3.0,
    symbolic_depth=2,
    collaboration_patterns=[
        "model_introspection",
        "architecture_adaptation",
        "compatibility_bridge",
    ],
)
class ModelArchitectureProcessor(BaseCore):
    """
    Model Architecture Processor with HOLO-1.5 integration.
    Provides comprehensive model analysis, architecture inspection,
    and compatibility code generation for GridFormer models.
    """

    # ------------------------------------------------------------------ #
    #  Encapsulated feature 1 – configurable logger injection            #
    # ------------------------------------------------------------------ #
    def __init__(self, vanta_core: Any, config: Dict[str, Any]):
        super().__init__(vanta_core, config)

        self.logger = config.get("logger", logging.getLogger(__name__))
        self.logger.setLevel(config.get("log_level", logging.INFO))

        # in-memory caches
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.generated_code_cache: Dict[str, str] = {}

        # ------------------------------------------------------------------ #
        #  Encapsulated feature 2 – optional disk-based analysis cache       #
        # ------------------------------------------------------------------ #
        self._disk_cache_file = config.get(
            "disk_cache_file", ".model_analysis_cache.json"
        )
        if os.path.exists(self._disk_cache_file):
            try:
                with open(self._disk_cache_file, "r") as fp:
                    self.analysis_cache.update(json.load(fp))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed loading disk cache: %s", exc)

    async def initialize(self) -> bool:  # noqa: D401
        """Initialize the ModelArchitectureProcessor for BaseCore compliance."""
        try:
            if hasattr(self.vanta_core, "register_component"):
                self.vanta_core.register_component(
                    "model_architecture_processor",
                    self,
                    {"type": "model_analysis_service"},
                )
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error initializing ModelArchitectureProcessor: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Encapsulated feature 3 – checkpoint integrity verification        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def verify_checkpoint_integrity(self, model_path: str) -> str:
        """
        Compute and return the SHA-256 checksum of a checkpoint.
        Useful for cache keys or corruption checks.
        """
        checksum = self._sha256(model_path)
        self.logger.debug("Checksum for %s: %s", model_path, checksum)
        return checksum

    def analyze_saved_model(self, model_path: str) -> Dict[str, Any]:  # noqa: C901
        """Analyze the architecture of a saved model."""
        self.logger.info("Analyzing model: %s", model_path)

        if model_path in self.analysis_cache:
            return self.analysis_cache[model_path]

        try:
            # Load checkpoint to CPU
            checkpoint = torch.load(model_path, map_location="cpu")

            analysis: Dict[str, Any] = {
                "file_path": model_path,
                "file_size_mb": os.path.getsize(model_path) / 1_048_576,
                "checksum": self.verify_checkpoint_integrity(model_path),
                "checkpoint_keys": list(checkpoint.keys()),
                "config": checkpoint.get("config", {}),
                "architecture_type": "unknown",
                "model_layers": {},
                "parameter_count": 0,
            }

            # Parameter inspection
            state_dict = checkpoint.get("model_state_dict", {})
            analysis["parameter_count"] = len(state_dict)

            layer_types: Dict[str, list[str]] = {}
            for key in state_dict:
                component = key.split(".")[0]
                layer_types.setdefault(component, []).append(key)
            analysis["model_layers"] = layer_types

            if "transformer_encoder" in layer_types:
                analysis["architecture_type"] = "enhanced_gridformer"
            elif "blocks" in layer_types:
                analysis["architecture_type"] = "basic_gridformer"

            # Cache (memory + optional disk)
            self.analysis_cache[model_path] = analysis
            try:
                with open(self._disk_cache_file, "w") as fp:
                    json.dump(self.analysis_cache, fp)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed writing disk cache: %s", exc)

            return analysis
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error analyzing model: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Encapsulated feature 4 – quick forward-pass sanity test           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quick_test_model(model: torch.nn.Module, grid_size: int = 3) -> bool:
        """
        Run a tiny dummy input through the model to ensure it produces output.
        Returns True on success, False otherwise.
        """
        try:
            dummy = torch.zeros(
                1,
                grid_size,
                grid_size,
                dtype=torch.long,
                device=next(model.parameters()).device,
            )
            _ = model(dummy)
            return True
        except Exception:  # noqa: BLE001
            return False

    def generate_compatible_model_class(self, analysis: Dict[str, Any]) -> str:
        """Generate Python code for a compatible model class."""
        cache_key = str(hash(json.dumps(analysis, sort_keys=True)))
        if cache_key in self.generated_code_cache:
            return self.generated_code_cache[cache_key]

        if analysis["architecture_type"] == "basic_gridformer":
            code = self._generate_basic_gridformer_class(analysis)
        elif analysis["architecture_type"] == "enhanced_gridformer":
            code = self._generate_enhanced_gridformer_class(analysis)
        else:
            code = "# Unknown architecture – manual inspection required\n"

        self.generated_code_cache[cache_key] = code
        return code

    # ------------------------------------------------------------------ #
    #  Encapsulated feature 5 – cache clearing utility                   #
    # ------------------------------------------------------------------ #
    def clear_caches(self) -> None:
        """Clear in-memory and on-disk caches."""
        self.analysis_cache.clear()
        self.generated_code_cache.clear()
        if os.path.exists(self._disk_cache_file):
            try:
                os.remove(self._disk_cache_file)
            except OSError as exc:  # noqa: BLE001
                self.logger.warning("Could not delete cache file: %s", exc)

    # ------------------------------------------------------------------ #
    #  Code generators                                                   #
    # ------------------------------------------------------------------ #
    def _generate_basic_gridformer_class(self, analysis: Dict[str, Any]) -> str:
        cfg = analysis.get("config", {})
        return f'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """2-D positional encoding for flattened grid tokens."""
    def __init__(self, hidden_dim: int, max_size: int = 30):
        super().__init__()
        self.register_parameter(
            "pe",
            nn.Parameter(torch.randn(1, max_size, max_size, hidden_dim) * 0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, D)
        b, s, d = x.size()
        g = int(math.sqrt(s))
        return x + self.pe[:, :g, :g, :].reshape(1, -1, d)[:, :s, :]


class BasicGridFormerConfig:
    vocab_size = {cfg.get("vocab_size", 10)}
    hidden_dim = {cfg.get("hidden_dim", 256)}
    num_layers = {cfg.get("num_layers", 6)}
    num_heads = {cfg.get("num_heads", 8)}
    max_grid_size = {cfg.get("max_grid_size", 30)}
    dropout = {cfg.get("dropout", 0.1)}
    device = "cuda" if torch.cuda.is_available() else "cpu"


class BasicGridFormer(nn.Module):
    def __init__(self, cfg: BasicGridFormerConfig | None = None):
        super().__init__()
        self.cfg = cfg or BasicGridFormerConfig()

        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.pos_encoding = PositionalEncoding2D(self.cfg.hidden_dim, self.cfg.max_grid_size)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.cfg.hidden_dim,
                    nhead=self.cfg.num_heads,
                    dim_feedforward=self.cfg.hidden_dim * 4,
                    dropout=self.cfg.dropout,
                    batch_first=True,
                )
                for _ in range(self.cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.head = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W)
        b, h, w = x.size()
        x = x.view(b, -1)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x.view(b, h, w, -1)


def load_basic_gridformer(model_path: str, device: str | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    chkpt = torch.load(model_path, map_location=device)
    model = BasicGridFormer()
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device).eval()
    return model, device
'''

    def _generate_enhanced_gridformer_class(self, analysis: Dict[str, Any]) -> str:
        cfg = analysis.get("config", {})
        hidden_dim = cfg.get("hidden_dim", 256)
        num_layers = cfg.get("num_layers", 6)
        return f"""
import torch
import torch.nn as nn


class EnhancedGridFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, {hidden_dim})
        encoder_layer = nn.TransformerEncoderLayer(d_model={hidden_dim}, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers={num_layers})
        self.head = nn.Linear({hidden_dim}, 10)

    def forward(self, x: torch.Tensor):
        b, h, w = x.size()
        x = self.embedding(x.view(b, -1))
        x = self.encoder(x)
        return self.head(x).view(b, h, w, -1)


def load_enhanced_gridformer(model_path: str, device: str | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    chkpt = torch.load(model_path, map_location=device)
    model = EnhancedGridFormer()
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device).eval()
    return model, device
"""

    def load_compatible_model(
        self, model_path: str
    ) -> Tuple[Optional[torch.nn.Module], Optional[torch.device]]:
        """Load a model using predefined architecture patterns (safer than exec)"""
        try:
            analysis = self.analyze_saved_model(model_path)
            if "error" in analysis:
                return (
                    None,
                    None,
                )  # Use predefined architecture loaders instead of exec()
            architecture_type = analysis.get("architecture_type", "unknown")

            if architecture_type == "basic_gridformer":
                return self._load_basic_gridformer(model_path, analysis)
            elif architecture_type == "enhanced_gridformer":
                return self._load_enhanced_gridformer(model_path, analysis)
            else:
                self.logger.warning(f"Unknown architecture type: {architecture_type}")
                return None, None

        except ImportError as e:
            self.logger.error(f"Missing dependencies for model loading: {e}")
            return None, None
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            return None, None
        except Exception as e:
            self.logger.error(
                f"Unexpected error loading compatible model: {e}", exc_info=True
            )
            return None, None

    def _load_basic_gridformer(
        self, model_path: str, analysis: dict
    ) -> Tuple[Optional[torch.nn.Module], Optional[torch.device]]:
        """Safely load a basic gridformer model without exec()"""
        try:
            # Implementation for basic gridformer loading
            # This is safer than exec() because it uses predefined, validated code paths
            self.logger.info("Loading basic gridformer model")
            # Add actual implementation here based on your architecture
            return None, None  # Placeholder
        except Exception as e:
            self.logger.error(f"Error loading basic gridformer: {e}")
            return None, None

    def _load_enhanced_gridformer(
        self, model_path: str, analysis: dict
    ) -> Tuple[Optional[torch.nn.Module], Optional[torch.device]]:
        """Safely load an enhanced gridformer model without exec()"""
        try:
            # Implementation for enhanced gridformer loading
            self.logger.info("Loading enhanced gridformer model")
            # Add actual implementation here based on your architecture
            return None, None  # Placeholder
        except Exception as e:
            self.logger.error(f"Error loading enhanced gridformer: {e}")
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
- **File**: {analysis.get("file_path", "Unknown")}
- **Size**: {analysis.get("file_size_mb", 0):.1f} MB
- **Architecture Type**: {analysis.get("architecture_type", "Unknown")}
- **Parameter Count**: {analysis.get("parameter_count", 0)}

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
