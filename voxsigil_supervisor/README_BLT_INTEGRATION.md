# BLT-Enhanced VoxSigil Supervisor with TinyLlama

This integration connects the hybrid BLT middleware system to TinyLlama and the VoxSigil supervisor, creating an adaptive RAG system that leverages byte-level transformations for improved performance.

## Components

- **BLTSupervisorRagInterface**: Provides entropy-aware retrieval of VoxSigil constructs.
- **TinyLlamaIntegration**: Connects TinyLlama with the BLT-enhanced supervisor.

## Features

- **Entropy-Based Processing**: Automatically routes queries to the most appropriate processing path based on their entropy.
- **Hybrid Embeddings**: Combines byte-level and token-level embeddings for improved recall.
- **Dynamic Computation**: Allocates more resources to complex, high-entropy content.
- **Adaptive Scaffolding**: Retrieves reasoning scaffolds with enhanced accuracy.

## Usage

```python
from voxsigil_supervisor.blt_supervisor_integration import TinyLlamaIntegration
from pathlib import Path

# Initialize the integration
integration = TinyLlamaIntegration(
    voxsigil_library_path=Path("./voxsigil-Library"),
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    blt_config={
        "entropy_threshold": 0.3,
        "blt_hybrid_weight": 0.7
    }
)

# Create a supervisor
supervisor = integration.create_supervisor()

# Use the supervisor
result = supervisor.solve("How does quantum entanglement work?")
print(result)
```

## Quick Start

For a simpler approach, use the helper function:

```python
from models.tinyllama_assistant import initialize_voxsigil_components

# Initialize components
supervisor, rag = initialize_voxsigil_components(
    library_path="./voxsigil-Library",
    use_blt=True  # Enable BLT enhancement
)

# Use the supervisor
result = supervisor.solve("Explain deep learning")
print(result)
```

## Documentation

- [Full BLT-TinyLlama Integration Guide](./docs/blt_supervisor_tinyllama_guide.md)
- [Architecture Diagram](./docs/blt_supervisor_architecture.md)

## Testing

Run the integration tests:

```bash
python tests/test_blt_supervisor_integration.py
```

Run the mock tests (for development):

```bash
python tests/mock_blt_supervisor_integration.py
```
