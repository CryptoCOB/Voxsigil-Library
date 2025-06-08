# Voxsigil-Library

Voxsigil-Library provides a collection of modules for building and running the
VoxSigil agent system.  It includes training utilities, a retrieval augmented
generation pipeline and GUI integration helpers.

## Running Tests

Use the following commands before submitting changes:

```bash
python test_integration.py
python test_step4.py
pytest -q
```

Some tests rely on optional GUI components.  If these are not available the
suite will still run but show warnings.

## Project Layout

- `voxsigil_integration.py` – main integration manager for GUI components
- `ART/` – Adaptive Resonance Theory modules
- `llm/` – utilities for language model handling
- `VoxSigilRag/` – retrieval augmented generation helpers

See `AGENTS.md` for contribution guidelines.
