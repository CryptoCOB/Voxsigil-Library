# ART Module

> **🔮 Schema 1.5-holo-alpha Upgrade Complete!**  
> *As of May 22, 2025, all VoxSigil files have been successfully upgraded to Schema 1.5-holo-alpha. This upgrade enhances support for immersive, experiential, reflective, and evolving cognitive systems within the Vanta ecosystem.*

This document provides an overview of the ART (Adaptive Resonance Theory) module within the VoxSigil supervisor component.

## Core Components

### art_controller.py
Central controller for ART operations, managing the adaptive resonance processes.

**Key Features:**
- Category formation and management
- Vigilance parameter control
- Resonance processing
- Stability-plasticity dynamics

### art_trainer.py
Trainer component for ART, handling the learning processes.

**Key Features:**
- Training data processing
- Learning rate management
- Category refinement
- Cross-validation and evaluation

## Bridge Components

### art_blt_bridge.py
Bridge between ART and BLT (Bidirectional Learning Transformer) components.

### art_entropy_bridge.py
Bridge for entropy-related processing, connecting ART with entropy analysis.

### art_hybrid_blt_bridge.py
Advanced bridge for the hybrid BLT implementation with enhanced capabilities.

### art_rag_bridge.py
Bridge connecting ART with RAG (Retrieval-Augmented Generation) components.

## Utility Components

### art_logger.py
Specialized logging for ART operations, providing detailed diagnostics.

### duplication_checker.py
Utility for checking and preventing duplicate pattern recognition.

### generative_art.py
Generative component for ART, creating new patterns based on learned categories.

### pattern_analysis.py
Tools for analyzing patterns recognized by ART.

## Tests

### tests/
Directory containing tests for ART functionality.

## Integration Points

ART integrates with the following components:
1. VANTA - for transformation operations
2. BLT - for enhanced bidirectional learning
3. RAG - for retrieval enhancement
4. SleepTimeCompute - for optimized memory consolidation
5. Entropy analysis - for information-theoretic processing

## Operational Flow

1. Input patterns are presented to ART
2. Controller determines if existing categories match the input
3. If matching category exists, it's updated (stability)
4. If no match, a new category may be created (plasticity)
5. The balance between stability and plasticity is managed by vigilance parameters
6. Bridge components enable interaction with other system elements
7. Trainer refines categories based on feedback and new examples
