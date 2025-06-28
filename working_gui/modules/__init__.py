"""
VoxSigil Working GUI Modules Package
===================================

This package contains core modules for the VoxSigil working GUI system.

Modules:
- training_worker: Asynchronous training worker with VantaCore integration
- (other modules will be added as they are developed)

Author: VoxSigil AI Assistant
"""

from .training_worker import TrainingWorker

__all__ = ["TrainingWorker"]
