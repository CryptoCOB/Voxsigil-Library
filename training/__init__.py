"""
Training Components

This package contains training components for the VoxSigil system,
including ARC (Abstraction and Reasoning Corpus) dataset integration,
GridFormer training, and Vanta-controlled async training.
"""

from .arc_grid_trainer import ARCGridTrainer, VantaGridFormerBridge

# Import GridFormer training components
try:
    from .gridformer_training import (
        GridFormerDataset,
        GridFormerModel,
        GridFormerTrainer,
        GridFormerTrainingConfig,
        create_gridformer_trainer,
        train_gridformer_model,
    )

    HAVE_GRIDFORMER_TRAINING = True
except ImportError:
    HAVE_GRIDFORMER_TRAINING = False
    GridFormerTrainer = None
    GridFormerModel = None
    GridFormerDataset = None
    GridFormerTrainingConfig = None

# Legacy compatibility exports for Gridformer.training imports
GridFormer = GridFormerModel  # Legacy alias
ARCGridDataProcessor = ARCGridTrainer  # Legacy alias


# Training utilities
def get_available_trainers():
    """Get list of available training components."""
    trainers = ["ARCGridTrainer"]
    if HAVE_GRIDFORMER_TRAINING:
        trainers.extend(["GridFormerTrainer", "GridFormerModel"])
    return trainers


def create_trainer(trainer_type: str, config: dict = None):
    """Factory function to create training components."""
    config = config or {}

    if trainer_type == "ARCGridTrainer":
        return ARCGridTrainer(**config)
    elif trainer_type == "GridFormerTrainer" and HAVE_GRIDFORMER_TRAINING:
        return create_gridformer_trainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


__all__ = [
    "ARCGridTrainer",
    "VantaGridFormerBridge",
    "GridFormerTrainer",
    "GridFormerModel",
    "GridFormerDataset",
    "GridFormerTrainingConfig",
    "create_gridformer_trainer",
    "train_gridformer_model",
    # Legacy compatibility
    "GridFormer",
    "ARCGridDataProcessor",
    # Utilities
    "get_available_trainers",
    "create_trainer",
]
