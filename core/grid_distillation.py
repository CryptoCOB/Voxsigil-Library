#!/usr/bin/env python
"""
grid_distillation.py - Teacher-Student Model Distillation for GRID-Former

Implements knowledge distillation techniques where a larger "teacher" model (LLM)
helps train a smaller "student" model (GRID-Former) on ARC tasks.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .grid_former import GRID_Former
from training.grid_model_trainer import GridFormerTrainer
from ARC.core.arc_data_processor import ARCGridDataProcessor, create_arc_dataloaders
from ARC.llm.arc_llm_interface import ARCAwareLLMInterface

logger = logging.getLogger("GRID-Former.Distillation")

class AugmentedARCDataset(Dataset):
    """
    Dataset class that augments ARC tasks with LLM-generated solutions.
    """
    
    def __init__(
        self,
        challenges_path: str,
        solutions_path: str,
        llm_interface: ARCAwareLLMInterface,
        cache_dir: str = "./llm_solutions_cache",
        regenerate: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            challenges_path: Path to ARC challenges JSON file
            solutions_path: Path to ARC solutions JSON file
            llm_interface: Interface to LLM for generating solutions
            cache_dir: Directory to cache LLM solutions
            regenerate: Whether to regenerate cached solutions
        """
        self.challenges_path = challenges_path
        self.solutions_path = solutions_path
        self.llm_interface = llm_interface
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.regenerate = regenerate
        
        # Load original data
        self.processor = ARCGridDataProcessor()
        self.task_data = self.processor.load_arc_data(challenges_path, solutions_path)
        self.task_ids = list(self.task_data.keys())
        
        # Generate or load LLM solutions
        self.llm_solutions = {}
        self._prepare_llm_solutions()
    
    def _prepare_llm_solutions(self) -> None:
        """
        Generate or load LLM solutions for all tasks.
        """
        cache_path = self.cache_dir / f"llm_solutions_{Path(self.challenges_path).stem}.json"
        
        # Load cached solutions if available
        if cache_path.exists() and not self.regenerate:
            logger.info(f"Loading cached LLM solutions from {cache_path}")
            with open(cache_path, 'r') as f:
                self.llm_solutions = json.load(f)
        else:
            # Generate solutions for each task
            logger.info("Generating LLM solutions for distillation")
            for i, task_id in enumerate(self.task_ids):
                if i % 10 == 0:
                    logger.info(f"Processing task {i+1}/{len(self.task_ids)}")
                
                task_data = self.task_data[task_id]
                
                # Skip if we already have a solution
                if task_id in self.llm_solutions and not self.regenerate:
                    continue
                
                # Generate LLM solution
                try:
                    llm_solution = self.llm_interface.solve_arc_task(task_data)
                    if llm_solution:
                        self.llm_solutions[task_id] = llm_solution
                except Exception as e:
                    logger.error(f"Error generating solution for task {task_id}: {e}")
            
            # Cache solutions
            with open(cache_path, 'w') as f:
                json.dump(self.llm_solutions, f)
    
    def __len__(self) -> int:
        return len(self.task_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset item with both original and LLM solutions.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with task data, original solution, and LLM solution
        """
        task_id = self.task_ids[idx]
        task_data = self.task_data[task_id]
        
        # Get LLM solution if available
        llm_solution = self.llm_solutions.get(task_id, None)
        
        return {
            "task_id": task_id,
            "task_data": task_data,
            "original_solution": task_data.get("solutions", None),
            "llm_solution": llm_solution
        }

class DistillationTrainer:
    """
    Trainer for distilling knowledge from LLM to GRID-Former.
    """
    
    def __init__(
        self,
        student_model: GRID_Former,
        llm_interface: ARCAwareLLMInterface,
        device: Optional[str] = None,
        alpha: float = 0.5,  # Weight for distillation loss
        temperature: float = 2.0,  # Temperature for softening probability distributions
        output_dir: str = "./distilled_models"
    ):
        """
        Initialize the trainer.
        
        Args:
            student_model: GRID-Former model to be trained
            llm_interface: Interface to LLM for generating solutions
            device: Device for computations
            alpha: Weight for distillation loss (0-1)
            temperature: Temperature for softening probability distributions
            output_dir: Directory for saving models
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        self.student_model = student_model.to(self.device)
        self.llm_interface = llm_interface
        self.alpha = alpha
        self.temperature = temperature
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard trainer
        self.base_trainer = GridFormerTrainer(
            model=student_model,
            device=self.device,
            output_dir=output_dir
        )
        
        logger.info(f"Distillation trainer initialized with alpha={alpha}, temperature={temperature}")
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Logits from student model [batch, height, width, num_colors]
            teacher_probs: Probabilities from teacher model [batch, height, width, num_colors]
            targets: True targets [batch, height, width]
            mask: Mask for valid positions [batch, height, width]
            
        Returns:
            Combined distillation loss
        """
        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting to num_colors dimension
            expanded_mask = mask.unsqueeze(-1).expand_as(student_logits)
            student_logits = student_logits * expanded_mask
            teacher_probs = teacher_probs * expanded_mask
            targets = targets * mask
        
        # Standard cross-entropy loss
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1).long(),
            ignore_index=-1  # Ignore padded positions
        )
        
        # Distillation loss with temperature scaling
        soft_targets = teacher_probs
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            soft_predictions.view(-1, soft_predictions.size(-1)),
            soft_targets.view(-1, soft_targets.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by temperature squared
        
        # Combine losses
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return loss
    
    def train_with_distillation(
        self,
        challenges_path: str,
        solutions_path: str,
        batch_size: int = 16,
        num_epochs: int = 50,
        lr: float = 0.0005,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train student model with distillation.
        
        Args:
            challenges_path: Path to challenges JSON file
            solutions_path: Path to solutions JSON file
            batch_size: Batch size for training
            num_epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            early_stopping_patience: Epochs to wait for improvement before stopping
            
        Returns:
            Dictionary with training history
        """
        # Prepare dataset with LLM solutions
        logger.info("Preparing dataset with LLM solutions")
        dataset = AugmentedARCDataset(
            challenges_path=challenges_path,
            solutions_path=solutions_path,
            llm_interface=self.llm_interface
        )
        
        # Create data loaders
        train_loader, val_loader = create_arc_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.8
        )
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
            # verbose parameter removed as it's not supported in this version
        )
        
        # Initialize tracking variables
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = None
        
        # Training loop
        logger.info(f"Starting distillation training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            self.student_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Get data
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to(self.device)
                
                # Get teacher predictions (can be pre-computed for efficiency)
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(input_grids)
                
                # Forward pass
                optimizer.zero_grad()
                student_logits = self.student_model(input_grids)
                
                # Calculate loss
                mask = (output_grids != -1)  # Mask for valid positions
                loss = self.distillation_loss(
                    student_logits,
                    teacher_outputs,
                    output_grids,
                    mask
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(student_logits, dim=-1)
                train_correct += (predictions[mask] == output_grids[mask]).sum().item()
                train_total += mask.sum().item()
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Validation phase
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics['val_loss']
            val_accuracy = val_metrics['val_accuracy']
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"train_acc={train_accuracy:.4f}, "
                f"val_acc={val_accuracy:.4f}, "
                f"time={epoch_time:.2f}s"
            )
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save best model
                best_model_path = self.output_dir / f"distilled_model_best.pt"
                torch.save({
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'distillation_params': {
                        'alpha': self.alpha,
                        'temperature': self.temperature
                    }
                }, best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs without improvement")
                break
        
        # Save final model
        final_model_path = self.output_dir / f"distilled_model_final.pt"
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'distillation_params': {
                'alpha': self.alpha,
                'temperature': self.temperature
            }
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Load best model if available
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path)
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
        
        return history
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.student_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to(self.device)
                
                # Get teacher predictions
                teacher_outputs = self._get_teacher_outputs(input_grids)
                
                # Forward pass
                student_logits = self.student_model(input_grids)
                
                # Calculate loss
                mask = (output_grids != -1)  # Mask for valid positions
                loss = self.distillation_loss(
                    student_logits,
                    teacher_outputs,
                    output_grids,
                    mask
                )
                
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(student_logits, dim=-1)
                val_correct += (predictions[mask] == output_grids[mask]).sum().item()
                val_total += mask.sum().item()
        
        return {
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_correct / val_total if val_total > 0 else 0
        }
    
    def _get_teacher_outputs(self, input_grids: torch.Tensor) -> torch.Tensor:
        """
        Get teacher model outputs for input grids.
        This could be pre-computed for efficiency in a real implementation.
        
        Args:
            input_grids: Batch of input grids [batch, height, width]
            
        Returns:
            Teacher probabilities [batch, height, width, num_colors]
        """
        # In a real implementation, this would use the LLM to generate probabilities
        # Here we're providing a simplified version that returns uniform probabilities
        # weighted toward the student model's predictions for demonstration
        
        # Generate pseudo-probabilities from student model
        with torch.no_grad():
            student_logits = self.student_model(input_grids)
            student_probs = F.softmax(student_logits, dim=-1)
        
        # For simplicity, we're using the student's own predictions as a placeholder
        # In a real implementation, these would come from the LLM
        return student_probs

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Teacher-Student Distillation for GRID-Former")
    parser.add_argument('--model-path', type=str, help='Path to pre-trained student model')
    parser.add_argument('--challenges', type=str, default='arc-agi_training_challenges.json', help='ARC challenges file')
    parser.add_argument('--solutions', type=str, default='arc-agi_training_solutions.json', help='ARC solutions file')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for distillation loss')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for softening')
    parser.add_argument('--output-dir', type=str, default='./distilled_models', help='Output directory')
    parser.add_argument('--llm-name', type=str, default='mistral', help='LLM to use as teacher')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize LLM interface
    from ..arc_llm_handler import ARCAwareLLMInterface
    llm_interface = ARCAwareLLMInterface(model_name=args.llm_name)
    
    # Initialize student model
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading student model from {args.model_path}")
        student_model = GRID_Former.load_from_file(args.model_path)
    else:
        logger.info("Creating new student model")
        student_model = GRID_Former()
    
    # Initialize trainer
    trainer = DistillationTrainer(
        student_model=student_model,
        llm_interface=llm_interface,
        alpha=args.alpha,
        temperature=args.temperature,
        output_dir=args.output_dir
    )
    
    # Run training
    history = trainer.train_with_distillation(
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    logger.info("Distillation training complete")
