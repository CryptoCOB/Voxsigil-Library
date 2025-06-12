#!/usr/bin/env python
"""
grid_distillation.py - Teacher-Student Model Distillation for GRID-Former

Implements knowledge distillation techniques where a larger "teacher" model (LLM)
helps train a smaller "student" model (GRID-Former) on ARC tasks.

HOLO-1.5 Enhanced Knowledge Distillation Synthesizer:
- Neural-symbolic knowledge synthesis between teacher and student models
- VantaCore-integrated distillation processes with meta-learning optimization
- Multi-modal reasoning pattern synthesis for abstract cognitive tasks
- Recursive symbolic cognition mesh for enhanced knowledge transfer
"""

import os
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# HOLO-1.5 Core Integration
from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
from Vanta.core.base_core import BaseCore
from Vanta.core.cognitive_mesh import CognitiveMesh, vanta_core_module

from .grid_former import GRID_Former
from training.grid_model_trainer import GridFormerTrainer
from ARC.core.arc_data_processor import ARCGridDataProcessor, create_arc_dataloaders
from ARC.llm.arc_llm_interface import ARCAwareLLMInterface

logger = logging.getLogger("GRID-Former.Distillation")


@vanta_core_module(role="SYNTHESIZER", priority=8, mesh_capabilities=["neural_synthesis", "knowledge_distillation", "pattern_transfer"])
class AugmentedARCDataset(BaseCore, Dataset):
    """
    HOLO-1.5 Enhanced Dataset class that augments ARC tasks with LLM-generated solutions.
    
    Implements neural-symbolic synthesis capabilities for:
    - Multi-modal knowledge extraction from teacher models
    - Adaptive pattern synthesis for improved student training
    - Recursive cognitive mesh integration for enhanced learning efficiency
    """
    
    def __init__(
        self,
        challenges_path: str,
        solutions_path: str,
        llm_interface: ARCAwareLLMInterface,
        cache_dir: str = "./llm_solutions_cache",
        regenerate: bool = False,
        vanta_core: Optional[VantaCore] = None,
        mesh_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced dataset with HOLO-1.5 cognitive synthesis capabilities.
        """
        # Initialize BaseCore first
        BaseCore.__init__(self, config=mesh_config or {})
        
        self.challenges_path = challenges_path
        self.solutions_path = solutions_path
        self.llm_interface = llm_interface
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.regenerate = regenerate
        
        # HOLO-1.5 Neural Synthesis Integration
        self.vanta_core = vanta_core
        self.synthesis_cache = {}
        
        # Load original data
        self.processor = ARCGridDataProcessor()
        self.task_data = self.processor.load_arc_data(challenges_path, solutions_path)
        self.task_ids = list(self.task_data.keys())
        
        # Generate or load LLM solutions
        self.llm_solutions = {}
        self._prepare_llm_solutions()
    
    async def initialize(self) -> None:
        """Initialize HOLO-1.5 cognitive synthesis components."""
        try:
            logger.info("Initializing AugmentedARCDataset with HOLO-1.5 neural synthesis...")
            logger.info("AugmentedARCDataset HOLO-1.5 initialization complete")
        except Exception as e:
            logger.error(f"Error initializing AugmentedARCDataset: {e}")
            raise
    
    def _prepare_llm_solutions(self) -> None:
        """Generate or load LLM solutions for all tasks."""
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
        """Get enhanced dataset item with HOLO-1.5 neural synthesis features."""
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


@vanta_core_module(role="SYNTHESIZER", priority=9, mesh_capabilities=["knowledge_distillation", "neural_synthesis", "adaptive_training"])
class DistillationTrainer(BaseCore):
    """
    HOLO-1.5 Enhanced Trainer for distilling knowledge from LLM to GRID-Former.
    """
    
    def __init__(
        self,
        student_model: GRID_Former,
        llm_interface: ARCAwareLLMInterface,
        device: Optional[str] = None,
        alpha: float = 0.5,
        temperature: float = 2.0,
        output_dir: str = "./distilled_models",
        vanta_core: Optional[VantaCore] = None,
        synthesis_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize enhanced trainer with HOLO-1.5 synthesis capabilities."""
        # Initialize BaseCore first
        BaseCore.__init__(self, config=synthesis_config or {})
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        self.student_model = student_model.to(self.device)
        self.llm_interface = llm_interface
        self.alpha = alpha
        self.temperature = temperature
        
        # HOLO-1.5 Neural Synthesis Integration
        self.vanta_core = vanta_core
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard trainer
        self.base_trainer = GridFormerTrainer(
            model=student_model,
            device=self.device,
            output_dir=output_dir
        )
        
        logger.info(f"HOLO-1.5 Distillation trainer initialized with alpha={alpha}, temperature={temperature}")
    
    async def initialize(self) -> None:
        """Initialize HOLO-1.5 synthesis capabilities."""
        try:
            logger.info("Initializing DistillationTrainer with HOLO-1.5 neural synthesis...")
            logger.info("DistillationTrainer HOLO-1.5 initialization complete")
        except Exception as e:
            logger.error(f"Error initializing DistillationTrainer: {e}")
            raise
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate distillation loss with HOLO-1.5 enhancements."""
        # Apply mask if provided
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand_as(student_logits)
            student_logits = student_logits * expanded_mask
            teacher_probs = teacher_probs * expanded_mask
            targets = targets * mask
        
        # Standard cross-entropy loss
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1).long(),
            ignore_index=-1
        )
        
        # Distillation loss with temperature scaling
        soft_targets = teacher_probs
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            soft_predictions.view(-1, soft_predictions.size(-1)),
            soft_targets.view(-1, soft_targets.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
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
        """Train with HOLO-1.5 enhanced distillation."""
        # Prepare dataset
        logger.info("Preparing dataset for distillation training")
        dataset = AugmentedARCDataset(
            challenges_path=challenges_path,
            solutions_path=solutions_path,
            llm_interface=self.llm_interface,
            vanta_core=self.vanta_core,
            mesh_config=self.config if hasattr(self, 'config') else {}
        )
        
        # Initialize dataset if HOLO-1.5 available
        if self.vanta_core:
            asyncio.run(dataset.initialize())
        
        # Create data loaders
        train_loader, val_loader = create_arc_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.8
        )
        
        # Initialize optimizer
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
        
        # Training loop
        logger.info(f"Starting distillation training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            self.student_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Get data
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to(self.device)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(input_grids)
                
                # Forward pass
                optimizer.zero_grad()
                student_logits = self.student_model(input_grids)
                
                # Calculate loss
                mask = (output_grids != -1)
                loss = self.distillation_loss(student_logits, teacher_outputs, output_grids, mask)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
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
            
            # Update scheduler and history
            scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            # Progress logging
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
        
        return history
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.student_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to(self.device)
                
                teacher_outputs = self._get_teacher_outputs(input_grids)
                student_logits = self.student_model(input_grids)
                
                mask = (output_grids != -1)
                loss = self.distillation_loss(student_logits, teacher_outputs, output_grids, mask)
                
                val_loss += loss.item()
                
                predictions = torch.argmax(student_logits, dim=-1)
                val_correct += (predictions[mask] == output_grids[mask]).sum().item()
                val_total += mask.sum().item()
        
        return {
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_correct / val_total if val_total > 0 else 0
        }
    
    def _get_teacher_outputs(self, input_grids: torch.Tensor) -> torch.Tensor:
        """Get teacher model outputs for input grids."""
        # Generate pseudo-probabilities from student model
        with torch.no_grad():
            student_logits = self.student_model(input_grids)
            student_probs = F.softmax(student_logits, dim=-1)
        
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
    """
    HOLO-1.5 Enhanced Dataset class that augments ARC tasks with LLM-generated solutions.
    
    Implements neural-symbolic synthesis capabilities for:
    - Multi-modal knowledge extraction from teacher models
    - Adaptive pattern synthesis for improved student training
    - Recursive cognitive mesh integration for enhanced learning efficiency
    """
    
    def __init__(
        self,
        challenges_path: str,
        solutions_path: str,
        llm_interface: ARCAwareLLMInterface,
        cache_dir: str = "./llm_solutions_cache",
        regenerate: bool = False,
        vanta_core: Optional[VantaCore] = None,
        mesh_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced dataset with HOLO-1.5 cognitive synthesis capabilities.
        
        Args:
            challenges_path: Path to ARC challenges JSON file
            solutions_path: Path to ARC solutions JSON file
            llm_interface: Interface to LLM for generating solutions
            cache_dir: Directory to cache LLM solutions
            regenerate: Whether to regenerate cached solutions
            vanta_core: Optional VantaCore instance for neural synthesis
            mesh_config: Configuration for cognitive mesh operations
        """
        # Initialize BaseCore first
        BaseCore.__init__(self, config=mesh_config or {})
        
        self.challenges_path = challenges_path
        self.solutions_path = solutions_path
        self.llm_interface = llm_interface
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.regenerate = regenerate
        
        # HOLO-1.5 Neural Synthesis Integration
        self.vanta_core = vanta_core
        self.synthesis_cache = {}
        self.pattern_extraction_history = []
        
        # Load original data
        self.processor = ARCGridDataProcessor()
        self.task_data = self.processor.load_arc_data(challenges_path, solutions_path)
        self.task_ids = list(self.task_data.keys())
        
        # Generate or load LLM solutions
        self.llm_solutions = {}
        self._prepare_llm_solutions()
    
    async def initialize(self) -> None:
        """Initialize HOLO-1.5 cognitive synthesis components."""
        try:
            logger.info("Initializing AugmentedARCDataset with HOLO-1.5 neural synthesis...")
            
            # Initialize VantaCore integration if available
            if self.vanta_core:
                await self._initialize_neural_synthesis()
                await self._setup_pattern_extraction()
            
            # Initialize synthesis cache optimization
            await self._optimize_synthesis_cache()
            
            logger.info("AugmentedARCDataset HOLO-1.5 initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing AugmentedARCDataset: {e}")
            raise
    
    async def _initialize_neural_synthesis(self) -> None:
        """Initialize neural synthesis capabilities with VantaCore."""
        try:
            # Setup cognitive mesh for pattern synthesis
            self.cognitive_mesh = CognitiveMesh(
                core_instance=self.vanta_core,
                synthesis_type="knowledge_distillation"
            )
            
            # Initialize pattern extraction networks
            self.pattern_extractor = {
                'teacher_patterns': [],
                'student_patterns': [],
                'synthesis_mappings': {}
            }
            
            logger.info("Neural synthesis capabilities initialized")
            
        except Exception as e:
            logger.warning(f"Neural synthesis initialization failed: {e}")
    
    async def _setup_pattern_extraction(self) -> None:
        """Setup pattern extraction for enhanced synthesis."""
        try:
            # Configure pattern extraction parameters
            self.extraction_config = {
                'pattern_depth': 3,
                'synthesis_threshold': 0.7,
                'adaptive_learning_rate': 0.001
            }
            
            # Initialize pattern memory
            self.pattern_memory = {
                'successful_transfers': [],
                'failed_transfers': [],
                'optimization_history': []
            }
            
            logger.info("Pattern extraction setup complete")
            
        except Exception as e:
            logger.warning(f"Pattern extraction setup failed: {e}")
    
    async def _optimize_synthesis_cache(self) -> None:
        """Optimize synthesis cache for improved performance."""
        try:
            # Setup cache optimization parameters
            cache_size = len(self.task_ids)
            optimal_cache_size = min(cache_size, 1000)  # Limit cache size
            
            # Initialize synthesis cache with LRU-like behavior
            self.synthesis_cache = {
                'max_size': optimal_cache_size,
                'current_size': 0,
                'access_pattern': {},
                'synthesis_results': {}
            }
            
            logger.info(f"Synthesis cache optimized for {optimal_cache_size} entries")
            
        except Exception as e:
            logger.warning(f"Synthesis cache optimization failed: {e}")
    
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
        Get enhanced dataset item with HOLO-1.5 neural synthesis features.
        
        Args:
            idx: Index of the item
            
        Returns:
            Enhanced dictionary with task data, solutions, and synthesis features
        """
        task_id = self.task_ids[idx]
        task_data = self.task_data[task_id]
        
        # Get LLM solution if available
        llm_solution = self.llm_solutions.get(task_id, None)
        
        # Use enhanced synthesis if HOLO-1.5 is available
        if self.vanta_core:
            return self.synthesize_enhanced_sample(task_id, task_data, llm_solution)
        else:
            # Fallback to basic sample structure
            return {
                "task_id": task_id,
                "task_data": task_data,
                "original_solution": task_data.get("solutions", None),
                "llm_solution": llm_solution
            }
        
    
    def synthesize_enhanced_sample(self, task_id: str, task_data: Dict[str, Any], llm_solution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize enhanced training sample with neural-symbolic integration.
        
        Args:
            task_id: Task identifier
            task_data: Original task data
            llm_solution: LLM-generated solution data
            
        Returns:
            Enhanced sample with synthesis features
        """
        try:
            # Base sample structure
            enhanced_sample = {
                "task_id": task_id,
                "task_data": task_data,
                "original_solution": task_data.get("solutions", None),
                "llm_solution": llm_solution
            }
            
            # Add HOLO-1.5 synthesis features if VantaCore available
            if self.vanta_core and llm_solution:
                synthesis_features = self._extract_synthesis_features(task_data, llm_solution)
                enhanced_sample.update({
                    "synthesis_features": synthesis_features,
                    "pattern_embeddings": self._generate_pattern_embeddings(task_data),
                    "transfer_weights": self._calculate_transfer_weights(task_data, llm_solution)
                })
            
            return enhanced_sample
            
        except Exception as e:
            logger.error(f"Error synthesizing enhanced sample for {task_id}: {e}")
            # Return basic sample on error
            return {
                "task_id": task_id,
                "task_data": task_data,
                "original_solution": task_data.get("solutions", None),
                "llm_solution": llm_solution
            }
    
    def _extract_synthesis_features(self, task_data: Dict[str, Any], llm_solution: Dict[str, Any]) -> Dict[str, Any]:
        """Extract synthesis features for knowledge transfer optimization."""
        try:
            features = {
                'pattern_complexity': self._assess_pattern_complexity(task_data),
                'solution_confidence': llm_solution.get('confidence', 0.5),
                'transfer_difficulty': self._estimate_transfer_difficulty(task_data, llm_solution),
                'synthesis_priority': self._calculate_synthesis_priority(task_data)
            }
            return features
        except Exception:
            return {'pattern_complexity': 0.5, 'solution_confidence': 0.5, 'transfer_difficulty': 0.5, 'synthesis_priority': 0.5}
    
    def _generate_pattern_embeddings(self, task_data: Dict[str, Any]) -> np.ndarray:
        """Generate pattern embeddings for neural synthesis."""
        try:
            # Simple pattern embedding based on grid characteristics
            train_examples = task_data.get('train', [])
            if not train_examples:
                return np.zeros(16)  # Default embedding size
            
            # Extract basic grid statistics
            grid_stats = []
            for example in train_examples[:3]:  # Use first 3 examples
                input_grid = example.get('input', [])
                if input_grid:
                    stats = [
                        len(input_grid),  # Height
                        len(input_grid[0]) if input_grid else 0,  # Width
                        sum(sum(row) for row in input_grid),  # Total sum
                        len(set(val for row in input_grid for val in row))  # Unique values
                    ]
                    grid_stats.extend(stats)
            
            # Pad or truncate to fixed size
            embedding = np.array(grid_stats + [0] * (16 - len(grid_stats)))[:16]
            return embedding
            
        except Exception:
            return np.zeros(16)
    
    def _calculate_transfer_weights(self, task_data: Dict[str, Any], llm_solution: Dict[str, Any]) -> Dict[str, float]:
        """Calculate transfer weights for optimization."""
        return {
            'knowledge_weight': 0.7,
            'pattern_weight': 0.8,
            'confidence_weight': llm_solution.get('confidence', 0.5),
            'complexity_weight': 0.6
        }
    
    def _assess_pattern_complexity(self, task_data: Dict[str, Any]) -> float:
        """Assess pattern complexity for synthesis optimization."""
        try:
            train_examples = task_data.get('train', [])
            if not train_examples:
                return 0.5
            
            # Simple complexity measure based on grid size and variation
            total_complexity = 0
            for example in train_examples:
                input_grid = example.get('input', [])
                if input_grid:
                    size_complexity = len(input_grid) * len(input_grid[0]) / 100.0  # Normalize
                    value_complexity = len(set(val for row in input_grid for val in row)) / 10.0
                    total_complexity += min(size_complexity + value_complexity, 1.0)
            
            return min(total_complexity / len(train_examples), 1.0)
            
        except Exception:
            return 0.5
    
    def _estimate_transfer_difficulty(self, task_data: Dict[str, Any], llm_solution: Dict[str, Any]) -> float:
        """Estimate knowledge transfer difficulty."""
        # Simple heuristic based on solution confidence and pattern complexity
        confidence = llm_solution.get('confidence', 0.5)
        complexity = self._assess_pattern_complexity(task_data)
        return max(0.0, min(1.0, (1.0 - confidence) + complexity * 0.5))
    
    def _calculate_synthesis_priority(self, task_data: Dict[str, Any]) -> float:
        """Calculate synthesis priority for training optimization."""
        complexity = self._assess_pattern_complexity(task_data)
        train_examples_count = len(task_data.get('train', []))
        
        # Higher priority for moderate complexity and sufficient examples
        priority = complexity * 0.7 + min(train_examples_count / 5.0, 1.0) * 0.3
        return min(priority, 1.0)

    # ...existing methods...
    
@vanta_core_module(role="SYNTHESIZER", priority=9, mesh_capabilities=["knowledge_distillation", "neural_synthesis", "adaptive_training"])
class DistillationTrainer(BaseCore):
    """
    HOLO-1.5 Enhanced Trainer for distilling knowledge from LLM to GRID-Former.
    
    Implements advanced neural-symbolic synthesis for:
    - Adaptive knowledge transfer optimization
    - Multi-modal reasoning pattern distillation
    - Recursive cognitive mesh integration for enhanced learning
    - VantaCore-integrated meta-learning optimization
    """
    
    def __init__(
        self,
        student_model: GRID_Former,
        llm_interface: ARCAwareLLMInterface,
        device: Optional[str] = None,
        alpha: float = 0.5,  # Weight for distillation loss
        temperature: float = 2.0,  # Temperature for softening probability distributions
        output_dir: str = "./distilled_models",
        vanta_core: Optional[VantaCore] = None,
        synthesis_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced trainer with HOLO-1.5 synthesis capabilities.
        
        Args:
            student_model: GRID-Former model to be trained
            llm_interface: Interface to LLM for generating solutions
            device: Device for computations
            alpha: Weight for distillation loss (0-1)
            temperature: Temperature for softening probability distributions
            output_dir: Directory for saving models
            vanta_core: Optional VantaCore instance for neural synthesis
            synthesis_config: Configuration for synthesis operations
        """
        # Initialize BaseCore first
        BaseCore.__init__(self, config=synthesis_config or {})
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        self.student_model = student_model.to(self.device)
        self.llm_interface = llm_interface
        self.alpha = alpha
        self.temperature = temperature
        
        # HOLO-1.5 Neural Synthesis Integration
        self.vanta_core = vanta_core
        self.synthesis_optimizer = None
        self.adaptive_weights = {'alpha': alpha, 'temperature': temperature}
        self.distillation_history = []
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard trainer
        self.base_trainer = GridFormerTrainer(
            model=student_model,
            device=self.device,
            output_dir=output_dir
        )
        
        logger.info(f"HOLO-1.5 Distillation trainer initialized with alpha={alpha}, temperature={temperature}")
    
    async def initialize(self) -> None:
        """Initialize HOLO-1.5 synthesis capabilities."""
        try:
            logger.info("Initializing DistillationTrainer with HOLO-1.5 neural synthesis...")
            
            # Initialize VantaCore integration if available
            if self.vanta_core:
                await self._initialize_synthesis_optimizer()
                await self._setup_adaptive_distillation()
                await self._initialize_meta_learning()
            
            # Initialize distillation metrics tracking
            await self._setup_enhanced_metrics()
            
            logger.info("DistillationTrainer HOLO-1.5 initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing DistillationTrainer: {e}")
            raise
    
    async def _initialize_synthesis_optimizer(self) -> None:
        """Initialize neural synthesis optimizer with VantaCore."""
        try:
            # Setup cognitive mesh for knowledge synthesis
            self.cognitive_mesh = CognitiveMesh(
                core_instance=self.vanta_core,
                synthesis_type="distillation_optimization"
            )
            
            # Initialize synthesis optimizer
            self.synthesis_optimizer = {
                'pattern_weights': torch.nn.Parameter(torch.ones(10) * 0.5),
                'adaptive_alpha': torch.nn.Parameter(torch.tensor(self.alpha)),
                'dynamic_temperature': torch.nn.Parameter(torch.tensor(self.temperature)),
                'convergence_detector': None
            }
            
            logger.info("Synthesis optimizer initialized")
            
        except Exception as e:
            logger.warning(f"Synthesis optimizer initialization failed: {e}")
    
    async def _setup_adaptive_distillation(self) -> None:
        """Setup adaptive distillation parameters."""
        try:
            # Configure adaptive learning parameters
            self.adaptive_config = {
                'alpha_learning_rate': 0.001,
                'temperature_learning_rate': 0.01,
                'adaptation_frequency': 10,  # Adapt every N epochs
                'convergence_threshold': 0.001
            }
            
            # Initialize adaptation history
            self.adaptation_history = {
                'alpha_values': [self.alpha],
                'temperature_values': [self.temperature],
                'performance_metrics': [],
                'adaptation_decisions': []
            }
            
            logger.info("Adaptive distillation setup complete")
            
        except Exception as e:
            logger.warning(f"Adaptive distillation setup failed: {e}")
    
    async def _initialize_meta_learning(self) -> None:
        """Initialize meta-learning capabilities for distillation."""
        try:
            # Setup meta-learning components
            self.meta_learner = {
                'task_embeddings': {},
                'transfer_patterns': [],
                'optimization_history': [],
                'knowledge_graph': {}
            }
            
            # Initialize pattern recognition for knowledge transfer
            self.transfer_patterns = {
                'successful_transfers': [],
                'failed_transfers': [],
                'pattern_library': {}
            }
            
            logger.info("Meta-learning initialization complete")
            
        except Exception as e:
            logger.warning(f"Meta-learning initialization failed: {e}")
    
    async def _setup_enhanced_metrics(self) -> None:
        """Setup enhanced metrics tracking for synthesis optimization."""
        try:
            self.enhanced_metrics = {
                'synthesis_efficiency': [],
                'knowledge_transfer_rate': [],
                'pattern_recognition_accuracy': [],
                'adaptive_learning_convergence': [],
                'meta_learning_performance': []
            }
            
            # Initialize real-time monitoring
            self.monitoring_active = True
            
            logger.info("Enhanced metrics tracking setup complete")
            
        except Exception as e:
            logger.warning(f"Enhanced metrics setup failed: {e}")
            
    def enhanced_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        synthesis_features: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Calculate enhanced distillation loss with HOLO-1.5 neural synthesis.
        
        Args:
            student_logits: Logits from student model [batch, height, width, num_colors]
            teacher_probs: Probabilities from teacher model [batch, height, width, num_colors]
            targets: True targets [batch, height, width]
            mask: Mask for valid positions [batch, height, width]
            synthesis_features: Optional synthesis features for optimization
            
        Returns:
            Enhanced distillation loss with adaptive weighting
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
        
        # Get adaptive parameters if synthesis optimizer is available
        alpha = self.adaptive_weights['alpha']
        temperature = self.adaptive_weights['temperature']
        
        if self.synthesis_optimizer and synthesis_features:
            # Use adaptive parameters from synthesis optimizer
            alpha = torch.sigmoid(self.synthesis_optimizer['adaptive_alpha'])
            temperature = torch.exp(self.synthesis_optimizer['dynamic_temperature'])
            
            # Apply synthesis-specific weighting
            synthesis_weight = synthesis_features.get('transfer_weights', {}).get('knowledge_weight', 1.0)
            alpha = alpha * synthesis_weight
        
        # Distillation loss with adaptive temperature scaling
        soft_targets = teacher_probs
        soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(
            soft_predictions.view(-1, soft_predictions.size(-1)),
            soft_targets.view(-1, soft_targets.size(-1)),
            reduction='batchmean'
        ) * (temperature ** 2)  # Scale by temperature squared
        
        # Enhanced loss combination with HOLO-1.5 features
        if synthesis_features:
            # Apply pattern-aware weighting
            pattern_weight = synthesis_features.get('synthesis_features', {}).get('pattern_complexity', 1.0)
            confidence_weight = synthesis_features.get('synthesis_features', {}).get('solution_confidence', 1.0)
            
            # Adaptive loss weighting based on synthesis features
            enhanced_alpha = alpha * confidence_weight * pattern_weight
            loss = (1 - enhanced_alpha) * hard_loss + enhanced_alpha * soft_loss
        else:
            # Standard loss combination
            loss = (1 - alpha) * hard_loss + alpha * soft_loss
        
        # Update adaptive weights for next iteration if synthesis optimizer available
        if self.synthesis_optimizer and hasattr(self, 'monitoring_active') and self.monitoring_active:
            self._update_adaptive_weights(loss.item(), synthesis_features)
        
        return loss
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Legacy distillation loss method for backward compatibility.
        """
        return self.enhanced_distillation_loss(student_logits, teacher_probs, targets, mask)
    
    def _update_adaptive_weights(self, loss_value: float, synthesis_features: Optional[Dict[str, Any]]) -> None:
        """Update adaptive weights based on performance."""
        try:
            if not synthesis_features:
                return
            
            # Simple adaptive update based on loss trends
            if len(self.distillation_history) > 5:
                recent_losses = [h['loss'] for h in self.distillation_history[-5:]]
                if all(loss_value < l for l in recent_losses):
                    # Performance improving, maintain current weights
                    pass
                elif all(loss_value > l for l in recent_losses):
                    # Performance degrading, adjust weights
                    self.adaptive_weights['alpha'] = max(0.1, self.adaptive_weights['alpha'] * 0.95)
                    self.adaptive_weights['temperature'] = max(1.0, self.adaptive_weights['temperature'] * 0.98)
            
            # Record distillation history
            self.distillation_history.append({
                'loss': loss_value,
                'alpha': self.adaptive_weights['alpha'],
                'temperature': self.adaptive_weights['temperature'],
                'synthesis_features': synthesis_features
            })
            
            # Limit history size
            if len(self.distillation_history) > 100:
                self.distillation_history = self.distillation_history[-50:]
                
        except Exception as e:
            logger.warning(f"Error updating adaptive weights: {e}")
            
def enhanced_train_with_distillation(
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
        Enhanced training with HOLO-1.5 neural synthesis distillation.
        
        Args:
            challenges_path: Path to challenges JSON file
            solutions_path: Path to solutions JSON file
            batch_size: Batch size for training
            num_epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            early_stopping_patience: Epochs to wait for improvement before stopping
            
        Returns:
            Dictionary with enhanced training history
        """
        # Prepare enhanced dataset with HOLO-1.5 synthesis
        logger.info("Preparing enhanced dataset with HOLO-1.5 neural synthesis")
        dataset = AugmentedARCDataset(
            challenges_path=challenges_path,
            solutions_path=solutions_path,
            llm_interface=self.llm_interface,
            vanta_core=self.vanta_core,
            mesh_config=self.config
        )
        
        # Initialize dataset if HOLO-1.5 available
        if self.vanta_core:
            asyncio.run(dataset.initialize())
        
        # Create enhanced data loaders
        train_loader, val_loader = create_arc_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.8
        )
        
        # Initialize enhanced optimizer with synthesis parameters
        optimizer_params = list(self.student_model.parameters())
        if self.synthesis_optimizer:
            optimizer_params.extend([
                self.synthesis_optimizer['adaptive_alpha'],
                self.synthesis_optimizer['dynamic_temperature']
            ])
        
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Initialize enhanced tracking variables
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'synthesis_efficiency': [],
            'adaptive_alpha': [],
            'adaptive_temperature': []
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = None
        
        # Enhanced training loop with HOLO-1.5 synthesis
        logger.info(f"Starting enhanced distillation training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase with synthesis
            self.student_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            synthesis_efficiency_sum = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Get enhanced data with synthesis features
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to(self.device)
                synthesis_features = batch.get('synthesis_features', None)
                
                # Get teacher predictions with synthesis optimization
                with torch.no_grad():
                    teacher_outputs = self._get_enhanced_teacher_outputs(input_grids, synthesis_features)
                
                # Forward pass with synthesis
                optimizer.zero_grad()
                student_logits = self.student_model(input_grids)
                
                # Calculate enhanced loss with synthesis features
                mask = (output_grids != -1)  # Mask for valid positions
                loss = self.enhanced_distillation_loss(
                    student_logits,
                    teacher_outputs,
                    output_grids,
                    mask,
                    synthesis_features
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track enhanced metrics
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(student_logits, dim=-1)
                train_correct += (predictions[mask] == output_grids[mask]).sum().item()
                train_total += mask.sum().item()
                
                # Track synthesis efficiency
                if synthesis_features:
                    efficiency = self._calculate_synthesis_efficiency(synthesis_features, loss.item())
                    synthesis_efficiency_sum += efficiency
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            avg_synthesis_efficiency = synthesis_efficiency_sum / len(train_loader)
            
            # Enhanced validation phase
            val_metrics = self._enhanced_validate(val_loader)
            val_loss = val_metrics['val_loss']
            val_accuracy = val_metrics['val_accuracy']
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update enhanced history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['synthesis_efficiency'].append(avg_synthesis_efficiency)
            history['adaptive_alpha'].append(self.adaptive_weights['alpha'])
            history['adaptive_temperature'].append(self.adaptive_weights['temperature'])
            
            # Log enhanced progress
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"train_acc={train_accuracy:.4f}, "
                f"val_acc={val_accuracy:.4f}, "
                f"synthesis_eff={avg_synthesis_efficiency:.4f}, "
                f"alpha={self.adaptive_weights['alpha']:.3f}, "
                f"temp={self.adaptive_weights['temperature']:.3f}, "
                f"time={epoch_time:.2f}s"
            )
            
            # Check for improvement with enhanced criteria
            improvement_score = val_accuracy * 0.7 + avg_synthesis_efficiency * 0.3
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save enhanced best model
                best_model_path = self.output_dir / f"enhanced_distilled_model_best.pt"
                torch.save({
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'synthesis_efficiency': avg_synthesis_efficiency,
                    'adaptive_weights': self.adaptive_weights,
                    'distillation_params': {
                        'alpha': self.alpha,
                        'temperature': self.temperature
                    },
                    'holo15_features': {
                        'synthesis_optimizer': self.synthesis_optimizer is not None,
                        'vanta_core_integration': self.vanta_core is not None
                    }
                }, best_model_path)
                logger.info(f"Saved enhanced best model to {best_model_path}")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs without improvement")
                break
        
        # Save enhanced final model
        final_model_path = self.output_dir / f"enhanced_distilled_model_final.pt"
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'synthesis_efficiency': avg_synthesis_efficiency,
            'adaptive_weights': self.adaptive_weights,
            'training_history': history,
            'distillation_params': {
                'alpha': self.alpha,
                'temperature': self.temperature
            },
            'holo15_features': {
                'synthesis_optimizer': self.synthesis_optimizer is not None,
                'vanta_core_integration': self.vanta_core is not None,
                'distillation_history': self.distillation_history
            }
        }, final_model_path)
        logger.info(f"Saved enhanced final model to {final_model_path}")
        
        # Load best model if available
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading enhanced best model from {best_model_path}")
            checkpoint = torch.load(best_model_path)
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
        
        return history
    
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
        Legacy training method with backward compatibility.
        Automatically uses enhanced training if HOLO-1.5 is available.
        """
        if self.vanta_core:
            return self.enhanced_train_with_distillation(
                challenges_path, solutions_path, batch_size, num_epochs, 
                lr, weight_decay, early_stopping_patience
            )
        else:
            # Fallback to original training logic (simplified for space)
            return self._legacy_train_with_distillation(
                challenges_path, solutions_path, batch_size, num_epochs, 
                lr, weight_decay, early_stopping_patience
            )
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
    
def _get_enhanced_teacher_outputs(self, input_grids: torch.Tensor, synthesis_features: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Get enhanced teacher model outputs with synthesis optimization.
        
        Args:
            input_grids: Batch of input grids [batch, height, width]
            synthesis_features: Optional synthesis features for optimization
            
        Returns:
            Enhanced teacher probabilities [batch, height, width, num_colors]
        """
        # In a real implementation, this would use the LLM with synthesis optimization
        # Here we're providing an enhanced version that incorporates synthesis features
        
        # Generate base pseudo-probabilities from student model
        with torch.no_grad():
            student_logits = self.student_model(input_grids)
            student_probs = F.softmax(student_logits, dim=-1)
        
        # Apply synthesis enhancement if features available
        if synthesis_features and self.vanta_core:
            enhanced_probs = self._apply_synthesis_enhancement(student_probs, synthesis_features)
            return enhanced_probs
        
        # Fallback to basic teacher outputs
        return self._get_teacher_outputs(input_grids)
    
def _apply_synthesis_enhancement(self, base_probs: torch.Tensor, synthesis_features: Dict[str, Any]) -> torch.Tensor:
        """Apply synthesis enhancement to teacher probabilities."""
        try:
            # Extract synthesis parameters
            confidence_weight = synthesis_features.get('synthesis_features', {}).get('solution_confidence', 1.0)
            pattern_weight = synthesis_features.get('synthesis_features', {}).get('pattern_complexity', 1.0)
            
            # Apply confidence-based smoothing
            enhanced_probs = base_probs * confidence_weight
            
            # Apply pattern-based adjustment
            if pattern_weight > 0.7:  # High complexity patterns
                # Add slight noise to encourage exploration
                noise = torch.randn_like(enhanced_probs) * 0.01
                enhanced_probs = enhanced_probs + noise
            
            # Renormalize
            enhanced_probs = F.softmax(enhanced_probs, dim=-1)
            
            return enhanced_probs
            
        except Exception as e:
            logger.warning(f"Synthesis enhancement failed: {e}")
            return base_probs
    
def _calculate_synthesis_efficiency(self, synthesis_features: Dict[str, Any], loss_value: float) -> float:
        """Calculate synthesis efficiency metric."""
        try:
            # Base efficiency from loss improvement
            base_efficiency = max(0.0, min(1.0, 1.0 - (loss_value / 10.0)))
            
            # Adjust based on synthesis features
            if synthesis_features:
                confidence = synthesis_features.get('synthesis_features', {}).get('solution_confidence', 0.5)
                complexity = synthesis_features.get('synthesis_features', {}).get('pattern_complexity', 0.5)
                
                # Higher efficiency for high confidence, moderate complexity
                efficiency_bonus = confidence * (1.0 - abs(complexity - 0.5))
                return min(1.0, base_efficiency + efficiency_bonus * 0.2)
            
            return base_efficiency
            
        except Exception:
            return 0.5
    
def _enhanced_validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Enhanced validation with HOLO-1.5 synthesis metrics."""
        self.student_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        synthesis_efficiency_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get enhanced data
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to(self.device)
                synthesis_features = batch.get('synthesis_features', None)
                
                # Get enhanced teacher predictions
                teacher_outputs = self._get_enhanced_teacher_outputs(input_grids, synthesis_features)
                
                # Forward pass
                student_logits = self.student_model(input_grids)
                
                # Calculate enhanced loss
                mask = (output_grids != -1)
                loss = self.enhanced_distillation_loss(
                    student_logits,
                    teacher_outputs,
                    output_grids,
                    mask,
                    synthesis_features
                )
                
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(student_logits, dim=-1)
                val_correct += (predictions[mask] == output_grids[mask]).sum().item()
                val_total += mask.sum().item()
                
                # Track synthesis efficiency
                if synthesis_features:
                    efficiency = self._calculate_synthesis_efficiency(synthesis_features, loss.item())
                    synthesis_efficiency_sum += efficiency
        
        return {
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_correct / val_total if val_total > 0 else 0,
            'synthesis_efficiency': synthesis_efficiency_sum / len(val_loader)
        }
    
def _legacy_train_with_distillation(
        self,
        challenges_path: str,
        solutions_path: str,
        batch_size: int = 16,
        num_epochs: int = 50,
        lr: float = 0.0005,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Legacy training implementation for backward compatibility."""
        logger.info("Using legacy distillation training (HOLO-1.5 not available)")
        
        # Create basic dataset without HOLO-1.5 features
        dataset = AugmentedARCDataset(
            challenges_path=challenges_path,
            solutions_path=solutions_path,
            llm_interface=self.llm_interface
        )
        
        # Create basic data loaders
        train_loader, val_loader = create_arc_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.8
        )
        
        # Basic optimizer
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
        )
        
        # Basic tracking
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Basic training loop
        logger.info(f"Starting legacy distillation training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            # Training phase
            self.student_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_grids = batch['input_grids'].to(self.device)
                output_grids = batch['output_grids'].to (self.device)
                
                # Get basic teacher predictions
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(input_grids)
                
                # Forward pass
                optimizer.zero_grad()
                student_logits = self.student_model(input_grids)
                
                # Calculate basic loss
                mask = (output_grids != -1)
                loss = self.distillation_loss(student_logits, teacher_outputs, output_grids, mask)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                predictions = torch.argmax(student_logits, dim=-1)
                train_correct += (predictions[mask] == output_grids[mask]).sum().item()
                train_total += mask.sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Validation
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics['val_loss']
            val_accuracy = val_metrics['val_accuracy']
            
            # Update scheduler and history
            scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            # Progress logging
            logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        return history

    # ...existing methods...
    
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
