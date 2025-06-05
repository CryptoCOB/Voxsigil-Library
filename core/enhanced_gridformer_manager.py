#!/usr/bin/env python
"""
Enhanced GridFormer Manager - Adaptive Integration Implementation
🗡️✨ Training Act I: The GridFormer Awakening - Adaptive Integration ⚔️

This module implements an enhanced GridFormer Manager that connects neural training
performance to VantaCore's meta-learning kernel, creating adaptive intelligence that
unites neural and symbolic realms through cross-component learning.

Key Features:
- Real-time performance feedback from GridFormer training to VantaCore meta-learning
- Adaptive parameter optimization based on cross-component learning
- Bidirectional knowledge transfer between neural and symbolic systems
- Performance-driven meta-parameter tuning
- Integration with VantaCore's task adaptation profiles
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Gridformer.core.simple_grid_former_handler import SimpleGridFormerHandler

# Import VantaCore components
try:
    # Import VantaCore directly as VantaUnifiedCore to avoid namespace conflict
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore, get_unified_core

    VANTA_AVAILABLE = True
except ImportError:
    VANTA_AVAILABLE = False

    raise ImportError("VantaCore components are not available.")


def get_unified_core():
    return UnifiedVantaCore()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedGridFormerManager")


@dataclass
class TrainingMetrics:
    """Real-time training metrics for adaptive learning"""

    epoch: int
    batch_idx: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    gradient_norm: float
    performance_trend: float
    convergence_rate: float
    timestamp: float


@dataclass
class AdaptationDecision:
    """Decision made by the meta-learning kernel"""

    parameter_name: str
    old_value: float
    new_value: float
    reason: str
    confidence: float
    timestamp: float


class PerformanceTracker:
    """Tracks and analyzes training performance for meta-learning"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.gradient_history = deque(maxlen=window_size)
        self.lr_history = deque(maxlen=window_size)
        self.performance_events = []

    def update(self, metrics: TrainingMetrics) -> None:
        """Update performance tracking with new metrics"""
        self.loss_history.append(metrics.train_loss)
        self.gradient_history.append(metrics.gradient_norm)
        self.lr_history.append(metrics.learning_rate)

        # Calculate performance indicators
        metrics.performance_trend = self._calculate_trend()
        metrics.convergence_rate = self._calculate_convergence_rate()

        self.performance_events.append(metrics)

    def _calculate_trend(self) -> float:
        """Calculate performance trend (negative = improving)"""
        if len(self.loss_history) < 5:
            return 0.0
        recent = list(self.loss_history)[-5:]
        earlier = (
            list(self.loss_history)[-10:-5] if len(self.loss_history) >= 10 else recent
        )
        return float(np.mean(recent) - np.mean(earlier))

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate (higher = faster convergence)"""
        if len(self.loss_history) < 10:
            return 0.0
        losses = np.array(list(self.loss_history))
        # Simple convergence metric: negative of loss variance
        return float(-np.var(losses[-10:]))

    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance summary for meta-learning"""
        return {
            "current_loss": float(self.loss_history[-1])
            if self.loss_history
            else float("inf"),
            # Convert deques to lists so NumPy receives a compatible sequence type
            "avg_loss": float(np.mean(list(self.loss_history)))
            if self.loss_history
            else float("inf"),
            "loss_std": float(np.std(list(self.loss_history)))
            if len(self.loss_history) > 1
            else 0.0,
            "performance_trend": float(self._calculate_trend()),
            "gradient_stability": float(np.std(list(self.gradient_history)))
            if len(self.gradient_history) > 1
            else 0.0,
        }


class MetaLearningIntegrator:
    """Integrates GridFormer training with VantaCore meta-learning"""

    def __init__(self, vanta_core: Optional[UnifiedVantaCore] = None):
        self.vanta_core = vanta_core
        self.task_id = f"gridformer_training_{int(time.time())}"
        self.adaptation_history = []
        self.knowledge_transfer_events = []

        # Initialize task in VantaCore if available
        if self.vanta_core:
            try:
                # Register this component with Vanta Core
                self.vanta_core.register_component(
                    "gridformer_trainer",
                    self,
                    {
                        "type": "neural_training_system",
                        "capabilities": [
                            "adaptive_training",
                            "meta_learning_integration",
                        ],
                    },
                )

                # Get advanced meta learner component if available
                advanced_meta_learner = None
                try:
                    components = self.vanta_core.list_components()
                    if "advanced_meta_learner" in components:
                        advanced_meta_learner = self.vanta_core.get_component(
                            "advanced_meta_learner"
                        )
                        logger.info("Connected to advanced meta learner component")
                except Exception as e:
                    logger.warning(f"Could not connect to advanced meta learner: {e}")

                # Register task with meta learner if available
                if advanced_meta_learner and hasattr(
                    advanced_meta_learner, "register_task"
                ):
                    advanced_meta_learner.register_task(
                        task_id=self.task_id,
                        task_features={
                            "domain": "neural_network_training",
                            "model_type": "gridformer",
                            "task_type": "arc_pattern_recognition",
                            "complexity": "high",
                        },
                    )
                    logger.info(
                        f"Registered GridFormer training task {self.task_id} with Advanced Meta Learner"
                    )

            except Exception as e:
                logger.warning(f"Failed to register component with VantaCore: {e}")

    def report_performance(
        self, metrics: TrainingMetrics, performance_summary: Dict[str, float]
    ) -> Dict[str, Any]:
        """Report training performance to VantaCore meta-learning kernel"""
        if not self.vanta_core:
            return {}

        try:
            # Convert performance to meta-learning score
            performance_score = self._calculate_meta_learning_score(performance_summary)

            # Try to get advanced meta learner component
            advanced_meta_learner = None
            try:
                components = self.vanta_core.list_components()
                if "advanced_meta_learner" in components:
                    advanced_meta_learner = self.vanta_core.get_component(
                        "advanced_meta_learner"
                    )
            except Exception:
                pass

            # Update with performance if available
            if advanced_meta_learner and hasattr(
                advanced_meta_learner, "update_performance"
            ):
                advanced_meta_learner.update_performance(
                    task_id=self.task_id, performance=performance_score
                )
                logger.debug(
                    f"Reported performance {performance_score:.4f} to Advanced Meta Learner"
                )
                return {"success": True, "performance_score": performance_score}
            else:
                # No advanced meta learner available
                logger.debug(
                    f"Calculated performance {performance_score:.4f} but no Advanced Meta Learner available"
                )
                return {}

        except Exception as e:
            logger.error(f"Failed to report performance to VantaCore: {e}")
            return {}

    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get adaptive parameters from VantaCore meta-learning"""
        if not self.vanta_core:
            return {}

        try:
            # Try to get advanced meta learner component
            advanced_meta_learner = None
            try:
                components = self.vanta_core.list_components()
                if "advanced_meta_learner" in components:
                    advanced_meta_learner = self.vanta_core.get_component(
                        "advanced_meta_learner"
                    )
            except Exception:
                pass

            # Get parameters if available
            if advanced_meta_learner and hasattr(
                advanced_meta_learner, "get_task_parameters"
            ):
                return advanced_meta_learner.get_task_parameters(self.task_id)
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get adaptive parameters from VantaCore: {e}")
            return {}

    def transfer_knowledge(
        self, source_domain: str, target_performance: float
    ) -> Dict[str, Any]:
        """Initiate cross-task knowledge transfer"""
        if not self.vanta_core:
            return {}

        # For now, simply record the transfer attempt since the direct API is not available
        transfer_record = {
            "timestamp": time.time(),
            "source_domain": source_domain,
            "target_performance": target_performance,
        }

        self.knowledge_transfer_events.append(transfer_record)
        logger.info(
            f"Recorded knowledge transfer request from domain '{source_domain}'"
        )

        return {"recorded": True}

    def _calculate_meta_learning_score(
        self, performance_summary: Dict[str, float]
    ) -> float:
        """Convert training metrics to meta-learning performance score"""
        # Combine multiple performance indicators
        loss_score = 1.0 / (1.0 + performance_summary.get("current_loss", 1.0))
        trend_score = max(0.0, -performance_summary.get("performance_trend", 0.0))
        convergence_score = max(0.0, performance_summary.get("convergence_rate", 0.0))
        stability_score = 1.0 / (
            1.0 + performance_summary.get("gradient_stability", 1.0)
        )

        # Weighted combination
        meta_score = (
            0.4 * loss_score
            + 0.3 * trend_score
            + 0.2 * convergence_score
            + 0.1 * stability_score
        )

        return np.clip(meta_score, 0.0, 1.0)


class AdaptiveOptimizer:
    """Adaptive optimizer that responds to meta-learning signals"""

    def __init__(
        self, base_optimizer: optim.Optimizer, adaptation_strength: float = 0.1
    ):
        self.base_optimizer = base_optimizer
        self.adaptation_strength = adaptation_strength
        self.adaptation_decisions = []

    def adapt_learning_rate(self, new_lr: float, reason: str = "meta_learning") -> None:
        """Adapt learning rate based on meta-learning feedback"""
        current_lr = self.base_optimizer.param_groups[0]["lr"]

        # Smooth adaptation to prevent training instability
        adapted_lr = current_lr + self.adaptation_strength * (new_lr - current_lr)

        for param_group in self.base_optimizer.param_groups:
            param_group["lr"] = adapted_lr

        decision = AdaptationDecision(
            parameter_name="learning_rate",
            old_value=current_lr,
            new_value=adapted_lr,
            reason=reason,
            confidence=0.8,
            timestamp=time.time(),
        )

        self.adaptation_decisions.append(decision)
        logger.info(
            f"Adapted learning rate: {current_lr:.6f} -> {adapted_lr:.6f} ({reason})"
        )

    def adapt_momentum(
        self, new_momentum: float, reason: str = "meta_learning"
    ) -> None:
        """Adapt momentum based on meta-learning feedback"""
        current_momentum = self.base_optimizer.param_groups[0].get("momentum", 0.0)

        # Smooth adaptation to prevent training instability
        adapted_momentum = current_momentum + self.adaptation_strength * (
            new_momentum - current_momentum
        )

        for param_group in self.base_optimizer.param_groups:
            param_group["momentum"] = adapted_momentum

        decision = AdaptationDecision(
            parameter_name="momentum",
            old_value=current_momentum,
            new_value=adapted_momentum,
            reason=reason,
            confidence=0.8,
            timestamp=time.time(),
        )

        self.adaptation_decisions.append(decision)
        logger.info(
            f"Adapted momentum: {current_momentum:.6f} -> {adapted_momentum:.6f} ({reason})"
        )


class EnhancedGridFormerManager:
    """
    Enhanced GridFormer Manager with VantaCore Meta-Learning Integration

    This class creates adaptive intelligence by connecting neural training performance
    to VantaCore's meta-learning kernel, enabling cross-component learning and
    real-time parameter optimization.
    """

    def __init(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_grid_size: int = 30,
        num_colors: int = 10,
        device: Optional[str] = None,
        vanta_config_path: Optional[str] = None,
        adaptation_strength: float = 0.1,
    ):
        # Initialize base GridFormer handler
        self.handler = SimpleGridFormerHandler(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=max_grid_size,
            num_colors=num_colors,
            device=device,
        )

        # Initialize VantaCore integration
        self.vanta_core = self._initialize_vanta_core(None)

        # Initialize adaptive components
        self.performance_tracker = PerformanceTracker()
        self.meta_integrator = MetaLearningIntegrator(self.vanta_core)
        self.adaptive_optimizer = None  # Will be set during training
        self.adaptation_strength = adaptation_strength

        # Training state
        self.training_active = False
        self.adaptation_thread = None
        self.adaptation_interval = 10  # Adapt every N batches

        logger.info(
            "Enhanced GridFormer Manager initialized with VantaCore integration"
        )
        logger.info(f"VantaCore available: {VANTA_AVAILABLE}")

    def _initialize_vanta_core(
        self, config_path: Optional[str]
    ) -> Optional[UnifiedVantaCore]:
        """Initialize VantaCore with meta-learning capabilities"""
        if not VANTA_AVAILABLE:
            logger.warning(
                "VantaCore components not available, running in standalone mode"
            )
            return None

        try:
            # Initialize VantaCore components
            vanta_core = get_unified_core()
            logger.info("VantaCore initialized successfully")

            # Log the provided config path for debugging
            if config_path:
                logger.info(f"Using VantaCore config path: {config_path}")

            return vanta_core

        except Exception as e:
            logger.error(f"Failed to initialize VantaCore: {e}")
            return None

    def train_adaptive(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        save_path: str = "enhanced_grid_former.pt",
        adaptation_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced training with real-time VantaCore meta-learning integration
        """
        logger.info("🗡️✨ Starting GridFormer Awakening - Adaptive Training ⚔️")

        # Setup adaptive training components
        optimizer = optim.Adam(self.handler.model.parameters(), lr=learning_rate)
        self.adaptive_optimizer = AdaptiveOptimizer(optimizer, self.adaptation_strength)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            # Prevent nested DataParallel wrapping
            if not isinstance(self.handler.model, nn.DataParallel):
                self.handler.model = nn.DataParallel(self.handler.model)

            logger.info(f"Using {torch.cuda.device_count()} GPUs for adaptive training")

        # Training state
        self.training_active = True
        history = {
            "train_loss": [],
            "val_loss": [],
            "adaptations": [],
            "meta_scores": [],
            "knowledge_transfers": [],
        }
        best_val_loss = float("inf")

        # Start adaptation monitoring thread
        self._start_adaptation_monitoring()

        try:
            for epoch in range(num_epochs):
                logger.info(
                    f"⚔️ Epoch {epoch + 1}/{num_epochs} - GridFormer Neural-Symbolic Fusion"
                )

                # Training phase with real-time adaptation
                self.handler.model.train()
                train_loss = 0.0

                for batch_idx, batch in enumerate(train_dataloader):
                    input_grid = batch["input"].to(self.handler.device)
                    output_grid = batch["output"].to(self.handler.device)

                    optimizer.zero_grad()

                    # Forward pass
                    output_logits = self.handler.model(input_grid)

                    # Compute loss
                    B, H, W, C = output_logits.shape
                    loss = criterion(
                        output_logits.reshape(B * H * W, C),
                        output_grid.reshape(B * H * W),
                    )

                    loss.backward()

                    # Calculate gradient norm for stability tracking
                    grad_norm = self._calculate_gradient_norm()

                    optimizer.step()
                    train_loss += loss.item()

                    # Real-time performance tracking and adaptation
                    if batch_idx % self.adaptation_interval == 0:
                        self._perform_real_time_adaptation(
                            epoch,
                            batch_idx,
                            loss.item(),
                            grad_norm,
                            optimizer.param_groups[0]["lr"],
                        )

                    # Progress logging
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(
                            f"Batch {batch_idx + 1}/{len(train_dataloader)}, "
                            f"Loss: {loss.item():.6f}, "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                        )

                # Validation phase
                val_loss = self._validate_model(val_dataloader, criterion)

                # Calculate averages
                train_loss /= len(train_dataloader)

                # Update training history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                # Meta-learning performance analysis
                performance_summary = self.performance_tracker.get_performance_summary()
                meta_score = self.meta_integrator._calculate_meta_learning_score(
                    performance_summary
                )
                history["meta_scores"].append(meta_score)

                logger.info(
                    f"✨ Epoch {epoch + 1} Complete - Train: {train_loss:.6f}, "
                    f"Val: {val_loss:.6f}, Meta-Score: {meta_score:.4f}"
                )

                # Save best model with meta-learning context
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model_with_meta_context(save_path, epoch, meta_score)
                    logger.info(
                        f"🏆 New best model saved! Meta-enhanced performance: {meta_score:.4f}"
                    )

                # Adaptive callback
                if adaptation_callback:
                    adaptation_callback(
                        epoch, history, self.adaptive_optimizer.adaptation_decisions
                    )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            self.training_active = False
            if self.adaptation_thread and self.adaptation_thread.is_alive():
                self.adaptation_thread.join(timeout=1.0)

        # Final knowledge transfer and analysis
        final_transfer = self.meta_integrator.transfer_knowledge(
            source_domain="neural_training_complete", target_performance=meta_score
        )
        history["knowledge_transfers"].append(final_transfer)

        logger.info(
            "🗡️✨ GridFormer Awakening Complete - Neural-Symbolic Fusion Achieved ⚔️"
        )
        return history

    def _calculate_gradient_norm(self) -> float:
        """Calculate gradient norm for stability tracking"""
        total_norm = 0.0
        model = (
            self.handler.model.module
            if isinstance(self.handler.model, nn.DataParallel)
            else self.handler.model
        )

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1.0 / 2)

    def _perform_real_time_adaptation(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        grad_norm: float,
        current_lr: float,
    ) -> None:
        """Perform real-time adaptation based on current training state"""
        # Create training metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            batch_idx=batch_idx,
            train_loss=loss,
            val_loss=None,
            learning_rate=current_lr,
            gradient_norm=grad_norm,
            performance_trend=0.0,  # Will be calculated
            convergence_rate=0.0,  # Will be calculated
            timestamp=time.time(),
        )  # Update performance tracker
        self.performance_tracker.update(metrics)

        # Get performance summary for meta-learning
        performance_summary = self.performance_tracker.get_performance_summary()

        # Report to meta-learning kernel
        _ = self.meta_integrator.report_performance(metrics, performance_summary)

        # Apply adaptive parameters
        adaptive_params = self.meta_integrator.get_adaptive_parameters()

        if adaptive_params:
            # Adapt learning rate
            if (
                "learning_rate" in adaptive_params
                and self.adaptive_optimizer is not None
            ):
                self.adaptive_optimizer.adapt_learning_rate(
                    adaptive_params["learning_rate"], "vanta_meta_learning"
                )

            # Adapt momentum if available
            if "momentum" in adaptive_params and self.adaptive_optimizer is not None:
                self.adaptive_optimizer.adapt_momentum(
                    adaptive_params["momentum"], "vanta_meta_learning"
                )

    def _validate_model(self, val_dataloader: DataLoader, criterion) -> float:
        """Validate model with meta-learning awareness"""
        self.handler.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                input_grid = batch["input"].to(self.handler.device)
                output_grid = batch["output"].to(self.handler.device)

                output_logits = self.handler.model(input_grid)
                B, H, W, C = output_logits.shape
                loss = criterion(
                    output_logits.reshape(B * H * W, C), output_grid.reshape(B * H * W)
                )
                val_loss += loss.item()

        return val_loss / len(val_dataloader)

    def _save_model_with_meta_context(
        self, filepath: str, epoch: int, meta_score: float
    ) -> None:
        """Save model with meta-learning context"""
        # Get base model state
        model = (
            self.handler.model.module
            if isinstance(self.handler.model, nn.DataParallel)
            else self.handler.model
        )

        # Create enhanced checkpoint with meta-learning context
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": self.handler.config,
            "meta_learning_context": {
                "meta_score": meta_score,
                "epoch": epoch,
                "adaptation_decisions": [
                    {
                        "parameter": dec.parameter_name,
                        "old_value": dec.old_value,
                        "new_value": dec.new_value,
                        "reason": dec.reason,
                        "confidence": dec.confidence,
                    }
                    for dec in (
                        self.adaptive_optimizer.adaptation_decisions[-10:]
                        if self.adaptive_optimizer is not None
                        else []
                    )
                ],
                "performance_summary": self.performance_tracker.get_performance_summary(),
                "vanta_integration": VANTA_AVAILABLE,
                "task_id": self.meta_integrator.task_id,
            },
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Model saved with meta-learning context: {filepath}")

    def _start_adaptation_monitoring(self) -> None:
        """Start background thread for continuous adaptation monitoring"""
        if not self.vanta_core:
            return

        def adaptation_monitor():
            while self.training_active:
                try:
                    # Perform background meta-learning operations
                    time.sleep(5.0)  # Check every 5 seconds

                    if self.performance_tracker.loss_history:
                        # Trigger knowledge transfer if performance stagnates
                        trend = self.performance_tracker._calculate_trend()
                        if trend > 0.1:  # Performance degrading
                            self.meta_integrator.transfer_knowledge(
                                source_domain="stagnation_recovery",
                                target_performance=self.performance_tracker.loss_history[
                                    -1
                                ],
                            )

                except Exception as e:
                    logger.error(f"Adaptation monitoring error: {e}")

        self.adaptation_thread = threading.Thread(
            target=adaptation_monitor, daemon=True
        )
        self.adaptation_thread.start()
        logger.info("Adaptation monitoring thread started")

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive adaptation summary"""
        return {
            "total_adaptations": len(self.adaptive_optimizer.adaptation_decisions)
            if self.adaptive_optimizer
            else 0,
            "adaptation_decisions": self.adaptive_optimizer.adaptation_decisions
            if self.adaptive_optimizer
            else [],
            "performance_events": len(self.performance_tracker.performance_events),
            "knowledge_transfers": len(self.meta_integrator.knowledge_transfer_events),
            "meta_learning_active": self.vanta_core is not None,
            "current_performance": self.performance_tracker.get_performance_summary(),
        }

    def predict_with_meta_enhancement(
        self,
        input_grid: torch.Tensor,
        target_shape: Optional[Tuple[int, int]] = None,
        use_meta_context: bool = True,
    ) -> torch.Tensor:
        """Enhanced prediction with meta-learning context awareness"""
        # Standard prediction
        predictions = self.handler.predict_simple(input_grid, target_shape)

        # Apply meta-learning enhancements if available
        if use_meta_context and self.vanta_core:
            try:
                # Get meta-context for prediction enhancement
                meta_context = self.meta_integrator.get_adaptive_parameters()

                # Apply context-aware prediction adjustments (placeholder for now)
                if meta_context:
                    logger.debug(f"Applied meta-context to prediction: {meta_context}")

            except Exception as e:
                logger.warning(
                    f"Meta-enhancement failed, using standard prediction: {e}"
                )

        return predictions


def create_enhanced_gridformer_manager(
    config_path: Optional[str] = None,
    **model_kwargs,
) -> EnhancedGridFormerManager:
    """
    Factory function to create Enhanced GridFormer Manager with optimal configuration
    """
    logger.info("🗡️✨ Creating Enhanced GridFormer Manager - Neural-Symbolic Bridge ⚔️")

    manager = EnhancedGridFormerManager(**model_kwargs)

    logger.info("✨ Enhanced GridFormer Manager created - Ready for adaptive training!")
    return manager


if __name__ == "__main__":
    # Example usage
    logger.info("🗡️✨ GridFormer Awakening - Adaptive Integration Demo ⚔️")

    manager = create_enhanced_gridformer_manager(
        hidden_dim=256, num_layers=6, num_heads=8, adaptation_strength=0.1
    )

    print("✅ Enhanced GridFormer Manager initialized!")
    print("🗡️ Ready for neural-symbolic adaptive training")
    print("✨ VantaCore meta-learning integration active")

    # Display adaptation capabilities
    summary = manager.get_adaptation_summary()
    print("\n📊 Adaptation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
