# CONSOLIDATED - EvolutionaryOptimizer moved to evolutionary_optimizer.py
# This file now contains only NeuralArchitectureSearch class

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import amp
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# HOLO-1.5 imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# -------------------------------------------------------------------------- #
#  Environment / logging
# -------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("EVO-NAS")

Path("logs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
writer = SummaryWriter("runs/evo_nas")

# -------------------------------------------------------------------------- #
#  Neural Architecture Search - Evolutionary approach to neural architecture discovery
# -------------------------------------------------------------------------- #
@vanta_core_module(
    name="neural_architecture_search",
    subsystem="optimization",
    mesh_role=CognitiveMeshRole.SYNTHESIZER,
    description="Evolutionary neural architecture search with genetic algorithms and adaptive optimization",
    capabilities=["architecture_evolution", "neural_search", "genetic_optimization", "model_synthesis", "performance_evaluation"],
    cognitive_load=4.5,
    symbolic_depth=4,
    collaboration_patterns=["evolutionary_synthesis", "adaptive_optimization", "architecture_discovery"]
)
class NeuralArchitectureSearch(BaseCore, nn.Module):
    def __init__(self,
                 vanta_core,
                 config: Optional[Dict[str, Any]] = None,
                 in_dim:  int = 784,
                 out_dim: int = 10,
                 device:  Optional[torch.device] = None):
        BaseCore.__init__(self, vanta_core, config or {})
        nn.Module.__init__(self)

        self.in_dim, self.out_dim = in_dim, out_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng    = random.Random()
        self.best   = []
        self.scaler = amp.GradScaler(enabled=self.device.type == "cuda")
        self.build_default()

    async def initialize(self) -> bool:
        """Initialize the Neural Architecture Search system."""
        log.info("NeuralArchitectureSearch initialized with HOLO-1.5 enhancement")
        return True

    def build_default(self):
        """Build a default neural network architecture."""
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64,  self.out_dim)
        ).to(self.device)

    # ---------- DNA ops -------------------------------------------------- #
    def propose(self) -> Dict:
        """Propose a new neural architecture configuration."""
        h = self.rng.choice([32, 64, 128, 256])
        return {
            "l1": {"type": "Linear", "in_features": self.in_dim, "out_features": h},
            "a1": {"type": "ReLU"},
            "l2": {"type": "Linear", "in_features": h, "out_features": h // 2},
            "a2": {"type": "ReLU"},
            "hd": {"type": "Linear", "in_features": h // 2, "out_features": self.out_dim}
        }

    def mutate(self, arch: Dict, p: float = .3) -> Dict:
        """Mutate an architecture configuration."""
        child = json.loads(json.dumps(arch))
        for cfg in child.values():
            if cfg.get("type") == "Linear" and self.rng.random() < p:
                delta = self.rng.choice([-32, -16, 16, 32])
                cfg["out_features"] = max(8, cfg["out_features"] + delta)
        return child

    def crossover(self, a1: Dict, a2: Dict) -> Dict:
        """Perform crossover between two architectures."""
        return {k: self.rng.choice([a1[k], a2[k]]) for k in a1}

    def _model_from_arch(self, arch: Dict) -> nn.Module:
        """Build a PyTorch model from architecture configuration."""
        layers = []
        for cfg in arch.values():
            t = cfg["type"]
            if t == "Linear":
                layers.append(nn.Linear(cfg["in_features"], cfg["out_features"]))
            elif t == "ReLU":
                layers.append(nn.ReLU())
        return nn.Sequential(*layers).to(self.device)

    def evaluate(self,
                 arch:  Dict,
                 data:  Tuple[Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor, torch.Tensor]],
                 epochs: int = 3) -> float:
        """Evaluate an architecture on given data."""
        (x_tr, y_tr), (x_va, y_va) = data
        mdl   = self._model_from_arch(arch)
        optim = Adam(mdl.parameters(), lr=1e-3)
        crit  = nn.MSELoss()
        tr_loader = DataLoader(TensorDataset(x_tr, y_tr), 64, shuffle=True)

        mdl.train()
        for _ in range(epochs):
            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                with amp.autocast(device_type=self.device.type):
                    loss = crit(mdl(xb), yb)
                self.scaler.scale(loss).backward()
                self.scaler.step(optim)
                self.scaler.update()

        mdl.eval()
        with torch.no_grad():
            val_loss = crit(mdl(x_va.to(self.device)), y_va.to(self.device)).item()

        self.best.append((val_loss, arch))
        self.best.sort(key=lambda t: t[0])
        self.best = self.best[:5]
        return val_loss

# -------------------------------------------------------------------------- #
#  NOTE: EvolutionaryOptimizer implementation consolidated to evolutionary_optimizer.py
#  For evolutionary optimization functionality, import from that module:
#  from .evolutionary_optimizer import EvolutionaryOptimizer
# -------------------------------------------------------------------------- #
