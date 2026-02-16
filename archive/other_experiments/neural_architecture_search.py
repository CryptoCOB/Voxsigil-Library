import logging
import random
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from modules.blt import get_blt_instance


class SearchSpace:
    """Simple search space definition for NAS."""

    HIDDEN_SIZES = [32, 64, 128]
    ACTIVATIONS = [nn.ReLU, nn.Tanh]
    NUM_LAYERS = [1, 2, 3]


class ArchitectureSampler:
    """Randomly samples architectures from a fixed search space."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def sample(self) -> Dict[str, Any]:
        layers: List[Dict[str, Any]] = []
        in_dim = self.input_dim
        for _ in range(random.choice(SearchSpace.NUM_LAYERS)):
            hidden = random.choice(SearchSpace.HIDDEN_SIZES)
            act = random.choice(SearchSpace.ACTIVATIONS)
            layers.append({"in": in_dim, "out": hidden, "activation": act})
            in_dim = hidden
        layers.append({"in": in_dim, "out": self.output_dim, "activation": None})
        return {"layers": layers}


class NASOptimizer:
    """Minimal neural architecture search optimizer."""

    def __init__(self, input_dim: int, output_dim: int, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sampler = ArchitectureSampler(input_dim, output_dim)
        self.best_arch: Dict[str, Any] | None = None
        self.best_score: float = -float("inf")
        self.logger = logging.getLogger("NASOptimizer")
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _create_model(self, arch: Dict[str, Any]) -> nn.Module:
        layers: List[nn.Module] = []
        for spec in arch["layers"]:
            layers.append(nn.Linear(spec["in"], spec["out"]))
            if spec["activation"]:
                layers.append(spec["activation"]())
        return nn.Sequential(*layers)

    def _evaluate(self, arch: Dict[str, Any]) -> float:
        model = self._create_model(arch).to(self.device)
        x = torch.randn(32, self.input_dim, device=self.device)
        y = torch.randint(0, self.output_dim, (32,), device=self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(x)
            score = torch.mean(torch.softmax(out, dim=1).max(dim=1).values).item()
        return score

    def search_step(self) -> Dict[str, Any]:
        arch = self.sampler.sample()
        score = self._evaluate(arch)
        if score > self.best_score:
            self.best_score = score
            self.best_arch = arch
        return {"architecture": arch, "score": score, "best_score": self.best_score}


class NASManager:
    """Manages NASOptimizer and integration hooks with enhanced BLT compression."""

    def __init__(self, input_dim: int, output_dim: int):
        self.optimizer = NASOptimizer(input_dim, output_dim)
        self.logger = logging.getLogger("NASManager")
        
        # Use global BLT instance for system-wide optimization
        self.blt = get_blt_instance()
        
        # Track NAS-specific metrics
        self.architecture_cache = {}
        self.compression_stats = {'total_architectures': 0, 'cache_hits': 0}

    def run_generation(self, steps: int = 1):
        """Run NAS generation with enhanced BLT compression and caching."""
        results = []
        
        for step in range(steps):
            # Perform search step
            res = self.optimizer.search_step()
            
            # Create architecture signature for caching
            arch_signature = str(sorted(res.items()))
            arch_hash = hash(arch_signature)
            
            # Check cache first
            if arch_hash in self.architecture_cache:
                self.compression_stats['cache_hits'] += 1
                cached_result = self.architecture_cache[arch_hash].copy()
                cached_result['cache_hit'] = True
                cached_result['step'] = step
                results.append(cached_result)
                continue
            
            # Compress architecture using enhanced BLT
            arch_bytes = arch_signature.encode('utf-8')
            compressed = self.blt.encode(arch_bytes)
            
            # Store compressed architecture with metadata
            enhanced_result = {
                **res,
                'compressed_architecture': compressed,
                'compression_ratio': len(arch_bytes) / len(compressed) if compressed else 1.0,
                'blt_latency': self.blt.latency_score(),
                'step': step,
                'cache_hit': False
            }
            
            # Cache for future use
            self.architecture_cache[arch_hash] = enhanced_result
            self.compression_stats['total_architectures'] += 1
            
            # Limit cache size
            if len(self.architecture_cache) > 1000:
                # Remove oldest 100 entries
                oldest_keys = list(self.architecture_cache.keys())[:100]
                for key in oldest_keys:
                    del self.architecture_cache[key]
            
            results.append(enhanced_result)
            
            # Optimize BLT periodically
            if step % 10 == 0:
                self.blt.optimize_for_latency()
        
        # Log performance statistics
        blt_state = self.blt.get_state()
        cache_hit_rate = self.compression_stats['cache_hits'] / max(steps, 1)
        
        self.logger.info(f"NAS Generation Complete - Steps: {steps}, "
                        f"Cache Hit Rate: {cache_hit_rate:.2%}, "
                        f"BLT Compression Ratio: {blt_state['compression_ratio']:.2f}, "
                        f"Latency Score: {blt_state['latency_score']:.3f}")
        
        return results
    
    def get_nas_performance(self) -> Dict[str, Any]:
        """Get comprehensive NAS performance metrics with BLT statistics."""
        blt_state = self.blt.get_state()
        
        return {
            'architecture_cache_size': len(self.architecture_cache),
            'total_architectures_processed': self.compression_stats['total_architectures'],
            'cache_hit_rate': self.compression_stats['cache_hits'] / max(self.compression_stats['total_architectures'], 1),
            'blt_metrics': {
                'latency_score': blt_state['latency_score'],
                'compression_ratio': blt_state['compression_ratio'],
                'avg_latency': blt_state['avg_latency'],
                'total_operations': blt_state['total_operations']
            }
        }
