#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for HOLO-1.5 ARC Ensemble

Exports comprehensive metrics for monitoring GPU RAM, latency, rule violations,
cognitive load, and ensemble performance to Prometheus/Grafana dashboards.

HOLO-1.5 Enhanced Telemetry:
- Recursive symbolic cognition metrics
- Neural-symbolic synthesis monitoring  
- VantaCore mesh collaboration telemetry
- Cognitive load and symbolic depth tracking
"""

import time
import psutil
import logging
import threading
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

try:
    import prometheus_client
    from prometheus_client import Gauge, Counter, Histogram, Info, CollectorRegistry
    from prometheus_client.core import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available. Install with: pip install prometheus-client")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.info("GPUtil not available. GPU metrics will be limited. Install with: pip install GPUtil")

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    collection_interval: float = 10.0  # seconds
    export_port: int = 8000
    export_host: str = "0.0.0.0"
    enable_detailed_traces: bool = True
    max_trace_history: int = 1000
    gpu_monitoring: bool = True
    memory_monitoring: bool = True
    performance_monitoring: bool = True

class HOLO15MetricsCollector:
    """Main metrics collector for HOLO-1.5 system"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.is_collecting = False
        self.collection_thread = None
        
        if not PROMETHEUS_AVAILABLE:
            logger.error("Prometheus client not available. Metrics collection disabled.")
            return
        
        self._initialize_metrics()
        logger.info("HOLO-1.5 metrics collector initialized")
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        
        # === GPU Metrics ===
        self.metrics['gpu_memory_used'] = Gauge(
            'holo15_gpu_memory_used_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.metrics['gpu_memory_total'] = Gauge(
            'holo15_gpu_memory_total_bytes',
            'Total GPU memory in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.metrics['gpu_utilization'] = Gauge(
            'holo15_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.metrics['gpu_temperature'] = Gauge(
            'holo15_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        # === System Memory Metrics ===
        self.metrics['system_memory_used'] = Gauge(
            'holo15_system_memory_used_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.metrics['system_memory_percent'] = Gauge(
            'holo15_system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        # === Performance Metrics ===
        self.metrics['inference_latency'] = Histogram(
            'holo15_inference_latency_seconds',
            'Task inference latency in seconds',
            ['agent_type', 'complexity_level'],
            registry=self.registry
        )
        
        self.metrics['tasks_processed_total'] = Counter(
            'holo15_tasks_processed_total',
            'Total number of tasks processed',
            ['agent_type', 'task_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['ensemble_accuracy'] = Gauge(
            'holo15_ensemble_accuracy_ratio',
            'Current ensemble accuracy ratio',
            ['complexity_level'],
            registry=self.registry
        )
        
        # === Cognitive Load Metrics ===
        self.metrics['cognitive_load'] = Gauge(
            'holo15_cognitive_load',
            'Current cognitive load of the ensemble',
            ['agent_type', 'subsystem'],
            registry=self.registry
        )
        
        self.metrics['symbolic_depth'] = Gauge(
            'holo15_symbolic_depth',
            'Current symbolic reasoning depth',
            ['agent_type', 'reasoning_type'],
            registry=self.registry
        )
        
        self.metrics['mesh_collaboration_score'] = Gauge(
            'holo15_mesh_collaboration_score',
            'VantaCore mesh collaboration effectiveness',
            ['mesh_role'],
            registry=self.registry
        )
        
        # === Rule Violation Metrics ===
        self.metrics['rule_violations_total'] = Counter(
            'holo15_rule_violations_total',
            'Total number of rule violations detected',
            ['rule_type', 'violation_severity'],
            registry=self.registry
        )
        
        self.metrics['logical_consistency_score'] = Gauge(
            'holo15_logical_consistency_score',
            'Logical consistency score (0-1)',
            ['reasoning_component'],
            registry=self.registry
        )
        
        # === Training Metrics ===
        self.metrics['stc_cycles_total'] = Counter(
            'holo15_stc_cycles_total',
            'Total Sleep Training Cycles completed',
            ['status'],
            registry=self.registry
        )
        
        self.metrics['canary_accuracy'] = Gauge(
            'holo15_canary_accuracy_ratio',
            'Canary grid validation accuracy',
            ['pattern_id'],
            registry=self.registry
        )
        
        self.metrics['model_drift_score'] = Gauge(
            'holo15_model_drift_score',
            'Model drift detection score',
            registry=self.registry
        )
        
        # === Component-Specific Metrics ===
        self.metrics['minicache_hit_ratio'] = Gauge(
            'holo15_minicache_hit_ratio',
            'MiniCache hit ratio for memory efficiency',
            registry=self.registry
        )
        
        self.metrics['deltanet_efficiency'] = Gauge(
            'holo15_deltanet_efficiency_ratio',
            'DeltaNet linear attention efficiency',
            registry=self.registry
        )
        
        self.metrics['splr_spike_rate'] = Gauge(
            'holo15_splr_spike_rate_hz',
            'SPLR spiking neural network activity rate',
            registry=self.registry
        )
        
        self.metrics['akorn_oscillation_sync'] = Gauge(
            'holo15_akorn_oscillation_sync_ratio',
            'AKOrN oscillator synchronization ratio',
            registry=self.registry
        )
        
        # === System Health Metrics ===
        self.metrics['system_health_score'] = Gauge(
            'holo15_system_health_score',
            'Overall system health score (0-1)',
            registry=self.registry
        )
        
        self.metrics['uptime_seconds'] = Gauge(
            'holo15_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        # === Info Metrics ===
        self.metrics['build_info'] = Info(
            'holo15_build_info',
            'Build information',
            registry=self.registry
        )
        
        # Set build info
        self.metrics['build_info'].info({
            'version': '1.5.0',
            'build_date': datetime.now(timezone.utc).isoformat(),
            'pytorch_version': torch.__version__ if TORCH_AVAILABLE else 'unknown',
            'cuda_available': str(torch.cuda.is_available()) if TORCH_AVAILABLE else 'unknown'
        })
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Cannot start collection - Prometheus client not available")
            return False
        
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return True
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection (interval: {self.config.collection_interval}s)")
        return True
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread"""
        start_time = time.time()
        
        while self.is_collecting:
            try:
                # Update uptime
                self.metrics['uptime_seconds'].set(time.time() - start_time)
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect GPU metrics
                if self.config.gpu_monitoring:
                    self._collect_gpu_metrics()
                
                # Sleep until next collection
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.config.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics['system_memory_used'].set(memory.used)
            self.metrics['system_memory_percent'].set(memory.percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # PyTorch GPU metrics
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    
                    # Memory metrics
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    self.metrics['gpu_memory_used'].labels(
                        gpu_id=str(i), gpu_name=device_name
                    ).set(memory_allocated)
            
            # Additional GPU metrics via GPUtil if available
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    self.metrics['gpu_memory_used'].labels(
                        gpu_id=str(gpu.id), gpu_name=gpu.name
                    ).set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes
                    
                    self.metrics['gpu_memory_total'].labels(
                        gpu_id=str(gpu.id), gpu_name=gpu.name
                    ).set(gpu.memoryTotal * 1024 * 1024)
                    
                    self.metrics['gpu_utilization'].labels(
                        gpu_id=str(gpu.id), gpu_name=gpu.name
                    ).set(gpu.load * 100)
                    
                    self.metrics['gpu_temperature'].labels(
                        gpu_id=str(gpu.id), gpu_name=gpu.name
                    ).set(gpu.temperature)
                    
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
    
    # === Public API for recording application metrics ===
    
    def record_inference_latency(self, agent_type: str, complexity_level: str, latency: float):
        """Record inference latency"""
        if 'inference_latency' in self.metrics:
            self.metrics['inference_latency'].labels(
                agent_type=agent_type, complexity_level=complexity_level
            ).observe(latency)
    
    def record_task_processed(self, agent_type: str, task_type: str, status: str):
        """Record task processing event"""
        if 'tasks_processed_total' in self.metrics:
            self.metrics['tasks_processed_total'].labels(
                agent_type=agent_type, task_type=task_type, status=status
            ).inc()
    
    def update_cognitive_load(self, agent_type: str, subsystem: str, load: float):
        """Update cognitive load metric"""
        if 'cognitive_load' in self.metrics:
            self.metrics['cognitive_load'].labels(
                agent_type=agent_type, subsystem=subsystem
            ).set(load)
    
    def update_symbolic_depth(self, agent_type: str, reasoning_type: str, depth: int):
        """Update symbolic reasoning depth"""
        if 'symbolic_depth' in self.metrics:
            self.metrics['symbolic_depth'].labels(
                agent_type=agent_type, reasoning_type=reasoning_type
            ).set(depth)
    
    def record_rule_violation(self, rule_type: str, severity: str):
        """Record rule violation"""
        if 'rule_violations_total' in self.metrics:
            self.metrics['rule_violations_total'].labels(
                rule_type=rule_type, violation_severity=severity
            ).inc()
    
    def update_ensemble_accuracy(self, complexity_level: str, accuracy: float):
        """Update ensemble accuracy"""
        if 'ensemble_accuracy' in self.metrics:
            self.metrics['ensemble_accuracy'].labels(complexity_level=complexity_level).set(accuracy)
    
    def update_canary_accuracy(self, pattern_id: str, accuracy: float):
        """Update canary pattern accuracy"""
        if 'canary_accuracy' in self.metrics:
            self.metrics['canary_accuracy'].labels(pattern_id=pattern_id).set(accuracy)
    
    def update_system_health(self, score: float):
        """Update overall system health score"""
        if 'system_health_score' in self.metrics:
            self.metrics['system_health_score'].set(score)
    
    def update_minicache_hit_ratio(self, ratio: float):
        """Update MiniCache hit ratio"""
        if 'minicache_hit_ratio' in self.metrics:
            self.metrics['minicache_hit_ratio'].set(ratio)
    
    def record_stc_cycle(self, status: str):
        """Record Sleep Training Cycle completion"""
        if 'stc_cycles_total' in self.metrics:
            self.metrics['stc_cycles_total'].labels(status=status).inc()
    
    def start_http_server(self) -> bool:
        """Start Prometheus HTTP server for metrics export"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Cannot start HTTP server - Prometheus client not available")
            return False
        
        try:
            prometheus_client.start_http_server(
                self.config.export_port, 
                self.config.export_host,
                registry=self.registry
            )
            logger.info(f"Prometheus metrics server started on {self.config.export_host}:{self.config.export_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {e}")
            return False

# Global metrics collector instance
_global_collector: Optional[HOLO15MetricsCollector] = None

def get_metrics_collector() -> Optional[HOLO15MetricsCollector]:
    """Get the global metrics collector instance"""
    return _global_collector

def initialize_metrics(config: MetricsConfig = None) -> HOLO15MetricsCollector:
    """Initialize global metrics collector"""
    global _global_collector
    
    if config is None:
        config = MetricsConfig()
    
    _global_collector = HOLO15MetricsCollector(config)
    return _global_collector

def start_metrics_server(port: int = 8000, host: str = "0.0.0.0") -> bool:
    """Convenience function to start metrics collection and HTTP server"""
    global _global_collector
    
    if _global_collector is None:
        config = MetricsConfig(export_port=port, export_host=host)
        _global_collector = initialize_metrics(config)
    
    # Start collection
    collection_started = _global_collector.start_collection()
    
    # Start HTTP server
    server_started = _global_collector.start_http_server()
    
    return collection_started and server_started

def main():
    """CLI entry point for metrics exporter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HOLO-1.5 Prometheus Metrics Exporter")
    parser.add_argument('--port', type=int, default=8000, help='HTTP server port')
    parser.add_argument('--host', default='0.0.0.0', help='HTTP server host')
    parser.add_argument('--interval', type=float, default=10.0, help='Collection interval in seconds')
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU monitoring')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create configuration
    config = MetricsConfig(
        collection_interval=args.interval,
        export_port=args.port,
        export_host=args.host,
        gpu_monitoring=not args.disable_gpu
    )
    
    # Initialize and start metrics
    collector = initialize_metrics(config)
    
    if not start_metrics_server(args.port, args.host):
        logger.error("Failed to start metrics server")
        exit(1)
    
    logger.info(f"🔍 HOLO-1.5 Metrics Exporter running...")
    logger.info(f"   HTTP Server: http://{args.host}:{args.port}/metrics")
    logger.info(f"   Collection Interval: {args.interval}s")
    logger.info(f"   GPU Monitoring: {'enabled' if not args.disable_gpu else 'disabled'}")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        collector.stop_collection()

if __name__ == "__main__":
    main()
