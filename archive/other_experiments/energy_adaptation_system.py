#!/usr/bin/env python3
"""
Energy Adaptation System Loading Integration
Implements dynamic system_load feedback integration for energy-aware processing across Nebula
"""

import os
import sys
import time
import psutil
import torch
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SystemLoadLevel(Enum):
    """System load levels for adaptive processing"""
    IDLE = "idle"           # < 25% CPU, abundant resources
    LIGHT = "light"         # 25-50% CPU, good resources  
    MODERATE = "moderate"   # 50-70% CPU, moderate resources
    HEAVY = "heavy"         # 70-85% CPU, limited resources
    CRITICAL = "critical"   # > 85% CPU, resource constrained

class ProcessingMode(Enum):
    """Processing modes based on energy/resource availability"""
    FULL_POWER = "full_power"           # All optimizations disabled, maximum quality
    BALANCED = "balanced"               # Standard mode with basic optimizations
    ENERGY_SAVER = "energy_saver"      # Reduced batch sizes, simplified models
    MINIMAL = "minimal"                 # Essential processing only
    HIBERNATION = "hibernation"        # Only critical operations

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_memory_percent: float = 0.0
    gpu_memory_available_gb: float = 0.0
    gpu_temperature: float = 0.0
    system_load_level: SystemLoadLevel = SystemLoadLevel.LIGHT
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    timestamp: float = 0.0
    
@dataclass
class AdaptationConfig:
    """Configuration for adaptive processing"""
    cpu_threshold_light: float = 25.0
    cpu_threshold_moderate: float = 50.0
    cpu_threshold_heavy: float = 70.0
    cpu_threshold_critical: float = 85.0
    memory_threshold_critical_gb: float = 2.0
    gpu_memory_threshold_critical_percent: float = 90.0
    adaptation_response_time_s: float = 5.0
    monitoring_interval_s: float = 2.0

class EnergyAdaptationManager:
    """Manages system load monitoring and adaptive processing responses"""
    
    def __init__(self, config: Optional[AdaptationConfig] = None, logger=None):
        self.config = config or AdaptationConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # System monitoring
        self.current_metrics = SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_available_gb=0.0,
            timestamp=time.time()
        )
        
        # Historical metrics for trend analysis
        self.metrics_history = deque(maxlen=60)  # Keep 2 minutes of history at 2s intervals
        
        # Adaptation state
        self.current_processing_mode = ProcessingMode.BALANCED
        self.adaptation_callbacks: List[callable] = []
        
        # Monitoring thread control
        self._monitoring_active = False
        self._monitoring_thread = None
        self._adaptation_lock = threading.Lock()
        
        # Component adaptations
        self.component_adaptations = {}
        
        self.logger.info("EnergyAdaptationManager initialized")
    
    def register_adaptation_callback(self, callback: callable, component_name: str):
        """Register a callback for system load adaptations"""
        self.adaptation_callbacks.append({
            'callback': callback,
            'component': component_name
        })
        self.logger.info(f"Registered adaptation callback for {component_name}")
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("System resource monitoring started")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("System resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in background thread)"""
        while self._monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Update current metrics with thread safety
                with self._adaptation_lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    
                # Determine if adaptation is needed
                if self._should_adapt(metrics):
                    self._trigger_adaptation(metrics)
                    
                time.sleep(self.config.monitoring_interval_s)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Short sleep on error to prevent spam
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 ** 3)
            
            # GPU metrics (if available)
            gpu_memory_percent = 0.0
            gpu_memory_available_gb = 0.0
            gpu_temperature = 0.0
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                    gpu_memory_available_gb = (gpu_memory_total - gpu_memory_used) / (1024 ** 3)
                    
                    # GPU temperature (platform dependent, may not work everywhere)
                    try:
                        import nvidia_ml_py3 as nvml
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        pass  # GPU temperature monitoring optional
                except Exception:
                    pass
            
            # Determine system load level
            load_level = self._determine_load_level(cpu_percent, memory_available_gb, gpu_memory_percent)
            
            # Determine processing mode
            processing_mode = self._determine_processing_mode(load_level, memory_available_gb)
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                gpu_memory_percent=gpu_memory_percent,
                gpu_memory_available_gb=gpu_memory_available_gb,
                gpu_temperature=gpu_temperature,
                system_load_level=load_level,
                processing_mode=processing_mode,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return safe fallback metrics
            return SystemMetrics(
                cpu_percent=50.0,
                memory_percent=50.0,
                memory_available_gb=4.0,
                system_load_level=SystemLoadLevel.MODERATE,
                processing_mode=ProcessingMode.BALANCED,
                timestamp=time.time()
            )
    
    def _determine_load_level(self, cpu_percent: float, memory_available_gb: float, gpu_memory_percent: float) -> SystemLoadLevel:
        """Determine current system load level"""
        # Critical conditions take precedence
        if (cpu_percent > self.config.cpu_threshold_critical or 
            memory_available_gb < self.config.memory_threshold_critical_gb or
            gpu_memory_percent > self.config.gpu_memory_threshold_critical_percent):
            return SystemLoadLevel.CRITICAL
            
        # Heavy load
        elif cpu_percent > self.config.cpu_threshold_heavy:
            return SystemLoadLevel.HEAVY
            
        # Moderate load
        elif cpu_percent > self.config.cpu_threshold_moderate:
            return SystemLoadLevel.MODERATE
            
        # Light load
        elif cpu_percent > self.config.cpu_threshold_light:
            return SystemLoadLevel.LIGHT
            
        # Idle
        else:
            return SystemLoadLevel.IDLE
    
    def _determine_processing_mode(self, load_level: SystemLoadLevel, memory_available_gb: float) -> ProcessingMode:
        """Determine appropriate processing mode based on system state"""
        if load_level == SystemLoadLevel.CRITICAL or memory_available_gb < 1.0:
            return ProcessingMode.HIBERNATION
        elif load_level == SystemLoadLevel.HEAVY:
            return ProcessingMode.MINIMAL
        elif load_level == SystemLoadLevel.MODERATE:
            return ProcessingMode.ENERGY_SAVER
        elif load_level == SystemLoadLevel.LIGHT:
            return ProcessingMode.BALANCED
        else:  # IDLE
            return ProcessingMode.FULL_POWER
    
    def _should_adapt(self, metrics: SystemMetrics) -> bool:
        """Determine if system adaptation should be triggered"""
        # Check if processing mode has changed
        if metrics.processing_mode != self.current_processing_mode:
            return True
            
        # Check for rapid system degradation
        if len(self.metrics_history) >= 3:
            recent_metrics = list(self.metrics_history)[-3:]
            cpu_trend = sum(m.cpu_percent for m in recent_metrics) / 3
            if cpu_trend > self.config.cpu_threshold_critical and metrics.cpu_percent > cpu_trend + 10:
                return True
                
        return False
    
    def _trigger_adaptation(self, metrics: SystemMetrics):
        """Trigger system adaptations based on current metrics"""
        try:
            old_mode = self.current_processing_mode
            self.current_processing_mode = metrics.processing_mode
            
            self.logger.info(f"System adaptation triggered: {old_mode.value} -> {metrics.processing_mode.value}")
            self.logger.info(f"System load: CPU={metrics.cpu_percent:.1f}%, Memory Available={metrics.memory_available_gb:.1f}GB")
            
            # Prepare adaptation parameters
            adaptation_params = self._get_adaptation_parameters(metrics)
            
            # Apply adaptations to registered components
            for callback_info in self.adaptation_callbacks:
                try:
                    callback = callback_info['callback']
                    component = callback_info['component']
                    
                    # Call the adaptation callback
                    if asyncio.iscoroutinefunction(callback):
                        # Handle async callbacks (run in thread to avoid blocking)
                        threading.Thread(
                            target=self._run_async_callback,
                            args=(callback, adaptation_params, component),
                            daemon=True
                        ).start()
                    else:
                        callback(adaptation_params)
                        
                    self.logger.debug(f"Applied adaptation to {component}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply adaptation to {callback_info['component']}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error triggering adaptations: {e}")
    
    def _run_async_callback(self, callback, adaptation_params, component):
        """Run async callback in a new event loop"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(callback(adaptation_params))
            loop.close()
        except Exception as e:
            self.logger.error(f"Error running async callback for {component}: {e}")
    
    def _get_adaptation_parameters(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Get adaptation parameters for components based on current metrics"""
        params = {
            'processing_mode': metrics.processing_mode,
            'system_load_level': metrics.system_load_level,
            'cpu_percent': metrics.cpu_percent,
            'memory_available_gb': metrics.memory_available_gb,
            'gpu_memory_percent': metrics.gpu_memory_percent,
            'timestamp': metrics.timestamp
        }
        
        # Add specific adaptation suggestions based on processing mode
        if metrics.processing_mode == ProcessingMode.HIBERNATION:
            params.update({
                'suggested_batch_size_multiplier': 0.1,  # Reduce batch size to 10%
                'suggested_model_precision': 'int8',      # Use lowest precision
                'suggested_inference_only': True,         # Disable training
                'suggested_disable_features': ['visualization', 'logging_verbose'],
                'suggested_processing_interval': 10.0     # Slow down processing
            })
        elif metrics.processing_mode == ProcessingMode.MINIMAL:
            params.update({
                'suggested_batch_size_multiplier': 0.25,
                'suggested_model_precision': 'fp16',
                'suggested_inference_only': False,
                'suggested_disable_features': ['visualization'],
                'suggested_processing_interval': 5.0
            })
        elif metrics.processing_mode == ProcessingMode.ENERGY_SAVER:
            params.update({
                'suggested_batch_size_multiplier': 0.5,
                'suggested_model_precision': 'fp16',
                'suggested_processing_interval': 2.0
            })
        elif metrics.processing_mode == ProcessingMode.BALANCED:
            params.update({
                'suggested_batch_size_multiplier': 0.75,
                'suggested_model_precision': 'fp32',
                'suggested_processing_interval': 1.0
            })
        else:  # FULL_POWER
            params.update({
                'suggested_batch_size_multiplier': 1.0,
                'suggested_model_precision': 'fp32',
                'suggested_processing_interval': 0.5
            })
            
        return params
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics (thread-safe)"""
        with self._adaptation_lock:
            return self.current_metrics
    
    def get_metrics_history(self) -> List[SystemMetrics]:
        """Get historical system metrics (thread-safe)"""
        with self._adaptation_lock:
            return list(self.metrics_history)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._adaptation_lock:
            current = self.current_metrics
            
            # Calculate trends if we have enough history
            trend_info = {}
            if len(self.metrics_history) >= 5:
                recent = list(self.metrics_history)[-5:]
                cpu_trend = (recent[-1].cpu_percent - recent[0].cpu_percent) / 4
                memory_trend = (recent[-1].memory_percent - recent[0].memory_percent) / 4
                trend_info = {
                    'cpu_trend_percent_per_sample': cpu_trend,
                    'memory_trend_percent_per_sample': memory_trend
                }
            
            return {
                'current_metrics': asdict(current),
                'processing_mode': current.processing_mode.value,
                'system_load_level': current.system_load_level.value,
                'monitoring_active': self._monitoring_active,
                'registered_components': len(self.adaptation_callbacks),
                'metrics_history_length': len(self.metrics_history),
                'trend_analysis': trend_info,
                'config': asdict(self.config)
            }

# Component-specific adaptation functions
class ComponentAdaptations:
    """Specific adaptation functions for different Nebula components"""
    
    @staticmethod
    async def adapt_file_processor(adaptation_params: Dict[str, Any]):
        """Adapt FileProcessor based on system load"""
        try:
            # This would be injected into FileProcessor instances
            processing_mode = adaptation_params['processing_mode']
            
            # Example adaptations (would need to be integrated into FileProcessor)
            if processing_mode == ProcessingMode.HIBERNATION:
                # Disable neural enhancements, use basic processing only
                pass
            elif processing_mode == ProcessingMode.MINIMAL:
                # Reduce batch sizes, disable advanced features
                pass
            
            logging.getLogger('FileProcessor').info(f"FileProcessor adapted to {processing_mode.value}")
            
        except Exception as e:
            logging.error(f"Failed to adapt FileProcessor: {e}")
    
    @staticmethod
    async def adapt_memory_learner(adaptation_params: Dict[str, Any]):
        """Adapt memory learner based on system load"""
        try:
            processing_mode = adaptation_params['processing_mode']
            
            if processing_mode == ProcessingMode.HIBERNATION:
                # Pause training, only do inference
                pass
            elif processing_mode == ProcessingMode.MINIMAL:
                # Reduce learning rate, smaller batches
                pass
                
            logging.getLogger('MemoryLearner').info(f"MemoryLearner adapted to {processing_mode.value}")
            
        except Exception as e:
            logging.error(f"Failed to adapt MemoryLearner: {e}")
    
    @staticmethod
    async def adapt_nas_evo_systems(adaptation_params: Dict[str, Any]):
        """Adapt NAS/EVO systems based on system load"""
        try:
            processing_mode = adaptation_params['processing_mode']
            
            if processing_mode == ProcessingMode.HIBERNATION:
                # Pause evolution, suspend NAS search
                pass
            elif processing_mode == ProcessingMode.MINIMAL:
                # Reduce population size, fewer generations
                pass
                
            logging.getLogger('NAS_EVO').info(f"NAS/EVO systems adapted to {processing_mode.value}")
            
        except Exception as e:
            logging.error(f"Failed to adapt NAS/EVO systems: {e}")

# Global energy adaptation manager instance
_global_adaptation_manager: Optional[EnergyAdaptationManager] = None

def get_global_adaptation_manager() -> EnergyAdaptationManager:
    """Get or create the global energy adaptation manager"""
    global _global_adaptation_manager
    
    if _global_adaptation_manager is None:
        _global_adaptation_manager = EnergyAdaptationManager()
        
        # Register default component adaptations
        _global_adaptation_manager.register_adaptation_callback(
            ComponentAdaptations.adapt_file_processor, 
            'FileProcessor'
        )
        _global_adaptation_manager.register_adaptation_callback(
            ComponentAdaptations.adapt_memory_learner,
            'MemoryLearner'
        )
        _global_adaptation_manager.register_adaptation_callback(
            ComponentAdaptations.adapt_nas_evo_systems,
            'NAS_EVO'
        )
        
        # Start monitoring
        _global_adaptation_manager.start_monitoring()
    
    return _global_adaptation_manager

def shutdown_global_adaptation_manager():
    """Shutdown the global energy adaptation manager"""
    global _global_adaptation_manager
    
    if _global_adaptation_manager:
        _global_adaptation_manager.stop_monitoring()
        _global_adaptation_manager = None