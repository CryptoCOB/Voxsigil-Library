#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ART Logger for the VoxSigil System.

This module provides a centralized logging facility for the ART module,
covering events, training progress, pattern detection, and errors.

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh for adaptive
logging, cognitive trace generation, and intelligent log analysis.
"""

import logging
import os
import sys
import asyncio
import time
import json
from typing import Optional, Dict, Any

# HOLO-1.5 Core Imports
try:
    from ..agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
except (ImportError, ValueError):
    # Fallback for non-HOLO environments
    def vanta_agent(role=None, name=None, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveMeshRole:
        PROCESSOR = "processor"
    
    class BaseAgent:
        pass

# --- Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Determine a base directory for logs, e.g., relative to this file or a common project logs dir
# For now, let's assume a logs directory at the project root or within the art module itself.
# This should be configurable or determined by the main application (Vanta).
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# Fallback if not set by Vanta
DEFAULT_ART_LOG_FILE = "art_module.log"

# Use the same base logger name as Vanta for easier integration
# This will allow handlers configured for "VoxSigilSupervisor" to also capture these logs.
VANTA_SUPERVISOR_LOGGER_NAME = "VoxSigilSupervisor"


@vanta_agent(role=CognitiveMeshRole.PROCESSOR)
class ARTLogger(BaseAgent):
    """
    ART Logger with HOLO-1.5 Recursive Symbolic Cognition Mesh enhancement.
    Provides adaptive logging, cognitive trace generation, and intelligent log analysis.
    """
    _instance = None

    def __new__(
        cls, logger_name="ARTModule", log_file=None, log_dir=None, level=LOG_LEVEL
    ):
        if cls._instance is None:
            cls._instance = super(ARTLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, logger_name="ARTModule", log_file=None, log_dir=None, level=LOG_LEVEL
    ):
        if self._initialized:
            return

        BaseAgent.__init__(self)
        
        self.logger_name = logger_name
        self.log_dir = log_dir if log_dir else DEFAULT_LOG_DIR
        self.log_file = (
            log_file if log_file else os.path.join(self.log_dir, DEFAULT_ART_LOG_FILE)
        )
        self.level = level

        # HOLO-1.5 Cognitive Metrics
        self.cognitive_metrics = {
            'log_events': 0,
            'error_events': 0,
            'warning_events': 0,
            'cognitive_traces': 0,
            'pattern_detections': 0,
            'adaptive_adjustments': 0,
            'cognitive_load': 0.0,
            'symbolic_depth': 0.0
        }
        
        self.vanta_core = None
        self._background_tasks = []
        self._log_patterns = []
        self._adaptive_thresholds = {
            'error_rate_threshold': 0.1,
            'warning_rate_threshold': 0.2,
            'cognitive_load_threshold': 0.8
        }

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except OSError as e:
                # Fallback to console logging if directory creation fails
                print(
                    f"Error creating log directory {self.log_dir}: {e}. Logging to console."
                )
                self._configure_console_logger()
                self._initialized = True
                return

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.level)

        # Prevent duplicate handlers if already configured (e.g., in a notebook environment)
        if not self.logger.handlers:
            # File Handler
            try:
                file_handler = logging.FileHandler(
                    self.log_file, mode="a"
                )  # Append mode
                file_handler.setLevel(self.level)
                formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except IOError as e:
                print(
                    f"Error setting up file handler for {self.log_file}: {e}. Logging to console."
                )
                self._configure_console_logger()  # Fallback to console
            else:
                # Console Handler (optional, for also printing to stdout/stderr)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self.level)  # Or a different level for console
                console_formatter = logging.Formatter(
                    LOG_FORMAT, datefmt=LOG_DATE_FORMAT
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)

        self._initialized = True
        
        # Start async initialization if possible
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                asyncio.run(self.async_init())
        except RuntimeError:
            # No event loop available, defer async initialization
            pass
    
    async def async_init(self):
        """Initialize async components and cognitive monitoring"""
        try:
            from ..core.vanta_core import VantaCore
            self.vanta_core = VantaCore.get_instance()
            await self.register_cognitive_capabilities()
            await self.start_cognitive_monitoring()
            self.logger.info("🧠 ARTLogger HOLO-1.5 initialization complete")
        except ImportError:
            if self.logger:
                self.logger.warning("VantaCore not available - running in standalone mode")
    
    async def register_cognitive_capabilities(self):
        """Register logging capabilities with VantaCore mesh"""
        if self.vanta_core:
            capabilities = {
                'adaptive_logging': 'Intelligent log level and pattern adaptation',
                'cognitive_trace_generation': 'Real-time cognitive state logging',
                'log_pattern_analysis': 'Pattern detection in log streams',
                'error_prediction': 'Predictive error analysis from log patterns',
                'performance_monitoring': 'System performance tracking through logs'
            }
            
            for capability, description in capabilities.items():
                await self.vanta_core.register_capability(
                    f"art_logger.{capability}",
                    description,
                    self
                )
    async def start_cognitive_monitoring(self):
        """Start background cognitive monitoring and adaptive logging"""
        async def monitor_loop():
            try:
                while not getattr(self, '_shutdown_requested', False):
                    await asyncio.sleep(60)  # Monitor every minute
                    
                    if getattr(self, '_shutdown_requested', False):
                        break
                        
                    self._update_cognitive_metrics()
                    self._adapt_logging_behavior()
                    
                    # Generate cognitive trace for mesh learning
                    if self.vanta_core:
                        try:
                            trace = self._generate_cognitive_trace()
                            await self.vanta_core.emit_cognitive_trace(trace)
                        except Exception as e:
                            logger.error(f"Error emitting cognitive trace: {e}")
                            
            except asyncio.CancelledError:
                logger.info("Cognitive monitoring cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in cognitive monitoring loop: {e}", exc_info=True)
        
        task = asyncio.create_task(monitor_loop())
        self._background_tasks.append(task)
    
    async def shutdown(self):
        """Gracefully shutdown the ART logger"""
        self._shutdown_requested = True
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling background task: {e}")
        
        self._background_tasks.clear()
        logger.info("ART logger shutdown complete")
    
    def _update_cognitive_metrics(self):
        """Update real-time cognitive metrics"""
        self.cognitive_metrics['cognitive_load'] = self._calculate_cognitive_load()
        self.cognitive_metrics['symbolic_depth'] = self._calculate_symbolic_depth()
        
        # Detect patterns in recent logs
        self._detect_log_patterns()
    
    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load based on logging activity"""
        base_load = 0.1
        error_load = min(self.cognitive_metrics['error_events'] * 0.1, 0.4)
        warning_load = min(self.cognitive_metrics['warning_events'] * 0.05, 0.3)
        volume_load = min(self.cognitive_metrics['log_events'] * 0.001, 0.2)
        return min(base_load + error_load + warning_load + volume_load, 1.0)
    
    def _calculate_symbolic_depth(self) -> float:
        """Calculate symbolic processing depth of logging"""
        base_depth = 0.1
        trace_depth = min(self.cognitive_metrics['cognitive_traces'] * 0.1, 0.5)
        pattern_depth = min(self.cognitive_metrics['pattern_detections'] * 0.15, 0.3)
        adaptation_depth = min(self.cognitive_metrics['adaptive_adjustments'] * 0.1, 0.1)
        return min(base_depth + trace_depth + pattern_depth + adaptation_depth, 1.0)
    
    def _detect_log_patterns(self):
        """Detect patterns in recent log activity"""
        # Analyze recent logging patterns and update metrics
        if self.cognitive_metrics['log_events'] > 0:
            error_rate = self.cognitive_metrics['error_events'] / self.cognitive_metrics['log_events']
            warning_rate = self.cognitive_metrics['warning_events'] / self.cognitive_metrics['log_events']
            
            # Detect concerning patterns
            if error_rate > self._adaptive_thresholds['error_rate_threshold']:
                self.cognitive_metrics['pattern_detections'] += 1
                if self.logger:
                    self.logger.warning(f"🔍 High error rate detected: {error_rate:.2%}")
            
            if warning_rate > self._adaptive_thresholds['warning_rate_threshold']:
                self.cognitive_metrics['pattern_detections'] += 1
                if self.logger:
                    self.logger.info(f"📊 High warning rate detected: {warning_rate:.2%}")
    
    def _adapt_logging_behavior(self):
        """Adapt logging behavior based on cognitive metrics"""
        current_load = self.cognitive_metrics['cognitive_load']
        
        # Adapt logging level based on cognitive load
        if current_load > self._adaptive_thresholds['cognitive_load_threshold']:
            # Reduce verbosity under high load
            if self.logger.level < logging.WARNING:
                self.logger.setLevel(logging.WARNING)
                self.cognitive_metrics['adaptive_adjustments'] += 1
                if self.logger:
                    self.logger.warning("🔧 Adapted to higher log level due to cognitive load")
        elif current_load < 0.3:
            # Increase verbosity under low load
            if self.logger.level > logging.INFO:
                self.logger.setLevel(logging.INFO)
                self.cognitive_metrics['adaptive_adjustments'] += 1
                if self.logger:
                    self.logger.info("🔧 Adapted to lower log level due to low cognitive load")
    
    def _generate_cognitive_trace(self) -> Dict[str, Any]:
        """Generate cognitive trace for mesh learning"""
        return {
            'component': 'ARTLogger',
            'role': 'PROCESSOR',
            'timestamp': time.time(),
            'metrics': self.cognitive_metrics.copy(),
            'cognitive_state': {
                'logging_efficiency': 1.0 - (self.cognitive_metrics['error_events'] / max(1, self.cognitive_metrics['log_events'])),
                'pattern_detection_rate': self.cognitive_metrics['pattern_detections'] / max(1, self.cognitive_metrics['log_events']),
                'adaptation_frequency': self.cognitive_metrics['adaptive_adjustments'] / max(1, time.time() - getattr(self, '_start_time', time.time()))
            },
            'adaptive_state': {
                'current_level': self.logger.level if self.logger else logging.INFO,
                'error_rate': self.cognitive_metrics['error_events'] / max(1, self.cognitive_metrics['log_events']),
                'warning_rate': self.cognitive_metrics['warning_events'] / max(1, self.cognitive_metrics['log_events'])
            }
        }
    
    def log_cognitive_trace(self, trace_data: Dict[str, Any]):
        """Log a cognitive trace with enhanced formatting"""
        self.cognitive_metrics['cognitive_traces'] += 1
        if self.logger:
            trace_str = json.dumps(trace_data, indent=2)
            self.logger.info(f"🧠 COGNITIVE_TRACE: {trace_str}")

    def _configure_console_logger(self):
        """Configures a basic console logger as a fallback."""
        self.logger = logging.getLogger(self.logger_name + "_console")
        self.logger.setLevel(self.level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# --- Public API for easy access ---
def get_art_logger(
    name: Optional[str] = None,
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    base_logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Returns a logger instance for ART components, with sensible defaults for easy use.

    Args:
        name: The name of the ART component (e.g., 'ARTController'). If None, uses 'ARTModule'.
        level: The logging level (e.g., logging.INFO, logging.DEBUG). If None, uses LOG_LEVEL.
        log_file: Optional path to a file for logs. If None, uses default ART log file.
        base_logger_name: The root logger name. If None, uses VANTA_SUPERVISOR_LOGGER_NAME.

    Returns:
        A configured logger instance.
    """
    # Set defaults for easier use
    if name is None:
        name = "ARTModule"
    if level is None:
        level = LOG_LEVEL
    if base_logger_name is None:
        base_logger_name = VANTA_SUPERVISOR_LOGGER_NAME

    logger_name = f"{base_logger_name}.art.{name}" if base_logger_name else name
    logger = logging.getLogger(logger_name)

    # Only configure handlers if none exist
    if not logger.handlers:
        logger.setLevel(level)

        # Add console handler if no parent handlers exist
        has_propagating_handlers = False
        temp_logger = logger
        while temp_logger:
            if temp_logger.handlers:
                has_propagating_handlers = True
                break
            if not getattr(temp_logger, "propagate", True):
                break
            temp_logger = getattr(temp_logger, "parent", None)

        if not has_propagating_handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s"
            )
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # Add file handler if requested
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s"
            )
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

    logger.propagate = True
    return logger


# --- Enhanced Public API with HOLO-1.5 Cognitive Integration ---
async def get_art_logger_async(
    name: Optional[str] = None,
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    base_logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Returns a logger instance for ART components with HOLO-1.5 cognitive enhancement.
    Enhanced with adaptive behavior and cognitive monitoring.

    Args:
        name: The name of the ART component (e.g., 'ARTController'). If None, uses 'ARTModule'.
        level: The logging level (e.g., logging.INFO, logging.DEBUG). If None, uses LOG_LEVEL.
        log_file: Optional path to a file for logs. If None, uses default ART log file.
        base_logger_name: The root logger name. If None, uses VANTA_SUPERVISOR_LOGGER_NAME.

    Returns:
        A configured logger instance with cognitive capabilities.
    """
    # Get the enhanced ARTLogger instance
    art_logger_instance = ARTLogger()
    await art_logger_instance.async_init()
    
    # Return the enhanced logger
    return get_art_logger(name, level, log_file, base_logger_name)

# Example Usage:
if __name__ == "__main__":
    # This demonstrates how other ART modules would use the logger.
    # Vanta or the main application would ideally configure the log_dir.

    # Using the default logger instance
    logger = get_art_logger("ARTModule")
    logger.info("ARTLogger initialized. This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    # Example of a specific logger for a sub-component
    controller_logger = get_art_logger("ARTController", log_file="art_controller.log")
    controller_logger.debug(
        "Debugging ART Controller operation."
    )  # Won't show if LOG_LEVEL is INFO

    # To see debug messages, you'd initialize with level=logging.DEBUG
    debug_logger = get_art_logger("ARTDebug", level=logging.DEBUG)
    debug_logger.debug("This is a detailed debug message for ART.")

    print(
        f"ART logs are being written to: {ARTLogger().log_file if hasattr(ARTLogger(), 'log_file') else 'console'}"
    )

    # Test with default base_logger_name ("VoxSigilSupervisor")
    logger1 = get_art_logger("TestComponent1")
    logger1.info(
        "This is an info message from TestComponent1 (under VoxSigilSupervisor)."
    )
    logger1.debug(
        "This is a debug message from TestComponent1 (under VoxSigilSupervisor)."
    )

    # Test with a specific log file for this logger instance
    logger2 = get_art_logger("TestComponent2", log_file="test_component2.log")
    logger2.warning("This is a warning from TestComponent2, also to its own file.")

    print(
        f"Logger1 name: {logger1.name}, Effective level: {logger1.getEffectiveLevel()}, Handlers: {logger1.handlers}"
    )
    print(
        f"Logger2 name: {logger2.name}, Effective level: {logger2.getEffectiveLevel()}, Handlers: {logger2.handlers}"
    )
