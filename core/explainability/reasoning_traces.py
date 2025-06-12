#!/usr/bin/env python3
"""
Explainability and Reasoning Trace System for HOLO-1.5

This module implements comprehensive explainability hooks that capture the minimal
proof slice leading to deductions, enabling debugging and safety reviews.

HOLO-1.5 Enhanced Explainability:
- Recursive symbolic reasoning trace capture
- Neural-symbolic synthesis path logging
- VantaCore mesh collaboration decision points
- Cognitive load and symbolic depth tracking
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class ReasoningStepType(Enum):
    """Types of reasoning steps in the trace"""
    PERCEPTION = "perception"
    SYMBOLIC_INFERENCE = "symbolic_inference"
    NEURAL_PROCESSING = "neural_processing"
    PATTERN_MATCHING = "pattern_matching"
    RULE_APPLICATION = "rule_application"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    DECISION = "decision"

class ConfidenceLevel(Enum):
    """Confidence levels for reasoning steps"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class ReasoningStep:
    """Individual step in the reasoning process"""
    step_id: str
    step_type: ReasoningStepType
    timestamp: str
    agent_name: str
    subsystem: str
    
    # Step content
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    rule_applied: Optional[str] = None
    
    # Confidence and metadata
    confidence: float = 0.5
    cognitive_load: float = 0.0
    symbolic_depth: int = 0
    processing_time_ms: float = 0.0
    
    # Dependency tracking
    depends_on: List[str] = field(default_factory=list)
    enables: List[str] = field(default_factory=list)
    
    # Evidence and justification
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    justification: str = ""
    alternative_considered: List[str] = field(default_factory=list)
    
    # Error handling
    error_log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a task"""
    trace_id: str
    task_id: str
    timestamp: str
    
    # Task information
    task_type: str
    complexity_level: str
    input_summary: str
    
    # Reasoning process
    steps: List[ReasoningStep] = field(default_factory=list)
    step_index: Dict[str, int] = field(default_factory=dict)
    
    # Final result
    final_decision: Optional[Dict[str, Any]] = None
    success: bool = False
    total_processing_time_ms: float = 0.0
    
    # Summary statistics
    total_steps: int = 0
    avg_confidence: float = 0.0
    max_symbolic_depth: int = 0
    total_cognitive_load: float = 0.0
    
    # Proof slice (minimal critical path)
    minimal_proof_slice: List[str] = field(default_factory=list)
    critical_decisions: List[str] = field(default_factory=list)

class ReasoningTraceCapture:
    """Main class for capturing and managing reasoning traces"""
    
    def __init__(self, output_directory: str = "logs/reasoning_traces", 
                 max_trace_files: int = 1000, detailed_mode: bool = True):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_trace_files = max_trace_files
        self.detailed_mode = detailed_mode
        
        # Current trace being built
        self.current_trace: Optional[ReasoningTrace] = None
        self.trace_lock = threading.Lock()
        
        # Performance tracking
        self.trace_stats = {
            'traces_created': 0,
            'steps_captured': 0,
            'avg_trace_length': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info(f"Reasoning trace capture initialized (output: {self.output_dir})")
    
    def start_trace(self, task_id: str, task_type: str, complexity_level: str, 
                   input_summary: str) -> str:
        """Start a new reasoning trace"""
        with self.trace_lock:
            trace_id = f"trace_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            self.current_trace = ReasoningTrace(
                trace_id=trace_id,
                task_id=task_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task_type=task_type,
                complexity_level=complexity_level,
                input_summary=input_summary
            )
            
            logger.debug(f"Started reasoning trace: {trace_id}")
            return trace_id
    
    def add_step(self, step_type: ReasoningStepType, agent_name: str, subsystem: str,
                description: str, input_data: Dict[str, Any], output_data: Dict[str, Any],
                **kwargs) -> str:
        """Add a reasoning step to the current trace"""
        if self.current_trace is None:
            logger.warning("No active trace - call start_trace() first")
            return ""
        
        with self.trace_lock:
            step_id = f"step_{len(self.current_trace.steps):03d}_{int(time.time() * 1000)}"
            
            step = ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_name=agent_name,
                subsystem=subsystem,
                description=description,
                input_data=input_data,
                output_data=output_data,
                **kwargs
            )
            
            self.current_trace.steps.append(step)
            self.current_trace.step_index[step_id] = len(self.current_trace.steps) - 1
            
            # Update statistics
            self.trace_stats['steps_captured'] += 1
            
            logger.debug(f"Added reasoning step: {step_id} ({step_type.value})")
            return step_id
    
    def add_dependency(self, step_id: str, depends_on: str):
        """Add dependency relationship between steps"""
        if self.current_trace is None:
            return
        
        with self.trace_lock:
            if step_id in self.current_trace.step_index:
                step_idx = self.current_trace.step_index[step_id]
                self.current_trace.steps[step_idx].depends_on.append(depends_on)
                
                # Update the enabling step
                if depends_on in self.current_trace.step_index:
                    dep_idx = self.current_trace.step_index[depends_on]
                    self.current_trace.steps[dep_idx].enables.append(step_id)
    
    def add_evidence(self, step_id: str, evidence: Dict[str, Any]):
        """Add evidence to a reasoning step"""
        if self.current_trace is None:
            return
        
        with self.trace_lock:
            if step_id in self.current_trace.step_index:
                step_idx = self.current_trace.step_index[step_id]
                self.current_trace.steps[step_idx].evidence.append(evidence)
    
    def set_final_decision(self, decision: Dict[str, Any], success: bool):
        """Set the final decision for the current trace"""
        if self.current_trace is None:
            return
        
        with self.trace_lock:
            self.current_trace.final_decision = decision
            self.current_trace.success = success
    
    def finalize_trace(self) -> Optional[str]:
        """Finalize and save the current trace"""
        if self.current_trace is None:
            logger.warning("No active trace to finalize")
            return None
        
        with self.trace_lock:
            trace = self.current_trace
            
            # Calculate summary statistics
            self._calculate_trace_statistics(trace)
            
            # Generate minimal proof slice
            self._generate_minimal_proof_slice(trace)
            
            # Save to file
            trace_file = self._save_trace(trace)
            
            # Update global statistics
            self._update_global_statistics(trace)
            
            # Cleanup old traces if needed
            self._cleanup_old_traces()
            
            # Clear current trace
            self.current_trace = None
            
            logger.info(f"Finalized reasoning trace: {trace.trace_id}")
            return trace_file
    
    def _calculate_trace_statistics(self, trace: ReasoningTrace):
        """Calculate summary statistics for the trace"""
        if not trace.steps:
            return
        
        trace.total_steps = len(trace.steps)
        trace.avg_confidence = sum(step.confidence for step in trace.steps) / len(trace.steps)
        trace.max_symbolic_depth = max((step.symbolic_depth for step in trace.steps), default=0)
        trace.total_cognitive_load = sum(step.cognitive_load for step in trace.steps)
        trace.total_processing_time_ms = sum(step.processing_time_ms for step in trace.steps)
    
    def _generate_minimal_proof_slice(self, trace: ReasoningTrace):
        """Generate minimal proof slice - critical path to final decision"""
        if not trace.steps or not trace.final_decision:
            return
        
        # Find critical decisions (high confidence, high impact)
        critical_steps = []
        high_confidence_threshold = 0.7
        
        for step in trace.steps:
            # Critical if:
            # 1. High confidence
            # 2. Enables multiple other steps
            # 3. Is a decision or synthesis step
            # 4. Has high symbolic depth
            
            is_critical = (
                step.confidence >= high_confidence_threshold or
                len(step.enables) >= 2 or
                step.step_type in [ReasoningStepType.DECISION, ReasoningStepType.SYNTHESIS] or
                step.symbolic_depth >= 3
            )
            
            if is_critical:
                critical_steps.append(step.step_id)
        
        trace.critical_decisions = critical_steps
        
        # Build minimal proof slice by following dependencies backwards from critical steps
        proof_slice = set()
        
        def add_dependencies(step_id: str):
            if step_id in trace.step_index:
                proof_slice.add(step_id)
                step_idx = trace.step_index[step_id]
                step = trace.steps[step_idx]
                
                for dep in step.depends_on:
                    add_dependencies(dep)
        
        # Add all critical steps and their dependencies
        for critical_step in critical_steps:
            add_dependencies(critical_step)
        
        # Convert to ordered list
        trace.minimal_proof_slice = [
            step.step_id for step in trace.steps 
            if step.step_id in proof_slice
        ]
    
    def _save_trace(self, trace: ReasoningTrace) -> str:
        """Save trace to JSON file"""
        try:
            # Create filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{trace.trace_id}_{timestamp_str}.json"
            filepath = self.output_dir / filename
            
            # Convert trace to dictionary
            trace_dict = self._trace_to_dict(trace)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(trace_dict, f, indent=2)
            
            logger.debug(f"Saved reasoning trace to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save reasoning trace: {e}")
            return ""
    
    def _trace_to_dict(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Convert ReasoningTrace to dictionary for JSON serialization"""
        return {
            'metadata': {
                'trace_id': trace.trace_id,
                'task_id': trace.task_id,
                'timestamp': trace.timestamp,
                'task_type': trace.task_type,
                'complexity_level': trace.complexity_level,
                'input_summary': trace.input_summary,
                'success': trace.success
            },
            'statistics': {
                'total_steps': trace.total_steps,
                'avg_confidence': trace.avg_confidence,
                'max_symbolic_depth': trace.max_symbolic_depth,
                'total_cognitive_load': trace.total_cognitive_load,
                'total_processing_time_ms': trace.total_processing_time_ms
            },
            'steps': [
                {
                    'step_id': step.step_id,
                    'step_type': step.step_type.value,
                    'timestamp': step.timestamp,
                    'agent_name': step.agent_name,
                    'subsystem': step.subsystem,
                    'description': step.description,
                    'input_data': step.input_data if self.detailed_mode else {'summary': 'detailed_data_omitted'},
                    'output_data': step.output_data if self.detailed_mode else {'summary': 'detailed_data_omitted'},
                    'rule_applied': step.rule_applied,
                    'confidence': step.confidence,
                    'cognitive_load': step.cognitive_load,
                    'symbolic_depth': step.symbolic_depth,
                    'processing_time_ms': step.processing_time_ms,
                    'depends_on': step.depends_on,
                    'enables': step.enables,
                    'evidence': step.evidence if self.detailed_mode else [],
                    'justification': step.justification,
                    'alternatives_considered': step.alternative_considered,
                    'error_log': step.error_log,
                    'warnings': step.warnings
                }
                for step in trace.steps
            ],
            'final_decision': trace.final_decision,
            'minimal_proof_slice': trace.minimal_proof_slice,
            'critical_decisions': trace.critical_decisions
        }
    
    def _update_global_statistics(self, trace: ReasoningTrace):
        """Update global statistics"""
        self.trace_stats['traces_created'] += 1
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        self.trace_stats['avg_trace_length'] = (
            alpha * trace.total_steps + 
            (1 - alpha) * self.trace_stats['avg_trace_length']
        )
        
        self.trace_stats['avg_processing_time'] = (
            alpha * trace.total_processing_time_ms + 
            (1 - alpha) * self.trace_stats['avg_processing_time']
        )
    
    def _cleanup_old_traces(self):
        """Remove old trace files if exceeding max limit"""
        try:
            trace_files = list(self.output_dir.glob("trace_*.json"))
            
            if len(trace_files) > self.max_trace_files:
                # Sort by modification time and remove oldest
                trace_files.sort(key=lambda f: f.stat().st_mtime)
                files_to_remove = trace_files[:-self.max_trace_files]
                
                for file_path in files_to_remove:
                    file_path.unlink()
                    logger.debug(f"Removed old trace file: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old traces: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'trace_capture_stats': self.trace_stats.copy(),
            'output_directory': str(self.output_dir),
            'max_trace_files': self.max_trace_files,
            'detailed_mode': self.detailed_mode,
            'current_trace_active': self.current_trace is not None
        }

# Global trace capture instance
_global_trace_capture: Optional[ReasoningTraceCapture] = None

def initialize_trace_capture(output_directory: str = "logs/reasoning_traces",
                           max_trace_files: int = 1000,
                           detailed_mode: bool = True) -> ReasoningTraceCapture:
    """Initialize global trace capture system"""
    global _global_trace_capture
    _global_trace_capture = ReasoningTraceCapture(
        output_directory=output_directory,
        max_trace_files=max_trace_files,
        detailed_mode=detailed_mode
    )
    return _global_trace_capture

def get_trace_capture() -> Optional[ReasoningTraceCapture]:
    """Get the global trace capture instance"""
    return _global_trace_capture

# Convenience functions for easy integration
def start_reasoning_trace(task_id: str, task_type: str, complexity_level: str, 
                         input_summary: str) -> str:
    """Start a new reasoning trace"""
    capture = get_trace_capture()
    if capture:
        return capture.start_trace(task_id, task_type, complexity_level, input_summary)
    return ""

def add_reasoning_step(step_type: ReasoningStepType, agent_name: str, subsystem: str,
                      description: str, input_data: Dict[str, Any], 
                      output_data: Dict[str, Any], **kwargs) -> str:
    """Add a reasoning step"""
    capture = get_trace_capture()
    if capture:
        return capture.add_step(step_type, agent_name, subsystem, description, 
                               input_data, output_data, **kwargs)
    return ""

def finalize_reasoning_trace() -> Optional[str]:
    """Finalize the current reasoning trace"""
    capture = get_trace_capture()
    if capture:
        return capture.finalize_trace()
    return None

def main():
    """CLI entry point for trace management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HOLO-1.5 Reasoning Trace Management")
    parser.add_argument('--output-dir', default='logs/reasoning_traces', 
                       help='Output directory for traces')
    parser.add_argument('--max-files', type=int, default=1000, 
                       help='Maximum number of trace files to keep')
    parser.add_argument('--detailed', action='store_true', 
                       help='Enable detailed trace capture')
    parser.add_argument('--stats', action='store_true', 
                       help='Show trace capture statistics')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up old trace files')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize trace capture
    capture = initialize_trace_capture(
        output_directory=args.output_dir,
        max_trace_files=args.max_files,
        detailed_mode=args.detailed
    )
    
    if args.stats:
        # Show statistics
        stats = capture.get_statistics()
        print("\n🔍 Reasoning Trace Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    if args.cleanup:
        # Force cleanup
        capture._cleanup_old_traces()
        print("🧹 Cleanup completed")
    
    print(f"\n📝 Reasoning Trace Capture Configuration:")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Max Files: {args.max_files}")
    print(f"   Detailed Mode: {args.detailed}")

if __name__ == "__main__":
    main()
