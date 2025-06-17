#!/usr/bin/env python3
"""
Task Analysis Utilities for VoxSigil System
Provides task analysis, decomposition, and cognitive processing functionality.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class TaskAnalyzer:
    """
    Advanced task analyzer for VoxSigil system.
    Handles task decomposition, analysis, and cognitive processing.
    """

    def __init__(self):
        self.analysis_history = []
        self.task_patterns = {}
        self.complexity_metrics = {
            "spatial": ["grid_size", "object_count", "transformation_complexity"],
            "logical": ["rule_complexity", "pattern_depth", "abstraction_level"],
            "temporal": ["sequence_length", "state_transitions", "memory_requirements"],
        }

    def analyze_task(self, task_data: Dict[str, Any], task_type: str = "arc") -> Dict[str, Any]:
        """
        Perform comprehensive task analysis.

        Args:
            task_data: Task data to analyze
            task_type: Type of task ('arc', 'reasoning', 'custom')

        Returns:
            Analysis results dictionary
        """
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        analysis = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "complexity": self._analyze_complexity(task_data),
            "patterns": self._identify_patterns(task_data),
            "cognitive_load": self._estimate_cognitive_load(task_data),
            "decomposition": self._decompose_task(task_data),
            "metadata": self._extract_metadata(task_data),
        }

        self.analysis_history.append(analysis)
        return analysis

    def _analyze_complexity(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task complexity across multiple dimensions."""
        complexity = {
            "overall": 0.0,
            "spatial": 0.0,
            "logical": 0.0,
            "temporal": 0.0,
            "details": {},
        }

        # Spatial complexity
        if "train" in task_data or "test" in task_data:
            spatial = self._calculate_spatial_complexity(task_data)
            complexity["spatial"] = spatial["score"]
            complexity["details"]["spatial"] = spatial

        # Logical complexity
        logical = self._calculate_logical_complexity(task_data)
        complexity["logical"] = logical["score"]
        complexity["details"]["logical"] = logical

        # Temporal complexity
        temporal = self._calculate_temporal_complexity(task_data)
        complexity["temporal"] = temporal["score"]
        complexity["details"]["temporal"] = temporal

        # Overall complexity (weighted average)
        complexity["overall"] = (
            complexity["spatial"] * 0.4 + complexity["logical"] * 0.4 + complexity["temporal"] * 0.2
        )

        return complexity

    def _calculate_spatial_complexity(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate spatial complexity metrics."""
        metrics = {
            "score": 0.0,
            "grid_sizes": [],
            "object_counts": [],
            "color_diversity": [],
            "transformation_types": [],
        }

        # Analyze training examples
        examples = task_data.get("train", []) + task_data.get("test", [])

        for example in examples:
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])

            if input_grid:
                # Grid size complexity
                height, width = len(input_grid), len(input_grid[0]) if input_grid else 0
                metrics["grid_sizes"].append(height * width)

                # Object count complexity
                unique_colors = set()
                for row in input_grid:
                    unique_colors.update(row)
                metrics["object_counts"].append(len(unique_colors))
                metrics["color_diversity"].append(len(unique_colors))

                # Transformation analysis
                if output_grid:
                    transform_type = self._analyze_transformation(input_grid, output_grid)
                    metrics["transformation_types"].append(transform_type)

        # Calculate score
        if metrics["grid_sizes"]:
            avg_size = sum(metrics["grid_sizes"]) / len(metrics["grid_sizes"])
            avg_objects = sum(metrics["object_counts"]) / len(metrics["object_counts"])

            # Normalize to 0-1 scale
            size_complexity = min(avg_size / 900, 1.0)  # Max grid size 30x30
            object_complexity = min(avg_objects / 10, 1.0)  # Max 10 colors

            metrics["score"] = (size_complexity + object_complexity) / 2

        return metrics

    def _calculate_logical_complexity(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate logical complexity metrics."""
        metrics = {
            "score": 0.0,
            "rule_types": [],
            "pattern_depth": 0,
            "abstraction_level": 0,
            "consistency": 0.0,
        }

        # Analyze patterns and rules
        examples = task_data.get("train", [])
        if len(examples) >= 2:
            # Pattern consistency across examples
            patterns = []
            for example in examples:
                pattern = self._extract_example_pattern(example)
                patterns.append(pattern)

            metrics["consistency"] = self._calculate_pattern_consistency(patterns)
            metrics["pattern_depth"] = self._estimate_pattern_depth(patterns)
            metrics["abstraction_level"] = self._estimate_abstraction_level(patterns)

            # Score based on complexity indicators
            metrics["score"] = (
                (1 - metrics["consistency"]) * 0.4  # Less consistent = more complex
                + metrics["pattern_depth"] * 0.3
                + metrics["abstraction_level"] * 0.3
            )

        return metrics

    def _calculate_temporal_complexity(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate temporal complexity metrics."""
        metrics = {"score": 0.0, "sequence_length": 0, "state_changes": 0, "memory_requirements": 0}

        examples = task_data.get("train", [])
        if examples:
            metrics["sequence_length"] = len(examples)

            # Estimate state changes and memory requirements
            total_changes = 0
            for example in examples:
                input_grid = example.get("input", [])
                output_grid = example.get("output", [])
                if input_grid and output_grid:
                    changes = self._count_state_changes(input_grid, output_grid)
                    total_changes += changes

            metrics["state_changes"] = total_changes
            metrics["memory_requirements"] = self._estimate_memory_requirements(examples)

            # Normalize score
            metrics["score"] = min(
                (
                    metrics["sequence_length"] / 10  # Max 10 examples
                    + metrics["state_changes"] / 100  # Normalize state changes
                    + metrics["memory_requirements"] / 50  # Normalize memory
                )
                / 3,
                1.0,
            )

        return metrics

    def _identify_patterns(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify patterns in the task data."""
        patterns = []

        examples = task_data.get("train", [])
        for i, example in enumerate(examples):
            pattern = {
                "example_id": i,
                "type": "transformation",
                "description": self._describe_transformation(example),
                "confidence": 0.8,  # Placeholder
                "features": self._extract_pattern_features(example),
            }
            patterns.append(pattern)

        return patterns

    def _estimate_cognitive_load(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cognitive load required for the task."""
        load = {"working_memory": 0.0, "attention": 0.0, "reasoning": 0.0, "total": 0.0}

        examples = task_data.get("train", [])
        if examples:
            # Working memory load
            avg_grid_size = 0
            for example in examples:
                input_grid = example.get("input", [])
                if input_grid:
                    size = len(input_grid) * len(input_grid[0]) if input_grid else 0
                    avg_grid_size += size

            if examples:
                avg_grid_size /= len(examples)
                load["working_memory"] = min(avg_grid_size / 900, 1.0)

            # Attention load (based on color diversity)
            total_colors = set()
            for example in examples:
                input_grid = example.get("input", [])
                for row in input_grid:
                    total_colors.update(row)

            load["attention"] = min(len(total_colors) / 10, 1.0)

            # Reasoning load (based on transformation complexity)
            reasoning_score = 0
            for example in examples:
                input_grid = example.get("input", [])
                output_grid = example.get("output", [])
                if input_grid and output_grid:
                    complexity = self._transformation_complexity(input_grid, output_grid)
                    reasoning_score += complexity

            if examples:
                load["reasoning"] = min(reasoning_score / len(examples), 1.0)

            # Total cognitive load
            load["total"] = (load["working_memory"] + load["attention"] + load["reasoning"]) / 3

        return load

    def _decompose_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose task into sub-tasks and components."""
        decomposition = {
            "subtasks": [],
            "dependencies": [],
            "execution_order": [],
            "complexity_hierarchy": [],
        }

        examples = task_data.get("train", [])

        # Identify common subtasks
        subtasks = [
            {"name": "perception", "description": "Perceive input grid structure"},
            {"name": "pattern_recognition", "description": "Identify transformation patterns"},
            {"name": "rule_extraction", "description": "Extract transformation rules"},
            {"name": "application", "description": "Apply rules to test input"},
            {"name": "verification", "description": "Verify output correctness"},
        ]

        decomposition["subtasks"] = subtasks
        decomposition["execution_order"] = [task["name"] for task in subtasks]

        return decomposition

    def _extract_metadata(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task metadata."""
        metadata = {
            "example_count": len(task_data.get("train", [])),
            "test_count": len(task_data.get("test", [])),
            "grid_dimensions": [],
            "color_usage": {},
            "transformation_types": [],
        }

        # Analyze all examples
        all_examples = task_data.get("train", []) + task_data.get("test", [])
        color_counts = {}

        for example in all_examples:
            input_grid = example.get("input", [])
            if input_grid:
                height, width = len(input_grid), len(input_grid[0]) if input_grid else 0
                metadata["grid_dimensions"].append((height, width))

                # Count colors
                for row in input_grid:
                    for color in row:
                        color_counts[color] = color_counts.get(color, 0) + 1

        metadata["color_usage"] = color_counts
        return metadata

    # Helper methods for analysis
    def _analyze_transformation(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> str:
        """Analyze transformation type between input and output."""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return "size_change"

        # Count changes
        changes = 0
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                if input_grid[i][j] != output_grid[i][j]:
                    changes += 1

        change_ratio = changes / (len(input_grid) * len(input_grid[0]))

        if change_ratio < 0.1:
            return "minimal"
        elif change_ratio < 0.5:
            return "moderate"
        else:
            return "major"

    def _extract_example_pattern(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern from a single example."""
        return {
            "input_size": len(example.get("input", [])),
            "output_size": len(example.get("output", [])),
            "colors_used": len(set(sum(example.get("input", []), []))),
            "transformation_type": self._analyze_transformation(
                example.get("input", []), example.get("output", [])
            ),
        }

    def _calculate_pattern_consistency(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across patterns."""
        if len(patterns) < 2:
            return 1.0

        # Compare transformation types
        transform_types = [p.get("transformation_type") for p in patterns]
        consistency = len(set(transform_types)) / len(transform_types)
        return 1.0 - consistency

    def _estimate_pattern_depth(self, patterns: List[Dict[str, Any]]) -> float:
        """Estimate pattern depth (complexity)."""
        # Simplified depth estimation
        avg_colors = (
            sum(p.get("colors_used", 0) for p in patterns) / len(patterns) if patterns else 0
        )
        return min(avg_colors / 10, 1.0)

    def _estimate_abstraction_level(self, patterns: List[Dict[str, Any]]) -> float:
        """Estimate abstraction level required."""
        # Simplified abstraction estimation
        size_variance = 0
        if patterns:
            sizes = [p.get("input_size", 0) for p in patterns]
            avg_size = sum(sizes) / len(sizes)
            size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)

        return min(size_variance / 100, 1.0)

    def _count_state_changes(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> int:
        """Count state changes between input and output."""
        changes = 0
        min_height = min(len(input_grid), len(output_grid))
        min_width = min(
            len(input_grid[0]) if input_grid else 0, len(output_grid[0]) if output_grid else 0
        )

        for i in range(min_height):
            for j in range(min_width):
                if input_grid[i][j] != output_grid[i][j]:
                    changes += 1

        return changes

    def _estimate_memory_requirements(self, examples: List[Dict[str, Any]]) -> int:
        """Estimate memory requirements for the task."""
        max_grid_size = 0
        total_colors = set()

        for example in examples:
            input_grid = example.get("input", [])
            if input_grid:
                size = len(input_grid) * len(input_grid[0]) if input_grid else 0
                max_grid_size = max(max_grid_size, size)

                for row in input_grid:
                    total_colors.update(row)

        return max_grid_size + len(total_colors) * 2  # Simplified memory calculation

    def _describe_transformation(self, example: Dict[str, Any]) -> str:
        """Generate description of the transformation."""
        input_grid = example.get("input", [])
        output_grid = example.get("output", [])

        if not input_grid or not output_grid:
            return "Unknown transformation"

        if len(input_grid) != len(output_grid):
            return "Grid size transformation"

        changes = self._count_state_changes(input_grid, output_grid)
        total_cells = len(input_grid) * len(input_grid[0])
        change_ratio = changes / total_cells if total_cells > 0 else 0

        if change_ratio < 0.1:
            return "Minor cell modifications"
        elif change_ratio < 0.5:
            return "Moderate pattern transformation"
        else:
            return "Major structural change"

    def _extract_pattern_features(self, example: Dict[str, Any]) -> List[str]:
        """Extract features from pattern."""
        features = []
        input_grid = example.get("input", [])

        if input_grid:
            height, width = len(input_grid), len(input_grid[0]) if input_grid else 0
            features.append(f"grid_size_{height}x{width}")

            colors = set(sum(input_grid, []))
            features.append(f"colors_{len(colors)}")

            # Add more feature extraction logic as needed

        return features

    def _transformation_complexity(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> float:
        """Calculate transformation complexity score."""
        changes = self._count_state_changes(input_grid, output_grid)
        total_cells = len(input_grid) * len(input_grid[0]) if input_grid else 1
        return changes / total_cells

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history.copy()

    def clear_history(self):
        """Clear analysis history."""
        self.analysis_history.clear()
