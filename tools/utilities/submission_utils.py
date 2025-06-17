#!/usr/bin/env python3
"""
Submission Utilities for VoxSigil System
Provides formatting, validation, and processing functionality for model submissions and outputs.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class SubmissionFormatter:
    """
    Advanced submission formatter for VoxSigil system.
    Handles formatting, validation, and processing of model outputs and submissions.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or str(project_root / "outputs" / "submissions")
        self.submission_history = []
        self.validation_rules = self._load_default_validation_rules()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_default_validation_rules(self) -> Dict[str, Any]:
        """Load default validation rules for submissions."""
        return {
            "required_fields": ["id", "predictions", "timestamp"],
            "max_predictions": 1000,
            "prediction_format": "grid",
            "allowed_values": list(range(10)),  # 0-9 for ARC tasks
            "max_grid_size": 30,
        }

    def format_submission(
        self,
        predictions: Union[List, Dict],
        task_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        format_type: str = "arc",
    ) -> Dict[str, Any]:
        """
        Format predictions into a standardized submission format.

        Args:
            predictions: Model predictions to format
            task_id: Unique identifier for the task
            metadata: Additional metadata to include
            format_type: Type of formatting ('arc', 'competition', 'custom')

        Returns:
            Formatted submission dictionary
        """
        submission_id = task_id or str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        if format_type == "arc":
            formatted = self._format_arc_submission(predictions, submission_id, timestamp, metadata)
        elif format_type == "competition":
            formatted = self._format_competition_submission(
                predictions, submission_id, timestamp, metadata
            )
        else:
            formatted = self._format_custom_submission(
                predictions, submission_id, timestamp, metadata
            )

        # Add to history
        self.submission_history.append(
            {
                "id": submission_id,
                "timestamp": timestamp,
                "format_type": format_type,
                "size": len(str(formatted)),
            }
        )

        return formatted

    def _format_arc_submission(
        self, predictions: Any, submission_id: str, timestamp: str, metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Format submission for ARC challenge format."""
        formatted = {
            "id": submission_id,
            "timestamp": timestamp,
            "format": "arc_challenge",
            "predictions": self._normalize_predictions(predictions),
            "metadata": metadata or {},
        }

        if isinstance(predictions, dict):
            # Handle task-specific predictions
            formatted["task_predictions"] = {}
            for task_id, pred in predictions.items():
                formatted["task_predictions"][task_id] = self._normalize_grid_prediction(pred)

        return formatted

    def _format_competition_submission(
        self, predictions: Any, submission_id: str, timestamp: str, metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Format submission for competition format."""
        return {
            "submission_id": submission_id,
            "created_at": timestamp,
            "model_output": self._normalize_predictions(predictions),
            "confidence_scores": self._extract_confidence_scores(predictions),
            "processing_time": metadata.get("processing_time") if metadata else None,
            "model_info": metadata.get("model_info", {}) if metadata else {},
        }

    def _format_custom_submission(
        self, predictions: Any, submission_id: str, timestamp: str, metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Format submission with custom format."""
        return {
            "id": submission_id,
            "timestamp": timestamp,
            "data": predictions,
            "metadata": metadata or {},
            "type": "custom_submission",
        }

    def _normalize_predictions(self, predictions: Any) -> Any:
        """Normalize predictions to a consistent format."""
        if isinstance(predictions, list):
            return [self._normalize_single_prediction(pred) for pred in predictions]
        elif isinstance(predictions, dict):
            return {k: self._normalize_single_prediction(v) for k, v in predictions.items()}
        else:
            return self._normalize_single_prediction(predictions)

    def _normalize_single_prediction(self, prediction: Any) -> Any:
        """Normalize a single prediction."""
        if isinstance(prediction, list) and prediction and isinstance(prediction[0], list):
            # Grid format - ensure all values are integers
            return [
                [int(cell) if isinstance(cell, (int, float)) else 0 for cell in row]
                for row in prediction
            ]
        elif isinstance(prediction, list):
            # List of values
            return [int(val) if isinstance(val, (int, float)) else 0 for val in prediction]
        elif isinstance(prediction, (int, float)):
            return int(prediction)
        else:
            return str(prediction)

    def _normalize_grid_prediction(self, prediction: Any) -> List[List[int]]:
        """Normalize prediction to grid format."""
        if isinstance(prediction, list) and prediction and isinstance(prediction[0], list):
            return [
                [int(cell) if isinstance(cell, (int, float)) else 0 for cell in row]
                for row in prediction
            ]
        elif isinstance(prediction, list):
            # Convert flat list to grid (assume square)
            size = int(len(prediction) ** 0.5)
            grid = []
            for i in range(size):
                row = prediction[i * size : (i + 1) * size]
                grid.append([int(val) if isinstance(val, (int, float)) else 0 for val in row])
            return grid
        else:
            # Single value - create 1x1 grid
            return [[int(prediction) if isinstance(prediction, (int, float)) else 0]]

    def _extract_confidence_scores(self, predictions: Any) -> Optional[List[float]]:
        """Extract confidence scores if available."""
        if isinstance(predictions, dict) and "confidence" in predictions:
            confidence = predictions["confidence"]
            if isinstance(confidence, list):
                return [float(score) for score in confidence]
            elif isinstance(confidence, (int, float)):
                return [float(confidence)]
        return None

    def validate_submission(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a submission against the validation rules.

        Args:
            submission: Submission to validate

        Returns:
            Validation result with success status and any errors
        """
        errors = []
        warnings = []

        # Check required fields
        for field in self.validation_rules["required_fields"]:
            if field not in submission:
                errors.append(f"Missing required field: {field}")

        # Validate predictions if present
        if "predictions" in submission:
            pred_validation = self._validate_predictions(submission["predictions"])
            errors.extend(pred_validation["errors"])
            warnings.extend(pred_validation["warnings"])

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checked_at": datetime.now().isoformat(),
        }

    def _validate_predictions(self, predictions: Any) -> Dict[str, List[str]]:
        """Validate prediction format and content."""
        errors = []
        warnings = []

        if isinstance(predictions, list):
            if len(predictions) > self.validation_rules["max_predictions"]:
                errors.append(
                    f"Too many predictions: {len(predictions)} > {self.validation_rules['max_predictions']}"
                )

            for i, pred in enumerate(predictions):
                if isinstance(pred, list) and pred and isinstance(pred[0], list):
                    # Grid prediction
                    grid_validation = self._validate_grid(pred, i)
                    errors.extend(grid_validation["errors"])
                    warnings.extend(grid_validation["warnings"])

        elif isinstance(predictions, dict):
            for task_id, pred in predictions.items():
                if isinstance(pred, list) and pred and isinstance(pred[0], list):
                    grid_validation = self._validate_grid(pred, task_id)
                    errors.extend(grid_validation["errors"])
                    warnings.extend(grid_validation["warnings"])

        return {"errors": errors, "warnings": warnings}

    def _validate_grid(
        self, grid: List[List[Any]], identifier: Union[str, int]
    ) -> Dict[str, List[str]]:
        """Validate a grid prediction."""
        errors = []
        warnings = []

        if len(grid) > self.validation_rules["max_grid_size"]:
            errors.append(f"Grid {identifier}: too many rows ({len(grid)})")

        for row_idx, row in enumerate(grid):
            if len(row) > self.validation_rules["max_grid_size"]:
                errors.append(f"Grid {identifier}, row {row_idx}: too many columns ({len(row)})")

            for col_idx, cell in enumerate(row):
                if not isinstance(cell, int) or cell not in self.validation_rules["allowed_values"]:
                    warnings.append(
                        f"Grid {identifier}[{row_idx}][{col_idx}]: invalid value {cell}"
                    )

        return {"errors": errors, "warnings": warnings}

    def save_submission(self, submission: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save submission to file.

        Args:
            submission: Submission to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_id = submission.get("id", "unknown")
            filename = f"submission_{submission_id}_{timestamp}.json"

        filepath = Path(self.output_dir) / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

        logger.info(f"Submission saved to: {filepath}")
        return str(filepath)

    def load_submission(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load submission from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_submission_history(self) -> List[Dict[str, Any]]:
        """Get submission history."""
        return self.submission_history.copy()

    def clear_history(self):
        """Clear submission history."""
        self.submission_history.clear()


def format_arc_solution(task_id: str, output_grids: List[List[List[int]]]) -> Dict[str, Any]:
    """
    Format solution for ARC challenge.

    Args:
        task_id: Task identifier
        output_grids: List of output grids for the task

    Returns:
        Formatted ARC solution
    """
    formatter = SubmissionFormatter()
    return formatter.format_submission(
        predictions={task_id: output_grids}, task_id=task_id, format_type="arc"
    )


def validate_arc_format(submission: Dict[str, Any]) -> bool:
    """
    Quick validation for ARC format.

    Args:
        submission: Submission to validate

    Returns:
        True if valid ARC format
    """
    formatter = SubmissionFormatter()
    result = formatter.validate_submission(submission)
    return result["valid"]
