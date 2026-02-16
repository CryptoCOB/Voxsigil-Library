"""
Tensor Validator Module

This module provides utilities for validating tensors and array-like data structures.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Set

# Configure logger
logger = logging.getLogger(__name__)

def validate_shape(tensor: np.ndarray, expected_shape: Union[Tuple[int, ...], List[int]]) -> bool:
    """
    Validate that a tensor has the expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (tuple or list of dimensions)
        
    Returns:
        True if tensor has the expected shape, False otherwise
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return False
    
    # Convert expected_shape to tuple to handle both tuples and lists
    expected_shape_tuple = tuple(expected_shape)
    
    # Check if shapes match
    if tensor.shape == expected_shape_tuple:
        return True
    
    # Otherwise, log the error and return False
    logger.warning(f"Shape mismatch: expected {expected_shape_tuple}, got {tensor.shape}")
    return False

def validate_dtype(tensor: np.ndarray, expected_dtype: Union[np.dtype, type, str]) -> bool:
    """
    Validate that a tensor has the expected data type.
    
    Args:
        tensor: Tensor to validate
        expected_dtype: Expected data type
        
    Returns:
        True if tensor has the expected dtype, False otherwise
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return False
    
    # Convert expected_dtype to numpy dtype for comparison
    try:
        if isinstance(expected_dtype, str):
            expected_np_dtype = np.dtype(expected_dtype)
        else:
            expected_np_dtype = np.dtype(expected_dtype)
    except TypeError:
        logger.error(f"Invalid dtype: {expected_dtype}")
        return False
    
    # Check if dtypes match
    if tensor.dtype == expected_np_dtype:
        return True
    
    # Otherwise, log the error and return False
    logger.warning(f"Dtype mismatch: expected {expected_np_dtype}, got {tensor.dtype}")
    return False

def validate_range(tensor: np.ndarray, min_val: Union[int, float], max_val: Union[int, float]) -> bool:
    """
    Validate that all values in a tensor are within the specified range.
    
    Args:
        tensor: Tensor to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        True if all values are within range, False otherwise
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return False
    
    # Check if all values are within range
    if np.all((tensor >= min_val) & (tensor <= max_val)):
        return True
    
    # Otherwise, log the error and return False
    min_actual = np.min(tensor)
    max_actual = np.max(tensor)
    
    out_of_range = np.sum((tensor < min_val) | (tensor > max_val))
    logger.warning(f"Range violation: expected [{min_val}, {max_val}], got [{min_actual}, {max_actual}]")
    logger.warning(f"{out_of_range} values out of {tensor.size} are out of range")
    
    return False

def validate_not_nan_or_inf(tensor: np.ndarray) -> bool:
    """
    Validate that a tensor does not contain NaN or infinite values.
    
    Args:
        tensor: Tensor to validate
        
    Returns:
        True if tensor has no NaN or infinite values, False otherwise
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return False
    
    # Check for NaN
    if np.any(np.isnan(tensor)):
        nan_count = np.sum(np.isnan(tensor))
        logger.warning(f"Tensor contains {nan_count} NaN values")
        return False
    
    # Check for infinity
    if np.any(np.isinf(tensor)):
        inf_count = np.sum(np.isinf(tensor))
        logger.warning(f"Tensor contains {inf_count} infinite values")
        return False
    
    return True

def validate_shape_compatibility(tensor1: np.ndarray, tensor2: np.ndarray, 
                                operation: str = "addition") -> bool:
    """
    Validate that two tensors have compatible shapes for a specified operation.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        operation: Operation to check compatibility for ('addition', 'multiplication', 'matmul', 'broadcasting')
        
    Returns:
        True if tensors have compatible shapes, False otherwise
    """
    if not isinstance(tensor1, np.ndarray) or not isinstance(tensor2, np.ndarray):
        logger.warning(f"Both inputs must be numpy arrays, got {type(tensor1)} and {type(tensor2)}")
        return False
    
    operation = operation.lower()
    
    if operation == "addition" or operation == "subtraction":
        # Shapes must be identical for element-wise addition/subtraction
        if tensor1.shape == tensor2.shape:
            return True
        
        # Check broadcast compatibility
        try:
            # Create dummy arrays to test broadcasting
            dummy1 = np.zeros(tensor1.shape, dtype=np.float32)
            dummy2 = np.zeros(tensor2.shape, dtype=np.float32)
            result_shape = (dummy1 + dummy2).shape
            logger.debug(f"Shapes {tensor1.shape} and {tensor2.shape} are broadcast-compatible for addition with result shape {result_shape}")
            return True
        except ValueError:
            logger.warning(f"Shapes {tensor1.shape} and {tensor2.shape} are not broadcast-compatible for addition")
            return False
    
    elif operation == "multiplication" or operation == "division":
        # For element-wise multiplication/division, same rules as addition
        return validate_shape_compatibility(tensor1, tensor2, "addition")
    
    elif operation == "matmul":
        # For matrix multiplication, check inner dimensions
        if tensor1.ndim < 1 or tensor2.ndim < 1:
            logger.warning(f"Matrix multiplication requires at least 1D tensors")
            return False
        
        # Handle vectors vs matrices
        if tensor1.ndim == 1 and tensor2.ndim == 1:
            # Vector dot product: dimensions must match
            if tensor1.shape[0] == tensor2.shape[0]:
                return True
            else:
                logger.warning(f"Vector dot product requires matching dimensions, got {tensor1.shape[0]} and {tensor2.shape[0]}")
                return False
        
        elif tensor1.ndim == 1:
            # Vector-matrix: vector length must match matrix columns
            if tensor1.shape[0] == tensor2.shape[-2]:
                return True
            else:
                logger.warning(f"Vector-matrix multiplication requires vector length {tensor1.shape[0]} to match matrix dimension {tensor2.shape[-2]}")
                return False
        
        elif tensor2.ndim == 1:
            # Matrix-vector: matrix columns must match vector length
            if tensor1.shape[-1] == tensor2.shape[0]:
                return True
            else:
                logger.warning(f"Matrix-vector multiplication requires matrix dimension {tensor1.shape[-1]} to match vector length {tensor2.shape[0]}")
                return False
        
        else:
            # Matrix-matrix: first matrix columns must match second matrix rows
            if tensor1.shape[-1] == tensor2.shape[-2]:
                return True
            else:
                logger.warning(f"Matrix multiplication requires dimensions {tensor1.shape[-1]} and {tensor2.shape[-2]} to match")
                return False
    
    elif operation == "broadcasting":
        try:
            # Use numpy's broadcasting rules directly
            np.broadcast(tensor1, tensor2)
            return True
        except ValueError:
            logger.warning(f"Shapes {tensor1.shape} and {tensor2.shape} are not broadcast-compatible")
            return False
    
    else:
        logger.warning(f"Unknown operation '{operation}'")
        return False

def validate_distribution(tensor: np.ndarray, distribution_type: str = "normal", 
                          params: Optional[Dict[str, float]] = None) -> bool:
    """
    Validate that a tensor follows a specific distribution (approximately).
    
    Args:
        tensor: Tensor to validate
        distribution_type: Type of distribution ('normal', 'uniform', 'binary')
        params: Distribution parameters (e.g., {'mean': 0, 'std': 1} for normal)
        
    Returns:
        True if tensor approximately follows the distribution, False otherwise
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return False
    
    # Flatten the tensor for distribution testing
    flat_tensor = tensor.flatten()
    
    if len(flat_tensor) < 10:
        logger.warning("Need at least 10 values for distribution validation")
        return False
    
    params = params or {}
    
    if distribution_type == "normal":
        # Check if approximately normally distributed
        mean = params.get("mean", 0)
        std = params.get("std", 1)
        
        # Get actual mean and std
        actual_mean = np.mean(flat_tensor)
        actual_std = np.std(flat_tensor)
        
        # Check if mean and std are close to expected
        mean_tolerance = params.get("mean_tolerance", 0.5)
        std_tolerance = params.get("std_tolerance", 0.5)
        
        mean_ok = abs(actual_mean - mean) <= mean_tolerance * std
        std_ok = abs(actual_std - std) <= std_tolerance * std
        
        if not mean_ok:
            logger.warning(f"Mean {actual_mean:.4f} not within tolerance of expected {mean:.4f}")
        if not std_ok:
            logger.warning(f"Std {actual_std:.4f} not within tolerance of expected {std:.4f}")
        
        return mean_ok and std_ok
    
    elif distribution_type == "uniform":
        # Check if approximately uniformly distributed
        min_val = params.get("min", 0)
        max_val = params.get("max", 1)
        
        # Check if values are within range
        in_range = validate_range(tensor, min_val, max_val)
        if not in_range:
            return False
        
        # Check if distribution is roughly uniform
        # This is a simplistic check - for a proper test, use a statistical test
        hist, _ = np.histogram(flat_tensor, bins=10, range=(min_val, max_val))
        hist_mean = np.mean(hist)
        hist_std = np.std(hist)
        
        # For uniform, std/mean should be relatively small
        uniformity = hist_std / hist_mean if hist_mean > 0 else float('inf')
        uniformity_threshold = params.get("uniformity_threshold", 0.5)
        
        is_uniform = uniformity <= uniformity_threshold
        
        if not is_uniform:
            logger.warning(f"Distribution not uniform enough: std/mean ratio {uniformity:.4f} > threshold {uniformity_threshold:.4f}")
        
        return is_uniform
    
    elif distribution_type == "binary":
        # Check if tensor contains mostly binary values (0 and 1, or -1 and 1)
        low_val = params.get("low", 0)
        high_val = params.get("high", 1)
        
        # Check if values are close to low or high
        tolerance = params.get("tolerance", 1e-5)
        
        low_mask = np.abs(flat_tensor - low_val) <= tolerance
        high_mask = np.abs(flat_tensor - high_val) <= tolerance
        binary_ratio = (np.sum(low_mask) + np.sum(high_mask)) / len(flat_tensor)
        
        threshold = params.get("threshold", 0.95)  # 95% of values should be binary
        
        is_binary = binary_ratio >= threshold
        
        if not is_binary:
            logger.warning(f"Tensor is not sufficiently binary: {binary_ratio:.2%} < threshold {threshold:.2%}")
        
        return is_binary
    
    else:
        logger.warning(f"Unknown distribution type '{distribution_type}'")
        return False

def validate_gradient(tensor: np.ndarray, gradient: np.ndarray) -> bool:
    """
    Validate that a gradient matches the expected shape and has valid values.
    
    Args:
        tensor: Original tensor
        gradient: Gradient tensor to validate
        
    Returns:
        True if gradient is valid, False otherwise
    """
    if not isinstance(tensor, np.ndarray) or not isinstance(gradient, np.ndarray):
        logger.warning(f"Both inputs must be numpy arrays")
        return False
    
    # Check shape compatibility
    if tensor.shape != gradient.shape:
        logger.warning(f"Gradient shape {gradient.shape} doesn't match tensor shape {tensor.shape}")
        return False
    
    # Check for NaN or Infinity
    if not validate_not_nan_or_inf(gradient):
        logger.warning("Gradient contains NaN or infinite values")
        return False
    
    # Check gradient magnitude
    grad_mag = np.mean(np.abs(gradient))
    if grad_mag > 1e10:
        logger.warning(f"Gradient magnitude too large: {grad_mag:.4e}")
        return False
    
    if grad_mag < 1e-10:
        logger.warning(f"Gradient magnitude suspiciously small: {grad_mag:.4e}")
        return False
    
    return True

def validate_custom(tensor: np.ndarray, validator_func: Callable[[np.ndarray], bool], 
                   name: str = "custom") -> bool:
    """
    Validate a tensor using a custom validation function.
    
    Args:
        tensor: Tensor to validate
        validator_func: Custom validation function that takes a tensor and returns a boolean
        name: Name of the custom validation for logging
        
    Returns:
        Result of the custom validation function
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return False
    
    try:
        is_valid = validator_func(tensor)
        if not is_valid:
            logger.warning(f"Custom validation '{name}' failed")
        return is_valid
    except Exception as e:
        logger.error(f"Error in custom validation '{name}': {e}")
        return False

def validate_tensor(tensor: np.ndarray, validations: Dict[str, Any]) -> Dict[str, bool]:
    """
    Run multiple validations on a tensor and return the results.
    
    Args:
        tensor: Tensor to validate
        validations: Dictionary mapping validation names to their parameters, e.g.,
                    {'shape': (3, 4), 'dtype': np.float32, 'range': (0, 1)}
        
    Returns:
        Dictionary mapping validation names to their results
    """
    if not isinstance(tensor, np.ndarray):
        logger.warning(f"Expected numpy array, got {type(tensor)}")
        return {name: False for name in validations}
    
    results = {}
    
    # Run each validation
    for name, params in validations.items():
        if name == 'shape':
            results['shape'] = validate_shape(tensor, params)
        elif name == 'dtype':
            results['dtype'] = validate_dtype(tensor, params)
        elif name == 'range':
            min_val, max_val = params
            results['range'] = validate_range(tensor, min_val, max_val)
        elif name == 'not_nan_or_inf':
            results['not_nan_or_inf'] = validate_not_nan_or_inf(tensor)
        elif name == 'custom':
            func, custom_name = params
            results[f'custom_{custom_name}'] = validate_custom(tensor, func, custom_name)
        elif name == 'distribution':
            dist_type = params.get('type', 'normal')
            dist_params = params.get('params', {})
            results['distribution'] = validate_distribution(tensor, dist_type, dist_params)
        elif name == 'gradient':
            gradient = params
            results['gradient'] = validate_gradient(tensor, gradient)
        elif name == 'shape_compatibility':
            other_tensor, operation = params
            results['shape_compatibility'] = validate_shape_compatibility(tensor, other_tensor, operation)
        else:
            logger.warning(f"Unknown validation type: {name}")
            results[name] = False
    
    return results

def create_validation_report(tensor: np.ndarray, validations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a detailed validation report for a tensor.
    
    Args:
        tensor: Tensor to validate
        validations: Dictionary mapping validation names to their parameters
        
    Returns:
        Dictionary with validation results and tensor information
    """
    if not isinstance(tensor, np.ndarray):
        return {"error": f"Expected numpy array, got {type(tensor)}"}
    
    # Get basic tensor information
    tensor_info = {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "size": tensor.size,
        "min": float(np.min(tensor)) if tensor.size > 0 else None,
        "max": float(np.max(tensor)) if tensor.size > 0 else None,
        "mean": float(np.mean(tensor)) if tensor.size > 0 else None,
        "has_nan": bool(np.any(np.isnan(tensor))),
        "has_inf": bool(np.any(np.isinf(tensor)))
    }
    
    # Add more tensor statistics for better insight
    tensor_info.update({
        "median": float(np.median(tensor)) if tensor.size > 0 else None,
        "percentile_25": float(np.percentile(tensor, 25)) if tensor.size > 0 else None,
        "percentile_75": float(np.percentile(tensor, 75)) if tensor.size > 0 else None,
        "std_dev": float(np.std(tensor)) if tensor.size > 0 else None,
        "sparsity": float(np.count_nonzero(tensor == 0) / tensor.size) if tensor.size > 0 else None
    })
    
    # Run validations
    validation_results = validate_tensor(tensor, validations)
    
    # Determine overall validity
    is_valid = all(validation_results.values())
    
    return {
        "is_valid": is_valid,
        "tensor_info": tensor_info,
        "validation_results": validation_results
    }

def validate_batch_of_tensors(tensors: List[np.ndarray], validations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a batch of tensors with the same validations.
    
    Args:
        tensors: List of tensors to validate
        validations: Dictionary mapping validation names to their parameters
        
    Returns:
        Dictionary with batch validation results
    """
    if not tensors:
        return {"error": "Empty tensor list"}
    
    # Validate each tensor
    individual_results = [validate_tensor(tensor, validations) for tensor in tensors]
    
    # Calculate statistics
    batch_size = len(tensors)
    valid_count = sum(1 for results in individual_results if all(results.values()))
    invalid_count = batch_size - valid_count
    
    # Aggregate results by validation type
    validation_stats = {}
    for name in validations:
        passed = sum(1 for results in individual_results if results.get(name, False))
        validation_stats[name] = {
            "passed": passed,
            "failed": batch_size - passed,
            "pass_rate": passed / batch_size if batch_size > 0 else 0
        }
    
    return {
        "batch_size": batch_size,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "success_rate": valid_count / batch_size if batch_size > 0 else 0,
        "validation_stats": validation_stats
    }

def visualize_tensor_statistics(tensor: np.ndarray, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize tensor statistics using histograms and other plots.
    
    Args:
        tensor: Tensor to visualize
        output_file: Optional file path to save the visualization
        
    Returns:
        Dictionary with visualization paths or base64-encoded images
    """
    if not isinstance(tensor, np.ndarray):
        return {"error": f"Expected numpy array, got {type(tensor)}"}
    
    result = {}
    
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        # Flatten the tensor for histogram
        flat_tensor = tensor.flatten()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Histogram
        axs[0, 0].hist(flat_tensor, bins=30)
        axs[0, 0].set_title('Value Distribution')
        axs[0, 0].set_xlabel('Value')
        axs[0, 0].set_ylabel('Frequency')
        
        # Cumulative distribution
        axs[0, 1].hist(flat_tensor, bins=30, cumulative=True, density=True)
        axs[0, 1].set_title('Cumulative Distribution')
        axs[0, 1].set_xlabel('Value')
        axs[0, 1].set_ylabel('Cumulative Probability')
        
        # Box plot
        axs[1, 0].boxplot(flat_tensor)
        axs[1, 0].set_title('Box Plot')
        
        # If 2D, show heatmap
        if tensor.ndim == 2:
            im = axs[1, 1].imshow(tensor, cmap='viridis')
            fig.colorbar(im, ax=axs[1, 1])
            axs[1, 1].set_title('2D Visualization')
        else:
            axs[1, 1].text(0.5, 0.5, f"Shape: {tensor.shape}\nDType: {tensor.dtype}", 
                          horizontalalignment='center', verticalalignment='center')
            axs[1, 1].set_title('Tensor Info')
            axs[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save or convert to base64
        if output_file:
            plt.savefig(output_file)
            result["visualization_path"] = output_file
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            result["visualization_base64"] = img_str
        
        plt.close(fig)
        result["success"] = True
        
    except ImportError:
        result["error"] = "matplotlib not available for visualization"
    except Exception as e:
        result["error"] = f"Error during visualization: {e}"
    
    return result

if __name__ == "__main__":
    # Example usage when run directly
    print("Tensor Validator Module")
    print("Create a sample tensor and run some validations:")
    
    sample_tensor = np.random.randn(3, 4)
    validations = {
        'shape': (3, 4),
        'dtype': np.float64,
        'range': (-10, 10),
        'not_nan_or_inf': True
    }
    
    results = validate_tensor(sample_tensor, validations)
    print("\nValidation Results:")
    for name, result in results.items():
        print(f"  {name}: {'✓' if result else '✗'}")
    
    report = create_validation_report(sample_tensor, validations)
    print("\nTensor Info:")
    for key, value in report["tensor_info"].items():
        print(f"  {key}: {value}")
