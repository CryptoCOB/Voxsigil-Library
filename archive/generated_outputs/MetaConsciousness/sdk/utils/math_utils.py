"""
Mathematical Utilities

This module provides common mathematical helper functions for the MetaConsciousness framework.
"""

import math
import logging
import numpy as np
from typing import List, Tuple, Optional, Union, Any, Dict, Callable

# Configure logger
logger = logging.getLogger(__name__)

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform division safely, returning a default value if the denominator is zero.
    
    Args:
        numerator: Division numerator
        denominator: Division denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator

def normalize_vector(vector: Union[List[float], np.ndarray], 
                    norm_type: int = 2) -> Union[List[float], np.ndarray]:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        norm_type: Type of norm (1=Manhattan, 2=Euclidean)
        
    Returns:
        Normalized vector
    """
    if isinstance(vector, list):
        vector = np.array(vector, dtype=float)
    
    # Handle zero vectors
    if np.all(np.abs(vector) < 1e-10):
        logger.warning("Attempted to normalize a zero vector")
        return vector
    
    # Calculate norm
    norm = np.linalg.norm(vector, ord=norm_type)
    
    # Normalize
    if norm > 1e-10:
        return vector / norm
    else:
        logger.warning(f"Vector norm too small ({norm}) for normalization")
        return vector

def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                     vec2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    if isinstance(vec1, list):
        vec1 = np.array(vec1, dtype=float)
    if isinstance(vec2, list):
        vec2 = np.array(vec2, dtype=float)
    
    # Check for zero vectors
    if np.all(np.abs(vec1) < 1e-10) or np.all(np.abs(vec2) < 1e-10):
        logger.warning("Cosine similarity calculated with a zero vector")
        return 0.0
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Handle numerical issues
    if similarity > 1.0:
        similarity = 1.0
    elif similarity < -1.0:
        similarity = -1.0
    
    return float(similarity)

def euclidean_distance(vec1: Union[List[float], np.ndarray],
                      vec2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    if isinstance(vec1, list):
        vec1 = np.array(vec1, dtype=float)
    if isinstance(vec2, list):
        vec2 = np.array(vec2, dtype=float)
    
    return float(np.linalg.norm(vec1 - vec2))

def calculate_ema(current: float, new_value: float, alpha: float = 0.1) -> float:
    """
    Calculate Exponential Moving Average.
    
    Args:
        current: Current EMA value
        new_value: New data point
        alpha: Smoothing factor (0-1)
        
    Returns:
        Updated EMA value
    """
    return alpha * new_value + (1 - alpha) * current

def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid value (0-1)
    """
    try:
        if x < -100:
            return 0
        elif x > 100:
            return 1
        return 1.0 / (1.0 + math.exp(-x))
    except (OverflowError, ValueError):
        if x < 0:
            return 0
        else:
            return 1

def softmax(values: Union[List[float], np.ndarray]) -> Union[List[float], np.ndarray]:
    """
    Apply the softmax function to a vector of values.
    
    Args:
        values: Input values
        
    Returns:
        Softmax probabilities (sum to 1)
    """
    is_list = isinstance(values, list)
    if is_list:
        values = np.array(values, dtype=float)
    
    # Subtract max value for numerical stability
    exp_values = np.exp(values - np.max(values))
    
    # Normalize
    result = exp_values / np.sum(exp_values)
    
    return result.tolist() if is_list else result

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between a minimum and maximum value.
    
    Args:
        value: Input value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))

def linear_interpolation(value: float, in_min: float, in_max: float, 
                        out_min: float, out_max: float) -> float:
    """
    Linearly interpolate a value from one range to another.
    
    Args:
        value: Input value in range [in_min, in_max]
        in_min: Input range minimum
        in_max: Input range maximum
        out_min: Output range minimum
        out_max: Output range maximum
        
    Returns:
        Interpolated value in range [out_min, out_max]
    """
    # Handle division by zero
    if abs(in_max - in_min) < 1e-10:
        logger.warning("Input range is too small for interpolation")
        return (out_min + out_max) / 2
    
    # Clamp input value to input range
    value = clamp(value, in_min, in_max)
    
    # Interpolate
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

def weighted_average(values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Calculate weighted average of values.
    
    Args:
        values: List of values
        weights: List of weights (None for simple average)
        
    Returns:
        Weighted average
    """
    if not values:
        return 0.0
    
    if weights is None:
        return sum(values) / len(values)
    
    if len(weights) != len(values):
        logger.warning(f"Weights length ({len(weights)}) does not match values length ({len(values)})")
        # Use as many weights as we have
        weights = weights[:len(values)]
        if not weights:
            return sum(values) / len(values)
    
    total_weight = sum(weights)
    if total_weight < 1e-10:
        logger.warning("Sum of weights is too small")
        return sum(values) / len(values)
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight

def exponential_decay(initial_value: float, decay_rate: float, steps: int) -> float:
    """
    Calculate exponential decay.
    
    Args:
        initial_value: Starting value
        decay_rate: Rate of decay
        steps: Number of steps to decay
        
    Returns:
        Decayed value
    """
    return initial_value * math.pow(1 - decay_rate, steps)

def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel.
    
    Args:
        size: Kernel size (odd number)
        sigma: Standard deviation
        
    Returns:
        Gaussian kernel array
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    # Generate kernel
    kernel = np.zeros(size)
    center = size // 2
    
    for i in range(size):
        x = i - center
        kernel[i] = math.exp(-(x * x) / (2 * sigma * sigma))
    
    # Normalize
    return kernel / np.sum(kernel)

def gaussian_kernel_2d(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        size: Kernel size (odd number)
        sigma: Standard deviation
        
    Returns:
        2D Gaussian kernel array
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    # Generate 1D kernel
    kernel_1d = gaussian_kernel(size, sigma)
    
    # Generate 2D kernel as outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Normalize
    return kernel_2d / np.sum(kernel_2d)

def calculate_moving_average(data: List[float], window_size: int = 3) -> List[float]:
    """
    Calculate simple moving average of a data series.
    
    Args:
        data: Input data series
        window_size: Window size for averaging
        
    Returns:
        Moving average series
    """
    if not data:
        return []
    
    # Ensure window size is valid
    window_size = min(window_size, len(data))
    window_size = max(1, window_size)
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window = data[start_idx:i+1]
        result.append(sum(window) / len(window))
    
    return result

def round_to_nearest(value: float, step: float = 1.0) -> float:
    """
    Round a value to the nearest multiple of a step value.
    
    Args:
        value: Value to round
        step: Step value
        
    Returns:
        Rounded value
    """
    if step <= 0:
        return value
    
    return round(value / step) * step
