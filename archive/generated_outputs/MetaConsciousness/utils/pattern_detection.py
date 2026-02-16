"""
Pattern Detection Module

This module provides functions for detecting various patterns in data.
"""
import cv2
import numpy as np
__all__ = []
import logging # Feature-6: Added for enhanced logging
from typing import Dict, Any, Tuple, List, Optional, Union, Sequence # Enhanced typing
from scipy import signal # Feature-1: Added for more robust correlation/convolution
from scipy import stats # Feature-3: Added for statistical significance

# --- Setup Logging ---
# EncapsulatedFeature-1: Configure Basic Logging
def _setup_logging() -> None:
    """Configure basic logging for the module."""
    logger = logging.getLogger(__name__)
    if not logger.handlers: # Avoid adding multiple handlers if already configured
        logger.setLevel(logging.INFO) # Default level, can be overridden
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = _setup_logging()

# --- Constants ---
# EncapsulatedFeature-2: Define Default Thresholds
DEFAULT_CHECKERBOARD_THRESHOLD = 0.7 # Adjusted based on potential improvements
DEFAULT_GRADIENT_THRESHOLD = 0.6 # Adjusted
DEFAULT_ADVERSARIAL_THRESHOLD = 0.5 # Adjusted
DEFAULT_REPEATING_THRESHOLD = 0.65 # Adjusted
DEFAULT_NOISE_THRESHOLD = 0.4 # Feature-2: Noise threshold
DEFAULT_EDGE_THRESHOLD = 0.5 # Feature-4: Edge threshold
DEFAULT_CYCLE_THRESHOLD = 0.6 # Feature-5: Cycle threshold

# --- Encapsulated Features (Helpers) ---

# EncapsulatedFeature-3: Robust Data Normalization
def _normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize numpy array data to a specific range (usually 0-1).

    Args:
        data: Input numpy array.
        method: Normalization method ('minmax', 'zscore', or 'none').

    Returns:
        Normalized numpy array. Returns original data if normalization fails or method is 'none'.
    """
    if not isinstance(data, np.ndarray) or data.size == 0:
        logger.debug("Normalization skipped: Input is not a valid numpy array.")
        return data

    original_dtype = data.dtype
    # Promote integer types to float for calculations
    if np.issubdtype(original_dtype, np.integer):
        data = data.astype(np.float64)

    try:
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                # Avoid division by zero; return array of zeros or 0.5s? Let's use zeros.
                logger.debug("Normalization skipped: Data range is zero (all values equal).")
                return np.zeros_like(data, dtype=original_dtype) # Return original type if possible
            normalized = (data - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                logger.debug("Normalization skipped: Standard deviation is zero.")
                return np.zeros_like(data, dtype=original_dtype)
            normalized = (data - mean_val) / std_val
            # Optional: Clip or scale z-scores if needed for specific applications
        elif method == 'none':
             normalized = data
        else:
            logger.warning(f"Unsupported normalization method: {method}. Returning original data.")
            normalized = data

        # Attempt to cast back if reasonable, otherwise return float
        if method != 'zscore' and np.issubdtype(original_dtype, np.integer):
             # Check if normalized data is effectively integer range (e.g., 0 or 1)
             if np.all(np.isin(normalized, [0, 1])):
                 return normalized.astype(original_dtype)
        return normalized

    except (ValueError, TypeError, FloatingPointError) as e:
        logger.error(f"Error during data normalization ({method}): {e}", exc_info=True)
        return data # Return original data on error

# EncapsulatedFeature-4: 2D Convolution Helper
def _convolve_2d(data: np.ndarray, kernel: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply 2D convolution using scipy.signal.

    Args:
        data: Input 2D numpy array.
        kernel: 2D kernel/filter.

    Returns:
        Convolved data array, or None if inputs are invalid.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in [data, kernel]):
        logger.warning("Invalid input for 2D convolution: Input and kernel must be 2D numpy arrays.")
        return None
    try:
        return signal.convolve2d(data, kernel, mode='same', boundary='wrap')
    except Exception as e:
        logger.error(f"Error during 2D convolution: {e}", exc_info=True)
        return None

# EncapsulatedFeature-5: Calculate Statistical Significance (Placeholder/Simple)
def _calculate_significance(confidence: float, n_samples: Optional[int] = None) -> float:
    """
    Estimate a p-value or significance score based on confidence.
    NOTE: This is a placeholder. Real significance testing requires specific statistical models.

    Args:
        confidence: The confidence score (e.g., correlation coefficient).
        n_samples: Optional number of samples used to calculate confidence.

    Returns:
        An estimated significance score (lower is more significant, e.g., p-value like).
    """
    # Very simplistic mapping: higher confidence -> lower p-value estimate
    # This does NOT reflect true statistical significance.
    p_value_estimate = max(0.0, 1.0 - confidence)
    # Optionally adjust based on N if provided (e.g., higher N makes same confidence more significant)
    # Example adjustment (needs proper statistical basis):
    # if n_samples and n_samples > 10:
    #    p_value_estimate /= np.log1p(n_samples) # Decrease p for larger N

    # Ensure p-value is within [0, 1]
    p_value_estimate = np.clip(p_value_estimate, 0.0, 1.0)
    return p_value_estimate

# EncapsulatedFeature-6: Data Validation (Dimension Check)
def _validate_2d_input(data: Any) -> Optional[np.ndarray]:
    """Validate if input is a 2D numpy array, converting if possible."""
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            return data
        else:
            logger.warning(f"Input data has incorrect dimensions ({data.ndim}D). Expected 2D.")
            return None
    elif isinstance(data, list):
        try:
            arr = np.array(data)
            if arr.ndim == 2:
                return arr
            else:
                logger.warning(f"Input list converted to array has incorrect dimensions ({arr.ndim}D). Expected 2D.")
                return None
        except Exception as e:
            logger.warning(f"Could not convert input list to 2D numpy array: {e}")
            return None
    else:
        logger.warning(f"Invalid input type ({type(data)}). Expected 2D numpy array or list.")
        return None

# EncapsulatedFeature-7: Root Mean Squared Error (RMSE)
def _calculate_rmse(data1: np.ndarray, data2: np.ndarray) -> Optional[float]:
    """Calculate the Root Mean Squared Error between two numpy arrays."""
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        logger.warning("RMSE calculation requires numpy arrays.")
        return None
    if data1.shape != data2.shape:
        logger.warning(f"RMSE calculation requires arrays of the same shape ({data1.shape} vs {data2.shape}).")
        return None
    try:
        return np.sqrt(np.mean((data1 - data2)**2))
    except Exception as e:
        logger.error(f"Error calculating RMSE: {e}", exc_info=True)
        return None

# EncapsulatedFeature-8: Peak Detection Helper
def _find_peaks(data: np.ndarray, min_distance: int = 1, min_prominence: float = 0.1) -> np.ndarray:
    """Find peaks in a 1D numpy array."""
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        logger.warning("Peak detection requires a 1D numpy array.")
        return np.array([], dtype=int)
    try:
        # Normalize prominence relative to data range
        data_range = np.ptp(data) if np.ptp(data) > 1e-6 else 1.0
        normalized_prominence = min_prominence * data_range
        peaks, _ = signal.find_peaks(data, distance=min_distance, prominence=normalized_prominence)
        return peaks
    except Exception as e:
        logger.error(f"Error during peak finding: {e}", exc_info=True)
        return np.array([], dtype=int)

# EncapsulatedFeature-9: Autocorrelation Function
def _calculate_autocorrelation(data: np.ndarray) -> Optional[np.ndarray]:
    """Calculate the autocorrelation of a 1D signal."""
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        logger.warning("Autocorrelation requires a 1D numpy array.")
        return None
    try:
        # Ensure data is float and subtract mean
        data_float = data.astype(float)
        data_centered = data_float - np.mean(data_float)
        # Use scipy's correlate function
        autocorr = signal.correlate(data_centered, data_centered, mode='full')
        # Normalize and return the second half (positive lags)
        autocorr = autocorr / (np.std(data_centered)**2 * len(data_centered))
        return autocorr[len(data_centered)-1:]
    except Exception as e:
        logger.error(f"Error calculating autocorrelation: {e}", exc_info=True)
        return None

# EncapsulatedFeature-10: Generate Ideal Checkerboard
def _generate_ideal_checkerboard(shape: Tuple[int, int], scale: int = 1, phase: int = 0) -> np.ndarray:
    """
    Generate an ideal checkerboard pattern of a given shape and scale.

    Args:
        shape: Tuple (height, width) of the desired pattern.
        scale: The size of each checker square.
        phase: Starting phase (0 or 1) for the top-left square.

    Returns:
        A numpy array with the checkerboard pattern (0s and 1s).
    """
    h, w = shape
    scale = max(1, int(scale))
    # Create indices scaled by the checker size
    y, x = np.indices((h, w)) // scale
    checkerboard = np.zeros((h, w))
    checkerboard[(x + y + phase) % 2 == 0] = 1
    return checkerboard

# EncapsulatedFeature-11: Calculate Image Entropy
def _calculate_entropy(data: np.ndarray, bins: int = 256) -> Optional[float]:
    """Calculate the Shannon entropy of an image (or data array)."""
    if not isinstance(data, np.ndarray) or data.size == 0:
        return None
    try:
        # Calculate histogram
        hist, _ = np.histogram(data.flatten(), bins=bins, density=True)
        # Calculate entropy
        entropy = stats.entropy(hist)
        return entropy
    except Exception as e:
        logger.error(f"Error calculating entropy: {e}", exc_info=True)
        return None

# EncapsulatedFeature-12: Sobel Edge Detection Filter
def _sobel_filter(data: np.ndarray) -> Optional[np.ndarray]:
    """Apply Sobel filter to detect edges."""
    if not isinstance(data, np.ndarray) or data.ndim != 2:
         logger.warning("Sobel filter requires 2D numpy array.")
         return None
    try:
        dx = signal.convolve2d(data, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], mode='same', boundary='symm')
        dy = signal.convolve2d(data, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], mode='same', boundary='symm')
        magnitude = np.sqrt(dx**2 + dy**2)
        return _normalize_data(magnitude) # Normalize magnitude to 0-1
    except Exception as e:
        logger.error(f"Error applying Sobel filter: {e}", exc_info=True)
        return None

# EncapsulatedFeature-13: Structural Similarity Index (SSIM) - Basic Placeholder
def _calculate_ssim_basic(data1: np.ndarray, data2: np.ndarray) -> Optional[float]:
    """Basic SSIM calculation (placeholder, use skimage for real implementation)."""
    if not all(isinstance(arr, np.ndarray) and arr.shape == data1.shape for arr in [data1, data2]):
        logger.warning("Basic SSIM requires two numpy arrays of the same shape.")
        return None
    # Very basic version using means and std devs - NOT a proper SSIM
    k1, k2 = 0.01, 0.03 # Default constants
    L = np.max([np.ptp(data1), np.ptp(data2)]) # Potential dynamic range (use 1 if normalized)
    c1 = (k1 * L)**2
    c2 = (k2 * L)**2

    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    sigma1_sq = np.var(data1)
    sigma2_sq = np.var(data2)
    sigma12 = np.mean((data1 - mu1) * (data2 - mu2)) # Covariance

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0 # Handle edge case

    ssim = numerator / denominator
    return np.clip(ssim, 0.0, 1.0)


# EncapsulatedFeature-14: Sequence Differencing
def _difference_sequence(sequence: Union[List[float], np.ndarray], order: int = 1) -> Optional[np.ndarray]:
    """Calculate the difference of a sequence."""
    try:
        data = np.array(sequence) if not isinstance(sequence, np.ndarray) else sequence
        if data.ndim != 1:
            logger.warning("Differencing requires a 1D sequence.")
            return None
        if order < 1:
            return data # No differencing needed
        diff_data = np.diff(data, n=order)
        # Pad with NaNs to maintain original length for some applications if needed
        # Or return the shorter differenced array directly
        return diff_data
    except Exception as e:
        logger.error(f"Error differencing sequence: {e}", exc_info=True)
        return None

# EncapsulatedFeature-15: Result Dictionary Factory
def _create_result_dict(detected: bool, confidence: Optional[float] = None, **kwargs) -> Dict[str, Any]:
    """Creates a standardized result dictionary."""
    result = {"detected": bool(detected)}
    if confidence is not None:
        result["confidence"] = float(np.clip(confidence, 0.0, 1.0)) if not np.isnan(confidence) else 0.0
    # Feature-3: Add significance estimate
    if confidence is not None:
        result["significance_estimate"] = _calculate_significance(result["confidence"])

    result.update(kwargs) # Add any extra information
    return result

# --- Main Detection Functions ---

def detect_checkerboard(data: np.ndarray,
                        threshold: float = DEFAULT_CHECKERBOARD_THRESHOLD,
                        scale: int = 1 # Feature-7: Add scale parameter
                        ) -> Dict[str, Any]: # Return dict instead of tuple
    """
    Detect checkerboard patterns using correlation with an ideal pattern.

    Args:
        data: Input data (2D array).
        threshold: Detection threshold (based on correlation coefficient).
        scale: The approximate size of the checker squares to look for.

    Returns:
        Dictionary with detection result, confidence (correlation), and scale used.
    """
    validated_data = _validate_2d_input(data) # E6
    if validated_data is None:
        return _create_result_dict(False, 0.0, error="Invalid 2D input data") # E15

    h, w = validated_data.shape
    if h < 2 or w < 2:
         return _create_result_dict(False, 0.0, reason="Data dimensions too small")

    # E3: Normalize input data robustly
    normalized_data = _normalize_data(validated_data, method='minmax')

    # E10: Create ideal checkerboards with phase 0 and 1
    checkerboard_p0 = _generate_ideal_checkerboard((h, w), scale=scale, phase=0)
    checkerboard_p1 = _generate_ideal_checkerboard((h, w), scale=scale, phase=1)

    # Calculate correlation with both phases
    try:
        corr_p0 = np.corrcoef(checkerboard_p0.flatten(), normalized_data.flatten())[0, 1]
        corr_p1 = np.corrcoef(checkerboard_p1.flatten(), normalized_data.flatten())[0, 1]

        # Handle potential NaN results from corrcoef if variance is zero
        corr_p0 = 0.0 if np.isnan(corr_p0) else corr_p0
        corr_p1 = 0.0 if np.isnan(corr_p1) else corr_p1

        # Use the phase with the higher absolute correlation
        abs_corr_p0 = abs(corr_p0)
        abs_corr_p1 = abs(corr_p1)

        if abs_corr_p0 >= abs_corr_p1:
            confidence = abs_corr_p0
            matched_phase = 0
        else:
            confidence = abs_corr_p1
            matched_phase = 1

    except Exception as e:
        logger.error(f"Error calculating checkerboard correlation: {e}", exc_info=True)
        return _create_result_dict(False, 0.0, error=f"Correlation calculation failed: {e}")

    detected = confidence > threshold
    logger.debug(f"Checkerboard detection: Confidence={confidence:.3f}, Detected={detected}, Scale={scale}, Phase={matched_phase}")
    return _create_result_dict(detected, confidence, scale=scale, phase=matched_phase) # E15


def detect_gradients(data: np.ndarray,
                     threshold: float = DEFAULT_GRADIENT_THRESHOLD,
                     method: str = 'sobel' # Feature-4: Allow different gradient methods
                     ) -> Dict[str, Any]: # Return dict
    """
    Detect significant gradient patterns in 2D data.

    Args:
        data: Input data (2D array).
        threshold: Detection threshold (based on gradient consistency/strength).
        method: Method for gradient calculation ('numpy' or 'sobel').

    Returns:
        Dictionary with detection result, confidence, gradient magnitude stats.
    """
    validated_data = _validate_2d_input(data) # E6
    if validated_data is None:
        return _create_result_dict(False, 0.0, error="Invalid 2D input data") # E15

    if validated_data.size < 4: # Need at least 2x2 for gradients
        return _create_result_dict(False, 0.0, reason="Data dimensions too small")

    try:
        if method == 'sobel':
             # Use Sobel magnitude (already normalized 0-1 by helper)
             gradient_magnitude = _sobel_filter(validated_data) # E12
             if gradient_magnitude is None: # Handle filter error
                 return _create_result_dict(False, 0.0, error="Sobel filter calculation failed")
        elif method == 'numpy':
            # Calculate gradients using numpy.gradient
            gx = np.gradient(validated_data.astype(float), axis=1)
            gy = np.gradient(validated_data.astype(float), axis=0)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            # Normalize magnitude
            gradient_magnitude = _normalize_data(gradient_magnitude, method='minmax') # E3
        else:
             logger.warning(f"Unsupported gradient method: {method}. Using 'numpy'.")
             gx = np.gradient(validated_data.astype(float), axis=1)
             gy = np.gradient(validated_data.astype(float), axis=0)
             gradient_magnitude = np.sqrt(gx**2 + gy**2)
             gradient_magnitude = _normalize_data(gradient_magnitude, method='minmax')

        # Calculate statistics on the normalized gradient magnitude
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        grad_max = np.max(gradient_magnitude)

        # Confidence score: Higher mean magnitude and lower standard deviation suggest a strong, consistent gradient.
        # Avoid division by zero if std is very small (e.g., perfectly flat gradient)
        # A high mean indicates strength, (1 - std) indicates consistency (for normalized data).
        consistency_score = 1.0 - grad_std if grad_mean > 1e-3 else 0.0 # Only consider consistency if gradient isn't near zero
        strength_score = grad_mean
        confidence = (strength_score * 0.6 + consistency_score * 0.4) # Weighted average

        # Ensure confidence is within [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)

    except Exception as e:
        logger.error(f"Error calculating gradients: {e}", exc_info=True)
        return _create_result_dict(False, 0.0, error=f"Gradient calculation failed: {e}")

    detected = confidence > threshold
    logger.debug(f"Gradient detection: Confidence={confidence:.3f}, Detected={detected}, MeanMag={grad_mean:.3f}, StdMag={grad_std:.3f}")
    return _create_result_dict(detected, confidence, mean_magnitude=grad_mean, std_magnitude=grad_std, max_magnitude=grad_max) # E15


def detect_adversarial_patterns(data: np.ndarray,
                                reference: Optional[np.ndarray] = None,
                                threshold: float = DEFAULT_ADVERSARIAL_THRESHOLD,
                                method: str = 'diff_std' # Feature-2: Add noise detection method
                                ) -> Dict[str, Any]: # Return dict
    """
    Detect potential high-frequency or anomalous patterns, possibly adversarial.

    Args:
        data: Input data (numpy array).
        reference: Optional reference (clean) data for comparison.
        threshold: Detection threshold.
        method: Method ('diff_std', 'fft_high_freq', 'entropy').

    Returns:
        Dictionary with detection result and confidence score.
    """
    if not isinstance(data, np.ndarray) or data.ndim < 1:
         return _create_result_dict(False, 0.0, error="Invalid input data")

    score = 0.0
    details = {}

    try:
        if reference is not None:
            if data.shape != reference.shape:
                logger.warning("Data and reference shapes mismatch for adversarial detection.")
                return _create_result_dict(False, 0.0, error="Data and reference shape mismatch")

            diff = data.astype(float) - reference.astype(float)
            # Normalize difference relative to data range? Or absolute difference?
            data_range = np.ptp(data) if np.ptp(data) > 1e-6 else 1.0
            norm_diff = diff / data_range if data_range > 0 else diff

            if method == 'diff_std':
                # Original approach: Higher std dev in difference indicates noise
                diff_std = np.std(norm_diff)
                diff_mean_abs = np.mean(np.abs(norm_diff))
                # Combine std and mean absolute difference
                score = (diff_std * 0.7 + diff_mean_abs * 0.3) # Weighted
                details = {'diff_std': diff_std, 'diff_mean_abs': diff_mean_abs}
            elif method == 'entropy':
                 # Calculate entropy of the difference signal
                 diff_entropy = _calculate_entropy(norm_diff) # E11
                 # Compare with baseline entropy? Lower entropy might indicate structured noise.
                 # This needs refinement - higher entropy usually means more random noise.
                 # Let's use deviation from mean entropy of reference as a metric? Complex.
                 # Simpler: assume high entropy diff => more noise. Needs validation.
                 score = np.clip(diff_entropy / 5.0, 0, 1) if diff_entropy is not None else 0.0 # Normalize roughly
                 details = {'difference_entropy': diff_entropy}
            elif method == 'ssim': # Feature-10: Use SSIM
                 ssim_score = _calculate_ssim_basic(data, reference) # E13
                 if ssim_score is not None:
                      score = 1.0 - ssim_score # Lower SSIM means more difference/potential pattern
                      details = {'ssim_score': ssim_score}
                 else: score = 0.0
            else:
                 logger.warning(f"Unsupported adversarial method with reference: {method}. Using 'diff_std'.")
                 diff_std = np.std(norm_diff)
                 diff_mean_abs = np.mean(np.abs(norm_diff))
                 score = (diff_std * 0.7 + diff_mean_abs * 0.3)
                 details = {'diff_std': diff_std, 'diff_mean_abs': diff_mean_abs}

        else: # No reference provided - check internal high-frequency content
            if data.ndim >= 2:
                # Use first channel/slice if more than 2D
                target_data = data if data.ndim == 2 else data[..., 0]
                target_data = target_data.astype(float)

                if method == 'fft_high_freq': # Feature-2 Method
                     # Calculate FFT and check energy in high frequencies
                    try:
                        fft_data = np.fft.fft2(target_data)
                        fft_shifted = np.fft.fftshift(fft_data)
                        rows, cols = target_data.shape
                        crow, ccol = rows // 2 , cols // 2
                        # Define high-frequency region (e.g., outer quarter)
                        mask = np.ones((rows, cols), dtype=bool)
                        radius_low = min(crow, ccol) // 4 # Example low freq radius
                        y, x = np.ogrid[:rows, :cols]
                        mask_area = (x - ccol)**2 + (y - crow)**2 <= radius_low**2
                        mask[mask_area] = False # Mask out low frequencies

                        total_energy = np.sum(np.abs(fft_shifted)**2)
                        high_freq_energy = np.sum(np.abs(fft_shifted[mask])**2)
                        if total_energy > 1e-9:
                            score = high_freq_energy / total_energy
                        else: score = 0.0
                        details = {'high_freq_energy_ratio': score}
                    except Exception as fft_e:
                         logger.error(f"FFT calculation failed: {fft_e}", exc_info=True)
                         score = 0.0 # Fallback if FFT fails
                         details = {'error': 'FFT calculation failed'}

                elif method == 'entropy': # Feature-2 Method
                     img_entropy = _calculate_entropy(target_data) # E11
                     # Higher entropy might suggest noise, needs baseline/calibration
                     score = np.clip(img_entropy / 8.0, 0, 1) if img_entropy is not None else 0.0 # Rough normalization (max entropy for 8bit ~ 8)
                     details = {'image_entropy': img_entropy}
                else: # Default: Use gradient std / mean (original approach refined)
                     gx = np.gradient(target_data, axis=1)
                     gy = np.gradient(target_data, axis=0)
                     gradient_magnitude = np.sqrt(gx**2 + gy**2)
                     grad_mean = np.mean(gradient_magnitude)
                     grad_std = np.std(gradient_magnitude)
                     # High std relative to mean suggests high variance/frequency
                     score = grad_std / (grad_mean + 1e-6) # Relative std dev
                     score = np.clip(score / 5.0, 0, 1) # Normalize roughly
                     details = {'gradient_relative_std': score * 5.0} # Store unnormalized rel std
            else: # 1D data
                logger.debug("Adversarial detection without reference on 1D data is limited.")
                # Check variance of differences
                diffs = _difference_sequence(data) # E14
                if diffs is not None and diffs.size > 0:
                     rel_std = np.std(diffs) / (np.mean(np.abs(data)) + 1e-6)
                     score = np.clip(rel_std, 0, 1) # Simple score based on relative diff std
                     details = {'difference_relative_std': rel_std}
                else: score = 0.0

    except Exception as e:
        logger.error(f"Error detecting adversarial patterns: {e}", exc_info=True)
        return _create_result_dict(False, 0.0, error=f"Detection failed: {e}")

    detected = score > threshold
    logger.debug(f"Adversarial detection (method={method}): Score={score:.3f}, Detected={detected}")
    return _create_result_dict(detected, score, method=method, **details) # E15


def detect_repeating_patterns(data: np.ndarray,
                             threshold: float = DEFAULT_REPEATING_THRESHOLD,
                             method: str = 'autocorr' # Feature-8: Add method option
                             ) -> Dict[str, Any]: # Return dict
    """
    Detect repeating patterns using autocorrelation or sequence matching.

    Args:
        data: Input data (numpy array, can be 1D or 2D).
        threshold: Detection threshold (based on autocorrelation peak or similarity).
        method: Method ('autocorr' or 'fft_peak').

    Returns:
        Dictionary with detection result, confidence, and detected period if found.
    """
    if not isinstance(data, np.ndarray) or data.ndim < 1:
        return _create_result_dict(False, 0.0, error="Invalid input data")

    # Flatten multi-dimensional data for sequence analysis if needed, or process 2D?
    # Let's prioritize 1D or use first row/column for 2D for simplicity with autocorrelation.
    if data.ndim >= 2:
        logger.debug("Using first row for repeating pattern detection on multi-dimensional data.")
        signal_data = data[0, :].flatten() # Use first row
    else:
        signal_data = data.flatten()

    if signal_data.size < 4: # Need enough points for correlation/fft
        return _create_result_dict(False, 0.0, reason="Insufficient data points")

    confidence = 0.0
    period = None
    details = {'method': method}

    try:
        if method == 'autocorr':
            autocorr = _calculate_autocorrelation(signal_data) # E9
            if autocorr is None or autocorr.size < 2:
                 return _create_result_dict(False, 0.0, error="Autocorrelation failed")

            # Find peaks in autocorrelation (excluding the zero lag peak)
            # E8: Use peak finding helper
            peaks = _find_peaks(autocorr[1:], min_distance=2, min_prominence=0.1) # Look for peaks after lag 0
            peaks = peaks + 1 # Adjust index back relative to original autocorr array

            if len(peaks) > 0:
                # Confidence is the height of the highest significant peak
                peak_heights = autocorr[peaks]
                best_peak_idx = np.argmax(peak_heights)
                confidence = peak_heights[best_peak_idx]
                period = peaks[best_peak_idx] # Period is the lag of the highest peak
                details['detected_period'] = int(period)
                details['peak_confidence'] = float(confidence)
            else:
                confidence = 0.0 # No significant peaks found

        elif method == 'fft_peak': # Feature-8 method
             # Find dominant frequency using FFT
             try:
                 fft_vals = np.fft.fft(signal_data)
                 fft_freq = np.fft.fftfreq(signal_data.size)
                 # Find peak frequency (excluding DC component at index 0)
                 idx = np.argmax(np.abs(fft_vals[1:])) + 1
                 dominant_freq = np.abs(fft_freq[idx])
                 if dominant_freq > 1e-6: # Avoid division by zero if flat signal
                     period = int(round(1.0 / dominant_freq))
                     # Confidence based on relative magnitude of the peak? Needs refinement.
                     peak_magnitude = np.abs(fft_vals[idx])
                     total_magnitude = np.sum(np.abs(fft_vals[1:]))
                     confidence = peak_magnitude / total_magnitude if total_magnitude > 0 else 0.0
                     details['detected_period'] = period
                     details['fft_peak_freq'] = float(dominant_freq)
                     details['fft_peak_confidence'] = float(confidence)
                 else:
                     confidence = 0.0
             except Exception as fft_e:
                 logger.error(f"FFT calculation failed for repeating patterns: {fft_e}", exc_info=True)
                 confidence = 0.0
                 details['error'] = 'FFT calculation failed'
        else:
             logger.warning(f"Unsupported repeating pattern method: {method}. Using 'autocorr'.")
             # Rerun autocorr logic (or refactor)
             autocorr = _calculate_autocorrelation(signal_data)
             if autocorr is not None and autocorr.size > 1:
                 peaks = _find_peaks(autocorr[1:], min_distance=2, min_prominence=0.1) + 1
                 if len(peaks) > 0:
                     peak_heights = autocorr[peaks]
                     best_peak_idx = np.argmax(peak_heights)
                     confidence = peak_heights[best_peak_idx]
                     period = peaks[best_peak_idx]
                     details = {'method': 'autocorr', 'detected_period': int(period), 'peak_confidence': float(confidence)}
                 else: confidence = 0.0
             else: confidence = 0.0

    except Exception as e:
        logger.error(f"Error detecting repeating patterns: {e}", exc_info=True)
        return _create_result_dict(False, 0.0, error=f"Detection failed: {e}", **details)

    detected = confidence > threshold
    logger.debug(f"Repeating pattern detection (method={details['method']}): Confidence={confidence:.3f}, Detected={detected}, Period={details.get('detected_period')}")
    return _create_result_dict(detected, confidence, **details) # E15


# --- New Functional Features ---

# Feature-2: Detect Noise Level
def detect_noise_level(data: np.ndarray,
                        threshold: float = DEFAULT_NOISE_THRESHOLD,
                        method: str = 'std_dev') -> Dict[str, Any]:
    """
    Estimate the noise level in the data.

    Args:
        data: Input numpy array.
        threshold: Threshold for considering noise 'high'.
        method: Method ('std_dev', 'entropy', 'high_freq').

    Returns:
        Dictionary with noise level estimate (0-1) and detection flag.
    """
    if not isinstance(data, np.ndarray) or data.size == 0:
        return _create_result_dict(False, 0.0, error="Invalid input data")

    noise_score = 0.0
    details = {'method': method}

    try:
        # Normalize data first to make scores comparable
        norm_data = _normalize_data(data, method='minmax')

        if method == 'std_dev':
             noise_score = np.std(norm_data)
             details['std_deviation'] = noise_score
        elif method == 'entropy':
             entropy = _calculate_entropy(norm_data) # E11
             noise_score = np.clip(entropy / 8.0, 0, 1) if entropy is not None else 0.0 # Rough normalization
             details['entropy'] = entropy
        elif method == 'high_freq' and norm_data.ndim == 2:
             # Reuse FFT logic from adversarial detection
             try:
                 fft_data = np.fft.fft2(norm_data)
                 fft_shifted = np.fft.fftshift(fft_data)
                 rows, cols = norm_data.shape
                 crow, ccol = rows // 2 , cols // 2
                 mask = np.ones((rows, cols), dtype=bool)
                 radius_low = min(crow, ccol) // 4
                 y, x = np.ogrid[:rows, :cols]
                 mask_area = (x - ccol)**2 + (y - crow)**2 <= radius_low**2
                 mask[mask_area] = False

                 total_energy = np.sum(np.abs(fft_shifted)**2)
                 high_freq_energy = np.sum(np.abs(fft_shifted[mask])**2)
                 noise_score = high_freq_energy / total_energy if total_energy > 1e-9 else 0.0
                 details['high_freq_energy_ratio'] = noise_score
             except Exception as fft_e:
                 logger.warning(f"FFT failed for noise detection, falling back to std_dev: {fft_e}")
                 noise_score = np.std(norm_data) # Fallback
                 details['method'] = 'std_dev' # Update method used
                 details['std_deviation'] = noise_score
        elif method == 'high_freq':
             logger.warning("High frequency noise detection requires 2D data. Falling back to std_dev.")
             noise_score = np.std(norm_data) # Fallback for non-2D
             details['method'] = 'std_dev' # Update method used
             details['std_deviation'] = noise_score
        else:
             logger.warning(f"Unsupported noise detection method: {method}. Using 'std_dev'.")
             noise_score = np.std(norm_data)
             details['method'] = 'std_dev' # Update method used
             details['std_deviation'] = noise_score

    except Exception as e:
        logger.error(f"Error detecting noise level: {e}", exc_info=True)
        return _create_result_dict(False, 0.0, error=f"Detection failed: {e}", **details)

    # Noise detected if score exceeds threshold
    detected = noise_score > threshold
    logger.debug(f"Noise level detection (method={details['method']}): Score={noise_score:.3f}, Detected={detected}")
    # Return score as confidence, detected flag based on threshold
    return _create_result_dict(detected, noise_score, **details) # E15

# Feature-4: Detect Edges
def detect_edges(data: np.ndarray,
                 threshold: float = DEFAULT_EDGE_THRESHOLD,
                 method: str = 'sobel' # Add option for other edge detectors later
                 ) -> Dict[str, Any]:
    """
    Detect edges in 2D data.

    Args:
        data: Input 2D numpy array.
        threshold: Threshold for edge density or strength.
        method: Edge detection method ('sobel').

    Returns:
        Dictionary with edge detection results.
    """
    validated_data = _validate_2d_input(data) # E6
    if validated_data is None:
        return _create_result_dict(False, 0.0, error="Invalid 2D input data")

    edge_map = None
    edge_density = 0.0
    details = {'method': method}

    try:
        if method == 'sobel':
            edge_map = _sobel_filter(validated_data) # E12
            if edge_map is None:
                 return _create_result_dict(False, 0.0, error="Sobel filter failed")
            # Confidence based on edge density (pixels above a certain magnitude threshold)
            edge_density = np.mean(edge_map > 0.5) # Example: density of pixels > 50% max magnitude
            details['edge_density'] = float(edge_density)
        # Add other methods like 'canny' if skimage is allowed later
        else:
            logger.warning(f"Unsupported edge detection method: {method}. Using 'sobel'.")
            edge_map = _sobel_filter(validated_data)
            if edge_map is not None:
                 edge_density = np.mean(edge_map > 0.5)
                 details = {'method': 'sobel', 'edge_density': float(edge_density)}
            else: # Handle Sobel failure even in fallback
                 return _create_result_dict(False, 0.0, error="Sobel filter failed")

    except Exception as e:
        logger.error(f"Error detecting edges: {e}", exc_info=True)
        return _create_result_dict(False, 0.0, error=f"Edge detection failed: {e}")

    # Detected if edge density exceeds threshold
    detected = edge_density > threshold
    confidence = edge_density # Use density as confidence score
    logger.debug(f"Edge detection (method={details['method']}): Confidence={confidence:.3f}, Detected={detected}")
    # Optionally return the edge map itself? For now, just metrics.
    return _create_result_dict(detected, confidence, **details) # E15


# --- Analysis Function ---

def analyze_patterns(data: np.ndarray,
                     config: Optional[Dict[str, Any]] = None # Feature-9: Add config
                     ) -> Dict[str, Any]:
    """
    Analyze data for multiple pattern types using configurable thresholds.

    Args:
        data: Input data (numpy array).
        config: Optional dictionary to override default thresholds and settings.
                Example: {'checkerboard_threshold': 0.75, 'gradient_method': 'sobel'}

    Returns:
        Dictionary with pattern analysis results for various types.
    """
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            logger.error(f"Input could not be converted to numpy array: {e}")
            return {"error": "Input could not be converted to numpy array"}
    if data.size == 0:
         return {"error": "Input data is empty"}

    cfg = config or {}
    logger.info(f"Starting pattern analysis with config: {cfg}")

    results = {}
    all_confidences = {}

    try:
        # --- Run individual pattern detectors ---
        # Checkerboard
        cb_thresh = cfg.get('checkerboard_threshold', DEFAULT_CHECKERBOARD_THRESHOLD)
        cb_scale = cfg.get('checkerboard_scale', 1)
        cb_result = detect_checkerboard(data, threshold=cb_thresh, scale=cb_scale)
        results["checkerboard"] = cb_result
        all_confidences["checkerboard"] = cb_result.get("confidence", 0.0)

        # Gradient
        grad_thresh = cfg.get('gradient_threshold', DEFAULT_GRADIENT_THRESHOLD)
        grad_method = cfg.get('gradient_method', 'sobel')
        grad_result = detect_gradients(data, threshold=grad_thresh, method=grad_method)
        results["gradient"] = grad_result
        all_confidences["gradient"] = grad_result.get("confidence", 0.0)

        # Adversarial / Noise-like
        adv_thresh = cfg.get('adversarial_threshold', DEFAULT_ADVERSARIAL_THRESHOLD)
        adv_method = cfg.get('adversarial_method', 'diff_std' if cfg.get('reference') else 'fft_high_freq')
        adv_ref = cfg.get('reference') # Allow passing reference via config
        adv_result = detect_adversarial_patterns(data, reference=adv_ref, threshold=adv_thresh, method=adv_method)
        results["adversarial"] = adv_result
        all_confidences["adversarial"] = adv_result.get("confidence", 0.0)

        # Repeating
        rep_thresh = cfg.get('repeating_threshold', DEFAULT_REPEATING_THRESHOLD)
        rep_method = cfg.get('repeating_method', 'autocorr')
        rep_result = detect_repeating_patterns(data, threshold=rep_thresh, method=rep_method)
        results["repeating"] = rep_result
        all_confidences["repeating"] = rep_result.get("confidence", 0.0)

        # Noise Level (Feature-2)
        noise_thresh = cfg.get('noise_threshold', DEFAULT_NOISE_THRESHOLD)
        noise_method = cfg.get('noise_method', 'std_dev')
        noise_result = detect_noise_level(data, threshold=noise_thresh, method=noise_method)
        results["noise"] = noise_result
        all_confidences["noise"] = noise_result.get("confidence", 0.0) # Use noise score as confidence

        # Edges (Feature-4)
        edge_thresh = cfg.get('edge_threshold', DEFAULT_EDGE_THRESHOLD)
        edge_method = cfg.get('edge_method', 'sobel')
        edge_result = detect_edges(data, threshold=edge_thresh, method=edge_method)
        results["edges"] = edge_result
        all_confidences["edges"] = edge_result.get("confidence", 0.0)

        # Cycles (Feature-5 - if applicable and implemented)
        # Ensure data is suitable for cycle detection (1D numeric)
        if data.ndim == 1 and np.issubdtype(data.dtype, np.number):
             cycle_thresh = cfg.get('cycle_threshold', DEFAULT_CYCLE_THRESHOLD)
             cycle_min_p = cfg.get('cycle_min_period', 3)
             cycle_max_p = cfg.get('cycle_max_period', data.size // 2 if data.size > 5 else 3)
             cycle_result = detect_cycles(data, min_period=cycle_min_p, max_period=cycle_max_p, threshold=cycle_thresh) # Use updated signature
             results["cycles"] = cycle_result
             all_confidences["cycles"] = cycle_result.get("confidence", 0.0)
        else:
             results["cycles"] = _create_result_dict(False, 0.0, reason="Requires 1D numeric data")
             all_confidences["cycles"] = 0.0

        # --- Determine dominant pattern (if any detected) ---
        detected_patterns = {k: v for k, v in all_confidences.items() if results[k]["detected"]}

        if detected_patterns:
            # Find the pattern with the highest confidence among those detected
            dominant_pattern = max(detected_patterns, key=detected_patterns.get)
            dominant_confidence = detected_patterns[dominant_pattern]
            # Only declare dominant if confidence is reasonably high? Add optional threshold?
            dominant_threshold = cfg.get('dominant_pattern_min_confidence', 0.5)
            if dominant_confidence >= dominant_threshold:
                 results["dominant_pattern"] = dominant_pattern
                 results["dominant_confidence"] = float(dominant_confidence)
                 logger.info(f"Dominant pattern identified: {dominant_pattern} (Confidence: {dominant_confidence:.3f})")
            else:
                 results["dominant_pattern"] = "none_dominant"
                 results["dominant_confidence"] = 0.0
                 logger.info(f"Multiple patterns detected, but none exceed dominance threshold ({dominant_threshold}).")

        else:
            results["dominant_pattern"] = "none"
            results["dominant_confidence"] = 0.0
            logger.info("No dominant pattern detected above thresholds.")

        results["analysis_config"] = cfg # Feature-9: Include config used
        results["input_shape"] = data.shape # Feature-10: Add input shape
        results["input_dtype"] = str(data.dtype) # Feature-10: Add input dtype

    except Exception as e:
        logger.critical(f"Critical error during pattern analysis: {e}", exc_info=True)
        return {"error": f"Analysis failed: {e}"}

    logger.info("Pattern analysis completed.")
    return results


# --- Stubbed / Partially Implemented Functions ---
# Need to fully implement these based on requirements

def detect_repetition(sequence: Sequence[Any], # Use Sequence for broader type hint
                      threshold: float = 0.8, # Adjust threshold
                      min_len: int = 3 # Feature-5: Min pattern length
                      ) -> Dict[str, Any]:
    """
    Detect repeating sub-sequences within a list or sequence.
    Uses a basic sliding window comparison.

    Args:
        sequence: Input sequence (list, tuple, etc.).
        threshold: Similarity threshold (0-1) for considering patterns a match (using detect_similarity).
        min_len: Minimum length of the repeating pattern to search for.

    Returns:
        Dict with repetition information (detected, pattern, locations, confidence).
    """
    n = len(sequence)
    if n < min_len * 2:
        return _create_result_dict(False, 0.0, reason="Sequence too short for repetition detection")

    best_pattern = None
    best_locations = []
    max_confidence = 0.0

    # Iterate through possible pattern lengths
    for length in range(min_len, n // 2 + 1):
        # Iterate through possible starting positions for the pattern
        for i in range(n - length):
            pattern = sequence[i : i + length]
            locations = [i]
            current_max_similarity = 0.0
            # Search for repetitions of this pattern later in the sequence
            for j in range(i + length, n - length + 1):
                sub_sequence = sequence[j : j + length]
                similarity = detect_similarity(pattern, sub_sequence) # Needs implementation
                if similarity >= threshold:
                    locations.append(j)
                    current_max_similarity = max(current_max_similarity, similarity)

            if len(locations) > 1: # Found at least one repetition
                 # Confidence could be based on number of repetitions and average similarity
                 confidence = (len(locations) / (n / length)) * current_max_similarity # Example confidence
                 if confidence > max_confidence:
                     max_confidence = confidence
                     best_pattern = pattern
                     best_locations = locations

    detected = max_confidence > 0 # Detected if any repetition found above threshold
    # Re-evaluate detected based on confidence vs some threshold? Let's use 0.5 as default.
    detected_final = max_confidence >= 0.5

    logger.debug(f"Repetition detection: Confidence={max_confidence:.3f}, Detected={detected_final}, Pattern={best_pattern}")
    return _create_result_dict(detected_final, max_confidence, pattern=best_pattern, locations=best_locations)


def detect_similarity(item1: Any, item2: Any) -> float:
    """
    Calculate similarity between two items (basic implementation).
    Needs significant enhancement based on item types.

    Args:
        item1: First item.
        item2: Second item.

    Returns:
        Similarity score (0-1).
    """
    # Very basic placeholder logic
    if type(item1) != type(item2):
        return 0.0 # Different types are dissimilar

    if isinstance(item1, (str, int, float, bool)):
        return 1.0 if item1 == item2 else 0.0
    elif isinstance(item1, (list, tuple)):
        if len(item1) != len(item2) or len(item1) == 0:
            return 0.0
        # Basic element-wise comparison average
        matches = sum(detect_similarity(e1, e2) for e1, e2 in zip(item1, item2))
        return matches / len(item1)
    elif isinstance(item1, np.ndarray):
         # Use basic SSIM or correlation for arrays? Or RMSE based?
         if item1.shape != item2.shape or item1.size == 0:
              return 0.0
         # Use normalized RMSE -> similarity = 1 - RMSE (if RMSE is normalized 0-1)
         norm1 = _normalize_data(item1)
         norm2 = _normalize_data(item2)
         rmse = _calculate_rmse(norm1, norm2)
         return max(0.0, 1.0 - rmse) if rmse is not None else 0.0

    # Add handling for dicts, sets, etc.
    logger.warning(f"Similarity detection not fully implemented for type: {type(item1)}")
    # Fallback to equality check
    try:
         return 1.0 if item1 == item2 else 0.0
    except Exception: # Handle non-comparable types
         return 0.0


def detect_gradient(data: Union[np.ndarray, List[float]], # Keep original name, refine implementation
                    axis: Optional[int] = None # Make axis optional, infer for 1D
                    ) -> Dict[str, Any]:
    """
    Detect gradients/trends in numerical data (1D focus).

    Args:
        data: Numerical data array or list.
        axis: Axis (ignored for 1D data).

    Returns:
        Dict with gradient trend information.
    """
    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data, dtype=float)
        except (ValueError, TypeError):
            return {
                "error": "Input data could not be converted to numerical array",
                "gradient_detected": False
            }

    # Infer axis or check dimension
    if data.ndim == 1:
        target_axis = 0
    elif data.ndim == 2 and axis is not None and axis < 2 :
         target_axis = axis
         logger.debug(f"Detecting gradient along specified axis {axis} for 2D data.")
    elif data.ndim == 2:
         logger.warning("Axis not specified for 2D data in detect_gradient, defaulting to axis 0 (rows).")
         target_axis = 0 # Default for 2D if not specified
    else:
         return {"error": f"Unsupported data dimension ({data.ndim}) for simple gradient detection.", "gradient_detected": False}


    # Check if data contains enough points for gradient analysis
    if data.shape[target_axis] < 2:
        return {
            "gradient_detected": False,
            "reason": "Insufficient data points along the specified axis for gradient detection"
        }

    try:
        # Calculate gradient using numpy's gradient function
        # Use edge_order=1 for potentially more stable results at edges
        gradients = np.gradient(data, axis=target_axis, edge_order=1)

        # For 2D data, np.gradient returns a list [grad_axis0, grad_axis1].
        # We analyze the gradient along the specified axis.
        if isinstance(gradients, list):
            target_gradients = gradients[target_axis]
        else: # Should be 1D input case
             target_gradients = gradients

        # Calculate statistics on the relevant gradient component
        avg_gradient = np.mean(target_gradients)
        abs_gradients = np.abs(target_gradients)
        max_abs_gradient = np.max(abs_gradients)
        min_abs_gradient = np.min(abs_gradients) # Minimum slope magnitude
        std_gradient = np.std(target_gradients)

        # Determine if there's a consistent trend
        # Consistency: low standard deviation relative to the average magnitude
        # Avoid division by zero if average gradient is tiny
        consistency_denominator = max(abs(avg_gradient), 1e-6)
        gradient_consistency = 1.0 - (std_gradient / consistency_denominator)
        gradient_consistency = np.clip(gradient_consistency, 0.0, 1.0) # Ensure valid range

        # Determine if a significant gradient exists (consistent and non-negligible average)
        # Thresholds might need tuning
        significant_gradient = abs(avg_gradient) > 0.05 and gradient_consistency > 0.4

        trend = "increasing" if avg_gradient > 1e-4 else "decreasing" if avg_gradient < -1e-4 else "flat"

        result = {
            "gradient_detected": significant_gradient,
            "average_gradient": float(avg_gradient),
            "max_abs_gradient": float(max_abs_gradient),
            "min_abs_gradient": float(min_abs_gradient),
            "std_dev_gradient": float(std_gradient),
            "gradient_consistency": float(gradient_consistency),
            "trend": trend,
            "axis": target_axis if data.ndim > 1 else 0
        }
        # Add detection dict structure
        return _create_result_dict(significant_gradient, gradient_consistency, **{k: v for k, v in result.items() if k != 'gradient_detected'})


    except Exception as e:
        logger.error(f"Error calculating gradient trend: {e}", exc_info=True)
        return {
            "error": f"Error calculating gradient: {str(e)}",
            "gradient_detected": False
        }


def detect_cycles(sequence: Union[List[float], np.ndarray], # Keep original name
                  min_period: int = 3,
                  max_period: Optional[int] = None, # Make max optional
                  threshold: float = DEFAULT_CYCLE_THRESHOLD # Add threshold param
                  ) -> Dict[str, Any]:
    """
    Detect cyclical patterns in 1D numeric data using autocorrelation peaks.

    Args:
        sequence: Numeric sequence (list or 1D numpy array).
        min_period: Minimum cycle period (lag) to detect.
        max_period: Maximum cycle period (lag) to detect (defaults to len(sequence)//2).
        threshold: Minimum autocorrelation peak height to consider a cycle detected.

    Returns:
        Dict with cycle detection results (detected, confidence, period).
    """
    if not isinstance(sequence, np.ndarray):
        try:
            sequence = np.array(sequence, dtype=float)
        except (ValueError, TypeError):
            return _create_result_dict(False, 0.0, error="Input sequence could not be converted to numerical array")

    if sequence.ndim != 1:
        return _create_result_dict(False, 0.0, error="Cycle detection requires a 1D sequence")

    n = sequence.size
    if max_period is None:
        max_period = n // 2
    max_period = min(max_period, n - 1) # Ensure max_period is valid

    if n < min_period * 2 or min_period < 2 or min_period > max_period:
        return _create_result_dict(False, 0.0, reason="Insufficient data or invalid period range")

    autocorr = _calculate_autocorrelation(sequence) # E9
    if autocorr is None or autocorr.size <= min_period:
        return _create_result_dict(False, 0.0, error="Autocorrelation calculation failed or too short")

    # Look for peaks within the specified period range
    lags_to_check = autocorr[min_period : max_period + 1]
    if lags_to_check.size == 0:
         return _create_result_dict(False, 0.0, reason="No valid lags in specified period range")

    # E8: Find peaks in the relevant autocorrelation part
    # Adjust prominence based on expected correlation values (0-1)
    peaks_indices, properties = signal.find_peaks(lags_to_check, height=threshold/2, prominence=0.1, distance=2)

    if peaks_indices.size > 0:
        # Find the peak with the highest prominence or height within the range
        peak_heights = properties['peak_heights']
        # best_peak_local_idx = np.argmax(properties['prominences']) # Or use height? Let's use height.
        best_peak_local_idx = np.argmax(peak_heights)

        confidence = peak_heights[best_peak_local_idx]
        # Adjust index back to original autocorrelation lags
        period = peaks_indices[best_peak_local_idx] + min_period
        detected = confidence >= threshold
        logger.debug(f"Cycle detection: Confidence={confidence:.3f}, Detected={detected}, Period={period}")
        return _create_result_dict(detected, confidence, detected_period=int(period))
    else:
        # No significant peaks found
        logger.debug("Cycle detection: No significant peaks found in autocorrelation within range.")
        return _create_result_dict(False, 0.0, reason="No significant autocorrelation peaks found")

def detect_low_correlation_risk(image1: np.ndarray, image2: np.ndarray, threshold=0.15) -> bool:
    """
    Detect if two images might have low correlation despite visual similarity.

    Args:
        image1: First image
        image2: Second image
        threshold: Correlation threshold below which risk is detected

    Returns:
        bool: True if there's risk of low correlation measures
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3 and image1.shape[2] in [3, 4]:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = image1.copy()

    if len(image2.shape) == 3 and image2.shape[2] in [3, 4]:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = image2.copy()

    # Ensure same shape for correlation
    if gray1.shape != gray2.shape:
        # Resize to match the smaller image
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        if h1 * w1 > h2 * w2:
            gray1 = cv2.resize(gray1, (w2, h2))
        else:
            gray2 = cv2.resize(gray2, (w1, h1))

    # Flatten arrays for correlation
    flat1 = gray1.flatten()
    flat2 = gray2.flatten()

    # Calculate correlation coefficient
    corr_coeff = np.corrcoef(flat1, flat2)[0, 1]

    # Check if correlation is below threshold (indicating risk)
    return np.abs(corr_coeff) < threshold

# If there's an __all__ list in the module, add the new function to it
try:
    __all__.append('detect_low_correlation_risk')
except NameError:
    __all__ = ['detect_low_correlation_risk']

def detect_image_pattern(image: np.ndarray) -> Dict[str, Any]:
    """
    Detect patterns in an image and classify it into pattern categories.

    Args:
        image: Input image

    Returns:
        Dict with pattern detection results
    """
    # Detect basic patterns
    has_checkerboard = detect_checkerboard(image)
    has_gradients, gradient_smoothness = detect_gradients(image)
    edge_info = detect_edges(image)
    noise_info = detect_noise_level(image)

    # Determine dominant pattern
    dominant_pattern = "unknown"
    pattern_confidences = {}

    if has_checkerboard:
        pattern_confidences["checkerboard"] = 0.8
        dominant_pattern = "checkerboard"

    if has_gradients:
        gradient_conf = gradient_smoothness
        pattern_confidences["gradient"] = gradient_conf
        if gradient_conf > pattern_confidences.get(dominant_pattern, 0.0):
            dominant_pattern = "gradient"

    if edge_info["detected"]:
        edge_conf = edge_info["confidence"]
        pattern_confidences["edge_rich"] = edge_conf
        if edge_conf > pattern_confidences.get(dominant_pattern, 0.0):
            dominant_pattern = "edge_rich"

    if noise_info["noise_level"] > 0.5:
        noise_conf = noise_info["noise_level"]
        pattern_confidences["noisy"] = noise_conf
        if noise_conf > pattern_confidences.get(dominant_pattern, 0.0):
            dominant_pattern = "noisy"

    # Check for flat/uniform regions
    if noise_info["noise_level"] < 0.1 and not has_gradients and not edge_info["detected"]:
        pattern_confidences["uniform"] = 0.9
        dominant_pattern = "uniform"

    # Return comprehensive detection results
    return {
        "detected_patterns": list(pattern_confidences.keys()),
        "pattern_confidences": pattern_confidences,
        "dominant_pattern": dominant_pattern,
        "checkerboard": has_checkerboard,
        "gradient": has_gradients,
        "gradient_smoothness": gradient_smoothness if has_gradients else 0.0,
        "edge_density": edge_info.get("edge_density", 0.0),
        "noise_level": noise_info["noise_level"]
    }

# If there's an __all__ list, add the new function to it
if '__all__' in globals():
    __all__.append('detect_image_pattern')

def detect_periodicity(data: np.ndarray, max_period: Optional[int] = None) -> Dict[str, Any]:
    """
    Detect periodic patterns in time series data.
    
    Args:
        data: Time series data
        max_period: Maximum period to check
        
    Returns:
        Dictionary with periodicity information
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.size < 4:
        return {"has_periodicity": False, "reason": "Insufficient data"}
    
    # Default max period is half the data length
    if max_period is None:
        max_period = data.size // 2
    else:
        max_period = min(max_period, data.size // 2)
    
    # Calculate autocorrelation
    autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    autocorr = autocorr[data.size-1:data.size-1+max_period]
    autocorr /= np.max(np.abs(autocorr))
    
    # Find peaks in autocorrelation
    try:
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr, height=0.5)
        
        if len(peaks) == 0:
            return {"has_periodicity": False, "reason": "No significant peaks"}
        
        # Get most significant peak
        peak_idx = peaks[np.argmax(properties["peak_heights"])]
        peak_height = properties["peak_heights"][np.argmax(properties["peak_heights"])]
        
        return {
            "has_periodicity": True,
            "period": int(peak_idx),
            "confidence": float(peak_height),
            "peaks": peaks.tolist(),
            "autocorrelation": autocorr.tolist()
        }
    except ImportError:
        # Fallback if scipy not available - simple peak finding
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > 0.5 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
        
        if not peaks:
            return {"has_periodicity": False, "reason": "No significant peaks (basic detection)"}
            
        # Find max peak
        max_peak = max(peaks, key=lambda i: autocorr[i])
        
        return {
            "has_periodicity": True,
            "period": int(max_peak),
            "confidence": float(autocorr[max_peak]),
            "peaks": peaks,
            "note": "Using basic peak detection (scipy not available)"
        }

def detect_trends(data: np.ndarray, window_size: int = 10) -> Dict[str, Any]:
    """
    Detect trends in time series data.
    
    Args:
        data: Time series data
        window_size: Window size for trend detection
        
    Returns:
        Dictionary with trend information
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.size < 3:
        return {"has_trend": False, "reason": "Insufficient data"}
    
    # Simple linear regression
    x = np.arange(data.size)
    slope, intercept = np.polyfit(x, data, 1)
    
    # Calculate R-squared
    predicted = intercept + slope * x
    ss_total = np.sum((data - np.mean(data)) ** 2)
    ss_residual = np.sum((data - predicted) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Determine trend strength
    trend_threshold = 0.5
    has_trend = r_squared > trend_threshold
    
    # Calculate short-term trend changes
    if data.size >= window_size:
        windows = [data[i:i+window_size] for i in range(0, data.size - window_size + 1, window_size // 2)]
        window_slopes = []
        
        for window in windows:
            window_x = np.arange(window.size)
            window_slope, _ = np.polyfit(window_x, window, 1)
            window_slopes.append(window_slope)
        
        trend_changes = np.sum(np.diff(np.sign(window_slopes)) != 0)
    else:
        trend_changes = 0
        window_slopes = []
    
    return {
        "has_trend": has_trend,
        "overall_slope": float(slope),
        "r_squared": float(r_squared),
        "trend_type": "upward" if slope > 0 else "downward",
        "trend_strength": float(r_squared),
        "trend_changes": int(trend_changes),
        "window_slopes": [float(s) for s in window_slopes]
    }

def detect_outliers(data: np.ndarray, threshold: float = 2.0) -> Dict[str, Any]:
    """
    Detect outliers in data using z-score method.
    
    Args:
        data: Data array
        threshold: Z-score threshold
        
    Returns:
        Dictionary with outlier information
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.size < 3:
        return {"outliers": [], "reason": "Insufficient data"}
    
    # Calculate z-scores
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return {"outliers": [], "reason": "Zero standard deviation"}
    
    z_scores = np.abs((data - mean) / std)
    
    # Find outliers
    outlier_indices = np.where(z_scores > threshold)[0]
    outlier_values = data[outlier_indices]
    outlier_z_scores = z_scores[outlier_indices]
    
    outliers = []
    for i, value, z in zip(outlier_indices, outlier_values, outlier_z_scores):
        outliers.append({
            "index": int(i),
            "value": float(value),
            "z_score": float(z)
        })
    
    return {
        "outliers": outliers,
        "count": len(outliers),
        "mean": float(mean),
        "std": float(std),
        "threshold": threshold
    }

def detect_image_pattern(image: np.ndarray) -> Dict[str, Any]:
    """
    Detect patterns in image data.
    
    Args:
        image: Image as numpy array (height, width, channels)
        
    Returns:
        Dictionary with pattern information
    """
    if not isinstance(image, np.ndarray):
        return {"error": "Input must be numpy array"}
    
    if len(image.shape) < 2:
        return {"error": "Input must be 2D or 3D array (image)"}
    
    # Basic image statistics
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    # Detection results
    results = {
        "dimensions": (width, height),
        "channels": channels,
        "detected_patterns": []
    }
    
    # Simple edge detection (gradient magnitude)
    try:
        # Compute gradients
        if channels == 1 or len(image.shape) == 2:
            # Grayscale image
            img = image if len(image.shape) == 2 else image.reshape(height, width)
            dx = np.gradient(img.astype(float), axis=1)
            dy = np.gradient(img.astype(float), axis=0)
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
        else:
            # Color image - convert to grayscale first
            gray = np.mean(image, axis=2) if channels >= 3 else image.squeeze()
            dx = np.gradient(gray.astype(float), axis=1)
            dy = np.gradient(gray.astype(float), axis=0)
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Calculate edge statistics
        edge_threshold = np.percentile(gradient_magnitude, 90)
        edge_mask = gradient_magnitude > edge_threshold
        edge_count = np.sum(edge_mask)
        edge_density = edge_count / (width * height)
        
        if edge_density > 0.1:
            results["detected_patterns"].append({
                "type": "edges",
                "confidence": min(1.0, edge_density * 2),
                "density": float(edge_density)
            })
    except Exception as e:
        logger.warning(f"Error in edge detection: {e}")
        
    # Check for symmetry
    try:
        # Simple horizontal symmetry check
        if len(image.shape) == 2:
            # Grayscale
            left_half = image[:, :width//2]
            right_half = np.fliplr(image[:, width//2:width//2*2])
            if right_half.shape == left_half.shape:  # Ensure halves are same size
                h_diff = np.mean(np.abs(left_half - right_half)) / 255.0
                h_symmetry = max(0, 1 - h_diff)
                
                if h_symmetry > 0.7:
                    results["detected_patterns"].append({
                        "type": "horizontal_symmetry",
                        "confidence": float(h_symmetry)
                    })
        else:
            # Average color channels for simplicity
            gray = np.mean(image, axis=2)
            left_half = gray[:, :width//2]
            right_half = np.fliplr(gray[:, width//2:width//2*2])
            if right_half.shape == left_half.shape:
                h_diff = np.mean(np.abs(left_half - right_half)) / 255.0
                h_symmetry = max(0, 1 - h_diff)
                
                if h_symmetry > 0.7:
                    results["detected_patterns"].append({
                        "type": "horizontal_symmetry",
                        "confidence": float(h_symmetry)
                    })
    except Exception as e:
        logger.warning(f"Error in symmetry detection: {e}")
    
    return results

def detect_clusters(data: np.ndarray, n_clusters: Optional[int] = None,
                   method: str = "kmeans") -> Dict[str, Any]:
    """
    Detect clusters in data.
    
    Args:
        data: Data array (samples, features)
        n_clusters: Number of clusters (None for automatic)
        method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
        
    Returns:
        Dictionary with cluster information
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if data.shape[0] < 3:
        return {"success": False, "reason": "Insufficient data"}
    
    # Automatic cluster count
    if n_clusters is None:
        # Simple method: square root of n samples
        n_clusters = max(2, min(10, int(np.sqrt(data.shape[0] / 2))))
    
    try:
        if method == "kmeans":
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(data)
                centers = model.cluster_centers_
                
                # Calculate silhouette score if more than one cluster
                if n_clusters > 1:
                    silhouette = silhouette_score(data, labels)
                else:
                    silhouette = 0
                
                # Count samples per cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                return {
                    "success": True,
                    "method": "kmeans",
                    "n_clusters": n_clusters,
                    "labels": labels.tolist(),
                    "centers": centers.tolist(),
                    "silhouette_score": float(silhouette),
                    "samples_per_cluster": counts.tolist()
                }
            except ImportError:
                return {"success": False, "reason": "sklearn not available for KMeans clustering"}
                
        elif method == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
                
                # Auto-epsilon based on nearest neighbors
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(10, data.shape[0] // 2)).fit(data)
                distances, _ = nn.kneighbors(data)
                epsilon = np.percentile(distances[:, 1], 90) * 1.5
                
                model = DBSCAN(eps=epsilon, min_samples=3)
                labels = model.fit_predict(data)
                
                # Calculate statistics
                unique_labels = np.unique(labels)
                n_clusters_found = len(unique_labels[unique_labels >= 0])
                n_noise = np.sum(labels == -1)
                
                return {
                    "success": True,
                    "method": "dbscan",
                    "n_clusters": n_clusters_found,
                    "labels": labels.tolist(),
                    "epsilon": float(epsilon),
                    "noise_points": int(n_noise)
                }
            except ImportError:
                return {"success": False, "reason": "sklearn not available for DBSCAN clustering"}
        else:
            return {"success": False, "reason": f"Unknown clustering method: {method}"}
    except Exception as e:
        logger.error(f"Error in cluster detection: {e}")
        return {"success": False, "reason": str(e)}

def detect_structural_patterns(data: Any) -> Dict[str, Any]:
    """
    Detect patterns in data structures.
    
    Args:
        data: Any data structure (list, dict, etc.)
        
    Returns:
        Dictionary with patterns detected
    """
    patterns = {
        "structure_type": type(data).__name__,
        "detected_patterns": []
    }
    
    if isinstance(data, (list, tuple)):
        patterns["length"] = len(data)
        
        # Check if all elements are of the same type
        if data:
            all_same_type = all(type(item) == type(data[0]) for item in data)
            if all_same_type:
                patterns["detected_patterns"].append({
                    "type": "homogeneous",
                    "element_type": type(data[0]).__name__,
                    "confidence": 1.0
                })
        
        # Check for increasing/decreasing sequences
        if data and all(isinstance(item, (int, float)) for item in data):
            diffs = [data[i+1] - data[i] for i in range(len(data)-1)]
            all_positive = all(d > 0 for d in diffs)
            all_negative = all(d < 0 for d in diffs)
            
            if all_positive:
                patterns["detected_patterns"].append({
                    "type": "increasing_sequence",
                    "confidence": 1.0
                })
            elif all_negative:
                patterns["detected_patterns"].append({
                    "type": "decreasing_sequence",
                    "confidence": 1.0
                })
    
    elif isinstance(data, dict):
        patterns["key_count"] = len(data)
        
        # Check for nested structure patterns
        nested_dicts = sum(1 for v in data.values() if isinstance(v, dict))
        nested_lists = sum(1 for v in data.values() if isinstance(v, (list, tuple)))
        
        if nested_dicts > 0:
            patterns["detected_patterns"].append({
                "type": "nested_dictionaries",
                "count": nested_dicts,
                "confidence": nested_dicts / len(data) if data else 0
            })
            
        if nested_lists > 0:
            patterns["detected_patterns"].append({
                "type": "nested_lists",
                "count": nested_lists,
                "confidence": nested_lists / len(data) if data else 0
            })
    
    return patterns
