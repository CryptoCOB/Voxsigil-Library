"""
Utility module to handle numpy import issues.
This ensures safe importing of numpy with proper error handling for circular imports.
"""

import logging
import sys

logger = logging.getLogger("Voxsigil.Utils.NumpyResolver")

# Initialize placeholder values
np = None
numpy = None
HAVE_NUMPY = False


def safe_import_numpy():
    """
    Safely import numpy and handle potential circular import issues.
    Returns a tuple of (numpy_module, np_alias, have_numpy_bool)
    """
    global np, numpy, HAVE_NUMPY

    if np is not None:
        # Already imported
        return numpy, np, HAVE_NUMPY

    try:
        # Check if numpy is already in sys.modules and potentially problematic
        if "numpy" in sys.modules:
            existing_numpy = sys.modules["numpy"]
            # Test if the existing numpy module has the attributes we need
            if not hasattr(existing_numpy, "bool") and not hasattr(
                existing_numpy, "bool_"
            ):
                logger.warning(
                    "Numpy module in sys.modules appears to be incomplete (circular import)"
                )
                # Remove the problematic module
                del sys.modules["numpy"]
                # Also remove related modules that might be causing issues
                for module_name in list(sys.modules.keys()):
                    if module_name.startswith("numpy"):
                        del sys.modules[module_name]

        # Try fresh import
        import numpy as _np
        import numpy as _numpy

        # Test for key attributes to ensure numpy is fully loaded
        if not hasattr(_numpy, "array"):
            logger.warning("Numpy imported but missing 'array' attribute")
            return None, None, False

        # Try to create a simple array to test functionality
        try:
            test_array = _numpy.array([1, 2, 3])
            if test_array is None:
                logger.warning("Numpy array creation test failed")
                return None, None, False
        except Exception as e:
            logger.warning(f"Numpy array creation test failed: {e}")
            return None, None, False

        # Everything looks good
        numpy = _numpy
        np = _np
        HAVE_NUMPY = True
        logger.debug(
            f"Successfully imported numpy {getattr(numpy, '__version__', 'unknown')}"
        )
        return numpy, np, HAVE_NUMPY

    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.warning(f"Failed to import numpy: {e}")
        return None, None, False
    except Exception as e:
        # Catch any other unexpected errors
        logger.warning(f"Unexpected error importing numpy: {e}")
        return None, None, False


def get_numpy():
    """
    Get the safely imported numpy module.
    Returns a tuple of (numpy_module, np_alias, have_numpy_bool)
    """
    return safe_import_numpy()


# Convenience functions for common numpy operations with fallbacks
def safe_array(data, dtype=None):
    """Safely create a numpy array with fallback to list."""
    _, np_module, have_numpy = safe_import_numpy()
    if have_numpy and np_module is not None:
        try:
            return np_module.array(data, dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to create numpy array: {e}")
    
    # Fallback to regular list
    return list(data) if data is not None else []


def safe_zeros(shape, dtype=None):
    """Safely create a numpy zeros array with fallback."""
    _, np_module, have_numpy = safe_import_numpy()
    if have_numpy and np_module is not None:
        try:
            return np_module.zeros(shape, dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to create numpy zeros: {e}")
    
    # Fallback for simple cases
    if isinstance(shape, (int, float)):
        return [0] * int(shape)
    elif isinstance(shape, (list, tuple)) and len(shape) == 1:
        return [0] * int(shape[0])
    else:
        logger.warning(f"Cannot create fallback for complex shape: {shape}")
        return []


def safe_concatenate(arrays, axis=0):
    """Safely concatenate arrays with fallback."""
    _, np_module, have_numpy = safe_import_numpy()
    if have_numpy and np_module is not None:
        try:
            return np_module.concatenate(arrays, axis=axis)
        except Exception as e:
            logger.warning(f"Failed to concatenate with numpy: {e}")
    
    # Simple fallback for 1D case
    if axis == 0:
        result = []
        for arr in arrays:
            if hasattr(arr, 'tolist'):
                result.extend(arr.tolist())
            elif isinstance(arr, (list, tuple)):
                result.extend(arr)
            else:
                result.append(arr)
        return result
    else:
        logger.warning("Cannot concatenate with fallback for non-zero axis")
        return arrays[0] if arrays else []


# Initialize on import
numpy, np, HAVE_NUMPY = safe_import_numpy()

# Export the safely imported modules
__all__ = [
    'safe_import_numpy',
    'get_numpy', 
    'safe_array',
    'safe_zeros',
    'safe_concatenate',
    'numpy',
    'np',
    'HAVE_NUMPY',
    'logger'
]
