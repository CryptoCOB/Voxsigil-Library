"""
Utility module to handle numpy import issues.
This ensures safe importing of numpy with proper error handling for circular imports.
"""

import logging
import sys

logger = logging.getLogger("Vanta.NumpyResolver")

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
