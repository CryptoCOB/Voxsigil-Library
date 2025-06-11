# Configuration Module
# Contains configuration management and production settings

__version__ = "1.0.0"

from .production_config import *

__all__ = ["ProductionConfig", "get_config", "load_config"]
