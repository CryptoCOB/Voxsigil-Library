"""
File I/O Utilities

This module provides utilities for safe file reading and writing operations.
"""

import os
import json
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, BinaryIO, TextIO

# Configure logger
logger = logging.getLogger(__name__)

def ensure_directory(directory_path: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Directory path to ensure
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)
        
        # Create directory if it doesn't exist
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error ensuring directory {directory_path}: {e}")
        return False

def safe_read_file(file_path: Union[str, Path], 
                  encoding: str = 'utf-8', 
                  binary_mode: bool = False) -> Optional[Union[str, bytes]]:
    """
    Safely read a file with error handling.
    
    Args:
        file_path: Path to the file
        encoding: File encoding (for text mode)
        binary_mode: Whether to read in binary mode
        
    Returns:
        File contents or None if error occurs
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None
    
    try:
        mode = 'rb' if binary_mode else 'r'
        
        with open(file_path, mode, encoding=None if binary_mode else encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def safe_write_file(file_path: Union[str, Path], 
                   content: Union[str, bytes],
                   encoding: str = 'utf-8',
                   binary_mode: bool = False,
                   atomic: bool = True) -> bool:
    """
    Safely write to a file with error handling and optional atomic write.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding (for text mode)
        binary_mode: Whether to write in binary mode
        atomic: Whether to do an atomic write
        
    Returns:
        True if successful, False otherwise
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    try:
        # Ensure parent directory exists
        ensure_directory(file_path.parent)
        
        mode = 'wb' if binary_mode else 'w'
        
        if atomic:
            # Write to a temporary file first
            temp_file = None
            try:
                # Create temp file in the same directory for atomic rename
                fd, temp_path = tempfile.mkstemp(dir=file_path.parent)
                
                with os.fdopen(fd, mode, encoding=None if binary_mode else encoding) as f:
                    f.write(content)
                
                # Atomic rename
                os.replace(temp_path, file_path)
                return True
            except Exception as e:
                logger.error(f"Error in atomic write to {file_path}: {e}")
                # Clean up temp file if it exists
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                return False
        else:
            # Direct write
            with open(file_path, mode, encoding=None if binary_mode else encoding) as f:
                f.write(content)
            return True
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False

def load_json(file_path: Union[str, Path], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load JSON data from a file with error handling.
    
    Args:
        file_path: Path to the file
        default: Default value to return if error occurs
        
    Returns:
        JSON data as dictionary or default value
    """
    default = default if default is not None else {}
    
    try:
        content = safe_read_file(file_path)
        if content is None:
            return default
            
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {file_path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return default

def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save JSON data to a file with error handling.
    
    Args:
        data: JSON-serializable data
        file_path: Path to the file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        content = json.dumps(data, indent=indent)
        return safe_write_file(file_path, content)
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def create_backup(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Directory to store backup (uses file's directory if None)
        
    Returns:
        Path to the backup file or None if error occurs
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File not found for backup: {file_path}")
        return None
    
    try:
        # Determine backup directory
        if backup_dir is None:
            backup_dir = file_path.parent
        elif isinstance(backup_dir, str):
            backup_dir = Path(backup_dir)
            
        # Ensure backup directory exists
        if not ensure_directory(backup_dir):
            return None
            
        # Create backup filename with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        # Create backup
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup of {file_path} to {backup_path}")
        
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {e}")
        return None
