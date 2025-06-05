#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ART Logger for the VoxSigil System.

This module provides a centralized logging facility for the ART module,
covering events, training progress, pattern detection, and errors.
"""

import logging
import os
import sys
from typing import Optional

# --- Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Determine a base directory for logs, e.g., relative to this file or a common project logs dir
# For now, let's assume a logs directory at the project root or within the art module itself.
# This should be configurable or determined by the main application (Vanta).
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# Fallback if not set by Vanta
DEFAULT_ART_LOG_FILE = "art_module.log"

# Use the same base logger name as Vanta for easier integration
# This will allow handlers configured for "VoxSigilSupervisor" to also capture these logs.
VANTA_SUPERVISOR_LOGGER_NAME = "VoxSigilSupervisor"


class ARTLogger:
    _instance = None

    def __new__(
        cls, logger_name="ARTModule", log_file=None, log_dir=None, level=LOG_LEVEL
    ):
        if cls._instance is None:
            cls._instance = super(ARTLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, logger_name="ARTModule", log_file=None, log_dir=None, level=LOG_LEVEL
    ):
        if self._initialized:
            return

        self.logger_name = logger_name
        self.log_dir = log_dir if log_dir else DEFAULT_LOG_DIR
        self.log_file = (
            log_file if log_file else os.path.join(self.log_dir, DEFAULT_ART_LOG_FILE)
        )
        self.level = level

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except OSError as e:
                # Fallback to console logging if directory creation fails
                print(
                    f"Error creating log directory {self.log_dir}: {e}. Logging to console."
                )
                self._configure_console_logger()
                self._initialized = True
                return

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.level)

        # Prevent duplicate handlers if already configured (e.g., in a notebook environment)
        if not self.logger.handlers:
            # File Handler
            try:
                file_handler = logging.FileHandler(
                    self.log_file, mode="a"
                )  # Append mode
                file_handler.setLevel(self.level)
                formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except IOError as e:
                print(
                    f"Error setting up file handler for {self.log_file}: {e}. Logging to console."
                )
                self._configure_console_logger()  # Fallback to console
            else:
                # Console Handler (optional, for also printing to stdout/stderr)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self.level)  # Or a different level for console
                console_formatter = logging.Formatter(
                    LOG_FORMAT, datefmt=LOG_DATE_FORMAT
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)

        self._initialized = True

    def _configure_console_logger(self):
        """Configures a basic console logger as a fallback."""
        self.logger = logging.getLogger(self.logger_name + "_console")
        self.logger.setLevel(self.level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# --- Public API for easy access ---
def get_art_logger(
    name: Optional[str] = None,
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    base_logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Returns a logger instance for ART components, with sensible defaults for easy use.

    Args:
        name: The name of the ART component (e.g., 'ARTController'). If None, uses 'ARTModule'.
        level: The logging level (e.g., logging.INFO, logging.DEBUG). If None, uses LOG_LEVEL.
        log_file: Optional path to a file for logs. If None, uses default ART log file.
        base_logger_name: The root logger name. If None, uses VANTA_SUPERVISOR_LOGGER_NAME.

    Returns:
        A configured logger instance.
    """
    # Set defaults for easier use
    if name is None:
        name = "ARTModule"
    if level is None:
        level = LOG_LEVEL
    if base_logger_name is None:
        base_logger_name = VANTA_SUPERVISOR_LOGGER_NAME

    logger_name = f"{base_logger_name}.art.{name}" if base_logger_name else name
    logger = logging.getLogger(logger_name)

    # Only configure handlers if none exist
    if not logger.handlers:
        logger.setLevel(level)

        # Add console handler if no parent handlers exist
        has_propagating_handlers = False
        temp_logger = logger
        while temp_logger:
            if temp_logger.handlers:
                has_propagating_handlers = True
                break
            if not getattr(temp_logger, "propagate", True):
                break
            temp_logger = getattr(temp_logger, "parent", None)

        if not has_propagating_handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s"
            )
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # Add file handler if requested
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s"
            )
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

    logger.propagate = True
    return logger


# Example Usage:
if __name__ == "__main__":
    # This demonstrates how other ART modules would use the logger.
    # Vanta or the main application would ideally configure the log_dir.

    # Using the default logger instance
    logger = get_art_logger("ARTModule")
    logger.info("ARTLogger initialized. This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    # Example of a specific logger for a sub-component
    controller_logger = get_art_logger("ARTController", log_file="art_controller.log")
    controller_logger.debug(
        "Debugging ART Controller operation."
    )  # Won't show if LOG_LEVEL is INFO

    # To see debug messages, you'd initialize with level=logging.DEBUG
    debug_logger = get_art_logger("ARTDebug", level=logging.DEBUG)
    debug_logger.debug("This is a detailed debug message for ART.")

    print(
        f"ART logs are being written to: {ARTLogger().log_file if hasattr(ARTLogger(), 'log_file') else 'console'}"
    )

    # Test with default base_logger_name ("VoxSigilSupervisor")
    logger1 = get_art_logger("TestComponent1")
    logger1.info(
        "This is an info message from TestComponent1 (under VoxSigilSupervisor)."
    )
    logger1.debug(
        "This is a debug message from TestComponent1 (under VoxSigilSupervisor)."
    )

    # Test with a specific log file for this logger instance
    logger2 = get_art_logger("TestComponent2", log_file="test_component2.log")
    logger2.warning("This is a warning from TestComponent2, also to its own file.")

    print(
        f"Logger1 name: {logger1.name}, Effective level: {logger1.getEffectiveLevel()}, Handlers: {logger1.handlers}"
    )
    print(
        f"Logger2 name: {logger2.name}, Effective level: {logger2.getEffectiveLevel()}, Handlers: {logger2.handlers}"
    )
