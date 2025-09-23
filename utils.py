#!/usr/bin/env python3
"""
Utilities for RRF Analytics Project (AI assisted file... complicated otherwise)
"""
import logging
import os


def setup_logger(name, level=None):
    """
    Set up logger with appropriate level and formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level override (if None, uses RRF_LOG_LEVEL env var or INFO)
    
    Returns:
        Configured logger
    """
    if level is None:
        # Check environment variable for log level
        env_level = os.environ.get('RRF_LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, env_level, logging.INFO)
    
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Minimal format for analysis scripts
    formatter = logging.Formatter('%(message)s')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent duplicate logs from parent loggers
    logger.propagate = False
    
    return logger