#!/usr/bin/env python3
# utilities for rrf analytics project

import logging
import os


def setup_logger(name, level=None):
    # set up logger with appropriate level and formatting
    # args:
    #   name: logger name (typically __name__)
    #   level: log level override (if None, uses RRF_LOG_LEVEL env var or INFO)
    # returns:
    #   configured logger
    if level is None:
        # check environment variable for log level
        env_level = os.environ.get('RRF_LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, env_level, logging.INFO)
    
    logger = logging.getLogger(name)
    
    # clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    # set level
    logger.setLevel(level)
    
    # create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # minimal format for analysis scripts
    formatter = logging.Formatter('%(message)s')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # prevent duplicate logs from parent loggers
    logger.propagate = False
    
    return logger