"""
Centralized configuration for Drain3 template miner.
This ensures consistency between parsing and server inference.
"""

class Drain3Config:
    """Drain3 template miner configuration."""
    
    # Drain algorithm parameters
    DEPTH = 4  # Drain tree depth
    SIMILARITY_THRESHOLD = 0.5  # Similarity threshold (st)
    
    # Performance settings
    PROFILING_ENABLED = False  # Disable profiling in production
    
    # State persistence
    STATE_FILE = "drain3_state.bin"  # Relative to parsers/ directory
    
    # Log format for HDFS logs
    LOG_FORMAT = r'<Date> <Time> <Pid> <Level> <Component>: <Content>'
