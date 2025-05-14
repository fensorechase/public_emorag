"""
utils.py
Utility functions for the LiveRAG challenge - adjusted for 114_synthetic.json
"""
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import os

from config import LOG_LEVEL, LOGS_DIR, MAX_RESPONSE_TIME

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f"liverag_{time.strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("liverag")

def setup_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(f"liverag.{name}")

def log_execution_time(func):
    """Decorator to log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def save_results(results: Dict[str, Any], filename: str, results_dir: str) -> str:
    """Save results to a JSON file"""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath

def format_passage(passage: Dict[str, Any], index: int = None) -> str:
    """Format a passage for inclusion in the prompt"""
    prefix = f"[{index}] " if index is not None else ""
    return f"{prefix}{passage['text']}\n\n"

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Placeholder function for truncating text to fit within token limit"""
    # For simplicity, we'll just truncate by character count as an approximation
    approx_chars_per_token = 4
    max_chars = max_tokens * approx_chars_per_token
    
    if len(text) <= max_chars:
        return text
    
    return text[:max_chars] + "..."

def get_unique_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate passages based on content"""
    seen_contents = set()
    unique_passages = []
    
    for passage in passages:
        if passage['text'] not in seen_contents:
            seen_contents.add(passage['text'])
            unique_passages.append(passage)
    
    return unique_passages

class ResponseTimeTracker:
    """Utility to track and manage response time"""
    
    def __init__(self, max_response_time: float = MAX_RESPONSE_TIME):
        self.max_response_time = max_response_time
        self.start_time = time.time()
    
    def time_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds"""
        return max(0.0, self.max_response_time - self.time_elapsed())
    
    def should_terminate_early(self, buffer_time: float = 0.5) -> bool:
        """Check if we should terminate early to meet time constraints"""
        return self.time_remaining() <= buffer_time
    
    def get_progress_percentage(self) -> float:
        """Get percentage of allowed time used"""
        return min(100.0, (self.time_elapsed() / self.max_response_time) * 100)

class SimpleCache:
    """Simple in-memory cache with time-based expiration"""
    
    def __init__(self, max_size: int = 1000, expiry_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.expiry_seconds = expiry_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and is not expired"""
        if key not in self.cache:
            return None
        
        access_time = self.access_times.get(key, 0)
        if time.time() - access_time > self.expiry_seconds:
            # Item has expired
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Add item to cache, evicting oldest items if necessary"""
        # Evict oldest items if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()