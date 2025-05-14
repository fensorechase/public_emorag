# auth.py
"""
Authentication utilities for accessing Hugging Face models
"""
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger("liverag.auth")

def setup_huggingface_auth(token: Optional[str] = None) -> None:
    """
    Set up Hugging Face authentication using either:
    1. Provided token parameter
    2. HF_TOKEN environment variable
    3. Token stored in ~/.huggingface/token file
    
    Args:
        token: Optional explicit Hugging Face token
    """
    if token:
        # Use provided token
        os.environ["HF_TOKEN"] = token
        logger.info("Using provided Hugging Face token")
        return
        
    # Check if token is already in environment
    if os.environ.get("HF_TOKEN"):
        logger.info("Using Hugging Face token from environment variable")
        return
        
    # Check for token file
    token_path = Path.home() / ".huggingface" / "token"
    if token_path.exists():
        try:
            with open(token_path, "r") as f:
                token = f.read().strip()
                if token:
                    os.environ["HF_TOKEN"] = token
                    logger.info(f"Using Hugging Face token from {token_path}")
                    return
        except Exception as e:
            logger.warning(f"Error reading token file: {e}")
    
    logger.warning(
        "No Hugging Face token found. You may need to log in manually or "
        "provide a token for accessing gated models like Falcon-3-10B-instruct."
    )

def login_huggingface_cli() -> None:
    """
    Login to Hugging Face using the CLI
    This is an alternative to setting the token directly
    """
    from huggingface_hub import login
    
    # Check if token exists in environment
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face using token")
    else:
        # Interactive login
        login()
        logger.info("Completed interactive Hugging Face login")