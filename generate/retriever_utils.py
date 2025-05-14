"""
Utility functions for retrieval
"""
import re
import os
import pyterrier as pt
from typing import Dict, List, Any

from config import BM25_INDEX_PATH
from utils import setup_logger

# Setup logger
logger = setup_logger("retriever_utils")

def cleanup(query):
    """Clean up query text"""
    # Replace special characters with spaces
    query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    # Replace multiple spaces with a single space
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def pt_reformulation():
    """
    Query reformulation pipeline using PyTerrier
    Removes stop words and expands query with synonyms
    """
    # Create a simple pipeline that reformulates queries
    # This is a placeholder for more complex query reformulation
    return pt.apply.generic(lambda row: {"query": cleanup(row["query"])})

def get_sparse_index(pt_index_path=BM25_INDEX_PATH):
    """
    Load the existing sparse index for FineWeb
    
    Args:
        pt_index_path: Path to the PyTerrier index directory or properties file
    
    Returns:
        PyTerrier index object
    """
    try:
        # First check if the provided path is a properties file
        if os.path.isfile(pt_index_path) and pt_index_path.endswith("data.properties"):
            properties_path = pt_index_path
        # Otherwise, check if it's a directory containing data.properties
        elif os.path.isdir(pt_index_path) and os.path.exists(os.path.join(pt_index_path, "data.properties")):
            properties_path = os.path.join(pt_index_path, "data.properties")
        # Last resort, assume it's directly the properties path
        else:
            properties_path = pt_index_path
            
        # Load the index using the properties file
        logger.info(f"Loading index from {properties_path}")
        index_ref = pt.IndexRef.of(properties_path)
        index = pt.IndexFactory.of(index_ref)
        
        # Log index statistics
        stats = index.getCollectionStatistics()
        logger.info(f"Loaded index with {stats.getNumberOfDocuments()} documents and {stats.getNumberOfTokens()} tokens")
        
        return index
    except Exception as e:
        logger.error(f"Error loading index from {pt_index_path}: {e}")
        raise