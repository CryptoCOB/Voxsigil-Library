from typing import List, Dict, Any, Optional
from MetaConsciousness.core.context import SDKContext


def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for the given text.
    
    Args:
        text: The text to embed
        
    Returns:
        List[float]: The embedding vector
    """
    model_router = SDKContext.get("model_router")
    
    if not model_router:
        from MetaConsciousness.models.router import default_router
        model_router = default_router
    
    # Explicitly specify this is an embedding task
    response = model_router.query(
        prompt=text,
        task_type="embedding",  # Specify this is an embedding task
        system_prompt="",  # No system prompt needed for embeddings
        temperature=0.0,   # Use deterministic output for embeddings
        max_tokens=0       # Not generating text
    )
    
    return response.get("embedding", [])

def get_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    import numpy as np
    
    # Get embeddings
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    if not emb1 or not emb2:
        return 0.0
    
    # Convert to numpy arrays for calculation
    vec1 = np.array(emb1)
    vec2 = np.array(emb2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
        
    return dot_product / norm_product
