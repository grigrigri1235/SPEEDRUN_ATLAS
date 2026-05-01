import torch
import torch.nn.functional as F

def get_unembedding_matrix(model):
    """Retrieve the unembedding matrix from the model."""
    return model.lm_head.weight

def calculate_dot_products(unembedding_matrix, target_token_id, comparison_token_ids):
    """
    Calculate dot products between a target embedding and multiple comparison embeddings.
    
    Args:
        unembedding_matrix: Tensor of shape [vocab_size, hidden_dim]
        target_token_id: ID of the reference token (e.g., 'owl')
        comparison_token_ids: List of IDs of tokens to compare against
        
    Returns:
        List of dot product values
    """
    target_embedding = unembedding_matrix[target_token_id]
    dot_products = []
    for tid in comparison_token_ids:
        comp_embedding = unembedding_matrix[tid]
        dot_products.append(torch.dot(target_embedding, comp_embedding).item())
    return dot_products

def calculate_cosine_similarities(unembedding_matrix, target_token_id, comparison_token_ids):
    """
    Calculate cosine similarities between a target embedding and multiple comparison embeddings.
    
    Args:
        unembedding_matrix: Tensor of shape [vocab_size, hidden_dim]
        target_token_id: ID of the reference token
        comparison_token_ids: List of IDs of tokens to compare against
        
    Returns:
        List of cosine similarity values
    """
    target_embedding = unembedding_matrix[target_token_id]
    target_norm = F.normalize(target_embedding, dim=0)
    
    cosine_sims = []
    for tid in comparison_token_ids:
        comp_embedding = unembedding_matrix[tid]
        comp_norm = F.normalize(comp_embedding, dim=0)
        cosine_sims.append(torch.dot(target_norm, comp_norm).item())
    return cosine_sims

def get_token_rankings_by_geometry(unembedding_matrix, target_token_id, comparison_token_ids, metric="dot"):
    """
    Rank comparison tokens by their geometric proximity to the target token.
    
    Args:
        unembedding_matrix: Tensor [vocab_size, hidden_dim]
        target_token_id: Reference token ID
        comparison_token_ids: List of comparison token IDs
        metric: "dot" or "cosine"
        
    Returns:
        Sorted list of tuples (token_id, metric_value)
    """
    if metric == "dot":
        vals = calculate_dot_products(unembedding_matrix, target_token_id, comparison_token_ids)
    elif metric == "cosine":
        vals = calculate_cosine_similarities(unembedding_matrix, target_token_id, comparison_token_ids)
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    data = list(zip(comparison_token_ids, vals))
    return sorted(data, key=lambda x: x[1], reverse=True)
