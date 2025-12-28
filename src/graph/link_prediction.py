"""
Link prediction algorithms for discovering new relations between entities.

This module implements baseline heuristic methods for predicting missing or 
future links in a knowledge graph. All methods operate on NetworkX graphs.
"""

import networkx as nx
from typing import List, Tuple, Optional, Set
from src.utils.logger import get_logger


logger = get_logger(__name__)


def common_neighbors(G: nx.Graph, u: str, v: str) -> float:
    """
    Count the number of common neighbors between two nodes.
    
    Higher scores indicate nodes with more shared connections.
    """
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    return float(len(neighbors_u & neighbors_v))


def jaccard_coefficient(G: nx.Graph, u: str, v: str) -> float:
    """
    Compute Jaccard similarity between node neighborhoods.
    
    Normalizes common neighbors by the size of the union of neighborhoods.
    Returns value between 0 and 1.
    """
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    
    intersection = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def adamic_adar_index(G: nx.Graph, u: str, v: str) -> float:
    """
    Compute Adamic-Adar index between two nodes.
    
    Weights common neighbors by the inverse log of their degree.
    Rare connections are weighted more heavily than common ones.
    """
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u & neighbors_v
    
    score = 0.0
    for neighbor in common:
        degree = G.degree(neighbor)
        if degree > 1:
            score += 1.0 / (degree ** 0.5)  # Using sqrt instead of log for numerical stability
    
    return score


def preferential_attachment(G: nx.Graph, u: str, v: str) -> float:
    """
    Compute preferential attachment score (product of node degrees).
    
    Implements the "rich get richer" principle - high-degree nodes 
    are more likely to form connections.
    """
    return float(G.degree(u) * G.degree(v))


def predict_links(
    G: nx.DiGraph,
    method: str = "adamic_adar",
    top_k: int = 10,
    exclude_existing: bool = True,
    node_filter: Optional[Set[str]] = None
) -> List[Tuple[str, str, float]]:
    """
    Predict missing or future links in a directed graph.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The directed graph to analyze
    method : str
        Link prediction method to use. Options:
        - 'common_neighbors': Count shared neighbors
        - 'jaccard': Jaccard similarity of neighborhoods
        - 'adamic_adar': Adamic-Adar index (default)
        - 'preferential_attachment': Product of node degrees
    top_k : int
        Number of top predictions to return (default: 10)
    exclude_existing : bool
        If True, exclude existing edges from predictions (default: True)
    node_filter : Optional[Set[str]]
        If provided, only consider node pairs within this set
        
    Returns:
    --------
    List[Tuple[str, str, float]]
        List of (source_node, target_node, score) tuples, sorted by score descending
    """
    # Convert to undirected for link prediction (treat graph as symmetric)
    G_undirected = G.to_undirected()
    
    # Select scoring function
    score_functions = {
        "common_neighbors": common_neighbors,
        "jaccard": jaccard_coefficient,
        "adamic_adar": adamic_adar_index,
        "preferential_attachment": preferential_attachment
    }
    
    if method not in score_functions:
        raise ValueError(f"Unknown method: {method}. Choose from {list(score_functions.keys())}")
    
    score_fn = score_functions[method]
    
    # Get nodes to consider
    nodes = list(node_filter) if node_filter else list(G_undirected.nodes())
    
    # Get existing edges (for exclusion)
    existing_edges = set()
    if exclude_existing:
        for u, v in G_undirected.edges():
            existing_edges.add((u, v))
            existing_edges.add((v, u))  # Both directions for undirected
    
    # Compute scores for all non-connected node pairs
    predictions = []
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:  # Only consider each pair once
            # Skip if edge already exists
            if exclude_existing and ((u, v) in existing_edges or (v, u) in existing_edges):
                continue
            
            # Compute score
            try:
                score = score_fn(G_undirected, u, v)
                if score > 0:  # Only include non-zero scores
                    predictions.append((u, v, score))
            except Exception as e:
                logger.warning(f"Error computing {method} for ({u}, {v}): {e}")
    
    # Sort by score descending and return top k
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    return predictions[:top_k]
