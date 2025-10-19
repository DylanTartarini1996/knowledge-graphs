import community
import networkx as nx

from igraph import Graph
from leidenalg import find_partition, ModularityVertexPartition
from src.utils.logger import get_logger
from neo4j import Query, Session
from typing import Any, Dict, Tuple


logger = get_logger(__name__)


def detect_louvain_communities(G: nx.DiGraph, return_modularity:bool=True) -> nx.DiGraph | Tuple[nx.DiGraph, float]:
    """ 
    Detects Louvain communities for a `networkx` Directed Graph. 
    If `return_modularity`, also return the modularity of the Graph according to 
    the Louvain distance measure.
    """
    G_undirected = G.to_undirected()

    partition = community.best_partition(G_undirected)  # Louvain method

    nx.set_node_attributes(G, partition, "community_louvain")  # Store communities in node attributes

    if not return_modularity:

        return G
    
    else: 
        modularity = community.modularity(partition, G_undirected)

        logger.info(f"Modularity based on Louvain communities: {modularity}")

        return G, modularity 
    

def detect_leiden_communities(G: nx.DiGraph, return_modularity:bool=True) -> nx.DiGraph | Tuple[nx.DiGraph, float]:
    """
    Detects Leiden communities for a `networkx` Directed Graph. 
    If `return_modularity`, also return the modularity of the Graph according to 
    the Louvain distance measure.
    """
    
    # Convert networkx to igraph
    mapping = {node: i for i, node in enumerate(G.nodes())}  # Node mapping
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    # Create igraph graph
    ig_G = Graph(directed=True)
    ig_G.add_vertices(len(G.nodes()))
    ig_G.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])
    
    partition = find_partition(ig_G, ModularityVertexPartition)

    # Assign community labels back to NetworkX
    for i, comm in enumerate(partition):
        for node in comm:
            G.nodes[reverse_mapping[node]]["community_leiden"] = i 
    
    if not return_modularity:
        return G
    
    else:
        modularity = partition.modularity

        logger.info(f"Modularity based on Leiden communities: {modularity}")

        return G, modularity
    

def compute_centralities(G: nx.DiGraph | nx.Graph) -> nx.DiGraph | nx.Graph:
    """
    Compute PageRank, Betweenness and Closeness Centralities and store them as metadata in the graph
    """
    
    pr = nx.pagerank(G, alpha=0.85)
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)

    nx.set_node_attributes(G, pr, "pagerank")
    nx.set_node_attributes(G, bc, "betweenness")
    nx.set_node_attributes(G, cc, "closeness")
    
    return G


def update_modularity(session: Session, mod: float, mod_type: str="leiden"):
    """
    Save Leiden or Louvain modularity score as a graph-wide property (inside a node).
    
    params: 
    -------
    `session`: `Session`  
        Neo4j Session
    `mod`: `float`  
        Modularity score
    `mod_type`: `str`
        Either `leiden` or `louvain`
    """
    if mod_type in ["leiden", "louvain"]:   
        try:
            session.run(
                f"""MERGE (m:GraphMetric {{name: '{mod_type}_modularity'}}) SET m.value = $modularity""", 
                modularity=mod
            )
        except Exception as e:
            logger.warning(f"Issue updating Leiden modularity property: {e}") 
    else:
        raise NotImplementedError("This Modularity type has not been implemented.")   
    
    
def build_update_query(
        node_id, 
        centralities=False, 
        leiden_communities=False, 
        louvain_communities=False,
        community_leiden: int=-1, 
        community_louvain: int=-1, 
        pagerank: float=0.0, 
        betweenness: float=0.0,
        closeness: float=0.0 
    ) -> Tuple[Query, Dict[str, Any]]:
    """ 
    Returns `Query` and `dict`with parameters to update node properies
    """
    
    # Base query
    query = "MATCH (n) WHERE elementId(n) = $node_id\n"

    # List to hold SET clauses
    set_clauses = []
    parameters = {"node_id": node_id}

    if leiden_communities:
        set_clauses.append("n.community_leiden = $community_leiden")
        parameters["community_leiden"] = community_leiden

    if louvain_communities:
        set_clauses.append("n.community_louvain = $community_louvain")
        parameters["community_louvain"] = community_louvain

    if centralities:
        set_clauses.append("n.pagerank = $pagerank")
        set_clauses.append("n.betweenness = $betweenness")
        set_clauses.append("n.closeness = $closeness")
        parameters.update(
            {"pagerank": pagerank, 
             "betweenness": betweenness, 
             "closeness": closeness
            }
        )

    # Only add SET if there's something to update
    if set_clauses:
        query += "SET " + ",\n    ".join(set_clauses)  # Join clauses with proper formatting

    return query, parameters