import re
import networkx as nx
import pandas as pd

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

from langchain.schema import Document
from langchain.load.serializable import Serializable
from langchain_neo4j.graphs.graph_document import Node, Relationship, GraphDocument

from src.schema import Chunk


class _Node(Serializable):
    id: str
    type: str
    properties: Optional[Dict[str, str]] = None


class _Relationship(Serializable):
    source: str
    target: str
    type: str
    properties: Optional[Dict[str, str]] = None


class _Graph(Serializable):
    """ 
    Represents a graph consisting of nodes and relationships.  
    
    -----------
    Attributes:
    -----------
        `nodes (List[_Node])`: A list of nodes in the graph.
        `relationships (List[_Relationship])`: A list of relationships in the graph.
    """
    nodes: List[_Node]
    relationships: List[_Relationship]


class Ontology(BaseModel):
    """     
    Used to describe arbitrary, project-specific allowed labels and relationships.

    Labels should map to `Node.type` and relationships to `Relationship.type` from 
    `langchain_neo4j.graphs.graph_document`. It is allowed to provide a functional 
    description of what labels and relationships represent in the domain.  
    
    """
    allowed_labels: Optional[List[str]]=None
    labels_descriptions: Optional[Dict[str, str]]=None
    allowed_relations: Optional[List[str]]=None
    
    
class Community(BaseModel):
    """ 
    Describes a community in the Knowledge Graph.
    
    -----------
    Attributes:
    -----------
    `community_type`: `str`
        The type of community, such as `leiden` or `louvain`
    `community_id`: `int`
        The identifier of this community in the graph nodes properties
    `community_size`: `Optional[int]`
        The number of nodes in the graph with attribute 'community_type: community_id'
    `entity_ids`: `Optional[List[str]]`
        List of entity IDs related to the community
    `relationship_ids`: `Optional[List[str]]`
        List of relationship IDs related to the community
    `table_repr`: `Optional[pd.DataFrame]`
        Table Representation of the community
    `attributes`: `Optional[Dict[str, Any]]`
        A dictionary of additional attributes associated with the community
    """
    community_type: str
    community_id: int
    community_size: Optional[int] = None
    entity_ids: Optional[List[str]] = None
    entity_names: Optional[List[str]] = None
    relationship_ids: Optional[List[str]] = None
    relationship_types:  Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    chunks: Optional[List[Chunk]] = None
    table_repr: Optional[pd.DataFrame] = None # TODO how to fetch this?
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    
class CommunityReport(BaseModel):
    """ 
    Summary report from a given `Community`
    
    -----------
    Attributes:
    -----------
    `community_type`: `str`
        The type of community, such as `leiden` or `louvain`
    `community_id`: `int`
        The identifier of this community in the graph nodes properties
    `summary`: `str`
        Summary of the report
    community_size`: `Optional[int]`
        The number of nodes in the graph with attribute 'community_type: community_id'
    `rank`: `float`
        Used for sorting. The higher the better. 
    `attributes`: `Optional[Dict[str, Any]]`
        A dictionary of additional attributes associated with the report
    """
    communtiy_type: str
    community_id: int
    summary: str = ""
    rank: float = 0.0
    community_size: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None   
    summary_embeddings: Optional[List[float]] = None
    
    
def graph_document_to_digraph(graph_doc: GraphDocument) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph_doc.nodes:
        G.add_node(
            node.id, 
            type=node.type
        )
    for relationship in graph_doc.relationships:
        G.add_edge(
            relationship.source.id, 
            relationship.target.id, 
            relationship=relationship.type, 
        )
    return G


def digraph_to_dict(G: nx.DiGraph, remove_unknown: bool=True) -> dict:

    graph_dict = {}
    
    for node in G.nodes(data=True):
        node_id = node[0]
        node_type = node[1]['type'] if 'type' in node[1].keys() else "unknown"
        graph_dict[node_id] = {'type': node_type, 'relationships': []}
        
    for node_id in G.nodes():
        successors = [
            (successor, G[node_id][successor].get('relationship', 'unknown')) 
            for successor in G.successors(node_id)
        ]        
        graph_dict[node_id]['relationships'] = successors
    
    if remove_unknown:
        graph_dict = remove_unknown_relationships(document_graph=graph_dict)
        
    return graph_dict


def dict_to_graph_document(graph_dict: Dict[str, Any], source_content: str="") -> GraphDocument:
    
    nodes = []
    nodes_map = {}  # To map node IDs to Node objects
    for node_id, node_info in graph_dict.items():
        node = Node(id=node_id, type=node_info['type'])
        nodes.append(node)
        nodes_map[node_id] = node
    
    relationships = []
    for node_id, node_info in graph_dict.items():
        for successor, relationship_type in node_info['relationships']:
            relationship = Relationship(
                source=nodes_map[node_id],
                target=nodes_map[successor],
                type=relationship_type
            )
            relationships.append(relationship)
    
    source = Document(page_content=source_content)
    
    graph_doc = GraphDocument(
        nodes=nodes, 
        relationships=relationships, 
        source=source
    )
    
    return graph_doc


def remove_unknown_relationships(document_graph: dict) -> dict:
    for key, value in document_graph.items():
        if 'relationships' in value:
            value['relationships'] = [
                relationship for relationship in value['relationships']
                if 'unknown' not in relationship
            ]
    return document_graph


def normalize_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """Normalize Nodes names"""
    mapping = {node: _normalize(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G
    

def _normalize(s: str) -> str:
    return re.sub(r'[.,;:!?@#$%^&*()\-_\[\]{}<>/\\\'"~\s]', ' ', s)


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties


def map_to_lc_node(node: _Node) -> Node:
    """Maps the `_Graph` `_Node` to the `langchain_neo4j.graphs.graph_document.Node`"""
    properties = node.properties if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return Node(
        id=node.id.title(), 
        type=node.type.capitalize(), 
        properties=properties
    )


def map_to_lc_relationship(rel: _Relationship, nodes: List[_Node]) -> Relationship:
    """Maps the `_Graph` `_Relationship`  to the `langchain_neo4j.graphs.graph_document.Relationship`"""
    
    source_node = [node for node in nodes if node.id == rel.source][0]
    target_node = [node for node in nodes if node.id == rel.target][0]

    source = map_to_lc_node(source_node)
    target = map_to_lc_node(target_node)

    properties = rel.properties if rel.properties else {}

    return Relationship(
        source=source, 
        target=target, 
        type=rel.type, 
        properties=properties
    )


def map_to_lc_graph(graph: _Graph, source_content: str) -> GraphDocument:
    """
    Maps the `_Graph` class to the 
    `langchain_neo4j.graphs.graph_document.GraphDocuemnt` class
    """
    nodes = [map_to_lc_node(node) for node in graph.nodes]

    relationships = [map_to_lc_relationship(rel, graph.nodes) for rel in graph.relationships]

    graph_doc = GraphDocument(
        nodes=nodes, 
        relationships=relationships,
        source=Document(page_content=source_content)
    )

    return graph_doc