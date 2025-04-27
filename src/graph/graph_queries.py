from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
from neo4j import Query, Session

from src.graph.graph_model import Node, Relationship, Community, CommunityReport
from src.graph.knowledge_graph import KnowledgeGraph
from src.schema import Chunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


def document_metadata(session: Session, filename: str, version: Optional[int]) -> dict:
    """ Returns a dictionary with metadata from a `Document` node in the Graph"""
    pass


def get_chunk_element_id(session: Session, chunk: Chunk) -> str | None:
    """ Returns the unique elementId in the graph for a given `Chunk`"""
    
    query = """ 
        MATCH (c:Chunk {filename: $filename, chunk_id: $chunk_id, text: $text})
        RETURN elementId(c) AS element_id
    """
    try:
        result = session.run(
            query, 
            filename=chunk.filename, 
            chunk_id=chunk.chunk_id,
            text=chunk.text
        )
        record = result.single()
        return record["element_id"]
    except Exception as e:
        logger.warning(f"Unable to retrieve elementId for Chunk: {chunk.chunk_id}")
        return None
        

def get_adjacent_chunks(
    session: Session, 
    chunk: Chunk, 
    use_elementId: bool=False
    ) -> Tuple[Chunk | None, Chunk , Chunk | None]:
    """
    Returns a tuple with the previous , current and following `Chunk` 
    given an initial node characterised by a `filename` and a `chunk_id`.  
    If `use_elementId` is set to `True`, will use the elementId of the chunk instead. 
    """
    if use_elementId:
        base_query = """ 
            MATCH (current:Chunk)
            WHERE elementId(current) = $elementId

            OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(current)
            OPTIONAL MATCH (current)-[:NEXT]->(next:Chunk)

            RETURN prev AS previous_chunk, current, next AS next_chunk
        """
        try: 
            result = session.run(base_query, elementId=chunk.chunk_id)
            record = result.single()
        except Exception as e:
            logger.warning(f"Unable to retrieve adjacent chunks for Chunk: {chunk.chunk_id}")
            return None, chunk, None
        
    else: 
        base_query = """ 
            MATCH (current:Chunk)
            WHERE current.chunk_id = $chunk_id AND current.filename = $filename

            OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(current)
            OPTIONAL MATCH (current)-[:NEXT]->(next:Chunk)

            RETURN prev AS previous_chunk, current, next AS next_chunk
        """
        
        try: 
            result = session.run(base_query, chunk_id=chunk.chunk_id, filename=chunk.filename)
            record = result.single()
        except Exception as e:
            logger.warning(f"Unable to retrieve adjacent chunks for Chunk: {chunk.chunk_id}")
            return None, chunk, None
    
    previous_chunk = dict(record["previous_chunk"]) if record["previous_chunk"] else None
    if previous_chunk:
        previous_chunk = Chunk(
            chunk_id=previous_chunk['chunk_id'],
            filename=previous_chunk['filename'],
            text=previous_chunk["text"],
        )
        chunk.chunk_id = previous_chunk.chunk_id + 1 # original chunk id
    next_chunk = dict(record["next_chunk"]) if record["next_chunk"] else None
    if next_chunk:
        next_chunk = Chunk(
            chunk_id=next_chunk['chunk_id'],
            filename=next_chunk['filename'],
            text=next_chunk["text"],
        )
        chunk.chunk_id = next_chunk.chunk_id-1 # original chunk id
    
    return previous_chunk, chunk, next_chunk



def get_mentioned_entities(
    session: Session, 
    chunk: Chunk,
    n_hops: int=1, 
    use_elementId: bool = False
    ) -> List[Dict[str, Any]]:
    """ 
    Follows the `MENTIONS` relationships of a given Chunk in the Graph and collects mentioned entities. 
    `n_hops` is used to indicate the number of relationship layers that could be done following entities linking.  
    """
    nodes = []
    
    # TODO perform n-hops retrieval
    if use_elementId:
        base_query = """
            MATCH (c:Chunk)
            WHERE elementId(c) = $elementId
            MATCH (c)-[:MENTIONS]->(mentioned)
            RETURN collect(mentioned) AS mentioned_nodes
        """
        try: 
            result= session.run(base_query, elementId=chunk.chunk_id)
            record = result.single()
            mentioned_nodes = record["mentioned_nodes"] if record else []
            for node in mentioned_nodes:
                nodes.append(dict(node))
        
            logger.info(f"Retrieved {len(nodes)} entities for chunk {chunk.chunk_id}")
            
            return nodes
        
        except Exception as e:
            logger.warning(f"No mentioned entities retrieved with exception: {e}")
            return []
    
    else: 
        base_query = """ 
            MATCH (c:Chunk)
            WHERE c.chunk_id = $chunk_id AND c.filename = $filename
            MATCH (c)-[:MENTIONS]->(mentioned)
            RETURN collect(mentioned) AS mentioned_nodes
        """
        try: 
            result= session.run(base_query, chunk_id=chunk.chunk_id, filename=chunk.filename)
            record = result.single()
            mentioned_nodes = record["mentioned_nodes"] if record else []
            for node in mentioned_nodes:
                nodes.append(dict(node))
        
            logger.info(f"Retrieved {len(nodes)} entities for chunk {chunk.chunk_id}")
            
            return nodes
        
        except Exception as e:
            logger.warning(f"No mentioned entities retrieved with exception: {e}")
            return []
        
        
def filter_graph_by_communities(session: Session, community_ids: List[int], community_type: str="leiden") -> List[Dict[str, Any]]:
    """
    Creates a temporary  view of the Knowledge Graph to filter it into subgraphs given community ids.
    """
    query = f"""
        MATCH (n)-[r]->(m)
        WHERE n.community_{community_type} IN $community_values
            AND NOT n:Chunk
            AND NOT m:Chunk
        RETURN n, r, m
    """
    
    keys_to_remove = {
        'community_louvain', 'community_leiden', 'pagerank',
        'id', 'betweenness', 'closeness'
    }
    
    try:
        result = session.run(query, community_values=community_ids)
        
        subgraph = []
    
        for record in result:
            node_1 = {k: v for k, v in dict(record["n"]).items() if k not in keys_to_remove}
            node_2 = {k: v for k, v in dict(record["m"]).items() if k not in keys_to_remove}
            relationship = dict(record["r"])
            
            subgraph.append({
                "node_1": node_1,
                "relationship": relationship,
                "node_2": node_2
            })
        
        return subgraph
    
    except Exception as e:
        print(f"Error while fetching subgraph: {e}")
        return []
    
    
def project_temporary_subgraph(session: Session, element_ids: List[str], n_hops: int=2, graph_name: str="tempGraph"):
    """ 
    Projects a temporary subgraph into Neo4j, based on elementIds from Chunks.  
    The temporary subgraph will disappear after the connection is closed or lost. 
    
    **WARNING**: this only works if your Neo4j instance is abilitated to use `gds` procedures
    """
    
    
    node_query = f"""
        UNWIND $element_ids AS eid
        MATCH (c:Chunk)
        WHERE elementId(c) = eid
        WITH collect(c) AS chunks
        UNWIND chunks AS c
        MATCH (c)-[*1..{n_hops}]-(e)
        RETURN id(c) AS id, labels(c) AS labels
        UNION
        RETURN id(e) AS id, labels(e) AS labels
    """

    rel_query = f"""
        UNWIND $element_ids AS eid
        MATCH (c:Chunk)
        WHERE elementId(c) = eid
        WITH collect(c) AS chunks
        UNWIND chunks AS c
        MATCH (c)-[r*1..{n_hops}]-(e)
        UNWIND r AS rel
        RETURN id(startNode(rel)) AS source, id(endNode(rel)) AS target, type(rel) AS type
    """

    session.run(
        """
        CALL gds.graph.project(
            $graph_name,
            {
              nodeProjection: $node_query
            },
            {
              relationshipProjection: $rel_query
            },
            {
              validateRelationships: false
            }
        )
        """,
        {
            "graph_name": graph_name,
            "node_query": node_query,
            "rel_query": rel_query,
            "element_ids": element_ids
        }
    )