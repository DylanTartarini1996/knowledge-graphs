import networkx as nx

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from neo4j import ManagedTransaction
from typing import List, Optional, Tuple

from src.config import KnowledgeGraphConfig
from src.graph.graph_model import Community, CommunityReport
from src.graph.graph_ds import (
    build_update_query,
    compute_centralities, 
    detect_leiden_communities, 
    detect_louvain_communities, 
    update_modularity
)
from src.schema import Chunk, ProcessedDocument
from src.utils.logger import get_logger


logger = get_logger(__name__)

BASE_ENTITY_LABEL = "__Entity__"


class KnowledgeGraph(Neo4jGraph):
    """
        Class used to represent a Knowledge Base under graph representation, 
        using `neo4j` as the backend for querying operations.
        
        If an `Ontology` is provided (see `KnowledgeGraphConfig.ontology`), will not allow for nodes and relationships
        to be created outside of the given sets of allowed labels and relationships.
    """

    def __init__(
            self, 
            conf: KnowledgeGraphConfig,
            embeddings_model: Embeddings,
            sanitize = False, 
            refresh_schema = True, 
            enhanced_schema = False
        ):
        if conf.uri is not None:
            self.url = conf.uri
        else: 
            self.url = f"{conf.db_schema}://{conf.host_name}:{conf.port}"
        self.username = conf.user
        self.password = conf.password
        self.database = conf.database
        self.timeout = conf.timeout
        self.index_name = conf.index_name

        if conf.ontology: # TODO 
            self.allowed_labels = conf.ontology.allowed_labels
            self.allowed_relationships = conf.ontology.allowed_relations
            
        self.embeddings = embeddings_model

        self._labels_ = None 
        self._number_of_entities_ = None
        self._number_of_labels_ = None
        self._number_of_relationships_ = None
        self._number_of_docs = None
        self._relationships_ = None
        self._leiden_modularity = None
        self._number_of_leiden_communities = None
        self._louvain_modularity = None
        self._number_of_louvain_communities = None

        try: 
            self.vector_store = Neo4jVector(
                embedding=self.embeddings,
                url=self.url,
                username=self.username, 
                database=self.database,
                password=self.password,
                index_name=self.index_name,
                node_label="Chunk",
                embedding_node_property="embedding",
                text_node_property="text",
            )
        except Exception as e:
            logger.warning(f"Error connecting to Neo4jVector: {e}")
            
        try:
            self.cr_store = Neo4jVector(
                embedding=self.embeddings,
                url=self.url,
                username=self.username, 
                database=self.database,
                password=self.password,
                index_name="reports",
                node_label="CommunityReport",
                embedding_node_property="summary_embeddings",
                text_node_property="summary",
            )
        except Exception as e:
            logger.warning(f"Error connecting to Neo4jVector: {e}")

        super().__init__(
            url=self.url, 
            username=self.username,
            password=self.password,
            database=self.database,
            timeout=self.timeout,
            sanitize=sanitize, 
            refresh_schema=refresh_schema,
            enhanced_schema=enhanced_schema
        )
        

    @property
    def labels(self) -> List[str]:
        """
        Returns a list of labels in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.labels() YIELD label RETURN COLLECT(label) AS labels"
            result = session.run(query)
            self._labels = result.single()["labels"]
        return self._labels
    

    @property
    def relationships(self) -> List[str]:
        """
        Returns a list of relationships in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.relationshipTypes() YIELD relationshipType RETURN COLLECT(relationshipType) AS relationship_types"
            result = session.run(query)
            self._relationships_ = result.single()["relationship_types"]
        return self._relationships_
    

    @property
    def number_of_nodes(self) -> int:
        """
        Returns the total number of nodes in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n) RETURN COUNT(n) AS nodes"
            result = session.run(query)
            self._number_of_entities = result.single()["nodes"]
        return self._number_of_entities
    

    @property
    def number_of_labels(self) -> int:
        """
        Returns the number of labels in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.labels() YIELD label RETURN COUNT(label) AS num_labels"
            result = session.run(query)
            self._number_of_labels = result.single()["num_labels"]
        return self._number_of_labels
    

    @property
    def number_of_relationships(self) -> int:
        """
        Returns the total number of relationships in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH ()-[r]-() RETURN COUNT(r) AS num_relationships"
            result = session.run(query)
            self._number_of_relationships = result.single()["num_relationships"]
        return self._number_of_relationships
    

    @property
    def number_of_docs(self) -> int:
        """
        Returns the current number of documents collected in the Knowledge Graph
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n: Document) RETURN COUNT(n) AS num_docs"
            result = session.run(query)
            self._number_of_docs = result.single()["num_docs"]
        return self._number_of_docs
    
    
    @property
    def leiden_modularity(self) -> float:
        query = """MATCH (m:GraphMetric WHERE m.name = 'leiden_modularity') RETURN m.value AS mod"""
        with self._driver.session(database=self._database) as session:
            try: 
                result = session.run(query)
                self._leiden_modularity = result.single().value()
                return self._leiden_modularity
            except Exception as e:
                logger.warning("Leiden Modularity has not been computed")
        
            
                
    @property
    def louvain_modularity(self) -> float:
        query = """MATCH (m:GraphMetric WHERE m.name = 'louvain_modularity') RETURN m.value AS mod"""
        with self._driver.session(database=self._database) as session:
            try: 
                result = session.run(query)
                self._louvain_modularity = result.single().value()
                return self._louvain_modularity
            except Exception as e:
                logger.warning("Louvain Modularity has not been computed")
                
    
    @property
    def number_of_louvain_communities(self) -> int:
        query = """
            MATCH (n)
            WHERE n.community_louvain IS NOT NULL
            RETURN count(DISTINCT n.community_louvain) AS num_communities
        """
        with self._driver.session(database=self._database) as session:
            try: 
                result = session.run(query)
                self._number_of_louvain_communities = result.single()["num_communities"]
                return self._number_of_louvain_communities
            except Exception as e:
                logger.warning("Louvain communities have not been detected yet")
                
                
    @property
    def number_of_leiden_communities(self) -> int:
        query = """
            MATCH (n)
            WHERE n.community_leiden IS NOT NULL
            RETURN count(DISTINCT n.community_leiden) AS num_communities
        """
        with self._driver.session(database=self._database) as session:
            try: 
                result = session.run(query)
                self._number_of_leiden_communities = result.single()["num_communities"]
                return self._number_of_leiden_communities
            except Exception as e:
                logger.warning("Leiden communities have not been detected yet")
                

    @staticmethod
    def _create_document_node(tx: ManagedTransaction, doc: ProcessedDocument):
        query = """
            CREATE (d:Document {
                filename: $filename,
                document_version: $document_version
            })
        """
        try:
            tx.run(
                query, 
                filename=doc.filename, 
                document_version=doc.document_version, 
                metadata=doc.metadata, 
            )
            logger.info(f"Document node created for file: {doc.filename}")
        except Exception as e:
            logger.warning(f"Error creating Document node for file: {doc.filename}: {e}")


    @staticmethod
    def _create_part_of_relationships(tx: ManagedTransaction, filename: str, document_version: int):
        query = """
            MATCH (d:Document {filename: $filename, document_version: $document_version})
            MATCH (c:Chunk {filename: $filename, document_version: $document_version})
            MERGE (c)-[:PART_OF]->(d)
        """
        try:
            tx.run(query, filename=filename, document_version=document_version)
            logger.info(f"PART_OF relationships created for Document {filename} version {document_version}")
        except Exception as e:
            logger.warning(f"Error creating PART_OF relationships for Document {filename}: {e}")
            

    @staticmethod
    def _create_next_relationships(
        tx: ManagedTransaction, 
        filename: str, 
        document_version: int
        ):
        query = """
            MATCH (c1:Chunk {filename: $filename, document_version: $document_version})
            WITH c1
            MATCH (c2:Chunk {filename: $filename, document_version: $document_version, chunk_id: c1.chunk_id + 1})
            MERGE (c1)-[:NEXT]->(c2)
        """
        try:
            tx.run(query, filename=filename, document_version=document_version)
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {filename}: {e}")


    @staticmethod
    def _create_mentions_relationships(
        tx: ManagedTransaction, 
        node_id: str,
        chunk_id: int,
        filename: str,
        document_version: int
        ):
        query = """
            MATCH (c:Chunk {chunk_id: $chunk_id, filename: $filename, document_version: $document_version})
            MATCH (e:__Entity__ {id: $node_id})
            MERGE (c)-[:MENTIONS]->(e)
        """
        try:
            tx.run(
                query, 
                node_id=node_id, 
                chunk_id=chunk_id, 
                filename=filename, 
                document_version=document_version
            )
        except Exception as e:
            logger.warning(f"Error creating MENTIONS relationships for {node_id}: {e}")
            
            
    @staticmethod
    def _fetch_communities(tx: ManagedTransaction, comm_type: str="leiden"): 
        query = f""" 
            MATCH (n)-[r]-(m)  
            WHERE n.community_{comm_type} IS NOT NULL
            OPTIONAL MATCH (chunk:Chunk) WHERE chunk.community_{comm_type} = n.community_{comm_type}  
            WITH 
                '{comm_type}' AS community_type, 
                n.community_{comm_type} AS community_id, 
                count(DISTINCT n) AS community_size, 
                collect(DISTINCT elementId(n)) AS entity_ids,
                collect(DISTINCT n.name) AS names,
                collect(DISTINCT elementId(r)) AS relationship_ids,  
                collect(DISTINCT type(r)) AS relationship_types,
                collect(DISTINCT elementId(chunk)) AS chunk_ids    
            RETURN 
                community_type, 
                community_id, 
                community_size, 
                entity_ids, 
                names, 
                relationship_ids, 
                relationship_types,
                chunk_ids
            ORDER BY community_size DESC
        """
        return list(tx.run(query))
    
    
    @staticmethod
    def _fetch_chunk(tx: ManagedTransaction, element_id: str):
        query = f"""
            MATCH (c:Chunk) 
            WHERE elementId(c) = "{element_id}"
            RETURN elementId(c) AS chunk_id, c.text AS text
        """
        return list(tx.run(query))
        

    def index_exists(self) -> bool:
        dimensions, index_ent_type = self.vector_store.retrieve_existing_index()
        if not dimensions:
            return False
        else:
            return True      
    
    
    def create_index(self) -> bool:
        try:
            self.vector_store.create_new_index()
            return True
        except:
            return False
    

    def create_document_node(self, doc: ProcessedDocument):
        """
        Creates a Document node in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_document_node, 
                doc
            )
            session.execute_write(
                self._create_part_of_relationships, 
                doc.filename, 
                doc.document_version
            )
            logger.info(f"Document node created for file: {doc.filename}")

    
    def create_next_relationships(self, filename: str, doc_version: int):
        """
        Creates NEXT relationships between Chunk Nodes from a Document.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_next_relationships, 
                filename,
                doc_version
            )
            logger.info(f"NEXT relationships created for Document {filename} version {doc_version}")


    def create_mentions_relationships(
            self, 
            node_id: str,
            chunk_id: int,
            filename: str,
            document_version: int
        ):
        """ Creates MENTIONS relationships between Chunk and __Entity__ nodes. """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_mentions_relationships, 
                node_id,
                chunk_id,
                filename,
                document_version
            )
            logger.info(f"MENTIONS relationships created!")


    def store_chunks_for_doc(self, doc: ProcessedDocument):
        """
        Stores Chunk nodes for a `ProcessedDocument` into the Knowledge Graph and updates the
        Knowledge Graph itself with the graphs extracted from each chunk, if any.
        """
        
        for chunk in doc.chunks:
            
            # doc level metadata
            if doc.metadata: 
                metadata = doc.metadata
            else: 
                metadata = {}
            metadata["filename"] = doc.filename
            metadata["document_version"] = doc.document_version
            # chunk level metadata
            metadata["chunk_id"] = chunk.chunk_id
            metadata["chunk_size"] = chunk.chunk_size
            metadata["chunk_overlap"] = chunk.chunk_overlap
            metadata["embeddings_model"] = chunk.embeddings_model

            try:
                self.vector_store.add_embeddings(
                    texts=[chunk.text],
                    embeddings=chunk.embedding,
                    metadatas=[metadata]
                )
            except Exception as e:
                logger.warning(f"Error storing chunk for document {doc.filename}: {e}")

            # store chunk's graph
            if chunk.nodes is not None :

                graph_doc: GraphDocument = GraphDocument(
                    nodes=chunk.nodes,
                    relationships=chunk.relationships if chunk.relationships is not None else [],
                    source=Document(
                        page_content=chunk.text
                    )
                )

                try:
                    self.add_graph_documents(
                        graph_documents=[graph_doc], 
                        include_source=False,
                        baseEntityLabel=True
                    )

                    for node in chunk.nodes:
                        self.create_mentions_relationships(
                            node_id=node.id, 
                            chunk_id=chunk.chunk_id, 
                            filename=doc.filename, 
                            document_version=doc.document_version
                        )
                except Exception as e:
                    logger.warning(f"Error storing graph for chunk {chunk.chunk_id} in document {doc.filename}: {e}")

        try:
            self.create_next_relationships(
                filename=doc.filename,
                doc_version=doc.document_version
            )
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {doc.filename}: {e}")

        try: 
            self.create_document_node(doc=doc)
        except Exception as e:
            logger.warning(f"Error creating Document source node for file: {doc.filename}: {e}")

        try:
            self.vector_store.create_new_index()
        except Exception as e:
            logger.warning(f"Error creating Index for chunks: {e}")


    def add_documents(self, docs: List[ProcessedDocument]): 
        for doc in docs:
            self.store_chunks_for_doc(doc)


    def get_digraph(self) -> nx.DiGraph:
        """ 
        Returns the Knowledge Graph under its `networkx.DiGraph` representation.
        """
        query_nodes = """
            MATCH (n)  
            RETURN elementId(n) AS node_id, labels(n) AS labels, properties(n) AS properties;
        """

        query_rels = """
            MATCH (n)-[r]->(m)  
            RETURN elementId(n) AS source, elementId(m) AS target, type(r) AS rel_type, properties(r) AS properties;
        """

        G = nx.DiGraph()
        
        with self._driver.session() as session:
            
            nodes = session.run(query_nodes)
            for record in nodes:
                G.add_node(record["node_id"], labels=record["labels"], **record["properties"])

            relationships = session.run(query_rels)
            for record in relationships:
                G.add_edge(record["source"], record["target"], type=record["rel_type"], **record["properties"])

        logger.info(f"DiGraph with {len(G.nodes)} nodes and {len(G.edges)} relationships")  

        return G
    
    
    def update_properties(
        self, 
        G: Optional[nx.DiGraph] = None, 
        centralities: bool=False,
        leiden_communities: bool=False, 
        louvain_communities: bool=False, 
        leiden_modularity: Optional[float] = None,
        louvain_modularity: Optional[float] = None, 
        ):
        """Update Neo4j nodes with Leiden/Louvain communities and centrality scores"""
        with self._driver.session() as session:
            
            if any([centralities, leiden_communities, louvain_communities]) == True: 
                
                for node, data in G.nodes(data=True):
                    
                    query, params = build_update_query(
                        node_id=node, 
                        centralities=centralities,
                        leiden_communities=leiden_communities,
                        louvain_communities=louvain_communities,
                        community_leiden=int(data.get("community_leiden", -1)),
                        community_louvain=int(data.get("community_louvain", -1)),
                        pagerank=float(data.get("pagerank", 0.0)), 
                        betweenness=float(data.get("betweenness", 0.0)),
                        closeness=float(data.get("closeness", 0.0))
                    )
                    try:
                        session.run(query, params)
                    except Exception as e:
                        logger.warning(f"Update Query failed for node_id: {node}")
                
                logger.info("Updated nodes properties in Graph")  
                
            if leiden_modularity is not None: 
                update_modularity(session, leiden_modularity, "leiden")
                logger.info("Updated Leiden Modularity property in Graph")  
                
            if louvain_modularity is not None:
                update_modularity(session, louvain_modularity, "louvain")
                logger.info("Updated Louvain Modularity property in Graph")  
                
    
    def update_centralities_and_communities(self):
        """ 
        Computes centralities measures and detects communities in nodes across the Knowledge Graph. 
        """

        lv = False
        louvain_mod = None
        ld = False
        leiden_mod = None
        centralities = False

        G = self.get_digraph()

        try: 
            G, louvain_mod = detect_louvain_communities(G, return_modularity=True)
            lv = True
        except Exception as e:
            logger.warning(f"Something went wrong detecting Louvain Communities: {e}")
        
        try:
            G, leiden_mod = detect_leiden_communities(G, return_modularity=True)
            ld = True
        except Exception as e:
            logger.warning(f"Something went wrong detecting Leiden Communities: {e}")

        try:
            G = compute_centralities(G)
            centralities = True
        except Exception as e:
            logger.warning(f"Something went wrong computing Centralities degrees on graph: {e}")
        
        try:
            self.update_properties(G, centralities, ld, lv, leiden_mod, louvain_mod)
        except Exception as e:
            logger.warning(f"Something went wrong while updating properties on graph nodes: {e}")


    def predict_links(
        self,
        method: str = "adamic_adar",
        top_k: int = 10,
        entity_only: bool = True
    ) -> List[Tuple[str, str, float]]:
        """
        Predicts missing or future links between entities in the Knowledge Graph.
        
        Uses baseline heuristic methods to identify potential relationships between
        nodes that are not currently connected.
        
        Parameters:
        -----------
        method : str
            Link prediction algorithm to use. Options:
            - 'common_neighbors': Count of shared neighbors
            - 'jaccard': Jaccard similarity of neighborhoods  
            - 'adamic_adar': Adamic-Adar index (default, weights rare connections higher)
            - 'preferential_attachment': Product of node degrees (rich-get-richer)
        top_k : int
            Number of top-ranked predictions to return (default: 10)
        entity_only : bool
            If True, only predict links between __Entity__ nodes, excluding
            Document, Chunk, and other metadata nodes (default: True)
            
        Returns:
        --------
        List[Tuple[str, str, float]]
            List of (source_id, target_id, score) tuples, ranked by prediction score
            
        Example:
        --------
        >>> predictions = kg.predict_links(method="adamic_adar", top_k=10)
        >>> for src, tgt, score in predictions:
        ...     print(f"{src} -> {tgt}: {score:.4f}")
        """
        from src.graph.link_prediction import predict_links as lp_predict_links
        
        # Get NetworkX representation of the graph
        G = self.get_digraph()
        
        # Filter to entity nodes if requested
        node_filter = None
        if entity_only:
            # Only consider nodes with __Entity__ label
            entity_nodes = {
                node_id for node_id, data in G.nodes(data=True)
                if BASE_ENTITY_LABEL in data.get("labels", [])
            }
            node_filter = entity_nodes
            logger.info(f"Filtering to {len(entity_nodes)} entity nodes for link prediction")
        
        # Run link prediction
        predictions = lp_predict_links(
            G=G,
            method=method,
            top_k=top_k,
            exclude_existing=True,
            node_filter=node_filter
        )
        
        logger.info(f"Generated {len(predictions)} link predictions using {method} method")
        
        return predictions



    def get_communities(self, comm_type: str = "leiden") -> List[Community]:
        """ 
        Fetches communities from the Knowledge Graph 
        """
        
        if comm_type in ["leiden", "louvain"]:
            
            communities = []
            
            with self._driver.session() as session:
                
                try:
                    results = session.execute_read(self._fetch_communities) 
                except Exception as e:
                    logger.warning(f"Issue fetching communities for type {comm_type}: {e}")
                    
                for r in results: 
                    
                    if r['names'] not in [["leiden_modularity"], ["louvain_modularity"]]: # avoid GraphMetric
                        
                        comm = Community(
                            community_type=comm_type, 
                            community_id=r["community_id"], 
                            community_size=r["community_size"],
                            entity_ids=r["entity_ids"],
                            entity_names=r["names"],
                            relationship_ids=r["relationship_ids"],
                            relationship_types=r["relationship_types"]
                        )
                        
                        # add chunks to community
                        comm.chunks = []
                        if len(r["chunk_ids"]) > 0:
                            for id in r["chunk_ids"]:
                                try: 
                                    c_res = session.execute_read(self._fetch_chunk, element_id=id)
                                    
                                    comm.chunks.append(Chunk(chunk_id=c_res[0]["chunk_id"], text=c_res[0]["text"]))
                                except Exception as e:
                                    logger.warning(f"Issue fetching chunk with elementId {id}: {e}")
                        
                        communities.append(comm)
                
                return communities
        
        else:
            raise NotImplementedError("This Community type has not been implemented.")  
        
        
    def store_community_reports(self, reports: List[CommunityReport]):
        """ 
        Stores Community Reports in the Graph, to make them available for GraphRAG strategies.
        """
        
        for report in reports:
            
            metadatas = {
                "community_type": report.communtiy_type,
                "community_id": report.community_id,
                "community_size": report.community_size
            }
            
            try:
                self.cr_store.add_embeddings(
                    texts=[report.summary],
                    embeddings=[report.summary_embeddings],
                    metadatas=[metadatas]
                )
            except Exception as e:
                logger.warning(f"Error saving Community Report: {e}")
                
        try:
            self.cr_store.create_new_index()
        except Exception as e:
            logger.warning(f"Error creating Index for CommunityReports: {e}")       