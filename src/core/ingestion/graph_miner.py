from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.core.agents.graph_extractor import GraphExtractor
from src.core.graph.graph_model import _Graph, Ontology, map_to_lc_graph
from src.schema import ProcessedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphMiner:
    """ Contains methods to mine graphs from a (list of) `ProcessedDocument`."""

    def __init__(self, llm: BaseChatModel, ontology: Optional[Ontology]=None):
        self.graph_extractor = GraphExtractor(llm=llm, ontology=ontology)

        if self.graph_extractor:
            logger.info(f"✅ GraphMiner initialized.")


    def mine_graph_from_doc_chunks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Mines a graph from a `ProcessedDocument` instance. 
        """
        
        for chunk in doc.chunks:
            try:
                graph: _Graph = self.graph_extractor.extract_graph(chunk.text)

                graph_doc = map_to_lc_graph(graph, source_content=chunk.text)

                chunk.nodes = graph_doc.nodes
                chunk.relationships = graph_doc.relationships
                
            except Exception as e:
                logger.warning(f"Error while mining graph: {e}")

            logger.info(f"✅ Created a graph representation for {len(doc.chunks)} chunks.")
        
        return doc


    def mine_graph_from_docs(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Mines graphs from a list of `ProcessedDocument` instances.
        """
        return [self.mine_graph_from_doc_chunks(doc) for doc in docs]
    