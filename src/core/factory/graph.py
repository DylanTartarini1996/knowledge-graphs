from langchain_core.embeddings import Embeddings

from src.config import KnowledgeGraphConfig
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_knowledge_graph(kg_conf: KnowledgeGraphConfig, embeddings: Embeddings):
    try:

        kg = KnowledgeGraph(
            conf=kg_conf, 
            embeddings_model=embeddings,
        )
        logger.info(f"✅ Intialized Knowledge Graph {kg_conf.database} with index {kg_conf.index_name}")
        return kg
    
    except Exception as e:
        logger.error(f"❌ Error while initializing Knowledge Graph interface: {e}")

    

