from src.config import KnowledgeGraphConfig, EmbedderConf
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.factory.embeddings import get_embeddings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_knowledge_graph(kg_conf: KnowledgeGraphConfig, embedder_conf: EmbedderConf):
    try:

        embeddings = get_embeddings(embedder_conf)
        
        kg = KnowledgeGraph(
            conf=kg_conf, 
            embeddings_model=embeddings
        )
        
        logger.info(f"✅ Intialized Knowledge Graph {kg_conf.database} with index {kg_conf.index_name}")
        
        return kg
    
    except Exception as e:
        logger.error(f"❌ Error while initializing Knowledge Graph interface: {e}")

    

