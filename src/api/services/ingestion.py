from src.config import Configuration
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.core.ingestion.local_ingestor import LocalIngestor
from src.core.ingestion.chunker import Chunker
from src.core.ingestion.cleaner import Cleaner
from src.core.ingestion.embedder import ChunkEmbedder
from src.core.ingestion.graph_miner import GraphMiner
from src.schema import File, ProcessedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IngestionService:
    """ 
    Ingestion service that parses uploaded files, 
    vectorizes them, and adds them to a Knowledge Graph while 
    extracting nodes and relationships from them. 
    """

    def __init__(self, config: Configuration):
        self.ingestor = LocalIngestor(source=config.source_conf)
        self.cleaner = Cleaner()
        self.chunker = Chunker(conf=config.chunker_conf)
        self.embedder = ChunkEmbedder(conf=config.embedder_conf)
        self.graph_miner = GraphMiner(conf=config.re_model_conf, ontology=config.graph_database.ontology)
        self.knowledge_graph = KnowledgeGraph(conf=config.graph_database, embeddings_model=self.embedder.embeddings)


    # TODO connect to a queue of files    
    async def ingest_files(self, task_id: str):
        """ 
        Ingest a batch of files into the Knowledge Graph. 

        -------
        params:
        -------
        - `task_id`: `str`
            Identifies the ingestion task to track the process. 
        """
        # TODO track status of files to be processed
        docs = self.ingestor.batch_ingest()
        docs = self.cleaner.clean_documents(docs)
        docs = self.chunker.chunk_documents(docs)
        docs = self.embedder.embed_documents_chunks(docs)
        docs = self.graph_miner.mine_graph_from_docs(docs=docs)

        # TODO what to do if adding files to the graph fails? Track a checkpoint?
        self.knowledge_graph.add_documents(docs)
        self.knowledge_graph.update_centralities_and_communities()

        # TODO update status of processed files in db

        # TODO cleanup uploaded files

    async def update_task_queue(self, task_id: str):
        
        pass

