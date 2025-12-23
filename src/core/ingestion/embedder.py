from typing import List

from src.config import EmbedderConf
from src.factory.embeddings import get_embeddings
from src.schema import ProcessedDocument
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ChunkEmbedder:
    """ Contains methods to embed Chunks from a (list of) `ProcessedDocument`."""
    def __init__(self, conf: EmbedderConf):
        self.conf = conf
        self.embeddings = get_embeddings(conf)

        if self.embeddings:
            logger.info(f"Embedder of type '{self.conf.type}' initialized.")
    

    def embed_document_chunks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Embeds the chunks of a `ProcessedDocument` instance.
        """
        if self.embeddings is not None:
            for chunk in doc.chunks:
                chunk.embedding = self.embeddings.embed_documents([chunk.text])
                chunk.embeddings_model = self.conf.model
            logger.info(f"Embedded {len(doc.chunks)} chunks.")
            return doc
        else: 
            logger.warning(f"Embedder type '{self.conf.type}' is not yet implemented")


    def embed_documents_chunks(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Embeds the chunks of a list of `ProcessedDocument` instances.
        """
        if self.embeddings is not None:
            for doc in docs:
                doc = self.embed_document_chunks(doc)
            return docs
        else: 
            logger.warning(f"Embedder type '{self.conf.type}' is not yet implemented")
            return docs