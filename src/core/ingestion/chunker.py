from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from src.utils.logger import get_logger

from src.config import ChunkerConf
from src.schema import ProcessedDocument, Chunk

logger = get_logger(__name__)


class Chunker:
    """
    Contains methods to chunk the text of a (list of) `ProcessedDocument`.
    """

    def __init__(self, conf: ChunkerConf):
        self.chunker_type = conf.type

        if self.chunker_type == "recursive":

            self.chunk_size = conf.chunk_size
            self.chunk_overlap = conf.chunk_overlap

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap, 
                is_separator_regex=False
            )
        
        else: 
            logger.warning(f"Chunker type '{self.chunker_type}' not supported.")


    def _chunk_document(self, text: str) -> list[str]:
        """Chunks the document and returns a list of chunks."""
        return self.splitter.split_text(text)


    def get_chunked_document_with_ids(
        self, 
        text: str, 
        ) -> list[dict]:
        """Chunks the document and returns a list of dictionaries with chunk ids and chunk text."""
        return [
            {
                "chunk_id": i + 1,
                "text": chunk,
                "chunk_size": self.chunk_size, 
                "chunk_overlap": self.chunk_overlap
            }
            for i, chunk in enumerate(self._chunk_document(text))
        ]
    

    def chunk_document(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Chunks the text of a `ProcessedDocument` instance.
        """
        chunks_dict = self.get_chunked_document_with_ids(doc.source)
        
        doc.chunks = [Chunk(**chunk) for chunk in chunks_dict]

        logger.info(f"DOcument {doc.filename} has been chunked into {len(doc.chunks)} chunks.")
        
        return doc

    
    def chunk_documents(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Chunks the text of a list of `ProcessedDocument` instances.
        """
        updated_docs = []
        for doc in docs:
            updated_docs.append(self.chunk_document(doc))
        return updated_docs
