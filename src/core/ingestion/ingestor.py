import os
import magic
from abc import abstractmethod
from typing import List, Tuple, Optional, Dict, Any

from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PDFPlumberLoader, BSHTMLLoader
from src.utils.logger import get_logger

from src.config import Source
from src.schema import ProcessedDocument

logger = get_logger(__name__)

MIME_TYPE_MAPPING = {
    'application/pdf': PDFPlumberLoader,
    'text/plain': TextLoader,
    'text/html': BSHTMLLoader,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': Docx2txtLoader
}


class Ingestor:
    """ 
    Base `Ingestor` Class with common methods. 
    Can be specialized by source.
    """ 
    def ___init__(self, source: Source):
        self.source = source
    
    @abstractmethod
    def list_files(self)-> List[str]:
        pass


    @abstractmethod
    def file_preparation(self, file) -> Tuple[str, dict]:
        pass

    
    @staticmethod
    def load_file(filepath: str, metadata: dict) -> List[Document]:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(filepath) or metadata.get('Content-Type')
        if mime_type == 'inode/x-empty':
            return []

        loader_class = MIME_TYPE_MAPPING.get(mime_type)
        if not loader_class:
            logger.warning(f'Unsupported MIME type: {mime_type} for file {filepath}, skipping.')
            return []
        
        if loader_class == PDFPlumberLoader:
            loader = loader_class(
                file_path=filepath,
                extract_images=False,
            )
        elif loader_class == Docx2txtLoader:
            loader = loader_class(
                file_path=filepath
            )
        elif loader_class == TextLoader:
            loader = loader_class(
                file_path=filepath
            )
        elif loader_class == BSHTMLLoader:
            loader = loader_class(
                file_path=filepath,
                open_encoding="utf-8",
            )
        try: 
            return loader.load()
        except Exception as e:
            logger.warning(f"Error loading file: {filepath} with exception: {e}")   
            pass 



    @staticmethod
    def merge_pages(pages: List[Document]) -> str:
        return "\n\n".join(page.page_content for page in pages)


    @staticmethod
    def create_processed_document(file: str, document_content: str, metadata: dict):
        processed_doc = ProcessedDocument(filename=file, source=document_content, metadata=metadata)
        return processed_doc

    
    def ingest(self, filename: str, metadata: Dict[str, Any]) -> ProcessedDocument | None:
        """ 
        Loads a file from a path and turn it into a `ProcessedDocument`
        """

        base_name = os.path.basename(filename)

        document_pages = self.load_file(filename, metadata)

        try: 
            document_content = self.merge_pages(document_pages)
        except(TypeError):
            logger.warning(f"Empty document {filename}, skipping..")
        
        if document_content is not None:
            processed_doc = self.create_processed_document(
                base_name, 
                document_content, 
                metadata
            )
            return processed_doc
        
    
    def batch_ingest(self) -> List[ProcessedDocument]:
        """
        Ingests all files in a folder
        """
        processed_documents = []
        for file in self.list_files():
            file, metadata = self.file_preparation(file)
            processed_doc = self.ingest(file, metadata)
            if processed_doc:
                processed_documents.append(processed_doc)
        return processed_documents