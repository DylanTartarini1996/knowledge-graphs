import os
from typing import List, Tuple

from src.config import Source
from src.core.ingestion.ingestor import Ingestor


class LocalIngestor(Ingestor):
    """
    `Ingestor` instance specialized in retrieving documents from 
    a local folder
    """
    def __init__(self, source: Source):
        self.folder = source.folder 


    def list_files(self) -> List[str]:
        files = []
        for folderpath, _, filenames in os.walk(self.folder):
            files.extend(os.path.join(folderpath, x) for x in filenames)
        return files
    
    
    def file_preparation(self, filepath: str) -> Tuple[str, dict]:
        """
        Retrieve the folder name
        Parameter
        - filepath: 
        Return
        - tuple(str, dict): contains filepath (str) and metadata (dict).
        """
        if foldername:=os.path.basename(os.path.dirname(filepath)):
            folder = foldername.lower()
        else:
            folder = None
        return filepath, {'folder': folder}