import re
from src.utils.logger import get_logger
from typing import List

from src.schema import ProcessedDocument


logger = get_logger(__name__)


class Cleaner:
    """
    Contains methods to clean the text of a (list of) `ProcessedDocument`.
    """

    def __init__(self):
        pass # TODO implement specific cleaning rules here 


    @staticmethod
    def _clean_text(text: str) -> str:
        # Removes one or more consecutive asterisks (*) from the text.
        text = re.sub(r'\*+', '', text)
        # Handle bullet points converted into 'l'.
        text = re.sub(r'(\n[l])(?=[A-Z])', ' \n*', text)
        # Replace one or more consecutive hyphens (-) followed by a space from the text with a single hyphens.
        text = re.sub(r'-+ ', ' ', text,  re.MULTILINE)
        # Replace one or more consecutive underscores (_) followed by a space from the text with a single underscore.
        text = re.sub(r'_+ ', '_', text)
        # Substitute all types of en dash characters with a standard hyphen (-).
        text = re.sub(r'[\u2013\u2014\u2212]', '-', text)
        # Substitute all types of apostrophes with the straight apostrophe ('), using the most common ASCII representation.
        text = re.sub(r'[\u2019\u02BC\u2032\u02B9\u00B4\u0060]', "'", text) 
        # Removes character that enforces UTF-8 (MOB)
        text = re.sub(r'[\ufeff]', '', text)
        # Regular expression substitution to replace control characters such as [\u0000, \u0001, \u0002] with space.
        text = re.sub(r'[\x00-\x09]', ' ', text)
        # Insert a white space after a number or lowercase letter if followed by an uppercase letter
        text = re.sub(r'([0-9a-z])(?=[A-Z])', r'\1 ', text)
        # Insert a white space after a accented letter if followed by an uppercase letter
        text = re.sub(r'([a-zà-ù])(?=[A-Z])', r'\1 ', text)
        # Replaces sequences of one or more consecutive newline characters followed by optional whitespace characters with a single newline character. This cleaner helps to condense multiple empty lines into a single empty line.
        text = re.sub(r'\n\s*\n', '\n', text, re.MULTILINE)
        # Replaces \nn characters with a space.
        text = re.sub(r'\nn', ' ', text)
        # Replaces newline characters with a space. This effectively removes line breaks by replacing them with spaces.
        text = re.sub(r'\n', ' ', text)
        # Delete useless footer text regarding page number of the current page.
        text = re.sub(r'Pagina (\d+) di (\d+)', ' ', text)
        # Replaces one or more consecutive spaces or tabs with a single space.
        text = re.sub(r'[ \t]+', ' ', text)
        # Removes leading and trailing whitespace from the text.
        text = re.sub(r'^\s+|\s+$', ' ', text)

        return text.strip()


    def clean_document(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Cleans the text of a `ProcessedDocument` instance.
        """
        doc.source = self._clean_text(doc.source)
        return doc
    

    def clean_documents(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Cleans the text of a list of `ProcessedDocument` instances.
        """
        return [self.clean_document(doc) for doc in docs]