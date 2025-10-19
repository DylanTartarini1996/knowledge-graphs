from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.core.graph.graph_model import Ontology, _Graph
from src.core.prompts.graph_extractor import get_graph_extractor_prompt
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, llm: BaseChatModel, ontology: Optional[Ontology]=None):
        self.llm = llm
        self.prompt = get_graph_extractor_prompt()

        self.prompt.partial_variables = {
            'allowed_labels':ontology.allowed_labels if ontology and ontology.allowed_labels else "", 
            'labels_descriptions': ontology.labels_descriptions if ontology and ontology.labels_descriptions else "", 
            'allowed_relationships': ontology.allowed_relations if ontology and ontology.allowed_relations else ""
        }


    def extract_graph(self, text: str) -> _Graph:
        """ 
        Extracts a graph from a text.
        """

        if self.llm is not None:
            try:
                graph: _Graph = self.llm.with_structured_output(
                    schema=_Graph
                    ).invoke(
                        input=self.prompt.format(input_text=text)
                    )

                return graph 
                
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")