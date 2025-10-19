import json
from src.utils.logger import get_logger

from enum import Enum
from pydantic import BaseModel
from typing import Optional

from src.core.graph.graph_model import Ontology


logger = get_logger(__name__)


class ModelType(str, Enum):
    """
    Type of embedders available in the toolkit
    """
    AZURE_OPENAI = "azure-openai"
    GOOGLE = "google"
    GROQ = "groq"
    OPENAI = "openai"
    OLLAMA = "ollama"
    TRANSFORMERS = "trf"



class ChunkerType(str, Enum):
    """
    Type of chunkers available in the toolkit
    """
    RECURSIVE = "recursive"
    # PLAIN = "plain" # TODO implement plain chunker
    # SEMANTIC = "semantic" # TODO implement SEMANTIC chunker


class Source(BaseModel):
    folder: str
    # TODO add specific source configurations


class ChunkerConf(BaseModel):
    type: ChunkerType = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 100


class LLMConf(BaseModel):
    """
    Configuration for an LLM
    -----------
    attributes:
    -----------
    `type`: LLM `ModelType` 
    `temperature`: LLM temperature param
    `deployment`: represents the name of the deployment.
    `model`: represents the name of the model
    `api_key`: reference to the OpenAI (or Groq, or Azure OpenAI) API key, if any
    `endpoint`: reference to the endpoint of the model, if any
    """
    model: str
    temperature: float = 0.0
    type: ModelType="openai"
    deployment: Optional[str]=None
    api_key: Optional[str]=None
    endpoint: Optional[str]=None
    api_version: Optional[str] = None


class EmbedderConf(BaseModel):
    """
    Embeddings model configuration  

    -----------
    attributes:
    -----------
    `type`: LLM `ModelType` 
    `deployment`: represents the name of the deployment.
    `model`: represents the name of the model
    `api_key`: reference to the OpenAI (or Azure OpenAI) API key, if any
    `endpoint`: reference to the endpoint of the model, if any
    """
    type: ModelType = "openai"
    model: Optional[str] = "text-embedding-ada-002"
    deployment: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None


class DBConfig(BaseModel):
    """ 
    Configuration for the backend Database used for memory management.

    -----------
    attributes:
    -----------
    `password`: `str`
    `host`: `str`
    `port`: `int`
    `user`: `str`
    `password`: `str`
    `database`: `str`
    `timeout`: `int`
    `url`: `str`
    """
    password: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    database: Optional[str] = None
    timeout: int=5000
    url: Optional[str] = None


class KnowledgeGraphConfig(BaseModel):
    """
    Configuration for the backend Database for the Knowledge Base.  
    It might come with an Ontology, a pre-estabilished set of allowed relationships
    and node labels.

    -----------
    attributes:
    -----------
    `schema`: `str`
    `host_name`: `str`
    `port`: `int`
    `user`: `str`
    `password`: `str`
    `database`: `str`
    `index_name`: `str`
    `timeout`: `int`
    `ontology`: `Ontology`
    `uri`: `str`
    """
    password: str
    db_schema : Optional[str] = None
    host_name: Optional[str] = None
    port:  Optional[int] = None
    user: Optional[str] = None
    database: Optional[str] = None
    index_name: str = "vector"
    timeout: int=5000
    ontology: Optional[Ontology] = None
    uri: Optional[str] = None


class Configuration(BaseModel):
    """
    Configuration for the Knowledge Base Project. 
    This will include configurations for the backend (the Graph DB of choice) as well as 
    configurations for users and for models in charge of producing and deleting entities in the KB.

    -----------
    attributes:
    -----------
    `graph_database`: configuration to access the Graph Database
    `rel_database`: configuration to access the Relational Database
    `document_source`: configuration storing informations on where to fetch documents from
    `re_model_conf`: configuration for the LLM in charge of extracting relationships from documents
    `embedder_conf`: configuration for the Embeddings model that will create vectors out of documents
    `summarizer_conf`: configuration for the LLM in charge of summarizing communities out of Chunks and other nodes
    `qa_model`: configuration for the Q&A model (LLM) that will interact with the user
    """
    graph_database: KnowledgeGraphConfig
    rel_database: DBConfig
    chunker_conf: Optional[ChunkerConf] = None
    source_conf: Optional[Source] = None
    re_model_conf: Optional[LLMConf] = None
    embedder_conf: Optional[EmbedderConf] = None
    summarizer_conf: Optional[LLMConf] = None
    qa_model: Optional[LLMConf] = None
    
    
    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            configuration_data = json.load(f)

        configuration = Configuration(**configuration_data)
        logger.info(f"Loaded configuration from {filename}")
        return configuration
