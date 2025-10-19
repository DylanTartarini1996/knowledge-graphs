from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings

from src.config import EmbedderConf
from src.utils.logger import get_logger


logger = get_logger(__name__)


# TODO CONDITIONAL IMPORTS HERE

def get_embeddings(conf: EmbedderConf) -> Embeddings:

        if conf.type == "ollama":
            embeddings = OllamaEmbeddings(
                model=conf.model
            )
        elif conf.type == "openai":
            embeddings = OpenAIEmbeddings(
                model=conf.model,
                api_key=conf.api_key,
                deployment=conf.deployment,
            )
        elif conf.type == "azure-openai":
            embeddings = AzureOpenAIEmbeddings(
                model=conf.model, 
                azure_endpoint=conf.endpoint,
                azure_deployment=conf.deployment,
                dimensions=1536,
                api_key=conf.api_key,
                api_version=conf.api_version
                
            )
        elif conf.type == "trf":
            embeddings = HuggingFaceEmbeddings(
                model=conf.model,
                endpoint=conf.endpoint,
            )
        else: 
            logger.warning(f"Embedder type '{conf.type}' not supported.")
            embeddings = None

        return embeddings