from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from src.utils.logger import get_logger

from src.config import LLMConf

logger = get_logger(__name__)

# TODO CONDITIONAL IMPORTS HERE

def fetch_llm(conf: LLMConf) -> BaseChatModel | None:
    """
    Fetches the LLM model.
    """
    logger.info(f"Fetching LLM model '{conf.model}'..")

    if conf.type == "ollama":
        llm = ChatOllama(
            model=conf.model,
            temperature=conf.temperature
        )
    elif conf.type == "openai":
        llm = ChatOpenAI(
            model=conf.model,
            api_key=conf.api_key,
            deployment=conf.deployment,
            temperature=conf.temperature,
        )
    elif conf.type == "azure-openai":
        llm = AzureChatOpenAI(
            model=conf.model,
            azure_endpoint=conf.endpoint,
            azure_deployment=conf.deployment,
            api_key=conf.api_key,
            temperature=conf.temperature,
            api_version=conf.api_version
        )
    elif conf.type == "groq":
        llm = ChatGroq(
            model=conf.model, 
            api_key=conf.api_key,
            temperature=conf.temperature,
            max_retries=3
        )
    elif conf.type == "google":
        llm = ChatGoogleGenerativeAI(
            model=conf.model,
            api_key=conf.api_key,
            temperature=conf.temperature,
        )
    elif conf.type == "trf":
        llm = ChatHuggingFace(
            model=conf.model,
            endpoint=conf.endpoint,
            temperature=conf.temperature
        )
    else:
        logger.warning(f"LLM type '{conf.type}' not supported.")
        llm = None
    
    logger.info(f"Initialized LLM of type: '{conf.type}'")
    return llm 