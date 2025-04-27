import os
from dotenv import load_dotenv

import streamlit as st

from src.agents.graph_qa import GraphAgentResponder
from src.config import Configuration, Source, ChunkerConf, LLMConf, EmbedderConf, KnowledgeGraphConfig
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.embedder import ChunkEmbedder

SOURCE_FOLDER = f"{os.getcwd()}/source_docs"

@st.cache_data
def get_configuration_from_env() -> Configuration:
    env = load_dotenv('config.env')
    if env:
        conf = Configuration(
            database=KnowledgeGraphConfig(
                uri=os.getenv("NEO4J_URI"),
                user=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                index_name=os.getenv("INDEX_NAME")
            ),
            source_conf=Source(folder=SOURCE_FOLDER),
            chunker_conf=ChunkerConf(
                type=os.getenv("CHUNKER_TYPE"), 
                chunk_size=os.getenv("CHUNKER_CHUNK_SIZE"), 
                chunk_overlap=os.getenv("CHUNKER_CHUNK_OVERLAP")
            ),
            embedder_conf = EmbedderConf(
                type=os.getenv("EMBEDDINGS_TYPE"),
                model=os.getenv("EMBEDDINGS_MODEL_NAME"),
                api_key=os.getenv("EMBEDDINGS_API_KEY"),
                deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"),
                endpoint=os.getenv("EMBEDDINGS_ENDPOINT"), 
                api_version=os.getenv("EMBEDDINGS_API_VERSION")
            ),
            re_model_conf=LLMConf(
                type=os.getenv("RE_MODEL_TYPE"),
                model=os.getenv("RE_MODEL_NAME"), 
                temperature=os.getenv("RE_MODEL_TEMPERATURE"), 
                deployment=os.getenv("RE_MODEL_DEPLOYMENT"),
                api_key=os.getenv("RE_API_KEY"),
                endpoint=os.getenv("RE_MODEL_ENDPOINT"),
                api_version=os.getenv("RE_MODEL_API_VERSION") or None
            ),
            qa_model=LLMConf(
                type=os.getenv("QA_MODEL_TYPE"),
                model=os.getenv("QA_MODEL_NAME"), 
                temperature=os.getenv("QA_MODEL_TEMPERATURE"), 
                deployment=os.getenv("QA_MODEL_DEPLOYMENT"),
                api_key=os.getenv("QA_API_KEY"),
                endpoint=os.getenv("QA_MODEL_ENDPOINT"),
                api_version=os.getenv("QA_MODEL_API_VERSION") or None
            )
        )
        return conf
    else: 
        st.error("Neither a Configuration file nor an Environment file has been passed!")


@st.cache_resource
def get_knowledge_graph(_conf: Configuration, _embedder: ChunkEmbedder):
    kg = KnowledgeGraph(
        conf=_conf.database, 
        embeddings_model=_embedder.embeddings,
    )
    return kg


@st.cache_resource
def get_embedder(_embedder_conf: EmbedderConf):
    embedder = ChunkEmbedder(conf=_embedder_conf)
    return embedder


@st.cache_resource
def get_responder(_conf: Configuration, _kg: KnowledgeGraph):
    responder = GraphAgentResponder(
        qa_llm_conf=_conf.qa_model,
        cypher_llm_conf=_conf.qa_model,
        graph=_kg
        # rephrase_llm_conf=conf.qa_model
    )
    return responder