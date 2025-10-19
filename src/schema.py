from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_neo4j.graphs.graph_document import Node, Relationship


class StatusEnum(str, Enum):
    QUEUED = "queued"
    INGESTED = "ingested"
    KO = "ko"


class ChatMode(Enum):
    """ Modalities and answering methods available for the LLM / Knowledge Graph combo. """
    SIMILARITY = "similarity-search"
    CYPHER = "text-to-cypher"
    COMBINE = "combine"
    COMMUNITIES = "communities"
    SUBGRAPH = "subgraph"


class Chunk(BaseModel):
    chunk_id: int
    text: str
    filename: Optional[str] = None
    embedding: Optional[List[float]] = None
    chunk_size: int=1000
    chunk_overlap: int=100
    embeddings_model: Optional[str] = None
    nodes: Optional[List[Node]] = None
    relationships: Optional[List[Relationship]] = None


class ProcessedDocument(BaseModel):
    filename: str = ""
    source: str = ""
    document_version: int = 1
    metadata: Optional[dict] = None
    chunks: Optional[List[Chunk]] = None


class File(BaseModel):
    filename: str
    source: str
    task_id: str
    status: str = StatusEnum.QUEUED.value


class Message(BaseModel):
    message_id: Optional[int] = 1
    user_input: Optional[str] = None
    bot_answer: Optional[str] = None
    chat_mode: Optional[ChatMode] = None
    sources: Optional[List[Chunk]] = Field(description="List of chunks used to answer to the user input", default=None)

class ChatHistory(BaseModel):
    chat_id: str
    user_id: str
    messages: List[Message] = Field(description="List of messages between the user and the AI", default=list())