import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Tuple

from src.api.services.session import ChatSessionManager
from src.config import LLMConf
from src.core.agents.graph_qa import GraphAgentResponder
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.schema import ChatMode, Chunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ChatService:
    """ 
    Chat Service powered by LLMs, capable of holding a conversation
    using graph-rag techniques for knowledge retrieval and track history 
    of the conversation thanks to its built-in memory system. 

    * memory of the conversation is hosted in a relational database 
    * knowledge is stored in a graph database (Knowledge Graph)
    """

    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        session: AsyncSession, 
        qa_llm_conf: Optional[LLMConf]=None, 
        cypher_llm_conf: Optional[LLMConf]=None,
        rephrase_llm_conf: Optional[LLMConf]=None,
        ):

        self.session_manager = ChatSessionManager(session=session)

        self.responder = GraphAgentResponder(
            graph=knowledge_graph,
            qa_llm_conf=qa_llm_conf,
            cypher_llm_conf=cypher_llm_conf,
            rephrase_llm_conf=rephrase_llm_conf
        )


    async def chat(
        self, 
        user_id: str, 
        chat_id: str, 
        user_message: str, 
        chat_mode: Optional[ChatMode]=ChatMode.COMBINE.value
        ) -> Tuple[str, List[Chunk]]:
        """ 
        Interfaces with a user to answer his/her queries
        """
        chat_history = await self.session_manager.fetch_history(chat_id, user_id)

        bot_answer, chunks = self.responder.get_answer(
            query=user_message, 
            chat_mode=chat_mode, 
            chat_history=chat_history.messages
        )

        asyncio.create_task(
            self.session_manager.log_history(
                chat_id=chat_id, 
                user_id=user_id, 
                user_message=user_message,
                answer_to_user=bot_answer, 
                sources=chunks,
                chat_mode=chat_mode
            )
        )

        return bot_answer, chunks
    
    