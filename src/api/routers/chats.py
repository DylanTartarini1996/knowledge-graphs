from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dtos.request_chat import ChatRequestDto
from src.api.dtos.response_chat import ChatResponseDto
from src.api.dtos.response_user_chats import UserChat, UserChatsResponseDto
from src.api.exceptions.api import ApiException
from src.api.factory.database import get_db_async
from src.api.services.chat import ChatService
from src.api.services.session import ChatSessionManager
from src.config import Configuration
from src.core.factory.configuration import get_configuration_from_env
from src.core.factory.graph import get_knowledge_graph
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.schema import ChatHistory
from src.utils.logger import get_logger


logger = get_logger(__name__)

CONFIGURATION_PATH = "config_example.env"
conf: Configuration = get_configuration_from_env(CONFIGURATION_PATH)

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("/")
async def chat(
    chat_request: ChatRequestDto, 
    kg: KnowledgeGraph = Depends(get_knowledge_graph(conf.graph_database)), 
    db: AsyncSession = Depends(get_db_async(conf.rel_database))
    ) -> ChatResponseDto:
    """ 
    Chat with the Knowledge Graph using LLMs. 
    """
    try:
        chat_service = ChatService(
            knowledge_graph=kg, 
            session=db, 
            qa_llm_conf=conf.qa_model,
            cypher_llm_conf=conf.qa_model,
            rephrase_llm_conf=conf.qa_model
        )
        
        answer, chunks = await chat_service.chat(
            user_id=chat_request.user_id,
            chat_id=chat_request.chat_id,
            user_message=chat_request.user_input,
            chat_mode=chat_request.chat_mode
        )

        return ChatResponseDto(
            user_id=chat_request.user_id, 
            chat_id=chat_request.chat_id,
            message_id=chat_request.message_id,
            user_input=chat_request.user_input,
            chat_mode=chat_request.chat_mode,
            bot_answer=answer,
            sources=chunks,
        )
    
    except Exception as e:
        return ApiException(
            message=f"❌ Error chatting with user: {e}",
            description=e,
            code=500
        )


@router.get(f'{{user_id}}', tags=["chats"])
async def get_user_chats(
    user_id: str, 
    db: AsyncSession = Depends(get_db_async(conf.rel_database))
    ) -> UserChatsResponseDto:
    """ Used to retrieve chats for a given user """
    session_manager = ChatSessionManager(session=db)
    user_chats = session_manager.list_user_chats(user_id)
    parsed_user_chats = [UserChat.model_validate(c) for c in user_chats]
    
    return UserChatsResponseDto(user_id=user_id, user_chats=parsed_user_chats)


@router.get(f'{{user_id}}/{{chat_id}}', tags=["chats"])
async def get_chat_history(
    user_id: str, 
    chat_id: str, 
    db: AsyncSession = Depends(get_db_async(conf.rel_database))
    ) -> ChatHistory:
    """ Used to retrieve chat history for a given chat session """
    session_manager = ChatSessionManager(session=db)
    history = await session_manager.fetch_history(chat_id=chat_id, user_id=user_id)
    return history
