from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.exceptions.api import ApiException
from src.api.models.chats import Chat
from src.schema import ChatHistory, ChatMode, Chunk, Message
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatSessionManager:
    """ 
    Class in charge of interacting with the relational database to log, 
    fetch and update with the history of the chat session.
    """
    def __init__(self, session: AsyncSession):
        self.session = session
        
    
    async def list_user_chats(self, user_id: str) -> list:
        """ Returns the list of user's chats in DB for a given user. """
        chats = []
        try:
            stmt = (
                select(Chat)
                .where(Chat.user_id==user_id)
                .order_by(Chat.updated_at)
            )
            result = await self.session.execute(stmt)
            for chat in result.scalars():
                chats.append({
                    "chat_id": chat.chat_id,
                    "created_at": chat.created_at.isoformat() if chat.created_at else ""
                })

            logger.info(f"ℹ️ Found {len(chats)} Chat sessions for user {user_id}")

            return chats
        except Exception as e:
            logger.warning(f"⚠️ Failed retrieving user's chats for user {user_id}: {e}")
            return chats
        
    
    async def fetch_user_chat(self, chat_id: str, user_id: str) -> Chat | None:
        """ Returns a `Chat` corresponding with a given `chat_id` for a give user """
        try: 
            stmt = (
                select(Chat)
                .where(Chat.chat_id==chat_id)
                .where(Chat.user_id==user_id)
            )
            result = await self.session.execute(stmt)
            chat = result.scalars().first()
            logger.info(f"ℹ️ Chat retrieved")
            return chat
        
        except Exception as e: 
            logger.warning(f"⚠️ No user chat retrieved for chat_id: {chat_id}")
            return None
            
        
    async def fetch_history(self, chat_id: str, user_id: str) -> ChatHistory:
        """ Fetches the history of the conversation based on `session_id`. """
        try: 
            
            chat: Chat | None = await self.fetch_user_chat(chat_id, user_id)
            
            if chat is not None: 
                history = ChatHistory(
                    messages=[Message(**msg) for msg in chat.history]
                )
                logger.info(f"ℹ️ Fetched history with {len(history.messages)} messages from DB")
            else: 
                logger.info("ℹ️ No history retrieved")
                history = ChatHistory(messages=[])
                
            return history
        except Exception as e: 
            logger.warning(f"⚠️ Error occurred while retrieving chat by chat_id {chat_id}: {e}")
            
            raise ApiException(
                message=f'❌ Error occurred while retrieving chat by chat_id {chat_id}: {e}',
                description=str(e.args),
                code=500
            )
    
    
    async def update_chat_session(self, chat_session: Chat, updated_history: ChatHistory) -> Chat: 
        """ Updates the row entry in table `Chat` with the new messages interaction """
        try: 
            chat_session.history = [msg.model_dump() for msg in updated_history.messages]
            
            await self.session.commit()
            await self.session.refresh(chat_session)
            
            logger.info("Updated history in DB")
            
            return chat_session
        
        except Exception as e:
            await self.session.rollback()
            raise ApiException(
                message="❌ Error during data update",
                description=str(e.args),
                code=500
            )
            
            
    async def insert_chat_session(self, chat_id: str, user_id: str, history: Optional[ChatHistory]=None) -> Chat:
        """ Creates a new entry in table `Chat` for a given `chat_id` and user """
        
        try: 
            new_chat = Chat(
                chat_id=chat_id, 
                history=history.model_dump()["messages"] if history is not None else [],
                user_id=user_id
            )
            
            self.session.add(new_chat)
            await self.session.commit()
            await self.session.refresh(new_chat)
    
            logger.info("✅ Created Chat Session in DB")
            
            return new_chat
        
        except Exception as e: 
            await self.session.rollback()
            raise ApiException(
                message="❌ Error during data insertion",
                description=str(e.args),
                code=500
               )
    
    @staticmethod
    def parse_history(chat_session: Chat) -> ChatHistory:
        """ Parses the history of a conversation """
        
        history = ChatHistory(
            chat_id=chat_session.chat_id,
            user_id=chat_session.user_id,
            messages=[Message.model_validate(msg) for msg in chat_session.history]
        )

        return history

        
    async def log_history(
        self, 
        chat_id: str, 
        user_id: str, 
        user_message: str, 
        answer_to_user: Optional[str]=None,
        sources: Optional[List[Chunk]]=None,
        chat_mode: Optional[ChatMode]=None
        ) -> Chat:

        """ Logs (or updates) the history of the conversation based on `chat_id` """
        
        chat: Chat | None = await self.fetch_user_chat(chat_id, user_id)
        # TODO review how logging history works

        # NOTE do it from chat service
        if chat is not None:
            chat_history = self.parse_history(chat.history)
            new_message = Message(
                message_id=chat_history.messages[-1].message_id +1,
                user_input=user_message, 
                bot_answer=answer_to_user,
                chat_mode=chat_mode,
                sources=sources
            )
            chat_history.messages.append(new_message)

            updt_chat = await self.update_chat_session(
                chat_session=chat,
                updated_history=chat_history
            ) 

        else:
            new_message = Message(
                user_id=user_id,
                user_input=user_message, 
                bot_answer=answer_to_user,
                chat_mode=chat_mode,
                sources=sources
            )

            chat_history = ChatHistory(
                chat_id=chat_id, 
                user_id=user_id, 
                messages=[new_message]
            )

            updt_chat = await self.insert_chat_session(
                chat_id=chat_id, 
                user_id=user_id,
                chat_history=chat_history
            )
            
        return updt_chat
    
