from typing import Optional
from pydantic import BaseModel

from src.schema import ChatMode


class ChatRequestDto(BaseModel):
    chat_id: str
    message_id: int 
    user_input: str
    user_id: str
    chat_mode: Optional[ChatMode] = ChatMode.COMBINE.value
