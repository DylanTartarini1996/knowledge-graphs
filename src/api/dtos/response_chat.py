from typing import List, Optional
from pydantic import BaseModel, Field

from src.schema import ChatMode, Chunk

class ChatResponseDto(BaseModel):
    chat_id: str
    message_id: int
    user_input: str
    user_id: str
    bot_answer: Optional[str] = None
    sources: Optional[List[Chunk]] = Field(description="List of chunks used to answer to the user input", default=None)
    chat_mode: Optional[ChatMode] = None