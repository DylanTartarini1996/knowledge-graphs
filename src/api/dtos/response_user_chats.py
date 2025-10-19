from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class UserChat(BaseModel):
    chat_id: str
    created_at: str


class UserChatsResponseDto(BaseModel):
    user_id: str
    chats: List[UserChat] = Field(default_factory=list,description="list of reference to a chats available for the user")
