from typing import List
from pydantic import BaseModel, Field


class AllUsersResponseDto(BaseModel):
    users: List[dict] = Field(default_factory=list)