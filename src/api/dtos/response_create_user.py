from pydantic import BaseModel


class CreateUserResponseDto(BaseModel):
    id: int
    username: str
