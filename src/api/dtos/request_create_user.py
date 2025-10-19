from pydantic import BaseModel


class CreateUserRequestDto(BaseModel):
    username: str
    password: str