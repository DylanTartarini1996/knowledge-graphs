from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dtos.request_create_user import CreateUserRequestDto
from src.api.dtos.response_create_user import CreateUserResponseDto
from src.api.dtos.response_users import AllUsersResponseDto
from src.api.exceptions.api import ApiException
from src.api.models.users import User
from src.api.services.users import UserService
from src.config import Configuration
from src.factory.configuration import get_configuration_from_env
from src.factory.database import get_db_async


CONFIGURATION_PATH = "config_example.env"
conf: Configuration = get_configuration_from_env(CONFIGURATION_PATH)

router = APIRouter(prefix="users", tags=["users"])

@router.get(f"/")
async def list_users(db: AsyncSession = Depends(get_db_async(conf.rel_database))):
    us_service = UserService(session=db)

    all_users = await us_service.get_all_users()

    return AllUsersResponseDto(users=all_users)


@router.post("/register", status_code=201)
async def register_user(
    create_user_request: CreateUserRequestDto, 
    db: AsyncSession = Depends(get_db_async(conf.rel_database))
    ) -> CreateUserResponseDto:
    
    us_service = UserService(session=db)

    db_user = await us_service.create_user(db=db, username=create_user_request.username, password=create_user_request.password)
    
    return CreateUserResponseDto(id=db_user.id, username=db_user.username)
    

@router.delete("/{user_id}", status_code=200)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_async(conf.rel_database))
    ):
    
    us_service = UserService(session=db)
    deleted = await us_service.delete_user(db=db, user_id=user_id)
    
    if not deleted:
        raise ApiException(
            code=404, 
            message="User not found",
            detail="User not found"
        )
    
    return {"message": "User deleted successfully"}
