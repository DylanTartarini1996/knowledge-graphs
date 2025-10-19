from typing import List
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.exceptions.api import ApiException
from src.api.models.users import User
from src.utils.logger import get_logger

logger = get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str):
    return pwd_context.hash(password)


class UserService:
    """ 
    In charge of interacting with the relational db to perform CRUD operations on users. 
    """

    def __init__(self, session: AsyncSession):
        self.session = session


    async def create_user(self, username: str, password: str):
        hashed_password = get_password_hash(password)
        db_user = User(username=username, hashed_password=hashed_password)
        
        try: 
            self.session.add(db_user)
            self.session.commit()
            self.session.refresh(db_user)

            logger.info(f"✅ Created user {username}")
        
        except Exception as e:
            await self.session.rollback()

            logger.error(f"❌ Error creating user: {e}")
            
            raise ApiException(
                code=400,
                message=f"❌ Error creating user: {e}",
                description="Error creating user. It's possible the username already exists."
            )

        return db_user


    async def delete_user(self, user_id: int):
        user = await self.session.get(User, user_id)
        
        if user:
            await self.session.delete(user)
            await self.session.commit()

            logger.info(f"ℹ️ Deleted user with ID {user_id}")

            return user
        
        logger.error(f"❌ User ID {user_id} not found")
        await self.session.rollback()

        return None


    async def get_all_users(self) -> List[dict]:

        users = []

        try: 
            stmt = select(User)

            results = await self.session.execute(stmt)

            for user in results.scalars():
                users.append(
                    {
                        "user_id": user.id, 
                        "username": user.username,
                        "created_at": user.created_at
                    }
                )

            logger.info(f"ℹ️ Found {len(users)} Users in DB")

        except Exception as e:

            await self.session.rollback()

            raise ApiException(
                message="❌ Error listing users",
                description=str(e.args),
                code=500
            )
        return users
