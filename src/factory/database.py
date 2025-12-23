from typing import AsyncGenerator

from contextlib import asynccontextmanager
from urllib.parse import quote_plus
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.config import DBConfig
from src.api.models import Base 
from src.api.models.chats import Chat
from src.api.models.users import User
from src.factory.configuration import get_configuration_from_env
from src.utils.logger import get_logger

CONFIGURATION_PATH = "config_example.env"
logger = get_logger(__name__)


class DatabaseFactory:
    def __init__(self, config: DBConfig):
        self.user = config.user
        self.password = config.password
        self.host = config.host
        self.port = str(config.port)
        self.database = config.database
        
        
        self.engine = create_async_engine(
            self.get_db_url(), 
            echo=False,
            pool_recycle=3600,  # Restart connection after an hour
            pool_pre_ping=True,  # Test connection before usage
            pool_size=50,         # Limit connection pool
            max_overflow=10      # Maximum number of connections
        )
            
        self.session_factory = async_sessionmaker(
                                autocommit=False,
                                autoflush=False,
                                bind=self.engine,
                                expire_on_commit=False
                            )

    async def init_models(self):
        """Initialize database models asynchronously"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


    def get_db_url(self) -> str:
        """Generate database URL with proper URL encoding and validation"""
        
        # 1. Validate required parameters (Ensure none are None)
        if not all([self.user, self.password, self.host, self.port, self.database]):
            logger.error("❌ Invalid DB configuration - one or more fields are missing")
            # Consider raising an exception here to stop execution early
            raise ValueError("Missing database configuration environment variables")

        # 2. URL encode to handle special characters (@, :, /, etc.)
        encoded_user = quote_plus(str(self.user))
        encoded_password = quote_plus(str(self.password))
        
        try:
            # 3. Validate port
            port = int(self.port)
            if not (0 < port <= 65535):
                raise ValueError("Port must be between 1 and 65535")
        except ValueError as e:
            logger.error(f"❌ Invalid DB port: {self.port}")
            raise

        # 4. Return PostgreSQL formatted URL
        url = f"postgresql+asyncpg://{encoded_user}:{encoded_password}@{self.host}:{port}/{self.database}"
        logger.info(f"Connecting to Postgres at {self.host}:{port} using database: {self.database}")
        return url


db_config = get_configuration_from_env(CONFIGURATION_PATH).rel_database
database_factory = DatabaseFactory(db_config)


async def initialize_database():
    """Initialize the database and create all tables"""
    await database_factory.init_models()


@asynccontextmanager
async def get_db_context(db_config: DBConfig):
    global database_factory
    try:
        session = database_factory.session_factory()
        # Check if connection is up
        await session.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Connection error: {e}")
        logger.info("Requesting new password and recreating factory")
        database_factory = DatabaseFactory(db_config)

        try:
            session = database_factory.session_factory()
            # Check if connection is up
            await session.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"❌ Error after connection refresh: {e}")
            raise Exception(message="Unable to connect to database")   
    try:
        yield session
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        await session.rollback()
        raise
    finally:
        await session.close()
        
        
async def get_db_async(db_config: DBConfig) -> AsyncGenerator[AsyncSession, None]:
    async with get_db_context(db_config) as session:
        yield session