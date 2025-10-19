from typing import AsyncGenerator

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker,AsyncSession

from src.config import DBConfig
from src.api.models import Base 
from src.utils.logger import get_logger


load_dotenv("config.env")
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


    def get_db_url(self)-> str:
        """Generate database URL with proper URL encoding and validation"""
        
        # Validate required parameters
        if not all([self.user, self.password, self.host, self.port, self.database]):
            logger.error("❌ Invalid DB configuration")
            
        # URL encode the password and username to handle special characters
        encoded_user = quote_plus(self.user)
        encoded_password = quote_plus(self.password)
        
        try:
            # Validate port is a valid integer
            port = int(self.port)
            if port <= 0 or port > 65535:
                raise ValueError("Port must be between 1 and 65535")
        except ValueError as e:
            logger.error(f"❌ Invalid DB port: {port}")
            
        return f"mysql+aiomysql://{encoded_user}:{encoded_password}@{self.host}:{port}/{self.database}" # TODO change to PostGre



db_config = DBConfig()
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