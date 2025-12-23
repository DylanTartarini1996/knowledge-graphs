from typing import Annotated, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.factory.database import get_db_async
from src.config import Configuration
from src.factory.configuration import get_configuration_from_env
from src.factory.graph import get_knowledge_graph
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIGURATION_PATH = "config_example.env"
conf: Configuration = get_configuration_from_env(CONFIGURATION_PATH)

router = APIRouter(prefix="/health",tags=["health"])

DBDependencyAsync = Annotated[AsyncSession, Depends(lambda: get_db_async(conf.rel_database))]
KnowledgeGraphDependency = Annotated[Any, Depends(lambda: get_knowledge_graph(conf.graph_database, embedder_conf=conf.embedder_conf))]

@router.get("/", tags=["health"])
async def healthcheck(db_async: DBDependencyAsync, kg: KnowledgeGraphDependency):
    # TODO Initialize GraphDB and check its availability 
    try: 
        # NOTE db async is already checking itself..?
        kg._driver.verify_connectivity()
        logger.info("✅ Application is healthy")
        return {"status": "OK"} 
    
    except Exception as e: 
        logger.error(f"❌ Healthcheck not passed: {e}")
        return {"status": "KO"}