import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dtos.request_upload import UploadFilesRequestDto
from src.api.dtos.response_upload import UploadFilesResponseDto
from src.api.factory.database import get_db_async
from src.api.services.ingestion import IngestionService
from src.config import Configuration
from src.core.factory.configuration import get_configuration_from_env
from src.core.factory.graph import get_knowledge_graph
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger


logger = get_logger(__name__)

CONFIGURATION_PATH = "config_example.env"
conf: Configuration = get_configuration_from_env(CONFIGURATION_PATH)

router = APIRouter(prefix="/files", tags=["files"])


@router.post(f"upload")
async def upload_file(
    upload_request: UploadFilesRequestDto, 
    kg: KnowledgeGraph = Depends(get_knowledge_graph(conf.graph_database)), 
    db: AsyncSession = Depends(get_db_async(conf.rel_database))
    ) -> UploadFilesResponseDto:
    """ 
    Receives an upload request with files specification and tracks it into a queue of file to be ingested.  

    Then, ingest the file into the Knowledge Graph.  
    
    If the ingestion process fails, tracks the status and leaves the file in the queue.
    """
    # TODO put file in queue

    # TODO ingest files from queue
    ingestion_service = IngestionService(config=conf)

    asyncio.create_task(ingestion_service.ingest_files(task_id=upload_request.task_id))

    response = UploadFilesResponseDto(task_id=upload_request.task_id, files=upload_request.files)

    return response

