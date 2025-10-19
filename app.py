from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.factory.database import initialize_database
from src.api.routers import chats, files, health 


KNOWLEDGE_GRAPH_APP = "knowledge-graph"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown events"""
    await initialize_database()
    yield


app = FastAPI(
    lifespan=lifespan,
    description="API for Knowledge Graph ingestion and querying of documents with integrated Swagger.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(router=health.router, prefix=KNOWLEDGE_GRAPH_APP)
app.include_router(prefix=chats.router, prefix=KNOWLEDGE_GRAPH_APP)
app.include_router(prefix=files.router, prefix=KNOWLEDGE_GRAPH_APP)
