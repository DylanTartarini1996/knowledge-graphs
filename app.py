from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.routers import chats, files, health 
from src.factory.database import initialize_database


KNOWLEDGE_GRAPH_APP = "/knowledge-graph"


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
app.include_router(router=chats.router, prefix=KNOWLEDGE_GRAPH_APP)
app.include_router(router=files.router, prefix=KNOWLEDGE_GRAPH_APP)
