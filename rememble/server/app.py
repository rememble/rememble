"""Combined HTTP app — REST API + MCP SSE on a single port."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from rememble.server.api import router
from rememble.server.mcp import mcp
from rememble.state import closeState, initState


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initState()
    yield
    closeState()


def createApp() -> FastAPI:
    app = FastAPI(title="Rememble", lifespan=lifespan)
    app.include_router(router, prefix="/api")
    mcp_app = mcp.http_app(transport="sse")
    app.mount("/mcp", mcp_app)
    return app
