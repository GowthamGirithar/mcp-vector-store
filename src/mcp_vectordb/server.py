"""FastMCP Vector Database Server - Main Application."""

from mcp.server.fastmcp import FastMCP
from .services import setup_services
from .config.config import get_settings
import asyncio


# Create FastMCP app
# Set up services (database, embeddings, etc.)
# lifespan hook. To tell what to do at the start of the session and end of the session
# mcp.settings.lifespan = setup_services
mcp = FastMCP(name="vectordb-server", lifespan=setup_services)


# Import tools to register them via decorators
# so all the available tools will be loaded
from . import tools  # noqa: E402

def get_mcp() -> FastMCP:
    """Get the FastMCP application instance."""
    return mcp


# http://127.0.0.1:8000/mcp for streamable http
# http://127.0.0.1:8000/sse for SSE
# we can use Starlette to mount multiple FactMCP server
def run_server() -> None:
    """Run the server (synchronous entry point)."""
    # more details abou working of stdio, sse, streamable-http are in client example files
    settings = get_settings()
    get_mcp().run(transport=settings.server.transport) # stdio, sse, streamable-http

if __name__ == "__main__":
    run_server()


