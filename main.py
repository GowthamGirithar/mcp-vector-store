"""Main entry point for MCP Vector Database Server."""

import asyncio
import sys
from src.mcp_vectordb.server import run_server

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server failed: {e}")
        sys.exit(1)