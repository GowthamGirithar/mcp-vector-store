"""MCP Vector Database Server

A Model Context Protocol server for vector database operations using FastMCP.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .server import get_mcp, run_server

__all__ = ["get_mcp",  "run_server"]