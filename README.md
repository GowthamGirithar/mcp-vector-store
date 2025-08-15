# MCP Vector Database Server

A Model Context Protocol (MCP) server that provides vector database functionality with automatic text embedding and similarity search capabilities. This server enables AI applications to store, index, and retrieve text documents using semantic similarity.

I have developed this for storing the user interactions as it is must for context engineering.

## Features

- **Text Storage**: Store text documents with automatic embedding generation
- **Similarity Search**: Find semantically similar documents using vector search
- **Multiple Embedding Providers**: Support for OpenAI and Sentence Transformers
- **Vector Database Support**: ChromaDB integration with extensible adapter pattern
- **Multiple Transport Modes**: STDIO, SSE, and Streamable HTTP support
- **Metadata Support**: Store and filter documents with custom metadata
- **Auto Collection Management**: Automatic collection creation and management


## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp-vectordb-server
```

2. Install dependencies:

For basic usage:
```bash
pip install -r requirements.txt
```

For development (includes testing and linting tools):
```bash
pip install -r requirements-dev.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Starting the Server

Run the server using the main entry point:

```bash
python main.py
```

The server supports three transport modes:
- **STDIO**: Standard input/output for direct process communication
- **SSE**: Server-Sent Events for web-based streaming
- **Streamable HTTP**: HTTP-based streaming protocol

### Available Tools

The server provides two main MCP tools:

#### 1. `store_text`
Store text documents with automatic embedding generation.

**Parameters:**
- `text` (required): The text content to store
- `collection` (optional): Collection name (default: "documents")
- `metadata` (optional): Custom metadata dictionary
- `document_id` (optional): Custom document ID


#### 2. `similarity_search`
Perform semantic similarity search to find relevant documents.

**Parameters:**
- `query` (required): The search query text
- `collection` (optional): Collection to search (default: "documents")
- `top_k` (optional): Number of results to return (1-100, default: 10)
- `filters` (optional): Metadata filters
- `include_scores` (optional): Include similarity scores (default: true)
- `include_metadata` (optional): Include document metadata (default: true)
- `min_score` (optional): Minimum similarity score threshold (0.0-1.0)


### Client Examples

The repository includes example clients for different transport modes:

#### STDIO Client
```bash
python example_mcp_client_stdio.py
```

#### SSE Client
```bash
python example_mcp_client_sse.py
```

#### Streamable HTTP Client
```bash
python example_mcp_client_streamable_http.py
```

## Configuration Options

### Vector Database Providers

Currently supported:
- **ChromaDB**: Local vector database with persistence

### Embedding Providers

- **OpenAI**: Uses OpenAI's embedding API (requires API key)
  - Models: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **Sentence Transformers**: Local embedding models (no API key required)
  - Models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc.

### Transport Modes

- **stdio**: Direct process communication
- **sse**: Server-Sent Events (accessible at `http://127.0.0.1:8000/sse`)
- **streamable-http**: HTTP streaming (accessible at `http://127.0.0.1:8000/mcp`)


### Adding New Vector Database Adapters

1. Create a new adapter in `src/mcp_vectordb/adapters/`
2. Implement the `VectorDBAdapter` interface from `base.py`
3. Register the adapter in `factory.py`
4. Update configuration to support the new provider

### Adding New Embedding Providers

1. Implement the `EmbeddingService` interface in `core/embedding.py`
2. Update the service factory in `services.py`
3. Add configuration options in `config/config.py`

### TODO List

1. Combine BM25 and vector results to provide accuracy
2. Use LLM to return the response from the BM25 and vector search results inaddition to the simillarity search results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
