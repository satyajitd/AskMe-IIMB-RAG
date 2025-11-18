# AskMe IIM Bangalore RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with LangGraph that answers questions about IIM Bangalore using hybrid search (vector + keyword) and continuous learning through web search fallback.

## Overview

This intelligent Q&A system routes queries about IIM Bangalore through a multi-stage workflow that combines vector database retrieval, web search fallback, and answer quality grading to provide accurate, grounded responses.

### Key Features

- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search for better retrieval
- **Continuous Learning**: Automatically ingests web search results into the vector database
- **Answer Quality Grading**: Validates responses and triggers regeneration if insufficient
- **Multi-LLM Support**: Works with Ollama (local/cloud) and Google Gemini
- **Async Web Search**: Non-blocking web search with configurable retry limits
- **Docker-Ready**: Fully containerized with Weaviate vector database

## Architecture

### Workflow Graph

```
┌─────────────┐
│   Question  │
└──────┬──────┘
       │
       v
┌─────────────┐
│   Router    │──────► Off-topic ──► Polite Refusal
└──────┬──────┘
       │ (IIM Bangalore related)
       v
┌─────────────┐
│  Retrieve   │ (Vector DB)
└──────┬──────┘
       │
       v
┌─────────────┐
│  Documents? │
└──────┬──────┘
       │
       ├─► Yes ──► Generate Answer
       │
       └─► No ──► Web Search ──► Generate Answer
                      │
                      └─► Persist to Vector DB
                            │
                            v
                     ┌─────────────┐
                     │ Grade Answer│
                     └──────┬──────┘
                            │
                            ├─► Sufficient ──► Return Answer
                            │
                            └─► Insufficient ──► Retry (max 3 attempts)
```

### Components

#### Agents (`agents/`)

1. **Router** (`router.py`)
   - Routes questions to vector store or marks as off-topic
   - Filters non-IIM Bangalore queries

2. **Generator** (`generator.py`)
   - Generates responses using retrieved context
   - Cites sources with bracketed indices
   - Formats answers with Summary → Details → Sources

3. **Answer Grader** (`answer_grader.py`)
   - Validates if answer sufficiently addresses the question
   - Returns 'yes' or 'no' with justification
   - Triggers web search or regeneration if insufficient

#### Vector Store (`store/vectorstore.py`)

- **Backend**: Weaviate vector database
- **Embeddings**: FastEmbed (BAAI/bge-base-en-v1.5)
- **Search Modes**:
  - `semantic`: Pure vector similarity search
  - `hybrid`: Combines BM25 + vector (configurable alpha)
- **Features**:
  - Async-ready upsert for continuous learning
  - Metadata tracking (source, URL, ingest timestamp)
  - Automatic embedding generation

#### Tools (`tools/`)

- **Web Search** (`search.py`)
  - Tavily API integration
  - Returns up to 5 results with titles, URLs, and snippets
  - Auto-persisted to vector database

#### Workflow (`workflow/workflow.py`)

Orchestrates the entire RAG pipeline using LangGraph:

- **Nodes**: `retrieve`, `generate`, `web_search`, `off_topic`
- **Conditional Edges**: Route based on document availability and answer quality
- **State Management**: Tracks attempts, documents, web search status
- **Retry Logic**: Maximum 3 generation attempts before finalizing

## Dependencies

### Core Frameworks

```
langchain-core          # LangChain framework
langgraph              # Workflow orchestration
langgraph-cli[inmem]   # Development server
```

### LLM Providers

```
langchain-ollama       # Ollama integration (local/cloud)
langchain-google-genai # Google Gemini integration
```

### Vector Database & Embeddings

```
weaviate-client                    # Weaviate Python client
llama-index-core                   # LlamaIndex framework
llama-index-vector-stores-weaviate # Weaviate integration
llama-index-embeddings-fastembed   # FastEmbed integration
fastembed                          # Fast embedding model
```

### External Tools

```
langchain-tavily       # Tavily web search API
```

## Environment Configuration

Create a `config.env` file in the parent directory:

```env
# LLM Provider (choose one: "ollama", "ollama-cloud", "gemini")
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://host.docker.internal:11434
MAX_OUTPUT_TOKENS=2048

# For Ollama Cloud
OLLAMA_CLOUD_API_KEY=your_key_here

# For Gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash

# Weaviate Vector Database
WEAVIATE_COLLECTION_NAME=AskmeIIMB
WEAVIATE_SERVER_HOST=weaviate        # Use 'localhost' for local dev
WEAVIATE_SERVER_PORT=8080            # Internal port (8081 for external)
WEAVIATE_N_RESULTS=5
WEAVIATE_SEARCH_MODE=hybrid          # "semantic" or "hybrid"
WEAVIATE_HYBRID_ALPHA=0.5            # 0.0=keyword, 1.0=vector

# Tavily Web Search
TAVILY_API_KEY=your_tavily_api_key

# LangSmith (optional tracing)
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=rag_project

# Logging
APP_LOG=INFO
```

## Installation & Setup

### Using Docker (Recommended)

1. **Start the services**:
```bash
docker-compose up -d
```

This starts:
- Weaviate vector database (port 8081)
- RAG application (port 8123)

2. **Check status**:
```bash
docker ps
docker logs askme-rag
```

3. **Verify Weaviate**:
```bash
curl http://localhost:8081/v1/meta
```

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start Weaviate**:
```bash
docker-compose up -d weaviate
```

3. **Update config.env**:
```env
WEAVIATE_SERVER_HOST=localhost
WEAVIATE_SERVER_PORT=8081
```

4. **Run the application**:
```bash
langgraph dev
```

The API will be available at `http://localhost:8123`

## API Endpoints

### LangGraph Server API

Once running, the system exposes standard LangGraph endpoints:

#### 1. **Stream Run** (Recommended)
```bash
POST http://localhost:8123/runs/stream

Body:
{
  "assistant_id": "agent",
  "input": {
    "question": "Who is the director of IIM Bangalore?"
  },
  "stream_mode": "values"
}
```

#### 2. **Invoke Run**
```bash
POST http://localhost:8123/runs/wait

Body:
{
  "assistant_id": "agent",
  "input": {
    "question": "What are the MBA programs offered?"
  }
}
```

#### 3. **Get Run Status**
```bash
GET http://localhost:8123/runs/{run_id}
```

#### 4. **Health Check**
```bash
GET http://localhost:8123/ok
```

### Response Format

```json
{
  "question": "Who is the director of IIM Bangalore?",
  "generation": "Summary\n[Answer text with citations [1], [2]]\n\nDetails\n- Bullet points\n\nSources\n[1] Source title - URL",
  "documents": [...],
  "attempts": 1,
  "web_search": "yes"
}
```

## Usage Examples

### Python Client

```python
import requests

url = "http://localhost:8123/runs/wait"
payload = {
    "assistant_id": "agent",
    "input": {
        "question": "What are the hostel facilities at IIM Bangalore?"
    }
}

response = requests.post(url, json=payload)
result = response.json()
print(result["output"]["generation"])
```

### cURL

```bash
curl -X POST http://localhost:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "question": "Tell me about the faculty at IIM Bangalore"
    }
  }'
```

## How It Works

### 1. Question Routing
- Router checks if the question is about IIM Bangalore
- Off-topic queries get a polite refusal response

### 2. Document Retrieval
- Retrieves top-k documents from Weaviate using hybrid search
- If no documents found, triggers web search immediately

### 3. Answer Generation
- Generates answer using retrieved context
- Includes source citations
- Tracks attempt count

### 4. Answer Grading
- Answer Grader validates if response sufficiently addresses question
- If insufficient and no web search done: trigger web search
- If insufficient after web search: regenerate (up to 3 attempts)
- After 3 attempts: return best available answer

### 5. Continuous Learning
- Web search results are automatically embedded and stored in Weaviate
- Includes metadata: source URL, title, query, timestamp
- Future queries can retrieve these learned facts

## Configuration Options

### Search Mode

**Semantic Search** (pure vector):
```env
WEAVIATE_SEARCH_MODE=semantic
```

**Hybrid Search** (vector + BM25):
```env
WEAVIATE_SEARCH_MODE=hybrid
WEAVIATE_HYBRID_ALPHA=0.5  # Balance between keyword (0.0) and vector (1.0)
```

### Retry Limits

Maximum regeneration attempts are set in `utils/constants.py`:
```python
MAX_ATTEMPTS=3
```

### LLM Provider

Switch between providers in `config.env`:
- `ollama`: Local Ollama instance
- `ollama-cloud`: Ollama cloud API
- `gemini`: Google Gemini

## Project Structure

```
rag/
├── agents/
│   ├── answer_grader.py      # Answer quality validation
│   ├── base_agent.py          # Base agent class
│   ├── generator.py           # Response generator
│   └── router.py              # Query router
├── model/
│   └── llm.py                 # LLM provider factory
├── store/
│   └── vectorstore.py         # Weaviate integration
├── tools/
│   └── search.py              # Tavily web search
├── utils/
│   ├── constants.py           # Application constants
│   ├── env.py                 # Environment variable names
│   └── logger.py              # Logging configuration
├── workflow/
│   ├── state.py               # LangGraph state definition
│   └── workflow.py            # Main workflow orchestration
├── docker-compose.yml         # Docker services
├── Dockerfile                 # Application container
├── langgraph.json             # LangGraph configuration
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Troubleshooting

### Connection Errors

**Problem**: `Connection to Weaviate failed`

**Solution**: Ensure Weaviate is running and check port configuration
```bash
# Check Weaviate status
docker ps | grep weaviate

# Test connection
curl http://localhost:8081/v1/meta

# For Docker, use internal port in config.env:
WEAVIATE_SERVER_HOST=weaviate
WEAVIATE_SERVER_PORT=8080
```

### Import Errors

**Problem**: Module not found errors

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Slow Responses

**Problem**: Queries taking too long

**Solution**: 
- Reduce `WEAVIATE_N_RESULTS` (default: 5)
- Use `semantic` mode instead of `hybrid`
- Check Ollama/Gemini API latency

### Web Search Not Working

**Problem**: Web search returns no results

**Solution**: Verify Tavily API key
```bash
# Test Tavily API
curl -X POST https://api.tavily.com/search \
  -H "Content-Type: application/json" \
  -d '{"api_key": "YOUR_KEY", "query": "IIM Bangalore"}'
```

## Development

### Running Tests
```bash
# Coming soon
pytest tests/
```

### Viewing Logs
```bash
# Docker
docker logs -f askme-rag

# Local
tail -f log/app.log
```

### Debugging Workflow
```bash
# LangGraph provides built-in debugging UI
langgraph dev --debug
```

## Migration Notes

This system recently migrated from ChromaDB to Weaviate. Key changes:
- Enhanced hybrid search capabilities (BM25 + vector)
- Better scalability and production readiness
- Richer metadata support
- See `WEAVIATE_MIGRATION.md` for details

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

## Contact

For questions about IIM Bangalore, ask the system!
For technical issues, [contact information].
