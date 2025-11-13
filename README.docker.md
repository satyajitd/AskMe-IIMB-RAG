# Docker Deployment Guide

This guide explains how to run the AskMe IIMB RAG application using Docker.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- Ollama running on your host machine (or accessible URL)
- Tavily API key for web search functionality

## Quick Start

### 1. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and set your values:
- `TAVILY_API_KEY`: Your Tavily API key
- `OLLAMA_MODEL`: Model to use (e.g., llama2, mistral)
- `OLLAMA_BASE_URL`: Ollama server URL
- Other optional configurations

### 2. Build and Run with Docker Compose

Start all services (ChromaDB + LangGraph app):

```bash
docker-compose up -d
```

This will:
- Start ChromaDB on port 8000
- Start LangGraph application on port 8123
- Create persistent volumes for data storage

### 3. Check Service Status

```bash
docker-compose ps
docker-compose logs -f langgraph-app
```

### 4. Access the Application

- **LangGraph Studio**: Open LangGraph Studio and connect to `http://localhost:8123`
- **API Endpoint**: The workflow is available at `http://localhost:8123`

## Alternative: Run with Docker Only

If you want to run just the LangGraph app (with external ChromaDB):

```bash
# Build the image
docker build -t askme-langgraph .

# Run the container
docker run -d \
  --name askme-langgraph \
  -p 8123:8123 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/log:/app/log \
  --env-file .env \
  askme-langgraph
```

## Data Management

### Feed Data to ChromaDB

Before using the RAG workflow, populate the vector store:

```bash
# Enter the container
docker-compose exec langgraph-app bash

# Run the data feeding script
python feed_data.py

# Or process a specific folder
python -c "from feed_data import DocumentProcessor; from store.vectorstore import vector_store; processor = DocumentProcessor(vector_store); processor.process_folder('data/')"
```

### Persist Data

The following directories are mounted as volumes:
- `./data` - Source documents (JSON/PDF files)
- `./log` - Application logs
- `./store` - Processed files tracker
- `chromadb_data` - ChromaDB vector database (Docker volume)

## Service Configuration

### ChromaDB
- **Port**: 8000
- **Health check**: `http://localhost:8000/api/v1/heartbeat`
- **Persistent storage**: Yes (Docker volume)

### LangGraph Application
- **Port**: 8123
- **Health check**: `http://localhost:8123/health`
- **Restart policy**: unless-stopped

## Troubleshooting

### ChromaDB Connection Issues

If the app can't connect to ChromaDB:

```bash
# Check ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# Check ChromaDB logs
docker-compose logs chromadb
```

### Ollama Connection Issues

The container connects to Ollama on your host machine using `host.docker.internal:11434`.

**For Linux**: You may need to use your host's IP address instead:

```bash
# Find your host IP
ip addr show docker0 | grep inet

# Update .env
OLLAMA_BASE_URL=http://172.17.0.1:11434
```

**Test Ollama connection from container**:

```bash
docker-compose exec langgraph-app curl http://host.docker.internal:11434/api/tags
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f langgraph-app
docker-compose logs -f chromadb

# Application logs (inside container)
docker-compose exec langgraph-app tail -f /app/log/app.log
```

### Rebuild After Code Changes

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Production Considerations

For production deployments:

1. **Use production WSGI server**: Modify `CMD` in Dockerfile to use gunicorn or similar
2. **Enable HTTPS**: Add reverse proxy (nginx/traefik)
3. **Secrets management**: Use Docker secrets or external vault
4. **Resource limits**: Add memory/CPU limits in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```
5. **Monitoring**: Add Prometheus/Grafana for metrics
6. **Backup**: Regular backup of `chromadb_data` volume

## Development Mode

To run in development with live code reload:

```bash
# Mount source code as volume
docker run -it \
  -p 8123:8123 \
  -v $(pwd):/app \
  --env-file .env \
  askme-langgraph \
  langgraph dev --host 0.0.0.0 --port 8123 --reload
```
