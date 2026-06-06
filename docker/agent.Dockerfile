# syntax=docker/dockerfile:1
# Cloud agent service (Component 4 / 6).
# Stack: FastAPI + RAG (TF-IDF on sklearn) + Mock LLM (MVP).
# Polish phase will add chromadb / sentence-transformers / langchain via a
# separate "agent-polish.txt" rather than bloating the MVP image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install pinned deps (rule §23). Agent needs:
#   - ml.txt        (sklearn for TF-IDF retrieval, jinja2 for prompts)
#   - agent.txt     (fastapi / uvicorn / httpx)
COPY requirements ./requirements
RUN pip install --upgrade pip \
 && pip install -r requirements/ml.txt \
 && pip install -r requirements/agent.txt

COPY api ./api
COPY agent ./agent
COPY configs ./configs
COPY rag ./rag
COPY tools ./tools
COPY utils ./utils

EXPOSE 8001

CMD ["uvicorn", "api.agent_service:app", "--host", "0.0.0.0", "--port", "8001"]
