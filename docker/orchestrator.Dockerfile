# syntax=docker/dockerfile:1
# Orchestrator: drives N simulator nodes against edge + agent services.
# Stateless container — exits when --duration expires (restart: "no" in compose).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install pinned deps (rule §23). Orchestrator needs:
#   - ml.txt        (numpy + simulation modules)
#   - agent.txt     (httpx for async edge / agent calls)
COPY requirements ./requirements
RUN pip install --upgrade pip \
 && pip install -r requirements/ml.txt \
 && pip install -r requirements/agent.txt

COPY api ./api
COPY configs ./configs
COPY orchestrator ./orchestrator
COPY simulation ./simulation
COPY utils ./utils

# Default runtime knobs (override via compose `environment` or CLI args).
ENV AGENTPV_EDGE_URL=http://edge-service:8000 \
    AGENTPV_AGENT_URL=http://agent-service:8001

# CLI honours --edge / --agent flags; compose passes them via command:.
CMD ["python", "-m", "orchestrator", "--nodes", "pv2_bess1", "--duration", "60"]
