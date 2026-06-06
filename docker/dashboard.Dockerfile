# syntax=docker/dockerfile:1
# Streamlit operator dashboard (Component 6).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install pinned deps (rule §23). Dashboard needs:
#   - dashboard.txt   (streamlit + altair + pyarrow)
COPY requirements ./requirements
RUN pip install --upgrade pip \
 && pip install -r requirements/dashboard.txt

COPY api ./api
COPY configs ./configs
COPY dashboard ./dashboard
COPY utils ./utils

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
