# ── FastAPI Backend ──
FROM python:3.11-slim AS backend

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ config/
COPY graph/ graph/
COPY prompts/ prompts/
COPY retriever/ retriever/
COPY utils/ utils/
COPY main.py .

# Default port (Railway overrides via PORT env var)
ENV PORT=8001
EXPOSE ${PORT}

# Run with uvicorn — reads PORT env var
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
