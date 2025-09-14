FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Create models directory
RUN mkdir -p /app/models

# Pre-download and cache BERT model ONLY
RUN uv run python -c "import os; \
os.environ['TRANSFORMERS_CACHE'] = '/app/models'; \
os.environ['HF_HOME'] = '/app/models'; \
print('Downloading BERT reranker...'); \
from sentence_transformers import CrossEncoder; \
bert_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'); \
print('BERT model cached'); \
print('Model successfully cached!')"


# Copy application source code
COPY app/ ./app/

# Copy data directory
COPY data/ ./data/

# Copy additional config files
COPY pyproject.toml ./

# Create directories for runtime
RUN mkdir -p /app/logs

RUN mkdir -p /home/appuser/.cache/uv 

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
