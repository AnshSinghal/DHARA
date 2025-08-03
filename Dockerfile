FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

#Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements
COPY . .

# Install Python dependencies
RUN uv sync --all-extras --all-groups

# Set environment variables
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uv", "run", "python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"]