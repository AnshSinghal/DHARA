#!/bin/bash
set -e

echo "🏗️  Building and running Legal RAG API..."

# Setup secrets if they don't exist
if [[ ! -d "secrets" || ! -f "secrets/PINECONE_API_KEY.txt" ]]; then
    echo "🔐 Setting up secrets first..."
    ./scripts/setup-secrets.sh
fi

# Build and run
echo "🐳 Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for service to be healthy..."
sleep 10

# Check health
echo "🏥 Checking service health..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Service is healthy!"
        echo "🌐 API available at: http://localhost:8000"
        echo "📚 API docs at: http://localhost:8000/docs"
        break
    fi
    echo "⏳ Waiting for service... ($i/30)"
    sleep 5
done

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20 legal-rag-api
