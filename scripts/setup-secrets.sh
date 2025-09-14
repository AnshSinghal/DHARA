#!/bin/bash
set -e

echo "🔐 Setting up Docker secrets for Legal RAG API..."

# Create secrets directory
mkdir -p secrets

# Function to get secret input
get_secret() {
    local secret_name=$1
    local secret_file="secrets/${secret_name}.txt"
    
    if [[ -f "$secret_file" ]]; then
        echo "✅ Secret $secret_name already exists"
        return 0
    fi
    
    echo "📝 Enter your $secret_name:"
    read -s secret_value
    echo ""
    
    if [[ -z "$secret_value" ]]; then
        echo "❌ Empty secret value. Please try again."
        get_secret "$secret_name"
        return
    fi
    
    echo "$secret_value" > "$secret_file"
    chmod 600 "$secret_file"
    echo "✅ Secret $secret_name saved securely"
}

# Get Pinecone API key
get_secret "PINECONE_API_KEY"

# Get Google API key (CORRECTED NAME)
get_secret "google_api_key"

echo ""
echo "🎉 All secrets configured successfully!"
echo "📁 Secrets stored in: ./secrets/"
echo "⚠️  Remember: Never commit the secrets/ directory to version control"

# Create .gitignore entry if it doesn't exist
if ! grep -q "secrets/" .gitignore 2>/dev/null; then
    echo "secrets/" >> .gitignore
    echo "📝 Added secrets/ to .gitignore"
fi

# Create .dockerignore entry if it doesn't exist
if ! grep -q "secrets/" .dockerignore 2>/dev/null; then
    echo "secrets/" >> .dockerignore
    echo "📝 Added secrets/ to .dockerignore for security"
fi

echo ""
echo "🚀 Ready to run: docker-compose up --build"
