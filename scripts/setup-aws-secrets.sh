#!/bin/bash
set -e

AWS_REGION="${AWS_REGION:-us-east-1}"
SECRET_NAME_PREFIX="${SECRET_NAME_PREFIX:-legal-rag-api}"

echo "🔐 Setting up AWS Secrets Manager for Legal RAG API..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first"
    exit 1
fi

# Function to create or update AWS secret
create_aws_secret() {
    local secret_name=$1
    local secret_description=$2
    local full_secret_name="${SECRET_NAME_PREFIX}/${secret_name}"
    
    echo "📝 Enter your $secret_name:"
    read -s secret_value
    echo ""
    
    if [[ -z "$secret_value" ]]; then
        echo "❌ Empty secret value. Skipping $secret_name"
        return 1
    fi
    
    # Check if secret exists
    if aws secretsmanager describe-secret --secret-id "$full_secret_name" --region "$AWS_REGION" >/dev/null 2>&1; then
        echo "🔄 Updating existing secret: $full_secret_name"
        aws secretsmanager update-secret \
            --secret-id "$full_secret_name" \
            --secret-string "$secret_value" \
            --region "$AWS_REGION" >/dev/null
    else
        echo "🆕 Creating new secret: $full_secret_name"
        aws secretsmanager create-secret \
            --name "$full_secret_name" \
            --description "$secret_description" \
            --secret-string "$secret_value" \
            --region "$AWS_REGION" >/dev/null
    fi
    
    echo "✅ Secret $secret_name configured in AWS"
    
    # Output the ARN for reference
    local secret_arn=$(aws secretsmanager describe-secret \
        --secret-id "$full_secret_name" \
        --region "$AWS_REGION" \
        --query 'ARN' \
        --output text)
    echo "   ARN: $secret_arn"
}

# Create AWS secrets
create_aws_secret "pinecone-api-key" "Pinecone API Key for Legal RAG system"
create_aws_secret "google-api-key" "Google Gemini API Key for Legal RAG system"

echo ""
echo "🎉 AWS Secrets Manager setup complete!"
echo ""
echo "📝 For ECS Task Definition, use these secret ARNs:"
echo "   Pinecone: arn:aws:secretsmanager:${AWS_REGION}:$(aws sts get-caller-identity --query Account --output text):secret:${SECRET_NAME_PREFIX}/pinecone-api-key"
echo "   Google: arn:aws:secretsmanager:${AWS_REGION}:$(aws sts get-caller-identity --query Account --output text):secret:${SECRET_NAME_PREFIX}/google-api-key"

echo ""
echo "🔐 IAM Policy needed for ECS task role:"
cat << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": [
                "arn:aws:secretsmanager:${AWS_REGION}:$(aws sts get-caller-identity --query Account --output text):secret:${SECRET_NAME_PREFIX}/*"
            ]
        }
    ]
}
EOF
