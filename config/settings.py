import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings:
# Model configurations
    LEGAL_EMBEDDING_MODEL = "law-ai/InLegalBERT"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GENERATION_MODEL = "microsoft/DialoGPT-medium"

    # Vector dimensions - FIXED to match your Pinecone index
    VECTOR_DIMENSION = 768  # Base embedding dimension (not 818)
    METADATA_FEATURES = 0   # Remove metadata expansion to fix dimension mismatch

    # Pinecone configuration - USE YOUR PROVIDED INDEX
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key-here")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = "legal-rag-enhanced"  # Your index name

    # Pipeline settings
    TOP_K_RETRIEVAL = 20
    TOP_K_RERANK = 10
    MAX_GENERATION_LENGTH = 512
    TEMPERATURE = 0.7

    # Chunk settings
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 50

    # File paths
    PROCESSED_DATA_DIR = "processed"
    BM25_INDEX_FILE = "legal_bm25_index.pkl"
    CHUNKS_METADATA_FILE = "chunks_metadata.json"

    DATA_DIR = "data"
    MERGED_DATA_DIR = f"{DATA_DIR}/merged"
    PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

    # UPDATED: Correct rhetorical roles mapping based on your specification
    RHETORICAL_ROLES = {
        'PREAMBLE': 'Preamble',
        'FAC': 'Facts',
        'RLC': 'Ruling by Lower Court',
        'ISSUE': 'Issues',
        'ARG_PETITIONER': 'Argument by Petitioner',
        'ARG_RESPONDENT': 'Argument by Respondent',
        'ANALYSIS': 'Analysis',
        'STA': 'Statute',
        'PRE_RELIED': 'Precedent Relied',
        'PRE_NOT_RELIED': 'Precedent Not Relied',
        'Ratio': 'Ratio of the decision',
        'RPC': 'Ruling by Present Court',
        'NONE': 'None'
    }

    # UPDATED: Correct entity types based on your specification
    LEGAL_ENTITY_TYPES = {
        'COURT': 'Court',
        'PETITIONER': 'Petitioner',
        'RESPONDENT': 'Respondent',
        'JUDGE': 'Judge',
        'LAWYER': 'Lawyer',
        'DATE': 'Date',
        'ORG': 'Organization',
        'GPE': 'Geopolitical Entity',
        'STATUTE': 'Statute',
        'PROVISION': 'Provision',
        'PRECEDENT': 'Precedent',
        'CASE_NUMBER': 'Case Number',
        'WITNESS': 'Witness',
        'OTHER_PERSON': 'Other Person'
    }

    # Roles that indicate precedent-heavy content
    PRECEDENT_ROLES = ['PRE_RELIED', 'PRE_NOT_RELIED', 'Ratio']

    # Roles that indicate factual content
    FACTUAL_ROLES = ['FAC', 'ISSUE']

    # Roles that indicate legal reasoning
    REASONING_ROLES = ['ANALYSIS', 'Ratio', 'RPC']

    # Roles that indicate arguments
    ARGUMENT_ROLES = ['ARG_PETITIONER', 'ARG_RESPONDENT']

    # High-priority entity types for legal analysis
    PRIORITY_ENTITY_TYPES = ['COURT', 'JUDGE', 'STATUTE', 'PROVISION', 'PRECEDENT', 'CASE_NUMBER']


settings = Settings()