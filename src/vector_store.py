from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import time
from tqdm import tqdm

from src.data_models import EnrichedChunk, SearchQuery

# Fixed import - now uses config.settings
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedPineconeStore:
    """Pinecone store with rich metadata support - CHROMADB REMOVED"""
    
    def __init__(self, index_name: str = None):
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        # FIXED: Use 818 dimensions for metadata-aware embeddings (768 text + 50 metadata)
        self.dimension = 818  # Text embeddings (768) + metadata features (50)
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._setup_index()
    
    def _setup_index(self):
        """Setup enhanced Pinecone index with new API"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating enhanced Pinecone index: {self.index_name}")
                
                # Create serverless index with 818 dimensions
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                while self.index_name not in [index.name for index in self.pc.list_indexes()]:
                    time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to enhanced Pinecone index: {self.index_name} (dimension: {self.dimension})")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def upsert_enriched_chunks(self, chunks: List[EnrichedChunk], 
                              embeddings: np.ndarray, batch_size: int = 100) -> bool:
        """Upsert enriched chunks with full metadata"""
        try:
            logger.info(f"Upserting {len(chunks)} enriched chunks to Pinecone")
            
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create comprehensive metadata for Pinecone
                metadata = {
                    # Basic info
                    'text': chunk.text[:1000],  # Truncated for Pinecone limits
                    'document_id': chunk.metadata.document_id,
                    'chunk_index': chunk.metadata.chunk_index,
                    
                    # Rhetorical role info
                    'primary_role': chunk.metadata.primary_role,
                    'rhetorical_roles': chunk.metadata.rhetorical_roles[:3],  # Top 3
                    
                    # Entity info
                    'entity_types': chunk.metadata.entity_types[:5],  # Top 5
                    'entity_count': chunk.metadata.entity_count,
                    
                    # Legal structure
                    'precedent_count': chunk.metadata.precedent_count,
                    'statute_count': chunk.metadata.statute_count,
                    'provision_count': chunk.metadata.provision_count,
                    
                    # Keywords and concepts
                    'keywords': chunk.keywords[:5],  # Top 5
                    'legal_concepts': chunk.legal_concepts[:3],  # Top 3
                    
                    # Source info
                    'source_file': chunk.metadata.source_file,
                }
                
                vector = {
                    'id': chunk.id,
                    'values': embedding.tolist(),
                    'metadata': metadata
                }
                
                vectors.append(vector)
            
            # Upsert in batches
            for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info("Successfully upserted all enriched chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting enriched chunks: {e}")
            return False
    
    def search_with_filters(self, query_vector: np.ndarray, 
                           search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Advanced search with metadata filters"""
        
        # Build Pinecone filter
        filter_dict = {}
        
        # Rhetorical role filter
        if search_query.rhetorical_roles:
            filter_dict['primary_role'] = {'$in': search_query.rhetorical_roles}
        
        # Entity type filter
        if search_query.entity_types:
            filter_dict['entity_types'] = {'$in': search_query.entity_types}
        
        # Numerical filters
        if search_query.min_precedent_count is not None:
            filter_dict['precedent_count'] = {'$gte': search_query.min_precedent_count}
        
        if search_query.min_statute_count is not None:
            filter_dict['statute_count'] = {'$gte': search_query.min_statute_count}
        
        try:
            # Execute search
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=search_query.top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Convert to standard format
            formatted_results = []
            for match in results.matches:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'metadata': match.metadata
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}