import asyncio
import logging
import time
import json
import os
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from src.data_models import (
    SearchQuery, EnrichedChunk, GenerationResult, 
    RerankingResult, LegalMetadata
)
from src.rhetorical_chunking import DocumentProcessor
from src.embeddings import LegalEmbeddingModel, MetadataAwareEmbedding  
from src.vector_store import EnhancedPineconeStore
from src.legal_bm25 import LegalBM25Retriever
from src.multi_faceted_search import LegalMultiFacetedSearch

# Fixed import - now uses config.settings
from config.settings import settings

logger = logging.getLogger(__name__)

class CompleteLegalRAGPipeline:
    """Complete production-ready legal RAG pipeline with smart index loading"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the complete RAG pipeline"""
        
        self.config = config or {}
        self.enable_gpu = self.config.get('enable_gpu', False)
        
        # Initialize core components
        self._initialize_components()
        
        # Pipeline state
        self.is_initialized = False
        self.index_built = False
        
        # Check for existing indices on startup
        self._check_existing_indices()
        
        logger.info("Complete Legal RAG Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        
        try:
            # Document processing
            self.document_processor = DocumentProcessor()
            
            # Embeddings - FIXED dimension
            self.embedding_model = LegalEmbeddingModel(
                device='cuda' if self.enable_gpu else 'cpu'
            )
            self.metadata_embedding = MetadataAwareEmbedding(self.embedding_model)
            
            # Vector store - ONLY PINECONE
            self.vector_store = EnhancedPineconeStore()
            
            # Sparse retrieval
            self.bm25_retriever = LegalBM25Retriever()
            
            # Multi-faceted search
            self.faceted_search = LegalMultiFacetedSearch(
                self.vector_store, self.embedding_model
            )
            
            # Lazy-loaded components to avoid circular imports
            self._hybrid_retriever = None
            self._reranker = None
            self._generator = None
            self._response_evaluator = None
            
            logger.info("Core pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            raise
    
    def _check_existing_indices(self):
        """Check for existing BM25 index and metadata files"""
        
        bm25_path = Path(settings.BM25_INDEX_FILE)
        metadata_path = Path(settings.CHUNKS_METADATA_FILE)
        
        if bm25_path.exists():
            logger.info(f"Found existing BM25 index: {bm25_path}")
            try:
                self.bm25_retriever.load_index(str(bm25_path))
                logger.info("✅ BM25 index loaded successfully")
                self.index_built = True
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
        
        if metadata_path.exists():
            logger.info(f"Found existing chunks metadata: {metadata_path}")
            self.is_initialized = True
        
        # Check Pinecone index
        try:
            stats = self.vector_store.get_index_stats()
            if stats.get('total_vector_count', 0) > 0:
                logger.info(f"✅ Pinecone index has {stats['total_vector_count']} vectors")
                self.index_built = True
                self.is_initialized = True
        except Exception as e:
            logger.warning(f"Could not check Pinecone index: {e}")
    
    @property
    def hybrid_retriever(self):
        """Lazy load hybrid retriever to avoid circular imports"""
        if self._hybrid_retriever is None:
            from src.advanced_hybrid_retrieval import AdvancedHybridRetriever
            self._hybrid_retriever = AdvancedHybridRetriever(
                self.vector_store, self.bm25_retriever, self.embedding_model
            )
        return self._hybrid_retriever
    
    @property
    def reranker(self):
        """Lazy load reranker to avoid circular imports"""
        if self._reranker is None:
            from src.legal_reranking import MultiStageReranker
            self._reranker = MultiStageReranker()
        return self._reranker
    
    @property 
    def generator(self):
        """Lazy load generator to avoid circular imports"""
        if self._generator is None:
            from src.context_aware_generation import RhetoricallyAwareGenerator
            self._generator = RhetoricallyAwareGenerator(
                device='cuda' if self.enable_gpu else 'cpu'
            )
        return self._generator
    
    async def build_index(self, data_directory: str, 
                         force_rebuild: bool = False) -> bool:
        """Build the complete index from merged JSON files"""
        
        if self.index_built and not force_rebuild:
            logger.info("Index already built, skipping...")
            return True
        
        start_time = time.time()
        
        try:
            logger.info("Starting index building process...")
            
            # Step 1: Process documents into chunks
            logger.info("Step 1: Processing documents into rhetorical chunks")
            chunks = self.document_processor.process_all_documents(data_directory)
            
            if not chunks:
                logger.error("No chunks created from documents")
                return False
            
            logger.info(f"Created {len(chunks)} rhetorical chunks")
            
            # Step 2: Generate embeddings
            logger.info("Step 2: Generating embeddings with metadata enrichment")
            embeddings = self.metadata_embedding.encode_chunks_with_metadata(chunks)
            
            # FIXED: Safe embedding dimension check
            def get_embedding_dimension(embeddings):
                """Safely get embedding dimension from various formats"""
                if embeddings is None:
                    return 'N/A'
                
                # Handle NumPy arrays
                if hasattr(embeddings, 'shape'):
                    if embeddings.size == 0:
                        return 'N/A'
                    elif len(embeddings.shape) == 1:
                        return embeddings.shape  # 1D array
                    elif len(embeddings.shape) == 2:
                        return embeddings.shape  # 2D array (N, D)
                    else:
                        return f"Shape: {embeddings.shape}"
                
                # Handle lists
                elif isinstance(embeddings, (list, tuple)):
                    if len(embeddings) == 0:
                        return 'N/A'
                    elif hasattr(embeddings[0], '__len__'):
                        return len(embeddings[0])
                    else:
                        return 'Unknown'
                else:
                    return 'Unknown'
            
            embedding_dim = get_embedding_dimension(embeddings)
            logger.info(f"Embedding dimension: {embedding_dim}")
            
            # Validate embeddings
            if embeddings is None or (hasattr(embeddings, 'size') and embeddings.size == 0):
                logger.error("No embeddings generated")
                return False
            
            # Step 3: Index in vector store
            logger.info("Step 3: Indexing in vector store")
            vector_success = self.vector_store.upsert_enriched_chunks(chunks, embeddings)
            
            if not vector_success:
                logger.error("Failed to index chunks in vector store")
                return False
            
            # Step 4: Build BM25 index
            logger.info("Step 4: Building BM25 sparse index")
            self.bm25_retriever.build_index(chunks)
            
            # Step 5: Save indices
            logger.info("Step 5: Saving indices to disk")
            os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
            bm25_index_path = f"{settings.PROCESSED_DATA_DIR}/{settings.BM25_INDEX_FILE}"
            self.bm25_retriever.save_index(bm25_index_path)
            
            # Save chunk metadata for evaluation
            chunks_metadata_path = f"{settings.PROCESSED_DATA_DIR}/{settings.CHUNKS_METADATA_FILE}"
            self._save_chunks_metadata(chunks, chunks_metadata_path)
            
            indexing_time = time.time() - start_time
            
            logger.info(f"Index building completed successfully in {indexing_time:.2f}s")
            
            self.index_built = True
            self.is_initialized = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _save_chunks_metadata(self, chunks: List[EnrichedChunk], file_path: str):
        """Save chunks metadata for evaluation and monitoring"""
        
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                'id': chunk.id,
                'document_id': chunk.metadata.document_id,
                'rhetorical_roles': chunk.metadata.rhetorical_roles,
                'entity_types': chunk.metadata.entity_types,
                'precedent_count': chunk.metadata.precedent_count,
                'statute_count': chunk.metadata.statute_count,
                'legal_concepts': chunk.legal_concepts,
                'keywords': chunk.keywords,
                'text_length': len(chunk.text)
            }
            chunks_data.append(chunk_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved chunks metadata to {file_path}")
    
    async def process_query(self, query: str, search_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a complete query through the entire RAG pipeline"""
        
        if not self.is_initialized:
            return {
                'query': query,
                'response': 'Pipeline not initialized. Please build index first.',
                'success': False,
                'error': 'Pipeline not initialized'
            }
        
        start_time = time.time()
        
        try:
            # Parse search options
            search_options = search_options or {}
            search_query = self._create_search_query(query, search_options)
            
            # Step 1: Hybrid Retrieval
            logger.info("Step 1: Executing hybrid retrieval")
            
            retrieval_results = self.hybrid_retriever.hybrid_search(
                query=query,
                search_query=search_query,
                top_k=search_options.get('retrieval_top_k', settings.TOP_K_RETRIEVAL),
                fusion_method=search_options.get('fusion_method', 'weighted_sum')
            )
            
            if not retrieval_results:
                return self._create_empty_response(query, start_time, "No relevant documents found")
            
            # Convert results to EnrichedChunk objects
            retrieved_chunks = self._convert_results_to_chunks(retrieval_results)
            
            # Step 2: Multi-stage Reranking
            logger.info("Step 2: Executing multi-stage reranking")
            
            query_context = {
                'rhetorical_roles': search_query.rhetorical_roles,
                'entity_types': search_query.entity_types
            }
            
            reranking_result = self.reranker.rerank_with_strategy(
                query=query,
                chunks=retrieved_chunks,
                query_context=query_context,
                strategy=search_options.get('reranking_strategy')
            )
            
            # Select top chunks for generation
            top_k_generation = search_options.get('generation_top_k', settings.TOP_K_RERANK)
            final_chunks = reranking_result.chunks[:top_k_generation]
            
            # Step 3: Context-Aware Generation
            logger.info("Step 3: Executing context-aware generation")
            
            generation_result = self.generator.generate_legal_response(
                query=query,
                chunks=final_chunks,
                max_length=search_options.get('max_generation_length'),
                temperature=search_options.get('temperature')
            )
            
            # Calculate overall metrics
            total_time = time.time() - start_time
            
            # Create comprehensive response
            response = {
                'query': query,
                'response': generation_result.response,
                'confidence_score': generation_result.confidence_score,
                'processing_time': total_time,
                
                # Context information
                'contexts_used': len(final_chunks),
                'supporting_documents': [
                    {
                        'id': chunk.metadata.document_id,
                        'rhetorical_role': chunk.metadata.primary_role,
                        'relevance_score': getattr(chunk.metadata, 'rerank_score', 0.0),
                        'text_preview': chunk.text[:200] + '...' if len(chunk.text) > 200 else chunk.text
                    }
                    for chunk in final_chunks
                ],
                
                # Citations and legal references
                'legal_citations': generation_result.legal_citations,
                'success': True
            }
            
            logger.info(f"Query processed successfully in {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'query': query,
                'response': f"I apologize, but I encountered an error while processing your legal query: {str(e)}",
                'error': str(e),
                'processing_time': error_time,
                'success': False
            }
    
    def _create_search_query(self, query: str, options: Dict[str, Any]) -> SearchQuery:
        """Create SearchQuery object from query and options"""
        
        return SearchQuery(
            query=query,
            rhetorical_roles=options.get('rhetorical_roles'),
            entity_types=options.get('entity_types'),
            court_levels=options.get('court_levels'),
            min_precedent_count=options.get('min_precedent_count'),
            min_statute_count=options.get('min_statute_count'),
            date_from=options.get('date_from'),
            date_to=options.get('date_to'),
            top_k=options.get('top_k', 10),
            search_method=options.get('search_method', 'hybrid'),
            enable_reranking=options.get('enable_reranking', True)
        )
    
    def _convert_results_to_chunks(self, results: List[Dict[str, Any]]) -> List[EnrichedChunk]:
        """Convert search results to EnrichedChunk objects"""
        
        chunks = []
        for result in results:
            metadata = LegalMetadata(
                document_id=result.get('metadata', {}).get('document_id', ''),
                chunk_id=result['id'],
                chunk_index=result.get('metadata', {}).get('chunk_index', 0),
                rhetorical_roles=result.get('metadata', {}).get('rhetorical_roles', []),
                primary_role=result.get('metadata', {}).get('primary_role', ''),
                entities=[],
                entity_types=result.get('metadata', {}).get('entity_types', []),
                entity_count=result.get('metadata', {}).get('entity_count', 0),
                precedent_count=result.get('metadata', {}).get('precedent_count', 0),
                statute_count=result.get('metadata', {}).get('statute_count', 0),
                provision_count=result.get('metadata', {}).get('provision_count', 0),
                source_file=result.get('metadata', {}).get('source_file', ''),
                original_start=0,
                original_end=0,
                # FIXED: Set retrieval score properly
                retrieval_score=result.get('score', 0.0)
            )
            
            chunk = EnrichedChunk(
                id=result['id'],
                text=result['text'],
                cleaned_text=result['text'],
                metadata=metadata,
                keywords=result.get('metadata', {}).get('keywords', []),
                legal_concepts=result.get('metadata', {}).get('legal_concepts', [])
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_empty_response(self, query: str, start_time: float, message: str) -> Dict[str, Any]:
        """Create response for cases with no results"""
        
        return {
            'query': query,
            'response': message,
            'confidence_score': 0.0,
            'processing_time': time.time() - start_time,
            'contexts_used': 0,
            'supporting_documents': [],
            'legal_citations': [],
            'success': False
        }
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        stats = {
            'pipeline_status': {
                'initialized': self.is_initialized,
                'index_built': self.index_built,
                'components_loaded': True
            },
            'configuration': {
                'enable_gpu': self.enable_gpu,
                'embedding_model': getattr(self.embedding_model, 'model_name', 'Unknown'),
                'vector_dimension': settings.VECTOR_DIMENSION
            },
            'system_resources': {
                'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                'cpu_count': psutil.cpu_count(),
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
        }
        
        # Add vector store stats
        try:
            stats['vector_store'] = self.vector_store.get_index_stats()
        except Exception as e:
            logger.warning(f"Could not get vector store stats: {e}")
        
        return stats
