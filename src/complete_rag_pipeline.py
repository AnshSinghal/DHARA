import asyncio
import logging
import time
import json
import os
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
import psutil

from src.data_models import (
    SearchQuery, EnrichedChunk, GenerationResult, 
    RerankingResult, LegalMetadata
)
from src.rhetorical_chunking import DocumentProcessor
from src.embeddings import LegalEmbeddingModel, MetadataAwareEmbedding  
from src.vector_store import EnhancedPineconeStore
from src.legal_bm25 import LegalBM25Retriever

# Fixed import - now uses config.settings
from config.settings import settings

logger = logging.getLogger(__name__)

class CompleteLegalRAGPipeline:
    """Complete production-ready legal RAG pipeline with comprehensive logging"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the complete RAG pipeline with detailed logging"""
        
        logger.info("🏗️ Initializing CompleteLegalRAGPipeline")
        
        self.config = config or {}
        self.enable_gpu = self.config.get('enable_gpu', False)
        
        logger.info(f"Configuration: {self.config}")
        logger.info(f"GPU enabled: {self.enable_gpu}")
        
        # Initialize core components
        self._initialize_components()
        
        # Pipeline state
        self.is_initialized = False
        self.index_built = False
        
        # Performance metrics
        self.metrics = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'last_query_time': None,
            'errors_count': 0
        }
        
        # Check for existing indices on startup
        self._check_existing_indices()
        
        logger.info("✅ CompleteLegalRAGPipeline initialized successfully")

        self.debug_and_fix_embeddings()

    def _initialize_components(self):
        """Initialize all pipeline components with detailed logging"""
        
        logger.info("🔧 Initializing pipeline components...")
        
        try:
            # Document processing
            logger.debug("Initializing DocumentProcessor...")
            self.document_processor = DocumentProcessor()
            logger.debug("✅ DocumentProcessor initialized")
            
            # Embeddings - FIXED dimension
            logger.debug(f"Initializing LegalEmbeddingModel (device: {'cuda' if self.enable_gpu else 'cpu'})...")
            self.embedding_model = LegalEmbeddingModel(
                device='cuda' if self.enable_gpu else 'cpu'
            )
            logger.debug("✅ LegalEmbeddingModel initialized")
            
            logger.debug("Initializing MetadataAwareEmbedding...")
            self.metadata_embedding = MetadataAwareEmbedding(self.embedding_model)
            logger.debug("✅ MetadataAwareEmbedding initialized")
            
            # Vector store - ONLY PINECONE
            logger.debug("Initializing EnhancedPineconeStore...")
            self.vector_store = EnhancedPineconeStore()
            logger.debug("✅ EnhancedPineconeStore initialized")
            
            # Sparse retrieval
            logger.debug("Initializing LegalBM25Retriever...")
            self.bm25_retriever = LegalBM25Retriever()
            logger.debug("✅ LegalBM25Retriever initialized")
            
            # FIXED: Preload all models during initialization instead of lazy loading
            logger.debug("Preloading AdvancedHybridRetriever...")
            from src.advanced_hybrid_retrieval import AdvancedHybridRetriever
            self.hybrid_retriever = AdvancedHybridRetriever(
                self.vector_store,
                self.bm25_retriever,
                self.embedding_model,        # LegalEmbeddingModel instance
                self.metadata_embedding      # MetadataAwareEmbedding instance
            )
            logger.debug("✅ AdvancedHybridRetriever preloaded")
            
            logger.debug("Preloading MultiStageReranker...")
            from src.legal_reranking import MultiStageReranker
            self.reranker = MultiStageReranker()
            logger.debug("✅ MultiStageReranker preloaded")
            
            logger.debug("Preloading RhetoricallyAwareGenerator...")
            from src.context_aware_generation import RhetoricallyAwareGenerator
            self.generator = RhetoricallyAwareGenerator(
                device='cuda' if self.enable_gpu else 'cpu'
            )
            logger.debug("✅ RhetoricallyAwareGenerator preloaded")
            
            # Optional response evaluator
            self._response_evaluator = None
            
            logger.info("✅ All pipeline components initialized and preloaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _check_existing_indices(self):
        """Check for existing BM25 index and metadata files with logging"""
        
        logger.info("🔍 Checking for existing indices...")
        
        bm25_path = Path(settings.BM25_INDEX_FILE)
        metadata_path = Path(settings.CHUNKS_METADATA_FILE)
        
        logger.debug(f"BM25 index path: {bm25_path}")
        logger.debug(f"Metadata path: {metadata_path}")
        
        # Check BM25 index
        if bm25_path.exists():
            logger.info(f"📁 Found existing BM25 index: {bm25_path}")
            try:
                self.bm25_retriever.load_index(str(bm25_path))
                logger.info("✅ BM25 index loaded successfully")
                self.index_built = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to load BM25 index: {e}")
        else:
            logger.info(f"📁 BM25 index not found at: {bm25_path}")
        
        # Check metadata
        if metadata_path.exists():
            logger.info(f"📄 Found existing chunks metadata: {metadata_path}")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ Chunks metadata loaded ({len(metadata)} chunks)")
                self.is_initialized = True
            except Exception as e:
                logger.warning(f"⚠️ Could not load chunks metadata: {e}")
        else:
            logger.info(f"📄 Chunks metadata not found at: {metadata_path}")
        
        # Check Pinecone index
        try:
            logger.debug("Checking Pinecone index status...")
            stats = self.vector_store.get_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            if vector_count > 0:
                logger.info(f"✅ Pinecone index active with {vector_count} vectors")
                logger.info(f"   Dimension: {stats.get('dimension', 'Unknown')}")
                logger.info(f"   Index fullness: {stats.get('index_fullness', 'Unknown')}")
                self.index_built = True
                self.is_initialized = True
            else:
                logger.info("📊 Pinecone index is empty")
        except Exception as e:
            logger.warning(f"⚠️ Could not check Pinecone index: {e}")
        
        logger.info(f"🏁 Index check complete - Initialized: {self.is_initialized}, Built: {self.index_built}")
    
    async def build_index(self, data_directory: str, 
                         force_rebuild: bool = False) -> bool:
        """Build the complete index with comprehensive logging"""
        
        logger.info("🏗️ STARTING INDEX BUILD PROCESS")
        logger.info("=" * 50)
        logger.info(f"Data directory: {data_directory}")
        logger.info(f"Force rebuild: {force_rebuild}")
        
        if self.index_built and not force_rebuild:
            logger.info("✅ Index already built and force_rebuild=False, skipping...")
            return True
        
        start_time = time.time()
        
        try:
            # Validate data directory
            if not os.path.exists(data_directory):
                logger.error(f"❌ Data directory does not exist: {data_directory}")
                return False
            
            json_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]
            logger.info(f"📁 Found {len(json_files)} JSON files in data directory")
            
            if len(json_files) == 0:
                logger.error("❌ No JSON files found in data directory")
                return False
            
            # Step 1: Process documents into chunks
            logger.info("📝 Step 1: Processing documents into rhetorical chunks")
            step_start = time.time()
            
            chunks = self.document_processor.process_all_documents(data_directory)
            
            step_time = time.time() - step_start
            logger.info(f"✅ Document processing completed in {step_time:.2f}s")
            
            if not chunks:
                logger.error("❌ No chunks created from documents")
                return False
            
            logger.info(f"📊 Created {len(chunks)} rhetorical chunks")
            logger.info(f"   Average chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} characters")
            
            # Log chunk distribution by rhetorical role
            role_distribution = {}
            for chunk in chunks:
                role = chunk.metadata.primary_role
                role_distribution[role] = role_distribution.get(role, 0) + 1
            
            logger.info("📈 Chunk distribution by rhetorical role:")
            for role, count in sorted(role_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(chunks)) * 100
                logger.info(f"   {role}: {count} ({percentage:.1f}%)")
            
            # Step 2: Generate embeddings
            logger.info("🔢 Step 2: Generating embeddings with metadata enrichment")
            step_start = time.time()
            
            embeddings = self.metadata_embedding.encode_chunks_with_metadata(chunks)
            
            step_time = time.time() - step_start
            logger.info(f"✅ Embedding generation completed in {step_time:.2f}s")
            
            # Log embedding statistics
            def get_embedding_stats(embeddings):
                if embeddings is None:
                    return "None"
                if hasattr(embeddings, 'shape'):
                    if len(embeddings.shape) == 2:
                        return f"Shape: {embeddings.shape}, Dtype: {embeddings.dtype}"
                elif isinstance(embeddings, (list, tuple)) and len(embeddings) > 0:
                    return f"List of {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if hasattr(embeddings[0], '__len__') else 'Unknown'}"
                return f"Type: {type(embeddings)}"
            
            embedding_stats = get_embedding_stats(embeddings)
            logger.info(f"📊 Embedding statistics: {embedding_stats}")
            
            # Validate embeddings
            if embeddings is None or (hasattr(embeddings, 'size') and embeddings.size == 0):
                logger.error("❌ No embeddings generated")
                return False
            
            # Step 3: Index in vector store
            logger.info("🗃️ Step 3: Indexing in vector store")
            step_start = time.time()
            
            vector_success = self.vector_store.upsert_enriched_chunks(chunks, embeddings)
            
            step_time = time.time() - step_start
            logger.info(f"✅ Vector store indexing completed in {step_time:.2f}s")
            
            if not vector_success:
                logger.error("❌ Failed to index chunks in vector store")
                return False
            
            # Step 4: Build BM25 index
            logger.info("🔍 Step 4: Building BM25 sparse index")
            step_start = time.time()
            
            self.bm25_retriever.build_index(chunks)
            
            step_time = time.time() - step_start
            logger.info(f"✅ BM25 index building completed in {step_time:.2f}s")
            
            # Step 5: Save indices
            logger.info("💾 Step 5: Saving indices to disk")
            step_start = time.time()
            
            os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
            
            # Save BM25 index
            bm25_index_path = f"{settings.PROCESSED_DATA_DIR}/{settings.BM25_INDEX_FILE}"
            self.bm25_retriever.save_index(bm25_index_path)
            logger.info(f"💾 BM25 index saved to: {bm25_index_path}")
            
            # Save chunk metadata
            chunks_metadata_path = f"{settings.PROCESSED_DATA_DIR}/{settings.CHUNKS_METADATA_FILE}"
            self._save_chunks_metadata(chunks, chunks_metadata_path)
            logger.info(f"💾 Chunks metadata saved to: {chunks_metadata_path}")
            
            step_time = time.time() - step_start
            logger.info(f"✅ Index saving completed in {step_time:.2f}s")
            
            # Final statistics
            total_time = time.time() - start_time
            
            logger.info("🎉 INDEX BUILD COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            logger.info(f"⏱️ Total time: {total_time:.2f}s")
            logger.info(f"📊 Documents processed: {len(json_files)}")
            logger.info(f"📊 Chunks created: {len(chunks)}")
            logger.info(f"📊 Embeddings generated: {len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'}")
            logger.info(f"📊 Processing rate: {len(chunks) / total_time:.1f} chunks/second")
            logger.info("=" * 50)
            
            self.index_built = True
            self.is_initialized = True
            
            return True
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error("❌ INDEX BUILD FAILED")
            logger.error("=" * 50)
            logger.error(f"⏱️ Failed after: {total_time:.2f}s")
            logger.error(f"❌ Error: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            logger.error("=" * 50)
            return False
    
    def _save_chunks_metadata(self, chunks: List[EnrichedChunk], file_path: str):
        """Save chunks metadata with detailed logging"""
        
        logger.debug(f"Saving chunks metadata to: {file_path}")
        
        try:
            chunks_data = []
            for chunk in chunks:
                chunk_data = {
                    'id': chunk.id,
                    'document_id': chunk.metadata.document_id,
                    'rhetorical_roles': chunk.metadata.rhetorical_roles,
                    'primary_role': chunk.metadata.primary_role,
                    'entity_types': chunk.metadata.entity_types,
                    'entity_count': chunk.metadata.entity_count,
                    'precedent_count': chunk.metadata.precedent_count,
                    'statute_count': chunk.metadata.statute_count,
                    'provision_count': chunk.metadata.provision_count,
                    'legal_concepts': chunk.legal_concepts,
                    'keywords': chunk.keywords,
                    'text_length': len(chunk.text),
                    'source_file': chunk.metadata.source_file
                }
                chunks_data.append(chunk_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"✅ Chunks metadata saved ({file_size:.1f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Error saving chunks metadata: {e}")
            raise
    
    async def process_query(self, query: str, search_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query with comprehensive logging and metrics"""
        
        query_start_time = time.time()
        query_id = self.metrics['queries_processed'] + 1
        
        logger.info(f"🔍 PROCESSING QUERY #{query_id}")
        logger.info(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"Options: {search_options}")
        
        if not self.is_initialized:
            error_msg = 'Pipeline not initialized. Please build index first.'
            logger.error(f"❌ {error_msg}")
            return {
                'query': query,
                'response': error_msg,
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - query_start_time
            }
        
        try:
            # Update metrics
            self.metrics['queries_processed'] += 1
            
            # Parse search options
            search_options = search_options or {}
            search_query = self._create_search_query(query, search_options)
            
            logger.debug(f"Search query object: {search_query}")
            
            # Step 1: Hybrid Retrieval
            logger.info("📚 Step 1: Executing hybrid retrieval")
            step1_start = time.time()
            
            retrieval_results = self.hybrid_retriever.hybrid_search(
                query=query,
                search_query=search_query,
                top_k=search_options.get('retrieval_top_k', settings.TOP_K_RETRIEVAL),
                fusion_method=search_options.get('fusion_method', 'weighted_sum')
            )
            
            step1_time = time.time() - step1_start
            logger.info(f"✅ Retrieval completed in {step1_time:.2f}s ({len(retrieval_results)} results)")
            
            if not retrieval_results:
                return self._create_empty_response(query, query_start_time, "No relevant documents found")
            
            # Convert results to EnrichedChunk objects
            retrieved_chunks = self._convert_results_to_chunks(retrieval_results)
            logger.debug(f"Converted to {len(retrieved_chunks)} enriched chunks")
            
            # Step 2: Multi-stage Reranking
            logger.info("🔄 Step 2: Executing multi-stage reranking")
            step2_start = time.time()
            
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
            
            step2_time = time.time() - step2_start
            logger.info(f"✅ Reranking completed in {step2_time:.2f}s")
            logger.debug(f"Reranking explanation: {reranking_result.reranking_explanation}")
            
            # Select top chunks for generation
            top_k_generation = search_options.get('generation_top_k', settings.TOP_K_RERANK)
            final_chunks = reranking_result.chunks[:top_k_generation]
            
            logger.info(f"📝 Using top {len(final_chunks)} chunks for generation")
            
            # Step 3: Context-Aware Generation
            logger.info("✍️ Step 3: Executing context-aware generation")
            step3_start = time.time()
            
            generation_result = self.generator.generate_legal_response(
                query=query,
                chunks=final_chunks,
                max_length=search_options.get('max_generation_length'),
                temperature=search_options.get('temperature')
            )
            
            step3_time = time.time() - step3_start
            logger.info(f"✅ Generation completed in {step3_time:.2f}s")
            
            # Calculate metrics
            total_time = time.time() - query_start_time
            self.metrics['total_processing_time'] += total_time
            self.metrics['average_processing_time'] = self.metrics['total_processing_time'] / self.metrics['queries_processed']
            self.metrics['last_query_time'] = total_time
            
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
                'precedent_references': [],  # Could be extracted from chunks
                'statute_references': [],    # Could be extracted from chunks
                'quality_metrics': {},       # Could be added later
                'pipeline_insights': {
                    'retrieval_method': 'hybrid',
                    'reranking_strategy': search_options.get('reranking_strategy', 'auto'),
                    'generation_template': generation_result.rhetorical_structure.get('template_used', 'general')
                },
                'related_queries': [],       # Could be generated
                
                # Performance breakdown
                'performance_metrics': {
                    'retrieval_time': step1_time,
                    'reranking_time': step2_time,
                    'generation_time': step3_time,
                    'total_time': total_time
                },
                
                'success': True
            }
            
            logger.info(f"✅ QUERY #{query_id} PROCESSED SUCCESSFULLY")
            logger.info(f"⏱️ Total time: {total_time:.2f}s")
            logger.info(f"📊 Performance: Retrieval({step1_time:.2f}s) + Reranking({step2_time:.2f}s) + Generation({step3_time:.2f}s)")
            logger.info(f"✨ Confidence: {generation_result.confidence_score:.3f}")
            
            return response
            
        except Exception as e:
            error_time = time.time() - query_start_time
            self.metrics['errors_count'] += 1
            
            logger.error(f"❌ QUERY #{query_id} PROCESSING FAILED")
            logger.error(f"⏱️ Failed after: {error_time:.2f}s")
            logger.error(f"❌ Error: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            
            return {
                'query': query,
                'response': f"I apologize, but I encountered an error while processing your legal query: {str(e)}",
                'error': str(e),
                'processing_time': error_time,
                'success': False
            }
    
    def _create_search_query(self, query: str, options: Dict[str, Any]) -> SearchQuery:
        """Create SearchQuery object with logging"""
        
        search_query = SearchQuery(
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
        
        logger.debug(f"Created search query: {search_query}")
        return search_query
    
    def _convert_results_to_chunks(self, results: List[Dict[str, Any]]) -> List[EnrichedChunk]:
        """Convert search results to EnrichedChunk objects with logging"""
        
        logger.debug(f"Converting {len(results)} results to enriched chunks")
        
        chunks = []
        for i, result in enumerate(results):
            try:
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
                
            except Exception as e:
                logger.warning(f"⚠️ Error converting result {i} to chunk: {e}")
                continue
        
        logger.debug(f"Successfully converted {len(chunks)} chunks")
        return chunks
    
    def _create_empty_response(self, query: str, start_time: float, message: str) -> Dict[str, Any]:
        """Create response for cases with no results"""
        
        logger.warning(f"Creating empty response: {message}")
        
        return {
            'query': query,
            'response': message,
            'confidence_score': 0.0,
            'processing_time': time.time() - start_time,
            'contexts_used': 0,
            'supporting_documents': [],
            'legal_citations': [],
            'precedent_references': [],
            'statute_references': [],
            'quality_metrics': {},
            'pipeline_insights': {},
            'performance_metrics': {},
            'related_queries': [],
            'success': False
        }
    
    # FIXED: Add missing health_check method
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all pipeline components"""
        
        logger.debug("Performing health check...")
        
        health_data = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        try:
            # Check pipeline initialization
            health_data['checks']['pipeline_initialized'] = self.is_initialized
            health_data['checks']['index_built'] = self.index_built
            
            # Check vector store
            try:
                vector_stats = self.vector_store.get_index_stats()
                health_data['checks']['vector_store'] = {
                    'status': 'healthy',
                    'vector_count': vector_stats.get('total_vector_count', 0)
                }
            except Exception as e:
                health_data['checks']['vector_store'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_data['status'] = 'degraded'
            
            # Check BM25 retriever
            health_data['checks']['bm25_retriever'] = {
                'status': 'healthy' if self.bm25_retriever.bm25 is not None else 'not_initialized'
            }
            
            # Check embedding model
            health_data['checks']['embedding_model'] = {
                'status': 'healthy',
                'model_name': self.embedding_model.model_name,
                'device': self.embedding_model.device
            }
            
            # Performance metrics
            health_data['checks']['performance'] = self.metrics
            
            # System resources
            health_data['checks']['system'] = {
                'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                'cpu_count': psutil.cpu_count(),
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_data['status'] = 'unhealthy'
            health_data['checks']['error'] = str(e)
        
        return health_data
    
    # FIXED: Add missing batch_process_queries method
    async def batch_process_queries(self, queries: List[str], options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        
        logger.info(f"Processing batch of {len(queries)} queries")
        batch_start_time = time.time()
        
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                result = await self.process_query(query, options)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch query {i+1}: {e}")
                results.append({
                    'query': query,
                    'response': f"Error processing query: {str(e)}",
                    'success': False,
                    'error': str(e),
                    'processing_time': 0.0
                })
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Batch processing completed in {batch_time:.2f}s")
        
        return results
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics with logging"""
        
        logger.debug("Gathering pipeline statistics...")
        
        try:
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
                'performance_metrics': self.metrics,
                'system_resources': {
                    'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                    'cpu_count': psutil.cpu_count(),
                    'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
                }
            }
            
            # Add vector store stats
            try:
                vector_stats = self.vector_store.get_index_stats()
                stats['vector_store'] = vector_stats
                logger.debug(f"Vector store stats: {vector_stats}")
            except Exception as e:
                logger.warning(f"Could not get vector store stats: {e}")
                stats['vector_store'] = {'error': str(e)}
            
            logger.debug("Pipeline statistics gathered successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error gathering pipeline statistics: {e}")
            return {'error': str(e)}

    def debug_and_fix_embeddings(self):
        """Debug and fix embedding dimension issues"""
        logger.info("🔍 Debugging embedding dimensions...")
        
        # Test embeddings
        test_chunk = EnrichedChunk(
            id="test",
            text="This is a test legal document about arbitration.",
            cleaned_text="This is a test legal document about arbitration.",
            metadata=LegalMetadata(
                document_id="test",
                chunk_id="test_0",
                chunk_index=0,
                rhetorical_roles=["FAC"],
                primary_role="FAC",
                entities=[],
                entity_types=["STATUTE"],
                entity_count=1,
                precedent_count=0,
                statute_count=1,
                provision_count=0,
                source_file="test.json",
                original_start=0,
                original_end=100
            ),
            keywords=["arbitration"],
            legal_concepts=["arbitration agreement"]
        )
        
        # Test text embedding
        text_emb = self.embedding_model.encode_chunks([test_chunk], use_enrichment=False)
        logger.info(f"Text embedding shape: {text_emb.shape}")
        
        # Test metadata-aware embedding
        meta_emb = self.metadata_embedding.encode_chunks_with_metadata([test_chunk])
        logger.info(f"Metadata embedding shape: {meta_emb.shape}")
        
        # Check Pinecone
        try:
            stats = self.vector_store.get_index_stats()
            logger.info(f"Pinecone dimension: {stats.get('dimension', 'Unknown')}")
            logger.info(f"Vector count: {stats.get('total_vector_count', 0)}")
        except Exception as e:
            logger.error(f"Pinecone stats error: {e}")
        
        return meta_emb.shape[1] if len(meta_emb.shape) > 1 else 0
