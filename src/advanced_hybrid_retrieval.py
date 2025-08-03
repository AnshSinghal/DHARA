import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.embeddings import LegalEmbeddingModel, MetadataAwareEmbedding
from src.vector_store import EnhancedPineconeStore 
from src.legal_bm25 import LegalBM25Retriever
from src.multi_faceted_search import LegalMultiFacetedSearch, AdvancedQueryProcessor
from src.data_models import SearchQuery, EnrichedChunk
from config.settings import settings

logger = logging.getLogger(__name__)

class AdvancedHybridRetriever:
    """Advanced hybrid retrieval combining dense, sparse, and metadata-aware search"""
    
    def __init__(self, vector_store: EnhancedPineconeStore, 
                 bm25_retriever: LegalBM25Retriever,
                 embedding_model: LegalEmbeddingModel):
        
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        
        # Initialize search components
        self.faceted_search = LegalMultiFacetedSearch(vector_store, embedding_model)
        self.query_processor = AdvancedQueryProcessor()
        
        # Metadata-aware embedding for enhanced dense retrieval
        self.metadata_embedding = MetadataAwareEmbedding(embedding_model)
        
        # Retrieval weights (can be dynamically adjusted)
        self.default_weights = {
            'dense': 0.5,
            'sparse': 0.3,
            'metadata': 0.2
        }
    
    def adaptive_weight_calculation(self, query: str, 
                                  search_context: Dict[str, Any] = None) -> Dict[str, float]:
        """Dynamically calculate retrieval weights based on query and context"""
        
        weights = self.default_weights.copy()
        query_lower = query.lower()
        
        # Boost sparse retrieval for exact term queries
        if any(pattern in query_lower for pattern in ['section ', 'article ', 'case number']):
            weights['sparse'] += 0.2
            weights['dense'] -= 0.1
            weights['metadata'] -= 0.1
        
        # Boost dense retrieval for conceptual queries
        elif any(pattern in query_lower for pattern in ['similar to', 'like', 'concept of']):
            weights['dense'] += 0.2
            weights['sparse'] -= 0.1
            weights['metadata'] -= 0.1
        
        # Boost metadata retrieval for filtered queries
        if search_context and (search_context.get('rhetorical_roles') or 
                              search_context.get('entity_types')):
            weights['metadata'] += 0.15
            weights['dense'] -= 0.075
            weights['sparse'] -= 0.075
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def execute_dense_retrieval(self, query: str, search_query: SearchQuery,
                               top_k: int = 20) -> List[Dict[str, Any]]:
        """Execute dense vector retrieval with metadata enrichment"""
        
        # Create query context for embeddings
        query_context = {
            'rhetorical_roles': search_query.rhetorical_roles,
            'entity_types': search_query.entity_types
        }
        
        # Encode query with context
        query_vector = self.embedding_model.encode_query(query, query_context)
        
        # Execute vector search with filters
        results = self.vector_store.search_with_filters(query_vector, search_query)
        
        # Add search type identifier
        for result in results:
            result['search_type'] = 'dense'
            result['dense_score'] = result['score']
        
        return results[:top_k]
    
    def execute_sparse_retrieval(self, query: str, search_query: SearchQuery,
                                top_k: int = 20) -> List[Dict[str, Any]]:
        """Execute BM25 sparse retrieval with legal optimization"""
        
        # Create query context for BM25
        query_context = {
            'rhetorical_roles': search_query.rhetorical_roles,
            'entity_types': search_query.entity_types
        }
        
        # Execute BM25 search
        results = self.bm25_retriever.search(query, top_k, query_context)
        
        # Add search type identifier  
        for result in results:
            result['search_type'] = 'sparse'
            result['sparse_score'] = result['score']
        
        return results
    
    def execute_metadata_retrieval(self, query: str, search_query: SearchQuery,
                                  top_k: int = 20) -> List[Dict[str, Any]]:
        """Execute metadata-aware retrieval using faceted search"""
        
        # Use faceted search for metadata-aware retrieval
        faceted_result = self.faceted_search.execute_faceted_search(search_query)
        
        results = []
        for chunk in faceted_result.chunks[:top_k]:
            result = {
                'id': chunk.id,
                'score': 1.0,  # Placeholder score
                'text': chunk.text,
                'search_type': 'metadata',
                'metadata_score': 1.0,
                'metadata': {
                    'document_id': chunk.metadata.document_id,
                    'primary_role': chunk.metadata.primary_role,
                    'entity_types': chunk.metadata.entity_types,
                    'precedent_count': chunk.metadata.precedent_count,
                    'statute_count': chunk.metadata.statute_count
                }
            }
            results.append(result)
        
        return results
    
    def normalize_scores(self, results: List[Dict[str, Any]], 
                        method: str = "min_max") -> List[Dict[str, Any]]:
        """Normalize scores across different retrieval methods"""
        
        if not results:
            return results
        
        scores = [result['score'] for result in results]
        
        if method == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for result in results:
                    result['normalized_score'] = (result['score'] - min_score) / score_range
            else:
                for result in results:
                    result['normalized_score'] = 1.0
        
        elif method == "z_score":
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            if std_score > 0:
                for result in results:
                    result['normalized_score'] = (result['score'] - mean_score) / std_score
            else:
                for result in results:
                    result['normalized_score'] = 0.0
        
        elif method == "softmax":
            exp_scores = np.exp(np.array(scores) - np.max(scores))  # Numerical stability
            softmax_scores = exp_scores / np.sum(exp_scores)
            
            for i, result in enumerate(results):
                result['normalized_score'] = float(softmax_scores[i])
        
        return results
    
    def fusion_strategies(self, dense_results: List[Dict[str, Any]],
                         sparse_results: List[Dict[str, Any]], 
                         metadata_results: List[Dict[str, Any]],
                         weights: Dict[str, float],
                         fusion_method: str = "weighted_sum") -> List[Dict[str, Any]]:
        """Advanced fusion strategies for combining retrieval results"""
        
        # Normalize scores within each method
        dense_results = self.normalize_scores(dense_results, "min_max")
        sparse_results = self.normalize_scores(sparse_results, "min_max") 
        metadata_results = self.normalize_scores(metadata_results, "min_max")
        
        # Create combined results dictionary
        combined = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result['id']
            combined[doc_id] = {
                **result,
                'dense_score': result['normalized_score'],
                'sparse_score': 0.0,
                'metadata_score': 0.0,
                'combined_score': weights['dense'] * result['normalized_score'],
                'fusion_components': ['dense']
            }
        
        # Add sparse results
        for result in sparse_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['sparse_score'] = result['normalized_score']
                combined[doc_id]['combined_score'] += weights['sparse'] * result['normalized_score']
                combined[doc_id]['fusion_components'].append('sparse')
                combined[doc_id]['search_type'] = 'hybrid'
            else:
                combined[doc_id] = {
                    **result,
                    'dense_score': 0.0,
                    'sparse_score': result['normalized_score'],
                    'metadata_score': 0.0,
                    'combined_score': weights['sparse'] * result['normalized_score'],
                    'fusion_components': ['sparse']
                }
        
        # Add metadata results
        for result in metadata_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['metadata_score'] = result['normalized_score']
                combined[doc_id]['combined_score'] += weights['metadata'] * result['normalized_score']
                combined[doc_id]['fusion_components'].append('metadata')
                combined[doc_id]['search_type'] = 'hybrid'
            else:
                combined[doc_id] = {
                    **result,
                    'dense_score': 0.0,
                    'sparse_score': 0.0,
                    'metadata_score': result['normalized_score'],
                    'combined_score': weights['metadata'] * result['normalized_score'],
                    'fusion_components': ['metadata']
                }
        
        # Apply fusion method
        if fusion_method == "rrf":  # Reciprocal Rank Fusion
            combined = self._apply_reciprocal_rank_fusion(
                dense_results, sparse_results, metadata_results, weights
            )
        elif fusion_method == "comb_sum":
            # Already implemented above as weighted sum
            pass
        elif fusion_method == "comb_max":
            for doc_id, doc_data in combined.items():
                scores = [doc_data['dense_score'] * weights['dense'],
                         doc_data['sparse_score'] * weights['sparse'], 
                         doc_data['metadata_score'] * weights['metadata']]
                doc_data['combined_score'] = max(scores)
        
        # Sort by combined score
        final_results = sorted(combined.values(), 
                             key=lambda x: x['combined_score'], 
                             reverse=True)
        
        # Add final ranking
        for i, result in enumerate(final_results):
            result['final_rank'] = i + 1
        
        return final_results
    
    def _apply_reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]],
                                    sparse_results: List[Dict[str, Any]],
                                    metadata_results: List[Dict[str, Any]],
                                    weights: Dict[str, float],
                                    k: int = 60) -> Dict[str, Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion (RRF) algorithm"""
        
        rrf_scores = {}
        
        # Create rank mappings
        result_sets = [
            (dense_results, 'dense', weights['dense']),
            (sparse_results, 'sparse', weights['sparse']),
            (metadata_results, 'metadata', weights['metadata'])
        ]
        
        for results, search_type, weight in result_sets:
            for rank, result in enumerate(results, 1):
                doc_id = result['id']
                rrf_score = weight * (1 / (k + rank))
                
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {
                        **result,
                        'rrf_score': rrf_score,
                        'rank_contributions': {search_type: rank}
                    }
                else:
                    rrf_scores[doc_id]['rrf_score'] += rrf_score
                    rrf_scores[doc_id]['rank_contributions'][search_type] = rank
        
        # Update combined scores to use RRF
        for doc_data in rrf_scores.values():
            doc_data['combined_score'] = doc_data['rrf_score']
        
        return rrf_scores
    
    def hybrid_search(self, query: str, search_query: SearchQuery,
                     top_k: int = 10, fusion_method: str = "weighted_sum") -> List[Dict[str, Any]]:
        """Execute full hybrid search with adaptive weighting"""
        
        start_time = time.time()
        
        # Calculate adaptive weights
        search_context = {
            'rhetorical_roles': search_query.rhetorical_roles,
            'entity_types': search_query.entity_types
        }
        weights = self.adaptive_weight_calculation(query, search_context)
        
        logger.info(f"Hybrid search with weights: {weights}")
        
        # Execute parallel retrieval
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit retrieval tasks
            dense_future = executor.submit(
                self.execute_dense_retrieval, query, search_query, top_k * 2
            )
            sparse_future = executor.submit(
                self.execute_sparse_retrieval, query, search_query, top_k * 2
            )
            metadata_future = executor.submit(
                self.execute_metadata_retrieval, query, search_query, top_k * 2
            )
            
            # Collect results
            dense_results = []
            sparse_results = []
            metadata_results = []
            
            for future in as_completed([dense_future, sparse_future, metadata_future]):
                try:
                    if future == dense_future:
                        dense_results = future.result()
                    elif future == sparse_future:
                        sparse_results = future.result()
                    else:
                        metadata_results = future.result()
                except Exception as e:
                    logger.error(f"Error in parallel retrieval: {e}")
        
        # Fusion
        fused_results = self.fusion_strategies(
            dense_results, sparse_results, metadata_results, 
            weights, fusion_method
        )
        
        # Select top-k
        final_results = fused_results[:top_k]
        
        # Add timing and metadata
        search_time = time.time() - start_time
        for result in final_results:
            result['search_time'] = search_time
            result['fusion_method'] = fusion_method
            result['adaptive_weights'] = weights
        
        logger.info(f"Hybrid search completed in {search_time:.3f}s")
        
        return final_results
    
    def explain_retrieval_decision(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide explanation for retrieval decisions"""
        
        if not results:
            return {"explanation": "No results found"}
        
        explanation = {
            "total_results": len(results),
            "search_methods_used": [],
            "fusion_analysis": {},
            "top_result_analysis": {},
            "retrieval_insights": []
        }
        
        # Analyze search methods used
        search_types = set()
        for result in results:
            if 'fusion_components' in result:
                search_types.update(result['fusion_components'])
            else:
                search_types.add(result.get('search_type', 'unknown'))
        
        explanation["search_methods_used"] = list(search_types)
        
        # Analyze top result
        top_result = results
        explanation["top_result_analysis"] = {
            "document_id": top_result.get('metadata', {}).get('document_id', 'unknown'),
            "combined_score": top_result.get('combined_score', 0),
            "fusion_components": top_result.get('fusion_components', []),
            "rhetorical_role": top_result.get('metadata', {}).get('primary_role', 'unknown')
        }
        
        # Generate insights
        insights = []
        
        if 'dense' in search_types and 'sparse' in search_types:
            insights.append("Hybrid approach used both semantic and lexical matching")
        
        if any('metadata' in result.get('fusion_components', []) for result in results):
            insights.append("Metadata filtering helped narrow results")
        
        high_precedent_results = [r for r in results 
                                if r.get('metadata', {}).get('precedent_count', 0) > 5]
        if high_precedent_results:
            insights.append(f"{len(high_precedent_results)} results have high precedent density")
        
        explanation["retrieval_insights"] = insights
        
        return explanation
