import torch
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np
import logging
import time
from dataclasses import dataclass

from src.data_models import EnrichedChunk, SearchQuery, RerankingResult
from config.settings import settings

logger = logging.getLogger(__name__)

class LegalAwareCrossEncoder:
    """Cross-encoder with legal domain awareness and metadata integration"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading legal-aware cross-encoder: {self.model_name}")
        
        # Load cross-encoder
        self.model = CrossEncoder(self.model_name, device=self.device)
        
        # Legal-specific scoring weights
        self.scoring_weights = {
            'base_relevance': 1.0,
            'rhetorical_role_match': 0.15,
            'entity_overlap': 0.10,
            'precedent_density': 0.08,
            'statute_density': 0.07,
            'legal_concept_match': 0.12
        }
        
        logger.info("Legal-aware cross-encoder initialized")
    
    def rerank_chunks(self, query: str, chunks: List[EnrichedChunk], 
                     query_context: Dict[str, Any] = None,
                     top_k: int = None) -> RerankingResult:
        """Comprehensive reranking with metadata awareness"""
        
        start_time = time.time()
        
        if not chunks:
            return RerankingResult(
                chunks=[], original_scores=[], rerank_scores=[],
                position_changes=[], processing_time=0.0,
                reranking_explanation="No chunks to rerank"
            )
        
        if top_k is None:
            top_k = len(chunks)
        
        # FIXED: Store original scores using getattr instead of .get()
        original_scores = [getattr(chunk.metadata, 'retrieval_score', 0.0) for chunk in chunks]
        
        # Create enhanced query-document pairs
        enhanced_pairs = self._create_enhanced_pairs(query, chunks, query_context)
        
        # Compute base cross-encoder scores
        base_scores = self._compute_base_scores(enhanced_pairs)
        
        # Compute metadata boosts
        metadata_boosts = self._compute_metadata_boosts(query, chunks, query_context)
        
        # Combine scores
        final_scores = [
            self.scoring_weights['base_relevance'] * base_score + boost
            for base_score, boost in zip(base_scores, metadata_boosts)
        ]
        
        # Create scored chunks with reranking info
        scored_chunks = []
        for i, (chunk, score) in enumerate(zip(chunks, final_scores)):
            # FIXED: Add reranking metadata using direct attribute assignment
            chunk.metadata.rerank_score = score
            chunk.metadata.original_position = i
            chunk.metadata.base_cross_encoder_score = base_scores[i]
            chunk.metadata.metadata_boost = metadata_boosts[i]
            
            scored_chunks.append((chunk, score))
        
        # Sort by final scores
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked chunks and calculate position changes
        reranked_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]
        rerank_scores = [score for _, score in scored_chunks[:top_k]]
        
        position_changes = []
        for new_pos, (chunk, _) in enumerate(scored_chunks[:top_k]):
            old_pos = getattr(chunk.metadata, 'original_position', new_pos)
            position_changes.append(new_pos - old_pos)
            chunk.metadata.new_position = new_pos
        
        # Generate explanation
        explanation = self._generate_reranking_explanation(
            query, reranked_chunks, position_changes, query_context
        )
        
        processing_time = time.time() - start_time
        
        return RerankingResult(
            chunks=reranked_chunks,
            original_scores=original_scores[:top_k],
            rerank_scores=rerank_scores,
            position_changes=position_changes,
            processing_time=processing_time,
            reranking_explanation=explanation
        )
    
    def _create_enhanced_pairs(self, query: str, chunks: List[EnrichedChunk], 
                             query_context: Dict[str, Any] = None) -> List[tuple]:
        """Create query-document pairs with metadata enrichment"""
        
        enhanced_pairs = []
        
        for chunk in chunks:
            # Start with base text
            enhanced_doc = chunk.cleaned_text
            
            # Add rhetorical role context
            if chunk.metadata.rhetorical_roles:
                role_context = f"[{chunk.metadata.primary_role}] "
                enhanced_doc = role_context + enhanced_doc
            
            # Add legal concept context
            if chunk.legal_concepts:
                concept_context = f"[Legal concepts: {', '.join(chunk.legal_concepts[:2])}] "
                enhanced_doc = concept_context + enhanced_doc
            
            # Enhance query with context if provided
            enhanced_query = query
            if query_context:
                if query_context.get('rhetorical_roles'):
                    role_hint = f"[Looking for: {', '.join(query_context['rhetorical_roles'])}] "
                    enhanced_query = role_hint + query
            
            enhanced_pairs.append((enhanced_query, enhanced_doc))
        
        return enhanced_pairs
    
    def _compute_base_scores(self, query_doc_pairs: List[tuple], 
                           batch_size: int = 16) -> List[float]:
        """Compute base cross-encoder scores"""
        
        all_scores = []
        
        # Process in batches for efficiency
        for i in range(0, len(query_doc_pairs), batch_size):
            batch_pairs = query_doc_pairs[i:i + batch_size]
            
            try:
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores.tolist())
            except Exception as e:
                logger.error(f"Error in cross-encoder batch {i//batch_size}: {e}")
                # Fallback to zero scores
                all_scores.extend([0.0] * len(batch_pairs))
        
        return all_scores
    
    def _compute_metadata_boosts(self, query: str, chunks: List[EnrichedChunk],
                               query_context: Dict[str, Any] = None) -> List[float]:
        """Compute metadata-based relevance boosts"""
        
        boosts = []
        query_lower = query.lower()
        
        for chunk in chunks:
            boost = 0.0
            
            # Rhetorical role matching boost
            if query_context and query_context.get('rhetorical_roles'):
                if chunk.metadata.primary_role in query_context['rhetorical_roles']:
                    boost += self.scoring_weights['rhetorical_role_match']
            
            # Entity overlap boost
            if query_context and query_context.get('entity_types'):
                entity_overlap = set(chunk.metadata.entity_types) & set(query_context['entity_types'])
                if entity_overlap:
                    boost += self.scoring_weights['entity_overlap'] * len(entity_overlap)
            
            # UPDATED: Precedent density boost using correct role names
            if any(term in query_lower for term in ['precedent', 'case law', 'decided', 'held']):
                if any(role in chunk.metadata.rhetorical_roles for role in PRECEDENT_ROLES):
                    boost += self.scoring_weights['precedent_density'] * 1.5
                elif chunk.metadata.precedent_count > 3:
                    boost += self.scoring_weights['precedent_density'] * min(chunk.metadata.precedent_count / 10, 1.0)
            
            # UPDATED: Statute density boost using correct entity and role names
            if any(term in query_lower for term in ['section', 'act', 'statute', 'provision']):
                if 'STA' in chunk.metadata.rhetorical_roles or 'STATUTE' in chunk.metadata.entity_types:
                    boost += self.scoring_weights['statute_density'] * 1.5
                elif chunk.metadata.statute_count > 2:
                    boost += self.scoring_weights['statute_density'] * min(chunk.metadata.statute_count / 5, 1.0)
            
            # Legal concept matching boost
            query_tokens = set(query_lower.split())
            concept_matches = sum(1 for concept in chunk.legal_concepts 
                                if any(token in concept.lower() for token in query_tokens))
            if concept_matches > 0:
                boost += self.scoring_weights['legal_concept_match'] * min(concept_matches / 5, 1.0)
            
            boosts.append(boost)
        
        return boosts
    
    def _generate_reranking_explanation(self, query: str, reranked_chunks: List[EnrichedChunk],
                                      position_changes: List[int], 
                                      query_context: Dict[str, Any] = None) -> str:
        """Generate human-readable explanation of reranking decisions"""
        
        if not reranked_chunks:
            return "No chunks available for reranking analysis."
        
        explanation_parts = []
        
        # Overall statistics
        moved_up = len([change for change in position_changes if change < 0])
        moved_down = len([change for change in position_changes if change > 0])
        stayed_same = len([change for change in position_changes if change == 0])
        
        explanation_parts.append(f"Reranking Results Summary:")
        explanation_parts.append(f"• {moved_up} documents moved up in ranking")
        explanation_parts.append(f"• {moved_down} documents moved down in ranking") 
        explanation_parts.append(f"• {stayed_same} documents maintained position")
        
        # Top result analysis
        top_chunk = reranked_chunks[0]
        explanation_parts.append(f"\nTop Result Analysis:")
        explanation_parts.append(f"• Document: {top_chunk.metadata.document_id}")
        explanation_parts.append(f"• Rhetorical role: {top_chunk.metadata.primary_role}")
        explanation_parts.append(f"• Position change: {position_changes[0]:+d}")
        explanation_parts.append(f"• Final score: {getattr(top_chunk.metadata, 'rerank_score', 0.0):.3f}")
        
        return "\n".join(explanation_parts)

class MultiStageReranker:
    """Multi-stage reranking pipeline with different strategies"""
    
    def __init__(self):
        self.legal_reranker = LegalAwareCrossEncoder()
        
        # Define reranking strategies
        self.strategies = {
            'legal_general': self._legal_general_reranking,
            'precedent_focused': self._precedent_focused_reranking,
            'statutory_focused': self._statutory_focused_reranking,
            'factual_focused': self._factual_focused_reranking,
            'reasoning_focused': self._reasoning_focused_reranking,  # NEW
            'argument_focused': self._argument_focused_reranking     # NEW
        }
    
    def rerank_with_strategy(self, query: str, chunks: List[EnrichedChunk],
                           query_context: Dict[str, Any] = None,
                           strategy: str = None) -> RerankingResult:
        """Execute reranking with specified or auto-determined strategy"""
        
        if strategy is None or strategy == 'auto':
            strategy = self._determine_strategy(query, query_context)
        
        logger.info(f"Using reranking strategy: {strategy}")
        
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy {strategy}, using legal_general")
            strategy = 'legal_general'
        
        return self.strategies[strategy](query, chunks, query_context)
    
    def _determine_strategy(self, query: str, query_context: Dict[str, Any] = None) -> str:
        """Determine optimal reranking strategy based on query and context"""
        
        query_lower = query.lower()
        
        # Precedent-focused strategy
        if any(term in query_lower for term in ['precedent', 'case law', 'decided', 'held', 'ruling']):
            return 'precedent_focused'
        
        # Statutory-focused strategy
        elif any(term in query_lower for term in ['section', 'act', 'statute', 'provision', 'law']):
            return 'statutory_focused'
        
        # Factual-focused strategy
        elif any(term in query_lower for term in ['facts', 'circumstances', 'background', 'incident']):
            return 'factual_focused'
        
        # Reasoning-focused strategy
        elif any(term in query_lower for term in ['analysis', 'reasoning', 'rationale', 'ratio']):
            return 'reasoning_focused'
        
        # Argument-focused strategy
        elif any(term in query_lower for term in ['argument', 'contention', 'submission', 'petition']):
            return 'argument_focused'
        
        # Default to general legal strategy
        else:
            return 'legal_general'
    
    def _legal_general_reranking(self, query: str, chunks: List[EnrichedChunk],
                                query_context: Dict[str, Any] = None) -> RerankingResult:
        """General legal reranking strategy"""
        return self.legal_reranker.rerank_chunks(query, chunks, query_context)
    
    def _precedent_focused_reranking(self, query: str, chunks: List[EnrichedChunk],
                                   query_context: Dict[str, Any] = None) -> RerankingResult:
        """Precedent-focused reranking strategy"""
        
        # UPDATED: Boost precedent-heavy chunks using correct role names
        for chunk in chunks:
            precedent_boost = False
            
            # Check for precedent-related roles
            if any(role in chunk.metadata.rhetorical_roles for role in PRECEDENT_ROLES):
                precedent_boost = True
            elif chunk.metadata.precedent_count > 5:
                precedent_boost = True
            
            if precedent_boost:
                original_score = getattr(chunk.metadata, 'retrieval_score', 0.5)
                chunk.metadata.retrieval_score = min(original_score * 1.2, 1.0)
        
        # Add precedent context to query context
        if query_context is None:
            query_context = {}
        query_context['focus'] = 'precedent'
        query_context['preferred_roles'] = PRECEDENT_ROLES
        
        return self.legal_reranker.rerank_chunks(query, chunks, query_context)
    
    def _statutory_focused_reranking(self, query: str, chunks: List[EnrichedChunk],
                                   query_context: Dict[str, Any] = None) -> RerankingResult:
        """Statutory-focused reranking strategy"""
        
        # UPDATED: Boost statute-heavy chunks using correct role and entity names
        for chunk in chunks:
            statute_boost = False
            
            # Check for statute-related roles and entities
            if 'STA' in chunk.metadata.rhetorical_roles:
                statute_boost = True
            elif any(entity_type in chunk.metadata.entity_types for entity_type in ['STATUTE', 'PROVISION']):
                statute_boost = True
            elif chunk.metadata.statute_count > 3:
                statute_boost = True
            
            if statute_boost:
                original_score = getattr(chunk.metadata, 'retrieval_score', 0.5)
                chunk.metadata.retrieval_score = min(original_score * 1.15, 1.0)
        
        if query_context is None:
            query_context = {}
        query_context['focus'] = 'statutory'
        query_context['preferred_entity_types'] = ['STATUTE', 'PROVISION']
        
        return self.legal_reranker.rerank_chunks(query, chunks, query_context)
    
    def _factual_focused_reranking(self, query: str, chunks: List[EnrichedChunk],
                                 query_context: Dict[str, Any] = None) -> RerankingResult:
        """Factual-focused reranking strategy"""
        
        # UPDATED: Boost chunks with factual roles using correct role names
        for chunk in chunks:
            if any(role in chunk.metadata.rhetorical_roles for role in FACTUAL_ROLES):
                original_score = getattr(chunk.metadata, 'retrieval_score', 0.5)
                chunk.metadata.retrieval_score = min(original_score * 1.3, 1.0)
        
        if query_context is None:
            query_context = {}
        query_context['focus'] = 'factual'
        query_context['preferred_roles'] = FACTUAL_ROLES
        
        return self.legal_reranker.rerank_chunks(query, chunks, query_context)
    
    def _reasoning_focused_reranking(self, query: str, chunks: List[EnrichedChunk],
                                   query_context: Dict[str, Any] = None) -> RerankingResult:
        """Reasoning-focused reranking strategy (NEW)"""
        
        # Boost chunks with reasoning roles
        for chunk in chunks:
            if any(role in chunk.metadata.rhetorical_roles for role in REASONING_ROLES):
                original_score = getattr(chunk.metadata, 'retrieval_score', 0.5)
                chunk.metadata.retrieval_score = min(original_score * 1.25, 1.0)
        
        if query_context is None:
            query_context = {}
        query_context['focus'] = 'reasoning'
        query_context['preferred_roles'] = REASONING_ROLES
        
        return self.legal_reranker.rerank_chunks(query, chunks, query_context)
    
    def _argument_focused_reranking(self, query: str, chunks: List[EnrichedChunk],
                                  query_context: Dict[str, Any] = None) -> RerankingResult:
        """Argument-focused reranking strategy (NEW)"""
        
        # Boost chunks with argument roles
        for chunk in chunks:
            if any(role in chunk.metadata.rhetorical_roles for role in ARGUMENT_ROLES):
                original_score = getattr(chunk.metadata, 'retrieval_score', 0.5)
                chunk.metadata.retrieval_score = min(original_score * 1.2, 1.0)
        
        if query_context is None:
            query_context = {}
        query_context['focus'] = 'arguments'
        query_context['preferred_roles'] = ARGUMENT_ROLES
        
        return self.legal_reranker.rerank_chunks(query, chunks, query_context)
