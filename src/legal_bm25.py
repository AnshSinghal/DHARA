import json
import pickle
import os
from typing import List, Dict, Any, Union, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import logging
from collections import Counter

from src.data_models import EnrichedChunk
from config.settings import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

class LegalBM25Retriever:
    """BM25 retriever optimized for legal documents with domain-specific preprocessing"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation (lower for legal text)
        self.b = b    # Length normalization
        
        self.bm25 = None
        self.chunks: List[EnrichedChunk] = []
        self.processed_docs: List[List[str]] = []
        
        # Legal-specific preprocessing
        self.legal_stopwords = self._build_legal_stopwords()
        self.legal_term_boosts = self._build_legal_term_boosts()
        
        # Compile regex patterns for efficiency
        self.legal_patterns = {
            'section_refs': re.compile(r'\b(?:section|sec\.?|s\.?)\s*\d+(?:\([a-zA-Z0-9]+\))?', re.IGNORECASE),
            'article_refs': re.compile(r'\b(?:article|art\.?)\s*\d+(?:\([a-zA-Z0-9]+\))?', re.IGNORECASE),
            'case_citations': re.compile(r'\b\d{4}\s+\w+\s+\d+\b'),
            'court_refs': re.compile(r'\b(?:supreme\s+court|high\s+court|district\s+court)', re.IGNORECASE),
            'legal_concepts': re.compile(r'\b(?:precedent|ratio|obiter|stare\s+decisis|res\s+judicata)\b', re.IGNORECASE)
        }
    
    def _build_legal_stopwords(self) -> set:
        """Build legal-domain stopwords (conservative removal)"""
        # Start with English stopwords
        base_stopwords = set(stopwords.words('english'))
        
        # Remove legal terms that might be important
        legal_keep_words = {
            'court', 'case', 'law', 'act', 'section', 'under', 'above', 
            'below', 'before', 'after', 'against', 'shall', 'may', 'must',
            'whereas', 'therefore', 'however', 'provided', 'subject', 'without'
        }
        
        # Add legal-specific stopwords that are rarely meaningful
        legal_stop_words = {
            'learned', 'counsel', 'respectfully', 'submitted', 'contended',
            'argued', 'stated', 'observed', 'noted', 'remarked', 'opined'
        }
        
        return (base_stopwords - legal_keep_words) | legal_stop_words
    
    def _build_legal_term_boosts(self) -> Dict[str, float]:
        """Define boost weights for important legal terms"""
        return {
            # Statutory references
            'section': 2.0, 'article': 2.0, 'provision': 1.8,
            'subsection': 1.5, 'clause': 1.5,
            
            # Legal concepts
            'precedent': 2.2, 'ratio': 2.0, 'obiter': 1.8,
            'stare_decisis': 2.5, 'res_judicata': 2.0,
            
            # Court terminology
            'held': 2.0, 'decided': 1.8, 'ruled': 1.8,
            'dismissed': 1.5, 'allowed': 1.5, 'quashed': 1.8,
            
            # Legal procedures
            'appeal': 1.5, 'revision': 1.5, 'writ': 1.8,
            'petition': 1.3, 'application': 1.2,
            
            # Evidence and burden
            'evidence': 1.5, 'proof': 1.5, 'burden': 1.8,
            'prima_facie': 2.0, 'preponderance': 1.8
        }
    
    def preprocess_legal_text(self, text: str, chunk_metadata: Dict[str, Any] = None) -> List[str]:
        """Advanced preprocessing for legal text with metadata awareness"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Extract and preserve legal references
        legal_refs = []
        
        # Section references
        for match in self.legal_patterns['section_refs'].finditer(text):
            ref = match.group().replace(' ', '_')
            legal_refs.append(ref)
        
        # Case citations  
        for match in self.legal_patterns['case_citations'].finditer(text):
            ref = match.group().replace(' ', '_')
            legal_refs.append(ref)
        
        # Court references
        for match in self.legal_patterns['court_refs'].finditer(text):
            ref = match.group().replace(' ', '_')
            legal_refs.append(ref)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Keep alphanumeric and legal symbols
            if re.match(r'^[a-zA-Z0-9._\-]+$', token):
                # Remove stopwords but keep legal terms
                if token not in self.legal_stopwords or len(token) > 2:
                    processed_tokens.append(token)
        
        # Add extracted legal references
        processed_tokens.extend(legal_refs)
        
        # Add metadata-based terms if available
        if chunk_metadata:
            # Add rhetorical role context
            if 'primary_role' in chunk_metadata:
                processed_tokens.append(f"role_{chunk_metadata['primary_role'].lower()}")
            
            # Add entity type context
            if 'entity_types' in chunk_metadata:
                for entity_type in chunk_metadata['entity_types'][:3]:  # Top 3
                    processed_tokens.append(f"entity_{entity_type.lower()}")
        
        return processed_tokens
    
    def build_index(self, chunks: List[EnrichedChunk]):
        """Build BM25 index from enriched chunks"""
        logger.info(f"Building legal BM25 index for {len(chunks)} chunks")
        
        self.chunks = chunks
        self.processed_docs = []
        
        # Process each chunk
        for chunk in chunks:
            # Create metadata dict for preprocessing
            metadata = {
                'primary_role': chunk.metadata.primary_role,
                'entity_types': chunk.metadata.entity_types,
                'precedent_count': chunk.metadata.precedent_count,
                'statute_count': chunk.metadata.statute_count
            }
            
            # Preprocess text with metadata awareness
            processed_tokens = self.preprocess_legal_text(chunk.cleaned_text, metadata)
            
            # Apply term boosting by duplicating important terms
            boosted_tokens = []
            for token in processed_tokens:
                boosted_tokens.append(token)
                
                # Apply boosts
                if token in self.legal_term_boosts:
                    boost_factor = int(self.legal_term_boosts[token])
                    boosted_tokens.extend([token] * (boost_factor - 1))
            
            self.processed_docs.append(boosted_tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.processed_docs, k1=self.k1, b=self.b)
        
        logger.info(f"Legal BM25 index built with vocabulary size: {len(self.bm25.idf)}")
    
    def search(self, query: str, top_k: int = 10, 
               query_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search with legal query preprocessing"""
        
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Preprocess query
        query_tokens = self.preprocess_legal_text(query)
        
        # Add context-based terms if provided
        if query_context:
            if query_context.get('rhetorical_roles'):
                for role in query_context['rhetorical_roles']:
                    query_tokens.append(f"role_{role.lower()}")
            
            if query_context.get('entity_types'):
                for entity_type in query_context['entity_types']:
                    query_tokens.append(f"entity_{entity_type.lower()}")
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only positive scores
                chunk = self.chunks[idx]
                
                # Calculate additional relevance signals
                relevance_signals = self._calculate_relevance_signals(
                    chunk, query_tokens, query_context
                )
                
                # Adjust score based on relevance signals
                adjusted_score = score * relevance_signals['multiplier']
                
                result = {
                    'id': chunk.id,
                    'score': float(adjusted_score),
                    'text': chunk.text,
                    'metadata': {
                        'document_id': chunk.metadata.document_id,
                        'primary_role': chunk.metadata.primary_role,
                        'entity_types': chunk.metadata.entity_types,
                        'precedent_count': chunk.metadata.precedent_count,
                        'statute_count': chunk.metadata.statute_count,
                        'legal_concepts': chunk.legal_concepts,
                        'keywords': chunk.keywords,
                        **relevance_signals['details']
                    }
                }
                results.append(result)
        
        return results
    
    def _calculate_relevance_signals(self, chunk: EnrichedChunk, 
                                   query_tokens: List[str],
                                   query_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate additional relevance signals beyond BM25"""
        
        signals = {'multiplier': 1.0, 'details': {}}
        
        # Rhetorical role matching
        if query_context and query_context.get('rhetorical_roles'):
            if chunk.metadata.primary_role in query_context['rhetorical_roles']:
                signals['multiplier'] *= 1.3
                signals['details']['role_match'] = True
        
        # Entity type matching
        if query_context and query_context.get('entity_types'):
            entity_overlap = set(chunk.metadata.entity_types) & set(query_context['entity_types'])
            if entity_overlap:
                signals['multiplier'] *= (1.0 + 0.1 * len(entity_overlap))
                signals['details']['entity_matches'] = list(entity_overlap)
        
        # Legal concept matching
        concept_matches = []
        for token in query_tokens:
            if token in chunk.legal_concepts:
                concept_matches.append(token)
        
        if concept_matches:
            signals['multiplier'] *= (1.0 + 0.2 * len(concept_matches))
            signals['details']['concept_matches'] = concept_matches
        
        # Precedent density bonus for precedent queries
        if any('precedent' in token for token in query_tokens):
            if chunk.metadata.precedent_count > 5:
                signals['multiplier'] *= 1.2
                signals['details']['high_precedent_density'] = True
        
        # Statute density bonus for statutory queries
        if any('section' in token or 'statute' in token for token in query_tokens):
            if chunk.metadata.statute_count > 3:
                signals['multiplier'] *= 1.15
                signals['details']['high_statute_density'] = True
        
        return signals
    
    def get_query_expansion_suggestions(self, query: str) -> List[str]:
        """Suggest query expansions based on legal domain knowledge"""
        suggestions = []
        query_lower = query.lower()
        
        # Synonym-based expansions
        synonyms = {
            'contract': ['agreement', 'covenant', 'pact'],
            'breach': ['violation', 'default', 'non-compliance'],
            'court': ['tribunal', 'bench', 'forum'],
            'appeal': ['revision', 'review', 'appellate'],
            'judgment': ['decision', 'order', 'ruling']
        }
        
        for term, synonym_list in synonyms.items():
            if term in query_lower:
                suggestions.extend([f"{query.replace(term, syn)}" for syn in synonym_list])
        
        # Role-based expansions
        if 'facts' in query_lower:
            suggestions.append(f"{query} background circumstances")
        elif 'decision' in query_lower:
            suggestions.append(f"{query} ruling judgment order")
        elif 'precedent' in query_lower:
            suggestions.append(f"{query} case law judicial precedent")
        
        return suggestions[:5]  # Top 5 suggestions
    
    def analyze_corpus_statistics(self) -> Dict[str, Any]:
        """Analyze corpus statistics for insights"""
        if not self.chunks:
            return {}
        
        stats = {
            'total_chunks': len(self.chunks),
            'vocabulary_size': len(self.bm25.idf) if self.bm25 else 0,
            'rhetorical_role_distribution': {},
            'entity_type_distribution': {},
            'average_precedent_count': 0,
            'average_statute_count': 0,
            'top_legal_concepts': {}
        }
        
        # Rhetorical role distribution
        role_counts = Counter()
        entity_counts = Counter()
        precedent_counts = []
        statute_counts = []
        concept_counts = Counter()
        
        for chunk in self.chunks:
            # Roles
            for role in chunk.metadata.rhetorical_roles:
                role_counts[role] += 1
            
            # Entities
            for entity_type in chunk.metadata.entity_types:
                entity_counts[entity_type] += 1
            
            # Counts
            precedent_counts.append(chunk.metadata.precedent_count)
            statute_counts.append(chunk.metadata.statute_count)
            
            # Concepts
            for concept in chunk.legal_concepts:
                concept_counts[concept] += 1
        
        stats['rhetorical_role_distribution'] = dict(role_counts.most_common(10))
        stats['entity_type_distribution'] = dict(entity_counts.most_common(10))
        stats['average_precedent_count'] = np.mean(precedent_counts) if precedent_counts else 0
        stats['average_statute_count'] = np.mean(statute_counts) if statute_counts else 0
        stats['top_legal_concepts'] = dict(concept_counts.most_common(20))
        
        return stats
    
    def save_index(self, file_path: str):
        """Save the BM25 index and related data"""
        index_data = {
            'bm25': self.bm25,
            'chunks': [chunk.model_dump() for chunk in self.chunks],
            'processed_docs': self.processed_docs,
            'parameters': {'k1': self.k1, 'b': self.b}
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Legal BM25 index saved to {file_path}")
    
    def load_index(self, file_path: str):
        """Load the BM25 index and related data"""
        with open(file_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.processed_docs = index_data['processed_docs']
        self.k1 = index_data['parameters']['k1']
        self.b = index_data['parameters']['b']
        
        # Reconstruct chunks
        from src.data_models import EnrichedChunk
        self.chunks = [EnrichedChunk(**chunk_data) for chunk_data in index_data['chunks']]
        
        logger.info(f"Legal BM25 index loaded from {file_path}")
