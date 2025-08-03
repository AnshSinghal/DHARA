import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from src.data_models import SearchQuery, EnrichedChunk, LegalMetadata
from src.embeddings import LegalEmbeddingModel
from src.vector_store import EnhancedPineconeStore
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchFacet:
    """Represents a search facet with its possible values"""
    name: str
    display_name: str
    values: List[str]
    count: int = 0

@dataclass
class FacetedSearchResult:
    """Enhanced search result with facet information"""
    chunks: List[EnrichedChunk]
    facets: List[SearchFacet]
    total_results: int
    query_intent: str
    search_time: float

class LegalMultiFacetedSearch:
    """Advanced multi-faceted search for legal documents using structured metadata"""
    
    def __init__(self, vector_store: EnhancedPineconeStore, embedding_model: LegalEmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        # Define available facets based on your data structure
        self.available_facets = {
            'rhetorical_roles': {
                'display_name': 'Document Sections',
                'values': list(settings.RHETORICAL_ROLES.keys()),
                'description': 'Filter by specific parts of legal judgments'
            },
            'entity_types': {
                'display_name': 'Legal Entities',
                'values': list(settings.LEGAL_ENTITY_TYPES.keys()),
                'description': 'Filter by types of legal entities mentioned'
            },
            'court_levels': {
                'display_name': 'Court Hierarchy',
                'values': ['Supreme Court', 'High Court', 'District Court', 'Sessions Court'],
                'description': 'Filter by court level'
            },
            'legal_areas': {
                'display_name': 'Areas of Law',
                'values': ['Criminal', 'Civil', 'Constitutional', 'Commercial', 'Family', 'Tax'],
                'description': 'Filter by legal domain'
            },
            'precedent_density': {
                'display_name': 'Precedent Citations',
                'values': ['High (10+)', 'Medium (5-9)', 'Low (1-4)', 'None (0)'],
                'description': 'Filter by number of precedent citations'
            },
            'statute_density': {
                'display_name': 'Statutory Provisions',
                'values': ['High (5+)', 'Medium (3-4)', 'Low (1-2)', 'None (0)'],
                'description': 'Filter by number of statutory references'
            }
        }
    
    def build_faceted_filters(self, search_query: SearchQuery) -> Dict[str, Any]:
        """Build Pinecone filters from search query facets"""
        filters = {}
        
        # Rhetorical role filters
        if search_query.rhetorical_roles:
            filters['primary_role'] = {'$in': search_query.rhetorical_roles}
        
        # Entity type filters
        if search_query.entity_types:
            filters['entity_types'] = {'$in': search_query.entity_types}
        
        # Precedent count filters
        if search_query.min_precedent_count is not None:
            filters['precedent_count'] = {'$gte': search_query.min_precedent_count}
        
        # Statute count filters
        if search_query.min_statute_count is not None:
            filters['statute_count'] = {'$gte': search_query.min_statute_count}
        
        # Date filters
        if search_query.date_from or search_query.date_to:
            date_filter = {}
            if search_query.date_from:
                date_filter['$gte'] = search_query.date_from
            if search_query.date_to:
                date_filter['$lte'] = search_query.date_to
            filters['judgment_date'] = date_filter
        
        return filters
    
    def execute_faceted_search(self, search_query: SearchQuery) -> FacetedSearchResult:
        """Execute multi-faceted search with rich metadata filtering"""
        import time
        start_time = time.time()
        
        # Encode query with context
        query_context = {
            'rhetorical_roles': search_query.rhetorical_roles,
            'entity_types': search_query.entity_types
        }
        
        query_vector = self.embedding_model.encode_query(search_query.query, query_context)
        
        # Build filters
        filters = self.build_faceted_filters(search_query)
        
        # Execute search
        search_results = self.vector_store.search_with_filters(query_vector, search_query)
        
        # Convert results to EnrichedChunk objects
        chunks = self._convert_results_to_chunks(search_results)
        
        # Generate facets for refinement 
        facets = self._generate_facets_from_results(chunks)
        
        # Classify query intent for better UX
        query_intent = self._classify_query_intent(search_query.query)
        
        search_time = time.time() - start_time
        
        return FacetedSearchResult(
            chunks=chunks,
            facets=facets,
            total_results=len(chunks),
            query_intent=query_intent,
            search_time=search_time
        )
    
    def _convert_results_to_chunks(self, search_results: List[Dict[str, Any]]) -> List[EnrichedChunk]:
        """Convert search results back to EnrichedChunk objects"""
        chunks = []
        
        for result in search_results:
            # Reconstruct metadata
            metadata = LegalMetadata(
                document_id=result['metadata'].get('document_id', ''),
                chunk_id=result['id'],
                chunk_index=result['metadata'].get('chunk_index', 0),
                rhetorical_roles=result['metadata'].get('rhetorical_roles', []),
                primary_role=result['metadata'].get('primary_role', ''),
                entities=[],  # Would need to reconstruct from metadata
                entity_types=result['metadata'].get('entity_types', []),
                entity_count=result['metadata'].get('entity_count', 0),
                precedent_count=result['metadata'].get('precedent_count', 0),
                statute_count=result['metadata'].get('statute_count', 0),
                provision_count=result['metadata'].get('provision_count', 0),
                source_file=result['metadata'].get('source_file', ''),
                original_start=0,
                original_end=0
            )
            
            chunk = EnrichedChunk(
                id=result['id'],
                text=result['text'],
                cleaned_text=result['text'],
                metadata=metadata,
                keywords=result['metadata'].get('keywords', []),
                legal_concepts=result['metadata'].get('legal_concepts', [])
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _generate_facets_from_results(self, chunks: List[EnrichedChunk]) -> List[SearchFacet]:
        """Generate available facets from current search results"""
        facets = []
        
        # Rhetorical roles facet
        role_counts = {}
        for chunk in chunks:
            for role in chunk.metadata.rhetorical_roles:
                role_counts[role] = role_counts.get(role, 0) + 1
        
        if role_counts:
            facets.append(SearchFacet(
                name='rhetorical_roles',
                display_name='Document Sections',
                values=list(role_counts.keys()),
                count=len(role_counts)
            ))
        
        # Entity types facet
        entity_counts = {}
        for chunk in chunks:
            for entity_type in chunk.metadata.entity_types:
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        if entity_counts:
            facets.append(SearchFacet(
                name='entity_types',
                display_name='Legal Entities',
                values=list(entity_counts.keys()),
                count=len(entity_counts)
            ))
        
        # Precedent density facet
        precedent_ranges = {'High (10+)': 0, 'Medium (5-9)': 0, 'Low (1-4)': 0, 'None (0)': 0}
        for chunk in chunks:
            count = chunk.metadata.precedent_count
            if count >= 10:
                precedent_ranges['High (10+)'] += 1
            elif count >= 5:
                precedent_ranges['Medium (5-9)'] += 1
            elif count >= 1:
                precedent_ranges['Low (1-4)'] += 1
            else:
                precedent_ranges['None (0)'] += 1
        
        facets.append(SearchFacet(
            name='precedent_density',
            display_name='Precedent Citations',
            values=[k for k, v in precedent_ranges.items() if v > 0],
            count=sum(v for v in precedent_ranges.values() if v > 0)
        ))
        
        return facets
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify query intent based on content and your NER/RRL data"""
        query_lower = query.lower()
        
        # Intent classification based on legal patterns
        if any(word in query_lower for word in ['precedent', 'case law', 'decided', 'held']):
            return "precedent_search"
        elif any(word in query_lower for word in ['section', 'act', 'statute', 'provision']):
            return "statutory_search"
        elif any(word in query_lower for word in ['facts', 'background', 'circumstances']):
            return "factual_search"
        elif any(word in query_lower for word in ['ratio', 'reasoning', 'decision', 'judgment']):
            return "ratio_search"
        elif any(word in query_lower for word in ['procedure', 'process', 'steps', 'how to']):
            return "procedural_search"
        else:
            return "general_search"
    

    def get_facet_statistics(self) -> Dict[str, Any]:
        """Get statistics about available facets with UPDATED role and entity types"""
        
        try:
            stats = {
                'rhetorical_roles': {
                    'total_roles': len(settings.RHETORICAL_ROLES),
                    'available_roles': list(settings.RHETORICAL_ROLES.keys()),
                    'role_categories': {
                        'content': ['PREAMBLE', 'FAC', 'ISSUE', 'ANALYSIS'],
                        'decisions': ['RLC', 'RPC', 'Ratio'],
                        'references': ['STA', 'PRE_RELIED', 'PRE_NOT_RELIED'],
                        'arguments': ['ARG_PETITIONER', 'ARG_RESPONDENT']
                    }
                },
                'entity_types': {
                    'total_types': len(settings.LEGAL_ENTITY_TYPES),
                    'available_types': list(settings.LEGAL_ENTITY_TYPES.keys()),
                    'priority_types': settings.PRIORITY_ENTITY_TYPES,
                    'type_categories': {
                        'people': ['JUDGE', 'LAWYER', 'PETITIONER', 'RESPONDENT', 'WITNESS', 'OTHER_PERSON'],
                        'legal_refs': ['STATUTE', 'PROVISION', 'PRECEDENT', 'CASE_NUMBER'],
                        'organizations': ['COURT', 'ORG', 'GPE']
                    }
                }
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error generating facet statistics: {e}")
            return {}

    def suggest_related_queries(self, query: str, chunks: List[EnrichedChunk]) -> List[str]:
        """Suggest related queries based on UPDATED rhetorical roles and entities in results"""
        
        if not chunks:
            return []
        
        suggestions = []
        query_lower = query.lower()
        
        # Analyze dominant roles in results
        role_counts = {}
        entity_type_counts = {}
        
        for chunk in chunks[:5]:  # Top 5 chunks
            for role in chunk.metadata.rhetorical_roles:
                role_counts[role] = role_counts.get(role, 0) + 1
            
            for entity_type in chunk.metadata.entity_types:
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        # Suggest based on dominant roles
        dominant_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for role, count in dominant_roles:
            if role in settings.RHETORICAL_ROLES:
                role_desc = settings.RHETORICAL_ROLES[role].lower()
                if role == 'PRE_RELIED':
                    suggestions.append(f"What precedents support {query_lower}?")
                elif role == 'STA':
                    suggestions.append(f"What statutes govern {query_lower}?")
                elif role == 'FAC':
                    suggestions.append(f"What are the key facts in {query_lower}?")
                elif role == 'ANALYSIS':
                    suggestions.append(f"How is {query_lower} analyzed legally?")
                elif role == 'Ratio':
                    suggestions.append(f"What is the legal principle in {query_lower}?")
        
        # Suggest based on entity types
        if 'COURT' in entity_type_counts:
            suggestions.append(f"Which courts have decided cases on {query_lower}?")
        
        if 'STATUTE' in entity_type_counts or 'PROVISION' in entity_type_counts:
            suggestions.append(f"What legal provisions apply to {query_lower}?")
        
        if 'PRECEDENT' in entity_type_counts:
            suggestions.append(f"What are similar cases to {query_lower}?")
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:5]
        
        return unique_suggestions


class AdvancedQueryProcessor:
    """Process and enhance queries using your NER and RRL data"""
    
    def __init__(self):
        # Legal concepts and their variations
        self.concept_expansions = {
            'contract': ['agreement', 'covenant', 'pact', 'deal'],
            'breach': ['violation', 'default', 'non-compliance'],
            'damages': ['compensation', 'remedy', 'restitution'],
            'precedent': ['case law', 'judicial precedent', 'stare decisis'],
            'appeal': ['revision', 'review', 'appellate proceedings']
        }
        
        # Rhetorical role synonyms
        self.role_synonyms = {
            'facts': ['FAC', 'background', 'circumstances'],
            'decision': ['RulingByPresentCourt', 'judgment', 'order'],
            'arguments': ['Argument', 'submissions', 'contentions'], 
            'precedents': ['Precedent', 'case law', 'citations'],
            'statutes': ['Statute', 'provisions', 'sections']
        }
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        """Expand query with legal concepts and role-based terms"""
        expanded_query = {
            'original': query,
            'expanded_terms': [],
            'suggested_roles': [],
            'suggested_entities': [],
            'query_type': 'general'
        }
        
        query_lower = query.lower()
        
        # Expand with concept synonyms
        for concept, synonyms in self.concept_expansions.items():
            if concept in query_lower:
                expanded_query['expanded_terms'].extend(synonyms)
        
        # Suggest relevant rhetorical roles
        for role_key, role_terms in self.role_synonyms.items():
            if any(term in query_lower for term in role_terms):
                expanded_query['suggested_roles'].append(role_key)
        
        # Suggest entity types based on query content
        if any(term in query_lower for term in ['court', 'judge', 'bench']):
            expanded_query['suggested_entities'].append('COURT')
        
        if any(term in query_lower for term in ['petitioner', 'appellant', 'plaintiff']):
            expanded_query['suggested_entities'].append('PETITIONER')
        
        if any(term in query_lower for term in ['respondent', 'defendant', 'accused']):
            expanded_query['suggested_entities'].append('RESPONDENT')
        
        # Classify query type
        if any(term in query_lower for term in ['section', 'act', 'provision']):
            expanded_query['query_type'] = 'statutory'
        elif any(term in query_lower for term in ['precedent', 'case law']):
            expanded_query['query_type'] = 'precedent'
        elif any(term in query_lower for term in ['facts', 'background']):
            expanded_query['query_type'] = 'factual'
        
        return expanded_query
