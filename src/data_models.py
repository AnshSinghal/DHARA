from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

class Entity(BaseModel):
    """NER Entity model"""
    text: Optional[str] = None
    label: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None

class LegalMetadata(BaseModel):
    """Enhanced metadata for legal documents with optional retrieval scores"""
    document_id: str
    chunk_id: str
    chunk_index: int
    
    # Rhetorical role information
    rhetorical_roles: List[str] = Field(default_factory=list)
    primary_role: str = ""
    
    # Entity information
    entities: List[Entity] = Field(default_factory=list)
    entity_types: List[str] = Field(default_factory=list)
    entity_count: int = 0
    
    # Legal structure metadata
    precedent_count: int = 0
    statute_count: int = 0
    provision_count: int = 0
    
    # Precedent and statute clusters
    precedent_clusters: Dict[str, Any] = Field(default_factory=dict)
    statute_clusters: Dict[str, Any] = Field(default_factory=dict)
    provision_statute_pairs: List[Dict[str, str]] = Field(default_factory=list)
    
    # Court and case information
    court_level: Optional[str] = None
    case_type: Optional[str] = None
    judgment_date: Optional[str] = None
    
    # Source information
    source_file: str = ""
    original_start: int = 0
    original_end: int = 0
    
    # FIXED: Add scoring attributes for reranking
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_position: Optional[int] = None
    new_position: Optional[int] = None
    base_cross_encoder_score: Optional[float] = None
    metadata_boost: Optional[float] = None

class EnrichedChunk(BaseModel):
    """Complete enriched chunk for RAG"""
    id: str
    text: str
    cleaned_text: str
    metadata: LegalMetadata
    
    # Search optimization fields
    keywords: List[str] = Field(default_factory=list)
    legal_concepts: List[str] = Field(default_factory=list)
    
    # Quality metrics  
    coherence_score: Optional[float] = None
    completeness_score: Optional[float] = None

class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    rhetorical_roles: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    court_levels: Optional[List[str]] = None
    min_precedent_count: Optional[int] = None
    min_statute_count: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_k: int = 10
    search_method: str = "hybrid"
    enable_reranking: bool = True

@dataclass
class GenerationResult:
    """Result of legal text generation"""
    response: str
    contexts_used: int
    rhetorical_structure: Dict[str, str]
    legal_citations: List[str]
    confidence_score: float
    generation_time: float
    metadata: Dict[str, Any]

@dataclass
class RerankingResult:
    """Result of reranking operation"""
    chunks: List[EnrichedChunk]
    original_scores: List[float]
    rerank_scores: List[float]
    position_changes: List[int]
    processing_time: float
    reranking_explanation: str
