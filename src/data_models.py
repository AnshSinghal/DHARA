from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

class Entity(BaseModel):
    """NER Entity model with optional fields for robust parsing"""
    text: Optional[str] = None
    label: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None

class RhetoricalAnnotation(BaseModel):
    """Rhetorical Roles Labeling (RRL) Annotation model with optional fields"""
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    id: Optional[str] = None
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
    # FIXED: Changed to List[Dict[str, str]] for proper provision-statute pairs
    provision_statute_pairs: List[Dict[str, str]] = Field(default_factory=list)
    
    # Court and case information
    court_level: Optional[str] = None
    case_type: Optional[str] = None
    judgment_date: Optional[str] = None
    
    # Source information
    source_file: str = ""
    original_start: int = 0
    original_end: int = 0
    
    # FIXED: Add scoring attributes for reranking (fixes .get() error)
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_position: Optional[int] = None
    new_position: Optional[int] = None
    base_cross_encoder_score: Optional[float] = None
    metadata_boost: Optional[float] = None

class EnrichedChunk(BaseModel):
    """Complete enriched chunk for RAG with all metadata"""
    id: str
    text: str
    cleaned_text: str
    embedding: Optional[List[float]] = None
    metadata: LegalMetadata
    
    # Search optimization fields
    keywords: List[str] = Field(default_factory=list)
    legal_concepts: List[str] = Field(default_factory=list)
    
    # Quality metrics  
    coherence_score: Optional[float] = None
    completeness_score: Optional[float] = None

class SearchQuery(BaseModel):
    """Enhanced search query model with comprehensive filtering options"""
    query: str
    
    # Filtering options using correct rhetorical roles and entity types
    rhetorical_roles: Optional[List[str]] = None  # PREAMBLE, FAC, RLC, ISSUE, etc.
    entity_types: Optional[List[str]] = None      # COURT, PETITIONER, JUDGE, etc.
    court_levels: Optional[List[str]] = None
    
    # Numerical filters
    min_precedent_count: Optional[int] = None
    min_statute_count: Optional[int] = None
    
    # Date filters
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    
    # Search parameters
    top_k: int = 10
    search_method: str = "hybrid"  # dense, sparse, hybrid
    enable_reranking: bool = True

@dataclass
class GenerationResult:
    """Result of legal text generation"""
    response: str
    contexts_used: int
    rhetorical_structure: Dict[str, Any]  # Changed from Dict[str, str] for flexibility
    legal_citations: List[str]
    confidence_score: float
    generation_time: float
    metadata: Dict[str, Any]

class PipelineMetrics(BaseModel):
    """Comprehensive pipeline performance metrics"""  
    total_processing_time: float
    indexing_time: float
    retrieval_time: float
    reranking_time: float
    generation_time: float
    
    documents_processed: int
    chunks_created: int
    chunks_retrieved: int
    chunks_reranked: int
    
    retrieval_quality_score: float
    generation_quality_score: float
    overall_pipeline_score: float
    
    memory_usage_mb: float
    gpu_utilization: Optional[float] = None

class RerankingResult(BaseModel):
    """Result of reranking operation"""
    chunks: List[EnrichedChunk]
    original_scores: List[float]
    rerank_scores: List[float] 
    position_changes: List[int]
    processing_time: float
    reranking_explanation: str

class FacetedSearchResult(BaseModel):
    """Enhanced search result with facet information"""
    chunks: List[EnrichedChunk]
    facets: List[Dict[str, Any]]
    total_results: int
    query_intent: str
    search_time: float

# Additional utility models for API responses
class HealthStatus(BaseModel):
    """System health check response"""
    status: str
    timestamp: float
    checks: Dict[str, Any]

class QueryResponse(BaseModel):
    """Complete API query response model"""
    query: str
    response: str
    confidence_score: float
    processing_time: float
    contexts_used: int
    supporting_documents: List[Dict[str, Any]]
    legal_citations: List[str]
    precedent_references: List[Dict[str, Any]]
    statute_references: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]
    pipeline_insights: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    related_queries: List[str]
    success: bool = True

class BatchQueryRequest(BaseModel):
    """Batch query request model"""
    queries: List[str] = Field(..., min_items=1, max_items=10)
    options: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    """Individual query request model"""
    query: str = Field(..., min_length=3, max_length=1000)
    rhetorical_roles: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    court_levels: Optional[List[str]] = None
    min_precedent_count: Optional[int] = Field(None, ge=0)
    min_statute_count: Optional[int] = Field(None, ge=0)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    
    # Search parameters
    retrieval_top_k: Optional[int] = Field(20, ge=1, le=100)
    generation_top_k: Optional[int] = Field(5, ge=1, le=20)
    fusion_method: Optional[str] = Field("weighted_sum")
    reranking_strategy: Optional[str] = None
    
    # Generation parameters
    max_generation_length: Optional[int] = Field(None, ge=50, le=1000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

# Legal domain specific models
class LegalCitation(BaseModel):
    """Legal citation reference"""
    citation_text: str
    case_name: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    citation_type: str = "case"  # case, statute, provision

class StatuteReference(BaseModel):
    """Statute reference model"""
    statute_name: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    act_year: Optional[int] = None

class PrecedentReference(BaseModel):
    """Precedent case reference"""
    case_name: str
    citation: str
    year: Optional[int] = None
    court: str
    ratio_summary: Optional[str] = None
    relevance_score: float = 0.0

# Evaluation models
class EvaluationResult(BaseModel):
    """Evaluation result for pipeline performance"""
    query: str
    expected_answer: Optional[str] = None
    generated_answer: str
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    legal_accuracy_score: float
    citation_accuracy: float
    processing_time: float
    chunks_used: int
    
class BenchmarkResult(BaseModel):
    """Benchmark evaluation results"""
    total_queries: int
    average_relevance: float
    average_accuracy: float
    average_processing_time: float
    success_rate: float
    evaluation_details: List[EvaluationResult]
    timestamp: datetime
