from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager

from src.complete_rag_pipeline import CompleteLegalRAGPipeline
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance
rag_pipeline: Optional[CompleteLegalRAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global rag_pipeline
    
    # Startup
    logger.info("Starting up Legal RAG API...")
    
    try:
        # Initialize pipeline
        config = {
            'use_pinecone': os.getenv('USE_PINECONE', 'true').lower() == 'true',
            'enable_gpu': os.getenv('ENABLE_GPU', 'true').lower() == 'true'
        }
        
        rag_pipeline = CompleteLegalRAGPipeline(config)
        
        # Build index if data directory exists
        data_dir = os.getenv('DATA_DIRECTORY', settings.MERGED_DATA_DIR)
        if os.path.exists(data_dir):
            logger.info(f"Building index from {data_dir}")
            success = await rag_pipeline.build_index(data_dir)
            if not success:
                logger.error("Failed to build index")
        else:
            logger.warning(f"Data directory not found: {data_dir}")
        
        logger.info("Legal RAG API startup completed")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal RAG API...")

# Create FastAPI app
app = FastAPI(
    title="Legal RAG System API",
    description="Advanced Legal Research Engine using Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Legal query")
    rhetorical_roles: Optional[List[str]] = Field(None, description="Filter by rhetorical roles")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    court_levels: Optional[List[str]] = Field(None, description="Filter by court levels")
    min_precedent_count: Optional[int] = Field(None, ge=0, description="Minimum precedent citations")
    min_statute_count: Optional[int] = Field(None, ge=0, description="Minimum statute references")
    date_from: Optional[str] = Field(None, description="Start date filter (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="End date filter (YYYY-MM-DD)")
    
    # Search parameters
    retrieval_top_k: Optional[int] = Field(20, ge=1, le=100, description="Number of documents to retrieve")
    generation_top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents for generation")
    fusion_method: Optional[str] = Field("weighted_sum", description="Fusion method for hybrid search")
    reranking_strategy: Optional[str] = Field(None, description="Reranking strategy")
    
    # Generation parameters
    max_generation_length: Optional[int] = Field(None, ge=50, le=1000, description="Maximum response length")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Generation temperature")

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=10, description="List of queries")
    options: Optional[Dict[str, Any]] = Field(None, description="Search options for all queries")

class QueryResponse(BaseModel):
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

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    checks: Dict[str, Any]

# Dependency to get pipeline
async def get_pipeline() -> CompleteLegalRAGPipeline:
    if rag_pipeline is None or not rag_pipeline.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    return rag_pipeline

# API Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "Legal RAG System API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(pipeline: CompleteLegalRAGPipeline = Depends(get_pipeline)):
    """Comprehensive health check of all pipeline components"""
    
    try:
        health_data = pipeline.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            checks={"error": str(e)}
        )

@app.get("/stats", summary="Pipeline statistics")
async def get_statistics(pipeline: CompleteLegalRAGPipeline = Depends(get_pipeline)):
    """Get comprehensive pipeline statistics"""
    
    try:
        stats = pipeline.get_pipeline_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving statistics: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, summary="Process legal query")
async def process_query(
    request: QueryRequest,
    pipeline: CompleteLegalRAGPipeline = Depends(get_pipeline)
):
    """Process a legal query through the complete RAG pipeline"""
    
    try:
        # Convert request to options dict
        search_options = {
            'rhetorical_roles': request.rhetorical_roles,
            'entity_types': request.entity_types,
            'court_levels': request.court_levels,
            'min_precedent_count': request.min_precedent_count,
            'min_statute_count': request.min_statute_count,
            'date_from': request.date_from,
            'date_to': request.date_to,
            'retrieval_top_k': request.retrieval_top_k,
            'generation_top_k': request.generation_top_k,
            'fusion_method': request.fusion_method,
            'reranking_strategy': request.reranking_strategy,
            'max_generation_length': request.max_generation_length,
            'temperature': request.temperature
        }
        
        # Remove None values
        search_options = {k: v for k, v in search_options.items() if v is not None}
        
        # Process query
        result = await pipeline.process_query(request.query, search_options)
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/batch-query", summary="Process multiple queries")
async def batch_process_queries(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    pipeline: CompleteLegalRAGPipeline = Depends(get_pipeline)
):
    """Process multiple legal queries in batch"""
    
    try:
        # Process queries
        results = await pipeline.batch_process_queries(request.queries, request.options)
        
        return {
            "batch_id": f"batch_{int(time.time())}",
            "total_queries": len(request.queries),
            "results": results,
            "processing_complete": True
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch processing: {str(e)}"
        )

# ... (keeping most of the existing code, updating only the facets endpoint)

@app.get("/facets", summary="Get available search facets")
async def get_available_facets(pipeline: CompleteLegalRAGPipeline = Depends(get_pipeline)):
    """Get available facets for filtered search with UPDATED role and entity types"""
    
    try:
        facets = {
            "rhetorical_roles": {
                "display_name": "Document Sections",
                "values": list(settings.RHETORICAL_ROLES.keys()),
                "descriptions": settings.RHETORICAL_ROLES,
                "groups": {
                    "Content Roles": ["PREAMBLE", "FAC", "ISSUE", "ANALYSIS"],
                    "Court Decisions": ["RLC", "RPC", "Ratio"],
                    "Legal References": ["STA", "PRE_RELIED", "PRE_NOT_RELIED"],
                    "Arguments": ["ARG_PETITIONER", "ARG_RESPONDENT"],
                    "Other": ["NONE"]
                },
                "description": "Filter by specific rhetorical roles in legal judgments"
            },
            "entity_types": {
                "display_name": "Legal Entities", 
                "values": list(settings.LEGAL_ENTITY_TYPES.keys()),
                "descriptions": settings.LEGAL_ENTITY_TYPES,
                "groups": {
                    "People": ["JUDGE", "LAWYER", "PETITIONER", "RESPONDENT", "WITNESS", "OTHER_PERSON"],
                    "Legal References": ["STATUTE", "PROVISION", "PRECEDENT", "CASE_NUMBER"],
                    "Organizations": ["COURT", "ORG", "GPE"],
                    "Other": ["DATE"]
                },
                "description": "Filter by types of legal entities mentioned"
            },
            "court_levels": {
                "display_name": "Court Hierarchy",
                "values": ["Supreme Court", "High Court", "District Court", "Sessions Court", "Tribunal"],
                "description": "Filter by court level"
            },
            "search_strategies": {
                "display_name": "Search Strategies",
                "values": ["legal_general", "precedent_focused", "statutory_focused", "factual_focused", "reasoning_focused", "argument_focused"],
                "descriptions": {
                    "legal_general": "General legal search with balanced relevance",
                    "precedent_focused": "Prioritize precedent and case law references",
                    "statutory_focused": "Focus on statutes and provisions",
                    "factual_focused": "Emphasize facts and issues",
                    "reasoning_focused": "Target legal analysis and reasoning",
                    "argument_focused": "Highlight arguments and contentions"
                },
                "description": "Specialized search and ranking strategies"
            }
        }
        
        return facets
        
    except Exception as e:
        logger.error(f"Error getting facets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving facets: {str(e)}"
        )

        
    except Exception as e:
        logger.error(f"Error getting facets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving facets: {str(e)}"
        )

@app.post("/rebuild-index", summary="Rebuild search index")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    pipeline: CompleteLegalRAGPipeline = Depends(get_pipeline)
):
    """Rebuild the search index (admin operation)"""
    
    try:
        # Add to background tasks
        background_tasks.add_task(
            rebuild_index_task, 
            pipeline, 
            settings.MERGED_DATA_DIR
        )
        
        return {
            "message": "Index rebuild started in background",
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Error starting index rebuild: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting index rebuild: {str(e)}"
        )

async def rebuild_index_task(pipeline: CompleteLegalRAGPipeline, data_dir: str):
    """Background task to rebuild index"""
    try:
        logger.info("Starting background index rebuild")
        success = await pipeline.build_index(data_dir, force_rebuild=True)
        if success:
            logger.info("Background index rebuild completed successfully")
        else:
            logger.error("Background index rebuild failed")
    except Exception as e:
        logger.error(f"Error in background index rebuild: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    # Run server
    uvicorn.run(
        "src.api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )
