from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import time
import traceback
from app.main import get_agent
from app.core.logging_config import get_logger, log_function_call

logger = get_logger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    complexity: Optional[str] = "advanced"

class QueryResponse(BaseModel):
    query: str
    final_analysis: str
    confidence_score: float
    iterations: int
    research_process: list
    processing_time: float

class DocumentSummaryRequest(BaseModel):
    document_text: str
    focus_area: Optional[str] = 'general'

class DocumentSummaryResponse(BaseModel):
    summary: str
    focus_area: str = 'general'

@router.post("/research", response_model=QueryResponse)
async def legal_research(request: QueryRequest, agent=Depends(get_agent)):
    """Endpoint to handle legal research queries with comprehensive logging."""
    request_start_time = time.time()
    
    logger.info(
        "Legal research request initiated",
        extra={
            "component": "research_endpoint",
            "query_length": len(request.query),
            "complexity": request.complexity,
            "query_preview": request.query[:100] + "..." if len(request.query) > 100 else request.query
        }
    )

    try:
        # Log the start of agent research
        logger.info(
            "Starting agent research process",
            extra={
                "component": "agent_research",
                "query_complexity": request.complexity
            }
        )

        agent_start_time = time.time()
        result = agent.research(request.query)
        agent_processing_time = time.time() - agent_start_time

        total_processing_time = time.time() - request_start_time
        
        # Log performance metrics
        log_function_call(
            func_name="agent.research",
            args={"query": request.query, "complexity": request.complexity},
            execution_time=agent_processing_time
        )

        logger.info(
            "Legal research completed successfully",
            extra={
                "component": "research_endpoint",
                "agent_processing_time_seconds": round(agent_processing_time, 2),
                "total_processing_time_seconds": round(total_processing_time, 2),
                "iterations": result.get("iterations", 0),
                "confidence_score": result.get("confidence_score", 0.0),
                "result_length": len(result.get("final_analysis", "")),
                "research_steps": len(result.get("research_process", []))
            }
        )

        return QueryResponse(
            query=result["query"],
            final_analysis=result["final_analysis"],
            confidence_score=result["confidence_score"],
            iterations=result["iterations"],
            research_process=result["research_process"],
            processing_time=total_processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - request_start_time
        
        logger.error(
            "Legal research request failed",
            extra={
                "component": "research_endpoint",
                "processing_time_seconds": round(processing_time, 2),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_length": len(request.query),
                "complexity": request.complexity,
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500, 
            detail="Internal Server Error: Failed to process legal research request"
        )

@router.post("/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(request: DocumentSummaryRequest, agent=Depends(get_agent)):
    """Endpoint to summarize legal documents with detailed logging."""
    request_start_time = time.time()
    
    logger.info(
        "Document summarization request initiated",
        extra={
            "component": "summarize_endpoint",
            "document_length": len(request.document_text),
            "focus_area": request.focus_area,
            "document_preview": request.document_text[:200] + "..." if len(request.document_text) > 200 else request.document_text
        }
    )
    
    try:
        logger.info(
            "Starting document summarization process",
            extra={
                "component": "document_summarizer",
                "focus_area": request.focus_area
            }
        )
        
        summarizer_start_time = time.time()
        summarizer = agent.summarizer
        
        # Use default focus area if None
        focus_area = request.focus_area or 'general'
        
        summary = summarizer._run(
            case_document=request.document_text,
            focus_area=focus_area
        )
        
        summarizer_processing_time = time.time() - summarizer_start_time
        total_processing_time = time.time() - request_start_time
        
        # Log performance metrics
        log_function_call(
            func_name="summarizer._run",
            args={"document_length": len(request.document_text), "focus_area": focus_area},
            execution_time=summarizer_processing_time
        )
        
        logger.info(
            "Document summarization completed successfully",
            extra={
                "component": "summarize_endpoint",
                "summarizer_processing_time_seconds": round(summarizer_processing_time, 2),
                "total_processing_time_seconds": round(total_processing_time, 2),
                "original_document_length": len(request.document_text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(request.document_text), 3),
                "focus_area": focus_area
            }
        )
        
        return DocumentSummaryResponse(
            summary=summary,
            focus_area=focus_area
        )
        
    except Exception as e:
        processing_time = time.time() - request_start_time
        
        logger.error(
            "Document summarization request failed",
            extra={
                "component": "summarize_endpoint",
                "processing_time_seconds": round(processing_time, 2),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "document_length": len(request.document_text),
                "focus_area": request.focus_area,
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500, 
            detail=f"Document Summarization Failed: {str(e)}"
        )
    
@router.post("/extract-entities")
async def extract_legal_entities(request: QueryRequest, agent=Depends(get_agent)):
    """Extract relevant legal entities from similar cases with comprehensive logging."""
    request_start_time = time.time()
    
    logger.info(
        "Entity extraction request initiated",
        extra={
            "component": "entity_extraction_endpoint",
            "query_length": len(request.query),
            "query_preview": request.query[:100] + "..." if len(request.query) > 100 else request.query
        }
    )
    
    try:
        logger.info(
            "Starting entity extraction process",
            extra={
                "component": "entity_extractor",
                "top_k_cases": 2
            }
        )
        
        extractor_start_time = time.time()
        
        # Use entity extractor tool
        result = agent.entity_extractor._run(query=request.query, top_k_cases=2)
        
        extractor_processing_time = time.time() - extractor_start_time
        total_processing_time = time.time() - request_start_time
        
        # Log performance metrics
        log_function_call(
            func_name="entity_extractor._run",
            args={"query": request.query, "top_k_cases": 2},
            execution_time=extractor_processing_time
        )
        
        logger.info(
            "Entity extraction completed successfully",
            extra={
                "component": "entity_extraction_endpoint",
                "extractor_processing_time_seconds": round(extractor_processing_time, 2),
                "total_processing_time_seconds": round(total_processing_time, 2),
                "entities_found": len(result) if isinstance(result, (list, dict)) else 0,
                "result_type": type(result).__name__
            }
        )
        
        return {
            "query": request.query,
            "extracted_entities": result
        }
        
    except Exception as e:
        processing_time = time.time() - request_start_time
        
        logger.error(
            "Entity extraction request failed",
            extra={
                "component": "entity_extraction_endpoint",
                "processing_time_seconds": round(processing_time, 2),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_length": len(request.query),
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500, 
            detail=f"Entity Extraction Failed: {str(e)}"
        )

@router.get("/status")
async def get_system_status(agent=Depends(get_agent)):
    """Get system status and configuration with detailed logging."""
    request_start_time = time.time()
    
    logger.info(
        "System status check initiated",
        extra={
            "component": "status_endpoint"
        }
    )
    
    try:
        status_data = {
            "agent_initialized": True,
            "retriever_status": "active",
            "model_status": "loaded",
            "data_path": "/app/data/processed/merged",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - request_start_time
        }
        
        logger.info(
            "System status check completed successfully",
            extra={
                "component": "status_endpoint",
                "agent_initialized": True,
                "retriever_status": "active",
                "model_status": "loaded"
            }
        )
        
        return status_data
        
    except Exception as e:
        logger.error(
            "System status check failed",
            extra={
                "component": "status_endpoint",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        return {
            "agent_initialized": False,
            "error": str(e),
            "timestamp": time.time()
        }