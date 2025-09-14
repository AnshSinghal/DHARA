from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Callable
from app.core.advanced_agent import AdvancedLegalAgent
from app.core.logging_config import setup_logging, get_logger, log_api_request
from dotenv import load_dotenv
load_dotenv() 


setup_logging()
logger = get_logger(__name__)

agent = None

def get_secret(secret_name: str) -> str:
    """
    Load secrets from Docker secrets or environment variables
    Docker secrets are mounted at /run/secrets/<secret_name>
    """
    # Try Docker secrets first (production)
    secret_path = Path(f"/run/secrets/{secret_name}")
    if secret_path.exists():
        logger.info(f"Loading secret {secret_name} from Docker secrets")
        return secret_path.read_text().strip()
    
    # Fallback to environment variable (development)
    env_var = secret_name.upper()
    value = os.getenv(env_var)
    if value:
        logger.info(f"Loading secret {secret_name} from environment variable")
        return value
    
    raise ValueError(f"Secret {secret_name} not found in Docker secrets or environment")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global agent
    logger.info(
        "Application startup initiated",
        extra={
            "component": "lifespan_manager",
            "phase": "startup",
            "app_title": "Agentic Legal RAG",
            "app_version": "1.0.0"
        }
    )

    try:
        logger.info("Loading API credentials...")
        pinecone_key = get_secret("PINECONE_API_KEY")
        google_key = get_secret("GOOGLE_API_KEY")
        logger.info("Credentials loaded successfully")
        
        os.environ["PINECONE_API_KEY"] = pinecone_key
        os.environ["GOOGLE_API_KEY"] = google_key

        model_cache_path = Path("/app/models")
        if model_cache_path.exists():
            cached_files = list(model_cache_path.rglob("*"))
            logger.info(f"Found {len(cached_files)} cached model files")
        else:
            logger.warning("Model cache directory not found")
        
        # Verify data directory
        data_path = Path("/app/data")
        if data_path.exists():
            json_files = list(data_path.rglob("*.json"))
            logger.info(f"Found {len(json_files)} JSON data files")
        else:
            logger.warning("Data directory not found")

        logger.info("Initializing AdvancedLegalAgent", extra={"component": "agent_initialization"})
        
        initialization_start = time.time()
        agent = AdvancedLegalAgent()
        initialization_time = time.time() - initialization_start
        
        logger.info(
            "AdvancedLegalAgent initialized successfully",
            extra={
                "component": "agent_initialization",
                "initialization_time_seconds": round(initialization_time, 2),
                "agent_status": "ready"
            }
        )
        
        yield
        
    except Exception as e:
        logger.critical(
            "Failed to initialize AdvancedLegalAgent during startup",
            extra={
                "component": "agent_initialization",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500, 
            detail="Internal Server Error: Failed to initialize legal research agent"
        )
    finally:
        logger.info(
            "Application shutdown initiated",
            extra={
                "component": "lifespan_manager",
                "phase": "shutdown"
            }
        )
        for key in ["PINECONE_API_KEY", "GOOGLE_API_KEY"]:
            os.environ.pop(key, None)
        
        # Cleanup resources
        if agent is not None:
            logger.info("Cleaning up agent resources", extra={"component": "cleanup"})
            # Add any cleanup logic here if needed
        
        logger.info(
            "Application shutdown completed",
            extra={
                "component": "lifespan_manager",
                "phase": "shutdown_complete"
            }
        )

app = FastAPI(
    title="Agentic Legal RAG",
    description="Production-ready Legal RAG system with hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to log all HTTP requests and responses."""
    start_time = time.time()
    
    # Extract request details
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    request_id = request.headers.get("x-request-id", f"req_{int(time.time() * 1000)}")
    
    # Log request start
    logger.info(
        "Request started",
        extra={
            "component": "http_middleware",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log successful response
        log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            execution_time=processing_time,
            user_id=None  # Add user identification logic here if available
        )
        
        logger.info(
            "Request completed successfully",
            extra={
                "component": "http_middleware",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time_seconds": round(processing_time, 4),
                "response_content_type": response.headers.get("content-type"),
                "response_content_length": response.headers.get("content-length")
            }
        )
        
        # Add request ID to response headers
        response.headers["x-request-id"] = request_id
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log error response
        logger.error(
            "Request failed with exception",
            extra={
                "component": "http_middleware",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "processing_time_seconds": round(processing_time, 4),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        # Log to access log as well
        log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            execution_time=processing_time
        )
        
        raise e


def get_agent():
    """Get the initialized agent instance."""
    global agent
    if agent is None:
        logger.error(
            "Agent access attempted but agent not initialized",
            extra={
                "component": "agent_accessor",
                "agent_status": "not_initialized"
            }
        )
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: Legal research agent not initialized"
        )
    
    logger.debug(
        "Agent instance accessed",
        extra={
            "component": "agent_accessor",
            "agent_status": "ready"
        }
    )
    return agent


from app.api.routes import router
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint providing basic API information."""
    logger.info(
        "Root endpoint accessed",
        extra={
            "component": "root_endpoint",
            "endpoint": "/"
        }
    )
    
    return {
        "message": "AI Legal Research Assistant API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    agent_status = agent is not None
    
    health_data = {
        "status": "healthy" if agent_status else "degraded",
        "agent_loaded": agent_status,
        "timestamp": time.time(),
        "version": "1.0.0"
    }
    
    if agent_status:
        logger.debug(
            "Health check passed",
            extra={
                "component": "health_check",
                "agent_status": "ready",
                "overall_status": "healthy"
            }
        )
    else:
        logger.warning(
            "Health check failed - agent not loaded",
            extra={
                "component": "health_check",
                "agent_status": "not_loaded",
                "overall_status": "degraded"
            }
        )
    
    return health_data