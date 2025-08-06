import google.generativeai as genai
import os
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from src.data_models import EnrichedChunk

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    response: str
    contexts_used: int
    rhetorical_structure: Dict[str, Any]
    legal_citations: List[str]
    confidence_score: float
    generation_time: float
    metadata: Dict[str, Any]

class RhetoricallyAwareGenerator:
    """Gemini-powered legal text generator"""
    
    def __init__(self, device: str = None):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini legal generator initialized")
    
    def generate_legal_response(self, query: str, chunks: List[EnrichedChunk], 
                              max_length: int = None, temperature: float = None) -> GenerationResult:
        """Generate legal response using Gemini"""
        start_time = time.time()
        
        try:
            # Create context from chunks
            context_parts = []
            for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
                context_parts.append(f"Context {i+1}: {chunk.text[:500]}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""As a legal expert, analyze the following context and answer the query comprehensively:

LEGAL CONTEXT:
{context}

QUERY: {query}

Provide a detailed legal analysis based on the context provided. Include relevant case law, statutes, and legal principles where applicable."""

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.2,
                    top_p=0.8
                )
            )
            
            generated_text = response.text if response.text else "Unable to generate response"
            generation_time = time.time() - start_time
            
            # Extract citations (basic)
            citations = []
            import re
            citation_patterns = [
                r'\b\d{4}\s+\w+\s+\d+\b',
                r'[Ss]ection\s+\d+',
                r'[Aa]rticle\s+\d+'
            ]
            
            for pattern in citation_patterns:
                citations.extend(re.findall(pattern, generated_text))
            
            return GenerationResult(
                response=generated_text,
                contexts_used=len(chunks),
                rhetorical_structure={'template_used': 'general'},
                legal_citations=list(set(citations)),
                confidence_score=0.8,
                generation_time=generation_time,
                metadata={'prompt_length': len(prompt)}
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return GenerationResult(
                response=f"I apologize, but I encountered an error: {str(e)}",
                contexts_used=0,
                rhetorical_structure={},
                legal_citations=[],
                confidence_score=0.0,
                generation_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
