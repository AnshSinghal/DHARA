import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    T5Tokenizer, T5ForConditionalGeneration,
    GenerationConfig, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
)
import numpy as np
import logging
import re
from dataclasses import dataclass

from src.data_models import EnrichedChunk, LegalMetadata
from config.settings import settings

logger = logging.getLogger(__name__)

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

class LegalStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for legal text generation"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.legal_endings = [
            "therefore", "accordingly", "in conclusion", "held that",
            "we conclude", "it is ordered", "judgment is", "appeal is"
        ]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] >= self.max_length:
            return True
        
        # Check for legal conclusion patterns
        if input_ids.shape[-1] > 20:  # Only check after some generation
            recent_text = self.tokenizer.decode(input_ids[0][-50:], skip_special_tokens=True).lower()
            if any(ending in recent_text for ending in self.legal_endings):
                # Check if sentence is complete
                if recent_text.endswith('.') or recent_text.endswith('!'):
                    return True
        
        return False

class RhetoricallyAwareGenerator:
    """Legal text generator with rhetorical role awareness"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.GENERATION_MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading rhetorically-aware generator: {self.model_name}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Legal prompt templates based on rhetorical roles
        self.rhetorical_templates = self._build_rhetorical_templates()
        
        # Legal citation patterns
        self.citation_patterns = [
            r'\b\d{4}\s+\w+\s+\d+\b',  # Case citations
            r'\b(?:section|sec\.?)\s*\d+(?:\([a-zA-Z0-9]+\))?',  # Section references
            r'\b(?:article|art\.?)\s*\d+(?:\([a-zA-Z0-9]+\))?',  # Article references
        ]
        
        logger.info("Rhetorically-aware generator initialized")
    
    def _load_model(self):
        """Load the generation model"""
        try:

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True, quantization_config=bnb_config,
                device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=True, quantization_config=bnb_config,
                device_map="auto")

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            
            # Generation configuration
            self.generation_config = GenerationConfig(
                max_length=settings.MAX_GENERATION_LENGTH,
                temperature=settings.TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        except Exception as e:
            logger.error(f"Error loading generation model: {e}")
            raise
    
    def _build_rhetorical_templates(self) -> Dict[str, str]:
        """Build templates for different rhetorical roles"""
        
        return {
            'FAC': """Based on the factual background provided:
{context}

Question: {query}

Factual Analysis: The key facts in this matter are""",
            
            'RulingByPresentCourt': """Given the court's decision context:
{context}

Question: {query}

Court's Ruling: The court held that""",
            
            'RatioOfTheDecision': """Based on the legal reasoning:
{context}

Question: {query}

Legal Ratio: The fundamental legal principle established is that""",
            
            'Precedent': """Considering the precedent citations:
{context}

Question: {query}

Precedent Analysis: The relevant case law establishes that""",
            
            'Statute': """Based on the statutory provisions:
{context}

Question: {query}

Statutory Interpretation: The applicable law provides that""",
            
            'Argument': """Given the legal arguments presented:
{context}

Question: {query}

Legal Argument: The key contention is that""",
            
            'general': """Based on the legal context provided:
{context}

Question: {query}

Legal Analysis:"""
        }
    
    def analyze_context_structure(self, chunks: List[EnrichedChunk]) -> Dict[str, Any]:
        """Analyze the rhetorical structure of retrieved contexts"""
        
        structure_analysis = {
            'dominant_roles': {},
            'role_distribution': {},
            'entity_distribution': {},
            'precedent_density': 0,
            'statute_density': 0,
            'context_coherence': 0.0
        }
        
        total_chunks = len(chunks)
        if total_chunks == 0:
            return structure_analysis
        
        # Analyze rhetorical roles
        all_roles = []
        for chunk in chunks:
            all_roles.extend(chunk.metadata.rhetorical_roles)
        
        role_counts = {}
        for role in all_roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        structure_analysis['role_distribution'] = role_counts
        structure_analysis['dominant_roles'] = sorted(
            role_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        
        # Analyze entities
        all_entities = []
        for chunk in chunks:
            all_entities.extend(chunk.metadata.entity_types)
        
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        structure_analysis['entity_distribution'] = entity_counts
        
        # Calculate precedent and statute density
        total_precedents = sum(chunk.metadata.precedent_count for chunk in chunks)
        total_statutes = sum(chunk.metadata.statute_count for chunk in chunks)
        
        structure_analysis['precedent_density'] = total_precedents / total_chunks
        structure_analysis['statute_density'] = total_statutes / total_chunks
        
        # Calculate context coherence (based on role consistency)
        if structure_analysis['dominant_roles']:
            dominant_role_count = structure_analysis['dominant_roles'][0][1]
            structure_analysis['context_coherence'] = dominant_role_count / len(all_roles)
        
        return structure_analysis
    
    def select_optimal_template(self, query: str, structure_analysis: Dict[str, Any]) -> str:
        """Select the most appropriate template based on query and context structure"""
        
        query_lower = query.lower()
        
        # Query-based template selection
        if any(term in query_lower for term in ['facts', 'circumstances', 'background']):
            if 'FAC' in [role for role, _ in structure_analysis['dominant_roles']]:
                return 'FAC'
        
        elif any(term in query_lower for term in ['held', 'decided', 'ruling', 'judgment']):
            if 'RulingByPresentCourt' in [role for role, _ in structure_analysis['dominant_roles']]:
                return 'RulingByPresentCourt'
        
        elif any(term in query_lower for term in ['ratio', 'principle', 'reasoning']):
            if 'RatioOfTheDecision' in [role for role, _ in structure_analysis['dominant_roles']]:
                return 'RatioOfTheDecision'
        
        elif any(term in query_lower for term in ['precedent', 'case law']):
            if structure_analysis['precedent_density'] > 2:
                return 'Precedent'
        
        elif any(term in query_lower for term in ['section', 'act', 'statute']):
            if structure_analysis['statute_density'] > 1:
                return 'Statute'
        
        # Context-based selection if query-based fails
        if structure_analysis['dominant_roles']:
            dominant_role = structure_analysis['dominant_roles'][0][0]
            if dominant_role in self.rhetorical_templates:
                return dominant_role
        
        return 'general'
    
    def create_structured_context(self, chunks: List[EnrichedChunk], 
                                 max_context_length: int = 1500) -> str:
        """Create well-structured context from chunks with rhetorical organization"""
        
        if not chunks:
            return ""
        
        # Group chunks by rhetorical roles
        role_groups = {}
        for chunk in chunks:
            primary_role = chunk.metadata.primary_role
            if primary_role not in role_groups:
                role_groups[primary_role] = []
            role_groups[primary_role].append(chunk)
        
        # Define preferred order for rhetorical roles
        preferred_order = ['FAC', 'Argument', 'Precedent', 'Statute', 
                          'RatioOfTheDecision', 'RulingByPresentCourt']
        
        context_parts = []
        current_length = 0
        
        # Add contexts in rhetorical order
        for role in preferred_order:
            if role in role_groups and current_length < max_context_length:
                role_display = settings.RHETORICAL_ROLES.get(role, role)
                context_parts.append(f"\n=== {role_display} ===")
                
                for chunk in role_groups[role]:
                    chunk_text = chunk.text
                    if current_length + len(chunk_text) > max_context_length:
                        # Truncate to fit
                        remaining = max_context_length - current_length
                        chunk_text = chunk_text[:remaining] + "..."
                    
                    context_parts.append(chunk_text)
                    current_length += len(chunk_text)
                    
                    if current_length >= max_context_length:
                        break
        
        # Add remaining roles if space allows
        for role, chunks_list in role_groups.items():
            if role not in preferred_order and current_length < max_context_length:
                role_display = settings.RHETORICAL_ROLES.get(role, role)
                context_parts.append(f"\n=== {role_display} ===")
                
                for chunk in chunks_list:
                    chunk_text = chunk.text
                    if current_length + len(chunk_text) > max_context_length:
                        remaining = max_context_length - current_length
                        chunk_text = chunk_text[:remaining] + "..."
                    
                    context_parts.append(chunk_text)
                    current_length += len(chunk_text)
                    
                    if current_length >= max_context_length:
                        break
        
        return "\n".join(context_parts)
    
    def generate_legal_response(self, query: str, chunks: List[EnrichedChunk],
                               max_length: int = None, temperature: float = None) -> GenerationResult:
        """Generate contextually-aware legal response"""
        
        import time
        start_time = time.time()
        
        try:
            # Analyze context structure
            structure_analysis = self.analyze_context_structure(chunks)
            
            # Select optimal template
            template_key = self.select_optimal_template(query, structure_analysis)
            template = self.rhetorical_templates[template_key]
            
            # Create structured context
            structured_context = self.create_structured_context(chunks)
            
            # Create prompt
            prompt = template.format(context=structured_context, query=query)
            
            # Tokenize
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512  # Leave room for generation
            ).to(self.device)
            
            # Set up generation parameters
            gen_config = self.generation_config
            if max_length:
                gen_config.max_length = max_length
            if temperature:
                gen_config.temperature = temperature
            
            # Create stopping criteria
            stopping_criteria = StoppingCriteriaList([
                LegalStoppingCriteria(self.tokenizer, gen_config.max_length)
            ])
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=gen_config,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if template_key == 'general':
                response = full_response.split("Legal Analysis:")[-1].strip()
            else:
                # Extract based on template structure
                template_end_markers = {
                    'FAC': 'Factual Analysis: The key facts in this matter are',
                    'RulingByPresentCourt': 'Court\'s Ruling: The court held that',
                    'RatioOfTheDecision': 'Legal Ratio: The fundamental legal principle established is that',
                    'Precedent': 'Precedent Analysis: The relevant case law establishes that',
                    'Statute': 'Statutory Interpretation: The applicable law provides that',
                    'Argument': 'Legal Argument: The key contention is that'
                }
                
                marker = template_end_markers.get(template_key, '')
                if marker and marker in full_response:
                    response = full_response.split(marker)[-1].strip()
                else:
                    response = full_response[len(prompt):].strip()
            
            # Post-process response
            response = self._post_process_response(response)
            
            # Extract legal citations
            legal_citations = self._extract_legal_citations(response)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(response, structure_analysis)
            
            # Create rhetorical structure analysis
            rhetorical_structure = {
                'template_used': template_key,
                'dominant_roles': [role for role, _ in structure_analysis['dominant_roles']],
                'context_coherence': structure_analysis['context_coherence']
            }
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                response=response,
                contexts_used=len(chunks),
                rhetorical_structure=rhetorical_structure,
                legal_citations=legal_citations,
                confidence_score=confidence_score,
                generation_time=generation_time,
                metadata={
                    'structure_analysis': structure_analysis,
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    'template_key': template_key
                }
            )
            
        except Exception as e:
            logger.error(f"Error in legal text generation: {e}")
            return GenerationResult(
                response="I apologize, but I encountered an error while generating a response to your legal query.",
                contexts_used=0,
                rhetorical_structure={},
                legal_citations=[],
                confidence_score=0.0,
                generation_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response"""
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Remove repetitive phrases
        lines = response.split('\n')
        seen_lines = set()
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                seen_lines.add(line)
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _extract_legal_citations(self, text: str) -> List[str]:
        """Extract legal citations from generated text"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def _calculate_confidence_score(self, response: str, 
                                   structure_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated response"""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on context coherence
        confidence += structure_analysis['context_coherence'] * 0.2
        
        # Boost confidence based on response length (within reasonable bounds)
        response_length = len(response.split())
        if 20 <= response_length <= 150:  # Reasonable length
            confidence += 0.1
        
        # Boost confidence if legal citations are present
        citations = self._extract_legal_citations(response)
        if citations:
            confidence += min(len(citations) * 0.05, 0.15)
        
        # Boost confidence based on precedent/statute density in context
        precedent_density = structure_analysis.get('precedent_density', 0)
        statute_density = structure_analysis.get('statute_density', 0)
        
        if precedent_density > 1:
            confidence += 0.1
        if statute_density > 0.5:
            confidence += 0.1
        
        # Penalize for very short or very long responses
        if response_length < 10:
            confidence -= 0.2
        elif response_length > 200:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)

class LegalResponseEvaluator:
    """Evaluate the quality of generated legal responses"""
    
    def __init__(self):
        self.legal_terms = {
            'procedural': ['petition', 'application', 'appeal', 'review', 'writ', 'motion'],
            'substantive': ['contract', 'tort', 'criminal', 'constitutional', 'property', 'liability'],
            'evidence': ['evidence', 'proof', 'burden', 'prima facie', 'preponderance'],
            'reasoning': ['held', 'ratio', 'obiter', 'principle', 'doctrine', 'precedent']
        }
    
    def evaluate_response(self, generation_result: GenerationResult, 
                         query: str, chunks: List[EnrichedChunk]) -> Dict[str, Any]:
        """Comprehensive evaluation of generated legal response"""
        
        response = generation_result.response
        evaluation = {
            'overall_score': 0.0,
            'dimensions': {},
            'strengths': [],
            'weaknesses': []
        }
        
        # Dimension 1: Legal Terminology Usage
        legal_term_score = self._evaluate_legal_terminology(response)
        evaluation['dimensions']['legal_terminology'] = legal_term_score
        
        # Dimension 2: Structural Coherence
        structure_score = self._evaluate_structure_coherence(
            response, generation_result.rhetorical_structure
        )
        evaluation['dimensions']['structural_coherence'] = structure_score
        
        # Dimension 3: Context Relevance
        relevance_score = self._evaluate_context_relevance(response, query, chunks)
        evaluation['dimensions']['context_relevance'] = relevance_score
        
        # Dimension 4: Citation Quality
        citation_score = self._evaluate_citation_quality(generation_result.legal_citations)
        evaluation['dimensions']['citation_quality'] = citation_score
        
        # Dimension 5: Response Completeness
        completeness_score = self._evaluate_completeness(response, query)
        evaluation['dimensions']['completeness'] = completeness_score
        
        # Calculate overall score
        dimension_weights = {
            'legal_terminology': 0.2,
            'structural_coherence': 0.2,
            'context_relevance': 0.25,
            'citation_quality': 0.15,
            'completeness': 0.2
        }
        
        evaluation['overall_score'] = sum(
            evaluation['dimensions'][dim] * weight
            for dim, weight in dimension_weights.items()
        )
        
        # Identify strengths and weaknesses
        evaluation['strengths'] = self._identify_strengths(evaluation['dimensions'])
        evaluation['weaknesses'] = self._identify_weaknesses(evaluation['dimensions'])
        
        return evaluation
    
    def _evaluate_legal_terminology(self, response: str) -> float:
        """Evaluate usage of legal terminology"""
        response_lower = response.lower()
        response_words = set(response_lower.split())
        
        total_legal_terms = 0
        found_legal_terms = 0
        
        for category, terms in self.legal_terms.items():
            total_legal_terms += len(terms)
            found_legal_terms += len([term for term in terms if term in response_lower])
        
        # Calculate density
        legal_density = found_legal_terms / len(response_words) if response_words else 0
        
        # Score based on appropriate usage (not too sparse, not too dense)
        if 0.05 <= legal_density <= 0.15:  # 5-15% legal terms
            return min(legal_density * 10, 1.0)
        elif legal_density < 0.05:
            return legal_density * 20  # Penalize sparse usage
        else:
            return max(1.0 - (legal_density - 0.15) * 5, 0.0)  # Penalize overuse
    
    def _evaluate_structure_coherence(self, response: str, 
                                    rhetorical_structure: Dict[str, Any]) -> float:
        """Evaluate structural coherence of the response"""
        score = 0.5  # Base score
        
        # Boost for coherent template usage
        if rhetorical_structure.get('template_used') != 'general':
            score += 0.2
        
        # Boost for high context coherence
        context_coherence = rhetorical_structure.get('context_coherence', 0)
        score += context_coherence * 0.3
        
        # Check for logical flow markers
        flow_markers = ['therefore', 'however', 'furthermore', 'consequently', 'accordingly']
        if any(marker in response.lower() for marker in flow_markers):
            score += 0.1
        
        # Check for proper sentence structure
        sentences = response.split('.')
        if len(sentences) >= 2:  # At least 2 sentences
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_context_relevance(self, response: str, query: str, 
                                   chunks: List[EnrichedChunk]) -> float:
        """Evaluate relevance to provided context"""
        
        # Extract key terms from query and context
        query_terms = set(query.lower().split())
        context_terms = set()
        
        for chunk in chunks:
            context_terms.update(chunk.text.lower().split())
            context_terms.update(chunk.keywords)
            context_terms.update(chunk.legal_concepts)
        
        # Check overlap between response and context terms
        response_terms = set(response.lower().split())
        
        query_overlap = len(query_terms.intersection(response_terms)) / max(len(query_terms), 1)
        context_overlap = len(context_terms.intersection(response_terms)) / max(len(context_terms), 1)
        
        # Weighted combination
        relevance_score = 0.6 * query_overlap + 0.4 * context_overlap
        
        return min(relevance_score, 1.0)
    
    def _evaluate_citation_quality(self, citations: List[str]) -> float:
        """Evaluate quality of legal citations"""
        if not citations:
            return 0.5  # Neutral score for no citations
        
        score = 0.3  # Base score for having citations
        
        # Boost for multiple citations
        if len(citations) > 1:
            score += min(len(citations) * 0.1, 0.4)
        
        # Check citation format quality
        for citation in citations:
            if re.match(r'\d{4}', citation):  # Year in citation
                score += 0.1
            if 'section' in citation.lower() or 'sec' in citation.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_completeness(self, response: str, query: str) -> float:
        """Evaluate completeness of the response"""
        
        # Check if response ends properly
        completeness = 0.0
        
        if response.endswith('.') or response.endswith('!'):
            completeness += 0.3
        
        # Check response length relative to query complexity
        query_words = len(query.split())
        response_words = len(response.split())
        
        expected_min_length = max(query_words * 2, 20)
        expected_max_length = query_words * 10
        
        if expected_min_length <= response_words <= expected_max_length:
            completeness += 0.4
        elif response_words < expected_min_length:
            completeness += (response_words / expected_min_length) * 0.4
        else:
            completeness += max(0.4 - (response_words - expected_max_length) / 100, 0)
        
        # Check for conclusion indicators
        if any(phrase in response.lower() for phrase in ['therefore', 'in conclusion', 'accordingly']):
            completeness += 0.3
        
        return min(completeness, 1.0)
    
    def _identify_strengths(self, dimensions: Dict[str, float]) -> List[str]:
        """Identify strengths based on dimension scores"""
        strengths = []
        
        if dimensions['legal_terminology'] > 0.8:
            strengths.append("Excellent use of legal terminology")
        
        if dimensions['structural_coherence'] > 0.8:
            strengths.append("Well-structured and coherent response")
        
        if dimensions['context_relevance'] > 0.8:
            strengths.append("Highly relevant to query and context")
        
        if dimensions['citation_quality'] > 0.8:
            strengths.append("Strong legal citations")
        
        if dimensions['completeness'] > 0.8:
            strengths.append("Comprehensive and complete response")
        
        return strengths
    
    def _identify_weaknesses(self, dimensions: Dict[str, float]) -> List[str]:
        """Identify weaknesses based on dimension scores"""
        weaknesses = []
        
        if dimensions['legal_terminology'] < 0.5:
            weaknesses.append("Insufficient legal terminology")
        
        if dimensions['structural_coherence'] < 0.5:
            weaknesses.append("Poor structural organization")
        
        if dimensions['context_relevance'] < 0.5:
            weaknesses.append("Limited relevance to provided context")
        
        if dimensions['citation_quality'] < 0.3:
            weaknesses.append("Lacking legal citations")
        
        if dimensions['completeness'] < 0.5:
            weaknesses.append("Incomplete or inadequate response")
        
        return weaknesses
