from langchain.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field, PrivateAttr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from typing import Type
import logging
import dotenv
import json
import os
import time
from .custom_retriever import HybridRetriever

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class DocumentSummarizerInput(BaseModel):
    '''Input schema for the DocumentSummarizerTool'''
    case_document: str = Field(description="Full text of the legal case document to summarize")
    focus_area: str = Field(default="general", description="Specific area to focus on: facts, judgment, analysis, or general")

class DocumentSummarizerTool(BaseTool):
    '''Tool for summarizing legal case documents'''

    name: str = "document_summarizer"
    description: str = "Summarizes legal case documents with focus on key facts, legal issues, and decisions"
    args_schema: Type[BaseModel] = DocumentSummarizerInput
    _llm: ChatGoogleGenerativeAI = PrivateAttr()
    _summary_prompt: ChatPromptTemplate = PrivateAttr()

    def __init__(self, GOOGLE_API_KEY: str, **kwargs):
        """Initialize DocumentSummarizerTool with Gemini LLM"""
        logger.info("Initializing DocumentSummarizerTool")
        start_time = time.time()
        
        try:
            super().__init__(**kwargs)
            
            logger.info("Configuring Gemini LLM for document summarization")
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
            
            logger.info("Creating document summarization prompt template")
            self._summary_prompt = ChatPromptTemplate.from_template(
                """You are a legal expert tasked with summarizing Indian court judgments.

Analyze the following legal case document and provide a structured summary covering:

1. **Case Details**: Case number, court, parties involved, judges
2. **Key Facts**: Chronological summary of events leading to the case
3. **Legal Issues**: Main questions/issues the court had to determine
4. **Legal Provisions**: Relevant statutes, sections, and articles cited
5. **Court's Analysis**: Key reasoning and legal principles applied
6. **Decision/Judgment**: Final ruling and orders passed
7. **Precedents Cited**: Important case laws referenced
8. **Significance**: Legal significance and implications

Focus Area: {focus_area}

Case Document:
{case_document}

Provide a comprehensive yet concise summary that captures all essential legal information."""
            )
            
            init_time = time.time() - start_time
            logger.info(f"DocumentSummarizerTool initialized successfully in {init_time:.3f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentSummarizerTool: {str(e)}")
            raise e

    def _run(self, case_document: str, focus_area: str = "general") -> str:
        '''Execute the document summarization'''
        logger.info("Starting document summarization task")
        start_time = time.time()
        
        # Log input parameters
        doc_length = len(case_document)
        logger.info(f"Input parameters - Document length: {doc_length} characters, Focus area: {focus_area}")
        
        # Validate input
        if not case_document or not case_document.strip():
            logger.error("Empty or invalid case document provided")
            return "Error: Empty case document provided"
        
        if doc_length > 100000:  # Log for very large documents
            logger.warning(f"Large document detected: {doc_length} characters")
        
        try:
            logger.info("Formatting summarization prompt")
            prompt = self._summary_prompt.format(
                case_document=case_document,
                focus_area=focus_area
            )
            
            logger.info("Sending request to Gemini LLM for document summarization")
            llm_start_time = time.time()
            
            response = self._llm.invoke(prompt)
            
            llm_time = time.time() - llm_start_time
            total_time = time.time() - start_time
            
            # Log response metrics
            response_length = len(response.content) if response.content else 0
            logger.info(f"Document summarization completed successfully")
            logger.info(f"Performance metrics - LLM call: {llm_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Output metrics - Response length: {response_length} characters")
            
            return response.content
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Document summarization failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error details - Document length: {doc_length}, Focus area: {focus_area}")
            return f"Error occurred during summarization: {str(e)}"
        
class LegalQueryInput(BaseModel):
    '''Input schema for the LegalQueryTool'''
    query: str = Field(description="Legal question or query to answer")
    context: str = Field(default="", description="Additional context or background information")

class BasicLegalQueryTool(BaseTool):
    '''Tool for answering legal queries'''

    name: str = "legal_query_tool"
    description: str = "Answers legal questions based on provided context and knowledge of Indian law"
    args_schema: Type[BaseModel] = LegalQueryInput
    _llm: ChatGoogleGenerativeAI = PrivateAttr()
    _query_prompt: ChatPromptTemplate = PrivateAttr()
    _retriever: HybridRetriever = PrivateAttr()

    def __init__(self, GOOGLE_API_KEY: str, retriever: HybridRetriever, model_name: str = "gemini-2.5-flash", **kwargs):
        """Initialize BasicLegalQueryTool with Gemini LLM"""
        logger.info("Initializing BasicLegalQueryTool")
        start_time = time.time()
        
        try:
            super().__init__(**kwargs)
            self._retriever = retriever
            
            logger.info(f"Configuring Gemini LLM with model: {model_name}")
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
            
            logger.info("Creating legal query response prompt template")
            self._query_prompt = ChatPromptTemplate.from_template(
            """You are an expert legal advisor specializing in Indian law. Answer the following legal query using the provided context from relevant legal cases and your knowledge of Indian law.


RELEVANT CASES AND CONTEXT:
{retrieved_context}


ADDITIONAL CONTEXT:
{additional_context}


LEGAL QUERY: {query}


Instructions:
1. Use the retrieved cases and context to provide accurate legal guidance
2. Cite relevant legal provisions (Acts, Sections, Articles) from the context
3. Reference case law and precedents mentioned in the retrieved cases
4. Provide practical implications and procedural aspects
5. Highlight key legal principles involved
6. Be precise, professional, and well-structured
7. If the retrieved context doesn't contain sufficient information, clearly state this and provide general legal guidance


Provide a detailed legal analysis and answer:"""
        )
            
            init_time = time.time() - start_time
            logger.info(f"BasicLegalQueryTool initialized successfully in {init_time:.3f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize BasicLegalQueryTool: {str(e)}")
            raise e

    def _run(self, query: str, context: str = "") -> str:
        '''Execute the legal query answering'''
        logger.info("Starting legal query processing")
        start_time = time.time()

        # Retrieve relevant cases
        logger.info(f"Retrieving relevant cases for query: {query}")
        retrieved_results = self._retriever.retrieve(query, top_k=3)
        
        # Log input parameters
        query_length = len(query)
        context_length = len(context)
        logger.info(f"Input parameters - Query length: {query_length} characters, Context length: {context_length} characters")
        
        # Validate input
        if not query or not query.strip():
            logger.error("Empty or invalid query provided")
            return "Error: Empty query provided"
        
        context_parts = []
        for i, result in enumerate(retrieved_results, 1):
            text_snippet = result['metadata'].get('text', '')
            relevance_score = round(result.get('bert_score', 0), 3)
            context_parts.append(f"Context {i}: {text_snippet} (Relevance: {relevance_score})")

        retrieved_context = "\n\n".join(context_parts)

        try:
            logger.info("Formatting legal query prompt")
            prompt = self._query_prompt.format(
                query=query,
                additional_context=context,
                retrieved_context=retrieved_context,
            )
            
            logger.info("Sending legal query to Gemini LLM")
            llm_start_time = time.time()
            
            response = self._llm.invoke(prompt)
            
            llm_time = time.time() - llm_start_time
            total_time = time.time() - start_time
            
            # Log response metrics
            response_length = len(response.content) if response.content else 0
            logger.info(f"Legal query processing completed successfully")
            logger.info(f"Performance metrics - LLM call: {llm_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Output metrics - Response length: {response_length} characters")
            
            return response.content
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Legal query processing failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error details - Query length: {query_length}, Context length: {context_length}")
            return f"Error occurred while processing legal query: {str(e)}"
        

class ExtractLegalEntitiesInput(BaseModel):
    '''Input schema for the LegalEntityExtractorTool'''
    query: str = Field(description="Legal query to find relevant statutes, provisions, and precedents for")
    top_k_cases: int = Field(default=2, description="Number of top cases to analyze for extraction")

class LegalEntityExtractorTool(BaseTool):
    '''Extracts relevant statutes, provisions, and precedents by analyzing top retrieved cases'''

    name: str = "legal_entity_extractor"
    description: str = "Finds relevant statutes, provisions, and precedents by analyzing similar cases from the database"
    args_schema: Type[BaseModel] = ExtractLegalEntitiesInput
    _llm: ChatGoogleGenerativeAI = PrivateAttr()   
    _retriever: HybridRetriever = PrivateAttr()
    _case_dir: str = PrivateAttr() 
    _extraction_prompt: ChatPromptTemplate = PrivateAttr()  

    def __init__(self, retriever: HybridRetriever, GOOGLE_API_KEY: str, model_name: str = "gemini-2.5-flash", case_dir: str = "cases", **kwargs):
        """Initialize LegalEntityExtractorTool with retriever and Gemini LLM"""
        logger.info("Initializing LegalEntityExtractorTool")
        start_time = time.time()
        
        try:
            super().__init__(**kwargs)
            
            # Validate and set retriever
            if not retriever:
                raise ValueError("Retriever instance is required")
            
            self._retriever = retriever
            self._case_dir = case_dir
            
            logger.info(f"Configuring case directory: {case_dir}")
            if not os.path.exists(case_dir):
                logger.warning(f"Case directory does not exist: {case_dir}")
            
            logger.info(f"Configuring Gemini LLM with model: {model_name}")
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
            
            logger.info("Creating legal entity extraction prompt template")
            self._extraction_prompt = ChatPromptTemplate.from_template(
                """You are a Indian legal research expert in Indian law. Based on the user query and the extracted legal elements from similar cases, identify the most relevant statutes, provisions, and precedents that apply to the query.
User Query: {query}

Extracted Legal Elements from Similar Cases:
Statutes: {statutes}
Provisions: {provisions}  
Precedents: {precedents}

Your task:
1. Analyze which statutes are most relevant to the query
2. Identify the specific provisions that apply
3. Determine which precedents are most applicable
4. Explain the relevance of each element to the query

Provide a structured response with:
- **Relevant Statutes**: List with brief explanation of relevance
- **Applicable Provisions**: Specific sections/articles with their significance  
- **Key Precedents**: Case names with their legal significance
- **Legal Framework**: How these elements work together for this query

Response:"""
            )
            
            init_time = time.time() - start_time
            logger.info(f"LegalEntityExtractorTool initialized successfully in {init_time:.3f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize LegalEntityExtractorTool: {str(e)}")
            raise e

    def _run(self, query: str, top_k_cases: int = 2) -> str:
        '''Extract relevant legal elements'''
        logger.info("Starting legal entity extraction process")
        start_time = time.time()
        
        # Log input parameters
        query_length = len(query)
        logger.info(f"Input parameters - Query length: {query_length} characters, Top K cases: {top_k_cases}")
        
        # Validate input
        if not query or not query.strip():
            logger.error("Empty or invalid query provided for legal entity extraction")
            return "Error: Empty query provided"
        
        if top_k_cases <= 0:
            logger.error(f"Invalid top_k_cases value: {top_k_cases}")
            return "Error: top_k_cases must be positive"
        
        # Initialize collections for legal elements
        all_statutes = set()
        all_provisions = set()
        all_precedents = set()
        
        try:
            logger.info(f"Retrieving {top_k_cases} similar cases using hybrid retriever")
            retrieval_start_time = time.time()
            
            retrieval_results = self._retriever.retrieve(query, top_k=top_k_cases)
            
            retrieval_time = time.time() - retrieval_start_time
            logger.info(f"Retrieved {len(retrieval_results)} cases in {retrieval_time:.3f} seconds")
            
            if not retrieval_results:
                logger.warning("No similar cases found for the query")
                return "No similar cases found in the database for this query"
            
            # Process each retrieved case
            processed_cases = 0
            failed_cases = 0
            
            for i, case in enumerate(retrieval_results, 1):
                try:
                    logger.debug(f"Processing case {i}/{len(retrieval_results)}")
                    
                    # Extract case metadata
                    case_metadata = case.get('metadata', {})
                    case_number_str = case_metadata.get('case_number', '0')
                    
                    # Handle case number conversion safely
                    try:
                        case_number = int(case_number_str) if case_number_str else 0
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid case number format: {case_number_str}, defaulting to 0")
                        case_number = 0
                    
                    case_file = os.path.join(self._case_dir, f"merged_case{case_number}.json")
                    
                    if not os.path.exists(case_file):
                        logger.warning(f"Case file not found: {case_file}")
                        failed_cases += 1
                        continue
                    
                    logger.debug(f"Processing case file: {case_file}")
                    file_start_time = time.time()
                    
                    with open(case_file, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                    
                    file_time = time.time() - file_start_time
                    logger.debug(f"Loaded case data in {file_time:.3f} seconds")
                    
                    # Extract precedents from precedent clusters
                    precedent_clusters = case_data.get("precedent_clusters", {})
                    if precedent_clusters:
                        for cluster_precedents in precedent_clusters.values():
                            if isinstance(cluster_precedents, (list, set)):
                                all_precedents.update(cluster_precedents)
                            else:
                                logger.warning(f"Unexpected precedent cluster format in case {case_number}")
                    
                    # Extract statutes and provisions from provision_counts
                    provision_counts = case_data.get("provision_counts", {})
                    if provision_counts:
                        for provision in provision_counts.keys():
                            all_statutes.add(provision)
                            all_provisions.add(provision)
                    
                    processed_cases += 1
                    logger.debug(f"Successfully processed case {case_number}")
                    
                except Exception as case_error:
                    failed_cases += 1
                    logger.error(f"Error processing case {i}: {str(case_error)}")
                    continue
            
            processing_time = time.time() - retrieval_start_time
            logger.info(f"Case processing completed - Processed: {processed_cases}, Failed: {failed_cases}, Time: {processing_time:.3f}s")
            
            if processed_cases == 0:
                logger.error("No cases could be processed successfully")
                return "Error: Unable to process any retrieved cases"

        except Exception as e:
            retrieval_time = time.time() - start_time
            logger.error(f"Error during case retrieval and processing after {retrieval_time:.3f}s: {str(e)}")
            return f"Error occurred during case processing: {str(e)}"
        
        # Prepare extracted elements for LLM analysis
        extracted_elements = {
            'statutes': list(all_statutes),
            'provisions': list(all_provisions),
            'precedents': list(all_precedents)
        }
        
        # Log extraction statistics
        logger.info(f"Legal elements extracted - Statutes: {len(extracted_elements['statutes'])}, "
                   f"Provisions: {len(extracted_elements['provisions'])}, "
                   f"Precedents: {len(extracted_elements['precedents'])}")
        
        try:
            logger.info("Formatting legal entity extraction prompt")
            prompt = self._extraction_prompt.format(
                query=query,
                statutes=", ".join(extracted_elements['statutes']) if extracted_elements['statutes'] else "None",
                provisions=", ".join(extracted_elements['provisions']) if extracted_elements['provisions'] else "None",
                precedents=", ".join(extracted_elements['precedents']) if extracted_elements['precedents'] else "None"
            )
            
            logger.info("Sending legal entity analysis request to Gemini LLM")
            llm_start_time = time.time()
            
            response = self._llm.invoke(prompt)
            
            llm_time = time.time() - llm_start_time
            total_time = time.time() - start_time
            
            # Log final metrics
            response_length = len(response.content) if response.content else 0
            logger.info(f"Legal entity extraction completed successfully")
            logger.info(f"Performance metrics - LLM call: {llm_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Output metrics - Response length: {response_length} characters")
            
            return response.content
        
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Legal entity extraction failed during LLM analysis after {total_time:.3f}s: {str(e)}")
            return f"Error occurred during legal entity analysis: {str(e)}"