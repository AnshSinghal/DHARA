from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Annotated, Dict, Any
import operator
from app.core.custom_tools import DocumentSummarizerTool, BasicLegalQueryTool, LegalEntityExtractorTool
from app.core.custom_retriever import HybridRetriever, RetrieverConfig
import json
import logging
import os
import time
import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalAgentState(TypedDict):
    messages: List[Any]  
    original_query: str
    query_analysis: str
    extracted_entities: str
    basic_answer: str
    final_analysis: str
    confidence_score: float
    iteration_count: int
    needs_more_analysis: bool



class AdvancedLegalAgent:
    '''Advanced legal research agent with multi-step reasoning using LangGraph'''

    def __init__(self):
        """Initialize AdvancedLegalAgent with LangGraph workflow and tools"""
        logger.info("Initializing AdvancedLegalAgent with LangGraph workflow")
        start_time = time.time()
        
        try:
            # Initialize configuration
            logger.info("Loading configuration from environment variables")
            PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
            
            if not PINECONE_API_KEY:
                logger.warning("PINECONE_API_KEY not found in environment variables")
            if not GOOGLE_API_KEY:
                logger.warning("GOOGLE_API_KEY not found in environment variables")
            
            self.config = RetrieverConfig(
                PINECONE_API_KEY=PINECONE_API_KEY,
                GOOGLE_API_KEY=GOOGLE_API_KEY
            )
            logger.info("Configuration loaded successfully")

            # Initialize retriever
            logger.info("Initializing HybridRetriever for advanced agent")
            retriever_start_time = time.time()
            self.retriever = HybridRetriever(self.config)
            retriever_time = time.time() - retriever_start_time
            logger.info(f"HybridRetriever initialized in {retriever_time:.3f} seconds")
            
            # Initialize LLM
            logger.info("Configuring Gemini LLM for workflow coordination")
            llm_start_time = time.time()
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
            llm_time = time.time() - llm_start_time
            logger.info(f"Gemini LLM configured in {llm_time:.3f} seconds")

            # Initialize tools
            logger.info("Initializing specialized legal tools for workflow")
            tools_start_time = time.time()
            
            case_dir = os.path.abspath("data/processed/merged")
            logger.info(f"Using case directory for entity extraction: {case_dir}")
            
            self.basic_query_tool = BasicLegalQueryTool(
                retriever=self.retriever,
                GOOGLE_API_KEY=GOOGLE_API_KEY
            )
            self.entity_extractor = LegalEntityExtractorTool(
                retriever=self.retriever,
                GOOGLE_API_KEY=GOOGLE_API_KEY,
                case_dir=case_dir
            )
            self.summarizer = DocumentSummarizerTool(
                GOOGLE_API_KEY=GOOGLE_API_KEY
            )
            
            tools_time = time.time() - tools_start_time
            logger.info(f"All specialized tools initialized in {tools_time:.3f} seconds")

            # Build workflow
            logger.info("Building LangGraph workflow for multi-step legal research")
            workflow_start_time = time.time()
            self.workflow = self._build_workflow()
            workflow_time = time.time() - workflow_start_time
            logger.info(f"LangGraph workflow compiled in {workflow_time:.3f} seconds")
            
            # Log initialization completion
            total_time = time.time() - start_time
            logger.info(f"AdvancedLegalAgent initialization completed successfully")
            logger.info(f"Total initialization time: {total_time:.3f} seconds")
            logger.info(f"Breakdown - Retriever: {retriever_time:.3f}s, LLM: {llm_time:.3f}s, Tools: {tools_time:.3f}s, Workflow: {workflow_time:.3f}s")
            
        except Exception as e:
            init_time = time.time() - start_time
            logger.error(f"Failed to initialize AdvancedLegalAgent after {init_time:.3f} seconds: {str(e)}")
            raise e

    def _build_workflow(self) -> StateGraph:
        '''Build the LangGraph workflow for the agent'''
        logger.info("Building LangGraph workflow with multi-step legal research nodes")
        
        workflow = StateGraph(LegalAgentState)

        logger.info("Adding workflow nodes for legal research pipeline")
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("extract_entities", self.extract_legal_entities)
        workflow.add_node("get_basic_answer", self.get_basic_legal_answer)
        workflow.add_node("synthesize_analysis", self.synthesize_final_analysis)
        workflow.add_node("quality_check", self.quality_check)

        logger.info("Configuring workflow entry point and edges")
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "extract_entities")
        workflow.add_edge("extract_entities", "get_basic_answer")
        workflow.add_edge("get_basic_answer", "synthesize_analysis")
        workflow.add_edge("synthesize_analysis", "quality_check")

        logger.info("Adding conditional edges for iterative improvement")
        workflow.add_conditional_edges(
            "quality_check",
            self.should_continue,
            {
                "continue": "extract_entities",
                "finish": END
            }
        )
        
        logger.info("Compiling LangGraph workflow")
        compiled_workflow = workflow.compile()
        logger.info("LangGraph workflow compilation completed successfully")
        
        return compiled_workflow

    def analyze_query(self, state: LegalAgentState) -> LegalAgentState:
        '''Analyze the original legal query'''
        logger.info("Starting query analysis phase")
        start_time = time.time()
        
        query = state["original_query"]
        query_length = len(query)
        logger.info(f"Analyzing query - Length: {query_length} characters")
        
        # Log query preview for audit trail (truncated for privacy)
        query_preview = query[:100] + "..." if len(query) > 100 else query
        logger.info(f"Query preview: {query_preview}")
        
        analysis_prompt = f"""
        As a legal research expert in Indian Law, analyze this query and determine:
        
        1. Type of legal research needed (case law, statutory analysis, procedural guidance)
        2. Key legal concepts and areas of law involved
        3. Specific search terms and legal entities to look for
        4. Complexity level (simple question vs complex analysis needed)
        
        Query: {query}
        
        Provide a structured analysis that will guide the research approach.
        """
        
        try:
            logger.info("Sending query analysis request to Gemini LLM")
            llm_start_time = time.time()
            
            response = self.llm.invoke(analysis_prompt)
            
            llm_time = time.time() - llm_start_time
            total_time = time.time() - start_time
            
            # Update state
            state["query_analysis"] = response.content
            state["iteration_count"] = 0
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            
            analysis_length = len(response.content)
            state["messages"] = state["messages"] + [AIMessage(content=f"Query Analysis Complete: {response.content[:200]}...")]

            logger.info(f"Query analysis completed successfully")
            logger.info(f"Performance metrics - LLM call: {llm_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Analysis output length: {analysis_length} characters")
            
            return state
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Query analysis failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error details - Query length: {query_length}")
            
            # Provide fallback analysis
            state["query_analysis"] = f"Failed to analyze query automatically. Manual analysis required for: {query_preview}"
            state["iteration_count"] = 0
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Query analysis failed - proceeding with fallback approach")]
            
            return state
    
    def extract_legal_entities(self, state: LegalAgentState) -> LegalAgentState:
        """Extract relevant legal entities using the extractor tool"""
        logger.info("Starting legal entity extraction phase")
        start_time = time.time()
        
        query = state["original_query"]
        iteration = state.get("iteration_count", 0)
        
        logger.info(f"Extracting legal entities - Iteration: {iteration + 1}")
        
        try:
            # Use the entity extractor tool
            logger.info("Invoking LegalEntityExtractorTool for case analysis")
            extraction_start_time = time.time()
            
            entities_result = self.entity_extractor._run(query=query, top_k_cases=3)
            
            extraction_time = time.time() - extraction_start_time
            total_time = time.time() - start_time
            
            # Update state
            state["extracted_entities"] = entities_result
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Legal entities extracted from similar cases")]

            # Log extraction metrics
            entities_length = len(entities_result)
            logger.info(f"Legal entity extraction completed successfully")
            logger.info(f"Performance metrics - Extraction call: {extraction_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Extracted entities length: {entities_length} characters")
            
            return state
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Legal entity extraction failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error details - Query: {query[:100]}..., Iteration: {iteration}")
            
            # Provide fallback entities
            state["extracted_entities"] = "Entity extraction failed - proceeding with basic legal analysis"
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Entity extraction failed - using fallback approach")]
            
            return state

    def get_basic_legal_answer(self, state: LegalAgentState) -> LegalAgentState:
        """Get basic legal answer with automatic context from retriever"""
        logger.info("Starting basic legal answer generation phase")
        start_time = time.time()
        
        query = state["original_query"]
        iteration = state.get("iteration_count", 0)
        
        logger.info(f"Generating basic legal answer - Iteration: {iteration + 1}")
        
        try:
            # Prepare additional context from extracted entities
            logger.info("Preparing context from extracted legal entities")
            try:
                entities_data = json.loads(state["extracted_entities"])
                additional_context = f"Additional extracted legal elements: {entities_data.get('llm_analysis', '')}"
                logger.info("Successfully parsed entities data for context")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse entities data: {str(e)}")
                additional_context = "Additional context from entity extraction available."
            
            # Truncate context to prevent token limit issues
            if len(additional_context) > 1000:
                additional_context = additional_context[:1000] + "..."
                logger.info("Truncated additional context to prevent token limit issues")
            
            logger.info("Invoking BasicLegalQueryTool with hybrid retrieval")
            query_start_time = time.time()
            
            basic_answer = self.basic_query_tool._run(
                query=query,
                context=additional_context
            )
            
            query_time = time.time() - query_start_time
            total_time = time.time() - start_time
            
            # Update state
            state["basic_answer"] = basic_answer
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Basic legal analysis completed with retrieved context")]

            # Log answer metrics
            answer_length = len(basic_answer)
            logger.info(f"Basic legal answer generation completed successfully")
            logger.info(f"Performance metrics - Query tool call: {query_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Generated answer length: {answer_length} characters")
            
            return state
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Basic legal answer generation failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error details - Query: {query[:100]}..., Iteration: {iteration}")
            
            # Provide fallback answer
            state["basic_answer"] = f"Unable to generate comprehensive answer. Basic guidance: Please consult relevant legal provisions for: {query[:200]}..."
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Basic legal analysis failed - using fallback response")]
            
            return state

    def synthesize_final_analysis(self, state: LegalAgentState) -> LegalAgentState:
        """Synthesize comprehensive final analysis from all gathered information"""
        logger.info("Starting final legal analysis synthesis")
        start_time = time.time()
        
        try:
            # Gather all analysis components
            query = state["original_query"]
            query_analysis = state.get("query_analysis", "")
            extracted_entities = state.get("extracted_entities", "")
            basic_answer = state.get("basic_answer", "")
            iteration = state.get("iteration_count", 0)
            
            logger.info(f"Synthesizing final analysis - Iteration: {iteration + 1}")
            logger.info(f"Analysis components - Query analysis: {len(query_analysis)} chars, Basic answer: {len(basic_answer)} chars")
            
            # Create synthesis prompt
            logger.info("Constructing comprehensive synthesis prompt")
            synthesis_prompt = f"""
            Based on all the research conducted, provide a comprehensive legal analysis:
            
            Original Query: {query}
            Query Analysis: {query_analysis}
            Extracted Legal Entities: {extracted_entities}
            Basic Legal Answer (with retrieved case context): {basic_answer}
            
            Synthesize this information into a final, comprehensive legal analysis that includes:
            
            1. **Executive Summary**: Key findings and recommendations
            2. **Legal Framework**: Relevant statutes, provisions, and legal principles
            3. **Case Law Analysis**: Precedents and their application (from retrieved context)
            4. **Practical Implications**: How this applies in practice
            5. **Recommendations**: Next steps or considerations
            
            Ensure the response is well-structured, professional, and actionable.
            The basic legal answer already includes context from relevant cases - incorporate this effectively.
            """
            
            # Generate final synthesis
            logger.info("Invoking LLM for final analysis synthesis")
            synthesis_start_time = time.time()
            
            response = self.llm.invoke(synthesis_prompt)
            final_analysis = response.content
            
            synthesis_time = time.time() - synthesis_start_time
            total_time = time.time() - start_time
            
            # Update state with final analysis
            state["final_analysis"] = final_analysis
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Final analysis synthesized")]
            
            # Log synthesis metrics
            analysis_length = len(final_analysis)
            logger.info(f"Final analysis synthesis completed successfully")
            logger.info(f"Performance metrics - LLM synthesis: {synthesis_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Final analysis length: {analysis_length} characters")
            logger.info(f"Synthesis incorporated {len([x for x in [query_analysis, extracted_entities, basic_answer] if x])} analysis components")
            
            return state
            
        except Exception as e:
            total_time = time.time() - start_time
            query = state.get("original_query", "Unknown query")
            basic_answer = state.get("basic_answer", "")
            
            logger.error(f"Final analysis synthesis failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error context - Query: {query[:100]}..., Available components: {list(state.keys())}")
            
            # Provide fallback synthesis
            fallback_analysis = f"""
            Final Analysis for: {query}
            
            Based on available information:
            {basic_answer if basic_answer else 'Basic legal guidance: Please consult relevant legal provisions and seek professional advice.'}
            
            Note: Complete synthesis unavailable due to processing limitations. 
            Recommendation: Consult with legal professionals for comprehensive guidance.
            """
            
            state["final_analysis"] = fallback_analysis
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"] = state["messages"] + [AIMessage(content="Final analysis synthesis completed with limitations")]
            
            return state

        logger.info("Final analysis synthesis completed")
        return state
    
    def quality_check(self, state: LegalAgentState) -> LegalAgentState:
        """Check if the analysis is comprehensive enough with detailed quality assessment"""
        logger.info("Starting quality assessment of legal analysis")
        start_time = time.time()
        
        try:
            final_analysis = state.get("final_analysis", "")
            basic_answer = state.get("basic_answer", "")
            iteration_count = state.get("iteration_count", 0)
            
            logger.info(f"Quality check - Iteration: {iteration_count + 1}")
            logger.info(f"Analysis lengths - Final: {len(final_analysis)} chars, Basic: {len(basic_answer)} chars")
            
            # Comprehensive confidence calculation
            logger.info("Calculating analysis confidence score")
            confidence = 0.6  # Base confidence score
            
            # Length-based quality indicators
            if len(final_analysis) > 800:
                confidence += 0.15
                logger.info("High quality: Final analysis length exceeds 800 characters")
            elif len(final_analysis) > 400:
                confidence += 0.08
                logger.info("Medium quality: Final analysis length exceeds 400 characters")
            
            if len(basic_answer) > 500:
                confidence += 0.15
                logger.info("High quality: Basic answer length exceeds 500 characters")
            elif len(basic_answer) > 250:
                confidence += 0.08
                logger.info("Medium quality: Basic answer length exceeds 250 characters")
            
            # Content quality indicators
            quality_indicators = [
                ("legal", "Legal terminology present"),
                ("section", "Statutory references found"),
                ("case", "Case law references found"),
                ("court", "Court references found"),
                ("provision", "Legal provisions mentioned")
            ]
            
            content_to_check = (final_analysis + " " + basic_answer).lower()
            found_indicators = 0
            
            for indicator, description in quality_indicators:
                if indicator in content_to_check:
                    found_indicators += 1
                    confidence += 0.02
                    logger.info(f"Quality indicator found: {description}")
            
            # Cap confidence at 1.0
            final_confidence = min(confidence, 1.0)
            
            # Update state
            state["confidence_score"] = final_confidence
            state["iteration_count"] = iteration_count + 1
            state["needs_more_analysis"] = False
            
            total_time = time.time() - start_time
            
            logger.info(f"Quality assessment completed successfully")
            logger.info(f"Performance metrics - Assessment time: {total_time:.3f}s")
            logger.info(f"Quality metrics - Final confidence: {final_confidence:.3f}, Quality indicators found: {found_indicators}/{len(quality_indicators)}")
            logger.info(f"Decision: Analysis deemed sufficient, proceeding to completion")
            
            return state
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Quality assessment failed after {total_time:.3f}s: {str(e)}")
            
            # Fallback quality assessment
            state["confidence_score"] = 0.7  # Default moderate confidence
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            state["needs_more_analysis"] = False
            
            logger.warning(f"Using fallback confidence score: 0.7")
            return state

    def should_continue(self, state: LegalAgentState) -> str:
        """Decide whether to continue analysis or finish with detailed workflow decision logging"""
        logger.info("Evaluating workflow continuation decision")
        start_time = time.time()
        
        try:
            iteration_count = state.get("iteration_count", 0)
            confidence_score = state.get("confidence_score", 0.0)
            needs_more_analysis = state.get("needs_more_analysis", False)
            
            logger.info(f"Workflow decision parameters - Iteration: {iteration_count}, Confidence: {confidence_score:.3f}, Needs more: {needs_more_analysis}")
            
            # Decision logic with comprehensive logging
            max_iterations = 3
            min_confidence = 0.6
            
            if iteration_count >= max_iterations:
                decision = "finish"
                reason = f"Maximum iterations reached ({max_iterations})"
                logger.info(f"Workflow decision: {decision} - {reason}")
            elif confidence_score >= min_confidence and not needs_more_analysis:
                decision = "finish"
                reason = f"Quality threshold met (confidence: {confidence_score:.3f} >= {min_confidence})"
                logger.info(f"Workflow decision: {decision} - {reason}")
            elif needs_more_analysis and iteration_count < max_iterations:
                decision = "continue"
                reason = f"Additional analysis required (iteration {iteration_count + 1}/{max_iterations})"
                logger.info(f"Workflow decision: {decision} - {reason}")
            else:
                decision = "finish"
                reason = "Default completion path"
                logger.info(f"Workflow decision: {decision} - {reason}")
            
            total_time = time.time() - start_time
            
            logger.info(f"Workflow decision completed in {total_time:.3f}s")
            logger.info(f"Final workflow status: {decision.upper()}")
            
            if decision == "finish":
                logger.info("Legal research workflow completing - preparing final results")
            else:
                logger.info("Legal research workflow continuing - initiating next iteration")
            
            return decision
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Workflow decision failed after {total_time:.3f}s: {str(e)}")
            logger.warning("Defaulting to workflow completion due to decision error")
            return "finish"

    
    def research(self, query: str) -> Dict[str, Any]:
        """Execute the complete legal research workflow with comprehensive monitoring"""
        logger.info("="*80)
        logger.info("INITIATING ADVANCED LEGAL RESEARCH WORKFLOW")
        logger.info("="*80)
        
        workflow_start_time = time.time()
        
        try:
            logger.info(f"Research query: {query}")
            logger.info(f"Query length: {len(query)} characters")
            
            # Initialize workflow state
            logger.info("Initializing workflow state")
            initial_state = LegalAgentState(
                messages=[HumanMessage(content=query)],
                original_query=query,
                query_analysis="",
                extracted_entities="",
                basic_answer="",
                final_analysis="",
                confidence_score=0.0,
                iteration_count=0,
                needs_more_analysis=True
            )
            
            logger.info("Workflow state initialized successfully")
            logger.info(f"Initial state components: {list(initial_state.keys())}")
            
            # Execute workflow
            logger.info("Invoking LangGraph workflow execution")
            workflow_execution_start = time.time()
            
            final_state = self.workflow.invoke(initial_state)
            
            workflow_execution_time = time.time() - workflow_execution_start
            total_workflow_time = time.time() - workflow_start_time
            
            # Log workflow completion metrics
            logger.info("LangGraph workflow execution completed successfully")
            logger.info(f"Performance metrics - Workflow execution: {workflow_execution_time:.3f}s, Total time: {total_workflow_time:.3f}s")
            logger.info(f"Workflow completion - Iterations: {final_state.get('iteration_count', 0)}, Confidence: {final_state.get('confidence_score', 0.0):.3f}")
            
            # Prepare result dictionary
            result_dict = {
                "query": query,
                "final_analysis": final_state.get("final_analysis", ""),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "iterations": final_state.get("iteration_count", 0),
                "extracted_entities": final_state.get("extracted_entities", ""),
                "research_process": [msg.content for msg in final_state.get("messages", [])]
            }
            
            # Log final results summary
            logger.info("="*80)
            logger.info("ADVANCED LEGAL RESEARCH WORKFLOW COMPLETED")
            logger.info("="*80)
            logger.info(f"Results summary:")
            logger.info(f"- Final analysis length: {len(result_dict['final_analysis'])} characters")
            logger.info(f"- Research confidence: {result_dict['confidence_score']:.3f}")
            logger.info(f"- Workflow iterations: {result_dict['iterations']}")
            logger.info(f"- Research process steps: {len(result_dict['research_process'])}")
            logger.info(f"- Total execution time: {total_workflow_time:.3f}s")
            
            return result_dict
            
        except Exception as e:
            total_workflow_time = time.time() - workflow_start_time
            logger.error("="*80)
            logger.error("ADVANCED LEGAL RESEARCH WORKFLOW FAILED")
            logger.error("="*80)
            logger.error(f"Workflow execution failed after {total_workflow_time:.3f}s: {str(e)}")
            logger.error(f"Query that caused failure: {query[:200]}...")
            
            # Return error result
            return {
                "query": query,
                "final_analysis": f"Legal research workflow encountered an error: {str(e)}. Please try rephrasing your query or contact support.",
                "confidence_score": 0.0,
                "iterations": 0,
                "extracted_entities": "",
                "research_process": ["Workflow execution failed"]
            }

# Test the advanced agent with comprehensive logging
def test_advanced_agent():
    """Test function for advanced legal agent with detailed logging"""
    logger.info("="*100)
    logger.info("STARTING ADVANCED LEGAL AGENT TEST")
    logger.info("="*100)
    
    test_start_time = time.time()
    
    try:
        logger.info("Initializing AdvancedLegalAgent instance")
        agent = AdvancedLegalAgent()
        
        test_query = "What are the legal requirements and precedents for granting anticipatory bail in cases involving domestic violence under Section 498A IPC?"
        
        logger.info(f"Test query: {test_query}")
        logger.info("="*100)
        logger.info("EXECUTING ADVANCED AGENT TEST")
        logger.info("="*100)
        
        # Execute research
        result = agent.research(test_query)
        
        test_execution_time = time.time() - test_start_time
        
        # Display results
        print(f"\n{'='*100}")
        print(f"ADVANCED AGENT TEST RESULTS")
        print(f"{'='*100}")
        print(f"QUERY: {test_query}")
        print(f"{'='*100}")
        
        print(f"\nFINAL ANALYSIS:")
        print(f"{'-'*50}")
        print(result["final_analysis"])
        
        print(f"\nMETADATA:")
        print(f"Confidence Score: {result['confidence_score']:.2f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Test Execution Time: {test_execution_time:.3f}s")
        
        print(f"\nRESEARCH PROCESS:")
        for i, step in enumerate(result["research_process"], 1):
            print(f"{i}. {step}")
        
        logger.info("="*100)
        logger.info("ADVANCED LEGAL AGENT TEST COMPLETED SUCCESSFULLY")
        logger.info("="*100)
        logger.info(f"Test metrics - Total time: {test_execution_time:.3f}s, Final confidence: {result['confidence_score']:.3f}")
        logger.info(f"Test results - Analysis length: {len(result['final_analysis'])} chars, Process steps: {len(result['research_process'])}")
        
    except Exception as e:
        test_execution_time = time.time() - test_start_time
        logger.error("="*100)
        logger.error("ADVANCED LEGAL AGENT TEST FAILED")
        logger.error("="*100)
        logger.error(f"Test failed after {test_execution_time:.3f}s: {str(e)}")
        print(f"Test failed: {str(e)}")
        raise
    agent = AdvancedLegalAgent()
    
    test_query = "What are the legal requirements and precedents for granting anticipatory bail in cases involving domestic violence under Section 498A IPC?"
    
    print(f"\n{'='*100}")
    print(f"ADVANCED AGENT TEST")
    print(f"{'='*100}")
    print(f"QUERY: {test_query}")
    print(f"{'='*100}")
    
    result = agent.research(test_query)
    
    print(f"\nFINAL ANALYSIS:")
    print(f"{'-'*50}")
    print(result["final_analysis"])
    
    print(f"\nMETADATA:")
    print(f"Confidence Score: {result['confidence_score']:.2f}")
    print(f"Iterations: {result['iterations']}")
    
    print(f"\nRESEARCH PROCESS:")
    for i, step in enumerate(result["research_process"], 1):
        print(f"{i}. {step}")

if __name__ == "__main__":
    test_advanced_agent()