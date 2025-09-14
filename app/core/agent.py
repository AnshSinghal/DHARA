from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.custom_tools import DocumentSummarizerTool, BasicLegalQueryTool, LegalEntityExtractorTool
from app.core.custom_retriever import HybridRetriever, RetrieverConfig
import os
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalAssistantAgent:
    '''Agent for assisting with legal queries and document analysis'''
    
    def __init__(self):
        """Initialize LegalAssistantAgent with retriever, LLM, and tools"""
        logger.info("Initializing LegalAssistantAgent")
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
            logger.info("Initializing HybridRetriever")
            retriever_start_time = time.time()
            self.retriever = HybridRetriever(self.config)
            retriever_time = time.time() - retriever_start_time
            logger.info(f"HybridRetriever initialized in {retriever_time:.3f} seconds")

            # Initialize LLM
            logger.info("Configuring Gemini LLM for agent coordination")
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
            logger.info("Initializing agent tools")
            tools_start_time = time.time()
            
            case_dir = os.path.abspath("data/processed/merged")
            logger.info(f"Using case directory: {case_dir}")
            
            self.tools = [
                DocumentSummarizerTool(GOOGLE_API_KEY=GOOGLE_API_KEY),
                BasicLegalQueryTool(
                    retriever=self.retriever,
                    GOOGLE_API_KEY=GOOGLE_API_KEY
                ),
                LegalEntityExtractorTool(
                    retriever=self.retriever,
                    GOOGLE_API_KEY=GOOGLE_API_KEY,
                    case_dir=case_dir
                )
            ]
            
            tools_time = time.time() - tools_start_time
            logger.info(f"Initialized {len(self.tools)} tools in {tools_time:.3f} seconds")
            
            # Create agent
            logger.info("Creating agent executor")
            agent_start_time = time.time()
            self.agent = self._create_agent()
            agent_time = time.time() - agent_start_time
            logger.info(f"Agent executor created in {agent_time:.3f} seconds")
            
            # Log initialization completion
            total_time = time.time() - start_time
            logger.info(f"LegalAssistantAgent initialization completed successfully")
            logger.info(f"Total initialization time: {total_time:.3f} seconds")
            logger.info(f"Breakdown - Retriever: {retriever_time:.3f}s, LLM: {llm_time:.3f}s, Tools: {tools_time:.3f}s, Agent: {agent_time:.3f}s")
            
        except Exception as e:
            init_time = time.time() - start_time
            logger.error(f"Failed to initialize LegalAssistantAgent after {init_time:.3f} seconds: {str(e)}")
            raise e


    def _create_agent(self):
        '''Create agent with tools and prompt'''
        logger.info("Creating agent with system prompt and tool configuration")
        
        system_prompt = """You are an expert AI legal research assistant specializing in Indian law. You help lawyers, legal professionals, and researchers with comprehensive legal analysis.

Your capabilities:
1. **legal_query_tool**: Answer legal questions using relevant cases from the database (automatically includes context from similar cases)
2. **legal_entity_extractor**: Find and analyze specific statutes, provisions, and precedents from similar cases
3. **document_summarizer**: Summarize legal documents with focus on specific areas

IMPORTANT GUIDELINES:
- The legal_query_tool now AUTOMATICALLY retrieves relevant cases and provides context - use this for most legal questions
- Use legal_entity_extractor when you need detailed analysis of specific legal elements (statutes, provisions, precedents)
- Use document_summarizer when the user provides a document to analyze
- For simple legal questions, the legal_query_tool is often sufficient as it includes relevant case context
- For complex research requiring deep analysis of legal elements, combine legal_query_tool with legal_entity_extractor
- Always prioritize accuracy and provide well-structured, professional responses

Your responses should be:
- Professional and legally accurate
- Well-structured with clear sections
- Include relevant case laws, statutes, and provisions (from retrieved context)
- Provide practical implications and recommendations
- Cite specific cases when available from the context
"""

        logger.info("Creating chat prompt template with system message and placeholders")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}") 
        ])

        logger.info("Creating tool-calling agent with LLM and tools")
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        logger.info("Creating agent executor with verbose output and error handling")
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
        
        logger.info("Agent executor created successfully")
        return agent_executor
    
    def query(self, question: str) -> str:
        '''Process a legal query through the agent'''
        logger.info("Processing legal query through agent")
        start_time = time.time()
        
        # Log input parameters
        question_length = len(question)
        logger.info(f"Input parameters - Query length: {question_length} characters")
        
        # Validate input
        if not question or not question.strip():
            logger.error("Empty or invalid question provided")
            return "Error: Please provide a valid legal question."
        
        # Log question for audit trail (truncated for security/privacy)
        question_preview = question[:100] + "..." if len(question) > 100 else question
        logger.info(f"Processing query: {question_preview}")
        
        try:
            logger.info("Invoking agent executor for query processing")
            agent_start_time = time.time()
            
            response = self.agent.invoke({"input": question})
            
            agent_time = time.time() - agent_start_time
            total_time = time.time() - start_time
            
            # Extract and validate response
            output = response.get('output', '')
            if not output:
                logger.warning("Agent returned empty response")
                return "I apologize, but I couldn't generate a response to your query. Please try rephrasing your question."
            
            # Log response metrics
            response_length = len(output)
            logger.info(f"Query processing completed successfully")
            logger.info(f"Performance metrics - Agent execution: {agent_time:.3f}s, Total time: {total_time:.3f}s")
            logger.info(f"Output metrics - Response length: {response_length} characters")
            
            # Log tools used if available in response
            if 'intermediate_steps' in response:
                tools_used = []
                for step in response['intermediate_steps']:
                    if hasattr(step, 'tool') and step.tool:
                        tools_used.append(step.tool)
                if tools_used:
                    logger.info(f"Tools utilized: {', '.join(set(tools_used))}")
            
            return output
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Query processing failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"Error details - Question length: {question_length}")
            
            # Return user-friendly error message
            return "I encountered an error while processing your query. Please try again or rephrase your question."
        