#!/usr/bin/env python3
"""
Complete Legal RAG System - Main execution script with full orchestration
"""

import argparse
import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.complete_rag_pipeline import CompleteLegalRAGPipeline
from src.api_server import app
from config.settings import settings

# Import DharaLogger
from logger import DharaLogger

# Initialize dharalogger for main script
dharalogger = DharaLogger("main", log_file="main_system.log")
logger = dharalogger.get_logger()

async def setup_complete_system(data_directory: str, force_rebuild: bool = False):
    """Setup the complete RAG system with all components"""
    
    logger.info("=" * 80)
    logger.info("LEGAL RAG SYSTEM - COMPLETE SETUP")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        logger.info(f"Setup parameters:")
        logger.info(f"  - Data directory: {data_directory}")
        logger.info(f"  - Force rebuild: {force_rebuild}")
        logger.info(f"  - Environment variables:")
        logger.info(f"    - USE_PINECONE: {os.getenv('USE_PINECONE', 'true')}")
        logger.info(f"    - ENABLE_GPU: {os.getenv('ENABLE_GPU', 'true')}")
        
        # Initialize pipeline with configuration
        config = {
            'use_pinecone': os.getenv('USE_PINECONE', 'true').lower() == 'true',
            'enable_gpu': os.getenv('ENABLE_GPU', 'true').lower() == 'true'
        }
        
        logger.info("Initializing Complete Legal RAG Pipeline...")
        logger.info(f"Configuration: {config}")
        
        init_start = time.time()
        pipeline = CompleteLegalRAGPipeline(config)
        init_time = time.time() - init_start
        logger.info(f"Pipeline initialized in {init_time:.2f}s")
        
        # Build comprehensive index
        logger.info("Building comprehensive index with all components...")
        build_start = time.time()
        success = await pipeline.build_index(data_directory, force_rebuild)
        build_time = time.time() - build_start
        
        if not success:
            logger.error("Failed to build complete system")
            logger.error(f"Index building failed after {build_time:.2f}s")
            return None
        
        logger.info(f"Index built successfully in {build_time:.2f}s")
        
        # Perform health check
        logger.info("Performing system health check...")
        health_start = time.time()
        health = pipeline.health_check()
        health_time = time.time() - health_start
        
        if health['status'] != 'healthy':
            logger.warning(f"System health check shows: {health['status']}")
            logger.warning(f"Health details: {health}")
        else:
            logger.info("✅ System health check passed")
        
        logger.info(f"Health check completed in {health_time:.2f}s")
        
        if health['status'] != 'healthy':
            logger.warning(f"System health check shows: {health['status']}")
            logger.warning(f"Health details: {health}")
        else:
            logger.info("✅ System health check passed")
        
        # Get and display statistics
        logger.info("Retrieving system statistics...")
        stats_start = time.time()
        stats = pipeline.get_pipeline_statistics()
        stats_time = time.time() - stats_start
        logger.info(f"Statistics retrieved in {stats_time:.2f}s")
        
        logger.info("=" * 60)
        logger.info("SYSTEM STATISTICS")
        logger.info("=" * 60)
        
        # Pipeline status
        status_info = stats.get('pipeline_status', {})
        logger.info(f"Pipeline Status:")
        logger.info(f"  Pipeline Initialized: {status_info.get('initialized', False)}")
        logger.info(f"  Index Built: {status_info.get('index_built', False)}")
        
        # Configuration
        config_info = stats.get('configuration', {})
        logger.info(f"Configuration:")
        logger.info(f"  Vector Store: {'Pinecone' if config_info.get('use_pinecone') else 'ChromaDB'}")
        logger.info(f"  GPU Enabled: {config_info.get('enable_gpu', False)}")
        logger.info(f"  Embedding Model: {config_info.get('embedding_model', 'N/A')}")
        logger.info(f"  Generation Model: {config_info.get('generation_model', 'N/A')}")
        
        # Vector store stats
        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            logger.info(f"Vector Store Statistics:")
            logger.info(f"  Indexed Vectors: {vs_stats.get('total_vector_count', 0)}")
        
        # BM25 stats
        if 'bm25' in stats:
            bm25_stats = stats['bm25']
            logger.info(f"BM25 Statistics:")
            logger.info(f"  BM25 Documents: {bm25_stats.get('total_chunks', 0)}")
            logger.info(f"  Vocabulary Size: {bm25_stats.get('vocabulary_size', 0)}")
        
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ COMPLETE LEGAL RAG SYSTEM SETUP SUCCESSFUL")
        logger.info(f"Total setup time: {total_time:.2f}s")
        logger.info("=" * 60)
        
        return pipeline
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error("SYSTEM SETUP FAILED")
        logger.error(f"Error after {error_time:.2f}s: {e}")
        logger.error("=" * 60)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

async def run_interactive_demo(pipeline: CompleteLegalRAGPipeline):
    """Run comprehensive interactive demo"""
    
    logger.info("Starting Interactive Legal RAG Demo")
    logger.info("Demo session initiated with comprehensive pipeline")
    
    print("\n" + "=" * 80)
    print("LEGAL RAG SYSTEM - INTERACTIVE DEMO")
    print("=" * 80)
    print("Ask legal questions and get AI-powered answers with:")
    print("• Rhetorical role-aware retrieval")  
    print("• Multi-faceted search with metadata")
    print("• Hybrid dense + sparse retrieval")
    print("• Cross-encoder reranking")
    print("• Context-aware legal text generation")
    print("• Comprehensive quality evaluation")
    print("\nType 'help' for advanced options, 'quit' to exit.")
    print("=" * 80)
    
    # Sample queries for demonstration
    sample_queries = [
        "What is the doctrine of separability in arbitration agreements?",
        "Under what circumstances can a contract be considered void?", 
        "What are the grounds for appeal in criminal cases?",
        "How is the burden of proof determined in civil cases?",
        "What constitutes breach of fiduciary duty?"
    ]
    
    logger.info(f"Demo initialized with {len(sample_queries)} sample queries")
    
    print(f"\n📝 Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    query_count = 0
    successful_queries = 0
    total_processing_time = 0
    
    while True:
        try:
            print("\n" + "-" * 60)
            query = input("🔍 Legal Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("User requested demo exit")
                break
            elif query.lower() == 'help':
                logger.info("User requested help information")
                print_help()
                continue
            elif query.lower().startswith('sample '):
                try:
                    sample_num = int(query.split()[1]) - 1
                    if 0 <= sample_num < len(sample_queries):
                        query = sample_queries[sample_num]
                        logger.info(f"User selected sample query {sample_num + 1}: {query[:50]}...")
                        print(f"Using sample query: {query}")
                    else:
                        print("Invalid sample number")
                        logger.warning(f"User entered invalid sample number: {sample_num + 1}")
                        continue
                except:
                    print("Usage: sample <number>")
                    logger.warning("User entered invalid sample command format")
                    continue
            elif not query:
                continue
            
            query_count += 1
            logger.info(f"Processing query #{query_count}: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            print(f"\n⚡ Processing query: {query}")
            print("🔄 Executing complete RAG pipeline...")
            
            # Process with advanced options
            options = {
                'retrieval_top_k': 15,
                'generation_top_k': 5,
                'fusion_method': 'weighted_sum'
            }
            
            logger.info(f"Query processing options: {options}")
            
            query_start_time = time.time()
            result = await pipeline.process_query(query, options)
            query_time = time.time() - query_start_time
            
            total_processing_time += query_time
            
            if result.get('success', True):
                successful_queries += 1
                logger.info(f"Query #{query_count} processed successfully in {query_time:.2f}s")
                logger.info(f"Confidence: {result.get('confidence_score', 0):.3f}")
            else:
                logger.error(f"Query #{query_count} failed: {result.get('error', 'Unknown error')}")
            
            # Display comprehensive results
            display_comprehensive_results(result)
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user (Ctrl+C)")
            break
        except Exception as e:
            logger.error(f"Demo error on query #{query_count}: {e}")
            try:
                logger.error(f"Query: {query}")
            except NameError:
                logger.error("Query variable not defined at error time")
            print(f"❌ Error: {e}")
    
    # Demo session summary
    logger.info("=" * 50)
    logger.info("DEMO SESSION SUMMARY")
    logger.info(f"Total queries processed: {query_count}")
    logger.info(f"Successful queries: {successful_queries}")
    logger.info(f"Failed queries: {query_count - successful_queries}")
    if query_count > 0:
        logger.info(f"Success rate: {successful_queries/query_count*100:.1f}%")
        logger.info(f"Average processing time: {total_processing_time/query_count:.2f}s")
    logger.info("=" * 50)
    
    print("\n👋 Thank you for using the Legal RAG System!")

def print_help():
    """Print help information"""
    
    print("\n" + "=" * 60)
    print("ADVANCED USAGE OPTIONS")
    print("=" * 60)
    print("Commands:")
    print("  help                    - Show this help")
    print("  quit/exit/q             - Exit the demo")
    print("  sample <1-5>            - Use sample query")
    print("\nQuery Examples:")
    print("  Basic: What is force majeure?")
    print("  Complex: Precedents on breach of contract in commercial disputes")
    print("\nThe system will automatically:")
    print("  • Use rhetorical role-based chunking")
    print("  • Apply hybrid search (dense + sparse)")
    print("  • Rerank results with cross-encoder")
    print("  • Generate context-aware responses")
    print("  • Provide quality metrics and explanations")
    print("=" * 60)

def display_comprehensive_results(result: dict):
    """Display comprehensive results with all pipeline insights"""
    
    print(f"\n📊 COMPREHENSIVE RESULTS")
    print("=" * 60)
    
    # Basic response
    print(f"💬 Response (Confidence: {result['confidence_score']:.2f}):")
    print("-" * 40)
    print(result['response'])
    
    # Quality metrics
    quality = result.get('quality_metrics', {})
    print(f"\n📈 Quality Metrics:")
    print(f"   Overall Score: {quality.get('overall_score', 0):.3f}")
    if 'response_dimensions' in quality:
        dims = quality['response_dimensions']
        print(f"   Legal Terminology: {dims.get('legal_terminology', 0):.3f}")
        print(f"   Context Relevance: {dims.get('context_relevance', 0):.3f}")
        print(f"   Structural Coherence: {dims.get('structural_coherence', 0):.3f}")
    
    # Processing details
    print(f"\n⚡ Processing Details:")
    print(f"   Total Time: {result['processing_time']:.2f}s")
    print(f"   Contexts Used: {result['contexts_used']}")
    
    # Performance breakdown
    perf = result.get('performance_metrics', {})
    if 'steps' in perf:
        steps = perf['steps']
        print(f"   Retrieval: {steps.get('retrieval', {}).get('time', 0):.2f}s")
        print(f"   Reranking: {steps.get('reranking', {}).get('time', 0):.2f}s") 
        print(f"   Generation: {steps.get('generation', {}).get('time', 0):.2f}s")
    
    # Supporting documents
    docs = result.get('supporting_documents', [])
    if docs:
        print(f"\n📄 Supporting Documents ({len(docs)}):")
        for i, doc in enumerate(docs[:3], 1):  # Show top 3
            print(f"   {i}. {doc.get('id', 'Unknown')} ({doc.get('rhetorical_role', 'N/A')})")
            print(f"      Score: {doc.get('relevance_score', 0):.3f}")
            print(f"      Preview: {doc.get('text_preview', '')[:100]}...")
    
    # Legal citations
    citations = result.get('legal_citations', [])
    if citations:
        print(f"\n⚖️  Legal Citations ({len(citations)}):")
        for citation in citations[:5]:  # Show top 5
            print(f"   • {citation}")
    
    # Pipeline insights
    insights = result.get('pipeline_insights', {})
    if 'rhetorical_structure' in insights:
        struct = insights['rhetorical_structure']
        print(f"\n🧠 Rhetorical Analysis:")
        print(f"   Template Used: {struct.get('template_used', 'N/A')}")
        print(f"   Dominant Roles: {', '.join(struct.get('dominant_roles', [])[:3])}")
        print(f"   Context Coherence: {struct.get('context_coherence', 0):.3f}")
    
    # Related queries
    related = result.get('related_queries', [])
    if related:
        print(f"\n🔗 Related Queries:")
        for query in related:
            print(f"   • {query}")

async def run_evaluation_suite(pipeline: CompleteLegalRAGPipeline):
    """Run comprehensive evaluation suite"""
    
    logger.info("Starting comprehensive evaluation suite")
    eval_start_time = time.time()
    
    # Test queries covering different aspects
    evaluation_queries = [
        # Factual queries
        ("What are the essential elements of a valid contract?", "factual"),
        ("Define the doctrine of res judicata", "definitional"),
        
        # Precedent-based queries  
        ("What precedents exist for breach of employment contracts?", "precedent"),
        ("Cases on fundamental rights violations", "precedent"),
        
        # Statutory queries
        ("Section 420 of Indian Penal Code provisions", "statutory"),
        ("Article 21 of Indian Constitution interpretation", "statutory"),
        
        # Procedural queries
        ("How to file an appeal in criminal cases?", "procedural"),
        ("Procedure for divorce under Hindu Marriage Act", "procedural"),
        
        # Complex analytical queries
        ("Comparative analysis of tort and contract law", "analytical"),
        ("Evolution of privacy rights in Indian jurisprudence", "analytical")
    ]
    
    logger.info(f"Evaluation suite configured with {len(evaluation_queries)} test queries")
    logger.info("Query distribution:")
    
    query_types = {}
    for _, qtype in evaluation_queries:
        query_types[qtype] = query_types.get(qtype, 0) + 1
    
    for qtype, count in query_types.items():
        logger.info(f"  {qtype}: {count} queries")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION SUITE")
    print("=" * 80)
    
    results = []
    total_start_time = time.time()
    
    for i, (query, query_type) in enumerate(evaluation_queries, 1):
        logger.info(f"Evaluating query {i}/{len(evaluation_queries)} [{query_type.upper()}]: {query}")
        print(f"\n[{i}/{len(evaluation_queries)}] Processing: {query_type.upper()} query")
        print(f"Query: {query}")
        
        try:
            start_time = time.time()
            result = await pipeline.process_query(query)
            processing_time = time.time() - start_time
            
            # Extract key metrics
            evaluation_result = {
                'query': query,
                'query_type': query_type,
                'processing_time': processing_time,
                'confidence_score': result['confidence_score'],
                'overall_quality': result.get('quality_metrics', {}).get('overall_score', 0),
                'contexts_used': result['contexts_used'],
                'success': result.get('success', True)
            }
            
            results.append(evaluation_result)
            
            logger.info(f"Query {i} completed successfully:")
            logger.info(f"  Processing time: {processing_time:.2f}s")
            logger.info(f"  Confidence: {result['confidence_score']:.3f}")
            logger.info(f"  Quality: {evaluation_result['overall_quality']:.3f}")
            logger.info(f"  Contexts used: {result['contexts_used']}")
            
            print(f"✅ Completed in {processing_time:.2f}s")
            print(f"   Confidence: {result['confidence_score']:.3f}")
            print(f"   Quality: {evaluation_result['overall_quality']:.3f}")
            
        except Exception as e:
            logger.error(f"Evaluation error for query {i} [{query_type}]: {e}")
            logger.error(f"Query: {query}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            results.append({
                'query': query,
                'query_type': query_type,
                'error': str(e),
                'success': False
            })
            print(f"❌ Failed: {str(e)}")
    
    total_time = time.time() - total_start_time
    
    logger.info("=" * 60)
    logger.info("EVALUATION SUITE COMPLETED")
    logger.info(f"Total evaluation time: {total_time:.2f}s")
    logger.info("=" * 60)
    
    # Generate evaluation report
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['confidence_score'] for r in successful_results) / len(successful_results)
        avg_quality = sum(r['overall_quality'] for r in successful_results) / len(successful_results)
        avg_contexts = sum(r['contexts_used'] for r in successful_results) / len(successful_results)
        
        print(f"📊 Overall Performance:")
        print(f"   Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"   Average Processing Time: {avg_processing_time:.2f}s")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Quality Score: {avg_quality:.3f}")
        print(f"   Average Contexts Used: {avg_contexts:.1f}")
        print(f"   Total Evaluation Time: {total_time:.2f}s")
        
        # Query type breakdown
        print(f"\n📈 Performance by Query Type:")
        query_types = {}
        for result in successful_results:
            qtype = result['query_type']
            if qtype not in query_types:
                query_types[qtype] = []
            query_types[qtype].append(result)
        
        for qtype, type_results in query_types.items():
            avg_qual = sum(r['overall_quality'] for r in type_results) / len(type_results)
            avg_conf = sum(r['confidence_score'] for r in type_results) / len(type_results) 
            print(f"   {qtype.capitalize()}: Quality={avg_qual:.3f}, Confidence={avg_conf:.3f}")
    
    print("=" * 80)
    
    return results

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("STARTING LEGAL RAG API SERVER")
    logger.info("=" * 60)
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Data directory: {settings.MERGED_DATA_DIR}")
    logger.info(f"  Workers: 1 (single worker for GPU/model sharing)")
    
    # Set environment variables for the server
    os.environ['DATA_DIRECTORY'] = settings.MERGED_DATA_DIR
    logger.info(f"Environment variable DATA_DIRECTORY set to: {settings.MERGED_DATA_DIR}")
    
    try:
        logger.info("Launching uvicorn server...")
        uvicorn.run(
            "src.api_server:app",
            host=host,
            port=port,
            workers=1,  # Single worker for now due to GPU/model sharing
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def main():
    """Main execution function"""
    
    logger.info("=" * 80)
    logger.info("LEGAL RAG SYSTEM - MAIN EXECUTION")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser(
        description="Complete Legal RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup --data-dir data/merged                    # Setup system
  python main.py demo --data-dir data/merged                     # Interactive demo
  python main.py evaluate --data-dir data/merged                 # Run evaluation
  python main.py api --host 0.0.0.0 --port 8000                 # Start API server
        """
    )
    
    parser.add_argument('command', 
                       choices=['setup', 'demo', 'evaluate', 'api'], 
                       help='Command to execute')
    
    parser.add_argument('--data-dir', 
                       default=settings.MERGED_DATA_DIR,
                       help='Directory containing merged JSON files')
    
    parser.add_argument('--force-rebuild', 
                       action='store_true',
                       help='Force rebuild of existing index')
    
    parser.add_argument('--host', 
                       default='0.0.0.0',
                       help='API server host')
    
    parser.add_argument('--port', 
                       type=int, 
                       default=8000,
                       help='API server port')
    
    args = parser.parse_args()
    
    logger.info(f"Command: {args.command}")
    logger.info(f"Arguments:")
    logger.info(f"  data-dir: {args.data_dir}")
    logger.info(f"  force-rebuild: {args.force_rebuild}")
    logger.info(f"  host: {args.host}")
    logger.info(f"  port: {args.port}")
    
    main_start_time = time.time()
    
    if args.command == 'setup':
        logger.info("Executing SETUP command")
        # Setup system
        pipeline = asyncio.run(setup_complete_system(args.data_dir, args.force_rebuild))
        if pipeline:
            logger.info("System setup completed successfully!")
            print("\n✅ System setup completed successfully!")
            print("\nNext steps:")
            print("  python main.py demo --data-dir", args.data_dir, "  # Interactive demo")
            print("  python main.py api --port 8000                    # Start API server")
        else:
            logger.error("System setup failed!")
            print("\n❌ System setup failed!")
            sys.exit(1)
    
    elif args.command == 'demo':
        logger.info("Executing DEMO command")
        # Setup and run demo
        pipeline = asyncio.run(setup_complete_system(args.data_dir))
        if pipeline:
            logger.info("System setup successful, starting demo")
            asyncio.run(run_interactive_demo(pipeline))
        else:
            logger.error("Failed to setup system for demo")
            print("❌ Failed to setup system for demo")
            sys.exit(1)
    
    elif args.command == 'evaluate':
        logger.info("Executing EVALUATE command")
        # Setup and run evaluation
        pipeline = asyncio.run(setup_complete_system(args.data_dir))
        if pipeline:
            logger.info("System setup successful, starting evaluation")
            results = asyncio.run(run_evaluation_suite(pipeline))
            
            # Save evaluation results
            import json
            eval_file = f"evaluation_results_{int(time.time())}.json"
            logger.info(f"Saving evaluation results to: {eval_file}")
            
            try:
                with open(eval_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Evaluation results saved successfully")
                print(f"\n💾 Evaluation results saved to: {eval_file}")
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")
        else:
            logger.error("Failed to setup system for evaluation")
            print("❌ Failed to setup system for evaluation")
            sys.exit(1)
    
    elif args.command == 'api':
        logger.info("Executing API command")
        # Run API server
        run_api_server(args.host, args.port)
    
    total_time = time.time() - main_start_time
    logger.info(f"Main execution completed in {total_time:.2f}s")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
