#!/usr/bin/env python3
"""
Complete Legal RAG System - Main Entry Point
Production-ready main file with proper settings integration
"""

import asyncio
import logging
import sys
import os
import time
import traceback
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Import settings class (as you mentioned you created a settings class)
from config import settings
from src.complete_rag_pipeline import CompleteLegalRAGPipeline
from src.data_models import QueryRequest, QueryResponse, HealthStatus
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('legal_rag_system.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

class LegalRAGSystemManager:
    """Main system manager for Legal RAG operations"""
    
    def __init__(self):
        self.pipeline: Optional[CompleteLegalRAGPipeline] = None
        self._initialized = False
    
    async def setup_complete_system(self, config: Optional[Dict[str, Any]] = None) -> CompleteLegalRAGPipeline:
        """Setup the complete Legal RAG system with proper configuration"""
        
        # Use provided config or create default
        config = config or {}
        
        # Set defaults based on settings
        config.setdefault('use_pinecone', True)
        config.setdefault('enable_gpu', False)
        config.setdefault('pinecone_index_name', settings.PINECONE_INDEX_NAME)
        config.setdefault('pinecone_api_key', settings.PINECONE_API_KEY)
        
        logger.info("🚀 Initializing Complete Legal RAG System")
        logger.info(f"📊 Configuration: {config}")
        
        start_time = time.time()
        
        try:
            # Initialize pipeline
            self.pipeline = CompleteLegalRAGPipeline(config)
            
            # Check if already initialized
            if self.pipeline.is_initialized and not config.get('force_rebuild', False):
                logger.info("✅ Pipeline already initialized, skipping setup")
                self._initialized = True
                return self.pipeline
            
            # Build indices if needed
            data_dir = config.get('data_dir', 'merged_output')
            force_rebuild = config.get('force_rebuild', False)
            
            logger.info(f"📁 Building index from data directory: {data_dir}")
            logger.info(f"🔄 Force rebuild: {force_rebuild}")
            
            success = await self.pipeline.build_index(data_dir, force_rebuild=force_rebuild)
            
            if not success:
                raise RuntimeError("❌ Failed to build pipeline index")
            
            setup_time = time.time() - start_time
            logger.info(f"✅ COMPLETE LEGAL RAG SYSTEM SETUP SUCCESSFUL")
            logger.info(f"⏱️ Total setup time: {setup_time:.2f}s")
            logger.info("=" * 60)
            
            self._initialized = True
            return self.pipeline
            
        except Exception as e:
            setup_time = time.time() - start_time
            logger.error(f"❌ SYSTEM SETUP FAILED")
            logger.error(f"⏱️ Error after {setup_time:.2f}s: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error("=" * 60)
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {
            'system_initialized': self._initialized,
            'pipeline_ready': self.pipeline is not None and self.pipeline.is_initialized,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'timestamp': time.time()
        }
        
        if self.pipeline:
            try:
                pipeline_stats = self.pipeline.get_pipeline_statistics()
                stats.update({'pipeline_stats': pipeline_stats})
            except Exception as e:
                logger.warning(f"Could not get pipeline stats: {e}")
        
        return stats

# Global system manager
system_manager = LegalRAGSystemManager()

async def run_setup_command(args):
    """Handle setup command"""
    logger.info("🔧 Running setup command")
    
    config = {
        'data_dir': args.data_dir,
        'force_rebuild': args.force_rebuild,
        'enable_gpu': args.enable_gpu,
        'use_pinecone': True
    }
    
    try:
        await system_manager.setup_complete_system(config)
        print("✅ System setup completed successfully!")
        print("\nNext steps:")
        print(f"  python main.py demo --data-dir {args.data_dir}   # Interactive demo")
        print(f"  python main.py api --port 8000                    # Start API server")
        
    except Exception as e:
        print(f"❌ System setup failed: {e}")
        sys.exit(1)

async def run_demo_command(args):
    """Handle demo command"""
    logger.info("🎯 Running interactive demo")
    
    config = {
        'data_dir': args.data_dir,
        'enable_gpu': args.enable_gpu,
        'use_pinecone': True
    }
    
    try:
        # Setup system if not already done
        pipeline = await system_manager.setup_complete_system(config)
        
        print("\n" + "=" * 60)
        print("🎯 LEGAL RAG INTERACTIVE DEMO")
        print("=" * 60)
        print("Ask legal questions about Indian court judgments!")
        print("Type 'quit' to exit, 'help' for sample queries, 'stats' for system info")
        
        # Sample queries for user guidance
        sample_queries = [
            "What is the doctrine of separability in arbitration agreements?",
            "Under what circumstances can a contract be void?",
            "What are the grounds for setting aside arbitration awards?",
            "Explain Section 34 of Arbitration and Conciliation Act",
            "What is the principle of res judicata in Indian law?"
        ]
        
        while True:
            try:
                query = input("\n🔍 Your Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using Legal RAG System!")
                    break
                elif query.lower() == 'help':
                    print("\n📝 Sample queries you can try:")
                    for i, sample in enumerate(sample_queries, 1):
                        print(f"{i}. {sample}")
                    continue
                elif query.lower() == 'stats':
                    stats = system_manager.get_system_stats()
                    print(f"\n📊 System Statistics:")
                    print(f"Memory Usage: {stats['memory_usage_mb']:.1f} MB")
                    print(f"CPU Usage: {stats['cpu_percent']:.1f}%")
                    print(f"Available Memory: {stats['available_memory_gb']:.1f} GB")
                    continue
                elif not query:
                    continue
                
                print(f"\n⚡ Processing: {query}")
                start_time = time.time()
                
                # Process query
                result = await pipeline.process_query(query)
                
                processing_time = time.time() - start_time
                
                # Display results
                print(f"\n📊 RESPONSE (⏱️ {processing_time:.2f}s)")
                print("=" * 50)
                print(result.get('response', 'No response generated'))
                
                if result.get('success', False):
                    print(f"\n✨ Confidence: {result.get('confidence_score', 0):.3f}")
                    print(f"📚 Contexts Used: {result.get('contexts_used', 0)}")
                    
                    # Show legal citations if available
                    citations = result.get('legal_citations', [])
                    if citations:
                        print(f"\n📖 Legal Citations:")
                        for citation in citations[:3]:
                            print(f"  • {citation}")
                
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting demo...")
                break
            except Exception as e:
                logger.error(f"Error in demo: {e}")
                print(f"❌ Error processing query: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start demo: {e}")
        print(f"❌ Failed to start demo: {e}")
        sys.exit(1)

async def run_api_command(args):
    """Handle API server command"""
    logger.info(f"🌐 Starting API server on port {args.port}")
    
    config = {
        'data_dir': args.data_dir,
        'enable_gpu': args.enable_gpu,
        'use_pinecone': True
    }
    
    try:
        # Setup system if not already done
        await system_manager.setup_complete_system(config)
        
        # Import and start API server
        from src.api_server import create_app
        import uvicorn
        
        app = create_app(system_manager.pipeline)
        
        print(f"🚀 Starting Legal RAG API Server")
        print(f"📍 Server URL: http://localhost:{args.port}")
        print(f"📖 API Documentation: http://localhost:{args.port}/docs")
        print(f"🏥 Health Check: http://localhost:{args.port}/health")
        
        # Run server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
            reload=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

async def run_evaluate_command(args):
    """Handle evaluation command"""
    logger.info("📈 Running system evaluation")
    
    config = {
        'data_dir': args.data_dir,
        'enable_gpu': args.enable_gpu,
        'use_pinecone': True
    }
    
    try:
        pipeline = await system_manager.setup_complete_system(config)
        
        # Import evaluation module
        from src.evaluation import LegalRAGEvaluator
        
        evaluator = LegalRAGEvaluator(pipeline)
        
        print("📊 Running comprehensive evaluation...")
        evaluation_results = await evaluator.run_comprehensive_evaluation()
        
        print("\n" + "=" * 60)
        print("📈 EVALUATION RESULTS")
        print("=" * 60)
        
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        
        # Save results
        import json
        with open(f"evaluation_results_{int(time.time())}.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Results saved to evaluation_results_{int(time.time())}.json")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Complete Legal RAG System for Indian Court Judgments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup --data-dir merged_output --force-rebuild
  python main.py demo --data-dir merged_output
  python main.py api --port 8000
  python main.py evaluate --data-dir merged_output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup and build indices')
    setup_parser.add_argument('--data-dir', default='merged_output', 
                             help='Directory containing merged JSON files')
    setup_parser.add_argument('--force-rebuild', action='store_true', 
                             help='Force rebuild of indices')
    setup_parser.add_argument('--enable-gpu', action='store_true', 
                             help='Enable GPU acceleration')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--data-dir', default='merged_output',
                            help='Directory containing merged JSON files')
    demo_parser.add_argument('--enable-gpu', action='store_true',
                            help='Enable GPU acceleration')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--port', type=int, default=8000,
                           help='Port to run API server on')
    api_parser.add_argument('--data-dir', default='merged_output',
                           help='Directory containing merged JSON files')
    api_parser.add_argument('--enable-gpu', action='store_true',
                           help='Enable GPU acceleration')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Run system evaluation')
    evaluate_parser.add_argument('--data-dir', default='merged_output',
                                help='Directory containing merged JSON files')
    evaluate_parser.add_argument('--enable-gpu', action='store_true',
                                help='Enable GPU acceleration')
    
    return parser

async def main():
    """Main execution function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set log level based on environment
    if os.getenv('DEBUG'):
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting Legal RAG System - Command: {args.command}")
    start_time = time.time()
    
    try:
        # Route to appropriate command handler
        if args.command == 'setup':
            await run_setup_command(args)
        elif args.command == 'demo':
            await run_demo_command(args)
        elif args.command == 'api':
            await run_api_command(args)
        elif args.command == 'evaluate':
            await run_evaluate_command(args)
        else:
            parser.print_help()
            return
        
        total_time = time.time() - start_time
        logger.info(f"Main execution completed in {total_time:.2f}s")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n👋 Goodbye!")
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Execution failed after {total_time:.2f}s")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Validate settings
    try:
        if not hasattr(settings, 'PINECONE_INDEX_NAME'):
            raise AttributeError("PINECONE_INDEX_NAME not found in settings")
        
        logger.info(f"Using Pinecone index: {settings.PINECONE_INDEX_NAME}")
        
    except Exception as e:
        logger.error(f"Settings validation failed: {e}")
        print(f"❌ Configuration error: {e}")
        print("Please ensure your config/settings.py is properly configured")
        sys.exit(1)
    
    # Run main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unhandled exception in main")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
