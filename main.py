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

# Fixed import - now properly imports settings
try:
    from config import settings
except ImportError:
    print("❌ Error: Could not import settings from config module")
    print("Please ensure you have:")
    print("1. config/settings.py file")
    print("2. config/__init__.py file")
    print("3. Proper directory structure")
    sys.exit(1)

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
        print("Type 'quit' to exit, 'help' for sample queries")
        
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
    
    return parser

async def main():
    """Main execution function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    logger.info(f"Starting Legal RAG System - Command: {args.command}")
    start_time = time.time()
    
    try:
        # Route to appropriate command handler
        if args.command == 'setup':
            await run_setup_command(args)
        elif args.command == 'demo':
            await run_demo_command(args)
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
    
    # Validate settings - FIXED validation
    try:
        # Test if we can access the settings attributes
        pinecone_index = getattr(settings, 'PINECONE_INDEX_NAME', None)
        if not pinecone_index:
            raise AttributeError("PINECONE_INDEX_NAME not found or empty in settings")
        
        logger.info(f"✅ Using Pinecone index: {pinecone_index}")
        logger.info(f"✅ Vector dimension: {settings.VECTOR_DIMENSION}")
        
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
