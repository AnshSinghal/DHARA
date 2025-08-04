#!/usr/bin/env python3
"""
Complete Legal RAG System - Main Entry Point
Production-ready main file with comprehensive logging and flexible argument parsing
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
import json

# Fixed import with better error handling
try:
    from config import settings
except ImportError as e:
    print("❌ Error: Could not import settings from config module")
    print("Please ensure you have:")
    print("1. config/settings.py file")
    print("2. config/__init__.py file")
    print("3. Proper directory structure")
    print(f"Import error: {e}")
    sys.exit(1)

try:
    from src.complete_rag_pipeline import CompleteLegalRAGPipeline
    from src.data_models import QueryRequest, QueryResponse, HealthStatus
    import psutil
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all source files are in the src/ directory")
    sys.exit(1)

# Configure comprehensive logging
def setup_logging(log_level: str = "INFO", log_file: str = "legal_rag_system.log"):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_file_path = f"logs/{log_file}"
    
    # Configure logging format
    log_format = '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
    
    # Configure handlers
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    ]
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Also log to rotating file handler for production
    from logging.handlers import RotatingFileHandler
    rotating_handler = RotatingFileHandler(
        f"logs/legal_rag_rotating.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    rotating_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(rotating_handler)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("LEGAL RAG SYSTEM STARTING")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("=" * 80)
    
    return logger

class LegalRAGSystemManager:
    """Main system manager for Legal RAG operations with comprehensive logging"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.pipeline: Optional[CompleteLegalRAGPipeline] = None
        self._initialized = False
        self.setup_start_time = None
        self.setup_metrics = {}
    
    async def setup_complete_system(self, config: Optional[Dict[str, Any]] = None) -> CompleteLegalRAGPipeline:
        """Setup the complete Legal RAG system with comprehensive logging"""
        
        self.setup_start_time = time.time()
        self.logger.info("🚀 INITIALIZING COMPLETE LEGAL RAG SYSTEM")
        self.logger.info("=" * 60)
        
        # Log system information
        self._log_system_info()
        
        # Use provided config or create default
        config = config or {}
        
        # Set defaults based on settings with logging
        config.setdefault('use_pinecone', True)
        config.setdefault('enable_gpu', False)
        config.setdefault('pinecone_index_name', settings.PINECONE_INDEX_NAME)
        config.setdefault('pinecone_api_key', settings.PINECONE_API_KEY)
        
        self.logger.info("📊 Configuration Details:")
        for key, value in config.items():
            if 'api_key' in key.lower():
                self.logger.info(f"  {key}: {'*' * 10}...{str(value)[-4:] if value else 'None'}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        try:
            # Step 1: Initialize pipeline
            self.logger.info("🔧 Step 1: Initializing pipeline components")
            component_start = time.time()
            
            self.pipeline = CompleteLegalRAGPipeline(config)
            
            component_time = time.time() - component_start
            self.setup_metrics['component_initialization'] = component_time
            self.logger.info(f"✅ Pipeline components initialized in {component_time:.2f}s")
            
            # Step 2: Check if already initialized
            if self.pipeline.is_initialized and not config.get('force_rebuild', False):
                self.logger.info("✅ Pipeline already initialized, skipping setup")
                self._log_existing_setup()
                self._initialized = True
                return self.pipeline
            
            # Step 3: Build indices if needed
            data_dir = config.get('data_dir', 'merged_output')
            force_rebuild = config.get('force_rebuild', False)
            
            self.logger.info(f"📁 Step 2: Building index from data directory")
            self.logger.info(f"  Data directory: {data_dir}")
            self.logger.info(f"  Force rebuild: {force_rebuild}")
            self.logger.info(f"  Directory exists: {os.path.exists(data_dir)}")
            
            if os.path.exists(data_dir):
                file_count = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
                self.logger.info(f"  JSON files found: {file_count}")
            else:
                self.logger.error(f"❌ Data directory does not exist: {data_dir}")
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
            # Build index with detailed logging
            index_start = time.time()
            success = await self.pipeline.build_index(data_dir, force_rebuild=force_rebuild)
            index_time = time.time() - index_start
            self.setup_metrics['index_building'] = index_time
            
            if not success:
                self.logger.error("❌ Failed to build pipeline index")
                raise RuntimeError("Failed to build pipeline index")
            
            self.logger.info(f"✅ Index built successfully in {index_time:.2f}s")
            
            # Step 4: Final setup
            setup_time = time.time() - self.setup_start_time
            self.setup_metrics['total_setup'] = setup_time
            
            self.logger.info("🎉 COMPLETE LEGAL RAG SYSTEM SETUP SUCCESSFUL")
            self.logger.info(f"⏱️ Total setup time: {setup_time:.2f}s")
            self._log_setup_metrics()
            self.logger.info("=" * 60)
            
            self._initialized = True
            return self.pipeline
            
        except Exception as e:
            setup_time = time.time() - self.setup_start_time
            self.logger.error("❌ SYSTEM SETUP FAILED")
            self.logger.error(f"⏱️ Error after {setup_time:.2f}s: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.logger.error("=" * 60)
            raise
    
    def _log_system_info(self):
        """Log comprehensive system information"""
        self.logger.info("💻 System Information:")
        self.logger.info(f"  OS: {os.name}")
        self.logger.info(f"  Python: {sys.version}")
        self.logger.info(f"  CPU cores: {psutil.cpu_count()}")
        self.logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        self.logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        self.logger.info(f"  Disk space: {psutil.disk_usage('/').free / (1024**3):.1f} GB free")
    
    def _log_existing_setup(self):
        """Log information about existing setup"""
        try:
            stats = self.pipeline.get_pipeline_statistics()
            self.logger.info("📊 Existing Pipeline Status:")
            self.logger.info(f"  Initialized: {stats.get('pipeline_status', {}).get('initialized', 'Unknown')}")
            self.logger.info(f"  Index built: {stats.get('pipeline_status', {}).get('index_built', 'Unknown')}")
            
            vector_stats = stats.get('vector_store', {})
            if vector_stats:
                self.logger.info(f"  Vector count: {vector_stats.get('total_vector_count', 'Unknown')}")
                self.logger.info(f"  Vector dimension: {vector_stats.get('dimension', 'Unknown')}")
        except Exception as e:
            self.logger.warning(f"Could not get existing pipeline stats: {e}")
    
    def _log_setup_metrics(self):
        """Log detailed setup metrics"""
        self.logger.info("📈 Setup Performance Metrics:")
        for metric, value in self.setup_metrics.items():
            self.logger.info(f"  {metric}: {value:.2f}s")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics with logging"""
        self.logger.debug("Getting system statistics...")
        
        try:
            stats = {
                'system_initialized': self._initialized,
                'pipeline_ready': self.pipeline is not None and self.pipeline.is_initialized,
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'timestamp': time.time(),
                'setup_metrics': self.setup_metrics
            }
            
            if self.pipeline:
                try:
                    pipeline_stats = self.pipeline.get_pipeline_statistics()
                    stats.update({'pipeline_stats': pipeline_stats})
                except Exception as e:
                    self.logger.warning(f"Could not get pipeline stats: {e}")
            
            self.logger.debug(f"System stats retrieved: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}

# Global system manager
system_manager = None
logger = None

async def run_setup_command(args):
    """Handle setup command with comprehensive logging"""
    logger.info("🔧 RUNNING SETUP COMMAND")
    logger.info(f"Arguments: {vars(args)}")
    
    config = {
        'data_dir': args.data_dir,
        'force_rebuild': args.force_rebuild,
        'enable_gpu': args.enable_gpu,
        'use_pinecone': True
    }
    
    try:
        await system_manager.setup_complete_system(config)
        
        print("\n" + "=" * 60)
        print("✅ SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Next steps:")
        print(f"  python main.py --demo --data-dir {args.data_dir}")
        print(f"  python main.py demo --data-dir {args.data_dir}")
        print("=" * 60)
        
        logger.info("Setup command completed successfully")
        
    except Exception as e:
        logger.error(f"Setup command failed: {e}")
        print(f"\n❌ SYSTEM SETUP FAILED: {e}")
        print("Check the logs for detailed error information")
        sys.exit(1)

async def run_demo_command(args):
    """Handle demo command with comprehensive logging"""
    logger.info("🎯 RUNNING INTERACTIVE DEMO COMMAND")
    logger.info(f"Arguments: {vars(args)}")
    
    config = {
        'data_dir': args.data_dir,
        'enable_gpu': args.enable_gpu,
        'use_pinecone': True
    }
    
    try:
        # Setup system if not already done
        logger.info("Setting up system for demo...")
        pipeline = await system_manager.setup_complete_system(config)
        
        print("\n" + "=" * 60)
        print("🎯 LEGAL RAG INTERACTIVE DEMO")
        print("=" * 60)
        print("Ask legal questions about Indian court judgments!")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the demo")
        print("  'help' - Show sample queries")
        print("  'stats' - Show system statistics")
        print("  'clear' - Clear screen")
        print("=" * 60)
        
        # Sample queries
        sample_queries = [
            "What is the doctrine of separability in arbitration agreements?",
            "Under what circumstances can a contract be void?",
            "What are the grounds for setting aside arbitration awards?",
            "Explain Section 34 of Arbitration and Conciliation Act",
            "What is the principle of res judicata in Indian law?"
        ]
        
        query_count = 0
        
        while True:
            try:
                query = input("\n🔍 Your Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info(f"Demo ended by user. Total queries processed: {query_count}")
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
                    print(f"Memory Usage: {stats.get('memory_usage_mb', 0):.1f} MB")
                    print(f"CPU Usage: {stats.get('cpu_percent', 0):.1f}%")
                    print(f"Available Memory: {stats.get('available_memory_gb', 0):.1f} GB")
                    print(f"Queries Processed: {query_count}")
                    continue
                    
                elif query.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                    
                elif not query:
                    continue
                
                query_count += 1
                logger.info(f"Processing query #{query_count}: {query[:50]}...")
                
                print(f"\n⚡ Processing query #{query_count}: {query}")
                start_time = time.time()
                
                # Process query
                result = await pipeline.process_query(query)
                
                processing_time = time.time() - start_time
                logger.info(f"Query #{query_count} processed in {processing_time:.2f}s")
                
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
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Query #{query_count} failed: {error_msg}")
                    print(f"❌ Error: {error_msg}")
                
            except KeyboardInterrupt:
                logger.info("Demo interrupted by user")
                print("\n👋 Exiting demo...")
                break
                
            except Exception as e:
                logger.error(f"Error in demo query processing: {e}")
                logger.error(traceback.format_exc())
                print(f"❌ Error processing query: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start demo: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Failed to start demo: {e}")
        sys.exit(1)

def create_parser():
    """Create flexible command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Complete Legal RAG System for Indian Court Judgments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct execution (recommended):
  python main.py --data-dir merged_output --force-rebuild  # Setup with rebuild
  python main.py --demo --data-dir merged_output           # Run demo
  python main.py --setup --data-dir merged_output         # Setup only
  
  # Subcommand style (also supported):
  python main.py setup --data-dir merged_output --force-rebuild
  python main.py demo --data-dir merged_output
  
  # Other options:
  python main.py --help                                    # Show help
  python main.py --log-level DEBUG --demo                 # Debug logging
        """
    )
    
    # Global arguments
    parser.add_argument('--data-dir', default='merged_output',
                       help='Directory containing merged JSON files (default: merged_output)')
    parser.add_argument('--enable-gpu', action='store_true',
                       help='Enable GPU acceleration if available')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level (default: INFO)')
    parser.add_argument('--log-file', default='legal_rag_system.log',
                       help='Log file name (default: legal_rag_system.log)')
    
    # Operation modes (mutually exclusive)
    operation_group = parser.add_mutually_exclusive_group()
    operation_group.add_argument('--setup', action='store_true',
                                help='Run setup (build indices)')
    operation_group.add_argument('--demo', action='store_true',
                                help='Run interactive demo')
    operation_group.add_argument('--force-rebuild', action='store_true',
                                help='Force rebuild indices (implies --setup)')
    
    # Subcommands (for backward compatibility)
    subparsers = parser.add_subparsers(dest='command', help='Available commands (alternative syntax)')
    
    # Setup subcommand
    setup_parser = subparsers.add_parser('setup', help='Setup and build indices')
    setup_parser.add_argument('--data-dir', default='merged_output',
                             help='Directory containing merged JSON files')
    setup_parser.add_argument('--force-rebuild', action='store_true',
                             help='Force rebuild of indices')
    setup_parser.add_argument('--enable-gpu', action='store_true',
                             help='Enable GPU acceleration')
    
    # Demo subcommand  
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--data-dir', default='merged_output',
                            help='Directory containing merged JSON files')
    demo_parser.add_argument('--enable-gpu', action='store_true',
                            help='Enable GPU acceleration')
    
    return parser

async def main():
    """Main execution function with comprehensive error handling"""
    global system_manager, logger
    
    # Parse arguments first to get log level
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    system_manager = LegalRAGSystemManager(logger)
    
    logger.info(f"Starting Legal RAG System with arguments: {vars(args)}")
    start_time = time.time()
    
    try:
        # Determine operation mode
        operation = None
        
        # Handle direct operation flags
        if args.force_rebuild:
            operation = 'setup'
            args.force_rebuild = True
        elif args.setup:
            operation = 'setup'
            args.force_rebuild = False
        elif args.demo:
            operation = 'demo'
        elif args.command:
            # Handle subcommand syntax
            operation = args.command
        else:
            # Default behavior - show help
            parser.print_help()
            logger.info("No operation specified, showing help")
            return
        
        logger.info(f"Operation mode: {operation}")
        
        # Route to appropriate command handler
        if operation == 'setup':
            await run_setup_command(args)
        elif operation == 'demo':
            await run_demo_command(args)
        else:
            logger.error(f"Unknown operation: {operation}")
            parser.print_help()
            return
        
        total_time = time.time() - start_time
        logger.info(f"✅ Main execution completed successfully in {total_time:.2f}s")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n👋 Goodbye!")
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Execution failed after {total_time:.2f}s")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"❌ System error: {e}")
        print("Check the logs for detailed error information")
        sys.exit(1)

if __name__ == '__main__':
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Validate settings before starting
    try:
        # Test settings access
        pinecone_index = getattr(settings, 'PINECONE_INDEX_NAME', None)
        if not pinecone_index:
            raise AttributeError("PINECONE_INDEX_NAME not found or empty in settings")
        
        print(f"✅ Configuration validated")
        print(f"✅ Using Pinecone index: {pinecone_index}")
        print(f"✅ Vector dimension: {settings.VECTOR_DIMENSION}")
        
    except Exception as e:
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
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
