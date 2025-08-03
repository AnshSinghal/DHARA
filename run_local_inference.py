#!/usr/bin/env python3
"""
Local CPU-only inference script using pre-built indices
"""

import os
import asyncio
import time
import json
from pathlib import Path

# Force CPU usage before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.complete_rag_pipeline import CompleteLegalRAGPipeline

async def setup_local_inference(bm25_index_path: str, chunks_metadata_path: str):
    """Setup pipeline for local inference using existing indices"""
    
    print("🚀 Setting up Local Legal RAG System (CPU-Only)")
    print("=" * 60)
    
    # Initialize pipeline with CPU-only config
    config = {
        'use_pinecone': True,  # You can still use Pinecone from local
        'enable_gpu': False    # Force CPU usage
    }
    
    pipeline = CompleteLegalRAGPipeline(config)
    
    # Load pre-built BM25 index (skips heavy indexing step)
    if os.path.exists(bm25_index_path):
        print(f"📂 Loading pre-built BM25 index from {bm25_index_path}")
        pipeline.bm25_retriever.load_index(bm25_index_path)
        print("✅ BM25 index loaded successfully")
    else:
        print(f"❌ BM25 index not found at {bm25_index_path}")
        return None
    
    # Load chunks metadata
    if os.path.exists(chunks_metadata_path):
        print(f"📄 Loading chunks metadata from {chunks_metadata_path}")
        with open(chunks_metadata_path, 'r') as f:
            chunks_metadata = json.load(f)
        print(f"✅ Loaded metadata for {len(chunks_metadata)} chunks")
    else:
        print(f"❌ Chunks metadata not found at {chunks_metadata_path}")
    
    # Mark as initialized (skip heavy setup)
    pipeline.is_initialized = True
    pipeline.index_built = True
    
    print("🎉 Local inference setup complete!")
    print("💡 Ready for fast CPU-only queries")
    
    return pipeline

async def run_local_demo(pipeline):
    """Run interactive demo with local pipeline"""
    
    print("\n" + "=" * 60)
    print("LOCAL LEGAL RAG DEMO (CPU-Only)")
    print("=" * 60)
    print("Ask legal questions - inference will be much faster now!")
    print("Type 'quit' to exit, 'help' for options")
    
    sample_queries = [
        "What is the doctrine of separability in arbitration agreements?",
        "Under what circumstances can a contract be void?",
        "What are the grounds for setting aside arbitration awards?",
        "Explain Section 34 of Arbitration and Conciliation Act",
        "What is the principle of res judicata in Indian law?"
    ]
    
    print("\n📝 Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    while True:
        try:
            query = input("\n🔍 Your Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'help':
                print("Available commands:")
                print("  quit/exit - Exit the demo")
                print("  help - Show this help")
                print("  sample <num> - Use sample query")
                continue
            elif not query:
                continue
            
            print(f"\n⚡ Processing: {query}")
            start_time = time.time()
            
            # Run inference (should be fast now)
            result = await pipeline.process_query(query)
            
            processing_time = time.time() - start_time
            
            # Display results
            print(f"\n📊 RESPONSE (⏱️ {processing_time:.2f}s)")
            print("=" * 50)
            print(result['response'])
            print(f"\n✨ Confidence: {result['confidence_score']:.3f}")
            print(f"📚 Contexts Used: {result['contexts_used']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Demo complete!")

async def main():
    """Main execution function"""
    
    # Paths to your downloaded files
    bm25_index_path = "legal_bm25_index.pkl"  # Update path as needed
    chunks_metadata_path = "chunks_metadata.json"  # Update path as needed

    # Setup local inference
    pipeline = await setup_local_inference(bm25_index_path, chunks_metadata_path)
    
    if pipeline:
        await run_local_demo(pipeline)
    else:
        print("❌ Failed to setup local inference")

if __name__ == "__main__":
    asyncio.run(main())
