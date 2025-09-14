from pinecone import Pinecone
from google import genai
from google.genai import types
import os
import logging
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
import torch
from typing import List, Dict, Any, Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
client = genai.Client(api_key=GOOGLE_API_KEY)

@dataclass
class RetrieverConfig:
    PINECONE_API_KEY: str
    GOOGLE_API_KEY: str
    sparse_index_name: str = "legal-cases-pincone-sparse-2048"
    dense_index_name: str = "legal-cases-gemini-3072"
    sparse_top_k: int = 20
    dense_top_k: int = 10
    final_top_k: int = 5
    bert_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    pinecone_rerank_model: str = "bge-reranker-v2-m3"
    pinecone_rerank_top_k: int = 10
    sparse_top_k: int = 20
    sparse_namespace: str = "namespace"
    dense_namespace: str = "__default__"

class HybridRetriever:
    '''
    Custom retriever that queries separate sparse and dense Pinecone indexes,
    applies Pinecone reranking to each, then combines and applies BERT reranking.
    '''
    def __init__(self, config: RetrieverConfig):
        self.config = config
        
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)

        
        self.sparse_index = self.pc.Index(config.sparse_index_name)
        self.dense_index = self.pc.Index(config.dense_index_name)
        logger.info("Pinecone client initialized.")

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Gemini client initialized.")

        self.bert_reranker = CrossEncoder(config.bert_rerank_model)
        logger.info("BERT reranker model loaded.")

    def get_dense_embedding(self, text) -> List[float]:
        '''Generate dense embedding using Gemini'''
        try:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=3072)
            )

            embedding = result.embeddings[0].values
            logger.info(f"Dense embedding generated with length {len(embedding)}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating dense embedding: {e}")
            return []

    def _query_sparse_index_with_reranking(self, query_text: str) -> List[Dict]:
        '''Query sparse index with Pinecone reranking using bge-reranker-v2-m3'''
        try:
            # Use the correct Pinecone search API with reranking
            response = self.sparse_index.search(
                namespace=self.config.sparse_namespace,
                query={
                    "inputs": {"text": query_text},
                    "top_k": self.config.sparse_top_k
                },
                rerank={
                    "model": self.config.pinecone_rerank_model,
                    "top_n": self.config.pinecone_rerank_top_k,
                    "rank_fields": ["text"],
                    "parameters": {
                        "truncate": "END"
                    }
                }
            )
            logger.info(f"Sparse index queried with reranking for top {self.config.sparse_top_k} results.")
            
            # Extract results from the correct response format
            if 'result' in response and 'hits' in response['result']:
                results = []
                for hit in response['result']['hits']:
                    result = {
                        'id': hit['_id'],
                        'score': hit['_score'],
                        'metadata': hit['fields'],
                        'source': 'sparse',
                        'pinecone_rerank_score': hit['_score']
                    }
                    results.append(result)
                logger.info(f"Retrieved {len(results)} results from sparse index with reranking.")
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error querying sparse index with reranking: {e}")
            return []

    def _query_dense_index(self, query: str) -> List[Dict]:
        '''Query dense index'''
        try:
            query_embedding = self.get_dense_embedding(query)
            if not query_embedding:
                logger.warning("Empty dense embedding, skipping dense index query.")
                return []
            response = self.dense_index.query(
                namespace=self.config.dense_namespace,
                vector=query_embedding,
                top_k=self.config.dense_top_k,
                include_metadata=True
            )
            logger.info(f"Dense index queried for top {self.config.dense_top_k} results.")
            initial_results = response.get('matches', [])
            if not initial_results:
                logger.warning("No results returned from dense index query.")
                return []

            results = []
            for match in initial_results:
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata'],
                    'source': 'dense',
                    'pinecone_rerank_score': match['score']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying dense index: {e}")
            return []

    def combine_and_deduplicate(self, sparse_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        '''Combine and deduplicate results from sparse and dense indexes'''
        combined = {res['id']: res for res in sparse_results + dense_results}
        combined_list = list(combined.values())
        logger.info(f"Combined results count after deduplication: {len(combined_list)}")
        return combined_list

    def rerank_with_bert(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        '''Rerank combined candidates using BERT cross-encoder'''
        if not candidates:
            logger.warning("No candidates to rerank with BERT.")
            return []            
        texts = [res['metadata'].get('text', '') for res in candidates]
        pairs = [[query, text] for text in texts]
        
        try:
            scores = self.bert_reranker.predict(pairs)
            for i, score in enumerate(scores):
                candidates[i]['bert_score'] = score
            
            ranked = sorted(candidates, key=lambda x: x['bert_score'], reverse=True)
            logger.info("Candidates reranked using BERT.")
            return ranked[:top_k]

        except Exception as e:
            logger.error(f"Error during BERT reranking: {e}")
            return candidates[:top_k]

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        '''Main retrieval method'''
        logger.info(f"Starting retrieval for query: {query}")
        
        sparse_results = self._query_sparse_index_with_reranking(query)
        dense_results = self._query_dense_index(query)
        
        combined = self.combine_and_deduplicate(sparse_results, dense_results)

        final_results = self.rerank_with_bert(query, combined, top_k=top_k)

        logger.info(f"Retrieval completed. Returning top {len(final_results)} results.")
        return final_results