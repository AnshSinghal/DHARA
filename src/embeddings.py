import torch
import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

from src.data_models import EnrichedChunk, LegalMetadata
from config.settings import settings

logger = logging.getLogger(__name__)

class LegalEmbeddingModel:
    """Enhanced embedding model with metadata enrichment"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.LEGAL_EMBEDDING_MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading legal embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(f"Failed to load as SentenceTransformer: {e}")
            # Fallback to transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.embedding_dimension = self.model.config.hidden_size
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
    
    def create_enriched_text(self, chunk: EnrichedChunk) -> str:
        """Create enriched text with metadata for better embedding"""
        
        enriched_parts = []
        
        # Add rhetorical role context
        if chunk.metadata.rhetorical_roles:
            role_context = f"[{chunk.metadata.primary_role}]"
            enriched_parts.append(role_context)
        
        # Add entity context
        if chunk.metadata.entities:
            entity_context = " ".join([
                f"[{entity.label}:{entity.text}]" 
                for entity in chunk.metadata.entities[:5]  # Limit to top 5
            ])
            enriched_parts.append(entity_context)
        
        # Add legal concepts
        if chunk.legal_concepts:
            concept_context = f"[CONCEPTS: {', '.join(chunk.legal_concepts[:3])}]"
            enriched_parts.append(concept_context)
        
        # Add main text
        enriched_parts.append(chunk.cleaned_text)
        
        # Add precedent/statute context if significant
        if chunk.metadata.precedent_count > 0:
            enriched_parts.append(f"[PRECEDENTS: {chunk.metadata.precedent_count}]")
        
        if chunk.metadata.statute_count > 0:
            enriched_parts.append(f"[STATUTES: {chunk.metadata.statute_count}]")
        
        return " ".join(enriched_parts)
    
    def encode_chunks(self, chunks: List[EnrichedChunk], 
                     batch_size: int = 16, use_enrichment: bool = True) -> np.ndarray:
        """Encode chunks with optional metadata enrichment"""
        
        if use_enrichment:
            texts = [self.create_enriched_text(chunk) for chunk in chunks]
        else:
            texts = [chunk.cleaned_text for chunk in chunks]
        
        logger.info(f"Encoding {len(texts)} chunks with enrichment={'on' if use_enrichment else 'off'}")
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                if hasattr(self.model, 'encode'):
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                else:
                    batch_embeddings = self._encode_with_transformers(batch_texts)
                
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size}: {e}")
                # Create zero embeddings as fallback
                batch_embeddings = np.zeros((len(batch_texts), self.embedding_dimension))
                embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding using transformers"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            # Mean pooling
            attention_mask = encodings['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().numpy()
    
    def encode_query(self, query: str, query_context: Dict[str, Any] = None) -> np.ndarray:
        """Encode query with optional context enrichment"""
        
        enriched_query = query
        
        if query_context:
            # Add rhetorical role context if specified
            if 'rhetorical_roles' in query_context and query_context['rhetorical_roles']:
                role_context = f"[{query_context['rhetorical_roles'][0]}]"
                enriched_query = f"{role_context} {query}"
            
            # Add entity type context if specified
            if 'entity_types' in query_context and query_context['entity_types']:
                entity_context = f"[{','.join(query_context['entity_types'])}]"
                enriched_query = f"{entity_context} {enriched_query}"
        
        if hasattr(self.model, 'encode'):
            return self.model.encode([enriched_query], normalize_embeddings=True)[0]
        else:
            return self._encode_with_transformers([enriched_query])[0]

class MetadataAwareEmbedding:
    """Embedding that combines text embeddings with metadata features"""
    
    def __init__(self, text_embedding_model: LegalEmbeddingModel):
        self.text_model = text_embedding_model
        self.metadata_dim = 50  # Dimension for metadata features
    
    def create_metadata_features(self, metadata: LegalMetadata) -> np.ndarray:
        """Create numerical features from metadata"""
        features = np.zeros(self.metadata_dim)
        
        # Rhetorical role features (one-hot encoding)
        role_mapping = {role: i for i, role in enumerate(settings.RHETORICAL_ROLES.keys())}
        for role in metadata.rhetorical_roles[:5]:  # Top 5 roles
            if role in role_mapping and role_mapping[role] < 10:
                features[role_mapping[role]] = 1.0
        
        # Entity type features  
        entity_mapping = {etype: i+10 for i, etype in enumerate(settings.LEGAL_ENTITY_TYPES.keys())}
        for etype in metadata.entity_types[:5]:  # Top 5 entity types
            if etype in entity_mapping and entity_mapping[etype] < 25:
                features[entity_mapping[etype]] = 1.0
        
        # Numerical features (normalized)
        features[25] = min(metadata.precedent_count / 10.0, 1.0)  # Precedent count
        features[26] = min(metadata.statute_count / 10.0, 1.0)    # Statute count
        features[27] = min(metadata.provision_count / 20.0, 1.0)  # Provision count
        features[28] = min(metadata.entity_count / 20.0, 1.0)     # Entity count
        
        # Text length features
        if hasattr(metadata, 'text_length'):
            features[29] = min(metadata.text_length / 1000.0, 1.0)
        
        return features
    
    def encode_chunks_with_metadata(self, chunks: List[EnrichedChunk]) -> np.ndarray:
        """Create combined text + metadata embeddings"""
        
        if not chunks:
            logger.warning("No chunks to encode")
            return np.array([])  # Return empty NumPy array instead of None
        
        try:
            # Get text embeddings
            text_embeddings = self.text_model.encode_chunks(chunks, use_enrichment=True)
            
            # Get metadata features
            metadata_features = []
            for chunk in chunks:
                meta_features = self.create_metadata_features(chunk.metadata)
                metadata_features.append(meta_features)
            
            metadata_features = np.array(metadata_features)
            
            # Combine embeddings
            combined_embeddings = np.concatenate([text_embeddings, metadata_features], axis=1)
            
            logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
            return combined_embeddings
        
        except Exception as e:
            logger.error(f"Error encoding chunks with metadata: {e}")
            return np.array([])
