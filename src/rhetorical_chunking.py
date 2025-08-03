import json
import re
import traceback  # Add missing import
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

from src.data_models import RhetoricalAnnotation, Entity, LegalMetadata, EnrichedChunk
from config.settings import settings

logger = logging.getLogger(__name__)

class RhetoricalChunker:
    """Advanced chunking based on existing RRL annotations"""
    
    def __init__(self):
        self.rhetorical_roles = settings.RHETORICAL_ROLES
        self.min_chunk_size = settings.MIN_CHUNK_SIZE
        self.max_chunk_size = settings.MAX_CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Ensure NLTK resources are available
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup required NLTK resources"""
        import nltk
        try:
            # Try punkt_tab first (newer versions)
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                logger.info("Downloading punkt_tab for sentence tokenization...")
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                # Fallback to punkt
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    logger.info("Downloading punkt for sentence tokenization...")
                    nltk.download('punkt', quiet=True)
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load merged JSON document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def parse_annotations(self, annotations: List[Dict[str, Any]]) -> List[RhetoricalAnnotation]:
        """Parse RRL annotations into structured format"""
        parsed_annotations = []
        invalid_count = 0
        
        for ann in annotations:
            try:
                annotation = RhetoricalAnnotation(
                    start=ann.get('start'),
                    end=ann.get('end'),
                    text=ann.get('text'),
                    labels=ann.get('labels', []),
                    id=ann.get('id')
                )
                parsed_annotations.append(annotation)
            except Exception as e:
                invalid_count += 1
                logger.warning(f"Failed to parse annotation {invalid_count}: {e}")
                continue
        
        logger.info(f"Parsed {len(parsed_annotations)} annotations (skipped {invalid_count} invalid)")
        return parsed_annotations
    
    def parse_entities(self, entities: List[Dict[str, Any]]) -> List[Entity]:
        """Parse NER entities with key remapping for 'start_char'/'end_char'"""
        parsed_entities = []
        invalid_count = 0
        
        for ent in entities:
            try:
                # Remap keys to match Entity model
                entity = Entity(
                    text=ent.get('text'),
                    label=ent.get('label'),
                    start=ent.get('start_char'),  # Remap 'start_char' to 'start'
                    end=ent.get('end_char'),      # Remap 'end_char' to 'end'
                    confidence=ent.get('confidence')
                )
                parsed_entities.append(entity)
            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:  # Log only first 5 errors to avoid spam
                    logger.warning(f"Failed to parse entity: {e}")
                continue
        
        logger.info(f"Parsed {len(parsed_entities)} entities (skipped {invalid_count} invalid)")
        return parsed_entities
    
    def find_entities_in_span(self, entities: List[Entity], start: int, end: int) -> List[Entity]:
        """Find entities within a text span"""
        span_entities = []
        
        for entity in entities:
            # Check if entity is within the span (handle None values)
            if entity.start is None or entity.end is None:
                continue
            if (entity.start >= start and entity.end <= end) or \
               (entity.start < end and entity.end > start):  # Overlapping
                span_entities.append(entity)
        
        return span_entities
    
    def create_rhetorical_chunks(self, document: Dict[str, Any]) -> List[EnrichedChunk]:
        """Create chunks based on rhetorical role annotations"""
        
        doc_id = document.get('id', 'unknown')
        cleaned_text = document.get('data', {}).get('cleaned_text', '')
        
        if not cleaned_text:
            logger.warning(f"No cleaned_text found in document {doc_id}")
            return []
        
        # Parse annotations and entities
        annotations = self.parse_annotations(document.get('annotations', []))
        entities = self.parse_entities(document.get('entities', []))
        
        chunks = []
        
        # Group annotations by rhetorical roles for better chunking
        role_groups = self._group_annotations_by_role(annotations)
        
        chunk_index = 0
        
        for role, role_annotations in role_groups.items():
            # Merge nearby annotations of the same role
            merged_spans = self._merge_nearby_spans(role_annotations)
            
            for span_start, span_end, span_annotations in merged_spans:
                # Extract text for this span (handle None offsets)
                if span_start is None or span_end is None:
                    continue
                    
                # Ensure span is within text bounds
                span_start = max(0, min(span_start, len(cleaned_text)))
                span_end = max(span_start, min(span_end, len(cleaned_text)))
                
                chunk_text = cleaned_text[span_start:span_end].strip()
                
                if len(chunk_text) < self.min_chunk_size:
                    continue
                
                # Split large chunks while preserving sentence boundaries
                if len(chunk_text) > self.max_chunk_size:
                    sub_chunks = self._split_large_chunk(chunk_text, span_start)
                    
                    for sub_start, sub_end, sub_text in sub_chunks:
                        chunk = self._create_enriched_chunk(
                            doc_id, chunk_index, sub_text, 
                            sub_start, sub_end, span_annotations,
                            entities, document
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                else:
                    chunk = self._create_enriched_chunk(
                        doc_id, chunk_index, chunk_text,
                        span_start, span_end, span_annotations,
                        entities, document
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        # Handle text not covered by annotations
        uncovered_chunks = self._create_uncovered_chunks(
            cleaned_text, annotations, entities, document, chunk_index
        )
        chunks.extend(uncovered_chunks)
        
        logger.info(f"Created {len(chunks)} rhetorical chunks for document {doc_id}")
        return chunks
    
    def _group_annotations_by_role(self, annotations: List[RhetoricalAnnotation]) -> Dict[str, List[RhetoricalAnnotation]]:
        """Group annotations by their primary rhetorical role"""
        role_groups = {}
        
        for annotation in annotations:
            # Use the first label as primary role (handle empty lists)
            primary_role = annotation.labels[0] if annotation.labels else "UNKNOWN"
            
            if primary_role not in role_groups:
                role_groups[primary_role] = []
            
            role_groups[primary_role].append(annotation)
        
        return role_groups
    
    def _merge_nearby_spans(self, annotations: List[RhetoricalAnnotation], 
                          max_gap: int = 100) -> List[Tuple[int, int, List[RhetoricalAnnotation]]]:
        """Merge nearby annotations of the same role"""
        if not annotations:
            return []
        
        # Sort by start position (skip if start is None)
        sorted_annotations = [ann for ann in annotations if ann.start is not None]
        sorted_annotations.sort(key=lambda x: x.start)
        
        if not sorted_annotations:
            return []
        
        merged_spans = []
        current_start = sorted_annotations[0].start
        current_end = sorted_annotations[0].end or current_start  # Handle None end
        current_annotations = [sorted_annotations[0]]
        
        for annotation in sorted_annotations[1:]:
            ann_start = annotation.start
            ann_end = annotation.end or ann_start
            
            # If the gap is small, merge
            if ann_start - current_end <= max_gap:
                current_end = max(current_end, ann_end)
                current_annotations.append(annotation)
            else:
                # Finalize current span
                merged_spans.append((current_start, current_end, current_annotations))
                
                # Start new span
                current_start = ann_start
                current_end = ann_end
                current_annotations = [annotation]
        
        # Add the last span
        merged_spans.append((current_start, current_end, current_annotations))
        
        return merged_spans
    
    def _split_large_chunk(self, text: str, offset: int) -> List[Tuple[int, int, str]]:
        """Split large chunks at sentence boundaries"""
        import nltk
        
        try:
            # Try new punkt_tab first
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}. Using simple sentence splitting.")
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        sub_chunks = []
        current_chunk = ""
        current_start = offset
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunk_end = current_start + len(current_chunk)
                    sub_chunks.append((current_start, chunk_end, current_chunk))
                
                # Start new chunk with overlap
                if len(sentence) > self.max_chunk_size:
                    # Handle extremely long sentences
                    sentence = sentence[:self.max_chunk_size]
                
                current_chunk = sentence
                current_start = max(0, current_start + len(current_chunk) - self.chunk_overlap)
        
        # Add remaining chunk
        if current_chunk:
            chunk_end = current_start + len(current_chunk)
            sub_chunks.append((current_start, chunk_end, current_chunk))
        
        return sub_chunks if sub_chunks else [(offset, offset + len(text), text)]
    
    def _create_enriched_chunk(self, doc_id: str, chunk_index: int, text: str,
                              start: int, end: int, annotations: List[RhetoricalAnnotation],
                              all_entities: List[Entity], document: Dict[str, Any]) -> EnrichedChunk:
        """Create an enriched chunk with full metadata"""
        
        # Find entities in this chunk
        chunk_entities = self.find_entities_in_span(all_entities, start, end)
        
        # Extract rhetorical roles
        rhetorical_roles = []
        for annotation in annotations:
            rhetorical_roles.extend(annotation.labels or [])
        
        rhetorical_roles = list(set(rhetorical_roles))  # Remove duplicates
        primary_role = rhetorical_roles[0] if rhetorical_roles else "UNKNOWN"
        
        # Extract legal concepts and keywords
        keywords = self._extract_keywords(text)
        legal_concepts = self._extract_legal_concepts(text, chunk_entities)
        
        # Convert provision_statute_pairs to list of dicts with robust handling
        pairs = document.get('provision_statute_pairs', [])
        converted_pairs = []
        
        for pair in pairs:
            if isinstance(pair, list):
                if len(pair) >= 2:
                    # Handle cases with duplicates or more than 2 elements
                    # Take the first two unique elements
                    unique_items = []
                    for item in pair:
                        if item not in unique_items:
                            unique_items.append(item)
                        if len(unique_items) == 2:
                            break
                    
                    if len(unique_items) == 2:
                        converted_pairs.append({
                            'provision': unique_items[0],
                            'statute': unique_items[1]
                        })
                    elif len(unique_items) == 1:
                        # If only one unique item, treat as both provision and statute
                        converted_pairs.append({
                            'provision': unique_items[0],
                            'statute': unique_items[0]
                        })
                elif len(pair) == 1:
                    # Single element case
                    converted_pairs.append({
                        'provision': pair[0],
                        'statute': pair[0]
                    })
            elif isinstance(pair, dict):
                # Already a dict
                converted_pairs.append(pair)
            else:
                # Invalid format - skip with less verbose logging
                if len(converted_pairs) < 5:  # Only log first 5 errors
                    logger.debug(f"Skipping invalid pair format: {pair}")
        
        # Create metadata with converted pairs
        metadata = LegalMetadata(
            document_id=doc_id,
            chunk_id=f"{doc_id}_{chunk_index}",
            chunk_index=chunk_index,
            rhetorical_roles=rhetorical_roles,
            primary_role=primary_role,
            entities=chunk_entities,
            entity_types=list(set([e.label for e in chunk_entities if e.label])),
            entity_count=len(chunk_entities),
            precedent_count=len(document.get('precedent_clusters', {})),
            statute_count=len(document.get('statute_clusters', {})),
            provision_count=len(converted_pairs),
            precedent_clusters=document.get('precedent_clusters', {}),
            statute_clusters=document.get('statute_clusters', {}),
            provision_statute_pairs=converted_pairs,
            source_file=document.get('source_file', ''),
            original_start=start,
            original_end=end
        )
        
        return EnrichedChunk(
            id=f"{doc_id}_{chunk_index}",
            text=text,
            cleaned_text=text,  # Already cleaned
            metadata=metadata,
            keywords=keywords,
            legal_concepts=legal_concepts
        )
    
    def _create_uncovered_chunks(self, full_text: str, annotations: List[RhetoricalAnnotation],
                                entities: List[Entity], document: Dict[str, Any], 
                                start_index: int) -> List[EnrichedChunk]:
        """Create chunks for text not covered by annotations"""
        
        # Find uncovered regions
        covered_spans = [(ann.start, ann.end) for ann in annotations 
                        if ann.start is not None and ann.end is not None]
        covered_spans.sort()
        
        uncovered_regions = []
        last_end = 0
        
        for start, end in covered_spans:
            if start > last_end:
                uncovered_regions.append((last_end, start))
            last_end = max(last_end, end)
        
        # Add final region if exists
        if last_end < len(full_text):
            uncovered_regions.append((last_end, len(full_text)))
        
        # Create chunks for uncovered regions
        chunks = []
        chunk_index = start_index
        
        for region_start, region_end in uncovered_regions:
            # Ensure bounds
            region_start = max(0, min(region_start, len(full_text)))
            region_end = max(region_start, min(region_end, len(full_text)))
            
            region_text = full_text[region_start:region_end].strip()
            
            if len(region_text) < self.min_chunk_size:
                continue
            
            # Split large uncovered regions
            if len(region_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(region_text, region_start)
                
                for sub_start, sub_end, sub_text in sub_chunks:
                    chunk = self._create_enriched_chunk(
                        document.get('id', 'unknown'), chunk_index, sub_text,
                        sub_start, sub_end, [],  # No specific annotations
                        entities, document
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                chunk = self._create_enriched_chunk(
                    document.get('id', 'unknown'), chunk_index, region_text,
                    region_start, region_end, [],
                    entities, document  
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using legal-specific patterns with UPDATED entity recognition"""
        # Legal keyword patterns - UPDATED to match correct entity types
        legal_patterns = [
            r'\b(?:section|sec\.?)\s+\d+(?:\([a-zA-Z0-9]+\))?',  # Section references (PROVISION)
            r'\b(?:article|art\.?)\s+\d+(?:\([a-zA-Z0-9]+\))?',   # Article references (PROVISION)
            r'\b(?:rule|r\.?)\s+\d+(?:\([a-zA-Z0-9]+\))?',        # Rule references (PROVISION)
            r'\b\d{4}\s+\w+\s+\d+\b',                           # Case citations (CASE_NUMBER)
            r'\b(?:held|ratio|obiter|precedent|doctrine)\b',      # Legal concepts
            r'\b(?:justice|judge|j\.)\s+[A-Z][a-z]+',            # Judge names (JUDGE)
            r'\b(?:court|tribunal|bench)\b',                     # Court references (COURT)
            r'\b(?:petitioner|appellant|plaintiff)\b',          # Petitioner references
            r'\b(?:respondent|defendant|appellee)\b',            # Respondent references
        ]
        
        keywords = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend([match.strip() for match in matches])
        
        return list(set(keywords))

    def _extract_legal_concepts(self, text: str, entities: List[Entity]) -> List[str]:
        """Extract legal concepts from text and entities with UPDATED entity types"""
        concepts = []
        
        # Add entity texts as concepts for relevant entity types
        for entity in entities:
            if entity.label in ['STATUTE', 'PROVISION', 'PRECEDENT', 'CASE_NUMBER'] and entity.text:
                concepts.append(entity.text)
        
        # Legal concept patterns - UPDATED
        concept_patterns = [
            r'\b(?:fundamental rights?|constitutional law|due process)\b',
            r'\b(?:natural justice|audi alteram partem|nemo judex)\b',
            r'\b(?:res judicata|double jeopardy|estoppel)\b',
            r'\b(?:burden of proof|prima facie|mens rea)\b',
            r'\b(?:ratio decidendi|obiter dicta|stare decisis)\b',  # Added more legal concepts
            r'\b(?:ultra vires|bona fide|mala fide)\b',
            r'\b(?:habeas corpus|certiorari|mandamus)\b',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))


class DocumentProcessor:
    """Process all merged documents through rhetorical chunking"""
    
    def __init__(self):
        self.chunker = RhetoricalChunker()
    
    def process_all_documents(self, input_dir: str) -> List[EnrichedChunk]:
        """Process all merged JSON files"""
        input_path = Path(input_dir)
        all_chunks = []
        
        json_files = list(input_path.glob("merged_*.json"))
        logger.info(f"📁 Found {len(json_files)} merged JSON files to process")
        
        if not json_files:
            logger.warning(f"⚠️ No merged_*.json files found in {input_dir}")
            return []
        
        processed_files = 0
        failed_files = 0
        
        for i, file_path in enumerate(json_files, 1):
            logger.info(f"📄 Processing file {i}/{len(json_files)}: {file_path.name}")
            try:
                document = self.chunker.load_document(str(file_path))
                
                # Add source file info
                document['source_file'] = str(file_path)
                
                chunks = self.chunker.create_rhetorical_chunks(document)
                all_chunks.extend(chunks)
                processed_files += 1
                
                # Progress update every 10 files
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(json_files)} files processed, {len(all_chunks)} chunks created")
                
            except Exception as e:
                failed_files += 1
                logger.error(f"❌ Error processing {file_path}: {e}")
                if failed_files <= 3:  # Only show traceback for first 3 errors
                    logger.error(traceback.format_exc())
                continue
        
        logger.info(f"✅ Processing complete: {processed_files} files processed, {failed_files} failed")
        logger.info(f"📊 Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[EnrichedChunk], output_file: str):
        """Save processed chunks to JSON"""
        try:
            chunks_data = [chunk.model_dump() for chunk in chunks]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved {len(chunks)} chunks to {output_file}")
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            raise
