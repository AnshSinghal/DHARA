#!/usr/bin/env python3
"""
JSON Case Merger - Merges RRL, NER, and cleaned text data for legal cases
"""

import json
import os
import time
import traceback

# Import DharaLogger
from logger import DharaLogger

# Initialize dharalogger for merge operations
dharalogger = DharaLogger("merge_json", log_file="merge_json.log")
logger = dharalogger.get_logger()

def merge_case(rrl_data, ner_data, cleaned_data):
    """Merge RRL, NER, and cleaned text data for a single case"""
    
    logger.debug("Starting case merge operation")
    logger.debug(f"RRL data keys: {list(rrl_data.keys()) if isinstance(rrl_data, dict) else 'Invalid format'}")
    logger.debug(f"NER data keys: {list(ner_data.keys()) if isinstance(ner_data, dict) else 'Invalid format'}")
    logger.debug(f"Cleaned data keys: {list(cleaned_data.keys()) if isinstance(cleaned_data, dict) else 'Invalid format'}")
    
    try:
        # Start with RRL as base
        merged = rrl_data.copy()
        logger.debug("Base merged structure created from RRL data")
        
        # Add cleaned_text directly to 'data' (create 'data' if missing)
        if 'data' not in merged:
            merged['data'] = {}
            logger.debug("Created 'data' section in merged structure")
        
        cleaned_text = cleaned_data.get('cleaned_text', '')
        merged['data']['cleaned_text'] = cleaned_text
        logger.debug(f"Added cleaned text ({len(cleaned_text)} characters)")
        
        # Merge NER keys directly into root (flatten)
        conflicts = 0
        direct_merges = 0
        
        for key, value in ner_data.items():
            if key in merged:
                # If conflict, nest NER version (e.g., 'ner_entities')
                merged[f'ner_{key}'] = value
                conflicts += 1
                logger.debug(f"Conflict resolved for key '{key}' -> 'ner_{key}'")
            else:
                merged[key] = value  # Direct merge if no conflict
                direct_merges += 1
                logger.debug(f"Direct merge for key '{key}'")
        
        logger.debug(f"NER merge completed: {direct_merges} direct merges, {conflicts} conflicts resolved")
        logger.debug(f"Final merged structure has {len(merged)} top-level keys")
        
        return merged
        
    except Exception as e:
        logger.error(f"Error during case merge: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def batch_merge_cases(rrl_dir, ner_dir, cleaned_dir, output_dir):
    """Batch merge cases from multiple directories"""
    
    logger.info("=" * 60)
    logger.info("STARTING BATCH CASE MERGE OPERATION")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    logger.info(f"Input directories:")
    logger.info(f"  RRL directory: {rrl_dir}")
    logger.info(f"  NER directory: {ner_dir}")
    logger.info(f"  Cleaned directory: {cleaned_dir}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Validate input directories
    for dir_path, dir_name in [(rrl_dir, "RRL"), (ner_dir, "NER"), (cleaned_dir, "Cleaned")]:
        if not os.path.exists(dir_path):
            logger.error(f"{dir_name} directory does not exist: {dir_path}")
            raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise
    
    # Get base names from RRL files (assuming RRL has the core files)
    try:
        rrl_files = [f for f in os.listdir(rrl_dir) if f.endswith('.json')]
        base_names = [f.split('.')[0] for f in rrl_files]  # e.g., 'case1'
        
        logger.info(f"Found {len(rrl_files)} RRL files to process")
        logger.info(f"Base names: {base_names[:10]}{' ...' if len(base_names) > 10 else ''}")
        
    except Exception as e:
        logger.error(f"Failed to list RRL files: {e}")
        raise
    
    # Processing statistics
    successful_merges = 0
    skipped_cases = 0
    failed_merges = 0
    
    for i, base in enumerate(base_names, 1):
        logger.info(f"Processing case {i}/{len(base_names)}: {base}")
        
        rrl_path = os.path.join(rrl_dir, f"{base}.json")
        ner_path = os.path.join(ner_dir, f"{base}_ner.json")
        cleaned_path = os.path.join(cleaned_dir, f"{base}.json")
        
        # Check if all required files exist
        missing_files = []
        for path, file_type in [(rrl_path, "RRL"), (ner_path, "NER"), (cleaned_path, "Cleaned")]:
            if not os.path.exists(path):
                missing_files.append(file_type)
        
        if missing_files:
            logger.warning(f"Skipping {base}: Missing files - {', '.join(missing_files)}")
            skipped_cases += 1
            continue
        
        try:
            # Load files
            logger.debug(f"Loading files for {base}")
            
            with open(rrl_path, 'r', encoding='utf-8') as f:
                rrl_data = json.load(f)
            logger.debug(f"Loaded RRL data: {len(json.dumps(rrl_data))} characters")
            
            with open(ner_path, 'r', encoding='utf-8') as f:
                ner_data = json.load(f)
            logger.debug(f"Loaded NER data: {len(json.dumps(ner_data))} characters")
            
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            logger.debug(f"Loaded cleaned data: {len(json.dumps(cleaned_data))} characters")
            
            # Merge
            logger.debug(f"Merging data for {base}")
            merge_start = time.time()
            merged_data = merge_case(rrl_data, ner_data, cleaned_data)
            merge_time = time.time() - merge_start
            
            # Save
            output_path = os.path.join(output_dir, f"merged_{base}.json")
            logger.debug(f"Saving merged data to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=4, ensure_ascii=False)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Merged {base} -> {output_path} ({file_size:,} bytes, {merge_time:.3f}s)")
            successful_merges += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to merge {base}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            failed_merges += 1
    
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("BATCH MERGE OPERATION COMPLETED")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Statistics:")
    logger.info(f"  Total cases found: {len(base_names)}")
    logger.info(f"  Successfully merged: {successful_merges}")
    logger.info(f"  Skipped (missing files): {skipped_cases}")
    logger.info(f"  Failed merges: {failed_merges}")
    logger.info(f"  Success rate: {successful_merges/len(base_names)*100:.1f}%")
    logger.info("=" * 60)
    
    return {
        'total_cases': len(base_names),
        'successful': successful_merges,
        'skipped': skipped_cases,
        'failed': failed_merges,
        'processing_time': total_time
    }

if __name__ == "__main__":
    # Configuration
    rrl_dir = './rrl/rrl_json_files/'         # Folder with case1.json, etc.
    ner_dir = './ner/ner_json_files/'         # Folder with case1_ner.json, etc.
    cleaned_dir = './cleaned_text/cleaned_cases/' # Folder with case1_cleaned.json, etc.
    output_dir = './merged_output/'   # Output folder
    
    logger.info("JSON Case Merger starting with configuration:")
    logger.info(f"  RRL directory: {rrl_dir}")
    logger.info(f"  NER directory: {ner_dir}")
    logger.info(f"  Cleaned directory: {cleaned_dir}")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        results = batch_merge_cases(rrl_dir, ner_dir, cleaned_dir, output_dir)
        
        if results['successful'] > 0:
            logger.info("✅ JSON Case Merger completed successfully!")
            print(f"\n✅ Merge operation completed!")
            print(f"Successfully merged: {results['successful']} cases")
            print(f"Total time: {results['processing_time']:.2f}s")
            print(f"Output directory: {output_dir}")
        else:
            logger.warning("⚠️ No cases were successfully merged")
            print("⚠️ No cases were successfully merged")
            
    except Exception as e:
        logger.error(f"❌ JSON Case Merger failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        print(f"❌ Merge operation failed: {e}")
        exit(1)
