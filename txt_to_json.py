#!/usr/bin/env python3
"""
Legal Document Ingestor - Converts legal text documents to structured JSON format
"""

import os
import json
import time
import traceback
from logger import DharaLogger

class LegalDocumentIngestor:
    '''
    A class to ingest legal text documents, extract text while preserving
    paragraph structure, and save as a structured JSON object.
    '''
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialize dharalogger for document ingestion
        self.dharalogger = DharaLogger("legal_document_ingestor", log_file="txt_to_json.log")
        self.logger = self.dharalogger.get_logger()
        
        self.logger.info("=" * 60)
        self.logger.info("LEGAL DOCUMENT INGESTOR INITIALIZED")
        self.logger.info("=" * 60)
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Validate input directory
        if not os.path.exists(input_dir):
            self.logger.error(f"Input directory does not exist: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")
            else:
                self.logger.info(f"Output directory already exists: {output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

    def process_document(self, txt_path):
        '''Processes a single TXT document'''
        doc_id = os.path.basename(txt_path).replace('.txt', '')
        self.logger.info(f"Processing document: {doc_id}")
        
        start_time = time.time()
        
        try:
            # Read the document
            self.logger.debug(f"Reading file: {txt_path}")
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            file_size = len(full_text)
            self.logger.debug(f"Read {file_size:,} characters from {doc_id}")
            
            # Process text into blocks
            self.logger.debug(f"Processing text into paragraph blocks...")
            blocks = [block.replace('\n', ' ').strip() for block in full_text.split('\n\n') if block.strip()]
            
            self.logger.debug(f"Created {len(blocks)} text blocks")
            if blocks:
                avg_block_length = sum(len(block) for block in blocks) / len(blocks)
                self.logger.debug(f"Average block length: {avg_block_length:.0f} characters")

            # Create structured output
            structured_output = {
                "document_id": doc_id,
                "source_file": txt_path,
                "processing_timestamp": time.time(),
                "content": {
                    "blocks": blocks,
                    "total_blocks": len(blocks),
                    "total_characters": file_size,
                    "processed_characters": sum(len(block) for block in blocks)
                }
            }

            # Save JSON output
            output_path = os.path.join(self.output_dir, f"{doc_id}.json")
            self.logger.debug(f"Saving structured output to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_output, f, ensure_ascii=False, indent=4)
            
            output_size = os.path.getsize(output_path)
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ Successfully processed {doc_id}:")
            self.logger.info(f"   Processing time: {processing_time:.3f}s")
            self.logger.info(f"   Input size: {file_size:,} characters")
            self.logger.info(f"   Output size: {output_size:,} bytes")
            self.logger.info(f"   Blocks created: {len(blocks)}")
            self.logger.info(f"   Output file: {output_path}")
            
            return True, None

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Error processing {doc_id} after {processing_time:.3f}s: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False, str(e)

    def run_pipeline(self):
        '''Runs the ingestion pipeline for all TXTs in the input directory.'''
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING DOCUMENT INGESTION PIPELINE")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Find all text files
        try:
            all_files = os.listdir(self.input_dir)
            txt_files = [f for f in all_files if f.endswith('.txt')]
            
            self.logger.info(f"Found {len(txt_files)} text files in {self.input_dir}")
            self.logger.info(f"Total files in directory: {len(all_files)}")
            
            if not txt_files:
                self.logger.warning("No .txt files found in input directory")
                return
                
        except Exception as e:
            self.logger.error(f"Failed to list files in input directory: {e}")
            return
        
        # Processing statistics
        successful_count = 0
        failed_count = 0
        total_input_size = 0
        total_output_size = 0
        
        for i, filename in enumerate(txt_files, 1):
            self.logger.info(f"Processing file {i}/{len(txt_files)}: {filename}")
            
            txt_path = os.path.join(self.input_dir, filename)
            
            try:
                # Get input file size
                input_size = os.path.getsize(txt_path)
                total_input_size += input_size
                
                # Process the document
                success, error = self.process_document(txt_path)
                
                if not success:
                    self.logger.error(f"Failed to process {filename}: {error}")
                    failed_count += 1
                else:
                    successful_count += 1
                    
                    # Get output file size
                    doc_id = os.path.basename(txt_path).replace('.txt', '')
                    output_path = os.path.join(self.output_dir, f"{doc_id}.json")
                    if os.path.exists(output_path):
                        output_size = os.path.getsize(output_path)
                        total_output_size += output_size
                        
            except Exception as e:
                self.logger.error(f"Unexpected error processing {filename}: {e}")
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                failed_count += 1
        
        total_time = time.time() - start_time
        
        # Final statistics
        self.logger.info("=" * 60)
        self.logger.info("DOCUMENT INGESTION PIPELINE COMPLETED")
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"Statistics:")
        self.logger.info(f"  Total files found: {len(txt_files)}")
        self.logger.info(f"  Successfully processed: {successful_count}")
        self.logger.info(f"  Failed to process: {failed_count}")
        self.logger.info(f"  Success rate: {successful_count/len(txt_files)*100:.1f}%")
        self.logger.info(f"  Total input size: {total_input_size:,} bytes")
        self.logger.info(f"  Total output size: {total_output_size:,} bytes")
        if total_input_size > 0:
            self.logger.info(f"  Compression ratio: {total_output_size/total_input_size:.2f}")
        if len(txt_files) > 0:
            self.logger.info(f"  Average processing time: {total_time/len(txt_files):.3f}s per file")
        self.logger.info("=" * 60)
        
        return {
            'total_files': len(txt_files),
            'successful': successful_count,
            'failed': failed_count,
            'processing_time': total_time,
            'input_size': total_input_size,
            'output_size': total_output_size
        }

if __name__ == "__main__":
    input_dir = 'cases'
    output_dir = 'json_files'
    
    print("Legal Document Ingestor")
    print("=" * 40)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        ingestor = LegalDocumentIngestor(input_dir=input_dir, output_dir=output_dir)
        results = ingestor.run_pipeline()
        
        if results and results['successful'] > 0:
            print(f"\n✅ Document ingestion completed successfully!")
            print(f"Processed: {results['successful']}/{results['total_files']} files")
            print(f"Processing time: {results['processing_time']:.2f}s")
            print(f"Output directory: {output_dir}")
        else:
            print("⚠️ No documents were successfully processed")
            
    except Exception as e:
        print(f"❌ Document ingestion failed: {e}")
        exit(1)


        