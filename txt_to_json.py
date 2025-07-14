import os
import json

class LegalDocumentIngestor:
    '''
    A class to ingest legal text documents, extract text while preserving
    paragraph structure, and save as a structured JSON object.
    '''
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_document(self, txt_path):
        '''Processes a single TXT document'''
        doc_id = os.path.basename(txt_path).replace('.txt', '')
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            blocks = [block.replace('\n', ' ').strip() for block in full_text.split('\n\n') if block.strip()]


            structured_output = {
                "document_id": doc_id,
                "source_file": txt_path,
                "content": {
                    "blocks": blocks
                }
            }

            output_path = os.path.join(self.output_dir, f"{doc_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_output, f, ensure_ascii=False, indent=4)
            
            return True, None

        except Exception as e:
            return False, str(e)

    def run_pipeline(self):
        '''Runs the ingestion pipeline for all TXTs in the input directory.'''
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.txt'):
                txt_path = os.path.join(self.input_dir, filename)
                success, error = self.process_document(txt_path)
                if not success:
                    print(f"Error processing {filename}: {error}")
                else:
                    print(f"Successfully processed {filename}")

if __name__ == "__main__":
    ingestor = LegalDocumentIngestor(input_dir='cases', output_dir='json_files')
    ingestor.run_pipeline()


        