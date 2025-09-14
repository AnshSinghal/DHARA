import os
import json
from glob import glob
from opennyai import Pipeline
from opennyai.utils import Data
from logger import DharaLogger

logger = DharaLogger(__name__).get_logger()

input_dir = "cases"
output_dir = "rrl"
os.makedirs(output_dir, exist_ok=True)

file_paths = sorted(glob(os.path.join(input_dir, "*.txt")))
texts = []
doc_ids = []

logger.info(f"Found {len(file_paths)} case files in '{input_dir}' directory.")
logger.debug(f"File paths: {file_paths}")

for path in file_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
            doc_ids.append(os.path.splitext(os.path.basename(path))[0])  # e.g., "case1"
        logger.debug(f"Loaded file: {path}")
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        logger.exception(e)

try:
    data = Data(texts, mini_batch_size=2048)
    logger.info("Data object created for pipeline.")
except Exception as e:
    logger.critical(f"Failed to create Data object: {e}")
    logger.exception(e)
    raise

try:
    pipe = Pipeline(components=["Rhetorical_Role"], use_gpu=True)
    logger.info("Pipeline initialized with Rhetorical_Role component.")
except Exception as e:
    logger.critical(f"Failed to initialize pipeline: {e}")
    logger.exception(e)
    raise

try:
    results = pipe(data)  # list of lists of {"sentence", "role"}
    logger.info("RRL inference completed.")
except Exception as e:
    logger.critical(f"Pipeline run failed: {e}")
    logger.exception(e)
    raise

for doc_id, result in zip(doc_ids, results):
    output_path = os.path.join(output_dir, f"{doc_id}.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved RRL result for {doc_id} to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save result for {doc_id}: {e}")
        logger.exception(e)

logger.info(f"Inference complete. {len(results)} files saved to {output_dir}")
