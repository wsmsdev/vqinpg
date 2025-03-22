from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
   
)
logger = logging.getLogger(__name__)

# Load the correct dataset configuration
dataset = load_dataset("mteb/nfcorpus", name="corpus")
dataset_corpus = dataset["corpus"]
logger.info(f"Dataset columns: {dataset_corpus.column_names}")
logger.info(f"Dataset entries: {len(dataset_corpus)}")

dimensions = 384

logger.info("Loading model")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-xsmall-v1", truncate_dim=dimensions)


ids = [doc["_id"] for doc in dataset_corpus]  
texts = [doc["text"] for doc in dataset_corpus] 

logger.info("Computing embeddings")
# Compute embeddings in batches
batch_size = 100
embeddings = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    logger.info(f"Processing batch {i//batch_size+1} of {len(texts)//batch_size}")
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
    embeddings.extend(batch_embeddings)
    
# Check vector point type and dimensions
logger.info("Checking vector point type and dimensions")
embeddings = np.array(embeddings, dtype=np.float32)
assert embeddings.shape[1] == dimensions, f"Expected {dimensions} dimensions, got {embeddings.shape[1]}"

#Save precomputed embeddins for later use
logger.info("Saving embeddings")
df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(dimensions)])
df["id"] = ids
df["text"] = texts
df.to_csv("./data/embeddings.csv", index=False)
logger.info(f"Saved {len(df)} embeddings to ./data/embeddings.csv")







