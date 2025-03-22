from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger(__name__)

def load_dataset_corpus(logger):
    dataset = load_dataset("mteb/nfcorpus", name="corpus")
    dataset_corpus = dataset["corpus"]
    logger.info(f"Dataset columns: {dataset_corpus.column_names}")
    logger.info(f"Dataset entries: {len(dataset_corpus)}")
    
    ids = [doc["_id"] for doc in dataset_corpus]  
    texts = [doc["text"] for doc in dataset_corpus]
    
    return ids, texts

def load_model(dimensions, logger):
    logger.info("Loading model")
    return SentenceTransformer("mixedbread-ai/mxbai-embed-xsmall-v1", truncate_dim=dimensions)

def compute_embeddings(model, texts, dimensions, logger):
    logger.info("Computing embeddings")
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
    
    return embeddings

def save_embeddings(embeddings, ids, texts, dimensions, logger, type):
    logger.info("Saving embeddings")
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(dimensions)])
    df["id"] = ids
    df["text"] = texts
    df.to_csv(f"./data/embeddings_{type}.csv", index=False)
    logger.info(f"Saved {len(df)} embeddings to ./data/embeddings_{type}.csv")
    

def quantize_embeddings_int8(embeddings, logger):
    logger.info("Quantizing embeddings to int8")
    min_val = embeddings.min()
    max_val = embeddings.max()
    scale = 127.0 / max(abs(min_val), abs(max_val))
    scaled = embeddings * scale
    
    int8_embeddings = scaled.astype(np.int8)
    return int8_embeddings

def quantize_embeddings_binary(embeddings, logger):
    logger.info("Quantizing embeddings to binary")
    binary_embeddings = (embeddings > 0).astype(np.bool)
    return binary_embeddings

def main():
    logger = setup_logging()
    dimensions = 384
    
    ids, texts = load_dataset_corpus(logger)
    model = load_model(dimensions, logger)
    
    embeddings = compute_embeddings(model, texts, dimensions, logger)
    
    save_embeddings(embeddings, ids, texts, dimensions, logger, "fp32")
    
    quantized_embeddings = quantize_embeddings_int8(embeddings, logger)
    save_embeddings(quantized_embeddings, ids, texts, dimensions, logger, "int8")
    
    quantized_embeddings = quantize_embeddings_binary(embeddings, logger)
    save_embeddings(quantized_embeddings, ids, texts, dimensions, logger, "binary")
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()







