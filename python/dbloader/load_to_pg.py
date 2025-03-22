import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
import logging
import time
from datetime import datetime


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger(__name__)
   

def connect_to_db():
    load_dotenv()
    conn = psycopg2.connect(
    host="localhost",
    database=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    port=5432
    )
    return conn
    

def get_table_size(cursor, table_name, logger):
    logger.info(f"Getting size for table {table_name}")
    cursor.execute(f"SELECT pg_table_size('{table_name}')")
    return cursor.fetchone()[0]

def get_index_size(cursor, index_name, logger):
    logger.info(f"Getting size for index {index_name}")
    cursor.execute(f"SELECT pg_relation_size('{index_name}')")
    return cursor.fetchone()[0]

def save_metrics_to_markdown(results, logger, include_halfvec=True, include_bit=True):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"./data/embedding_metrics_{timestamp}.md"
    
    with open(filename, "w") as f:
        f.write("# Embedding Performance Metrics\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Comparison\n\n")
        headers = ["Metric", "FP32"]
        if include_halfvec:
            headers.append("HALFVEC")
        if include_bit:
            headers.append("BIT")
        
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["----" for _ in headers]) + "|\n")
        
        metrics = ["total_vectors", "dimensions", "insert_time", "index_time", "table_size", "index_size"]
        labels = ["Total Vectors", "Dimensions", "Insert Time (s)", "Index Creation Time (s)", 
                  "Table Size (MB)", "Index Size (MB)"]
        
        for metric, label in zip(metrics, labels):
            line = f"| {label} | {results['fp32'][metric]}"
            if metric in ["table_size", "index_size"]:
                line = f"| {label} | {results['fp32'][metric]/1024/1024:.2f}"
                if include_halfvec:
                    line += f" | {results['halfvec'][metric]/1024/1024:.2f}"
                if include_bit:
                    line += f" | {results['bit'][metric]/1024/1024:.2f}"
            else:
                if include_halfvec:
                    line += f" | {results['halfvec'][metric]}"
                if include_bit:
                    line += f" | {results['bit'][metric]}"
            line += " |"
            f.write(line + "\n")
        
        emb_types = ["fp32"]
        if include_halfvec:
            emb_types.append("halfvec")
        if include_bit:
            emb_types.append("bit")
        
        for emb_type in emb_types:
            f.write(f"\n## {emb_type.upper()} Embeddings\n\n")
            f.write(f"- **Vectors:** {results[emb_type]['total_vectors']}\n")
            f.write(f"- **Dimensions:** {results[emb_type]['dimensions']}\n")
            f.write(f"- **Insert Time:** {results[emb_type]['insert_time']:.2f} seconds\n")
            f.write(f"- **Index Creation Time:** {results[emb_type]['index_time']:.2f} seconds\n")
            f.write(f"- **Table Size:** {results[emb_type]['table_size']/1024/1024:.2f} MB\n")
            f.write(f"- **Index Size:** {results[emb_type]['index_size']/1024/1024:.2f} MB\n")
    
    logger.info(f"Metrics saved to {filename}")
    return filename


def process_embeddings(cursor, logger, dimensions=384):
    file_path = "./data/embeddings_fp32.csv"
    logger.info(f"Loading original fp32 embeddings from {file_path}")
    df = pd.read_csv(file_path)
    vector_columns = [col for col in df.columns if col.startswith("dim_")]
    total_vectors = len(df)
    logger.info(f"Total vectors: {total_vectors}")
    
    
    if len(vector_columns) != dimensions:
        dimensions = len(vector_columns)
        logger.info(f"Adjusted dimensions to {dimensions} based on data")
    
    results = {}
    
    fp32_results = process_fp32_vectors(cursor, logger, df, vector_columns, dimensions)
    results["fp32"] = fp32_results
    
    halfvec_results = process_halfvec_vectors(cursor, logger, df, vector_columns, dimensions)
    results["halfvec"] = halfvec_results
    
    bit_results = process_bit_vectors(cursor, logger, df, vector_columns, dimensions)
    results["bit"] = bit_results
    
    return results

def process_fp32_vectors(cursor, logger, df, vector_columns, dimensions):
    table_name = "embeddings_fp32"
    index_name = f"{table_name}_idx"
    total_vectors = len(df)
    
    logger.info(f"Creating table for fp32 embeddings")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding vector({dimensions})
        )
    """)
    
    start_insert_time = time.time()
    for index, row in df.iterrows():
        id_val = row["id"]
        text = row["text"]
        vector_str = '[' + ','.join([str(float(row[col])) for col in vector_columns]) + ']'
        
        cursor.execute(f"INSERT INTO {table_name} (id, text, embedding) VALUES (%s, %s, %s::vector)", 
                      (id_val, text, vector_str))
        if index % 100 == 0:
            logger.info(f"Inserted {index} of {total_vectors} fp32 embeddings")
    
    insert_time = time.time() - start_insert_time
    logger.info(f"Inserted {total_vectors} fp32 embeddings in {insert_time:.2f} seconds")
    
    cursor.connection.commit()
    
    logger.info(f"Creating HNSW index for fp32 embeddings")
    start_index_time = time.time()
    
    cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
    cursor.execute(f"""
    CREATE INDEX {index_name} ON {table_name} 
    USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 64)
    """)
    
    index_time = time.time() - start_index_time
    logger.info(f"Created index for fp32 embeddings in {index_time:.2f} seconds")
    
    cursor.connection.commit()
    
    table_size = get_table_size(cursor, table_name, logger)
    index_size = get_index_size(cursor, index_name, logger)
    
    logger.info(f"fp32 table size: {table_size/1024/1024:.2f} MB")
    logger.info(f"fp32 index size: {index_size/1024/1024:.2f} MB")
    
    return {
        "embedding_type": "fp32",
        "total_vectors": total_vectors,
        "dimensions": dimensions,
        "insert_time": insert_time,
        "index_time": index_time,
        "table_size": table_size,
        "index_size": index_size
    }

def process_halfvec_vectors(cursor, logger, df, vector_columns, dimensions):
    table_name = "embeddings_halfvec"
    index_name = f"{table_name}_idx"
    total_vectors = len(df)
    
    logger.info(f"Creating table for halfvec embeddings")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding halfvec({dimensions})
        )
    """)
    
    start_insert_time = time.time()
    for index, row in df.iterrows():
        id_val = row["id"]
        text = row["text"]
        # The vector format is the same as for fp32, but will be cast to halfvec
        vector_str = '[' + ','.join([str(float(row[col])) for col in vector_columns]) + ']'
        
        cursor.execute(f"INSERT INTO {table_name} (id, text, embedding) VALUES (%s, %s, %s::halfvec)", 
                      (id_val, text, vector_str))
        if index % 100 == 0:
            logger.info(f"Inserted {index} of {total_vectors} halfvec embeddings")
    
    insert_time = time.time() - start_insert_time
    logger.info(f"Inserted {total_vectors} halfvec embeddings in {insert_time:.2f} seconds")
    
    cursor.connection.commit()
    
    logger.info(f"Creating HNSW index for halfvec embeddings")
    start_index_time = time.time()
    
    cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
    cursor.execute(f"""
    CREATE INDEX {index_name} ON {table_name} 
    USING hnsw (embedding halfvec_l2_ops) 
    WITH (m = 16, ef_construction = 64)
    """)
    
    index_time = time.time() - start_index_time
    logger.info(f"Created index for halfvec embeddings in {index_time:.2f} seconds")
    
    cursor.connection.commit()
    
    table_size = get_table_size(cursor, table_name, logger)
    index_size = get_index_size(cursor, index_name, logger)
    
    logger.info(f"halfvec table size: {table_size/1024/1024:.2f} MB")
    logger.info(f"halfvec index size: {index_size/1024/1024:.2f} MB")
    
    return {
        "embedding_type": "halfvec",
        "total_vectors": total_vectors,
        "dimensions": dimensions,
        "insert_time": insert_time,
        "index_time": index_time,
        "table_size": table_size,
        "index_size": index_size
    }

def process_bit_vectors(cursor, logger, df, vector_columns, dimensions):
    table_name = "embeddings_bit"
    index_name = f"{table_name}_idx"
    total_vectors = len(df)
    
    logger.info(f"Creating table for bit embeddings")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding BIT({dimensions})
        )
    """)
    
    start_insert_time = time.time()
    for index, row in df.iterrows():
        id_val = row["id"]
        text = row["text"]
        # Use simple thresholding to convert float values to binary
        # You can use a more sophisticated method like mean or median of vector
        # Check https://www.sbert.net/examples/applications/embedding-quantization/ for more details
        bit_string = ''.join(['1' if float(row[col]) > 0 else '0' for col in vector_columns])
        
        cursor.execute(f"INSERT INTO {table_name} (id, text, embedding) VALUES (%s, %s, %s::BIT({dimensions}))", 
                     (id_val, text, bit_string))
        if index % 100 == 0:
            logger.info(f"Inserted {index} of {total_vectors} bit embeddings")
    
    insert_time = time.time() - start_insert_time
    logger.info(f"Inserted {total_vectors} bit embeddings in {insert_time:.2f} seconds")
    
    cursor.connection.commit()
    
    logger.info(f"Creating HNSW index for bit embeddings")
    start_index_time = time.time()
    
    cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
    cursor.execute(f"""
    CREATE INDEX {index_name} ON {table_name} 
    USING hnsw (embedding bit_hamming_ops) 
    WITH (m = 16, ef_construction = 64)
    """)
    
    
    index_time = time.time() - start_index_time
    logger.info(f"Created index for bit embeddings in {index_time:.2f} seconds")
    
    cursor.connection.commit()
    
    table_size = get_table_size(cursor, table_name, logger)
    index_size = get_index_size(cursor, index_name, logger)
    
    logger.info(f"bit table size: {table_size/1024/1024:.2f} MB")
    logger.info(f"bit index size: {index_size/1024/1024:.2f} MB")
    
    return {
        "embedding_type": "bit",
        "total_vectors": total_vectors,
        "dimensions": dimensions,
        "insert_time": insert_time,
        "index_time": index_time,
        "table_size": table_size,
        "index_size": index_size
    }

def create_search_examples(cursor, logger):
    logger.info("Creating example search functions")
    
    cursor.execute("""
    CREATE OR REPLACE FUNCTION search_fp32(query_vector FLOAT[], k INTEGER DEFAULT 10)
    RETURNS TABLE(id TEXT, text TEXT, distance FLOAT) AS $$
    BEGIN
        RETURN QUERY 
        SELECT id, text, embedding <-> query_vector::vector AS distance
        FROM embeddings_fp32
        ORDER BY distance
        LIMIT k;
    END;
    $$ LANGUAGE plpgsql;
    """)
    
    cursor.execute("""
    CREATE OR REPLACE FUNCTION search_halfvec(query_vector FLOAT[], k INTEGER DEFAULT 10)
    RETURNS TABLE(id TEXT, text TEXT, distance FLOAT) AS $$
    BEGIN
        RETURN QUERY 
        SELECT id, text, embedding <-> query_vector::halfvec AS distance
        FROM embeddings_halfvec
        ORDER BY distance
        LIMIT k;
    END;
    $$ LANGUAGE plpgsql;
    """)
    
    cursor.execute("""
    CREATE OR REPLACE FUNCTION search_bit(query_bit BIT, k INTEGER DEFAULT 10)
    RETURNS TABLE(id TEXT, text TEXT, distance FLOAT) AS $$
    BEGIN
        RETURN QUERY 
        SELECT id, text, embedding <-> query_bit AS distance
        FROM embeddings_bit
        ORDER BY distance
        LIMIT k;
    END;
    $$ LANGUAGE plpgsql;
    """)
    
    # Reranking function for bit vectors (using original fp32 for reranking)
   
    
    logger.info("Example search functions created")

def main():
    logger = setup_logging()
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
      
        results = process_embeddings(cursor, logger)
        logger.info("Processing embeddings")
        create_search_examples(cursor, logger)
        logger.info("Creating search examples")
        metrics_file = save_metrics_to_markdown(results, logger)
        logger.info(f"\nMetrics saved to markdown file: {metrics_file}")
        
        logger.info("\n=== PERFORMANCE SUMMARY ===")
        for emb_type, metrics in results.items():
            logger.info(f"\n{emb_type.upper()} EMBEDDINGS:")
            logger.info(f"  Vectors: {metrics['total_vectors']}")
            logger.info(f"  Dimensions: {metrics['dimensions']}")
            logger.info(f"  Insert time: {metrics['insert_time']:.2f} seconds")
            logger.info(f"  Index creation time: {metrics['index_time']:.2f} seconds")
            logger.info(f"  Table size: {metrics['table_size']/1024/1024:.2f} MB")
            logger.info(f"  Index size: {metrics['index_size']/1024/1024:.2f} MB")
        
        logger.info(f"\nMetrics saved to markdown file: {metrics_file}")
        
        # logger.info("\n=== EXAMPLE QUERIES ===")
        # logger.info("FP32 Search:  SELECT * FROM search_fp32('[0.1, 0.2, ...]'::float[], 10);")
        # logger.info("HalfVec Search: SELECT * FROM search_halfvec('[0.1, 0.2, ...]'::float[], 10);")
        # logger.info("Bit Search: SELECT * FROM search_bit('101010...'::bit, 10);")
        # logger.info("Bit Search w/Rerank: SELECT * FROM search_bit_rerank('101010...'::bit, '[0.1, 0.2, ...]'::float[], 100, 10);")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    main()
