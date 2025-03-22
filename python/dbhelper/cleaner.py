import psycopg2
import os
from dotenv import load_dotenv
import logging

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

def clean_database(cursor, logger):
    """Drop all existing tables and indexes to allow clean reloading of data"""
    logger.info("Cleaning database before loading data")
    
        
    cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_name LIKE 'embeddings%' 
    AND table_schema = 'public'
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    
    if not tables:
        logger.info("No existing embedding tables found")
        return
    
    cursor.execute("SET session_replication_role = 'replica';")
    
    for table in tables:
        logger.info(f"Dropping table {table}")
        cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
    
    cursor.execute("""
    SELECT indexname 
    FROM pg_indexes 
    WHERE indexname LIKE 'embeddings%' 
    AND schemaname = 'public'
    """)
    
    indexes = [row[0] for row in cursor.fetchall()]
    
    for index in indexes:
        logger.info(f"Dropping index {index}")
        cursor.execute(f"DROP INDEX IF EXISTS {index};")
    
   
    
    cursor.execute("SET session_replication_role = 'origin';")
    
    cursor.connection.commit()
    logger.info("Database cleaning completed")

def main():
    logger = setup_logging()
    conn = connect_to_db()
    cursor = conn.cursor()
    clean_database(cursor, logger)
    conn.close()

if __name__ == "__main__":
    main()

