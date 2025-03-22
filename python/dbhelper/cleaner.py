import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from python.utils.script import setup_logging, connect_to_db


def clean_database(cursor, logger):
    logger.info("Cleaning database before loading data")
    cursor.execute(
        """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_name LIKE 'embeddings%' 
    AND table_schema = 'public'
    """
    )

    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        logger.info("No existing embedding tables found")
        return
    cursor.execute("SET session_replication_role = 'replica';")
    for table in tables:
        logger.info(f"Dropping table {table}")
        cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")

    cursor.execute(
        """
    SELECT indexname 
    FROM pg_indexes 
    WHERE indexname LIKE 'embeddings%' 
    AND schemaname = 'public'
    """
    )

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
