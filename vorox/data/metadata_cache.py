"""
Module: metadata_cache.py
Purpose: Provides a metadata caching mechanism using PostgreSQL to record processed sample IDs for each epoch.
"""

import psycopg2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MetadataCache:
    """
    A metadata cache implementation using PostgreSQL.
    
    Records sample IDs for each epoch in the 'metadata_cache' table.
    """

    def __init__(self, dsn: str):
        """
        Initialize the MetadataCache with a PostgreSQL DSN.
        
        Parameters:
            dsn (str): PostgreSQL DSN string (e.g., postgresql://user:pass@host:port/db)
        """
        self.dsn = dsn
        try:
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = True
            self._init_table()
            logger.info("Connected to PostgreSQL metadata cache.")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata_cache (
                epoch INTEGER NOT NULL,
                sample_id INTEGER NOT NULL
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_epoch ON metadata_cache (epoch);")
    
    def add_sample(self, epoch: int, sample_id: int):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO metadata_cache (epoch, sample_id) VALUES (%s, %s);", 
                (epoch, sample_id)
            )
    
    def get_samples(self, epoch: int):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT sample_id FROM metadata_cache WHERE epoch = %s ORDER BY sample_id;",
                (epoch,)
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]
    
    def close(self):
        self.conn.close()
        logger.info("Closed PostgreSQL connection for metadata cache.") 