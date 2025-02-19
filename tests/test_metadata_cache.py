"""
Tests for metadata_cache.py using PostgreSQL.
Ensure you have a running Postgres instance and set the TEST_PG_DSN environment variable
or use the default below (which must point to a valid test database).
"""

import os
from data.metadata_cache import MetadataCache

def clear_test_data(cache, epochs):
    with cache.conn.cursor() as cur:
        cur.execute("DELETE FROM metadata_cache WHERE epoch = ANY(%s);", (epochs,))

def test_metadata_cache():
    # Use a DSN from the environment or default to a test database.
    test_dsn = os.getenv("TEST_PG_DSN", "postgresql://postgres:postgres@localhost:5432/test_db")
    cache = MetadataCache(test_dsn)
    
    # Clear any existing test data for epochs 1 and 2.
    clear_test_data(cache, [1, 2])
    
    cache.add_sample(1, 100)
    cache.add_sample(1, 101)
    cache.add_sample(2, 200)
    
    samples_epoch1 = cache.get_samples(1)
    samples_epoch2 = cache.get_samples(2)
    
    assert samples_epoch1 == [100, 101], f"Expected [100, 101] but got {samples_epoch1}"
    assert samples_epoch2 == [200], f"Expected [200] but got {samples_epoch2}"
    
    cache.close()

if __name__ == "__main__":
    test_metadata_cache()
    print("test_metadata_cache passed!") 