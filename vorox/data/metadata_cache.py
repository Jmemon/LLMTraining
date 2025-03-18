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
    A PostgreSQL-backed metadata cache for tracking processed sample IDs during training epochs.
    
    Provides persistent storage and retrieval of sample processing metadata with O(1) insertion
    and O(n) retrieval complexity, enabling resumable training and dataset auditing capabilities.
    
    Architecture:
        - Implements a thin wrapper over PostgreSQL with connection pooling via psycopg2
        - Uses a simple two-column schema (epoch, sample_id) with O(log n) indexed lookups
        - Maintains a single persistent connection with autocommit semantics for write durability
        - Thread-safe for concurrent reads but not for concurrent writes to the same epoch/sample
    
    Interface:
        - add_sample(epoch: int, sample_id: int) -> None: Records a processed sample
          Parameters:
            epoch: Non-negative integer identifying the training epoch
            sample_id: Arbitrary integer uniquely identifying the sample within its source
          Raises:
            psycopg2.Error: On database connection or transaction failures
        
        - get_samples(epoch: int) -> List[int]: Retrieves all sample IDs for an epoch
          Parameters:
            epoch: Non-negative integer identifying the training epoch
          Returns:
            Ordered list of sample IDs processed during the specified epoch
          Raises:
            psycopg2.Error: On database connection or query failures
        
        - close() -> None: Releases the database connection
          Should be called when the cache is no longer needed to prevent connection leaks
    
    Behavior:
        - Connection is established at initialization and maintained until explicitly closed
        - Table and index are created automatically if they don't exist
        - Writes are immediately durable due to autocommit mode
        - Not thread-safe for concurrent writes to the same epoch/sample combination
        - Thread-safe for concurrent reads or writes to different epoch/sample combinations
    
    Integration:
        - Initialize with a PostgreSQL DSN string: cache = MetadataCache("postgresql://user:pass@host:port/db")
        - Typically integrated with data loading pipelines via mapper functions:
          ```
          def record_metadata(sample):
              cache.add_sample(current_epoch, sample_id)
              return sample
          dataset = dataset.map(record_metadata)
          ```
        - Should be used with context management or explicit close() calls to prevent connection leaks
    
    Limitations:
        - No built-in connection pooling for multi-worker scenarios; each worker should create its own instance
        - No batch insertion capability; high-frequency writes may cause performance bottlenecks
        - No automatic cleanup mechanism for old epochs; manual table maintenance required
        - Requires PostgreSQL instance with appropriate permissions and network accessibility
        - No support for complex metadata beyond epoch and sample ID; extend schema for additional needs
    """

    def __init__(self, dsn: str):
        """
        Initializes a PostgreSQL connection for metadata persistence with automatic schema creation.
        
        Establishes a persistent database connection with autocommit semantics for immediate durability,
        creates the required schema if not present, and configures the connection for metadata tracking
        with O(1) connection initialization complexity.
        
        Architecture:
            - Implements eager connection establishment pattern with O(1) time complexity
            - Automatic schema initialization with idempotent DDL operations
            - Single-connection design with connection pooling via psycopg2
            - Exception propagation with contextual logging for connection failures
            - Memory complexity: O(1) with single connection object allocation
            
        Parameters:
            dsn (str): PostgreSQL connection string in standard URI format.
                Must contain valid credentials and connection parameters.
                Format: postgresql://[user[:password]@][host][:port][/database][?param=value]
                Example: "postgresql://user:pass@localhost:5432/metadata_db"
                
        Raises:
            psycopg2.OperationalError: When connection cannot be established due to network,
                authentication, or database availability issues.
            psycopg2.ProgrammingError: When schema creation fails due to permission issues
                or invalid SQL syntax.
            Exception: For any other unexpected errors during initialization.
            
        Behavior:
            - Stateful operation establishing persistent connection
            - Not thread-safe for concurrent initialization with same DSN
            - Connection remains open until explicitly closed via close()
            - Automatic table and index creation with IF NOT EXISTS guards
            - Logs connection success at INFO level and failures at ERROR level
            
        Integration:
            - Called during cache instantiation: cache = MetadataCache(dsn_string)
            - Typically wrapped in context management for resource cleanup:
              ```
              with contextlib.closing(MetadataCache(dsn)) as cache:
                  # Use cache for metadata tracking
              ```
            - DSN typically sourced from configuration or environment variables
            
        Limitations:
            - Single connection design limits throughput for high-concurrency scenarios
            - No connection retry logic for transient database failures
            - No connection pooling for multi-process environments
            - Requires PostgreSQL instance with appropriate network accessibility
            - Schema creation requires database user with DDL privileges
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
        """
        Initializes the database schema for metadata tracking with idempotent DDL operations.
        
        Creates the core two-column table structure and supporting index for efficient epoch-based
        queries with O(1) schema initialization complexity and O(log n) query performance characteristics.
        
        Architecture:
            - Implements idempotent schema creation pattern with IF NOT EXISTS guards
            - Two-phase DDL execution with separate table and index creation
            - Minimal schema design with two non-nullable INTEGER columns
            - B-tree index on epoch column for O(log n) lookup complexity
            - Single cursor execution context with implicit transaction semantics
            
        Interface:
            - No parameters: Operates on the pre-established connection
            - No return value: Success indicated by absence of exceptions
            - Exceptions: Propagates any psycopg2 exceptions from DDL operations
            
        Behavior:
            - Stateful operation modifying database schema if not already present
            - Thread-safe for schema creation due to PostgreSQL transaction isolation
            - Idempotent: Safe to call multiple times without side effects
            - Executes within the connection's current transaction context
            - No explicit logging; relies on connection-level error propagation
            
        Integration:
            - Called automatically during MetadataCache initialization
            - Requires active database connection with DDL privileges
            - Depends on self.conn being a valid psycopg2 connection object
            
        Limitations:
            - No schema versioning or migration capabilities
            - No composite or unique constraints on (epoch, sample_id) pairs
            - No partitioning strategy for high-volume epoch data
            - Requires PostgreSQL-specific DDL syntax; not database-agnostic
            - No explicit error handling for permission or resource issues
        """
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata_cache (
                epoch INTEGER NOT NULL,
                sample_id INTEGER NOT NULL
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_epoch ON metadata_cache (epoch);")
    
    def add_sample(self, epoch: int, sample_id: int):
        """
        Records a processed sample in the metadata cache with O(1) insertion complexity.
        
        Persists the epoch and sample identifier pair to PostgreSQL using a parameterized query
        for SQL injection protection, enabling resumable training and dataset auditing with
        minimal performance overhead.
        
        Architecture:
            - Implements single-row insertion pattern with O(1) time complexity
            - Uses parameterized queries for SQL injection protection
            - Cursor-based execution with automatic resource management
            - Leverages connection's autocommit mode for immediate durability
            - Memory complexity: O(1) with fixed-size parameter tuple allocation
            
        Parameters:
            epoch (int): Non-negative integer identifying the training epoch.
                Must be a valid 32-bit signed integer within PostgreSQL's INTEGER range.
                Typically corresponds to the current training epoch number.
                
            sample_id (int): Arbitrary integer uniquely identifying the sample within its source.
                Must be a valid 32-bit signed integer within PostgreSQL's INTEGER range.
                Typically corresponds to the sample's position in the dataset or a unique identifier.
                
        Raises:
            psycopg2.Error: When insertion fails due to connection issues, constraint violations,
                or other database errors. Not explicitly caught within the method.
            psycopg2.IntegrityError: If a unique constraint exists and the (epoch, sample_id) pair
                violates it. (Note: default schema does not enforce uniqueness)
            
        Behavior:
            - Stateful operation modifying database content
            - Thread-safe for different epoch/sample_id combinations
            - Not thread-safe for identical epoch/sample_id combinations
            - Immediate durability due to connection's autocommit mode
            - No explicit transaction management; relies on autocommit
            
        Integration:
            - Typically called during dataset iteration:
              ```
              for i, sample in enumerate(dataset):
                  cache.add_sample(current_epoch, i)
                  process_sample(sample)
              ```
            - Often wrapped in try/except blocks for error handling:
              ```
              try:
                  cache.add_sample(epoch, sample_id)
              except psycopg2.Error as e:
                  logger.error(f"Failed to record sample {sample_id}: {e}")
              ```
            
        Limitations:
            - No batch insertion capability; high-frequency calls may cause performance bottlenecks
            - No deduplication logic; duplicate entries will be created if called multiple times
            - No validation of epoch or sample_id values beyond PostgreSQL INTEGER constraints
            - Requires active database connection; fails if connection is closed
            - No retry logic for transient database failures
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO metadata_cache (epoch, sample_id) VALUES (%s, %s);", 
                (epoch, sample_id)
            )
    
    def get_samples(self, epoch: int):
        """
        Retrieves all sample IDs processed during a specific epoch with O(n) retrieval complexity.
        
        Executes an indexed query against the metadata cache to fetch all sample identifiers
        for the specified epoch, returning them in ascending order for deterministic iteration
        and resumable processing workflows.
        
        Architecture:
            - Implements indexed retrieval pattern with O(log n) lookup and O(n) result complexity
            - Uses parameterized queries for SQL injection protection
            - Cursor-based execution with automatic resource management
            - List comprehension for efficient result transformation
            - Memory complexity: O(n) where n is the number of samples in the epoch
            
        Parameters:
            epoch (int): Non-negative integer identifying the training epoch to query.
                Must be a valid 32-bit signed integer within PostgreSQL's INTEGER range.
                Typically corresponds to a specific training epoch number.
                
        Returns:
            list[int]: Ordered list of sample IDs processed during the specified epoch.
                Empty list if no samples were recorded for the epoch.
                Sample IDs are returned in ascending numerical order for deterministic iteration.
                
        Raises:
            psycopg2.Error: When query fails due to connection issues, invalid parameters,
                or other database errors. Not explicitly caught within the method.
            psycopg2.ProgrammingError: When the metadata_cache table doesn't exist,
                indicating initialization failure or schema corruption.
            
        Behavior:
            - Read-only operation with no database state modification
            - Thread-safe for concurrent calls with different or same epoch values
            - Blocking operation; execution time scales with result set size
            - No result caching; repeated calls will re-query the database
            - Returns empty list rather than None when no samples exist
            
        Integration:
            - Typically used for resuming interrupted training:
              ```
              processed_samples = cache.get_samples(current_epoch)
              start_idx = len(processed_samples)
              for i, sample in enumerate(dataset, start=start_idx):
                  process_sample(sample)
              ```
            - Often used for dataset auditing and validation:
              ```
              epoch1_samples = cache.get_samples(1)
              epoch2_samples = cache.get_samples(2)
              missed_samples = set(epoch1_samples) - set(epoch2_samples)
              ```
            
        Limitations:
            - Full result set loaded into memory; may cause OOM for very large epochs
            - No pagination support for large result sets
            - No filtering capabilities beyond epoch selection
            - Performance degrades linearly with number of samples in epoch
            - Requires active database connection; fails if connection is closed
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT sample_id FROM metadata_cache WHERE epoch = %s ORDER BY sample_id;",
                (epoch,)
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]
    
    def close(self):
        """
        Releases the PostgreSQL connection resources with O(1) cleanup complexity.
        
        Terminates the persistent database connection established during initialization,
        ensuring proper resource cleanup and connection pool slot availability with
        minimal overhead and deterministic completion semantics.
        
        Architecture:
            - Implements resource release pattern with O(1) time complexity
            - Delegates to psycopg2's connection.close() for native resource cleanup
            - Explicit logging for operational visibility and debugging
            - Memory complexity: O(1) with fixed overhead regardless of usage history
            
        Interface:
            - No parameters: Operates on the instance's connection attribute
            - No return value: Success indicated by absence of exceptions
            - Exceptions: Propagates any psycopg2 exceptions from connection closure
            
        Raises:
            psycopg2.Error: When connection closure fails due to pending transactions
                or other database-specific issues. Not explicitly caught.
            
        Behavior:
            - Stateful operation modifying instance's connection state
            - Idempotent: Safe to call multiple times (subsequent calls are no-ops)
            - Thread-safe if no other methods are concurrently accessing the connection
            - Blocking operation until all server-side resources are released
            - Logs successful closure at INFO level for operational visibility
            
        Integration:
            - Typically called during application shutdown or context manager exit:
              ```
              try:
                  # Use cache for operations
              finally:
                  cache.close()  # Ensure connection is released
              ```
            - Essential for proper resource management in long-running applications:
              ```
              caches = [MetadataCache(dsn) for _ in range(worker_count)]
              try:
                  # Parallel operations using caches
              finally:
                  for cache in caches:
                      cache.close()  # Release all connections
              ```
            
        Limitations:
            - No transaction finalization; uncommitted changes may be lost
            - No connection pooling awareness; closes physical connection
            - No error recovery for connection closure failures
            - No automatic reconnection capability after closure
            - Cannot be used with async/await; synchronous blocking operation
        """
        self.conn.close()
        logger.info("Closed PostgreSQL connection for metadata cache.") 
