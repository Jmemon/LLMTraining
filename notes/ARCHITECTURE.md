# Vorox: A Scalable LLM Training Architecture

## Overview

This repository implements a scalable architecture for training Large Language Models (LLMs) with the following key features:

- **Efficient Data Pipeline**: Streams large datasets from remote storage (S3, GCP, HuggingFace) with configurable prefetching and buffering
- **Modular Transformer Implementation**: Implements a decoder-only transformer with Grouped Query Attention (GQA) and configurable components
- **Robust Configuration System**: Uses Pydantic models for type-safe configuration of all training aspects
- **Production-Ready Monitoring**: Includes PostgreSQL-based metadata tracking and comprehensive logging

## Core Components

### 1. Data Pipeline (data/*)

#### Remote Streaming Dataset
- **Implementation**: `RemoteIterableDataset` in data/dataset.py
- **Key Features**:
  - Streams data from multiple remote sources (S3, GCP, HuggingFace)
  - Configurable prefetch buffer with optional shuffling
  - Integrated metadata tracking
  - Transformation pipeline support

#### Metadata Tracking
- **Implementation**: `MetadataCache` in data/metadata_cache.py
- **Features**:
  - PostgreSQL-based sample tracking
  - Per-epoch metadata storage
  - Indexing for efficient querying

### 2. Model Architecture (vorox/*)

#### Transformer Implementation
- **Core Components**:
  - Grouped Query Attention (GQA) for efficient attention computation
  - Rotary Position Embeddings (RoPE)
  - Configurable activation functions (GELU, ReLU, SiLU, SwiGLU)

#### Training Loop
- **Features**:
  - Distributed training support
  - Gradient accumulation
  - Mixed-precision training
  - Comprehensive logging

### 3. Configuration System

The system uses Pydantic for type-safe configuration management:

#### Model Configuration
- Architecture parameters (layers, dimensions, heads)
- Optimizer settings (type, learning rate, weight decay)
- Loss function configuration

#### Data Configuration
```yaml
data:
  settings:
    prefetch_size: 1000
    cache_dsn: "postgresql://user:pass@host:port/db"
    shuffle_buffer: true
  source:
    name: "dolma" # or other supported datasets
    urls:
      - "s3://bucket/train_part1.txt"
      - "s3://bucket/train_part2.txt"
```

## Workflow

1. **Configuration Initialization**
   - Load and validate YAML config using Pydantic models
   - Initialize PostgreSQL connection for metadata tracking
   - Set up logging and monitoring

2. **Data Pipeline Setup**
   - Initialize RemoteIterableDataset with configured sources
   - Set up prefetch buffer and transformation pipeline
   - Connect to metadata tracking system

3. **Model Initialization**
   - Build transformer model based on architecture config
   - Initialize optimizer and loss function
   - Move model to appropriate device (CPU/GPU/MPS)

4. **Training Loop**
   - Stream data from remote sources
   - Apply transformations and tokenization
   - Execute forward/backward passes
   - Track metadata and log progress

## Performance Considerations

- **Memory Efficiency**: Streaming architecture prevents OOM issues with large datasets
- **Throughput Optimization**: 
  - Configurable prefetch buffer size
  - Optional buffer shuffling for improved randomization
  - Grouped Query Attention for reduced memory usage
- **Monitoring**: 
  - Comprehensive logging of buffer statistics
  - Timing measurements for data loading
  - PostgreSQL-based metadata tracking for reproducibility

## Extension Points

The architecture is designed for extensibility:

1. **Data Sources**: Add new source types by extending the remote streaming module
2. **Transformations**: Implement custom data transformations
3. **Model Architecture**: Add new attention mechanisms or layer types
4. **Monitoring**: Extend metadata tracking with additional metrics

## Dependencies

- PyTorch: Core deep learning framework
- Pydantic: Configuration management and validation
- PostgreSQL: Metadata tracking
- boto3: AWS S3 integration
- smart_open: Remote file streaming
- transformers: Tokenization and model components

This architecture provides a robust foundation for training large language models while maintaining code quality, reproducibility, and scalability.' > ARCHITECTURE.md