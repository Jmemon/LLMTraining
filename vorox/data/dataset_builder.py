from typing import List
from datasets import IterableDataset
from vorox.configs import RunConfig as Config, DatasetType
from vorox.data.dclm_baseline import dclm_baseline

class DatasetBuilder:
    """
    Factory class for building datasets based on configuration.
    
    Provides a unified interface for instantiating different types of datasets
    based on configuration parameters, abstracting implementation details from client code.
    
    Architecture:
        - Implements the Factory Method pattern with O(1) lookup complexity for dataset types
        - Stateless class design with no instance variables, operating purely through class methods
        - Maps configuration enums directly to dataset implementations with minimal overhead
    """
    
    @staticmethod
    def build(config: Config) -> List[IterableDataset]:
        """
        Builds a list of iterable datasets based on the provided configuration.
        
        Parameters:
            config (Config): Configuration object containing dataset specifications
            
        Returns:
            List[IterableDataset]: List of iterable datasets ready for training
            
        Raises:
            ValueError: If an unknown dataset type is specified
        """
        datasets = []
        
        if not config.data.train_data:
            return datasets
            
        for dataset_type in config.data.train_data:
            if dataset_type == DatasetType.dclm_baseline:
                datasets.append(dclm_baseline())
            elif dataset_type == DatasetType.thestack:
                # TODO: Implement thestack dataset
                pass
            elif dataset_type == DatasetType.dolma:
                # TODO: Implement dolma dataset
                pass
            elif dataset_type == DatasetType.redpajama:
                # TODO: Implement redpajama dataset
                pass
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
                
        return datasets
