"""
Base Loader

Abstract base class for data loaders.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.
    
    All loaders should inherit from this class and implement the load method.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize the loader.
        
        Args:
            output_path: Path where to save the data
        """
        self.output_path = output_path
    
    @abstractmethod
    def load(self, df: pd.DataFrame) -> None:
        """
        Load data to the target destination.
        
        Args:
            df: DataFrame to load
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement load method")
