"""
Base Transformer

Abstract base class for data transformers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.
    
    All transformers should inherit from this class and implement the transform method.
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
    
    @abstractmethod
    def transform(self, *args, **kwargs) -> pd.DataFrame:
        """
        Transform the input data.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            DataFrame with transformed data
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement transform method")
