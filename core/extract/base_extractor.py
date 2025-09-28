"""
Base Extractor

Abstract base class for data extractors.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseExtractor(ABC):
    """
    Abstract base class for data extractors.
    
    All extractors should inherit from this class and implement the extract method.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the extractor.
        
        Args:
            file_path: Path to the file to extract data from
        """
        self.file_path = file_path
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Extract data from the source.
        
        Returns:
            DataFrame with extracted data
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement extract method")
