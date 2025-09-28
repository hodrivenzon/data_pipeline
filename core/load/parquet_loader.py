"""
Parquet Loader

Loader for saving DataFrames to Parquet format.
"""

import pandas as pd
import logging
from pathlib import Path
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ParquetLoader(BaseLoader):
    """
    Loader for saving DataFrames to Parquet format.
    
    Handles Parquet file creation with proper error handling and logging.
    """
    
    def load(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to Parquet format.
        
        Args:
            df: DataFrame to save
            
        Raises:
            Exception: If saving fails
        """
        output_path = Path(self.output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving {len(df)} records to {output_path}")
        
        try:
            df.to_parquet(output_path, index=False, engine='pyarrow')
            logger.info(f"✅ Successfully saved Parquet file: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save Parquet file {output_path}: {e}")
            raise
