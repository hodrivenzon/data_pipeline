"""
Summary Loader

Loader for saving summary DataFrames to CSV format.
"""

import pandas as pd
import logging
from pathlib import Path
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class SummaryLoader(BaseLoader):
    """
    Loader for saving summary DataFrames to CSV format.
    
    Handles CSV file creation with proper error handling and logging.
    """
    
    def load(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to CSV format.
        
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
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"✅ Successfully saved CSV file: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save CSV file {output_path}: {e}")
            raise
