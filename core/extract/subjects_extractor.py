"""
Subjects Extractor

Extractor for subjects data from CSV files.
"""

import pandas as pd
import logging
from pathlib import Path
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class SubjectsExtractor(BaseExtractor):
    """
    Extractor for subjects data.
    
    Reads subjects data from CSV files and handles various encoding issues.
    """
    
    def extract(self) -> pd.DataFrame:
        """
        Extract subjects data from CSV file.
        
        Returns:
            DataFrame with subjects data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file can't be read
        """
        file_path = Path(self.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Subjects file not found: {file_path}")
        
        logger.info(f"📥 Extracting subjects data from {file_path}")
        
        try:
            # Try UTF-8 encoding first
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                low_memory=False
            )
            
            logger.info(f"✅ Extracted {len(df)} subjects records")
            return df
            
        except UnicodeDecodeError:
            # Try with latin-1 encoding if UTF-8 fails
            try:
                df = pd.read_csv(
                    file_path,
                    encoding='latin-1',
                    low_memory=False
                )
                logger.info(f"✅ Extracted {len(df)} subjects records (latin-1 encoding)")
                return df
            except Exception as e:
                logger.error(f"❌ Failed to read {file_path} with multiple encodings: {e}")
                raise
        
        except Exception as e:
            logger.error(f"❌ Failed to extract subjects from {file_path}: {e}")
            raise
