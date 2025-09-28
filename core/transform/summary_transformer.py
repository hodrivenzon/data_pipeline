"""
Summary Transformer

Component for generating summary statistics from merged data.
"""

import pandas as pd
import logging
from typing import Dict, Any
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


class SummaryTransformer(BaseTransformer):
    """
    Summary transformer for generating aggregate statistics.
    
    Creates summary statistics per project, study, and cohort including:
    - Number of samples detected
    - Percentage of finished samples
    - Lowest detection value
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the summary transformer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics from merged data.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Summary DataFrame
        """
        logger.info("📊 Generating summary statistics...")
        
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return pd.DataFrame()
        
        # Group by project, study, cohort
        groupby_cols = ['project_code', 'study_code', 'study_cohort_code']
        
        # Filter to only existing groupby columns
        existing_groupby_cols = [col for col in groupby_cols if col in df.columns]
        
        if not existing_groupby_cols:
            from utils.exceptions.pipeline_exceptions import SummaryGenerationException
            raise SummaryGenerationException("No groupby columns found in DataFrame")
        
        # Check for required columns
        required_cols = ['project_code', 'study_code', 'study_cohort_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            from utils.exceptions.pipeline_exceptions import SummaryGenerationException
            raise SummaryGenerationException(f"Missing required columns: {missing_cols}")
        
        summary_data = []
        
        for group_keys, group_df in df.groupby(existing_groupby_cols):
            # Ensure group_keys is always a tuple
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            
            # Calculate metrics
            samples_detected = len(group_df)
            
            # Calculate percentage of finished samples
            finished_percentage = self._calculate_finished_percentage(group_df)
            
            # Calculate lowest detection value
            lowest_detection = self._calculate_lowest_detection(group_df)
            
            # Create Code column by combining project-study-cohort
            code_parts = []
            for i, col in enumerate(existing_groupby_cols):
                if i < len(group_keys):
                    code_parts.append(str(group_keys[i]))
                else:
                    code_parts.append('')
            code = '-'.join(code_parts)
            
            # Create summary record with correct column names
            summary_record = {
                'Code': code,
                'Total_Samples': samples_detected,
                'Finished_Percentage': finished_percentage,
                'Lowest_Detection': lowest_detection
            }
            
            # Add groupby columns to summary record for reference
            for i, col in enumerate(existing_groupby_cols):
                if i < len(group_keys):
                    summary_record[col] = group_keys[i]
                else:
                    summary_record[col] = None
            
            summary_data.append(summary_record)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Reorder columns to put groupby columns first
        column_order = existing_groupby_cols + [col for col in summary_df.columns if col not in existing_groupby_cols]
        summary_df = summary_df[column_order]
        
        logger.info(f"✅ Generated summary: {len(summary_df)} records")
        return summary_df
    
    def _calculate_finished_percentage(self, group_df: pd.DataFrame) -> float:
        """
        Calculate percentage of finished samples in the group.
        
        Args:
            group_df: Group DataFrame
            
        Returns:
            Percentage of finished samples
        """
        if 'sample_status' not in group_df.columns:
            return 0.0
        
        total_samples = len(group_df)
        if total_samples == 0:
            return 0.0
        
        # Count samples with finished status
        finished_samples = group_df['sample_status'].str.contains(
            'Finished|Completed', case=False, na=False
        ).sum()
        
        return round((finished_samples / total_samples) * 100, 2)
    
    def _calculate_lowest_detection(self, group_df: pd.DataFrame) -> float:
        """
        Calculate lowest detection value in the group.
        
        Args:
            group_df: Group DataFrame
            
        Returns:
            Lowest detection value or None if no valid values
        """
        if 'detection_value' not in group_df.columns:
            return None
        
        # Convert to numeric and get valid values
        detection_values = pd.to_numeric(group_df['detection_value'], errors='coerce')
        valid_values = detection_values.dropna()
        
        if len(valid_values) == 0:
            return None
        
        return float(valid_values.min())
