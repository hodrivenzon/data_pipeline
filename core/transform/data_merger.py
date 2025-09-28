"""
Data Merger

Component for merging multiple DataFrames.
"""

import pandas as pd
import logging
from typing import Dict, Any
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


class DataMerger(BaseTransformer):
    """
    Data merger for combining multiple DataFrames.
    
    Merges projects, subjects, and results data into a single DataFrame
    based on common key columns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data merger.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
    
    def transform(self, projects_df: pd.DataFrame, subjects_df: pd.DataFrame, 
                 results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge projects, subjects, and results data.
        
        Args:
            projects_df: Projects DataFrame
            subjects_df: Subjects DataFrame
            results_df: Results DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("🔄 Merging data from all sources...")
        
        # Step 1: Merge subjects with results on sample_id
        subjects_results = pd.merge(
            subjects_df,
            results_df,
            on='sample_id',
            how='inner',
            suffixes=('', '_results')
        )
        
        logger.debug(f"Subjects-Results merge: {len(subjects_results)} records")
        
        # Step 2: Merge with projects on project/study/cohort codes
        merge_keys = ['project_code', 'study_code', 'study_cohort_code']
        
        # Ensure merge keys exist in both DataFrames
        projects_keys = [key for key in merge_keys if key in projects_df.columns]
        subjects_keys = [key for key in merge_keys if key in subjects_results.columns]
        
        if not projects_keys or not subjects_keys:
            logger.warning("⚠️ Missing merge keys, performing cartesian product")
            merged_df = pd.concat([subjects_results, projects_df], ignore_index=True)
        else:
            common_keys = list(set(projects_keys) & set(subjects_keys))
            
            # Remove duplicates from projects_df before merge
            projects_df_clean = projects_df.drop_duplicates(subset=common_keys, keep='first')
            
            merged_df = pd.merge(
                subjects_results,
                projects_df_clean,
                on=common_keys,
                how='inner',  # Use inner join to only include valid relationships
                suffixes=('', '_projects')
            )
        
        # Step 3: Select essential columns for final output
        essential_columns = [
            'project_code', 'study_code', 'study_cohort_code',
            'project_name', 'study_name', 'study_cohort_name',
            'project_manager_name', 'disease_name',
            'subject_id', 'sample_id', 'type',
            'detection_value', 'cancer_detected', 'sample_status',
            'sample_quality', 'sample_quality_threshold', 'fail_reason', 'date_of_run'
        ]
        
        # Filter to only existing essential columns
        existing_essential = [col for col in essential_columns if col in merged_df.columns]
        if existing_essential:
            merged_df = merged_df[existing_essential]
        
        logger.info(f"✅ Merged data: {len(merged_df)} final records")
        return merged_df
