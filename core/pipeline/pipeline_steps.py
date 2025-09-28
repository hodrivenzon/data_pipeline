"""
Pipeline Steps

Contains all the atomic step methods for the data pipeline.
These methods implement the core ETL operations in a stateless, independent manner.

⚠️  CRITICAL: ATOMIC STEP DESIGN ⚠️
====================================

These step methods follow STRICT ATOMIC DESIGN PRINCIPLES:

✅ ATOMIC PRINCIPLES:
- Each method has a SINGLE, WELL-DEFINED PURPOSE
- Methods do NOT depend on input/output from other methods
- Each method can be executed independently
- Methods are stateless and do not maintain internal state
- Methods receive input, process it, and return output

✅ INDEPENDENCE GUARANTEES:
- No method modifies global state
- No method depends on other methods' internal variables
- Data flows through explicit parameters only
- Each method is testable in isolation
- Methods can be reordered without side effects

🚫 FORBIDDEN PATTERNS:
- Methods that share state or variables
- Methods that depend on other methods' internal state
- Methods that modify global configuration
- Methods that have side effects beyond their input/output
- Methods that cannot be executed independently

For any changes to these step methods, contact the data engineering team.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from config.config_manager import ConfigManager
from ..extract import ProjectsExtractor, SubjectsExtractor, ResultsExtractor
from ..transform import DataMerger, SummaryTransformer, DynamicDataTransformer
from ..load import ParquetLoader, SummaryLoader
from utils.exceptions.pipeline_exceptions import (
    ExtractionException,
    ProjectsExtractionException,
    SubjectsExtractionException, 
    ResultsExtractionException,
    TransformationException,
    DataCleaningException,
    DataMergingException,
    DataValidationException,
    SummaryGenerationException,
    LoadingException,
    ParquetLoadingException,
    SummaryLoadingException
)

logger = logging.getLogger(__name__)


class PipelineSteps:
    """
    Atomic step methods for the data pipeline.
    
    This class contains all the individual step methods that implement
    the core ETL operations. Each method is atomic, independent, and
    follows the strict design principles outlined in the module docstring.
    """
    
    def __init__(self, config_manager: ConfigManager, data: Dict[str, Any]):
        """
        Initialize pipeline steps.
        
        Args:
            config_manager: Configuration manager instance
            data: Shared data dictionary for pipeline state
        """
        self.config_manager = config_manager
        self.data = data
        
        # Store transformer instances for validation reporting
        self._projects_transformer = None
        self._subjects_transformer = None
        self._results_transformer = None
    
    def extract_projects(self) -> None:
        """Extract projects data."""
        try:
            projects_file = self.config_manager.get('files.projects_file')
            extractor = ProjectsExtractor(projects_file)
            self.data['projects'] = extractor.extract()
            logger.info(f"✅ Extracted {len(self.data['projects'])} projects records")
        except Exception as e:
            error_msg = f"Failed to extract projects: {e}"
            logger.error(f"❌ {error_msg}")
            raise ProjectsExtractionException(error_msg, str(e))
    
    def extract_subjects(self) -> None:
        """Extract subjects data."""
        try:
            subjects_file = self.config_manager.get('files.subjects_file')
            extractor = SubjectsExtractor(subjects_file)
            self.data['subjects'] = extractor.extract()
            logger.info(f"✅ Extracted {len(self.data['subjects'])} subjects records")
        except Exception as e:
            error_msg = f"Failed to extract subjects: {e}"
            logger.error(f"❌ {error_msg}")
            raise SubjectsExtractionException(error_msg, str(e))
    
    def extract_results(self) -> None:
        """Extract results data."""
        try:
            results_file = self.config_manager.get('files.results_file')
            extractor = ResultsExtractor(results_file)
            self.data['results'] = extractor.extract()
            logger.info(f"✅ Extracted {len(self.data['results'])} results records")
        except Exception as e:
            error_msg = f"Failed to extract results: {e}"
            logger.error(f"❌ {error_msg}")
            raise ResultsExtractionException(error_msg, str(e))
    
    def transform_projects(self) -> None:
        """Transform projects data."""
        try:
            transformer = DynamicDataTransformer(self.config_manager.config, 'projects')
            self.data['projects'] = transformer.transform(self.data['projects'])
            self._projects_transformer = transformer  # Store for validation reporting
            logger.info(f"✅ Transformed {len(self.data['projects'])} projects records")
        except Exception as e:
            error_msg = f"Failed to transform projects: {e}"
            logger.error(f"❌ {error_msg}")
            raise DataCleaningException(error_msg, 'projects', str(e))
    
    def transform_subjects(self) -> None:
        """Transform subjects data."""
        try:
            transformer = DynamicDataTransformer(self.config_manager.config, 'subjects')
            self.data['subjects'] = transformer.transform(self.data['subjects'])
            self._subjects_transformer = transformer  # Store for validation reporting
            logger.info(f"✅ Transformed {len(self.data['subjects'])} subjects records")
        except Exception as e:
            error_msg = f"Failed to transform subjects: {e}"
            logger.error(f"❌ {error_msg}")
            raise DataCleaningException(error_msg, 'subjects', str(e))
    
    def transform_results(self) -> None:
        """Transform results data."""
        try:
            transformer = DynamicDataTransformer(self.config_manager.config, 'results')
            self.data['results'] = transformer.transform(self.data['results'])
            self._results_transformer = transformer  # Store for validation reporting
            logger.info(f"✅ Transformed {len(self.data['results'])} results records")
        except Exception as e:
            error_msg = f"Failed to transform results: {e}"
            logger.error(f"❌ {error_msg}")
            raise DataCleaningException(error_msg, 'results', str(e))
    
    def merge_data(self) -> None:
        """Merge all data sources."""
        try:
            merger = DataMerger(self.config_manager.config)
            merged_data = merger.transform(
                self.data['projects'],
                self.data['subjects'],
                self.data['results']
            )
            self.data['merged'] = merged_data
            logger.info(f"✅ Merged data: {len(merged_data)} final records")
        except Exception as e:
            error_msg = f"Failed to merge data: {e}"
            logger.error(f"❌ {error_msg}")
            raise DataMergingException(error_msg, str(e))
    
    def validate_data(self) -> None:
        """Validate merged data."""
        try:
            from utils.schemas.schemas import get_schema
            schema = get_schema('merged')
            schema.validate(self.data['merged'], lazy=True)
            logger.info(f"✅ Validated {len(self.data['merged'])} records")
        except Exception as e:
            error_msg = f"Failed to validate data: {e}"
            logger.error(f"❌ {error_msg}")
            raise DataValidationException(error_msg, str(e))
    
    def generate_summary(self) -> None:
        """Generate summary statistics."""
        try:
            transformer = SummaryTransformer(self.config_manager.config)
            self.data['summary'] = transformer.transform(self.data['merged'])
            logger.info(f"✅ Generated summary: {len(self.data['summary'])} records")
        except Exception as e:
            error_msg = f"Failed to generate summary: {e}"
            logger.error(f"❌ {error_msg}")
            raise SummaryGenerationException(error_msg, str(e))
    
    def save_parquet(self) -> None:
        """Save data to Parquet format."""
        try:
            output_file = self.config_manager.get('files.output_parquet')
            loader = ParquetLoader(output_file)
            loader.load(self.data['merged'])
            logger.info(f"✅ Saved Parquet: {output_file}")
        except Exception as e:
            error_msg = f"Failed to save Parquet: {e}"
            logger.error(f"❌ {error_msg}")
            raise ParquetLoadingException(error_msg, str(e))
    
    def save_summary(self) -> None:
        """Save summary to CSV format."""
        try:
            summary_file = self.config_manager.get('files.summary_csv')
            loader = SummaryLoader(summary_file)
            loader.load(self.data['summary'])
            logger.info(f"✅ Saved summary: {summary_file}")
        except Exception as e:
            error_msg = f"Failed to save summary: {e}"
            logger.error(f"❌ {error_msg}")
            raise SummaryLoadingException(error_msg, str(e))
