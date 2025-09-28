"""
Data Pipeline

Main pipeline orchestration for the data processing workflow.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from config.config_manager import ConfigManager
from .pipeline_steps import PipelineSteps

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main data pipeline for processing CSV files into Parquet and summary outputs.
    
    The pipeline follows ETL pattern:
    1. Extract: Read CSV files 
    2. Transform: Clean, merge, and validate data
    3. Load: Save processed data to Parquet and CSV
    
    ⚠️  CRITICAL DESIGN PRINCIPLES ⚠️
    ================================================
    
    This pipeline implements a STRICT ATOMIC STEP ARCHITECTURE that is CRUCIAL
    for maintaining data integrity and pipeline reliability. The design principles
    below are FUNDAMENTAL and MUST NOT be changed by AI agents or developers
    without explicit approval from the data engineering team.
    
    🔒 STEP ORDER IS CRITICAL:
    =========================
    
    The `run()` method executes steps in a SPECIFIC, PREDETERMINED ORDER that
    has been carefully designed to ensure data quality and processing integrity.
    This order is NOT arbitrary and reflects the logical dependencies between
    data processing stages:
    
    1. EXTRACT → TRANSFORM → LOAD (ETL Pattern)
    2. Within TRANSFORM: Clean → Merge → Validate → Summarize
    3. Each step builds upon the previous step's output
    
    🚫 DO NOT CHANGE STEP ORDER:
    - The sequence is optimized for data quality
    - Changing order can introduce data corruption
    - Dependencies between steps are intentional
    - The order reflects business logic requirements
    
    🔒 ATOMIC STEP DESIGN:
    =====================
    
    Each step in the pipeline is ATOMIC and INDEPENDENT:
    
    ✅ ATOMIC PRINCIPLES:
    - Each step has a SINGLE, WELL-DEFINED PURPOSE
    - Steps do NOT depend on input/output from other steps
    - Each step can be executed independently
    - Steps are stateless and do not maintain internal state
    - Steps receive input, process it, and return output
    
    ✅ INDEPENDENCE GUARANTEES:
    - No step modifies global state
    - No step depends on previous step's internal variables
    - Data flows through explicit parameters only
    - Each step is testable in isolation
    - Steps can be reordered without side effects
    
    🚫 FORBIDDEN PATTERNS:
    - Steps that share state or variables
    - Steps that depend on previous step's internal state
    - Steps that modify global configuration
    - Steps that have side effects beyond their input/output
    - Steps that cannot be executed independently
    
    🔒 PIPELINE ORCHESTRATION:
    ==========================
    
    The `run()` method serves as the ORCHESTRATOR that:
    - Coordinates step execution in the correct order
    - Manages data flow between steps
    - Handles error propagation and logging
    - Ensures atomic step execution
    - Maintains pipeline state and metadata
    
    The orchestrator is responsible for:
    - Step sequencing and coordination
    - Data passing between steps
    - Error handling and recovery
    - Logging and monitoring
    - Pipeline state management
    
    🚫 DO NOT MODIFY:
    - Step execution order in `run()` method
    - Atomic step design principles
    - Data flow patterns between steps
    - Error handling and logging structure
    - Pipeline orchestration logic
    
    For any changes to the pipeline architecture, contact the data engineering team.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data pipeline.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.to_dict()
        
        # Setup logging from configuration
        self._setup_logging()
        
        # Initialize data storage
        self.data = {
            'projects': None,
            'subjects': None,
            'results': None,
            'merged': None,
            'summary': None
        }
        
        # Initialize pipeline steps
        self.steps = PipelineSteps(self.config_manager, self.data)
        
        logger.info("🚀 Data Pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = self.config_manager.get('logging.log_level', 'INFO')
        log_format = self.config_manager.get('logging.log_format', '%(asctime)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            force=True
        )
    
    def run(self) -> bool:
        """
        Run the complete data pipeline.
        
        🔒 CRITICAL: STEP ORDER IS FIXED AND MUST NOT BE CHANGED 🔒
        ============================================================
        
        This method orchestrates the pipeline execution in a SPECIFIC ORDER
        that has been carefully designed for data quality and integrity.
        
        The order reflects:
        1. Logical dependencies between processing stages
        2. Data quality requirements
        3. Business logic constraints
        4. Error handling and recovery patterns
        
        🚫 DO NOT MODIFY THE STEP ORDER:
        - Changing order can introduce data corruption
        - Dependencies between steps are intentional
        - The sequence is optimized for data quality
        - Each step builds upon the previous step's output
        
        Returns:
            True if pipeline succeeds, False otherwise
        """
        try:
            logger.info("🔄 Starting data pipeline execution")
            
            # Extract phase - MUST BE FIRST
            self.steps.extract_projects()
            self.steps.extract_subjects() 
            self.steps.extract_results()
            
            # Check data quality immediately after extraction - fail if too much data is corrupted
            if not self._check_data_quality():
                logger.error("❌ Data quality check failed - too much corrupted data")
                return False
            
            # Transform phase - MUST FOLLOW EXTRACT
            self.steps.transform_projects()
            self.steps.transform_subjects()
            self.steps.transform_results()
            self.steps.merge_data()
            self.steps.validate_data()
            self.steps.generate_summary()
            
            # Load phase - MUST FOLLOW TRANSFORM
            self.steps.save_parquet()
            self.steps.save_summary()
            
            # Generate validation report - MUST BE LAST
            self.generate_validation_report()
            
            logger.info("✅ Pipeline execution completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return False
    
    def _check_data_quality(self) -> bool:
        """
        Check data quality and fail if too much data is corrupted.
        
        Returns:
            True if data quality is acceptable, False if too much corruption
        """
        try:
            # Check for excessive null values in critical columns
            if self.data['projects'] is not None:
                # Check for null values in critical columns specifically
                critical_cols = ['project_code', 'study_code', 'study_cohort_code']
                for col in critical_cols:
                    if col in self.data['projects'].columns:
                        null_count = self.data['projects'][col].isnull().sum()
                        if null_count > 0:  # Any null values in critical columns
                            logger.warning(f"⚠️ Null values found in critical column '{col}': {null_count} nulls")
                            return False
            
            if self.data['subjects'] is not None:
                null_ratio = self.data['subjects'].isnull().sum().sum() / (len(self.data['subjects']) * len(self.data['subjects'].columns))
                if null_ratio > 0.2:  # More than 20% null values
                    logger.warning(f"⚠️ High null ratio in subjects data: {null_ratio:.2%}")
                    return False
            
            if self.data['results'] is not None:
                null_ratio = self.data['results'].isnull().sum().sum() / (len(self.data['results']) * len(self.data['results'].columns))
                if null_ratio > 0.2:  # More than 20% null values
                    logger.warning(f"⚠️ High null ratio in results data: {null_ratio:.2%}")
                    return False
            
            # Check for missing critical columns
            if self.data['projects'] is not None:
                required_cols = ['project_code', 'study_code', 'study_cohort_code']
                missing_cols = [col for col in required_cols if col not in self.data['projects'].columns]
                if missing_cols:
                    logger.warning(f"⚠️ Missing critical columns in projects: {missing_cols}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data quality check failed: {e}")
            return False
    
    def generate_validation_report(self) -> str:
        """
        Generate and save validation reports.
        
        Returns:
            Path to the saved validation report
        """
        try:
            logger.info("📊 Generating validation reports")
            
            # Generate reports from the last transformer that ran
            if hasattr(self.steps, '_results_transformer') and self.steps._results_transformer:
                logger.info(f"📊 Generating validation reports from results transformer")
                results_reporter = self.steps._results_transformer.get_validation_reporter()
                report_paths = results_reporter.generate_reports()
                
                # Generate comprehensive data quality report
                data_quality_report_path = results_reporter.generate_data_quality_report(
                    self.config_manager.config
                )
                logger.info(f"📊 Data quality report generated: {data_quality_report_path}")
                
                logger.info(f"✅ Validation reports generated: {report_paths}")
                return report_paths.get('technical_report', '')
            else:
                logger.warning("⚠️ No transformer instances found for validation reporting")
                return ""
            
        except Exception as e:
            logger.error(f"❌ Failed to generate validation report: {e}")
            return ""
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of processed data.
        
        Returns:
            Dictionary with data statistics
        """
        summary = {}
        for key, df in self.data.items():
            if df is not None:
                summary[key] = len(df)
            else:
                summary[key] = 0
        return summary
