"""
Validation Reporter

Main orchestrator for validation reporting system.
Handles the collection and generation of both technical and summary reports.

REPORTING SYSTEM LOGIC:
======================

This validation reporter implements a dual-report system designed to serve
different stakeholder needs while maintaining a single source of truth for
validation data. The system logic is based on the following principles:

1. DUAL REPORT ARCHITECTURE:
   - TECHNICAL REPORT: Comprehensive, developer-focused analysis
     * Contains complete Pandera schema exception details
     * Provides comprehensive table, column, value, issue, type information
     * Points exactly where and why validation failures occurred
     * Includes complete schema definition violations and constraint failures
     * Designed for deep debugging and comprehensive analysis

   - SUMMARY REPORT: Practical, error-focused actionable information
     * Focuses on specific error details: table, column, value, failure reason
     * Provides actionable recommendations for immediate fixing
     * Includes error severity, priority, and recommended actions
     * Designed for immediate error resolution and data cleaning

2. DATA COLLECTION STRATEGY:
   - Captures validation events throughout the pipeline
   - Records both successes and failures with context
   - Maintains detailed error information for technical analysis
   - Aggregates data for practical error resolution

3. REPORT GENERATION LOGIC:
   - Technical reports: Maximum detail for comprehensive debugging
   - Summary reports: Practical error details for immediate fixing
   - Both reports use the same underlying validation data
   - Reports are generated independently to serve different needs

4. INTEGRATION WITH VALIDATION PIPELINE:
   - Seamlessly integrates with existing validation flow
   - Captures pre-validation and post-validation results
   - Tracks cleaning effectiveness and improvement metrics
   - Provides comprehensive validation lifecycle reporting

TARGET USERS: Technical Teams (Technical Report) + Data Engineers (Summary Report)
USE CASES: Deep debugging + Immediate error resolution and data cleaning
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .data_quality_report_generator import DataQualityReportGenerator

from .technical_report_generator import TechnicalReportGenerator
from .summary_report_generator import SummaryReportGenerator
from .report_cleaner import ReportCleaner

logger = logging.getLogger(__name__)

class ValidationReporter:
    """
    Main validation reporter that orchestrates the generation of both
    technical and summary validation reports.
    """
    
    def __init__(self, output_dir: str = "file_store/reports"):
        """
        Initialize the validation reporter.
        
        Args:
            output_dir: Directory to save validation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.technical_generator = TechnicalReportGenerator()
        self.summary_generator = SummaryReportGenerator()
        self.report_cleaner = ReportCleaner(str(self.output_dir))
        
        self.validation_data = {
            'run_info': {
                'start_time': None,
                'end_time': None,
                'total_tables': 0,
                'total_validations': 0
            },
            'tables': [],
            'errors': [],
            'successes': []
        }
    
    def start_validation_run(self):
        """Start a new validation run and record start time."""
        self.validation_data['run_info']['start_time'] = datetime.now().isoformat()
        logger.info("🔍 Starting validation run")
    
    def end_validation_run(self):
        """End the validation run and record end time."""
        self.validation_data['run_info']['end_time'] = datetime.now().isoformat()
        logger.info("✅ Validation run completed")
    
    def add_table_validation(self, table_name: str, validation_result: Dict[str, Any]):
        """
        Add validation result for a specific table.
        
        Args:
            table_name: Name of the table being validated
            validation_result: Dictionary containing validation details
        """
        table_data = {
            'table_name': table_name,
            'timestamp': datetime.now().isoformat(),
            'validation_result': validation_result
        }
        
        self.validation_data['tables'].append(table_data)
        self.validation_data['run_info']['total_tables'] += 1
        
        logger.debug(f"📊 Added validation result for table: {table_name}")
    
    def add_validation_error(self, error_data: Dict[str, Any]):
        """
        Add a validation error to the collection.
        
        Args:
            error_data: Dictionary containing error details
        """
        error_data['timestamp'] = datetime.now().isoformat()
        self.validation_data['errors'].append(error_data)
        
        logger.debug(f"❌ Added validation error: {error_data.get('error_type', 'Unknown')}")
    
    def add_validation_success(self, success_data: Dict[str, Any]):
        """
        Add a validation success to the collection.
        
        Args:
            success_data: Dictionary containing success details
        """
        success_data['timestamp'] = datetime.now().isoformat()
        self.validation_data['successes'].append(success_data)
        
        logger.debug(f"✅ Added validation success: {success_data.get('table', 'Unknown')}")
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate both technical and summary reports.
        
        LOGIC: This method orchestrates the generation of both report types
        using the same underlying validation data. The dual-report approach
        serves different stakeholder needs:
        
        1. TECHNICAL REPORT: Generated for developers and technical teams
           - Contains detailed Pandera schema exception information
           - Provides exact table, column, value, issue, type details
           - Points exactly where and why validation failures occurred
           - Includes schema definition violations and constraint failures
        
        2. SUMMARY REPORT: Generated for business stakeholders and managers
           - Provides overall picture without overwhelming detail
           - Focuses on success rates, error categories, and trends
           - Includes actionable recommendations and next steps
           - Designed for executive reporting and business decisions
        
        Both reports use the same validation data but present it in formats
        optimized for their respective audiences.
        
        Returns:
            Dictionary with paths to generated reports
        """
        logger.info("📝 Generating validation reports")
        
        # Generate technical report
        technical_report_path = self.technical_generator.generate_report(
            self.validation_data, 
            self.output_dir
        )
        
        # Generate summary report
        summary_report_path = self.summary_generator.generate_report(
            self.validation_data,
            self.output_dir
        )
        
        # Clean up old reports to prevent file overflow
        cleanup_stats = self.report_cleaner.clean_old_reports()
        if cleanup_stats["cleaned"] > 0:
            logger.info(f"🧹 Cleaned up {cleanup_stats['cleaned']} old reports, kept {cleanup_stats['kept']} latest reports")
        
        return {
            'technical_report': str(technical_report_path),
            'summary_report': str(summary_report_path)
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a quick summary of the validation run.
        
        Returns:
            Dictionary with validation summary statistics
        """
        total_errors = len(self.validation_data['errors'])
        total_successes = len(self.validation_data['successes'])
        total_tables = len(self.validation_data['tables'])
        
        return {
            'total_tables': total_tables,
            'total_errors': total_errors,
            'total_successes': total_successes,
            'success_rate': (total_successes / (total_successes + total_errors) * 100) if (total_successes + total_errors) > 0 else 0,
            'run_duration': self._calculate_run_duration()
        }
    
    def _calculate_run_duration(self) -> Optional[str]:
        """Calculate the duration of the validation run."""
        start_time = self.validation_data['run_info'].get('start_time')
        end_time = self.validation_data['run_info'].get('end_time')
        
        if start_time and end_time:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            return str(duration)
        
        return None
    
    def print_summary(self):
        """Print a quick summary of the validation run to console."""
        summary = self.get_validation_summary()
        
        print("\n" + "="*60)
        print("📊 VALIDATION RUN SUMMARY")
        print("="*60)
        print(f"📋 Tables Validated: {summary['total_tables']}")
        print(f"✅ Successful Validations: {summary['total_successes']}")
        print(f"❌ Validation Errors: {summary['total_errors']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        if summary['run_duration']:
            print(f"⏱️  Run Duration: {summary['run_duration']}")
        print("="*60)
    
    def generate_data_quality_report(self, config: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive data quality and transformation report.
        
        Args:
            config: Configuration dictionary for transformation data
            
        Returns:
            Path to generated data quality report
        """
        try:
            report_generator = DataQualityReportGenerator()
            report_path = report_generator.generate_report(config)
            logger.info(f"📊 Data quality report generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"❌ Error generating data quality report: {e}")
            raise
