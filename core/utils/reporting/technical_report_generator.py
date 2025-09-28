"""
Technical Report Generator

Generates comprehensive technical validation reports with detailed
Pandera schema exception information for deep debugging and analysis.

REPORT LOGIC AND PURPOSE:
========================

This technical report is designed for developers, data engineers, and technical
stakeholders who need comprehensive technical details for debugging and fixing
validation issues. The report focuses on providing maximum technical detail:

1. COMPREHENSIVE PANDERA EXCEPTIONS:
   - Complete Pandera exception details with full stack traces
   - Schema definition violations with exact column specifications
   - Data type mismatches with detailed type information
   - Check failures with complete validation context

2. DETAILED TECHNICAL ANALYSIS:
   - Complete table, column, and value-level error information
   - Full actual vs expected values for all failed validations
   - Complete schema rule violations with check specifications
   - Exact data locations and row-level error details

3. COMPREHENSIVE DEBUGGING INFORMATION:
   - Complete schema definitions that were violated
   - Full validation rule specifications that failed
   - Complete Pandera exception details for debugging
   - Detailed technical analysis and recommendations

4. COMPLETE ERROR TRACKING:
   - Full error categorization by type, severity, and impact
   - Complete column-level validation status
   - Comprehensive data quality metrics and scoring
   - Complete table-level analysis with all technical details

The report provides maximum technical detail for comprehensive debugging
and analysis of validation failures.

TARGET AUDIENCE: Developers, Data Engineers, Technical Teams
USE CASE: Deep debugging, comprehensive analysis, technical issue resolution
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TechnicalReportGenerator:
    """
    Generates detailed technical validation reports with comprehensive
    information about validation failures, schema violations, and data quality issues.
    """
    
    def __init__(self):
        """Initialize the technical report generator."""
        self.report_data = {}
    
    def generate_report(self, validation_data: Dict[str, Any], output_dir: Path) -> Path:
        """
        Generate a detailed technical validation report.
        
        Args:
            validation_data: Raw validation data from the validation process
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated technical report
        """
        logger.info("🔧 Generating technical validation report")
        
        # Process validation data into technical report format
        technical_report = self._process_validation_data(validation_data)
        
        # Create technical reports subdirectory
        technical_dir = output_dir / "technical"
        technical_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"technical_validation_report_{timestamp}.json"
        report_path = technical_dir / filename
        
        # Save the report
        with open(report_path, 'w') as f:
            json.dump(technical_report, f, indent=2, default=str)
        
        logger.info(f"✅ Technical report saved: {report_path}")
        return report_path
    
    def _process_validation_data(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw validation data into technical report format.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Processed technical report data
        """
        technical_report = {
            'report_metadata': {
                'report_type': 'technical_validation_report',
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0.0',
                'focus': 'pandera_schema_exceptions_and_data_quality_issues'
            },
            'run_info': validation_data.get('run_info', {}),
            'validation_summary': self._generate_validation_summary(validation_data),
            'detailed_errors': self._process_validation_errors(validation_data),
            'schema_violations': self._process_schema_violations(validation_data),
            'data_quality_issues': self._process_data_quality_issues(validation_data),
            'table_analysis': self._process_table_analysis(validation_data)
        }
        
        return technical_report
    
    def _generate_validation_summary(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the validation run."""
        total_errors = len(validation_data.get('errors', []))
        total_successes = len(validation_data.get('successes', []))
        total_tables = len(validation_data.get('tables', []))
        
        return {
            'total_tables_validated': total_tables,
            'total_validation_errors': total_errors,
            'total_validation_successes': total_successes,
            'overall_success_rate': (total_successes / (total_successes + total_errors) * 100) if (total_successes + total_errors) > 0 else 0,
            'error_categories': self._categorize_errors(validation_data.get('errors', []))
        }
    
    def _process_validation_errors(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process validation errors with detailed technical information.
        
        LOGIC: This method extracts and structures all validation errors to provide
        maximum technical detail for debugging. Each error is enriched with:
        - Exact Pandera exception details
        - Schema definition context
        - Actual vs expected values
        - Validation rule information
        - Recommended technical fixes
        
        This approach ensures developers have all the information needed to:
        1. Understand exactly what validation rule failed
        2. See the actual data that caused the failure
        3. Know which schema definition was violated
        4. Get specific recommendations for fixing the issue
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of detailed error information with technical debugging context
        """
        detailed_errors = []
        
        for error in validation_data.get('errors', []):
            detailed_error = {
                'error_id': f"ERR_{len(detailed_errors) + 1:04d}",
                'timestamp': error.get('timestamp'),
                'table_name': error.get('table_name', 'Unknown'),
                'column_name': error.get('column_name', 'Unknown'),
                'error_type': error.get('error_type', 'Unknown'),
                'error_message': error.get('error_message', 'No message provided'),
                'schema_definition': error.get('schema_definition', {}),
                'actual_value': error.get('actual_value'),
                'expected_value': error.get('expected_value'),
                'validation_rule': error.get('validation_rule', {}),
                'pandera_exception': error.get('pandera_exception', {}),
                'data_type_issues': error.get('data_type_issues', {}),
                'constraint_violations': error.get('constraint_violations', []),
                'recommended_fix': error.get('recommended_fix', 'Manual review required')
            }
            
            detailed_errors.append(detailed_error)
        
        return detailed_errors
    
    def _process_schema_violations(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process schema violations with detailed information.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of schema violation details
        """
        schema_violations = []
        
        for table_data in validation_data.get('tables', []):
            table_name = table_data.get('table_name', 'Unknown')
            validation_result = table_data.get('validation_result', {})
            
            if not validation_result.get('is_valid', True):
                violations = {
                    'table_name': table_name,
                    'timestamp': table_data.get('timestamp'),
                    'violation_count': validation_result.get('error_count', 0),
                    'schema_definition': validation_result.get('schema_definition', {}),
                    'column_violations': validation_result.get('column_violations', []),
                    'data_type_violations': validation_result.get('data_type_violations', []),
                    'constraint_violations': validation_result.get('constraint_violations', []),
                    'missing_columns': validation_result.get('missing_columns', []),
                    'extra_columns': validation_result.get('extra_columns', [])
                }
                
                schema_violations.append(violations)
        
        return schema_violations
    
    def _process_data_quality_issues(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process data quality issues with detailed information.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of data quality issue details
        """
        data_quality_issues = []
        
        for error in validation_data.get('errors', []):
            if error.get('error_type') in ['DataQualityError', 'ValidationError']:
                issue = {
                    'issue_id': f"DQ_{len(data_quality_issues) + 1:04d}",
                    'table_name': error.get('table_name', 'Unknown'),
                    'column_name': error.get('column_name', 'Unknown'),
                    'issue_type': error.get('error_type', 'Unknown'),
                    'severity': error.get('severity', 'Medium'),
                    'description': error.get('error_message', 'No description provided'),
                    'affected_rows': error.get('affected_rows', 0),
                    'data_samples': error.get('data_samples', []),
                    'quality_metrics': error.get('quality_metrics', {}),
                    'recommended_action': error.get('recommended_action', 'Manual review required')
                }
                
                data_quality_issues.append(issue)
        
        return data_quality_issues
    
    def _process_table_analysis(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process table-level analysis with detailed information.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of table analysis details
        """
        table_analysis = []
        
        for table_data in validation_data.get('tables', []):
            table_name = table_data.get('table_name', 'Unknown')
            validation_result = table_data.get('validation_result', {})
            
            analysis = {
                'table_name': table_name,
                'timestamp': table_data.get('timestamp'),
                'validation_status': 'PASSED' if validation_result.get('is_valid', True) else 'FAILED',
                'total_rows': validation_result.get('total_rows', 0),
                'valid_rows': validation_result.get('valid_rows', 0),
                'invalid_rows': validation_result.get('invalid_rows', 0),
                'data_quality_score': validation_result.get('data_quality_score', 0),
                'column_analysis': validation_result.get('column_analysis', []),
                'schema_compliance': validation_result.get('schema_compliance', {}),
                'recommendations': validation_result.get('recommendations', [])
            }
            
            table_analysis.append(analysis)
        
        return table_analysis
    
    def _categorize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize errors by type for summary statistics.
        
        Args:
            errors: List of error dictionaries
            
        Returns:
            Dictionary with error counts by category
        """
        categories = {}
        
        for error in errors:
            error_type = error.get('error_type', 'Unknown')
            categories[error_type] = categories.get(error_type, 0) + 1
        
        return categories
