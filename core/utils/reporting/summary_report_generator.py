"""
Summary Report Generator

Generates practical, error-focused validation reports that provide
actionable information about data quality issues for immediate resolution.

REPORT LOGIC AND PURPOSE:
=========================

This summary report is designed for data engineers, developers, and technical
stakeholders who need to quickly identify and fix validation issues. The report
focuses on providing PRACTICAL ERROR DETAILS for immediate action:

1. ERROR-FOCUSED CONTENT:
   - Detailed error information: table, column, value, failure reason
   - Specific data values that caused validation failures
   - Exact locations of problematic data
   - Clear error descriptions for immediate fixing

2. PRACTICAL ERROR DETAILS:
   - Table names where errors occurred
   - Column names with validation failures
   - Actual values that failed validation
   - Specific validation rules that were violated
   - Row numbers or data locations for quick identification

3. ACTIONABLE ERROR INFORMATION:
   - Error categorization by severity and type
   - Priority ranking of issues to fix
   - Specific recommendations for each error type
   - Quick reference for data cleaning tasks

4. MINIMAL SUCCESS OVERVIEW:
   - Brief summary of successful validations
   - Overall data quality status
   - Success rate for context only

The report prioritizes ERROR DETAILS over general overviews, providing
the specific information needed to fix data quality issues immediately.

TARGET AUDIENCE: Data Engineers, Developers, Technical Teams
USE CASE: Error resolution, data cleaning, immediate issue fixing
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SummaryReportGenerator:
    """
    Generates human-readable summary validation reports that provide
    a clear overview of validation results and data quality status.
    """
    
    def __init__(self):
        """Initialize the summary report generator."""
        self.report_data = {}
    
    def generate_report(self, validation_data: Dict[str, Any], output_dir: Path) -> Path:
        """
        Generate a human-readable summary validation report.
        
        Args:
            validation_data: Raw validation data from the validation process
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated summary report
        """
        logger.info("📋 Generating summary validation report")
        
        # Process validation data into summary report format
        summary_report = self._process_validation_data(validation_data)
        
        # Create summary reports subdirectory
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_validation_report_{timestamp}.json"
        report_path = summary_dir / filename
        
        # Save the report
        with open(report_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        logger.info(f"✅ Summary report saved: {report_path}")
        return report_path
    
    def _process_validation_data(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw validation data into summary report format.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Processed summary report data
        """
        summary_report = {
            'report_metadata': {
                'report_type': 'summary_validation_report',
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0.0',
                'focus': 'practical_error_details_for_immediate_resolution'
            },
            'error_details': self._generate_detailed_errors(validation_data),
            'error_summary': self._generate_error_summary(validation_data),
            'quick_overview': self._generate_quick_overview(validation_data),
            'actionable_recommendations': self._generate_actionable_recommendations(validation_data)
        }
        
        return summary_report
    
    def _generate_detailed_errors(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate detailed error information for immediate fixing.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of detailed error information
        """
        detailed_errors = []
        errors = validation_data.get('errors', [])
        
        for error in errors:
            error_detail = {
                'table_name': error.get('table_name', 'Unknown'),
                'column_name': error.get('column_name', 'Unknown'),
                'error_type': error.get('error_type', 'Unknown'),
                'error_message': error.get('error_message', 'No message'),
                'failed_value': error.get('actual_value', 'Not specified'),
                'expected_value': error.get('expected_value', 'Not specified'),
                'validation_rule': error.get('validation_rule', 'Not specified'),
                'row_location': error.get('row_index', 'Not specified'),
                'severity': self._determine_error_severity(error),
                'fix_priority': self._determine_fix_priority(error),
                'recommended_action': self._get_recommended_action(error)
            }
            detailed_errors.append(error_detail)
        
        return detailed_errors
    
    def _generate_quick_overview(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a quick overview with minimal success information.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Quick overview dictionary
        """
        errors = validation_data.get('errors', [])
        successes = validation_data.get('successes', [])
        
        total_errors = len(errors)
        total_successes = len(successes)
        total_validations = total_errors + total_successes
        
        success_rate = (total_successes / total_validations * 100) if total_validations > 0 else 0.0
        
        return {
            'total_errors': total_errors,
            'total_successes': total_successes,
            'success_rate_percentage': round(success_rate, 1),
            'overall_status': 'PASSED' if total_errors == 0 else 'FAILED',
            'data_quality_status': 'EXCELLENT' if success_rate >= 95 else 'GOOD' if success_rate >= 80 else 'POOR' if success_rate >= 50 else 'CRITICAL'
        }
    
    def _generate_actionable_recommendations(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations for fixing errors.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        errors = validation_data.get('errors', [])
        
        # Group errors by type for targeted recommendations
        error_types = {}
        for error in errors:
            error_type = error.get('error_type', 'Unknown')
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        for error_type, error_list in error_types.items():
            recommendation = {
                'error_type': error_type,
                'count': len(error_list),
                'affected_tables': list(set([e.get('table_name', 'Unknown') for e in error_list])),
                'priority': self._get_error_type_priority(error_type),
                'recommended_fix': self._get_error_type_fix(error_type),
                'estimated_effort': self._get_estimated_effort(error_type, len(error_list))
            }
            recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        return recommendations
    
    def _determine_error_severity(self, error: Dict[str, Any]) -> str:
        """Determine error severity level."""
        error_type = error.get('error_type', '')
        if 'MissingColumn' in error_type or 'SchemaError' in error_type:
            return 'CRITICAL'
        elif 'TypeMismatch' in error_type or 'ValueError' in error_type:
            return 'HIGH'
        elif 'FormatError' in error_type or 'ConstraintError' in error_type:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _determine_fix_priority(self, error: Dict[str, Any]) -> int:
        """Determine fix priority (1-5, 5 being highest)."""
        severity = self._determine_error_severity(error)
        if severity == 'CRITICAL':
            return 5
        elif severity == 'HIGH':
            return 4
        elif severity == 'MEDIUM':
            return 3
        else:
            return 2
    
    def _get_recommended_action(self, error: Dict[str, Any]) -> str:
        """Get recommended action for fixing the error."""
        error_type = error.get('error_type', '')
        if 'MissingColumn' in error_type:
            return 'Add missing column to data source or update schema'
        elif 'TypeMismatch' in error_type:
            return 'Convert data type or clean invalid values'
        elif 'ValueError' in error_type:
            return 'Clean or replace invalid values'
        elif 'ConstraintError' in error_type:
            return 'Fix constraint violations in data'
        else:
            return 'Review and fix data quality issues'
    
    def _get_error_type_priority(self, error_type: str) -> int:
        """Get priority for error type."""
        if 'MissingColumn' in error_type:
            return 5
        elif 'TypeMismatch' in error_type:
            return 4
        elif 'ValueError' in error_type:
            return 3
        else:
            return 2
    
    def _get_error_type_fix(self, error_type: str) -> str:
        """Get fix recommendation for error type."""
        if 'MissingColumn' in error_type:
            return 'Add missing columns to data source'
        elif 'TypeMismatch' in error_type:
            return 'Implement data type conversion and cleaning'
        elif 'ValueError' in error_type:
            return 'Clean invalid values and implement validation'
        else:
            return 'Review data quality and implement appropriate fixes'
    
    def _get_estimated_effort(self, error_type: str, count: int) -> str:
        """Get estimated effort for fixing errors."""
        if 'MissingColumn' in error_type:
            return 'High - requires schema changes'
        elif 'TypeMismatch' in error_type:
            return f'Medium - {count} type conversions needed'
        elif 'ValueError' in error_type:
            return f'Medium - {count} values to clean'
        else:
            return 'Low to Medium - depends on specific issues'
    
    def _generate_executive_summary(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an executive summary of the validation run.
        
        LOGIC: This method creates a high-level business summary that provides
        executives and managers with the essential information they need to
        understand validation status without technical details. The summary includes:
        
        1. OVERALL STATUS: Simple PASSED/FAILED/WARNING classification
        2. KEY METRICS: Success rates, error counts, data quality scores
        3. BUSINESS IMPACT: Assessment of data quality status
        4. KEY FINDINGS: Critical issues that need attention
        
        This approach ensures business stakeholders can quickly understand:
        - Whether the validation was successful overall
        - How many issues were found and their severity
        - What the data quality status means for the business
        - What critical issues need immediate attention
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Executive summary dictionary with business-focused insights
        """
        total_errors = len(validation_data.get('errors', []))
        total_successes = len(validation_data.get('successes', []))
        total_tables = len(validation_data.get('tables', []))
        
        success_rate = (total_successes / (total_successes + total_errors) * 100) if (total_successes + total_errors) > 0 else 0
        
        return {
            'overall_status': 'PASSED' if success_rate >= 90 else 'FAILED' if success_rate < 70 else 'WARNING',
            'total_tables_validated': total_tables,
            'total_errors': total_errors,
            'total_successes': total_successes,
            'success_rate_percentage': round(success_rate, 1),
            'data_quality_status': self._assess_data_quality_status(success_rate),
            'key_findings': self._extract_key_findings(validation_data)
        }
    
    def _generate_validation_overview(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a validation overview section.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Validation overview dictionary
        """
        run_info = validation_data.get('run_info', {})
        
        return {
            'validation_run_id': f"VR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': run_info.get('start_time'),
            'end_time': run_info.get('end_time'),
            'duration': self._calculate_duration(run_info.get('start_time'), run_info.get('end_time')),
            'total_validations': run_info.get('total_validations', 0),
            'validation_scope': 'Full dataset validation',
            'validation_method': 'Pandera schema validation'
        }
    
    def _generate_table_summary(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a summary for each table.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of table summary dictionaries
        """
        table_summaries = []
        
        for table_data in validation_data.get('tables', []):
            table_name = table_data.get('table_name', 'Unknown')
            validation_result = table_data.get('validation_result', {})
            
            summary = {
                'table_name': table_name,
                'status': 'PASSED' if validation_result.get('is_valid', True) else 'FAILED',
                'total_rows': validation_result.get('total_rows', 0),
                'valid_rows': validation_result.get('valid_rows', 0),
                'invalid_rows': validation_result.get('invalid_rows', 0),
                'data_quality_score': validation_result.get('data_quality_score', 0),
                'primary_issues': validation_result.get('primary_issues', []),
                'recommendations': validation_result.get('recommendations', [])
            }
            
            table_summaries.append(summary)
        
        return table_summaries
    
    def _generate_error_summary(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of validation errors.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Error summary dictionary
        """
        errors = validation_data.get('errors', [])
        
        if not errors:
            return {
                'total_errors': 0,
                'error_categories': {},
                'top_error_types': [],
                'most_problematic_tables': [],
                'error_trends': {}
            }
        
        # Categorize errors
        error_categories = {}
        table_errors = {}
        
        for error in errors:
            error_type = error.get('error_type', 'Unknown')
            table_name = error.get('table_name', 'Unknown')
            
            error_categories[error_type] = error_categories.get(error_type, 0) + 1
            table_errors[table_name] = table_errors.get(table_name, 0) + 1
        
        # Get top error types
        top_error_types = sorted(error_categories.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get most problematic tables
        most_problematic_tables = sorted(table_errors.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_errors': len(errors),
            'error_categories': error_categories,
            'top_error_types': [{'error_type': error_type, 'count': count} for error_type, count in top_error_types],
            'most_problematic_tables': [{'table_name': table_name, 'error_count': count} for table_name, count in most_problematic_tables],
            'error_trends': self._analyze_error_trends(errors)
        }
    
    def _generate_success_summary(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of validation successes.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            Success summary dictionary
        """
        successes = validation_data.get('successes', [])
        
        return {
            'total_successes': len(successes),
            'successful_tables': [success.get('table_name', 'Unknown') for success in successes],
            'success_categories': self._categorize_successes(successes),
            'quality_highlights': self._extract_quality_highlights(successes)
        }
    
    def _generate_recommendations(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on validation results.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Analyze errors to generate recommendations
        errors = validation_data.get('errors', [])
        
        if errors:
            # High priority recommendations
            if any(error.get('error_type') == 'SchemaViolation' for error in errors):
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Schema Compliance',
                    'recommendation': 'Review and update data to match schema requirements',
                    'impact': 'Critical for data quality'
                })
            
            # Medium priority recommendations
            if any(error.get('error_type') == 'DataQualityError' for error in errors):
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Data Quality',
                    'recommendation': 'Implement data cleaning processes',
                    'impact': 'Improve overall data quality'
                })
        
        return recommendations
    
    def _generate_next_steps(self, validation_data: Dict[str, Any]) -> List[str]:
        """
        Generate next steps based on validation results.
        
        Args:
            validation_data: Raw validation data
            
        Returns:
            List of next step strings
        """
        next_steps = []
        
        total_errors = len(validation_data.get('errors', []))
        total_successes = len(validation_data.get('successes', []))
        
        if total_errors > 0:
            next_steps.append("Review technical report for detailed error analysis")
            next_steps.append("Implement data cleaning processes for identified issues")
            next_steps.append("Re-run validation after data cleaning")
        
        if total_successes > 0:
            next_steps.append("Monitor data quality metrics for successful tables")
            next_steps.append("Implement automated validation checks")
        
        return next_steps
    
    def _assess_data_quality_status(self, success_rate: float) -> str:
        """Assess overall data quality status based on success rate."""
        if success_rate >= 95:
            return "EXCELLENT"
        elif success_rate >= 90:
            return "GOOD"
        elif success_rate >= 80:
            return "FAIR"
        elif success_rate >= 70:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _extract_key_findings(self, validation_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from validation data."""
        findings = []
        
        total_errors = len(validation_data.get('errors', []))
        total_successes = len(validation_data.get('successes', []))
        
        if total_errors == 0:
            findings.append("All tables passed validation successfully")
        elif total_successes == 0:
            findings.append("All tables failed validation - immediate attention required")
        else:
            findings.append(f"Mixed validation results: {total_successes} successes, {total_errors} errors")
        
        return findings
    
    def _calculate_duration(self, start_time: Optional[str], end_time: Optional[str]) -> Optional[str]:
        """Calculate duration between start and end times."""
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(start_time)
                end = datetime.fromisoformat(end_time)
                duration = end - start
                return str(duration)
            except ValueError:
                return None
        return None
    
    def _analyze_error_trends(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error trends and patterns."""
        if not errors:
            return {}
        
        # Analyze error distribution over time
        error_timestamps = [error.get('timestamp') for error in errors if error.get('timestamp')]
        
        return {
            'error_frequency': len(errors),
            'time_span': 'Analysis of error distribution over time',
            'patterns': 'Error pattern analysis would go here'
        }
    
    def _categorize_successes(self, successes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize successes by type."""
        categories = {}
        
        for success in successes:
            success_type = success.get('success_type', 'Unknown')
            categories[success_type] = categories.get(success_type, 0) + 1
        
        return categories
    
    def _extract_quality_highlights(self, successes: List[Dict[str, Any]]) -> List[str]:
        """Extract quality highlights from successful validations."""
        highlights = []
        
        for success in successes:
            if success.get('data_quality_score', 0) >= 95:
                highlights.append(f"Excellent data quality in {success.get('table_name', 'Unknown')} table")
        
        return highlights
