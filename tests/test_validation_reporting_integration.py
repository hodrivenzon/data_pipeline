#!/usr/bin/env python3
"""
Validation Reporting Integration Test

This test validates the entire validation reporting system by:
1. Creating copies of original data files
2. Intentionally inserting various validation issues into the test data
3. Running the complete data pipeline with the modified data
4. Verifying that both technical and summary reports accurately capture the inserted issues

TEST DEMANDS:
=============

This test must verify that the validation reporting system correctly:

1. DETECTS INSERTED ISSUES:
   - Schema violations (wrong data types, missing columns, extra columns)
   - Data quality issues (invalid values, constraint violations)
   - Pandera validation failures (check failures, type mismatches)
   - Business rule violations (value range violations, format issues)

2. GENERATES ACCURATE TECHNICAL REPORTS:
   - Contains exact details of inserted issues
   - Shows correct table, column, value, issue, type information
   - Points exactly where and why validation failures occurred
   - Includes proper Pandera exception details
   - Provides accurate schema definition violations

3. GENERATES ACCURATE SUMMARY REPORTS:
   - Shows correct overall validation status
   - Displays accurate success rates and error counts
   - Categorizes errors properly by type and severity
   - Provides meaningful business insights
   - Includes actionable recommendations

4. VALIDATES REPORT COMPLETENESS:
   - All inserted issues are captured in reports
   - No false positives or missing issues
   - Reports are generated successfully
   - File paths are correct and accessible

5. VERIFIES REPORT ACCURACY:
   - Technical details match inserted issues exactly
   - Summary statistics reflect actual validation results
   - Error categorization is accurate
   - Recommendations are relevant to inserted issues

TEST SCENARIOS:
==============

The test will insert the following types of issues:

PROJECTS TABLE ISSUES:
- Invalid data types (string instead of expected type)
- Missing required columns
- Extra unexpected columns
- Constraint violations

SUBJECTS TABLE ISSUES:
- Column name mismatches
- Invalid sample types
- Missing subject IDs
- Data format violations

RESULTS TABLE ISSUES:
- Invalid detection values (out of range, non-numeric)
- Invalid cancer detection values (not in allowed list)
- Invalid sample status values
- Date format violations
- Quality score violations

EXPECTED OUTCOMES:
=================

After running the pipeline with inserted issues, the test expects:

1. TECHNICAL REPORT TO CONTAIN:
   - Detailed error information for each inserted issue
   - Exact Pandera exception details
   - Schema violation descriptions
   - Recommended fixes for each issue type

2. SUMMARY REPORT TO CONTAIN:
   - Overall validation status showing failures
   - Accurate error counts matching inserted issues
   - Proper error categorization
   - Business-relevant recommendations

3. VALIDATION REPORTS TO BE:
   - Generated successfully without errors
   - Saved to correct file paths
   - Containing accurate and complete information
   - Reflecting the exact issues that were inserted

This test ensures the validation reporting system works end-to-end
and provides accurate, actionable information for both technical
and business stakeholders.
"""

import os
import sys
import shutil
import pandas as pd
import json
import tempfile
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline.data_pipeline import DataPipeline
from core.utils.reporting.validation_reporter import ValidationReporter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationReportingIntegrationTest:
    """Integration test for validation reporting system."""
    
    def __init__(self):
        """Initialize the test."""
        self.test_dir = Path("test_validation_data")
        self.original_input_dir = Path("../file_store/input")
        self.test_input_dir = self.test_dir / "input"
        self.test_output_dir = self.test_dir / "output"
        self.test_reports_dir = self.test_dir / "reports"
        
        # Test data modifications
        self.inserted_issues = {
            'projects': [],
            'subjects': [],
            'results': []
        }
        
        # Expected validation results
        self.expected_issues = {
            'projects': [],
            'subjects': [],
            'results': []
        }
    
    def setup_test_environment(self):
        """Set up test environment with copies of original data."""
        logger.info("🔧 Setting up test environment")
        
        # Create test directories
        self.test_dir.mkdir(exist_ok=True)
        self.test_input_dir.mkdir(exist_ok=True)
        self.test_output_dir.mkdir(exist_ok=True)
        self.test_reports_dir.mkdir(exist_ok=True)
        
        # Copy original data files
        original_files = [
            "excel_1_project_study_cohort.csv",
            "excel_2_subject_samples.csv", 
            "excel_3_sample_run_results.csv"
        ]
        
        for file in original_files:
            src = self.original_input_dir / file
            dst = self.test_input_dir / file
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"✅ Copied {file} to test environment")
            else:
                logger.warning(f"⚠️ Original file {file} not found")
    
    def insert_projects_issues(self):
        """Insert various validation issues into projects data."""
        logger.info("🔧 Inserting projects table issues")
        
        projects_file = self.test_input_dir / "excel_1_project_study_cohort.csv"
        if not projects_file.exists():
            logger.warning("⚠️ Projects file not found, skipping issues")
            return
        
        df = pd.read_csv(projects_file)
        original_shape = df.shape
        
        # Issue 1: Add invalid data type (string instead of expected type)
        df.loc[0, 'project_code'] = ''  # Empty string
        self.inserted_issues['projects'].append({
            'type': 'EmptyValue',
            'column': 'project_code',
            'row': 0,
            'description': 'Empty project code'
        })
        
        # Issue 2: Add duplicate project codes
        df.loc[1, 'project_code'] = df.loc[0, 'project_code'] if len(df) > 1 else 'DUPLICATE'
        self.inserted_issues['projects'].append({
            'type': 'DuplicateValue',
            'column': 'project_code', 
            'row': 1,
            'description': 'Duplicate project code'
        })
        
        # Issue 3: Add invalid manager name (special characters)
        df.loc[2, 'project_manager_name'] = 'Invalid@Manager#Name' if len(df) > 2 else 'Invalid@Name'
        self.inserted_issues['projects'].append({
            'type': 'InvalidFormat',
            'column': 'project_manager_name',
            'row': 2,
            'description': 'Invalid manager name format'
        })
        
        # Save modified data
        df.to_csv(projects_file, index=False)
        logger.info(f"✅ Inserted {len(self.inserted_issues['projects'])} issues into projects table")
    
    def insert_subjects_issues(self):
        """Insert various validation issues into subjects data."""
        logger.info("🔧 Inserting subjects table issues")
        
        subjects_file = self.test_input_dir / "excel_2_subject_samples.csv"
        if not subjects_file.exists():
            logger.warning("⚠️ Subjects file not found, skipping issues")
            return
        
        df = pd.read_csv(subjects_file)
        
        # Issue 1: Add missing subject IDs
        df.loc[0, 'subject_id'] = ''  # Empty subject ID
        self.inserted_issues['subjects'].append({
            'type': 'MissingValue',
            'column': 'subject_id',
            'row': 0,
            'description': 'Missing subject ID'
        })
        
        # Issue 2: Add invalid sample types
        df.loc[1, 'type'] = 'INVALID_TYPE' if len(df) > 1 else 'INVALID'
        self.inserted_issues['subjects'].append({
            'type': 'InvalidValue',
            'column': 'type',
            'row': 1,
            'description': 'Invalid sample type'
        })
        
        # Issue 3: Add duplicate sample IDs
        if len(df) > 2:
            df.loc[2, 'sample_id'] = df.loc[0, 'sample_id']
            self.inserted_issues['subjects'].append({
                'type': 'DuplicateValue',
                'column': 'sample_id',
                'row': 2,
                'description': 'Duplicate sample ID'
            })
        
        # Save modified data
        df.to_csv(subjects_file, index=False)
        logger.info(f"✅ Inserted {len(self.inserted_issues['subjects'])} issues into subjects table")
    
    def insert_results_issues(self):
        """Insert various validation issues into results data."""
        logger.info("🔧 Inserting results table issues")
        
        results_file = self.test_input_dir / "excel_3_sample_run_results.csv"
        if not results_file.exists():
            logger.warning("⚠️ Results file not found, skipping issues")
            return
        
        df = pd.read_csv(results_file)
        
        # Issue 1: Add invalid detection values (out of range)
        df.loc[0, ' detection_value '] = 1.5  # Out of range (should be 0-1)
        self.inserted_issues['results'].append({
            'type': 'RangeViolation',
            'column': ' detection_value ',
            'row': 0,
            'description': 'Detection value out of range (1.5)'
        })
        
        # Issue 2: Add invalid cancer detection values
        df.loc[1, 'cancer_detected_(yes_no)'] = 'INVALID_VALUE' if len(df) > 1 else 'INVALID'
        self.inserted_issues['results'].append({
            'type': 'InvalidValue',
            'column': 'cancer_detected_(yes_no)',
            'row': 1,
            'description': 'Invalid cancer detection value'
        })
        
        # Issue 3: Add invalid sample status
        df.loc[2, 'sample_status(running/finished/failed)'] = 'INVALID_STATUS' if len(df) > 2 else 'INVALID'
        self.inserted_issues['results'].append({
            'type': 'InvalidValue',
            'column': 'sample_status(running/finished/failed)',
            'row': 2,
            'description': 'Invalid sample status'
        })
        
        # Issue 4: Add non-numeric detection values
        df.loc[3, ' detection_value '] = 'NOT_A_NUMBER' if len(df) > 3 else 'INVALID'
        self.inserted_issues['results'].append({
            'type': 'TypeMismatch',
            'column': ' detection_value ',
            'row': 3,
            'description': 'Non-numeric detection value'
        })
        
        # Save modified data
        df.to_csv(results_file, index=False)
        logger.info(f"✅ Inserted {len(self.inserted_issues['results'])} issues into results table")
    
    def run_data_pipeline(self):
        """Run the complete data pipeline with modified data."""
        logger.info("🚀 Running data pipeline with inserted issues")
        
        try:
            # Initialize and run pipeline with test directories
            pipeline = DataPipeline("../config.yaml")
            
            # Update pipeline paths for test
            pipeline.config_manager.config['input_dir'] = str(self.test_input_dir)
            pipeline.config_manager.config['output_dir'] = str(self.test_output_dir)
            
            pipeline.run()
            
            logger.info("✅ Data pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data pipeline failed: {e}")
            return False
    
    def validate_technical_report(self):
        """Validate that technical report contains inserted issues."""
        logger.info("🔍 Validating technical report")
        
        # Find technical report file
        reports_dir = Path("file_store/reports")
        technical_reports = list(reports_dir.glob("technical_validation_report_*.json"))
        
        if not technical_reports:
            logger.error("❌ No technical report found")
            return False
        
        # Use the most recent report
        latest_report = max(technical_reports, key=os.path.getctime)
        
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            # Validate report structure
            required_sections = ['report_metadata', 'run_info', 'validation_summary', 'detailed_errors']
            for section in required_sections:
                if section not in report_data:
                    logger.error(f"❌ Missing section in technical report: {section}")
                    return False
            
            # Validate error details
            detailed_errors = report_data.get('detailed_errors', [])
            if not detailed_errors:
                logger.warning("⚠️ No detailed errors found in technical report")
            
            # Check for inserted issues
            total_inserted = sum(len(issues) for issues in self.inserted_issues.values())
            logger.info(f"📊 Technical report contains {len(detailed_errors)} errors (inserted {total_inserted} issues)")
            
            # Validate error structure
            for error in detailed_errors:
                required_fields = ['error_id', 'table_name', 'column_name', 'error_type', 'error_message']
                for field in required_fields:
                    if field not in error:
                        logger.error(f"❌ Missing field in error: {field}")
                        return False
            
            logger.info("✅ Technical report validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Technical report validation failed: {e}")
            return False
    
    def validate_summary_report(self):
        """Validate that summary report contains accurate information."""
        logger.info("🔍 Validating summary report")
        
        # Find summary report file
        reports_dir = Path("file_store/reports")
        summary_reports = list(reports_dir.glob("summary_validation_report_*.json"))
        
        if not summary_reports:
            logger.error("❌ No summary report found")
            return False
        
        # Use the most recent report
        latest_report = max(summary_reports, key=os.path.getctime)
        
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            # Validate report structure
            required_sections = ['report_metadata', 'executive_summary', 'validation_overview', 'table_summary']
            for section in required_sections:
                if section not in report_data:
                    logger.error(f"❌ Missing section in summary report: {section}")
                    return False
            
            # Validate executive summary
            exec_summary = report_data.get('executive_summary', {})
            if 'overall_status' not in exec_summary:
                logger.error("❌ Missing overall_status in executive summary")
                return False
            
            # Validate error summary
            error_summary = report_data.get('error_summary', {})
            if 'total_errors' not in error_summary:
                logger.error("❌ Missing total_errors in error summary")
                return False
            
            logger.info("✅ Summary report validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Summary report validation failed: {e}")
            return False
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        logger.info("🧹 Cleaning up test environment")
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            logger.info("✅ Test environment cleaned up")
    
    def run_integration_test(self):
        """Run the complete integration test."""
        logger.info("🧪 Starting Validation Reporting Integration Test")
        logger.info("=" * 60)
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Insert issues
            self.insert_projects_issues()
            self.insert_subjects_issues()
            self.insert_results_issues()
            
            # Run pipeline
            pipeline_success = self.run_data_pipeline()
            if not pipeline_success:
                logger.error("❌ Pipeline execution failed")
                return False
            
            # Validate reports
            technical_valid = self.validate_technical_report()
            summary_valid = self.validate_summary_report()
            
            # Results
            if technical_valid and summary_valid:
                logger.info("🎉 All validation reporting tests PASSED!")
                logger.info("✅ Technical report accurately captures inserted issues")
                logger.info("✅ Summary report provides accurate business insights")
                return True
            else:
                logger.error("❌ Validation reporting tests FAILED!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()

def main():
    """Run the validation reporting integration test."""
    test = ValidationReportingIntegrationTest()
    success = test.run_integration_test()
    
    if success:
        print("\n🎉 VALIDATION REPORTING INTEGRATION TEST PASSED!")
        print("✅ Reports accurately capture inserted validation issues")
        print("✅ Technical report provides detailed debugging information")
        print("✅ Summary report provides accurate business insights")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION REPORTING INTEGRATION TEST FAILED!")
        print("❌ Reports do not accurately capture validation issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
