"""
Data Validation Tests

This test suite focuses on:
- Schema validation correctness
- Data type enforcement
- Range and constraint validation
- Enum value validation
- Null value handling
- Data quality enforcement
"""

import unittest
import pandas as pd
import tempfile
import shutil
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.schemas.schemas import get_schema
from pandera.errors import SchemaError, SchemaErrors


class TestSchemaValidation(unittest.TestCase):
    """Test schema validation functionality."""
    
    def test_projects_schema_validation(self):
        """Test projects schema validation."""
        schema = get_schema('projects')
        
        # Valid data
        valid_data = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B'],
            'study_name': ['Study A', 'Study B'],
            'study_cohort_name': ['Cohort A', 'Cohort B'],
            'project_manager_name': ['Manager A', 'Manager B'],
            'disease_name': ['Disease A', 'Disease B']
        })
        
        # Should pass validation
        try:
            schema.validate(valid_data, lazy=True)
            self.assertTrue(True, "Valid data should pass validation")
        except Exception as e:
            self.fail(f"Valid data should not fail validation: {e}")
    
    def test_projects_schema_missing_columns(self):
        """Test projects schema with missing columns."""
        schema = get_schema('projects')
        
        # Missing required columns
        invalid_data = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002']
            # Missing other required columns
        })
        
        # Should fail validation
        with self.assertRaises((SchemaError, SchemaErrors)):
            schema.validate(invalid_data, lazy=True)
    
    def test_projects_schema_wrong_types(self):
        """Test projects schema with wrong data types."""
        schema = get_schema('projects')
        
        # Wrong data types
        invalid_data = pd.DataFrame({
            'project_code': [1, 2],  # Should be string
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B'],
            'study_name': ['Study A', 'Study B'],
            'study_cohort_name': ['Cohort A', 'Cohort B'],
            'project_manager_name': ['Manager A', 'Manager B'],
            'disease_name': ['Disease A', 'Disease B']
        })
        
        # Should fail validation
        with self.assertRaises((SchemaError, SchemaErrors)):
            schema.validate(invalid_data, lazy=True)
    
    def test_subjects_schema_validation(self):
        """Test subjects schema validation."""
        schema = get_schema('subjects')
        
        # Valid data
        valid_data = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'subject_id': ['SUBJ001', 'SUBJ002'],
            'sample_id': ['SAMP001', 'SAMP002'],
            'type': ['Type1', 'Type2']
        })
        
        # Should pass validation
        try:
            schema.validate(valid_data, lazy=True)
            self.assertTrue(True, "Valid data should pass validation")
        except Exception as e:
            self.fail(f"Valid data should not fail validation: {e}")
    
    def test_results_schema_validation(self):
        """Test results schema validation."""
        schema = get_schema('results')
        
        # Valid data
        valid_data = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.5, 0.8],
            'cancer_detected': ['Yes', 'No'],
            'sample_quality': [0.9, 0.7],
            'sample_quality_threshold': [0.1, 0.1],
            'sample_status': ['Finished', 'Running'],
            'fail_reason': [None, None],
            'date_of_run': ['2023-01-01', '2023-01-02']
        })
        
        # Should pass validation
        try:
            schema.validate(valid_data, lazy=True)
            self.assertTrue(True, "Valid data should pass validation")
        except Exception as e:
            self.fail(f"Valid data should not fail validation: {e}")
    
    def test_results_schema_range_validation(self):
        """Test results schema range validation."""
        schema = get_schema('results')
        
        # Invalid range values
        invalid_data = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [1.5, -0.5],  # Should be 0-1
            'cancer_detected': ['Yes', 'No'],
            'sample_quality': [1.5, -0.5],  # Should be 0-1
            'sample_quality_threshold': [0.1, 0.1],
            'sample_status': ['Finished', 'Running'],
            'fail_reason': [None, None],
            'date_of_run': ['2023-01-01', '2023-01-02']
        })
        
        # Should fail validation
        with self.assertRaises((SchemaError, SchemaErrors)):
            schema.validate(invalid_data, lazy=True)
    
    def test_results_schema_enum_validation(self):
        """Test results schema enum validation."""
        schema = get_schema('results')
        
        # Invalid enum values
        invalid_data = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.5, 0.8],
            'cancer_detected': ['Maybe', 'Unknown'],  # Should be Yes/No
            'sample_quality': [0.9, 0.7],
            'sample_quality_threshold': [0.1, 0.1],
            'sample_status': ['Pending', 'Processing'],  # Should be Finished/Running/Failed
            'fail_reason': [None, None],
            'date_of_run': ['2023-01-01', '2023-01-02']
        })
        
        # Should fail validation
        with self.assertRaises((SchemaError, SchemaErrors)):
            schema.validate(invalid_data, lazy=True)
    
    def test_merged_schema_validation(self):
        """Test merged schema validation."""
        schema = get_schema('merged')
        
        # Valid merged data
        valid_data = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A'],
            'study_name': ['Study A'],
            'study_cohort_name': ['Cohort A'],
            'project_manager_name': ['Manager A'],
            'disease_name': ['Disease A'],
            'subject_id': ['SUBJ001'],
            'sample_id': ['SAMP001'],
            'type': ['Type1'],
            'detection_value': [0.5],
            'cancer_detected': ['Yes'],
            'sample_quality': [0.9],
            'sample_quality_threshold': [0.1],
            'sample_status': ['Finished'],
            'fail_reason': [None],
            'date_of_run': ['2023-01-01']
        })
        
        # Should pass validation
        try:
            schema.validate(valid_data, lazy=True)
            self.assertTrue(True, "Valid data should pass validation")
        except Exception as e:
            self.fail(f"Valid data should not fail validation: {e}")
    
    def test_summary_schema_validation(self):
        """Test summary schema validation."""
        schema = get_schema('summary')
        
        # Valid summary data
        valid_data = pd.DataFrame({
            'Code': ['P001-S001-C001', 'P002-S002-C002'],
            'Total_Samples': [2, 3],
            'Finished_Percentage': [100.0, 66.7],
            'Lowest_Detection': [0.2, 0.1]
        })
        
        # Should pass validation
        try:
            schema.validate(valid_data, lazy=True)
            self.assertTrue(True, "Valid data should pass validation")
        except Exception as e:
            self.fail(f"Valid data should not fail validation: {e}")


class TestDataQualityEnforcement(unittest.TestCase):
    """Test data quality enforcement."""
    
    def test_null_value_handling(self):
        """Test null value handling."""
        schema = get_schema('projects')
        
        # Data with null values
        null_data = pd.DataFrame({
            'project_code': ['P001', None],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B'],
            'study_name': ['Study A', 'Study B'],
            'study_cohort_name': ['Cohort A', 'Cohort B'],
            'project_manager_name': ['Manager A', 'Manager B'],
            'disease_name': ['Disease A', 'Disease B']
        })
        
        # Should fail validation due to null values
        with self.assertRaises((SchemaError, SchemaErrors)):
            schema.validate(null_data, lazy=True)
    
    def test_duplicate_handling(self):
        """Test duplicate record handling."""
        schema = get_schema('projects')
        
        # Data with duplicates
        duplicate_data = pd.DataFrame({
            'project_code': ['P001', 'P001'],
            'study_code': ['S001', 'S001'],
            'study_cohort_code': ['C001', 'C001'],
            'project_name': ['Project A', 'Project A'],
            'study_name': ['Study A', 'Study A'],
            'study_cohort_name': ['Cohort A', 'Cohort A'],
            'project_manager_name': ['Manager A', 'Manager A'],
            'disease_name': ['Disease A', 'Disease A']
        })
        
        # Should fail validation due to duplicates
        with self.assertRaises((SchemaError, SchemaErrors)):
            schema.validate(duplicate_data, lazy=True)
    
    def test_data_consistency_validation(self):
        """Test data consistency validation."""
        schema = get_schema('results')
        
        # Inconsistent data
        inconsistent_data = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.5, 0.8],
            'cancer_detected': ['Yes', 'No'],
            'sample_quality': [0.9, 0.7],
            'sample_quality_threshold': [0.1, 0.1],
            'sample_status': ['Finished', 'Running'],
            'fail_reason': [None, 'Technical'],  # Inconsistent with status
            'date_of_run': ['2023-01-01', '2023-01-02']
        })
        
        # Should pass validation (fail_reason can be None or string)
        try:
            schema.validate(inconsistent_data, lazy=True)
            self.assertTrue(True, "Data should pass validation")
        except Exception as e:
            self.fail(f"Data should not fail validation: {e}")


class TestSchemaEdgeCases(unittest.TestCase):
    """Test schema edge cases."""
    
    def test_empty_dataframe_validation(self):
        """Test validation of empty dataframes."""
        schema = get_schema('projects')
        
        # Empty dataframe
        empty_data = pd.DataFrame(columns=[
            'project_code', 'study_code', 'study_cohort_code',
            'project_name', 'study_name', 'study_cohort_name',
            'project_manager_name', 'disease_name'
        ])
        
        # Should pass validation (empty is valid)
        try:
            schema.validate(empty_data, lazy=True)
            self.assertTrue(True, "Empty dataframe should pass validation")
        except Exception as e:
            self.fail(f"Empty dataframe should not fail validation: {e}")
    
    def test_single_record_validation(self):
        """Test validation of single record."""
        schema = get_schema('projects')
        
        # Single record
        single_data = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A'],
            'study_name': ['Study A'],
            'study_cohort_name': ['Cohort A'],
            'project_manager_name': ['Manager A'],
            'disease_name': ['Disease A']
        })
        
        # Should pass validation
        try:
            schema.validate(single_data, lazy=True)
            self.assertTrue(True, "Single record should pass validation")
        except Exception as e:
            self.fail(f"Single record should not fail validation: {e}")
    
    def test_large_dataset_validation(self):
        """Test validation of large dataset."""
        schema = get_schema('projects')
        
        # Large dataset
        large_data = pd.DataFrame({
            'project_code': [f'P{i:03d}' for i in range(1000)],
            'study_code': [f'S{i:03d}' for i in range(1000)],
            'study_cohort_code': [f'C{i:03d}' for i in range(1000)],
            'project_name': [f'Project {i}' for i in range(1000)],
            'study_name': [f'Study {i}' for i in range(1000)],
            'study_cohort_name': [f'Cohort {i}' for i in range(1000)],
            'project_manager_name': [f'Manager {i}' for i in range(1000)],
            'disease_name': [f'Disease {i}' for i in range(1000)]
        })
        
        # Should pass validation
        try:
            schema.validate(large_data, lazy=True)
            self.assertTrue(True, "Large dataset should pass validation")
        except Exception as e:
            self.fail(f"Large dataset should not fail validation: {e}")


if __name__ == '__main__':
    unittest.main()
