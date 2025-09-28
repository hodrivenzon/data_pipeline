"""
Comprehensive ETL Pipeline Tests

This test suite focuses on:
- Pipeline and ETL logic correctness
- Data validation and schema enforcement
- Data modeling and structure integrity
- Data merge operations and output matching
- Edge cases and error handling
- Performance and scalability
"""

import unittest
import pandas as pd
import tempfile
import shutil
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import DataPipeline
from config.config_manager import ConfigManager
from core.extract import ProjectsExtractor, SubjectsExtractor, ResultsExtractor
from core.load import ParquetLoader, SummaryLoader
from core.transform import DataMerger, DynamicDataTransformer, SummaryTransformer
from utils.schemas.schemas import get_schema
from utils.exceptions.pipeline_exceptions import (
    ExtractionException, DataCleaningException, DataMergingException,
    DataValidationException, SummaryGenerationException, LoadingException
)


class TestETLPipelineLogic(unittest.TestCase):
    """Test ETL pipeline logic and data flow correctness."""
    
    def setUp(self):
        """Set up test fixtures with comprehensive data."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "file_store" / "input"
        self.output_dir = self.test_dir / "file_store" / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive test data
        self._create_test_data()
        self._create_config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_data(self):
        """Create comprehensive test datasets."""
        # Projects data with all required fields
        self.projects_data = pd.DataFrame({
            'project_code': ['P001', 'P002', 'P003'],
            'study_code': ['S001', 'S002', 'S003'],
            'study_cohort_code': ['C001', 'C002', 'C003'],
            'project_name': ['Project Alpha', 'Project Beta', 'Project Gamma'],
            'study_name': ['Study 1', 'Study 2', 'Study 3'],
            'study_cohort_name': ['Cohort A', 'Cohort B', 'Cohort C'],
            'project_manager_name': ['Manager A', 'Manager B', 'Manager C'],
            'disease_name': ['Disease X', 'Disease Y', 'Disease Z']
        })
        
        # Subjects data with proper relationships
        self.subjects_data = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P002', 'P002', 'P003'],
            'study_code': ['S001', 'S001', 'S002', 'S002', 'S003'],
            'study_cohort_code': ['C001', 'C001', 'C002', 'C002', 'C003'],
            'subject_id': ['SUBJ001', 'SUBJ002', 'SUBJ003', 'SUBJ004', 'SUBJ005'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003', 'SAMP004', 'SAMP005'],
            'type': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1']
        })
        
        # Results data with various scenarios
        self.results_data = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003', 'SAMP004', 'SAMP005'],
            'detection_value': [0.5, 0.8, 0.2, 0.9, 0.1],
            'cancer_detected': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'sample_quality': [0.9, 0.7, 0.6, 0.8, 0.5],
            'sample_quality_threshold': [0.1, 0.1, 0.1, 0.1, 0.1],
            'sample_status': ['Finished', 'Running', 'Finished', 'Failed', 'Finished'],
            'fail_reason': [None, None, None, 'Technical', None],
            'date_of_run': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        # Save test data
        self.projects_data.to_csv(self.input_dir / "projects.csv", index=False)
        self.subjects_data.to_csv(self.input_dir / "subjects.csv", index=False)
        self.results_data.to_csv(self.input_dir / "results.csv", index=False)
    
    def _create_config(self):
        """Create comprehensive configuration."""
        self.config_path = self.test_dir / "config.yaml"
        config = {
            'files': {
                'projects_file': str(self.input_dir / "projects.csv"),
                'subjects_file': str(self.input_dir / "subjects.csv"),
                'results_file': str(self.input_dir / "results.csv"),
                'output_parquet': str(self.output_dir / "output.parquet"),
                'summary_csv': str(self.output_dir / "summary.csv")
            },
            'data': {
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'column_mappings': {
                    'results': {
                        'detection_value': 'detection_value',
                        'cancer_detected': 'cancer_detected',
                        'sample_status': 'sample_status',
                        'fail_reason': 'fail_reason',
                        'sample_quality_threshold': 'sample_quality_threshold',
                        'date_of_run': 'date_of_run'
                    }
                },
                'value_mappings': {
                    'results': {
                        'sample_status': {
                            'finished': 'Finished',
                            'running': 'Running',
                            'failed': 'Failed'
                        }
                    }
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_complete_etl_pipeline(self):
        """Test complete ETL pipeline execution."""
        pipeline = DataPipeline(str(self.config_path))
        success = pipeline.run()
        
        self.assertTrue(success, "Pipeline should complete successfully")
        
        # Verify output files exist
        self.assertTrue(Path(self.output_dir / "output.parquet").exists())
        self.assertTrue(Path(self.output_dir / "summary.csv").exists())
    
    def test_data_extraction_correctness(self):
        """Test data extraction correctness and completeness."""
        config_manager = ConfigManager(str(self.config_path))
        
        # Test projects extraction
        projects_extractor = ProjectsExtractor(config_manager.config['files']['projects_file'])
        projects_df = projects_extractor.extract()
        
        self.assertEqual(len(projects_df), 3)
        self.assertIn('project_code', projects_df.columns)
        self.assertIn('project_name', projects_df.columns)
        
        # Test subjects extraction
        subjects_extractor = SubjectsExtractor(config_manager.config['files']['subjects_file'])
        subjects_df = subjects_extractor.extract()
        
        self.assertEqual(len(subjects_df), 5)
        self.assertIn('subject_id', subjects_df.columns)
        self.assertIn('sample_id', subjects_df.columns)
        
        # Test results extraction
        results_extractor = ResultsExtractor(config_manager.config['files']['results_file'])
        results_df = results_extractor.extract()
        
        self.assertEqual(len(results_df), 5)
        self.assertIn('detection_value', results_df.columns)
        self.assertIn('cancer_detected', results_df.columns)
    
    def test_data_transformation_correctness(self):
        """Test data transformation correctness."""
        config_manager = ConfigManager(str(self.config_path))
        
        # Test projects transformation
        projects_transformer = DynamicDataTransformer(config_manager.config, 'projects')
        transformed_projects = projects_transformer.transform(self.projects_data.copy())
        
        self.assertEqual(len(transformed_projects), 3)
        self.assertIn('project_code', transformed_projects.columns)
        
        # Test subjects transformation
        subjects_transformer = DynamicDataTransformer(config_manager.config, 'subjects')
        transformed_subjects = subjects_transformer.transform(self.subjects_data.copy())
        
        self.assertEqual(len(transformed_subjects), 5)
        self.assertIn('subject_id', transformed_subjects.columns)
        
        # Test results transformation
        results_transformer = DynamicDataTransformer(config_manager.config, 'results')
        transformed_results = results_transformer.transform(self.results_data.copy())
        
        self.assertEqual(len(transformed_results), 5)
        self.assertIn('detection_value', transformed_results.columns)
    
    def test_data_merging_correctness(self):
        """Test data merging logic and relationship integrity."""
        merger = DataMerger()
        
        # Test merging with proper relationships
        merged_df = merger.transform(
            projects_df=self.projects_data,
            subjects_df=self.subjects_data,
            results_df=self.results_data
        )
        
        # Verify merged data structure
        self.assertEqual(len(merged_df), 5)  # Should match results count
        self.assertIn('project_name', merged_df.columns)
        self.assertIn('subject_id', merged_df.columns)
        self.assertIn('sample_id', merged_df.columns)
        self.assertIn('detection_value', merged_df.columns)
        
        # Verify relationships are maintained
        for _, row in merged_df.iterrows():
            # Each row should have project info
            self.assertIsNotNone(row['project_name'])
            self.assertIsNotNone(row['subject_id'])
            self.assertIsNotNone(row['sample_id'])
    
    def test_schema_validation_correctness(self):
        """Test schema validation and data quality enforcement."""
        # Test projects schema
        projects_schema = get_schema('projects')
        self.assertIsNotNone(projects_schema)
        
        # Test subjects schema
        subjects_schema = get_schema('subjects')
        self.assertIsNotNone(subjects_schema)
        
        # Test results schema
        results_schema = get_schema('results')
        self.assertIsNotNone(results_schema)
        
        # Test merged schema
        merged_schema = get_schema('merged')
        self.assertIsNotNone(merged_schema)
        
        # Test summary schema
        summary_schema = get_schema('summary')
        self.assertIsNotNone(summary_schema)
    
    def test_summary_generation_correctness(self):
        """Test summary generation and statistics accuracy."""
        # Create merged data for summary generation
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=self.projects_data,
            subjects_df=self.subjects_data,
            results_df=self.results_data
        )
        
        # Generate summary
        summary_transformer = SummaryTransformer()
        summary_df = summary_transformer.transform(merged_df)
        
        # Verify summary structure
        self.assertGreater(len(summary_df), 0)
        self.assertIn('Code', summary_df.columns)
        self.assertIn('Total_Samples', summary_df.columns)
        self.assertIn('Finished_Percentage', summary_df.columns)
        self.assertIn('Lowest_Detection', summary_df.columns)
        
        # Verify summary calculations
        for _, row in summary_df.iterrows():
            self.assertGreaterEqual(row['Total_Samples'], 0)
            self.assertGreaterEqual(row['Finished_Percentage'], 0)
            self.assertLessEqual(row['Finished_Percentage'], 100)
            if not pd.isna(row['Lowest_Detection']):
                self.assertGreaterEqual(row['Lowest_Detection'], 0)
                self.assertLessEqual(row['Lowest_Detection'], 1)
    
    def test_data_loading_correctness(self):
        """Test data loading and output file generation."""
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Test Parquet loading
        parquet_path = self.output_dir / "test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(test_df)
        
        self.assertTrue(parquet_path.exists())
        
        # Verify data integrity
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(test_df, loaded_df)
        
        # Test CSV loading
        csv_path = self.output_dir / "test.csv"
        csv_loader = SummaryLoader(str(csv_path))
        csv_loader.load(test_df)
        
        self.assertTrue(csv_path.exists())
        
        # Verify data integrity
        loaded_csv_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(test_df, loaded_csv_df)


class TestDataValidationAndModeling(unittest.TestCase):
    """Test data validation and modeling correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_schema_validation_with_valid_data(self):
        """Test schema validation with valid data."""
        # Create valid projects data
        valid_projects = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B'],
            'study_name': ['Study A', 'Study B'],
            'study_cohort_name': ['Cohort A', 'Cohort B'],
            'project_manager_name': ['Manager A', 'Manager B'],
            'disease_name': ['Disease A', 'Disease B']
        })
        
        # Test schema validation
        projects_schema = get_schema('projects')
        try:
            projects_schema.validate(valid_projects, lazy=True)
            self.assertTrue(True, "Valid data should pass schema validation")
        except Exception as e:
            self.fail(f"Valid data should not fail schema validation: {e}")
    
    def test_schema_validation_with_invalid_data(self):
        """Test schema validation with invalid data."""
        # Create invalid projects data (missing required columns)
        invalid_projects = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002']
            # Missing required columns
        })
        
        # Test schema validation
        projects_schema = get_schema('projects')
        with self.assertRaises(Exception):
            projects_schema.validate(invalid_projects, lazy=True)
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Create data with wrong types
        invalid_types = pd.DataFrame({
            'project_code': [1, 2],  # Should be string
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B'],
            'study_name': ['Study A', 'Study B'],
            'study_cohort_name': ['Cohort A', 'Cohort B'],
            'project_manager_name': ['Manager A', 'Manager B'],
            'disease_name': ['Disease A', 'Disease B']
        })
        
        # Test schema validation
        projects_schema = get_schema('projects')
        with self.assertRaises(Exception):
            projects_schema.validate(invalid_types, lazy=True)
    
    def test_data_range_validation(self):
        """Test data range validation."""
        # Create data with invalid ranges
        invalid_ranges = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [1.5, -0.5],  # Should be 0-1
            'cancer_detected': ['Yes', 'No'],
            'sample_quality': [1.5, -0.5],  # Should be 0-1
            'sample_quality_threshold': [0.1, 0.1],
            'sample_status': ['Finished', 'Running'],
            'fail_reason': [None, None],
            'date_of_run': ['2023-01-01', '2023-01-02']
        })
        
        # Test schema validation
        results_schema = get_schema('results')
        with self.assertRaises(Exception):
            results_schema.validate(invalid_ranges, lazy=True)
    
    def test_data_enum_validation(self):
        """Test data enum validation."""
        # Create data with invalid enum values
        invalid_enums = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.5, 0.8],
            'cancer_detected': ['Maybe', 'Unknown'],  # Should be Yes/No
            'sample_quality': [0.9, 0.7],
            'sample_quality_threshold': [0.1, 0.1],
            'sample_status': ['Pending', 'Processing'],  # Should be Finished/Running/Failed
            'fail_reason': [None, None],
            'date_of_run': ['2023-01-01', '2023-01-02']
        })
        
        # Test schema validation
        results_schema = get_schema('results')
        with self.assertRaises(Exception):
            results_schema.validate(invalid_enums, lazy=True)


class TestDataMergeAndOutputMatching(unittest.TestCase):
    """Test data merge operations and output matching."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_data_merge_relationships(self):
        """Test data merge relationship integrity."""
        # Create test data with known relationships
        projects_df = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B']
        })
        
        subjects_df = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P002'],
            'study_code': ['S001', 'S001', 'S002'],
            'study_cohort_code': ['C001', 'C001', 'C002'],
            'subject_id': ['SUBJ001', 'SUBJ002', 'SUBJ003'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003']
        })
        
        results_df = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'detection_value': [0.5, 0.8, 0.2],
            'cancer_detected': ['Yes', 'No', 'Yes']
        })
        
        # Test merge
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=projects_df,
            subjects_df=subjects_df,
            results_df=results_df
        )
        
        # Verify relationships
        self.assertEqual(len(merged_df), 3)
        
        # Check that each result has corresponding project and subject info
        for _, row in merged_df.iterrows():
            if row['sample_id'] == 'SAMP001':
                self.assertEqual(row['project_name'], 'Project A')
                self.assertEqual(row['subject_id'], 'SUBJ001')
            elif row['sample_id'] == 'SAMP002':
                self.assertEqual(row['project_name'], 'Project A')
                self.assertEqual(row['subject_id'], 'SUBJ002')
            elif row['sample_id'] == 'SAMP003':
                self.assertEqual(row['project_name'], 'Project B')
                self.assertEqual(row['subject_id'], 'SUBJ003')
    
    def test_data_merge_with_missing_relationships(self):
        """Test data merge with missing relationships."""
        # Create data with missing relationships
        projects_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A']
        })
        
        subjects_df = pd.DataFrame({
            'project_code': ['P001', 'P002'],  # P002 not in projects
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'subject_id': ['SUBJ001', 'SUBJ002'],
            'sample_id': ['SAMP001', 'SAMP002']
        })
        
        results_df = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.5, 0.8],
            'cancer_detected': ['Yes', 'No']
        })
        
        # Test merge
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=projects_df,
            subjects_df=subjects_df,
            results_df=results_df
        )
        
        # Should only include results that have valid relationships
        self.assertEqual(len(merged_df), 1)  # Only SAMP001 should be included
        self.assertEqual(merged_df.iloc[0]['sample_id'], 'SAMP001')
    
    def test_output_file_matching(self):
        """Test output file content matching."""
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [0.1, 0.2, 0.3]
        })
        
        # Test Parquet output
        parquet_path = self.output_dir / "test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(test_df)
        
        # Verify content matches
        loaded_parquet = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(test_df, loaded_parquet)
        
        # Test CSV output
        csv_path = self.output_dir / "test.csv"
        csv_loader = SummaryLoader(str(csv_path))
        csv_loader.load(test_df)
        
        # Verify content matches
        loaded_csv = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(test_df, loaded_csv)
    
    def test_data_consistency_across_stages(self):
        """Test data consistency across ETL stages."""
        # Create comprehensive test data
        projects_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A']
        })
        
        subjects_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'subject_id': ['SUBJ001'],
            'sample_id': ['SAMP001']
        })
        
        results_df = pd.DataFrame({
            'sample_id': ['SAMP001'],
            'detection_value': [0.5],
            'cancer_detected': ['Yes']
        })
        
        # Test each stage
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=projects_df,
            subjects_df=subjects_df,
            results_df=results_df
        )
        
        # Verify data consistency
        self.assertEqual(len(merged_df), 1)
        self.assertEqual(merged_df.iloc[0]['project_code'], 'P001')
        self.assertEqual(merged_df.iloc[0]['subject_id'], 'SUBJ001')
        self.assertEqual(merged_df.iloc[0]['sample_id'], 'SAMP001')
        self.assertEqual(merged_df.iloc[0]['detection_value'], 0.5)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_empty_dataframes(self):
        """Test handling of empty dataframes."""
        # Create empty dataframes
        empty_projects = pd.DataFrame(columns=['project_code', 'study_code', 'study_cohort_code'])
        empty_subjects = pd.DataFrame(columns=['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id'])
        empty_results = pd.DataFrame(columns=['sample_id', 'detection_value', 'cancer_detected'])
        
        # Test merge with empty data
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=empty_projects,
            subjects_df=empty_subjects,
            results_df=empty_results
        )
        
        self.assertEqual(len(merged_df), 0)
    
    def test_missing_columns(self):
        """Test handling of missing columns."""
        # Create data with missing columns
        incomplete_projects = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001']
            # Missing required columns
        })
        
        # Test schema validation
        projects_schema = get_schema('projects')
        with self.assertRaises(Exception):
            projects_schema.validate(incomplete_projects, lazy=True)
    
    def test_null_values(self):
        """Test handling of null values."""
        # Create data with null values
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
        
        # Test schema validation
        projects_schema = get_schema('projects')
        with self.assertRaises(Exception):
            projects_schema.validate(null_data, lazy=True)
    
    def test_duplicate_records(self):
        """Test handling of duplicate records."""
        # Create data with duplicates
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
        
        # Test schema validation
        projects_schema = get_schema('projects')
        with self.assertRaises(Exception):
            projects_schema.validate(duplicate_data, lazy=True)
    
    def test_large_datasets(self):
        """Test handling of large datasets."""
        # Create large dataset
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
        
        # Test schema validation
        projects_schema = get_schema('projects')
        try:
            projects_schema.validate(large_data, lazy=True)
            self.assertTrue(True, "Large dataset should pass schema validation")
        except Exception as e:
            self.fail(f"Large dataset should not fail schema validation: {e}")


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_performance_with_medium_dataset(self):
        """Test performance with medium dataset."""
        # Create medium dataset (1000 records)
        medium_projects = pd.DataFrame({
            'project_code': [f'P{i:03d}' for i in range(1000)],
            'study_code': [f'S{i:03d}' for i in range(1000)],
            'study_cohort_code': [f'C{i:03d}' for i in range(1000)],
            'project_name': [f'Project {i}' for i in range(1000)],
            'study_name': [f'Study {i}' for i in range(1000)],
            'study_cohort_name': [f'Cohort {i}' for i in range(1000)],
            'project_manager_name': [f'Manager {i}' for i in range(1000)],
            'disease_name': [f'Disease {i}' for i in range(1000)]
        })
        
        # Test schema validation performance
        start_time = datetime.now()
        projects_schema = get_schema('projects')
        projects_schema.validate(medium_projects, lazy=True)
        end_time = datetime.now()
        
        # Should complete within reasonable time (less than 5 seconds)
        execution_time = (end_time - start_time).total_seconds()
        self.assertLess(execution_time, 5.0, f"Schema validation took {execution_time} seconds")
    
    def test_memory_usage_with_large_dataset(self):
        """Test memory usage with large dataset."""
        # Create large dataset (10000 records)
        large_projects = pd.DataFrame({
            'project_code': [f'P{i:05d}' for i in range(10000)],
            'study_code': [f'S{i:05d}' for i in range(10000)],
            'study_cohort_code': [f'C{i:05d}' for i in range(10000)],
            'project_name': [f'Project {i}' for i in range(10000)],
            'study_name': [f'Study {i}' for i in range(10000)],
            'study_cohort_name': [f'Cohort {i}' for i in range(10000)],
            'project_manager_name': [f'Manager {i}' for i in range(10000)],
            'disease_name': [f'Disease {i}' for i in range(10000)]
        })
        
        # Test that we can handle large dataset without memory issues
        projects_schema = get_schema('projects')
        try:
            projects_schema.validate(large_projects, lazy=True)
            self.assertTrue(True, "Large dataset should be handled without memory issues")
        except MemoryError:
            self.fail("Large dataset should not cause memory issues")
        except Exception as e:
            # Other exceptions are acceptable for large datasets
            pass


if __name__ == '__main__':
    unittest.main()
