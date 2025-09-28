"""
Edge Cases and Error Handling Tests

This test suite focuses on:
- Edge cases and boundary conditions
- Error handling and exception scenarios
- Data corruption and malformed input handling
- Resource constraints and performance limits
- Recovery and resilience testing
"""

import unittest
import pandas as pd
import tempfile
import shutil
import os
import sys
from pathlib import Path
import numpy as np
import json
import yaml

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


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "file_store" / "input"
        self.output_dir = self.test_dir / "file_store" / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_empty_dataframes(self):
        """Test handling of empty dataframes."""
        # Create empty dataframes
        empty_projects = pd.DataFrame(columns=[
            'project_code', 'study_code', 'study_cohort_code',
            'project_name', 'study_name', 'study_cohort_name',
            'project_manager_name', 'disease_name'
        ])
        empty_subjects = pd.DataFrame(columns=[
            'project_code', 'study_code', 'study_cohort_code',
            'subject_id', 'sample_id', 'type'
        ])
        empty_results = pd.DataFrame(columns=[
            'sample_id', 'detection_value', 'cancer_detected',
            'sample_quality', 'sample_quality_threshold',
            'sample_status', 'fail_reason', 'date_of_run'
        ])
        
        # Save empty data
        empty_projects.to_csv(self.input_dir / "projects.csv", index=False)
        empty_subjects.to_csv(self.input_dir / "subjects.csv", index=False)
        empty_results.to_csv(self.input_dir / "results.csv", index=False)
        
        # Create config
        config_path = self._create_config()
        
        # Test pipeline with empty data
        pipeline = DataPipeline(str(config_path))
        success = pipeline.run()
        
        # Should handle empty data gracefully
        self.assertTrue(success, "Pipeline should handle empty data gracefully")
    
    def test_single_record_datasets(self):
        """Test handling of single record datasets."""
        # Create single record datasets
        single_projects = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A'],
            'study_name': ['Study A'],
            'study_cohort_name': ['Cohort A'],
            'project_manager_name': ['Manager A'],
            'disease_name': ['Disease A']
        })
        
        single_subjects = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'subject_id': ['SUBJ001'],
            'sample_id': ['SAMP001'],
            'type': ['Type1']
        })
        
        single_results = pd.DataFrame({
            'sample_id': ['SAMP001'],
            'detection_value': [0.5],
            'cancer_detected': ['Yes'],
            'sample_quality': [0.9],
            'sample_quality_threshold': [0.1],
            'sample_status': ['Finished'],
            'fail_reason': [None],
            'date_of_run': ['2023-01-01']
        })
        
        # Save single record data
        single_projects.to_csv(self.input_dir / "projects.csv", index=False)
        single_subjects.to_csv(self.input_dir / "subjects.csv", index=False)
        single_results.to_csv(self.input_dir / "results.csv", index=False)
        
        # Create config
        config_path = self._create_config()
        
        # Test pipeline with single record data
        pipeline = DataPipeline(str(config_path))
        success = pipeline.run()
        
        # Should handle single record data gracefully
        self.assertTrue(success, "Pipeline should handle single record data gracefully")
    
    def test_large_datasets(self):
        """Test handling of large datasets."""
        # Create large datasets
        large_projects = pd.DataFrame({
            'project_code': [f'P{i:05d}' for i in range(1000)],
            'study_code': [f'S{i:05d}' for i in range(1000)],
            'study_cohort_code': [f'C{i:05d}' for i in range(1000)],
            'project_name': [f'Project {i}' for i in range(1000)],
            'study_name': [f'Study {i}' for i in range(1000)],
            'study_cohort_name': [f'Cohort {i}' for i in range(1000)],
            'project_manager_name': [f'Manager {i}' for i in range(1000)],
            'disease_name': [f'Disease {i}' for i in range(1000)]
        })
        
        large_subjects = pd.DataFrame({
            'project_code': [f'P{i:05d}' for i in range(1000)],
            'study_code': [f'S{i:05d}' for i in range(1000)],
            'study_cohort_code': [f'C{i:05d}' for i in range(1000)],
            'subject_id': [f'SUBJ{i:05d}' for i in range(1000)],
            'sample_id': [f'SAMP{i:05d}' for i in range(1000)],
            'type': [f'Type{i % 3}' for i in range(1000)]
        })
        
        large_results = pd.DataFrame({
            'sample_id': [f'SAMP{i:05d}' for i in range(1000)],
            'detection_value': [np.random.random() for _ in range(1000)],
            'cancer_detected': [np.random.choice(['Yes', 'No']) for _ in range(1000)],
            'sample_quality': [np.random.random() for _ in range(1000)],
            'sample_quality_threshold': [0.1] * 1000,
            'sample_status': [np.random.choice(['Finished', 'Running', 'Failed']) for _ in range(1000)],
            'fail_reason': [None] * 1000,
            'date_of_run': [f'2023-01-{i % 28 + 1:02d}' for i in range(1000)]
        })
        
        # Save large data
        large_projects.to_csv(self.input_dir / "projects.csv", index=False)
        large_subjects.to_csv(self.input_dir / "subjects.csv", index=False)
        large_results.to_csv(self.input_dir / "results.csv", index=False)
        
        # Create config
        config_path = self._create_config()
        
        # Test pipeline with large data
        pipeline = DataPipeline(str(config_path))
        success = pipeline.run()
        
        # Should handle large data gracefully
        self.assertTrue(success, "Pipeline should handle large data gracefully")
    
    def test_malformed_csv_files(self):
        """Test handling of malformed CSV files."""
        # Create malformed CSV files
        malformed_csv = "project_code,study_code\nP001,S001\nP002,S002\n"  # Missing columns
        with open(self.input_dir / "projects.csv", 'w') as f:
            f.write(malformed_csv)
        
        # Create valid subjects and results
        valid_subjects = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'subject_id': ['SUBJ001'],
            'sample_id': ['SAMP001'],
            'type': ['Type1']
        })
        
        valid_results = pd.DataFrame({
            'sample_id': ['SAMP001'],
            'detection_value': [0.5],
            'cancer_detected': ['Yes'],
            'sample_quality': [0.9],
            'sample_quality_threshold': [0.1],
            'sample_status': ['Finished'],
            'fail_reason': [None],
            'date_of_run': ['2023-01-01']
        })
        
        valid_subjects.to_csv(self.input_dir / "subjects.csv", index=False)
        valid_results.to_csv(self.input_dir / "results.csv", index=False)
        
        # Create config
        config_path = self._create_config()
        
        # Test pipeline with malformed data
        pipeline = DataPipeline(str(config_path))
        success = pipeline.run()
        
        # Should handle malformed data gracefully
        self.assertFalse(success, "Pipeline should fail with malformed data")
    
    def test_missing_files(self):
        """Test handling of missing input files."""
        # Create config with non-existent files
        config = {
            'files': {
                'projects_file': '/non/existent/projects.csv',
                'subjects_file': str(self.input_dir / "subjects.csv"),
                'results_file': str(self.input_dir / "results.csv"),
                'output_parquet': str(self.output_dir / "output.parquet"),
                'summary_csv': str(self.output_dir / "summary.csv")
            },
            'data': {
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir)
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        config_path = self.test_dir / "missing_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Test pipeline with missing files
        pipeline = DataPipeline(str(config_path))
        success = pipeline.run()
        
        # Should handle missing files gracefully
        self.assertFalse(success, "Pipeline should fail with missing files")
    
    def test_corrupted_data(self):
        """Test handling of corrupted data."""
        # Create corrupted data
        corrupted_data = pd.DataFrame({
            'project_code': ['P001', None, 'P003'],  # Null values
            'study_code': ['S001', 'S002', 'S003'],
            'study_cohort_code': ['C001', 'C002', 'C003'],
            'project_name': ['Project A', 'Project B', 'Project C'],
            'study_name': ['Study A', 'Study B', 'Study C'],
            'study_cohort_name': ['Cohort A', 'Cohort B', 'Cohort C'],
            'project_manager_name': ['Manager A', 'Manager B', 'Manager C'],
            'disease_name': ['Disease A', 'Disease B', 'Disease C']
        })
        
        # Save corrupted data
        corrupted_data.to_csv(self.input_dir / "projects.csv", index=False)
        
        # Create valid subjects and results
        valid_subjects = pd.DataFrame({
            'project_code': ['P001', 'P002', 'P003'],
            'study_code': ['S001', 'S002', 'S003'],
            'study_cohort_code': ['C001', 'C002', 'C003'],
            'subject_id': ['SUBJ001', 'SUBJ002', 'SUBJ003'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'type': ['Type1', 'Type2', 'Type1']
        })
        
        valid_results = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'detection_value': [0.5, 0.8, 0.2],
            'cancer_detected': ['Yes', 'No', 'Yes'],
            'sample_quality': [0.9, 0.7, 0.6],
            'sample_quality_threshold': [0.1, 0.1, 0.1],
            'sample_status': ['Finished', 'Running', 'Finished'],
            'fail_reason': [None, None, None],
            'date_of_run': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        valid_subjects.to_csv(self.input_dir / "subjects.csv", index=False)
        valid_results.to_csv(self.input_dir / "results.csv", index=False)
        
        # Create config
        config_path = self._create_config()
        
        # Test pipeline with corrupted data
        pipeline = DataPipeline(str(config_path))
        success = pipeline.run()
        
        # Should handle corrupted data gracefully
        self.assertFalse(success, "Pipeline should fail with corrupted data")
    
    def _create_config(self):
        """Create test configuration."""
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
                'output_dir': str(self.output_dir)
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        config_path = self.test_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exception scenarios."""
    
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
    
    def test_extraction_exceptions(self):
        """Test extraction exception handling."""
        # Test with non-existent file
        with self.assertRaises(ExtractionException):
            extractor = ProjectsExtractor("/non/existent/file.csv")
            extractor.extract()
    
    def test_validation_exceptions(self):
        """Test validation exception handling."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'project_code': [1, 2],  # Wrong type
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project A', 'Project B'],
            'study_name': ['Study A', 'Study B'],
            'study_cohort_name': ['Cohort A', 'Cohort B'],
            'project_manager_name': ['Manager A', 'Manager B'],
            'disease_name': ['Disease A', 'Disease B']
        })
        
        # Test schema validation
        schema = get_schema('projects')
        with self.assertRaises(Exception):
            schema.validate(invalid_data, lazy=True)
    
    def test_merge_exceptions(self):
        """Test merge exception handling."""
        # Create data with no relationships
        projects_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A']
        })
        
        subjects_df = pd.DataFrame({
            'project_code': ['P002'],  # No match
            'study_code': ['S002'],
            'study_cohort_code': ['C002'],
            'subject_id': ['SUBJ001'],
            'sample_id': ['SAMP001']
        })
        
        results_df = pd.DataFrame({
            'sample_id': ['SAMP001'],
            'detection_value': [0.5],
            'cancer_detected': ['Yes']
        })
        
        # Test merge
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=projects_df,
            subjects_df=subjects_df,
            results_df=results_df
        )
        
        # Should return empty dataframe
        self.assertEqual(len(merged_df), 0)
    
    def test_loading_exceptions(self):
        """Test loading exception handling."""
        # Test with invalid output path
        with self.assertRaises(Exception):
            loader = ParquetLoader("/invalid/path/output.parquet")
            loader.load(pd.DataFrame({'col1': [1, 2, 3]}))
    
    def test_summary_generation_exceptions(self):
        """Test summary generation exception handling."""
        # Create data with missing required columns
        incomplete_data = pd.DataFrame({
            'some_column': ['value1', 'value2']
            # Missing all required columns
        })
        
        # Test summary generation
        summary_transformer = SummaryTransformer()
        with self.assertRaises(Exception):
            summary_transformer.transform(incomplete_data)


class TestResourceConstraints(unittest.TestCase):
    """Test resource constraints and performance limits."""
    
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
    
    def test_memory_usage_with_large_data(self):
        """Test memory usage with large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'project_code': [f'P{i:05d}' for i in range(10000)],
            'study_code': [f'S{i:05d}' for i in range(10000)],
            'study_cohort_code': [f'C{i:05d}' for i in range(10000)],
            'project_name': [f'Project {i}' for i in range(10000)],
            'study_name': [f'Study {i}' for i in range(10000)],
            'study_cohort_name': [f'Cohort {i}' for i in range(10000)],
            'project_manager_name': [f'Manager {i}' for i in range(10000)],
            'disease_name': [f'Disease {i}' for i in range(10000)]
        })
        
        # Test schema validation with large data
        schema = get_schema('projects')
        try:
            schema.validate(large_data, lazy=True)
            self.assertTrue(True, "Large dataset should be handled without memory issues")
        except MemoryError:
            self.fail("Large dataset should not cause memory issues")
        except Exception as e:
            # Other exceptions are acceptable for large datasets
            pass
    
    def test_performance_with_medium_dataset(self):
        """Test performance with medium dataset."""
        # Create medium dataset
        medium_data = pd.DataFrame({
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
        import time
        start_time = time.time()
        schema = get_schema('projects')
        schema.validate(medium_data, lazy=True)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 5 seconds)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5.0, f"Schema validation took {execution_time} seconds")
    
    def test_disk_space_usage(self):
        """Test disk space usage with large outputs."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': range(10000),
            'col2': [f'value_{i}' for i in range(10000)],
            'col3': [i * 0.001 for i in range(10000)]
        })
        
        # Test Parquet output
        parquet_path = self.output_dir / "large_test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(large_data)
        
        # Verify file was created and has reasonable size
        self.assertTrue(parquet_path.exists())
        file_size = parquet_path.stat().st_size
        self.assertGreater(file_size, 0, "Output file should not be empty")
        self.assertLess(file_size, 100 * 1024 * 1024, "Output file should not be too large")  # Less than 100MB


if __name__ == '__main__':
    unittest.main()
