"""
Test suite for the data pipeline system.

This module contains comprehensive tests for the ETL pipeline,
including unit tests, integration tests, and edge case scenarios.
"""

import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
import json

from core.pipeline import DataPipeline
from config.config_manager import ConfigManager
from core.extract import ProjectsExtractor, SubjectsExtractor, ResultsExtractor
from core.load import ParquetLoader, SummaryLoader
from core.transform import DataMerger, SummaryTransformer, DynamicDataTransformer
from utils.schemas.schemas import get_schema
from utils.exceptions.pipeline_exceptions import (
    ExtractionException,
    TransformationException,
    LoadingException
)


class TestPipeline(unittest.TestCase):
    """Test the main pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "test_config.yaml"
        
        # Create test CSV files
        self.projects_file = self.test_dir / "projects.csv"
        self.subjects_file = self.test_dir / "subjects.csv"
        self.results_file = self.test_dir / "results.csv"
        
        # Create test data
        self.projects_data = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project 1', 'Project 2'],
            'manager_name': ['Manager 1', 'Manager 2']
        })
        
        self.subjects_data = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P002'],
            'study_code': ['S001', 'S001', 'S002'],
            'study_cohort_code': ['C001', 'C001', 'C002'],
            'subject_id': ['SUB001', 'SUB002', 'SUB003'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'sample_type': ['Blood', 'Tissue', 'Blood']
        })
        
        self.results_data = pd.DataFrame({
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'detection_value': [0.8, 0.6, 0.9],
            'cancer_detected': ['Yes', 'No', 'Yes'],
            'sample_status': ['Finished', 'Running', 'Finished']
        })
        
        # Save test data
        self.projects_data.to_csv(self.projects_file, index=False)
        self.subjects_data.to_csv(self.subjects_file, index=False)
        self.results_data.to_csv(self.results_file, index=False)
        
        # Create test config
        self.create_test_config()
    
    def create_test_config(self):
        """Create a test configuration file."""
        config = {
            'files': {
                'projects_file': str(self.projects_file),
                'subjects_file': str(self.subjects_file),
                'results_file': str(self.results_file)
            },
            'data': {
                'input_dir': str(self.test_dir),
                'output_dir': str(self.test_dir / "output"),
                'column_mappings': {
                    'results': {
                        ' detection_value ': 'detection_value',
                        'cancer_detected_(yes_no)': 'cancer_detected'
                    }
                },
                'value_mappings': {
                    'results': {
                        'sample_status': {
                            'finished': 'Finished',
                            'running': 'Running'
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
            import yaml
            yaml.dump(config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config_manager = ConfigManager(str(self.config_path))
        
        # Test basic loading
        self.assertIsInstance(config_manager.config, dict)
        self.assertIn('files', config_manager.config)
        self.assertIn('data', config_manager.config)
        
        # Test file paths
        files_config = config_manager.config['files']
        self.assertIn('projects_file', files_config)
        self.assertIn('subjects_file', files_config)
        self.assertIn('results_file', files_config)
    
    def test_data_extraction(self):
        """Test data extraction functionality."""
        config_manager = ConfigManager(str(self.config_path))
        
        # Test projects extraction
        projects_extractor = ProjectsExtractor(config_manager.config)
        projects_df = projects_extractor.extract()
        self.assertEqual(len(projects_df), 2)
        self.assertIn('project_code', projects_df.columns)
        
        # Test subjects extraction
        subjects_extractor = SubjectsExtractor(config_manager.config)
        subjects_df = subjects_extractor.extract()
        self.assertEqual(len(subjects_df), 3)
        self.assertIn('subject_id', subjects_df.columns)
        
        # Test results extraction
        results_extractor = ResultsExtractor(config_manager.config)
        results_df = results_extractor.extract()
        self.assertEqual(len(results_df), 3)
        self.assertIn('sample_id', results_df.columns)
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        config_manager = ConfigManager(str(self.config_path))
        
        # Test dynamic data transformer
        transformer = DynamicDataTransformer(config_manager.config, 'results')
        transformed_df = transformer.transform(self.results_data.copy())
        
        # Should have same number of rows
        self.assertEqual(len(transformed_df), len(self.results_data))
        
        # Should have cleaned column names
        self.assertIn('detection_value', transformed_df.columns)
    
    def test_data_merging(self):
        """Test data merging functionality."""
        merger = DataMerger()
        
        # Test merging
        merged_df = merger.transform({
            'projects': self.projects_data,
            'subjects': self.subjects_data,
            'results': self.results_data
        })
        
        # Should have merged data
        self.assertGreater(len(merged_df), 0)
        self.assertIn('project_code', merged_df.columns)
        self.assertIn('sample_id', merged_df.columns)
    
    def test_summary_generation(self):
        """Test summary generation functionality."""
        transformer = SummaryTransformer()
        
        # Create merged data for summary
        merged_data = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P002'],
            'study_code': ['S001', 'S001', 'S002'],
            'study_cohort_code': ['C001', 'C001', 'C002'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'detection_value': [0.8, 0.6, 0.9],
            'cancer_detected': ['Yes', 'No', 'Yes'],
            'sample_status': ['Finished', 'Running', 'Finished']
        })
        
        summary_df = transformer.transform(merged_data)
        
        # Should have summary statistics
        self.assertGreater(len(summary_df), 0)
        self.assertIn('project_code', summary_df.columns)
        self.assertIn('samples_detected', summary_df.columns)
    
    def test_data_loading(self):
        """Test data loading functionality."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Test Parquet loading
        parquet_loader = ParquetLoader()
        parquet_path = self.test_dir / "test.parquet"
        parquet_loader.load(test_df, str(parquet_path))
        
        # Verify file exists
        self.assertTrue(parquet_path.exists())
        
        # Test CSV loading
        csv_loader = SummaryLoader()
        csv_path = self.test_dir / "test.csv"
        csv_loader.load(test_df, str(csv_path))
        
        # Verify file exists
        self.assertTrue(csv_path.exists())
    
    def test_schema_validation(self):
        """Test schema validation functionality."""
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
    
    def test_full_pipeline(self):
        """Test the complete pipeline execution."""
        pipeline = DataPipeline(str(self.config_path))
        
        # Test pipeline execution
        success = pipeline.run()
        self.assertTrue(success)
        
        # Verify output files exist
        output_dir = self.test_dir / "output"
        self.assertTrue((output_dir / "output.parquet").exists())
        self.assertTrue((output_dir / "summary.csv").exists())
    
    def test_pipeline_with_invalid_data(self):
        """Test pipeline behavior with invalid data."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'project_code': [None, 'P002'],
            'study_code': ['S001', None],
            'study_cohort_code': ['C001', 'C002']
        })
        
        invalid_file = self.test_dir / "invalid_projects.csv"
        invalid_data.to_csv(invalid_file, index=False)
        
        # Update config to use invalid file
        config = {
            'files': {
                'projects_file': str(invalid_file),
                'subjects_file': str(self.subjects_file),
                'results_file': str(self.results_file)
            },
            'data': {
                'input_dir': str(self.test_dir),
                'output_dir': str(self.test_dir / "output")
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        invalid_config_path = self.test_dir / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            import yaml
            yaml.dump(config, f)
        
        # Test pipeline with invalid data
        pipeline = DataPipeline(str(invalid_config_path))
        
        # Should handle invalid data gracefully
        with self.assertRaises(Exception):
            pipeline.run()
    
    def test_pipeline_with_missing_files(self):
        """Test pipeline behavior with missing input files."""
        # Create config with non-existent files
        config = {
            'files': {
                'projects_file': '/non/existent/projects.csv',
                'subjects_file': str(self.subjects_file),
                'results_file': str(self.results_file)
            },
            'data': {
                'input_dir': str(self.test_dir),
                'output_dir': str(self.test_dir / "output")
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        missing_config_path = self.test_dir / "missing_config.yaml"
        with open(missing_config_path, 'w') as f:
            import yaml
            yaml.dump(config, f)
        
        # Test pipeline with missing files
        pipeline = DataPipeline(str(missing_config_path))
        
        # Should handle missing files gracefully
        with self.assertRaises(Exception):
            pipeline.run()


class TestExtractors(unittest.TestCase):
    """Test individual extractor components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.csv"
        
        # Create test data
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_data.to_csv(self.test_file, index=False)
        
        self.config = {
            'files': {
                'projects_file': str(self.test_file),
                'subjects_file': str(self.test_file),
                'results_file': str(self.test_file)
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_projects_extractor(self):
        """Test projects extractor."""
        extractor = ProjectsExtractor(self.config)
        df = extractor.extract()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
    
    def test_subjects_extractor(self):
        """Test subjects extractor."""
        extractor = SubjectsExtractor(self.config)
        df = extractor.extract()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
    
    def test_results_extractor(self):
        """Test results extractor."""
        extractor = ResultsExtractor(self.config)
        df = extractor.extract()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)


class TestTransformers(unittest.TestCase):
    """Test transformer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        self.config = {
            'data': {
                'column_mappings': {},
                'value_mappings': {}
            }
        }
    
    def test_data_merger(self):
        """Test data merger."""
        merger = DataMerger()
        
        data = {
            'projects': self.test_data,
            'subjects': self.test_data,
            'results': self.test_data
        }
        
        result = merger.transform(data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_summary_transformer(self):
        """Test summary transformer."""
        transformer = SummaryTransformer()
        
        # Create merged-like data
        merged_data = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.8, 0.6],
            'cancer_detected': ['Yes', 'No'],
            'sample_status': ['Finished', 'Running']
        })
        
        result = transformer.transform(merged_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
    
    def test_dynamic_data_transformer(self):
        """Test dynamic data transformer."""
        transformer = DynamicDataTransformer(self.config, 'test')
        result = transformer.transform(self.test_data.copy())
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_data))


class TestLoaders(unittest.TestCase):
    """Test loader components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_parquet_loader(self):
        """Test Parquet loader."""
        loader = ParquetLoader()
        output_path = self.test_dir / "test.parquet"
        
        loader.load(self.test_data, str(output_path))
        
        # Verify file exists
        self.assertTrue(output_path.exists())
        
        # Verify data can be read back
        loaded_data = pd.read_parquet(output_path)
        self.assertEqual(len(loaded_data), len(self.test_data))
    
    def test_summary_loader(self):
        """Test summary loader."""
        loader = SummaryLoader()
        output_path = self.test_dir / "test.csv"
        
        loader.load(self.test_data, str(output_path))
        
        # Verify file exists
        self.assertTrue(output_path.exists())
        
        # Verify data can be read back
        loaded_data = pd.read_csv(output_path)
        self.assertEqual(len(loaded_data), len(self.test_data))


if __name__ == '__main__':
    unittest.main()