"""
Data Merge and Output Matching Tests

This test suite focuses on:
- Data merge operations correctness
- Relationship integrity
- Output file generation and matching
- Data consistency across stages
- Merge edge cases and error handling
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

from core.transform import DataMerger, SummaryTransformer
from core.load import ParquetLoader, SummaryLoader
from utils.schemas.schemas import get_schema


class TestDataMergeOperations(unittest.TestCase):
    """Test data merge operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_basic_data_merge(self):
        """Test basic data merge functionality."""
        # Create test data
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
        
        # Verify merge results
        self.assertEqual(len(merged_df), 3)
        self.assertIn('project_name', merged_df.columns)
        self.assertIn('subject_id', merged_df.columns)
        self.assertIn('sample_id', merged_df.columns)
        self.assertIn('detection_value', merged_df.columns)
        
        # Verify relationships
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
    
    def test_merge_with_missing_relationships(self):
        """Test merge with missing relationships."""
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
        
        # Should only include results with valid relationships
        self.assertEqual(len(merged_df), 1)
        self.assertEqual(merged_df.iloc[0]['sample_id'], 'SAMP001')
        self.assertEqual(merged_df.iloc[0]['project_name'], 'Project A')
    
    def test_merge_with_duplicate_keys(self):
        """Test merge with duplicate keys."""
        # Create data with duplicate keys
        projects_df = pd.DataFrame({
            'project_code': ['P001', 'P001'],  # Duplicate
            'study_code': ['S001', 'S001'],
            'study_cohort_code': ['C001', 'C001'],
            'project_name': ['Project A', 'Project A']
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
        
        # Test merge
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=projects_df,
            subjects_df=subjects_df,
            results_df=results_df
        )
        
        # Should handle duplicates gracefully
        self.assertEqual(len(merged_df), 1)
        self.assertEqual(merged_df.iloc[0]['sample_id'], 'SAMP001')
    
    def test_merge_with_empty_dataframes(self):
        """Test merge with empty dataframes."""
        # Create empty dataframes
        empty_projects = pd.DataFrame(columns=['project_code', 'study_code', 'study_cohort_code', 'project_name'])
        empty_subjects = pd.DataFrame(columns=['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id'])
        empty_results = pd.DataFrame(columns=['sample_id', 'detection_value', 'cancer_detected'])
        
        # Test merge
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=empty_projects,
            subjects_df=empty_subjects,
            results_df=empty_results
        )
        
        # Should return empty dataframe
        self.assertEqual(len(merged_df), 0)
    
    def test_merge_data_consistency(self):
        """Test merge data consistency."""
        # Create comprehensive test data
        projects_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'project_name': ['Project A'],
            'study_name': ['Study A'],
            'study_cohort_name': ['Cohort A'],
            'project_manager_name': ['Manager A'],
            'disease_name': ['Disease A']
        })
        
        subjects_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'subject_id': ['SUBJ001'],
            'sample_id': ['SAMP001'],
            'type': ['Type1']
        })
        
        results_df = pd.DataFrame({
            'sample_id': ['SAMP001'],
            'detection_value': [0.5],
            'cancer_detected': ['Yes'],
            'sample_quality': [0.9],
            'sample_quality_threshold': [0.1],
            'sample_status': ['Finished'],
            'fail_reason': [None],
            'date_of_run': ['2023-01-01']
        })
        
        # Test merge
        merger = DataMerger()
        merged_df = merger.transform(
            projects_df=projects_df,
            subjects_df=subjects_df,
            results_df=results_df
        )
        
        # Verify data consistency
        self.assertEqual(len(merged_df), 1)
        row = merged_df.iloc[0]
        
        # Check all relationships are maintained
        self.assertEqual(row['project_code'], 'P001')
        self.assertEqual(row['study_code'], 'S001')
        self.assertEqual(row['study_cohort_code'], 'C001')
        self.assertEqual(row['project_name'], 'Project A')
        self.assertEqual(row['subject_id'], 'SUBJ001')
        self.assertEqual(row['sample_id'], 'SAMP001')
        self.assertEqual(row['detection_value'], 0.5)
        self.assertEqual(row['cancer_detected'], 'Yes')


class TestOutputMatching(unittest.TestCase):
    """Test output file generation and matching."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_parquet_output_matching(self):
        """Test Parquet output file generation and matching."""
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [0.1, 0.2, 0.3]
        })
        
        # Test Parquet loading
        parquet_path = self.output_dir / "test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(test_df)
        
        # Verify file exists
        self.assertTrue(parquet_path.exists())
        
        # Verify content matches
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(test_df, loaded_df)
    
    def test_csv_output_matching(self):
        """Test CSV output file generation and matching."""
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [0.1, 0.2, 0.3]
        })
        
        # Test CSV loading
        csv_path = self.output_dir / "test.csv"
        csv_loader = SummaryLoader(str(csv_path))
        csv_loader.load(test_df)
        
        # Verify file exists
        self.assertTrue(csv_path.exists())
        
        # Verify content matches
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(test_df, loaded_df)
    
    def test_large_dataset_output_matching(self):
        """Test output matching with large dataset."""
        # Create large test data
        large_df = pd.DataFrame({
            'id': range(1000),
            'value': [f'value_{i}' for i in range(1000)],
            'score': [i * 0.001 for i in range(1000)]
        })
        
        # Test Parquet output
        parquet_path = self.output_dir / "large_test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(large_df)
        
        # Verify content matches
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(large_df, loaded_df)
        
        # Test CSV output
        csv_path = self.output_dir / "large_test.csv"
        csv_loader = SummaryLoader(str(csv_path))
        csv_loader.load(large_df)
        
        # Verify content matches
        loaded_csv_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(large_df, loaded_csv_df)
    
    def test_complex_data_output_matching(self):
        """Test output matching with complex data types."""
        # Create complex test data
        complex_df = pd.DataFrame({
            'string_col': ['a', 'b', 'c'],
            'int_col': [1, 2, 3],
            'float_col': [0.1, 0.2, 0.3],
            'bool_col': [True, False, True],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'null_col': [None, 'value', None]
        })
        
        # Test Parquet output
        parquet_path = self.output_dir / "complex_test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(complex_df)
        
        # Verify content matches
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(complex_df, loaded_df)
    
    def test_output_file_permissions(self):
        """Test output file permissions and accessibility."""
        # Create test data
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Test Parquet output
        parquet_path = self.output_dir / "permissions_test.parquet"
        parquet_loader = ParquetLoader(str(parquet_path))
        parquet_loader.load(test_df)
        
        # Verify file is readable
        self.assertTrue(parquet_path.exists())
        self.assertTrue(parquet_path.is_file())
        
        # Verify file can be read
        loaded_df = pd.read_parquet(parquet_path)
        self.assertEqual(len(loaded_df), 3)


class TestSummaryGeneration(unittest.TestCase):
    """Test summary generation and statistics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_summary_generation_basic(self):
        """Test basic summary generation."""
        # Create test merged data
        merged_df = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P002'],
            'study_code': ['S001', 'S001', 'S002'],
            'study_cohort_code': ['C001', 'C001', 'C002'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'detection_value': [0.5, 0.8, 0.2],
            'sample_status': ['Finished', 'Running', 'Finished']
        })
        
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
    
    def test_summary_generation_with_no_finished_samples(self):
        """Test summary generation with no finished samples."""
        # Create test data with no finished samples
        merged_df = pd.DataFrame({
            'project_code': ['P001'],
            'study_code': ['S001'],
            'study_cohort_code': ['C001'],
            'sample_id': ['SAMP001'],
            'detection_value': [0.5],
            'sample_status': ['Running']
        })
        
        # Generate summary
        summary_transformer = SummaryTransformer()
        summary_df = summary_transformer.transform(merged_df)
        
        # Verify summary handles no finished samples
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]['Finished_Percentage'], 0.0)
        self.assertEqual(summary_df.iloc[0]['Lowest_Detection'], 0.5)
    
    def test_summary_generation_with_all_finished_samples(self):
        """Test summary generation with all finished samples."""
        # Create test data with all finished samples
        merged_df = pd.DataFrame({
            'project_code': ['P001', 'P001'],
            'study_code': ['S001', 'S001'],
            'study_cohort_code': ['C001', 'C001'],
            'sample_id': ['SAMP001', 'SAMP002'],
            'detection_value': [0.5, 0.8],
            'sample_status': ['Finished', 'Finished']
        })
        
        # Generate summary
        summary_transformer = SummaryTransformer()
        summary_df = summary_transformer.transform(merged_df)
        
        # Verify summary handles all finished samples
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]['Finished_Percentage'], 100.0)
        self.assertEqual(summary_df.iloc[0]['Lowest_Detection'], 0.5)
    
    def test_summary_generation_with_mixed_statuses(self):
        """Test summary generation with mixed sample statuses."""
        # Create test data with mixed statuses
        merged_df = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P001'],
            'study_code': ['S001', 'S001', 'S001'],
            'study_cohort_code': ['C001', 'C001', 'C001'],
            'sample_id': ['SAMP001', 'SAMP002', 'SAMP003'],
            'detection_value': [0.5, 0.8, 0.2],
            'sample_status': ['Finished', 'Running', 'Failed']
        })
        
        # Generate summary
        summary_transformer = SummaryTransformer()
        summary_df = summary_transformer.transform(merged_df)
        
        # Verify summary calculations
        self.assertEqual(len(summary_df), 1)
        row = summary_df.iloc[0]
        self.assertEqual(row['Total_Samples'], 3)
        self.assertEqual(row['Finished_Percentage'], 33.33)  # 1 out of 3
        self.assertEqual(row['Lowest_Detection'], 0.2)
    
    def test_summary_generation_with_empty_data(self):
        """Test summary generation with empty data."""
        # Create empty dataframe
        empty_df = pd.DataFrame(columns=[
            'project_code', 'study_code', 'study_cohort_code',
            'sample_id', 'detection_value', 'sample_status'
        ])
        
        # Generate summary
        summary_transformer = SummaryTransformer()
        summary_df = summary_transformer.transform(empty_df)
        
        # Verify summary handles empty data
        self.assertEqual(len(summary_df), 0)


if __name__ == '__main__':
    unittest.main()
