"""
Tests for Extract Components

Test cases for data extraction components.
"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path

from core.extract import ProjectsExtractor, SubjectsExtractor, ResultsExtractor


class TestExtractors(unittest.TestCase):
    """Test the extraction components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test CSV files
        self.projects_file = self.test_dir / "projects.csv"
        self.subjects_file = self.test_dir / "subjects.csv"
        self.results_file = self.test_dir / "results.csv"
        
        self._create_test_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_files(self):
        """Create test CSV files."""
        # Projects data
        projects_data = pd.DataFrame({
            'project_code': ['P001', 'P002'],
            'study_code': ['S001', 'S002'],
            'study_cohort_code': ['C001', 'C002'],
            'project_name': ['Project One', 'Project Two'],
            'study_name': ['Study One', 'Study Two'],
            'study_cohort_name': ['Cohort One', 'Cohort Two'],
            'project_manager_name': ['Manager One', 'Manager Two'],
            'disease_name': ['Disease A', 'Disease B']
        })
        projects_data.to_csv(self.projects_file, index=False)
        
        # Subjects data
        subjects_data = pd.DataFrame({
            'project_code': ['P001', 'P001', 'P002'],
            'study_code': ['S001', 'S001', 'S002'],
            'study_cohort_code': ['C001', 'C001', 'C002'],
            'subject_id': ['SUB001', 'SUB002', 'SUB003'],
            'sample_id': ['SAM001', 'SAM002', 'SAM003'],
            'type': ['Blood', 'Tissue', 'Blood']
        })
        subjects_data.to_csv(self.subjects_file, index=False)
        
        # Results data
        results_data = pd.DataFrame({
            'sample_id': ['SAM001', 'SAM002', 'SAM003'],
            'detection_value': [0.75, 0.82, 0.65],
            'cancer_detected': ['Yes', 'Yes', 'No'],
            'sample_status': ['Finished', 'Finished', 'Completed'],
            'fail_reason': [None, None, None],
            'sample_quality': [0.95, 0.87, 0.92],
            'sample_quality_threshold': [0.8, 0.8, 0.8],
            'date_of_run': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        results_data.to_csv(self.results_file, index=False)
    
    def test_projects_extractor(self):
        """Test projects data extraction."""
        extractor = ProjectsExtractor(str(self.projects_file))
        df = extractor.extract()
        
        self.assertEqual(len(df), 2)
        self.assertIn('project_code', df.columns)
        self.assertIn('study_code', df.columns)
        self.assertEqual(df.iloc[0]['project_code'], 'P001')
    
    def test_subjects_extractor(self):
        """Test subjects data extraction."""
        extractor = SubjectsExtractor(str(self.subjects_file))
        df = extractor.extract()
        
        self.assertEqual(len(df), 3)
        self.assertIn('sample_id', df.columns)
        self.assertIn('subject_id', df.columns)
        self.assertEqual(df.iloc[0]['sample_id'], 'SAM001')
    
    def test_results_extractor(self):
        """Test results data extraction."""
        extractor = ResultsExtractor(str(self.results_file))
        df = extractor.extract()
        
        self.assertEqual(len(df), 3)
        self.assertIn('sample_id', df.columns)
        self.assertIn('detection_value', df.columns)
        self.assertEqual(df.iloc[0]['sample_id'], 'SAM001')
    
    def test_file_not_found(self):
        """Test handling of missing files."""
        with self.assertRaises(FileNotFoundError):
            extractor = ProjectsExtractor("nonexistent.csv")
            extractor.extract()


if __name__ == '__main__':
    unittest.main()
