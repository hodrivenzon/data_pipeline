#!/usr/bin/env python3
"""
Test Cancer Detected Validation

Comprehensive tests for cancer_detected field validation with Yes/No only values.
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas.schemas import get_schema, get_schema_metadata
from config.config_manager import ConfigManager
from core.transform.dynamic_data_transformer import DynamicDataTransformer
from core.transform.transformation_recipe import TransformationRecipe


class TestCancerDetectedValidation:
    """Test cancer_detected field validation with Yes/No only values."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager(str(project_root / "config.yaml"))
        self.config = self.config_manager.config
        
    def test_schema_allows_only_yes_no(self):
        """Test that schema only allows Yes and No values."""
        results_schema = get_schema('results')
        
        # Test valid values (include all required columns)
        valid_data = pd.DataFrame({
            'sample_id': ['S001', 'S002'],
            'cancer_detected': ['Yes', 'No'],
            'detection_value': [0.5, 0.3],
            'sample_status': ['Finished', 'Running'],
            'fail_reason': [None, None],
            'sample_quality': [0.8, 0.9],
            'sample_quality_threshold': [0.7, 0.8],
            'date_of_run': [None, None]
        })
        
        # This should pass validation
        try:
            results_schema.validate(valid_data, lazy=True)
            print("✅ Valid Yes/No values passed schema validation")
        except Exception as e:
            pytest.fail(f"Valid Yes/No values failed validation: {e}")
        
        # Test invalid values (include all required columns)
        invalid_data = pd.DataFrame({
            'sample_id': ['S001', 'S002', 'S003'],
            'cancer_detected': ['Yes', 'Y', 'True'],  # Y and True should be invalid
            'detection_value': [0.5, 0.3, 0.7],
            'sample_status': ['Finished', 'Running', 'Failed'],
            'fail_reason': [None, None, None],
            'sample_quality': [0.8, 0.9, 0.6],
            'sample_quality_threshold': [0.7, 0.8, 0.5],
            'date_of_run': [None, None, None]
        })
        
        # This should fail validation
        try:
            results_schema.validate(invalid_data, lazy=True)
            pytest.fail("Invalid values should have failed schema validation")
        except Exception as e:
            print("✅ Invalid values correctly failed schema validation")
            assert "cancer_detected" in str(e)
    
    def test_config_value_mapping_standardization(self):
        """Test that config value mappings standardize to Yes/No only."""
        # Test various input values and their expected outputs
        test_cases = [
            ('Yes', 'Yes'),
            ('No', 'No'),
            ('Y', 'Yes'),
            ('N', 'No'),
            ('True', 'Yes'),
            ('False', 'No'),
            ('1', 'Yes'),
            ('0', 'No'),
            ('yes', 'Yes'),
            ('no', 'No'),
            ('NA', None),
            ('nan', None),
            ('InvalidValue', 'InvalidValue')  # Should remain unchanged if not in mapping
        ]
        
        # Get value mappings from config
        value_mappings = self.config_manager.get('data.value_mappings.results.cancer_detected', {})
        
        for input_val, expected_output in test_cases:
            if input_val in value_mappings:
                mapped_value = value_mappings[input_val]
                assert mapped_value == expected_output, f"Mapping failed for {input_val}: expected {expected_output}, got {mapped_value}"
                print(f"✅ {input_val} → {mapped_value}")
            else:
                print(f"⚠️ No mapping found for {input_val}")
    
    def test_constraint_validation_with_transformer(self):
        """Test constraint validation with the data transformer."""
        # Create test data with various cancer_detected values
        test_data = pd.DataFrame({
            'sample_id': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007'],
            'cancer_detected': ['Yes', 'No', 'Y', 'N', 'True', 'False', 'InvalidValue'],
            'detection_value': [0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.6]
        })
        
        print(f"📊 Initial data: {len(test_data)} records")
        print(f"📊 Initial cancer_detected values: {test_data['cancer_detected'].unique()}")
        
        # Test transformer
        transformer = DynamicDataTransformer(self.config, 'results')
        
        try:
            cleaned_data = transformer.transform(test_data)
            
            print(f"📊 Cleaned data: {len(cleaned_data)} records")
            print(f"📊 Cleaned cancer_detected values: {cleaned_data['cancer_detected'].unique()}")
            
            # Check that only Yes/No values remain
            if 'cancer_detected' in cleaned_data.columns:
                valid_cancer = cleaned_data['cancer_detected'].dropna()
                if not valid_cancer.empty:
                    allowed_values = ['Yes', 'No']
                    invalid_values = valid_cancer[~valid_cancer.isin(allowed_values)]
                    
                    if len(invalid_values) > 0:
                        print(f"❌ Invalid cancer_detected values still present: {invalid_values.unique()}")
                        pytest.fail(f"Invalid values found: {invalid_values.unique()}")
                    else:
                        print("✅ All cancer_detected values are Yes or No only")
                        
                        # Check that values were properly standardized
                        unique_values = set(valid_cancer.unique())
                        expected_values = {'Yes', 'No'}
                        assert unique_values.issubset(expected_values), f"Unexpected values found: {unique_values - expected_values}"
                        print("✅ All values are properly standardized to Yes/No")
            
        except Exception as e:
            print(f"❌ Transformer test failed: {e}")
            pytest.fail(f"Transformer failed: {e}")
    
    def test_schema_constraint_extraction(self):
        """Test that schema constraints are properly extracted."""
        recipe = TransformationRecipe('results', self.config)
        analysis = recipe.analyze_schema()
        
        # Check that cancer_detected constraints are extracted
        assert 'cancer_detected' in analysis['constraints']
        
        cancer_constraints = analysis['constraints']['cancer_detected']
        assert 'allowed_values' in cancer_constraints
        
        # Verify only Yes/No are allowed
        allowed_values = set(cancer_constraints['allowed_values'])
        expected_values = {'Yes', 'No'}
        assert allowed_values == expected_values, f"Expected {expected_values}, got {allowed_values}"
        
        print("✅ Schema constraint extraction working correctly")
        print(f"📊 Allowed values: {allowed_values}")
    
    def test_config_constraint_definitions(self):
        """Test that config constraint definitions are correct."""
        constraints = self.config_manager.get('data.constraints.results.cancer_detected', {})
        
        assert 'allowed_values' in constraints
        assert 'description' in constraints
        
        allowed_values = set(constraints['allowed_values'])
        expected_values = {'Yes', 'No'}
        assert allowed_values == expected_values, f"Expected {expected_values}, got {allowed_values}"
        
        assert constraints['description'] == "Cancer detected must be Yes or No only"
        
        print("✅ Config constraint definitions are correct")
        print(f"📊 Allowed values: {allowed_values}")
        print(f"📊 Description: {constraints['description']}")
    
    def test_value_mapping_completeness(self):
        """Test that value mappings cover all expected input values."""
        value_mappings = self.config_manager.get('data.value_mappings.results.cancer_detected', {})
        
        # Expected input values that should be mapped
        expected_inputs = [
            'Yes', 'No', 'Y', 'N', 'True', 'False', '1', '0', 'yes', 'no',
            'NA', 'nan', 'NaN', 'N/A', 'n/a'
        ]
        
        missing_mappings = []
        for input_val in expected_inputs:
            if input_val not in value_mappings:
                missing_mappings.append(input_val)
        
        if missing_mappings:
            print(f"❌ Missing value mappings: {missing_mappings}")
            pytest.fail(f"Missing mappings for: {missing_mappings}")
        else:
            print("✅ All expected input values have mappings")
        
        # Check that all mappings result in Yes, No, or null
        for input_val, mapped_val in value_mappings.items():
            if mapped_val is not None:
                assert mapped_val in ['Yes', 'No'], f"Invalid mapping result: {input_val} → {mapped_val}"
        
        print("✅ All mappings result in valid values (Yes, No, or null)")
    
    def test_error_reporting_for_invalid_values(self):
        """Test that invalid values are properly reported in validation errors."""
        # Create data with invalid cancer_detected values
        invalid_data = pd.DataFrame({
            'sample_id': ['S001', 'S002', 'S003'],
            'cancer_detected': ['Yes', 'InvalidValue', 'Maybe'],
            'detection_value': [0.5, 0.3, 0.7],
            'sample_status': ['Finished', 'Running', 'Failed'],
            'fail_reason': [None, None, None],
            'sample_quality': [0.8, 0.9, 0.6],
            'sample_quality_threshold': [0.7, 0.8, 0.5],
            'date_of_run': [None, None, None]
        })
        
        results_schema = get_schema('results')
        
        try:
            results_schema.validate(invalid_data, lazy=True)
            pytest.fail("Should have failed validation")
        except Exception as e:
            error_str = str(e)
            print(f"📊 Validation error: {error_str}")
            
            # Check that the error mentions cancer_detected
            assert "cancer_detected" in error_str, "Error should mention cancer_detected field"
            
            # Check that invalid values are mentioned
            assert "InvalidValue" in error_str or "Maybe" in error_str, "Error should mention invalid values"
            
            print("✅ Validation errors properly report invalid cancer_detected values")


def run_manual_tests():
    """Run manual tests for cancer_detected validation."""
    print("🧪 Running Cancer Detected Validation Tests")
    print("=" * 60)
    
    test_instance = TestCancerDetectedValidation()
    test_instance.setup_method()
    
    tests_passed = 0
    total_tests = 7
    
    try:
        test_instance.test_schema_allows_only_yes_no()
        print("✅ Schema Yes/No validation test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Schema Yes/No validation test failed: {e}")
    
    try:
        test_instance.test_config_value_mapping_standardization()
        print("✅ Config value mapping test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Config value mapping test failed: {e}")
    
    try:
        test_instance.test_constraint_validation_with_transformer()
        print("✅ Transformer constraint validation test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Transformer constraint validation test failed: {e}")
    
    try:
        test_instance.test_schema_constraint_extraction()
        print("✅ Schema constraint extraction test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Schema constraint extraction test failed: {e}")
    
    try:
        test_instance.test_config_constraint_definitions()
        print("✅ Config constraint definitions test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Config constraint definitions test failed: {e}")
    
    try:
        test_instance.test_value_mapping_completeness()
        print("✅ Value mapping completeness test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Value mapping completeness test failed: {e}")
    
    try:
        test_instance.test_error_reporting_for_invalid_values()
        print("✅ Error reporting test passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Error reporting test failed: {e}")
    
    print(f"\n🎉 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎊 All tests passed! Cancer detected validation is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")


if __name__ == "__main__":
    run_manual_tests()
