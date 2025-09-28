"""
Dynamic Data Transformer

Core dynamic transformer that leverages TransformationRecipe for schema-driven transformation.
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from pandera.errors import SchemaError, SchemaErrors

from .base_transformer import BaseTransformer
from .transformation_recipe import TransformationRecipe
from utils.schemas.schemas import get_schema
from ..utils.reporting.validation_reporter import ValidationReporter
from utils.exceptions.pipeline_exceptions import DataCleaningException

logger = logging.getLogger(__name__)


class DynamicDataTransformer(BaseTransformer):
    """
    Dynamic data transformer using schema-driven transformation recipes.
    
    This transformer orchestrates pre-validation, recipe generation, recipe execution,
    and post-validation using Pandera schemas and TransformationRecipe.
    """
    
    def __init__(self, config: Dict[str, Any] = None, file_type: str = None):
        """
        Initialize the dynamic data transformer.
        
        Args:
            config: Configuration dictionary
            file_type: Type of file being transformed ('projects', 'subjects', 'results')
        """
        super().__init__(config)
        self.file_type = file_type
        if not self.file_type:
            raise ValueError("file_type must be provided")
        
        self.recipe_generator = TransformationRecipe(file_type, config)
        self.validation_reporter = ValidationReporter()
        self.pre_validation_errors = []
        self.post_validation_errors = []
        self.validation_report = {}
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame using the validation-recipe-transformation-validation flow.
        
        This method implements the core validation pipeline flow:
        1. PRE-VALIDATION: Identify data quality issues using Pandera schemas
        2. RECIPE GENERATION: Create transformation recipe based on validation errors
        3. DATA TRANSFORMATION: Apply the generated recipe to fix identified issues
        4. POST-VALIDATION: Verify that transformation successfully resolved the issues
        
        The key principle: Fix the DATA to match the schema, never change the schema.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame with improved data quality
        """
        if df.empty:
            logger.warning(f"⚠️ DataFrame is empty for {self.file_type}")
            return df
        
        logger.info(f"🔄 Starting dynamic transformation for {self.file_type}: {len(df)} records")
        
        try:
            # Start validation reporting
            self.validation_reporter.start_validation_run()
            
            # Step 0: Apply column mapping first (if available)
            df = self._apply_column_mapping(df)
            
            # Step 1: Pre-validation
            self._perform_pre_validation(df)
            
            # Step 2: Generate and execute transformation recipe
            cleaned_df = self._execute_transformation_recipe(df)
            
            # Step 3: Post-validation
            self._perform_post_validation(cleaned_df)
            
            # Step 4: Generate validation reports
            self._generate_validation_report(df, cleaned_df)
            
            # End validation reporting
            self.validation_reporter.end_validation_run()
            
            logger.info(f"✅ Dynamic transformation completed for {self.file_type}: {len(cleaned_df)} records")
            return cleaned_df
            
        except Exception as e:
            error_msg = f"Unexpected error during {self.file_type} transformation: {e}"
            logger.error(f"❌ {error_msg}")
            raise DataCleaningException(error_msg, self.file_type, str(e))
    
    def _apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping from config if available."""
        try:
            column_mappings = self.config.get('data', {}).get('column_mappings', {}).get(self.file_type, {})
            if column_mappings:
                logger.debug(f"  ➤ Applying column mapping for {self.file_type}")
                df = df.rename(columns=column_mappings)
                logger.debug(f"    ✓ Column mapping applied: {len(column_mappings)} mappings")
            return df
        except Exception as e:
            logger.warning(f"    ⚠️ Error in column mapping: {e}")
            return df

    def _perform_pre_validation(self, df: pd.DataFrame) -> None:
        """Perform validation before transformation."""
        logger.debug(f"  ➤ Pre-validation for {self.file_type}")
        
        try:
            schema = get_schema(self.file_type)
            schema.validate(df, lazy=True)
            logger.debug(f"    ✓ Pre-validation passed")
            self.pre_validation_errors = []
            # Add successful pre-validation result
            self.validation_reporter.add_validation_success({
                'table_name': self.file_type,
                'stage': 'before_cleaning',
                'total_rows': len(df),
                'valid_rows': len(df),
                'invalid_rows': 0,
                'data_quality_score': 100.0
            })
        except SchemaErrors as e:
            self.pre_validation_errors = self._parse_schema_errors(e, df)
            error_details = self._parse_schema_errors_to_details(e, df)
            column_status = self._parse_schema_errors_to_column_status(e, df)
            logger.debug(f"    ⚠️ Pre-validation issues: {len(self.pre_validation_errors)} errors")
            # Add pre-validation errors to reporter
            for schema_error in e.schema_errors:
                self.validation_reporter.add_validation_error({
                    'table_name': self.file_type,
                    'column_name': getattr(schema_error, 'column', 'Unknown'),
                    'error_type': type(schema_error).__name__,
                    'error_message': str(schema_error),
                    'stage': 'before_cleaning',
                    'schema_definition': self._get_schema_definition(),
                    'actual_value': self._get_actual_value(df, getattr(schema_error, 'column', None)),
                    'validation_rule': self._get_validation_rule(schema_error),
                    'pandera_exception': {
                        'exception_type': type(schema_error).__name__,
                        'exception_message': str(schema_error),
                        'check_name': getattr(schema_error, 'check', 'Unknown')
                    }
                })
        except Exception as e:
            logger.debug(f"    ⚠️ Pre-validation error: {e}")
            self.pre_validation_errors = [f"Unexpected pre-validation error: {e}"]
            # Add unexpected error to reporter
            self.validation_reporter.add_validation_error({
                'table_name': self.file_type,
                'column_name': 'Unknown',
                'error_type': 'UnexpectedError',
                'error_message': str(e),
                'stage': 'before_cleaning',
                'pandera_exception': {
                    'exception_type': 'UnexpectedError',
                    'exception_message': str(e)
                }
            })
        else:
            # This case is already handled in the try block above
            pass
    
    def _execute_transformation_recipe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate and execute transformation recipe."""
        logger.debug(f"  ➤ Executing transformation recipe for {self.file_type}")
        
        try:
            # Generate recipe based on schema and data
            recipe = self.recipe_generator.generate_recipe(df)
            
            # Execute the recipe
            cleaned_df = self.recipe_generator.execute_recipe(df)
            
            logger.debug(f"    ✓ Transformation recipe executed successfully")
            return cleaned_df
            
        except Exception as e:
            error_msg = f"Recipe execution failed for {self.file_type}: {e}"
            logger.error(f"    ❌ {error_msg}")
            raise DataCleaningException(error_msg, self.file_type, str(e))
    
    def _perform_post_validation(self, df: pd.DataFrame) -> None:
        """Perform validation after transformation."""
        logger.debug(f"  ➤ Post-validation for {self.file_type}")
        
        try:
            schema = get_schema(self.file_type)
            schema.validate(df, lazy=True)
            logger.info(f"✅ Schema validation passed for {self.file_type}")
            self.post_validation_errors = []
            # Add successful post-validation result
            self.validation_reporter.add_validation_success({
                'table_name': self.file_type,
                'stage': 'after_cleaning',
                'total_rows': len(df),
                'valid_rows': len(df),
                'invalid_rows': 0,
                'data_quality_score': 100.0
            })
        except SchemaErrors as e:
            self.post_validation_errors = self._parse_schema_errors(e, df)
            error_details = self._parse_schema_errors_to_details(e, df)
            column_status = self._parse_schema_errors_to_column_status(e, df)
            logger.warning(f"⚠️ Schema validation issues for {self.file_type}: {', '.join(error[:100] for error in self.post_validation_errors[:3])}")
            # Add post-validation errors to reporter
            for schema_error in e.schema_errors:
                self.validation_reporter.add_validation_error({
                    'table_name': self.file_type,
                    'column_name': getattr(schema_error, 'column', 'Unknown'),
                    'error_type': type(schema_error).__name__,
                    'error_message': str(schema_error),
                    'stage': 'after_cleaning',
                    'schema_definition': self._get_schema_definition(),
                    'actual_value': self._get_actual_value(df, getattr(schema_error, 'column', None)),
                    'validation_rule': self._get_validation_rule(schema_error),
                    'pandera_exception': {
                        'exception_type': type(schema_error).__name__,
                        'exception_message': str(schema_error),
                        'check_name': getattr(schema_error, 'check', 'Unknown')
                    }
                })
        except Exception as e:
            logger.warning(f"⚠️ Post-validation error for {self.file_type}: {e}")
            self.post_validation_errors = [f"Unexpected post-validation error: {e}"]
            # Add unexpected post-validation error to reporter
            self.validation_reporter.add_validation_error({
                'table_name': self.file_type,
                'column_name': 'Unknown',
                'error_type': 'UnexpectedError',
                'error_message': str(e),
                'stage': 'after_cleaning',
                'pandera_exception': {
                    'exception_type': 'UnexpectedError',
                    'exception_message': str(e)
                }
            })
        else:
            # This case is already handled in the try block above
            pass
    
    def _parse_schema_errors(self, error: SchemaErrors, df: pd.DataFrame) -> List[str]:
        """
        Parse Pandera SchemaErrors into a list of informative strings.
        
        Args:
            error: The SchemaErrors exception
            df: The DataFrame that was being validated
            
        Returns:
            List of error messages
        """
        error_messages = []
        
        try:
            # Handle different types of schema errors
            error_str = str(error)
            
            # Check for missing columns
            if "column" in error_str and "not in dataframe" in error_str:
                error_messages.append(error_str)
            elif hasattr(error, 'failure_cases') and error.failure_cases is not None:
                # Handle failure cases
                failure_cases = error.failure_cases
                if hasattr(failure_cases, 'empty') and not failure_cases.empty:
                    for _, row in failure_cases.iterrows():
                        column = getattr(row, 'column', 'Unknown')
                        check = getattr(row, 'check', 'Unknown check')
                        case = getattr(row, 'failure_case', 'No details')
                        error_messages.append(f"Column '{column}' failed check '{check}': {case}")
                else:
                    error_messages.append(str(error))
            else:
                error_messages.append(str(error))
                
        except Exception as parse_error:
            logger.debug(f"Error parsing schema error: {parse_error}")
            error_messages.append(f"Schema validation failed: {error}")
        
        return error_messages
    
    def _parse_schema_errors_to_details(self, error: SchemaErrors, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Parse Pandera SchemaErrors into detailed error information.
        
        Args:
            error: The SchemaErrors exception
            df: The DataFrame that was being validated
            
        Returns:
            List of detailed error dictionaries
        """
        error_details = []
        
        try:
            error_str = str(error)
            
            # Check for missing columns
            if "column" in error_str and "not in dataframe" in error_str:
                error_details.append({
                    'error_type': 'MissingColumn',
                    'column': 'Unknown',
                    'error_message': error_str,
                    'check_name': 'ColumnExists',
                    'failed_cases_count': 1
                })
            elif hasattr(error, 'failure_cases') and error.failure_cases is not None:
                # Handle failure cases
                failure_cases = error.failure_cases
                if hasattr(failure_cases, 'empty') and not failure_cases.empty:
                    for _, row in failure_cases.iterrows():
                        column = getattr(row, 'column', 'Unknown')
                        check = getattr(row, 'check', 'Unknown check')
                        case = getattr(row, 'failure_case', 'No details')
                        
                        error_details.append({
                            'error_type': 'ValidationError',
                            'column': column,
                            'error_message': f"Column '{column}' failed check '{check}': {case}",
                            'check_name': str(check),
                            'failed_cases_count': 1
                        })
                else:
                    error_details.append({
                        'error_type': 'SchemaError',
                        'column': 'Unknown',
                        'error_message': str(error),
                        'check_name': 'SchemaValidation',
                        'failed_cases_count': 1
                    })
            else:
                error_details.append({
                    'error_type': 'SchemaError',
                    'column': 'Unknown',
                    'error_message': str(error),
                    'check_name': 'SchemaValidation',
                    'failed_cases_count': 1
                })
                
        except Exception as parse_error:
            logger.debug(f"Error parsing schema error: {parse_error}")
            error_details.append({
                'error_type': 'ParseError',
                'column': 'Unknown',
                'error_message': f"Schema validation failed: {error}",
                'check_name': 'Unknown',
                'failed_cases_count': 1
            })
        
        return error_details
    
    def _parse_schema_errors_to_column_status(self, error: SchemaErrors, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Parse Pandera SchemaErrors into detailed column-level status information.
        
        Args:
            error: The SchemaErrors exception
            df: The DataFrame that was being validated
            
        Returns:
            Dictionary with column-level status information
        """
        column_status = {}
        
        try:
            # Get all columns in the DataFrame
            df_columns = set(df.columns.tolist())
            
            # Get schema columns from the schema
            schema = get_schema(self.file_type)
            schema_columns = set(schema.columns.keys())
            
            # Check for missing columns
            missing_columns = schema_columns - df_columns
            for col in missing_columns:
                column_status[col] = {
                    'is_valid': False,
                    'error_count': 1,
                    'error_type': 'MissingColumn',
                    'error_message': f"Column '{col}' not found in dataframe",
                    'check_name': 'ColumnExists',
                    'validation_details': [{
                        'stage': 'column_existence',
                        'expected': True,
                        'actual': False,
                        'error_code': 'COLUMN_MISSING'
                    }]
                }
            
            # Check for extra columns
            extra_columns = df_columns - schema_columns
            for col in extra_columns:
                column_status[col] = {
                    'is_valid': False,
                    'error_count': 1,
                    'error_type': 'ExtraColumn',
                    'error_message': f"Unexpected column '{col}' found in dataframe",
                    'check_name': 'ColumnExists',
                    'validation_details': [{
                        'stage': 'column_existence',
                        'expected': False,
                        'actual': True,
                        'error_code': 'EXTRA_COLUMN'
                    }]
                }
            
            # Parse failure cases for validation errors
            if hasattr(error, 'failure_cases') and error.failure_cases is not None:
                failure_cases = error.failure_cases
                if hasattr(failure_cases, 'empty') and not failure_cases.empty:
                    for _, row in failure_cases.iterrows():
                        column = getattr(row, 'column', 'Unknown')
                        check = getattr(row, 'check', 'Unknown check')
                        case = getattr(row, 'failure_case', 'No details')
                        index = getattr(row, 'index', None)
                        
                        if column not in column_status:
                            column_status[column] = {
                                'is_valid': False,
                                'error_count': 0,
                                'error_type': 'ValidationError',
                                'error_message': f"Column '{column}' failed check '{check}': {case}",
                                'check_name': str(check),
                                'validation_details': []
                            }
                        
                        column_status[column]['error_count'] += 1
                        column_status[column]['validation_details'].append({
                            'stage': 'value_validation',
                            'check_name': str(check),
                            'failed_value': str(case),
                            'row_index': int(index) if index is not None else None,
                            'error_code': 'VALIDATION_FAILED'
                        })
            
            # Mark valid columns (present in both schema and dataframe)
            valid_columns = df_columns & schema_columns
            for col in valid_columns:
                if col not in column_status:  # Only if no errors were found
                    column_status[col] = {
                        'is_valid': True,
                        'error_count': 0,
                        'error_type': None,
                        'error_message': None,
                        'check_name': None,
                        'validation_details': [{
                            'stage': 'column_existence',
                            'expected': True,
                            'actual': True,
                            'error_code': 'SUCCESS'
                        }]
                    }
                    
        except Exception as parse_error:
            logger.debug(f"Error parsing schema errors to column status: {parse_error}")
            # Fallback: mark all columns as having unknown status
            for col in df.columns:
                column_status[col] = {
                    'is_valid': False,
                    'error_count': 1,
                    'error_type': 'ParseError',
                    'error_message': f"Error parsing validation results: {parse_error}",
                    'check_name': 'Unknown',
                    'validation_details': [{
                        'stage': 'parsing',
                        'error_code': 'PARSE_ERROR'
                    }]
                }
        
        return column_status
    
    def _get_all_columns_status(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all columns in the DataFrame, marking them as valid.
        
        Args:
            df: The DataFrame to analyze
            
        Returns:
            Dictionary with column-level status information for all columns
        """
        column_status = {}
        
        try:
            # Get schema columns from the schema
            schema = get_schema(self.file_type)
            schema_columns = set(schema.columns.keys())
            
            # Get all columns in the DataFrame
            df_columns = set(df.columns.tolist())
            
            # Mark all columns as valid
            for col in df_columns:
                column_status[col] = {
                    'is_valid': True,
                    'error_count': 0,
                    'error_type': None,
                    'error_message': None,
                    'check_name': None,
                    'validation_details': [{
                        'stage': 'column_existence',
                        'expected': True,
                        'actual': True,
                        'error_code': 'SUCCESS'
                    }]
                }
            
            # Check for missing schema columns
            missing_columns = schema_columns - df_columns
            for col in missing_columns:
                column_status[col] = {
                    'is_valid': False,
                    'error_count': 1,
                    'error_type': 'MissingColumn',
                    'error_message': f"Column '{col}' not found in dataframe",
                    'check_name': 'ColumnExists',
                    'validation_details': [{
                        'stage': 'column_existence',
                        'expected': True,
                        'actual': False,
                        'error_code': 'COLUMN_MISSING'
                    }]
                }
                
        except Exception as e:
            logger.debug(f"Error getting all columns status: {e}")
            # Fallback: mark all columns as having unknown status
            for col in df.columns:
                column_status[col] = {
                    'is_valid': False,
                    'error_count': 1,
                    'error_type': 'ParseError',
                    'error_message': f"Error getting column status: {e}",
                    'check_name': 'Unknown',
                    'validation_details': [{
                        'stage': 'parsing',
                        'error_code': 'PARSE_ERROR'
                    }]
                }
        
        return column_status
    
    def _generate_validation_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
        """Generate a comprehensive validation report."""
        self.validation_report = {
            'file_type': self.file_type,
            'original_data': {
                'shape': original_df.shape,
                'columns': original_df.columns.tolist()
            },
            'cleaned_data': {
                'shape': cleaned_df.shape,
                'columns': cleaned_df.columns.tolist()
            },
            'pre_validation': {
                'error_count': len(self.pre_validation_errors),
                'errors': self.pre_validation_errors[:10]  # Limit to first 10 errors
            },
            'post_validation': {
                'error_count': len(self.post_validation_errors),
                'errors': self.post_validation_errors[:10]  # Limit to first 10 errors
            },
            'transformation_summary': {
                'rows_removed': len(original_df) - len(cleaned_df),
                'columns_added': len(cleaned_df.columns) - len(original_df.columns),
                'is_valid': len(self.post_validation_errors) == 0
            }
        }
        
        logger.debug(f"  ➤ Generated validation report for {self.file_type}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get the validation report from the last cleaning operation.
        
        Returns:
            Dictionary containing validation report
        """
        return self.validation_report
    
    def get_pre_validation_errors(self) -> List[str]:
        """Get pre-validation errors."""
        return self.pre_validation_errors
    
    def get_post_validation_errors(self) -> List[str]:
        """Get post-validation errors."""
        return self.post_validation_errors
    
    def is_valid(self) -> bool:
        """Check if the last cleaning operation resulted in valid data."""
        return len(self.post_validation_errors) == 0
    
    def get_validation_reporter(self) -> ValidationReporter:
        """Get the validation reporter."""
        return self.validation_reporter
    
    def print_validation_summary(self):
        """Print validation summary to console."""
        self.validation_reporter.print_summary()
    
    def _get_schema_definition(self) -> Dict[str, Any]:
        """Get schema definition for the current file type."""
        try:
            schema = get_schema(self.file_type)
            return {
                'columns': list(schema.columns.keys()),
                'dtypes': {col: str(schema.columns[col].dtype) for col in schema.columns},
                'checks': {col: str(schema.columns[col].checks) for col in schema.columns if schema.columns[col].checks}
            }
        except Exception:
            return {}
    
    def _get_actual_value(self, df: pd.DataFrame, column: Optional[str]) -> Any:
        """Get actual value from DataFrame for error reporting."""
        if column and column in df.columns:
            # Return a sample of the actual values
            sample_values = df[column].dropna().head(3).tolist()
            return sample_values if sample_values else None
        return None
    
    def _get_validation_rule(self, schema_error) -> Dict[str, Any]:
        """Get validation rule information from schema error."""
        try:
            return {
                'check_name': getattr(schema_error, 'check', 'Unknown'),
                'check_description': str(getattr(schema_error, 'check', 'Unknown')),
                'constraint': getattr(schema_error, 'constraint', None)
            }
        except Exception:
            return {'check_name': 'Unknown', 'check_description': 'Unknown'}