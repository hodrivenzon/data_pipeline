"""
Transformation Recipe

Dynamic transformation recipe generator based on Pandera schemas.
Analyzes schema rules to generate transformation steps automatically.
"""

import pandas as pd
import logging
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path
from pandera.pandas import DataFrameSchema, Column

from utils.schemas.schemas import get_schema
from utils.exceptions.pipeline_exceptions import SchemaRuleException

logger = logging.getLogger(__name__)


class TransformationRecipe:
    """
    Generates dynamic transformation recipes based on Pandera schema analysis.
    
    This class analyzes Pandera schemas to create transformation steps including:
    - Pre/post validation
    - Column mapping
    - String standardization
    - Missing value handling
    - Type conversion
    - Constraint application
    - Duplicate removal
    """
    
    def __init__(self, file_type: str, config: Dict[str, Any] = None):
        """
        Initialize the transformation recipe generator.
        
        Args:
            file_type: Type of file/data ('projects', 'subjects', 'results')
            config: Configuration dictionary
        """
        self.file_type = file_type
        self.config = config or {}
        self.schema = get_schema(file_type)
        self.recipe = None
        self.output_dir = Path("file_store")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure pandas global options for date parsing
        self._configure_pandas_date_parsing()
    
    def _configure_pandas_date_parsing(self):
        """Configure pandas global options for date parsing to avoid warnings."""
        # Set pandas global options for date parsing
        pd.set_option('display.date_dayfirst', True)
        
        # Suppress specific date parsing warnings
        warnings.filterwarnings('ignore', 
                               message='Parsing dates in .* format when dayfirst=False.*',
                               category=UserWarning)
        
        # Suppress FutureWarning about DataFrame concatenation
        warnings.filterwarnings('ignore', 
                               message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*',
                               category=FutureWarning)
        
    def generate_recipe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a transformation recipe based on the schema and data.
        
        Args:
            df: DataFrame to analyze for recipe generation
            
        Returns:
            Dictionary containing the transformation recipe
        """
        logger.info(f"🔍 Generating transformation recipe for {self.file_type}")
        
        # Analyze schema to extract rules
        schema_analysis = self.analyze_schema()
        
        # Generate transformation steps
        recipe = {
            'metadata': {
                'file_type': self.file_type,
                'generated_at': datetime.now().isoformat(),
                'data_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
                'data_columns': df.columns.tolist(),
                'schema_columns': list(self.schema.columns.keys())
            },
            'transformation_steps': self._generate_transformation_steps(df, schema_analysis),
            'schema_analysis': schema_analysis,
            'column_mappings': self._generate_column_mappings(df)
        }
        
        self.recipe = recipe
        
        # Save recipe to file
        self._save_recipe_to_file(recipe)
        
        logger.info(f"✅ Generated transformation recipe with {len(recipe['transformation_steps'])} steps")
        return recipe
    
    def analyze_schema(self) -> Dict[str, Any]:
        """
        Analyze the Pandera schema to extract all rules and constraints.
        
        Returns:
            Dictionary containing schema analysis results
        """
        analysis = {
            'total_columns': len(self.schema.columns),
            'required_columns': [],
            'nullable_columns': [],
            'data_types': {},
            'constraints': {},
            'checks': {},
            'missing_critical_rules': []
        }
        
        # Get config-based constraints if available
        config_constraints = self.config.get('data.constraints', {}).get(self.file_type, {})
        
        for col_name, col_def in self.schema.columns.items():
            # Nullability
            if col_def.nullable:
                analysis['nullable_columns'].append(col_name)
            else:
                analysis['required_columns'].append(col_name)
            
            # Data types
            analysis['data_types'][col_name] = str(col_def.dtype)
            
            # Constraints and checks
            if col_def.checks:
                analysis['checks'][col_name] = []
                for check in col_def.checks:
                    check_info = {
                        'check_type': check.__class__.__name__,
                        'description': getattr(check, 'description', 'No description')
                    }
                    
                    # Extract specific check parameters from statistics
                    if hasattr(check, 'statistics') and check.statistics:
                        check_info.update(check.statistics)
                        
                        # Extract constraint values from statistics
                        if col_name not in analysis['constraints']:
                            analysis['constraints'][col_name] = {}
                        
                        # Map statistics to constraint keys
                        for stat_key, stat_value in check.statistics.items():
                            if stat_key == 'min_value':
                                analysis['constraints'][col_name]['min_value'] = stat_value
                            elif stat_key == 'max_value':
                                analysis['constraints'][col_name]['max_value'] = stat_value
                            elif stat_key == 'allowed_values':
                                analysis['constraints'][col_name]['allowed_values'] = stat_value
                            elif stat_key == 'greater_than':
                                analysis['constraints'][col_name]['greater_than'] = stat_value
                            elif stat_key == 'less_than':
                                analysis['constraints'][col_name]['less_than'] = stat_value
                    
                    analysis['checks'][col_name].append(check_info)
            else:
                # Check for missing critical rules
                if col_name in analysis['required_columns']:
                    dtype = analysis['data_types'][col_name]
                    if 'str' in dtype or 'object' in dtype:
                        # String columns should have length or pattern checks
                        if col_name not in ['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id']:
                            analysis['missing_critical_rules'].append({
                                'column': col_name,
                                'rule_type': 'string_validation',
                                'suggestion': 'Consider adding string length or pattern validation'
                            })
                    elif 'float' in dtype or 'int' in dtype:
                        # Numeric columns should have range checks
                        if col_name not in ['detection_value', 'sample_quality']:
                            analysis['missing_critical_rules'].append({
                                'column': col_name,
                                'rule_type': 'range_validation',
                                'suggestion': 'Consider adding min/max value constraints'
                            })
        
        # Merge config-based constraints with schema constraints
        for col_name, config_constraint in config_constraints.items():
            if col_name not in analysis['constraints']:
                analysis['constraints'][col_name] = {}
            
            # Add config constraints
            for constraint_type, constraint_value in config_constraint.items():
                if constraint_type != 'description':  # Skip description
                    analysis['constraints'][col_name][constraint_type] = constraint_value
        
        return analysis
    
    def _generate_transformation_steps(self, df: pd.DataFrame, schema_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate the sequence of transformation steps."""
        steps = []
        
        # Step 1: Pre-validation
        steps.append({
            'step_number': 1,
            'step_name': 'pre_validation',
            'description': 'Validate data before cleaning',
            'enabled': True,
            'parameters': {
                'schema_type': self.file_type,
                'lazy': True,
                'capture_errors': True
            }
        })
        
        # Step 2: Column mapping
        steps.append({
            'step_number': 2,
            'step_name': 'column_mapping',
            'description': 'Map and standardize column names',
            'enabled': True,
            'parameters': {
                'column_mappings': self._get_column_mappings_from_config()
            }
        })
        
        # Step 3: Value mapping and standardization
        value_mappings = self._get_value_mappings_from_config()
        if value_mappings:
            steps.append({
                'step_number': 3,
                'step_name': 'value_mapping',
                'description': 'Standardize and map data values',
                'enabled': True,
                'parameters': {
                    'value_mappings': value_mappings
                }
            })
        
        # Step 4: String standardization
        string_columns = [col for col, dtype in schema_analysis['data_types'].items() 
                         if 'str' in dtype or 'object' in dtype]
        if string_columns:
            steps.append({
                'step_number': 4,
                'step_name': 'string_standardization',
                'description': 'Clean and standardize string columns',
                'enabled': True,
                'parameters': {
                    'columns': string_columns,
                    'operations': ['strip', 'lower', 'remove_extra_spaces']
                }
            })
        
        # Step 5: Missing value handling
        steps.append({
            'step_number': 5,
            'step_name': 'missing_value_handling',
            'description': 'Handle missing values based on schema rules',
            'enabled': True,
            'parameters': {
                'required_columns': schema_analysis['required_columns'],
                'nullable_columns': schema_analysis['nullable_columns'],
                'default_strategies': {
                    'string': 'empty_string',
                    'numeric': 'none',
                    'datetime': 'none'
                }
            }
        })
        
        # Step 6: Type conversion
        steps.append({
            'step_number': 6,
            'step_name': 'type_conversion',
            'description': 'Convert data types to match schema',
            'enabled': True,
            'parameters': {
                'type_mappings': schema_analysis['data_types'],
                'coerce_errors': True
            }
        })
        
        # Step 7: Constraint application
        if schema_analysis['constraints']:
            steps.append({
                'step_number': 7,
                'step_name': 'constraint_application',
                'description': 'Apply schema constraints and checks',
                'enabled': True,
                'parameters': {
                    'constraints': schema_analysis['constraints']
                }
            })
        
        # Step 7: Duplicate removal
        key_columns = self._get_key_columns_for_file_type()
        if key_columns:
            steps.append({
                'step_number': 7,
                'step_name': 'duplicate_removal',
                'description': f'Remove duplicates based on key columns: {key_columns}',
                'enabled': True,
                'parameters': {
                    'subset': key_columns,
                    'keep': 'first'
                }
            })
        
        # Step 8: Post-validation
        steps.append({
            'step_number': 8,
            'step_name': 'post_validation',
            'description': 'Validate cleaned data against schema',
            'enabled': True,
            'parameters': {
                'schema_type': self.file_type,
                'lazy': True,
                'strict_mode': False
            }
        })
        
        return steps
    
    def _generate_column_mappings(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate column mappings between DataFrame and schema."""
        mappings = {}
        df_columns = df.columns.tolist()
        schema_columns = list(self.schema.columns.keys())
        
        # Direct mappings from config
        config_mappings = self._get_column_mappings_from_config()
        
        for schema_col in schema_columns:
            if schema_col in df_columns:
                mappings[schema_col] = schema_col
            elif schema_col in config_mappings:
                if config_mappings[schema_col] in df_columns:
                    mappings[schema_col] = config_mappings[schema_col]
            else:
                # Try to find fuzzy matches
                for df_col in df_columns:
                    if schema_col.lower().replace('_', '') in df_col.lower().replace('_', '').replace(' ', ''):
                        mappings[schema_col] = df_col
                        break
        
        return mappings
    
    def _get_column_mappings_from_config(self) -> Dict[str, str]:
        """Get column mappings from configuration."""
        # Get file-specific column mappings from the new config structure
        column_mappings = self.config.get('data', {}).get('column_mappings', {})
        file_mappings = column_mappings.get(self.file_type, {})
        # Handle None values from YAML config
        if file_mappings is None:
            file_mappings = {}
        logger.debug(f"    📋 Column mappings for {self.file_type}: {file_mappings}")
        return file_mappings
    
    def _get_value_mappings_from_config(self) -> Dict[str, Dict[str, str]]:
        """Get value mappings from configuration."""
        # Get file-specific value mappings from the new config structure
        value_mappings = self.config.get('data', {}).get('value_mappings', {})
        file_mappings = value_mappings.get(self.file_type, {})
        # Handle None values from YAML config
        if file_mappings is None:
            file_mappings = {}
        logger.debug(f"    📋 Value mappings for {self.file_type}: {file_mappings}")
        return file_mappings
    
    def _get_key_columns_for_file_type(self) -> List[str]:
        """Get key columns for duplicate detection based on file type."""
        key_mappings = {
            'projects': ['project_code', 'study_code', 'study_cohort_code'],
            'subjects': ['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id'],
            'results': ['sample_id']
        }
        return key_mappings.get(self.file_type, [])
    
    def _save_recipe_to_file(self, recipe: Dict[str, Any]) -> str:
        """Save the generated recipe to a JSON file."""
        # Create recipes subdirectory
        recipes_dir = self.output_dir / "recipes"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"transformation_recipe_{self.file_type}.json"
        file_path = recipes_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(recipe, f, indent=2, default=str)
            logger.info(f"📄 Saved transformation recipe to {file_path}")
            
            # Clean up old transformation recipes to prevent file overflow
            self._cleanup_old_recipes()
            
            return str(file_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not save transformation recipe: {e}")
            return ""
    
    def _cleanup_old_recipes(self):
        """Clean up old transformation recipe files, keeping only the latest 3."""
        try:
            from ..utils.reporting.report_cleaner import ReportCleaner
            cleaner = ReportCleaner(str(self.output_dir))
            cleaned_count = cleaner.clean_transformation_recipes()
            if cleaned_count > 0:
                logger.debug(f"🧹 Cleaned up {cleaned_count} old transformation recipe files")
        except Exception as e:
            logger.debug(f"⚠️ Could not cleanup old recipes: {e}")
    
    def execute_recipe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the generated transformation recipe on a DataFrame.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.recipe:
            self.generate_recipe(df)
        
        logger.info(f"🔄 Executing transformation recipe for {self.file_type}")
        
        transformed_df = df.copy()
        
        for step in self.recipe['transformation_steps']:
            if not step['enabled']:
                continue
                
            step_name = step['step_name']
            logger.debug(f"  ➤ Step {step['step_number']}: {step_name}")
            
            try:
                if step_name == 'pre_validation':
                    # Pre-validation is handled separately
                    pass
                elif step_name == 'column_mapping':
                    transformed_df = self._apply_column_mapping(transformed_df, step['parameters'])
                elif step_name == 'value_mapping':
                    transformed_df = self._apply_value_mapping(transformed_df, step['parameters'])
                elif step_name == 'string_standardization':
                    transformed_df = self._apply_string_standardization(transformed_df, step['parameters'])
                elif step_name == 'missing_value_handling':
                    transformed_df = self._apply_missing_value_handling(transformed_df, step['parameters'])
                elif step_name == 'type_conversion':
                    transformed_df = self._apply_type_conversion(transformed_df, step['parameters'])
                elif step_name == 'constraint_application':
                    transformed_df = self._apply_constraints(transformed_df, step['parameters'])
                elif step_name == 'duplicate_removal':
                    transformed_df = self._apply_duplicate_removal(transformed_df, step['parameters'])
                elif step_name == 'post_validation':
                    # Post-validation is handled separately
                    pass
                    
            except Exception as e:
                logger.warning(f"⚠️ Error in step {step_name}: {e}")
                continue
        
        logger.info(f"✅ Recipe execution completed for {self.file_type}")
        return transformed_df
    
    def _apply_column_mapping(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply column mappings."""
        mappings = params.get('column_mappings', {})
        
        # Direct mapping: CSV column names to schema column names
        rename_dict = {}
        for csv_col, schema_col in mappings.items():
            if csv_col in df.columns:
                rename_dict[csv_col] = schema_col
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.debug(f"    ✓ Renamed columns: {rename_dict}")
        
        return df
    
    def _apply_value_mapping(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply value mappings to standardize data values."""
        value_mappings = params.get('value_mappings', {})
        
        for column, mapping in value_mappings.items():
            if column in df.columns:
                # Apply value mapping for this column
                original_values = df[column].unique()
                mapped_count = 0
                
                for old_value, new_value in mapping.items():
                    # Handle null values specially
                    if new_value is None:
                        # Convert to pandas NaN for proper null handling
                        mask = df[column].astype(str).str.lower() == old_value.lower()
                        if mask.any():
                            df.loc[mask, column] = pd.NA
                            mapped_count += mask.sum()
                    else:
                        # Apply case-insensitive mapping for non-null values
                        mask = df[column].astype(str).str.lower() == old_value.lower()
                        if mask.any():
                            df.loc[mask, column] = new_value
                            mapped_count += mask.sum()
                
                if mapped_count > 0:
                    logger.debug(f"    ✓ Mapped {mapped_count} values in column '{column}'")
        
        return df
    
    def _apply_string_standardization(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply string standardization operations."""
        columns = params.get('columns', [])
        operations = params.get('operations', [])
        
        # Get columns that have value mappings - exclude them from case conversion
        value_mappings = self._get_value_mappings_from_config()
        columns_with_value_mappings = set(value_mappings.keys())
        
        for col in columns:
            if col in df.columns:
                if 'strip' in operations:
                    df[col] = df[col].astype(str).str.strip()
                if 'lower' in operations and col not in columns_with_value_mappings:
                    # Don't convert to lowercase if column has value mappings
                    df[col] = df[col].astype(str).str.lower()
                if 'remove_extra_spaces' in operations:
                    df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
        
        return df
    
    def _apply_missing_value_handling(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply missing value handling strategies."""
        required_columns = params.get('required_columns', [])
        default_strategies = params.get('default_strategies', {})
        
        for col in required_columns:
            if col in df.columns:
                if df[col].isna().any():
                    # Apply default strategy based on column type
                    if col in self.schema.columns:
                        dtype = str(self.schema.columns[col].dtype)
                        if 'str' in dtype or 'object' in dtype:
                            df[col] = df[col].fillna('')
                        # For numeric and datetime, leave as NaN for now
        
        return df
    
    def _apply_type_conversion(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply data type conversions."""
        type_mappings = params.get('type_mappings', {})
        coerce_errors = params.get('coerce_errors', True)
        
        for col, target_type in type_mappings.items():
            if col in df.columns:
                try:
                    if 'float' in target_type:
                        df[col] = pd.to_numeric(df[col], errors='coerce' if coerce_errors else 'raise')
                    elif 'int' in target_type:
                        df[col] = pd.to_numeric(df[col], errors='coerce' if coerce_errors else 'raise').astype('Int64')
                    elif 'datetime' in target_type:
                        df[col] = pd.to_datetime(df[col], errors='coerce' if coerce_errors else 'raise', dayfirst=True)
                    elif 'str' in target_type or 'object' in target_type:
                        df[col] = df[col].astype(str)
                except Exception as e:
                    if not coerce_errors:
                        raise
                    logger.debug(f"    ⚠️ Type conversion failed for {col}: {e}")
        
        return df
    
    def _apply_constraints(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply schema constraints with enhanced config-based validation."""
        constraints = params.get('constraints', {})
        initial_count = len(df)
        
        for col, constraint_dict in constraints.items():
            if col in df.columns:
                logger.debug(f"    🔍 Applying constraints to column: {col}")
                
                # Apply allowed values constraint
                if 'allowed_values' in constraint_dict:
                    allowed = constraint_dict['allowed_values']
                    before_count = len(df)
                    # Handle NaN values properly - keep NaN values if column is nullable
                    mask = df[col].isin(allowed) | df[col].isna()
                    df = df[mask]
                    after_count = len(df)
                    if before_count != after_count:
                        logger.debug(f"      ✓ Filtered {before_count - after_count} records with invalid {col} values")
                
                # Apply numeric range constraints
                if 'min_value' in constraint_dict:
                    min_val = constraint_dict['min_value']
                    before_count = len(df)
                    # Handle NaN values properly - keep NaN values if column is nullable
                    mask = (df[col] >= min_val) | df[col].isna()
                    df = df[mask]
                    after_count = len(df)
                    if before_count != after_count:
                        logger.debug(f"      ✓ Filtered {before_count - after_count} records with {col} < {min_val}")
                
                if 'max_value' in constraint_dict:
                    max_val = constraint_dict['max_value']
                    before_count = len(df)
                    # Handle NaN values properly - keep NaN values if column is nullable
                    mask = (df[col] <= max_val) | df[col].isna()
                    df = df[mask]
                    after_count = len(df)
                    if before_count != after_count:
                        logger.debug(f"      ✓ Filtered {before_count - after_count} records with {col} > {max_val}")
                
                # Apply additional constraints
                if 'greater_than' in constraint_dict:
                    gt_val = constraint_dict['greater_than']
                    before_count = len(df)
                    # Handle NaN values properly - keep NaN values if column is nullable
                    mask = (df[col] > gt_val) | df[col].isna()
                    df = df[mask]
                    after_count = len(df)
                    if before_count != after_count:
                        logger.debug(f"      ✓ Filtered {before_count - after_count} records with {col} <= {gt_val}")
                
                if 'less_than' in constraint_dict:
                    lt_val = constraint_dict['less_than']
                    before_count = len(df)
                    # Handle NaN values properly - keep NaN values if column is nullable
                    mask = (df[col] < lt_val) | df[col].isna()
                    df = df[mask]
                    after_count = len(df)
                    if before_count != after_count:
                        logger.debug(f"      ✓ Filtered {before_count - after_count} records with {col} >= {lt_val}")
        
        final_count = len(df)
        if initial_count != final_count:
            logger.info(f"    📊 Constraint application: {initial_count} → {final_count} records ({initial_count - final_count} filtered)")
        
        return df
    
    def _apply_duplicate_removal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply duplicate removal."""
        subset = params.get('subset', [])
        keep = params.get('keep', 'first')
        
        # Only use subset columns that exist in the DataFrame
        existing_subset = [col for col in subset if col in df.columns]
        
        if existing_subset:
            initial_count = len(df)
            df = df.drop_duplicates(subset=existing_subset, keep=keep)
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate records")
        
        return df