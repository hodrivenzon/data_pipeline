"""
Pandera Schemas

Data validation schemas for different file types using Pandera DataFrameSchema.
Organized as classes for better structure and maintainability.

⚠️  CRITICAL: SCHEMA PROTECTION NOTICE ⚠️
===============================================

This file contains business-critical data schemas that define the expected
structure and validation rules for the ETL pipeline. These schemas should
ONLY be modified by human developers when there are actual business logic
changes, NOT to solve validation issues.

🚫 AI AGENTS ARE PROHIBITED FROM MODIFYING THIS FILE 🚫
========================================================

AI agents should:
- ✅ READ and analyze these schemas
- ✅ INFORM users about potential schema-related issues
- ✅ SUGGEST schema changes to users for consideration
- ❌ NEVER directly modify these schemas
- ❌ NEVER change validation rules to fix data issues
- ❌ NEVER alter business logic without human approval

If validation issues are detected, AI agents should:
1. Identify if the issue is schema-related
2. Inform the user about potential schema mismatches
3. Suggest that the user review the schema if business logic has changed
4. Focus on data cleaning and transformation logic instead

The correct approach is to fix the DATA to match the schema,
not to change the schema to match the data.

For schema modifications, contact the data engineering team.
"""

import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check
from typing import Optional
from datetime import datetime


class ProjectsSchema:
    """Schema for projects data validation."""
    
    @staticmethod
    def get_schema():
        """Get DataFrameSchema for projects data validation."""
        return DataFrameSchema({
            "project_code": Column(str, description="Project code identifier"),
            "study_code": Column(str, description="Study code identifier"),
            "study_cohort_code": Column(str, description="Study cohort code identifier"),
            "project_name": Column(str, description="Project name"),
            "study_name": Column(str, description="Study name"),
            "study_cohort_name": Column(str, description="Study cohort name"),
            "project_manager_name": Column(str, description="Project manager name"),
            "disease_name": Column(str, description="Disease name"),
        }, coerce=False, strict=True, checks=Check(lambda df: df.duplicated(subset=['project_code', 'study_code', 'study_cohort_code']).sum() == 0, error="Duplicate records found"))
    
    @staticmethod
    def get_metadata():
        """Get metadata about the projects schema."""
        return {
            'file_type': 'projects',
            'model_class': 'ProjectsSchema',
            'fields': [
                'project_code', 'study_code', 'study_cohort_code', 'project_name',
                'study_name', 'study_cohort_name', 'project_manager_name', 'disease_name'
            ],
            'required_fields': ['project_code', 'study_code', 'study_cohort_code', 'project_name', 'study_name', 'study_cohort_name', 'project_manager_name', 'disease_name'],
            'nullable_fields': [],
            'field_types': {
                'project_code': 'str', 'study_code': 'str', 'study_cohort_code': 'str',
                'project_name': 'str', 'study_name': 'str', 'study_cohort_name': 'str',
                'project_manager_name': 'str', 'disease_name': 'str'
            }
        }


class SubjectsSchema:
    """Schema for subjects data validation."""
    
    @staticmethod
    def get_schema():
        """Get DataFrameSchema for subjects data validation."""
        return DataFrameSchema({
            "project_code": Column(str, description="Project code identifier"),
            "study_code": Column(str, description="Study code identifier"),
            "study_cohort_code": Column(str, description="Study cohort code identifier"),
            "subject_id": Column(str, description="Subject identifier"),
            "sample_id": Column(str, description="Sample identifier"),
            "sample_type": Column(str, description="Sample type"),
        }, coerce=True, strict=False)
    
    @staticmethod
    def get_metadata():
        """Get metadata about the subjects schema."""
        return {
            'file_type': 'subjects',
            'model_class': 'SubjectsSchema',
            'fields': ['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id', 'sample_type'],
            'required_fields': ['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id', 'sample_type'],
            'nullable_fields': [],
            'field_types': {
                'project_code': 'str', 'study_code': 'str', 'study_cohort_code': 'str',
                'subject_id': 'str', 'sample_id': 'str', 'sample_type': 'str'
            }
        }


class ResultsSchema:
    """Schema for results data validation."""
    
    @staticmethod
    def get_schema():
        """Get DataFrameSchema for results data validation."""
        return DataFrameSchema({
            "sample_id": Column(str, description="Sample identifier"),
            "detection_value": Column(
                float,
                description="Detection value",
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True
            ),
            "cancer_detected": Column(
                str,
                description="Cancer detection result",
                checks=Check.isin(['Yes', 'No']),
                nullable=True
            ),
            "sample_status": Column(
                str,
                description="Sample status",
                checks=Check.isin(['Running', 'Finished', 'Failed', 'InProgress', 'Completed', 'Error']),
                nullable=True
            ),
            "fail_reason": Column(str, nullable=True, description="Failure reason"),
            "sample_quality": Column(
                float,
                description="Sample quality score",
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True
            ),
            "sample_quality_threshold": Column(
                float,
                description="Sample quality threshold",
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True
            ),
            "date_of_run": Column(pa.DateTime, nullable=True, description="Date of run"),
        }, coerce=True, strict=False)
    
    @staticmethod
    def get_metadata():
        """Get metadata about the results schema."""
        return {
            'file_type': 'results',
            'model_class': 'ResultsSchema',
            'fields': [
                'sample_id', 'detection_value', 'cancer_detected', 'sample_status',
                'fail_reason', 'sample_quality', 'sample_quality_threshold', 'date_of_run'
            ],
            'required_fields': ['sample_id'],
            'nullable_fields': [
                'detection_value', 'cancer_detected', 'sample_status', 'fail_reason',
                'sample_quality', 'sample_quality_threshold', 'date_of_run'
            ],
            'field_types': {
                'sample_id': 'str', 'detection_value': 'float', 'cancer_detected': 'str',
                'sample_status': 'str', 'fail_reason': 'str', 'sample_quality': 'float',
                'sample_quality_threshold': 'float', 'date_of_run': 'datetime'
            }
        }


class MergedSchema:
    """Schema for merged data validation."""
    
    @staticmethod
    def get_schema():
        """Get DataFrameSchema for merged data validation."""
        return DataFrameSchema({
            "project_code": Column(str, description="Project code identifier"),
            "study_code": Column(str, description="Study code identifier"),
            "study_cohort_code": Column(str, description="Study cohort code identifier"),
            "subject_id": Column(str, description="Subject identifier"),
            "sample_id": Column(str, description="Sample identifier"),
            "detection_value": Column(
                float,
                description="Detection value",
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True
            ),
            "cancer_detected": Column(str, nullable=True, description="Cancer detection result"),
            "sample_status": Column(str, nullable=True, description="Sample status"),
        }, coerce=True, strict=False)
    
    @staticmethod
    def get_metadata():
        """Get metadata about the merged schema."""
        return {
            'file_type': 'merged',
            'model_class': 'MergedSchema',
            'fields': [
                'project_code', 'study_code', 'study_cohort_code', 'subject_id',
                'sample_id', 'detection_value', 'cancer_detected', 'sample_status'
            ],
            'required_fields': ['project_code', 'study_code', 'study_cohort_code', 'subject_id', 'sample_id'],
            'nullable_fields': ['detection_value', 'cancer_detected', 'sample_status'],
            'field_types': {
                'project_code': 'str', 'study_code': 'str', 'study_cohort_code': 'str',
                'subject_id': 'str', 'sample_id': 'str', 'detection_value': 'float',
                'cancer_detected': 'str', 'sample_status': 'str'
            }
        }


class SummarySchema:
    """Schema for summary data validation."""
    
    @staticmethod
    def get_schema():
        """Get DataFrameSchema for summary data validation."""
        return DataFrameSchema({
            "Code": Column(str, description="Combined project-study-cohort code"),
            "Total_Samples": Column(
                int,
                description="Number of samples detected",
                checks=Check.ge(0)
            ),
            "Finished_Percentage": Column(
                float,
                description="Percentage of finished samples",
                checks=[Check.ge(0.0), Check.le(100.0)]
            ),
            "Lowest_Detection": Column(
                float,
                description="Lowest detection value",
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True
            ),
        }, coerce=True, strict=False)
    
    @staticmethod
    def get_metadata():
        """Get metadata about the summary schema."""
        return {
            'file_type': 'summary',
            'model_class': 'SummarySchema',
            'fields': ['Code', 'Total_Samples', 'Finished_Percentage', 'Lowest_Detection'],
            'required_fields': ['Code', 'Total_Samples', 'Finished_Percentage'],
            'nullable_fields': ['Lowest_Detection'],
            'field_types': {
                'Code': 'str', 'Total_Samples': 'int', 'Finished_Percentage': 'float', 'Lowest_Detection': 'float'
            }
        }


# Schema registry for easy access
SCHEMAS = {
    'projects': ProjectsSchema.get_schema,
    'subjects': SubjectsSchema.get_schema,
    'results': ResultsSchema.get_schema,
    'merged': MergedSchema.get_schema,
    'summary': SummarySchema.get_schema
}

# Schema model classes for direct access
SCHEMA_MODELS = {
    'projects': ProjectsSchema,
    'subjects': SubjectsSchema,
    'results': ResultsSchema,
    'merged': MergedSchema,
    'summary': SummarySchema
}


def get_schema(file_type: str):
    """
    Get schema for a specific file type.
    
    Args:
        file_type: Type of file ('projects', 'subjects', 'results', 'merged', 'summary')
        
    Returns:
        Pandera DataFrameSchema
        
    Raises:
        ValueError: If file_type is not supported
    """
    if file_type not in SCHEMAS:
        available_types = list(SCHEMAS.keys())
        raise ValueError(f"Unsupported file type: {file_type}. Available types: {available_types}")
    
    return SCHEMAS[file_type]()


def get_schema_model(file_type: str):
    """
    Get schema model class for a specific file type.
    
    Args:
        file_type: Type of file ('projects', 'subjects', 'results', 'merged', 'summary')
        
    Returns:
        Schema class
        
    Raises:
        ValueError: If file_type is not supported
    """
    if file_type not in SCHEMA_MODELS:
        available_types = list(SCHEMA_MODELS.keys())
        raise ValueError(f"Unsupported file type: {file_type}. Available types: {available_types}")
    
    return SCHEMA_MODELS[file_type]


def get_schema_metadata(file_type: str):
    """
    Get schema metadata for a specific file type.
    
    Args:
        file_type: Type of file ('projects', 'subjects', 'results', 'merged', 'summary')
        
    Returns:
        Dictionary containing schema metadata
    """
    schema_model = get_schema_model(file_type)
    return schema_model.get_metadata()


# Legacy function names for backward compatibility
def get_projects_schema():
    """Get schema for projects data validation."""
    return ProjectsSchema.get_schema()


def get_subjects_schema():
    """Get schema for subjects data validation."""
    return SubjectsSchema.get_schema()


def get_results_schema():
    """Get schema for results data validation."""
    return ResultsSchema.get_schema()


def get_merged_schema():
    """Get schema for merged data validation."""
    return MergedSchema.get_schema()


def get_summary_schema():
    """Get schema for summary data validation."""
    return SummarySchema.get_schema()