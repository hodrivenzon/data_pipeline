"""
Pipeline Exceptions

Custom exception classes for the data pipeline.
"""


class PipelineException(Exception):
    """Base exception for pipeline-related errors."""
    
    def __init__(self, message: str, details: str = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class SchemaRuleException(PipelineException):
    """Exception raised when schema validation rules fail."""
    pass


class ValidationException(PipelineException):
    """Exception raised when data validation fails."""
    pass


class DataQualityException(PipelineException):
    """Exception raised when data quality checks fail."""
    pass


class ConfigurationException(PipelineException):
    """Exception raised when configuration is invalid."""
    pass


class DataCleaningException(PipelineException):
    """Exception raised when data cleaning operations fail."""
    pass


class ExtractionException(PipelineException):
    """Exception raised when data extraction fails."""
    pass


class ProjectsExtractionException(ExtractionException):
    """Exception raised when projects data extraction fails."""
    pass


class SubjectsExtractionException(ExtractionException):
    """Exception raised when subjects data extraction fails."""
    pass


class ResultsExtractionException(ExtractionException):
    """Exception raised when results data extraction fails."""
    pass


class TransformationException(PipelineException):
    """Exception raised when data transformation fails."""
    pass


class DataMergingException(PipelineException):
    """Exception raised when data merging fails."""
    pass


class DataValidationException(PipelineException):
    """Exception raised when data validation fails."""
    pass


class SummaryGenerationException(PipelineException):
    """Exception raised when summary generation fails."""
    pass


class LoadingException(PipelineException):
    """Exception raised when data loading fails."""
    pass


class ParquetLoadingException(LoadingException):
    """Exception raised when Parquet loading fails."""
    pass


class SummaryLoadingException(LoadingException):
    """Exception raised when summary loading fails."""
    pass
