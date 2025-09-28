"""
Report Cleaner

Utility for managing report retention and cleaning up old files.
Keeps only the 3 latest reports from each type to prevent file overflow.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportCleaner:
    """
    Manages report retention and cleanup to prevent file overflow.
    Keeps only the 3 latest reports from each type.
    """
    
    def __init__(self, reports_dir: str = "file_store/reports"):
        """
        Initialize the report cleaner.
        
        Args:
            reports_dir: Directory containing reports
        """
        self.reports_dir = Path(reports_dir)
        self.max_reports_per_type = 3
        
    def clean_old_reports(self) -> Dict[str, int]:
        """
        Clean up old reports, keeping only the latest 3 of each type.
        
        Returns:
            Dictionary with cleanup statistics
        """
        if not self.reports_dir.exists():
            logger.debug(f"📁 Reports directory does not exist: {self.reports_dir}")
            return {"cleaned": 0, "kept": 0, "errors": 0}
        
        cleanup_stats = {"cleaned": 0, "kept": 0, "errors": 0}
        
        try:
            # Group reports by type
            report_groups = self._group_reports_by_type()
            
            for report_type, report_files in report_groups.items():
                if len(report_files) <= self.max_reports_per_type:
                    logger.debug(f"📊 {report_type}: {len(report_files)} files (no cleanup needed)")
                    cleanup_stats["kept"] += len(report_files)
                    continue
                
                # Sort by modification time (newest first)
                sorted_files = sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Keep the latest files
                files_to_keep = sorted_files[:self.max_reports_per_type]
                files_to_remove = sorted_files[self.max_reports_per_type:]
                
                # Remove old files
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        logger.debug(f"🗑️  Removed old report: {file_path.name}")
                        cleanup_stats["cleaned"] += 1
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to remove {file_path.name}: {e}")
                        cleanup_stats["errors"] += 1
                
                # Log kept files
                kept_names = [f.name for f in files_to_keep]
                logger.debug(f"📊 {report_type}: Kept {len(files_to_keep)} files: {kept_names}")
                cleanup_stats["kept"] += len(files_to_keep)
            
            if cleanup_stats["cleaned"] > 0:
                logger.info(f"🧹 Report cleanup completed: {cleanup_stats['cleaned']} files removed, {cleanup_stats['kept']} files kept")
            
        except Exception as e:
            logger.error(f"❌ Error during report cleanup: {e}")
            cleanup_stats["errors"] += 1
        
        return cleanup_stats
    
    def _group_reports_by_type(self) -> Dict[str, List[Path]]:
        """
        Group report files by their type (technical, summary, etc.).
        
        Returns:
            Dictionary mapping report types to lists of file paths
        """
        report_groups = {}
        
        # Check subdirectories for organized reports
        for subdir in ['technical', 'summary']:
            subdir_path = self.reports_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.iterdir():
                    if file_path.is_file():
                        report_type = subdir  # Use subdirectory name as type
                        if report_type not in report_groups:
                            report_groups[report_type] = []
                        report_groups[report_type].append(file_path)
        
        # Also check for any files in the main reports directory (backward compatibility)
        for file_path in self.reports_dir.iterdir():
            if not file_path.is_file():
                continue
            
            # Determine report type from filename
            report_type = self._get_report_type(file_path.name)
            
            if report_type not in report_groups:
                report_groups[report_type] = []
            
            report_groups[report_type].append(file_path)
        
        return report_groups
    
    def _get_report_type(self, filename: str) -> str:
        """
        Determine report type from filename.
        
        Args:
            filename: Name of the report file
            
        Returns:
            Report type (technical, summary, etc.)
        """
        if "technical_validation_report" in filename:
            return "technical_validation"
        elif "summary_validation_report" in filename:
            return "summary_validation"
        elif "transformation_recipe" in filename:
            return "transformation_recipe"
        else:
            return "other"
    
    def get_report_stats(self) -> Dict[str, int]:
        """
        Get statistics about current reports.
        
        Returns:
            Dictionary with report statistics
        """
        if not self.reports_dir.exists():
            return {"total_files": 0, "by_type": {}}
        
        report_groups = self._group_reports_by_type()
        
        stats = {
            "total_files": sum(len(files) for files in report_groups.values()),
            "by_type": {report_type: len(files) for report_type, files in report_groups.items()}
        }
        
        return stats
    
    def clean_transformation_recipes(self) -> int:
        """
        Clean up old transformation recipe files, keeping only the latest 3.
        
        Returns:
            Number of files cleaned
        """
        if not self.reports_dir.exists():
            return 0
        
        # Check recipes subdirectory first
        recipes_dir = self.reports_dir / "recipes"
        if recipes_dir.exists():
            recipe_files = []
            for file_path in recipes_dir.iterdir():
                if file_path.is_file() and "transformation_recipe" in file_path.name:
                    recipe_files.append(file_path)
        else:
            # Fallback to main directory for backward compatibility
            recipe_files = []
            for file_path in self.reports_dir.iterdir():
                if file_path.is_file() and "transformation_recipe" in file_path.name:
                    recipe_files.append(file_path)
        
        if len(recipe_files) <= self.max_reports_per_type:
            return 0
        
        # Sort by modification time (newest first)
        sorted_files = sorted(recipe_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        files_to_remove = sorted_files[self.max_reports_per_type:]
        cleaned_count = 0
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"🗑️  Removed old transformation recipe: {file_path.name}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to remove {file_path.name}: {e}")
        
        return cleaned_count
