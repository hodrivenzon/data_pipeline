#!/usr/bin/env python3
"""
Data Pipeline Main Entry Point

Command-line interface for running the data pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path

from core.pipeline import DataPipeline
from utils.exceptions.pipeline_exceptions import PipelineException


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Data Pipeline - Process CSV files into Parquet and summary outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run with default config.yaml
  %(prog)s --config custom.yaml      # Run with custom config
  %(prog)s --input /data/input       # Override input directory
  %(prog)s --output /data/output     # Override output directory
  %(prog)s --dry-run                 # Test run without saving files
  %(prog)s --verbose                 # Enable detailed logging
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Override input directory path"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        help="Override output directory path"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Run pipeline without saving output files"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    return parser


def setup_console_logging(verbose: bool = False) -> None:
    """Set up basic console logging before config is loaded."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )


def main() -> int:
    """
    Main entry point for the data pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup initial logging
    setup_console_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            return 1
        
        # Initialize pipeline
        logger.info("🚀 Starting Data Pipeline")
        logger.info(f"📁 Configuration: {config_path}")
        if args.dry_run:
            logger.info("🧪 Dry run mode enabled")
        
        pipeline = DataPipeline(str(config_path))
        
        # Apply command-line overrides
        if args.input:
            logger.info(f"📂 Overriding input directory: {args.input}")
            # Update file paths in config
            files_config = pipeline.config_manager.get_section('files')
            for key, value in files_config.items():
                if key.endswith('_file'):
                    filename = Path(value).name
                    new_path = Path(args.input) / filename
                    pipeline.config_manager.set(f'files.{key}', str(new_path))
        
        if args.output:
            logger.info(f"📂 Overriding output directory: {args.output}")
            # Update output paths in config
            pipeline.config_manager.set('files.output_parquet', str(Path(args.output) / "output.parquet"))
            pipeline.config_manager.set('files.summary_csv', str(Path(args.output) / "summary.csv"))
        
        if args.dry_run:
            # Set dry run mode in config
            pipeline.config_manager.set('pipeline.dry_run', True)
        
        # Run pipeline
        success = pipeline.run()
        
        if success:
            logger.info("✅ Pipeline completed successfully!")
            
            if not args.dry_run:
                # Show output summary
                summary = pipeline.get_data_summary()
                logger.info("📊 Processing Summary:")
                for data_type, count in summary.items():
                    if count > 0:
                        logger.info(f"  {data_type}: {count} records")
                
                # Show output files
                files_config = pipeline.config_manager.get_section('files')
                logger.info("📁 Output Files:")
                logger.info(f"  Parquet: {files_config.get('output_parquet')}")
                logger.info(f"  Summary: {files_config.get('summary_csv')}")
            else:
                logger.info("🧪 Dry run completed - no files were saved")
            
            return 0
        else:
            logger.error("❌ Pipeline failed!")
            return 1
    
    except KeyboardInterrupt:
        logger.info("⚠️ Pipeline interrupted by user")
        return 1
    
    except PipelineException as e:
        logger.error(f"❌ Pipeline error: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())