#!/usr/bin/env python3
"""
Main script to run the entire audio processing pipeline.

This script orchestrates the full pipeline:
1. Ingest audio files from Cloudflare R2
2. Transcribe audio using OpenAI Whisper
3. Tokenize transcriptions
4. Store results
5. Generate summary visualizations (optional)
"""
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

# Add the project root to sys.path to allow imports from modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    logger, validate_config, PIPELINE_CONFIG, 
    OUTPUT_DIR, TEMP_DIR, get_openai_api_key
)
from pipeline.ingest import ingest_audio_files
from pipeline.transcribe import batch_transcribe
from pipeline.tokenize import tokenize_batch
from pipeline.store import store_data, validate_data
from pipeline.summarize import generate_all_visualizations

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio Processing Pipeline for transcription and tokenization"
    )
    
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Optional prefix to filter R2 files by folder"
    )
    
    parser.add_argument(
        "--batch-size", type=int, default=PIPELINE_CONFIG["batch_size"],
        help="Number of files to process in each batch"
    )
    
    parser.add_argument(
        "--max-workers", type=int, default=PIPELINE_CONFIG["max_workers"],
        help="Maximum number of parallel workers"
    )
    
    parser.add_argument(
        "--output-format", type=str, default=PIPELINE_CONFIG["output_format"],
        choices=["json", "parquet", "sqlite"],
        help="Output format for the processed data"
    )
    
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching for transcription"
    )
    
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of files to process (for testing)"
    )
    
    parser.add_argument(
        "--skip-summary", action="store_true",
        help="Skip generating summary visualizations"
    )
    
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip ingestion step and use already downloaded files"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Create file handler
    log_file = OUTPUT_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    return logger

def run_pipeline(args):
    """
    Run the complete audio processing pipeline.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    
    logger.info("Starting audio processing pipeline")
    logger.info(f"Configuration: {args}")
    
    try:
        # Validate configuration
        validate_config()
        
        # Step 1: Ingest audio files from R2
        logger.info("Step 1: Ingesting audio files from R2")
        if args.skip_ingestion:
            logger.info("Skipping ingestion step and using already downloaded files")
            # Get all audio files from TEMP_DIR
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                audio_files.extend(list(Path(TEMP_DIR).glob(f"**/*{ext}")))
            
            if not audio_files:
                logger.error("No audio files found in temp directory")
                return
            
            logger.info(f"Found {len(audio_files)} existing audio files")
        else:
            audio_files = ingest_audio_files(
                prefix=args.prefix,
                batch_size=args.batch_size,
                max_workers=args.max_workers
            )
        
        if not audio_files:
            logger.error("No audio files found to process")
            return
        
        logger.info(f"Using {len(audio_files)} audio files")
        
        # Apply limit if specified
        if args.limit and args.limit > 0:
            audio_files = audio_files[:args.limit]
            logger.info(f"Limited to {len(audio_files)} files as requested")
        
        # Step 2: Transcribe audio files using Whisper
        logger.info("Step 2: Transcribing audio files using Whisper")
        transcription_results = batch_transcribe(
            audio_files,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            use_cache=not args.no_cache
        )
        
        logger.info(f"Transcribed {len(transcription_results)} audio files")
        
        # Step 3: Tokenize transcriptions
        logger.info("Step 3: Tokenizing transcriptions")
        tokenized_results = tokenize_batch(
            transcription_results,
            max_workers=args.max_workers
        )
        
        logger.info(f"Tokenized {len(tokenized_results)} transcriptions")
        
        # Step 4: Store results
        logger.info(f"Step 4: Storing results in {args.output_format} format")
        
        # Validate data before storage
        if not validate_data(tokenized_results):
            logger.error("Data validation failed")
            return
        
        output_path = store_data(
            tokenized_results,
            format=args.output_format
        )
        
        logger.info(f"Stored results in {args.output_format} format at {output_path}")
        
        # Step 5: Generate summary visualizations (optional)
        if not args.skip_summary:
            logger.info("Step 5: Generating summary visualizations")
            visualizations = generate_all_visualizations(tokenized_results)
            
            logger.info("Generated visualizations:")
            for viz_type, path in visualizations.items():
                logger.info(f"- {viz_type}: {path}")
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        
        print("\n" + "=" * 50)
        print(f"Pipeline completed successfully!")
        print(f"- Processed {len(audio_files)} audio files")
        print(f"- Results stored in {args.output_format} format at {output_path}")
        
        if not args.skip_summary:
            print("- Visualizations generated in:", OUTPUT_DIR / "visualizations")
        
        print(f"- Total time: {total_time:.2f} seconds")
        print("=" * 50)
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"\nError: Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    run_pipeline(args)
