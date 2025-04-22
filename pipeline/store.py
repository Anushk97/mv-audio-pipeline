"""
Storage module for saving processed data in various formats.

This module handles storing the processed audio data (transcriptions and tokens)
in different formats like JSON, Parquet, or SQLite.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from utils.config import OUTPUT_DIR, logger

# Define the supported output formats
OutputFormat = Literal["json", "parquet", "sqlite"]

def store_as_json(data: List[Dict[str, Any]], 
                 output_path: Optional[Path] = None) -> Path:
    """
    Store data as JSON file.
    
    Args:
        data: List of dictionaries to store
        output_path: Custom output path
        
    Returns:
        Path: Path to the saved file
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "results.json"
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean the data for JSON serialization
    cleaned_data = []
    for item in data:
        # Create a copy of the item to avoid modifying the original
        clean_item = item.copy()
        
        # Only keep the required fields for the final output
        final_item = {
            "id": clean_item.get("id", ""),
            "transcription": clean_item.get("transcription", ""),
            "tokens": clean_item.get("tokens", [])
        }
        
        cleaned_data.append(final_item)
    
    # Write the data to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(cleaned_data)} records to JSON: {output_path}")
    return output_path

def store_as_parquet(data: List[Dict[str, Any]], 
                    output_path: Optional[Path] = None) -> Path:
    """
    Store data as Parquet file.
    
    Args:
        data: List of dictionaries to store
        output_path: Custom output path
        
    Returns:
        Path: Path to the saved file
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "results.parquet"
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean the data and prepare for Parquet
    cleaned_data = []
    for item in data:
        # Create a copy of the item to avoid modifying the original
        clean_item = item.copy()
        
        # Only keep the required fields for the final output
        final_item = {
            "id": clean_item.get("id", ""),
            "transcription": clean_item.get("transcription", ""),
            "tokens": clean_item.get("tokens", [])
        }
        
        cleaned_data.append(final_item)
    
    # Convert to DataFrame
    df = pd.DataFrame(cleaned_data)
    
    # Handle lists (tokens) for Parquet storage
    # Convert token lists to strings for storage
    if 'tokens' in df.columns:
        df['tokens_json'] = df['tokens'].apply(lambda x: json.dumps(x))
        df.drop('tokens', axis=1, inplace=True)
    
    # Write to Parquet
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(cleaned_data)} records to Parquet: {output_path}")
    return output_path

def store_as_sqlite(data: List[Dict[str, Any]], 
                   output_path: Optional[Path] = None) -> Path:
    """
    Store data in a SQLite database.
    
    Args:
        data: List of dictionaries to store
        output_path: Custom output path
        
    Returns:
        Path: Path to the saved database
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "results.db"
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean the data for SQLite storage
    cleaned_data = []
    for item in data:
        # Create a copy of the item to avoid modifying the original
        clean_item = item.copy()
        
        # Convert tokens list to JSON string
        if 'tokens' in clean_item:
            clean_item['tokens'] = json.dumps(clean_item['tokens'])
        
        # Only keep the required fields for the final output
        final_item = {
            "id": clean_item.get("id", ""),
            "transcription": clean_item.get("transcription", ""),
            "tokens": clean_item.get("tokens", "[]")
        }
        
        cleaned_data.append(final_item)
    
    # Convert to DataFrame for easier SQLite insertion
    df = pd.DataFrame(cleaned_data)
    
    # Connect to SQLite database
    conn = sqlite3.connect(str(output_path))
    
    # Create table and insert data
    df.to_sql('audio_data', conn, if_exists='replace', index=False)
    
    # Create indices for better query performance
    with conn:
        conn.execute('CREATE INDEX IF NOT EXISTS idx_id ON audio_data(id)')
    
    conn.close()
    
    logger.info(f"Saved {len(cleaned_data)} records to SQLite: {output_path}")
    return output_path

def store_data(data: List[Dict[str, Any]], 
              format: OutputFormat = "json",
              output_path: Optional[Path] = None) -> Path:
    """
    Store data in the specified format.
    
    Args:
        data: List of dictionaries to store
        format: Output format (json, parquet, or sqlite)
        output_path: Custom output path
        
    Returns:
        Path: Path to the saved file
    """
    if not data:
        logger.warning("No data to store")
        return None
    
    # Determine output path if not provided
    if output_path is None:
        if format == "json":
            output_path = OUTPUT_DIR / "results.json"
        elif format == "parquet":
            output_path = OUTPUT_DIR / "results.parquet"
        elif format == "sqlite":
            output_path = OUTPUT_DIR / "results.db"
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # Store data in the specified format
    if format == "json":
        return store_as_json(data, output_path)
    elif format == "parquet":
        return store_as_parquet(data, output_path)
    elif format == "sqlite":
        return store_as_sqlite(data, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def validate_data(data: List[Dict[str, Any]]) -> bool:
    """
    Validate that data contains the required fields.
    
    Args:
        data: List of dictionaries to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if not data:
        return False
    
    for item in data:
        # Check required fields
        if 'id' not in item:
            logger.error(f"Missing required field 'id' in item: {item}")
            return False
        
        if 'transcription' not in item:
            logger.error(f"Missing required field 'transcription' in item: {item}")
            return False
        
        if 'tokens' not in item:
            logger.error(f"Missing required field 'tokens' in item: {item}")
            return False
        
        # Check field types
        if not isinstance(item['id'], str):
            logger.error(f"Field 'id' must be a string, got {type(item['id'])} in item: {item}")
            return False
        
        if not isinstance(item['transcription'], str):
            logger.error(f"Field 'transcription' must be a string, got {type(item['transcription'])} in item: {item}")
            return False
        
        if not isinstance(item['tokens'], list):
            logger.error(f"Field 'tokens' must be a list, got {type(item['tokens'])} in item: {item}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    from utils.config import PIPELINE_CONFIG
    from pipeline.ingest import ingest_audio_files
    from pipeline.transcribe import batch_transcribe
    from pipeline.tokenize import tokenize_batch
    
    # Get audio files
    audio_files = ingest_audio_files(
        batch_size=PIPELINE_CONFIG['batch_size'],
        max_workers=PIPELINE_CONFIG['max_workers']
    )
    
    if audio_files:
        # Transcribe them
        transcription_results = batch_transcribe(
            audio_files[:5],  # Only process first 5 for testing
            batch_size=PIPELINE_CONFIG['batch_size'],
            max_workers=PIPELINE_CONFIG['max_workers'],
            use_cache=PIPELINE_CONFIG['enable_cache']
        )
        
        # Tokenize the transcriptions
        tokenized_results = tokenize_batch(
            transcription_results,
            max_workers=PIPELINE_CONFIG['max_workers']
        )
        
        # Store the results
        output_format = PIPELINE_CONFIG.get('output_format', 'json')
        output_path = store_data(tokenized_results, format=output_format)
        print(f"Stored results in {output_format} format at {output_path}")
