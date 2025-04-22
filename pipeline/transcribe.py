"""
Transcription module using OpenAI's Whisper API.
"""
import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tempfile

from tqdm import tqdm
import openai
from openai import OpenAI

from utils.config import TEMP_DIR, OUTPUT_DIR, get_openai_api_key, logger

# Initialize cache directory
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def get_openai_client() -> OpenAI:
    """
    Get authenticated OpenAI client.
    
    Returns:
        OpenAI: Authenticated OpenAI client
    """
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    return client

def generate_cache_key(file_path: Path) -> str:
    """
    Generate a cache key for a file based on its path and modification time.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        str: Cache key
    """
    mod_time = os.path.getmtime(file_path)
    file_size = os.path.getsize(file_path)
    
    # Create a hash based on the file path, size, and modification time
    hash_content = f"{file_path}:{file_size}:{mod_time}"
    return hashlib.md5(hash_content.encode()).hexdigest()

def get_cached_transcription(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Try to get cached transcription result for a file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Optional[Dict]: Cached transcription result or None if not found
    """
    cache_key = generate_cache_key(file_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_path}: {e}")
    
    return None

def save_to_cache(file_path: Path, transcription_result: Dict[str, Any]) -> None:
    """
    Save transcription result to cache.
    
    Args:
        file_path: Path to the audio file
        transcription_result: Transcription result to cache
    """
    cache_key = generate_cache_key(file_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(transcription_result, f)
    except Exception as e:
        logger.warning(f"Failed to save cache for {file_path}: {e}")

def transcribe_audio_file(file_path: Path, 
                         use_cache: bool = True, 
                         model: str = "whisper-1") -> Dict[str, Any]:
    """
    Transcribe a single audio file using OpenAI's Whisper API.
    
    Args:
        file_path: Path to the audio file
        use_cache: Whether to use caching
        model: Whisper model to use
        
    Returns:
        Dict: Transcription result including file ID and text
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Generate file ID (relative path from TEMP_DIR)
    try:
        file_id = str(file_path.relative_to(TEMP_DIR))
    except ValueError:
        # If file is not in TEMP_DIR, use the filename
        file_id = file_path.name
    
    # Check cache if enabled
    if use_cache:
        cached_result = get_cached_transcription(file_path)
        if cached_result:
            logger.debug(f"Using cached transcription for {file_id}")
            return cached_result
    
    client = get_openai_client()
    
    try:
        logger.info(f"Transcribing {file_id}")
        with open(file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json"
            )
            
        # Extract the text from the response
        if hasattr(response, 'text'):
            transcription_text = response.text
        else:
            # For verbose_json format
            transcription_text = response.text if hasattr(response, 'text') else ""
            
        result = {
            "id": file_id,
            "transcription": transcription_text,
            "file_path": str(file_path),
            "model": model
        }
        
        # Save to cache if enabled
        if use_cache:
            save_to_cache(file_path, result)
            
        return result
    
    except Exception as e:
        logger.error(f"Error transcribing {file_id}: {e}")
        return {
            "id": file_id,
            "transcription": "",
            "error": str(e),
            "file_path": str(file_path)
        }

def transcribe_audio_files_parallel(file_paths: List[Path], 
                                   max_workers: int = 4, 
                                   use_cache: bool = True,
                                   model: str = "whisper-1") -> List[Dict[str, Any]]:
    """
    Transcribe multiple audio files in parallel.
    
    Args:
        file_paths: List of paths to audio files
        max_workers: Maximum number of parallel workers
        use_cache: Whether to use caching
        model: Whisper model to use
        
    Returns:
        List[Dict]: List of transcription results
    """
    transcribe_func = partial(transcribe_audio_file, 
                             use_cache=use_cache, 
                             model=model)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress tracking
        results = list(tqdm(
            executor.map(transcribe_func, file_paths),
            total=len(file_paths),
            desc="Transcribing audio"
        ))
    
    # Count successful transcriptions
    successful = sum(1 for r in results if "error" not in r)
    logger.info(f"Transcribed {successful}/{len(results)} files successfully")
    
    return results

def batch_transcribe(file_paths: List[Path],
                    batch_size: int = 10,
                    max_workers: int = 4,
                    use_cache: bool = True,
                    model: str = "whisper-1") -> List[Dict[str, Any]]:
    """
    Transcribe audio files in batches to manage memory and API rate limits.
    
    Args:
        file_paths: List of paths to audio files
        batch_size: Number of files to process in each batch
        max_workers: Maximum number of parallel workers
        use_cache: Whether to use caching
        model: Whisper model to use
        
    Returns:
        List[Dict]: Combined results from all batches
    """
    all_results = []
    
    # Process in batches
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
        
        batch_results = transcribe_audio_files_parallel(
            batch,
            max_workers=max_workers,
            use_cache=use_cache,
            model=model
        )
        
        all_results.extend(batch_results)
        
        # Small delay between batches to avoid API rate limits
        if i + batch_size < len(file_paths):
            time.sleep(1)
    
    return all_results

if __name__ == "__main__":
    # Example usage
    from utils.config import PIPELINE_CONFIG
    from pipeline.ingest import ingest_audio_files
    
    # Get audio files
    audio_files = ingest_audio_files(
        batch_size=PIPELINE_CONFIG['batch_size'],
        max_workers=PIPELINE_CONFIG['max_workers']
    )
    
    if audio_files:
        # Transcribe them
        results = batch_transcribe(
            audio_files[:5],  # Only process first 5 for testing
            batch_size=PIPELINE_CONFIG['batch_size'],
            max_workers=PIPELINE_CONFIG['max_workers'],
            use_cache=PIPELINE_CONFIG['enable_cache']
        )
        
        # Print results
        for result in results:
            print(f"File: {result['id']}")
            print(f"Transcription: {result['transcription'][:100]}...")
            print("-" * 50)
