"""
Ingestion module for audio files from Cloudflare R2 bucket.
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Iterator, Tuple, Optional, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from utils.config import R2_CONFIG, TEMP_DIR, logger

def create_s3_client():
    """
    Create an S3 client connected to the Cloudflare R2 bucket.
    
    Returns:
        boto3.client: S3 client object
    """
    try:
        client = boto3.client(
            's3',
            endpoint_url=R2_CONFIG['endpoint_url'],
            aws_access_key_id=R2_CONFIG['aws_access_key_id'],
            aws_secret_access_key=R2_CONFIG['aws_secret_access_key'],
            region_name=R2_CONFIG['region_name']
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise

def list_audio_files(client, prefix: str = '', extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    List all audio files in the R2 bucket.
    
    Args:
        client: S3 client
        prefix: Optional prefix to filter files by folder
        extensions: List of file extensions to filter (e.g., ['.wav', '.mp3'])
    
    Returns:
        List[Dict]: List of dictionaries containing file information
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    bucket_name = R2_CONFIG['bucket_name']
    all_files = []
    
    try:
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext) for ext in extensions):
                        all_files.append({
                            'Key': key,
                            'Size': obj['Size'],
                            'LastModified': obj['LastModified']
                        })
        
        logger.info(f"Found {len(all_files)} audio files in R2 bucket")
        return all_files
    
    except ClientError as e:
        logger.error(f"Error listing files from bucket: {e}")
        raise

def download_file(client, file_info: Dict[str, Any], 
                 destination_dir: Path = TEMP_DIR, 
                 force_download: bool = False) -> Path:
    """
    Download a file from the R2 bucket.
    
    Args:
        client: S3 client
        file_info: Dictionary with file information from list_audio_files
        destination_dir: Directory to save downloaded files
        force_download: Whether to force download even if file exists
    
    Returns:
        Path: Path to the downloaded file
    """
    key = file_info['Key']
    bucket_name = R2_CONFIG['bucket_name']
    
    # Create destination directory
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a destination path preserving the directory structure
    destination_path = destination_dir / key
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if destination_path.exists() and not force_download:
        logger.debug(f"File already exists: {destination_path}")
        return destination_path
    
    try:
        logger.debug(f"Downloading {key} to {destination_path}")
        client.download_file(bucket_name, key, str(destination_path))
        return destination_path
    except ClientError as e:
        logger.error(f"Error downloading file {key}: {e}")
        raise

def download_files_parallel(file_infos: List[Dict[str, Any]], 
                           destination_dir: Path = TEMP_DIR,
                           max_workers: int = 4,
                           force_download: bool = False) -> List[Path]:
    """
    Download multiple files in parallel.
    
    Args:
        file_infos: List of file info dictionaries from list_audio_files
        destination_dir: Directory to save downloaded files
        max_workers: Maximum number of parallel downloads
        force_download: Whether to force download even if files exist
    
    Returns:
        List[Path]: List of paths to the downloaded files
    """
    client = create_s3_client()
    download_func = partial(download_file, client, 
                          destination_dir=destination_dir, 
                          force_download=force_download)
    
    downloaded_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(download_func, file_info): file_info['Key']
            for file_info in file_infos
        }
        
        with tqdm(total=len(file_infos), desc="Downloading files") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                file_key = future_to_file[future]
                try:
                    file_path = future.result()
                    downloaded_files.append(file_path)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to download {file_key}: {e}")
    
    logger.info(f"Downloaded {len(downloaded_files)} files successfully")
    return downloaded_files

def batch_generator(file_infos: List[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    """
    Generate batches of files for processing.
    
    Args:
        file_infos: List of file info dictionaries
        batch_size: Size of each batch
    
    Yields:
        Iterator[List[Dict]]: Batches of file info dictionaries
    """
    for i in range(0, len(file_infos), batch_size):
        yield file_infos[i:i + batch_size]

def ingest_audio_files(prefix: str = '', 
                      extensions: List[str] = None,
                      batch_size: int = 10,
                      max_workers: int = 4,
                      force_download: bool = False) -> List[Path]:
    """
    Main function to ingest audio files from R2.
    
    Args:
        prefix: Optional prefix to filter files by folder
        extensions: List of file extensions to filter
        batch_size: Size of batches for processing
        max_workers: Maximum number of parallel downloads
        force_download: Whether to force download even if files exist
    
    Returns:
        List[Path]: List of paths to the downloaded files
    """
    client = create_s3_client()
    
    # List all audio files
    file_infos = list_audio_files(client, prefix, extensions)
    
    if not file_infos:
        logger.warning("No audio files found to process")
        return []
    
    # Download files in batches to manage memory
    all_downloaded_files = []
    
    for batch in batch_generator(file_infos, batch_size):
        logger.info(f"Processing batch of {len(batch)} files")
        batch_files = download_files_parallel(
            batch, 
            destination_dir=TEMP_DIR,
            max_workers=max_workers,
            force_download=force_download
        )
        all_downloaded_files.extend(batch_files)
        
    return all_downloaded_files

if __name__ == "__main__":
    # Example usage
    from utils.config import PIPELINE_CONFIG
    
    files = ingest_audio_files(
        batch_size=PIPELINE_CONFIG['batch_size'],
        max_workers=PIPELINE_CONFIG['max_workers']
    )
    print(f"Ingested {len(files)} audio files")
