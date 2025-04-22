"""
Tokenization module for processing transcribed text.

This module handles the tokenization of transcribed text from audio files,
converting the raw text into tokens (words) for further analysis.
"""
import re
import string
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import nltk
from tqdm import tqdm

from utils.config import logger, OUTPUT_DIR

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize NLTK stopwords
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Preprocess text before tokenization:
    - Convert to lowercase
    - Remove numeric characters
    - Remove punctuation
    - Remove extra whitespace
    
    Args:
        text: Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numeric characters
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text: str, 
                 remove_stopwords: bool = True,
                 min_token_length: int = 2) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text to tokenize
        remove_stopwords: Whether to remove common stopwords
        min_token_length: Minimum token length to include
        
    Returns:
        List[str]: List of tokens
    """
    if not text:
        return []
    
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize the text
    tokens = word_tokenize(preprocessed_text)
    
    # Filter tokens
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    
    # Filter by length
    tokens = [token for token in tokens if len(token) >= min_token_length]
    
    return tokens

def tokenize_transcription(transcription_result: Dict[str, Any],
                          remove_stopwords: bool = True,
                          min_token_length: int = 2) -> Dict[str, Any]:
    """
    Tokenize a transcription result.
    
    Args:
        transcription_result: Dictionary with transcription result
        remove_stopwords: Whether to remove common stopwords
        min_token_length: Minimum token length to include
        
    Returns:
        Dict: Updated dictionary with tokens added
    """
    # Make a copy of the input dictionary to avoid modifying the original
    result = transcription_result.copy()
    
    # Get the transcription text
    text = result.get('transcription', '')
    
    # Tokenize the text
    tokens = tokenize_text(
        text, 
        remove_stopwords=remove_stopwords,
        min_token_length=min_token_length
    )
    
    # Add tokens to the result
    result['tokens'] = tokens
    
    return result

def tokenize_batch(transcription_results: List[Dict[str, Any]],
                  remove_stopwords: bool = True,
                  min_token_length: int = 2,
                  max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Tokenize a batch of transcription results in parallel.
    
    Args:
        transcription_results: List of transcription result dictionaries
        remove_stopwords: Whether to remove common stopwords
        min_token_length: Minimum token length to include
        max_workers: Maximum number of parallel workers
        
    Returns:
        List[Dict]: Updated list of dictionaries with tokens added
    """
    if not transcription_results:
        return []
    
    tokenize_func = partial(
        tokenize_transcription,
        remove_stopwords=remove_stopwords,
        min_token_length=min_token_length
    )
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(tokenize_func, transcription_results),
            total=len(transcription_results),
            desc="Tokenizing transcriptions"
        ))
    
    return results

def save_tokenized_results(results: List[Dict[str, Any]], 
                          output_path: Optional[Path] = None) -> Path:
    """
    Save tokenized results to a JSON file.
    
    Args:
        results: List of tokenized result dictionaries
        output_path: Custom output path for the results
        
    Returns:
        Path: Path to the saved file
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "tokenized_results.json"
    
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved tokenized results to {output_path}")
    return output_path

def get_token_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about the tokens.
    
    Args:
        results: List of tokenized result dictionaries
        
    Returns:
        Dict: Statistics about the tokens
    """
    if not results:
        return {}
    
    # Count the total number of tokens
    all_tokens = []
    for result in results:
        all_tokens.extend(result.get('tokens', []))
    
    # Count token frequencies
    token_freq = {}
    for token in all_tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    
    # Calculate statistics
    stats = {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(token_freq),
        "avg_tokens_per_file": len(all_tokens) / len(results) if results else 0,
        "top_tokens": sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:50]
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    from utils.config import PIPELINE_CONFIG
    from pipeline.ingest import ingest_audio_files
    from pipeline.transcribe import batch_transcribe
    
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
        
        # Print sample results
        for result in tokenized_results[:2]:
            print(f"File: {result['id']}")
            print(f"Tokens: {result['tokens'][:20]}...")
            print("-" * 50)
        
        # Save the results
        save_tokenized_results(tokenized_results)
        
        # Get token statistics
        stats = get_token_stats(tokenized_results)
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Unique tokens: {stats['unique_tokens']}")
        print(f"Average tokens per file: {stats['avg_tokens_per_file']:.2f}")
        print("Top 10 tokens:")
        for token, count in stats['top_tokens'][:10]:
            print(f"  {token}: {count}")
