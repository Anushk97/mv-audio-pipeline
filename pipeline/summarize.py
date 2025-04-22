"""
Summarization module for generating aggregated statistics and visualizations.

This module handles the optional bonus requirement of generating token frequency
histograms, audio duration stats, word clouds, and other useful analytics.
"""
import json
import math
import logging
import string
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from utils.config import OUTPUT_DIR, logger

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_token_frequency(data: List[Dict[str, Any]], 
                            top_n: int = 50,
                            min_count: int = 2) -> Dict[str, int]:
    """
    Generate token frequency counts from processed data.
    
    Args:
        data: List of processed data dictionaries
        top_n: Number of top tokens to include
        min_count: Minimum count for tokens to include
        
    Returns:
        Dict: Token frequency dictionary
    """
    # Collect all tokens from all records
    all_tokens = []
    for item in data:
        tokens = item.get('tokens', [])
        if tokens:
            all_tokens.extend(tokens)
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Filter by minimum count
    token_counts = {token: count for token, count in token_counts.items() 
                   if count >= min_count}
    
    # Get top N tokens
    top_tokens = dict(sorted(token_counts.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:top_n])
    
    return top_tokens

def plot_token_histogram(token_freq: Dict[str, int], 
                        output_path: Optional[Path] = None,
                        top_n: int = 30,
                        figsize: Tuple[int, int] = (12, 8)) -> Path:
    """
    Plot a histogram of token frequencies.
    
    Args:
        token_freq: Token frequency dictionary
        output_path: Path to save the plot
        top_n: Number of top tokens to include in the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path: Path to the saved plot
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "token_histogram.png"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get top N tokens sorted by frequency
    top_tokens = dict(sorted(token_freq.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:top_n])
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.bar(range(len(top_tokens)), list(top_tokens.values()), align='center')
    plt.xticks(range(len(top_tokens)), list(top_tokens.keys()), rotation=45, ha='right')
    plt.title(f'Top {len(top_tokens)} Token Frequencies')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved token histogram to {output_path}")
    return output_path

def generate_wordcloud(token_freq: Dict[str, int], 
                      output_path: Optional[Path] = None,
                      figsize: Tuple[int, int] = (12, 8)) -> Path:
    """
    Generate a word cloud from token frequencies.
    
    Args:
        token_freq: Token frequency dictionary
        output_path: Path to save the word cloud
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path: Path to the saved word cloud
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "wordcloud.png"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                        background_color='white',
                        max_words=200,
                        colormap='viridis',
                        contour_width=3,
                        contour_color='steelblue')
    
    # Generate the word cloud from token frequencies
    wordcloud.generate_from_frequencies(token_freq)
    
    # Display the word cloud
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the word cloud
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved word cloud to {output_path}")
    return output_path

def generate_transcription_length_stats(data: List[Dict[str, Any]], 
                                       output_path: Optional[Path] = None,
                                       figsize: Tuple[int, int] = (12, 8)) -> Tuple[Dict[str, Any], Path]:
    """
    Generate statistics and plots about transcription lengths.
    
    Args:
        data: List of processed data dictionaries
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        Tuple: (Statistics dictionary, Path to saved plot)
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "transcription_length_stats.png"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract transcription lengths
    lengths = []
    for item in data:
        transcription = item.get('transcription', '')
        if transcription:
            lengths.append(len(transcription))
    
    if not lengths:
        logger.warning("No transcription data to generate statistics")
        return {}, output_path
    
    # Calculate statistics
    stats = {
        'count': len(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'q25': np.percentile(lengths, 25),
        'q75': np.percentile(lengths, 75)
    }
    
    # Create histogram
    plt.figure(figsize=figsize)
    sns.histplot(lengths, bins=30, kde=True)
    plt.title('Distribution of Transcription Lengths')
    plt.xlabel('Length (characters)')
    plt.ylabel('Count')
    plt.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
    plt.axvline(stats['median'], color='g', linestyle='-', label=f"Median: {stats['median']:.1f}")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved transcription length stats to {output_path}")
    return stats, output_path

def generate_token_count_stats(data: List[Dict[str, Any]], 
                              output_path: Optional[Path] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> Tuple[Dict[str, Any], Path]:
    """
    Generate statistics and plots about token counts per file.
    
    Args:
        data: List of processed data dictionaries
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        Tuple: (Statistics dictionary, Path to saved plot)
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "token_count_stats.png"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract token counts
    counts = []
    for item in data:
        tokens = item.get('tokens', [])
        if tokens:
            counts.append(len(tokens))
    
    if not counts:
        logger.warning("No token data to generate statistics")
        return {}, output_path
    
    # Calculate statistics
    stats = {
        'count': len(counts),
        'mean': np.mean(counts),
        'median': np.median(counts),
        'std': np.std(counts),
        'min': np.min(counts),
        'max': np.max(counts),
        'q25': np.percentile(counts, 25),
        'q75': np.percentile(counts, 75)
    }
    
    # Create histogram
    plt.figure(figsize=figsize)
    sns.histplot(counts, bins=30, kde=True)
    plt.title('Distribution of Token Counts per File')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Count')
    plt.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
    plt.axvline(stats['median'], color='g', linestyle='-', label=f"Median: {stats['median']:.1f}")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved token count stats to {output_path}")
    return stats, output_path

def generate_summary_report(data: List[Dict[str, Any]], 
                           output_path: Optional[Path] = None) -> Path:
    """
    Generate a comprehensive summary report of the processed data.
    
    Args:
        data: List of processed data dictionaries
        output_path: Path to save the report
        
    Returns:
        Path: Path to the saved report
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "summary_report.json"
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate token frequency
    token_freq = generate_token_frequency(data)
    
    # Get top tokens
    top_tokens = [(token, count) for token, count in 
                 sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:50]]
    
    # Generate transcription length stats
    transcription_stats, _ = generate_transcription_length_stats(data)
    
    # Generate token count stats
    token_count_stats, _ = generate_token_count_stats(data)
    
    # Compile report
    report = {
        'dataset_summary': {
            'num_files': len(data),
            'total_tokens': sum(len(item.get('tokens', [])) for item in data),
            'unique_tokens': len(token_freq),
        },
        'transcription_length_stats': transcription_stats,
        'token_count_stats': token_count_stats,
        'top_tokens': top_tokens
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved summary report to {output_path}")
    return output_path

def generate_all_visualizations(data: List[Dict[str, Any]], 
                               output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate all available visualizations for the processed data.
    
    Args:
        data: List of processed data dictionaries
        output_dir: Directory to save visualizations
        
    Returns:
        Dict: Dictionary mapping visualization types to file paths
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "visualizations"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizations = {}
    
    # Generate token frequency histogram
    token_freq = generate_token_frequency(data)
    viz_path = output_dir / "token_histogram.png"
    visualizations['token_histogram'] = plot_token_histogram(token_freq, viz_path)
    
    # Generate word cloud
    viz_path = output_dir / "wordcloud.png"
    visualizations['wordcloud'] = generate_wordcloud(token_freq, viz_path)
    
    # Generate transcription length histogram
    viz_path = output_dir / "transcription_length.png"
    _, viz_path = generate_transcription_length_stats(data, viz_path)
    visualizations['transcription_length'] = viz_path
    
    # Generate token count histogram
    viz_path = output_dir / "token_count.png"
    _, viz_path = generate_token_count_stats(data, viz_path)
    visualizations['token_count'] = viz_path
    
    # Generate summary report
    report_path = output_dir / "summary_report.json"
    visualizations['summary_report'] = generate_summary_report(data, report_path)
    
    return visualizations

if __name__ == "__main__":
    # Example usage
    from utils.config import PIPELINE_CONFIG
    from pipeline.store import store_data
    import json
    
    # Load processed data
    results_path = OUTPUT_DIR / "results.json"
    
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Generate visualizations
        visualizations = generate_all_visualizations(data)
        
        print("Generated visualizations:")
        for viz_type, path in visualizations.items():
            print(f"- {viz_type}: {path}")
    else:
        print(f"No processed data found at {results_path}. Run the pipeline first.")
