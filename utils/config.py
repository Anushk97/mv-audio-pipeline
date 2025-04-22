"""
Configuration module for the audio processing pipeline.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Cloudflare R2 configuration
R2_CONFIG = {
    "bucket_name": "mv-data-engineer-test",
    "endpoint_url": "https://bdadc4417ecd7714dd7d42a104a276c2.r2.cloudflarestorage.com",
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "cb5eb34e7737acb9296ff550121d1d6b"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "2081f620ecea1cf0166389256f1ef2208865bb5349216faa2898b42fffe8a2d4"),
    "region_name": os.getenv("AWS_REGION", "auto")
}

# Whisper API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pipeline configuration
PIPELINE_CONFIG = {
    # Number of worker processes for parallel processing
    "max_workers": os.cpu_count() or 4,
    
    # Batch size for processing (number of files to process in parallel)
    "batch_size": int(os.getenv("BATCH_SIZE", "10")),
    
    # Output file format: 'json', 'parquet', 'sqlite'
    "output_format": os.getenv("OUTPUT_FORMAT", "json"),
    
    # Whether to enable caching to avoid reprocessing files
    "enable_cache": os.getenv("ENABLE_CACHE", "True").lower() in ("true", "1", "yes"),
    
    # Whether to run summary analysis
    "run_summary": os.getenv("RUN_SUMMARY", "True").lower() in ("true", "1", "yes"),
}

def get_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment "
            "variable or add it to a .env file."
        )
    return OPENAI_API_KEY

def validate_config() -> bool:
    # Check if R2 credentials are set
    if not R2_CONFIG["aws_access_key_id"] or not R2_CONFIG["aws_secret_access_key"]:
        raise ValueError("R2 credentials not set properly")
    
    # Attempt to get OpenAI API key (will raise ValueError if not set)
    get_openai_api_key()
    
    return True
