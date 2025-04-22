# Audio Data Preprocessing Pipeline

A robust, horizontally scalable data pipeline for audio preprocessing that ingests audio files from Cloudflare R2, transcribes them using OpenAI Whisper, tokenizes the transcripts, and stores the results in a structured format.

## 🏗️ Architecture

1. **Data Ingestion**: 
   - Connects to Cloudflare R2 using boto3/S3 API
   - Efficiently downloads audio files in batches
   - Preserves directory structure and metadata

2. **Transcription**: 
   - Uses OpenAI Whisper API for high-quality transcription
   - Implements intelligent caching to avoid reprocessing files
   - Handles batch processing with error recovery

3. **Tokenization**: 
   - Processes transcribed text into tokens
   - Removes stopwords and applies text normalization
   - Preserves token order and context

4. **Storage**: 
   - Flexible storage options (JSON, Parquet, SQLite)
   - Schema validation and data integrity checks
   - Optimized for query performance

5. **Summarization (Bonus)**:
   - Token frequency analysis
   - Statistical insights (length distribution, etc.)
   - Visualizations (word clouds, histograms)

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/api-keys)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mv-audio-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Alternatively, create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t mv-audio-pipeline .
   ```

2. Run the container:
   ```bash
   docker run -v "$(pwd)/output:/app/output" -e OPENAI_API_KEY="your-api-key-here" mv-audio-pipeline
   ```

## 📊 Usage

### Basic Usage

Run the pipeline with default settings:

```bash
python run_pipeline.py
```

### Command Line Options

```bash
python run_pipeline.py --help
```

Key options:
- `--prefix`: Filter files by folder prefix in R2 bucket
- `--batch-size`: Number of files to process in each batch (default: 10)
- `--max-workers`: Maximum number of parallel workers (default: CPU count)
- `--output-format`: Storage format (json, parquet, sqlite)
- `--no-cache`: Disable transcription caching
- `--limit`: Process only N files (useful for testing)
- `--skip-summary`: Skip generating visualizations
- `--skip-ingestion`: Skip downloading files and use already downloaded files in the temp directory
- `--verbose`: Enable detailed logging

### Example Commands

Process all files with JSON output:
```bash
python run_pipeline.py --output-format json
```

Test with a small sample:
```bash
python run_pipeline.py --limit 5 --verbose
```

Process specific folder:
```bash
python run_pipeline.py --prefix "folder_name/"
```

Use already downloaded files (skip downloading from R2):
```bash
python run_pipeline.py --skip-ingestion --verbose
```

Process a small sample of already downloaded files:
```bash
python run_pipeline.py --skip-ingestion --limit 10 --verbose
```

### Using Pre-Downloaded Files

If you've already downloaded audio files (either from a previous run or from another source), you can skip the ingestion step and process files directly from the `temp` directory:

1. Ensure your audio files are placed in the `temp` directory (following the same structure they would have in the R2 bucket)
2. Run the pipeline with the `--skip-ingestion` flag:

```bash
python run_pipeline.py --skip-ingestion
```

This approach is useful for:
- Continuing processing after a previous run was interrupted
- Working offline without R2 access
- Testing with your own audio files
- Reducing API calls and bandwidth usage

The pipeline will automatically detect all audio files (with .wav, .mp3, .flac, .m4a, or .ogg extensions) in the temp directory and process them.

## �� Project Structure

```
mv-audio-pipeline/
├── pipeline/
│   ├── ingest.py     # R2 bucket connection and file downloading
│   ├── transcribe.py # OpenAI Whisper transcription
│   ├── tokenize.py   # Text tokenization and processing
│   ├── store.py      # Data storage in various formats
│   └── summarize.py  # Statistical analysis and visualization
├── utils/
│   └── config.py     # Configuration management
├── requirements.txt  # Dependencies
├── Dockerfile        # Container definition
├── README.md         # Documentation
└── run_pipeline.py   # Main orchestration script
```

## 🔄 Data Flow

1. **Ingestion**: Audio files are identified and downloaded from R2 storage
2. **Transcription**: Audio is converted to text using OpenAI Whisper
3. **Tokenization**: Transcribed text is processed into tokens
4. **Storage**: Results are saved in the requested format
5. **Summarization**: Aggregate statistics and visualizations are generated (optional)

## 🔌 Scalability Considerations

The pipeline is designed with horizontal scalability in mind:

- **Stateless Processing**: Each component is designed to operate independently
- **Parallelization**: Multi-processing at each step for greater throughput
- **Batching**: Intelligent batching to limit memory usage
- **Caching**: Avoid redundant processing with smart caching
- **Error Resilience**: Robust error handling and recovery mechanisms

To scale further:
- Deploy on cloud infrastructure with auto-scaling
- Integrate with distributed processing frameworks (Apache Spark, Ray)
- Add message queue for better load balancing (RabbitMQ, Kafka)

## 🔬 Technical Details

### Data Format

The pipeline produces structured output in the following format:

```json
{
  "id": "folder/audio123.wav",
  "transcription": "This is the spoken content.",
  "tokens": ["this", "is", "the", "spoken", "content"]
}
```

### Error Handling

- Network failures during ingestion trigger automatic retries
- API errors are logged and reported
- Batch processing continues even if individual files fail
- Comprehensive logging for troubleshooting

### Performance Optimization

- Parallel downloads and processing
- Efficient memory management with batched processing
- Caching to avoid redundant API calls
- Optimized storage formats (Parquet, SQLite indexes)

## 📝 Possible Improvements

- Add support for more transcription services (AssemblyAI, Google Speech, etc.)
- Implement more sophisticated tokenization algorithms
- Add sentiment analysis and entity extraction
- Integrate with a workflow engine like Airflow for scheduling
- Add support for streaming processing for real-time applications
- Implement more robust data validation and quality checks

## ⏱️ Time Spent

- Architecture design: 2 hours
- Implementation: 8 hours
- Testing and debugging: 3 hours
- Documentation: 2 hours
- Total: ~15 hours

## 🧩 Trade-offs and Decisions

- **S3 Client vs Direct HTTP**: Used boto3 S3 client for better compatibility and error handling
- **File Handling**: Downloaded files to temp storage rather than streaming to simplify implementation
- **Parallelization**: Used Python's native multiprocessing for simplicity, could be replaced with Ray for more advanced distributed processing
- **Storage**: Implemented multiple formats to allow flexibility based on downstream needs
- **Error Handling**: Chose to continue processing on individual file failures to maximize throughput

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
