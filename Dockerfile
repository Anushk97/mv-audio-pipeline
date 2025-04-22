FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create directories required by the application
RUN mkdir -p /app/temp /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV AWS_ACCESS_KEY_ID=cb5eb34e7737acb9296ff550121d1d6b
ENV AWS_SECRET_ACCESS_KEY=2081f620ecea1cf0166389256f1ef2208865bb5349216faa2898b42fffe8a2d4

# Set the entrypoint
ENTRYPOINT ["python", "run_pipeline.py"]

# Default command (can be overridden at runtime)
CMD ["--help"]
