#!/usr/bin/env python3
"""
Download required NLTK data for the pipeline.
"""
import nltk

# Download required NLTK datasets
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

print("NLTK data downloaded successfully.") 