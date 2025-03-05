import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import json

def download_models():
    # Create cache directory
    cache_dir = "./models"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("Downloading models (this may take a while)...")
    
    # Download relation extraction model
    print("Downloading relation extraction model...")
    AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir)
    AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir)
    
    # Download summarization model
    print("Downloading summarization model...")
    AutoTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir=cache_dir)
    AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn', cache_dir=cache_dir)
    
    # Download embedding model
    print("Downloading embedding model...")
    SentenceTransformer('all-mpnet-base-v2', cache_folder=cache_dir)
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()