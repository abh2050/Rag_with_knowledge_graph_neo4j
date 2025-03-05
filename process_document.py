#!/usr/bin/env python
# filepath: /Users/abhishekshah/Desktop/technical_paper_extraction/process_document.py
"""
Script to process a document and add it to the knowledge graph.
"""
import argparse
import logging
import sys
import os
import json
from dataclasses import asdict
from pathlib import Path

# Import your NER and processing classes
from ner import ScientificDocumentPipeline, load_config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Process a scientific document and add it to the knowledge graph')
    parser.add_argument('--file', type=str, required=True, help='Path to the document file')
    parser.add_argument('--title', type=str, help='Document title (optional)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed document')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Create output directory if it doesn't exist
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        logger.info("Initializing document processing pipeline")
        pipeline = ScientificDocumentPipeline(config)
        
        # Process the document
        logger.info(f"Processing document: {args.file}")
        processed_doc = pipeline.process_document(args.file)
        
        # Override title if provided
        if args.title:
            processed_doc.metadata.title = args.title
        
        # Save processed document to JSON
        output_path = Path(config.output_dir) / f"{Path(args.file).name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(processed_doc), f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        logger.info(f"Saved processed document to {output_path}")
        
        # Add to Neo4j knowledge graph
        logger.info("Loading document into Neo4j knowledge graph")
        doc_id = pipeline.load_document_to_neo4j(processed_doc)
        logger.info(f"Document ID: {doc_id}")
        
        print(f"Document ID: {doc_id}")
        return 0
        
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        return 1

# Custom JSON encoder for numpy arrays, etc.
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super(CustomJSONEncoder, self).default(obj)

if __name__ == "__main__":
    sys.exit(main())