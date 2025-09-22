#!/usr/bin/env python3
"""
Script to process agricultural text files and create embeddings
"""
import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append('backend')

from text_processor import AgriculturalTextProcessor
from enhanced_rag_store import EnhancedRAGStore

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agricultural_processing.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main processing function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if data directory exists
    data_dir = "data/extracted_text/content/extracted_text"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found")
        return
    
    # Initialize text processor
    logger.info("Initializing text processor...")
    processor = AgriculturalTextProcessor()
    
    # Process all text files
    logger.info(f"Processing text files in {data_dir}...")
    chunks = processor.process_directory(data_dir)
    
    if not chunks:
        logger.error("No chunks were processed")
        return
    
    # Create embeddings
    logger.info("Creating embeddings...")
    chunks_with_embeddings = processor.create_embeddings(chunks)
    
    # Save processed chunks
    output_file = "processed_agricultural_chunks.json"
    logger.info(f"Saving processed chunks to {output_file}...")
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(chunks_with_embeddings)} processed chunks to {output_file}")
    
    # Initialize enhanced RAG store
    logger.info("Initializing enhanced RAG store...")
    rag_store = EnhancedRAGStore()
    
    # Load processed chunks into RAG store
    logger.info("Loading chunks into RAG store...")
    rag_store.load_processed_chunks(output_file)
    
    # Save RAG store index
    index_path = "agricultural_rag_index"
    logger.info(f"Saving RAG store index to {index_path}...")
    rag_store.save_index(index_path)
    
    # Print statistics
    stats = rag_store.get_statistics()
    logger.info("Processing completed successfully!")
    logger.info(f"Total documents: {stats.get('total_documents', 0)}")
    logger.info(f"Document types: {stats.get('document_types', {})}")
    logger.info(f"Crop types: {stats.get('crop_types', {})}")
    logger.info(f"Practice types: {stats.get('practice_types', {})}")
    logger.info(f"Source files: {len(stats.get('source_files', []))}")
    
    # Test search functionality
    logger.info("\nTesting search functionality...")
    test_queries = [
        "organic farming practices",
        "crop diseases and treatment",
        "soil management techniques",
        "irrigation methods",
        "pest control strategies"
    ]
    
    for query in test_queries:
        logger.info(f"\nSearching for: '{query}'")
        results = rag_store.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Score: {result['score']:.3f}")
            logger.info(f"     Text: {result['text'][:100]}...")
            logger.info(f"     Type: {result['metadata'].get('document_type', 'unknown')}")

if __name__ == "__main__":
    main()
