#!/usr/bin/env python3
"""
Test script for improved chunking and RAG search quality
"""
import sys
import os
sys.path.append('backend')

from enhanced_rag_store import EnhancedRAGStore
import json
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_chunking_improvements():
    """Test the improvements in chunking strategy"""
    logger = logging.getLogger(__name__)
    
    # Load the processed chunks
    with open('processed_agricultural_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Total chunks: {len(chunks)}")
    
    # Analyze chunking strategies
    strategies = {}
    chunk_lengths = []
    
    for chunk in chunks:
        strategy = chunk['metadata'].get('chunking_strategy', 'unknown')
        strategies[strategy] = strategies.get(strategy, 0) + 1
        chunk_lengths.append(len(chunk['text']))
    
    logger.info("Chunking strategies distribution:")
    for strategy, count in strategies.items():
        logger.info(f"  {strategy}: {count} chunks")
    
    logger.info(f"Average chunk length: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
    logger.info(f"Min chunk length: {min(chunk_lengths)} characters")
    logger.info(f"Max chunk length: {max(chunk_lengths)} characters")
    
    # Test RAG search quality
    logger.info("\nTesting RAG search quality...")
    rag_store = EnhancedRAGStore()
    rag_store.load_processed_chunks('processed_agricultural_chunks.json')
    
    test_queries = [
        "organic farming benefits",
        "crop diseases prevention",
        "soil health management",
        "irrigation techniques",
        "pest control methods",
        "fertilizer application",
        "crop rotation benefits",
        "water conservation"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = rag_store.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Score: {result['score']:.3f}")
            logger.info(f"     Strategy: {result['metadata'].get('chunking_strategy', 'unknown')}")
            logger.info(f"     Type: {result['metadata'].get('document_type', 'unknown')}")
            logger.info(f"     Length: {len(result['text'])} chars")
            logger.info(f"     Text: {result['text'][:100]}...")

def test_chunk_diversity():
    """Test the diversity of chunks created"""
    logger = logging.getLogger(__name__)
    
    with open('processed_agricultural_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Group chunks by source file
    by_source = {}
    for chunk in chunks:
        source = chunk['metadata']['source_file']
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(chunk)
    
    logger.info("\nChunks per source file:")
    for source, source_chunks in by_source.items():
        strategies = set(chunk['metadata'].get('chunking_strategy', 'unknown') for chunk in source_chunks)
        logger.info(f"  {source}: {len(source_chunks)} chunks, strategies: {strategies}")
    
    # Check for Q&A chunks
    qa_chunks = [chunk for chunk in chunks if chunk['metadata'].get('chunking_strategy') == 'qa']
    logger.info(f"\nQ&A chunks found: {len(qa_chunks)}")
    
    if qa_chunks:
        logger.info("Sample Q&A chunks:")
        for i, chunk in enumerate(qa_chunks[:3], 1):
            logger.info(f"  {i}. {chunk['text'][:150]}...")

if __name__ == "__main__":
    setup_logging()
    test_chunking_improvements()
    test_chunk_diversity()
