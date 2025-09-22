#!/usr/bin/env python3
"""
Test script to verify RAG content filtering for TTS
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.graph_definition import ConversationGraph
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_rag_content_filtering():
    """Test that RAG content is properly filtered before TTS"""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing RAG content filtering...")
    graph = ConversationGraph()
    
    # Test cases with RAG content that should be filtered
    test_cases = [
        {
            "name": "Source file reference",
            "input": "Based on the agricultural extension report, organic farming practices include crop rotation. Source: AGRICULTURAL-EXTENSION-IN-KERALA-.txt",
            "expected_removed": ["Source: AGRICULTURAL-EXTENSION-IN-KERALA-.txt"]
        },
        {
            "name": "Chunk metadata",
            "input": "Crop rotation is important. Chunk ID: abc123 Score: 0.85 Document Type: technical",
            "expected_removed": ["Chunk ID: abc123", "Score: 0.85", "Document Type: technical"]
        },
        {
            "name": "RAG formatting",
            "input": "Here's information about soil management. [RAG] [SEARCH] This content was retrieved from the knowledge base.",
            "expected_removed": ["[RAG]", "[SEARCH]"]
        },
        {
            "name": "Multiple metadata",
            "input": "Organic farming benefits include better soil health. File: organicfarmingpolicyenglish.txt Section: benefits Chunk ID: def456",
            "expected_removed": ["File: organicfarmingpolicyenglish.txt", "Section: benefits", "Chunk ID: def456"]
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        
        original_text = test_case['input']
        filtered_text = graph._filter_rag_content_for_tts(original_text)
        
        logger.info(f"Original: {original_text}")
        logger.info(f"Filtered: {filtered_text}")
        
        # Check that expected content was removed
        removed_content = []
        for expected_removed in test_case['expected_removed']:
            if expected_removed not in filtered_text:
                removed_content.append(expected_removed)
            else:
                logger.warning(f"Expected removal failed: '{expected_removed}' still present")
        
        if len(removed_content) == len(test_case['expected_removed']):
            logger.info(f"✅ All expected content removed: {removed_content}")
        else:
            logger.error(f"❌ Some content not removed. Expected: {test_case['expected_removed']}, Removed: {removed_content}")

def test_tts_with_filtering():
    """Test TTS node with RAG content filtering"""
    logger = logging.getLogger(__name__)
    
    logger.info("\nTesting TTS node with RAG content filtering...")
    graph = ConversationGraph()
    
    # Create a mock state with RAG content
    mock_state = {
        "ai_response": "Based on the agricultural knowledge base, organic farming practices include crop rotation and soil management. Source: organicfarmingpolicyenglish.txt Chunk ID: abc123 Score: 0.85",
        "language": "en"
    }
    
    logger.info(f"Original AI response: {mock_state['ai_response']}")
    
    # Process through TTS node
    result_state = graph.tts_node(mock_state)
    
    # Check that audio was generated
    audio_response = result_state.get("audio_response", b"")
    logger.info(f"Audio generated: {len(audio_response)} bytes")
    
    if len(audio_response) > 0:
        logger.info("✅ TTS generated audio with filtered content")
    else:
        logger.warning("⚠️ No audio generated")

def test_edge_cases():
    """Test edge cases for content filtering"""
    logger = logging.getLogger(__name__)
    
    logger.info("\nTesting edge cases...")
    graph = ConversationGraph()
    
    edge_cases = [
        ("", "Empty string"),
        ("Short text", "Very short text"),
        ("No RAG content here", "Normal text without RAG metadata"),
        ("Source: test.txt", "Only metadata, no content")
    ]
    
    for text, description in edge_cases:
        logger.info(f"Testing: {description}")
        filtered = graph._filter_rag_content_for_tts(text)
        logger.info(f"  Original: '{text}'")
        logger.info(f"  Filtered: '{filtered}'")
        
        # For very short text, should return original
        if len(text) < 20 and text != filtered:
            logger.warning(f"  ⚠️ Short text was modified: '{text}' -> '{filtered}'")

if __name__ == "__main__":
    setup_logging()
    test_rag_content_filtering()
    test_tts_with_filtering()
    test_edge_cases()
