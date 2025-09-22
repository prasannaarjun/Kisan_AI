#!/usr/bin/env python3
"""
Test script to verify specific RAG content filtering for the user's example
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

def test_user_example():
    """Test the specific example provided by the user"""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing user's specific RAG content example...")
    graph = ConversationGraph()
    
    # The exact text from the user's example
    user_text = """For successful crop cultivation, focus on proper soil preparation, appropriate planting times, adequate spacing, and regular monitoring. Each crop has specific requirements for sunlight, water, and nutrients. Based on agricultural knowledge: PACKAGE OF PRACTICES RECOMMENDATIONS : CROPS 2016 15th edition Chief Editor S. ESTELITTA Editors BINOO P. BONNY, S. HELEN, A. SUMA DIRECTORATE OF EXTENSION KERALA AGRICULTURAL UNIVERSITY THRISSUR – 680 651, KERALA, INDIA English Package of Practices Recommendations : Crops 2016 First published 1 MEMBERS OF APEX COMMITTEE 1 Dr.K.Vasuki IAS, Director of Agriculture 2 Shri, K.P. Purushothaman, Senior Administrative Officer 3 Smt. Sheela Panicker P K, Additional Director of Agriculture (Extension) 4 Shri.P.Shaji, Senior Finance Officer 5 Smt. Latha G Panicker, Additional Directo"""
    
    logger.info(f"Original text length: {len(user_text)}")
    logger.info(f"Original text: {user_text[:200]}...")
    
    # Filter the text
    filtered_text = graph._filter_rag_content_for_tts(user_text)
    
    logger.info(f"Filtered text length: {len(filtered_text)}")
    logger.info(f"Filtered text: {filtered_text}")
    
    # Check if the filtering worked
    if "Based on agricultural knowledge:" not in filtered_text:
        logger.info("✅ 'Based on agricultural knowledge:' removed")
    else:
        logger.error("❌ 'Based on agricultural knowledge:' still present")
    
    if "PACKAGE OF PRACTICES" not in filtered_text:
        logger.info("✅ 'PACKAGE OF PRACTICES' removed")
    else:
        logger.error("❌ 'PACKAGE OF PRACTICES' still present")
    
    if "DIRECTORATE OF EXTENSION" not in filtered_text:
        logger.info("✅ 'DIRECTORATE OF EXTENSION' removed")
    else:
        logger.error("❌ 'DIRECTORATE OF EXTENSION' still present")
    
    if "KERALA AGRICULTURAL UNIVERSITY" not in filtered_text:
        logger.info("✅ 'KERALA AGRICULTURAL UNIVERSITY' removed")
    else:
        logger.error("❌ 'KERALA AGRICULTURAL UNIVERSITY' still present")
    
    # Check if the useful content is preserved
    if "soil preparation" in filtered_text and "planting times" in filtered_text:
        logger.info("✅ Useful agricultural content preserved")
    else:
        logger.warning("⚠️ Useful content may have been removed")
    
    # Test the TTS node with this text
    logger.info("\nTesting TTS node with filtered text...")
    mock_state = {
        "ai_response": user_text,
        "language": "en"
    }
    
    result_state = graph.tts_node(mock_state)
    audio_response = result_state.get("audio_response", b"")
    logger.info(f"Audio generated: {len(audio_response)} bytes")

if __name__ == "__main__":
    setup_logging()
    test_user_example()
