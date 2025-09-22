"""
Test script for the TTS engine with indic-parler-tts
"""
import sys
import logging
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from tts_engine import TTSEngine, CachedTTSEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_tts():
    """Test basic TTS functionality"""
    logger.info("Testing basic TTS engine...")
    
    tts = TTSEngine()
    
    if not tts.is_available():
        logger.warning("TTS model not available, skipping tests")
        return False
    
    # Test supported languages
    languages = tts.get_supported_languages()
    logger.info(f"Supported languages: {languages}")
    
    # Test synthesis
    test_text = "Hello, this is a test of the text-to-speech system."
    result = tts.synthesize_speech(test_text, language="en")
    
    if result['duration'] > 0 and len(result['audio_data']) > 0:
        logger.info(f"✓ TTS synthesis successful: {result['duration']:.2f}s audio")
        return True
    else:
        logger.error("✗ TTS synthesis failed")
        return False

def test_multilingual_tts():
    """Test TTS with different languages"""
    logger.info("Testing multilingual TTS...")
    
    tts = TTSEngine()
    
    if not tts.is_available():
        logger.warning("TTS model not available, skipping multilingual tests")
        return False
    
    # Test different languages
    test_cases = [
        ("Hello, how are you?", "en"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("হ্যালো, আপনি কেমন আছেন?", "bn"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "ta"),
        ("హలో, మీరు ఎలా ఉన్నారు?", "te")
    ]
    
    success_count = 0
    for text, lang in test_cases:
        logger.info(f"Testing {lang}: {text}")
        result = tts.synthesize_speech(text, language=lang)
        
        if result['duration'] > 0 and len(result['audio_data']) > 0:
            logger.info(f"✓ {lang} synthesis successful: {result['duration']:.2f}s")
            success_count += 1
        else:
            logger.error(f"✗ {lang} synthesis failed")
    
    logger.info(f"Multilingual test: {success_count}/{len(test_cases)} languages successful")
    return success_count == len(test_cases)

def test_cached_tts():
    """Test cached TTS functionality"""
    logger.info("Testing cached TTS engine...")
    
    tts = CachedTTSEngine(cache_size=10)
    
    if not tts.is_available():
        logger.warning("TTS model not available, skipping cache tests")
        return False
    
    # Test caching
    test_text = "This is a test for caching functionality."
    
    # First synthesis (should be cached)
    result1 = tts.synthesize_speech(test_text, language="en")
    logger.info(f"First synthesis: {result1['duration']:.2f}s")
    
    # Second synthesis (should use cache)
    result2 = tts.synthesize_speech(test_text, language="en")
    logger.info(f"Second synthesis: {result2['duration']:.2f}s")
    
    # Results should be identical
    if result1['audio_data'] == result2['audio_data']:
        logger.info("✓ Cache functionality working correctly")
        return True
    else:
        logger.error("✗ Cache functionality not working")
        return False

def test_voice_settings():
    """Test TTS with different voice settings"""
    logger.info("Testing voice settings...")
    
    tts = TTSEngine()
    
    if not tts.is_available():
        logger.warning("TTS model not available, skipping voice settings tests")
        return False
    
    test_text = "Testing different voice settings."
    
    # Test different settings
    settings_cases = [
        {"speed": 0.8, "pitch": 0.9, "volume": 1.0},
        {"speed": 1.2, "pitch": 1.1, "volume": 1.0},
        {"speed": 1.0, "pitch": 1.0, "volume": 0.8}
    ]
    
    success_count = 0
    for i, settings in enumerate(settings_cases):
        logger.info(f"Testing settings {i+1}: {settings}")
        result = tts.synthesize_speech(test_text, language="en", voice_settings=settings)
        
        if result['duration'] > 0 and len(result['audio_data']) > 0:
            logger.info(f"✓ Settings {i+1} synthesis successful: {result['duration']:.2f}s")
            success_count += 1
        else:
            logger.error(f"✗ Settings {i+1} synthesis failed")
    
    logger.info(f"Voice settings test: {success_count}/{len(settings_cases)} settings successful")
    return success_count == len(settings_cases)

def test_batch_synthesis():
    """Test batch synthesis functionality"""
    logger.info("Testing batch synthesis...")
    
    tts = TTSEngine()
    
    if not tts.is_available():
        logger.warning("TTS model not available, skipping batch tests")
        return False
    
    # Test batch synthesis
    texts = [
        "First sentence for batch testing.",
        "Second sentence for batch testing.",
        "Third sentence for batch testing."
    ]
    
    results = tts.synthesize_batch(texts, language="en")
    
    success_count = 0
    for i, result in enumerate(results):
        if result['duration'] > 0 and len(result['audio_data']) > 0:
            logger.info(f"✓ Batch item {i+1} successful: {result['duration']:.2f}s")
            success_count += 1
        else:
            logger.error(f"✗ Batch item {i+1} failed")
    
    logger.info(f"Batch synthesis test: {success_count}/{len(texts)} items successful")
    return success_count == len(texts)

def test_voice_info():
    """Test voice information functionality"""
    logger.info("Testing voice information...")
    
    tts = TTSEngine()
    
    # Test voice info for different languages
    languages = ["en", "hi", "bn", "ta", "te"]
    
    for lang in languages:
        info = tts.get_voice_info(lang)
        logger.info(f"Voice info for {lang}: {info}")
        
        if info['model_available'] == tts.is_available():
            logger.info(f"✓ Voice info for {lang} correct")
        else:
            logger.error(f"✗ Voice info for {lang} incorrect")

def main():
    """Run all TTS tests"""
    logger.info("Starting TTS engine tests...")
    
    tests = [
        ("Basic TTS", test_basic_tts),
        ("Multilingual TTS", test_multilingual_tts),
        ("Cached TTS", test_cached_tts),
        ("Voice Settings", test_voice_settings),
        ("Batch Synthesis", test_batch_synthesis),
        ("Voice Info", test_voice_info)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
