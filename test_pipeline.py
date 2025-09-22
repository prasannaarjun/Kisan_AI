"""
Test the complete voice pipeline with TTS
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_voice_pipeline():
    """Test the complete voice pipeline"""
    try:
        from graph_definition import ConversationGraph
        
        print("🚀 Testing complete voice pipeline...")
        
        # Initialize the conversation graph
        graph = ConversationGraph()
        
        # Test TTS engine availability
        tts_available = graph.tts_engine.is_available()
        print(f"✅ TTS Engine available: {tts_available}")
        
        if not tts_available:
            print("⚠️ TTS engine not available, testing without audio synthesis")
        
        # Test with dummy audio data
        dummy_audio = b"dummy_audio_data"
        session_id = "test_session_123"
        
        print("🔄 Processing conversation through pipeline...")
        
        # Process conversation
        result = await graph.process_conversation(
            audio_data=dummy_audio,
            session_id=session_id,
            language="en"
        )
        
        # Check results
        print("\n📊 Pipeline results:")
        print(f"  Session ID: {result.get('session_id')}")
        print(f"  Language: {result.get('language')}")
        print(f"  User Text: {result.get('user_text', 'N/A')}")
        print(f"  AI Text: {result.get('ai_text', 'N/A')}")
        print(f"  Audio Response Length: {len(result.get('audio_response', b''))} bytes")
        print(f"  Context Docs: {len(result.get('context_docs', []))}")
        
        # Test TTS directly if available
        if tts_available:
            print("\n🎵 Testing TTS engine directly...")
            test_text = "Hello, this is a test of the text-to-speech system."
            tts_result = graph.tts_engine.synthesize_speech(test_text, language="en")
            
            print(f"TTS Result:")
            print(f"  Duration: {tts_result['duration']:.2f}s")
            print(f"  Sample Rate: {tts_result['sample_rate']} Hz")
            print(f"  Audio Data Length: {len(tts_result['audio_data'])} bytes")
            print(f"  Language: {tts_result['language']}")
        
        print("\n✅ Voice pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Voice pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_only():
    """Test just the TTS engine"""
    try:
        from tts_engine import TTSEngine
        
        print("🎵 Testing TTS engine only...")
        
        tts = TTSEngine()
        
        if not tts.is_available():
            print("⚠️ TTS model not available")
            return False
        
        # Test basic synthesis
        test_text = "Hello, this is a test of the indic-parler-tts model."
        result = tts.synthesize_speech(test_text, language="en")
        
        print(f"TTS Test Results:")
        print(f"  Text: {test_text}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Sample Rate: {result['sample_rate']} Hz")
        print(f"  Audio Data Length: {len(result['audio_data'])} bytes")
        print(f"  Language: {result['language']}")
        
        # Test voice info
        info = tts.get_voice_info("en")
        print(f"Voice Info: {info}")
        
        # Test supported languages
        languages = tts.get_supported_languages()
        print(f"Supported Languages: {languages}")
        
        print("✅ TTS engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ TTS engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Starting voice pipeline tests...")
    
    # Test TTS engine only first
    print("\n" + "="*60)
    print("TEST 1: TTS Engine Only")
    print("="*60)
    tts_success = test_tts_only()
    
    # Test complete pipeline
    print("\n" + "="*60)
    print("TEST 2: Complete Voice Pipeline")
    print("="*60)
    pipeline_success = await test_voice_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"TTS Engine Test: {'PASSED' if tts_success else 'FAILED'}")
    print(f"Complete Pipeline Test: {'PASSED' if pipeline_success else 'FAILED'}")
    
    overall_success = tts_success and pipeline_success
    print(f"Overall Result: {'PASSED' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("🎉 All tests passed! Voice pipeline is working correctly.")
    else:
        print("❌ Some tests failed. Check the logs above for details.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
