"""
Debug script to test the voice pipeline step by step
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

async def debug_pipeline():
    """Debug the voice pipeline step by step"""
    try:
        from graph_definition import ConversationGraph
        from conversation_manager import ConversationManager
        
        print("🔍 Debugging voice pipeline...")
        
        # Test conversation manager directly
        print("\n1. Testing Conversation Manager directly...")
        conv_manager = ConversationManager()
        
        # Test with a simple input
        test_input = "Hello, I need help with my crops"
        print(f"Input: {test_input}")
        
        response = conv_manager.generate_response(
            user_input=test_input,
            session_id="debug_session",
            context_docs=[],
            language="en"
        )
        
        print(f"Response: {response}")
        print(f"Response length: {len(response)}")
        
        # Test the full pipeline
        print("\n2. Testing full pipeline...")
        graph = ConversationGraph()
        
        # Test with dummy audio data
        dummy_audio = b"dummy_audio_data"
        session_id = "debug_session_2"
        
        print("Processing conversation through pipeline...")
        result = await graph.process_conversation(
            audio_data=dummy_audio,
            session_id=session_id,
            language="en"
        )
        
        print("\n📊 Pipeline Debug Results:")
        print(f"  Session ID: {result.get('session_id')}")
        print(f"  Language: {result.get('language')}")
        print(f"  User Text: '{result.get('user_text', 'N/A')}'")
        print(f"  AI Text: '{result.get('ai_text', 'N/A')}'")
        print(f"  AI Text Length: {len(result.get('ai_text', ''))}")
        print(f"  Audio Response Length: {len(result.get('audio_response', b''))} bytes")
        print(f"  Context Docs: {len(result.get('context_docs', []))}")
        
        # Test TTS with the AI response
        if result.get('ai_text'):
            print(f"\n3. Testing TTS with AI response...")
            tts_result = graph.tts_engine.synthesize_speech(
                result['ai_text'], 
                language=result.get('language', 'en')
            )
            print(f"TTS Duration: {tts_result['duration']:.2f}s")
            print(f"TTS Audio Length: {len(tts_result['audio_data'])} bytes")
        else:
            print("\n3. No AI text to test TTS with")
        
        print("\n✅ Debug completed!")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_pipeline())
    sys.exit(0 if success else 1)
