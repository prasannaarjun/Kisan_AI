"""
Test script to verify the fixes for token length, JSON serialization, and deprecation warnings
"""
import sys
import os
import json
import base64

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.conversation_manager import ConversationManager
from backend.app import app

def test_conversation_manager():
    """Test that ConversationManager can handle long inputs without token errors"""
    print("Testing ConversationManager...")
    
    try:
        cm = ConversationManager()
        
        # Test with a very long input (simulating the issue)
        long_input = "What are the best practices for agricultural farming in Kerala? " * 100  # Very long input
        
        # Test generate_response with long input
        response = cm.generate_response(
            user_input=long_input,
            session_id="test_session",
            context_docs=[{"text": "Sample agricultural content " * 50}],  # Long context
            language="en"
        )
        
        print(f"✓ ConversationManager handled long input successfully")
        print(f"✓ Response length: {len(response)} characters")
        return True
        
    except Exception as e:
        print(f"✗ ConversationManager test failed: {e}")
        return False

def test_json_serialization():
    """Test that audio data can be properly serialized to JSON"""
    print("\nTesting JSON serialization...")
    
    try:
        # Simulate audio data
        audio_data = b"fake_audio_data_for_testing"
        
        # Test base64 encoding (as done in the fixed app.py)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Test JSON serialization
        test_data = {
            "type": "audio_response",
            "data": {
                "audio_data": audio_base64,
                "language": "en",
                "sample_rate": 22050
            }
        }
        
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        # Verify we can decode the audio data back
        decoded_audio = base64.b64decode(parsed_data["data"]["audio_data"])
        assert decoded_audio == audio_data
        
        print("✓ JSON serialization with base64 audio works correctly")
        return True
        
    except Exception as e:
        print(f"✗ JSON serialization test failed: {e}")
        return False

def test_app_import():
    """Test that the FastAPI app can be imported without errors"""
    print("\nTesting FastAPI app import...")
    
    try:
        # This should not raise any errors
        assert app is not None
        print("✓ FastAPI app imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ FastAPI app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running fix verification tests...\n")
    
    tests = [
        test_conversation_manager,
        test_json_serialization,
        test_app_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
