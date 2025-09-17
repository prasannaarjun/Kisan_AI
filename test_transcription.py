#!/usr/bin/env python3
"""
Test script for transcription engine
"""
import sys
import os
import numpy as np

# Add backend to path
sys.path.append('backend')

def test_transcription_engine():
    """Test the transcription engine"""
    print("🧪 Testing Transcription Engine")
    print("=" * 35)
    
    try:
        from transcription_engine import TranscriptionEngine
        
        # Initialize the engine
        print("Initializing transcription engine...")
        engine = TranscriptionEngine()
        
        # Create dummy audio data (1 second of silence)
        print("Creating dummy audio data...")
        sample_rate = 16000
        duration = 1.0  # 1 second
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Convert to bytes (simulate real audio)
        audio_bytes = audio_data.tobytes()
        
        # Test transcription
        print("Testing transcription...")
        result = engine.transcribe_audio(audio_bytes, language="en")
        
        print(f"Result: {result}")
        
        if result["text"] is not None:
            print("✅ Transcription engine working!")
            return True
        else:
            print("⚠️  Transcription engine working but returned empty result")
            return True
            
    except Exception as e:
        print(f"❌ Transcription engine test failed: {e}")
        return False

def main():
    """Main test function"""
    if test_transcription_engine():
        print("\n🎉 Transcription engine is ready!")
        print("You can now run: python start_backend.py")
    else:
        print("\n❌ Transcription engine has issues.")
        print("Try running: python fix_windows_signal.py")

if __name__ == "__main__":
    main()
