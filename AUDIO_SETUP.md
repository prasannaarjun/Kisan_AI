# Audio Setup for KisanAI

## Required Dependencies

The transcription engine requires several audio processing libraries to handle different audio formats from the frontend.

### Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### FFmpeg Installation (Required for WebM/Opus support)

FFmpeg is essential for handling WebM/Opus audio from the browser's MediaRecorder API.

#### Windows
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH
4. Verify installation: `ffmpeg -version`

#### Alternative: Use Chocolatey
```bash
choco install ffmpeg
```

#### Alternative: Use Scoop
```bash
scoop install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

### Verify Installation

Run the test script to verify everything is working:
```bash
python test_audio_loading.py
```

## Audio Format Support

The system supports multiple audio formats with fallback chain:

1. **FFmpeg** (best for WebM/Opus from frontend)
2. **Pydub** (good for WebM/Opus, MP3, WAV)
3. **Librosa** (good for various formats)
4. **Soundfile** (fast for WAV, FLAC)
5. **Raw PCM** (fallback for uncompressed audio)

## Troubleshooting

### "All audio loading methods failed"
- Ensure FFmpeg is installed and in PATH
- Check that audio data is not empty
- Verify Python audio libraries are installed

### "No module named 'librosa'"
- Run: `pip install librosa`

### "No module named 'pydub'"
- Run: `pip install pydub`

### "ffmpeg binary not found"
- Install FFmpeg and add to PATH
- Restart your terminal/IDE after installation

## Frontend Audio Format

The frontend sends audio in WebM format with Opus codec:
- Format: `audio/webm;codecs=opus`
- Sample Rate: 16kHz
- Channels: Mono
- Chunk Size: 200ms

This format is automatically converted to WAV (16kHz, mono, PCM16) for the STT model.
