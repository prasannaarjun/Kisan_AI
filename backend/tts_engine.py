"""
Simple Text-to-Speech engine using pyttsx3
Supports multiple languages with basic voice synthesis
"""
import logging
import io
import wave
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Configure logging first
logger = logging.getLogger(__name__)

# Suppress warnings to reduce log spam
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not available. Install with: pip install pyttsx3")

class TTSEngine:
    """Simple Text-to-Speech engine using pyttsx3"""
    
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate
        self.engine = None
        
        logger.info("Initializing TTS engine with pyttsx3")
        
        try:
            if not PYTTSX3_AVAILABLE:
                raise ImportError("pyttsx3 library not available")
            
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a suitable voice
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
            
            # Set speech rate (words per minute)
            self.engine.setProperty('rate', 150)
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', 0.8)
            
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            self.engine = None
    
    def is_available(self) -> bool:
        """Check if TTS engine is available"""
        return self.engine is not None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return ['en', 'hi']  # Basic support for English and Hindi
    
    def synthesize_speech(self, text: str, language: str = "en", 
                         voice_settings: Optional[Dict] = None) -> Dict[str, any]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            language: Language code (e.g., 'en', 'hi')
            voice_settings: Optional voice parameters (rate, volume)
            
        Returns:
            Dict with 'audio_data', 'sample_rate', 'duration', 'language'
        """
        try:
            if not self.is_available():
                logger.warning("TTS engine not available, returning empty audio")
                return self._create_empty_audio()
            
            if not text or not text.strip():
                logger.warning("Empty text provided for synthesis")
                return self._create_empty_audio()
            
            # Apply voice settings if provided
            if voice_settings:
                if 'rate' in voice_settings:
                    self.engine.setProperty('rate', voice_settings['rate'])
                if 'volume' in voice_settings:
                    self.engine.setProperty('volume', voice_settings['volume'])
            
            logger.info(f"Synthesizing speech for text: '{text[:50]}...' in {language}")
            
            # Create a temporary buffer to capture audio
            audio_buffer = io.BytesIO()
            
            # Configure engine to save to buffer
            self.engine.save_to_file(text, 'temp_audio.wav')
            self.engine.runAndWait()
            
            # Read the generated audio file
            try:
                with open('temp_audio.wav', 'rb') as f:
                    audio_data = f.read()
                
                # Convert to numpy array and get actual sample rate
                audio_array, actual_sample_rate = self._wav_to_numpy(audio_data)
                
                # Clean up temp file
                import os
                os.remove('temp_audio.wav')
                
                # Convert to bytes for transmission
                audio_bytes = self._audio_array_to_bytes(audio_array)
                
                duration = len(audio_array) / actual_sample_rate
                
                logger.info(f"Speech synthesized successfully: {duration:.2f}s audio")
                
                return {
                    'audio_data': audio_bytes,
                    'sample_rate': actual_sample_rate,  # Use actual sample rate from WAV file
                    'duration': duration,
                    'language': language,
                    'text': text,
                    'settings': voice_settings or {}
                }
                
            except Exception as e:
                logger.error(f"Error processing generated audio: {e}")
                return self._create_empty_audio()
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return self._create_empty_audio()
    
    def _wav_to_numpy(self, wav_data: bytes) -> Tuple[np.ndarray, int]:
        """Convert WAV data to numpy array and return sample rate"""
        try:
            with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
                # Get audio parameters
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Read audio data
                frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert to numpy array based on sample width
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16
                
                audio_array = np.frombuffer(frames, dtype=dtype)
                
                # Convert to mono if stereo
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2)
                    audio_array = np.mean(audio_array, axis=1)
                
                # Convert to float32 and normalize
                if dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                    if dtype == np.uint8:
                        audio_array = (audio_array - 128) / 128.0
                    elif dtype == np.int16:
                        audio_array = audio_array / 32768.0
                    elif dtype == np.int32:
                        audio_array = audio_array / 2147483648.0
                
                # Don't resample - use the actual sample rate from the WAV file
                # This ensures the audio plays at the correct speed
                return audio_array, sample_rate
                
        except Exception as e:
            logger.error(f"Error converting WAV to numpy: {e}")
            # Return empty audio with default sample rate
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32), self.sample_rate
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation"""
        if orig_sr == target_sr:
            return audio
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        # Create new indices
        old_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio)
        
        return resampled.astype(np.float32)
    
    def _create_empty_audio(self) -> Dict[str, any]:
        """Create empty audio response"""
        empty_audio = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)  # 0.1s silence
        audio_bytes = self._audio_array_to_bytes(empty_audio)
        logger.warning(f"Creating empty audio: {len(audio_bytes)} bytes, sample rate: {self.sample_rate}Hz")
        return {
            'audio_data': audio_bytes,
            'sample_rate': self.sample_rate,
            'duration': 0.1,
            'language': 'en',
            'text': '',
            'settings': {}
        }
    
    def _audio_array_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert audio array to bytes for transmission"""
        # Convert to 16-bit PCM
        audio_16bit = (audio_array * 32767).astype(np.int16)
        return audio_16bit.tobytes()
    
    def get_voice_info(self, language: str = "en") -> Dict[str, any]:
        """Get information about available voices for a language"""
        return {
            'language': language,
            'sample_rate': self.sample_rate,
            'supported_settings': ['rate', 'volume'],
            'engine_available': self.is_available()
        }
    
    def test_synthesis(self, text: str = "Hello, this is a test of the text-to-speech system.", 
                      language: str = "en") -> bool:
        """Test the TTS system with a sample text"""
        try:
            result = self.synthesize_speech(text, language)
            success = result['duration'] > 0 and len(result['audio_data']) > 0
            logger.info(f"TTS test {'passed' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"TTS test failed: {e}")
            return False

class CachedTTSEngine(TTSEngine):
    """TTS Engine with caching support"""
    
    def __init__(self, cache_size: int = 100):
        super().__init__()
        self.cache = {}
        self.max_cache_size = cache_size
        self.access_order = []
    
    def synthesize_speech(self, text: str, language: str = "en", 
                         voice_settings: Optional[Dict] = None) -> Dict[str, any]:
        """Synthesize speech with caching"""
        if voice_settings is None:
            voice_settings = {}
        
        # Create cache key
        cache_key = f"{text}_{language}_{str(sorted(voice_settings.items()))}"
        
        # Check cache first
        if cache_key in self.cache:
            logger.debug("Using cached TTS result")
            return self.cache[cache_key]
        
        # Generate new audio
        result = super().synthesize_speech(text, language, voice_settings)
        
        # Store in cache (with size limit)
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        self.access_order.append(cache_key)
        
        return result
