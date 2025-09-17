"""
Speech-to-Text engine using ai4bharat/indic-conformer-600m-multilingual
Enhanced with robust audio loading and language detection
"""
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
import logging
from typing import Dict, List, Tuple, Optional
import io
import subprocess
import platform
import warnings

# Suppress warnings to reduce log spam
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class AudioLoader:
    """Robust audio loading with multiple fallback methods"""
    
    def __init__(self):
        self.loaders_available = {}
        self._check_available_loaders()
    
    def _check_available_loaders(self):
        """Check which audio loaders are available"""
        # Check soundfile
        try:
            import soundfile as sf
            self.loaders_available['soundfile'] = sf
            logger.info("✓ soundfile available")
        except ImportError:
            logger.warning("✗ soundfile not available")
        
        # Check librosa
        try:
            import librosa
            self.loaders_available['librosa'] = librosa
            logger.info("✓ librosa available")
        except ImportError:
            logger.warning("✗ librosa not available")
        
        # Check pydub
        try:
            from pydub import AudioSegment
            self.loaders_available['pydub'] = AudioSegment
            logger.info("✓ pydub available")
        except ImportError:
            logger.warning("✗ pydub not available")
        
        # Check ffmpeg
        try:
            import ffmpeg
            self.loaders_available['ffmpeg'] = ffmpeg
            logger.info("✓ ffmpeg-python available")
        except ImportError:
            logger.warning("✗ ffmpeg-python not available")
        
        # Check if ffmpeg binary is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            self.loaders_available['ffmpeg_binary'] = True
            logger.info("✓ ffmpeg binary available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("✗ ffmpeg binary not found - install ffmpeg for better audio support")
    
    def load_audio(self, audio_data: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio with fallback chain: ffmpeg → pydub → librosa → soundfile → raw PCM
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_array = None
        sample_rate = None
        used_loader = None
        
        # Debug: Log audio data info
        logger.info(f"Loading audio data: {len(audio_data)} bytes")
        if len(audio_data) > 0:
            # Check for common audio format signatures
            if audio_data.startswith(b'RIFF'):
                logger.info("Detected WAV format")
            elif audio_data.startswith(b'OggS'):
                logger.info("Detected OGG format")
            elif audio_data.startswith(b'\x1a\x45\xdf\xa3'):
                logger.info("Detected WebM format")
            elif audio_data.startswith(b'ID3') or (len(audio_data) > 8 and audio_data[4:8] == b'ftyp'):
                logger.info("Detected MP4/M4A format")
            else:
                logger.info(f"Unknown format, first 16 bytes: {audio_data[:16].hex()}")
                logger.info(f"First 32 bytes as text: {audio_data[:32]}")
        else:
            logger.warning("Empty audio data received")
            return np.array([]), 0
        
        # Method 1: Try ffmpeg first (best for WebM/Opus from frontend)
        if 'ffmpeg' in self.loaders_available and 'ffmpeg_binary' in self.loaders_available:
            try:
                logger.info("Trying ffmpeg...")
                # Use ffmpeg to convert to WAV format - handles WebM/Opus well
                # Only try ffmpeg for known audio formats, not raw PCM
                if (audio_data.startswith(b'RIFF') or audio_data.startswith(b'OggS') or 
                    audio_data.startswith(b'\x1a\x45\xdf\xa3') or audio_data.startswith(b'ID3') or
                    (len(audio_data) > 8 and audio_data[4:8] == b'ftyp')):
                    
                    process = (
                        self.loaders_available['ffmpeg']
                        .input('pipe:')
                        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sr)
                        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                    )
                    
                    stdout, stderr = process.communicate(input=audio_data)
                    logger.info(f"ffmpeg return code: {process.returncode}")
                    logger.info(f"ffmpeg stdout length: {len(stdout)}")
                    logger.info(f"ffmpeg stderr: {stderr.decode()[:200]}")
                    
                    if process.returncode == 0:
                        # Parse WAV data (skip 44-byte header)
                        if len(stdout) > 44:
                            audio_array = np.frombuffer(stdout[44:], dtype=np.int16).astype(np.float32) / 32768.0
                            sample_rate = target_sr
                            used_loader = 'ffmpeg'
                            logger.info("Audio loaded with ffmpeg")
                        else:
                            logger.warning("ffmpeg returned empty audio data")
                    else:
                        logger.warning(f"ffmpeg failed with return code {process.returncode}")
                else:
                    logger.info("Skipping ffmpeg for raw PCM data")
            except Exception as e:
                logger.warning(f"ffmpeg failed: {e}")
        else:
            logger.info("ffmpeg not available")
        
        # Method 2: Try pydub (good for WebM/Opus)
        if audio_array is None and 'pydub' in self.loaders_available:
            try:
                logger.info("Trying pydub...")
                audio_segment = self.loaders_available['pydub'].from_file(io.BytesIO(audio_data))
                # Convert to mono and get raw data
                audio_segment = audio_segment.set_channels(1)
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                sample_rate = audio_segment.frame_rate
                # Normalize to [-1, 1]
                if audio_segment.sample_width == 1:
                    audio_array = audio_array / 128.0 - 1.0
                elif audio_segment.sample_width == 2:
                    audio_array = audio_array / 32768.0
                elif audio_segment.sample_width == 4:
                    audio_array = audio_array / 2147483648.0
                used_loader = 'pydub'
                logger.info("Audio loaded with pydub")
            except Exception as e:
                logger.warning(f"pydub failed: {e}")
        else:
            logger.info("pydub not available")
        
        # Method 3: Try librosa
        if audio_array is None and 'librosa' in self.loaders_available:
            try:
                audio_array, sample_rate = self.loaders_available['librosa'].load(
                    io.BytesIO(audio_data), sr=None, mono=True
                )
                used_loader = 'librosa'
                logger.debug("Audio loaded with librosa")
            except Exception as e:
                logger.debug(f"librosa failed: {e}")
        
        # Method 4: Try soundfile
        if audio_array is None and 'soundfile' in self.loaders_available:
            try:
                audio_array, sample_rate = self.loaders_available['soundfile'].read(
                    io.BytesIO(audio_data), dtype='float32'
                )
                used_loader = 'soundfile'
                logger.debug("Audio loaded with soundfile")
            except Exception as e:
                logger.debug(f"soundfile failed: {e}")
        
        # Method 5: Try WebM/Opus specific handling
        if audio_array is None and audio_data.startswith(b'\x1a\x45\xdf\xa3'):
            try:
                # WebM format detected, try to handle with pydub specifically
                if 'pydub' in self.loaders_available:
                    # Force WebM format detection
                    audio_segment = self.loaders_available['pydub'].from_file(
                        io.BytesIO(audio_data), 
                        format="webm"
                    )
                    audio_segment = audio_segment.set_channels(1)
                    audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    sample_rate = audio_segment.frame_rate
                    # Normalize
                    if audio_segment.sample_width == 2:
                        audio_array = audio_array / 32768.0
                    used_loader = 'pydub_webm'
                    logger.debug("Audio loaded with pydub (WebM format)")
            except Exception as e:
                logger.debug(f"WebM-specific loading failed: {e}")
        
        # Method 6: Assume raw PCM data as last resort
        if audio_array is None:
            try:
                # Try different PCM formats
                for dtype in [np.int16, np.int32, np.float32]:
                    try:
                        audio_array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
                        if dtype == np.int16:
                            audio_array = audio_array / 32768.0
                        elif dtype == np.int32:
                            audio_array = audio_array / 2147483648.0
                        sample_rate = 16000  # Assume 16kHz
                        used_loader = 'raw_pcm'
                        logger.debug(f"Audio loaded as raw PCM ({dtype})")
                        break
                    except:
                        continue
            except Exception as e:
                logger.debug(f"Raw PCM loading failed: {e}")
        
        if audio_array is None or len(audio_array) == 0:
            logger.error("All audio loading methods failed")
            return np.array([]), 0
        
        # Log successful loader only once
        if not hasattr(self, '_logged_loaders'):
            self._logged_loaders = set()
        
        if used_loader and used_loader not in self._logged_loaders:
            logger.info(f"Audio loader chain: {used_loader} succeeded")
            self._logged_loaders.add(used_loader)
        
        return audio_array, sample_rate

class LanguageDetector:
    """Lightweight language detection"""
    
    def __init__(self):
        self.detectors_available = {}
        self._check_available_detectors()
    
    def _check_available_detectors(self):
        """Check which language detectors are available"""
        # Check langdetect
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # For consistent results
            self.detectors_available['langdetect'] = detect
            logger.info("✓ langdetect available")
        except ImportError:
            logger.debug("langdetect not available")
        
        # Check langid
        try:
            import langid
            self.detectors_available['langid'] = langid.classify
            logger.info("✓ langid available")
        except ImportError:
            logger.debug("langid not available")
    
    def detect_language(self, text: str) -> str:
        """Detect language from text with fallbacks"""
        if not text or not text.strip():
            return "unknown"
        
        text = text.strip()
        
        # Try langdetect first
        if 'langdetect' in self.detectors_available:
            try:
                lang_code = self.detectors_available['langdetect'](text)
                return lang_code
            except Exception as e:
                logger.debug(f"langdetect failed: {e}")
        
        # Try langid
        if 'langid' in self.detectors_available:
            try:
                lang_code, confidence = self.detectors_available['langid'](text)
                if confidence > 0.5:  # Only trust if confidence > 50%
                    return lang_code
            except Exception as e:
                logger.debug(f"langid failed: {e}")
        
        # Fallback to character-based detection
        return self._detect_by_characters(text)
    
    def _detect_by_characters(self, text: str) -> str:
        """Simple character-based language detection"""
        # Count Devanagari characters (Hindi, Marathi, etc.)
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        # Count Bengali characters
        bengali_count = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        # Count Tamil characters
        tamil_count = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
        # Count Telugu characters
        telugu_count = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
        # Count Gujarati characters
        gujarati_count = sum(1 for char in text if '\u0A80' <= char <= '\u0AFF')
        # Count Kannada characters
        kannada_count = sum(1 for char in text if '\u0C80' <= char <= '\u0CFF')
        # Count Malayalam characters
        malayalam_count = sum(1 for char in text if '\u0D00' <= char <= '\u0D7F')
        
        total_indic = (devanagari_count + bengali_count + tamil_count + 
                      telugu_count + gujarati_count + kannada_count + malayalam_count)
        
        if total_indic > len(text) * 0.3:  # If more than 30% Indic characters
            # Find the script with most characters
            script_counts = {
                'hi': devanagari_count,
                'bn': bengali_count,
                'ta': tamil_count,
                'te': telugu_count,
                'gu': gujarati_count,
                'kn': kannada_count,
                'ml': malayalam_count
            }
            return max(script_counts, key=script_counts.get)
        else:
            return "en"  # Default to English

class TranscriptionEngine:
    def __init__(self, model_name: str = "ai4bharat/indic-conformer-600m-multilingual"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        
        # Initialize audio loader and language detector
        self.audio_loader = AudioLoader()
        self.language_detector = LanguageDetector()
        
        logger.info(f"Loading STT model: {model_name} on {self.device}")
        
        # Fix for Windows signal.SIGALRM issue
        if platform.system() == "Windows":
            # Use a simpler model for Windows to avoid signal issues
            self.model_name = "openai/whisper-tiny"
            logger.info(f"Windows detected, using fallback model: {self.model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=False
            )
            
            # Check if it's a Whisper model
            if "whisper" in self.model_name.lower():
                from transformers import WhisperForConditionalGeneration
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    trust_remote_code=False
                )
                self.model_type = "whisper"
            else:
                self.model = AutoModelForCTC.from_pretrained(
                    self.model_name,
                    trust_remote_code=False
                )
                self.model_type = "ctc"
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            # Fallback to a very simple model
            logger.info("Falling back to basic model...")
            self._load_fallback_model()
        
        # Language mapping for the model
        self.lang_codes = {
            'hi': 'hindi',
            'en': 'english', 
            'bn': 'bengali',
            'te': 'telugu',
            'ta': 'tamil',
            'gu': 'gujarati',
            'kn': 'kannada',
            'ml': 'malayalam',
            'mr': 'marathi',
            'pa': 'punjabi',
            'or': 'odia',
            'as': 'assamese'
        }
    
    def _load_fallback_model(self):
        """Load a very simple fallback model"""
        try:
            # Use a basic CTC model that should work
            self.model_name = "facebook/wav2vec2-base-960h"
            logger.info(f"Loading fallback model: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=False
            )
            self.model = AutoModelForCTC.from_pretrained(
                self.model_name,
                trust_remote_code=False
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_type = "ctc"
            
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            # Create a dummy model that just returns empty strings
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model that returns empty transcriptions"""
        logger.warning("Creating dummy model - transcription will be disabled")
        self.model = None
        self.processor = None
    
    def preprocess_audio(self, audio_data: bytes, target_sr: int = 16000) -> np.ndarray:
        """Convert raw audio bytes to the format expected by the model"""
        try:
            # Use the robust audio loader
            audio_array, sr = self.audio_loader.load_audio(audio_data, target_sr)
            
            if len(audio_array) == 0:
                logger.error("No audio data loaded")
                return np.array([])
            
            # Resample if necessary
            if sr != target_sr:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_array) * target_sr / sr)
                    audio_array = signal.resample(audio_array, num_samples)
                    logger.debug(f"Resampled from {sr}Hz to {target_sr}Hz")
                except Exception as e:
                    logger.warning(f"Resampling failed: {e}")
                    # Simple linear interpolation as fallback
                    indices = np.linspace(0, len(audio_array) - 1, int(len(audio_array) * target_sr / sr))
                    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Ensure it's float32
            audio_array = audio_array.astype(np.float32)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return np.array([])
    
    def transcribe_audio(self, audio_data: bytes, language: str = "auto") -> Dict[str, str]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio bytes
            language: Language code or "auto" for automatic detection
            
        Returns:
            Dict with 'text', 'language', 'confidence'
        """
        try:
            # Check if we have a working model
            if self.model is None or self.processor is None:
                logger.warning("No model available, returning empty transcription")
                return {"text": "", "language": "unknown", "confidence": 0.0}
            
            # Preprocess audio
            audio_array = self.preprocess_audio(audio_data)
            if len(audio_array) == 0:
                logger.warning("Empty audio array after preprocessing")
                return {"text": "", "language": "unknown", "confidence": 0.0}
            
            # Prepare inputs with language specification for Whisper
            if hasattr(self, 'model_type') and self.model_type == "whisper":
                # For Whisper, we can specify language in the processor
                if language == "en" or language == "auto":
                    # Force English language detection and transcription
                    inputs = self.processor(
                        audio_array, 
                        sampling_rate=self.sample_rate, 
                        return_tensors="pt",
                        language="en"
                    ).to(self.device)
                else:
                    inputs = self.processor(
                        audio_array, 
                        sampling_rate=self.sample_rate, 
                        return_tensors="pt"
                    ).to(self.device)
            else:
                inputs = self.processor(
                    audio_array, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                ).to(self.device)
            
            # Generate transcription based on model type
            with torch.no_grad():
                if hasattr(self, 'model_type') and self.model_type == "whisper":
                    # Whisper model - force English transcription if language is detected as English
                    generation_kwargs = {
                        "max_length": 448,
                        "num_beams": 1,
                        "do_sample": False
                    }
                    
                    # Force English language if detected language is English
                    if language == "en" or language == "auto":
                        # Set language to English to force English transcription
                        generation_kwargs["language"] = "en"
                        generation_kwargs["task"] = "transcribe"  # Force transcription, not translation
                    
                    outputs = self.model.generate(
                        inputs.input_features,
                        **generation_kwargs
                    )
                    transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                else:
                    # CTC model
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Clean up transcription
            transcription = transcription.strip()
            
            # Log transcription for debugging
            logger.info(f"Raw transcription: {transcription}")
            logger.info(f"Has non-Latin script: {self._has_non_latin_script(transcription)}")
            
            # Determine language if auto
            if language == "auto":
                detected_lang = self.language_detector.detect_language(transcription)
                logger.info(f"Initial language detection: {detected_lang}")
                
                # If transcription is in non-Latin script but detected as English, 
                # it's likely a language detection error - force re-detection
                if detected_lang == "en" and self._has_non_latin_script(transcription):
                    # Try to detect the actual script
                    script_lang = self._detect_script_language(transcription)
                    logger.info(f"Script-based language detection: {script_lang}")
                    if script_lang != "en":
                        detected_lang = script_lang
                        logger.info(f"Corrected language to: {detected_lang}")
            else:
                detected_lang = language
                logger.info(f"Using specified language: {detected_lang}")
            
            # Calculate confidence based on transcription length and content
            confidence = min(0.9, max(0.1, len(transcription) / 50.0)) if transcription else 0.0
            
            return {
                "text": transcription,
                "language": detected_lang,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0}
    
    def transcribe_streaming(self, audio_chunk: bytes, language: str = "auto") -> Dict[str, str]:
        """
        Transcribe a single audio chunk for streaming
        """
        return self.transcribe_audio(audio_chunk, language)
    
    def _has_non_latin_script(self, text: str) -> bool:
        """Check if text contains non-Latin characters"""
        if not text:
            return False
        
        # Check for common non-Latin scripts
        for char in text:
            # Devanagari (Hindi, Marathi, etc.)
            if '\u0900' <= char <= '\u097F':
                return True
            # Arabic/Persian/Urdu
            elif '\u0600' <= char <= '\u06FF':
                return True
            # Bengali
            elif '\u0980' <= char <= '\u09FF':
                return True
            # Tamil
            elif '\u0B80' <= char <= '\u0BFF':
                return True
            # Telugu
            elif '\u0C00' <= char <= '\u0C7F':
                return True
            # Gujarati
            elif '\u0A80' <= char <= '\u0AFF':
                return True
            # Kannada
            elif '\u0C80' <= char <= '\u0CFF':
                return True
            # Malayalam
            elif '\u0D00' <= char <= '\u0D7F':
                return True
        
        return False
    
    def _detect_script_language(self, text: str) -> str:
        """Detect language based on script characters"""
        if not text:
            return "en"
        
        # Count characters by script
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        arabic_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        bengali_count = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        tamil_count = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
        telugu_count = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
        gujarati_count = sum(1 for char in text if '\u0A80' <= char <= '\u0AFF')
        kannada_count = sum(1 for char in text if '\u0C80' <= char <= '\u0CFF')
        malayalam_count = sum(1 for char in text if '\u0D00' <= char <= '\u0D7F')
        
        # Find the script with most characters
        script_counts = {
            'hi': devanagari_count,  # Hindi/Urdu (Devanagari script)
            'ur': arabic_count,      # Urdu (Arabic script)
            'bn': bengali_count,
            'ta': tamil_count,
            'te': telugu_count,
            'gu': gujarati_count,
            'kn': kannada_count,
            'ml': malayalam_count
        }
        
        max_script = max(script_counts, key=script_counts.get)
        return max_script if script_counts[max_script] > 0 else "en"

    def get_audio_info(self, audio_data: bytes) -> Dict[str, any]:
        """Get information about the audio data"""
        try:
            audio_array, sample_rate = self.audio_loader.load_audio(audio_data)
            return {
                "duration": len(audio_array) / sample_rate if sample_rate > 0 else 0,
                "sample_rate": sample_rate,
                "channels": 1,  # We always convert to mono
                "samples": len(audio_array)
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {"duration": 0, "sample_rate": 0, "channels": 0, "samples": 0}