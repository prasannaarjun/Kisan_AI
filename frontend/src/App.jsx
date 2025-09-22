import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from './api';
import MicButton from './components/MicButton';
import TranscriptPanel from './components/TranscriptPanel';
import ChatHistory from './components/ChatHistory';
import ContextPanel from './components/ContextPanel';
import StatsPanel from './components/StatsPanel';

function App() {
  // State management
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isMicEnabled, setIsMicEnabled] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isPartialTranscript, setIsPartialTranscript] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState('');
  const [status, setStatus] = useState('Ready to start');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [contextDocs, setContextDocs] = useState([]);
  const [showStats, setShowStats] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // Refs
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);

  // Initialize app
  useEffect(() => {
    loadSessions();
    checkMicrophonePermission();
  }, []);

  // Load sessions from API
  const loadSessions = async () => {
    try {
      setIsLoading(true);
      const sessionsData = await apiService.getSessions();
      setSessions(sessionsData);
      
      if (sessionsData.length > 0 && !currentSession) {
        await selectSession(sessionsData[0].session_id);
      }
    } catch (err) {
      setError('Failed to load sessions');
      console.error('Error loading sessions:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Check microphone permission
  const checkMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsMicEnabled(true);
      stream.getTracks().forEach(track => track.stop());
    } catch (err) {
      setIsMicEnabled(false);
      setError('Microphone access denied. Please enable microphone permissions.');
    }
  };

  // Create new session
  const createNewSession = async () => {
    try {
      setIsLoading(true);
      const newSession = await apiService.startSession();
      setSessions(prev => [newSession, ...prev]);
      await selectSession(newSession.session_id);
      setError('');
    } catch (err) {
      setError('Failed to create new session');
      console.error('Error creating session:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Select a session
  const selectSession = async (sessionId) => {
    try {
      setIsLoading(true);
      const history = await apiService.getSessionHistory(sessionId);
      setCurrentSession(sessionId);
      setMessages(history);
      setError('');
    } catch (err) {
      setError('Failed to load session history');
      console.error('Error loading session:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // WebSocket connection
  const connectWebSocket = useCallback((sessionId) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = apiService.createWebSocketConnection(sessionId);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('Connected - Ready to speak');
      setConnectionStatus('connected');
      setError('');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message received:', data);
        handleWebSocketMessage(data);
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
        setError('Error processing server response');
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setStatus('Disconnected');
      setConnectionStatus('disconnected');
      
      // Attempt to reconnect if not a clean close
      if (event.code !== 1000) {
        setError('Connection lost. Attempting to reconnect...');
        setTimeout(() => {
          if (currentSession) {
            connectWebSocket(currentSession);
          }
        }, 3000);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error. Please check your internet connection.');
      setConnectionStatus('error');
    };

    wsRef.current = ws;
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'transcription':
        setTranscript(data.data.text);
        setIsPartialTranscript(!data.data.is_final);
        setDetectedLanguage(data.data.language || '');
        break;
      
      case 'ai_response':
        setMessages(prev => [...prev, {
          role: 'assistant',
          text: data.data.text,
          language: data.data.language,
          timestamp: new Date().toISOString()
        }]);
        setTranscript('');
        setIsPartialTranscript(false);
        break;
      
      case 'audio_response':
        // Handle audio response playback
        if (data.data.audio_data) {
          console.log('Received audio response:', {
            dataLength: data.data.audio_data.length,
            language: data.data.language,
            sampleRate: data.data.sample_rate
          });
          playAudioResponse(data.data.audio_data, data.data.sample_rate);
        } else {
          console.log('No audio data in response');
        }
        break;
      
      case 'conversation_response':
        // Handle full conversation response
        if (data.data.user_text) {
          setMessages(prev => [...prev, {
            role: 'user',
            text: data.data.user_text,
            language: data.data.language,
            timestamp: new Date().toISOString()
          }]);
        }
        
        // Handle AI response in conversation_response
        if (data.data.ai_text) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            text: data.data.ai_text,
            language: data.data.language,
            timestamp: new Date().toISOString()
          }]);
        }
        
        // Handle context documents
        if (data.data.context_docs) {
          setContextDocs(data.data.context_docs);
        }
        break;
      
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  // Start recording
  const startRecording = async () => {
    try {
      if (!currentSession) {
        await createNewSession();
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      // Connect WebSocket if not connected
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        connectWebSocket(currentSession);
        // Wait a bit for connection to establish
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        sendAudioData(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      // Start recording with 200ms chunks
      mediaRecorderRef.current.start(200);
      setIsRecording(true);
      setStatus('Recording... Speak now');
      setTranscript('');
      setError('');

    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to start recording');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    setStatus('Processing...');
  };

  // Send audio data via WebSocket
  const sendAudioData = async (audioBlob) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Convert to array buffer and send
        const arrayBuffer = await audioBlob.arrayBuffer();
        wsRef.current.send(arrayBuffer);
        setStatus('Processing audio...');
      } else {
        setError('Connection lost. Please try again.');
      }
    } catch (err) {
      console.error('Error sending audio:', err);
      setError('Failed to send audio data');
    }
  };

  // Handle microphone errors
  const handleMicError = (message) => {
    setError(message);
    setIsRecording(false);
    setStatus('Microphone error');
  };

  // Play audio response
  const playAudioResponse = (audioData, sampleRate = 22050) => {
    try {
      // Check if audio data is empty or too small
      if (!audioData || audioData.length < 100) {
        console.log('Empty or very small audio data received, skipping playback');
        setIsPlayingAudio(false);
        return;
      }

      console.log(`Playing audio: ${audioData.length} bytes, sample rate: ${sampleRate}Hz`);

      // Decode base64 audio data
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Debug: Check if this looks like valid PCM data
      console.log('Audio data analysis:', {
        totalBytes: bytes.length,
        expectedSamples: bytes.length / 2, // 16-bit = 2 bytes per sample
        expectedDuration: (bytes.length / 2) / sampleRate,
        firstFewBytes: Array.from(bytes.slice(0, 16)),
        lastFewBytes: Array.from(bytes.slice(-16))
      });
      
      // Use the sample rate from the backend directly
      // The backend now sends the correct sample rate from the TTS engine
      let actualSampleRate = sampleRate || 22050; // Fallback to 22050 if not provided
      console.log(`Using sample rate from backend: ${actualSampleRate}Hz`);
      
      const wavBlob = createWavBlob(bytes, actualSampleRate);
      const audioUrl = URL.createObjectURL(wavBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setIsPlayingAudio(true);
        
        audioRef.current.onended = () => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
        };
        
        audioRef.current.onerror = (e) => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
          console.error('Error playing audio response:', e);
          setError('Failed to play audio response');
        };
      }
    } catch (err) {
      console.error('Error playing audio response:', err);
      setIsPlayingAudio(false);
      setError('Failed to process audio response');
    }
  };

  // Convert raw PCM data to WAV format
  const createWavBlob = (pcmData, sampleRate) => {
    const length = pcmData.length;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    // RIFF header
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true); // File size - 8
    writeString(8, 'WAVE');
    
    // fmt chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true);  // audio format (PCM)
    view.setUint16(22, 1, true);  // number of channels
    view.setUint32(24, sampleRate, true); // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate (sample rate * channels * bytes per sample)
    view.setUint16(32, 2, true);  // block align (channels * bytes per sample)
    view.setUint16(34, 16, true); // bits per sample
    
    // data chunk
    writeString(36, 'data');
    view.setUint32(40, length, true); // data size
    
    // Copy PCM data
    const pcmView = new Uint8Array(buffer, 44);
    pcmView.set(pcmData);
    
    console.log('WAV header created:', {
      fileSize: 44 + length,
      sampleRate: sampleRate,
      channels: 1,
      bitsPerSample: 16,
      byteRate: sampleRate * 2,
      dataSize: length
    });
    
    return new Blob([buffer], { type: 'audio/wav' });
  };

  // Delete session
  const deleteSession = async (sessionId) => {
    try {
      setIsLoading(true);
      await apiService.deleteSession(sessionId);
      
      // Remove from local state
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
      
      // If deleted session was current, select another or create new
      if (currentSession === sessionId) {
        const remainingSessions = sessions.filter(s => s.session_id !== sessionId);
        if (remainingSessions.length > 0) {
          await selectSession(remainingSessions[0].session_id);
        } else {
          setCurrentSession(null);
          setMessages([]);
        }
      }
      
      setError('');
    } catch (err) {
      setError('Failed to delete session');
      console.error('Error deleting session:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Retry connection
  const retryConnection = () => {
    if (currentSession) {
      setError('');
      setStatus('Retrying connection...');
      connectWebSocket(currentSession);
    }
  };

  // Clear error
  const clearError = () => {
    setError('');
  };

  // Test audio playback with a simple tone
  const testAudioPlayback = () => {
    try {
      // Generate a simple 440Hz tone for 2 seconds
      const sampleRate = 22050;
      const duration = 2; // 2 seconds
      const frequency = 440; // A4 note
      const samples = sampleRate * duration;
      const audioData = new Int16Array(samples);
      
      for (let i = 0; i < samples; i++) {
        audioData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 32767 * 0.3;
      }
      
      // Convert to bytes
      const bytes = new Uint8Array(audioData.buffer);
      
      console.log('Test audio generated:', {
        samples: samples,
        duration: duration,
        sampleRate: sampleRate,
        frequency: frequency,
        bytes: bytes.length
      });
      
      // Create WAV blob and play
      const wavBlob = createWavBlob(bytes, sampleRate);
      const audioUrl = URL.createObjectURL(wavBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setIsPlayingAudio(true);
        
        audioRef.current.onended = () => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
          console.log('Test audio playback completed successfully');
        };
        
        audioRef.current.onerror = (e) => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
          console.error('Test audio playback failed:', e);
          setError('Audio playback test failed');
        };
      }
    } catch (err) {
      console.error('Error in test audio playback:', err);
      setError('Failed to test audio playback');
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>🌾 KisanAI - Agricultural Assistant</h1>
      </header>

      <main className="main-content">
        <aside className="sidebar">
          <div className="sidebar-controls">
            <button 
              className="new-session-btn" 
              onClick={createNewSession}
              disabled={isLoading}
            >
              + New Session
            </button>
            <button 
              className="stats-btn" 
              onClick={() => setShowStats(true)}
              title="View Statistics"
            >
              📊 Stats
            </button>
            <button 
              className="test-audio-btn" 
              onClick={testAudioPlayback}
              title="Test Audio Playback"
            >
              🔊 Test
            </button>
          </div>

          <h3>Sessions</h3>
          <ul className="session-list">
            {sessions.map((session) => (
              <li
                key={session.session_id}
                className={`session-item ${
                  currentSession === session.session_id ? 'active' : ''
                }`}
              >
                <div 
                  className="session-content"
                  onClick={() => selectSession(session.session_id)}
                >
                  <div className="session-title">
                    Session {session.session_id.slice(-8)}
                  </div>
                  <div className="session-meta">
                    {session.message_count} messages
                  </div>
                </div>
                <button
                  className="delete-session-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(session.session_id);
                  }}
                  title="Delete session"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <div className="chat-container">
          {error && (
            <div className="error">
              <div className="error-content">
                <span className="error-message">{error}</span>
                {connectionStatus === 'error' && (
                  <div className="error-actions">
                    <button 
                      className="retry-btn" 
                      onClick={retryConnection}
                      disabled={isLoading}
                    >
                      🔄 Retry
                    </button>
                    <button 
                      className="clear-error-btn" 
                      onClick={clearError}
                    >
                      ✕
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          <TranscriptPanel
            transcript={transcript}
            isPartial={isPartialTranscript}
            language={detectedLanguage}
          />

          <ChatHistory
            messages={messages}
            isLoading={isLoading}
          />

          <ContextPanel
            contextDocs={contextDocs}
            language={detectedLanguage}
          />

          <div className="controls">
            <MicButton
              isRecording={isRecording}
              isEnabled={isMicEnabled}
              onStart={startRecording}
              onStop={stopRecording}
              onError={handleMicError}
            />
            <div className="status">
              {isPlayingAudio ? '🔊 Playing response...' : status}
            </div>
          </div>
        </div>
      </main>
      
      {/* Hidden audio element for playback */}
      <audio ref={audioRef} />
      
      {/* Stats Panel */}
      <StatsPanel 
        isVisible={showStats} 
        onClose={() => setShowStats(false)} 
      />
    </div>
  );
}

export default App;
