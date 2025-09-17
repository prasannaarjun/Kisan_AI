import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from './api';
import MicButton from './components/MicButton';
import TranscriptPanel from './components/TranscriptPanel';
import ChatHistory from './components/ChatHistory';

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

  // Refs
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

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
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus('Disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error');
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
          <button 
            className="new-session-btn" 
            onClick={createNewSession}
            disabled={isLoading}
          >
            + New Session
          </button>

          <h3>Sessions</h3>
          <ul className="session-list">
            {sessions.map((session) => (
              <li
                key={session.session_id}
                className={`session-item ${
                  currentSession === session.session_id ? 'active' : ''
                }`}
                onClick={() => selectSession(session.session_id)}
              >
                <div className="session-title">
                  Session {session.session_id.slice(-8)}
                </div>
                <div className="session-meta">
                  {session.message_count} messages
                </div>
              </li>
            ))}
          </ul>
        </aside>

        <div className="chat-container">
          {error && (
            <div className="error">
              {error}
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

          <div className="controls">
            <MicButton
              isRecording={isRecording}
              isEnabled={isMicEnabled}
              onStart={startRecording}
              onStop={stopRecording}
              onError={handleMicError}
            />
            <div className="status">
              {status}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
