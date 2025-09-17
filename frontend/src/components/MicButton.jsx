import React from 'react';
import { Mic, MicOff } from 'lucide-react';

const MicButton = ({ isRecording, isEnabled, onStart, onStop, onError }) => {
  const handleClick = () => {
    if (!isEnabled) {
      onError('Microphone not available');
      return;
    }

    if (isRecording) {
      onStop();
    } else {
      onStart();
    }
  };

  return (
    <button
      className={`mic-button ${isRecording ? 'recording' : ''}`}
      onClick={handleClick}
      disabled={!isEnabled}
      title={isRecording ? 'Stop recording' : 'Start recording'}
    >
      {isRecording ? <MicOff size={24} /> : <Mic size={24} />}
    </button>
  );
};

export default MicButton;
