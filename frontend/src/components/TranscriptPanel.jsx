import React from 'react';

const TranscriptPanel = ({ transcript, isPartial, language }) => {
  if (!transcript) {
    return (
      <div className="transcript-panel">
        <h3>Live Transcript</h3>
        <div className="transcript-text">
          Click the microphone to start speaking...
        </div>
      </div>
    );
  }

  return (
    <div className="transcript-panel">
      <h3>
        Live Transcript
        {language && (
          <span style={{ fontSize: '0.8rem', fontWeight: 'normal', marginLeft: '0.5rem' }}>
            ({language.toUpperCase()})
          </span>
        )}
      </h3>
      <div className="transcript-text">
        {transcript}
        {isPartial && <span style={{ opacity: 0.7 }}>...</span>}
      </div>
    </div>
  );
};

export default TranscriptPanel;
