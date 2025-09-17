import React from 'react';

const ChatHistory = ({ messages, isLoading }) => {
  if (isLoading) {
    return (
      <div className="chat-messages">
        <div className="loading">
          <div className="spinner"></div>
          Loading messages...
        </div>
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className="chat-messages">
        <div style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
          No messages yet. Start a conversation by clicking the microphone!
        </div>
      </div>
    );
  }

  return (
    <div className="chat-messages">
      {messages.map((message, index) => (
        <div key={index} className={`message ${message.role}`}>
          <div className="message-header">
            {message.role === 'user' ? 'You' : 'KisanAI'}
            {message.language && (
              <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem' }}>
                ({message.language.toUpperCase()})
              </span>
            )}
            <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem' }}>
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
          </div>
          <div className="message-text">{message.text}</div>
        </div>
      ))}
    </div>
  );
};

export default ChatHistory;
