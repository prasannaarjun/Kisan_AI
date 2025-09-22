import React from 'react';

const ContextPanel = ({ contextDocs, language }) => {
  if (!contextDocs || contextDocs.length === 0) {
    return null;
  }

  return (
    <div className="context-panel">
      <h4>📚 Knowledge Sources</h4>
      <div className="context-docs">
        {contextDocs.map((doc, index) => (
          <div key={index} className="context-doc">
            <div className="context-text">
              {doc.text}
            </div>
            {doc.metadata && (
              <div className="context-meta">
                {doc.metadata.topic && (
                  <span className="context-topic">
                    {doc.metadata.topic.replace('_', ' ')}
                  </span>
                )}
                {doc.metadata.crop && (
                  <span className="context-crop">
                    {doc.metadata.crop}
                  </span>
                )}
                {doc.score && (
                  <span className="context-score">
                    {Math.round(doc.score * 100)}% match
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ContextPanel;
