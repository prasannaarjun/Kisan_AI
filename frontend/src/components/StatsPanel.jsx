import React, { useState, useEffect } from 'react';
import { apiService } from '../api';

const StatsPanel = ({ isVisible, onClose }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isVisible) {
      loadStats();
    }
  }, [isVisible]);

  const loadStats = async () => {
    try {
      setLoading(true);
      const statsData = await apiService.getStats();
      setStats(statsData);
    } catch (err) {
      console.error('Error loading stats:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!isVisible) return null;

  return (
    <div className="stats-overlay">
      <div className="stats-panel">
        <div className="stats-header">
          <h3>📊 Session Statistics</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        {loading ? (
          <div className="loading">Loading statistics...</div>
        ) : stats ? (
          <div className="stats-content">
            <div className="stat-item">
              <span className="stat-label">Total Sessions:</span>
              <span className="stat-value">{stats.total_sessions || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Total Messages:</span>
              <span className="stat-value">{stats.total_messages || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Active Sessions:</span>
              <span className="stat-value">{stats.active_sessions || 0}</span>
            </div>
            {stats.avg_messages_per_session && (
              <div className="stat-item">
                <span className="stat-label">Avg Messages/Session:</span>
                <span className="stat-value">
                  {Math.round(stats.avg_messages_per_session * 10) / 10}
                </span>
              </div>
            )}
            {stats.most_common_language && (
              <div className="stat-item">
                <span className="stat-label">Most Common Language:</span>
                <span className="stat-value">
                  {stats.most_common_language.toUpperCase()}
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="error">Failed to load statistics</div>
        )}
      </div>
    </div>
  );
};

export default StatsPanel;
