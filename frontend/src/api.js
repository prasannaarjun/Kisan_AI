import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const apiService = {
  // Session management
  startSession: async (sessionId = null) => {
    const response = await api.post('/start_session', { session_id: sessionId });
    return response.data;
  },

  getSessions: async (limit = 20) => {
    const response = await api.get(`/sessions?limit=${limit}`);
    return response.data;
  },

  getSessionHistory: async (sessionId, limit = 50) => {
    const response = await api.get(`/sessions/${sessionId}/history?limit=${limit}`);
    return response.data;
  },

  deleteSession: async (sessionId) => {
    const response = await api.delete(`/sessions/${sessionId}`);
    return response.data;
  },

  getStats: async () => {
    const response = await api.get('/stats');
    return response.data;
  },

  // WebSocket connection
  createWebSocketConnection: (sessionId) => {
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/conversation/${sessionId}`;
    return new WebSocket(wsUrl);
  },
};

export default api;
