/**
 * API Service - REST API client
 */

const API_URL = 'http://localhost:8765/api';

class ApiService {
  constructor() {
    this.baseUrl = API_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Request failed');
      }

      return data;
    } catch (error) {
      console.error('[API] Error', endpoint, error);
      throw error;
    }
  }

  // Session methods
  async createSession(sessionData) {
    return this.request('/session', {
      method: 'POST',
      body: JSON.stringify(sessionData),
    });
  }

  async getSession(sessionId) {
    return this.request(`/session/${sessionId}`);
  }

  async listSessions() {
    return this.request('/sessions');
  }

  async deleteSession(sessionId) {
    return this.request(`/sessions/${sessionId}`, { method: 'DELETE' });
  }

  // Chat methods
  async sendMessage(message, sessionId) {
    return this.request('/chat', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId }),
    });
  }

  // File methods
  async getFiles(sessionId) {
    return this.request(`/sessions/${sessionId}/files`);
  }

  async getFileContent(sessionId, filePath) {
    return this.request(`/sessions/${sessionId}/files/${encodeURIComponent(filePath)}`);
  }

  async updateFile(sessionId, filePath, content, language) {
    return this.request(`/sessions/${sessionId}/files/${encodeURIComponent(filePath)}`, {
      method: 'PUT',
      body: JSON.stringify({ content, language }),
    });
  }

  // Task methods
  async getTasks(sessionId) {
    return this.request(`/sessions/${sessionId}/tasks`);
  }

  async updateTask(sessionId, taskId, updates) {
    return this.request(`/sessions/${sessionId}/tasks/${taskId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  // Model methods
  async getModels() {
    return this.request('/models');
  }

  // Health check
  async health() {
    return this.request('/health');
  }
}

// Singleton instance
let instance = null;

export function getApiService() {
  if (!instance) {
    instance = new ApiService();
  }
  return instance;
}
