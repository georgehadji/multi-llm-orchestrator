/**
 * WebSocket Service - Manages WebSocket connection
 */

const WS_URL = 'ws://localhost:8765/ws';

class WebSocketService {
  constructor() {
    this.ws = null;
    this.sessionId = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.pingInterval = null;
  }

  connect(sessionId) {
    this.sessionId = sessionId;
    const url = `${WS_URL}/${sessionId}`;
    console.log('[WS] Connecting to:', url);
    
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
          console.log('[WS] Connected');
          this.reconnectAttempts = 0;
          this.startPing();
          resolve();
        };
        
        this.ws.onclose = (event) => {
          console.log('[WS] Disconnected', event.code, event.reason);
          this.stopPing();
          this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
          console.error('[WS] Error', error);
          reject(error);
        };
        
        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            console.log('[WS] Received:', message);
            this.handleMessage(message);
          } catch (e) {
            console.error('[WS] Parse error', e);
          }
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  disconnect() {
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendMessage(event, data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ event, data, session_id: this.sessionId }));
    } else {
      console.warn('[WS] Not connected, cannot send', event);
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    };
  }

  off(event, callback) {
    if (callback) {
      const callbacks = this.listeners.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    } else {
      this.listeners.delete(event);
    }
  }

  handleMessage(message) {
    const { event, data } = message;
    const callbacks = this.listeners.get(event) || [];
    callbacks.forEach(cb => cb(data));
    
    // Also emit generic 'message' event
    const allCallbacks = this.listeners.get('message') || [];
    allCallbacks.forEach(cb => cb(message));
  }

  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] Max reconnect attempts reached');
      const callbacks = this.listeners.get('disconnected') || [];
      callbacks.forEach(cb => cb());
      return;
    }
    
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;
    
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (this.sessionId) {
        this.connect(this.sessionId).catch(console.error);
      }
    }, delay);
  }

  startPing() {
    this.pingInterval = setInterval(() => {
      this.sendMessage('ping', {});
    }, 30000);
  }

  stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let instance = null;

export function getWebSocketService() {
  if (!instance) {
    instance = new WebSocketService();
  }
  return instance;
}

export function resetWebSocketService() {
  if (instance) {
    instance.disconnect();
    instance = null;
  }
}
