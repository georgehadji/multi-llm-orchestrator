/**
 * WebSocket Hook - React hook for WebSocket communication
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { getWebSocketService } from '../services/websocket';

export function useWebSocket(sessionId) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const listenersRef = useRef([]);

  // Connect to WebSocket
  useEffect(() => {
    if (!sessionId) return;

    const ws = getWebSocketService();
    wsRef.current = ws;

    const connect = async () => {
      try {
        await ws.connect(sessionId);
        setIsConnected(true);
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        setIsConnected(false);
      }
    };

    connect();

    // Set up message listener
    const unsubscribe = ws.on('message', (message) => {
      setLastMessage(message);
      listenersRef.current.forEach(({ callback }) => callback(message));
    });

    const disconnectListener = ws.on('disconnected', () => {
      setIsConnected(false);
    });

    return () => {
      unsubscribe();
      disconnectListener();
    };
  }, [sessionId]);

  // Send message
  const sendMessage = useCallback((event, data) => {
    if (wsRef.current) {
      wsRef.current.sendMessage(event, data);
    }
  }, []);

  // Subscribe to specific event
  const on = useCallback((event, callback) => {
    if (wsRef.current) {
      return wsRef.current.on(event, callback);
    }
    return () => {};
  }, []);

  // Unsubscribe from event
  const off = useCallback((event, callback) => {
    if (wsRef.current) {
      wsRef.current.off(event, callback);
    }
  }, []);

  // Disconnect
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      setIsConnected(false);
    }
  }, []);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    on,
    off,
    disconnect,
  };
}
