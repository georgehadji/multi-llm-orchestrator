/**
 * Session Hook - React hook for session management
 */
import { useState, useEffect, useCallback } from 'react';
import { getApiService } from '../services/api';
import { useWebSocket } from './useWebSocket';

export function useSession(initialSessionId) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(initialSessionId);

  const api = getApiService();
  const {
    isConnected,
    sendMessage,
    on,
    off,
  } = useWebSocket(sessionId);

  // Load session
  useEffect(() => {
    const loadSession = async () => {
      if (!sessionId) {
        // Create new session
        try {
          const result = await api.createSession({});
          setSessionId(result.session.id);
          setSession(result.session);
        } catch (err) {
          setError(err.message);
        }
        setLoading(false);
        return;
      }

      try {
        const result = await api.getSession(sessionId);
        setSession(result.session);
      } catch (err) {
        setError(err.message);
        // Create new session if load fails
        try {
          const result = await api.createSession({});
          setSessionId(result.session.id);
          setSession(result.session);
        } catch (createErr) {
          setError(createErr.message);
        }
      }
      setLoading(false);
    };

    loadSession();
  }, [sessionId]);

  // Listen for WebSocket updates
  useEffect(() => {
    if (!isConnected) return;

    const handleSessionState = (data) => {
      setSession(data);
    };

    const handleMessagesUpdate = (data) => {
      setSession(prev => prev ? { ...prev, messages: data.messages } : null);
    };

    const handleFilesUpdate = (data) => {
      setSession(prev => prev ? { ...prev, files: data.files } : null);
    };

    const handleTerminalUpdate = (data) => {
      setSession(prev => prev ? { ...prev, terminal_lines: data.lines } : null);
    };

    on('session_state', handleSessionState);
    on('messages_update', handleMessagesUpdate);
    on('files_update', handleFilesUpdate);
    on('terminal_update', handleTerminalUpdate);

    return () => {
      off('session_state', handleSessionState);
      off('messages_update', handleMessagesUpdate);
      off('files_update', handleFilesUpdate);
      off('terminal_update', handleTerminalUpdate);
    };
  }, [isConnected, on, off]);

  // Send chat message
  const sendChatMessage = useCallback(async (message) => {
    if (!sessionId) return;

    // Send via WebSocket for real-time (API call adds duplicate)
    sendMessage('chat_message', { message, session_id: sessionId });
  }, [sessionId, sendMessage]);

  // Update session settings
  const updateSettings = useCallback((settings) => {
    sendMessage('session_update', { ...settings, session_id: sessionId });
  }, [sessionId, sendMessage]);

  // Get file content
  const getFileContent = useCallback(async (filePath) => {
    if (!sessionId) return null;
    return api.getFileContent(sessionId, filePath);
  }, [sessionId]);

  // Update file content
  const updateFileContent = useCallback(async (filePath, content) => {
    if (!sessionId) return false;
    const result = await api.updateFile(sessionId, filePath, content);
    sendMessage('file_update', { path: filePath, content, session_id: sessionId });
    return result;
  }, [sessionId, sendMessage]);

  // Execute terminal command
  const runTerminalCommand = useCallback((command) => {
    sendMessage('terminal_command', { command, session_id: sessionId });
  }, [sessionId, sendMessage]);

  return {
    session,
    sessionId,
    isConnected,
    loading,
    error,
    sendMessage,
    sendChatMessage,
    updateSettings,
    getFileContent,
    updateFileContent,
    runTerminalCommand,
  };
}
