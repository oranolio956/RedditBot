/**
 * WebSocket Hook for Real-time Updates
 * Handles connection states, reconnection, and real-time message updates
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export type WebSocketState = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface UseWebSocketOptions {
  url: string;
  protocols?: string | string[];
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  shouldReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  debug?: boolean;
}

export interface UseWebSocketReturn {
  state: WebSocketState;
  send: (message: any) => void;
  lastMessage: WebSocketMessage | null;
  connect: () => void;
  disconnect: () => void;
  reconnectAttempts: number;
  isConnected: boolean;
}

export const useWebSocket = (options: UseWebSocketOptions): UseWebSocketReturn => {
  const {
    url,
    protocols,
    onMessage,
    onError,
    onOpen,
    onClose,
    shouldReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    heartbeatInterval = 30000,
    debug = false
  } = options;

  const [state, setState] = useState<WebSocketState>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isManualClose = useRef(false);

  const log = useCallback((message: string, ...args: any[]) => {
    if (debug) {
      console.log(`[WebSocket] ${message}`, ...args);
    }
  }, [debug]);

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    if (heartbeatInterval > 0) {
      heartbeatTimeoutRef.current = setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
          log('Heartbeat sent');
          startHeartbeat();
        }
      }, heartbeatInterval);
    }
  }, [heartbeatInterval, log]);

  const handleReconnect = useCallback(() => {
    if (!shouldReconnect || isManualClose.current || reconnectAttempts >= maxReconnectAttempts) {
      log('Reconnection stopped', { shouldReconnect, isManualClose: isManualClose.current, reconnectAttempts, maxReconnectAttempts });
      return;
    }

    log(`Attempting to reconnect (${reconnectAttempts + 1}/${maxReconnectAttempts})`);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      setReconnectAttempts(prev => prev + 1);
      connect();
    }, reconnectInterval);
  }, [shouldReconnect, reconnectAttempts, maxReconnectAttempts, reconnectInterval, log]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      log('Already connected');
      return;
    }

    log('Connecting to', url);
    setState('connecting');
    clearTimeouts();

    try {
      wsRef.current = new WebSocket(url, protocols);

      wsRef.current.onopen = (event) => {
        log('Connected');
        setState('connected');
        setReconnectAttempts(0);
        isManualClose.current = false;
        startHeartbeat();
        onOpen?.(event);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Handle pong responses
          if (message.type === 'pong') {
            log('Heartbeat received');
            return;
          }

          log('Message received', message);
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          log('Failed to parse message', error, event.data);
        }
      };

      wsRef.current.onerror = (event) => {
        log('Error occurred', event);
        setState('error');
        onError?.(event);
      };

      wsRef.current.onclose = (event) => {
        log('Connection closed', event.code, event.reason);
        setState('disconnected');
        clearTimeouts();
        onClose?.(event);

        // Only attempt to reconnect if it wasn't a manual close
        if (!isManualClose.current && shouldReconnect) {
          handleReconnect();
        }
      };
    } catch (error) {
      log('Failed to create WebSocket', error);
      setState('error');
    }
  }, [url, protocols, onOpen, onMessage, onError, onClose, shouldReconnect, startHeartbeat, handleReconnect, log]);

  const disconnect = useCallback(() => {
    log('Manually disconnecting');
    isManualClose.current = true;
    clearTimeouts();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setState('disconnected');
    setReconnectAttempts(0);
  }, [clearTimeouts, log]);

  const send = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const messageString = typeof message === 'string' ? message : JSON.stringify(message);
      wsRef.current.send(messageString);
      log('Message sent', message);
    } else {
      log('Cannot send message - WebSocket not connected', { state: wsRef.current?.readyState });
    }
  }, [log]);

  // Auto-connect on mount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [url]); // Only reconnect when URL changes

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimeouts();
      if (wsRef.current) {
        isManualClose.current = true;
        wsRef.current.close();
      }
    };
  }, [clearTimeouts]);

  return {
    state,
    send,
    lastMessage,
    connect,
    disconnect,
    reconnectAttempts,
    isConnected: state === 'connected'
  };
};

// Specialized hook for Kelly real-time updates
export const useKellyWebSocket = () => {
  const [conversations, setConversations] = useState<any[]>([]);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [systemStatus, setSystemStatus] = useState<any>(null);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'conversation_update':
        setConversations(prev => 
          prev.map(conv => 
            conv.id === message.data.id ? { ...conv, ...message.data } : conv
          )
        );
        break;

      case 'new_conversation':
        setConversations(prev => [message.data, ...prev]);
        break;

      case 'conversation_ended':
        setConversations(prev => 
          prev.filter(conv => conv.id !== message.data.conversation_id)
        );
        break;

      case 'new_message':
        // This would be handled by individual conversation components
        break;

      case 'safety_alert':
        setNotifications(prev => [
          {
            id: Date.now().toString(),
            type: 'safety',
            message: message.data.alert,
            timestamp: new Date(),
            severity: message.data.severity
          },
          ...prev.slice(0, 9)
        ]);
        break;

      case 'system_status':
        setSystemStatus(message.data);
        break;

      case 'intervention_required':
        setNotifications(prev => [
          {
            id: Date.now().toString(),
            type: 'intervention',
            message: `Conversation ${message.data.conversation_id} requires human review`,
            timestamp: new Date(),
            conversationId: message.data.conversation_id
          },
          ...prev.slice(0, 9)
        ]);
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, []);

  const websocket = useWebSocket({
    url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v1/kelly/ws`,
    onMessage: handleMessage,
    shouldReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 10,
    heartbeatInterval: 30000,
    debug: process.env.NODE_ENV === 'development'
  });

  // Subscribe to conversation updates when connected
  useEffect(() => {
    if (websocket.isConnected) {
      websocket.send({
        type: 'subscribe',
        channels: ['conversations', 'safety_alerts', 'system_status']
      });
    }
  }, [websocket.isConnected]);

  return {
    ...websocket,
    conversations,
    notifications,
    systemStatus,
    subscribeToConversation: (conversationId: string) => {
      websocket.send({
        type: 'subscribe_conversation',
        conversation_id: conversationId
      });
    },
    unsubscribeFromConversation: (conversationId: string) => {
      websocket.send({
        type: 'unsubscribe_conversation',
        conversation_id: conversationId
      });
    },
    clearNotifications: () => setNotifications([])
  };
};