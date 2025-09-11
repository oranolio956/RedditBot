/**
 * WebSocket Manager for Real-time Features
 * Handles all real-time updates for consciousness, metrics, and quantum features
 */

import { io, Socket } from 'socket.io-client';
import {
  WebSocketMessage,
  ConsciousnessUpdate,
  MetricsUpdate,
  EmotionalStateUpdate,
  MemoryPalaceEvent,
  QuantumEvent,
} from '@/types';

export type WebSocketEventType = 
  | 'consciousness_update'
  | 'metrics_update'
  | 'emotional_state_update'
  | 'memory_palace_event'
  | 'quantum_event'
  | 'notification'
  | 'system_alert'
  | 'kelly_conversation_update'
  | 'kelly_safety_alert'
  | 'claude_response_generation'
  | 'claude_cost_update';

export type WebSocketEventHandler<T = any> = (data: T) => void;

class WebSocketManager {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers = new Map<string, Set<WebSocketEventHandler>>();
  private isConnected = false;
  private connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';

  constructor() {
    this.handleVisibilityChange();
  }

  // Connection management
  connect(userId?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve();
        return;
      }

      this.connectionStatus = 'connecting';
      
      const wsUrl = import.meta.env.VITE_WS_URL || 'http://localhost:8000';
      
      this.socket = io(wsUrl, {
        transports: ['websocket', 'polling'],
        auth: {
          userId: userId,
          token: localStorage.getItem('auth_token'),
        },
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
      });

      this.setupEventHandlers(resolve, reject);
    });
  }

  private setupEventHandlers(resolve: () => void, reject: (error: any) => void) {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('🔗 WebSocket connected');
      this.isConnected = true;
      this.connectionStatus = 'connected';
      this.reconnectAttempts = 0;
      resolve();
    });

    this.socket.on('disconnect', (reason) => {
      console.log('💔 WebSocket disconnected:', reason);
      this.isConnected = false;
      this.connectionStatus = 'disconnected';
      this.notifyConnectionChange(false);
    });

    this.socket.on('connect_error', (error) => {
      console.error('❌ WebSocket connection error:', error);
      this.connectionStatus = 'error';
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        reject(error);
      }
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`🔄 WebSocket reconnected after ${attemptNumber} attempts`);
      this.isConnected = true;
      this.connectionStatus = 'connected';
      this.notifyConnectionChange(true);
    });

    // Set up message routing
    this.setupMessageRouting();
  }

  private setupMessageRouting() {
    if (!this.socket) return;

    // Consciousness updates
    this.socket.on('consciousness_update', (data: ConsciousnessUpdate) => {
      this.notifyHandlers('consciousness_update', data);
    });

    // Telegram metrics updates
    this.socket.on('metrics_update', (data: MetricsUpdate) => {
      this.notifyHandlers('metrics_update', data);
    });

    // Emotional state updates
    this.socket.on('emotional_state_update', (data: EmotionalStateUpdate) => {
      this.notifyHandlers('emotional_state_update', data);
    });

    // Memory palace events
    this.socket.on('memory_palace_event', (data: MemoryPalaceEvent) => {
      this.notifyHandlers('memory_palace_event', data);
    });

    // Quantum consciousness events
    this.socket.on('quantum_event', (data: QuantumEvent) => {
      this.notifyHandlers('quantum_event', data);
    });

    // General notifications
    this.socket.on('notification', (data: any) => {
      this.notifyHandlers('notification', data);
    });

    // System alerts
    this.socket.on('system_alert', (data: any) => {
      this.notifyHandlers('system_alert', data);
    });

    // Kelly conversation updates
    this.socket.on('kelly_conversation_update', (data: any) => {
      this.notifyHandlers('kelly_conversation_update', data);
    });

    // Kelly safety alerts
    this.socket.on('kelly_safety_alert', (data: any) => {
      this.notifyHandlers('kelly_safety_alert', data);
    });

    // Claude response generation updates
    this.socket.on('claude_response_generation', (data: any) => {
      this.notifyHandlers('claude_response_generation', data);
    });

    // Claude cost updates
    this.socket.on('claude_cost_update', (data: any) => {
      this.notifyHandlers('claude_cost_update', data);
    });

    // Generic message handler
    this.socket.onAny((eventName: string, data: any) => {
      console.log(`📨 WebSocket event: ${eventName}`, data);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.connectionStatus = 'disconnected';
    }
  }

  // Event subscription
  subscribe<T>(eventType: WebSocketEventType, handler: WebSocketEventHandler<T>): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }
    
    this.eventHandlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.eventHandlers.get(eventType);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.eventHandlers.delete(eventType);
        }
      }
    };
  }

  // Room management for targeted updates
  joinRoom(room: string) {
    if (this.socket?.connected) {
      this.socket.emit('join_room', room);
      console.log(`🏠 Joined WebSocket room: ${room}`);
    }
  }

  leaveRoom(room: string) {
    if (this.socket?.connected) {
      this.socket.emit('leave_room', room);
      console.log(`🚪 Left WebSocket room: ${room}`);
    }
  }

  // Specialized room management
  joinUserRoom(userId: string) {
    this.joinRoom(`user:${userId}`);
  }

  joinConsciousnessRoom(userId: string) {
    this.joinRoom(`consciousness:${userId}`);
  }

  joinMemoryPalaceRoom(palaceId: string) {
    this.joinRoom(`memory_palace:${palaceId}`);
  }

  joinQuantumNetworkRoom(userId: string) {
    this.joinRoom(`quantum:${userId}`);
  }

  joinTelegramMetricsRoom() {
    this.joinRoom('telegram:metrics');
  }

  // Kelly-specific room management
  joinKellyAccountRoom(accountId: string) {
    this.joinRoom(`kelly:account:${accountId}`);
  }

  joinKellyConversationRoom(conversationId: string) {
    this.joinRoom(`kelly:conversation:${conversationId}`);
  }

  joinKellySafetyRoom() {
    this.joinRoom('kelly:safety');
  }

  joinClaudeMetricsRoom() {
    this.joinRoom('claude:metrics');
  }

  // Send messages to server
  emit(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('⚠️ Attempted to emit event while disconnected:', event);
    }
  }

  // Status getters
  get connected(): boolean {
    return this.isConnected;
  }

  get status(): string {
    return this.connectionStatus;
  }

  get socketId(): string | undefined {
    return this.socket?.id;
  }

  // Private helper methods
  private notifyHandlers(eventType: string, data: any) {
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${eventType}:`, error);
        }
      });
    }
  }

  private notifyConnectionChange(connected: boolean) {
    this.notifyHandlers('connection_change', { connected, status: this.connectionStatus });
  }

  private handleVisibilityChange() {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        // Page is hidden - reduce connection activity
        console.log('📱 Page hidden, reducing WebSocket activity');
      } else {
        // Page is visible - restore full activity
        console.log('👁️ Page visible, restoring WebSocket activity');
        if (!this.isConnected && this.socket) {
          this.socket.connect();
        }
      }
    });
  }
}

// Export singleton instance
export const wsManager = new WebSocketManager();

// React hook for WebSocket connection
export function useWebSocket() {
  return {
    connect: wsManager.connect.bind(wsManager),
    disconnect: wsManager.disconnect.bind(wsManager),
    subscribe: wsManager.subscribe.bind(wsManager),
    joinRoom: wsManager.joinRoom.bind(wsManager),
    leaveRoom: wsManager.leaveRoom.bind(wsManager),
    joinUserRoom: wsManager.joinUserRoom.bind(wsManager),
    joinConsciousnessRoom: wsManager.joinConsciousnessRoom.bind(wsManager),
    joinMemoryPalaceRoom: wsManager.joinMemoryPalaceRoom.bind(wsManager),
    joinQuantumNetworkRoom: wsManager.joinQuantumNetworkRoom.bind(wsManager),
    joinTelegramMetricsRoom: wsManager.joinTelegramMetricsRoom.bind(wsManager),
    emit: wsManager.emit.bind(wsManager),
    connected: wsManager.connected,
    status: wsManager.status,
    socketId: wsManager.socketId,
  };
}

// Specialized hooks for different features
export function useConsciousnessUpdates(userId: string, onUpdate: WebSocketEventHandler<ConsciousnessUpdate>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!userId) return;
    
    ws.joinConsciousnessRoom(userId);
    const unsubscribe = ws.subscribe('consciousness_update', onUpdate);
    
    return () => {
      unsubscribe();
    };
  }, [userId, onUpdate]);
}

export function useTelegramMetrics(onUpdate: WebSocketEventHandler<MetricsUpdate>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    ws.joinTelegramMetricsRoom();
    const unsubscribe = ws.subscribe('metrics_update', onUpdate);
    
    return () => {
      unsubscribe();
    };
  }, [onUpdate]);
}

export function useEmotionalStateUpdates(userId: string, onUpdate: WebSocketEventHandler<EmotionalStateUpdate>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!userId) return;
    
    ws.joinUserRoom(userId);
    const unsubscribe = ws.subscribe('emotional_state_update', onUpdate);
    
    return () => {
      unsubscribe();
    };
  }, [userId, onUpdate]);
}

export function useMemoryPalaceEvents(palaceId: string, onEvent: WebSocketEventHandler<MemoryPalaceEvent>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!palaceId) return;
    
    ws.joinMemoryPalaceRoom(palaceId);
    const unsubscribe = ws.subscribe('memory_palace_event', onEvent);
    
    return () => {
      unsubscribe();
    };
  }, [palaceId, onEvent]);
}

export function useQuantumNetwork(userId: string, onEvent: WebSocketEventHandler<QuantumEvent>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!userId) return;
    
    ws.joinQuantumNetworkRoom(userId);
    const unsubscribe = ws.subscribe('quantum_event', onEvent);
    
    return () => {
      unsubscribe();
    };
  }, [userId, onEvent]);
}

export function useNotifications(onNotification: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    const unsubscribe = ws.subscribe('notification', onNotification);
    return unsubscribe;
  }, [onNotification]);
}

export function useSystemAlerts(onAlert: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    const unsubscribe = ws.subscribe('system_alert', onAlert);
    return unsubscribe;
  }, [onAlert]);
}

// Kelly-specific hooks
export function useKellyConversationUpdates(conversationId: string, onUpdate: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!conversationId) return;
    
    ws.joinKellyConversationRoom(conversationId);
    const unsubscribe = ws.subscribe('kelly_conversation_update', onUpdate);
    
    return () => {
      unsubscribe();
    };
  }, [conversationId, onUpdate]);
}

export function useKellySafetyAlerts(onAlert: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    ws.joinKellySafetyRoom();
    const unsubscribe = ws.subscribe('kelly_safety_alert', onAlert);
    
    return () => {
      unsubscribe();
    };
  }, [onAlert]);
}

export function useClaudeResponseGeneration(conversationId: string, onUpdate: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!conversationId) return;
    
    const unsubscribe = ws.subscribe('claude_response_generation', (data) => {
      if (data.payload.conversation_id === conversationId) {
        onUpdate(data);
      }
    });
    
    return () => {
      unsubscribe();
    };
  }, [conversationId, onUpdate]);
}

export function useClaudeCostUpdates(accountId: string, onUpdate: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    if (!accountId) return;
    
    ws.joinKellyAccountRoom(accountId);
    const unsubscribe = ws.subscribe('claude_cost_update', (data) => {
      if (data.payload.account_id === accountId) {
        onUpdate(data);
      }
    });
    
    return () => {
      unsubscribe();
    };
  }, [accountId, onUpdate]);
}

export function useClaudeMetrics(onUpdate: WebSocketEventHandler<any>) {
  const ws = useWebSocket();
  
  React.useEffect(() => {
    ws.joinClaudeMetricsRoom();
    const unsubscribe = ws.subscribe('claude_cost_update', onUpdate);
    
    return () => {
      unsubscribe();
    };
  }, [onUpdate]);
}

export default wsManager;