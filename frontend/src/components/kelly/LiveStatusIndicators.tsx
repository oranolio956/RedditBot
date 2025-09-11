/**
 * Live Status Indicators for Kelly
 * AI thinking indicators, human agent typing, multi-agent status
 * Real-time status broadcasting with Apple-inspired animations
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { useWebSocket, useClaudeResponseGeneration } from '@/lib/websocket';
import { cn } from '@/lib/utils';

interface StatusIndicatorState {
  type: 'ai_thinking' | 'ai_generating' | 'human_typing' | 'multi_agent' | 'idle' | 'error';
  message: string;
  progress?: number;
  estimatedCompletion?: Date;
  agentInfo?: {
    name: string;
    type: 'ai' | 'human';
    avatar?: string;
  };
  metadata?: any;
}

interface TypingIndicatorProps {
  show: boolean;
  className?: string;
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({ show, className }) => {
  if (!show) return null;

  return (
    <div className={cn("flex items-center space-x-1", className)}>
      <div className="flex space-x-1">
        <div 
          className="h-2 w-2 bg-consciousness-primary rounded-full animate-bounce"
          style={{ animationDelay: '0ms', animationDuration: '1.4s' }}
        />
        <div 
          className="h-2 w-2 bg-consciousness-primary rounded-full animate-bounce"
          style={{ animationDelay: '160ms', animationDuration: '1.4s' }}
        />
        <div 
          className="h-2 w-2 bg-consciousness-primary rounded-full animate-bounce"
          style={{ animationDelay: '320ms', animationDuration: '1.4s' }}
        />
      </div>
    </div>
  );
};

interface ThinkingIndicatorProps {
  show: boolean;
  progress?: number;
  className?: string;
}

const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ 
  show, 
  progress,
  className 
}) => {
  if (!show) return null;

  return (
    <div className={cn("flex items-center space-x-3", className)}>
      {/* Pulsing brain icon */}
      <div className="relative">
        <span className="text-xl animate-pulse">🧠</span>
        <div className="absolute -inset-1 bg-consciousness-primary/20 rounded-full animate-ping opacity-30" />
      </div>
      
      {/* Progress indicator if available */}
      {progress !== undefined && (
        <div className="flex-1 min-w-0">
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className="h-1.5 bg-consciousness-primary rounded-full transition-all duration-300 animate-pulse"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}
      
      {/* Animated dots */}
      <div className="flex space-x-1">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-1.5 w-1.5 bg-consciousness-primary rounded-full animate-pulse"
            style={{ animationDelay: `${i * 200}ms` }}
          />
        ))}
      </div>
    </div>
  );
};

interface AgentAvatarProps {
  agent: {
    name: string;
    type: 'ai' | 'human';
    avatar?: string;
  };
  isActive?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const AgentAvatar: React.FC<AgentAvatarProps> = ({ 
  agent, 
  isActive = false,
  size = 'md' 
}) => {
  const sizeClasses = useMemo(() => {
    switch (size) {
      case 'sm': return 'h-6 w-6 text-xs';
      case 'lg': return 'h-10 w-10 text-base';
      default: return 'h-8 w-8 text-sm';
    }
  }, [size]);

  const getDefaultIcon = useCallback(() => {
    return agent.type === 'ai' ? '🤖' : '👤';
  }, [agent.type]);

  return (
    <div className="relative">
      {agent.avatar ? (
        <img
          src={agent.avatar}
          alt={agent.name}
          className={cn(
            "rounded-full border-2 transition-all duration-200",
            sizeClasses,
            isActive ? "border-consciousness-primary shadow-glow" : "border-gray-300"
          )}
        />
      ) : (
        <div
          className={cn(
            "rounded-full border-2 flex items-center justify-center font-medium transition-all duration-200",
            sizeClasses,
            isActive ? "border-consciousness-primary bg-consciousness-primary/10" : "border-gray-300 bg-gray-100",
            agent.type === 'ai' ? "text-consciousness-primary" : "text-text-primary"
          )}
        >
          {agent.name.charAt(0) || getDefaultIcon()}
        </div>
      )}
      
      {/* Active indicator */}
      {isActive && (
        <div className="absolute -inset-0.5 rounded-full bg-consciousness-primary/20 animate-ping" />
      )}
    </div>
  );
};

export interface LiveStatusIndicatorsProps {
  conversationId: string;
  className?: string;
  compact?: boolean;
  showEstimatedTime?: boolean;
  onStatusChange?: (status: StatusIndicatorState) => void;
}

export const LiveStatusIndicators: React.FC<LiveStatusIndicatorsProps> = ({
  conversationId,
  className,
  compact = false,
  showEstimatedTime = true,
  onStatusChange
}) => {
  const [currentStatus, setCurrentStatus] = useState<StatusIndicatorState>({
    type: 'idle',
    message: 'Ready'
  });
  const [multiAgentStatus, setMultiAgentStatus] = useState<{
    ai: { active: boolean; thinking: boolean; confidence?: number };
    human: { active: boolean; typing: boolean; name?: string };
  }>({
    ai: { active: false, thinking: false },
    human: { active: false, typing: false }
  });

  const ws = useWebSocket();

  // Handle Claude response generation
  useClaudeResponseGeneration(conversationId, useCallback((data: any) => {
    const payload = data.payload;
    
    switch (payload.status) {
      case 'thinking':
        setCurrentStatus({
          type: 'ai_thinking',
          message: 'Claude is analyzing the conversation...',
          agentInfo: {
            name: 'Claude',
            type: 'ai'
          },
          metadata: payload
        });
        
        setMultiAgentStatus(prev => ({
          ...prev,
          ai: { active: true, thinking: true, confidence: payload.confidence_so_far }
        }));
        break;
        
      case 'generating':
        setCurrentStatus({
          type: 'ai_generating',
          message: payload.partial_response 
            ? `Claude: "${payload.partial_response.slice(0, 50)}${payload.partial_response.length > 50 ? '...' : ''}"`
            : 'Claude is writing a response...',
          progress: payload.progress,
          estimatedCompletion: payload.estimated_completion_time 
            ? new Date(Date.now() + payload.estimated_completion_time) 
            : undefined,
          agentInfo: {
            name: 'Claude',
            type: 'ai'
          },
          metadata: payload
        });
        
        setMultiAgentStatus(prev => ({
          ...prev,
          ai: { active: true, thinking: false, confidence: payload.confidence_so_far }
        }));
        break;
        
      case 'complete':
        setCurrentStatus({
          type: 'idle',
          message: 'Response ready',
          agentInfo: {
            name: 'Claude',
            type: 'ai'
          }
        });
        
        setMultiAgentStatus(prev => ({
          ...prev,
          ai: { active: false, thinking: false }
        }));
        
        // Auto-clear after 3 seconds
        setTimeout(() => {
          setCurrentStatus({ type: 'idle', message: 'Ready' });
        }, 3000);
        break;
        
      case 'error':
        setCurrentStatus({
          type: 'error',
          message: payload.error_message || 'AI response failed',
          agentInfo: {
            name: 'Claude',
            type: 'ai'
          }
        });
        
        setMultiAgentStatus(prev => ({
          ...prev,
          ai: { active: false, thinking: false }
        }));
        break;
    }
  }, []));

  // Handle human agent typing
  useEffect(() => {
    const unsubscribeTyping = ws.subscribe('human_agent_typing', (data: any) => {
      if (data.conversation_id !== conversationId) return;
      
      if (data.typing) {
        setCurrentStatus({
          type: 'human_typing',
          message: `${data.agent_name} is typing...`,
          agentInfo: {
            name: data.agent_name,
            type: 'human',
            avatar: data.agent_avatar
          }
        });
        
        setMultiAgentStatus(prev => ({
          ...prev,
          human: { active: true, typing: true, name: data.agent_name }
        }));
      } else {
        setMultiAgentStatus(prev => ({
          ...prev,
          human: { active: false, typing: false }
        }));
        
        if (currentStatus.type === 'human_typing') {
          setCurrentStatus({ type: 'idle', message: 'Ready' });
        }
      }
    });

    // Handle multi-agent scenarios
    const unsubscribeMultiAgent = ws.subscribe('multi_agent_status', (data: any) => {
      if (data.conversation_id !== conversationId) return;
      
      setCurrentStatus({
        type: 'multi_agent',
        message: data.message || 'Multiple agents are working...',
        metadata: data
      });
    });

    return () => {
      unsubscribeTyping();
      unsubscribeMultiAgent();
    };
  }, [ws, conversationId, currentStatus.type]);

  // Broadcast status changes
  useEffect(() => {
    onStatusChange?.(currentStatus);
    
    // Broadcast via WebSocket for other clients
    ws.emit('status_update', {
      conversation_id: conversationId,
      status: currentStatus,
      multi_agent_status: multiAgentStatus
    });
  }, [currentStatus, multiAgentStatus, conversationId, ws, onStatusChange]);

  // Time remaining calculation
  const timeRemaining = useMemo(() => {
    if (!currentStatus.estimatedCompletion || !showEstimatedTime) return null;
    
    const remaining = currentStatus.estimatedCompletion.getTime() - Date.now();
    if (remaining <= 0) return null;
    
    const seconds = Math.ceil(remaining / 1000);
    return seconds > 60 ? `${Math.ceil(seconds / 60)}m` : `${seconds}s`;
  }, [currentStatus.estimatedCompletion, showEstimatedTime]);

  // Render different status types
  const renderStatusContent = useCallback(() => {
    const baseClasses = "flex items-center space-x-3 py-2 px-4 rounded-lg transition-all duration-300";
    
    switch (currentStatus.type) {
      case 'ai_thinking':
        return (
          <div className={cn(baseClasses, "bg-consciousness-primary/5 border border-consciousness-primary/20")}>
            <ThinkingIndicator 
              show={true} 
              progress={currentStatus.progress}
              className="flex-1"
            />
            {!compact && (
              <div className="flex items-center space-x-2">
                <AgentAvatar 
                  agent={currentStatus.agentInfo!} 
                  isActive={true}
                  size="sm"
                />
                <span className="text-caption-text text-consciousness-primary font-medium">
                  {currentStatus.message}
                </span>
                {timeRemaining && (
                  <span className="text-caption-text text-text-tertiary">
                    ~{timeRemaining}
                  </span>
                )}
              </div>
            )}
          </div>
        );
        
      case 'ai_generating':
        return (
          <div className={cn(baseClasses, "bg-consciousness-secondary/5 border border-consciousness-secondary/20")}>
            <AgentAvatar 
              agent={currentStatus.agentInfo!} 
              isActive={true}
              size="sm"
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2">
                <span className="text-caption-text text-consciousness-secondary font-medium truncate">
                  {currentStatus.message}
                </span>
                {timeRemaining && (
                  <span className="text-caption-text text-text-tertiary">
                    ~{timeRemaining}
                  </span>
                )}
              </div>
              {currentStatus.progress !== undefined && (
                <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                  <div
                    className="h-1 bg-consciousness-secondary rounded-full transition-all duration-300"
                    style={{ width: `${currentStatus.progress * 100}%` }}
                  />
                </div>
              )}
            </div>
          </div>
        );
        
      case 'human_typing':
        return (
          <div className={cn(baseClasses, "bg-consciousness-accent/5 border border-consciousness-accent/20")}>
            <AgentAvatar 
              agent={currentStatus.agentInfo!} 
              isActive={true}
              size="sm"
            />
            <span className="text-caption-text text-consciousness-accent font-medium">
              {currentStatus.message}
            </span>
            <TypingIndicator show={true} />
          </div>
        );
        
      case 'multi_agent':
        return (
          <div className={cn(baseClasses, "bg-purple-50 border border-purple-200")}>
            <div className="flex items-center space-x-1">
              <AgentAvatar 
                agent={{ name: 'Claude', type: 'ai' }} 
                isActive={multiAgentStatus.ai.active}
                size="sm"
              />
              <AgentAvatar 
                agent={{ name: multiAgentStatus.human.name || 'Agent', type: 'human' }} 
                isActive={multiAgentStatus.human.active}
                size="sm"
              />
            </div>
            <span className="text-caption-text text-purple-700 font-medium">
              {currentStatus.message}
            </span>
            {(multiAgentStatus.ai.thinking || multiAgentStatus.human.typing) && (
              <div className="flex items-center space-x-2">
                {multiAgentStatus.ai.thinking && <ThinkingIndicator show={true} />}
                {multiAgentStatus.human.typing && <TypingIndicator show={true} />}
              </div>
            )}
          </div>
        );
        
      case 'error':
        return (
          <div className={cn(baseClasses, "bg-red-50 border border-red-200")}>
            <span className="text-xl">⚠️</span>
            <span className="text-caption-text text-red-700 font-medium">
              {currentStatus.message}
            </span>
          </div>
        );
        
      case 'idle':
      default:
        return (
          <div className={cn(baseClasses, "bg-gray-50 border border-gray-200 opacity-50")}>
            <div className="h-2 w-2 bg-states-flow rounded-full" />
            <span className="text-caption-text text-text-tertiary">
              {currentStatus.message}
            </span>
          </div>
        );
    }
  }, [currentStatus, multiAgentStatus, compact, timeRemaining]);

  if (currentStatus.type === 'idle' && compact) {
    return null;
  }

  return (
    <div className={cn("", className)}>
      {compact ? (
        renderStatusContent()
      ) : (
        <Card variant="consciousness" glassmorphism className="border-0 shadow-sm">
          <CardContent className="p-0">
            {renderStatusContent()}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

// Standalone mini status indicator for headers/toolbars
export interface MiniStatusIndicatorProps {
  conversationId: string;
  className?: string;
}

export const MiniStatusIndicator: React.FC<MiniStatusIndicatorProps> = ({
  conversationId,
  className
}) => {
  const [status, setStatus] = useState<'idle' | 'ai_active' | 'human_active' | 'multi_active'>('idle');
  
  const ws = useWebSocket();
  
  useEffect(() => {
    const unsubscribe = ws.subscribe('status_update', (data: any) => {
      if (data.conversation_id !== conversationId) return;
      
      const multiAgent = data.multi_agent_status;
      if (multiAgent?.ai?.active && multiAgent?.human?.active) {
        setStatus('multi_active');
      } else if (multiAgent?.ai?.active) {
        setStatus('ai_active');
      } else if (multiAgent?.human?.active) {
        setStatus('human_active');
      } else {
        setStatus('idle');
      }
    });

    return unsubscribe;
  }, [ws, conversationId]);

  const statusConfig = useMemo(() => {
    switch (status) {
      case 'ai_active':
        return { color: 'bg-consciousness-primary', icon: '🤖', pulse: true };
      case 'human_active':
        return { color: 'bg-consciousness-accent', icon: '👤', pulse: true };
      case 'multi_active':
        return { color: 'bg-purple-500', icon: '👥', pulse: true };
      default:
        return { color: 'bg-gray-300', icon: '●', pulse: false };
    }
  }, [status]);

  return (
    <div className={cn("flex items-center space-x-2", className)}>
      <div className="relative">
        <div
          className={cn(
            "h-2 w-2 rounded-full transition-colors",
            statusConfig.color,
            statusConfig.pulse && "animate-pulse"
          )}
        />
        {statusConfig.pulse && (
          <div className={cn(
            "absolute -inset-1 h-4 w-4 rounded-full opacity-20 animate-ping",
            statusConfig.color
          )} />
        )}
      </div>
      
      <span className="text-caption-text text-text-tertiary">
        {status === 'idle' ? 'Ready' : 
         status === 'ai_active' ? 'AI working' :
         status === 'human_active' ? 'Agent active' :
         'Multi-agent'}
      </span>
    </div>
  );
};

export default LiveStatusIndicators;