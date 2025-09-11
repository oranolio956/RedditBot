/**
 * Intervention Controls Panel for Kelly
 * Prominent take control functionality with AI confidence monitoring
 * Quick action buttons, emergency overrides, and progress indicators
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button, BreakthroughButton, FlowStateButton } from '@/components/ui/Button';
import { useWebSocket, useClaudeResponseGeneration } from '@/lib/websocket';
import { KellyConversation, ConversationMessage } from '@/types/kelly';
import { cn } from '@/lib/utils';

interface InterventionState {
  status: 'ai_active' | 'monitoring' | 'human_control' | 'handoff_in_progress';
  aiConfidence: number;
  humanAgent?: string;
  startedAt?: Date;
  lastAction?: string;
}

interface QuickAction {
  id: string;
  label: string;
  icon: string;
  description: string;
  action: () => Promise<void>;
  variant: 'primary' | 'secondary' | 'warning' | 'success';
  requiresConfirmation?: boolean;
}

export interface InterventionControlsPanelProps {
  conversationId: string;
  conversation: KellyConversation;
  className?: string;
  onInterventionChange?: (state: InterventionState) => void;
  onActionComplete?: (action: string, success: boolean) => void;
}

export const InterventionControlsPanel: React.FC<InterventionControlsPanelProps> = ({
  conversationId,
  conversation,
  className,
  onInterventionChange,
  onActionComplete
}) => {
  const [interventionState, setInterventionState] = useState<InterventionState>({
    status: 'ai_active',
    aiConfidence: conversation.ai_confidence || 85,
  });
  const [loading, setLoading] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState<string | null>(null);
  const [claudeGenerating, setClaudeGenerating] = useState(false);
  const [handoffProgress, setHandoffProgress] = useState(0);

  const ws = useWebSocket();

  // Handle Claude response generation updates
  useClaudeResponseGeneration(conversationId, useCallback((data: any) => {
    setClaudeGenerating(data.payload.status === 'thinking' || data.payload.status === 'generating');
    
    if (data.payload.confidence_so_far !== undefined) {
      setInterventionState(prev => ({
        ...prev,
        aiConfidence: data.payload.confidence_so_far * 100
      }));
    }
  }, []));

  // Monitor conversation updates for confidence changes
  useEffect(() => {
    const unsubscribe = ws.subscribe('kelly_conversation_update', (data: any) => {
      if (data.conversation_id === conversationId && data.ai_confidence_change) {
        setInterventionState(prev => ({
          ...prev,
          aiConfidence: data.ai_confidence_change
        }));
      }
    });

    return unsubscribe;
  }, [ws, conversationId]);

  // Update parent when intervention state changes
  useEffect(() => {
    onInterventionChange?.(interventionState);
  }, [interventionState, onInterventionChange]);

  // Take control action
  const handleTakeControl = useCallback(async () => {
    try {
      setLoading(true);
      setHandoffProgress(25);

      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/take-control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: 'current_user', // Would be actual user ID
          reason: 'Manual intervention requested'
        })
      });

      if (!response.ok) throw new Error('Failed to take control');
      
      setHandoffProgress(75);
      
      setInterventionState(prev => ({
        ...prev,
        status: 'human_control',
        humanAgent: 'You',
        startedAt: new Date(),
        lastAction: 'Took control'
      }));

      setHandoffProgress(100);
      onActionComplete?.('take_control', true);
      
      // Simulate handoff completion
      setTimeout(() => {
        setHandoffProgress(0);
      }, 2000);
      
    } catch (error) {
      console.error('Failed to take control:', error);
      onActionComplete?.('take_control', false);
    } finally {
      setLoading(false);
    }
  }, [conversationId, onActionComplete]);

  // Return control to AI
  const handleReturnControl = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/return-control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) throw new Error('Failed to return control');
      
      setInterventionState(prev => ({
        ...prev,
        status: 'ai_active',
        humanAgent: undefined,
        startedAt: undefined,
        lastAction: 'Returned control to AI'
      }));

      onActionComplete?.('return_control', true);
    } catch (error) {
      console.error('Failed to return control:', error);
      onActionComplete?.('return_control', false);
    } finally {
      setLoading(false);
    }
  }, [conversationId, onActionComplete]);

  // Monitor mode toggle
  const handleToggleMonitoring = useCallback(async () => {
    try {
      setLoading(true);
      
      const newStatus = interventionState.status === 'monitoring' ? 'ai_active' : 'monitoring';
      
      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/monitoring`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: newStatus === 'monitoring' })
      });

      if (!response.ok) throw new Error('Failed to toggle monitoring');
      
      setInterventionState(prev => ({
        ...prev,
        status: newStatus,
        lastAction: newStatus === 'monitoring' ? 'Started monitoring' : 'Stopped monitoring'
      }));

      onActionComplete?.('toggle_monitoring', true);
    } catch (error) {
      console.error('Failed to toggle monitoring:', error);
      onActionComplete?.('toggle_monitoring', false);
    } finally {
      setLoading(false);
    }
  }, [conversationId, interventionState.status, onActionComplete]);

  // Suggest response action
  const handleSuggestResponse = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/suggest-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) throw new Error('Failed to generate suggestion');
      
      setInterventionState(prev => ({
        ...prev,
        lastAction: 'Generated response suggestion'
      }));

      onActionComplete?.('suggest_response', true);
    } catch (error) {
      console.error('Failed to suggest response:', error);
      onActionComplete?.('suggest_response', false);
    } finally {
      setLoading(false);
    }
  }, [conversationId, onActionComplete]);

  // Emergency override
  const handleEmergencyOverride = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/emergency-override`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason: 'Emergency intervention',
          immediate: true
        })
      });

      if (!response.ok) throw new Error('Failed to execute emergency override');
      
      setInterventionState(prev => ({
        ...prev,
        status: 'human_control',
        humanAgent: 'You (Emergency)',
        startedAt: new Date(),
        lastAction: 'Emergency override activated'
      }));

      onActionComplete?.('emergency_override', true);
    } catch (error) {
      console.error('Failed to execute emergency override:', error);
      onActionComplete?.('emergency_override', false);
    } finally {
      setLoading(false);
      setShowConfirmation(null);
    }
  }, [conversationId, onActionComplete]);

  // Quick actions configuration
  const quickActions = useMemo((): QuickAction[] => [
    {
      id: 'monitor',
      label: interventionState.status === 'monitoring' ? 'Stop Monitoring' : 'Monitor',
      icon: '👀',
      description: interventionState.status === 'monitoring' 
        ? 'Stop monitoring this conversation'
        : 'Monitor conversation without taking control',
      action: handleToggleMonitoring,
      variant: interventionState.status === 'monitoring' ? 'secondary' : 'primary'
    },
    {
      id: 'suggest',
      label: 'Suggest Response',
      icon: '💡',
      description: 'Generate a response suggestion for the AI',
      action: handleSuggestResponse,
      variant: 'secondary'
    },
    {
      id: 'escalate',
      label: 'Escalate',
      icon: '📈',
      description: 'Escalate conversation to senior agent',
      action: async () => {
        // Implementation would escalate to senior agent
        console.log('Escalating conversation');
      },
      variant: 'warning'
    }
  ], [interventionState.status, handleToggleMonitoring, handleSuggestResponse]);

  // AI confidence level indicator
  const confidenceLevel = useMemo(() => {
    if (interventionState.aiConfidence >= 90) return 'high';
    if (interventionState.aiConfidence >= 70) return 'medium';
    if (interventionState.aiConfidence >= 50) return 'low';
    return 'very_low';
  }, [interventionState.aiConfidence]);

  const confidenceColor = useMemo(() => {
    switch (confidenceLevel) {
      case 'high': return 'text-states-flow';
      case 'medium': return 'text-consciousness-primary';
      case 'low': return 'text-consciousness-accent';
      case 'very_low': return 'text-states-stress';
    }
  }, [confidenceLevel]);

  // Status indicator
  const statusConfig = useMemo(() => {
    switch (interventionState.status) {
      case 'ai_active':
        return {
          label: 'AI Active',
          color: 'bg-states-flow',
          description: 'AI is handling this conversation',
          icon: '🤖'
        };
      case 'monitoring':
        return {
          label: 'Monitoring',
          color: 'bg-consciousness-primary',
          description: 'Monitoring AI responses',
          icon: '👀'
        };
      case 'human_control':
        return {
          label: 'Human Control',
          color: 'bg-consciousness-accent',
          description: `Controlled by ${interventionState.humanAgent}`,
          icon: '👨‍💼'
        };
      case 'handoff_in_progress':
        return {
          label: 'Handoff in Progress',
          color: 'bg-consciousness-secondary',
          description: 'Transferring control...',
          icon: '🔄'
        };
    }
  }, [interventionState]);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Main control panel */}
      <Card variant="breakthrough" glassmorphism>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>{statusConfig.icon}</span>
            <span>Intervention Controls</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Status display */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-surface-secondary">
            <div className="flex items-center space-x-3">
              <div className={cn(
                "h-4 w-4 rounded-full",
                statusConfig.color,
                claudeGenerating && "animate-pulse"
              )} />
              <div>
                <p className="text-body-text font-medium text-consciousness-primary">
                  {statusConfig.label}
                </p>
                <p className="text-caption-text text-text-tertiary">
                  {statusConfig.description}
                </p>
              </div>
            </div>
            
            {interventionState.startedAt && (
              <div className="text-right">
                <p className="text-caption-text text-text-tertiary">
                  Started
                </p>
                <p className="text-caption-text font-medium">
                  {interventionState.startedAt.toLocaleTimeString()}
                </p>
              </div>
            )}
          </div>

          {/* AI Confidence monitoring */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-body-text font-medium">AI Confidence Level</span>
              <span className={cn("text-body-text font-bold", confidenceColor)}>
                {Math.round(interventionState.aiConfidence)}%
              </span>
            </div>
            
            <div className="relative">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={cn(
                    "h-3 rounded-full transition-all duration-300",
                    confidenceLevel === 'high' && "bg-states-flow",
                    confidenceLevel === 'medium' && "bg-consciousness-primary",
                    confidenceLevel === 'low' && "bg-consciousness-accent",
                    confidenceLevel === 'very_low' && "bg-states-stress"
                  )}
                  style={{ width: `${interventionState.aiConfidence}%` }}
                />
              </div>
              
              {/* Confidence threshold indicators */}
              <div className="absolute top-0 left-1/2 w-0.5 h-3 bg-gray-400 opacity-50" />
              <div className="absolute top-0 left-3/4 w-0.5 h-3 bg-gray-400 opacity-50" />
              <div className="absolute -bottom-5 left-1/2 transform -translate-x-1/2 text-caption-text text-text-tertiary">
                50%
              </div>
              <div className="absolute -bottom-5 left-3/4 transform -translate-x-1/2 text-caption-text text-text-tertiary">
                75%
              </div>
            </div>

            {claudeGenerating && (
              <div className="flex items-center space-x-2 text-consciousness-primary">
                <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                <span className="text-caption-text">Claude is thinking...</span>
              </div>
            )}
          </div>

          {/* Handoff progress */}
          {handoffProgress > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-body-text">Handoff Progress</span>
                <span className="text-caption-text">{handoffProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 bg-consciousness-primary rounded-full transition-all duration-500"
                  style={{ width: `${handoffProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Primary action button */}
          <div className="pt-4">
            {interventionState.status === 'human_control' ? (
              <FlowStateButton
                onClick={handleReturnControl}
                loading={loading}
                className="w-full"
                size="lg"
              >
                Return Control to AI
              </FlowStateButton>
            ) : (
              <BreakthroughButton
                onClick={handleTakeControl}
                loading={loading}
                className="w-full"
                size="lg"
              >
                Take Control
              </BreakthroughButton>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick actions */}
      <Card variant="consciousness" glassmorphism>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>⚡</span>
            <span>Quick Actions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {quickActions.map((action) => (
              <Button
                key={action.id}
                variant={action.variant}
                onClick={action.action}
                loading={loading}
                className="flex flex-col items-center space-y-2 h-auto py-4"
                title={action.description}
              >
                <span className="text-xl">{action.icon}</span>
                <span className="text-caption-text">{action.label}</span>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Emergency override */}
      <Card variant="warning" className="border-2 border-states-stress/30">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-states-stress">
            <span>🚨</span>
            <span>Emergency Override</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-caption-text text-text-secondary mb-4">
            Immediately stop AI and take control. Use only in emergencies.
          </p>
          
          {showConfirmation === 'emergency' ? (
            <div className="flex items-center space-x-3">
              <Button
                variant="destructive"
                onClick={handleEmergencyOverride}
                loading={loading}
              >
                Confirm Emergency Override
              </Button>
              <Button
                variant="ghost"
                onClick={() => setShowConfirmation(null)}
              >
                Cancel
              </Button>
            </div>
          ) : (
            <Button
              variant="destructive"
              onClick={() => setShowConfirmation('emergency')}
              disabled={loading}
            >
              Emergency Override
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Last action indicator */}
      {interventionState.lastAction && (
        <div className="text-center">
          <p className="text-caption-text text-text-tertiary">
            Last action: {interventionState.lastAction}
          </p>
        </div>
      )}
    </div>
  );
};

export default InterventionControlsPanel;