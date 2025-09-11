/**
 * Emergency Override Panel for Kelly
 * Large accessible emergency stop functionality with multi-level override system
 * Double-tap confirmation, audit trail, and safety-first design
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useWebSocket } from '@/lib/websocket';
import { cn } from '@/lib/utils';

interface OverrideAction {
  id: string;
  type: 'soft' | 'hard' | 'emergency' | 'reset';
  timestamp: Date;
  reason: string;
  initiatedBy: string;
  conversationId?: string;
  accountId?: string;
  success: boolean;
  duration?: number; // in milliseconds
  metadata?: any;
}

interface OverrideLevel {
  level: 'soft' | 'hard' | 'emergency' | 'reset';
  label: string;
  description: string;
  icon: string;
  color: string;
  confirmationRequired: boolean;
  doubleConfirmation: boolean;
  adminOnly?: boolean;
  cooldownSeconds: number;
}

const OVERRIDE_LEVELS: Record<string, OverrideLevel> = {
  soft: {
    level: 'soft',
    label: 'Soft Override',
    description: 'Pause AI responses temporarily, allow graceful handoff',
    icon: '⏸️',
    color: 'bg-consciousness-primary',
    confirmationRequired: true,
    doubleConfirmation: false,
    cooldownSeconds: 5
  },
  hard: {
    level: 'hard',
    label: 'Hard Override',
    description: 'Immediately stop AI, prevent new responses',
    icon: '🛑',
    color: 'bg-consciousness-accent',
    confirmationRequired: true,
    doubleConfirmation: true,
    cooldownSeconds: 10
  },
  emergency: {
    level: 'emergency',
    label: 'Emergency Stop',
    description: 'Complete system shutdown, alert all supervisors',
    icon: '🚨',
    color: 'bg-states-stress',
    confirmationRequired: true,
    doubleConfirmation: true,
    adminOnly: true,
    cooldownSeconds: 30
  },
  reset: {
    level: 'reset',
    label: 'System Reset',
    description: 'Restore normal AI operations, clear overrides',
    icon: '🔄',
    color: 'bg-states-flow',
    confirmationRequired: true,
    doubleConfirmation: false,
    cooldownSeconds: 0
  }
};

interface ConfirmationModalProps {
  overrideLevel: OverrideLevel;
  show: boolean;
  onConfirm: (reason: string) => void;
  onCancel: () => void;
  requiresDoubleConfirmation: boolean;
}

const ConfirmationModal: React.FC<ConfirmationModalProps> = ({
  overrideLevel,
  show,
  onConfirm,
  onCancel,
  requiresDoubleConfirmation
}) => {
  const [reason, setReason] = useState('');
  const [confirmationStep, setConfirmationStep] = useState<1 | 2>(1);
  const [countdown, setCountdown] = useState(5);
  const [canProceed, setCanProceed] = useState(false);

  useEffect(() => {
    if (show && requiresDoubleConfirmation) {
      setCountdown(5);
      setCanProceed(false);
      
      const timer = setInterval(() => {
        setCountdown(prev => {
          if (prev <= 1) {
            setCanProceed(true);
            clearInterval(timer);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    } else {
      setCanProceed(true);
    }
  }, [show, requiresDoubleConfirmation]);

  const handleConfirm = useCallback(() => {
    if (requiresDoubleConfirmation && confirmationStep === 1) {
      setConfirmationStep(2);
      setCanProceed(false);
      setCountdown(3);
      
      const timer = setTimeout(() => {
        setCanProceed(true);
      }, 3000);
      
      return;
    }
    
    onConfirm(reason || `${overrideLevel.label} activated by user`);
  }, [confirmationStep, reason, requiresDoubleConfirmation, overrideLevel.label, onConfirm]);

  const handleReset = useCallback(() => {
    setConfirmationStep(1);
    setReason('');
    setCanProceed(!requiresDoubleConfirmation);
  }, [requiresDoubleConfirmation]);

  useEffect(() => {
    if (show) {
      handleReset();
    }
  }, [show, handleReset]);

  if (!show) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card variant="breakthrough" className="max-w-md w-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-states-stress">
            <span className="text-2xl">{overrideLevel.icon}</span>
            <span>
              {requiresDoubleConfirmation && confirmationStep === 2 
                ? 'Final Confirmation Required' 
                : `Confirm ${overrideLevel.label}`
              }
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-body-text text-red-800 font-medium">
              {overrideLevel.description}
            </p>
            {overrideLevel.level === 'emergency' && (
              <p className="text-caption-text text-red-600 mt-2">
                ⚠️ This will alert all supervisors and create an incident report
              </p>
            )}
          </div>

          <div className="space-y-2">
            <label className="block text-body-text font-medium text-text-primary">
              Reason for override:
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-consciousness-primary focus:border-consciousness-primary"
              rows={3}
              placeholder="Please provide a reason for this action..."
              required={overrideLevel.level === 'emergency'}
            />
          </div>

          {requiresDoubleConfirmation && (
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-caption-text text-yellow-800">
                {confirmationStep === 1 ? (
                  <>
                    Step {confirmationStep} of 2: Please review the action above carefully.
                  </>
                ) : (
                  <>
                    Step {confirmationStep} of 2: Are you absolutely certain you want to proceed?
                  </>
                )}
              </p>
              {countdown > 0 && (
                <p className="text-caption-text text-yellow-600 mt-1">
                  Please wait {countdown} seconds...
                </p>
              )}
            </div>
          )}

          <div className="flex items-center justify-end space-x-3 pt-4">
            <Button
              variant="ghost"
              onClick={onCancel}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirm}
              disabled={!canProceed || (overrideLevel.level === 'emergency' && !reason.trim())}
              className="min-w-[120px]"
            >
              {requiresDoubleConfirmation && confirmationStep === 1 ? (
                'Continue'
              ) : countdown > 0 ? (
                `Confirm (${countdown})`
              ) : (
                'Confirm Override'
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export interface EmergencyOverridePanelProps {
  conversationId?: string;
  accountId?: string;
  className?: string;
  onOverrideComplete?: (action: OverrideAction) => void;
  onOverrideError?: (error: string) => void;
  currentOverrideStatus?: 'none' | 'soft' | 'hard' | 'emergency';
}

export const EmergencyOverridePanel: React.FC<EmergencyOverridePanelProps> = ({
  conversationId,
  accountId,
  className,
  onOverrideComplete,
  onOverrideError,
  currentOverrideStatus = 'none'
}) => {
  const [activeOverride, setActiveOverride] = useState<OverrideAction | null>(null);
  const [showConfirmation, setShowConfirmation] = useState<OverrideLevel | null>(null);
  const [auditTrail, setAuditTrail] = useState<OverrideAction[]>([]);
  const [loading, setLoading] = useState(false);
  const [cooldowns, setCooldowns] = useState<Record<string, number>>({});

  const ws = useWebSocket();
  const emergencyButtonRef = useRef<HTMLButtonElement>(null);

  // Handle cooldowns
  useEffect(() => {
    const interval = setInterval(() => {
      setCooldowns(prev => {
        const updated = { ...prev };
        let changed = false;
        
        Object.keys(updated).forEach(level => {
          if (updated[level] > 0) {
            updated[level] -= 1;
            changed = true;
          }
        });
        
        return changed ? updated : prev;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Load audit trail
  useEffect(() => {
    const loadAuditTrail = async () => {
      try {
        const params = new URLSearchParams();
        if (conversationId) params.append('conversation_id', conversationId);
        if (accountId) params.append('account_id', accountId);
        
        const response = await fetch(`/api/v1/kelly/overrides/audit?${params}`);
        const data = await response.json();
        
        const actions: OverrideAction[] = data.actions?.map((action: any) => ({
          ...action,
          timestamp: new Date(action.timestamp)
        })) || [];
        
        setAuditTrail(actions);
        
        // Check for active override
        const active = actions.find(action => 
          action.type !== 'reset' && 
          action.success &&
          !actions.some(resetAction => 
            resetAction.type === 'reset' && 
            resetAction.timestamp > action.timestamp
          )
        );
        
        setActiveOverride(active || null);
      } catch (error) {
        console.error('Failed to load audit trail:', error);
      }
    };

    loadAuditTrail();
  }, [conversationId, accountId]);

  // Execute override
  const executeOverride = useCallback(async (level: OverrideLevel, reason: string) => {
    try {
      setLoading(true);
      setShowConfirmation(null);

      const startTime = Date.now();
      
      const response = await fetch('/api/v1/kelly/overrides/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: level.level,
          reason,
          conversation_id: conversationId,
          account_id: accountId,
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Override failed');
      }

      const result = await response.json();
      const duration = Date.now() - startTime;

      const action: OverrideAction = {
        id: result.action_id || Date.now().toString(),
        type: level.level,
        timestamp: new Date(),
        reason,
        initiatedBy: 'Current User',
        conversationId,
        accountId,
        success: true,
        duration,
        metadata: result.metadata
      };

      setActiveOverride(action);
      setAuditTrail(prev => [action, ...prev]);
      
      // Set cooldown
      setCooldowns(prev => ({
        ...prev,
        [level.level]: level.cooldownSeconds
      }));

      // Emit WebSocket event
      ws.emit('emergency_override', {
        action,
        conversation_id: conversationId,
        account_id: accountId
      });

      onOverrideComplete?.(action);

      // Auto-focus emergency button for accessibility
      if (level.level === 'emergency') {
        emergencyButtonRef.current?.focus();
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Override failed:', error);
      onOverrideError?.(errorMessage);

      const action: OverrideAction = {
        id: Date.now().toString(),
        type: level.level,
        timestamp: new Date(),
        reason,
        initiatedBy: 'Current User',
        conversationId,
        accountId,
        success: false,
        metadata: { error: errorMessage }
      };

      setAuditTrail(prev => [action, ...prev]);
    } finally {
      setLoading(false);
    }
  }, [conversationId, accountId, onOverrideComplete, onOverrideError, ws]);

  // Handle override button click
  const handleOverrideClick = useCallback((level: OverrideLevel) => {
    if (cooldowns[level.level] > 0) return;
    
    if (level.confirmationRequired) {
      setShowConfirmation(level);
    } else {
      executeOverride(level, `${level.label} activated`);
    }
  }, [cooldowns, executeOverride]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (event: KeyboardEvent) => {
      // Emergency stop: Ctrl+Shift+E
      if (event.ctrlKey && event.shiftKey && event.key === 'E') {
        event.preventDefault();
        handleOverrideClick(OVERRIDE_LEVELS.emergency);
      }
      
      // Hard stop: Ctrl+Alt+S
      if (event.ctrlKey && event.altKey && event.key === 'S') {
        event.preventDefault();
        handleOverrideClick(OVERRIDE_LEVELS.hard);
      }
    };

    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, [handleOverrideClick]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Current status */}
      {activeOverride && (
        <Card variant="warning" className="border-2 border-states-stress/50">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-states-stress">
              <span className="animate-pulse">🚨</span>
              <span>Override Active</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-body-text font-medium">
                {OVERRIDE_LEVELS[activeOverride.type].label} is currently active
              </p>
              <p className="text-caption-text text-text-secondary">
                Initiated: {activeOverride.timestamp.toLocaleString()}
              </p>
              <p className="text-caption-text text-text-secondary">
                Reason: {activeOverride.reason}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Override controls */}
      <Card variant="breakthrough" glassmorphism>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>🛡️</span>
            <span>Emergency Override Controls</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.values(OVERRIDE_LEVELS).map((level) => {
              const isActive = activeOverride?.type === level.level;
              const isCooldown = cooldowns[level.level] > 0;
              const isDisabled = loading || isCooldown || (level.adminOnly && false); // TODO: Add admin check

              return (
                <Button
                  key={level.level}
                  ref={level.level === 'emergency' ? emergencyButtonRef : undefined}
                  variant={level.level === 'reset' ? 'success' : 'destructive'}
                  size="lg"
                  onClick={() => handleOverrideClick(level)}
                  disabled={isDisabled}
                  loading={loading && showConfirmation?.level === level.level}
                  className={cn(
                    "flex flex-col items-center space-y-2 h-auto py-6 text-center",
                    level.level === 'emergency' && "text-xl animate-breathing",
                    isActive && "ring-2 ring-states-stress animate-pulse"
                  )}
                  title={`Keyboard: ${level.level === 'emergency' ? 'Ctrl+Shift+E' : level.level === 'hard' ? 'Ctrl+Alt+S' : ''}`}
                  aria-label={`${level.label}: ${level.description}`}
                >
                  <span className="text-2xl">{level.icon}</span>
                  <div>
                    <div className="font-medium">
                      {level.label}
                      {isActive && ' (Active)'}
                    </div>
                    <div className="text-caption-text opacity-80 mt-1">
                      {level.description}
                    </div>
                    {isCooldown && (
                      <div className="text-caption-text text-yellow-300 mt-1">
                        Cooldown: {cooldowns[level.level]}s
                      </div>
                    )}
                  </div>
                </Button>
              );
            })}
          </div>

          {/* Accessibility note */}
          <div className="text-caption-text text-text-tertiary text-center border-t pt-4">
            Keyboard shortcuts: Emergency Stop (Ctrl+Shift+E), Hard Stop (Ctrl+Alt+S)
          </div>
        </CardContent>
      </Card>

      {/* Audit trail */}
      <Card variant="consciousness" glassmorphism>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>📋</span>
            <span>Audit Trail</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {auditTrail.length === 0 ? (
            <p className="text-center text-text-tertiary py-4">
              No override actions recorded
            </p>
          ) : (
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {auditTrail.slice(0, 10).map((action) => (
                <div
                  key={action.id}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg border",
                    action.success ? "bg-gray-50 border-gray-200" : "bg-red-50 border-red-200"
                  )}
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span>{OVERRIDE_LEVELS[action.type].icon}</span>
                      <span className="text-body-text font-medium">
                        {OVERRIDE_LEVELS[action.type].label}
                      </span>
                      <span className={cn(
                        "px-2 py-0.5 rounded text-caption-text",
                        action.success ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                      )}>
                        {action.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                    <p className="text-caption-text text-text-secondary mt-1">
                      {action.reason}
                    </p>
                    <p className="text-caption-text text-text-tertiary mt-1">
                      {action.timestamp.toLocaleString()} • {action.initiatedBy}
                      {action.duration && ` • ${action.duration}ms`}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Confirmation modal */}
      <ConfirmationModal
        overrideLevel={showConfirmation!}
        show={showConfirmation !== null}
        onConfirm={(reason) => executeOverride(showConfirmation!, reason)}
        onCancel={() => setShowConfirmation(null)}
        requiresDoubleConfirmation={showConfirmation?.doubleConfirmation || false}
      />
    </div>
  );
};

export default EmergencyOverridePanel;