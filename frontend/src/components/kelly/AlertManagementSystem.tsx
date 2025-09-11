/**
 * Alert Management System for Kelly
 * 3-level alert hierarchy with sound notifications, smart grouping, and escalation workflow
 * Apple-inspired design with accessibility features
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button, BreakthroughButton } from '@/components/ui/Button';
import { useWebSocket, useKellySafetyAlerts } from '@/lib/websocket';
import { SafetyAlert } from '@/types/kelly';
import { cn } from '@/lib/utils';

interface Alert extends SafetyAlert {
  acknowledged: boolean;
  acknowledgedAt?: Date;
  acknowledgedBy?: string;
  resolved: boolean;
  resolvedAt?: Date;
  resolvedBy?: string;
  escalated: boolean;
  escalatedAt?: Date;
  escalatedTo?: string;
  groupId?: string;
}

interface AlertGroup {
  id: string;
  type: string;
  count: number;
  highestSeverity: 'low' | 'medium' | 'high' | 'critical';
  alerts: Alert[];
  firstAlert: Date;
  lastAlert: Date;
}

interface AlertSoundConfig {
  critical: string;
  high: string;
  medium: string;
  enabled: boolean;
}

export interface AlertManagementSystemProps {
  className?: string;
  maxAlerts?: number;
  groupSimilarAlerts?: boolean;
  soundNotifications?: boolean;
  onAlertClick?: (alert: Alert) => void;
  onAlertResolved?: (alertId: string) => void;
  onEscalateAlert?: (alertId: string, escalateTo: string) => void;
}

export const AlertManagementSystem: React.FC<AlertManagementSystemProps> = ({
  className,
  maxAlerts = 100,
  groupSimilarAlerts = true,
  soundNotifications = true,
  onAlertClick,
  onAlertResolved,
  onEscalateAlert
}) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [soundConfig, setSoundConfig] = useState<AlertSoundConfig>({
    critical: '/sounds/critical-alert.mp3',
    high: '/sounds/high-alert.mp3',
    medium: '/sounds/medium-alert.mp3',
    enabled: soundNotifications
  });
  const [filter, setFilter] = useState<'all' | 'unacknowledged' | 'critical' | 'resolved'>('unacknowledged');
  const [selectedAlerts, setSelectedAlerts] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);

  const ws = useWebSocket();
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Handle new alerts
  useKellySafetyAlerts(useCallback((newAlert: SafetyAlert) => {
    const alert: Alert = {
      ...newAlert,
      acknowledged: false,
      resolved: false,
      escalated: false
    };

    setAlerts(prev => {
      const updated = [alert, ...prev.slice(0, maxAlerts - 1)];
      return updated;
    });

    // Play sound notification
    if (soundConfig.enabled && soundConfig[newAlert.severity as keyof AlertSoundConfig]) {
      playAlertSound(newAlert.severity);
    }

    // Show browser notification for critical alerts
    if (newAlert.severity === 'critical' && 'Notification' in window) {
      if (Notification.permission === 'granted') {
        new Notification('Critical Safety Alert', {
          body: newAlert.description,
          icon: '/favicon.ico',
          tag: newAlert.id
        });
      } else if (Notification.permission === 'default') {
        Notification.requestPermission();
      }
    }
  }, [soundConfig, maxAlerts]));

  // Play alert sound
  const playAlertSound = useCallback((severity: string) => {
    if (!soundConfig.enabled) return;

    const soundFile = soundConfig[severity as keyof AlertSoundConfig];
    if (!soundFile) return;

    if (audioRef.current) {
      audioRef.current.src = soundFile;
      audioRef.current.play().catch(() => {
        // Silently handle audio play failures (browser restrictions)
      });
    }
  }, [soundConfig]);

  // Group similar alerts
  const alertGroups = useMemo((): AlertGroup[] => {
    if (!groupSimilarAlerts) {
      return alerts.map(alert => ({
        id: alert.id,
        type: alert.alert_type,
        count: 1,
        highestSeverity: alert.severity,
        alerts: [alert],
        firstAlert: new Date(alert.timestamp),
        lastAlert: new Date(alert.timestamp)
      }));
    }

    const groups = new Map<string, AlertGroup>();
    
    alerts.forEach(alert => {
      const groupKey = `${alert.alert_type}-${alert.conversation_id || 'global'}`;
      
      if (!groups.has(groupKey)) {
        groups.set(groupKey, {
          id: groupKey,
          type: alert.alert_type,
          count: 0,
          highestSeverity: 'low',
          alerts: [],
          firstAlert: new Date(alert.timestamp),
          lastAlert: new Date(alert.timestamp)
        });
      }

      const group = groups.get(groupKey)!;
      group.count++;
      group.alerts.push(alert);
      
      // Update highest severity
      const severityOrder = ['low', 'medium', 'high', 'critical'];
      if (severityOrder.indexOf(alert.severity) > severityOrder.indexOf(group.highestSeverity)) {
        group.highestSeverity = alert.severity;
      }

      // Update timestamps
      const alertTime = new Date(alert.timestamp);
      if (alertTime < group.firstAlert) group.firstAlert = alertTime;
      if (alertTime > group.lastAlert) group.lastAlert = alertTime;
    });

    return Array.from(groups.values()).sort((a, b) => {
      // Sort by severity, then by latest alert
      const severityOrder = ['critical', 'high', 'medium', 'low'];
      const severityDiff = severityOrder.indexOf(a.highestSeverity) - severityOrder.indexOf(b.highestSeverity);
      if (severityDiff !== 0) return severityDiff;
      
      return b.lastAlert.getTime() - a.lastAlert.getTime();
    });
  }, [alerts, groupSimilarAlerts]);

  // Filter alerts
  const filteredGroups = useMemo(() => {
    switch (filter) {
      case 'unacknowledged':
        return alertGroups.filter(group => 
          group.alerts.some(alert => !alert.acknowledged)
        );
      case 'critical':
        return alertGroups.filter(group => 
          group.highestSeverity === 'critical'
        );
      case 'resolved':
        return alertGroups.filter(group =>
          group.alerts.every(alert => alert.resolved)
        );
      default:
        return alertGroups;
    }
  }, [alertGroups, filter]);

  // Acknowledge alert(s)
  const handleAcknowledge = useCallback(async (alertIds: string[]) => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/v1/kelly/alerts/acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_ids: alertIds })
      });

      if (!response.ok) throw new Error('Failed to acknowledge alerts');

      setAlerts(prev => prev.map(alert =>
        alertIds.includes(alert.id)
          ? { ...alert, acknowledged: true, acknowledgedAt: new Date(), acknowledgedBy: 'Current User' }
          : alert
      ));

      setSelectedAlerts(new Set());
    } catch (error) {
      console.error('Failed to acknowledge alerts:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Resolve alert(s)
  const handleResolve = useCallback(async (alertIds: string[]) => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/v1/kelly/alerts/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_ids: alertIds })
      });

      if (!response.ok) throw new Error('Failed to resolve alerts');

      setAlerts(prev => prev.map(alert =>
        alertIds.includes(alert.id)
          ? { 
              ...alert, 
              resolved: true, 
              resolvedAt: new Date(), 
              resolvedBy: 'Current User',
              acknowledged: true,
              acknowledgedAt: alert.acknowledgedAt || new Date()
            }
          : alert
      ));

      alertIds.forEach(id => onAlertResolved?.(id));
      setSelectedAlerts(new Set());
    } catch (error) {
      console.error('Failed to resolve alerts:', error);
    } finally {
      setLoading(false);
    }
  }, [onAlertResolved]);

  // Escalate alert
  const handleEscalate = useCallback(async (alertId: string, escalateTo: string = 'senior_agent') => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/v1/kelly/alerts/escalate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_id: alertId, escalate_to: escalateTo })
      });

      if (!response.ok) throw new Error('Failed to escalate alert');

      setAlerts(prev => prev.map(alert =>
        alert.id === alertId
          ? { ...alert, escalated: true, escalatedAt: new Date(), escalatedTo: escalateTo }
          : alert
      ));

      onEscalateAlert?.(alertId, escalateTo);
    } catch (error) {
      console.error('Failed to escalate alert:', error);
    } finally {
      setLoading(false);
    }
  }, [onEscalateAlert]);

  // Bulk actions
  const handleBulkAcknowledge = useCallback(() => {
    if (selectedAlerts.size > 0) {
      handleAcknowledge(Array.from(selectedAlerts));
    }
  }, [selectedAlerts, handleAcknowledge]);

  const handleBulkResolve = useCallback(() => {
    if (selectedAlerts.size > 0) {
      handleResolve(Array.from(selectedAlerts));
    }
  }, [selectedAlerts, handleResolve]);

  // Toggle alert selection
  const toggleAlertSelection = useCallback((alertId: string) => {
    setSelectedAlerts(prev => {
      const newSet = new Set(prev);
      if (newSet.has(alertId)) {
        newSet.delete(alertId);
      } else {
        newSet.add(alertId);
      }
      return newSet;
    });
  }, []);

  // Load initial alerts
  useEffect(() => {
    const loadAlerts = async () => {
      try {
        const response = await fetch('/api/v1/kelly/alerts');
        const data = await response.json();
        
        const formattedAlerts: Alert[] = data.alerts?.map((alert: any) => ({
          ...alert,
          acknowledged: alert.acknowledged || false,
          resolved: alert.resolved || false,
          escalated: alert.escalated || false,
          acknowledgedAt: alert.acknowledged_at ? new Date(alert.acknowledged_at) : undefined,
          resolvedAt: alert.resolved_at ? new Date(alert.resolved_at) : undefined,
          escalatedAt: alert.escalated_at ? new Date(alert.escalated_at) : undefined
        })) || [];

        setAlerts(formattedAlerts);
      } catch (error) {
        console.error('Failed to load alerts:', error);
      }
    };

    loadAlerts();
  }, []);

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  // Statistics
  const stats = useMemo(() => {
    const unacknowledged = alerts.filter(a => !a.acknowledged).length;
    const critical = alerts.filter(a => a.severity === 'critical').length;
    const resolved = alerts.filter(a => a.resolved).length;
    
    return { unacknowledged, critical, resolved, total: alerts.length };
  }, [alerts]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Hidden audio element for notifications */}
      <audio ref={audioRef} preload="auto" />

      {/* Header with statistics */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-insight-title text-consciousness-primary font-sf-pro">
            Alert Management
          </h1>
          <div className="flex items-center space-x-6 mt-2">
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 bg-states-stress rounded-full animate-pulse" />
              <span className="text-body-text text-text-secondary">
                {stats.unacknowledged} unacknowledged
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 bg-consciousness-accent rounded-full" />
              <span className="text-body-text text-text-secondary">
                {stats.critical} critical
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 bg-states-flow rounded-full" />
              <span className="text-body-text text-text-secondary">
                {stats.resolved} resolved
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSoundConfig(prev => ({ ...prev, enabled: !prev.enabled }))}
            className={cn(
              soundConfig.enabled ? "text-consciousness-primary" : "text-text-tertiary"
            )}
          >
            {soundConfig.enabled ? '🔊' : '🔇'}
          </Button>
        </div>
      </div>

      {/* Filters and bulk actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {(['all', 'unacknowledged', 'critical', 'resolved'] as const).map((filterOption) => (
            <Button
              key={filterOption}
              variant={filter === filterOption ? 'primary' : 'outline'}
              size="sm"
              onClick={() => setFilter(filterOption)}
            >
              {filterOption.charAt(0).toUpperCase() + filterOption.slice(1)}
            </Button>
          ))}
        </div>

        {selectedAlerts.size > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-caption-text text-text-tertiary">
              {selectedAlerts.size} selected
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleBulkAcknowledge}
              loading={loading}
            >
              Acknowledge
            </Button>
            <Button
              variant="success"
              size="sm"
              onClick={handleBulkResolve}
              loading={loading}
            >
              Resolve
            </Button>
          </div>
        )}
      </div>

      {/* Alerts list */}
      <div className="space-y-4">
        {filteredGroups.length === 0 ? (
          <Card variant="consciousness" glassmorphism>
            <CardContent className="text-center py-12">
              <span className="text-4xl mb-4 block">
                {filter === 'resolved' ? '✅' : '🛡️'}
              </span>
              <p className="text-insight-subtitle text-text-secondary">
                {filter === 'resolved' ? 'No resolved alerts' : 'No active alerts'}
              </p>
              <p className="text-body-text text-text-tertiary mt-2">
                {filter === 'resolved' 
                  ? 'Resolved alerts will appear here'
                  : 'All systems are operating normally'
                }
              </p>
            </CardContent>
          </Card>
        ) : (
          filteredGroups.map((group) => (
            <Card
              key={group.id}
              variant={group.highestSeverity === 'critical' ? 'breakthrough' : 'consciousness'}
              glassmorphism
              className={cn(
                "transition-all duration-200",
                group.highestSeverity === 'critical' && "animate-pulse-glow"
              )}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      className="h-4 w-4 text-consciousness-primary rounded border-gray-300"
                      checked={group.alerts.every(alert => selectedAlerts.has(alert.id))}
                      onChange={(e) => {
                        const alertIds = group.alerts.map(a => a.id);
                        if (e.target.checked) {
                          setSelectedAlerts(prev => new Set([...prev, ...alertIds]));
                        } else {
                          setSelectedAlerts(prev => {
                            const newSet = new Set(prev);
                            alertIds.forEach(id => newSet.delete(id));
                            return newSet;
                          });
                        }
                      }}
                    />
                    
                    <CardTitle className="flex items-center space-x-2">
                      <span className="text-xl">
                        {group.highestSeverity === 'critical' ? '🚨' :
                         group.highestSeverity === 'high' ? '⚠️' :
                         group.highestSeverity === 'medium' ? '⚡' : 'ℹ️'}
                      </span>
                      <span>{group.type.replace('_', ' ')}</span>
                      {group.count > 1 && (
                        <span className="bg-consciousness-primary/10 text-consciousness-primary px-2 py-1 rounded-full text-caption-text">
                          {group.count}
                        </span>
                      )}
                    </CardTitle>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className={cn(
                      "px-2 py-1 rounded-full text-caption-text font-medium",
                      group.highestSeverity === 'critical' && "bg-states-stress text-white",
                      group.highestSeverity === 'high' && "bg-consciousness-accent text-white",
                      group.highestSeverity === 'medium' && "bg-yellow-500 text-white",
                      group.highestSeverity === 'low' && "bg-gray-400 text-white"
                    )}>
                      {group.highestSeverity}
                    </span>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                {/* Show latest alert or all alerts */}
                {group.alerts.slice(0, group.count > 3 ? 1 : group.count).map((alert) => (
                  <div
                    key={alert.id}
                    className={cn(
                      "p-4 rounded-lg border transition-colors cursor-pointer",
                      alert.resolved && "bg-green-50 border-green-200",
                      alert.acknowledged && !alert.resolved && "bg-blue-50 border-blue-200",
                      !alert.acknowledged && "bg-red-50 border-red-200",
                      "hover:shadow-sm"
                    )}
                    onClick={() => onAlertClick?.(alert)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-body-text font-medium text-consciousness-primary">
                          {alert.description}
                        </p>
                        
                        {alert.conversation_id && (
                          <p className="text-caption-text text-text-tertiary mt-1">
                            Conversation: {alert.conversation_id}
                          </p>
                        )}
                        
                        <div className="flex items-center space-x-4 mt-2 text-caption-text text-text-tertiary">
                          <span>
                            {new Date(alert.timestamp).toLocaleString()}
                          </span>
                          
                          {alert.acknowledged && (
                            <span className="text-states-flow">
                              ✓ Acknowledged {alert.acknowledgedAt?.toLocaleTimeString()}
                            </span>
                          )}
                          
                          {alert.resolved && (
                            <span className="text-states-flow">
                              ✓ Resolved {alert.resolvedAt?.toLocaleTimeString()}
                            </span>
                          )}
                          
                          {alert.escalated && (
                            <span className="text-consciousness-accent">
                              ↗ Escalated to {alert.escalatedTo}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2 ml-4">
                        {!alert.acknowledged && (
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleAcknowledge([alert.id]);
                            }}
                          >
                            Acknowledge
                          </Button>
                        )}
                        
                        {alert.acknowledged && !alert.resolved && (
                          <Button
                            variant="success"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleResolve([alert.id]);
                            }}
                          >
                            Resolve
                          </Button>
                        )}
                        
                        {!alert.escalated && alert.severity !== 'low' && (
                          <Button
                            variant="warning"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEscalate(alert.id);
                            }}
                          >
                            Escalate
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                
                {group.count > 3 && (
                  <div className="text-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        // Show all alerts in group
                        console.log('Show all alerts in group:', group.id);
                      }}
                    >
                      Show {group.count - 1} more alerts
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertManagementSystem;