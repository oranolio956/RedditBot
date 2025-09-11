/**
 * Kelly Safety Dashboard Component
 * Real-time safety monitoring with Claude AI analysis
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  CheckCircleIcon,
  EyeIcon,
  HandRaisedIcon,
  DocumentTextIcon,
  ClockIcon,
  UserGroupIcon,
  ChartBarIcon,
  FlagIcon,
  BellIcon,
} from '@heroicons/react/24/outline';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { useKellyStore } from '@/store';
import { apiClient } from '@/lib/api';
import { useKellySafetyAlerts } from '@/lib/websocket';
import type { SafetyStatus, SafetyViolation, SafetyAlert } from '@/types/kelly';

interface SafetyDashboardProps {
  accountId?: string;
  className?: string;
}

export default function SafetyDashboard({ accountId, className = '' }: SafetyDashboardProps) {
  const { selectedAccount } = useKellyStore();
  const currentAccountId = accountId || selectedAccount?.id || '';

  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus | null>(null);
  const [activeAlerts, setActiveAlerts] = useState<SafetyAlert[]>([]);
  const [recentViolations, setRecentViolations] = useState<SafetyViolation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState<SafetyAlert | null>(null);
  const [reviewNotes, setReviewNotes] = useState('');

  // Real-time safety alerts
  useKellySafetyAlerts((alert) => {
    setActiveAlerts(prev => [alert.payload, ...prev]);
    
    // Play notification sound for critical alerts
    if (alert.payload.severity === 'critical') {
      new Audio('/sounds/alert.mp3').play().catch(() => {});
    }
  });

  // Load safety data
  useEffect(() => {
    if (!currentAccountId) return;

    const loadSafetyData = async () => {
      try {
        setIsLoading(true);
        const status = await apiClient.getKellySafetyStatus(currentAccountId);
        setSafetyStatus(status);
        setRecentViolations(status.recent_violations || []);
      } catch (error) {
        console.error('Failed to load safety data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadSafetyData();
  }, [currentAccountId]);

  const handleReviewAlert = async (
    alertId: string, 
    action: 'approve' | 'reject' | 'escalate',
    notes?: string
  ) => {
    try {
      await apiClient.reviewSafetyAlert(alertId, action, notes);
      
      // Remove from active alerts
      setActiveAlerts(prev => prev.filter(alert => alert.id !== alertId));
      setSelectedAlert(null);
      setReviewNotes('');
    } catch (error) {
      console.error('Failed to review alert:', error);
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'low': return CheckCircleIcon;
      case 'medium': return ExclamationTriangleIcon;
      case 'high': return ExclamationTriangleIcon;
      case 'critical': return XCircleIcon;
      default: return CheckCircleIcon;
    }
  };

  const getViolationTypeIcon = (type: string) => {
    switch (type) {
      case 'harassment': return HandRaisedIcon;
      case 'inappropriate_content': return EyeIcon;
      case 'payment_pressure': return DocumentTextIcon;
      case 'underage': return UserGroupIcon;
      default: return FlagIcon;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading safety data..." />
      </div>
    );
  }

  if (!safetyStatus) {
    return (
      <Card className="p-6">
        <div className="text-center text-text-secondary">
          <ShieldCheckIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Safety data not available</p>
        </div>
      </Card>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Safety Overview */}
      <Card variant="default" className="overflow-hidden">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-lg bg-green-100 flex items-center justify-center">
                <ShieldCheckIcon className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <CardTitle className="text-xl">Safety Overview</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  Current Risk Level: {safetyStatus.current_risk_level}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-4 py-2 rounded-full text-sm font-medium ${getRiskLevelColor(safetyStatus.current_risk_level)}`}>
                {safetyStatus.current_risk_level.toUpperCase()}
              </div>
              <div className="text-right">
                <div className="text-2xl font-light text-green-600">
                  {Math.round(safetyStatus.safety_score)}
                </div>
                <div className="text-caption-text text-text-tertiary">Safety Score</div>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-metric-value font-light text-green-600">
                {activeAlerts.length}
              </div>
              <div className="text-caption-text text-text-tertiary">Active Alerts</div>
            </div>
            <div className="text-center">
              <div className="text-metric-value font-light text-orange-600">
                {recentViolations.length}
              </div>
              <div className="text-caption-text text-text-tertiary">Recent Violations</div>
            </div>
            <div className="text-center">
              <div className="text-metric-value font-light text-red-600">
                {safetyStatus.blocked_users_count}
              </div>
              <div className="text-caption-text text-text-tertiary">Blocked Users</div>
            </div>
            <div className="text-center">
              <div className="text-metric-value font-light text-blue-600">
                {safetyStatus.escalated_conversations_count}
              </div>
              <div className="text-caption-text text-text-tertiary">Escalated</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Active Safety Alerts */}
      {activeAlerts.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <BellIcon className="w-6 h-6 text-red-600" />
                <CardTitle>Active Safety Alerts ({activeAlerts.length})</CardTitle>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setActiveAlerts([])}
              >
                Clear All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {activeAlerts.map((alert) => {
                const SeverityIcon = getSeverityIcon(alert.severity);
                return (
                  <motion.div
                    key={alert.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`p-4 rounded-lg border-l-4 ${
                      alert.severity === 'critical' 
                        ? 'border-red-500 bg-red-50'
                        : alert.severity === 'high'
                        ? 'border-orange-500 bg-orange-50'
                        : 'border-yellow-500 bg-yellow-50'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <SeverityIcon className={`w-5 h-5 mt-1 ${
                          alert.severity === 'critical' 
                            ? 'text-red-600'
                            : alert.severity === 'high'
                            ? 'text-orange-600'
                            : 'text-yellow-600'
                        }`} />
                        <div className="flex-1">
                          <h4 className="font-medium text-text-primary">
                            {alert.alert_type.replace('_', ' ').toUpperCase()}
                          </h4>
                          <p className="text-sm text-text-secondary mt-1">
                            {alert.description}
                          </p>
                          <div className="flex items-center space-x-4 mt-2 text-xs text-text-tertiary">
                            <span>Conversation: {alert.conversation_id.slice(0, 8)}...</span>
                            <span>Account: {alert.account_id.slice(0, 8)}...</span>
                            <span>{new Date().toLocaleTimeString()}</span>
                          </div>
                          {alert.requires_immediate_action && (
                            <div className="mt-2">
                              <span className="px-2 py-1 bg-red-600 text-white text-xs rounded-full">
                                IMMEDIATE ACTION REQUIRED
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedAlert(alert)}
                        >
                          Review
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleReviewAlert(alert.id, 'approve')}
                          className="text-green-600 hover:text-green-700"
                        >
                          <CheckCircleIcon className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleReviewAlert(alert.id, 'reject')}
                          className="text-red-600 hover:text-red-700"
                        >
                          <XCircleIcon className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Violations */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-3">
            <ExclamationTriangleIcon className="w-6 h-6 text-orange-600" />
            <CardTitle>Recent Safety Violations</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          {recentViolations.length === 0 ? (
            <div className="text-center py-8 text-text-secondary">
              <CheckCircleIcon className="w-12 h-12 mx-auto mb-4 text-green-500" />
              <p>No recent safety violations. Great job!</p>
            </div>
          ) : (
            <div className="space-y-4">
              {recentViolations.map((violation) => {
                const ViolationIcon = getViolationTypeIcon(violation.type);
                return (
                  <div
                    key={violation.id}
                    className="flex items-start space-x-4 p-4 bg-surface-secondary/30 rounded-lg"
                  >
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      violation.severity === 'critical' 
                        ? 'bg-red-100'
                        : violation.severity === 'high'
                        ? 'bg-orange-100'
                        : 'bg-yellow-100'
                    }`}>
                      <ViolationIcon className={`w-5 h-5 ${
                        violation.severity === 'critical' 
                          ? 'text-red-600'
                          : violation.severity === 'high'
                          ? 'text-orange-600'
                          : 'text-yellow-600'
                      }`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h4 className="font-medium text-text-primary">
                          {violation.type.replace('_', ' ').toUpperCase()}
                        </h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskLevelColor(violation.severity)}`}>
                          {violation.severity}
                        </span>
                      </div>
                      <p className="text-sm text-text-secondary mt-1">
                        {violation.description}
                      </p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-text-tertiary">
                        <span>Confidence: {Math.round(violation.confidence_score * 100)}%</span>
                        <span>Action: {violation.action_taken}</span>
                        <span>{new Date(violation.detected_at).toLocaleString()}</span>
                      </div>
                      {violation.message_content && (
                        <div className="mt-3 p-3 bg-surface-secondary rounded text-sm">
                          <p className="text-text-tertiary text-xs mb-1">Message content:</p>
                          <p className="text-text-secondary italic">"{violation.message_content}"</p>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alert Review Modal */}
      <AnimatePresence>
        {selectedAlert && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedAlert(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-surface-primary rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold">Review Safety Alert</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedAlert(null)}
                >
                  <XCircleIcon className="w-4 h-4" />
                </Button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-text-secondary">Alert Type</label>
                  <p className="text-text-primary">{selectedAlert.alert_type}</p>
                </div>

                <div>
                  <label className="text-sm font-medium text-text-secondary">Description</label>
                  <p className="text-text-primary">{selectedAlert.description}</p>
                </div>

                <div>
                  <label className="text-sm font-medium text-text-secondary">Severity</label>
                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getRiskLevelColor(selectedAlert.severity)}`}>
                    {selectedAlert.severity}
                  </span>
                </div>

                {selectedAlert.suggested_actions && selectedAlert.suggested_actions.length > 0 && (
                  <div>
                    <label className="text-sm font-medium text-text-secondary">Suggested Actions</label>
                    <ul className="list-disc list-inside text-text-primary">
                      {selectedAlert.suggested_actions.map((action, index) => (
                        <li key={index}>{action}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div>
                  <label className="text-sm font-medium text-text-secondary">Review Notes</label>
                  <textarea
                    value={reviewNotes}
                    onChange={(e) => setReviewNotes(e.target.value)}
                    className="w-full p-3 border border-surface-tertiary rounded-lg mt-1"
                    rows={3}
                    placeholder="Add your review notes..."
                  />
                </div>

                <div className="flex items-center space-x-3 pt-4 border-t">
                  <Button
                    variant="primary"
                    onClick={() => handleReviewAlert(selectedAlert.id, 'approve', reviewNotes)}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    Approve
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleReviewAlert(selectedAlert.id, 'reject', reviewNotes)}
                    className="text-red-600 hover:text-red-700 border-red-600"
                  >
                    Reject
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleReviewAlert(selectedAlert.id, 'escalate', reviewNotes)}
                    className="text-orange-600 hover:text-orange-700 border-orange-600"
                  >
                    Escalate
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}