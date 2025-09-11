/**
 * Kelly Safety Dashboard
 * Comprehensive safety monitoring and threat management system
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield,
  AlertTriangle,
  UserX,
  Eye,
  Flag,
  Activity,
  TrendingUp,
  BarChart3,
  Clock,
  Users,
  MessageCircle,
  RefreshCw,
  Filter,
  Search,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Settings,
  Download,
  ExternalLink,
  Play,
  Pause,
  StopCircle,
  MoreVertical,
  Calendar,
  Target,
  Layers,
  Database,
  Cpu,
  Brain,
  Lock,
  Unlock,
  Ban,
  UserCheck,
  FileText,
  Bell,
  BellOff
} from 'lucide-react';
import { useKellyStore } from '@/store';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { KellyErrorBoundary } from '@/components/ui/ErrorBoundary';
import { SafetyStatus, SafetyViolation, RedFlag } from '@/types/kelly';
import { formatDistanceToNow, format } from 'date-fns';

const SafetyDashboard: React.FC = () => {
  const {
    safetyStatus,
    activeConversations,
    isLoading,
    setSafetyStatus,
    setLoading
  } = useKellyStore();

  const [timeRange, setTimeRange] = useState('24h');
  const [filterSeverity, setFilterSeverity] = useState<'all' | 'low' | 'medium' | 'high' | 'critical'>('all');
  const [filterType, setFilterType] = useState<'all' | 'spam' | 'harassment' | 'inappropriate' | 'payment' | 'boundary' | 'underage' | 'threats'>('all');
  const [selectedViolation, setSelectedViolation] = useState<SafetyViolation | null>(null);
  const [blockedUsers, setBlockedUsers] = useState<any[]>([]);
  const [safetyMetrics, setSafetyMetrics] = useState<any>(null);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [autoActionsEnabled, setAutoActionsEnabled] = useState(true);

  useEffect(() => {
    loadSafetyData();
    
    // Set up real-time updates
    const interval = setInterval(loadSafetyData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [timeRange]);

  const loadSafetyData = async () => {
    try {
      setLoading(true);
      
      const [statusRes, metricsRes, blockedRes] = await Promise.all([
        fetch('/api/v1/kelly/safety/status'),
        fetch(`/api/v1/kelly/safety/metrics?timeRange=${timeRange}`),
        fetch('/api/v1/kelly/safety/blocked-users')
      ]);
      
      const [statusData, metricsData, blockedData] = await Promise.all([
        statusRes.json(),
        metricsRes.json(),
        blockedRes.json()
      ]);
      
      setSafetyStatus(statusData.safety_status);
      setSafetyMetrics(metricsData.metrics);
      setBlockedUsers(blockedData.blocked_users || []);
      
    } catch (error) {
      console.error('Failed to load safety data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViolation = async (violationId: string, action: 'dismiss' | 'escalate' | 'block_user') => {
    try {
      const response = await fetch(`/api/v1/kelly/safety/violations/${violationId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });

      if (response.ok) {
        loadSafetyData();
        setSelectedViolation(null);
      }
    } catch (error) {
      console.error('Failed to handle violation:', error);
    }
  };

  const unblockUser = async (userId: string) => {
    try {
      const response = await fetch(`/api/v1/kelly/safety/users/${userId}/unblock`, {
        method: 'POST'
      });

      if (response.ok) {
        loadSafetyData();
      }
    } catch (error) {
      console.error('Failed to unblock user:', error);
    }
  };

  const exportSafetyReport = async () => {
    try {
      const response = await fetch(`/api/v1/kelly/safety/export?timeRange=${timeRange}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `safety-report-${timeRange}.csv`;
      a.click();
    } catch (error) {
      console.error('Failed to export safety report:', error);
    }
  };

  const filteredViolations = safetyStatus?.recent_violations?.filter(violation => {
    if (filterSeverity !== 'all' && violation.severity !== filterSeverity) return false;
    if (filterType !== 'all' && violation.type !== filterType) return false;
    return true;
  }) || [];

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
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'high': return <AlertCircle className="h-4 w-4 text-orange-500" />;
      case 'medium': return <Flag className="h-4 w-4 text-yellow-500" />;
      case 'low': return <Eye className="h-4 w-4 text-blue-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  if (isLoading && !safetyStatus) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading safety dashboard..." />
      </div>
    );
  }

  return (
    <KellyErrorBoundary>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Safety Dashboard</h1>
              <p className="mt-2 text-gray-600">
                Monitor threats, manage violations, and ensure conversation safety
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              
              <Button onClick={exportSafetyReport} variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export Report
              </Button>
              
              <Button onClick={loadSafetyData} variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {/* Safety Status Alert */}
        {safetyStatus && safetyStatus.current_risk_level !== 'low' && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mb-6 border rounded-md p-4 ${
              safetyStatus.current_risk_level === 'critical'
                ? 'bg-red-50 border-red-200'
                : safetyStatus.current_risk_level === 'high'
                ? 'bg-orange-50 border-orange-200'
                : 'bg-yellow-50 border-yellow-200'
            }`}
          >
            <div className="flex items-center">
              <AlertTriangle className={`h-5 w-5 mr-3 ${
                safetyStatus.current_risk_level === 'critical'
                  ? 'text-red-500'
                  : safetyStatus.current_risk_level === 'high'
                  ? 'text-orange-500'
                  : 'text-yellow-500'
              }`} />
              <div>
                <h3 className={`text-sm font-medium ${
                  safetyStatus.current_risk_level === 'critical'
                    ? 'text-red-800'
                    : safetyStatus.current_risk_level === 'high'
                    ? 'text-orange-800'
                    : 'text-yellow-800'
                }`}>
                  Elevated Risk Level: {safetyStatus.current_risk_level.toUpperCase()}
                </h3>
                <p className={`text-sm mt-1 ${
                  safetyStatus.current_risk_level === 'critical'
                    ? 'text-red-700'
                    : safetyStatus.current_risk_level === 'high'
                    ? 'text-orange-700'
                    : 'text-yellow-700'
                }`}>
                  {safetyStatus.escalated_conversations_count} conversations require immediate attention.
                  {safetyStatus.manual_review_required && ' Manual review has been enabled.'}
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Safety Score</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {Math.round((safetyStatus?.safety_score || 0) * 100)}%
                  </p>
                </div>
                <div className={`p-3 rounded-full ${
                  (safetyStatus?.safety_score || 0) >= 0.8 ? 'bg-green-100' :
                  (safetyStatus?.safety_score || 0) >= 0.6 ? 'bg-yellow-100' : 'bg-red-100'
                }`}>
                  <Shield className={`h-6 w-6 ${
                    (safetyStatus?.safety_score || 0) >= 0.8 ? 'text-green-600' :
                    (safetyStatus?.safety_score || 0) >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                  }`} />
                </div>
              </div>
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      (safetyStatus?.safety_score || 0) >= 0.8 ? 'bg-green-600' :
                      (safetyStatus?.safety_score || 0) >= 0.6 ? 'bg-yellow-600' : 'bg-red-600'
                    }`}
                    style={{ width: `${(safetyStatus?.safety_score || 0) * 100}%` }}
                  />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Risk Level</p>
                  <p className="text-2xl font-bold text-gray-900 capitalize">
                    {safetyStatus?.current_risk_level || 'Unknown'}
                  </p>
                </div>
                <div className={`p-3 rounded-full ${getRiskLevelColor(safetyStatus?.current_risk_level || 'low')}`}>
                  <AlertTriangle className="h-6 w-6" />
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <span className={`${getRiskLevelColor(safetyStatus?.current_risk_level || 'low')} px-2 py-1 rounded-full`}>
                    Last check: {formatDistanceToNow(new Date(safetyStatus?.last_safety_check || Date.now()), { addSuffix: true })}
                  </span>
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Blocked Users</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {safetyStatus?.blocked_users_count || 0}
                  </p>
                </div>
                <div className="p-3 rounded-full bg-red-100">
                  <UserX className="h-6 w-6 text-red-600" />
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                  <span className="text-green-600">
                    {blockedUsers.length} total blocked
                  </span>
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Escalated</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {safetyStatus?.escalated_conversations_count || 0}
                  </p>
                </div>
                <div className="p-3 rounded-full bg-orange-100">
                  <Flag className="h-6 w-6 text-orange-600" />
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <Eye className="h-4 w-4 text-blue-500 mr-1" />
                  <span className="text-blue-600">
                    Require review
                  </span>
                </div>
              </div>
            </Card>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Violations */}
          <div className="lg:col-span-2">
            <Card>
              <div className="p-6 border-b border-gray-200">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Recent Safety Violations</h3>
                  <div className="flex items-center space-x-2">
                    <select
                      value={filterSeverity}
                      onChange={(e) => setFilterSeverity(e.target.value as any)}
                      className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="all">All Severities</option>
                      <option value="critical">Critical</option>
                      <option value="high">High</option>
                      <option value="medium">Medium</option>
                      <option value="low">Low</option>
                    </select>
                    
                    <select
                      value={filterType}
                      onChange={(e) => setFilterType(e.target.value as any)}
                      className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="all">All Types</option>
                      <option value="spam">Spam</option>
                      <option value="harassment">Harassment</option>
                      <option value="inappropriate">Inappropriate</option>
                      <option value="payment">Payment Pressure</option>
                      <option value="boundary">Boundary Violation</option>
                      <option value="underage">Underage</option>
                      <option value="threats">Threats</option>
                    </select>
                  </div>
                </div>
              </div>
              
              <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                {filteredViolations.length > 0 ? (
                  filteredViolations.map((violation) => (
                    <motion.div
                      key={violation.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="p-6 hover:bg-gray-50 cursor-pointer"
                      onClick={() => setSelectedViolation(violation)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-3">
                          {getSeverityIcon(violation.severity)}
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-1">
                              <h4 className="text-sm font-medium text-gray-900">
                                {violation.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </h4>
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                                violation.severity === 'critical' ? 'bg-red-100 text-red-800' :
                                violation.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                                violation.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-blue-100 text-blue-800'
                              }`}>
                                {violation.severity}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600 mb-2">
                              {violation.description}
                            </p>
                            <div className="flex items-center space-x-4 text-xs text-gray-500">
                              <span>Confidence: {Math.round(violation.confidence_score * 100)}%</span>
                              <span>{formatDistanceToNow(new Date(violation.detected_at), { addSuffix: true })}</span>
                              {violation.action_taken && (
                                <span className="text-blue-600">Action: {violation.action_taken}</span>
                              )}
                            </div>
                          </div>
                        </div>
                        
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <MoreVertical className="h-4 w-4" />
                        </button>
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <div className="p-12 text-center">
                    <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Violations Found</h3>
                    <p className="text-gray-600">
                      No safety violations detected in the selected time range and filters.
                    </p>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Safety Controls & Settings */}
          <div className="space-y-6">
            {/* Safety Controls */}
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Safety Controls</h3>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">Real-time Alerts</h4>
                      <p className="text-sm text-gray-500">Get notified of safety violations</p>
                    </div>
                    <button
                      onClick={() => setAlertsEnabled(!alertsEnabled)}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                        alertsEnabled ? 'bg-blue-600' : 'bg-gray-200'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          alertsEnabled ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">Auto Actions</h4>
                      <p className="text-sm text-gray-500">Automatically handle violations</p>
                    </div>
                    <button
                      onClick={() => setAutoActionsEnabled(!autoActionsEnabled)}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                        autoActionsEnabled ? 'bg-blue-600' : 'bg-gray-200'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          autoActionsEnabled ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                  
                  <div className="pt-4 border-t border-gray-200">
                    <Button variant="outline" className="w-full mb-2">
                      <Settings className="h-4 w-4 mr-2" />
                      Configure Thresholds
                    </Button>
                    <Button variant="outline" className="w-full">
                      <FileText className="h-4 w-4 mr-2" />
                      Safety Policies
                    </Button>
                  </div>
                </div>
              </div>
            </Card>

            {/* Blocked Users */}
            <Card>
              <div className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Blocked Users</h3>
                  <span className="text-sm text-gray-500">
                    {blockedUsers.length} blocked
                  </span>
                </div>
                
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {blockedUsers.slice(0, 10).map((user, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">
                          {user.username || user.first_name || 'Unknown User'}
                        </h4>
                        <p className="text-xs text-gray-500">
                          Blocked {formatDistanceToNow(new Date(user.blocked_at), { addSuffix: true })}
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => unblockUser(user.user_id)}
                      >
                        <Unlock className="h-3 w-3 mr-1" />
                        Unblock
                      </Button>
                    </div>
                  ))}
                  
                  {blockedUsers.length === 0 && (
                    <div className="text-center py-4">
                      <UserCheck className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-500">No blocked users</p>
                    </div>
                  )}
                </div>
              </div>
            </Card>

            {/* Safety Metrics Summary */}
            {safetyMetrics && (
              <Card>
                <div className="p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Safety Metrics</h3>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Total Violations</span>
                      <span className="text-sm font-medium text-gray-900">
                        {safetyMetrics.total_violations || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Auto-resolved</span>
                      <span className="text-sm font-medium text-green-600">
                        {safetyMetrics.auto_resolved || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Manual Review</span>
                      <span className="text-sm font-medium text-yellow-600">
                        {safetyMetrics.manual_review || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">False Positives</span>
                      <span className="text-sm font-medium text-gray-600">
                        {safetyMetrics.false_positives || 0}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>

        {/* Violation Detail Modal */}
        <AnimatePresence>
          {selectedViolation && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
              onClick={() => setSelectedViolation(null)}
            >
              <motion.div
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
                className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-start space-x-3">
                      {getSeverityIcon(selectedViolation.severity)}
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                          {selectedViolation.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </h3>
                        <p className="text-sm text-gray-500">
                          Detected {format(new Date(selectedViolation.detected_at), 'PPp')}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedViolation(null)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <XCircle className="h-6 w-6" />
                    </button>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Description</h4>
                      <p className="text-sm text-gray-700">{selectedViolation.description}</p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 mb-1">Severity</h4>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                          selectedViolation.severity === 'critical' ? 'bg-red-100 text-red-800' :
                          selectedViolation.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                          selectedViolation.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {selectedViolation.severity}
                        </span>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 mb-1">Confidence</h4>
                        <span className="text-sm text-gray-700">
                          {Math.round(selectedViolation.confidence_score * 100)}%
                        </span>
                      </div>
                    </div>
                    
                    {selectedViolation.message_content && (
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 mb-2">Message Content</h4>
                        <div className="bg-gray-50 border border-gray-200 rounded-md p-3">
                          <p className="text-sm text-gray-700">
                            {selectedViolation.message_content}
                          </p>
                        </div>
                      </div>
                    )}
                    
                    {selectedViolation.action_taken && (
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 mb-1">Action Taken</h4>
                        <p className="text-sm text-gray-700">{selectedViolation.action_taken}</p>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex justify-end space-x-3 mt-6 pt-6 border-t border-gray-200">
                    <Button
                      variant="outline"
                      onClick={() => handleViolation(selectedViolation.id, 'dismiss')}
                    >
                      Dismiss
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => handleViolation(selectedViolation.id, 'escalate')}
                      className="text-yellow-600 hover:text-yellow-700"
                    >
                      Escalate
                    </Button>
                    <Button
                      onClick={() => handleViolation(selectedViolation.id, 'block_user')}
                      className="bg-red-600 hover:bg-red-700 text-white"
                    >
                      Block User
                    </Button>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </KellyErrorBoundary>
  );
};

export default SafetyDashboard;