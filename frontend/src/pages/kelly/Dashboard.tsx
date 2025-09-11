/**
 * Kelly Dashboard - Main Overview
 * Comprehensive dashboard for Kelly Telegram system with real-time metrics
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Users,
  MessageCircle,
  Shield,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Brain,
  Zap,
  Heart,
  Eye,
  Settings,
  ChevronRight,
  Plus,
  RefreshCw,
  Bell,
  Star,
  BarChart3,
  Calendar,
  Filter,
  Search,
  AlertCircle,
  PlayCircle,
  PauseCircle,
  StopCircle
} from 'lucide-react';
import { useKellyStore } from '@/store';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { KellyErrorBoundary } from '@/components/ui/ErrorBoundary';
import { KellyDashboardOverview, ConversationStage } from '@/types/kelly';
import { formatDistanceToNow, format } from 'date-fns';

const KellyDashboard: React.FC = () => {
  const {
    overview,
    accounts,
    activeConversations,
    safetyStatus,
    isLoading,
    setOverview,
    setAccounts,
    setActiveConversations,
    setSafetyStatus,
    setLoading
  } = useKellyStore();

  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState('24h');
  const [selectedMetric, setSelectedMetric] = useState('all');

  useEffect(() => {
    loadDashboardData();
    
    // Set up real-time updates
    const interval = setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load all dashboard data in parallel
      const [overviewRes, accountsRes, conversationsRes, safetyRes] = await Promise.all([
        fetch('/api/v1/kelly/dashboard/overview'),
        fetch('/api/v1/kelly/accounts'),
        fetch('/api/v1/kelly/conversations/active'),
        fetch('/api/v1/kelly/safety/status')
      ]);
      
      const [overviewData, accountsData, conversationsData, safetyData] = await Promise.all([
        overviewRes.json(),
        accountsRes.json(),
        conversationsRes.json(),
        safetyRes.json()
      ]);
      
      setOverview(overviewData.overview);
      setAccounts(accountsData.accounts || []);
      setActiveConversations(conversationsData.conversations || []);
      setSafetyStatus(safetyData.safety_status);
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const getSystemHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStageColor = (stage: ConversationStage) => {
    switch (stage) {
      case 'initial_contact': return 'bg-blue-100 text-blue-800';
      case 'rapport_building': return 'bg-green-100 text-green-800';
      case 'qualification': return 'bg-yellow-100 text-yellow-800';
      case 'engagement': return 'bg-purple-100 text-purple-800';
      case 'advanced_engagement': return 'bg-indigo-100 text-indigo-800';
      case 'payment_discussion': return 'bg-emerald-100 text-emerald-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading && !overview) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading Kelly Dashboard..." />
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
              <h1 className="text-3xl font-bold text-gray-900">Kelly Dashboard</h1>
              <p className="mt-2 text-gray-600">
                AI-powered Telegram conversation management system
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
              
              <Button
                onClick={handleRefresh}
                disabled={refreshing}
                variant="outline"
                size="sm"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {/* System Health Alert */}
        {overview && overview.system_health !== 'healthy' && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 bg-red-50 border border-red-200 rounded-md p-4"
          >
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-red-500 mr-3" />
              <div>
                <h3 className="text-sm font-medium text-red-800">
                  System Health: {overview.system_health}
                </h3>
                <p className="text-sm text-red-700 mt-1">
                  Some accounts may require attention. Check the accounts tab for details.
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Active Accounts</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {overview?.active_accounts || 0}
                    <span className="text-sm font-normal text-gray-500 ml-1">
                      / {overview?.total_accounts || 0}
                    </span>
                  </p>
                </div>
                <div className="p-3 rounded-full bg-blue-100">
                  <Users className="h-6 w-6 text-blue-600" />
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <CheckCircle className="h-4 w-4 text-green-500 mr-1" />
                  <span className="text-green-600">
                    {overview?.connected_accounts || 0} connected
                  </span>
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
                  <p className="text-sm font-medium text-gray-600">Today's Conversations</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {overview?.total_conversations_today || 0}
                  </p>
                </div>
                <div className="p-3 rounded-full bg-green-100">
                  <MessageCircle className="h-6 w-6 text-green-600" />
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <TrendingUp className="h-4 w-4 text-blue-500 mr-1" />
                  <span className="text-blue-600">
                    {overview?.total_messages_today || 0} messages sent
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
                  <p className="text-sm font-medium text-gray-600">Engagement Score</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {Math.round((overview?.average_engagement_score || 0) * 100)}%
                  </p>
                </div>
                <div className="p-3 rounded-full bg-purple-100">
                  <Heart className="h-6 w-6 text-purple-600" />
                </div>
              </div>
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full"
                    style={{ width: `${(overview?.average_engagement_score || 0) * 100}%` }}
                  />
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
                  <p className="text-sm font-medium text-gray-600">Safety Score</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {Math.round((overview?.average_safety_score || 0) * 100)}%
                  </p>
                </div>
                <div className="p-3 rounded-full bg-emerald-100">
                  <Shield className="h-6 w-6 text-emerald-600" />
                </div>
              </div>
              <div className="mt-4">
                {overview?.safety_alerts_count > 0 ? (
                  <div className="flex items-center text-sm">
                    <AlertTriangle className="h-4 w-4 text-red-500 mr-1" />
                    <span className="text-red-600">
                      {overview.safety_alerts_count} alerts
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-1" />
                    <span className="text-green-600">All clear</span>
                  </div>
                )}
              </div>
            </Card>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Active Conversations */}
          <div className="lg:col-span-2">
            <Card>
              <div className="p-6 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-medium text-gray-900">Active Conversations</h3>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">
                      {activeConversations?.length || 0} active
                    </span>
                    <Button variant="outline" size="sm">
                      <Eye className="h-4 w-4 mr-2" />
                      View All
                    </Button>
                  </div>
                </div>
              </div>
              
              <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                {activeConversations?.slice(0, 6).map((conversation) => (
                  <motion.div
                    key={conversation.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-6 hover:bg-gray-50 cursor-pointer"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className="relative">
                          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                            <MessageCircle className="h-5 w-5 text-blue-600" />
                          </div>
                          <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full ${
                            conversation.status === 'active' ? 'bg-green-400' : 
                            conversation.status === 'paused' ? 'bg-yellow-400' : 'bg-gray-400'
                          }`} />
                        </div>
                        
                        <div>
                          <h4 className="text-sm font-medium text-gray-900">
                            {conversation.user_info.username || conversation.user_info.first_name || 'Unknown User'}
                          </h4>
                          <p className="text-sm text-gray-500">
                            Stage: {conversation.stage.replace('_', ' ')}
                          </p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            getStageColor(conversation.stage)
                          }`}>
                            {conversation.message_count} msgs
                          </span>
                        </div>
                        <p className="text-xs text-gray-500">
                          {formatDistanceToNow(new Date(conversation.last_activity), { addSuffix: true })}
                        </p>
                      </div>
                    </div>
                    
                    {/* Conversation metrics */}
                    <div className="mt-4 flex items-center space-x-4">
                      <div className="flex items-center space-x-1">
                        <span className="text-xs text-gray-500">Engagement:</span>
                        <div className="w-16 bg-gray-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-600 h-1.5 rounded-full"
                            style={{ width: `${conversation.engagement_score}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600">{conversation.engagement_score}%</span>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <span className="text-xs text-gray-500">Safety:</span>
                        <div className={`w-2 h-2 rounded-full ${
                          conversation.safety_score >= 80 ? 'bg-green-400' :
                          conversation.safety_score >= 60 ? 'bg-yellow-400' : 'bg-red-400'
                        }`} />
                        <span className="text-xs text-gray-600">{conversation.safety_score}%</span>
                      </div>
                      
                      {conversation.red_flags.length > 0 && (
                        <div className="flex items-center space-x-1">
                          <AlertTriangle className="h-3 w-3 text-red-500" />
                          <span className="text-xs text-red-600">
                            {conversation.red_flags.length} flag{conversation.red_flags.length !== 1 ? 's' : ''}
                          </span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </div>

          {/* Quick Stats & Actions */}
          <div className="space-y-6">
            {/* Stage Distribution */}
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Conversation Stages</h3>
                <div className="space-y-3">
                  {overview?.stage_distribution && Object.entries(overview.stage_distribution).map(([stage, count]) => (
                    <div key={stage} className="flex items-center justify-between">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        getStageColor(stage as ConversationStage)
                      }`}>
                        {stage.replace('_', ' ')}
                      </span>
                      <span className="text-sm text-gray-900">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Card>

            {/* System Status */}
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">System Status</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Health</span>
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      getSystemHealthColor(overview?.system_health || 'unknown')
                    }`}>
                      {overview?.system_health || 'Unknown'}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">AI Performance</span>
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      overview?.ai_performance === 'optimal' ? 'text-green-600 bg-green-100' :
                      overview?.ai_performance === 'good' ? 'text-blue-600 bg-blue-100' :
                      'text-yellow-600 bg-yellow-100'
                    }`}>
                      {overview?.ai_performance || 'Unknown'}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Requires Review</span>
                    <span className="text-sm text-gray-900">
                      {overview?.conversations_requiring_review || 0}
                    </span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Quick Actions */}
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h3>
                <div className="space-y-2">
                  <Button variant="outline" className="w-full justify-start">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Account
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Settings className="h-4 w-4 mr-2" />
                    AI Features
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Shield className="h-4 w-4 mr-2" />
                    Safety Settings
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <BarChart3 className="h-4 w-4 mr-2" />
                    View Analytics
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </KellyErrorBoundary>
  );
};

export default KellyDashboard;