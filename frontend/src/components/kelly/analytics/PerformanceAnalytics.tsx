/**
 * Performance Analytics - Team productivity and AI vs human performance comparison
 * Response time analytics, cost tracking, and conversation quality trends
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '../../ui/Card';
import { LoadingSpinner } from '../../ui/LoadingSpinner';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  PieChart, Pie, Cell
} from 'recharts';
import {
  ClockIcon,
  CpuChipIcon,
  CurrencyDollarIcon,
  UserIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  SparklesIcon,
  ChartBarIcon,
  BoltIcon,
  FireIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  PlayIcon,
  PauseIcon,
  Cog6ToothIcon,
  CalendarIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  EyeIcon,
  ChatBubbleLeftRightIcon,
  UserGroupIcon,
  AcademicCapIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { format, subDays, subHours, startOfDay, endOfDay, differenceInDays } from 'date-fns';
import {
  AIPerformanceMetrics,
  ModelPerformanceMetrics,
  ConversationMetrics,
  UserEngagementMetrics,
  RevenueMetrics
} from '../../../types/analytics';
import { ClaudeUsageMetrics, ConversationStage } from '../../../types/kelly';

interface PerformanceAnalyticsProps {
  className?: string;
  timeRange?: '24h' | '7d' | '30d' | '90d';
  comparisonMode?: 'ai_vs_human' | 'model_comparison' | 'team_performance' | 'cost_analysis';
  agentId?: string;
  showRealTime?: boolean;
}

interface TeamMember {
  id: string;
  name: string;
  role: 'ai_agent' | 'human_agent' | 'supervisor';
  conversations_handled: number;
  avg_response_time: number;
  satisfaction_score: number;
  conversion_rate: number;
  cost_per_conversation: number;
  availability_hours: number;
  quality_score: number;
  escalation_rate: number;
}

interface PerformanceComparison {
  metric: string;
  ai_value: number;
  human_value: number;
  advantage: 'ai' | 'human' | 'neutral';
  improvement_potential: number;
}

interface CostBreakdown {
  category: string;
  ai_cost: number;
  human_cost: number;
  total_cost: number;
  cost_per_outcome: number;
  roi: number;
}

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon: React.ElementType;
  color?: string;
  loading?: boolean;
  comparison?: {
    ai: number;
    human: number;
    advantage: 'ai' | 'human' | 'neutral';
  };
}> = ({ title, value, change, changeLabel, icon: Icon, color = 'blue', loading, comparison }) => {
  const changeColor = change && change > 0 ? 'text-green-600' : 'text-red-600';
  const ChangeIcon = change && change > 0 ? TrendingUpIcon : TrendingDownIcon;

  return (
    <Card className="p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className={`p-3 rounded-lg bg-${color}-100`}>
            <Icon className={`w-6 h-6 text-${color}-600`} />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            {loading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <p className="text-2xl font-bold text-gray-900">
                {typeof value === 'number' ? value.toLocaleString() : value}
              </p>
            )}
          </div>
        </div>
        
        <div className="text-right">
          {change !== undefined && !loading && (
            <div className={`flex items-center space-x-1 ${changeColor}`}>
              <ChangeIcon className="w-4 h-4" />
              <span className="text-sm font-medium">
                {Math.abs(change).toFixed(1)}%
              </span>
            </div>
          )}
          {changeLabel && (
            <span className="text-xs text-gray-500">{changeLabel}</span>
          )}
        </div>
      </div>
      
      {comparison && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2">
              <CpuChipIcon className="w-4 h-4 text-blue-500" />
              <span className="text-gray-600">AI: {comparison.ai.toLocaleString()}</span>
            </div>
            <div className="flex items-center space-x-2">
              <UserIcon className="w-4 h-4 text-green-500" />
              <span className="text-gray-600">Human: {comparison.human.toLocaleString()}</span>
            </div>
          </div>
          <div className="mt-2">
            <div className={`text-xs font-medium ${
              comparison.advantage === 'ai' ? 'text-blue-600' :
              comparison.advantage === 'human' ? 'text-green-600' :
              'text-gray-600'
            }`}>
              {comparison.advantage === 'ai' ? 'AI Advantage' :
               comparison.advantage === 'human' ? 'Human Advantage' :
               'Balanced Performance'}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

const TeamPerformanceTable: React.FC<{
  teamMembers: TeamMember[];
  sortBy: string;
  onSort: (field: string) => void;
  onMemberClick: (member: TeamMember) => void;
}> = ({ teamMembers, sortBy, onSort, onMemberClick }) => {
  const getSortIcon = (field: string) => {
    if (sortBy === field) {
      return <ArrowTrendingDownIcon className="w-4 h-4" />;
    }
    return null;
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'ai_agent': return 'bg-blue-100 text-blue-800';
      case 'human_agent': return 'bg-green-100 text-green-800';
      case 'supervisor': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Agent
            </th>
            <th 
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              onClick={() => onSort('conversations_handled')}
            >
              <div className="flex items-center space-x-1">
                <span>Conversations</span>
                {getSortIcon('conversations_handled')}
              </div>
            </th>
            <th 
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              onClick={() => onSort('avg_response_time')}
            >
              <div className="flex items-center space-x-1">
                <span>Avg Response Time</span>
                {getSortIcon('avg_response_time')}
              </div>
            </th>
            <th 
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              onClick={() => onSort('satisfaction_score')}
            >
              <div className="flex items-center space-x-1">
                <span>Satisfaction</span>
                {getSortIcon('satisfaction_score')}
              </div>
            </th>
            <th 
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              onClick={() => onSort('conversion_rate')}
            >
              <div className="flex items-center space-x-1">
                <span>Conversion Rate</span>
                {getSortIcon('conversion_rate')}
              </div>
            </th>
            <th 
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              onClick={() => onSort('cost_per_conversation')}
            >
              <div className="flex items-center space-x-1">
                <span>Cost/Conv</span>
                {getSortIcon('cost_per_conversation')}
              </div>
            </th>
            <th 
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              onClick={() => onSort('quality_score')}
            >
              <div className="flex items-center space-x-1">
                <span>Quality Score</span>
                {getSortIcon('quality_score')}
              </div>
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {teamMembers.map((member) => (
            <motion.tr
              key={member.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              whileHover={{ backgroundColor: '#F9FAFB' }}
              onClick={() => onMemberClick(member)}
              className="cursor-pointer"
            >
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex items-center">
                  <div className="flex-shrink-0 h-10 w-10">
                    <div className={`h-10 w-10 rounded-full flex items-center justify-center ${
                      member.role === 'ai_agent' ? 'bg-blue-100' :
                      member.role === 'human_agent' ? 'bg-green-100' :
                      'bg-purple-100'
                    }`}>
                      {member.role === 'ai_agent' ? (
                        <CpuChipIcon className="h-5 w-5 text-blue-600" />
                      ) : (
                        <UserIcon className="h-5 w-5 text-green-600" />
                      )}
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className="text-sm font-medium text-gray-900">{member.name}</div>
                    <div className="text-sm text-gray-500">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRoleColor(member.role)}`}>
                        {member.role.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                {member.conversations_handled.toLocaleString()}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                {member.avg_response_time.toFixed(1)}s
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                <div className="flex items-center">
                  <span>{member.satisfaction_score.toFixed(1)}/5</span>
                  <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(member.satisfaction_score / 5) * 100}%` }}
                    />
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                {(member.conversion_rate * 100).toFixed(1)}%
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${member.cost_per_conversation.toFixed(2)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                <div className="flex items-center">
                  <span>{member.quality_score}/100</span>
                  <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        member.quality_score >= 90 ? 'bg-green-500' :
                        member.quality_score >= 70 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${member.quality_score}%` }}
                    />
                  </div>
                </div>
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const PerformanceAnalytics: React.FC<PerformanceAnalyticsProps> = ({
  className = '',
  timeRange = '7d',
  comparisonMode = 'ai_vs_human',
  agentId,
  showRealTime = true
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRange);
  const [currentComparisonMode, setCurrentComparisonMode] = useState(comparisonMode);
  const [sortBy, setSortBy] = useState('conversations_handled');
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);

  // Mock performance data
  const [performanceMetrics, setPerformanceMetrics] = useState<AIPerformanceMetrics>({
    claude_usage: {
      total_tokens_used_today: 245789,
      total_cost_today: 47.23,
      requests_by_model: {
        opus: 45,
        sonnet: 289,
        haiku: 1567
      },
      average_response_time: 847,
      success_rate: 98.7,
      cost_trend: Array.from({ length: 7 }, (_, i) => ({
        date: format(subDays(new Date(), 6 - i), 'yyyy-MM-dd'),
        cost: 35 + Math.random() * 20
      })),
      token_usage_trend: Array.from({ length: 7 }, (_, i) => ({
        date: format(subDays(new Date(), 6 - i), 'yyyy-MM-dd'),
        tokens: 200000 + Math.random() * 100000
      })),
      model_performance: {
        opus: { avg_confidence: 96.2, avg_quality: 94.8 },
        sonnet: { avg_confidence: 89.4, avg_quality: 87.3 },
        haiku: { avg_confidence: 78.9, avg_quality: 76.5 }
      }
    },
    response_quality_avg: 87.6,
    ai_vs_human_performance: {
      ai_success_rate: 84.7,
      human_success_rate: 76.3,
      ai_response_time: 1.2,
      human_response_time: 45.8,
      ai_satisfaction: 4.5,
      human_satisfaction: 4.7
    },
    model_performance: {
      opus: {
        usage_count: 45,
        average_confidence: 96.2,
        average_quality: 94.8,
        success_rate: 97.8,
        cost_per_use: 1.05,
        response_time: 1240,
        preferred_stages: ['qualification', 'engagement']
      },
      sonnet: {
        usage_count: 289,
        average_confidence: 89.4,
        average_quality: 87.3,
        success_rate: 91.2,
        cost_per_use: 0.23,
        response_time: 890,
        preferred_stages: ['rapport_building', 'maintenance']
      },
      haiku: {
        usage_count: 1567,
        average_confidence: 78.9,
        average_quality: 76.5,
        success_rate: 85.4,
        cost_per_use: 0.03,
        response_time: 450,
        preferred_stages: ['initial_contact', 'closing']
      }
    },
    cost_efficiency: {
      cost_per_successful_conversation: 2.34,
      roi_on_ai_investment: 456.7,
      automation_rate: 89.3
    }
  });

  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([
    {
      id: 'ai_001',
      name: 'Claude Opus Agent',
      role: 'ai_agent',
      conversations_handled: 1247,
      avg_response_time: 1.2,
      satisfaction_score: 4.5,
      conversion_rate: 0.847,
      cost_per_conversation: 1.05,
      availability_hours: 24,
      quality_score: 94,
      escalation_rate: 0.03
    },
    {
      id: 'ai_002',
      name: 'Claude Sonnet Agent',
      role: 'ai_agent',
      conversations_handled: 2156,
      avg_response_time: 0.9,
      satisfaction_score: 4.3,
      conversion_rate: 0.782,
      cost_per_conversation: 0.23,
      availability_hours: 24,
      quality_score: 87,
      escalation_rate: 0.08
    },
    {
      id: 'ai_003',
      name: 'Claude Haiku Agent',
      role: 'ai_agent',
      conversations_handled: 4523,
      avg_response_time: 0.45,
      satisfaction_score: 4.1,
      conversion_rate: 0.689,
      cost_per_conversation: 0.03,
      availability_hours: 24,
      quality_score: 76,
      escalation_rate: 0.15
    },
    {
      id: 'human_001',
      name: 'Sarah Chen',
      role: 'human_agent',
      conversations_handled: 234,
      avg_response_time: 45.8,
      satisfaction_score: 4.7,
      conversion_rate: 0.763,
      cost_per_conversation: 12.50,
      availability_hours: 8,
      quality_score: 89,
      escalation_rate: 0.02
    },
    {
      id: 'human_002',
      name: 'Mike Rodriguez',
      role: 'human_agent',
      conversations_handled: 198,
      avg_response_time: 38.2,
      satisfaction_score: 4.6,
      conversion_rate: 0.801,
      cost_per_conversation: 13.25,
      availability_hours: 8,
      quality_score: 92,
      escalation_rate: 0.01
    }
  ]);

  const performanceComparisons: PerformanceComparison[] = useMemo(() => [
    {
      metric: 'Response Time',
      ai_value: 1.2,
      human_value: 45.8,
      advantage: 'ai',
      improvement_potential: 97.4
    },
    {
      metric: 'Availability',
      ai_value: 24,
      human_value: 8,
      advantage: 'ai',
      improvement_potential: 66.7
    },
    {
      metric: 'Cost per Conversation',
      ai_value: 0.44,
      human_value: 12.88,
      advantage: 'ai',
      improvement_potential: 96.6
    },
    {
      metric: 'Success Rate',
      ai_value: 84.7,
      human_value: 76.3,
      advantage: 'ai',
      improvement_potential: 11.0
    },
    {
      metric: 'Customer Satisfaction',
      ai_value: 4.5,
      human_value: 4.7,
      advantage: 'human',
      improvement_potential: 4.4
    },
    {
      metric: 'Quality Score',
      ai_value: 85.7,
      human_value: 90.5,
      advantage: 'human',
      improvement_potential: 5.6
    }
  ], []);

  const costBreakdown: CostBreakdown[] = useMemo(() => [
    {
      category: 'Conversation Handling',
      ai_cost: 3456.78,
      human_cost: 2567.45,
      total_cost: 6024.23,
      cost_per_outcome: 4.23,
      roi: 578.9
    },
    {
      category: 'Training & Development',
      ai_cost: 245.67,
      human_cost: 1234.56,
      total_cost: 1480.23,
      cost_per_outcome: 12.34,
      roi: 123.4
    },
    {
      category: 'Infrastructure',
      ai_cost: 567.89,
      human_cost: 234.56,
      total_cost: 802.45,
      cost_per_outcome: 2.78,
      roi: 234.5
    }
  ], []);

  const responseTimeTrend = useMemo(() => {
    return Array.from({ length: 24 }, (_, i) => {
      const hour = i;
      return {
        hour: format(new Date().setHours(hour, 0, 0, 0), 'HH:mm'),
        ai_response_time: 0.8 + Math.random() * 0.8,
        human_response_time: 35 + Math.random() * 20,
        ai_volume: Math.floor(Math.random() * 50) + 20,
        human_volume: Math.floor(Math.random() * 15) + 5
      };
    });
  }, []);

  const modelComparisonData = useMemo(() => {
    return [
      {
        model: 'Claude Opus',
        quality: 94.8,
        speed: 1240,
        cost: 1.05,
        usage: 45,
        satisfaction: 4.5
      },
      {
        model: 'Claude Sonnet',
        quality: 87.3,
        speed: 890,
        cost: 0.23,
        usage: 289,
        satisfaction: 4.3
      },
      {
        model: 'Claude Haiku',
        quality: 76.5,
        speed: 450,
        cost: 0.03,
        usage: 1567,
        satisfaction: 4.1
      }
    ];
  }, []);

  const timeRanges = [
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' },
    { label: '90D', value: '90d' }
  ];

  const comparisonModes = [
    { id: 'ai_vs_human', label: 'AI vs Human', icon: CpuChipIcon },
    { id: 'model_comparison', label: 'Model Comparison', icon: ChartBarIcon },
    { id: 'team_performance', label: 'Team Performance', icon: UserGroupIcon },
    { id: 'cost_analysis', label: 'Cost Analysis', icon: CurrencyDollarIcon }
  ];

  const handleSort = (field: string) => {
    setSortBy(field);
    setTeamMembers(prev => [...prev].sort((a, b) => {
      const aValue = a[field as keyof TeamMember] as number;
      const bValue = b[field as keyof TeamMember] as number;
      return bValue - aValue;
    }));
  };

  const handleMemberClick = (member: TeamMember) => {
    setSelectedMember(member);
  };

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Performance Analytics</h2>
            <p className="text-gray-600 mt-1">
              AI vs human performance, cost analysis, and team productivity insights
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Time Range Selector */}
            <div className="flex items-center space-x-2 bg-white rounded-lg p-1 border border-gray-200">
              {timeRanges.map((range) => (
                <button
                  key={range.value}
                  onClick={() => setSelectedTimeRange(range.value as any)}
                  className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                    selectedTimeRange === range.value
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  {range.label}
                </button>
              ))}
            </div>

            {/* Real-time indicator */}
            {showRealTime && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>Live Data</span>
              </div>
            )}
          </div>
        </div>

        {/* Comparison Mode Toggle */}
        <div className="flex items-center space-x-2 bg-gray-100 rounded-lg p-1">
          {comparisonModes.map((mode) => {
            const Icon = mode.icon;
            return (
              <button
                key={mode.id}
                onClick={() => setCurrentComparisonMode(mode.id as any)}
                className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  currentComparisonMode === mode.id
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{mode.label}</span>
              </button>
            );
          })}
        </div>

        {/* Key Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Average Response Time"
            value={`${performanceMetrics.ai_vs_human_performance.ai_response_time}s`}
            change={-23.5}
            changeLabel="vs human agents"
            icon={ClockIcon}
            color="blue"
            comparison={{
              ai: performanceMetrics.ai_vs_human_performance.ai_response_time,
              human: performanceMetrics.ai_vs_human_performance.human_response_time,
              advantage: 'ai'
            }}
          />
          <MetricCard
            title="Success Rate"
            value={`${performanceMetrics.ai_vs_human_performance.ai_success_rate}%`}
            change={8.4}
            changeLabel="vs human agents"
            icon={CheckCircleIcon}
            color="green"
            comparison={{
              ai: performanceMetrics.ai_vs_human_performance.ai_success_rate,
              human: performanceMetrics.ai_vs_human_performance.human_success_rate,
              advantage: 'ai'
            }}
          />
          <MetricCard
            title="Cost per Conversation"
            value={`$${performanceMetrics.cost_efficiency.cost_per_successful_conversation}`}
            change={-96.6}
            changeLabel="vs human agents"
            icon={CurrencyDollarIcon}
            color="purple"
          />
          <MetricCard
            title="AI Automation Rate"
            value={`${performanceMetrics.cost_efficiency.automation_rate}%`}
            change={15.3}
            changeLabel="this month"
            icon={BoltIcon}
            color="orange"
          />
        </div>

        {/* Main Analytics Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentComparisonMode}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {currentComparisonMode === 'ai_vs_human' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Response Time Comparison */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Response Time Comparison</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={responseTimeTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis dataKey="hour" stroke="#6B7280" fontSize={12} />
                      <YAxis yAxisId="time" stroke="#6B7280" fontSize={12} />
                      <YAxis yAxisId="volume" orientation="right" stroke="#6B7280" fontSize={12} />
                      <Tooltip />
                      <Legend />
                      <Area
                        yAxisId="time"
                        type="monotone"
                        dataKey="human_response_time"
                        fill="#F59E0B"
                        fillOpacity={0.3}
                        stroke="#F59E0B"
                        name="Human Response Time (s)"
                      />
                      <Line
                        yAxisId="time"
                        type="monotone"
                        dataKey="ai_response_time"
                        stroke="#3B82F6"
                        strokeWidth={3}
                        name="AI Response Time (s)"
                      />
                      <Bar
                        yAxisId="volume"
                        dataKey="ai_volume"
                        fill="#3B82F6"
                        fillOpacity={0.6}
                        name="AI Volume"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </Card>

                {/* Performance Comparison Radar */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Comparison</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={performanceComparisons.map(comp => ({
                      metric: comp.metric,
                      ai: comp.advantage === 'ai' ? 100 : (comp.ai_value / comp.human_value) * 100,
                      human: comp.advantage === 'human' ? 100 : (comp.human_value / comp.ai_value) * 100
                    }))}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="metric" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar
                        name="AI Performance"
                        dataKey="ai"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.3}
                      />
                      <Radar
                        name="Human Performance"
                        dataKey="human"
                        stroke="#10B981"
                        fill="#10B981"
                        fillOpacity={0.3}
                      />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </Card>

                {/* Detailed Comparison Table */}
                <Card className="p-6 lg:col-span-2">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Performance Metrics</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Metric
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            AI Performance
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Human Performance
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Advantage
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Improvement Potential
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {performanceComparisons.map((comparison, index) => (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {comparison.metric}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              <div className="flex items-center">
                                <CpuChipIcon className="w-4 h-4 text-blue-500 mr-2" />
                                {comparison.metric.includes('Time') ? `${comparison.ai_value}s` :
                                 comparison.metric.includes('Cost') ? `$${comparison.ai_value}` :
                                 comparison.metric.includes('Rate') || comparison.metric.includes('Satisfaction') ? `${comparison.ai_value}` :
                                 comparison.ai_value}
                                {comparison.metric === 'Availability' && 'h/day'}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              <div className="flex items-center">
                                <UserIcon className="w-4 h-4 text-green-500 mr-2" />
                                {comparison.metric.includes('Time') ? `${comparison.human_value}s` :
                                 comparison.metric.includes('Cost') ? `$${comparison.human_value}` :
                                 comparison.metric.includes('Rate') || comparison.metric.includes('Satisfaction') ? `${comparison.human_value}` :
                                 comparison.human_value}
                                {comparison.metric === 'Availability' && 'h/day'}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                comparison.advantage === 'ai' ? 'bg-blue-100 text-blue-800' :
                                comparison.advantage === 'human' ? 'bg-green-100 text-green-800' :
                                'bg-gray-100 text-gray-800'
                              }`}>
                                {comparison.advantage === 'ai' ? 'AI' : comparison.advantage === 'human' ? 'Human' : 'Neutral'}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              <div className="flex items-center">
                                <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                  <div 
                                    className="bg-blue-600 h-2 rounded-full"
                                    style={{ width: `${Math.min(comparison.improvement_potential, 100)}%` }}
                                  />
                                </div>
                                <span>{comparison.improvement_potential.toFixed(1)}%</span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {currentComparisonMode === 'model_comparison' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Model Performance Scatter Plot */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality vs Cost Analysis</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis 
                        dataKey="cost" 
                        type="number" 
                        domain={[0, 'dataMax']}
                        stroke="#6B7280" 
                        fontSize={12}
                        label={{ value: 'Cost per Use ($)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        dataKey="quality" 
                        type="number" 
                        domain={[70, 100]}
                        stroke="#6B7280" 
                        fontSize={12}
                        label={{ value: 'Quality Score', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip 
                        formatter={(value, name) => [value, name]}
                        labelFormatter={(value) => `Model: ${value}`}
                      />
                      <Scatter 
                        name="Claude Models" 
                        data={modelComparisonData}
                        fill="#3B82F6"
                      >
                        {modelComparisonData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={
                            entry.model === 'Claude Opus' ? '#3B82F6' :
                            entry.model === 'Claude Sonnet' ? '#10B981' :
                            '#F59E0B'
                          } />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </Card>

                {/* Model Usage Distribution */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Distribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={modelComparisonData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ model, usage }) => `${model.split(' ')[1]}: ${usage}`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="usage"
                      >
                        {modelComparisonData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={
                            entry.model === 'Claude Opus' ? '#3B82F6' :
                            entry.model === 'Claude Sonnet' ? '#10B981' :
                            '#F59E0B'
                          } />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Card>

                {/* Model Performance Comparison */}
                <Card className="p-6 lg:col-span-2">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Comparison</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={modelComparisonData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis dataKey="model" stroke="#6B7280" fontSize={12} />
                      <YAxis stroke="#6B7280" fontSize={12} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="quality" fill="#3B82F6" name="Quality Score" />
                      <Bar dataKey="satisfaction" fill="#10B981" name="Satisfaction (scaled 20x)" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            )}

            {currentComparisonMode === 'team_performance' && (
              <Card className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-gray-900">Team Performance Overview</h3>
                  <div className="flex items-center space-x-4 text-sm text-gray-600">
                    <span>Total conversations: {teamMembers.reduce((sum, m) => sum + m.conversations_handled, 0).toLocaleString()}</span>
                    <span>Avg satisfaction: {(teamMembers.reduce((sum, m) => sum + m.satisfaction_score, 0) / teamMembers.length).toFixed(1)}/5</span>
                  </div>
                </div>
                
                <TeamPerformanceTable
                  teamMembers={teamMembers}
                  sortBy={sortBy}
                  onSort={handleSort}
                  onMemberClick={handleMemberClick}
                />
              </Card>
            )}

            {currentComparisonMode === 'cost_analysis' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Cost Breakdown */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Breakdown Analysis</h3>
                  <div className="space-y-4">
                    {costBreakdown.map((item, index) => (
                      <div key={index} className="p-4 bg-gray-50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium text-gray-900">{item.category}</h4>
                          <span className="text-lg font-bold text-gray-900">
                            ${item.total_cost.toLocaleString()}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                          <div>
                            <span className="font-medium">AI Cost:</span> ${item.ai_cost.toLocaleString()}
                          </div>
                          <div>
                            <span className="font-medium">Human Cost:</span> ${item.human_cost.toLocaleString()}
                          </div>
                          <div>
                            <span className="font-medium">Cost/Outcome:</span> ${item.cost_per_outcome}
                          </div>
                          <div>
                            <span className="font-medium">ROI:</span> {item.roi.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* Cost Trends */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Trends</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={performanceMetrics.claude_usage.cost_trend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                      <YAxis stroke="#6B7280" fontSize={12} />
                      <Tooltip formatter={(value) => [`$${value}`, 'Daily Cost']} />
                      <Area
                        type="monotone"
                        dataKey="cost"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>

                {/* ROI Comparison */}
                <Card className="p-6 lg:col-span-2">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">ROI Comparison</h3>
                  <div className="grid grid-cols-3 gap-6 text-center">
                    <div>
                      <div className="text-3xl font-bold text-green-600">
                        {performanceMetrics.cost_efficiency.roi_on_ai_investment.toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">AI Investment ROI</div>
                    </div>
                    <div>
                      <div className="text-3xl font-bold text-blue-600">
                        ${performanceMetrics.cost_efficiency.cost_per_successful_conversation}
                      </div>
                      <div className="text-sm text-gray-600">Cost per Success</div>
                    </div>
                    <div>
                      <div className="text-3xl font-bold text-purple-600">
                        {performanceMetrics.cost_efficiency.automation_rate.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Automation Rate</div>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Member Detail Modal */}
        {selectedMember && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
            >
              <div className="flex items-center justify-between p-6 border-b border-gray-200">
                <h3 className="text-xl font-bold text-gray-900">{selectedMember.name}</h3>
                <button
                  onClick={() => setSelectedMember(null)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <XCircleIcon className="w-6 h-6" />
                </button>
              </div>
              
              <div className="p-6 space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <span className="block text-sm font-medium text-gray-700 mb-1">Role</span>
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      selectedMember.role === 'ai_agent' ? 'bg-blue-100 text-blue-800' :
                      selectedMember.role === 'human_agent' ? 'bg-green-100 text-green-800' :
                      'bg-purple-100 text-purple-800'
                    }`}>
                      {selectedMember.role.replace('_', ' ')}
                    </span>
                  </div>
                  <div>
                    <span className="block text-sm font-medium text-gray-700 mb-1">Availability</span>
                    <span className="text-gray-900">{selectedMember.availability_hours}h/day</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-6 text-center">
                  <div>
                    <div className="text-2xl font-bold text-blue-600">
                      {selectedMember.conversations_handled.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600">Conversations</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-600">
                      {selectedMember.satisfaction_score.toFixed(1)}/5
                    </div>
                    <div className="text-sm text-gray-600">Satisfaction</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-600">
                      {(selectedMember.conversion_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Conversion Rate</div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Quality Score</span>
                      <span className="text-sm text-gray-900">{selectedMember.quality_score}/100</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 ${
                          selectedMember.quality_score >= 90 ? 'bg-green-500' :
                          selectedMember.quality_score >= 70 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${selectedMember.quality_score}%` }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Escalation Rate</span>
                      <span className="text-sm text-gray-900">{(selectedMember.escalation_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-red-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${selectedMember.escalation_rate * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};