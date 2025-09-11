/**
 * User Journey Mapping - Visual conversation touchpoint analysis
 * Cross-channel conversation flow tracking and user behavior patterns
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '../../ui/Card';
import { LoadingSpinner } from '../../ui/LoadingSpinner';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Sankey, Flow, Node, Link
} from 'recharts';
import {
  MapIcon,
  ChatBubbleLeftRightIcon,
  ClockIcon,
  UserIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  EyeIcon,
  ArrowRightIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  ChartBarIcon,
  SparklesIcon,
  LightBulbIcon,
  FireIcon,
  ShieldCheckIcon,
  BoltIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  ForwardIcon,
  BackwardIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow, differenceInMinutes, differenceInHours, subDays } from 'date-fns';
import {
  UserJourney,
  JourneyStage,
  JourneyTouchpoint,
  OptimizationOpportunity,
  FunnelAnalysis,
  FunnelStage,
  DropOffAnalysis,
  CohortAnalysis
} from '../../../types/analytics';

interface UserJourneyMappingProps {
  className?: string;
  journeyId?: string;
  viewMode?: 'timeline' | 'funnel' | 'heatmap' | 'flows';
  timeRange?: '24h' | '7d' | '30d' | '90d';
  segmentBy?: 'all' | 'source' | 'stage' | 'outcome';
  onJourneySelect?: (journey: UserJourney) => void;
}

interface JourneyTimelineProps {
  journey: UserJourney;
  onTouchpointClick: (touchpoint: JourneyTouchpoint) => void;
}

interface TouchpointNode {
  id: string;
  type: string;
  timestamp: string;
  value: number;
  satisfaction: number;
  outcome: string;
  channel: string;
  x: number;
  y: number;
}

interface FlowConnection {
  source: string;
  target: string;
  value: number;
  users: number;
  conversionRate: number;
}

const JourneyTimeline: React.FC<JourneyTimelineProps> = ({ journey, onTouchpointClick }) => {
  const timelineData = useMemo(() => {
    const startTime = new Date(journey.start_date).getTime();
    
    return journey.touchpoints.map((touchpoint, index) => {
      const touchpointTime = new Date(touchpoint.timestamp).getTime();
      const relativeTime = (touchpointTime - startTime) / (1000 * 60 * 60); // Hours from start
      
      return {
        ...touchpoint,
        relativeTime,
        index,
        color: getTouchpointColor(touchpoint.type, touchpoint.satisfaction_score || 0.5)
      };
    });
  }, [journey]);

  const getTouchpointColor = (type: string, satisfaction: number) => {
    const baseColors = {
      conversation: '#3B82F6',
      feature_interaction: '#10B981',
      content_consumption: '#F59E0B',
      support_interaction: '#EF4444'
    };
    
    const alpha = Math.max(0.4, satisfaction);
    return `${baseColors[type as keyof typeof baseColors] || '#6B7280'}${Math.round(alpha * 255).toString(16)}`;
  };

  const maxTime = timelineData.length > 0 ? Math.max(...timelineData.map(t => t.relativeTime)) : 0;

  return (
    <div className="relative">
      {/* Timeline Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Journey Timeline</h3>
          <p className="text-sm text-gray-600">
            {format(new Date(journey.start_date), 'MMM dd, yyyy')} - 
            {journey.end_date ? format(new Date(journey.end_date), 'MMM dd, yyyy') : 'Ongoing'}
          </p>
        </div>
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span className="text-gray-600">Conversation</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-gray-600">Feature Use</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span className="text-gray-600">Content</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span className="text-gray-600">Support</span>
          </div>
        </div>
      </div>

      {/* Timeline Visualization */}
      <div className="relative bg-gray-50 rounded-lg p-6 overflow-x-auto">
        <div className="relative" style={{ minWidth: Math.max(800, maxTime * 40) }}>
          {/* Time axis */}
          <div className="absolute top-0 left-0 right-0 h-8 border-b border-gray-300">
            {Array.from({ length: Math.ceil(maxTime / 6) + 1 }, (_, i) => i * 6).map(hour => (
              <div
                key={hour}
                className="absolute top-0 bottom-0 border-l border-gray-200"
                style={{ left: `${(hour / maxTime) * 100}%` }}
              >
                <span className="absolute -top-6 text-xs text-gray-500 transform -translate-x-1/2">
                  {hour === 0 ? 'Start' : `+${hour}h`}
                </span>
              </div>
            ))}
          </div>

          {/* Journey stages */}
          <div className="mt-12 space-y-4">
            {journey.journey_stages.map((stage, stageIndex) => (
              <div key={stage.stage_id} className="relative">
                <div className="flex items-center space-x-3 mb-2">
                  <div className={`w-4 h-4 rounded-full ${
                    stage.stage_completion ? 'bg-green-500' : 'bg-yellow-500'
                  }`}></div>
                  <h4 className="font-medium text-gray-900">{stage.stage_name}</h4>
                  <span className="text-sm text-gray-500">
                    {stage.duration ? `${Math.round(stage.duration / (1000 * 60 * 60))}h` : 'In progress'}
                  </span>
                </div>
                
                {/* Stage timeline bar */}
                <div className="relative h-8 bg-white rounded border border-gray-200">
                  {timelineData
                    .filter(touchpoint => {
                      const touchpointTime = new Date(touchpoint.timestamp);
                      const stageStart = new Date(stage.entry_time);
                      const stageEnd = stage.exit_time ? new Date(stage.exit_time) : new Date();
                      return touchpointTime >= stageStart && touchpointTime <= stageEnd;
                    })
                    .map((touchpoint, touchpointIndex) => (
                      <motion.div
                        key={touchpoint.touchpoint_id}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: touchpointIndex * 0.1 }}
                        className="absolute top-1 w-6 h-6 rounded-full cursor-pointer hover:scale-110 transition-transform"
                        style={{
                          left: `${(touchpoint.relativeTime / maxTime) * 100}%`,
                          backgroundColor: touchpoint.color,
                          transform: 'translateX(-50%)'
                        }}
                        onClick={() => onTouchpointClick(touchpoint)}
                        title={`${touchpoint.type} - ${touchpoint.description}`}
                      >
                        <div className="w-full h-full rounded-full border-2 border-white shadow-sm flex items-center justify-center">
                          {touchpoint.type === 'conversation' && <ChatBubbleLeftRightIcon className="w-3 h-3 text-white" />}
                          {touchpoint.type === 'feature_interaction' && <SparklesIcon className="w-3 h-3 text-white" />}
                          {touchpoint.type === 'content_consumption' && <EyeIcon className="w-3 h-3 text-white" />}
                          {touchpoint.type === 'support_interaction' && <LightBulbIcon className="w-3 h-3 text-white" />}
                        </div>
                      </motion.div>
                    ))}
                </div>
              </div>
            ))}
          </div>

          {/* Journey outcomes */}
          <div className="mt-8 pt-4 border-t border-gray-300">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <span className="text-sm font-medium text-gray-700">Journey Outcome:</span>
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  journey.status === 'completed' ? 'bg-green-100 text-green-800' :
                  journey.status === 'active' ? 'bg-blue-100 text-blue-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {journey.status}
                </span>
              </div>
              <div className="flex items-center space-x-6 text-sm text-gray-600">
                <span>Value: ${journey.journey_value.toLocaleString()}</span>
                <span>Satisfaction: {journey.journey_satisfaction.toFixed(1)}/5</span>
                <span>Efficiency: {journey.journey_efficiency_score.toFixed(0)}/100</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const FunnelVisualization: React.FC<{
  funnelData: FunnelAnalysis;
  onStageClick: (stage: FunnelStage) => void;
}> = ({ funnelData, onStageClick }) => {
  const chartData = funnelData.stages.map(stage => ({
    name: stage.stage_name,
    users: stage.users_entered,
    completed: stage.users_completed,
    conversionRate: stage.conversion_rate * 100,
    avgTime: stage.avg_time_to_complete
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Conversion Funnel</h3>
        <div className="text-sm text-gray-600">
          Overall conversion: {((funnelData.stages[funnelData.stages.length - 1]?.users_completed || 0) / (funnelData.stages[0]?.users_entered || 1) * 100).toFixed(1)}%
        </div>
      </div>

      {/* Funnel stages */}
      <div className="space-y-2">
        {funnelData.stages.map((stage, index) => {
          const width = (stage.users_entered / funnelData.stages[0].users_entered) * 100;
          const conversionRate = stage.conversion_rate * 100;
          
          return (
            <motion.div
              key={stage.stage_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => onStageClick(stage)}
              className="cursor-pointer hover:bg-gray-50 p-3 rounded-lg transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-gray-900">{stage.stage_name}</h4>
                <div className="flex items-center space-x-4 text-sm text-gray-600">
                  <span>{stage.users_entered.toLocaleString()} users</span>
                  <span>{conversionRate.toFixed(1)}% conversion</span>
                  <span>{Math.round(stage.avg_time_to_complete / 60)} min avg</span>
                </div>
              </div>
              
              <div className="relative">
                <div className="w-full bg-gray-200 rounded-full h-8">
                  <div 
                    className={`h-8 rounded-full flex items-center justify-end pr-4 text-white text-sm font-medium transition-all duration-500 ${
                      conversionRate >= 80 ? 'bg-green-500' :
                      conversionRate >= 60 ? 'bg-blue-500' :
                      conversionRate >= 40 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${width}%` }}
                  >
                    {stage.users_completed.toLocaleString()}
                  </div>
                </div>
                
                {/* Drop-off indicator */}
                {index < funnelData.stages.length - 1 && (
                  <div className="absolute right-0 top-full mt-1 text-xs text-red-600">
                    -{(stage.users_entered - stage.users_completed).toLocaleString()} dropped off
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Drop-off analysis */}
      <div className="mt-8">
        <h4 className="font-medium text-gray-900 mb-4">Drop-off Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {funnelData.drop_off_analysis.map((analysis, index) => (
            <Card key={index} className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-900">
                  {funnelData.stages.find(s => s.stage_id === analysis.stage_id)?.stage_name}
                </span>
                <span className="text-red-600 font-medium">
                  {(analysis.drop_off_rate * 100).toFixed(1)}% drop-off
                </span>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="text-gray-600">
                  Revenue Impact: ${analysis.revenue_impact.toLocaleString()}
                </div>
                <div className="text-gray-600">
                  Segments Affected: {analysis.user_segments_affected.join(', ')}
                </div>
                
                <div className="mt-3">
                  <span className="font-medium text-gray-700">Top Reasons:</span>
                  <ul className="mt-1 space-y-1">
                    {analysis.drop_off_reasons.slice(0, 3).map((reason, reasonIndex) => (
                      <li key={reasonIndex} className="text-sm text-gray-600 flex items-center space-x-2">
                        <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                        <span>{reason.reason} ({(reason.frequency * 100).toFixed(0)}%)</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

const FlowVisualization: React.FC<{
  flowData: FlowConnection[];
  nodes: TouchpointNode[];
}> = ({ flowData, nodes }) => {
  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">User Flow Analysis</h3>
      
      {/* Flow chart would be implemented here with a library like react-flow or D3 */}
      <div className="bg-gray-50 rounded-lg p-8 min-h-96 flex items-center justify-center">
        <div className="text-center">
          <MapIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">
            Interactive flow visualization would be implemented here using react-flow or D3.js
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Showing {flowData.length} connections between {nodes.length} touchpoints
          </p>
        </div>
      </div>
      
      {/* Flow statistics */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">{nodes.length}</div>
          <div className="text-sm text-gray-600">Touchpoints</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-green-600">{flowData.length}</div>
          <div className="text-sm text-gray-600">Connections</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">
            {flowData.length > 0 ? (flowData.reduce((sum, flow) => sum + flow.conversionRate, 0) / flowData.length).toFixed(1) : 0}%
          </div>
          <div className="text-sm text-gray-600">Avg Conversion</div>
        </Card>
      </div>
    </div>
  );
};

const OptimizationInsights: React.FC<{
  opportunities: OptimizationOpportunity[];
  onImplement: (opportunityId: string) => void;
}> = ({ opportunities, onImplement }) => {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Optimization Opportunities</h3>
      
      {opportunities.map((opportunity, index) => (
        <motion.div
          key={opportunity.opportunity_id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <Card className="p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h4 className="font-semibold text-gray-900">{opportunity.description}</h4>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    opportunity.type === 'friction_reduction' ? 'bg-red-100 text-red-800' :
                    opportunity.type === 'engagement_increase' ? 'bg-blue-100 text-blue-800' :
                    opportunity.type === 'conversion_improvement' ? 'bg-green-100 text-green-800' :
                    'bg-purple-100 text-purple-800'
                  }`}>
                    {opportunity.type.replace('_', ' ')}
                  </span>
                </div>
                
                <div className="grid grid-cols-4 gap-4 text-sm text-gray-600 mb-4">
                  <div>
                    <span className="font-medium">Current:</span> {opportunity.current_performance.toFixed(1)}%
                  </div>
                  <div>
                    <span className="font-medium">Potential:</span> +{opportunity.potential_improvement.toFixed(1)}%
                  </div>
                  <div>
                    <span className="font-medium">Effort:</span> {opportunity.implementation_effort}
                  </div>
                  <div>
                    <span className="font-medium">ROI:</span> {opportunity.expected_roi.toFixed(1)}x
                  </div>
                </div>
                
                <div className="space-y-2">
                  <span className="text-sm font-medium text-gray-700">Success Metrics:</span>
                  <div className="flex flex-wrap gap-2">
                    {opportunity.success_metrics.map((metric, metricIndex) => (
                      <span
                        key={metricIndex}
                        className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800"
                      >
                        {metric}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => onImplement(opportunity.opportunity_id)}
                  className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Implement
                </button>
              </div>
            </div>
          </Card>
        </motion.div>
      ))}
    </div>
  );
};

export const UserJourneyMapping: React.FC<UserJourneyMappingProps> = ({
  className = '',
  journeyId,
  viewMode = 'timeline',
  timeRange = '7d',
  segmentBy = 'all',
  onJourneySelect
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedJourney, setSelectedJourney] = useState<UserJourney | null>(null);
  const [selectedTouchpoint, setSelectedTouchpoint] = useState<JourneyTouchpoint | null>(null);
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);

  // Mock data
  const [userJourneys, setUserJourneys] = useState<UserJourney[]>([
    {
      journey_id: 'journey_001',
      contact_id: 'contact_001',
      start_date: subDays(new Date(), 5).toISOString(),
      end_date: subDays(new Date(), 1).toISOString(),
      status: 'completed',
      journey_stages: [
        {
          stage_id: 'awareness',
          stage_name: 'Awareness',
          entry_time: subDays(new Date(), 5).toISOString(),
          exit_time: subDays(new Date(), 4).toISOString(),
          duration: 24 * 60 * 60 * 1000,
          stage_completion: true,
          stage_satisfaction: 4.2,
          actions_taken: ['Initial conversation', 'Information gathering'],
          barriers_encountered: ['Price concerns'],
          assistance_provided: ['Personalized consultation']
        },
        {
          stage_id: 'consideration',
          stage_name: 'Consideration',
          entry_time: subDays(new Date(), 4).toISOString(),
          exit_time: subDays(new Date(), 2).toISOString(),
          duration: 2 * 24 * 60 * 60 * 1000,
          stage_completion: true,
          stage_satisfaction: 4.5,
          actions_taken: ['Feature exploration', 'Pricing comparison'],
          barriers_encountered: ['Complex features'],
          assistance_provided: ['Demo session', 'FAQ resources']
        },
        {
          stage_id: 'decision',
          stage_name: 'Decision',
          entry_time: subDays(new Date(), 2).toISOString(),
          exit_time: subDays(new Date(), 1).toISOString(),
          duration: 24 * 60 * 60 * 1000,
          stage_completion: true,
          stage_satisfaction: 4.8,
          actions_taken: ['Purchase decision', 'Payment process'],
          barriers_encountered: [],
          assistance_provided: ['Payment support']
        }
      ],
      touchpoints: [
        {
          touchpoint_id: 'tp_001',
          type: 'conversation',
          description: 'Initial contact about privacy concerns',
          timestamp: subDays(new Date(), 5).toISOString(),
          channel: 'telegram',
          engagement_quality: 0.8,
          satisfaction_score: 4.2,
          conversation_id: 'conv_001',
          outcome: 'Interested in learning more',
          value_created: 0.3
        },
        {
          touchpoint_id: 'tp_002',
          type: 'feature_interaction',
          description: 'Explored broker scan feature',
          timestamp: subDays(new Date(), 4).toISOString(),
          channel: 'web_app',
          engagement_quality: 0.9,
          satisfaction_score: 4.5,
          outcome: 'Impressed with capabilities',
          value_created: 0.5
        },
        {
          touchpoint_id: 'tp_003',
          type: 'conversation',
          description: 'Pricing discussion and customization',
          timestamp: subDays(new Date(), 3).toISOString(),
          channel: 'telegram',
          engagement_quality: 0.7,
          satisfaction_score: 4.0,
          conversation_id: 'conv_002',
          outcome: 'Price concerns addressed',
          value_created: 0.4
        },
        {
          touchpoint_id: 'tp_004',
          type: 'content_consumption',
          description: 'Read privacy protection guide',
          timestamp: subDays(new Date(), 2).toISOString(),
          channel: 'web_app',
          engagement_quality: 0.6,
          satisfaction_score: 4.3,
          outcome: 'Better understanding of process',
          value_created: 0.2
        },
        {
          touchpoint_id: 'tp_005',
          type: 'conversation',
          description: 'Final consultation and purchase',
          timestamp: subDays(new Date(), 1).toISOString(),
          channel: 'telegram',
          engagement_quality: 0.95,
          satisfaction_score: 4.8,
          conversation_id: 'conv_003',
          outcome: 'Purchase completed',
          value_created: 1.0
        }
      ],
      conversion_events: [
        {
          event_id: 'conv_001',
          event_type: 'purchase',
          event_value: 2500,
          timestamp: subDays(new Date(), 1).toISOString(),
          attribution_data: {
            first_touch: {
              timestamp: subDays(new Date(), 5).toISOString(),
              type: 'conversation',
              description: 'Initial contact',
              value: 0.3
            },
            last_touch: {
              timestamp: subDays(new Date(), 1).toISOString(),
              type: 'conversation',
              description: 'Purchase decision',
              value: 1.0
            },
            all_touchpoints: [],
            conversion_path: ['conversation', 'feature_interaction', 'conversation', 'content_consumption', 'conversation'],
            time_to_conversion: 4 * 24 * 60 * 60 * 1000,
            attribution_model: 'time_decay'
          }
        }
      ],
      journey_value: 2500,
      journey_satisfaction: 4.5,
      journey_efficiency_score: 87.3,
      optimization_opportunities: [
        {
          opportunity_id: 'opt_001',
          type: 'friction_reduction',
          description: 'Simplify pricing presentation in initial conversations',
          current_performance: 65,
          potential_improvement: 15,
          implementation_effort: 'low',
          expected_roi: 2.3,
          success_metrics: ['Reduced price objections', 'Faster decision making', 'Higher satisfaction scores']
        },
        {
          opportunity_id: 'opt_002',
          type: 'engagement_increase',
          description: 'Add interactive demo early in the journey',
          current_performance: 72,
          potential_improvement: 18,
          implementation_effort: 'medium',
          expected_roi: 3.1,
          success_metrics: ['Increased feature understanding', 'Higher engagement scores', 'Better conversion rates']
        }
      ]
    }
  ]);

  const [funnelAnalysis, setFunnelAnalysis] = useState<FunnelAnalysis>({
    funnel_id: 'funnel_001',
    funnel_name: 'Conversation to Purchase Funnel',
    stages: [
      {
        stage_id: 'initial_contact',
        stage_name: 'Initial Contact',
        stage_order: 1,
        users_entered: 1000,
        users_completed: 750,
        conversion_rate: 0.75,
        avg_time_to_complete: 30 * 60, // 30 minutes
        typical_actions: ['First message', 'Problem identification'],
        success_factors: ['Quick response', 'Empathetic approach'],
        common_barriers: ['Skepticism', 'Time constraints']
      },
      {
        stage_id: 'interest_development',
        stage_name: 'Interest Development',
        stage_order: 2,
        users_entered: 750,
        users_completed: 450,
        conversion_rate: 0.60,
        avg_time_to_complete: 2 * 60 * 60, // 2 hours
        typical_actions: ['Feature exploration', 'Questions about service'],
        success_factors: ['Clear explanations', 'Relevant examples'],
        common_barriers: ['Complex terminology', 'Information overload']
      },
      {
        stage_id: 'evaluation',
        stage_name: 'Evaluation',
        stage_order: 3,
        users_entered: 450,
        users_completed: 270,
        conversion_rate: 0.60,
        avg_time_to_complete: 24 * 60 * 60, // 24 hours
        typical_actions: ['Pricing review', 'Feature comparison'],
        success_factors: ['Transparent pricing', 'Value demonstration'],
        common_barriers: ['Price concerns', 'Decision paralysis']
      },
      {
        stage_id: 'purchase_decision',
        stage_name: 'Purchase Decision',
        stage_order: 4,
        users_entered: 270,
        users_completed: 189,
        conversion_rate: 0.70,
        avg_time_to_complete: 3 * 60 * 60, // 3 hours
        typical_actions: ['Final consultation', 'Payment process'],
        success_factors: ['Personal attention', 'Easy payment'],
        common_barriers: ['Payment complexity', 'Last-minute doubts']
      }
    ],
    conversion_rates: {
      'initial_contact': 0.75,
      'interest_development': 0.60,
      'evaluation': 0.60,
      'purchase_decision': 0.70
    },
    drop_off_analysis: [
      {
        stage_id: 'interest_development',
        drop_off_rate: 0.40,
        drop_off_reasons: [
          {
            reason: 'Information overload',
            frequency: 0.35,
            impact_score: 0.7,
            addressable: true,
            solution_strategies: ['Simplify initial explanations', 'Progressive disclosure']
          },
          {
            reason: 'Not seeing immediate value',
            frequency: 0.25,
            impact_score: 0.8,
            addressable: true,
            solution_strategies: ['Lead with benefits', 'Quick wins demonstration']
          }
        ],
        user_segments_affected: ['tech professionals', 'small business owners'],
        revenue_impact: 45000,
        recovery_strategies: ['Retargeting campaigns', 'Simplified onboarding']
      }
    ],
    cohort_analysis: {
      cohort_periods: [],
      retention_curves: [],
      cohort_performance_comparison: [],
      lifecycle_value_by_cohort: {}
    },
    segment_performance: [],
    optimization_insights: [],
    benchmark_comparison: {
      industry_benchmarks: {},
      competitor_benchmarks: {},
      our_performance: {},
      performance_gaps: []
    }
  });

  const handleTouchpointClick = (touchpoint: JourneyTouchpoint) => {
    setSelectedTouchpoint(touchpoint);
  };

  const handleStageClick = (stage: FunnelStage) => {
    console.log('Stage clicked:', stage);
  };

  const handleImplementOptimization = (opportunityId: string) => {
    console.log('Implementing optimization:', opportunityId);
  };

  const viewModes = [
    { id: 'timeline', label: 'Timeline', icon: ClockIcon },
    { id: 'funnel', label: 'Funnel', icon: FunnelIcon },
    { id: 'flows', label: 'User Flows', icon: MapIcon },
    { id: 'optimization', label: 'Optimization', icon: LightBulbIcon }
  ];

  // Generate flow data for visualization
  const flowData: FlowConnection[] = useMemo(() => {
    // Mock flow connections between touchpoints
    return [
      { source: 'conversation', target: 'feature_interaction', value: 85, users: 650, conversionRate: 78.5 },
      { source: 'feature_interaction', target: 'conversation', value: 75, users: 520, conversionRate: 82.1 },
      { source: 'conversation', target: 'content_consumption', value: 45, users: 340, conversionRate: 65.3 },
      { source: 'content_consumption', target: 'conversation', value: 60, users: 280, conversionRate: 71.2 }
    ];
  }, []);

  const touchpointNodes: TouchpointNode[] = useMemo(() => {
    return [
      { id: 'conversation', type: 'conversation', timestamp: '', value: 850, satisfaction: 4.2, outcome: 'engaged', channel: 'telegram', x: 100, y: 100 },
      { id: 'feature_interaction', type: 'feature_interaction', timestamp: '', value: 650, satisfaction: 4.5, outcome: 'impressed', channel: 'web_app', x: 300, y: 150 },
      { id: 'content_consumption', type: 'content_consumption', timestamp: '', value: 340, satisfaction: 4.1, outcome: 'informed', channel: 'web_app', x: 200, y: 250 },
      { id: 'support_interaction', type: 'support_interaction', timestamp: '', value: 120, satisfaction: 4.7, outcome: 'resolved', channel: 'telegram', x: 400, y: 200 }
    ];
  }, []);

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">User Journey Mapping</h2>
            <p className="text-gray-600 mt-1">
              Visualize conversation touchpoints and optimize user experiences
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* View Mode Toggle */}
            <div className="flex items-center space-x-2 bg-white rounded-lg p-1 border border-gray-200">
              {viewModes.map((mode) => {
                const Icon = mode.icon;
                return (
                  <button
                    key={mode.id}
                    onClick={() => setCurrentViewMode(mode.id as any)}
                    className={`flex items-center space-x-2 px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                      currentViewMode === mode.id
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{mode.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Time Range Selector */}
            <select
              value={timeRange}
              onChange={(e) => {/* Handle time range change */}}
              className="text-sm border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last 90 Days</option>
            </select>
          </div>
        </div>

        {/* Summary Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{userJourneys.length}</div>
            <div className="text-sm text-gray-600">Active Journeys</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {userJourneys.reduce((sum, j) => sum + j.touchpoints.length, 0)}
            </div>
            <div className="text-sm text-gray-600">Touchpoints</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">
              {(userJourneys.reduce((sum, j) => sum + j.journey_satisfaction, 0) / userJourneys.length).toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Avg Satisfaction</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-orange-600">
              {(userJourneys.reduce((sum, j) => sum + j.journey_efficiency_score, 0) / userJourneys.length).toFixed(0)}
            </div>
            <div className="text-sm text-gray-600">Efficiency Score</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-red-600">
              ${userJourneys.reduce((sum, j) => sum + j.journey_value, 0).toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Total Value</div>
          </Card>
        </div>

        {/* Main Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentViewMode}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {currentViewMode === 'timeline' && userJourneys[0] && (
              <Card className="p-6">
                <JourneyTimeline
                  journey={userJourneys[0]}
                  onTouchpointClick={handleTouchpointClick}
                />
              </Card>
            )}

            {currentViewMode === 'funnel' && (
              <Card className="p-6">
                <FunnelVisualization
                  funnelData={funnelAnalysis}
                  onStageClick={handleStageClick}
                />
              </Card>
            )}

            {currentViewMode === 'flows' && (
              <Card className="p-6">
                <FlowVisualization
                  flowData={flowData}
                  nodes={touchpointNodes}
                />
              </Card>
            )}

            {currentViewMode === 'optimization' && userJourneys[0] && (
              <Card className="p-6">
                <OptimizationInsights
                  opportunities={userJourneys[0].optimization_opportunities}
                  onImplement={handleImplementOptimization}
                />
              </Card>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Touchpoint Detail Modal */}
        {selectedTouchpoint && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
            >
              <div className="flex items-center justify-between p-6 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Touchpoint Details</h3>
                <button
                  onClick={() => setSelectedTouchpoint(null)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <XCircleIcon className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-6 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="block text-sm font-medium text-gray-700 mb-1">Type</span>
                    <span className="text-gray-900 capitalize">{selectedTouchpoint.type.replace('_', ' ')}</span>
                  </div>
                  <div>
                    <span className="block text-sm font-medium text-gray-700 mb-1">Channel</span>
                    <span className="text-gray-900 capitalize">{selectedTouchpoint.channel}</span>
                  </div>
                  <div>
                    <span className="block text-sm font-medium text-gray-700 mb-1">Satisfaction</span>
                    <span className="text-gray-900">{selectedTouchpoint.satisfaction_score?.toFixed(1)}/5</span>
                  </div>
                  <div>
                    <span className="block text-sm font-medium text-gray-700 mb-1">Engagement Quality</span>
                    <span className="text-gray-900">{((selectedTouchpoint.engagement_quality || 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
                
                <div>
                  <span className="block text-sm font-medium text-gray-700 mb-1">Description</span>
                  <p className="text-gray-900">{selectedTouchpoint.description}</p>
                </div>
                
                <div>
                  <span className="block text-sm font-medium text-gray-700 mb-1">Outcome</span>
                  <p className="text-gray-900">{selectedTouchpoint.outcome}</p>
                </div>
                
                <div>
                  <span className="block text-sm font-medium text-gray-700 mb-1">Timestamp</span>
                  <p className="text-gray-900">{format(new Date(selectedTouchpoint.timestamp), 'MMM dd, yyyy HH:mm')}</p>
                </div>
                
                <div>
                  <span className="block text-sm font-medium text-gray-700 mb-1">Value Created</span>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${(selectedTouchpoint.value_created || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-600 mt-1">
                    {((selectedTouchpoint.value_created || 0) * 100).toFixed(0)}% of journey value
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};