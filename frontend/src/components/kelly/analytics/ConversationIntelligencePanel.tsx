/**
 * Conversation Intelligence Panel - AI-powered conversation analysis and insights
 * Real-time sentiment analysis, success predictions, and coaching recommendations
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '../../ui/Card';
import { LoadingSpinner } from '../../ui/LoadingSpinner';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, 
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, Cell, PieChart, Pie
} from 'recharts';
import {
  SparklesIcon,
  ChartBarIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  EyeIcon,
  ChatBubbleLeftRightIcon,
  CpuChipIcon,
  ArrowTrendingUpIcon,
  FireIcon,
  ShieldCheckIcon,
  AcademicCapIcon,
  UserGroupIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow } from 'date-fns';
import {
  ConversationIntelligence,
  SentimentAnalysis,
  TopicAnalysis,
  SuccessPrediction,
  CoachingRecommendation,
  PatternAnalysis,
  ConversationPattern,
  Anomaly
} from '../../../types/analytics';
import { ConversationStage, KellyConversation } from '../../../types/kelly';

interface ConversationIntelligencePanelProps {
  conversationId?: string;
  conversations?: KellyConversation[];
  className?: string;
  mode?: 'single' | 'aggregate' | 'comparison';
  autoRefresh?: boolean;
}

interface IntelligenceInsight {
  type: 'success_factor' | 'risk_factor' | 'optimization' | 'prediction';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  confidence: number;
  impact: number;
  actionable: boolean;
  recommendations: string[];
}

const SentimentGauge: React.FC<{
  sentiment: number;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}> = ({ sentiment, size = 'md', label }) => {
  const radius = size === 'sm' ? 30 : size === 'md' ? 40 : 50;
  const strokeWidth = size === 'sm' ? 4 : size === 'md' ? 6 : 8;
  const normalizedSentiment = (sentiment + 1) / 2; // Convert from -1,1 to 0,1
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = `${normalizedSentiment * circumference} ${circumference}`;
  
  const getColor = (value: number) => {
    if (value > 0.6) return '#10B981'; // Green
    if (value > 0.4) return '#F59E0B'; // Yellow
    return '#EF4444'; // Red
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <svg width={radius * 2.5} height={radius * 2.5} className="transform -rotate-90">
          <circle
            cx={radius * 1.25}
            cy={radius * 1.25}
            r={radius}
            stroke="#E5E7EB"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <circle
            cx={radius * 1.25}
            cy={radius * 1.25}
            r={radius}
            stroke={getColor(normalizedSentiment)}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={strokeDasharray}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`font-bold ${size === 'sm' ? 'text-sm' : size === 'md' ? 'text-lg' : 'text-xl'}`}>
            {(sentiment * 100).toFixed(0)}
          </span>
        </div>
      </div>
      {label && <span className="text-sm text-gray-600 mt-2">{label}</span>}
    </div>
  );
};

const TopicBubble: React.FC<{
  topic: string;
  relevance: number;
  sentiment: number;
  engagement: number;
  onClick?: () => void;
}> = ({ topic, relevance, sentiment, engagement, onClick }) => {
  const size = Math.max(40, relevance * 80);
  const opacity = Math.max(0.3, engagement);
  
  return (
    <motion.div
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className="cursor-pointer relative"
      style={{
        width: size,
        height: size,
      }}
    >
      <div
        className={`w-full h-full rounded-full flex items-center justify-center border-2 ${
          sentiment > 0.5 ? 'bg-green-100 border-green-300' :
          sentiment > -0.2 ? 'bg-yellow-100 border-yellow-300' :
          'bg-red-100 border-red-300'
        }`}
        style={{ opacity }}
      >
        <span className="text-xs font-medium text-center px-1 break-words">
          {topic.length > 8 ? topic.substring(0, 8) + '...' : topic}
        </span>
      </div>
      <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-blue-500 rounded-full text-xs text-white flex items-center justify-center">
        {Math.round(relevance * 100)}
      </div>
    </motion.div>
  );
};

const PatternCard: React.FC<{
  pattern: ConversationPattern;
  onClick?: () => void;
}> = ({ pattern, onClick }) => {
  const successColor = pattern.success_rate > 80 ? 'green' : pattern.success_rate > 60 ? 'yellow' : 'red';
  
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className="cursor-pointer"
    >
      <Card className="p-4 border-l-4 border-l-blue-500 hover:bg-gray-50 transition-colors">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 capitalize">
              {pattern.pattern_type.replace('_', ' ')}
            </h4>
            <p className="text-sm text-gray-600 mt-1">{pattern.pattern_description}</p>
            <div className="flex items-center space-x-4 mt-3">
              <div className="flex items-center space-x-1">
                <div className={`w-2 h-2 rounded-full bg-${successColor}-500`}></div>
                <span className="text-xs text-gray-600">
                  {pattern.success_rate.toFixed(1)}% success
                </span>
              </div>
              <div className="text-xs text-gray-600">
                Used {pattern.frequency} times
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-lg font-bold text-gray-900">
              {pattern.avg_conversation_impact.toFixed(1)}
            </div>
            <div className="text-xs text-gray-600">Impact Score</div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

const RecommendationCard: React.FC<{
  recommendation: CoachingRecommendation;
  onImplement?: (id: string) => void;
  onDismiss?: (id: string) => void;
}> = ({ recommendation, onImplement, onDismiss }) => {
  const priorityColors = {
    low: 'bg-blue-100 text-blue-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800'
  };

  const typeIcons = {
    messaging: ChatBubbleLeftRightIcon,
    timing: ClockIcon,
    topic: LightBulbIcon,
    tone: SparklesIcon,
    strategy: ChartBarIcon
  };

  const Icon = typeIcons[recommendation.type];

  return (
    <Card className="p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start space-x-3">
        <div className="p-2 bg-blue-100 rounded-lg">
          <Icon className="w-5 h-5 text-blue-600" />
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <h4 className="font-medium text-gray-900">{recommendation.recommendation}</h4>
            <span className={`px-2 py-1 text-xs font-medium rounded-full ${priorityColors[recommendation.priority]}`}>
              {recommendation.priority}
            </span>
          </div>
          <p className="text-sm text-gray-600 mb-3">{recommendation.rationale}</p>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-xs text-gray-500">
              <span>Impact: {(recommendation.expected_impact * 100).toFixed(0)}%</span>
              <span>Confidence: {(recommendation.ai_confidence * 100).toFixed(0)}%</span>
              <span className="capitalize">Effort: {recommendation.implementation_difficulty}</span>
            </div>
            
            <div className="flex items-center space-x-2">
              {onDismiss && (
                <button
                  onClick={() => onDismiss(recommendation.id)}
                  className="px-3 py-1 text-xs font-medium text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition-colors"
                >
                  Dismiss
                </button>
              )}
              {onImplement && (
                <button
                  onClick={() => onImplement(recommendation.id)}
                  className="px-3 py-1 text-xs font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors"
                >
                  Implement
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export const ConversationIntelligencePanel: React.FC<ConversationIntelligencePanelProps> = ({
  conversationId,
  conversations = [],
  className = '',
  mode = 'aggregate',
  autoRefresh = true
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'sentiment' | 'topics' | 'patterns' | 'coaching' | 'predictions'>('overview');
  const [timeRange, setTimeRange] = useState('24h');

  // Mock conversation intelligence data
  const [intelligence, setIntelligence] = useState<ConversationIntelligence>({
    conversation_id: conversationId || 'aggregate',
    quality_score: 87.3,
    sentiment_analysis: {
      overall_sentiment: 'positive',
      sentiment_score: 0.67,
      sentiment_timeline: [
        { timestamp: '2024-01-15T10:00:00Z', value: 0.2 },
        { timestamp: '2024-01-15T11:00:00Z', value: 0.45 },
        { timestamp: '2024-01-15T12:00:00Z', value: 0.67 },
        { timestamp: '2024-01-15T13:00:00Z', value: 0.72 },
        { timestamp: '2024-01-15T14:00:00Z', value: 0.58 }
      ],
      emotional_journey: [
        {
          timestamp: '2024-01-15T10:00:00Z',
          emotion: 'curiosity',
          intensity: 0.6,
          context: 'Initial engagement about services'
        },
        {
          timestamp: '2024-01-15T11:30:00Z',
          emotion: 'interest',
          intensity: 0.8,
          context: 'Discussing specific needs'
        },
        {
          timestamp: '2024-01-15T13:00:00Z',
          emotion: 'confidence',
          intensity: 0.9,
          context: 'Understanding value proposition'
        }
      ],
      sentiment_drivers: {
        positive_factors: ['personalized responses', 'quick resolution', 'helpful suggestions'],
        negative_factors: ['initial confusion', 'price concern'],
        neutral_factors: ['technical questions', 'standard procedures']
      },
      mood_transitions: [
        {
          from_mood: 'neutral',
          to_mood: 'interested',
          timestamp: '2024-01-15T10:15:00Z',
          trigger: 'personalized greeting',
          impact_score: 0.4
        }
      ]
    },
    topic_analysis: {
      primary_topics: [
        {
          topic: 'pricing',
          relevance_score: 0.9,
          frequency: 12,
          sentiment_score: 0.3,
          engagement_impact: 0.7,
          conversion_correlation: 0.85,
          first_mentioned: '2024-01-15T10:30:00Z',
          last_mentioned: '2024-01-15T14:00:00Z'
        },
        {
          topic: 'features',
          relevance_score: 0.8,
          frequency: 8,
          sentiment_score: 0.8,
          engagement_impact: 0.9,
          conversion_correlation: 0.92,
          first_mentioned: '2024-01-15T10:45:00Z',
          last_mentioned: '2024-01-15T13:45:00Z'
        },
        {
          topic: 'integration',
          relevance_score: 0.6,
          frequency: 5,
          sentiment_score: 0.5,
          engagement_impact: 0.6,
          conversion_correlation: 0.75,
          first_mentioned: '2024-01-15T11:00:00Z',
          last_mentioned: '2024-01-15T12:30:00Z'
        }
      ],
      topic_progression: [],
      topic_sentiment: {
        'pricing': 0.3,
        'features': 0.8,
        'integration': 0.5,
        'support': 0.9,
        'timeline': 0.2
      },
      engagement_by_topic: {
        'pricing': 0.7,
        'features': 0.9,
        'integration': 0.6,
        'support': 0.8,
        'timeline': 0.4
      },
      successful_topics: ['features', 'support', 'customization'],
      problematic_topics: ['pricing', 'timeline', 'complexity'],
      topic_recommendations: [
        'Focus more on feature benefits rather than technical details',
        'Address pricing concerns early with value demonstration',
        'Provide clear timeline expectations'
      ]
    },
    success_prediction: {
      probability_of_success: 0.78,
      confidence_interval: [0.68, 0.88],
      key_success_factors: [
        'High engagement with feature discussions',
        'Positive sentiment trend',
        'Multiple touchpoints maintained'
      ],
      risk_factors: [
        'Price sensitivity detected',
        'Timeline concerns expressed',
        'Limited budget authority'
      ],
      recommended_actions: [
        'Schedule demo focusing on ROI',
        'Provide flexible pricing options',
        'Connect with decision maker'
      ],
      time_to_conversion_estimate: 72,
      optimal_next_message_timing: 4,
      conversation_health_score: 84.2
    },
    coaching_recommendations: [
      {
        id: 'rec_001',
        type: 'messaging',
        priority: 'high',
        recommendation: 'Emphasize ROI and cost savings in next message',
        rationale: 'User showed price sensitivity but high feature interest',
        expected_impact: 0.34,
        implementation_difficulty: 'easy',
        success_examples: ['Similar users converted 68% better with ROI focus'],
        ai_confidence: 0.89
      },
      {
        id: 'rec_002',
        type: 'timing',
        priority: 'medium',
        recommendation: 'Follow up in 4-6 hours during their active window',
        rationale: 'User typically responds between 2-4 PM based on pattern analysis',
        expected_impact: 0.23,
        implementation_difficulty: 'easy',
        success_examples: ['40% higher response rate during active windows'],
        ai_confidence: 0.76
      },
      {
        id: 'rec_003',
        type: 'strategy',
        priority: 'high',
        recommendation: 'Offer personalized demo or trial',
        rationale: 'High feature engagement suggests hands-on approach would be effective',
        expected_impact: 0.45,
        implementation_difficulty: 'medium',
        success_examples: ['67% conversion rate for users with similar engagement patterns'],
        ai_confidence: 0.82
      }
    ],
    pattern_analysis: {
      conversation_patterns: [
        {
          pattern_id: 'pattern_001',
          pattern_type: 'engagement',
          pattern_description: 'Progressive interest building through feature exploration',
          frequency: 23,
          success_rate: 78.3,
          avg_conversation_impact: 0.67,
          contexts_where_effective: ['technical users', 'feature-focused discussions'],
          similar_patterns: ['pattern_045', 'pattern_067']
        },
        {
          pattern_id: 'pattern_002',
          pattern_type: 'objection_handling',
          pattern_description: 'Price concern addressed with value demonstration',
          frequency: 15,
          success_rate: 65.4,
          avg_conversation_impact: 0.54,
          contexts_where_effective: ['price-sensitive users', 'budget-conscious segments'],
          similar_patterns: ['pattern_089', 'pattern_123']
        }
      ],
      user_behavior_patterns: [],
      temporal_patterns: [],
      success_patterns: [],
      failure_patterns: [],
      anomaly_detection: {
        detected_anomalies: [
          {
            anomaly_id: 'anom_001',
            type: 'behavior',
            description: 'Unusually high engagement with technical features for non-technical user',
            severity: 0.6,
            detected_at: '2024-01-15T13:45:00Z',
            confidence: 0.78,
            potential_causes: ['User has hidden technical background', 'Strong influence from technical colleague'],
            recommended_investigation: ['Ask about technical team involvement', 'Gauge decision-making process']
          }
        ],
        anomaly_score: 0.3,
        baseline_comparison: {
          current_performance: 87.3,
          baseline_performance: 73.5,
          variance: 18.8,
          variance_explanation: 'Significantly above average engagement and progression',
          is_significant: true
        },
        investigation_priority: 'medium'
      }
    },
    competitive_insights: {
      market_position: 'leading',
      performance_vs_competitors: [
        {
          competitor: 'DeleteMe',
          metric: 'response_time',
          our_performance: 1.2,
          competitor_performance: 2880, // 48 minutes
          advantage_factor: 2400,
          improvement_potential: 0
        }
      ],
      unique_advantages: ['Real-time AI coaching', 'Predictive analytics', '100x faster responses'],
      improvement_opportunities: ['Enhanced topic modeling', 'Multi-language sentiment analysis'],
      market_trend_alignment: 0.92
    }
  });

  const sentimentTimelineData = useMemo(() => {
    return intelligence.sentiment_analysis.sentiment_timeline.map(point => ({
      time: format(new Date(point.timestamp), 'HH:mm'),
      sentiment: point.value * 100,
      emotion: intelligence.sentiment_analysis.emotional_journey.find(
        ej => Math.abs(new Date(ej.timestamp).getTime() - new Date(point.timestamp).getTime()) < 30 * 60 * 1000
      )?.emotion || 'neutral'
    }));
  }, [intelligence.sentiment_analysis]);

  const topicEngagementData = useMemo(() => {
    return intelligence.topic_analysis.primary_topics.map(topic => ({
      topic: topic.topic,
      relevance: topic.relevance_score * 100,
      sentiment: topic.sentiment_score * 100,
      engagement: topic.engagement_impact * 100,
      conversion: topic.conversion_correlation * 100
    }));
  }, [intelligence.topic_analysis.primary_topics]);

  const handleImplementRecommendation = (id: string) => {
    console.log('Implementing recommendation:', id);
    // Implementation logic here
  };

  const handleDismissRecommendation = (id: string) => {
    console.log('Dismissing recommendation:', id);
    // Dismissal logic here
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: ChartBarIcon },
    { id: 'sentiment', label: 'Sentiment', icon: SparklesIcon },
    { id: 'topics', label: 'Topics', icon: MagnifyingGlassIcon },
    { id: 'patterns', label: 'Patterns', icon: TrendingUpIcon },
    { id: 'coaching', label: 'Coaching', icon: AcademicCapIcon },
    { id: 'predictions', label: 'Predictions', icon: CpuChipIcon }
  ];

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Conversation Intelligence</h2>
            <p className="text-gray-600 mt-1">
              AI-powered analysis and coaching recommendations
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Real-time Analysis</span>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4 text-center">
            <div className="flex items-center justify-center w-12 h-12 mx-auto mb-2 bg-green-100 rounded-lg">
              <CheckCircleIcon className="w-6 h-6 text-green-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {(intelligence.success_prediction.probability_of_success * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-gray-600">Success Probability</div>
          </Card>

          <Card className="p-4 text-center">
            <div className="flex items-center justify-center w-12 h-12 mx-auto mb-2 bg-blue-100 rounded-lg">
              <SparklesIcon className="w-6 h-6 text-blue-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {intelligence.quality_score.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Quality Score</div>
          </Card>

          <Card className="p-4 text-center">
            <SentimentGauge 
              sentiment={intelligence.sentiment_analysis.sentiment_score}
              size="sm"
              label="Sentiment"
            />
          </Card>

          <Card className="p-4 text-center">
            <div className="flex items-center justify-center w-12 h-12 mx-auto mb-2 bg-purple-100 rounded-lg">
              <ClockIcon className="w-6 h-6 text-purple-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {intelligence.success_prediction.time_to_conversion_estimate}h
            </div>
            <div className="text-sm text-gray-600">Est. Conversion</div>
          </Card>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setSelectedTab(tab.id as any)}
                  className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    selectedTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {selectedTab === 'overview' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Sentiment Timeline */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Journey</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={sentimentTimelineData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis dataKey="time" stroke="#6B7280" fontSize={12} />
                      <YAxis stroke="#6B7280" fontSize={12} domain={[-100, 100]} />
                      <Tooltip 
                        formatter={(value, name) => [`${value}%`, 'Sentiment']}
                        labelFormatter={(label) => `Time: ${label}`}
                      />
                      <Area
                        type="monotone"
                        dataKey="sentiment"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>

                {/* Success Factors */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Success Factors</h3>
                  <div className="space-y-4">
                    {intelligence.success_prediction.key_success_factors.map((factor, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <CheckCircleIcon className="w-5 h-5 text-green-500 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{factor}</span>
                      </div>
                    ))}
                  </div>
                  
                  <h4 className="font-medium text-gray-900 mt-6 mb-3">Risk Factors</h4>
                  <div className="space-y-3">
                    {intelligence.success_prediction.risk_factors.map((factor, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{factor}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {selectedTab === 'sentiment' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Sentiment Overview */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Overview</h3>
                  <div className="text-center">
                    <SentimentGauge 
                      sentiment={intelligence.sentiment_analysis.sentiment_score}
                      size="lg"
                    />
                    <div className="mt-4">
                      <div className="text-2xl font-bold text-gray-900 capitalize">
                        {intelligence.sentiment_analysis.overall_sentiment}
                      </div>
                      <div className="text-sm text-gray-600">Overall Sentiment</div>
                    </div>
                  </div>
                </Card>

                {/* Sentiment Drivers */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Drivers</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-green-700 mb-2">Positive Factors</h4>
                      <div className="space-y-1">
                        {intelligence.sentiment_analysis.sentiment_drivers.positive_factors.map((factor, index) => (
                          <div key={index} className="text-sm text-gray-600 flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span>{factor}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-red-700 mb-2">Negative Factors</h4>
                      <div className="space-y-1">
                        {intelligence.sentiment_analysis.sentiment_drivers.negative_factors.map((factor, index) => (
                          <div key={index} className="text-sm text-gray-600 flex items-center space-x-2">
                            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                            <span>{factor}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Emotional Journey */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Emotional Journey</h3>
                  <div className="space-y-4">
                    {intelligence.sentiment_analysis.emotional_journey.map((journey, index) => (
                      <div key={index} className="border-l-4 border-blue-500 pl-4">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900 capitalize">{journey.emotion}</span>
                          <span className="text-sm text-gray-500">
                            {format(new Date(journey.timestamp), 'HH:mm')}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 mt-1">{journey.context}</div>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${journey.intensity * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {selectedTab === 'topics' && (
              <div className="space-y-6">
                {/* Topic Visualization */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Topic Engagement Map</h3>
                  <div className="flex flex-wrap gap-4 justify-center">
                    {intelligence.topic_analysis.primary_topics.map((topic, index) => (
                      <TopicBubble
                        key={index}
                        topic={topic.topic}
                        relevance={topic.relevance_score}
                        sentiment={topic.sentiment_score}
                        engagement={topic.engagement_impact}
                        onClick={() => console.log('Topic clicked:', topic.topic)}
                      />
                    ))}
                  </div>
                </Card>

                {/* Topic Performance */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Topic Performance</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={topicEngagementData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis dataKey="topic" stroke="#6B7280" fontSize={12} />
                      <YAxis stroke="#6B7280" fontSize={12} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="engagement" fill="#3B82F6" name="Engagement %" />
                      <Bar dataKey="sentiment" fill="#10B981" name="Sentiment %" />
                      <Bar dataKey="conversion" fill="#F59E0B" name="Conversion %" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                {/* Topic Recommendations */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Topic Recommendations</h3>
                  <div className="space-y-3">
                    {intelligence.topic_analysis.topic_recommendations.map((recommendation, index) => (
                      <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                        <LightBulbIcon className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-700">{recommendation}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {selectedTab === 'patterns' && (
              <div className="space-y-6">
                {/* Pattern Analysis */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Successful Patterns</h3>
                    <div className="space-y-4">
                      {intelligence.pattern_analysis.conversation_patterns.map((pattern, index) => (
                        <PatternCard 
                          key={index}
                          pattern={pattern}
                          onClick={() => console.log('Pattern clicked:', pattern.pattern_id)}
                        />
                      ))}
                    </div>
                  </Card>

                  {/* Anomaly Detection */}
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Anomaly Detection</h3>
                    <div className="space-y-4">
                      {intelligence.pattern_analysis.anomaly_detection.detected_anomalies.map((anomaly, index) => (
                        <div key={index} className="border border-yellow-200 rounded-lg p-4 bg-yellow-50">
                          <div className="flex items-start space-x-3">
                            <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                            <div className="flex-1">
                              <h4 className="font-medium text-gray-900">{anomaly.description}</h4>
                              <div className="text-sm text-gray-600 mt-1">
                                Confidence: {(anomaly.confidence * 100).toFixed(0)}% | 
                                Severity: {(anomaly.severity * 100).toFixed(0)}%
                              </div>
                              <div className="mt-2">
                                <p className="text-sm text-gray-700">Potential causes:</p>
                                <ul className="text-sm text-gray-600 mt-1 space-y-1">
                                  {anomaly.potential_causes.map((cause, causeIndex) => (
                                    <li key={causeIndex} className="flex items-center space-x-2">
                                      <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                                      <span>{cause}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              </div>
            )}

            {selectedTab === 'coaching' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 gap-4">
                  {intelligence.coaching_recommendations.map((recommendation, index) => (
                    <RecommendationCard
                      key={index}
                      recommendation={recommendation}
                      onImplement={handleImplementRecommendation}
                      onDismiss={handleDismissRecommendation}
                    />
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'predictions' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Success Prediction */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Success Prediction</h3>
                  <div className="text-center mb-6">
                    <div className="text-4xl font-bold text-green-600">
                      {(intelligence.success_prediction.probability_of_success * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-600">Probability of Success</div>
                    <div className="text-xs text-gray-500 mt-1">
                      Confidence: {((intelligence.success_prediction.confidence_interval[1] - intelligence.success_prediction.confidence_interval[0]) * 100).toFixed(0)}% range
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Recommended Actions</h4>
                      <div className="space-y-2">
                        {intelligence.success_prediction.recommended_actions.map((action, index) => (
                          <div key={index} className="flex items-center space-x-2 text-sm">
                            <CheckCircleIcon className="w-4 h-4 text-blue-500 flex-shrink-0" />
                            <span className="text-gray-700">{action}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Timing Predictions */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Timing Intelligence</h3>
                  <div className="space-y-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600">
                        {intelligence.success_prediction.optimal_next_message_timing}h
                      </div>
                      <div className="text-sm text-gray-600">Optimal Next Contact</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-3xl font-bold text-purple-600">
                        {intelligence.success_prediction.time_to_conversion_estimate}h
                      </div>
                      <div className="text-sm text-gray-600">Estimated Conversion Time</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-600">
                        {intelligence.success_prediction.conversation_health_score.toFixed(1)}
                      </div>
                      <div className="text-sm text-gray-600">Conversation Health Score</div>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </ErrorBoundary>
  );
};