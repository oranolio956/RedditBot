/**
 * CRM Contact Profiles - Comprehensive contact management with AI-powered insights
 * Complete conversation history, lead scoring, and relationship intelligence
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '../../ui/Card';
import { LoadingSpinner } from '../../ui/LoadingSpinner';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, 
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import {
  UserCircleIcon,
  ChatBubbleLeftRightIcon,
  StarIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  ClockIcon,
  MapPinIcon,
  PhoneIcon,
  EnvelopeIcon,
  GlobeAltIcon,
  HeartIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  EyeIcon,
  TagIcon,
  CalendarIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  UserGroupIcon,
  FireIcon,
  ShieldCheckIcon,
  BoltIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow, startOfMonth, endOfMonth } from 'date-fns';
import {
  CRMContact,
  ContactProfile,
  LeadScore,
  ContactEngagementMetrics,
  RelationshipIntelligence,
  RevenueAttribution,
  ContactTimelineEvent,
  ConversationSummary,
  LifecycleStage
} from '../../../types/analytics';

interface CRMContactProfilesProps {
  className?: string;
  contactId?: string;
  viewMode?: 'list' | 'grid' | 'detail';
  filterBy?: {
    lifecycle_stage?: LifecycleStage[];
    lead_score_range?: [number, number];
    last_activity_days?: number;
    tags?: string[];
  };
  sortBy?: 'lead_score' | 'last_activity' | 'revenue' | 'engagement';
  onContactSelect?: (contact: CRMContact) => void;
}

interface ContactListItemProps {
  contact: CRMContact;
  onClick: () => void;
  selected?: boolean;
}

const LeadScoreGauge: React.FC<{
  score: number;
  maxScore: number;
  size?: 'sm' | 'md' | 'lg';
}> = ({ score, maxScore, size = 'md' }) => {
  const radius = size === 'sm' ? 25 : size === 'md' ? 35 : 45;
  const strokeWidth = size === 'sm' ? 3 : size === 'md' ? 4 : 5;
  const normalizedScore = score / maxScore;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = `${normalizedScore * circumference} ${circumference}`;
  
  const getColor = (value: number) => {
    if (value > 0.8) return '#10B981'; // Green
    if (value > 0.6) return '#3B82F6'; // Blue
    if (value > 0.4) return '#F59E0B'; // Yellow
    return '#EF4444'; // Red
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={radius * 2.2} height={radius * 2.2} className="transform -rotate-90">
        <circle
          cx={radius * 1.1}
          cy={radius * 1.1}
          r={radius}
          stroke="#E5E7EB"
          strokeWidth={strokeWidth}
          fill="none"
        />
        <circle
          cx={radius * 1.1}
          cy={radius * 1.1}
          r={radius}
          stroke={getColor(normalizedScore)}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={strokeDasharray}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`font-bold ${size === 'sm' ? 'text-xs' : size === 'md' ? 'text-sm' : 'text-base'}`}>
          {score}
        </span>
      </div>
    </div>
  );
};

const LifecycleStageIndicator: React.FC<{
  stage: LifecycleStage;
  size?: 'sm' | 'md';
}> = ({ stage, size = 'md' }) => {
  const stageConfig = {
    subscriber: { color: 'bg-gray-100 text-gray-800', label: 'Subscriber' },
    lead: { color: 'bg-blue-100 text-blue-800', label: 'Lead' },
    marketing_qualified_lead: { color: 'bg-indigo-100 text-indigo-800', label: 'MQL' },
    sales_qualified_lead: { color: 'bg-purple-100 text-purple-800', label: 'SQL' },
    opportunity: { color: 'bg-yellow-100 text-yellow-800', label: 'Opportunity' },
    customer: { color: 'bg-green-100 text-green-800', label: 'Customer' },
    evangelist: { color: 'bg-pink-100 text-pink-800', label: 'Evangelist' },
    churned: { color: 'bg-red-100 text-red-800', label: 'Churned' }
  };

  const config = stageConfig[stage];
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full ${textSize} font-medium ${config.color}`}>
      {config.label}
    </span>
  );
};

const ContactListItem: React.FC<ContactListItemProps> = ({ contact, onClick, selected }) => {
  const lastActivity = contact.engagement_metrics.last_interaction_date 
    ? formatDistanceToNow(new Date(contact.engagement_metrics.last_interaction_date), { addSuffix: true })
    : 'No activity';

  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      onClick={onClick}
      className={`cursor-pointer transition-all duration-200 ${selected ? 'ring-2 ring-blue-500' : ''}`}
    >
      <Card className={`p-4 hover:shadow-md ${selected ? 'bg-blue-50 border-blue-200' : 'bg-white hover:bg-gray-50'}`}>
        <div className="flex items-center space-x-4">
          {/* Avatar */}
          <div className="relative">
            {contact.profile.basic_info.profile_photo_url ? (
              <img
                src={contact.profile.basic_info.profile_photo_url}
                alt={`${contact.profile.basic_info.first_name} ${contact.profile.basic_info.last_name}`}
                className="w-12 h-12 rounded-full object-cover"
              />
            ) : (
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center">
                <UserCircleIcon className="w-8 h-8 text-white" />
              </div>
            )}
            <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-white ${
              contact.engagement_metrics.last_interaction_date && 
              new Date(contact.engagement_metrics.last_interaction_date) > new Date(Date.now() - 24 * 60 * 60 * 1000)
                ? 'bg-green-500' : 'bg-gray-400'
            }`}></div>
          </div>

          {/* Main Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-3">
              <h3 className="text-lg font-semibold text-gray-900 truncate">
                {contact.profile.basic_info.first_name} {contact.profile.basic_info.last_name}
              </h3>
              <LifecycleStageIndicator stage={contact.lifecycle_stage} size="sm" />
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
              {contact.profile.basic_info.email && (
                <div className="flex items-center space-x-1">
                  <EnvelopeIcon className="w-4 h-4" />
                  <span className="truncate">{contact.profile.basic_info.email}</span>
                </div>
              )}
              {contact.profile.basic_info.location && (
                <div className="flex items-center space-x-1">
                  <MapPinIcon className="w-4 h-4" />
                  <span>{contact.profile.basic_info.location}</span>
                </div>
              )}
            </div>
            <div className="flex items-center space-x-6 mt-2 text-sm text-gray-500">
              <span>Last activity: {lastActivity}</span>
              <span>{contact.conversation_history.length} conversations</span>
              <span>${contact.revenue_attribution.total_revenue.toLocaleString()}</span>
            </div>
          </div>

          {/* Metrics */}
          <div className="flex items-center space-x-6">
            <div className="text-center">
              <LeadScoreGauge 
                score={contact.lead_score.current_score}
                maxScore={contact.lead_score.max_score}
                size="sm"
              />
              <div className="text-xs text-gray-600 mt-1">Lead Score</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-bold text-gray-900">
                {contact.engagement_metrics.engagement_score_trend.length > 0 
                  ? contact.engagement_metrics.engagement_score_trend[contact.engagement_metrics.engagement_score_trend.length - 1].value.toFixed(0)
                  : '0'
                }
              </div>
              <div className="text-xs text-gray-600">Engagement</div>
            </div>

            <div className="text-center">
              <div className="text-lg font-bold text-green-600">
                ${contact.revenue_attribution.total_revenue.toLocaleString()}
              </div>
              <div className="text-xs text-gray-600">Revenue</div>
            </div>
          </div>
        </div>

        {/* Tags */}
        {contact.tags.length > 0 && (
          <div className="flex items-center space-x-2 mt-3">
            <TagIcon className="w-4 h-4 text-gray-400" />
            <div className="flex flex-wrap gap-1">
              {contact.tags.slice(0, 3).map((tag, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800"
                >
                  {tag}
                </span>
              ))}
              {contact.tags.length > 3 && (
                <span className="text-xs text-gray-500">+{contact.tags.length - 3} more</span>
              )}
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
};

const ContactDetailPanel: React.FC<{
  contact: CRMContact;
  onEdit?: () => void;
  onDelete?: () => void;
}> = ({ contact, onEdit, onDelete }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'conversations' | 'timeline' | 'analytics'>('overview');

  const personalityRadarData = useMemo(() => [
    { trait: 'Openness', value: contact.profile.psychographics.personality_traits.openness * 100 || 50 },
    { trait: 'Conscientiousness', value: contact.profile.psychographics.personality_traits.conscientiousness * 100 || 50 },
    { trait: 'Extraversion', value: contact.profile.psychographics.personality_traits.extraversion * 100 || 50 },
    { trait: 'Agreeableness', value: contact.profile.psychographics.personality_traits.agreeableness * 100 || 50 },
    { trait: 'Neuroticism', value: contact.profile.psychographics.personality_traits.neuroticism * 100 || 50 }
  ], [contact.profile.psychographics.personality_traits]);

  const engagementTrendData = useMemo(() => {
    return contact.engagement_metrics.engagement_score_trend.map(point => ({
      date: format(new Date(point.timestamp), 'MMM dd'),
      engagement: point.value,
      interactions: Math.floor(Math.random() * 10) + 1 // Mock data
    }));
  }, [contact.engagement_metrics.engagement_score_trend]);

  const revenueTimelineData = useMemo(() => {
    return contact.revenue_attribution.revenue_by_period.map(point => ({
      date: format(new Date(point.timestamp), 'MMM dd'),
      revenue: point.value,
      cumulative: contact.revenue_attribution.revenue_by_period
        .slice(0, contact.revenue_attribution.revenue_by_period.indexOf(point) + 1)
        .reduce((sum, p) => sum + p.value, 0)
    }));
  }, [contact.revenue_attribution.revenue_by_period]);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: EyeIcon },
    { id: 'conversations', label: 'Conversations', icon: ChatBubbleLeftRightIcon },
    { id: 'timeline', label: 'Timeline', icon: CalendarIcon },
    { id: 'analytics', label: 'Analytics', icon: ChartBarIcon }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-6">
            {/* Avatar */}
            <div className="relative">
              {contact.profile.basic_info.profile_photo_url ? (
                <img
                  src={contact.profile.basic_info.profile_photo_url}
                  alt={`${contact.profile.basic_info.first_name} ${contact.profile.basic_info.last_name}`}
                  className="w-20 h-20 rounded-full object-cover"
                />
              ) : (
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center">
                  <UserCircleIcon className="w-12 h-12 text-white" />
                </div>
              )}
              <div className={`absolute -bottom-1 -right-1 w-6 h-6 rounded-full border-4 border-white ${
                contact.engagement_metrics.last_interaction_date && 
                new Date(contact.engagement_metrics.last_interaction_date) > new Date(Date.now() - 24 * 60 * 60 * 1000)
                  ? 'bg-green-500' : 'bg-gray-400'
              }`}></div>
            </div>

            {/* Basic Info */}
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-gray-900">
                {contact.profile.basic_info.first_name} {contact.profile.basic_info.last_name}
              </h1>
              <div className="flex items-center space-x-3 mt-2">
                <LifecycleStageIndicator stage={contact.lifecycle_stage} />
                <LeadScoreGauge 
                  score={contact.lead_score.current_score}
                  maxScore={contact.lead_score.max_score}
                  size="md"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                {contact.profile.basic_info.email && (
                  <div className="flex items-center space-x-2">
                    <EnvelopeIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600">{contact.profile.basic_info.email}</span>
                  </div>
                )}
                {contact.profile.basic_info.phone_number && (
                  <div className="flex items-center space-x-2">
                    <PhoneIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600">{contact.profile.basic_info.phone_number}</span>
                  </div>
                )}
                {contact.profile.basic_info.location && (
                  <div className="flex items-center space-x-2">
                    <MapPinIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600">{contact.profile.basic_info.location}</span>
                  </div>
                )}
                {contact.profile.basic_info.timezone && (
                  <div className="flex items-center space-x-2">
                    <GlobeAltIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600">{contact.profile.basic_info.timezone}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-2">
            {onEdit && (
              <button
                onClick={onEdit}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <PencilIcon className="w-5 h-5" />
              </button>
            )}
            {onDelete && (
              <button
                onClick={onDelete}
                className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <TrashIcon className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-4 gap-6 mt-6 pt-6 border-t border-gray-200">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {contact.conversation_history.length}
            </div>
            <div className="text-sm text-gray-600">Conversations</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              ${contact.revenue_attribution.total_revenue.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Total Revenue</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {contact.engagement_metrics.total_interactions}
            </div>
            <div className="text-sm text-gray-600">Interactions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {contact.relationship_intelligence.relationship_strength.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Relationship Score</div>
          </div>
        </div>

        {/* Tags */}
        {contact.tags.length > 0 && (
          <div className="flex items-center space-x-2 mt-6 pt-6 border-t border-gray-200">
            <TagIcon className="w-5 h-5 text-gray-400" />
            <div className="flex flex-wrap gap-2">
              {contact.tags.map((tag, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </Card>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
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
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Personality Profile */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Personality Profile</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={personalityRadarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="trait" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Personality"
                      dataKey="value"
                      stroke="#3B82F6"
                      fill="#3B82F6"
                      fillOpacity={0.3}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </Card>

              {/* Lead Score Breakdown */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Lead Score Breakdown</h3>
                <div className="space-y-4">
                  {Object.entries(contact.lead_score.score_breakdown).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700 capitalize">
                        {key.replace('_', ' ')}
                      </span>
                      <div className="flex items-center space-x-3">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${(value / 100) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-900 w-8 text-right">
                          {value}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="mt-6 pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    <span className="text-base font-semibold text-gray-900">Total Score</span>
                    <span className="text-xl font-bold text-blue-600">
                      {contact.lead_score.current_score}/{contact.lead_score.max_score}
                    </span>
                  </div>
                </div>
              </Card>

              {/* Relationship Intelligence */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Relationship Intelligence</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {contact.relationship_intelligence.relationship_strength.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-600">Relationship Strength</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {contact.relationship_intelligence.trust_level.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-600">Trust Level</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {contact.relationship_intelligence.influence_score.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-600">Influence Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {contact.relationship_intelligence.referral_potential.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-600">Referral Potential</div>
                  </div>
                </div>
                
                <div className="mt-4">
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    contact.relationship_intelligence.relationship_health === 'strong' ? 'bg-green-100 text-green-800' :
                    contact.relationship_intelligence.relationship_health === 'good' ? 'bg-blue-100 text-blue-800' :
                    contact.relationship_intelligence.relationship_health === 'weak' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {contact.relationship_intelligence.relationship_health} relationship
                  </div>
                </div>
              </Card>

              {/* Revenue Attribution */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Revenue Attribution</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Total Revenue</span>
                    <span className="text-lg font-bold text-green-600">
                      ${contact.revenue_attribution.total_revenue.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Customer Lifetime Value</span>
                    <span className="text-lg font-bold text-blue-600">
                      ${contact.revenue_attribution.customer_lifetime_value.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Predicted Future Value</span>
                    <span className="text-lg font-bold text-purple-600">
                      ${contact.revenue_attribution.predicted_future_value.toLocaleString()}
                    </span>
                  </div>
                  
                  {contact.revenue_attribution.churn_risk.risk_level !== 'low' && (
                    <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600" />
                        <span className="text-sm font-medium text-yellow-800">
                          {contact.revenue_attribution.churn_risk.risk_level} churn risk
                        </span>
                      </div>
                      <div className="text-sm text-yellow-700 mt-1">
                        Risk Score: {contact.revenue_attribution.churn_risk.risk_score}/100
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}

          {activeTab === 'conversations' && (
            <div className="space-y-4">
              {contact.conversation_history.map((conversation, index) => (
                <Card key={index} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h4 className="font-medium text-gray-900">
                          Conversation {index + 1}
                        </h4>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          conversation.status === 'active' ? 'bg-green-100 text-green-800' :
                          conversation.status === 'completed' ? 'bg-blue-100 text-blue-800' :
                          conversation.status === 'abandoned' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {conversation.status}
                        </span>
                        <span className="text-sm text-gray-500 capitalize">
                          {conversation.stage_reached.replace('_', ' ')}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3 text-sm text-gray-600">
                        <div>
                          <span className="font-medium">Duration:</span> {conversation.start_date} - {conversation.end_date || 'Ongoing'}
                        </div>
                        <div>
                          <span className="font-medium">Messages:</span> {conversation.message_count}
                        </div>
                        <div>
                          <span className="font-medium">Engagement:</span> {conversation.engagement_score.toFixed(1)}/100
                        </div>
                        <div>
                          <span className="font-medium">Revenue:</span> ${conversation.revenue_generated?.toLocaleString() || '0'}
                        </div>
                      </div>

                      {conversation.key_topics.length > 0 && (
                        <div className="flex items-center space-x-2 mt-3">
                          <span className="text-sm font-medium text-gray-700">Topics:</span>
                          <div className="flex flex-wrap gap-1">
                            {conversation.key_topics.map((topic, topicIndex) => (
                              <span
                                key={topicIndex}
                                className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800"
                              >
                                {topic}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors">
                        <EyeIcon className="w-4 h-4" />
                      </button>
                      <button className="p-2 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors">
                        <ChatBubbleLeftRightIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {activeTab === 'timeline' && (
            <div className="space-y-4">
              <div className="relative">
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200"></div>
                {contact.contact_timeline.map((event, index) => (
                  <div key={index} className="relative flex items-start space-x-6 pb-6">
                    <div className="relative z-10 flex items-center justify-center w-8 h-8 bg-blue-100 rounded-full">
                      <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <Card className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900">{event.title}</h4>
                            <p className="text-sm text-gray-600 mt-1">{event.description}</p>
                            <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                              <span>{format(new Date(event.timestamp), 'MMM dd, yyyy HH:mm')}</span>
                              <span>Impact: {event.impact_score}/100</span>
                            </div>
                          </div>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            event.type === 'conversion' ? 'bg-green-100 text-green-800' :
                            event.type === 'conversation' ? 'bg-blue-100 text-blue-800' :
                            event.type === 'milestone' ? 'bg-purple-100 text-purple-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {event.type}
                          </span>
                        </div>
                      </Card>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'analytics' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Engagement Trend */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Engagement Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={engagementTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                    <YAxis stroke="#6B7280" fontSize={12} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="engagement"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              {/* Revenue Timeline */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Revenue Timeline</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={revenueTimelineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="date" stroke="#6B7280" fontSize={12} />
                    <YAxis stroke="#6B7280" fontSize={12} />
                    <Tooltip formatter={(value) => [`$${value}`, 'Revenue']} />
                    <Area
                      type="monotone"
                      dataKey="cumulative"
                      stroke="#10B981"
                      fill="#10B981"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export const CRMContactProfiles: React.FC<CRMContactProfilesProps> = ({
  className = '',
  contactId,
  viewMode = 'list',
  filterBy,
  sortBy = 'lead_score',
  onContactSelect
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedContact, setSelectedContact] = useState<CRMContact | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);

  // Mock contacts data
  const [contacts, setContacts] = useState<CRMContact[]>([
    // Mock contact data would be populated here from API
    {
      contact_id: 'contact_001',
      telegram_user_id: 123456789,
      profile: {
        basic_info: {
          first_name: 'Sarah',
          last_name: 'Johnson',
          username: 'sarah_j',
          email: 'sarah.johnson@email.com',
          phone_number: '+1-555-0123',
          location: 'San Francisco, CA',
          timezone: 'America/Los_Angeles',
          language: 'en',
          profile_photo_url: undefined
        },
        demographics: {
          age_range: '25-34',
          gender: 'female',
          occupation: 'Software Engineer',
          income_level: '$75k-$100k',
          education_level: 'Bachelor\'s Degree'
        },
        psychographics: {
          personality_traits: {
            openness: 0.8,
            conscientiousness: 0.7,
            extraversion: 0.6,
            agreeableness: 0.9,
            neuroticism: 0.3
          },
          interests: ['technology', 'productivity', 'privacy'],
          values: ['security', 'efficiency', 'innovation'],
          lifestyle_indicators: ['remote work', 'tech-savvy'],
          communication_style: 'professional'
        },
        behavioral_data: {
          online_activity_pattern: [],
          response_patterns: [],
          engagement_preferences: [],
          purchase_history: [],
          content_preferences: []
        },
        enrichment_data: {
          social_media_profiles: [],
          professional_background: {
            current_position: 'Senior Software Engineer',
            company: 'TechCorp Inc.',
            industry: 'Technology',
            experience_level: 'Senior',
            skills: ['React', 'Python', 'AWS'],
            professional_interests: ['AI', 'Privacy', 'Security']
          },
          network_connections: [],
          digital_footprint_score: 78,
          data_sources: ['telegram', 'web_scraping'],
          last_enriched: new Date().toISOString()
        }
      },
      conversation_history: [
        {
          conversation_id: 'conv_001',
          start_date: '2024-01-10',
          end_date: '2024-01-15',
          status: 'completed',
          stage_reached: 'payment_discussion',
          message_count: 47,
          avg_response_time: 1.2,
          engagement_score: 89.5,
          satisfaction_score: 4.8,
          outcome: 'successful',
          revenue_generated: 2500,
          key_topics: ['privacy protection', 'data removal', 'pricing'],
          sentiment_summary: 'positive',
          ai_confidence_avg: 92.3,
          human_interventions: 2
        }
      ],
      lead_score: {
        current_score: 85,
        max_score: 100,
        score_history: [
          { timestamp: '2024-01-10T00:00:00Z', value: 45 },
          { timestamp: '2024-01-12T00:00:00Z', value: 67 },
          { timestamp: '2024-01-15T00:00:00Z', value: 85 }
        ],
        scoring_factors: [],
        score_breakdown: {
          demographic_score: 78,
          behavioral_score: 89,
          engagement_score: 92,
          intent_score: 85,
          fit_score: 76
        },
        prediction_confidence: 0.87,
        recommended_actions: ['Schedule follow-up call', 'Send pricing proposal'],
        score_trend: 'increasing'
      },
      lifecycle_stage: 'customer',
      engagement_metrics: {
        total_interactions: 47,
        last_interaction_date: '2024-01-15T10:30:00Z',
        interaction_frequency: 3.2,
        avg_session_duration: 25.6,
        engagement_score_trend: [
          { timestamp: '2024-01-10T00:00:00Z', value: 65 },
          { timestamp: '2024-01-12T00:00:00Z', value: 78 },
          { timestamp: '2024-01-15T00:00:00Z', value: 89 }
        ],
        preferred_contact_times: ['14:00-16:00', '19:00-21:00'],
        response_rate: 0.94,
        click_through_rate: 0.67,
        conversion_events: [],
        engagement_quality_score: 87.3
      },
      relationship_intelligence: {
        relationship_strength: 87.5,
        trust_level: 92.1,
        influence_score: 76.3,
        network_value: 45.2,
        relationship_timeline: [],
        mutual_connections: [],
        referral_potential: 78.9,
        relationship_health: 'strong',
        recommended_relationship_actions: ['Request referrals', 'Upsell premium features']
      },
      revenue_attribution: {
        total_revenue: 2500,
        revenue_by_period: [
          { timestamp: '2024-01-15T00:00:00Z', value: 2500 }
        ],
        revenue_sources: [],
        customer_lifetime_value: 7500,
        predicted_future_value: 5000,
        revenue_quality_score: 89.2,
        upsell_opportunities: [],
        churn_risk: {
          risk_level: 'low',
          risk_score: 15,
          risk_factors: [],
          early_warning_indicators: [],
          retention_strategies: []
        }
      },
      contact_timeline: [
        {
          event_id: 'timeline_001',
          type: 'conversation',
          title: 'Initial Contact',
          description: 'First conversation about data privacy concerns',
          timestamp: '2024-01-10T09:00:00Z',
          impact_score: 65,
          conversation_id: 'conv_001',
          metadata: {}
        },
        {
          event_id: 'timeline_002',
          type: 'conversion',
          title: 'Service Purchase',
          description: 'Purchased premium data removal service',
          timestamp: '2024-01-15T14:30:00Z',
          impact_score: 95,
          metadata: { revenue: 2500 }
        }
      ],
      tags: ['high-value', 'tech-professional', 'privacy-focused', 'quick-decision'],
      custom_fields: {},
      created_at: '2024-01-10T09:00:00Z',
      updated_at: '2024-01-15T14:30:00Z'
    }
    // Additional mock contacts would be added here
  ]);

  const filteredContacts = useMemo(() => {
    let filtered = contacts;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(contact =>
        `${contact.profile.basic_info.first_name} ${contact.profile.basic_info.last_name}`
          .toLowerCase()
          .includes(searchQuery.toLowerCase()) ||
        contact.profile.basic_info.email?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        contact.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    // Apply additional filters
    if (filterBy) {
      if (filterBy.lifecycle_stage?.length) {
        filtered = filtered.filter(contact =>
          filterBy.lifecycle_stage!.includes(contact.lifecycle_stage)
        );
      }
      
      if (filterBy.lead_score_range) {
        filtered = filtered.filter(contact =>
          contact.lead_score.current_score >= filterBy.lead_score_range![0] &&
          contact.lead_score.current_score <= filterBy.lead_score_range![1]
        );
      }
      
      if (filterBy.last_activity_days) {
        const cutoffDate = new Date(Date.now() - filterBy.last_activity_days * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(contact =>
          contact.engagement_metrics.last_interaction_date &&
          new Date(contact.engagement_metrics.last_interaction_date) >= cutoffDate
        );
      }
      
      if (filterBy.tags?.length) {
        filtered = filtered.filter(contact =>
          filterBy.tags!.some(tag => contact.tags.includes(tag))
        );
      }
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'lead_score':
          return b.lead_score.current_score - a.lead_score.current_score;
        case 'last_activity':
          const aDate = a.engagement_metrics.last_interaction_date ? new Date(a.engagement_metrics.last_interaction_date) : new Date(0);
          const bDate = b.engagement_metrics.last_interaction_date ? new Date(b.engagement_metrics.last_interaction_date) : new Date(0);
          return bDate.getTime() - aDate.getTime();
        case 'revenue':
          return b.revenue_attribution.total_revenue - a.revenue_attribution.total_revenue;
        case 'engagement':
          const aEngagement = a.engagement_metrics.engagement_score_trend.length > 0 
            ? a.engagement_metrics.engagement_score_trend[a.engagement_metrics.engagement_score_trend.length - 1].value 
            : 0;
          const bEngagement = b.engagement_metrics.engagement_score_trend.length > 0 
            ? b.engagement_metrics.engagement_score_trend[b.engagement_metrics.engagement_score_trend.length - 1].value 
            : 0;
          return bEngagement - aEngagement;
        default:
          return 0;
      }
    });

    return filtered;
  }, [contacts, searchQuery, filterBy, sortBy]);

  useEffect(() => {
    if (contactId) {
      const contact = contacts.find(c => c.contact_id === contactId);
      if (contact) {
        setSelectedContact(contact);
      }
    }
  }, [contactId, contacts]);

  const handleContactClick = (contact: CRMContact) => {
    setSelectedContact(contact);
    onContactSelect?.(contact);
  };

  if (selectedContact) {
    return (
      <div className={className}>
        <div className="mb-4">
          <button
            onClick={() => setSelectedContact(null)}
            className="flex items-center space-x-2 text-blue-600 hover:text-blue-800 transition-colors"
          >
            <ArrowPathIcon className="w-4 h-4" />
            <span>Back to contacts</span>
          </button>
        </div>
        <ContactDetailPanel
          contact={selectedContact}
          onEdit={() => console.log('Edit contact')}
          onDelete={() => console.log('Delete contact')}
        />
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Contact Profiles</h2>
            <p className="text-gray-600 mt-1">
              AI-powered contact management with conversation insights
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Search */}
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search contacts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            
            {/* Add Contact Button */}
            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <PlusIcon className="w-4 h-4" />
              <span>Add Contact</span>
            </button>
          </div>
        </div>

        {/* Filters and View Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <FunnelIcon className="w-5 h-5 text-gray-400" />
            <span className="text-sm text-gray-600">
              {filteredContacts.length} contacts found
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Sort by:</span>
            <select 
              value={sortBy}
              onChange={(e) => {/* Handle sort change */}}
              className="text-sm border border-gray-300 rounded-md px-3 py-1 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="lead_score">Lead Score</option>
              <option value="last_activity">Last Activity</option>
              <option value="revenue">Revenue</option>
              <option value="engagement">Engagement</option>
            </select>
          </div>
        </div>

        {/* Contact List */}
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <LoadingSpinner size="lg" />
          </div>
        ) : error ? (
          <div className="text-center text-red-600 h-64 flex items-center justify-center">
            {error}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredContacts.map((contact) => (
              <ContactListItem
                key={contact.contact_id}
                contact={contact}
                onClick={() => handleContactClick(contact)}
                selected={selectedContact?.contact_id === contact.contact_id}
              />
            ))}
            
            {filteredContacts.length === 0 && (
              <div className="text-center py-12">
                <UserGroupIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No contacts found</h3>
                <p className="text-gray-600">
                  {searchQuery ? 'Try adjusting your search criteria' : 'Start by adding your first contact'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};