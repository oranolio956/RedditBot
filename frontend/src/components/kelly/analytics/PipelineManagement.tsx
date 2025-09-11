/**
 * Pipeline Management - Visual drag-and-drop interface for conversation-driven deals
 * Automated stage advancement and revenue attribution
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Card } from '../../ui/Card';
import { LoadingSpinner } from '../../ui/LoadingSpinner';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  FunnelChart, Funnel, LabelList
} from 'recharts';
import {
  CurrencyDollarIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  EyeIcon,
  ChatBubbleLeftRightIcon,
  CalendarIcon,
  UserIcon,
  TagIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  FunnelIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  FireIcon,
  ShieldCheckIcon,
  BoltIcon,
  InformationCircleIcon,
  PlayIcon,
  PauseIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow, differenceInDays, addDays } from 'date-fns';
import {
  Pipeline,
  Deal,
  PipelineStage,
  StageHistoryEntry,
  DealActivity,
  DealRiskAssessment,
  NextAction,
  Bottleneck,
  AutomationRule
} from '../../../types/analytics';

interface PipelineManagementProps {
  className?: string;
  pipelineId?: string;
  viewMode?: 'kanban' | 'list' | 'analytics';
  autoRefresh?: boolean;
  onDealSelect?: (deal: Deal) => void;
  onStageChange?: (dealId: string, newStageId: string) => void;
}

interface DragState {
  isDragging: boolean;
  draggedDeal: Deal | null;
  dragOverStage: string | null;
}

interface StageColumnProps {
  stage: PipelineStage;
  deals: Deal[];
  onDealClick: (deal: Deal) => void;
  onDealDrop: (dealId: string, stageId: string) => void;
  dragState: DragState;
  setDragState: React.Dispatch<React.SetStateAction<DragState>>;
}

interface DealCardProps {
  deal: Deal;
  onClick: () => void;
  onDragStart: (deal: Deal) => void;
  onDragEnd: () => void;
  isDragging?: boolean;
}

const DealCard: React.FC<DealCardProps> = ({ 
  deal, 
  onClick, 
  onDragStart, 
  onDragEnd, 
  isDragging = false 
}) => {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const daysUntilClose = differenceInDays(new Date(deal.expected_close_date), new Date());
  const isOverdue = daysUntilClose < 0;
  const isUrgent = daysUntilClose <= 7 && daysUntilClose >= 0;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`transition-all duration-200 ${isDragging ? 'opacity-50 rotate-3 scale-105' : ''}`}
    >
      <Card
        className={`p-4 cursor-move hover:shadow-md transition-shadow border-l-4 ${
          isOverdue ? 'border-l-red-500 bg-red-50' :
          isUrgent ? 'border-l-yellow-500 bg-yellow-50' :
          'border-l-blue-500'
        }`}
        draggable
        onDragStart={() => onDragStart(deal)}
        onDragEnd={onDragEnd}
        onClick={onClick}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h4 className="font-semibold text-gray-900 truncate">
              Deal #{deal.deal_id.slice(-6)}
            </h4>
            <p className="text-sm text-gray-600 truncate">
              {deal.contact_id}
            </p>
          </div>
          <div className="flex items-center space-x-1">
            {deal.risk_assessment.risk_level !== 'low' && (
              <div className={`w-3 h-3 rounded-full ${getRiskColor(deal.risk_assessment.risk_level).split(' ')[1]}`}></div>
            )}
            <span className="text-xs font-medium text-gray-500">
              {deal.probability}%
            </span>
          </div>
        </div>

        {/* Value and Expected Close */}
        <div className="space-y-2 mb-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Value:</span>
            <span className="font-semibold text-green-600">
              ${deal.deal_value.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Close:</span>
            <span className={`text-sm font-medium ${
              isOverdue ? 'text-red-600' : isUrgent ? 'text-yellow-600' : 'text-gray-700'
            }`}>
              {isOverdue ? `${Math.abs(daysUntilClose)} days overdue` :
               isUrgent ? `${daysUntilClose} days left` :
               format(new Date(deal.expected_close_date), 'MMM dd')}
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
            <span>Progress</span>
            <span>{deal.probability}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                deal.probability >= 80 ? 'bg-green-500' :
                deal.probability >= 60 ? 'bg-blue-500' :
                deal.probability >= 40 ? 'bg-yellow-500' :
                'bg-red-500'
              }`}
              style={{ width: `${deal.probability}%` }}
            />
          </div>
        </div>

        {/* Tags */}
        {deal.deal_tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {deal.deal_tags.slice(0, 2).map((tag, index) => (
              <span
                key={index}
                className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800"
              >
                {tag}
              </span>
            ))}
            {deal.deal_tags.length > 2 && (
              <span className="text-xs text-gray-500">+{deal.deal_tags.length - 2}</span>
            )}
          </div>
        )}

        {/* Last Activity */}
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-1">
            <ChatBubbleLeftRightIcon className="w-3 h-3" />
            <span>{deal.conversation_ids.length} conversations</span>
          </div>
          <span>{formatDistanceToNow(new Date(deal.last_activity), { addSuffix: true })}</span>
        </div>

        {/* Next Actions */}
        {deal.next_actions.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <div className="flex items-center space-x-2 text-xs">
              <ClockIcon className="w-3 h-3 text-blue-500" />
              <span className="text-gray-600 truncate">
                {deal.next_actions[0].description}
              </span>
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
};

const StageColumn: React.FC<StageColumnProps> = ({
  stage,
  deals,
  onDealClick,
  onDealDrop,
  dragState,
  setDragState
}) => {
  const stageValue = deals.reduce((sum, deal) => sum + deal.deal_value, 0);
  const averageProbability = deals.length > 0 
    ? deals.reduce((sum, deal) => sum + deal.probability, 0) / deals.length 
    : 0;

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (dragState.isDragging) {
      setDragState(prev => ({ ...prev, dragOverStage: stage.stage_id }));
    }
  };

  const handleDragLeave = () => {
    setDragState(prev => ({ ...prev, dragOverStage: null }));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (dragState.draggedDeal) {
      onDealDrop(dragState.draggedDeal.deal_id, stage.stage_id);
    }
    setDragState({
      isDragging: false,
      draggedDeal: null,
      dragOverStage: null
    });
  };

  const handleDealDragStart = (deal: Deal) => {
    setDragState({
      isDragging: true,
      draggedDeal: deal,
      dragOverStage: null
    });
  };

  const handleDealDragEnd = () => {
    setDragState({
      isDragging: false,
      draggedDeal: null,
      dragOverStage: null
    });
  };

  return (
    <div 
      className={`flex-1 min-w-80 transition-all duration-200 ${
        dragState.dragOverStage === stage.stage_id ? 'bg-blue-50 border-blue-200' : 'bg-gray-50'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Card className="h-full">
        {/* Stage Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900">{stage.name}</h3>
              <p className="text-sm text-gray-600">{stage.description}</p>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-gray-900">{deals.length}</div>
              <div className="text-xs text-gray-600">deals</div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 mt-3 text-sm">
            <div>
              <span className="text-gray-600">Value:</span>
              <div className="font-semibold text-green-600">
                ${stageValue.toLocaleString()}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Avg Probability:</span>
              <div className="font-semibold text-blue-600">
                {averageProbability.toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Deals List */}
        <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
          <AnimatePresence>
            {deals.map((deal) => (
              <DealCard
                key={deal.deal_id}
                deal={deal}
                onClick={() => onDealClick(deal)}
                onDragStart={handleDealDragStart}
                onDragEnd={handleDealDragEnd}
                isDragging={dragState.draggedDeal?.deal_id === deal.deal_id}
              />
            ))}
          </AnimatePresence>
          
          {deals.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <FunnelIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No deals in this stage</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

const DealDetailModal: React.FC<{
  deal: Deal;
  isOpen: boolean;
  onClose: () => void;
  onUpdate?: (deal: Deal) => void;
}> = ({ deal, isOpen, onClose, onUpdate }) => {
  const [activeTab, setActiveTab] = useState<'details' | 'history' | 'activities' | 'risk'>('details');

  if (!isOpen) return null;

  const tabs = [
    { id: 'details', label: 'Details', icon: InformationCircleIcon },
    { id: 'history', label: 'History', icon: ClockIcon },
    { id: 'activities', label: 'Activities', icon: ChatBubbleLeftRightIcon },
    { id: 'risk', label: 'Risk Assessment', icon: ShieldCheckIcon }
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-bold text-gray-900">
              Deal #{deal.deal_id.slice(-6)}
            </h2>
            <p className="text-gray-600">Contact: {deal.contact_id}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <XCircleIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-4 gap-6 p-6 bg-gray-50 border-b border-gray-200">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              ${deal.deal_value.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Deal Value</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {deal.probability}%
            </div>
            <div className="text-sm text-gray-600">Probability</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {differenceInDays(new Date(deal.expected_close_date), new Date())}
            </div>
            <div className="text-sm text-gray-600">Days to Close</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {deal.conversation_ids.length}
            </div>
            <div className="text-sm text-gray-600">Conversations</div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center space-x-2 py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
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
        <div className="p-6 max-h-96 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === 'details' && (
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Deal Source
                      </label>
                      <p className="text-gray-900">{deal.deal_source}</p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Created Date
                      </label>
                      <p className="text-gray-900">
                        {format(new Date(deal.created_date), 'MMM dd, yyyy')}
                      </p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Expected Close Date
                      </label>
                      <p className="text-gray-900">
                        {format(new Date(deal.expected_close_date), 'MMM dd, yyyy')}
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Current Stage
                      </label>
                      <p className="text-gray-900">{deal.current_stage}</p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Last Activity
                      </label>
                      <p className="text-gray-900">
                        {formatDistanceToNow(new Date(deal.last_activity), { addSuffix: true })}
                      </p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Tags
                      </label>
                      <div className="flex flex-wrap gap-1">
                        {deal.deal_tags.map((tag, index) => (
                          <span
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'history' && (
                <div className="space-y-4">
                  {deal.stage_history.map((entry, index) => (
                    <div key={index} className="flex items-start space-x-4 p-4 bg-gray-50 rounded-lg">
                      <div className="flex-shrink-0">
                        <div className={`w-3 h-3 rounded-full ${
                          entry.conversion_successful ? 'bg-green-500' : 'bg-gray-400'
                        }`}></div>
                      </div>
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900">{entry.stage_name}</h4>
                        <p className="text-sm text-gray-600">
                          {format(new Date(entry.entered_at), 'MMM dd, yyyy HH:mm')}
                          {entry.exited_at && (
                            <span> - {format(new Date(entry.exited_at), 'MMM dd, yyyy HH:mm')}</span>
                          )}
                        </p>
                        {entry.duration && (
                          <p className="text-sm text-gray-500">
                            Duration: {Math.round(entry.duration / (1000 * 60 * 60 * 24))} days
                          </p>
                        )}
                        {entry.exit_reason && (
                          <p className="text-sm text-gray-700 mt-1">
                            Exit reason: {entry.exit_reason}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'activities' && (
                <div className="space-y-4">
                  {deal.activities.map((activity, index) => (
                    <div key={index} className="flex items-start space-x-4 p-4 border border-gray-200 rounded-lg">
                      <div className="flex-shrink-0">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                          <ChatBubbleLeftRightIcon className="w-4 h-4 text-blue-600" />
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-gray-900 capitalize">{activity.type}</h4>
                          <span className="text-sm text-gray-500">
                            {format(new Date(activity.timestamp), 'MMM dd, HH:mm')}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{activity.description}</p>
                        <p className="text-sm text-gray-700 mt-1">Outcome: {activity.outcome}</p>
                        <div className="flex items-center mt-2">
                          <span className="text-xs text-gray-500">Impact Score:</span>
                          <div className="ml-2 w-16 bg-gray-200 rounded-full h-1">
                            <div 
                              className="bg-blue-600 h-1 rounded-full"
                              style={{ width: `${activity.impact_score}%` }}
                            />
                          </div>
                          <span className="ml-1 text-xs text-gray-600">{activity.impact_score}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'risk' && (
                <div className="space-y-6">
                  <div className="flex items-center space-x-4">
                    <div className={`px-4 py-2 rounded-lg font-medium ${
                      deal.risk_assessment.risk_level === 'low' ? 'bg-green-100 text-green-800' :
                      deal.risk_assessment.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                      deal.risk_assessment.risk_level === 'high' ? 'bg-orange-100 text-orange-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {deal.risk_assessment.risk_level.toUpperCase()} RISK
                    </div>
                    <div className="text-sm text-gray-600">
                      Probability adjustment: {deal.risk_assessment.probability_adjustment > 0 ? '+' : ''}{deal.risk_assessment.probability_adjustment}%
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Risk Factors</h4>
                    <div className="space-y-3">
                      {deal.risk_assessment.risk_factors.map((factor, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium text-gray-900">{factor.factor}</span>
                            <span className="text-sm text-gray-600">
                              Impact: {(factor.impact * 100).toFixed(0)}% | 
                              Probability: {(factor.probability * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="space-y-1">
                            {factor.mitigation_actions.map((action, actionIndex) => (
                              <div key={actionIndex} className="text-sm text-gray-700 flex items-center space-x-2">
                                <CheckCircleIcon className="w-3 h-3 text-green-500 flex-shrink-0" />
                                <span>{action}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Mitigation Strategies</h4>
                    <div className="space-y-2">
                      {deal.risk_assessment.mitigation_strategies.map((strategy, index) => (
                        <div key={index} className="flex items-center space-x-2 text-sm text-gray-700">
                          <BoltIcon className="w-4 h-4 text-blue-500 flex-shrink-0" />
                          <span>{strategy}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};

export const PipelineManagement: React.FC<PipelineManagementProps> = ({
  className = '',
  pipelineId,
  viewMode = 'kanban',
  autoRefresh = true,
  onDealSelect,
  onStageChange
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDeal, setSelectedDeal] = useState<Deal | null>(null);
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);
  const [dragState, setDragState] = useState<DragState>({
    isDragging: false,
    draggedDeal: null,
    dragOverStage: null
  });

  // Mock pipeline data
  const [pipeline, setPipeline] = useState<Pipeline>({
    pipeline_id: 'pipeline_001',
    name: 'Conversation-Driven Sales Pipeline',
    stages: [
      {
        stage_id: 'stage_001',
        name: 'Initial Contact',
        description: 'First conversation initiated',
        order: 1,
        probability: 10,
        expected_duration: 2,
        automation_rules: [],
        stage_requirements: [],
        exit_criteria: ['Qualified interest', 'Budget confirmed']
      },
      {
        stage_id: 'stage_002',
        name: 'Qualification',
        description: 'Needs assessment and fit evaluation',
        order: 2,
        probability: 25,
        expected_duration: 5,
        automation_rules: [],
        stage_requirements: [],
        exit_criteria: ['Pain points identified', 'Authority confirmed']
      },
      {
        stage_id: 'stage_003',
        name: 'Proposal',
        description: 'Solution presentation and pricing',
        order: 3,
        probability: 50,
        expected_duration: 7,
        automation_rules: [],
        stage_requirements: [],
        exit_criteria: ['Proposal delivered', 'Feedback received']
      },
      {
        stage_id: 'stage_004',
        name: 'Negotiation',
        description: 'Terms and pricing discussion',
        order: 4,
        probability: 75,
        expected_duration: 10,
        automation_rules: [],
        stage_requirements: [],
        exit_criteria: ['Terms agreed', 'Contract ready']
      },
      {
        stage_id: 'stage_005',
        name: 'Closing',
        description: 'Final approval and contract signing',
        order: 5,
        probability: 90,
        expected_duration: 3,
        automation_rules: [],
        stage_requirements: [],
        exit_criteria: ['Contract signed', 'Payment processed']
      }
    ],
    deals: [
      {
        deal_id: 'deal_001',
        contact_id: 'Sarah Johnson',
        pipeline_id: 'pipeline_001',
        current_stage: 'stage_003',
        deal_value: 2500,
        probability: 65,
        expected_close_date: addDays(new Date(), 14).toISOString(),
        created_date: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
        last_activity: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        conversation_ids: ['conv_001', 'conv_002'],
        deal_source: 'Telegram conversation',
        deal_tags: ['high-value', 'tech-professional'],
        stage_history: [
          {
            stage_id: 'stage_001',
            stage_name: 'Initial Contact',
            entered_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
            exited_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
            duration: 2 * 24 * 60 * 60 * 1000,
            conversion_successful: true
          },
          {
            stage_id: 'stage_002',
            stage_name: 'Qualification',
            entered_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
            exited_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
            duration: 4 * 24 * 60 * 60 * 1000,
            conversion_successful: true
          },
          {
            stage_id: 'stage_003',
            stage_name: 'Proposal',
            entered_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
            conversion_successful: false
          }
        ],
        activities: [
          {
            activity_id: 'activity_001',
            type: 'conversation',
            description: 'Initial discussion about privacy concerns',
            timestamp: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
            outcome: 'Interested in service',
            impact_score: 75,
            conversation_id: 'conv_001'
          },
          {
            activity_id: 'activity_002',
            type: 'proposal',
            description: 'Sent customized proposal with pricing',
            timestamp: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
            outcome: 'Under review',
            impact_score: 85
          }
        ],
        revenue_attribution: {
          attributed_conversations: ['conv_001', 'conv_002'],
          conversation_impact_scores: { 'conv_001': 75, 'conv_002': 85 },
          primary_driver_conversation: 'conv_002',
          revenue_journey: [],
          attribution_confidence: 0.87
        },
        risk_assessment: {
          risk_level: 'medium',
          risk_factors: [
            {
              factor: 'Price sensitivity',
              impact: 0.3,
              probability: 0.6,
              mitigation_actions: ['Emphasize ROI', 'Offer payment plans']
            }
          ],
          mitigation_strategies: ['Focus on value proposition', 'Provide case studies'],
          probability_adjustment: -10,
          risk_monitoring_alerts: ['No response for 3 days']
        },
        next_actions: [
          {
            action_id: 'action_001',
            type: 'follow_up',
            description: 'Follow up on proposal feedback',
            priority: 'high',
            due_date: addDays(new Date(), 2).toISOString(),
            assigned_to: 'AI Agent',
            success_criteria: ['Response received', 'Next steps defined']
          }
        ]
      },
      {
        deal_id: 'deal_002',
        contact_id: 'Michael Chen',
        pipeline_id: 'pipeline_001',
        current_stage: 'stage_002',
        deal_value: 1800,
        probability: 40,
        expected_close_date: addDays(new Date(), 21).toISOString(),
        created_date: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
        last_activity: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        conversation_ids: ['conv_003'],
        deal_source: 'Telegram conversation',
        deal_tags: ['small-business', 'price-sensitive'],
        stage_history: [
          {
            stage_id: 'stage_001',
            stage_name: 'Initial Contact',
            entered_at: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
            exited_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
            duration: 2 * 24 * 60 * 60 * 1000,
            conversion_successful: true
          },
          {
            stage_id: 'stage_002',
            stage_name: 'Qualification',
            entered_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
            conversion_successful: false
          }
        ],
        activities: [
          {
            activity_id: 'activity_003',
            type: 'conversation',
            description: 'Discussed business needs and budget constraints',
            timestamp: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
            outcome: 'Budget concerns raised',
            impact_score: 60,
            conversation_id: 'conv_003'
          }
        ],
        revenue_attribution: {
          attributed_conversations: ['conv_003'],
          conversation_impact_scores: { 'conv_003': 60 },
          primary_driver_conversation: 'conv_003',
          revenue_journey: [],
          attribution_confidence: 0.65
        },
        risk_assessment: {
          risk_level: 'high',
          risk_factors: [
            {
              factor: 'Budget constraints',
              impact: 0.5,
              probability: 0.8,
              mitigation_actions: ['Offer starter package', 'Payment plan options']
            }
          ],
          mitigation_strategies: ['Focus on essential features', 'ROI demonstration'],
          probability_adjustment: -20,
          risk_monitoring_alerts: ['Long response times', 'Price objections']
        },
        next_actions: [
          {
            action_id: 'action_002',
            type: 'send_proposal',
            description: 'Send budget-friendly proposal',
            priority: 'medium',
            due_date: addDays(new Date(), 3).toISOString(),
            assigned_to: 'AI Agent',
            success_criteria: ['Proposal sent', 'Budget concerns addressed']
          }
        ]
      }
    ],
    conversion_rates: {
      'stage_001': 0.75,
      'stage_002': 0.60,
      'stage_003': 0.45,
      'stage_004': 0.80,
      'stage_005': 0.95
    },
    avg_stage_duration: {
      'stage_001': 2.5,
      'stage_002': 5.2,
      'stage_003': 7.8,
      'stage_004': 12.1,
      'stage_005': 3.2
    },
    pipeline_velocity: 28.6,
    pipeline_value: 4300,
    pipeline_health_score: 78.5,
    bottlenecks: [
      {
        stage_id: 'stage_003',
        stage_name: 'Proposal',
        bottleneck_type: 'duration',
        severity: 0.6,
        description: 'Deals staying too long in proposal stage',
        affected_deals_count: 3,
        revenue_impact: 12000,
        root_causes: ['Complex pricing discussions', 'Multiple stakeholders'],
        recommended_solutions: ['Simplify pricing tiers', 'Stakeholder mapping']
      }
    ],
    optimization_recommendations: [
      'Automate proposal generation based on conversation insights',
      'Implement risk scoring for early intervention',
      'Create conversation-to-deal progression triggers'
    ]
  });

  const dealsByStage = useMemo(() => {
    const grouped: Record<string, Deal[]> = {};
    pipeline.stages.forEach(stage => {
      grouped[stage.stage_id] = pipeline.deals.filter(deal => deal.current_stage === stage.stage_id);
    });
    return grouped;
  }, [pipeline]);

  const pipelineMetrics = useMemo(() => {
    const totalValue = pipeline.deals.reduce((sum, deal) => sum + deal.deal_value, 0);
    const weightedValue = pipeline.deals.reduce((sum, deal) => sum + (deal.deal_value * deal.probability / 100), 0);
    const averageDealSize = totalValue / pipeline.deals.length || 0;
    const averageCloseTime = pipeline.deals.reduce((sum, deal) => {
      const daysToClose = differenceInDays(new Date(deal.expected_close_date), new Date(deal.created_date));
      return sum + daysToClose;
    }, 0) / pipeline.deals.length || 0;

    return {
      totalDeals: pipeline.deals.length,
      totalValue,
      weightedValue,
      averageDealSize,
      averageCloseTime,
      conversionRate: Object.values(pipeline.conversion_rates).reduce((sum, rate) => sum + rate, 0) / Object.keys(pipeline.conversion_rates).length
    };
  }, [pipeline]);

  const handleDealClick = (deal: Deal) => {
    setSelectedDeal(deal);
    onDealSelect?.(deal);
  };

  const handleDealDrop = useCallback((dealId: string, newStageId: string) => {
    setPipeline(prev => {
      const newPipeline = { ...prev };
      const dealIndex = newPipeline.deals.findIndex(d => d.deal_id === dealId);
      
      if (dealIndex !== -1) {
        const deal = newPipeline.deals[dealIndex];
        const newStage = newPipeline.stages.find(s => s.stage_id === newStageId);
        
        if (newStage && deal.current_stage !== newStageId) {
          // Update deal stage
          newPipeline.deals[dealIndex] = {
            ...deal,
            current_stage: newStageId,
            probability: newStage.probability,
            last_activity: new Date().toISOString(),
            stage_history: [
              ...deal.stage_history,
              {
                stage_id: newStageId,
                stage_name: newStage.name,
                entered_at: new Date().toISOString(),
                conversion_successful: false
              }
            ]
          };
          
          // Close previous stage
          const previousStageIndex = deal.stage_history.length - 1;
          if (previousStageIndex >= 0) {
            newPipeline.deals[dealIndex].stage_history[previousStageIndex] = {
              ...newPipeline.deals[dealIndex].stage_history[previousStageIndex],
              exited_at: new Date().toISOString(),
              conversion_successful: true
            };
          }
          
          onStageChange?.(dealId, newStageId);
        }
      }
      
      return newPipeline;
    });
  }, [onStageChange]);

  const funnelData = useMemo(() => {
    return pipeline.stages.map(stage => ({
      name: stage.name,
      deals: dealsByStage[stage.stage_id]?.length || 0,
      value: dealsByStage[stage.stage_id]?.reduce((sum, deal) => sum + deal.deal_value, 0) || 0,
      probability: stage.probability
    }));
  }, [pipeline.stages, dealsByStage]);

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{pipeline.name}</h2>
            <p className="text-gray-600 mt-1">
              Conversation-driven deal management with AI insights
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* View Mode Toggle */}
            <div className="flex items-center space-x-2 bg-white rounded-lg p-1 border border-gray-200">
              <button
                onClick={() => setCurrentViewMode('kanban')}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  currentViewMode === 'kanban'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                Kanban
              </button>
              <button
                onClick={() => setCurrentViewMode('analytics')}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  currentViewMode === 'analytics'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                Analytics
              </button>
            </div>

            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <PlusIcon className="w-4 h-4" />
              <span>New Deal</span>
            </button>
          </div>
        </div>

        {/* Pipeline Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{pipelineMetrics.totalDeals}</div>
            <div className="text-sm text-gray-600">Total Deals</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              ${pipelineMetrics.totalValue.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Pipeline Value</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">
              ${pipelineMetrics.weightedValue.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Weighted Value</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-orange-600">
              ${pipelineMetrics.averageDealSize.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Avg Deal Size</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-red-600">
              {pipelineMetrics.averageCloseTime.toFixed(0)}d
            </div>
            <div className="text-sm text-gray-600">Avg Close Time</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-indigo-600">
              {(pipelineMetrics.conversionRate * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-gray-600">Conversion Rate</div>
          </Card>
        </div>

        {/* Main Content */}
        {currentViewMode === 'kanban' ? (
          <div className="flex space-x-6 overflow-x-auto pb-4">
            {pipeline.stages.map((stage) => (
              <StageColumn
                key={stage.stage_id}
                stage={stage}
                deals={dealsByStage[stage.stage_id] || []}
                onDealClick={handleDealClick}
                onDealDrop={handleDealDrop}
                dragState={dragState}
                setDragState={setDragState}
              />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Funnel Chart */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Funnel</h3>
              <ResponsiveContainer width="100%" height={300}>
                <FunnelChart>
                  <Funnel
                    dataKey="deals"
                    data={funnelData}
                    isAnimationActive
                    fill="#3B82F6"
                  >
                    <LabelList position="center" />
                  </Funnel>
                  <Tooltip />
                </FunnelChart>
              </ResponsiveContainer>
            </Card>

            {/* Conversion Rates */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Stage Conversion Rates</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={funnelData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis dataKey="name" stroke="#6B7280" fontSize={12} />
                  <YAxis stroke="#6B7280" fontSize={12} />
                  <Tooltip formatter={(value) => [`${value}%`, 'Conversion Rate']} />
                  <Bar dataKey="probability" fill="#10B981" />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Pipeline Health */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Health</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Overall Health Score</span>
                  <span className="text-2xl font-bold text-green-600">
                    {pipeline.pipeline_health_score.toFixed(1)}/100
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-green-500 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${pipeline.pipeline_health_score}%` }}
                  />
                </div>
                
                <div className="mt-6">
                  <h4 className="font-medium text-gray-900 mb-3">Bottlenecks</h4>
                  {pipeline.bottlenecks.map((bottleneck, index) => (
                    <div key={index} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg mb-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-yellow-800">{bottleneck.stage_name}</span>
                        <span className="text-sm text-yellow-700">
                          Severity: {(bottleneck.severity * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-sm text-yellow-700">{bottleneck.description}</p>
                      <div className="mt-2 text-xs text-yellow-600">
                        Affected deals: {bottleneck.affected_deals_count} | 
                        Revenue impact: ${bottleneck.revenue_impact.toLocaleString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>

            {/* Optimization Recommendations */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Recommendations</h3>
              <div className="space-y-3">
                {pipeline.optimization_recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                    <BoltIcon className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <span className="text-sm text-blue-800">{recommendation}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* Deal Detail Modal */}
        <AnimatePresence>
          {selectedDeal && (
            <DealDetailModal
              deal={selectedDeal}
              isOpen={!!selectedDeal}
              onClose={() => setSelectedDeal(null)}
              onUpdate={(updatedDeal) => {
                setPipeline(prev => ({
                  ...prev,
                  deals: prev.deals.map(d => d.deal_id === updatedDeal.deal_id ? updatedDeal : d)
                }));
                setSelectedDeal(updatedDeal);
              }}
            />
          )}
        </AnimatePresence>
      </div>
    </ErrorBoundary>
  );
};