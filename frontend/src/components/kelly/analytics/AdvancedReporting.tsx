/**
 * Advanced Reporting System - Automated insight generation and custom reports
 * AI-generated recommendations, anomaly detection, and predictive analytics
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '../../ui/Card';
import { LoadingSpinner } from '../../ui/LoadingSpinner';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine
} from 'recharts';
import {
  DocumentChartBarIcon,
  SparklesIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  CpuChipIcon,
  ChartBarIcon,
  DocumentTextIcon,
  CalendarIcon,
  FunnelIcon,
  Cog6ToothIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  EyeIcon,
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  BoltIcon,
  FireIcon,
  ShieldCheckIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { format, subDays, subHours, addDays } from 'date-fns';
import {
  ReportTemplate,
  ReportMetric,
  ReportVisualization,
  AdvancedInsight,
  InsightRecommendation,
  PredictedOutcome,
  RootCauseAnalysis
} from '../../../types/analytics';

interface AdvancedReportingProps {
  className?: string;
  reportTemplateId?: string;
  viewMode?: 'dashboard' | 'builder' | 'insights' | 'schedule';
  autoGenerate?: boolean;
}

interface ReportBuilderState {
  template: Partial<ReportTemplate>;
  selectedMetrics: ReportMetric[];
  selectedVisualizations: ReportVisualization[];
  previewData: any[];
}

interface InsightCard {
  insight: AdvancedInsight;
  onExpand: () => void;
  onImplement: (recommendationId: string) => void;
  onDismiss: () => void;
}

const InsightCard: React.FC<InsightCard> = ({ insight, onExpand, onImplement, onDismiss }) => {
  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'border-red-500 bg-red-50';
      case 'high': return 'border-orange-500 bg-orange-50';
      case 'medium': return 'border-yellow-500 bg-yellow-50';
      default: return 'border-blue-500 bg-blue-50';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'trend': return TrendingUpIcon;
      case 'anomaly': return ExclamationTriangleIcon;
      case 'prediction': return CpuChipIcon;
      case 'recommendation': return LightBulbIcon;
      case 'alert': return FireIcon;
      default: return SparklesIcon;
    }
  };

  const Icon = getTypeIcon(insight.type);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`border-l-4 rounded-lg p-4 ${getUrgencyColor(insight.urgency)}`}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <Icon className={`w-6 h-6 ${
            insight.urgency === 'critical' ? 'text-red-600' :
            insight.urgency === 'high' ? 'text-orange-600' :
            insight.urgency === 'medium' ? 'text-yellow-600' :
            'text-blue-600'
          }`} />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-900 mb-1">{insight.title}</h3>
              <p className="text-gray-700 mb-3">{insight.description}</p>
              
              <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                <span>Confidence: {(insight.confidence_score * 100).toFixed(0)}%</span>
                <span>Impact: {insight.business_impact.toFixed(1)}/10</span>
                <span className="capitalize">{insight.type}</span>
              </div>
              
              {insight.affected_metrics.length > 0 && (
                <div className="mb-3">
                  <span className="text-sm font-medium text-gray-700">Affected Metrics:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {insight.affected_metrics.map((metric, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-white text-gray-800 border border-gray-200"
                      >
                        {metric}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={onExpand}
                className="p-2 text-gray-400 hover:text-blue-600 hover:bg-white rounded-lg transition-colors"
              >
                <EyeIcon className="w-4 h-4" />
              </button>
              <button
                onClick={onDismiss}
                className="p-2 text-gray-400 hover:text-red-600 hover:bg-white rounded-lg transition-colors"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          {insight.recommendations.length > 0 && (
            <div className="space-y-2">
              {insight.recommendations.slice(0, 2).map((rec, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{rec.title}</h4>
                    <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                    <div className="flex items-center space-x-3 mt-2 text-xs text-gray-500">
                      <span>Impact: +{(rec.expected_impact * 100).toFixed(0)}%</span>
                      <span className="capitalize">Effort: {rec.implementation_effort}</span>
                      <span className="capitalize">Priority: {rec.priority}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => onImplement(rec.recommendation_id)}
                    className="ml-3 px-3 py-1 bg-blue-600 text-white text-sm font-medium rounded hover:bg-blue-700 transition-colors"
                  >
                    Implement
                  </button>
                </div>
              ))}
              
              {insight.recommendations.length > 2 && (
                <button
                  onClick={onExpand}
                  className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                >
                  View {insight.recommendations.length - 2} more recommendations
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

const ReportBuilder: React.FC<{
  onSave: (template: ReportTemplate) => void;
  onCancel: () => void;
  existingTemplate?: ReportTemplate;
}> = ({ onSave, onCancel, existingTemplate }) => {
  const [builderState, setBuilderState] = useState<ReportBuilderState>({
    template: existingTemplate || {
      name: '',
      description: '',
      report_type: 'dashboard',
      frequency: 'daily',
      metrics: [],
      visualizations: [],
      automated_insights: true
    },
    selectedMetrics: [],
    selectedVisualizations: [],
    previewData: []
  });

  const availableMetrics = [
    { metric_id: 'conversations_total', metric_name: 'Total Conversations', metric_type: 'count' },
    { metric_id: 'conversion_rate', metric_name: 'Conversion Rate', metric_type: 'percentage' },
    { metric_id: 'revenue_total', metric_name: 'Total Revenue', metric_type: 'currency' },
    { metric_id: 'ai_confidence_avg', metric_name: 'Average AI Confidence', metric_type: 'percentage' },
    { metric_id: 'response_time_avg', metric_name: 'Average Response Time', metric_type: 'duration' },
    { metric_id: 'satisfaction_score', metric_name: 'Customer Satisfaction', metric_type: 'number' }
  ];

  const availableVisualizations = [
    { viz_id: 'line_chart', type: 'line_chart', title: 'Trend Line Chart' },
    { viz_id: 'bar_chart', type: 'bar_chart', title: 'Bar Chart' },
    { viz_id: 'area_chart', type: 'area_chart', title: 'Area Chart' },
    { viz_id: 'pie_chart', type: 'pie_chart', title: 'Pie Chart' },
    { viz_id: 'gauge', type: 'gauge', title: 'Gauge Chart' },
    { viz_id: 'table', type: 'table', title: 'Data Table' }
  ];

  const handleSave = () => {
    const template: ReportTemplate = {
      template_id: existingTemplate?.template_id || `template_${Date.now()}`,
      name: builderState.template.name || 'Untitled Report',
      description: builderState.template.description || '',
      report_type: builderState.template.report_type || 'dashboard',
      frequency: builderState.template.frequency || 'daily',
      metrics: builderState.selectedMetrics.map(metric => ({
        ...metric,
        aggregation: 'sum',
        data_source: 'kelly_analytics',
        calculation_method: 'real_time',
        display_format: 'number',
        thresholds: []
      })),
      filters: [],
      visualizations: builderState.selectedVisualizations.map(viz => ({
        ...viz,
        metrics: builderState.selectedMetrics.map(m => m.metric_id),
        dimensions: ['time'],
        configuration: {},
        interactive: true,
        drill_down_enabled: false
      })),
      automated_insights: builderState.template.automated_insights || false,
      distribution_list: [],
      custom_fields: []
    };
    
    onSave(template);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Report Builder</h3>
        <div className="flex items-center space-x-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Save Report
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Panel */}
        <div className="space-y-6">
          <Card className="p-6">
            <h4 className="font-medium text-gray-900 mb-4">Report Configuration</h4>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Report Name
                </label>
                <input
                  type="text"
                  value={builderState.template.name || ''}
                  onChange={(e) => setBuilderState(prev => ({
                    ...prev,
                    template: { ...prev.template, name: e.target.value }
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter report name"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={builderState.template.description || ''}
                  onChange={(e) => setBuilderState(prev => ({
                    ...prev,
                    template: { ...prev.template, description: e.target.value }
                  }))}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter report description"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Report Type
                  </label>
                  <select
                    value={builderState.template.report_type || 'dashboard'}
                    onChange={(e) => setBuilderState(prev => ({
                      ...prev,
                      template: { ...prev.template, report_type: e.target.value as any }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="dashboard">Dashboard</option>
                    <option value="detailed">Detailed Report</option>
                    <option value="executive">Executive Summary</option>
                    <option value="operational">Operational Report</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Frequency
                  </label>
                  <select
                    value={builderState.template.frequency || 'daily'}
                    onChange={(e) => setBuilderState(prev => ({
                      ...prev,
                      template: { ...prev.template, frequency: e.target.value as any }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="real_time">Real-time</option>
                    <option value="hourly">Hourly</option>
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                  </select>
                </div>
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="automated_insights"
                  checked={builderState.template.automated_insights || false}
                  onChange={(e) => setBuilderState(prev => ({
                    ...prev,
                    template: { ...prev.template, automated_insights: e.target.checked }
                  }))}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="automated_insights" className="ml-2 block text-sm text-gray-900">
                  Enable automated insights and recommendations
                </label>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h4 className="font-medium text-gray-900 mb-4">Select Metrics</h4>
            <div className="space-y-3">
              {availableMetrics.map((metric) => (
                <div key={metric.metric_id} className="flex items-center">
                  <input
                    type="checkbox"
                    id={metric.metric_id}
                    checked={builderState.selectedMetrics.some(m => m.metric_id === metric.metric_id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setBuilderState(prev => ({
                          ...prev,
                          selectedMetrics: [...prev.selectedMetrics, metric as ReportMetric]
                        }));
                      } else {
                        setBuilderState(prev => ({
                          ...prev,
                          selectedMetrics: prev.selectedMetrics.filter(m => m.metric_id !== metric.metric_id)
                        }));
                      }
                    }}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label htmlFor={metric.metric_id} className="ml-3 block text-sm text-gray-900">
                    {metric.metric_name}
                    <span className="text-gray-500 text-xs ml-2">({metric.metric_type})</span>
                  </label>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h4 className="font-medium text-gray-900 mb-4">Select Visualizations</h4>
            <div className="grid grid-cols-2 gap-3">
              {availableVisualizations.map((viz) => (
                <div
                  key={viz.viz_id}
                  className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                    builderState.selectedVisualizations.some(v => v.viz_id === viz.viz_id)
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                  onClick={() => {
                    const isSelected = builderState.selectedVisualizations.some(v => v.viz_id === viz.viz_id);
                    if (isSelected) {
                      setBuilderState(prev => ({
                        ...prev,
                        selectedVisualizations: prev.selectedVisualizations.filter(v => v.viz_id !== viz.viz_id)
                      }));
                    } else {
                      setBuilderState(prev => ({
                        ...prev,
                        selectedVisualizations: [...prev.selectedVisualizations, viz as ReportVisualization]
                      }));
                    }
                  }}
                >
                  <ChartBarIcon className="w-6 h-6 text-gray-600 mb-2" />
                  <div className="text-sm font-medium text-gray-900">{viz.title}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Preview Panel */}
        <div className="space-y-6">
          <Card className="p-6">
            <h4 className="font-medium text-gray-900 mb-4">Report Preview</h4>
            <div className="space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <h5 className="font-medium text-gray-900">
                  {builderState.template.name || 'Untitled Report'}
                </h5>
                <p className="text-sm text-gray-600 mt-1">
                  {builderState.template.description || 'No description provided'}
                </p>
                <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                  <span className="capitalize">{builderState.template.report_type}</span>
                  <span className="capitalize">{builderState.template.frequency}</span>
                  <span>{builderState.selectedMetrics.length} metrics</span>
                  <span>{builderState.selectedVisualizations.length} charts</span>
                </div>
              </div>
              
              {builderState.selectedMetrics.length > 0 && (
                <div>
                  <h6 className="font-medium text-gray-700 mb-2">Selected Metrics</h6>
                  <div className="flex flex-wrap gap-2">
                    {builderState.selectedMetrics.map((metric) => (
                      <span
                        key={metric.metric_id}
                        className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                      >
                        {metric.metric_name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {builderState.selectedVisualizations.length > 0 && (
                <div>
                  <h6 className="font-medium text-gray-700 mb-2">Selected Visualizations</h6>
                  <div className="flex flex-wrap gap-2">
                    {builderState.selectedVisualizations.map((viz) => (
                      <span
                        key={viz.viz_id}
                        className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800"
                      >
                        {viz.title}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export const AdvancedReporting: React.FC<AdvancedReportingProps> = ({
  className = '',
  reportTemplateId,
  viewMode = 'dashboard',
  autoGenerate = true
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);
  const [showReportBuilder, setShowReportBuilder] = useState(false);
  const [selectedInsight, setSelectedInsight] = useState<AdvancedInsight | null>(null);

  // Mock data
  const [reportTemplates, setReportTemplates] = useState<ReportTemplate[]>([
    {
      template_id: 'template_001',
      name: 'Executive Dashboard',
      description: 'High-level KPIs and insights for executive team',
      report_type: 'executive',
      frequency: 'daily',
      metrics: [],
      filters: [],
      visualizations: [],
      automated_insights: true,
      distribution_list: ['exec@company.com'],
      custom_fields: []
    },
    {
      template_id: 'template_002',
      name: 'Operational Performance Report',
      description: 'Detailed operational metrics and performance indicators',
      report_type: 'operational',
      frequency: 'hourly',
      metrics: [],
      filters: [],
      visualizations: [],
      automated_insights: true,
      distribution_list: ['ops@company.com'],
      custom_fields: []
    }
  ]);

  const [insights, setInsights] = useState<AdvancedInsight[]>([
    {
      insight_id: 'insight_001',
      type: 'trend',
      title: 'Conversation Quality Improvement Detected',
      description: 'AI conversation quality scores have increased by 15% over the past week, driven by enhanced context understanding and more personalized responses.',
      confidence_score: 0.92,
      business_impact: 8.7,
      urgency: 'medium',
      affected_metrics: ['ai_confidence_avg', 'conversation_quality', 'customer_satisfaction'],
      root_cause_analysis: {
        primary_causes: [
          {
            cause_id: 'cause_001',
            description: 'Model fine-tuning improvements',
            probability: 0.8,
            impact_score: 0.9,
            evidence: ['Recent model updates', 'Training data quality improvements'],
            validation_method: 'A/B testing results'
          }
        ],
        contributing_factors: [],
        correlation_analysis: [],
        external_factors: [],
        confidence_in_analysis: 0.87
      },
      recommendations: [
        {
          recommendation_id: 'rec_001',
          title: 'Expand model improvements to all conversation types',
          description: 'Apply the successful fine-tuning approach to other conversation categories to maintain quality gains.',
          priority: 'high',
          implementation_effort: 'medium',
          expected_impact: 0.25,
          success_metrics: ['Consistent quality scores across all conversation types', '20% reduction in human interventions'],
          implementation_steps: ['Analyze successful patterns', 'Create training data for other categories', 'Deploy incremental updates'],
          resource_requirements: ['2 ML engineers', '1 week development time'],
          timeline_estimate: '2-3 weeks',
          risk_assessment: ['Temporary dip in performance during transition']
        },
        {
          recommendation_id: 'rec_002',
          title: 'Implement real-time quality monitoring',
          description: 'Set up automated monitoring to detect and maintain quality improvements proactively.',
          priority: 'medium',
          implementation_effort: 'low',
          expected_impact: 0.15,
          success_metrics: ['Early detection of quality degradation', 'Automated alerts for anomalies'],
          implementation_steps: ['Deploy monitoring infrastructure', 'Configure alerting thresholds', 'Create response protocols'],
          resource_requirements: ['1 DevOps engineer', '3 days setup time'],
          timeline_estimate: '1 week',
          risk_assessment: ['False positive alerts initially']
        }
      ],
      predicted_outcomes: [
        {
          scenario: 'Recommendations implemented',
          probability: 0.8,
          impact_description: 'Quality improvements sustained and expanded across all conversation types',
          timeline: '4-6 weeks',
          confidence_interval: [0.75, 0.85],
          assumptions: ['No major model architecture changes', 'Continued training data quality'],
          monitoring_indicators: ['Quality score trends', 'Customer satisfaction metrics']
        }
      ],
      supporting_data: {
        quality_trend: [85.2, 87.1, 89.3, 91.7, 93.4],
        confidence_scores: [0.89, 0.91, 0.93, 0.95, 0.97],
        customer_feedback: 'positive'
      },
      generated_at: new Date().toISOString(),
      valid_until: addDays(new Date(), 7).toISOString()
    },
    {
      insight_id: 'insight_002',
      type: 'anomaly',
      title: 'Unusual Drop in Response Time Performance',
      description: 'Average response times have increased by 40% in the last 4 hours, indicating potential system performance issues.',
      confidence_score: 0.95,
      business_impact: 7.2,
      urgency: 'critical',
      affected_metrics: ['response_time_avg', 'system_performance', 'user_satisfaction'],
      root_cause_analysis: {
        primary_causes: [
          {
            cause_id: 'cause_002',
            description: 'Database query performance degradation',
            probability: 0.7,
            impact_score: 0.8,
            evidence: ['Slow query logs', 'Database connection timeouts'],
            validation_method: 'Database performance monitoring'
          }
        ],
        contributing_factors: [],
        correlation_analysis: [],
        external_factors: [],
        confidence_in_analysis: 0.82
      },
      recommendations: [
        {
          recommendation_id: 'rec_003',
          title: 'Immediate database optimization',
          description: 'Optimize slow queries and add missing indexes to restore performance.',
          priority: 'critical',
          implementation_effort: 'low',
          expected_impact: 0.8,
          success_metrics: ['Response times under 2 seconds', 'Database query optimization'],
          implementation_steps: ['Identify slow queries', 'Add database indexes', 'Optimize query patterns'],
          resource_requirements: ['1 database administrator', 'Immediate action required'],
          timeline_estimate: '2-4 hours',
          risk_assessment: ['Brief downtime during index creation']
        }
      ],
      predicted_outcomes: [
        {
          scenario: 'Quick optimization implemented',
          probability: 0.9,
          impact_description: 'Response times return to normal within 4 hours',
          timeline: '2-4 hours',
          confidence_interval: [0.85, 0.95],
          assumptions: ['Issue is database-related', 'No hardware failures'],
          monitoring_indicators: ['Response time metrics', 'Database performance counters']
        }
      ],
      supporting_data: {
        response_time_trend: [1.2, 1.3, 1.8, 2.4, 2.8],
        database_metrics: { cpu_usage: 85, connection_count: 247, slow_queries: 23 }
      },
      generated_at: new Date().toISOString(),
      valid_until: addHours(new Date(), 12).toISOString()
    },
    {
      insight_id: 'insight_003',
      type: 'prediction',
      title: 'Revenue Growth Forecast Exceeds Targets',
      description: 'Based on current conversation trends and conversion patterns, projected revenue for next month is 23% above target.',
      confidence_score: 0.88,
      business_impact: 9.1,
      urgency: 'medium',
      affected_metrics: ['revenue_total', 'conversion_rate', 'pipeline_value'],
      root_cause_analysis: {
        primary_causes: [
          {
            cause_id: 'cause_003',
            description: 'Improved conversation quality driving higher conversions',
            probability: 0.85,
            impact_score: 0.9,
            evidence: ['Higher conversion rates', 'Better customer engagement scores'],
            validation_method: 'Correlation analysis'
          }
        ],
        contributing_factors: [],
        correlation_analysis: [],
        external_factors: [],
        confidence_in_analysis: 0.91
      },
      recommendations: [
        {
          recommendation_id: 'rec_004',
          title: 'Scale successful conversation patterns',
          description: 'Identify and replicate the most successful conversation patterns to maximize revenue potential.',
          priority: 'high',
          implementation_effort: 'medium',
          expected_impact: 0.35,
          success_metrics: ['Consistent high conversion rates', 'Scaled successful patterns across all agents'],
          implementation_steps: ['Analyze top-performing patterns', 'Create training materials', 'Deploy pattern recognition'],
          resource_requirements: ['2 data analysts', '1 ML engineer', '2 weeks analysis time'],
          timeline_estimate: '3-4 weeks',
          risk_assessment: ['Over-optimization risk', 'Pattern staleness over time']
        }
      ],
      predicted_outcomes: [
        {
          scenario: 'Current trends continue',
          probability: 0.75,
          impact_description: 'Revenue exceeds target by 20-25% next month',
          timeline: '30 days',
          confidence_interval: [0.70, 0.80],
          assumptions: ['No major market changes', 'Continued AI performance'],
          monitoring_indicators: ['Daily revenue tracking', 'Conversion rate trends']
        }
      ],
      supporting_data: {
        revenue_forecast: [125000, 135000, 147000, 162000, 178000],
        conversion_trends: [0.67, 0.71, 0.74, 0.78, 0.82]
      },
      generated_at: new Date().toISOString(),
      valid_until: addDays(new Date(), 30).toISOString()
    }
  ]);

  const handleImplementRecommendation = (recommendationId: string) => {
    console.log('Implementing recommendation:', recommendationId);
    // Implementation logic here
  };

  const handleDismissInsight = (insightId: string) => {
    setInsights(prev => prev.filter(insight => insight.insight_id !== insightId));
  };

  const handleSaveReport = (template: ReportTemplate) => {
    setReportTemplates(prev => {
      const existing = prev.find(t => t.template_id === template.template_id);
      if (existing) {
        return prev.map(t => t.template_id === template.template_id ? template : t);
      } else {
        return [...prev, template];
      }
    });
    setShowReportBuilder(false);
  };

  const viewModes = [
    { id: 'dashboard', label: 'Dashboard', icon: DocumentChartBarIcon },
    { id: 'insights', label: 'AI Insights', icon: SparklesIcon },
    { id: 'builder', label: 'Report Builder', icon: Cog6ToothIcon },
    { id: 'schedule', label: 'Scheduled Reports', icon: CalendarIcon }
  ];

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Advanced Reporting</h2>
            <p className="text-gray-600 mt-1">
              AI-powered insights, custom reports, and automated analytics
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

            {currentViewMode === 'builder' && (
              <button
                onClick={() => setShowReportBuilder(true)}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <PlusIcon className="w-4 h-4" />
                <span>New Report</span>
              </button>
            )}
          </div>
        </div>

        {/* Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentViewMode}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {currentViewMode === 'insights' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">AI-Generated Insights</h3>
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <SparklesIcon className="w-4 h-4" />
                    <span>Auto-refreshing every 15 minutes</span>
                  </div>
                </div>

                <div className="space-y-4">
                  {insights.map((insight) => (
                    <InsightCard
                      key={insight.insight_id}
                      insight={insight}
                      onExpand={() => setSelectedInsight(insight)}
                      onImplement={handleImplementRecommendation}
                      onDismiss={() => handleDismissInsight(insight.insight_id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {currentViewMode === 'builder' && (
              <div>
                {showReportBuilder ? (
                  <ReportBuilder
                    onSave={handleSaveReport}
                    onCancel={() => setShowReportBuilder(false)}
                  />
                ) : (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-gray-900">Report Templates</h3>
                      <button
                        onClick={() => setShowReportBuilder(true)}
                        className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        <PlusIcon className="w-4 h-4" />
                        <span>Create Report</span>
                      </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {reportTemplates.map((template) => (
                        <Card key={template.template_id} className="p-6 hover:shadow-md transition-shadow">
                          <div className="flex items-start justify-between mb-4">
                            <div>
                              <h4 className="font-semibold text-gray-900">{template.name}</h4>
                              <p className="text-sm text-gray-600 mt-1">{template.description}</p>
                            </div>
                            <div className="flex items-center space-x-1">
                              <button className="p-1 text-gray-400 hover:text-blue-600 rounded">
                                <PencilIcon className="w-4 h-4" />
                              </button>
                              <button className="p-1 text-gray-400 hover:text-red-600 rounded">
                                <TrashIcon className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                          
                          <div className="space-y-3">
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">Type:</span>
                              <span className="capitalize font-medium">{template.report_type}</span>
                            </div>
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">Frequency:</span>
                              <span className="capitalize font-medium">{template.frequency}</span>
                            </div>
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">AI Insights:</span>
                              <span className={`font-medium ${template.automated_insights ? 'text-green-600' : 'text-gray-600'}`}>
                                {template.automated_insights ? 'Enabled' : 'Disabled'}
                              </span>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2 mt-4 pt-4 border-t border-gray-200">
                            <button className="flex-1 px-3 py-2 text-sm font-medium text-blue-600 border border-blue-600 rounded hover:bg-blue-50 transition-colors">
                              Generate Report
                            </button>
                            <button className="p-2 text-gray-400 hover:text-gray-600 border border-gray-300 rounded hover:bg-gray-50 transition-colors">
                              <ArrowDownTrayIcon className="w-4 h-4" />
                            </button>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {currentViewMode === 'dashboard' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Quick Metrics */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">System Performance</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Reports Generated Today</span>
                      <span className="text-2xl font-bold text-blue-600">127</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">AI Insights Created</span>
                      <span className="text-2xl font-bold text-green-600">23</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Automated Actions</span>
                      <span className="text-2xl font-bold text-purple-600">45</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Data Processing Speed</span>
                      <span className="text-2xl font-bold text-orange-600">847ms</span>
                    </div>
                  </div>
                </Card>

                {/* Recent Insights */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Insights</h3>
                  <div className="space-y-3">
                    {insights.slice(0, 3).map((insight) => (
                      <div key={insight.insight_id} className="p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 text-sm">{insight.title}</h4>
                            <p className="text-xs text-gray-600 mt-1 line-clamp-2">{insight.description}</p>
                            <div className="flex items-center space-x-3 mt-2 text-xs text-gray-500">
                              <span>Confidence: {(insight.confidence_score * 100).toFixed(0)}%</span>
                              <span className="capitalize">{insight.urgency}</span>
                            </div>
                          </div>
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            insight.type === 'trend' ? 'bg-blue-100 text-blue-800' :
                            insight.type === 'anomaly' ? 'bg-red-100 text-red-800' :
                            insight.type === 'prediction' ? 'bg-purple-100 text-purple-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {insight.type}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {currentViewMode === 'schedule' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Scheduled Reports</h3>
                <div className="bg-gray-50 rounded-lg p-8 text-center">
                  <CalendarIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">Scheduled reporting interface would be implemented here</p>
                  <p className="text-sm text-gray-500 mt-2">
                    Manage automated report generation and distribution schedules
                  </p>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Insight Detail Modal */}
        {selectedInsight && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
            >
              <div className="flex items-center justify-between p-6 border-b border-gray-200">
                <h3 className="text-xl font-bold text-gray-900">{selectedInsight.title}</h3>
                <button
                  onClick={() => setSelectedInsight(null)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <XCircleIcon className="w-6 h-6" />
                </button>
              </div>
              
              <div className="p-6 max-h-96 overflow-y-auto">
                <div className="space-y-6">
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Analysis</h4>
                    <p className="text-gray-700">{selectedInsight.description}</p>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-blue-600">
                        {(selectedInsight.confidence_score * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">Confidence</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-green-600">
                        {selectedInsight.business_impact.toFixed(1)}/10
                      </div>
                      <div className="text-sm text-gray-600">Business Impact</div>
                    </div>
                    <div>
                      <div className={`text-2xl font-bold capitalize ${
                        selectedInsight.urgency === 'critical' ? 'text-red-600' :
                        selectedInsight.urgency === 'high' ? 'text-orange-600' :
                        selectedInsight.urgency === 'medium' ? 'text-yellow-600' :
                        'text-blue-600'
                      }`}>
                        {selectedInsight.urgency}
                      </div>
                      <div className="text-sm text-gray-600">Urgency</div>
                    </div>
                  </div>
                  
                  {selectedInsight.root_cause_analysis.primary_causes.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Root Cause Analysis</h4>
                      <div className="space-y-3">
                        {selectedInsight.root_cause_analysis.primary_causes.map((cause, index) => (
                          <div key={index} className="p-4 bg-gray-50 rounded-lg">
                            <h5 className="font-medium text-gray-900">{cause.description}</h5>
                            <div className="grid grid-cols-2 gap-4 mt-2 text-sm text-gray-600">
                              <div>Probability: {(cause.probability * 100).toFixed(0)}%</div>
                              <div>Impact: {(cause.impact_score * 100).toFixed(0)}%</div>
                            </div>
                            <div className="mt-2">
                              <span className="text-sm font-medium text-gray-700">Evidence:</span>
                              <ul className="text-sm text-gray-600 mt-1 space-y-1">
                                {cause.evidence.map((evidence, evidenceIndex) => (
                                  <li key={evidenceIndex} className="flex items-center space-x-2">
                                    <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                                    <span>{evidence}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
                    <div className="space-y-4">
                      {selectedInsight.recommendations.map((rec, index) => (
                        <div key={index} className="p-4 border border-gray-200 rounded-lg">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h5 className="font-medium text-gray-900">{rec.title}</h5>
                              <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                              <div className="grid grid-cols-3 gap-4 mt-3 text-sm text-gray-600">
                                <div>Impact: +{(rec.expected_impact * 100).toFixed(0)}%</div>
                                <div className="capitalize">Effort: {rec.implementation_effort}</div>
                                <div className="capitalize">Priority: {rec.priority}</div>
                              </div>
                            </div>
                            <button
                              onClick={() => handleImplementRecommendation(rec.recommendation_id)}
                              className="ml-4 px-3 py-1 bg-blue-600 text-white text-sm font-medium rounded hover:bg-blue-700 transition-colors"
                            >
                              Implement
                            </button>
                          </div>
                        </div>
                      ))}
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