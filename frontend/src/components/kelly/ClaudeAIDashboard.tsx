/**
 * Claude AI Integration Dashboard
 * Real-time monitoring and control of Claude AI features for Kelly
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  CpuChipIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  CogIcon,
} from '@heroicons/react/24/outline';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { useKellyStore } from '@/store';
import { apiClient } from '@/lib/api';
import { useClaudeMetrics, useClaudeResponseGeneration } from '@/lib/websocket';
import type { ClaudeUsageMetrics, ClaudeIntegrationConfig } from '@/types/kelly';

interface ClaudeAIDashboardProps {
  accountId?: string;
  className?: string;
}

export default function ClaudeAIDashboard({ accountId, className = '' }: ClaudeAIDashboardProps) {
  const { selectedAccount } = useKellyStore();
  const currentAccountId = accountId || selectedAccount?.id || '';

  const [metrics, setMetrics] = useState<ClaudeUsageMetrics | null>(null);
  const [config, setConfig] = useState<ClaudeIntegrationConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeGeneration, setActiveGeneration] = useState<any>(null);

  // Real-time metrics updates
  useClaudeMetrics((update) => {
    if (update.payload) {
      setMetrics(prev => ({ ...prev, ...update.payload }));
    }
  });

  // Real-time response generation updates
  useClaudeResponseGeneration('', (update) => {
    setActiveGeneration(update.payload);
  });

  // Load initial data
  useEffect(() => {
    if (!currentAccountId) return;

    const loadData = async () => {
      try {
        setIsLoading(true);
        const [metricsData, configData] = await Promise.all([
          apiClient.getClaudeUsageMetrics(currentAccountId),
          apiClient.getClaudeConfig(currentAccountId),
        ]);
        setMetrics(metricsData);
        setConfig(configData);
      } catch (error) {
        console.error('Failed to load Claude AI data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [currentAccountId]);

  const handleModelChange = async (model: 'opus' | 'sonnet' | 'haiku') => {
    if (!config || !currentAccountId) return;

    try {
      const updatedConfig = await apiClient.updateClaudeConfig(currentAccountId, {
        model_selection: model
      });
      setConfig(updatedConfig);
    } catch (error) {
      console.error('Failed to update Claude model:', error);
    }
  };

  const handleTemperatureChange = async (temperature: number) => {
    if (!config || !currentAccountId) return;

    try {
      const updatedConfig = await apiClient.updateClaudeConfig(currentAccountId, {
        temperature
      });
      setConfig(updatedConfig);
    } catch (error) {
      console.error('Failed to update Claude temperature:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading Claude AI dashboard..." />
      </div>
    );
  }

  if (!metrics || !config) {
    return (
      <Card className="p-6">
        <div className="text-center text-text-secondary">
          <CpuChipIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Claude AI integration not available</p>
        </div>
      </Card>
    );
  }

  const modelColors = {
    opus: 'text-purple-600 bg-purple-100',
    sonnet: 'text-blue-600 bg-blue-100',
    haiku: 'text-green-600 bg-green-100',
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Real-time Status */}
      <Card variant="consciousness" className="overflow-hidden">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-lg bg-consciousness-primary/10 flex items-center justify-center">
                <CpuChipIcon className="w-6 h-6 text-consciousness-primary" />
              </div>
              <div>
                <CardTitle className="text-xl">Claude AI Integration</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  {config.enabled ? 'Active' : 'Inactive'} • Model: {config.model_selection}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {activeGeneration && (
                <div className="flex items-center space-x-2 text-sm text-consciousness-primary">
                  <div className="w-2 h-2 bg-consciousness-primary rounded-full animate-pulse"></div>
                  <span>Generating response...</span>
                </div>
              )}
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                config.enabled 
                  ? 'bg-states-flow/20 text-states-flow'
                  : 'bg-states-stress/20 text-states-stress'
              }`}>
                {config.enabled ? 'Enabled' : 'Disabled'}
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Real-time Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Daily Usage */}
        <Card className="group hover:shadow-elevated transition-all duration-300">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                <ChartBarIcon className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <CardTitle className="text-lg">Daily Usage</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  {metrics.total_tokens_used_today.toLocaleString()} tokens
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-caption-text">
                <span>Requests:</span>
                <span className="font-medium">
                  {Object.values(metrics.requests_by_model).reduce((a, b) => a + b, 0)}
                </span>
              </div>
              <div className="flex justify-between text-caption-text">
                <span>Success Rate:</span>
                <span className="font-medium">{Math.round(metrics.success_rate * 100)}%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Cost Tracking */}
        <Card className="group hover:shadow-elevated transition-all duration-300">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center">
                <CurrencyDollarIcon className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <CardTitle className="text-lg">Daily Cost</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  ${metrics.total_cost_today.toFixed(4)}
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-caption-text">
                <span>Budget:</span>
                <span className="font-medium">${config.cost_limit_daily}</span>
              </div>
              <div className="w-full bg-surface-secondary rounded-full h-2">
                <motion.div
                  className="bg-green-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ 
                    width: `${Math.min((metrics.total_cost_today / config.cost_limit_daily) * 100, 100)}%` 
                  }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Response Time */}
        <Card className="group hover:shadow-elevated transition-all duration-300">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-yellow-100 flex items-center justify-center">
                <ClockIcon className="w-5 h-5 text-yellow-600" />
              </div>
              <div>
                <CardTitle className="text-lg">Response Time</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  {Math.round(metrics.average_response_time)}ms avg
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-caption-text">Status:</span>
                {metrics.average_response_time < 2000 ? (
                  <div className="flex items-center space-x-1 text-states-flow">
                    <CheckCircleIcon className="w-4 h-4" />
                    <span className="text-caption-text">Fast</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-1 text-states-stress">
                    <ExclamationTriangleIcon className="w-4 h-4" />
                    <span className="text-caption-text">Slow</span>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* AI Confidence */}
        <Card className="group hover:shadow-elevated transition-all duration-300">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-purple-100 flex items-center justify-center">
                <LightBulbIcon className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <CardTitle className="text-lg">AI Confidence</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  {Math.round(
                    Object.values(metrics.model_performance).reduce(
                      (sum, model) => sum + model.avg_confidence, 0
                    ) / Object.keys(metrics.model_performance).length * 100
                  )}% avg
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(metrics.model_performance).map(([model, perf]) => (
                <div key={model} className="flex justify-between text-caption-text">
                  <span className="capitalize">{model}:</span>
                  <span className="font-medium">{Math.round(perf.avg_confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Selection and Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Selection */}
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-3">
              <CpuChipIcon className="w-6 h-6 text-consciousness-primary" />
              <CardTitle>Model Selection</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                {(['opus', 'sonnet', 'haiku'] as const).map((model) => (
                  <Button
                    key={model}
                    variant={config.model_selection === model ? 'primary' : 'outline'}
                    size="sm"
                    onClick={() => handleModelChange(model)}
                    className={`${modelColors[model]} ${
                      config.model_selection === model ? 'ring-2 ring-offset-2' : ''
                    }`}
                  >
                    <div className="text-center">
                      <div className="font-medium capitalize">{model}</div>
                      <div className="text-xs opacity-75">
                        {model === 'opus' && 'Most Capable'}
                        {model === 'sonnet' && 'Balanced'}
                        {model === 'haiku' && 'Fastest'}
                      </div>
                    </div>
                  </Button>
                ))}
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">Creativity Level</label>
                  <span className="text-sm text-text-secondary">{config.temperature}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={config.temperature}
                  onChange={(e) => handleTemperatureChange(Number(e.target.value))}
                  className="w-full h-2 bg-surface-secondary rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-text-tertiary">
                  <span>Conservative</span>
                  <span>Creative</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* AI Features Status */}
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-3">
              <CogIcon className="w-6 h-6 text-consciousness-primary" />
              <CardTitle>AI Features</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Personality Matching</span>
                <div className={`w-2 h-2 rounded-full ${
                  config.response_optimization.personality_matching 
                    ? 'bg-states-flow' 
                    : 'bg-surface-tertiary'
                }`} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Emotional Intelligence</span>
                <div className={`w-2 h-2 rounded-full ${
                  config.response_optimization.emotional_intelligence 
                    ? 'bg-states-flow' 
                    : 'bg-surface-tertiary'
                }`} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Memory Integration</span>
                <div className={`w-2 h-2 rounded-full ${
                  config.response_optimization.memory_integration 
                    ? 'bg-states-flow' 
                    : 'bg-surface-tertiary'
                }`} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Real-time Adaptation</span>
                <div className={`w-2 h-2 rounded-full ${
                  config.response_optimization.real_time_adaptation 
                    ? 'bg-states-flow' 
                    : 'bg-surface-tertiary'
                }`} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Safety Filtering</span>
                <div className={`w-2 h-2 rounded-full ${
                  config.safety_features.content_filtering 
                    ? 'bg-states-flow' 
                    : 'bg-surface-tertiary'
                }`} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Red Flag Detection</span>
                <div className={`w-2 h-2 rounded-full ${
                  config.safety_features.red_flag_detection 
                    ? 'bg-states-flow' 
                    : 'bg-surface-tertiary'
                }`} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Response Generation */}
      {activeGeneration && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
        >
          <Card variant="consciousness" className="border-consciousness-primary/30">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-consciousness-primary/10 flex items-center justify-center">
                  <CpuChipIcon className="w-5 h-5 text-consciousness-primary animate-pulse" />
                </div>
                <div>
                  <CardTitle>Claude is Thinking...</CardTitle>
                  <p className="text-caption-text text-text-tertiary">
                    Using {activeGeneration.model_used} • 
                    {activeGeneration.estimated_completion_time && 
                      ` ETA: ${Math.round(activeGeneration.estimated_completion_time / 1000)}s`
                    }
                  </p>
                </div>
              </div>
            </CardHeader>
            {(activeGeneration.thinking_process || activeGeneration.partial_response) && (
              <CardContent>
                <div className="space-y-3">
                  {activeGeneration.thinking_process && (
                    <div>
                      <p className="text-sm font-medium mb-2">Thinking Process:</p>
                      <p className="text-sm text-text-secondary bg-surface-secondary/50 p-3 rounded-lg">
                        {activeGeneration.thinking_process}
                      </p>
                    </div>
                  )}
                  {activeGeneration.partial_response && (
                    <div>
                      <p className="text-sm font-medium mb-2">Generating Response:</p>
                      <p className="text-sm text-text-secondary bg-surface-secondary/50 p-3 rounded-lg">
                        {activeGeneration.partial_response}
                        <span className="animate-pulse">|</span>
                      </p>
                    </div>
                  )}
                  {activeGeneration.confidence_so_far && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm">Confidence:</span>
                      <div className="flex-1 bg-surface-secondary rounded-full h-2">
                        <motion.div
                          className="bg-consciousness-primary h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${activeGeneration.confidence_so_far * 100}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                      <span className="text-sm font-medium">
                        {Math.round(activeGeneration.confidence_so_far * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
            )}
          </Card>
        </motion.div>
      )}
    </div>
  );
}