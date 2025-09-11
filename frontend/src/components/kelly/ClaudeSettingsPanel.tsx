/**
 * Claude AI Settings Panel
 * Configuration and fine-tuning of Claude integration
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  CogIcon,
  CpuChipIcon,
  ShieldCheckIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { useKellyStore } from '@/store';
import { apiClient } from '@/lib/api';
import type { 
  ClaudeIntegrationConfig, 
  ClaudeStageConfig
} from '@/types/kelly';

interface ClaudeSettingsPanelProps {
  accountId?: string;
  onConfigChange?: (config: ClaudeIntegrationConfig) => void;
  className?: string;
}

export default function ClaudeSettingsPanel({ 
  accountId, 
  onConfigChange,
  className = '' 
}: ClaudeSettingsPanelProps) {
  const { selectedAccount } = useKellyStore();
  const currentAccountId = accountId || selectedAccount?.id || '';

  const [config, setConfig] = useState<ClaudeIntegrationConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<'general' | 'stages' | 'safety' | 'optimization'>('general');
  const [hasChanges, setHasChanges] = useState(false);

  // Load configuration
  useEffect(() => {
    if (!currentAccountId) return;

    const loadConfig = async () => {
      try {
        setIsLoading(true);
        const claudeConfig = await apiClient.getClaudeConfig(currentAccountId);
        setConfig(claudeConfig);
      } catch (error) {
        console.error('Failed to load Claude config:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadConfig();
  }, [currentAccountId]);

  const handleConfigUpdate = (updates: Partial<ClaudeIntegrationConfig>) => {
    if (!config) return;

    const updatedConfig = { ...config, ...updates };
    setConfig(updatedConfig);
    setHasChanges(true);
    onConfigChange?.(updatedConfig);
  };

  const handleStageConfigUpdate = (stage: keyof ClaudeIntegrationConfig['conversation_stages'], updates: Partial<ClaudeStageConfig>) => {
    if (!config) return;

    const updatedConfig = {
      ...config,
      conversation_stages: {
        ...config.conversation_stages,
        [stage]: { ...config.conversation_stages[stage], ...updates }
      }
    };
    setConfig(updatedConfig);
    setHasChanges(true);
    onConfigChange?.(updatedConfig);
  };

  const handleSaveConfig = async () => {
    if (!config || !currentAccountId) return;

    try {
      setIsSaving(true);
      const savedConfig = await apiClient.updateClaudeConfig(currentAccountId, config);
      setConfig(savedConfig);
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save Claude config:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const getModelColor = (model: string) => {
    switch (model) {
      case 'opus': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'sonnet': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'haiku': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const tabs = [
    { id: 'general', label: 'General', icon: CogIcon },
    { id: 'stages', label: 'Conversation Stages', icon: ClockIcon },
    { id: 'safety', label: 'Safety', icon: ShieldCheckIcon },
    { id: 'optimization', label: 'Optimization', icon: CpuChipIcon },
  ] as const;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading Claude settings..." />
      </div>
    );
  }

  if (!config) {
    return (
      <Card className="p-6">
        <div className="text-center text-text-secondary">
          <CpuChipIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Claude integration not configured</p>
        </div>
      </Card>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-lg bg-consciousness-primary/10 flex items-center justify-center">
                <CpuChipIcon className="w-6 h-6 text-consciousness-primary" />
              </div>
              <div>
                <CardTitle className="text-xl">Claude AI Settings</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  Configure Claude integration for optimal performance
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              {hasChanges && (
                <div className="flex items-center space-x-2 text-orange-600">
                  <ExclamationTriangleIcon className="w-4 h-4" />
                  <span className="text-sm">Unsaved changes</span>
                </div>
              )}
              <Button
                variant="primary"
                onClick={handleSaveConfig}
                disabled={!hasChanges || isSaving}
              >
                {isSaving ? 'Saving...' : 'Save Changes'}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Tab Navigation */}
      <Card>
        <CardContent className="p-0">
          <div className="flex border-b border-surface-tertiary">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-6 py-4 text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'border-b-2 border-consciousness-primary text-consciousness-primary bg-consciousness-primary/5'
                      : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {activeTab === 'general' && (
          <div className="space-y-6">
            {/* Basic Settings */}
            <Card>
              <CardHeader>
                <CardTitle>Basic Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Enable/Disable Toggle */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Enable Claude Integration</h4>
                    <p className="text-sm text-text-secondary">
                      Turn Claude AI responses on or off for this account
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.enabled}
                      onChange={(e) => handleConfigUpdate({ enabled: e.target.checked })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Model Selection */}
                <div>
                  <h4 className="font-medium mb-3">Default Model</h4>
                  <div className="grid grid-cols-4 gap-3">
                    {(['auto', 'opus', 'sonnet', 'haiku'] as const).map((model) => (
                      <button
                        key={model}
                        onClick={() => handleConfigUpdate({ model_selection: model })}
                        className={`p-3 border rounded-lg text-center transition-all ${
                          config.model_selection === model
                            ? getModelColor(model)
                            : 'border-surface-tertiary hover:border-consciousness-primary/30'
                        }`}
                      >
                        <div className="font-medium capitalize">{model}</div>
                        <div className="text-xs opacity-75 mt-1">
                          {model === 'auto' && 'Smart Selection'}
                          {model === 'opus' && 'Most Capable'}
                          {model === 'sonnet' && 'Balanced'}
                          {model === 'haiku' && 'Fastest'}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Temperature */}
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <h4 className="font-medium">Creativity Level</h4>
                    <span className="text-sm text-text-secondary">{config.temperature}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.temperature}
                    onChange={(e) => handleConfigUpdate({ temperature: Number(e.target.value) })}
                    className="w-full h-2 bg-surface-secondary rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-text-tertiary mt-1">
                    <span>Conservative</span>
                    <span>Balanced</span>
                    <span>Creative</span>
                  </div>
                </div>

                {/* Budget Settings */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">Daily Cost Limit ($)</label>
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={config.cost_limit_daily}
                      onChange={(e) => handleConfigUpdate({ cost_limit_daily: Number(e.target.value) })}
                      className="w-full p-3 border border-surface-tertiary rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Max Tokens per Response</label>
                    <input
                      type="number"
                      min="100"
                      max="4000"
                      step="100"
                      value={config.max_tokens}
                      onChange={(e) => handleConfigUpdate({ max_tokens: Number(e.target.value) })}
                      className="w-full p-3 border border-surface-tertiary rounded-lg"
                    />
                  </div>
                </div>

                {/* Confidence Threshold */}
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <h4 className="font-medium">Minimum Confidence Threshold</h4>
                    <span className="text-sm text-text-secondary">{Math.round(config.confidence_threshold * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={config.confidence_threshold}
                    onChange={(e) => handleConfigUpdate({ confidence_threshold: Number(e.target.value) })}
                    className="w-full h-2 bg-surface-secondary rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-text-tertiary mt-1">
                    Responses below this confidence level will require human review
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'stages' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Conversation Stage Configuration</CardTitle>
                <p className="text-caption-text text-text-tertiary">
                  Customize Claude's behavior for different conversation stages
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {Object.entries(config.conversation_stages).map(([stage, stageConfig]) => (
                    <div key={stage} className="border border-surface-tertiary rounded-lg p-4">
                      <h4 className="font-medium mb-4 capitalize">
                        {stage.replace('_', ' ')}
                      </h4>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Model Preference */}
                        <div>
                          <label className="block text-sm font-medium mb-2">Model</label>
                          <select
                            value={stageConfig.model_preference}
                            onChange={(e) => handleStageConfigUpdate(stage as keyof ClaudeIntegrationConfig['conversation_stages'], {
                              model_preference: e.target.value as 'opus' | 'sonnet' | 'haiku'
                            })}
                            className="w-full p-2 border border-surface-tertiary rounded"
                          >
                            <option value="haiku">Haiku (Fast)</option>
                            <option value="sonnet">Sonnet (Balanced)</option>
                            <option value="opus">Opus (Capable)</option>
                          </select>
                        </div>

                        {/* Temperature */}
                        <div>
                          <label className="block text-sm font-medium mb-2">
                            Temperature ({stageConfig.temperature})
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={stageConfig.temperature}
                            onChange={(e) => handleStageConfigUpdate(stage as keyof ClaudeIntegrationConfig['conversation_stages'], {
                              temperature: Number(e.target.value)
                            })}
                            className="w-full h-2 bg-surface-secondary rounded-lg appearance-none cursor-pointer"
                          />
                        </div>

                        {/* Personality Weight */}
                        <div>
                          <label className="block text-sm font-medium mb-2">
                            Personality Weight ({stageConfig.personality_weight})
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={stageConfig.personality_weight}
                            onChange={(e) => handleStageConfigUpdate(stage as keyof ClaudeIntegrationConfig['conversation_stages'], {
                              personality_weight: Number(e.target.value)
                            })}
                            className="w-full h-2 bg-surface-secondary rounded-lg appearance-none cursor-pointer"
                          />
                        </div>

                        {/* Safety Level */}
                        <div>
                          <label className="block text-sm font-medium mb-2">Safety Level</label>
                          <select
                            value={stageConfig.safety_level}
                            onChange={(e) => handleStageConfigUpdate(stage as keyof ClaudeIntegrationConfig['conversation_stages'], {
                              safety_level: e.target.value as 'low' | 'medium' | 'high' | 'maximum'
                            })}
                            className="w-full p-2 border border-surface-tertiary rounded"
                          >
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                            <option value="maximum">Maximum</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'safety' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Safety Features</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Content Filtering */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Content Filtering</h4>
                    <p className="text-sm text-text-secondary">
                      Automatically filter inappropriate content before responses
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.safety_features.content_filtering}
                      onChange={(e) => handleConfigUpdate({
                        safety_features: {
                          ...config.safety_features,
                          content_filtering: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Red Flag Detection */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Red Flag Detection</h4>
                    <p className="text-sm text-text-secondary">
                      Use Claude to identify potential safety concerns in conversations
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.safety_features.red_flag_detection}
                      onChange={(e) => handleConfigUpdate({
                        safety_features: {
                          ...config.safety_features,
                          red_flag_detection: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Manual Override */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Manual Override</h4>
                    <p className="text-sm text-text-secondary">
                      Allow manual override of Claude responses when needed
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.safety_features.manual_override_enabled}
                      onChange={(e) => handleConfigUpdate({
                        safety_features: {
                          ...config.safety_features,
                          manual_override_enabled: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Escalation Triggers */}
                <div>
                  <h4 className="font-medium mb-3">Escalation Triggers</h4>
                  <div className="space-y-2">
                    {config.safety_features.escalation_triggers.map((trigger, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <input
                          type="text"
                          value={trigger}
                          onChange={(e) => {
                            const newTriggers = [...config.safety_features.escalation_triggers];
                            newTriggers[index] = e.target.value;
                            handleConfigUpdate({
                              safety_features: {
                                ...config.safety_features,
                                escalation_triggers: newTriggers
                              }
                            });
                          }}
                          className="flex-1 p-2 border border-surface-tertiary rounded"
                          placeholder="Enter escalation trigger keyword..."
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            const newTriggers = config.safety_features.escalation_triggers.filter((_, i) => i !== index);
                            handleConfigUpdate({
                              safety_features: {
                                ...config.safety_features,
                                escalation_triggers: newTriggers
                              }
                            });
                          }}
                        >
                          Remove
                        </Button>
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        handleConfigUpdate({
                          safety_features: {
                            ...config.safety_features,
                            escalation_triggers: [...config.safety_features.escalation_triggers, '']
                          }
                        });
                      }}
                    >
                      Add Trigger
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'optimization' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Response Optimization</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Personality Matching */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Personality Matching</h4>
                    <p className="text-sm text-text-secondary">
                      Adapt responses to match user's communication style and personality
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.response_optimization.personality_matching}
                      onChange={(e) => handleConfigUpdate({
                        response_optimization: {
                          ...config.response_optimization,
                          personality_matching: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Emotional Intelligence */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Emotional Intelligence</h4>
                    <p className="text-sm text-text-secondary">
                      Consider emotional context and respond with appropriate empathy
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.response_optimization.emotional_intelligence}
                      onChange={(e) => handleConfigUpdate({
                        response_optimization: {
                          ...config.response_optimization,
                          emotional_intelligence: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Memory Integration */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Memory Integration</h4>
                    <p className="text-sm text-text-secondary">
                      Use conversation history and learned patterns in responses
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.response_optimization.memory_integration}
                      onChange={(e) => handleConfigUpdate({
                        response_optimization: {
                          ...config.response_optimization,
                          memory_integration: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>

                {/* Real-time Adaptation */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Real-time Adaptation</h4>
                    <p className="text-sm text-text-secondary">
                      Continuously adapt based on conversation feedback and outcomes
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.response_optimization.real_time_adaptation}
                      onChange={(e) => handleConfigUpdate({
                        response_optimization: {
                          ...config.response_optimization,
                          real_time_adaptation: e.target.checked
                        }
                      })}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-consciousness-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-consciousness-primary"></div>
                  </label>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </motion.div>
    </div>
  );
}