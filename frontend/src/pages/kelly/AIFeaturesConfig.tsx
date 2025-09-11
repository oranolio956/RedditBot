/**
 * Kelly AI Features Configuration
 * Comprehensive AI feature management with revolutionary capabilities
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Eye,
  Heart,
  Clock,
  Zap,
  Sparkles,
  Palette,
  Moon,
  Activity,
  Settings,
  ToggleLeft,
  ToggleRight,
  Sliders,
  Info,
  TrendingUp,
  Target,
  Cpu,
  Database,
  BarChart3,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Save,
  RotateCcw,
  Play,
  Pause,
  StopCircle,
  Layers,
  Network,
  Waves,
  Atom,
  Microscope,
  Telescope,
  Lightbulb,
  Puzzle
} from 'lucide-react';
import { useKellyStore } from '@/store';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { AIFeatureConfig } from '@/types/kelly';

const AIFeaturesConfig: React.FC = () => {
  const {
    aiFeatures,
    isLoading,
    setAIFeatures,
    updateAIFeature,
    setLoading
  } = useKellyStore();

  const [activeFeature, setActiveFeature] = useState<string | null>(null);
  const [unsavedChanges, setUnsavedChanges] = useState(false);
  const [tempConfig, setTempConfig] = useState<AIFeatureConfig | null>(null);
  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [testing, setTesting] = useState<Record<string, boolean>>({});

  useEffect(() => {
    loadAIFeatures();
  }, []);

  useEffect(() => {
    if (aiFeatures && !tempConfig) {
      setTempConfig(aiFeatures);
    }
  }, [aiFeatures]);

  const loadAIFeatures = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/kelly/ai-features');
      const data = await response.json();
      setAIFeatures(data.features);
    } catch (error) {
      console.error('Failed to load AI features:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveConfiguration = async () => {
    if (!tempConfig) return;

    try {
      const response = await fetch('/api/v1/kelly/ai-features', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: tempConfig })
      });

      if (response.ok) {
        setAIFeatures(tempConfig);
        setUnsavedChanges(false);
      }
    } catch (error) {
      console.error('Failed to save AI features:', error);
    }
  };

  const resetToDefaults = () => {
    if (aiFeatures) {
      setTempConfig(aiFeatures);
      setUnsavedChanges(false);
    }
  };

  const updateFeature = (feature: keyof AIFeatureConfig, updates: any) => {
    if (!tempConfig) return;
    
    setTempConfig({
      ...tempConfig,
      [feature]: { ...tempConfig[feature], ...updates }
    });
    setUnsavedChanges(true);
  };

  const testFeature = async (featureName: string) => {
    setTesting({ ...testing, [featureName]: true });
    
    try {
      const response = await fetch(`/api/v1/kelly/ai-features/${featureName}/test`, {
        method: 'POST'
      });
      const data = await response.json();
      setTestResults({ ...testResults, [featureName]: data.result });
    } catch (error) {
      console.error(`Failed to test ${featureName}:`, error);
    } finally {
      setTesting({ ...testing, [featureName]: false });
    }
  };

  const aiFeaturesList = [
    {
      key: 'consciousness_mirror',
      title: 'Consciousness Mirror',
      description: 'Mirrors user personality and adapts communication style in real-time',
      icon: Brain,
      color: 'from-purple-500 to-pink-500',
      category: 'Core Intelligence'
    },
    {
      key: 'memory_palace',
      title: 'Memory Palace',
      description: 'Advanced contextual memory with cross-conversation pattern recognition',
      icon: Database,
      color: 'from-blue-500 to-indigo-500',
      category: 'Memory Systems'
    },
    {
      key: 'emotional_intelligence',
      title: 'Emotional Intelligence',
      description: 'Deep emotional understanding with mood prediction and empathetic responses',
      icon: Heart,
      color: 'from-red-500 to-pink-500',
      category: 'Emotional Processing'
    },
    {
      key: 'temporal_archaeology',
      title: 'Temporal Archaeology',
      description: 'Analyzes conversation patterns over time to predict optimal engagement strategies',
      icon: Clock,
      color: 'from-green-500 to-teal-500',
      category: 'Pattern Analysis'
    },
    {
      key: 'digital_telepathy',
      title: 'Digital Telepathy',
      description: 'Predicts user responses and optimal timing with quantum-inspired algorithms',
      icon: Zap,
      color: 'from-yellow-500 to-orange-500',
      category: 'Predictive Systems'
    },
    {
      key: 'quantum_consciousness',
      title: 'Quantum Consciousness',
      description: 'Multi-dimensional decision making with parallel response generation',
      icon: Atom,
      color: 'from-indigo-500 to-purple-500',
      category: 'Advanced Cognition'
    },
    {
      key: 'synesthesia',
      title: 'Synesthesia Engine',
      description: 'Cross-modal sensory interpretation for richer conversation understanding',
      icon: Palette,
      color: 'from-pink-500 to-rose-500',
      category: 'Sensory Processing'
    },
    {
      key: 'neural_dreams',
      title: 'Neural Dreams',
      description: 'Subconscious pattern recognition with creative response generation',
      icon: Moon,
      color: 'from-teal-500 to-cyan-500',
      category: 'Creative Intelligence'
    }
  ];

  if (isLoading && !tempConfig) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading AI features..." />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">AI Features Configuration</h1>
            <p className="mt-2 text-gray-600">
              Configure Kelly's revolutionary AI capabilities for optimal conversation management
            </p>
          </div>
          
          {unsavedChanges && (
            <div className="flex items-center space-x-2">
              <Button variant="outline" onClick={resetToDefaults}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset
              </Button>
              <Button onClick={saveConfiguration}>
                <Save className="h-4 w-4 mr-2" />
                Save Changes
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Unsaved Changes Alert */}
      {unsavedChanges && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-yellow-50 border border-yellow-200 rounded-md p-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-yellow-500 mr-3" />
              <span className="text-sm text-yellow-800">You have unsaved changes to your AI configuration</span>
            </div>
          </div>
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Features Overview */}
        <div className="lg:col-span-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {aiFeaturesList.map((feature) => {
              const config = tempConfig?.[feature.key as keyof AIFeatureConfig];
              const isEnabled = config?.enabled;
              const isActive = activeFeature === feature.key;
              const Icon = feature.icon;
              
              return (
                <motion.div
                  key={feature.key}
                  layout
                  className={`cursor-pointer transition-all duration-200 ${
                    isActive ? 'transform scale-105' : 'hover:transform hover:scale-102'
                  }`}
                  onClick={() => setActiveFeature(isActive ? null : feature.key)}
                >
                  <Card className={`h-full ${
                    isEnabled ? 'ring-2 ring-blue-200 bg-blue-50' : 'bg-gray-50'
                  }`}>
                    <div className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className={`p-3 rounded-lg bg-gradient-to-r ${feature.color}`}>
                          <Icon className="h-6 w-6 text-white" />
                        </div>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            updateFeature(feature.key as keyof AIFeatureConfig, {
                              enabled: !isEnabled
                            });
                          }}
                          className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            isEnabled ? 'bg-blue-600' : 'bg-gray-200'
                          }`}
                        >
                          <span
                            className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                              isEnabled ? 'translate-x-5' : 'translate-x-0'
                            }`}
                          />
                        </button>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                          {feature.title}
                        </h3>
                        <p className="text-sm text-gray-600 mb-3">
                          {feature.description}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            {feature.category}
                          </span>
                          
                          <div className="flex items-center space-x-2">
                            {testResults[feature.key] && (
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            )}
                            
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                testFeature(feature.key);
                              }}
                              disabled={testing[feature.key] || !isEnabled}
                            >
                              {testing[feature.key] ? (
                                <RefreshCw className="h-3 w-3 animate-spin" />
                              ) : (
                                <Play className="h-3 w-3" />
                              )}
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Feature Configuration Panel */}
        <div>
          <Card className="sticky top-8">
            <div className="p-6">
              {activeFeature ? (
                <FeatureConfigPanel
                  featureKey={activeFeature}
                  config={tempConfig?.[activeFeature as keyof AIFeatureConfig]}
                  onUpdate={(updates) => updateFeature(activeFeature as keyof AIFeatureConfig, updates)}
                  testResult={testResults[activeFeature]}
                  onTest={() => testFeature(activeFeature)}
                  testing={testing[activeFeature]}
                />
              ) : (
                <div className="text-center py-8">
                  <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Select a Feature
                  </h3>
                  <p className="text-gray-600">
                    Click on any AI feature to configure its settings
                  </p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

// Feature Configuration Panel Component
const FeatureConfigPanel: React.FC<{
  featureKey: string;
  config: any;
  onUpdate: (updates: any) => void;
  testResult?: any;
  onTest: () => void;
  testing?: boolean;
}> = ({ featureKey, config, onUpdate, testResult, onTest, testing }) => {
  if (!config) return null;

  const getFeatureTitle = (key: string) => {
    return key.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const renderConfigOptions = () => {
    switch (featureKey) {
      case 'consciousness_mirror':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sensitivity ({Math.round((config.sensitivity || 0.5) * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.sensitivity || 0.5}
                onChange={(e) => onUpdate({ sensitivity: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                How quickly to adapt to user personality changes
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mirroring Strength ({Math.round((config.mirroring_strength || 0.7) * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.mirroring_strength || 0.7}
                onChange={(e) => onUpdate({ mirroring_strength: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                Balance between mirroring and maintaining Kelly's personality
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Adaptation Speed ({Math.round((config.adaptation_speed || 0.6) * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.adaptation_speed || 0.6}
                onChange={(e) => onUpdate({ adaptation_speed: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        );
        
      case 'memory_palace':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Context Window Size
              </label>
              <input
                type="number"
                min="10"
                max="1000"
                value={config.context_window_size || 100}
                onChange={(e) => onUpdate({ context_window_size: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Number of messages to keep in active memory
              </p>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Long-term Storage</span>
              <button
                onClick={() => onUpdate({ long_term_storage_enabled: !config.long_term_storage_enabled })}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  config.long_term_storage_enabled ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    config.long_term_storage_enabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Memory Consolidation
              </label>
              <select
                value={config.memory_consolidation_frequency || 'daily'}
                onChange={(e) => onUpdate({ memory_consolidation_frequency: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="hourly">Hourly</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
              </select>
            </div>
          </div>
        );
        
      default:
        return (
          <div className="text-center py-4">
            <Info className="h-8 w-8 text-gray-400 mx-auto mb-2" />
            <p className="text-sm text-gray-500">
              Configuration options for {getFeatureTitle(featureKey)} will be available soon.
            </p>
          </div>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          {getFeatureTitle(featureKey)}
        </h3>
        <div className="flex items-center justify-between">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
            config.enabled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
          }`}>
            {config.enabled ? 'Enabled' : 'Disabled'}
          </span>
          
          <Button
            variant="outline"
            size="sm"
            onClick={onTest}
            disabled={testing || !config.enabled}
          >
            {testing ? (
              <RefreshCw className="h-3 w-3 animate-spin mr-2" />
            ) : (
              <Play className="h-3 w-3 mr-2" />
            )}
            Test Feature
          </Button>
        </div>
      </div>
      
      {config.enabled && (
        <div className="space-y-6">
          {renderConfigOptions()}
          
          {testResult && (
            <div className="bg-green-50 border border-green-200 rounded-md p-4">
              <div className="flex items-start">
                <CheckCircle className="h-5 w-5 text-green-500 mr-3 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-green-800">Test Successful</h4>
                  <div className="mt-2 text-sm text-green-700">
                    <pre className="whitespace-pre-wrap">
                      {JSON.stringify(testResult, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
};

export default AIFeaturesConfig;