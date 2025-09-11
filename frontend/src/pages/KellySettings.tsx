/**
 * Kelly Settings Page
 * 
 * Main Kelly dashboard with comprehensive tabbed interface for all Kelly system management
 */

import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Users, 
  MessageCircle, 
  Shield, 
  Brain, 
  Settings,
  DollarSign,
  Activity,
  Zap,
  RefreshCw
} from 'lucide-react';
import {
  CpuChipIcon,
  ShieldCheckIcon,
  CogIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { useKellyStore } from '@/store';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { Button } from '@/components/ui/Button';

// Import Kelly dashboard components
import KellyDashboard from './kelly/Dashboard';
import AccountSettings from './kelly/AccountSettings';
import ConversationManagement from './kelly/ConversationManagement';
import AIFeaturesConfig from './kelly/AIFeaturesConfig';

// Import new Claude AI components
import ClaudeAIDashboard from '@/components/kelly/ClaudeAIDashboard';
import ConversationManager from '@/components/kelly/ConversationManager';
import SafetyDashboard from '@/components/kelly/SafetyDashboard';
import ClaudeSettingsPanel from '@/components/kelly/ClaudeSettingsPanel';

const KellySettings: React.FC = () => {
  const { isLoading, setLoading, setActiveTab, activeTab } = useKellyStore();
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Initialize Kelly system data
    loadKellySystemData();
  }, []);

  const loadKellySystemData = async () => {
    try {
      setLoading(true);
      // System-wide initialization will be handled by individual dashboard components
    } catch (error) {
      console.error('Failed to initialize Kelly system:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadKellySystemData();
    setRefreshing(false);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <KellyDashboard />;
      case 'claude-ai':
        return <ClaudeAIDashboard />;
      case 'conversations':
        return <ConversationManager />;
      case 'ai-features':
        return <AIFeaturesConfig />;
      case 'safety':
        return <SafetyDashboard />;
      case 'claude-settings':
        return <ClaudeSettingsPanel />;
      case 'accounts':
        return <AccountSettings />;
      default:
        return <KellyDashboard />;
    }
  };

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'claude-ai', label: 'Claude AI', icon: CpuChipIcon },
    { id: 'conversations', label: 'Conversations', icon: MessageCircle },
    { id: 'ai-features', label: 'AI Features', icon: Brain },
    { id: 'safety', label: 'Safety', icon: ShieldCheckIcon },
    { id: 'claude-settings', label: 'Claude Settings', icon: CogIcon },
    { id: 'accounts', label: 'Accounts', icon: Users },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Kelly Brain System</h1>
        <p className="mt-2 text-gray-600">
          Claude AI-powered Telegram conversation management with real-time intelligence
        </p>
      </div>

      {/* Global Actions */}
      <div className="mb-8 flex justify-end">
        <Button
          onClick={handleRefresh}
          disabled={refreshing}
          variant="outline"
          size="sm"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh All
        </Button>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-8">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-5 w-5 mr-2" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {isLoading && activeTab === 'dashboard' ? (
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner size="lg" text="Loading Kelly system..." />
        </div>
      ) : (
        renderTabContent()
      )}
    </div>
  );
};

export default KellySettings;