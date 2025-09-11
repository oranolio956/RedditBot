/**
 * Kelly Account Settings
 * Comprehensive account management with personality configuration and safety settings
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users,
  Settings,
  Shield,
  Heart,
  Brain,
  Zap,
  Clock,
  DollarSign,
  Plus,
  Edit,
  Trash2,
  Play,
  Pause,
  Power,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff,
  Save,
  RotateCcw,
  Copy,
  ExternalLink,
  Smartphone,
  Wifi,
  WifiOff,
  Activity,
  BarChart3,
  Sliders,
  MessageCircle
} from 'lucide-react';
import { useKellyStore } from '@/store';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { KellyAccount, KellyPersonalityConfig } from '@/types/kelly';
import { formatDistanceToNow } from 'date-fns';
import TelegramConnectModal from '@/components/kelly/TelegramConnectModal';

const AccountSettings: React.FC = () => {
  const {
    accounts,
    selectedAccount,
    isLoading,
    setAccounts,
    setSelectedAccount,
    updateAccount,
    addAccount,
    removeAccount,
    setLoading
  } = useKellyStore();

  const [activeTab, setActiveTab] = useState('general');
  const [showAddAccount, setShowAddAccount] = useState(false);
  const [showApiCredentials, setShowApiCredentials] = useState(false);
  const [unsavedChanges, setUnsavedChanges] = useState(false);
  const [tempConfig, setTempConfig] = useState<KellyPersonalityConfig | null>(null);

  useEffect(() => {
    loadAccounts();
  }, []);

  useEffect(() => {
    if (selectedAccount && !tempConfig) {
      setTempConfig(selectedAccount.config);
    }
  }, [selectedAccount]);

  const loadAccounts = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/kelly/accounts');
      const data = await response.json();
      setAccounts(data.accounts || []);
      
      // Auto-select first account if none selected
      if (!selectedAccount && data.accounts?.length > 0) {
        setSelectedAccount(data.accounts[0]);
      }
    } catch (error) {
      console.error('Failed to load accounts:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleAccount = async (accountId: string) => {
    try {
      const account = accounts.find(a => a.id === accountId);
      if (!account) return;

      const response = await fetch(`/api/v1/kelly/accounts/${accountId}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: !account.is_active })
      });

      if (response.ok) {
        updateAccount(accountId, { is_active: !account.is_active });
      }
    } catch (error) {
      console.error('Failed to toggle account:', error);
    }
  };

  const savePersonalityConfig = async () => {
    if (!selectedAccount || !tempConfig) return;

    try {
      const response = await fetch(`/api/v1/kelly/accounts/${selectedAccount.id}/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: tempConfig })
      });

      if (response.ok) {
        updateAccount(selectedAccount.id, { config: tempConfig });
        setUnsavedChanges(false);
      }
    } catch (error) {
      console.error('Failed to save personality config:', error);
    }
  };

  const resetToDefaults = () => {
    if (selectedAccount) {
      setTempConfig(selectedAccount.config);
      setUnsavedChanges(false);
    }
  };

  const updateTempConfig = (updates: Partial<KellyPersonalityConfig>) => {
    setTempConfig(prev => prev ? { ...prev, ...updates } : null);
    setUnsavedChanges(true);
  };

  const getConnectionStatus = (account: KellyAccount) => {
    if (!account.is_active) return { status: 'inactive', color: 'bg-gray-400', text: 'Inactive' };
    if (account.is_connected) return { status: 'connected', color: 'bg-green-400', text: 'Connected' };
    return { status: 'disconnected', color: 'bg-red-400', text: 'Disconnected' };
  };

  const tabs = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'personality', label: 'Personality', icon: Brain },
    { id: 'response', label: 'Response Settings', icon: Clock },
    { id: 'safety', label: 'Safety', icon: Shield },
    { id: 'payment', label: 'Payment', icon: DollarSign },
  ];

  if (isLoading && accounts.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading accounts..." />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Account Settings</h1>
            <p className="mt-2 text-gray-600">
              Manage Kelly accounts, personality traits, and conversation settings
            </p>
          </div>
          
          <Button
            onClick={() => setShowAddAccount(true)}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Account
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Account List */}
        <div className="lg:col-span-1">
          <Card>
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Telegram Accounts</h3>
            </div>
            
            <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
              {accounts.map((account) => {
                const status = getConnectionStatus(account);
                const isSelected = selectedAccount?.id === account.id;
                
                return (
                  <motion.div
                    key={account.id}
                    layoutId={`account-${account.id}`}
                    className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                      isSelected ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                    }`}
                    onClick={() => setSelectedAccount(account)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${status.color}`} />
                        <span className="text-sm font-medium text-gray-900">
                          {account.phone_number.replace(/^\+/, '+')}
                        </span>
                      </div>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleAccount(account.id);
                        }}
                        className={`p-1 rounded ${account.is_active ? 'text-green-600' : 'text-gray-400'}`}
                      >
                        {account.is_active ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                      </button>
                    </div>
                    
                    <div className="text-xs text-gray-500 space-y-1">
                      <div className="flex justify-between">
                        <span>Status:</span>
                        <span className={status.status === 'connected' ? 'text-green-600' : 'text-red-600'}>
                          {status.text}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Today:</span>
                        <span>{account.messages_sent_today}/{account.max_daily_messages}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Mode:</span>
                        <span>{account.dm_only_mode ? 'DM Only' : 'All Chats'}</span>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* Account Configuration */}
        <div className="lg:col-span-3">
          {selectedAccount ? (
            <>
              {/* Tab Navigation */}
              <div className="border-b border-gray-200 mb-6">
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
                      <span className="text-sm text-yellow-800">You have unsaved changes</span>
                    </div>
                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm" onClick={resetToDefaults}>
                        <RotateCcw className="h-4 w-4 mr-1" />
                        Reset
                      </Button>
                      <Button size="sm" onClick={savePersonalityConfig}>
                        <Save className="h-4 w-4 mr-1" />
                        Save
                      </Button>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Tab Content */}
              <AnimatePresence mode="wait">
                {activeTab === 'general' && (
                  <GeneralSettings
                    account={selectedAccount}
                    onUpdate={(updates) => updateAccount(selectedAccount.id, updates)}
                    showApiCredentials={showApiCredentials}
                    setShowApiCredentials={setShowApiCredentials}
                  />
                )}
                
                {activeTab === 'personality' && tempConfig && (
                  <PersonalitySettings
                    config={tempConfig}
                    onUpdate={updateTempConfig}
                  />
                )}
                
                {activeTab === 'response' && tempConfig && (
                  <ResponseSettings
                    config={tempConfig}
                    onUpdate={updateTempConfig}
                  />
                )}
                
                {activeTab === 'safety' && tempConfig && (
                  <SafetySettings
                    config={tempConfig}
                    onUpdate={updateTempConfig}
                  />
                )}
                
                {activeTab === 'payment' && tempConfig && (
                  <PaymentSettings
                    config={tempConfig}
                    onUpdate={updateTempConfig}
                  />
                )}
              </AnimatePresence>
            </>
          ) : (
            <Card className="p-12 text-center">
              <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Account Selected</h3>
              <p className="text-gray-600 mb-4">
                Select an account from the list to configure its settings
              </p>
              <Button onClick={() => setShowAddAccount(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Your First Account
              </Button>
            </Card>
          )}
        </div>
      </div>

      {/* Telegram Connect Modal */}
      <TelegramConnectModal
        isOpen={showAddAccount}
        onClose={() => setShowAddAccount(false)}
        onAccountAdded={(account) => {
          addAccount(account);
          setShowAddAccount(false);
          setSelectedAccount(account);
        }}
      />
    </div>
  );
};

// General Settings Component
const GeneralSettings: React.FC<{
  account: KellyAccount;
  onUpdate: (updates: Partial<KellyAccount>) => void;
  showApiCredentials: boolean;
  setShowApiCredentials: (show: boolean) => void;
}> = ({ account, onUpdate, showApiCredentials, setShowApiCredentials }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Account Information</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Phone Number
              </label>
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  value={account.phone_number}
                  disabled
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md bg-gray-50 text-gray-500"
                />
                <Smartphone className="h-5 w-5 text-gray-400" />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session Name
              </label>
              <input
                type="text"
                value={account.session_name}
                onChange={(e) => onUpdate({ session_name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Friendly name for this session"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Daily Messages
              </label>
              <input
                type="number"
                min="1"
                max="1000"
                value={account.max_daily_messages}
                onChange={(e) => onUpdate({ max_daily_messages: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Connection Status
              </label>
              <div className="flex items-center space-x-2">
                {account.is_connected ? (
                  <Wifi className="h-5 w-5 text-green-500" />
                ) : (
                  <WifiOff className="h-5 w-5 text-red-500" />
                )}
                <span className={account.is_connected ? 'text-green-600' : 'text-red-600'}>
                  {account.is_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
          
          <div className="mt-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium text-gray-900">Account Active</h4>
                <p className="text-sm text-gray-500">Enable or disable this account for messaging</p>
              </div>
              <button
                onClick={() => onUpdate({ is_active: !account.is_active })}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  account.is_active ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    account.is_active ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium text-gray-900">DM Only Mode</h4>
                <p className="text-sm text-gray-500">Only respond to direct messages, ignore group chats</p>
              </div>
              <button
                onClick={() => onUpdate({ dm_only_mode: !account.dm_only_mode })}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  account.dm_only_mode ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    account.dm_only_mode ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
      </Card>
      
      {/* API Credentials */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">API Credentials</h3>
            <button
              onClick={() => setShowApiCredentials(!showApiCredentials)}
              className="text-sm text-blue-600 hover:text-blue-700 flex items-center"
            >
              {showApiCredentials ? <EyeOff className="h-4 w-4 mr-1" /> : <Eye className="h-4 w-4 mr-1" />}
              {showApiCredentials ? 'Hide' : 'Show'}
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                API ID
              </label>
              <div className="flex items-center space-x-2">
                <input
                  type={showApiCredentials ? 'text' : 'password'}
                  value={account.api_id}
                  disabled
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md bg-gray-50 text-gray-500"
                />
                <button className="p-2 text-gray-400 hover:text-gray-600">
                  <Copy className="h-4 w-4" />
                </button>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                API Hash
              </label>
              <div className="flex items-center space-x-2">
                <input
                  type={showApiCredentials ? 'text' : 'password'}
                  value={account.api_hash}
                  disabled
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md bg-gray-50 text-gray-500"
                />
                <button className="p-2 text-gray-400 hover:text-gray-600">
                  <Copy className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
          
          <p className="mt-4 text-sm text-gray-500">
            These credentials are used to connect to Telegram's API. Never share them with anyone.
          </p>
        </div>
      </Card>
      
      {/* Account Metrics */}
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Account Metrics</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {account.metrics.total_messages_sent.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Total Messages</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {account.metrics.total_conversations.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Conversations</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(account.metrics.engagement_success_rate * 100)}%
              </div>
              <div className="text-sm text-gray-500">Success Rate</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-emerald-600">
                {Math.round(account.metrics.conversation_quality_score)}
              </div>
              <div className="text-sm text-gray-500">Quality Score</div>
            </div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

// Personality Settings Component  
const PersonalitySettings: React.FC<{
  config: KellyPersonalityConfig;
  onUpdate: (updates: Partial<KellyPersonalityConfig>) => void;
}> = ({ config, onUpdate }) => {
  const personalityTraits = [
    { key: 'warmth', label: 'Warmth', description: 'How friendly and approachable Kelly appears', color: 'text-red-600' },
    { key: 'empathy', label: 'Empathy', description: 'Ability to understand and respond to emotions', color: 'text-pink-600' },
    { key: 'playfulness', label: 'Playfulness', description: 'Use of humor, jokes, and playful language', color: 'text-yellow-600' },
    { key: 'professionalism', label: 'Professionalism', description: 'Formal vs casual communication style', color: 'text-blue-600' },
    { key: 'confidence', label: 'Confidence', description: 'How assertive and self-assured responses are', color: 'text-purple-600' },
    { key: 'creativity', label: 'Creativity', description: 'Originality and uniqueness in responses', color: 'text-green-600' },
    { key: 'patience', label: 'Patience', description: 'Tolerance for repetitive or difficult conversations', color: 'text-teal-600' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Personality Traits</h3>
          <p className="text-sm text-gray-600 mb-6">
            Adjust Kelly's personality to match your desired conversation style. Each trait affects how Kelly responds to users.
          </p>
          
          <div className="space-y-6">
            {personalityTraits.map((trait) => {
              const value = config[trait.key as keyof KellyPersonalityConfig] as number;
              
              return (
                <div key={trait.key}>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-gray-700">
                      {trait.label}
                    </label>
                    <span className={`text-sm font-semibold ${trait.color}`}>
                      {Math.round(value * 100)}%
                    </span>
                  </div>
                  
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={value}
                    onChange={(e) => onUpdate({ [trait.key]: parseFloat(e.target.value) })}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  
                  <p className="text-xs text-gray-500 mt-1">
                    {trait.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </Card>
      
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Communication Style</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Message Length Preference
              </label>
              <select
                value={config.message_length_preference}
                onChange={(e) => onUpdate({ message_length_preference: e.target.value as any })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="short">Short (1-2 sentences)</option>
                <option value="medium">Medium (2-4 sentences)</option>
                <option value="long">Long (4+ sentences)</option>
                <option value="adaptive">Adaptive (Match user's style)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Punctuation Style
              </label>
              <select
                value={config.punctuation_style}
                onChange={(e) => onUpdate({ punctuation_style: e.target.value as any })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="minimal">Minimal</option>
                <option value="standard">Standard</option>
                <option value="expressive">Expressive (!!! ...)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Emoji Frequency ({Math.round(config.emoji_frequency * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.emoji_frequency}
                onChange={(e) => onUpdate({ emoji_frequency: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Typing Speed (WPM)
              </label>
              <input
                type="number"
                min="20"
                max="120"
                value={config.typing_speed_wpm}
                onChange={(e) => onUpdate({ typing_speed_wpm: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

// Response Settings Component
const ResponseSettings: React.FC<{
  config: KellyPersonalityConfig;
  onUpdate: (updates: Partial<KellyPersonalityConfig>) => void;
}> = ({ config, onUpdate }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Response Timing</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Minimum Response Delay (seconds)
              </label>
              <input
                type="number"
                min="1"
                max="300"
                value={config.response_delay_min}
                onChange={(e) => onUpdate({ response_delay_min: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Maximum Response Delay (seconds)
              </label>
              <input
                type="number"
                min="1"
                max="3600"
                value={config.response_delay_max}
                onChange={(e) => onUpdate({ response_delay_max: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          <div className="mt-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Initiation Probability ({Math.round(config.initiation_probability * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.initiation_probability}
                onChange={(e) => onUpdate({ initiation_probability: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                Likelihood of starting new conversations
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Follow-up Aggressiveness ({Math.round(config.follow_up_aggressiveness * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.follow_up_aggressiveness}
                onChange={(e) => onUpdate({ follow_up_aggressiveness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                How persistent Kelly is when users don't respond
              </p>
            </div>
          </div>
        </div>
      </Card>
      
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Conversation Behavior</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Topic Change Frequency ({Math.round(config.topic_change_frequency * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.topic_change_frequency}
                onChange={(e) => onUpdate({ topic_change_frequency: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                How often Kelly introduces new conversation topics
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Question Asking Rate ({Math.round(config.question_asking_rate * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.question_asking_rate}
                onChange={(e) => onUpdate({ question_asking_rate: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                Frequency of asking questions to keep conversations engaging
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Context Memory Depth
              </label>
              <input
                type="number"
                min="10"
                max="1000"
                value={config.context_memory_depth}
                onChange={(e) => onUpdate({ context_memory_depth: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Number of previous messages to remember for context
              </p>
            </div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

// Safety Settings Component
const SafetySettings: React.FC<{
  config: KellyPersonalityConfig;
  onUpdate: (updates: Partial<KellyPersonalityConfig>) => void;
}> = ({ config, onUpdate }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Red Flag Detection</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Detection Sensitivity ({Math.round(config.red_flag_sensitivity * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.red_flag_sensitivity}
                onChange={(e) => onUpdate({ red_flag_sensitivity: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                Higher sensitivity detects more potential safety issues
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Auto-block Threshold ({Math.round(config.auto_block_threshold * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.auto_block_threshold}
                onChange={(e) => onUpdate({ auto_block_threshold: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                Confidence level required to automatically block users
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Escalation Threshold ({Math.round(config.escalation_threshold * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.escalation_threshold}
                onChange={(e) => onUpdate({ escalation_threshold: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-1">
                When to escalate conversations for human review
              </p>
            </div>
          </div>
        </div>
      </Card>
      
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Content Filtering</h3>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filtering Level
            </label>
            <select
              value={config.content_filtering_level}
              onChange={(e) => onUpdate({ content_filtering_level: e.target.value as any })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="low">Low - Basic safety checks</option>
              <option value="medium">Medium - Standard filtering</option>
              <option value="high">High - Strict content policies</option>
              <option value="maximum">Maximum - Highest safety level</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Higher levels may reduce conversation flow but increase safety
            </p>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

// Payment Settings Component
const PaymentSettings: React.FC<{
  config: KellyPersonalityConfig;
  onUpdate: (updates: Partial<KellyPersonalityConfig>) => void;
}> = ({ config, onUpdate }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Payment Discussion Settings</h3>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Discussion Threshold ({config.payment_discussion_threshold} messages)
            </label>
            <input
              type="range"
              min="10"
              max="100"
              step="1"
              value={config.payment_discussion_threshold}
              onChange={(e) => onUpdate({ payment_discussion_threshold: parseInt(e.target.value) })}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-xs text-gray-500 mt-1">
              Minimum number of messages before payment topics can be discussed
            </p>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

export default AccountSettings;