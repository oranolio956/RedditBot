/**
 * Kelly Conversation Management
 * Real-time conversation monitoring with intervention capabilities
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageCircle,
  Users,
  Eye,
  AlertTriangle,
  CheckCircle,
  Clock,
  ThumbsUp,
  ThumbsDown,
  Play,
  Pause,
  StopCircle,
  Edit,
  Send,
  Filter,
  Search,
  MoreVertical,
  Star,
  Flag,
  UserX,
  Volume2,
  VolumeX,
  ChevronDown,
  ChevronRight,
  Activity,
  BarChart3,
  RefreshCw,
  Zap,
  Brain,
  Heart,
  Shield,
  TrendingUp,
  Calendar,
  User,
  Bot,
  ArrowRight,
  ExternalLink
} from 'lucide-react';
import { useKellyStore } from '@/store';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { KellyConversation, ConversationMessage, ConversationStage, RedFlag } from '@/types/kelly';
import { formatDistanceToNow, format } from 'date-fns';
import { ManualInterventionPanel } from '@/components/kelly/ManualInterventionPanel';

const ConversationManagement: React.FC = () => {
  const {
    activeConversations,
    selectedConversation,
    isLoading,
    setActiveConversations,
    setSelectedConversation,
    updateConversation,
    setLoading
  } = useKellyStore();

  const [filterStage, setFilterStage] = useState<ConversationStage | 'all'>('all');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'flagged' | 'review'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'recent' | 'engagement' | 'safety' | 'stage'>('recent');
  const [realTimeEnabled, setRealTimeEnabled] = useState(true);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([]);
  const [showInterventionPanel, setShowInterventionPanel] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConversations();
    
    // Set up real-time updates
    if (realTimeEnabled) {
      const interval = setInterval(loadConversations, 10000); // Refresh every 10 seconds
      return () => clearInterval(interval);
    }
  }, [realTimeEnabled]);

  useEffect(() => {
    if (selectedConversation) {
      loadConversationMessages(selectedConversation.id);
      loadAiSuggestions(selectedConversation.id);
    }
  }, [selectedConversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadConversations = async () => {
    try {
      const response = await fetch('/api/v1/kelly/conversations/active');
      const data = await response.json();
      setActiveConversations(data.conversations || []);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversationMessages = async (conversationId: string) => {
    try {
      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/messages`);
      const data = await response.json();
      setMessages(data.messages || []);
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  const loadAiSuggestions = async (conversationId: string) => {
    try {
      const response = await fetch(`/api/v1/kelly/conversations/${conversationId}/suggestions`);
      const data = await response.json();
      setAiSuggestions(data.suggestions || []);
    } catch (error) {
      console.error('Failed to load AI suggestions:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const filteredConversations = activeConversations.filter(conversation => {
    // Stage filter
    if (filterStage !== 'all' && conversation.stage !== filterStage) return false;
    
    // Status filter
    if (filterStatus === 'flagged' && conversation.red_flags.length === 0) return false;
    if (filterStatus === 'review' && !conversation.requires_human_review) return false;
    if (filterStatus === 'active' && conversation.status !== 'active') return false;
    
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const username = (conversation.user_info.username || '').toLowerCase();
      const firstName = (conversation.user_info.first_name || '').toLowerCase();
      if (!username.includes(query) && !firstName.includes(query)) return false;
    }
    
    return true;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'recent':
        return new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime();
      case 'engagement':
        return b.engagement_score - a.engagement_score;
      case 'safety':
        return b.safety_score - a.safety_score;
      case 'stage':
        return a.stage.localeCompare(b.stage);
      default:
        return 0;
    }
  });

  const getStageColor = (stage: ConversationStage) => {
    switch (stage) {
      case 'initial_contact': return 'bg-blue-100 text-blue-800';
      case 'rapport_building': return 'bg-green-100 text-green-800';
      case 'qualification': return 'bg-yellow-100 text-yellow-800';
      case 'engagement': return 'bg-purple-100 text-purple-800';
      case 'advanced_engagement': return 'bg-indigo-100 text-indigo-800';
      case 'payment_discussion': return 'bg-emerald-100 text-emerald-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStageProgress = (stage: ConversationStage) => {
    const stages = ['initial_contact', 'rapport_building', 'qualification', 'engagement', 'advanced_engagement', 'payment_discussion'];
    return ((stages.indexOf(stage) + 1) / stages.length) * 100;
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !selectedConversation) return;

    try {
      const response = await fetch(`/api/v1/kelly/conversations/${selectedConversation.id}/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: newMessage, manual_override: true })
      });

      if (response.ok) {
        setNewMessage('');
        loadConversationMessages(selectedConversation.id);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const pauseConversation = async (conversationId: string) => {
    try {
      await fetch(`/api/v1/kelly/conversations/${conversationId}/pause`, {
        method: 'POST'
      });
      updateConversation(conversationId, { status: 'paused' });
    } catch (error) {
      console.error('Failed to pause conversation:', error);
    }
  };

  const resumeConversation = async (conversationId: string) => {
    try {
      await fetch(`/api/v1/kelly/conversations/${conversationId}/resume`, {
        method: 'POST'
      });
      updateConversation(conversationId, { status: 'active' });
    } catch (error) {
      console.error('Failed to resume conversation:', error);
    }
  };

  const endConversation = async (conversationId: string) => {
    try {
      await fetch(`/api/v1/kelly/conversations/${conversationId}/end`, {
        method: 'POST'
      });
      updateConversation(conversationId, { status: 'ended' });
    } catch (error) {
      console.error('Failed to end conversation:', error);
    }
  };

  if (isLoading && activeConversations.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading conversations..." />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Conversation Management</h1>
            <p className="mt-2 text-gray-600">
              Monitor and manage ongoing conversations in real-time
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500">Real-time</span>
              <button
                onClick={() => setRealTimeEnabled(!realTimeEnabled)}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  realTimeEnabled ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    realTimeEnabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
            
            <Button onClick={loadConversations} variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            
            <Button 
              onClick={() => setShowInterventionPanel(true)}
              variant="outline" 
              size="sm"
              className="text-red-600 hover:text-red-700"
            >
              <Shield className="h-4 w-4 mr-2" />
              Emergency Controls
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        {/* Conversation List */}
        <div className="lg:col-span-2">
          {/* Filters */}
          <Card className="mb-6">
            <div className="p-4">
              <div className="space-y-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search conversations..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                
                {/* Filters */}
                <div className="grid grid-cols-3 gap-2">
                  <select
                    value={filterStage}
                    onChange={(e) => setFilterStage(e.target.value as any)}
                    className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Stages</option>
                    <option value="initial_contact">Initial</option>
                    <option value="rapport_building">Rapport</option>
                    <option value="qualification">Qualification</option>
                    <option value="engagement">Engagement</option>
                    <option value="advanced_engagement">Advanced</option>
                    <option value="payment_discussion">Payment</option>
                  </select>
                  
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value as any)}
                    className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="flagged">Flagged</option>
                    <option value="review">Needs Review</option>
                  </select>
                  
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="recent">Recent</option>
                    <option value="engagement">Engagement</option>
                    <option value="safety">Safety</option>
                    <option value="stage">Stage</option>
                  </select>
                </div>
              </div>
            </div>
          </Card>

          {/* Conversation List */}
          <Card>
            <div className="p-4 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-medium text-gray-900">Active Conversations</h3>
                <span className="text-sm text-gray-500">
                  {filteredConversations.length} of {activeConversations.length}
                </span>
              </div>
            </div>
            
            <div className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
              {filteredConversations.map((conversation) => {
                const isSelected = selectedConversation?.id === conversation.id;
                
                return (
                  <motion.div
                    key={conversation.id}
                    layoutId={`conversation-${conversation.id}`}
                    className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                      isSelected ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                    }`}
                    onClick={() => setSelectedConversation(conversation)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <div className="relative">
                          <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                            <span className="text-white font-medium text-sm">
                              {(conversation.user_info.first_name || conversation.user_info.username || 'U')[0].toUpperCase()}
                            </span>
                          </div>
                          <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${
                            conversation.status === 'active' ? 'bg-green-400' :
                            conversation.status === 'paused' ? 'bg-yellow-400' : 'bg-red-400'
                          }`} />
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <h4 className="text-sm font-medium text-gray-900 truncate">
                            {conversation.user_info.username || conversation.user_info.first_name || 'Unknown User'}
                          </h4>
                          <p className="text-sm text-gray-500 truncate">
                            Stage: {conversation.stage.replace('_', ' ')}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        {conversation.red_flags.length > 0 && (
                          <AlertTriangle className="h-4 w-4 text-red-500" />
                        )}
                        {conversation.requires_human_review && (
                          <Eye className="h-4 w-4 text-yellow-500" />
                        )}
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <MoreVertical className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                    
                    {/* Stage Progress */}
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                        <span>Progress</span>
                        <span>{Math.round(getStageProgress(conversation.stage))}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${getStageProgress(conversation.stage)}%` }}
                        />
                      </div>
                    </div>
                    
                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="text-center">
                        <div className="text-gray-500">Messages</div>
                        <div className="font-semibold">{conversation.message_count}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-500">Engagement</div>
                        <div className={`font-semibold ${
                          conversation.engagement_score >= 80 ? 'text-green-600' :
                          conversation.engagement_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {conversation.engagement_score}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-500">Safety</div>
                        <div className={`font-semibold ${
                          conversation.safety_score >= 80 ? 'text-green-600' :
                          conversation.safety_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {conversation.safety_score}%
                        </div>
                      </div>
                    </div>
                    
                    {/* Last Activity */}
                    <div className="mt-2 text-xs text-gray-500">
                      {formatDistanceToNow(new Date(conversation.last_activity), { addSuffix: true })}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* Conversation Detail */}
        <div className="lg:col-span-3">
          {selectedConversation ? (
            <div className="space-y-6">
              {/* Conversation Header */}
              <Card>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                        <span className="text-white font-medium">
                          {(selectedConversation.user_info.first_name || selectedConversation.user_info.username || 'U')[0].toUpperCase()}
                        </span>
                      </div>
                      
                      <div>
                        <h2 className="text-xl font-bold text-gray-900">
                          {selectedConversation.user_info.username || selectedConversation.user_info.first_name || 'Unknown User'}
                        </h2>
                        <p className="text-sm text-gray-500">
                          {selectedConversation.user_info.username && selectedConversation.user_info.first_name && (
                            `${selectedConversation.user_info.first_name} (@${selectedConversation.user_info.username})`
                          )}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {selectedConversation.status === 'active' ? (
                        <Button
                          onClick={() => pauseConversation(selectedConversation.id)}
                          variant="outline"
                          size="sm"
                        >
                          <Pause className="h-4 w-4 mr-2" />
                          Pause
                        </Button>
                      ) : (
                        <Button
                          onClick={() => resumeConversation(selectedConversation.id)}
                          variant="outline"
                          size="sm"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Resume
                        </Button>
                      )}
                      
                      <Button
                        onClick={() => endConversation(selectedConversation.id)}
                        variant="outline"
                        size="sm"
                        className="text-red-600 hover:text-red-700"
                      >
                        <StopCircle className="h-4 w-4 mr-2" />
                        End
                      </Button>
                    </div>
                  </div>
                  
                  {/* Stage and Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <span className="text-sm text-gray-500">Stage</span>
                      <div className={`mt-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        getStageColor(selectedConversation.stage)
                      }`}>
                        {selectedConversation.stage.replace('_', ' ')}
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-sm text-gray-500">Messages</span>
                      <div className="text-lg font-semibold text-gray-900">
                        {selectedConversation.message_count}
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-sm text-gray-500">Engagement</span>
                      <div className={`text-lg font-semibold ${
                        selectedConversation.engagement_score >= 80 ? 'text-green-600' :
                        selectedConversation.engagement_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {selectedConversation.engagement_score}%
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-sm text-gray-500">Safety Score</span>
                      <div className={`text-lg font-semibold ${
                        selectedConversation.safety_score >= 80 ? 'text-green-600' :
                        selectedConversation.safety_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {selectedConversation.safety_score}%
                      </div>
                    </div>
                  </div>
                  
                  {/* Red Flags */}
                  {selectedConversation.red_flags.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-red-700 mb-2">
                        Red Flags ({selectedConversation.red_flags.length})
                      </h4>
                      <div className="space-y-2">
                        {selectedConversation.red_flags.slice(0, 3).map((flag, index) => (
                          <div key={index} className="bg-red-50 border border-red-200 rounded-md p-2">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-red-800">
                                {flag.type.replace('_', ' ')}
                              </span>
                              <span className={`text-xs px-2 py-1 rounded-full ${
                                flag.severity === 'critical' ? 'bg-red-100 text-red-800' :
                                flag.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                                'bg-yellow-100 text-yellow-800'
                              }`}>
                                {flag.severity}
                              </span>
                            </div>
                            <p className="text-sm text-red-700 mt-1">{flag.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
              
              {/* Messages */}
              <Card>
                <div className="p-4 border-b border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900">Messages</h3>
                </div>
                
                <div className="h-96 overflow-y-auto p-4 space-y-4">
                  {messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${message.sender === 'kelly' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.sender === 'kelly'
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-100 text-gray-900'
                      }`}>
                        <div className="flex items-center space-x-2 mb-1">
                          {message.sender === 'kelly' ? (
                            <Bot className="h-3 w-3" />
                          ) : (
                            <User className="h-3 w-3" />
                          )}
                          <span className="text-xs opacity-75">
                            {format(new Date(message.timestamp), 'HH:mm')}
                          </span>
                          {message.ai_confidence && (
                            <span className="text-xs opacity-75">
                              {Math.round(message.ai_confidence)}%
                            </span>
                          )}
                        </div>
                        <p className="text-sm">{message.content}</p>
                        
                        {/* Safety flags */}
                        {message.safety_flags.length > 0 && (
                          <div className="mt-2 flex items-center space-x-1">
                            <AlertTriangle className="h-3 w-3 text-red-300" />
                            <span className="text-xs opacity-75">
                              {message.safety_flags.length} flag(s)
                            </span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
                
                {/* Message Input */}
                <div className="p-4 border-t border-gray-200">
                  <div className="flex space-x-2">
                    <input
                      type="text"
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                      placeholder="Type a message to send manually..."
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <Button onClick={sendMessage} disabled={!newMessage.trim()}>
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {/* AI Suggestions */}
                  {aiSuggestions.length > 0 && (
                    <div className="mt-3">
                      <p className="text-sm text-gray-500 mb-2">AI Suggestions:</p>
                      <div className="space-y-1">
                        {aiSuggestions.slice(0, 3).map((suggestion, index) => (
                          <button
                            key={index}
                            onClick={() => setNewMessage(suggestion)}
                            className="block w-full text-left text-sm bg-gray-50 hover:bg-gray-100 p-2 rounded-md transition-colors"
                          >
                            <Zap className="h-3 w-3 inline mr-2 text-blue-500" />
                            {suggestion}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            </div>
          ) : (
            <Card className="p-12 text-center">
              <MessageCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Conversation Selected</h3>
              <p className="text-gray-600">
                Select a conversation from the list to view messages and manage the interaction
              </p>
            </Card>
          )}
        </div>
      </div>

      {/* Manual Intervention Panel */}
      <AnimatePresence>
        {showInterventionPanel && (
          <ManualInterventionPanel 
            onClose={() => setShowInterventionPanel(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default ConversationManagement;