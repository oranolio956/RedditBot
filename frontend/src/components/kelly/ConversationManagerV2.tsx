/**
 * Kelly AI Conversation Manager - Phase 1
 * WhatsApp Web/Slack-inspired two-pane layout with real-time updates
 * 30/70 split: conversation list / message viewer with manual intervention
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  MessageSquare,
  AlertTriangle,
  Filter,
  MoreHorizontal,
  ChevronDown,
  RefreshCw
} from 'lucide-react';
import { useKellyStore } from '@/store';
import { Button } from '@/components/ui/Button';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { ConversationList } from './ConversationList';
import { MessageViewer } from './MessageViewer';
import ManualInterventionPanel from './ManualInterventionPanel';
import { useWebSocket, useKellyConversationUpdates, useKellySafetyAlerts } from '@/lib/websocket';
import { KellyConversation, ConversationStage, SafetyAlert } from '@/types/kelly';
import { cn, formatRelativeTime, debounce } from '@/lib/utils';

interface ConversationManagerV2Props {
  className?: string;
  onConversationSelect?: (conversation: KellyConversation) => void;
  onSafetyAlert?: (alert: SafetyAlert) => void;
}

export const ConversationManagerV2: React.FC<ConversationManagerV2Props> = ({
  className,
  onConversationSelect,
  onSafetyAlert
}) => {
  const {
    activeConversations,
    selectedConversation,
    isLoading,
    error,
    setActiveConversations,
    setSelectedConversation,
    updateConversation,
    setLoading,
    setError
  } = useKellyStore();

  // Local state for UI
  const [searchQuery, setSearchQuery] = useState('');
  const [stageFilter, setStageFilter] = useState<ConversationStage | 'all'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'flagged' | 'review'>('all');
  const [sortBy, setSortBy] = useState<'recent' | 'engagement' | 'safety'>('recent');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [manualMode, setManualMode] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // WebSocket connection for real-time updates
  const ws = useWebSocket();

  // Real-time conversation updates
  useKellyConversationUpdates(
    selectedConversation?.id || '',
    useCallback((update) => {
      if (update.payload.conversation_id && selectedConversation?.id === update.payload.conversation_id) {
        updateConversation(update.payload.conversation_id, {
          recent_messages: update.payload.new_messages || [],
          engagement_score: update.payload.engagement_score_change || selectedConversation?.engagement_score || 0,
          safety_score: update.payload.safety_score_change || selectedConversation?.safety_score || 0
        });
        setLastUpdate(new Date());
      }
    }, [selectedConversation, updateConversation])
  );

  // Safety alerts monitoring
  useKellySafetyAlerts(
    useCallback((alert) => {
      if (alert.payload.requires_immediate_action) {
        onSafetyAlert?.(alert);
        // Auto-select conversation with critical safety issues
        if (alert.payload.severity === 'critical') {
          const conversation = activeConversations.find(c => c.id === alert.payload.conversation_id);
          if (conversation) {
            setSelectedConversation(conversation);
            setManualMode(true); // Enable manual mode for critical issues
          }
        }
      }
    }, [activeConversations, onSafetyAlert, setSelectedConversation])
  );

  // Debounced search to improve performance
  const debouncedSearch = useMemo(
    () => debounce((query: string) => {
      setSearchQuery(query);
    }, 300),
    []
  );

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
    // Set up periodic refresh (every 30 seconds)
    const interval = setInterval(loadConversations, 30000);
    return () => clearInterval(interval);
  }, []);

  // Connect to real-time updates
  useEffect(() => {
    if (!ws.connected) {
      ws.connect().catch(console.error);
    }
  }, [ws]);

  const loadConversations = async () => {
    if (isLoading) return; // Prevent multiple simultaneous requests
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/kelly/conversations/active', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to load conversations: ${response.statusText}`);
      }
      
      const data = await response.json();
      setActiveConversations(data.conversations || []);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Error loading conversations:', err);
      setError(err instanceof Error ? err.message : 'Failed to load conversations');
    } finally {
      setLoading(false);
    }
  };

  const handleConversationSelect = useCallback((conversation: KellyConversation) => {
    setSelectedConversation(conversation);
    onConversationSelect?.(conversation);
  }, [setSelectedConversation, onConversationSelect]);

  const handleRefresh = useCallback(() => {
    loadConversations();
  }, []);

  const handleManualModeToggle = useCallback((enabled: boolean) => {
    setManualMode(enabled);
  }, []);

  // Filtered and sorted conversations
  const filteredConversations = useMemo(() => {
    return activeConversations
      .filter(conversation => {
        // Search filter
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          const searchableText = [
            conversation.user_info.username,
            conversation.user_info.first_name,
            conversation.user_info.last_name,
            ...conversation.topics_discussed
          ].filter(Boolean).join(' ').toLowerCase();
          
          if (!searchableText.includes(query)) return false;
        }
        
        // Stage filter
        if (stageFilter !== 'all' && conversation.stage !== stageFilter) return false;
        
        // Status filter
        if (statusFilter === 'flagged' && conversation.red_flags.length === 0) return false;
        if (statusFilter === 'review' && !conversation.requires_human_review) return false;
        if (statusFilter === 'active' && conversation.status !== 'active') return false;
        
        return true;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case 'recent':
            return new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime();
          case 'engagement':
            return b.engagement_score - a.engagement_score;
          case 'safety':
            return a.safety_score - b.safety_score; // Lower safety scores first
          default:
            return 0;
        }
      });
  }, [activeConversations, searchQuery, stageFilter, statusFilter, sortBy]);

  // Summary stats
  const stats = useMemo(() => {
    const total = activeConversations.length;
    const flagged = activeConversations.filter(c => c.red_flags.length > 0).length;
    const needsReview = activeConversations.filter(c => c.requires_human_review).length;
    const activeCount = activeConversations.filter(c => c.status === 'active').length;
    
    return { total, flagged, needsReview, active: activeCount };
  }, [activeConversations]);

  if (isLoading && activeConversations.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" text="Loading conversations..." />
      </div>
    );
  }

  return (
    <div className={cn(
      "flex h-screen bg-surface-primary overflow-hidden",
      className
    )}>
      {/* Left Sidebar - Conversation List (30%) */}
      <motion.div
        initial={false}
        animate={{ width: sidebarCollapsed ? '80px' : '30%' }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="bg-surface-secondary border-r border-gray-200 flex flex-col min-w-0"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 bg-white">
          <div className="flex items-center justify-between mb-4">
            <motion.div
              animate={{ opacity: sidebarCollapsed ? 0 : 1 }}
              transition={{ duration: 0.2 }}
              className="flex items-center space-x-2"
            >
              <MessageSquare className="h-6 w-6 text-consciousness-primary" />
              <h1 className="text-lg font-semibold text-text-primary">
                Conversations
              </h1>
            </motion.div>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRefresh}
                disabled={isLoading}
                className="h-8 w-8 p-0"
              >
                <RefreshCw className={cn(
                  "h-4 w-4",
                  isLoading && "animate-spin"
                )} />
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="h-8 w-8 p-0"
              >
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </div>
          </div>
          
          {/* Stats Row */}
          {!sidebarCollapsed && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="grid grid-cols-4 gap-2 text-xs"
            >
              <div className="text-center">
                <div className="font-semibold text-consciousness-primary">{stats.total}</div>
                <div className="text-text-tertiary">Total</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-states-flow">{stats.active}</div>
                <div className="text-text-tertiary">Active</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-consciousness-accent">{stats.needsReview}</div>
                <div className="text-text-tertiary">Review</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-states-stress">{stats.flagged}</div>
                <div className="text-text-tertiary">Flagged</div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Search and Filters */}
        {!sidebarCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-4 space-y-3 bg-white border-b border-gray-200"
          >
            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-text-tertiary" />
              <input
                type="text"
                placeholder="Search conversations..."
                onChange={(e) => debouncedSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-consciousness-primary focus:border-transparent"
              />
            </div>
            
            {/* Quick Filters */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Button
                  variant={statusFilter === 'all' ? 'primary' : 'ghost'}
                  size="sm"
                  onClick={() => setStatusFilter('all')}
                  className="text-xs h-7"
                >
                  All
                </Button>
                <Button
                  variant={statusFilter === 'flagged' ? 'warning' : 'ghost'}
                  size="sm"
                  onClick={() => setStatusFilter('flagged')}
                  className="text-xs h-7"
                >
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Flagged
                </Button>
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowFilters(!showFilters)}
                className="text-xs h-7"
              >
                <Filter className="h-3 w-3 mr-1" />
                <ChevronDown className={cn(
                  "h-3 w-3 transition-transform",
                  showFilters && "rotate-180"
                )} />
              </Button>
            </div>
            
            {/* Advanced Filters */}
            <AnimatePresence>
              {showFilters && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="grid grid-cols-2 gap-2"
                >
                  <select
                    value={stageFilter}
                    onChange={(e) => setStageFilter(e.target.value as any)}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-consciousness-primary"
                  >
                    <option value="all">All Stages</option>
                    <option value="initial_contact">Initial</option>
                    <option value="rapport_building">Rapport</option>
                    <option value="qualification">Qualification</option>
                    <option value="engagement">Engagement</option>
                    <option value="advanced_engagement">Advanced</option>
                  </select>
                  
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-consciousness-primary"
                  >
                    <option value="recent">Recent</option>
                    <option value="engagement">Engagement</option>
                    <option value="safety">Safety Issues</option>
                  </select>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Conversation List */}
        <div className="flex-1 overflow-hidden">
          <ConversationList
            conversations={filteredConversations}
            selectedConversation={selectedConversation}
            onConversationSelect={handleConversationSelect}
            isLoading={isLoading}
            collapsed={sidebarCollapsed}
            searchQuery={searchQuery}
          />
        </div>
      </motion.div>

      {/* Right Panel - Message Viewer & Manual Intervention (70%) */}
      <div className="flex-1 flex flex-col min-w-0 bg-white">
        {selectedConversation ? (
          <div className="flex flex-col h-full">
            {/* Manual Intervention Panel - Top */}
            <ManualInterventionPanel
              conversation={selectedConversation}
              manualMode={manualMode}
              onManualModeToggle={handleManualModeToggle}
              className="border-b border-gray-200"
            />
            
            {/* Message Viewer - Main */}
            <div className="flex-1 overflow-hidden">
              <MessageViewer
                conversation={selectedConversation}
                manualMode={manualMode}
                onMessageSent={() => {
                  // Refresh conversation data after sending message
                  handleRefresh();
                }}
              />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center max-w-md mx-auto p-8"
            >
              <div className="w-16 h-16 bg-consciousness-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <MessageSquare className="w-8 h-8 text-consciousness-primary" />
              </div>
              <h3 className="text-xl font-semibold text-text-primary mb-2">
                Select a Conversation
              </h3>
              <p className="text-text-secondary">
                Choose a conversation from the list to view messages and manage the interaction.
              </p>
              {error && (
                <div className="mt-4 p-3 bg-states-stress/10 text-states-stress rounded-lg text-sm">
                  <div className="flex items-center justify-center space-x-2">
                    <AlertTriangle className="h-4 w-4" />
                    <span>{error}</span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefresh}
                    className="mt-2"
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Retry
                  </Button>
                </div>
              )}
            </motion.div>
          </div>
        )}
      </div>

      {/* Real-time Status Indicator */}
      <div className="fixed bottom-4 right-4 z-50">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className={cn(
            "flex items-center space-x-2 px-3 py-2 rounded-full text-xs font-medium backdrop-blur-sm",
            ws.connected 
              ? "bg-states-flow/90 text-white"
              : "bg-states-stress/90 text-white"
          )}
        >
          <div className={cn(
            "w-2 h-2 rounded-full",
            ws.connected ? "bg-white animate-pulse" : "bg-white/70"
          )} />
          <span>
            {ws.connected ? 'Real-time' : 'Disconnected'}
          </span>
          <span className="text-white/70">
            {formatRelativeTime(lastUpdate)}
          </span>
        </motion.div>
      </div>
    </div>
  );
};

export default ConversationManagerV2;