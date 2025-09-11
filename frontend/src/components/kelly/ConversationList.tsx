/**
 * Kelly Conversation List Component
 * Virtual scrolling conversation list with real-time updates and performance optimization
 * Supports 10,000+ conversations with smooth scrolling
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { FixedSizeList as List, ListChildComponentProps } from 'react-window';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageCircle,
  AlertTriangle,
  User,
  Bot,
  MoreVertical,
  ArrowRight,
  Zap,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { KellyConversation, ConversationStage } from '@/types/kelly';
import { cn, formatRelativeTime, truncate } from '@/lib/utils';

interface ConversationListProps {
  conversations: KellyConversation[];
  selectedConversation: KellyConversation | null;
  onConversationSelect: (conversation: KellyConversation) => void;
  isLoading?: boolean;
  collapsed?: boolean;
  searchQuery?: string;
  className?: string;
}

interface ConversationItemData {
  conversations: KellyConversation[];
  selectedConversation: KellyConversation | null;
  onConversationSelect: (conversation: KellyConversation) => void;
  collapsed: boolean;
  searchQuery: string;
}

type ConversationItemProps = ListChildComponentProps<ConversationItemData>;

// Individual conversation item component (optimized for virtual scrolling)
const ConversationItem = ({ index, style, data }: ConversationItemProps) => {
  const { conversations, selectedConversation, onConversationSelect, collapsed, searchQuery } = data;
  const conversation = conversations[index];
  const isSelected = selectedConversation?.id === conversation.id;
  const [isHovered, setIsHovered] = useState(false);

  const getStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'active': return 'bg-states-flow';
      case 'paused': return 'bg-consciousness-accent';
      case 'ended': return 'bg-text-tertiary';
      case 'blocked': return 'bg-states-stress';
      default: return 'bg-text-tertiary';
    }
  }, []);

  const getStageColor = useCallback((stage: ConversationStage) => {
    switch (stage) {
      case 'initial_contact': return 'text-blue-600 bg-blue-50';
      case 'rapport_building': return 'text-green-600 bg-green-50';
      case 'qualification': return 'text-yellow-600 bg-yellow-50';
      case 'engagement': return 'text-purple-600 bg-purple-50';
      case 'advanced_engagement': return 'text-indigo-600 bg-indigo-50';
      case 'payment_discussion': return 'text-emerald-600 bg-emerald-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  }, []);

  const getEngagementIcon = useCallback((score: number) => {
    if (score >= 80) return <TrendingUp className="h-3 w-3 text-states-flow" />;
    if (score >= 60) return <ArrowRight className="h-3 w-3 text-consciousness-accent" />;
    return <TrendingDown className="h-3 w-3 text-states-stress" />;
  }, []);

  const highlightSearchTerm = useCallback((text: string, query: string) => {
    if (!query || !text) return text;
    const regex = new RegExp(`(${query})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, i) => 
      regex.test(part) ? 
        <mark key={i} className="bg-consciousness-accent/20 text-consciousness-accent font-medium">
          {part}
        </mark> : 
        part
    );
  }, []);

  const displayName = conversation.user_info.first_name || 
                     conversation.user_info.username || 
                     `User ${conversation.user_info.user_id}`;

  const lastMessage = conversation.recent_messages?.[conversation.recent_messages.length - 1];
  const messagePreview = lastMessage ? 
    truncate(lastMessage.content, 50) : 
    'No messages yet';

  const unreadCount = conversation.recent_messages?.filter(
    m => m.sender === 'user' && !m.read
  ).length || 0;

  if (collapsed) {
    return (
      <div style={style} className="px-2">
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className={cn(
            "relative w-full p-2 rounded-lg cursor-pointer transition-all duration-200",
            isSelected 
              ? "bg-consciousness-primary text-white shadow-md" 
              : "hover:bg-surface-elevated"
          )}
          onClick={() => onConversationSelect(conversation)}
        >
          {/* Avatar only */}
          <div className="relative mx-auto">
            <div className={cn(
              "w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold",
              isSelected 
                ? "bg-white/20 text-white" 
                : "bg-gradient-to-br from-consciousness-primary to-consciousness-secondary text-white"
            )}>
              {displayName[0].toUpperCase()}
            </div>
            
            {/* Status indicator */}
            <div className={cn(
              "absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2",
              isSelected ? "border-consciousness-primary" : "border-white",
              getStatusColor(conversation.status)
            )} />
            
            {/* Unread indicator */}
            {unreadCount > 0 && (
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-states-stress text-white text-xs rounded-full flex items-center justify-center">
                {unreadCount > 9 ? '9+' : unreadCount}
              </div>
            )}
          </div>
          
          {/* Red flags indicator */}
          {conversation.red_flags.length > 0 && (
            <div className="absolute top-1 left-1">
              <AlertTriangle className="h-3 w-3 text-states-stress" />
            </div>
          )}
        </motion.div>
      </div>
    );
  }

  return (
    <div style={style} className="px-4">
      <motion.div
        layout
        whileHover={{ scale: 1.005 }}
        whileTap={{ scale: 0.995 }}
        className={cn(
          "relative w-full p-3 rounded-xl cursor-pointer transition-all duration-200 border",
          isSelected 
            ? "bg-consciousness-primary text-white shadow-lg border-consciousness-primary/20" 
            : "bg-white hover:bg-surface-elevated border-gray-100 hover:border-consciousness-primary/30 hover:shadow-md"
        )}
        onClick={() => onConversationSelect(conversation)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div className="flex items-start space-x-3">
          {/* Avatar with status */}
          <div className="relative flex-shrink-0">
            <div className={cn(
              "w-12 h-12 rounded-full flex items-center justify-center text-sm font-semibold",
              isSelected 
                ? "bg-white/20 text-white" 
                : "bg-gradient-to-br from-consciousness-primary to-consciousness-secondary text-white"
            )}>
              {displayName[0].toUpperCase()}
            </div>
            
            {/* Status indicator */}
            <div className={cn(
              "absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full border-2",
              isSelected ? "border-consciousness-primary" : "border-white",
              getStatusColor(conversation.status)
            )} />
          </div>

          {/* Main content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between mb-1">
              <div className="flex-1 min-w-0">
                <h4 className={cn(
                  "text-sm font-semibold truncate",
                  isSelected ? "text-white" : "text-text-primary"
                )}>
                  {highlightSearchTerm(displayName, searchQuery)}
                </h4>
                {conversation.user_info.username && conversation.user_info.first_name && (
                  <p className={cn(
                    "text-xs truncate",
                    isSelected ? "text-white/70" : "text-text-tertiary"
                  )}>
                    @{highlightSearchTerm(conversation.user_info.username, searchQuery)}
                  </p>
                )}
              </div>
              
              <div className="flex items-center space-x-1 ml-2">
                {/* Time */}
                <span className={cn(
                  "text-xs whitespace-nowrap",
                  isSelected ? "text-white/70" : "text-text-tertiary"
                )}>
                  {formatRelativeTime(conversation.last_activity)}
                </span>
                
                {/* More menu (on hover) */}
                <AnimatePresence>
                  {isHovered && (
                    <motion.button
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className={cn(
                        "p-1 rounded-full transition-colors",
                        isSelected 
                          ? "hover:bg-white/10 text-white/70" 
                          : "hover:bg-gray-100 text-text-tertiary"
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle more menu
                      }}
                    >
                      <MoreVertical className="h-3 w-3" />
                    </motion.button>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Message preview */}
            <div className="flex items-center space-x-2 mb-2">
              {lastMessage && (
                <div className="flex items-center space-x-1 flex-1 min-w-0">
                  {lastMessage.sender === 'kelly' ? (
                    <Bot className={cn(
                      "h-3 w-3 flex-shrink-0",
                      isSelected ? "text-white/70" : "text-consciousness-primary"
                    )} />
                  ) : (
                    <User className={cn(
                      "h-3 w-3 flex-shrink-0",
                      isSelected ? "text-white/70" : "text-text-tertiary"
                    )} />
                  )}
                  <p className={cn(
                    "text-xs truncate",
                    isSelected ? "text-white/70" : "text-text-secondary"
                  )}>
                    {highlightSearchTerm(messagePreview, searchQuery)}
                  </p>
                </div>
              )}
              
              {/* Unread count */}
              {unreadCount > 0 && (
                <div className="bg-states-stress text-white text-xs px-1.5 py-0.5 rounded-full font-medium">
                  {unreadCount > 99 ? '99+' : unreadCount}
                </div>
              )}
            </div>

            {/* Stage and metrics */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {/* Stage badge */}
                <span className={cn(
                  "px-2 py-0.5 rounded-full text-xs font-medium",
                  isSelected 
                    ? "bg-white/20 text-white" 
                    : getStageColor(conversation.stage)
                )}>
                  {conversation.stage.replace('_', ' ')}
                </span>
                
                {/* Safety indicators */}
                {conversation.red_flags.length > 0 && (
                  <div className="flex items-center space-x-1">
                    <AlertTriangle className={cn(
                      "h-3 w-3",
                      isSelected ? "text-white" : "text-states-stress"
                    )} />
                    <span className={cn(
                      "text-xs",
                      isSelected ? "text-white/70" : "text-states-stress"
                    )}>
                      {conversation.red_flags.length}
                    </span>
                  </div>
                )}
                
                {/* AI confidence */}
                {conversation.ai_confidence && (
                  <div className="flex items-center space-x-1">
                    <Zap className={cn(
                      "h-3 w-3",
                      isSelected ? "text-white" : "text-consciousness-secondary"
                    )} />
                    <span className={cn(
                      "text-xs",
                      isSelected ? "text-white/70" : "text-text-tertiary"
                    )}>
                      {Math.round(conversation.ai_confidence)}%
                    </span>
                  </div>
                )}
              </div>
              
              {/* Engagement indicator */}
              <div className="flex items-center space-x-1">
                {getEngagementIcon(conversation.engagement_score)}
                <span className={cn(
                  "text-xs font-medium",
                  isSelected ? "text-white/70" : "text-text-tertiary"
                )}>
                  {conversation.engagement_score}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Progress bar for stage */}
        <div className="mt-2">
          <div className={cn(
            "w-full h-1 rounded-full overflow-hidden",
            isSelected ? "bg-white/20" : "bg-gray-200"
          )}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(conversation.message_count / 50) * 100}%` }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className={cn(
                "h-full rounded-full",
                isSelected ? "bg-white/60" : "bg-consciousness-primary"
              )}
            />
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  selectedConversation,
  onConversationSelect,
  isLoading = false,
  collapsed = false,
  searchQuery = '',
  className
}) => {
  const listRef = useRef<List>(null);
  const [listHeight, setListHeight] = useState(600);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate item height based on collapsed state
  const itemHeight = collapsed ? 60 : 140;

  // Update list height when container resizes
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setListHeight(rect.height);
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Scroll to selected conversation
  useEffect(() => {
    if (selectedConversation && listRef.current) {
      const index = conversations.findIndex(c => c.id === selectedConversation.id);
      if (index !== -1) {
        listRef.current.scrollToItem(index, 'smart');
      }
    }
  }, [selectedConversation, conversations]);

  // Memoized item data to prevent unnecessary re-renders
  const itemData: ConversationItemData = useMemo(() => ({
    conversations,
    selectedConversation,
    onConversationSelect,
    collapsed,
    searchQuery
  }), [conversations, selectedConversation, onConversationSelect, collapsed, searchQuery]);

  if (isLoading && conversations.length === 0) {
    return (
      <div className={cn("flex items-center justify-center h-full", className)}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-consciousness-primary mx-auto mb-4"></div>
          <p className="text-sm text-text-tertiary">Loading conversations...</p>
        </div>
      </div>
    );
  }

  if (conversations.length === 0) {
    return (
      <div className={cn("flex items-center justify-center h-full p-8", className)}>
        <div className="text-center">
          <MessageCircle className="h-12 w-12 text-text-tertiary mx-auto mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">
            No Conversations
          </h3>
          <p className="text-sm text-text-tertiary">
            {searchQuery 
              ? `No conversations match "${searchQuery}"`
              : "No active conversations found"
            }
          </p>
        </div>
      </div>
    );
  }

  return (
    <div 
      ref={containerRef}
      className={cn("h-full overflow-hidden", className)}
    >
      <List
        ref={listRef}
        height={listHeight}
        itemCount={conversations.length}
        itemSize={itemHeight}
        itemData={itemData}
        overscanCount={5} // Render extra items for smooth scrolling
        className="scrollbar-apple"
      >
        {ConversationItem}
      </List>
      
      {/* Loading overlay for refresh */}
      <AnimatePresence>
        {isLoading && conversations.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute top-0 left-0 right-0 bg-white/80 backdrop-blur-sm p-2 flex items-center justify-center"
          >
            <div className="flex items-center space-x-2 text-sm text-consciousness-primary">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-consciousness-primary"></div>
              <span>Updating...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ConversationList;