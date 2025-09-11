/**
 * Kelly Message Viewer Component
 * Real-time message display with typing indicators, bubble UI, and manual takeover controls
 * Optimized for large message histories with virtual scrolling
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { FixedSizeList as List } from 'react-window';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Bot,
  User,
  AlertTriangle,
  Shield,
  Clock,
  CheckCircle,
  XCircle,
  Zap,
  Brain,
  Heart,
  Star,
  Flag,
  MoreVertical,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Edit,
  Trash2,
  RefreshCw,
  Eye,
  EyeOff,
  Volume2,
  VolumeX,
  Image,
  File,
  Download,
  MessageCircle
} from 'lucide-react';
import { KellyConversation, ConversationMessage, GeneratedResponse } from '@/types/kelly';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { useKellyConversationUpdates, useClaudeResponseGeneration } from '@/lib/websocket';
import { cn, formatRelativeTime, formatDateTime } from '@/lib/utils';

interface MessageViewerProps {
  conversation: KellyConversation;
  manualMode?: boolean;
  onMessageSent?: () => void;
  className?: string;
}

interface MessageItemProps {
  index: number;
  style: React.CSSProperties;
  data: {
    messages: ConversationMessage[];
    conversation: KellyConversation;
    onMessageAction: (messageId: string, action: string) => void;
  };
}

// Typing indicator component
const TypingIndicator: React.FC<{ visible: boolean }> = ({ visible }) => {
  if (!visible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex items-center space-x-2 px-4 py-2"
    >
      <div className="flex items-center space-x-2">
        <Bot className="h-4 w-4 text-consciousness-primary" />
        <div className="bg-gray-100 rounded-full px-4 py-2">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
          </div>
        </div>
      </div>
      <span className="text-xs text-text-tertiary">Kelly is thinking...</span>
    </motion.div>
  );
};

// Individual message bubble component
const MessageBubble: React.FC<{
  message: ConversationMessage;
  conversation: KellyConversation;
  onAction: (messageId: string, action: string) => void;
}> = ({ message, conversation, onAction }) => {
  const [showActions, setShowActions] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const isKelly = message.sender === 'kelly';
  const isUser = message.sender === 'user';

  const getSafetyColor = useCallback((flags: any[]) => {
    if (flags.some(f => f.severity === 'critical')) return 'text-states-stress';
    if (flags.some(f => f.severity === 'high')) return 'text-consciousness-accent';
    if (flags.some(f => f.severity === 'medium')) return 'text-yellow-600';
    return 'text-text-tertiary';
  }, []);

  const getConfidenceColor = useCallback((confidence: number) => {
    if (confidence >= 90) return 'text-states-flow';
    if (confidence >= 70) return 'text-consciousness-primary';
    if (confidence >= 50) return 'text-consciousness-accent';
    return 'text-states-stress';
  }, []);

  const handleCopyMessage = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    onAction(message.id, 'copy');
  }, [message.content, message.id, onAction]);

  const handleFlagMessage = useCallback(() => {
    onAction(message.id, 'flag');
  }, [message.id, onAction]);

  const handleRateMessage = useCallback((rating: 'up' | 'down') => {
    onAction(message.id, `rate_${rating}`);
  }, [message.id, onAction]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex mb-4 px-4",
        isKelly ? "justify-end" : "justify-start"
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className={cn(
        "max-w-[70%] space-y-1",
        isKelly ? "items-end" : "items-start"
      )}>
        {/* Message bubble */}
        <div className={cn(
          "relative group rounded-2xl px-4 py-2 shadow-sm",
          isKelly 
            ? "bg-consciousness-primary text-white rounded-br-md"
            : "bg-gray-100 text-text-primary rounded-bl-md"
        )}>
          {/* Message content */}
          <div className="space-y-2">
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
            
            {/* Media attachments */}
            {message.message_type !== 'text' && (
              <div className="flex items-center space-x-2 text-xs opacity-75">
                {message.message_type === 'photo' && <Image className="h-3 w-3" />}
                {message.message_type === 'voice' && <Volume2 className="h-3 w-3" />}
                {message.message_type === 'video' && <Volume2 className="h-3 w-3" />}
                {message.message_type === 'document' && <File className="h-3 w-3" />}
                <span className="capitalize">{message.message_type}</span>
              </div>
            )}
          </div>

          {/* Quick actions */}
          <AnimatePresence>
            {showActions && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className={cn(
                  "absolute top-0 flex items-center space-x-1 bg-white rounded-lg shadow-lg border p-1",
                  isKelly ? "-left-20" : "-right-20"
                )}
              >
                <button
                  onClick={handleCopyMessage}
                  className="p-1 text-text-tertiary hover:text-consciousness-primary rounded"
                  title="Copy message"
                >
                  <Copy className="h-3 w-3" />
                </button>
                
                {isKelly && (
                  <>
                    <button
                      onClick={() => handleRateMessage('up')}
                      className="p-1 text-text-tertiary hover:text-states-flow rounded"
                      title="Good response"
                    >
                      <ThumbsUp className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => handleRateMessage('down')}
                      className="p-1 text-text-tertiary hover:text-states-stress rounded"
                      title="Poor response"
                    >
                      <ThumbsDown className="h-3 w-3" />
                    </button>
                  </>
                )}
                
                <button
                  onClick={handleFlagMessage}
                  className="p-1 text-text-tertiary hover:text-consciousness-accent rounded"
                  title="Flag message"
                >
                  <Flag className="h-3 w-3" />
                </button>
                
                <button
                  onClick={() => setShowDetails(!showDetails)}
                  className="p-1 text-text-tertiary hover:text-consciousness-primary rounded"
                  title="Message details"
                >
                  <MoreVertical className="h-3 w-3" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Message metadata */}
        <div className={cn(
          "flex items-center space-x-2 text-xs text-text-tertiary px-2",
          isKelly ? "justify-end" : "justify-start"
        )}>
          {/* Sender icon */}
          {isKelly ? (
            <Bot className="h-3 w-3 text-consciousness-primary" />
          ) : (
            <User className="h-3 w-3" />
          )}
          
          {/* Timestamp */}
          <span>{formatRelativeTime(message.timestamp)}</span>
          
          {/* AI confidence for Kelly's messages */}
          {isKelly && message.ai_confidence && (
            <div className="flex items-center space-x-1">
              <Zap className={cn("h-3 w-3", getConfidenceColor(message.ai_confidence))} />
              <span className={getConfidenceColor(message.ai_confidence)}>
                {Math.round(message.ai_confidence)}%
              </span>
            </div>
          )}
          
          {/* Safety flags */}
          {message.safety_flags.length > 0 && (
            <div className="flex items-center space-x-1">
              <Shield className={cn("h-3 w-3", getSafetyColor(message.safety_flags))} />
              <span className={getSafetyColor(message.safety_flags)}>
                {message.safety_flags.length}
              </span>
            </div>
          )}
          
          {/* Response quality for Kelly's messages */}
          {isKelly && message.response_quality_score && (
            <div className="flex items-center space-x-1">
              <Star className="h-3 w-3 text-consciousness-accent" />
              <span>{Math.round(message.response_quality_score)}%</span>
            </div>
          )}
        </div>

        {/* Detailed metadata (expandable) */}
        <AnimatePresence>
          {showDetails && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-gray-50 rounded-lg p-3 text-xs space-y-2 max-w-sm"
            >
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <span className="font-medium">Timestamp:</span>
                  <div className="text-text-tertiary">{formatDateTime(message.timestamp)}</div>
                </div>
                
                {message.sentiment_score && (
                  <div>
                    <span className="font-medium">Sentiment:</span>
                    <div className={cn(
                      "font-medium",
                      message.sentiment_score > 0.6 ? "text-states-flow" :
                      message.sentiment_score < -0.6 ? "text-states-stress" : "text-consciousness-accent"
                    )}>
                      {message.sentiment_score > 0.6 ? 'Positive' :
                       message.sentiment_score < -0.6 ? 'Negative' : 'Neutral'}
                    </div>
                  </div>
                )}
                
                {message.emotional_tone && (
                  <div>
                    <span className="font-medium">Emotion:</span>
                    <div className="text-text-tertiary capitalize">{message.emotional_tone}</div>
                  </div>
                )}
                
                {message.topics_mentioned.length > 0 && (
                  <div className="col-span-2">
                    <span className="font-medium">Topics:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {message.topics_mentioned.slice(0, 3).map((topic, i) => (
                        <span key={i} className="bg-consciousness-primary/10 text-consciousness-primary px-2 py-0.5 rounded-full text-xs">
                          {topic}
                        </span>
                      ))}
                      {message.topics_mentioned.length > 3 && (
                        <span className="text-text-tertiary">+{message.topics_mentioned.length - 3} more</span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Safety flags details */}
              {message.safety_flags.length > 0 && (
                <div>
                  <span className="font-medium text-states-stress">Safety Flags:</span>
                  <div className="space-y-1 mt-1">
                    {message.safety_flags.map((flag, i) => (
                      <div key={i} className="flex items-center justify-between bg-states-stress/10 rounded px-2 py-1">
                        <span className="text-states-stress capitalize">{flag.type.replace('_', ' ')}</span>
                        <span className={cn(
                          "text-xs px-1.5 py-0.5 rounded-full",
                          flag.severity === 'critical' ? "bg-states-stress text-white" :
                          flag.severity === 'high' ? "bg-consciousness-accent text-white" :
                          "bg-yellow-500 text-white"
                        )}>
                          {flag.severity}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

// Main MessageViewer component
export const MessageViewer: React.FC<MessageViewerProps> = ({
  conversation,
  manualMode = false,
  onMessageSent,
  className
}) => {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [aiSuggestions, setAiSuggestions] = useState<GeneratedResponse[]>([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [claudeGenerating, setClaudeGenerating] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<List>(null);
  const [listHeight, setListHeight] = useState(400);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load conversation messages
  useEffect(() => {
    if (conversation.id) {
      loadMessages();
      loadAISuggestions();
    }
  }, [conversation.id]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update list height when container resizes
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setListHeight(rect.height - 150); // Account for input area
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Real-time message updates
  useKellyConversationUpdates(
    conversation.id,
    useCallback((update) => {
      if (update.payload.new_messages) {
        setMessages(prev => [...prev, ...update.payload.new_messages]);
      }
    }, [])
  );

  // Claude response generation monitoring
  useClaudeResponseGeneration(
    conversation.id,
    useCallback((update) => {
      const { status, thinking_process, partial_response } = update.payload;
      
      if (status === 'thinking') {
        setIsTyping(true);
        setClaudeGenerating(true);
      } else if (status === 'generating') {
        setIsTyping(true);
        // Could show partial response here
      } else if (status === 'complete') {
        setIsTyping(false);
        setClaudeGenerating(false);
        loadMessages(); // Refresh messages
        loadAISuggestions(); // Refresh suggestions
      } else if (status === 'error') {
        setIsTyping(false);
        setClaudeGenerating(false);
      }
    }, [])
  );

  const loadMessages = async () => {
    setIsLoadingMessages(true);
    try {
      const response = await fetch(`/api/v1/kelly/conversations/${conversation.id}/messages`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages || []);
      }
    } catch (error) {
      console.error('Failed to load messages:', error);
    } finally {
      setIsLoadingMessages(false);
    }
  };

  const loadAISuggestions = async () => {
    if (!manualMode) return;
    
    try {
      const response = await fetch(`/api/v1/kelly/conversations/${conversation.id}/suggestions`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setAiSuggestions(data.suggestions || []);
      }
    } catch (error) {
      console.error('Failed to load AI suggestions:', error);
    }
  };

  const sendMessage = async () => {
    if (!newMessage.trim()) return;

    try {
      const response = await fetch(`/api/v1/kelly/conversations/${conversation.id}/send`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          message: newMessage, 
          manual_override: true,
          sender: 'kelly'
        })
      });

      if (response.ok) {
        setNewMessage('');
        onMessageSent?.();
        loadMessages();
        loadAISuggestions();
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const useSuggestion = useCallback((suggestion: GeneratedResponse) => {
    setNewMessage(suggestion.content);
    setShowSuggestions(false);
  }, []);

  const handleMessageAction = useCallback((messageId: string, action: string) => {
    console.log(`Message ${messageId} action: ${action}`);
    // Handle message actions (copy, rate, flag, etc.)
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className={cn("flex flex-col h-full bg-white", className)}>
      {/* Messages area */}
      <div ref={containerRef} className="flex-1 overflow-hidden">
        {isLoadingMessages ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-consciousness-primary mx-auto mb-4"></div>
              <p className="text-sm text-text-tertiary">Loading messages...</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md mx-auto p-8">
              <div className="w-16 h-16 bg-consciousness-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <MessageCircle className="w-8 h-8 text-consciousness-primary" />
              </div>
              <h3 className="text-lg font-semibold text-text-primary mb-2">
                Start the Conversation
              </h3>
              <p className="text-text-secondary">
                No messages yet. {manualMode ? 'Send a message to begin.' : 'Kelly will start the conversation automatically.'}
              </p>
            </div>
          </div>
        ) : (
          <div className="h-full overflow-y-auto scrollbar-apple">
            <div className="py-4">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  conversation={conversation}
                  onAction={handleMessageAction}
                />
              ))}
              
              {/* Typing indicator */}
              <TypingIndicator visible={isTyping} />
              
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}
      </div>

      {/* Manual mode input area */}
      {manualMode && (
        <div className="border-t border-gray-200 p-4 bg-white">
          {/* AI Suggestions */}
          {aiSuggestions.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Brain className="h-4 w-4 text-consciousness-secondary" />
                  <span className="text-sm font-medium text-text-primary">AI Suggestions</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSuggestions(!showSuggestions)}
                  className="text-xs"
                >
                  {showSuggestions ? 'Hide' : 'Show'} ({aiSuggestions.length})
                </Button>
              </div>
              
              <AnimatePresence>
                {showSuggestions && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-2 max-h-32 overflow-y-auto"
                  >
                    {aiSuggestions.slice(0, 3).map((suggestion) => (
                      <motion.button
                        key={suggestion.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => useSuggestion(suggestion)}
                        className="w-full text-left p-3 bg-consciousness-primary/5 hover:bg-consciousness-primary/10 rounded-lg border border-consciousness-primary/20 transition-colors"
                      >
                        <div className="flex items-start justify-between">
                          <p className="text-sm text-text-primary flex-1 pr-2">
                            {suggestion.content}
                          </p>
                          <div className="flex items-center space-x-2 text-xs text-text-tertiary">
                            <Zap className="h-3 w-3" />
                            <span>{Math.round(suggestion.confidence_score)}%</span>
                          </div>
                        </div>
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-xs text-consciousness-primary capitalize">
                            {suggestion.response_type.replace('_', ' ')}
                          </span>
                          <span className="text-xs text-text-tertiary">
                            Quality: {Math.round(suggestion.estimated_quality)}%
                          </span>
                        </div>
                      </motion.button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Message input */}
          <div className="flex space-x-3">
            <div className="flex-1">
              <textarea
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={claudeGenerating ? "Claude is generating a response..." : "Type your message..."}
                disabled={claudeGenerating}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-consciousness-primary focus:border-transparent resize-none"
                rows={3}
              />
            </div>
            
            <div className="flex flex-col space-y-2">
              <Button
                onClick={sendMessage}
                disabled={!newMessage.trim() || claudeGenerating}
                variant="primary"
                size="lg"
                className="px-6"
              >
                {claudeGenerating ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
              
              <Button
                onClick={loadAISuggestions}
                variant="outline"
                size="sm"
                className="px-3"
                title="Refresh AI suggestions"
              >
                <Brain className="h-3 w-3" />
              </Button>
            </div>
          </div>
          
          {/* Quick stats */}
          <div className="flex items-center justify-between mt-3 text-xs text-text-tertiary">
            <div className="flex items-center space-x-4">
              <span>Messages: {messages.length}</span>
              <span>Stage: {conversation.stage.replace('_', ' ')}</span>
              <span>Safety: {conversation.safety_score}%</span>
            </div>
            
            <div className="flex items-center space-x-2">
              {claudeGenerating && (
                <div className="flex items-center space-x-1 text-consciousness-primary">
                  <Brain className="h-3 w-3 animate-pulse" />
                  <span>AI thinking...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageViewer;