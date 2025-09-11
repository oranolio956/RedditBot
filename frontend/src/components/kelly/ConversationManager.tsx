/**
 * Kelly Conversation Manager Component
 * Real-time conversation monitoring with Claude AI integration
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChatBubbleLeftRightIcon,
  UserIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  HandRaisedIcon,
  PauseIcon,
  PlayIcon,
  XMarkIcon,
  ArrowPathIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { useKellyStore } from '@/store';
import { apiClient } from '@/lib/api';
import { 
  useKellyConversationUpdates, 
  useClaudeResponseGeneration,
  useKellySafetyAlerts 
} from '@/lib/websocket';
import type { 
  ConversationMessage, 
  GeneratedResponse,
  SafetyAlert 
} from '@/types/kelly';

interface ConversationManagerProps {
  conversationId?: string;
  className?: string;
}

export default function ConversationManager({ 
  conversationId, 
  className = '' 
}: ConversationManagerProps) {
  const { selectedConversation, updateConversation } = useKellyStore();
  const currentConversation = conversationId 
    ? selectedConversation?.id === conversationId ? selectedConversation : null
    : selectedConversation;

  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [generatedResponses, setGeneratedResponses] = useState<GeneratedResponse[]>([]);
  const [isGeneratingResponse, setIsGeneratingResponse] = useState(false);
  const [safetyAlerts, setSafetyAlerts] = useState<SafetyAlert[]>([]);
  const [userMessage, setUserMessage] = useState('');
  const [responseGeneration, setResponseGeneration] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Real-time conversation updates
  useKellyConversationUpdates(currentConversation?.id || '', (update) => {
    if (update.payload.new_messages) {
      setMessages(prev => [...prev, ...update.payload.new_messages]);
      scrollToBottom();
    }
    if (update.payload.stage_change) {
      console.log('Conversation stage changed:', update.payload.stage_change);
    }
  });

  // Real-time Claude response generation
  useClaudeResponseGeneration(currentConversation?.id || '', (update) => {
    setResponseGeneration(update.payload);
    if (update.payload.status === 'complete') {
      setIsGeneratingResponse(false);
      // Reload generated responses
      loadGeneratedResponses();
    } else if (update.payload.status === 'generating') {
      setIsGeneratingResponse(true);
    }
  });

  // Safety alerts
  useKellySafetyAlerts((alert) => {
    if (alert.payload.conversation_id === currentConversation?.id) {
      setSafetyAlerts(prev => [alert.payload, ...prev]);
    }
  });

  // Load conversation data
  useEffect(() => {
    if (!currentConversation?.id) return;

    const loadConversationData = async () => {
      try {
        setIsLoading(true);
        const conversation = await apiClient.getKellyConversation(currentConversation.id);
        setMessages(conversation.recent_messages || []);
        await loadGeneratedResponses();
      } catch (error) {
        console.error('Failed to load conversation data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadConversationData();
  }, [currentConversation?.id]);

  const loadGeneratedResponses = async () => {
    if (!currentConversation?.id) return;
    
    try {
      // This would load pending responses for the conversation
      // Implementation depends on backend API
      setGeneratedResponses([]);
    } catch (error) {
      console.error('Failed to load generated responses:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim() || !currentConversation?.id) return;

    try {
      await apiClient.sendKellyMessage(currentConversation.id, userMessage, {
        useClaudeGeneration: true
      });
      setUserMessage('');
      setIsGeneratingResponse(true);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleGenerateResponse = async () => {
    if (!currentConversation?.id) return;

    try {
      setIsGeneratingResponse(true);
      await apiClient.generateClaudeResponse(currentConversation.id, 'Generate appropriate response');
    } catch (error) {
      console.error('Failed to generate response:', error);
      setIsGeneratingResponse(false);
    }
  };

  const handleSelectResponse = async (responseId: string) => {
    if (!currentConversation?.id) return;

    try {
      await apiClient.selectClaudeResponse(currentConversation.id, responseId);
      // Remove from generated responses
      setGeneratedResponses(prev => prev.filter(r => r.id !== responseId));
    } catch (error) {
      console.error('Failed to select response:', error);
    }
  };

  const handlePauseConversation = async () => {
    if (!currentConversation?.id) return;

    try {
      await apiClient.pauseKellyConversation(currentConversation.id, 'Manual pause');
      updateConversation(currentConversation.id, { status: 'paused' });
    } catch (error) {
      console.error('Failed to pause conversation:', error);
    }
  };

  const handleResumeConversation = async () => {
    if (!currentConversation?.id) return;

    try {
      await apiClient.resumeKellyConversation(currentConversation.id);
      updateConversation(currentConversation.id, { status: 'active' });
    } catch (error) {
      console.error('Failed to resume conversation:', error);
    }
  };

  const handleEscalateConversation = async () => {
    if (!currentConversation?.id) return;

    try {
      await apiClient.escalateKellyConversation(currentConversation.id, 'Manual escalation');
      updateConversation(currentConversation.id, { status: 'escalated' });
    } catch (error) {
      console.error('Failed to escalate conversation:', error);
    }
  };

  if (!currentConversation) {
    return (
      <Card className="p-6">
        <div className="text-center text-text-secondary">
          <ChatBubbleLeftRightIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select a conversation to start monitoring</p>
        </div>
      </Card>
    );
  }

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'initial_contact': return 'bg-blue-100 text-blue-800';
      case 'rapport_building': return 'bg-green-100 text-green-800';
      case 'qualification': return 'bg-yellow-100 text-yellow-800';
      case 'engagement': return 'bg-purple-100 text-purple-800';
      case 'advanced_engagement': return 'bg-pink-100 text-pink-800';
      case 'payment_discussion': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-states-flow/20 text-states-flow';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      case 'escalated': return 'bg-states-stress/20 text-states-stress';
      case 'ended': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className={`h-full flex flex-col ${className}`}>
      {/* Conversation Header */}
      <Card className="flex-shrink-0 mb-4">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-full bg-consciousness-primary/10 flex items-center justify-center">
                <UserIcon className="w-6 h-6 text-consciousness-primary" />
              </div>
              <div>
                <CardTitle className="text-lg">
                  {currentConversation.user_info.first_name || 'Unknown User'}
                </CardTitle>
                <div className="flex items-center space-x-3 text-sm">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStageColor(currentConversation.stage)}`}>
                    {currentConversation.stage.replace('_', ' ')}
                  </span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(currentConversation.status)}`}>
                    {currentConversation.status}
                  </span>
                  <span className="text-text-tertiary">
                    {currentConversation.message_count} messages
                  </span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              {/* Conversation Controls */}
              {currentConversation.status === 'active' && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePauseConversation}
                  className="text-yellow-600 hover:text-yellow-700"
                >
                  <PauseIcon className="w-4 h-4" />
                </Button>
              )}
              
              {currentConversation.status === 'paused' && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleResumeConversation}
                  className="text-green-600 hover:text-green-700"
                >
                  <PlayIcon className="w-4 h-4" />
                </Button>
              )}
              
              <Button
                variant="outline"
                size="sm"
                onClick={handleEscalateConversation}
                className="text-orange-600 hover:text-orange-700"
              >
                <HandRaisedIcon className="w-4 h-4" />
              </Button>
              
              {/* Metrics */}
              <div className="flex items-center space-x-4 text-sm text-text-secondary">
                <div className="flex items-center space-x-1">
                  <ShieldCheckIcon className="w-4 h-4" />
                  <span>{Math.round(currentConversation.safety_score)}% safe</span>
                </div>
                <div className="flex items-center space-x-1">
                  <CheckCircleIcon className="w-4 h-4" />
                  <span>{Math.round(currentConversation.engagement_score)}% engaged</span>
                </div>
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Safety Alerts */}
      <AnimatePresence>
        {safetyAlerts.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-4"
          >
            <Card variant="default" className="border-red-200 bg-red-50">
              <CardContent className="p-4">
                <div className="flex items-center space-x-3">
                  <ExclamationTriangleIcon className="w-6 h-6 text-red-600" />
                  <div className="flex-1">
                    <h4 className="font-medium text-red-800">Safety Alert</h4>
                    <p className="text-sm text-red-600">
                      {safetyAlerts[0].payload.description}
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSafetyAlerts([])}
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages Area */}
      <Card className="flex-1 flex flex-col min-h-0">
        <CardContent className="flex-1 flex flex-col min-h-0 p-4">
          {isLoading ? (
            <div className="flex-1 flex items-center justify-center">
              <LoadingSpinner size="lg" text="Loading conversation..." />
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto space-y-4 mb-4">
              {messages.map((message, index) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex ${message.sender === 'kelly' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-xs lg:max-w-md xl:max-w-lg px-4 py-2 rounded-lg ${
                    message.sender === 'kelly' 
                      ? 'bg-consciousness-primary text-white'
                      : 'bg-surface-secondary text-text-primary'
                  }`}>
                    <p className="text-sm">{message.content}</p>
                    <div className="flex items-center justify-between mt-2 text-xs opacity-75">
                      <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                      {message.sender === 'kelly' && message.ai_confidence && (
                        <span>{Math.round(message.ai_confidence * 100)}% conf</span>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}

              {/* Claude Response Generation Status */}
              {responseGeneration && responseGeneration.status === 'thinking' && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-end"
                >
                  <div className="max-w-xs lg:max-w-md xl:max-w-lg px-4 py-2 rounded-lg bg-consciousness-primary/20 border border-consciousness-primary/30">
                    <div className="flex items-center space-x-2">
                      <CpuChipIcon className="w-4 h-4 text-consciousness-primary animate-pulse" />
                      <span className="text-sm text-consciousness-primary">
                        Claude is thinking...
                      </span>
                    </div>
                    {responseGeneration.thinking_process && (
                      <p className="text-xs text-consciousness-primary/80 mt-2">
                        {responseGeneration.thinking_process}
                      </p>
                    )}
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Generated Responses */}
          {generatedResponses.length > 0 && (
            <div className="border-t pt-4 mb-4">
              <h4 className="text-sm font-medium mb-3 text-text-secondary">
                Generated Responses ({generatedResponses.length})
              </h4>
              <div className="space-y-2">
                {generatedResponses.map((response) => (
                  <motion.div
                    key={response.id}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-start space-x-3 p-3 bg-surface-secondary/50 rounded-lg"
                  >
                    <div className="flex-1">
                      <p className="text-sm">{response.content}</p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-text-tertiary">
                        <span>Confidence: {Math.round(response.confidence_score * 100)}%</span>
                        <span>Quality: {Math.round(response.estimated_quality * 100)}%</span>
                        {response.claude_metadata && (
                          <>
                            <span>Model: {response.claude_metadata.model_used}</span>
                            <span>Cost: ${response.claude_metadata.cost_usd.toFixed(4)}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => handleSelectResponse(response.id)}
                    >
                      Select
                    </Button>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Message Input */}
          <div className="border-t pt-4">
            <div className="flex space-x-3">
              <div className="flex-1">
                <textarea
                  ref={textareaRef}
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  placeholder="Type a message to simulate user input..."
                  className="w-full p-3 border border-surface-tertiary rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-consciousness-primary/20"
                  rows={2}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                />
              </div>
              <div className="flex flex-col space-y-2">
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleSendMessage}
                  disabled={!userMessage.trim()}
                >
                  Send
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleGenerateResponse}
                  disabled={isGeneratingResponse}
                  className="text-consciousness-primary hover:text-consciousness-secondary"
                >
                  {isGeneratingResponse ? (
                    <ArrowPathIcon className="w-4 h-4 animate-spin" />
                  ) : (
                    <CpuChipIcon className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}