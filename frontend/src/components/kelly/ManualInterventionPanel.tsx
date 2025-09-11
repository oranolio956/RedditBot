/**
 * Manual Intervention Panel Component
 * Emergency controls for Kelly AI with human oversight capabilities
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  X,
  AlertTriangle,
  Pause,
  Play,
  StopCircle,
  Shield,
  MessageCircle,
  Clock,
  Users,
  Send,
  CheckCircle,
  RefreshCw
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useKellyStore } from '@/store';

interface ManualInterventionPanelProps {
  conversation?: any;
  manualMode?: boolean;
  onManualModeToggle?: (enabled: boolean) => void;
  className?: string;
}

interface QuickReply {
  id: string;
  text: string;
  category: 'safety' | 'redirect' | 'closure' | 'escalation';
  icon: React.ReactNode;
}

const ManualInterventionPanel: React.FC<ManualInterventionPanelProps> = ({ 
  onManualModeToggle
}) => {
  const { selectedConversation, activeConversations, setActiveConversations } = useKellyStore();
  const [isKellyPaused, setIsKellyPaused] = useState(false);
  const [emergencyMode, setEmergencyMode] = useState(false);
  const [customMessage, setCustomMessage] = useState('');
  const [selectedQuickReply, setSelectedQuickReply] = useState<string | null>(null);
  const [interventionLog, setInterventionLog] = useState<Array<{
    id: string;
    timestamp: Date;
    action: string;
    conversation_id?: string;
  }>>([]);

  const quickReplies: QuickReply[] = [
    {
      id: 'safety_pause',
      text: "I appreciate your interest, but I need to take a break from our conversation right now. Have a great day!",
      category: 'safety',
      icon: <Shield className="h-4 w-4" />
    },
    {
      id: 'redirect_topic',
      text: "That's an interesting topic! I'd love to hear more about your thoughts on [topic]. What got you interested in that?",
      category: 'redirect',
      icon: <MessageCircle className="h-4 w-4" />
    },
    {
      id: 'polite_closure',
      text: "It's been really nice chatting with you! I hope you have a wonderful rest of your day.",
      category: 'closure',
      icon: <CheckCircle className="h-4 w-4" />
    },
    {
      id: 'escalation_human',
      text: "Let me connect you with someone who can better help you with that specific question.",
      category: 'escalation',
      icon: <Users className="h-4 w-4" />
    },
    {
      id: 'boundary_setting',
      text: "I prefer to keep our conversation friendly and appropriate. What else would you like to talk about?",
      category: 'safety',
      icon: <Shield className="h-4 w-4" />
    },
    {
      id: 'time_boundary',
      text: "I should probably get some rest soon, but I'm enjoying our conversation! What's one thing that made you smile today?",
      category: 'redirect',
      icon: <Clock className="h-4 w-4" />
    }
  ];

  useEffect(() => {
    checkKellyStatus();
    loadInterventionLog();
  }, []);

  const checkKellyStatus = async () => {
    try {
      const response = await fetch('/api/v1/kelly/status');
      const data = await response.json();
      setIsKellyPaused(data.is_paused || false);
      setEmergencyMode(data.emergency_mode || false);
    } catch (error) {
      console.error('Failed to check Kelly status:', error);
    }
  };

  const loadInterventionLog = async () => {
    try {
      const response = await fetch('/api/v1/kelly/interventions/recent');
      const data = await response.json();
      setInterventionLog(data.interventions || []);
    } catch (error) {
      console.error('Failed to load intervention log:', error);
    }
  };

  const pauseKellyGlobally = async () => {
    try {
      const response = await fetch('/api/v1/kelly/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'manual_intervention' })
      });

      if (response.ok) {
        setIsKellyPaused(true);
        logIntervention('Global Kelly pause activated');
      }
    } catch (error) {
      console.error('Failed to pause Kelly:', error);
    }
  };

  const resumeKellyGlobally = async () => {
    try {
      const response = await fetch('/api/v1/kelly/resume', {
        method: 'POST'
      });

      if (response.ok) {
        setIsKellyPaused(false);
        logIntervention('Global Kelly pause deactivated');
      }
    } catch (error) {
      console.error('Failed to resume Kelly:', error);
    }
  };

  const activateEmergencyStop = async () => {
    try {
      const response = await fetch('/api/v1/kelly/emergency-stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'manual_emergency_stop' })
      });

      if (response.ok) {
        setEmergencyMode(true);
        setIsKellyPaused(true);
        logIntervention('EMERGENCY STOP activated - All conversations terminated');
        
        // Update local state to show all conversations as ended
        setActiveConversations([]);
      }
    } catch (error) {
      console.error('Failed to activate emergency stop:', error);
    }
  };

  const sendQuickReply = async (replyId: string) => {
    if (!selectedConversation) return;

    const reply = quickReplies.find(r => r.id === replyId);
    if (!reply) return;

    try {
      const response = await fetch(`/api/v1/kelly/conversations/${selectedConversation.id}/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: reply.text, 
          manual_override: true,
          intervention_type: reply.category
        })
      });

      if (response.ok) {
        setSelectedQuickReply(replyId);
        logIntervention(`Quick reply sent: ${reply.category}`, selectedConversation.id);
        setTimeout(() => setSelectedQuickReply(null), 2000);
      }
    } catch (error) {
      console.error('Failed to send quick reply:', error);
    }
  };

  const sendCustomMessage = async () => {
    if (!selectedConversation || !customMessage.trim()) return;

    try {
      const response = await fetch(`/api/v1/kelly/conversations/${selectedConversation.id}/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: customMessage, 
          manual_override: true,
          intervention_type: 'custom'
        })
      });

      if (response.ok) {
        setCustomMessage('');
        logIntervention(`Custom message sent`, selectedConversation.id);
      }
    } catch (error) {
      console.error('Failed to send custom message:', error);
    }
  };

  const logIntervention = (action: string, conversationId?: string) => {
    const newEntry = {
      id: Date.now().toString(),
      timestamp: new Date(),
      action,
      conversation_id: conversationId
    };
    setInterventionLog(prev => [newEntry, ...prev.slice(0, 9)]);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'safety': return 'bg-red-50 border-red-200 hover:bg-red-100';
      case 'redirect': return 'bg-blue-50 border-blue-200 hover:bg-blue-100';
      case 'closure': return 'bg-green-50 border-green-200 hover:bg-green-100';
      case 'escalation': return 'bg-yellow-50 border-yellow-200 hover:bg-yellow-100';
      default: return 'bg-gray-50 border-gray-200 hover:bg-gray-100';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
      onClick={() => onManualModeToggle?.(false)}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-red-100 rounded-full">
                <Shield className="h-6 w-6 text-red-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">Manual Intervention Controls</h2>
                <p className="text-sm text-gray-600">Emergency controls and human oversight for Kelly AI</p>
              </div>
            </div>
            <button
              onClick={() => onManualModeToggle?.(false)}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-6 w-6" />
            </button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Global Controls */}
          <Card>
            <div className="p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Global Controls</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Kelly Status */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <div className="text-sm font-medium text-gray-900">Kelly Status</div>
                    <div className={`text-xs ${isKellyPaused ? 'text-red-600' : 'text-green-600'}`}>
                      {isKellyPaused ? 'Paused' : 'Active'}
                    </div>
                  </div>
                  <div className={`w-3 h-3 rounded-full ${isKellyPaused ? 'bg-red-500' : 'bg-green-500'}`} />
                </div>

                {/* Active Conversations */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <div className="text-sm font-medium text-gray-900">Active Conversations</div>
                    <div className="text-xs text-gray-600">{activeConversations.length} ongoing</div>
                  </div>
                  <MessageCircle className="h-5 w-5 text-blue-500" />
                </div>

                {/* Emergency Mode */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <div className="text-sm font-medium text-gray-900">Emergency Mode</div>
                    <div className={`text-xs ${emergencyMode ? 'text-red-600' : 'text-gray-600'}`}>
                      {emergencyMode ? 'Active' : 'Inactive'}
                    </div>
                  </div>
                  <AlertTriangle className={`h-5 w-5 ${emergencyMode ? 'text-red-500' : 'text-gray-400'}`} />
                </div>
              </div>

              <div className="mt-4 flex items-center space-x-3">
                {!isKellyPaused ? (
                  <Button
                    onClick={pauseKellyGlobally}
                    variant="outline"
                    className="text-yellow-600 hover:text-yellow-700"
                  >
                    <Pause className="h-4 w-4 mr-2" />
                    Pause Kelly
                  </Button>
                ) : (
                  <Button
                    onClick={resumeKellyGlobally}
                    variant="outline"
                    className="text-green-600 hover:text-green-700"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Resume Kelly
                  </Button>
                )}

                <Button
                  onClick={activateEmergencyStop}
                  variant="outline"
                  className="text-red-600 hover:text-red-700"
                  disabled={emergencyMode}
                >
                  <StopCircle className="h-4 w-4 mr-2" />
                  Emergency Stop
                </Button>
              </div>
            </div>
          </Card>

          {/* Quick Replies */}
          {selectedConversation && (
            <Card>
              <div className="p-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Quick Replies for {selectedConversation.user_info.username || 'Selected Conversation'}
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {quickReplies.map((reply) => (
                    <button
                      key={reply.id}
                      onClick={() => sendQuickReply(reply.id)}
                      disabled={selectedQuickReply === reply.id}
                      className={`p-3 text-left border rounded-lg transition-all ${
                        selectedQuickReply === reply.id 
                          ? 'bg-green-100 border-green-300' 
                          : getCategoryColor(reply.category)
                      }`}
                    >
                      <div className="flex items-start space-x-2">
                        <div className="flex-shrink-0 mt-0.5">
                          {selectedQuickReply === reply.id ? (
                            <CheckCircle className="h-4 w-4 text-green-600" />
                          ) : (
                            reply.icon
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium text-gray-600 capitalize mb-1">
                            {reply.category.replace('_', ' ')}
                          </div>
                          <div className="text-sm text-gray-900">
                            {reply.text}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Custom Message */}
                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Custom Message
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="text"
                      value={customMessage}
                      onChange={(e) => setCustomMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && sendCustomMessage()}
                      placeholder="Type a custom intervention message..."
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <Button onClick={sendCustomMessage} disabled={!customMessage.trim()}>
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* Intervention Log */}
          <Card>
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Recent Interventions</h3>
                <Button onClick={loadInterventionLog} variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </div>
              
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {interventionLog.length > 0 ? (
                  interventionLog.map((entry) => (
                    <div key={entry.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex-1">
                        <div className="text-sm font-medium text-gray-900">{entry.action}</div>
                        <div className="text-xs text-gray-500">
                          {entry.timestamp.toLocaleString()}
                          {entry.conversation_id && ` • Conversation: ${entry.conversation_id.slice(-8)}`}
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No recent interventions</p>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Footer */}
        <div className="p-4 bg-gray-50 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-500">
              All interventions are logged for safety and compliance
            </div>
            <Button 
              onClick={() => onManualModeToggle?.(false)} 
              variant="outline"
            >
              Close Panel
            </Button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ManualInterventionPanel;