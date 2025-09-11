/**
 * Kelly AI Conversation Manager V2 - Phase 1 Page
 * Full-screen conversation management interface with WhatsApp Web/Slack UX patterns
 */

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  MessageSquare, 
  Settings, 
  AlertTriangle, 
  Users, 
  Activity,
  BarChart3,
  Shield,
  Clock,
  Zap,
  Brain
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { KellyConversation, SafetyAlert } from '@/types/kelly';
import { useKellyStore } from '@/store';
import { KellyErrorBoundary } from '@/components/ui/ErrorBoundary';
import ConversationManagement from './ConversationManagement';

const ConversationManagerV2Page: React.FC = () => {
  const { activeConversations, dashboardOverview } = useKellyStore();
  const [showHeader, setShowHeader] = useState(true);
  const [selectedConversation, setSelectedConversation] = useState<KellyConversation | null>(null);

  const handleConversationSelect = useCallback((conversation: KellyConversation) => {
    setSelectedConversation(conversation);
  }, []);

  const handleSafetyAlert = useCallback((alert: SafetyAlert) => {
    // Handle safety alerts
    console.warn('Safety Alert:', alert);
  }, []);

  const stats = {
    total: activeConversations?.length || 0,
    active: activeConversations?.filter(c => c.status === 'active').length || 0,
    flagged: activeConversations?.filter(c => c.red_flags.length > 0).length || 0,
    needsReview: activeConversations?.filter(c => c.requires_human_review).length || 0,
    avgEngagement: activeConversations?.length > 0 
      ? Math.round(activeConversations.reduce((sum, c) => sum + c.engagement_score, 0) / activeConversations.length)
      : 0,
    avgSafety: activeConversations?.length > 0
      ? Math.round(activeConversations.reduce((sum, c) => sum + c.safety_score, 0) / activeConversations.length)
      : 0
  };

  return (
    <KellyErrorBoundary>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {/* Optional Header - Can be hidden for full-screen mode */}
        {showHeader && (
          <motion.header
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white border-b border-gray-200 px-6 py-4"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                    <MessageSquare className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">
                      Kelly AI Conversations
                    </h1>
                    <p className="text-sm text-gray-600">
                      Phase 1 - Real-time conversation management
                    </p>
                  </div>
                </div>
              </div>

              {/* Quick Stats */}
              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
                  <div className="text-sm text-gray-500">Total</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{stats.active}</div>
                  <div className="text-sm text-gray-500">Active</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-600">{stats.flagged}</div>
                  <div className="text-sm text-gray-500">Flagged</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-600">{stats.needsReview}</div>
                  <div className="text-sm text-gray-500">Review</div>
                </div>

                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setShowHeader(false)}
                >
                  Full Screen
                </Button>
              </div>
            </div>
          </motion.header>
        )}

        {/* Main Conversation Interface */}
        <div className="flex-1">
          <ConversationManagement />
        </div>

        {/* Full Screen Toggle - Hidden Header Mode */}
        {!showHeader && (
          <button
            onClick={() => setShowHeader(true)}
            className="fixed top-4 right-4 z-50 bg-white shadow-lg rounded-full p-2 hover:bg-gray-50 transition-colors"
          >
            <Settings className="w-5 h-5 text-gray-600" />
          </button>
        )}
      </div>
    </KellyErrorBoundary>
  );
};

export default ConversationManagerV2Page;