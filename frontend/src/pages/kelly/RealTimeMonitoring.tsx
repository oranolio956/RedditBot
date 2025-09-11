/**
 * Kelly Real-Time Monitoring Page
 * Comprehensive dashboard showcasing Phase 2 components
 * Apple-inspired design with fluid animations and real-time updates
 */

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import {
  RealTimeMonitoringDashboard,
  LiveActivityFeed,
  InterventionControlsPanel,
  AlertManagementSystem,
  LiveStatusIndicators,
  MiniStatusIndicator,
  EmergencyOverridePanel
} from '@/components/kelly';
import { useWebSocket } from '@/lib/websocket';
import { KellyConversation, SafetyAlert } from '@/types/kelly';
import { cn } from '@/lib/utils';

interface ViewState {
  activeView: 'overview' | 'activities' | 'alerts' | 'intervention' | 'emergency';
  selectedConversation: KellyConversation | null;
  showSidebar: boolean;
}

const RealTimeMonitoringPage: React.FC = () => {
  const [viewState, setViewState] = useState<ViewState>({
    activeView: 'overview',
    selectedConversation: null,
    showSidebar: true
  });
  const [notifications, setNotifications] = useState<any[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const ws = useWebSocket();

  // Connect WebSocket on mount
  useEffect(() => {
    if (!ws.connected) {
      ws.connect();
    }
  }, [ws]);

  // Handle drill-down from monitoring dashboard
  const handleDashboardDrillDown = useCallback((metric: string, value: any) => {
    console.log('Drill down:', metric, value);
    
    switch (metric) {
      case 'active_conversations':
        setViewState(prev => ({ ...prev, activeView: 'activities' }));
        break;
      case 'safety_score':
        setViewState(prev => ({ ...prev, activeView: 'alerts' }));
        break;
      default:
        // Show detailed view for the specific metric
        break;
    }
  }, []);

  // Handle activity item clicks
  const handleActivityClick = useCallback((item: any) => {
    if (item.conversationId) {
      // Load conversation details and show intervention panel
      setViewState(prev => ({ 
        ...prev, 
        activeView: 'intervention',
        selectedConversation: { 
          id: item.conversationId,
          // Mock conversation data - would be loaded from API
          account_id: item.accountId || 'mock-account',
          telegram_user_id: 12345,
          chat_id: -1001234567,
          user_info: {
            user_id: 12345,
            first_name: item.user?.name || 'Unknown User',
            is_premium: false
          },
          stage: 'engagement',
          status: 'active',
          message_count: 25,
          started_at: new Date().toISOString(),
          last_activity: new Date().toISOString(),
          engagement_score: 78,
          safety_score: 92,
          progression_score: 65,
          ai_confidence: 85,
          red_flags: [],
          requires_human_review: false,
          conversation_context: {
            preferred_communication_style: 'casual',
            interests: ['technology', 'music'],
            conversation_history_summary: 'User interested in AI and technology',
            key_topics: ['AI', 'technology'],
            emotional_state_history: [],
            response_patterns: [],
            availability_patterns: []
          },
          user_personality_profile: {
            detected_traits: {
              openness: 0.7,
              conscientiousness: 0.6,
              extraversion: 0.8,
              agreeableness: 0.9,
              neuroticism: 0.3
            },
            message_length_preference: 'medium',
            emoji_usage: 0.6,
            response_speed: 'medium',
            conversation_style: 'playful',
            online_activity_pattern: [],
            topic_preferences: [],
            engagement_triggers: [],
            disengagement_signals: [],
            confidence_level: 0.8,
            last_updated: new Date().toISOString()
          },
          topics_discussed: ['AI', 'technology', 'music'],
          recent_messages: []
        } as KellyConversation
      }));
    }
  }, []);

  // Handle alert clicks
  const handleAlertClick = useCallback((alert: SafetyAlert) => {
    if (alert.conversation_id) {
      handleActivityClick({ 
        conversationId: alert.conversation_id,
        accountId: alert.account_id
      });
    }
  }, [handleActivityClick]);

  // Navigation items
  const navigationItems = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'activities', label: 'Live Feed', icon: '🔥' },
    { id: 'alerts', label: 'Alerts', icon: '🚨' },
    { id: 'intervention', label: 'Controls', icon: '🎛️' },
    { id: 'emergency', label: 'Emergency', icon: '🛑' }
  ] as const;

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  return (
    <div className={cn(
      "min-h-screen bg-gradient-to-br from-surface-primary to-surface-secondary",
      isFullscreen && "fixed inset-0 z-50"
    )}>
      {/* Header */}
      <div className="bg-surface-primary/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-40">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-insight-title text-consciousness-primary font-sf-pro">
                Kelly Real-Time Monitoring
              </h1>
              
              {viewState.selectedConversation && (
                <MiniStatusIndicator
                  conversationId={viewState.selectedConversation.id}
                  className="ml-4"
                />
              )}
            </div>

            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <div className={cn(
                  "h-2 w-2 rounded-full transition-colors",
                  ws.connected ? "bg-states-flow animate-pulse" : "bg-states-stress"
                )} />
                <span className="text-caption-text text-text-tertiary">
                  {ws.connected ? 'Live' : 'Disconnected'}
                </span>
              </div>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => setViewState(prev => ({ ...prev, showSidebar: !prev.showSidebar }))}
              >
                {viewState.showSidebar ? '←' : '→'}
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={toggleFullscreen}
              >
                {isFullscreen ? '⤷' : '⤢'}
              </Button>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center space-x-1 mt-4">
            {navigationItems.map((item) => (
              <Button
                key={item.id}
                variant={viewState.activeView === item.id ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => setViewState(prev => ({ ...prev, activeView: item.id }))}
                className="flex items-center space-x-2"
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </Button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Sidebar */}
        {viewState.showSidebar && (
          <div className="w-80 bg-surface-primary border-r border-gray-200 h-[calc(100vh-140px)] overflow-y-auto">
            <div className="p-4 space-y-4">
              {/* Quick stats */}
              <Card variant="consciousness" glassmorphism className="p-4">
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-metric-value text-consciousness-primary">24</div>
                    <div className="text-caption-text text-text-tertiary">Active</div>
                  </div>
                  <div>
                    <div className="text-metric-value text-consciousness-accent">3</div>
                    <div className="text-caption-text text-text-tertiary">Alerts</div>
                  </div>
                </div>
              </Card>

              {/* Activity preview */}
              {viewState.activeView !== 'activities' && (
                <Card variant="consciousness" glassmorphism>
                  <CardHeader>
                    <CardTitle className="text-body-text">Recent Activity</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {[1, 2, 3].map(i => (
                      <div key={i} className="flex items-center space-x-3 p-2 rounded bg-surface-secondary">
                        <div className="h-2 w-2 bg-consciousness-primary rounded-full" />
                        <div className="flex-1 text-caption-text text-text-secondary">
                          New message in conversation #{i}
                        </div>
                        <div className="text-caption-text text-text-tertiary">
                          {i}m
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}

        {/* Main content */}
        <div className="flex-1 p-6 overflow-y-auto h-[calc(100vh-140px)]">
          {viewState.activeView === 'overview' && (
            <RealTimeMonitoringDashboard
              onDrillDown={handleDashboardDrillDown}
              refreshInterval={30000}
            />
          )}

          {viewState.activeView === 'activities' && (
            <LiveActivityFeed
              maxItems={100}
              onItemClick={handleActivityClick}
              onItemPreview={(item) => {
                // Show preview tooltip or sidebar
                console.log('Preview:', item);
              }}
              groupByTime={true}
            />
          )}

          {viewState.activeView === 'alerts' && (
            <AlertManagementSystem
              maxAlerts={100}
              groupSimilarAlerts={true}
              soundNotifications={true}
              onAlertClick={handleAlertClick}
              onAlertResolved={(alertId) => {
                console.log('Alert resolved:', alertId);
              }}
              onEscalateAlert={(alertId, escalateTo) => {
                console.log('Alert escalated:', alertId, escalateTo);
              }}
            />
          )}

          {viewState.activeView === 'intervention' && viewState.selectedConversation && (
            <div className="space-y-6">
              <Card variant="consciousness" glassmorphism className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-insight-subtitle text-consciousness-primary">
                      Conversation Control
                    </h2>
                    <p className="text-body-text text-text-secondary mt-1">
                      ID: {viewState.selectedConversation.id}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    onClick={() => setViewState(prev => ({ ...prev, selectedConversation: null }))}
                  >
                    ✕
                  </Button>
                </div>
              </Card>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <LiveStatusIndicators
                    conversationId={viewState.selectedConversation.id}
                    showEstimatedTime={true}
                    onStatusChange={(status) => {
                      console.log('Status changed:', status);
                    }}
                  />
                </div>

                <InterventionControlsPanel
                  conversationId={viewState.selectedConversation.id}
                  conversation={viewState.selectedConversation}
                  onInterventionChange={(state) => {
                    console.log('Intervention state changed:', state);
                  }}
                  onActionComplete={(action, success) => {
                    console.log('Action completed:', action, success);
                    if (success) {
                      setNotifications(prev => [...prev, {
                        id: Date.now(),
                        type: 'success',
                        message: `${action} completed successfully`
                      }]);
                    }
                  }}
                />
              </div>
            </div>
          )}

          {viewState.activeView === 'emergency' && (
            <EmergencyOverridePanel
              conversationId={viewState.selectedConversation?.id}
              onOverrideComplete={(action) => {
                console.log('Override completed:', action);
                setNotifications(prev => [...prev, {
                  id: Date.now(),
                  type: 'warning',
                  message: `${action.type} override executed`
                }]);
              }}
              onOverrideError={(error) => {
                console.error('Override error:', error);
                setNotifications(prev => [...prev, {
                  id: Date.now(),
                  type: 'error',
                  message: `Override failed: ${error}`
                }]);
              }}
            />
          )}
        </div>
      </div>

      {/* Toast notifications */}
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        {notifications.slice(-3).map((notification) => (
          <Card
            key={notification.id}
            variant="breakthrough"
            className={cn(
              "p-4 animate-insight-arrive shadow-dramatic",
              notification.type === 'success' && "bg-green-50 border-green-200",
              notification.type === 'warning' && "bg-yellow-50 border-yellow-200",
              notification.type === 'error' && "bg-red-50 border-red-200"
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-body-text">{notification.message}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setNotifications(prev => prev.filter(n => n.id !== notification.id));
                }}
              >
                ✕
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default RealTimeMonitoringPage;