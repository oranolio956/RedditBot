/**
 * Live Activity Feed for Kelly
 * Chronological timeline with smart grouping, VIP notifications, and real-time updates
 * Apple-inspired design with smooth animations
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useWebSocket, useKellyConversationUpdates } from '@/lib/websocket';
import { ConversationMessage, KellyConversation, SafetyAlert } from '@/types/kelly';
import { cn } from '@/lib/utils';

interface ActivityItem {
  id: string;
  type: 'new_conversation' | 'message' | 'escalation' | 'intervention' | 'safety_alert' | 'ai_insight';
  timestamp: Date;
  title: string;
  description: string;
  metadata: any;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  isVIP?: boolean;
  conversationId?: string;
  accountId?: string;
  user?: {
    name?: string;
    avatar?: string;
  };
}

interface ActivityGroupProps {
  title: string;
  items: ActivityItem[];
  isVIP?: boolean;
  onItemClick: (item: ActivityItem) => void;
  onItemHover: (item: ActivityItem | null) => void;
}

const ActivityGroup: React.FC<ActivityGroupProps> = ({
  title,
  items,
  isVIP = false,
  onItemClick,
  onItemHover
}) => {
  if (items.length === 0) return null;

  return (
    <div className={cn(
      "space-y-3",
      isVIP && "order-first"
    )}>
      <div className="flex items-center space-x-2 sticky top-0 bg-surface-primary/80 backdrop-blur-sm py-2 z-10">
        {isVIP && <span className="text-consciousness-accent">⭐</span>}
        <h3 className="text-insight-subtitle text-consciousness-primary font-sf-pro">
          {title}
        </h3>
        <span className="bg-consciousness-primary/10 text-consciousness-primary px-2 py-1 rounded-full text-caption-text">
          {items.length}
        </span>
      </div>

      <div className="space-y-2">
        {items.map((item, index) => (
          <ActivityItemComponent
            key={item.id}
            item={item}
            index={index}
            onClick={() => onItemClick(item)}
            onMouseEnter={() => onItemHover(item)}
            onMouseLeave={() => onItemHover(null)}
          />
        ))}
      </div>
    </div>
  );
};

interface ActivityItemComponentProps {
  item: ActivityItem;
  index: number;
  onClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const ActivityItemComponent: React.FC<ActivityItemComponentProps> = ({
  item,
  index,
  onClick,
  onMouseEnter,
  onMouseLeave
}) => {
  const getActivityIcon = useMemo(() => {
    switch (item.type) {
      case 'new_conversation': return '💬';
      case 'message': return '💌';
      case 'escalation': return '⚠️';
      case 'intervention': return '🚨';
      case 'safety_alert': return '🛡️';
      case 'ai_insight': return '🧠';
      default: return '📝';
    }
  }, [item.type]);

  const getSeverityColor = useMemo(() => {
    switch (item.severity) {
      case 'critical': return 'border-l-states-stress bg-red-50/50';
      case 'high': return 'border-l-consciousness-accent bg-orange-50/50';
      case 'medium': return 'border-l-yellow-500 bg-yellow-50/50';
      case 'low': return 'border-l-gray-400 bg-gray-50/50';
      default: return 'border-l-consciousness-primary bg-blue-50/50';
    }
  }, [item.severity]);

  const timeAgo = useMemo(() => {
    const now = new Date();
    const diff = now.getTime() - item.timestamp.getTime();
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  }, [item.timestamp]);

  return (
    <div
      className={cn(
        "group relative p-4 rounded-lg border-l-4 cursor-pointer transition-all duration-200 ease-in-out",
        "hover:shadow-elevated hover:-translate-y-0.5 hover:scale-[1.02]",
        "animate-insight-arrive",
        getSeverityColor,
        item.isVIP && "ring-2 ring-consciousness-accent ring-opacity-50"
      )}
      style={{ animationDelay: `${index * 100}ms` }}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <span className="text-xl">{getActivityIcon}</span>
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="text-body-text font-medium text-consciousness-primary group-hover:text-consciousness-accent transition-colors">
                {item.title}
              </p>
              <p className="text-caption-text text-text-secondary mt-1 line-clamp-2">
                {item.description}
              </p>
              
              {item.user && (
                <div className="flex items-center space-x-2 mt-2">
                  {item.user.avatar ? (
                    <img
                      src={item.user.avatar}
                      alt={item.user.name}
                      className="h-5 w-5 rounded-full"
                    />
                  ) : (
                    <div className="h-5 w-5 rounded-full bg-consciousness-primary/20 flex items-center justify-center">
                      <span className="text-caption-text text-consciousness-primary">
                        {item.user.name?.charAt(0) || 'U'}
                      </span>
                    </div>
                  )}
                  <span className="text-caption-text text-text-tertiary">
                    {item.user.name || 'Unknown User'}
                  </span>
                </div>
              )}
            </div>
            
            <div className="flex flex-col items-end space-y-1">
              <span className="text-caption-text text-text-tertiary">
                {timeAgo}
              </span>
              
              {item.isVIP && (
                <span className="bg-consciousness-accent text-white px-2 py-0.5 rounded-full text-caption-text">
                  VIP
                </span>
              )}
              
              {item.severity && (
                <span className={cn(
                  "px-2 py-0.5 rounded-full text-caption-text font-medium",
                  item.severity === 'critical' && "bg-states-stress text-white",
                  item.severity === 'high' && "bg-consciousness-accent text-white",
                  item.severity === 'medium' && "bg-yellow-500 text-white",
                  item.severity === 'low' && "bg-gray-400 text-white"
                )}>
                  {item.severity}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Hover preview indicator */}
      <div className="absolute right-4 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="text-consciousness-primary">→</span>
      </div>
    </div>
  );
};

export interface LiveActivityFeedProps {
  className?: string;
  maxItems?: number;
  refreshInterval?: number;
  onItemClick?: (item: ActivityItem) => void;
  onItemPreview?: (item: ActivityItem | null) => void;
  groupByTime?: boolean;
}

export const LiveActivityFeed: React.FC<LiveActivityFeedProps> = ({
  className,
  maxItems = 50,
  refreshInterval = 10000,
  onItemClick,
  onItemPreview,
  groupByTime = true
}) => {
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'vip' | 'critical' | 'conversations'>('all');
  const [hoveredItem, setHoveredItem] = useState<ActivityItem | null>(null);

  const ws = useWebSocket();

  // Handle real-time activity updates
  useEffect(() => {
    const unsubscribeConversation = ws.subscribe('kelly_conversation_update', (data: any) => {
      const activity: ActivityItem = {
        id: `conv-${data.conversation_id}-${Date.now()}`,
        type: data.new_messages?.length > 0 ? 'message' : 'new_conversation',
        timestamp: new Date(),
        title: data.new_messages?.length > 0 ? 'New Message' : 'New Conversation Started',
        description: data.new_messages?.length > 0 
          ? `${data.new_messages.length} new message(s) in conversation`
          : 'A new conversation has been initiated',
        metadata: data,
        conversationId: data.conversation_id,
        accountId: data.account_id,
        user: data.user_info ? {
          name: data.user_info.first_name || data.user_info.username,
          avatar: data.user_info.profile_photo_url
        } : undefined
      };

      setActivities(prev => [activity, ...prev.slice(0, maxItems - 1)]);
    });

    const unsubscribeSafety = ws.subscribe('kelly_safety_alert', (alert: SafetyAlert) => {
      const activity: ActivityItem = {
        id: `safety-${alert.id}`,
        type: 'safety_alert',
        timestamp: new Date(),
        title: 'Safety Alert',
        description: alert.description,
        metadata: alert,
        severity: alert.severity,
        isVIP: alert.severity === 'critical',
        conversationId: alert.conversation_id,
        accountId: alert.account_id
      };

      setActivities(prev => [activity, ...prev.slice(0, maxItems - 1)]);
    });

    const unsubscribeIntervention = ws.subscribe('intervention_required', (data: any) => {
      const activity: ActivityItem = {
        id: `intervention-${data.conversation_id}-${Date.now()}`,
        type: 'intervention',
        timestamp: new Date(),
        title: 'Human Intervention Required',
        description: data.reason || 'Manual review needed for conversation',
        metadata: data,
        severity: 'high',
        isVIP: true,
        conversationId: data.conversation_id
      };

      setActivities(prev => [activity, ...prev.slice(0, maxItems - 1)]);
    });

    return () => {
      unsubscribeConversation();
      unsubscribeSafety();
      unsubscribeIntervention();
    };
  }, [ws, maxItems]);

  // Initial data load
  useEffect(() => {
    const loadActivities = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/v1/kelly/activities/feed');
        const data = await response.json();
        
        const formattedActivities: ActivityItem[] = data.activities?.map((activity: any) => ({
          id: activity.id,
          type: activity.type,
          timestamp: new Date(activity.timestamp),
          title: activity.title,
          description: activity.description,
          metadata: activity.metadata,
          severity: activity.severity,
          isVIP: activity.is_vip,
          conversationId: activity.conversation_id,
          accountId: activity.account_id,
          user: activity.user
        })) || [];

        setActivities(formattedActivities);
      } catch (error) {
        console.error('Failed to load activities:', error);
      } finally {
        setLoading(false);
      }
    };

    loadActivities();

    const interval = setInterval(loadActivities, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  // WebSocket connection
  useEffect(() => {
    if (!ws.connected) {
      ws.connect();
    }

    ws.joinRoom('kelly:activities');
    
    return () => {
      ws.leaveRoom('kelly:activities');
    };
  }, [ws]);

  // Filter activities
  const filteredActivities = useMemo(() => {
    switch (filter) {
      case 'vip':
        return activities.filter(item => item.isVIP);
      case 'critical':
        return activities.filter(item => item.severity === 'critical' || item.severity === 'high');
      case 'conversations':
        return activities.filter(item => item.type === 'new_conversation' || item.type === 'message');
      default:
        return activities;
    }
  }, [activities, filter]);

  // Group activities by time
  const groupedActivities = useMemo(() => {
    if (!groupByTime) {
      return { 'All Activities': filteredActivities };
    }

    const now = new Date();
    const groups: Record<string, ActivityItem[]> = {};

    filteredActivities.forEach(item => {
      const diff = now.getTime() - item.timestamp.getTime();
      const hours = diff / (1000 * 60 * 60);
      
      let groupKey: string;
      if (item.isVIP) {
        groupKey = 'VIP Activities';
      } else if (hours < 1) {
        groupKey = 'Last Hour';
      } else if (hours < 24) {
        groupKey = 'Today';
      } else if (hours < 48) {
        groupKey = 'Yesterday';
      } else {
        groupKey = 'Earlier';
      }

      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(item);
    });

    return groups;
  }, [filteredActivities, groupByTime]);

  const handleItemClick = useCallback((item: ActivityItem) => {
    onItemClick?.(item);
  }, [onItemClick]);

  const handleItemHover = useCallback((item: ActivityItem | null) => {
    setHoveredItem(item);
    onItemPreview?.(item);
  }, [onItemPreview]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header with filters */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-insight-subtitle text-consciousness-primary font-sf-pro">
            Live Activity Feed
          </h2>
          <p className="text-body-text text-text-secondary mt-1">
            Real-time updates from all Kelly accounts
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {(['all', 'vip', 'critical', 'conversations'] as const).map((filterOption) => (
            <Button
              key={filterOption}
              variant={filter === filterOption ? 'primary' : 'outline'}
              size="sm"
              onClick={() => setFilter(filterOption)}
            >
              {filterOption.charAt(0).toUpperCase() + filterOption.slice(1)}
            </Button>
          ))}
        </div>
      </div>

      {/* Activity feed */}
      <Card variant="consciousness" glassmorphism className="min-h-[600px]">
        <CardContent className="p-0">
          {loading ? (
            <div className="p-6 space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse flex space-x-4">
                  <div className="h-4 w-4 bg-gray-200 rounded-full"></div>
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : filteredActivities.length === 0 ? (
            <div className="text-center py-12">
              <span className="text-4xl mb-4 block">📱</span>
              <p className="text-insight-subtitle text-text-secondary">No activities yet</p>
              <p className="text-body-text text-text-tertiary mt-2">
                Activity will appear here as conversations happen
              </p>
            </div>
          ) : (
            <div className="p-6 space-y-6 max-h-[800px] overflow-y-auto">
              {Object.entries(groupedActivities).map(([groupTitle, items]) => (
                <ActivityGroup
                  key={groupTitle}
                  title={groupTitle}
                  items={items}
                  isVIP={groupTitle === 'VIP Activities'}
                  onItemClick={handleItemClick}
                  onItemHover={handleItemHover}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Live indicator */}
      <div className="flex items-center justify-center space-x-2 text-caption-text text-text-tertiary">
        <div className={cn(
          "h-2 w-2 rounded-full transition-colors",
          ws.connected ? "bg-states-flow animate-pulse" : "bg-states-stress"
        )} />
        <span>
          {ws.connected ? 'Live updates enabled' : 'Disconnected from live updates'}
        </span>
      </div>
    </div>
  );
};

export default LiveActivityFeed;