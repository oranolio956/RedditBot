/**
 * Real-Time Monitoring Dashboard for Kelly
 * Apple-inspired design with live metrics, glass effects, and fluid animations
 * WebSocket integration for real-time updates
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useWebSocket, useKellySafetyAlerts, useKellyConversationUpdates } from '@/lib/websocket';
import { KellyDashboardOverview, ConversationStage, SafetyAlert } from '@/types/kelly';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  title: string;
  value: number | string;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
  loading?: boolean;
  onClick?: () => void;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  trend = 'neutral',
  icon,
  loading,
  onClick
}) => {
  const trendColor = useMemo(() => {
    switch (trend) {
      case 'up': return 'text-states-flow';
      case 'down': return 'text-states-stress';
      default: return 'text-text-secondary';
    }
  }, [trend]);

  const trendIcon = useMemo(() => {
    switch (trend) {
      case 'up': return '↗';
      case 'down': return '↘';
      default: return '→';
    }
  }, [trend]);

  return (
    <Card
      variant="consciousness"
      glassmorphism
      interactive={!!onClick}
      className={cn(
        "group transition-all duration-300",
        loading && "animate-breathing"
      )}
      onClick={onClick}
    >
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              {icon && <span className="text-consciousness-accent text-xl">{icon}</span>}
              <p className="text-caption-text text-text-tertiary font-medium uppercase tracking-wide">
                {title}
              </p>
            </div>
            
            {loading ? (
              <div className="h-8 w-16 bg-gray-200 rounded animate-pulse" />
            ) : (
              <p className="text-metric-value font-sf-pro text-consciousness-primary">
                {value}
              </p>
            )}
            
            {change !== undefined && !loading && (
              <div className={cn("flex items-center space-x-1", trendColor)}>
                <span className="text-sm font-medium">{trendIcon}</span>
                <span className="text-sm font-medium">
                  {Math.abs(change)}%
                </span>
                <span className="text-caption-text text-text-tertiary">vs last hour</span>
              </div>
            )}
          </div>
          
          {onClick && (
            <Button
              variant="ghost"
              size="icon"
              className="opacity-0 group-hover:opacity-100 transition-opacity"
            >
              🔍
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

interface StatusIndicatorProps {
  status: 'healthy' | 'warning' | 'critical';
  label: string;
  description?: string;
  pulse?: boolean;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  description,
  pulse = false
}) => {
  const statusConfig = useMemo(() => {
    switch (status) {
      case 'healthy':
        return { color: 'bg-states-flow', textColor: 'text-states-flow', icon: '●' };
      case 'warning':
        return { color: 'bg-consciousness-accent', textColor: 'text-consciousness-accent', icon: '●' };
      case 'critical':
        return { color: 'bg-states-stress', textColor: 'text-states-stress', icon: '●' };
    }
  }, [status]);

  return (
    <div className="flex items-center space-x-3">
      <div className="relative">
        <div
          className={cn(
            "h-3 w-3 rounded-full",
            statusConfig.color,
            pulse && "animate-pulse"
          )}
        />
        {pulse && (
          <div className={cn(
            "absolute -inset-1 h-5 w-5 rounded-full opacity-30 animate-ping",
            statusConfig.color
          )} />
        )}
      </div>
      
      <div className="flex-1 min-w-0">
        <p className={cn("text-sm font-medium", statusConfig.textColor)}>
          {label}
        </p>
        {description && (
          <p className="text-caption-text text-text-tertiary truncate">
            {description}
          </p>
        )}
      </div>
    </div>
  );
};

export interface RealTimeMonitoringDashboardProps {
  className?: string;
  onDrillDown?: (metric: string, value: any) => void;
  refreshInterval?: number;
}

export const RealTimeMonitoringDashboard: React.FC<RealTimeMonitoringDashboardProps> = ({
  className,
  onDrillDown,
  refreshInterval = 30000
}) => {
  const [dashboardData, setDashboardData] = useState<KellyDashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [alerts, setAlerts] = useState<SafetyAlert[]>([]);

  const ws = useWebSocket();

  // Handle real-time safety alerts
  useKellySafetyAlerts(useCallback((alert: SafetyAlert) => {
    setAlerts(prev => [alert, ...prev.slice(0, 4)]); // Keep last 5 alerts

    // Play sound for critical alerts
    if (alert.severity === 'critical') {
      const audio = new Audio('/sounds/critical-alert.mp3');
      audio.play().catch(() => {}); // Ignore audio failures
    }
  }, []));

  // Handle dashboard updates
  useEffect(() => {
    const unsubscribe = ws.subscribe('kelly_dashboard_update', (data: KellyDashboardOverview) => {
      setDashboardData(data);
      setLastUpdate(new Date());
      setLoading(false);
    });

    return unsubscribe;
  }, [ws]);

  // Initial data load and periodic refresh
  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/v1/kelly/dashboard/overview');
        const data = await response.json();
        setDashboardData(data);
        setLastUpdate(new Date());
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();

    const interval = setInterval(loadDashboardData, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  // WebSocket connection on mount
  useEffect(() => {
    if (!ws.connected) {
      ws.connect();
    }

    // Subscribe to dashboard updates
    ws.joinRoom('kelly:dashboard');
    
    return () => {
      ws.leaveRoom('kelly:dashboard');
    };
  }, [ws]);

  // Calculate response time trend
  const responseTimeTrend = useMemo(() => {
    if (!dashboardData) return 'neutral';
    // This would typically compare with historical data
    return 'neutral';
  }, [dashboardData]);

  // Calculate engagement trend
  const engagementTrend = useMemo(() => {
    if (!dashboardData || dashboardData.average_engagement_score === 0) return 'neutral';
    // This would typically compare with previous period
    return dashboardData.average_engagement_score >= 75 ? 'up' : 'down';
  }, [dashboardData]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-insight-title text-consciousness-primary font-sf-pro">
            Real-Time Monitoring
          </h1>
          <p className="text-body-text text-text-secondary mt-1">
            Live system metrics and conversation insights
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-caption-text text-text-tertiary">
            Last update: {lastUpdate.toLocaleTimeString()}
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={cn(
              "h-2 w-2 rounded-full",
              ws.connected ? "bg-states-flow animate-pulse" : "bg-states-stress"
            )} />
            <span className="text-caption-text">
              {ws.connected ? 'Live' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Conversations"
          value={dashboardData?.total_conversations_today || 0}
          icon="💬"
          loading={loading}
          onClick={() => onDrillDown?.('active_conversations', dashboardData?.total_conversations_today)}
        />
        
        <MetricCard
          title="Messages Today"
          value={dashboardData?.total_messages_today || 0}
          change={12}
          trend="up"
          icon="📱"
          loading={loading}
          onClick={() => onDrillDown?.('messages_today', dashboardData?.total_messages_today)}
        />
        
        <MetricCard
          title="Avg Engagement"
          value={dashboardData ? `${Math.round(dashboardData.average_engagement_score)}%` : '0%'}
          change={8}
          trend={engagementTrend}
          icon="🎯"
          loading={loading}
          onClick={() => onDrillDown?.('engagement', dashboardData?.average_engagement_score)}
        />
        
        <MetricCard
          title="Safety Score"
          value={dashboardData ? `${Math.round(dashboardData.average_safety_score)}%` : '0%'}
          change={-2}
          trend="down"
          icon="🛡️"
          loading={loading}
          onClick={() => onDrillDown?.('safety_score', dashboardData?.average_safety_score)}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Health Status */}
        <Card variant="consciousness" glassmorphism className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <span>🏥</span>
              <span>System Health</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <StatusIndicator
              status={dashboardData?.system_health || 'healthy'}
              label="Overall System"
              description="All services operational"
              pulse={dashboardData?.system_health === 'critical'}
            />
            
            <StatusIndicator
              status={dashboardData?.ai_performance === 'optimal' ? 'healthy' : 
                     dashboardData?.ai_performance === 'good' ? 'warning' : 'critical'}
              label="AI Performance"
              description={`Claude response time: ${dashboardData?.claude_metrics?.average_response_time || 0}ms`}
            />
            
            <StatusIndicator
              status={dashboardData?.safety_alerts_count === 0 ? 'healthy' : 
                     dashboardData?.safety_alerts_count < 3 ? 'warning' : 'critical'}
              label="Safety Monitoring"
              description={`${dashboardData?.safety_alerts_count || 0} active alerts`}
              pulse={dashboardData?.safety_alerts_count > 0}
            />

            <StatusIndicator
              status={dashboardData?.conversations_requiring_review === 0 ? 'healthy' : 
                     dashboardData?.conversations_requiring_review < 5 ? 'warning' : 'critical'}
              label="Review Queue"
              description={`${dashboardData?.conversations_requiring_review || 0} conversations pending`}
            />
          </CardContent>
        </Card>

        {/* Recent Alerts */}
        <Card variant="breakthrough" glassmorphism>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <span>🚨</span>
              <span>Recent Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <p className="text-text-tertiary text-sm text-center py-4">
                No recent alerts
              </p>
            ) : (
              <div className="space-y-3">
                {alerts.map((alert, index) => (
                  <div
                    key={`${alert.id}-${index}`}
                    className={cn(
                      "p-3 rounded-lg border-l-4 transition-colors",
                      alert.severity === 'critical' && "bg-red-50 border-l-states-stress",
                      alert.severity === 'high' && "bg-orange-50 border-l-consciousness-accent",
                      alert.severity === 'medium' && "bg-yellow-50 border-l-yellow-500",
                      alert.severity === 'low' && "bg-gray-50 border-l-gray-400"
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900">
                          {alert.alert_type}
                        </p>
                        <p className="text-caption-text text-text-tertiary mt-1">
                          {alert.description}
                        </p>
                      </div>
                      <span className={cn(
                        "inline-flex items-center px-2 py-1 rounded-full text-caption-text font-medium",
                        alert.severity === 'critical' && "bg-states-stress text-white",
                        alert.severity === 'high' && "bg-consciousness-accent text-white",
                        alert.severity === 'medium' && "bg-yellow-500 text-white",
                        alert.severity === 'low' && "bg-gray-400 text-white"
                      )}>
                        {alert.severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Conversation Stage Distribution */}
      <Card variant="consciousness" glassmorphism>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>📊</span>
            <span>Conversation Stages</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {dashboardData?.stage_distribution ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(dashboardData.stage_distribution).map(([stage, count]) => (
                <div
                  key={stage}
                  className="text-center p-4 rounded-lg bg-surface-secondary hover:bg-surface-elevated transition-colors cursor-pointer"
                  onClick={() => onDrillDown?.('stage_distribution', { stage, count })}
                >
                  <div className="text-metric-value text-consciousness-primary font-light mb-1">
                    {count}
                  </div>
                  <div className="text-caption-text text-text-tertiary capitalize">
                    {stage.replace('_', ' ')}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="animate-pulse space-y-2">
                <div className="h-8 bg-gray-200 rounded mx-auto w-16"></div>
                <div className="h-4 bg-gray-200 rounded mx-auto w-24"></div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default RealTimeMonitoringDashboard;