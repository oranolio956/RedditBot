/**
 * Kelly Analytics Components - Phase 3 Export Index
 * Advanced analytics and CRM features for Kelly conversation management system
 */

export { AnalyticsDashboard } from './AnalyticsDashboard';
export { ConversationIntelligencePanel } from './ConversationIntelligencePanel';
export { CRMContactProfiles } from './CRMContactProfiles';
export { PipelineManagement } from './PipelineManagement';
export { UserJourneyMapping } from './UserJourneyMapping';
export { AdvancedReporting } from './AdvancedReporting';
export { PerformanceAnalytics } from './PerformanceAnalytics';

// Re-export analytics types for convenience
export type {
  AnalyticsTimeframe,
  MetricPoint,
  ConversationMetrics,
  UserEngagementMetrics,
  RevenueMetrics,
  AIPerformanceMetrics,
  ConversationIntelligence,
  SentimentAnalysis,
  TopicAnalysis,
  SuccessPrediction,
  CoachingRecommendation,
  PatternAnalysis,
  CRMContact,
  ContactProfile,
  LeadScore,
  ContactEngagementMetrics,
  RelationshipIntelligence,
  Pipeline,
  Deal,
  PipelineStage,
  UserJourney,
  FunnelAnalysis,
  ReportTemplate,
  AdvancedInsight,
  InsightRecommendation
} from '../../../types/analytics';