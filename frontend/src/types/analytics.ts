/**
 * Analytics and CRM Types for Kelly AI System
 * Phase 3: Advanced analytics, conversation intelligence, and CRM features
 */

import { ConversationStage, ConversationMessage, KellyConversation, ClaudeUsageMetrics } from './kelly';

// Core Analytics Interfaces
export interface AnalyticsTimeframe {
  start: string;
  end: string;
  granularity: 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';
}

export interface MetricPoint {
  timestamp: string;
  value: number;
  metadata?: Record<string, any>;
}

export interface ConversationMetrics {
  total_conversations: number;
  active_conversations: number;
  completed_conversations: number;
  abandoned_conversations: number;
  average_duration: number;
  average_messages_per_conversation: number;
  conversion_rate: number;
  engagement_score: number;
  ai_confidence_avg: number;
  response_time_avg: number;
  satisfaction_score: number;
}

export interface UserEngagementMetrics {
  daily_active_users: number;
  weekly_active_users: number;
  monthly_active_users: number;
  retention_rate_7day: number;
  retention_rate_30day: number;
  churn_rate: number;
  time_to_conversion: number;
  pages_per_session: number;
  bounce_rate: number;
}

export interface RevenueMetrics {
  total_revenue: number;
  recurring_revenue: number;
  revenue_per_conversation: number;
  customer_lifetime_value: number;
  cost_per_acquisition: number;
  revenue_growth_rate: number;
  mrr_growth: number;
  churn_impact: number;
}

export interface AIPerformanceMetrics {
  claude_usage: ClaudeUsageMetrics;
  response_quality_avg: number;
  ai_vs_human_performance: {
    ai_success_rate: number;
    human_success_rate: number;
    ai_response_time: number;
    human_response_time: number;
    ai_satisfaction: number;
    human_satisfaction: number;
  };
  model_performance: {
    opus: ModelPerformanceMetrics;
    sonnet: ModelPerformanceMetrics;
    haiku: ModelPerformanceMetrics;
  };
  cost_efficiency: {
    cost_per_successful_conversation: number;
    roi_on_ai_investment: number;
    automation_rate: number;
  };
}

export interface ModelPerformanceMetrics {
  usage_count: number;
  average_confidence: number;
  average_quality: number;
  success_rate: number;
  cost_per_use: number;
  response_time: number;
  preferred_stages: ConversationStage[];
}

// Conversation Intelligence
export interface ConversationIntelligence {
  conversation_id: string;
  quality_score: number;
  sentiment_analysis: SentimentAnalysis;
  topic_analysis: TopicAnalysis;
  success_prediction: SuccessPrediction;
  coaching_recommendations: CoachingRecommendation[];
  pattern_analysis: PatternAnalysis;
  competitive_insights: CompetitiveInsights;
}

export interface SentimentAnalysis {
  overall_sentiment: 'very_positive' | 'positive' | 'neutral' | 'negative' | 'very_negative';
  sentiment_score: number; // -1 to 1
  sentiment_timeline: MetricPoint[];
  emotional_journey: EmotionalJourneyPoint[];
  sentiment_drivers: {
    positive_factors: string[];
    negative_factors: string[];
    neutral_factors: string[];
  };
  mood_transitions: MoodTransition[];
}

export interface EmotionalJourneyPoint {
  timestamp: string;
  emotion: string;
  intensity: number;
  trigger_message?: string;
  context: string;
}

export interface MoodTransition {
  from_mood: string;
  to_mood: string;
  timestamp: string;
  trigger: string;
  impact_score: number;
}

export interface TopicAnalysis {
  primary_topics: TopicInsight[];
  topic_progression: TopicProgression[];
  topic_sentiment: Record<string, number>;
  engagement_by_topic: Record<string, number>;
  successful_topics: string[];
  problematic_topics: string[];
  topic_recommendations: string[];
}

export interface TopicInsight {
  topic: string;
  relevance_score: number;
  frequency: number;
  sentiment_score: number;
  engagement_impact: number;
  conversion_correlation: number;
  first_mentioned: string;
  last_mentioned: string;
}

export interface TopicProgression {
  timestamp: string;
  topic: string;
  transition_type: 'introduction' | 'development' | 'conclusion' | 'abandonment';
  success_indicator: number;
}

export interface SuccessPrediction {
  probability_of_success: number;
  confidence_interval: [number, number];
  key_success_factors: string[];
  risk_factors: string[];
  recommended_actions: string[];
  time_to_conversion_estimate: number;
  optimal_next_message_timing: number;
  conversation_health_score: number;
}

export interface CoachingRecommendation {
  id: string;
  type: 'messaging' | 'timing' | 'topic' | 'tone' | 'strategy';
  priority: 'low' | 'medium' | 'high' | 'critical';
  recommendation: string;
  rationale: string;
  expected_impact: number;
  implementation_difficulty: 'easy' | 'medium' | 'hard';
  success_examples: string[];
  ai_confidence: number;
}

export interface PatternAnalysis {
  conversation_patterns: ConversationPattern[];
  user_behavior_patterns: UserBehaviorPattern[];
  temporal_patterns: TemporalPattern[];
  success_patterns: SuccessPattern[];
  failure_patterns: FailurePattern[];
  anomaly_detection: AnomalyDetection;
}

export interface ConversationPattern {
  pattern_id: string;
  pattern_type: 'opening' | 'engagement' | 'objection_handling' | 'closing' | 'follow_up';
  pattern_description: string;
  frequency: number;
  success_rate: number;
  avg_conversation_impact: number;
  contexts_where_effective: string[];
  similar_patterns: string[];
}

export interface UserBehaviorPattern {
  behavior_id: string;
  behavior_description: string;
  frequency: number;
  correlation_with_success: number;
  typical_user_segments: string[];
  response_strategies: string[];
}

export interface TemporalPattern {
  pattern_name: string;
  time_periods: string[];
  activity_level: number;
  success_rate: number;
  optimal_response_times: number[];
  peak_engagement_hours: number[];
}

export interface SuccessPattern {
  pattern_id: string;
  description: string;
  frequency_in_successful_convos: number;
  frequency_in_failed_convos: number;
  impact_score: number;
  replication_strategy: string;
}

export interface FailurePattern {
  pattern_id: string;
  description: string;
  frequency: number;
  damage_score: number;
  prevention_strategy: string;
  early_warning_indicators: string[];
}

export interface AnomalyDetection {
  detected_anomalies: Anomaly[];
  anomaly_score: number;
  baseline_comparison: BaselineComparison;
  investigation_priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface Anomaly {
  anomaly_id: string;
  type: 'performance' | 'behavior' | 'pattern' | 'technical';
  description: string;
  severity: number;
  detected_at: string;
  confidence: number;
  potential_causes: string[];
  recommended_investigation: string[];
}

export interface BaselineComparison {
  current_performance: number;
  baseline_performance: number;
  variance: number;
  variance_explanation: string;
  is_significant: boolean;
}

export interface CompetitiveInsights {
  market_position: 'leading' | 'competitive' | 'lagging';
  performance_vs_competitors: CompetitorComparison[];
  unique_advantages: string[];
  improvement_opportunities: string[];
  market_trend_alignment: number;
}

export interface CompetitorComparison {
  competitor: string;
  metric: string;
  our_performance: number;
  competitor_performance: number;
  advantage_factor: number;
  improvement_potential: number;
}

// CRM and Contact Management
export interface CRMContact {
  contact_id: string;
  telegram_user_id: number;
  profile: ContactProfile;
  conversation_history: ConversationSummary[];
  lead_score: LeadScore;
  lifecycle_stage: LifecycleStage;
  engagement_metrics: ContactEngagementMetrics;
  relationship_intelligence: RelationshipIntelligence;
  revenue_attribution: RevenueAttribution;
  contact_timeline: ContactTimelineEvent[];
  tags: string[];
  custom_fields: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface ContactProfile {
  basic_info: {
    first_name?: string;
    last_name?: string;
    username?: string;
    phone_number?: string;
    email?: string;
    location?: string;
    timezone?: string;
    language: string;
    profile_photo_url?: string;
  };
  demographics: {
    age_range?: string;
    gender?: string;
    occupation?: string;
    income_level?: string;
    education_level?: string;
    relationship_status?: string;
  };
  psychographics: {
    personality_traits: Record<string, number>;
    interests: string[];
    values: string[];
    lifestyle_indicators: string[];
    communication_style: 'formal' | 'casual' | 'playful' | 'professional';
  };
  behavioral_data: {
    online_activity_pattern: ActivityPattern[];
    response_patterns: ResponsePattern[];
    engagement_preferences: EngagementPreference[];
    purchase_history: PurchaseHistory[];
    content_preferences: ContentPreference[];
  };
  enrichment_data: {
    social_media_profiles: SocialMediaProfile[];
    professional_background: ProfessionalBackground;
    network_connections: NetworkConnection[];
    digital_footprint_score: number;
    data_sources: string[];
    last_enriched: string;
  };
}

export interface ConversationSummary {
  conversation_id: string;
  start_date: string;
  end_date?: string;
  status: 'active' | 'completed' | 'abandoned' | 'blocked';
  stage_reached: ConversationStage;
  message_count: number;
  avg_response_time: number;
  engagement_score: number;
  satisfaction_score?: number;
  outcome: 'successful' | 'unsuccessful' | 'ongoing';
  outcome_reason?: string;
  revenue_generated?: number;
  key_topics: string[];
  sentiment_summary: 'positive' | 'neutral' | 'negative';
  ai_confidence_avg: number;
  human_interventions: number;
}

export interface LeadScore {
  current_score: number;
  max_score: number;
  score_history: MetricPoint[];
  scoring_factors: ScoringFactor[];
  score_breakdown: {
    demographic_score: number;
    behavioral_score: number;
    engagement_score: number;
    intent_score: number;
    fit_score: number;
  };
  prediction_confidence: number;
  recommended_actions: string[];
  score_trend: 'increasing' | 'decreasing' | 'stable';
}

export interface ScoringFactor {
  factor: string;
  weight: number;
  current_value: number;
  impact_on_score: number;
  last_updated: string;
}

export type LifecycleStage = 
  | 'subscriber' 
  | 'lead' 
  | 'marketing_qualified_lead' 
  | 'sales_qualified_lead' 
  | 'opportunity' 
  | 'customer' 
  | 'evangelist' 
  | 'churned';

export interface ContactEngagementMetrics {
  total_interactions: number;
  last_interaction_date: string;
  interaction_frequency: number;
  avg_session_duration: number;
  engagement_score_trend: MetricPoint[];
  preferred_contact_times: string[];
  response_rate: number;
  click_through_rate: number;
  conversion_events: ConversionEvent[];
  engagement_quality_score: number;
}

export interface ConversionEvent {
  event_id: string;
  event_type: string;
  event_value: number;
  timestamp: string;
  conversation_id?: string;
  attribution_data: AttributionData;
}

export interface AttributionData {
  first_touch: TouchPoint;
  last_touch: TouchPoint;
  all_touchpoints: TouchPoint[];
  conversion_path: string[];
  time_to_conversion: number;
  attribution_model: 'first_touch' | 'last_touch' | 'linear' | 'time_decay' | 'position_based';
}

export interface TouchPoint {
  timestamp: string;
  type: 'conversation' | 'message' | 'content_view' | 'feature_use';
  description: string;
  value: number;
  conversation_id?: string;
}

export interface RelationshipIntelligence {
  relationship_strength: number;
  trust_level: number;
  influence_score: number;
  network_value: number;
  relationship_timeline: RelationshipMilestone[];
  mutual_connections: string[];
  referral_potential: number;
  relationship_health: 'strong' | 'good' | 'weak' | 'at_risk';
  recommended_relationship_actions: string[];
}

export interface RelationshipMilestone {
  milestone_id: string;
  type: 'first_contact' | 'engagement_milestone' | 'conversion' | 'referral' | 'expansion' | 'renewal';
  description: string;
  date: string;
  impact_score: number;
  conversation_id?: string;
}

export interface RevenueAttribution {
  total_revenue: number;
  revenue_by_period: MetricPoint[];
  revenue_sources: RevenueSource[];
  customer_lifetime_value: number;
  predicted_future_value: number;
  revenue_quality_score: number;
  upsell_opportunities: UpsellOpportunity[];
  churn_risk: ChurnRisk;
}

export interface RevenueSource {
  source: string;
  amount: number;
  date: string;
  conversation_id?: string;
  recurring: boolean;
}

export interface UpsellOpportunity {
  opportunity_id: string;
  product_service: string;
  probability: number;
  potential_value: number;
  recommended_approach: string;
  timing_recommendation: string;
}

export interface ChurnRisk {
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  risk_factors: string[];
  early_warning_indicators: string[];
  retention_strategies: string[];
  predicted_churn_date?: string;
}

export interface ContactTimelineEvent {
  event_id: string;
  type: 'conversation' | 'conversion' | 'milestone' | 'interaction' | 'system_event';
  title: string;
  description: string;
  timestamp: string;
  impact_score: number;
  conversation_id?: string;
  metadata: Record<string, any>;
}

// Pipeline and Deal Management
export interface Pipeline {
  pipeline_id: string;
  name: string;
  stages: PipelineStage[];
  deals: Deal[];
  conversion_rates: Record<string, number>;
  avg_stage_duration: Record<string, number>;
  pipeline_velocity: number;
  pipeline_value: number;
  pipeline_health_score: number;
  bottlenecks: Bottleneck[];
  optimization_recommendations: string[];
}

export interface PipelineStage {
  stage_id: string;
  name: string;
  description: string;
  order: number;
  probability: number;
  expected_duration: number;
  automation_rules: AutomationRule[];
  stage_requirements: StageRequirement[];
  exit_criteria: string[];
}

export interface Deal {
  deal_id: string;
  contact_id: string;
  pipeline_id: string;
  current_stage: string;
  deal_value: number;
  probability: number;
  expected_close_date: string;
  created_date: string;
  last_activity: string;
  conversation_ids: string[];
  deal_source: string;
  deal_tags: string[];
  stage_history: StageHistoryEntry[];
  activities: DealActivity[];
  revenue_attribution: DealRevenueAttribution;
  risk_assessment: DealRiskAssessment;
  next_actions: NextAction[];
}

export interface StageHistoryEntry {
  stage_id: string;
  stage_name: string;
  entered_at: string;
  exited_at?: string;
  duration?: number;
  exit_reason?: string;
  conversion_successful: boolean;
}

export interface DealActivity {
  activity_id: string;
  type: 'conversation' | 'email' | 'call' | 'meeting' | 'proposal' | 'contract';
  description: string;
  timestamp: string;
  outcome: string;
  impact_score: number;
  conversation_id?: string;
}

export interface DealRevenueAttribution {
  attributed_conversations: string[];
  conversation_impact_scores: Record<string, number>;
  primary_driver_conversation: string;
  revenue_journey: RevenueJourneyPoint[];
  attribution_confidence: number;
}

export interface RevenueJourneyPoint {
  timestamp: string;
  event: string;
  impact: number;
  conversation_id?: string;
  cumulative_impact: number;
}

export interface DealRiskAssessment {
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_factors: RiskFactor[];
  mitigation_strategies: string[];
  probability_adjustment: number;
  risk_monitoring_alerts: string[];
}

export interface RiskFactor {
  factor: string;
  impact: number;
  probability: number;
  mitigation_actions: string[];
}

export interface NextAction {
  action_id: string;
  type: 'follow_up' | 'send_proposal' | 'schedule_demo' | 'negotiate' | 'close';
  description: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  due_date: string;
  assigned_to: string;
  conversation_context?: string;
  success_criteria: string[];
}

export interface Bottleneck {
  stage_id: string;
  stage_name: string;
  bottleneck_type: 'conversion' | 'duration' | 'activity';
  severity: number;
  description: string;
  affected_deals_count: number;
  revenue_impact: number;
  root_causes: string[];
  recommended_solutions: string[];
}

export interface AutomationRule {
  rule_id: string;
  name: string;
  trigger: AutomationTrigger;
  conditions: AutomationCondition[];
  actions: AutomationAction[];
  enabled: boolean;
  success_rate: number;
}

export interface AutomationTrigger {
  type: 'conversation_event' | 'time_based' | 'stage_change' | 'metric_threshold';
  parameters: Record<string, any>;
}

export interface AutomationCondition {
  field: string;
  operator: 'equals' | 'greater_than' | 'less_than' | 'contains' | 'not_equals';
  value: any;
}

export interface AutomationAction {
  type: 'move_stage' | 'send_message' | 'create_task' | 'update_field' | 'trigger_workflow';
  parameters: Record<string, any>;
}

export interface StageRequirement {
  requirement_id: string;
  description: string;
  required: boolean;
  validation_rule: string;
  auto_check: boolean;
}

// User Journey and Funnel Analysis
export interface UserJourney {
  journey_id: string;
  contact_id: string;
  start_date: string;
  end_date?: string;
  status: 'active' | 'completed' | 'abandoned';
  journey_stages: JourneyStage[];
  touchpoints: JourneyTouchpoint[];
  conversion_events: ConversionEvent[];
  journey_value: number;
  journey_satisfaction: number;
  journey_efficiency_score: number;
  optimization_opportunities: OptimizationOpportunity[];
}

export interface JourneyStage {
  stage_id: string;
  stage_name: string;
  entry_time: string;
  exit_time?: string;
  duration?: number;
  stage_completion: boolean;
  stage_satisfaction: number;
  actions_taken: string[];
  barriers_encountered: string[];
  assistance_provided: string[];
}

export interface JourneyTouchpoint {
  touchpoint_id: string;
  type: 'conversation' | 'feature_interaction' | 'content_consumption' | 'support_interaction';
  description: string;
  timestamp: string;
  channel: string;
  engagement_quality: number;
  satisfaction_score?: number;
  conversation_id?: string;
  outcome: string;
  value_created: number;
}

export interface OptimizationOpportunity {
  opportunity_id: string;
  type: 'friction_reduction' | 'engagement_increase' | 'conversion_improvement' | 'satisfaction_boost';
  description: string;
  current_performance: number;
  potential_improvement: number;
  implementation_effort: 'low' | 'medium' | 'high';
  expected_roi: number;
  success_metrics: string[];
}

// Funnel Analysis
export interface FunnelAnalysis {
  funnel_id: string;
  funnel_name: string;
  stages: FunnelStage[];
  conversion_rates: Record<string, number>;
  drop_off_analysis: DropOffAnalysis[];
  cohort_analysis: CohortAnalysis;
  segment_performance: SegmentPerformance[];
  optimization_insights: FunnelOptimizationInsight[];
  benchmark_comparison: BenchmarkComparison;
}

export interface FunnelStage {
  stage_id: string;
  stage_name: string;
  stage_order: number;
  users_entered: number;
  users_completed: number;
  conversion_rate: number;
  avg_time_to_complete: number;
  typical_actions: string[];
  success_factors: string[];
  common_barriers: string[];
}

export interface DropOffAnalysis {
  stage_id: string;
  drop_off_rate: number;
  drop_off_reasons: DropOffReason[];
  user_segments_affected: string[];
  revenue_impact: number;
  recovery_strategies: string[];
}

export interface DropOffReason {
  reason: string;
  frequency: number;
  impact_score: number;
  addressable: boolean;
  solution_strategies: string[];
}

export interface CohortAnalysis {
  cohort_periods: CohortPeriod[];
  retention_curves: RetentionCurve[];
  cohort_performance_comparison: CohortComparison[];
  lifecycle_value_by_cohort: Record<string, number>;
}

export interface CohortPeriod {
  period_id: string;
  start_date: string;
  end_date: string;
  cohort_size: number;
  conversion_performance: Record<string, number>;
  engagement_metrics: Record<string, number>;
  revenue_metrics: Record<string, number>;
}

export interface RetentionCurve {
  cohort_id: string;
  retention_points: MetricPoint[];
  churn_points: MetricPoint[];
  predicted_retention: MetricPoint[];
}

export interface CohortComparison {
  metric: string;
  cohort_performance: Record<string, number>;
  best_performing_cohort: string;
  worst_performing_cohort: string;
  improvement_opportunities: string[];
}

export interface SegmentPerformance {
  segment_id: string;
  segment_name: string;
  segment_criteria: Record<string, any>;
  user_count: number;
  conversion_performance: Record<string, number>;
  engagement_metrics: Record<string, number>;
  revenue_metrics: Record<string, number>;
  optimization_recommendations: string[];
}

export interface FunnelOptimizationInsight {
  insight_id: string;
  type: 'bottleneck' | 'opportunity' | 'anomaly' | 'trend';
  description: string;
  affected_stages: string[];
  impact_estimation: number;
  confidence_level: number;
  recommended_actions: string[];
  implementation_priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface BenchmarkComparison {
  industry_benchmarks: Record<string, number>;
  competitor_benchmarks: Record<string, number>;
  our_performance: Record<string, number>;
  performance_gaps: PerformanceGap[];
  competitive_advantages: string[];
}

export interface PerformanceGap {
  metric: string;
  our_value: number;
  benchmark_value: number;
  gap_size: number;
  improvement_potential: number;
  action_plan: string[];
}

// Reporting and Insights
export interface ReportTemplate {
  template_id: string;
  name: string;
  description: string;
  report_type: 'dashboard' | 'detailed' | 'executive' | 'operational';
  frequency: 'real_time' | 'hourly' | 'daily' | 'weekly' | 'monthly';
  metrics: ReportMetric[];
  filters: ReportFilter[];
  visualizations: ReportVisualization[];
  automated_insights: boolean;
  distribution_list: string[];
  custom_fields: CustomField[];
}

export interface ReportMetric {
  metric_id: string;
  metric_name: string;
  metric_type: 'number' | 'percentage' | 'currency' | 'duration' | 'count';
  aggregation: 'sum' | 'average' | 'median' | 'count' | 'min' | 'max';
  data_source: string;
  calculation_method: string;
  display_format: string;
  thresholds: MetricThreshold[];
}

export interface MetricThreshold {
  type: 'target' | 'warning' | 'critical';
  value: number;
  comparison: 'greater_than' | 'less_than' | 'equals';
  action: string;
}

export interface ReportFilter {
  filter_id: string;
  field: string;
  operator: string;
  value: any;
  required: boolean;
  user_configurable: boolean;
}

export interface ReportVisualization {
  viz_id: string;
  type: 'line_chart' | 'bar_chart' | 'pie_chart' | 'heatmap' | 'table' | 'gauge' | 'funnel';
  title: string;
  metrics: string[];
  dimensions: string[];
  configuration: Record<string, any>;
  interactive: boolean;
  drill_down_enabled: boolean;
}

export interface CustomField {
  field_id: string;
  field_name: string;
  field_type: 'text' | 'number' | 'date' | 'boolean' | 'select' | 'multi_select';
  required: boolean;
  default_value?: any;
  validation_rules?: string[];
  options?: string[];
}

export interface AdvancedInsight {
  insight_id: string;
  type: 'trend' | 'anomaly' | 'prediction' | 'recommendation' | 'alert';
  title: string;
  description: string;
  confidence_score: number;
  business_impact: number;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  affected_metrics: string[];
  root_cause_analysis: RootCauseAnalysis;
  recommendations: InsightRecommendation[];
  predicted_outcomes: PredictedOutcome[];
  supporting_data: Record<string, any>;
  generated_at: string;
  valid_until?: string;
}

export interface RootCauseAnalysis {
  primary_causes: Cause[];
  contributing_factors: Cause[];
  correlation_analysis: CorrelationAnalysis[];
  external_factors: ExternalFactor[];
  confidence_in_analysis: number;
}

export interface Cause {
  cause_id: string;
  description: string;
  probability: number;
  impact_score: number;
  evidence: string[];
  validation_method: string;
}

export interface CorrelationAnalysis {
  variable_a: string;
  variable_b: string;
  correlation_coefficient: number;
  significance_level: number;
  causal_relationship: 'strong' | 'weak' | 'none' | 'unclear';
}

export interface ExternalFactor {
  factor: string;
  impact_assessment: number;
  controllability: 'high' | 'medium' | 'low' | 'none';
  monitoring_strategy: string;
}

export interface InsightRecommendation {
  recommendation_id: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  implementation_effort: 'low' | 'medium' | 'high';
  expected_impact: number;
  success_metrics: string[];
  implementation_steps: string[];
  resource_requirements: string[];
  timeline_estimate: string;
  risk_assessment: string[];
}

export interface PredictedOutcome {
  scenario: string;
  probability: number;
  impact_description: string;
  timeline: string;
  confidence_interval: [number, number];
  assumptions: string[];
  monitoring_indicators: string[];
}

// Additional helper types
export interface ActivityPattern {
  pattern_id: string;
  day_of_week: number;
  hour_of_day: number;
  activity_level: number;
  typical_duration: number;
  engagement_quality: number;
}

export interface ResponsePattern {
  message_type: string;
  avg_response_time: number;
  typical_length: number;
  sentiment_tendency: string;
  engagement_impact: number;
}

export interface EngagementPreference {
  preference_type: string;
  preference_value: string;
  confidence: number;
  source: string;
  last_updated: string;
}

export interface PurchaseHistory {
  purchase_id: string;
  product_service: string;
  amount: number;
  date: string;
  payment_method: string;
  satisfaction_score?: number;
}

export interface ContentPreference {
  content_type: string;
  engagement_score: number;
  frequency_preference: string;
  optimal_timing: string[];
}

export interface SocialMediaProfile {
  platform: string;
  profile_url: string;
  follower_count?: number;
  activity_level: 'high' | 'medium' | 'low';
  content_themes: string[];
  influence_score?: number;
}

export interface ProfessionalBackground {
  current_position?: string;
  company?: string;
  industry?: string;
  experience_level?: string;
  skills: string[];
  professional_interests: string[];
}

export interface NetworkConnection {
  connection_id: string;
  connection_type: 'direct' | 'second_degree' | 'group_member';
  platform: string;
  strength: number;
  mutual_connections?: number;
  interaction_frequency: 'high' | 'medium' | 'low';
}

// Export all analytics interfaces
export type {
  // Core analytics
  AnalyticsTimeframe,
  MetricPoint,
  ConversationMetrics,
  UserEngagementMetrics,
  RevenueMetrics,
  AIPerformanceMetrics,
  
  // Conversation intelligence
  ConversationIntelligence,
  SentimentAnalysis,
  TopicAnalysis,
  SuccessPrediction,
  CoachingRecommendation,
  PatternAnalysis,
  
  // CRM
  CRMContact,
  ContactProfile,
  LeadScore,
  ContactEngagementMetrics,
  RelationshipIntelligence,
  
  // Pipeline
  Pipeline,
  Deal,
  PipelineStage,
  
  // Journey analysis
  UserJourney,
  FunnelAnalysis,
  
  // Reporting
  ReportTemplate,
  AdvancedInsight,
  InsightRecommendation
};