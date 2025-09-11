/**
 * Kelly Telegram Bot Types
 * Comprehensive type definitions for Kelly's AI brain and conversation management
 */

// Core Kelly Account Types
export interface KellyAccount {
  id: string;
  user_id: string;
  phone_number: string;
  session_name: string;
  api_id: string;
  api_hash: string;
  is_active: boolean;
  is_connected: boolean;
  dm_only_mode: boolean;
  max_daily_messages: number;
  messages_sent_today: number;
  last_activity: string;
  created_at: string;
  updated_at: string;
  config: KellyPersonalityConfig;
  metrics: KellyAccountMetrics;
  safety_status: SafetyStatus;
  conversations: KellyConversation[];
}

// Kelly Personality Configuration
export interface KellyPersonalityConfig {
  // Core personality traits (0-1 scale)
  warmth: number;
  empathy: number;
  playfulness: number;
  professionalism: number;
  confidence: number;
  creativity: number;
  patience: number;
  
  // Communication style
  message_length_preference: 'short' | 'medium' | 'long' | 'adaptive';
  emoji_frequency: number; // 0-1 scale
  punctuation_style: 'minimal' | 'standard' | 'expressive';
  typing_speed_wpm: number;
  response_delay_min: number; // seconds
  response_delay_max: number; // seconds
  
  // Conversation behavior
  initiation_probability: number; // chance to start conversations
  follow_up_aggressiveness: number; // how persistent to be
  topic_change_frequency: number; // how often to introduce new topics
  question_asking_rate: number; // how many questions to ask
  
  // Safety and boundaries
  payment_discussion_threshold: number; // conversation stage when payments can be discussed
  red_flag_sensitivity: number;
  auto_block_threshold: number;
  escalation_threshold: number;
  content_filtering_level: 'low' | 'medium' | 'high' | 'maximum';
  
  // Advanced settings
  context_memory_depth: number; // how many messages to remember
  personality_adaptation_rate: number; // how quickly to adapt to user
  emotional_responsiveness: number;
  humor_detection_enabled: boolean;
  sarcasm_usage_level: number;
}

// Account Performance Metrics
export interface KellyAccountMetrics {
  // Daily stats
  messages_sent_today: number;
  conversations_started_today: number;
  successful_engagements_today: number;
  safety_violations_today: number;
  
  // Weekly stats
  weekly_message_count: number;
  weekly_conversation_count: number;
  weekly_engagement_rate: number;
  weekly_safety_score: number;
  
  // All-time stats
  total_messages_sent: number;
  total_conversations: number;
  average_conversation_length: number;
  engagement_success_rate: number;
  safety_violation_count: number;
  
  // Performance indicators
  response_time_avg: number; // seconds
  conversation_quality_score: number; // 0-100
  user_satisfaction_rating: number; // 0-5
  ai_confidence_score: number; // 0-100
  
  // Stage progression
  stage_1_10_success_rate: number; // initial contact
  stage_11_20_success_rate: number; // qualification
  stage_21_30_success_rate: number; // engagement
  stage_31_plus_success_rate: number; // advanced engagement
}

// Safety and Risk Management
export interface SafetyStatus {
  current_risk_level: 'low' | 'medium' | 'high' | 'critical';
  safety_score: number; // 0-100
  recent_violations: SafetyViolation[];
  blocked_users_count: number;
  escalated_conversations_count: number;
  auto_responses_disabled: boolean;
  manual_review_required: boolean;
  last_safety_check: string;
}

export interface SafetyViolation {
  id: string;
  type: 'spam' | 'harassment' | 'inappropriate_content' | 'payment_pressure' | 'boundary_violation' | 'underage' | 'threats';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  conversation_id: string;
  message_content: string;
  detected_at: string;
  action_taken: 'warning' | 'temporary_pause' | 'conversation_ended' | 'user_blocked' | 'escalated';
  confidence_score: number;
}

// Conversation Management
export interface KellyConversation {
  id: string;
  account_id: string;
  telegram_user_id: number;
  chat_id: number;
  user_info: TelegramUserInfo;
  
  // Conversation state
  stage: ConversationStage;
  status: 'active' | 'paused' | 'ended' | 'blocked' | 'escalated';
  message_count: number;
  started_at: string;
  last_activity: string;
  
  // Quality metrics
  engagement_score: number; // 0-100
  safety_score: number; // 0-100
  progression_score: number; // how well it's advancing through stages
  ai_confidence: number; // how confident AI is about responses
  
  // Flags and alerts
  red_flags: RedFlag[];
  requires_human_review: boolean;
  escalation_reason?: string;
  
  // Context and memory
  conversation_context: ConversationContext;
  user_personality_profile: UserPersonalityProfile;
  topics_discussed: string[];
  recent_messages: ConversationMessage[];
}

export type ConversationStage = 
  | 'initial_contact' // 1-10 messages
  | 'rapport_building' // 11-20 messages  
  | 'qualification' // 21-30 messages
  | 'engagement' // 31-50 messages
  | 'advanced_engagement' // 51+ messages
  | 'payment_discussion' // when payment topics arise
  | 'closing' // ending conversation
  | 'maintenance'; // ongoing relationship

export interface TelegramUserInfo {
  user_id: number;
  username?: string;
  first_name?: string;
  last_name?: string;
  phone_number?: string;
  is_premium: boolean;
  language_code?: string;
  profile_photo_url?: string;
  bio?: string;
  joined_date?: string;
}

export interface ConversationContext {
  user_timezone?: string;
  preferred_communication_style: 'formal' | 'casual' | 'playful' | 'professional';
  interests: string[];
  conversation_history_summary: string;
  key_topics: string[];
  emotional_state_history: EmotionalStateEntry[];
  response_patterns: ResponsePattern[];
  availability_patterns: AvailabilityPattern[];
}

export interface UserPersonalityProfile {
  // Detected personality traits
  detected_traits: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
  
  // Communication patterns
  message_length_preference: 'short' | 'medium' | 'long';
  emoji_usage: number;
  response_speed: 'fast' | 'medium' | 'slow';
  conversation_style: 'direct' | 'narrative' | 'question_heavy' | 'playful';
  
  // Behavioral patterns
  online_activity_pattern: string[];
  topic_preferences: TopicPreference[];
  engagement_triggers: string[];
  disengagement_signals: string[];
  
  confidence_level: number; // how confident we are in this profile
  last_updated: string;
}

export interface TopicPreference {
  topic: string;
  interest_level: number; // 0-100
  expertise_level: number; // 0-100
  frequency_discussed: number;
  emotional_response: 'positive' | 'neutral' | 'negative';
}

export interface EmotionalStateEntry {
  timestamp: string;
  detected_emotion: string;
  confidence: number;
  context: string;
  trigger?: string;
}

export interface ResponsePattern {
  pattern_type: 'greeting' | 'question_response' | 'topic_change' | 'goodbye';
  typical_response: string;
  response_time_avg: number;
  frequency: number;
}

export interface AvailabilityPattern {
  day_of_week: number; // 0-6
  hour_ranges: { start: number; end: number }[];
  typical_response_time: number;
  activity_level: 'high' | 'medium' | 'low';
}

export interface ConversationMessage {
  id: string;
  conversation_id: string;
  sender: 'kelly' | 'user';
  content: string;
  message_type: 'text' | 'sticker' | 'photo' | 'voice' | 'video' | 'document';
  timestamp: string;
  read: boolean;
  
  // AI analysis
  sentiment_score: number;
  emotional_tone: string;
  topics_mentioned: string[];
  ai_confidence: number;
  response_quality_score?: number; // for Kelly's messages
  
  // Safety analysis
  safety_flags: SafetyFlag[];
  content_warnings: string[];
  
  // Conversation flow
  response_to_message_id?: string;
  generated_responses?: GeneratedResponse[];
}

export interface SafetyFlag {
  type: 'inappropriate_content' | 'payment_request' | 'personal_info' | 'aggressive_language' | 'spam' | 'underage_indicator';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  confidence: number;
  auto_action?: 'none' | 'warning' | 'pause' | 'escalate' | 'block';
}

export interface RedFlag {
  id: string;
  type: 'payment_pressure' | 'aggressive_behavior' | 'inappropriate_requests' | 'underage_user' | 'spam_behavior' | 'harassment';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  detected_at: string;
  message_id: string;
  confidence_score: number;
  action_taken?: string;
  human_reviewed: boolean;
}

export interface GeneratedResponse {
  id: string;
  content: string;
  confidence_score: number;
  response_type: 'engagement' | 'question' | 'topic_change' | 'emotional_support' | 'humor' | 'continuation';
  estimated_quality: number;
  ai_features_used: string[];
  selected: boolean;
  human_modified: boolean;
  claude_metadata?: ClaudeResponseMetadata;
}

export interface ClaudeResponseMetadata {
  model_used: 'opus' | 'sonnet' | 'haiku';
  tokens_used: number;
  cost_usd: number;
  processing_time_ms: number;
  thinking_process?: string; // Claude's reasoning
  confidence_breakdown: {
    content_quality: number;
    personality_match: number;
    safety_score: number;
    engagement_potential: number;
  };
  alternative_responses?: {
    content: string;
    confidence: number;
    reasoning: string;
  }[];
}

// AI Features Configuration
export interface AIFeatureConfig {
  consciousness_mirror: ConsciousnessMirrorConfig;
  memory_palace: MemoryPalaceConfig;
  emotional_intelligence: EmotionalIntelligenceConfig;
  temporal_archaeology: TemporalArchaeologyConfig;
  digital_telepathy: DigitalTelepathyConfig;
  quantum_consciousness: QuantumConsciousnessConfig;
  synesthesia: SynesthesiaConfig;
  neural_dreams: NeuralDreamsConfig;
  claude_integration: ClaudeIntegrationConfig;
}

// Claude AI Integration Configuration
export interface ClaudeIntegrationConfig {
  enabled: boolean;
  model_selection: 'opus' | 'sonnet' | 'haiku' | 'auto';
  temperature: number; // 0-1 scale for creativity
  max_tokens: number;
  confidence_threshold: number; // minimum confidence to use response
  cost_limit_daily: number; // daily spending limit in USD
  conversation_stages: {
    initial_contact: ClaudeStageConfig;
    rapport_building: ClaudeStageConfig;
    qualification: ClaudeStageConfig;
    engagement: ClaudeStageConfig;
    advanced_engagement: ClaudeStageConfig;
    payment_discussion: ClaudeStageConfig;
  };
  safety_features: {
    content_filtering: boolean;
    red_flag_detection: boolean;
    escalation_triggers: string[];
    manual_override_enabled: boolean;
  };
  response_optimization: {
    personality_matching: boolean;
    emotional_intelligence: boolean;
    memory_integration: boolean;
    real_time_adaptation: boolean;
  };
}

export interface ClaudeStageConfig {
  model_preference: 'opus' | 'sonnet' | 'haiku';
  temperature: number;
  max_response_length: number;
  personality_weight: number; // how much to adapt to user personality
  safety_level: 'low' | 'medium' | 'high' | 'maximum';
}

export interface ConsciousnessMirrorConfig {
  enabled: boolean;
  sensitivity: number; // how quickly to adapt to user personality
  mirroring_strength: number; // how much to mirror vs maintain Kelly's personality
  adaptation_speed: number; // how fast to change based on new information
  personality_lock_threshold: number; // when to stop major adaptations
}

export interface MemoryPalaceConfig {
  enabled: boolean;
  context_window_size: number; // how many messages to keep in active memory
  long_term_storage_enabled: boolean;
  memory_consolidation_frequency: 'hourly' | 'daily' | 'weekly';
  pattern_detection_enabled: boolean;
  cross_conversation_memory: boolean;
}

export interface EmotionalIntelligenceConfig {
  enabled: boolean;
  mood_detection_sensitivity: number;
  emotional_response_strength: number;
  empathy_level: number;
  emotional_memory_enabled: boolean;
  mood_prediction_enabled: boolean;
}

export interface TemporalArchaeologyConfig {
  enabled: boolean;
  pattern_analysis_depth: number; // how far back to analyze
  behavioral_prediction_enabled: boolean;
  conversation_archaeology_enabled: boolean;
  timeline_reconstruction_enabled: boolean;
}

export interface DigitalTelepathyConfig {
  enabled: boolean;
  response_prediction_enabled: boolean;
  optimal_timing_prediction: boolean;
  conversation_flow_prediction: boolean;
  engagement_optimization: boolean;
  predictive_confidence_threshold: number;
}

export interface QuantumConsciousnessConfig {
  enabled: boolean;
  decision_complexity_level: number;
  multi_dimensional_thinking: boolean;
  parallel_response_generation: boolean;
  consciousness_coherence_check: boolean;
  quantum_uncertainty_handling: boolean;
}

export interface SynesthesiaConfig {
  enabled: boolean;
  sensory_interpretation_modes: string[];
  cross_modal_translation: boolean;
  sensory_memory_enhancement: boolean;
  multi_sensory_response_generation: boolean;
}

export interface NeuralDreamsConfig {
  enabled: boolean;
  creativity_level: number;
  subconscious_pattern_recognition: boolean;
  dream_logic_integration: boolean;
  creative_response_generation: boolean;
  inspiration_mode_enabled: boolean;
}

// Dashboard and Analytics Types
export interface KellyDashboardOverview {
  total_accounts: number;
  active_accounts: number;
  connected_accounts: number;
  total_conversations_today: number;
  total_messages_today: number;
  average_engagement_score: number;
  average_safety_score: number;
  
  // Status indicators
  system_health: 'healthy' | 'warning' | 'critical';
  ai_performance: 'optimal' | 'good' | 'degraded';
  safety_alerts_count: number;
  conversations_requiring_review: number;
  
  // Claude AI specific metrics
  claude_metrics: ClaudeUsageMetrics;
  
  // Quick stats
  stage_distribution: Record<ConversationStage, number>;
  daily_message_trend: { date: string; count: number }[];
  engagement_trend: { date: string; score: number }[];
  safety_trend: { date: string; score: number }[];
}

export interface ConversationAnalytics {
  conversation_id: string;
  analytics: {
    message_frequency: { hour: number; count: number }[];
    topic_distribution: { topic: string; percentage: number }[];
    sentiment_timeline: { timestamp: string; sentiment: number }[];
    engagement_progression: { stage: string; score: number }[];
    ai_confidence_timeline: { timestamp: string; confidence: number }[];
    response_quality_scores: { message_id: string; quality: number }[];
  };
}

// Real-time Updates
export interface KellyWebSocketMessage {
  type: 'conversation_update' | 'safety_alert' | 'account_status' | 'message_received' | 'ai_insight' | 'claude_response_generation' | 'claude_cost_update';
  payload: any;
  timestamp: string;
  account_id?: string;
  conversation_id?: string;
}

export interface ClaudeResponseGenerationUpdate extends KellyWebSocketMessage {
  type: 'claude_response_generation';
  payload: {
    conversation_id: string;
    status: 'thinking' | 'generating' | 'complete' | 'error';
    model_used?: 'opus' | 'sonnet' | 'haiku';
    thinking_process?: string;
    partial_response?: string;
    confidence_so_far?: number;
    estimated_completion_time?: number;
    error_message?: string;
  };
}

export interface ClaudeCostUpdate extends KellyWebSocketMessage {
  type: 'claude_cost_update';
  payload: {
    account_id: string;
    daily_cost: number;
    monthly_cost: number;
    tokens_used_today: number;
    cost_per_conversation: Record<string, number>;
    budget_warning?: boolean;
    budget_exceeded?: boolean;
  };
}

export interface ClaudeUsageMetrics {
  total_tokens_used_today: number;
  total_cost_today: number;
  requests_by_model: {
    opus: number;
    sonnet: number;
    haiku: number;
  };
  average_response_time: number;
  success_rate: number;
  cost_trend: { date: string; cost: number }[];
  token_usage_trend: { date: string; tokens: number }[];
  model_performance: {
    opus: { avg_confidence: number; avg_quality: number };
    sonnet: { avg_confidence: number; avg_quality: number };
    haiku: { avg_confidence: number; avg_quality: number };
  };
}

export interface ConversationUpdate extends KellyWebSocketMessage {
  type: 'conversation_update';
  payload: {
    conversation_id: string;
    new_messages: ConversationMessage[];
    stage_change?: { from: ConversationStage; to: ConversationStage };
    engagement_score_change?: number;
    safety_score_change?: number;
    red_flags_added?: RedFlag[];
  };
}

export interface SafetyAlert extends KellyWebSocketMessage {
  type: 'safety_alert';
  payload: {
    alert_id: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    alert_type: string;
    description: string;
    conversation_id: string;
    account_id: string;
    requires_immediate_action: boolean;
    suggested_actions: string[];
  };
}

// Payment and Monetization
export interface PaymentSettings {
  enabled: boolean;
  discussion_allowed_after_stage: ConversationStage;
  pricing_tiers: PricingTier[];
  payment_methods: string[];
  conversation_value_threshold: number; // minimum conversation quality before discussing payment
  auto_payment_requests: boolean;
  custom_pricing_enabled: boolean;
}

export interface PricingTier {
  id: string;
  name: string;
  description: string;
  price_per_hour: number;
  price_per_message: number;
  minimum_commitment: string;
  features_included: string[];
  target_conversation_stages: ConversationStage[];
}

// Export main Kelly system interface
export interface KellySystem {
  accounts: KellyAccount[];
  ai_features: AIFeatureConfig;
  payment_settings: PaymentSettings;
  dashboard_overview: KellyDashboardOverview;
  active_conversations: KellyConversation[];
  safety_status: SafetyStatus;
  system_metrics: {
    uptime: number;
    performance_score: number;
    ai_response_time: number;
    error_rate: number;
    last_health_check: string;
  };
}