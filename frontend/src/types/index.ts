// Core API Types for AI Consciousness Platform

export interface User {
  id: string;
  telegram_id?: number;
  username?: string;
  full_name?: string;
  first_name?: string;
  last_name?: string;
  email?: string;
  is_premium: boolean;
  subscription_type: 'free' | 'premium' | 'enterprise';
  created_at: string;
  updated_at: string;
  last_activity?: string;
  is_active: boolean;
  preferences: UserPreferences;
  stats: UserStats;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  notifications: {
    insights: boolean;
    patterns: boolean;
    breakthroughs: boolean;
    daily_summary: boolean;
  };
  privacy: {
    data_sharing: boolean;
    analytics: boolean;
    export_enabled: boolean;
  };
  consciousness: {
    update_frequency: number;
    calibration_sensitivity: number;
    pattern_detection: boolean;
  };
}

export interface UserStats {
  total_sessions: number;
  consciousness_sessions: number;
  memory_palace_rooms: number;
  stored_memories: number;
  insights_generated: number;
  patterns_discovered: number;
  quantum_connections: number;
  dreams_recorded: number;
}

// Consciousness Mirroring Types
export interface CognitiveProfile {
  user_id: string;
  big_five: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
  cognitive_load: number;
  focus_state: number;
  flow_state: number;
  energy_level: number;
  clarity_score: number;
  confidence_level: number;
  last_updated: string;
  calibration_accuracy: number;
}

export interface ConsciousnessUpdate {
  keystroke_patterns?: KeystrokePattern[];
  response_times?: number[];
  decision_context?: string;
  emotional_state?: EmotionalState;
  cognitive_load?: number;
}

export interface KeystrokePattern {
  timestamp: number;
  duration: number;
  pressure?: number;
  rhythm_pattern: number[];
}

export interface TwinResponse {
  message: string;
  confidence: number;
  reasoning: string;
  emotional_tone: string;
  predicted_user_response?: string;
}

export interface PersonalityEvolution {
  user_id: string;
  timeline: EvolutionPoint[];
  significant_changes: SignificantChange[];
  prediction_accuracy: number;
}

export interface EvolutionPoint {
  timestamp: string;
  traits: CognitiveProfile['big_five'];
  trigger_event?: string;
  confidence: number;
}

export interface SignificantChange {
  trait: keyof CognitiveProfile['big_five'];
  change_magnitude: number;
  date_detected: string;
  potential_causes: string[];
  confidence: number;
}

// Memory Palace Types
export interface MemoryPalace {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  total_rooms: number;
  total_memories: number;
  recall_accuracy: number;
  spatial_layout: SpatialLayout;
  created_at: string;
  updated_at: string;
  backup_status: 'none' | 'pending' | 'complete' | 'failed';
}

export interface MemoryRoom {
  id: string;
  palace_id: string;
  name: string;
  description?: string;
  position: Vector3D;
  size: Vector3D;
  color_scheme: string;
  memory_count: number;
  associations: string[];
  created_at: string;
}

export interface StoredMemory {
  id: string;
  room_id: string;
  title: string;
  content: string;
  memory_type: 'text' | 'image' | 'audio' | 'video' | 'concept';
  position: Vector3D;
  associations: string[];
  recall_strength: number;
  created_at: string;
  last_accessed: string;
  access_count: number;
}

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface SpatialLayout {
  dimensions: Vector3D;
  room_positions: Record<string, Vector3D>;
  connection_paths: ConnectionPath[];
}

export interface ConnectionPath {
  from_room: string;
  to_room: string;
  path_type: 'corridor' | 'stairs' | 'portal' | 'bridge';
  waypoints: Vector3D[];
}

// Temporal Archaeology Types
export interface ConversationFragment {
  id: string;
  user_id: string;
  content: string;
  timestamp: string;
  source_platform: string;
  confidence: number;
  reconstruction_status: 'original' | 'reconstructed' | 'partially_reconstructed';
  linguistic_markers: LinguisticMarker[];
}

export interface LinguisticMarker {
  type: 'vocabulary' | 'syntax' | 'style' | 'emotion' | 'topic';
  value: string;
  confidence: number;
  frequency: number;
}

export interface TemporalPattern {
  user_id: string;
  pattern_type: 'communication_style' | 'topic_evolution' | 'emotional_patterns' | 'linguistic_drift';
  pattern_data: Record<string, any>;
  time_range: {
    start: string;
    end: string;
  };
  significance: number;
  confidence: number;
}

export interface GhostConversation {
  id: string;
  user_id: string;
  reconstructed_messages: ReconstructedMessage[];
  confidence_score: number;
  time_period: string;
  conversation_theme: string;
}

export interface ReconstructedMessage {
  content: string;
  estimated_timestamp: string;
  confidence: number;
  reconstruction_method: string;
  original_fragments: string[];
}

// Emotional Intelligence Types
export interface EmotionalProfile {
  user_id: string;
  current_state: EmotionalState;
  baseline_traits: {
    emotional_stability: number;
    empathy_level: number;
    social_awareness: number;
    self_regulation: number;
    motivation: number;
  };
  recent_patterns: EmotionalPattern[];
  growth_areas: string[];
  strengths: string[];
}

export interface EmotionalState {
  primary_emotion: string;
  secondary_emotions: string[];
  intensity: number;
  valence: number; // positive/negative
  arousal: number; // calm/excited
  confidence: number;
  timestamp: string;
  context?: string;
}

export interface MoodEntry {
  id: string;
  user_id: string;
  mood_score: number; // -5 to 5
  emotions: string[];
  energy_level: number;
  stress_level: number;
  notes?: string;
  timestamp: string;
  context_tags: string[];
}

export interface EmotionalPattern {
  pattern_type: 'daily_cycle' | 'weekly_trend' | 'trigger_response' | 'seasonal';
  description: string;
  strength: number;
  time_range: string;
  predictions: EmotionalPrediction[];
}

export interface EmotionalPrediction {
  predicted_state: EmotionalState;
  probability: number;
  time_horizon: string;
  influencing_factors: string[];
}

export interface CompatibilityAnalysis {
  user1_id: string;
  user2_id: string;
  overall_score: number;
  compatibility_areas: {
    communication_style: number;
    emotional_resonance: number;
    value_alignment: number;
    conflict_resolution: number;
  };
  improvement_suggestions: string[];
  relationship_type: 'friendship' | 'romantic' | 'professional';
}

// Synesthesia Engine Types
export interface SynestheticProfile {
  user_id: string;
  natural_associations: SensoryMapping[];
  trained_associations: SensoryMapping[];
  sensitivity_levels: {
    visual: number;
    auditory: number;
    tactile: number;
    gustatory: number;
    olfactory: number;
  };
  dominant_modalities: string[];
  conversion_accuracy: number;
}

export interface SensoryMapping {
  id: string;
  from_modality: SensoryModality;
  to_modality: SensoryModality;
  trigger_value: any;
  response_value: any;
  strength: number;
  consistency: number;
  created_at: string;
  usage_count: number;
}

export interface SensoryModality {
  type: 'visual' | 'auditory' | 'tactile' | 'gustatory' | 'olfactory' | 'temporal' | 'spatial';
  subtype?: string;
  properties: Record<string, any>;
}

export interface SynestheticExperience {
  id: string;
  user_id: string;
  input_data: any;
  converted_data: any;
  conversion_type: string;
  quality_score: number;
  shareable: boolean;
  created_at: string;
  view_count: number;
  likes: number;
}

// Neural Dreams Types
export interface DreamSession {
  id: string;
  user_id: string;
  dream_data: DreamData;
  interpretation: DreamInterpretation;
  session_duration: number;
  quality_score: number;
  lucidity_level: number;
  created_at: string;
  shared: boolean;
}

export interface DreamData {
  visual_elements: DreamElement[];
  emotional_tone: EmotionalState;
  narrative_structure: NarrativeElement[];
  symbolic_content: Symbol[];
  recurring_themes: string[];
}

export interface DreamElement {
  type: 'visual' | 'auditory' | 'tactile' | 'conceptual';
  content: any;
  vividness: number;
  emotional_charge: number;
  timestamp_in_dream: number;
}

export interface NarrativeElement {
  sequence: number;
  description: string;
  characters: string[];
  setting: string;
  action: string;
  emotional_arc: EmotionalState[];
}

export interface Symbol {
  symbol: string;
  personal_meaning?: string;
  archetypal_meaning?: string;
  frequency: number;
  emotional_association: number;
}

export interface DreamInterpretation {
  summary: string;
  key_themes: string[];
  psychological_insights: string[];
  recommendations: string[];
  confidence: number;
  interpretation_method: string;
}

// Quantum Consciousness Types
export interface QuantumNetwork {
  user_id: string;
  entangled_users: EntanglementPair[];
  coherence_level: number;
  network_stability: number;
  total_connections: number;
  active_connections: number;
  quantum_state: QuantumState;
}

export interface EntanglementPair {
  id: string;
  user1_id: string;
  user2_id: string;
  entanglement_strength: number;
  coherence_duration: number;
  last_interaction: string;
  synchronization_events: SynchronizationEvent[];
  created_at: string;
}

export interface QuantumState {
  superposition_active: boolean;
  measurement_count: number;
  decoherence_rate: number;
  quantum_information: any;
  observation_effects: ObservationEffect[];
}

export interface SynchronizationEvent {
  timestamp: string;
  event_type: string;
  participants: string[];
  correlation_strength: number;
  confidence: number;
}

export interface ObservationEffect {
  observer_id: string;
  observed_system: string;
  effect_magnitude: number;
  measurement_type: string;
  timestamp: string;
}

export interface ThoughtTeleportation {
  id: string;
  sender_id: string;
  receiver_id: string;
  thought_content: any;
  transmission_fidelity: number;
  reception_confirmation: boolean;
  timestamp: string;
  quantum_channel_quality: number;
}

// Meta Reality Types
export interface RealityLayer {
  id: string;
  user_id: string;
  layer_name: string;
  description: string;
  reality_filters: RealityFilter[];
  perception_modifications: PerceptionMod[];
  consensus_level: number;
  stability: number;
  created_at: string;
}

export interface RealityFilter {
  filter_type: 'visual' | 'auditory' | 'conceptual' | 'temporal' | 'spatial';
  parameters: Record<string, any>;
  intensity: number;
  active: boolean;
}

export interface PerceptionMod {
  modality: string;
  modification_type: 'enhancement' | 'suppression' | 'alteration';
  target: string;
  effect_parameters: Record<string, any>;
  strength: number;
}

export interface ConsensusReality {
  reality_anchor_points: AnchorPoint[];
  participant_count: number;
  consensus_strength: number;
  reality_stability: number;
  shared_elements: SharedElement[];
  divergence_points: DivergencePoint[];
}

export interface AnchorPoint {
  id: string;
  coordinates: Vector3D;
  anchor_type: string;
  stability: number;
  participant_agreements: number;
}

export interface SharedElement {
  element_id: string;
  element_type: string;
  consensus_level: number;
  participants: string[];
  properties: Record<string, any>;
}

export interface DivergencePoint {
  location: Vector3D;
  divergence_type: string;
  affected_participants: string[];
  severity: number;
  resolution_status: 'pending' | 'resolved' | 'persistent';
}

// Telegram Bot Types
export interface TelegramBotStatus {
  is_running: boolean;
  uptime: number;
  active_sessions: number;
  total_users: number;
  messages_processed_today: number;
  current_rate_limit: number;
  webhook_status: WebhookStatus;
  circuit_breakers: CircuitBreakerStatus[];
  anti_ban_metrics: AntiBanMetrics;
}

export interface WebhookStatus {
  url: string;
  is_active: boolean;
  last_error?: string;
  pending_updates: number;
  max_connections: number;
}

export interface CircuitBreakerStatus {
  name: string;
  state: 'closed' | 'open' | 'half-open';
  failure_count: number;
  last_failure?: string;
  next_attempt?: string;
}

export interface AntiBanMetrics {
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  message_rate: number;
  user_reports: number;
  behavioral_score: number;
  evasion_tactics_active: string[];
  last_risk_assessment: string;
}

export interface TelegramSession {
  id: string;
  user_id: string;
  telegram_user_id: number;
  chat_id: number;
  session_type: 'private' | 'group' | 'channel';
  status: 'active' | 'paused' | 'expired';
  started_at: string;
  last_activity: string;
  message_count: number;
  context: SessionContext;
}

export interface SessionContext {
  current_feature?: string;
  conversation_state: Record<string, any>;
  user_preferences: Record<string, any>;
  active_flows: string[];
  context_history: ContextSnapshot[];
}

export interface ContextSnapshot {
  timestamp: string;
  feature: string;
  state: Record<string, any>;
  user_action: string;
}

// WebSocket Types
export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: string;
  user_id?: string;
  session_id?: string;
}

export interface ConsciousnessUpdate extends WebSocketMessage {
  type: 'consciousness_update';
  payload: {
    profile: CognitiveProfile;
    changes: Partial<CognitiveProfile>;
    confidence: number;
  };
}

export interface MetricsUpdate extends WebSocketMessage {
  type: 'metrics_update';
  payload: {
    telegram_metrics: TelegramBotStatus;
    user_metrics: Record<string, number>;
    system_metrics: Record<string, number>;
  };
}

export interface EmotionalStateUpdate extends WebSocketMessage {
  type: 'emotional_state_update';
  payload: {
    emotional_state: EmotionalState;
    insights: string[];
    recommendations: string[];
  };
}

export interface MemoryPalaceEvent extends WebSocketMessage {
  type: 'memory_palace_event';
  payload: {
    event_type: 'room_created' | 'memory_stored' | 'palace_backup' | 'recall_session';
    palace_id: string;
    details: Record<string, any>;
  };
}

export interface QuantumEvent extends WebSocketMessage {
  type: 'quantum_event';
  payload: {
    event_type: 'entanglement_created' | 'coherence_lost' | 'thought_received' | 'synchronization_detected';
    participants: string[];
    quantum_data: Record<string, any>;
  };
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T = any> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    per_page: number;
    total: number;
    pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  timestamp: string;
  request_id: string;
}

// Component Props Types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  text?: string;
}

export interface ErrorProps extends BaseComponentProps {
  error: Error | string;
  retry?: () => void;
  showDetails?: boolean;
}

export interface CardProps extends BaseComponentProps {
  title?: string;
  subtitle?: string;
  footer?: React.ReactNode;
  actions?: React.ReactNode;
  elevation?: 'low' | 'medium' | 'high';
  glassmorphism?: boolean;
}

// Chart and Visualization Types
export interface ChartDataPoint {
  x: number | string | Date;
  y: number;
  label?: string;
  color?: string;
  metadata?: Record<string, any>;
}

export interface TimeSeriesData {
  timestamp: string | Date;
  value: number;
  label?: string;
  category?: string;
}

export interface NetworkGraphNode {
  id: string;
  label: string;
  group?: string;
  size?: number;
  color?: string;
  position?: { x: number; y: number };
  metadata?: Record<string, any>;
}

export interface NetworkGraphEdge {
  id: string;
  source: string;
  target: string;
  weight?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
}

// Form Types
export interface FormFieldProps {
  name: string;
  label: string;
  placeholder?: string;
  required?: boolean;
  disabled?: boolean;
  error?: string;
  helper?: string;
}

export interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
  group?: string;
}

// Theme and UI Types
export type ThemeMode = 'light' | 'dark' | 'system';

export interface UITheme {
  mode: ThemeMode;
  colors: Record<string, string>;
  typography: Record<string, any>;
  spacing: Record<string, string>;
  animation: Record<string, any>;
}

export interface NotificationProps {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    handler: () => void;
  };
  persistent?: boolean;
  read?: boolean;
}