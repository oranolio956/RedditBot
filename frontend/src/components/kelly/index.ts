/**
 * Kelly AI Components - Complete Export Index
 * Real-time monitoring and intervention system with Apple-inspired design
 */

// Phase 1 Components - Core conversation management
export { default as ConversationManagerV2 } from './ConversationManagerV2';
export { default as ConversationList } from './ConversationList';
export { default as MessageViewer } from './MessageViewer';
export { default as ManualInterventionPanel } from './ManualInterventionPanel';

// Phase 2 Components - Real-time monitoring and intervention
export { default as RealTimeMonitoringDashboard } from './RealTimeMonitoringDashboard';
export { default as LiveActivityFeed } from './LiveActivityFeed';
export { default as InterventionControlsPanel } from './InterventionControlsPanel';
export { default as AlertManagementSystem } from './AlertManagementSystem';
export { default as LiveStatusIndicators, MiniStatusIndicator } from './LiveStatusIndicators';
export { default as EmergencyOverridePanel } from './EmergencyOverridePanel';

// Existing Components
export { default as ConversationManager } from './ConversationManager';
export { default as ClaudeAIDashboard } from './ClaudeAIDashboard';
export { default as ClaudeSettingsPanel } from './ClaudeSettingsPanel';
export { default as SafetyDashboard } from './SafetyDashboard';
export { default as TelegramConnectModal } from './TelegramConnectModal';

// Type exports for component props
export type {
  ConversationManagerV2Props,
  ConversationListProps,
  MessageViewerProps,
  ManualInterventionPanelProps
} from './types';

// Phase 2 type exports
export type {
  RealTimeMonitoringDashboardProps
} from './RealTimeMonitoringDashboard';

export type {
  LiveActivityFeedProps
} from './LiveActivityFeed';

export type {
  InterventionControlsPanelProps
} from './InterventionControlsPanel';

export type {
  AlertManagementSystemProps
} from './AlertManagementSystem';

export type {
  LiveStatusIndicatorsProps,
  MiniStatusIndicatorProps
} from './LiveStatusIndicators';

export type {
  EmergencyOverridePanelProps
} from './EmergencyOverridePanel';