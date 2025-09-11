/**
 * Kelly AI Components - Phase 1 Export Index
 * WhatsApp Web/Slack-inspired conversation management system
 */

// Phase 1 Components
export { default as ConversationManagerV2 } from './ConversationManagerV2';
export { default as ConversationList } from './ConversationList';
export { default as MessageViewer } from './MessageViewer';
export { default as ManualInterventionPanel } from './ManualInterventionPanel';

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