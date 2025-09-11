/**
 * Kelly Component Type Definitions
 * TypeScript interfaces for Phase 1 conversation management components
 */

import { KellyConversation, SafetyAlert } from '@/types/kelly';

// ConversationManagerV2 Props
export interface ConversationManagerV2Props {
  className?: string;
  onConversationSelect?: (conversation: KellyConversation) => void;
  onSafetyAlert?: (alert: SafetyAlert) => void;
}

// ConversationList Props
export interface ConversationListProps {
  conversations: KellyConversation[];
  selectedConversation: KellyConversation | null;
  onConversationSelect: (conversation: KellyConversation) => void;
  isLoading?: boolean;
  collapsed?: boolean;
  searchQuery?: string;
  className?: string;
}

// MessageViewer Props
export interface MessageViewerProps {
  conversation: KellyConversation;
  manualMode?: boolean;
  onMessageSent?: () => void;
  className?: string;
}

// ManualInterventionPanel Props
export interface ManualInterventionPanelProps {
  conversation: KellyConversation;
  manualMode: boolean;
  onManualModeToggle: (enabled: boolean) => void;
  className?: string;
}

// Virtual List Item Props
export interface VirtualListItemData<T = any> {
  items: T[];
  selectedItem?: T | null;
  onItemSelect?: (item: T) => void;
  collapsed?: boolean;
  searchQuery?: string;
}

// Message Actions
export type MessageAction = 
  | 'copy'
  | 'flag'
  | 'rate_up'
  | 'rate_down'
  | 'edit'
  | 'delete'
  | 'reply';

// Quick Reply Template
export interface QuickReplyTemplate {
  id: string;
  text: string;
  category: 'greeting' | 'question' | 'closing' | 'concern' | 'redirect';
  confidence: number;
}

// Component State Interfaces
export interface ConversationFilterState {
  searchQuery: string;
  stageFilter: 'all' | string;
  statusFilter: 'all' | 'active' | 'flagged' | 'review';
  sortBy: 'recent' | 'engagement' | 'safety';
}

export interface UIState {
  sidebarCollapsed: boolean;
  showFilters: boolean;
  showQuickReplies: boolean;
  showAdvanced: boolean;
  manualMode: boolean;
  notifications: boolean;
}

// Real-time Update Handlers
export type ConversationUpdateHandler = (update: any) => void;
export type SafetyAlertHandler = (alert: SafetyAlert) => void;
export type ClaudeGenerationHandler = (update: any) => void;