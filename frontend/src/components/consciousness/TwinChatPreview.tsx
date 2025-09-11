/**
 * Twin Chat Preview Component
 * Shows recent conversations with digital twin
 */

// 
import { motion } from 'framer-motion';
import { formatRelativeTime } from '@/lib/utils';

interface Conversation {
  id: string;
  message: string;
  response: string;
  confidence: number;
  timestamp: string;
  emotional_tone?: string;
}

interface TwinChatPreviewProps {
  conversations: Conversation[];
  className?: string;
}

export default function TwinChatPreview({ conversations, className }: TwinChatPreviewProps) {
  if (!conversations || conversations.length === 0) {
    return (
      <div className={`text-center py-8 text-text-tertiary ${className}`}>
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-consciousness-primary/10 flex items-center justify-center">
          <span className="text-2xl">🧠</span>
        </div>
        <p className="text-sm">No conversations yet</p>
        <p className="text-xs mt-1">Start chatting with your digital twin!</p>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {conversations.map((conversation, index) => (
        <motion.div
          key={conversation.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="border border-gray-200 rounded-lg p-4 hover:shadow-card transition-shadow"
        >
          {/* User message */}
          <div className="flex justify-end mb-3">
            <div className="bg-consciousness-primary text-white rounded-lg px-3 py-2 max-w-xs">
              <p className="text-sm">{conversation.message}</p>
            </div>
          </div>

          {/* Twin response */}
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-consciousness-gradient flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm">🤖</span>
            </div>
            <div className="flex-1">
              <div className="bg-surface-secondary rounded-lg px-3 py-2">
                <p className="text-sm text-text-primary">{conversation.response}</p>
              </div>
              
              {/* Metadata */}
              <div className="flex items-center justify-between mt-2 text-xs text-text-tertiary">
                <div className="flex items-center space-x-3">
                  <span>
                    Confidence: {Math.round(conversation.confidence * 100)}%
                  </span>
                  {conversation.emotional_tone && (
                    <span>
                      Tone: {conversation.emotional_tone}
                    </span>
                  )}
                </div>
                <span>{formatRelativeTime(conversation.timestamp)}</span>
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}