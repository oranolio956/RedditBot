/**
 * Connection Status Component
 * Shows WebSocket connection status with visual indicators
 */

import React from 'react';
import { motion } from 'framer-motion';
import { WifiIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface ConnectionStatusProps {
  connected: boolean;
  className?: string;
}

export default function ConnectionStatus({ connected, className }: ConnectionStatusProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`flex items-center space-x-2 ${className}`}
    >
      {connected ? (
        <>
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-2 h-2 bg-states-flow rounded-full"
          />
          <WifiIcon className="w-4 h-4 text-states-flow" />
          <span className="text-xs text-states-flow font-medium hidden sm:block">
            Real-time
          </span>
        </>
      ) : (
        <>
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
            className="w-2 h-2 bg-consciousness-accent rounded-full"
          />
          <ExclamationTriangleIcon className="w-4 h-4 text-consciousness-accent" />
          <span className="text-xs text-consciousness-accent font-medium hidden sm:block">
            Reconnecting...
          </span>
        </>
      )}
    </motion.div>
  );
}