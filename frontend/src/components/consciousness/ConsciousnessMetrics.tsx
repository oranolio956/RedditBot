/**
 * Consciousness Metrics Component
 * Real-time display of consciousness state indicators
 */

// import React from 'react';
import { motion } from 'framer-motion';
import { formatRelativeTime, getConsciousnessColor } from '@/lib/utils';

interface ConsciousnessMetricsProps {
  cognitiveLoad: number;
  focusState: number;
  energyLevel: number;
  lastUpdate?: string;
  className?: string;
}

export default function ConsciousnessMetrics({
  cognitiveLoad,
  focusState,
  energyLevel,
  lastUpdate,
  className,
}: ConsciousnessMetricsProps) {
  const metrics = [
    {
      label: 'Cognitive Load',
      value: cognitiveLoad,
      unit: '%',
      color: cognitiveLoad > 0.8 ? 'text-states-stress' : cognitiveLoad > 0.6 ? 'text-consciousness-accent' : 'text-states-flow',
      description: 'Mental processing demand',
    },
    {
      label: 'Focus State',
      value: focusState,
      unit: '%',
      color: getConsciousnessColor(focusState),
      description: 'Attention concentration level',
    },
    {
      label: 'Energy Level',
      value: energyLevel,
      unit: '%',
      color: getConsciousnessColor(energyLevel),
      description: 'Cognitive energy reserves',
    },
  ];

  return (
    <div className={`space-y-4 ${className}`}>
      {metrics.map((metric, index) => (
        <motion.div
          key={metric.label}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="space-y-2"
        >
          {/* Metric header */}
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-text-primary">
                {metric.label}
              </h4>
              <p className="text-xs text-text-tertiary">
                {metric.description}
              </p>
            </div>
            <div className={`text-lg font-semibold ${metric.color}`}>
              {Math.round(metric.value * 100)}{metric.unit}
            </div>
          </div>

          {/* Progress bar */}
          <div className="relative">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <motion.div
                className={`h-2 rounded-full transition-all duration-500 ${
                  metric.value > 0.8 ? 'bg-states-flow' :
                  metric.value > 0.6 ? 'bg-consciousness-primary' :
                  metric.value > 0.4 ? 'bg-consciousness-accent' : 'bg-states-stress'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${metric.value * 100}%` }}
                transition={{ delay: index * 0.1 + 0.3, duration: 0.8 }}
              />
            </div>
            
            {/* Pulse animation for active metrics */}
            {metric.value > 0.7 && (
              <motion.div
                className="absolute inset-0 bg-white/30 rounded-full h-2"
                animate={{ opacity: [0, 0.5, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}
          </div>

          {/* Trend indicator */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-tertiary">
              {metric.value > 0.8 ? 'High' : metric.value > 0.6 ? 'Optimal' : metric.value > 0.4 ? 'Moderate' : 'Low'}
            </span>
            {lastUpdate && index === 0 && (
              <span className="text-text-tertiary">
                Updated {formatRelativeTime(lastUpdate)}
              </span>
            )}
          </div>
        </motion.div>
      ))}

      {/* Overall status indicator */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-6 p-3 rounded-lg bg-consciousness-primary/5 border border-consciousness-primary/20"
      >
        <div className="flex items-center space-x-3">
          <div className="w-3 h-3 rounded-full bg-states-flow animate-breathing" />
          <div>
            <p className="text-sm font-medium text-text-primary">
              Consciousness Active
            </p>
            <p className="text-xs text-text-tertiary">
              Real-time monitoring enabled
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}