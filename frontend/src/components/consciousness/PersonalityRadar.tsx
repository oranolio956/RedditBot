/**
 * Personality Radar Chart Component
 * Visualizes Big Five personality traits in a radar/spider chart
 */

// import React from 'react';
import { motion } from 'framer-motion';

interface PersonalityRadarProps {
  traits: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
  size?: number;
  className?: string;
}

export default function PersonalityRadar({ traits, size = 250, className }: PersonalityRadarProps) {
  const center = size / 2;
  const radius = (size / 2) - 40;
  const angleStep = (2 * Math.PI) / 5;

  // Trait labels and colors
  const traitConfig = [
    { key: 'openness', label: 'Openness', color: '#007AFF' },
    { key: 'conscientiousness', label: 'Conscientiousness', color: '#5856D6' },
    { key: 'extraversion', label: 'Extraversion', color: '#FF9500' },
    { key: 'agreeableness', label: 'Agreeableness', color: '#30D158' },
    { key: 'neuroticism', label: 'Neuroticism', color: '#FF453A' },
  ];

  // Calculate points for each trait
  const points = traitConfig.map((trait, index) => {
    const angle = index * angleStep - Math.PI / 2; // Start from top
    const value = traits[trait.key as keyof typeof traits];
    const distance = value * radius;
    
    return {
      ...trait,
      x: center + Math.cos(angle) * distance,
      y: center + Math.sin(angle) * distance,
      labelX: center + Math.cos(angle) * (radius + 25),
      labelY: center + Math.sin(angle) * (radius + 25),
      angle,
      value,
    };
  });

  // Create the polygon path
  const polygonPath = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ') + ' Z';

  // Create concentric guide circles
  const guideCircles = [0.2, 0.4, 0.6, 0.8, 1.0].map(scale => ({
    radius: radius * scale,
    opacity: 0.1 + (scale * 0.1),
  }));

  // Create guide lines from center to each axis
  const guideLines = traitConfig.map((_, index) => {
    const angle = index * angleStep - Math.PI / 2;
    return {
      x1: center,
      y1: center,
      x2: center + Math.cos(angle) * radius,
      y2: center + Math.sin(angle) * radius,
    };
  });

  return (
    <div className={className}>
      <svg width={size} height={size} className="overflow-visible">
        {/* Guide circles */}
        {guideCircles.map((circle, index) => (
          <circle
            key={index}
            cx={center}
            cy={center}
            r={circle.radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
            opacity={circle.opacity}
            className="text-gray-300"
          />
        ))}

        {/* Guide lines */}
        {guideLines.map((line, index) => (
          <line
            key={index}
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            stroke="currentColor"
            strokeWidth="1"
            opacity="0.2"
            className="text-gray-300"
          />
        ))}

        {/* Personality polygon */}
        <motion.path
          d={polygonPath}
          fill="rgba(0, 122, 255, 0.1)"
          stroke="var(--consciousness-primary)"
          strokeWidth="2"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.5, ease: 'easeInOut' }}
        />

        {/* Data points */}
        {points.map((point, index) => (
          <motion.g key={point.key}>
            {/* Point circle */}
            <motion.circle
              cx={point.x}
              cy={point.y}
              r="4"
              fill={point.color}
              stroke="white"
              strokeWidth="2"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="drop-shadow-sm"
            />
            
            {/* Value indicator */}
            <motion.circle
              cx={point.x}
              cy={point.y}
              r="8"
              fill="none"
              stroke={point.color}
              strokeWidth="1"
              opacity="0.3"
              initial={{ scale: 0 }}
              animate={{ scale: point.value }}
              transition={{ duration: 1, delay: index * 0.1 }}
            />
          </motion.g>
        ))}

        {/* Labels */}
        {points.map((point, index) => (
          <motion.g key={`label-${point.key}`}>
            <motion.text
              x={point.labelX}
              y={point.labelY}
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-xs font-medium fill-current text-text-secondary"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
            >
              {point.label}
            </motion.text>
            <motion.text
              x={point.labelX}
              y={point.labelY + 12}
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-xs font-bold fill-current text-text-primary"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.7 + index * 0.1 }}
            >
              {Math.round(point.value * 100)}%
            </motion.text>
          </motion.g>
        ))}

        {/* Center point */}
        <circle
          cx={center}
          cy={center}
          r="2"
          fill="var(--consciousness-primary)"
          className="drop-shadow-sm"
        />
      </svg>
    </div>
  );
}