/**
 * Loading Spinner Component
 * Apple-inspired loading animations
 */


import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'secondary' | 'accent' | 'white';
  text?: string;
  className?: string;
  variant?: 'spinner' | 'dots' | 'consciousness' | 'quantum';
}

export default function LoadingSpinner({
  size = 'md',
  color = 'primary',
  text,
  className,
  variant = 'spinner',
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12',
  };

  const colorClasses = {
    primary: 'text-consciousness-primary',
    secondary: 'text-consciousness-secondary',
    accent: 'text-consciousness-accent',
    white: 'text-white',
  };

  if (variant === 'consciousness') {
    return (
      <div className={cn('flex flex-col items-center space-y-4', className)}>
        <motion.div
          className="relative"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        >
          <div className={cn('rounded-full border-2 border-gray-200', sizeClasses[size])}>
            <motion.div
              className={cn(
                'rounded-full border-t-2 border-r-2 border-transparent',
                sizeClasses[size],
                colorClasses[color]
              )}
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            />
          </div>
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <span className="text-xs">🧠</span>
          </motion.div>
        </motion.div>
        {text && (
          <p className="text-sm text-text-secondary font-medium">{text}</p>
        )}
      </div>
    );
  }

  if (variant === 'quantum') {
    return (
      <div className={cn('flex flex-col items-center space-y-4', className)}>
        <div className="relative">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className={cn(
                'absolute rounded-full border-2',
                sizeClasses[size],
                colorClasses[color]
              )}
              style={{
                left: i * 8,
                top: i * 2,
              }}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.7, 1, 0.7],
                rotate: 360,
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.2,
              }}
            />
          ))}
        </div>
        {text && (
          <p className="text-sm text-text-secondary font-medium">{text}</p>
        )}
      </div>
    );
  }

  if (variant === 'dots') {
    return (
      <div className={cn('flex items-center space-x-2', className)}>
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={cn(
              'rounded-full',
              size === 'sm' ? 'w-2 h-2' : size === 'md' ? 'w-3 h-3' : 'w-4 h-4',
              colorClasses[color]
            )}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.5, 1, 0.5],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              delay: i * 0.2,
            }}
            style={{ backgroundColor: 'currentColor' }}
          />
        ))}
        {text && (
          <span className="ml-3 text-sm text-text-secondary font-medium">{text}</span>
        )}
      </div>
    );
  }

  // Default spinner variant
  return (
    <div className={cn('flex flex-col items-center space-y-4', className)}>
      <motion.div
        className={cn(
          'rounded-full border-2 border-gray-200 border-t-transparent',
          sizeClasses[size],
          colorClasses[color]
        )}
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        style={{ borderTopColor: 'currentColor' }}
      />
      {text && (
        <p className="text-sm text-text-secondary font-medium">{text}</p>
      )}
    </div>
  );
}

// Specialized loading components
export function ConsciousnessLoader({ text = 'Calibrating consciousness...' }) {
  return <LoadingSpinner variant="consciousness" size="lg" text={text} />;
}

export function QuantumLoader({ text = 'Entangling quantum states...' }) {
  return <LoadingSpinner variant="quantum" size="lg" color="secondary" text={text} />;
}

export function ProcessingDots({ text = 'Processing' }) {
  return <LoadingSpinner variant="dots" size="sm" text={text} />;
}