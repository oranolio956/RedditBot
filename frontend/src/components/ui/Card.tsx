/**
 * Apple-inspired Card Component
 * Implements design system specifications with glassmorphism support
 */

import React, { forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const cardVariants = cva(
  // Base styles
  'rounded-lg border bg-surface-primary text-text-primary shadow-card transition-all duration-300 ease-in-out',
  {
    variants: {
      variant: {
        default: 'border-gray-200',
        consciousness: 'border-consciousness-primary/20 shadow-consciousness',
        breakthrough: 'border-consciousness-accent/20 shadow-glow animate-pulse-glow',
        flow: 'border-states-flow/20 shadow-elevated animate-breathing',
        quantum: 'border-consciousness-secondary/20 shadow-dramatic',
        emotional: 'border-pink-200 shadow-card',
        memory: 'border-purple-200 shadow-elevated',
        dream: 'border-indigo-200 shadow-card',
        synesthesia: 'border-rainbow shadow-elevated',
      },
      size: {
        sm: 'p-4',
        md: 'p-6',
        lg: 'p-8',
        xl: 'p-10',
      },
      elevation: {
        low: 'shadow-card',
        medium: 'shadow-elevated',
        high: 'shadow-dramatic',
      },
      glassmorphism: {
        true: 'bg-surface-elevated backdrop-blur-md border-white/20',
        false: '',
      },
      interactive: {
        true: 'hover:shadow-elevated hover:-translate-y-1 cursor-pointer',
        false: '',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'md',
      elevation: 'low',
      glassmorphism: false,
      interactive: false,
    },
  }
);

export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {
  asChild?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, size, elevation, glassmorphism, interactive, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(cardVariants({ variant, size, elevation, glassmorphism, interactive }), className)}
      {...props}
    />
  )
);

Card.displayName = 'Card';

// Card Header
export interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  subtitle?: string;
}

const CardHeader = forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className, children, subtitle, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('flex flex-col space-y-1.5 pb-6', className)}
      {...props}
    >
      {children}
      {subtitle && (
        <p className="text-caption-text text-text-secondary">{subtitle}</p>
      )}
    </div>
  )
);

CardHeader.displayName = 'CardHeader';

// Card Title
const CardTitle = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, children, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn('text-insight-subtitle font-semibold leading-none tracking-tight', className)}
      {...props}
    >
      {children}
    </h3>
  )
);

CardTitle.displayName = 'CardTitle';

// Card Description
const CardDescription = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p
      ref={ref}
      className={cn('text-body-text text-text-secondary', className)}
      {...props}
    />
  )
);

CardDescription.displayName = 'CardDescription';

// Card Content
const CardContent = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('pt-0', className)} {...props} />
  )
);

CardContent.displayName = 'CardContent';

// Card Footer
const CardFooter = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('flex items-center pt-6', className)}
      {...props}
    />
  )
);

CardFooter.displayName = 'CardFooter';

// Specialized consciousness cards
export interface InsightCardProps extends CardProps {
  title: string;
  confidence: number;
  icon?: React.ReactNode;
  onExplore?: () => void;
}

const InsightCard = forwardRef<HTMLDivElement, InsightCardProps>(
  ({ title, confidence, icon, onExplore, children, className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="breakthrough"
      interactive={!!onExplore}
      className={cn('group', className)}
      onClick={onExplore}
      {...props}
    >
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-2">
            {icon && <span className="text-consciousness-accent">{icon}</span>}
            <div>
              <CardTitle className="group-hover:text-consciousness-accent transition-colors">
                {title}
              </CardTitle>
              <div className="flex items-center space-x-2 mt-2">
                <span className="text-caption-text text-text-tertiary">Confidence:</span>
                <span className="text-caption-text font-medium">{Math.round(confidence * 100)}%</span>
                <div className="w-12 h-1 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-consciousness-accent transition-all duration-300"
                    style={{ width: `${confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
          {onExplore && (
            <span className="text-consciousness-accent group-hover:text-consciousness-primary transition-colors">
              🔍
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  )
);

InsightCard.displayName = 'InsightCard';

// Consciousness state card
export interface ConsciousnessCardProps extends CardProps {
  state: {
    focus: number;
    flow: number;
    clarity: number;
    energy: number;
  };
  status: string;
}

const ConsciousnessCard = forwardRef<HTMLDivElement, ConsciousnessCardProps>(
  ({ state, status, className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="consciousness"
      className={cn('', className)}
      {...props}
    >
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <span>🧠</span>
          <span>Consciousness State</span>
        </CardTitle>
        <CardDescription>{status}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(state).map(([key, value]) => (
            <div key={key} className="space-y-1">
              <div className="flex justify-between text-caption-text">
                <span className="capitalize">{key}:</span>
                <span className="font-medium">{Math.round(value * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all duration-300',
                    value >= 0.8 ? 'bg-states-flow' :
                    value >= 0.6 ? 'bg-consciousness-primary' :
                    value >= 0.4 ? 'bg-consciousness-accent' : 'bg-states-stress'
                  )}
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
);

ConsciousnessCard.displayName = 'ConsciousnessCard';

// Memory palace card
export interface MemoryPalaceCardProps extends CardProps {
  palace: {
    name: string;
    total_rooms: number;
    total_memories: number;
    recall_accuracy: number;
  };
  onVisit?: () => void;
}

const MemoryPalaceCard = forwardRef<HTMLDivElement, MemoryPalaceCardProps>(
  ({ palace, onVisit, className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="memory"
      interactive={!!onVisit}
      className={cn('group', className)}
      onClick={onVisit}
      {...props}
    >
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <span>🏛️</span>
          <span>{palace.name}</span>
        </CardTitle>
        <CardDescription>
          {palace.total_rooms} rooms • {palace.total_memories} items • {Math.round(palace.recall_accuracy * 100)}% recall
        </CardDescription>
      </CardHeader>
      {onVisit && (
        <CardFooter>
          <div className="text-consciousness-primary group-hover:text-consciousness-accent transition-colors">
            Visit Palace →
          </div>
        </CardFooter>
      )}
    </Card>
  )
);

MemoryPalaceCard.displayName = 'MemoryPalaceCard';

// Quantum entanglement card
export interface QuantumCardProps extends CardProps {
  entanglement: {
    user2_id: string;
    entanglement_strength: number;
    coherence_duration: number;
    last_interaction: string;
  };
  onInteract?: () => void;
}

const QuantumCard = forwardRef<HTMLDivElement, QuantumCardProps>(
  ({ entanglement, onInteract, className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="quantum"
      interactive={!!onInteract}
      glassmorphism
      className={cn('group', className)}
      onClick={onInteract}
      {...props}
    >
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <span>🔗</span>
          <span>Quantum Entanglement</span>
        </CardTitle>
        <CardDescription>
          Strength: {Math.round(entanglement.entanglement_strength * 100)}% • 
          Duration: {entanglement.coherence_duration}s
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between text-caption-text">
            <span>Connected to:</span>
            <span className="font-medium">{entanglement.user2_id}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="h-2 bg-consciousness-secondary rounded-full transition-all duration-300 animate-pulse-glow"
              style={{ width: `${entanglement.entanglement_strength * 100}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
);

QuantumCard.displayName = 'QuantumCard';

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
  InsightCard,
  ConsciousnessCard,
  MemoryPalaceCard,
  QuantumCard,
  cardVariants,
};