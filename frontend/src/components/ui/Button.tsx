/**
 * Apple-inspired Button Component
 * Implements design system specifications with accessibility
 */

import React, { forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  // Base styles
  'inline-flex items-center justify-center rounded-lg text-body-text font-sf-pro font-medium ring-offset-surface-primary transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-consciousness-primary focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 active:scale-95',
  {
    variants: {
      variant: {
        // Primary consciousness button
        primary: 
          'bg-consciousness-primary text-white hover:bg-blue-600 shadow-card hover:shadow-elevated',
        
        // Secondary consciousness button  
        secondary:
          'bg-consciousness-secondary text-white hover:bg-purple-600 shadow-card hover:shadow-elevated',
          
        // Accent breakthrough button
        accent:
          'bg-consciousness-accent text-white hover:bg-orange-600 shadow-card hover:shadow-elevated',
          
        // Success/flow state button
        success:
          'bg-states-flow text-white hover:bg-green-600 shadow-card hover:shadow-elevated',
          
        // Warning/stress button
        warning:
          'bg-states-stress text-white hover:bg-red-600 shadow-card hover:shadow-elevated',
          
        // Outlined button
        outline:
          'border border-consciousness-primary text-consciousness-primary hover:bg-consciousness-primary hover:text-white',
          
        // Ghost button
        ghost:
          'text-consciousness-primary hover:bg-consciousness-primary/10',
          
        // Surface button
        surface:
          'bg-surface-secondary text-text-primary hover:bg-gray-300 border border-gray-200 shadow-sm',
          
        // Link style
        link:
          'text-consciousness-primary underline-offset-4 hover:underline p-0 h-auto',
          
        // Destructive
        destructive:
          'bg-states-stress text-white hover:bg-red-600 shadow-card',
      },
      size: {
        sm: 'h-9 rounded-md px-3 text-sm',
        md: 'h-10 px-4 py-2',
        lg: 'h-11 rounded-lg px-8 text-lg',
        xl: 'h-12 rounded-lg px-10 text-xl',
        icon: 'h-10 w-10',
      },
      animation: {
        none: '',
        breathing: 'animate-breathing',
        glow: 'animate-pulse-glow',
        bounce: 'hover:animate-bounce',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
      animation: 'none',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant, 
    size, 
    animation,
    asChild = false, 
    loading = false,
    leftIcon,
    rightIcon,
    children,
    disabled,
    ...props 
  }, ref) => {
    const isDisabled = disabled || loading;

    return (
      <button
        className={cn(buttonVariants({ variant, size, animation }), className)}
        ref={ref}
        disabled={isDisabled}
        {...props}
      >
        {loading && (
          <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
        )}
        {leftIcon && !loading && (
          <span className="mr-2">{leftIcon}</span>
        )}
        {children}
        {rightIcon && (
          <span className="ml-2">{rightIcon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// Specialized consciousness buttons
export const ConsciousnessButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => (
    <Button
      ref={ref}
      variant="primary"
      animation="breathing"
      className="shadow-consciousness"
      {...props}
    />
  )
);

export const BreakthroughButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => (
    <Button
      ref={ref}
      variant="accent"
      animation="glow"
      className="shadow-glow"
      {...props}
    />
  )
);

export const FlowStateButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => (
    <Button
      ref={ref}
      variant="success"
      animation="breathing"
      {...props}
    />
  )
);

export { Button, buttonVariants };