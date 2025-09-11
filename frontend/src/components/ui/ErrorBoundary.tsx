/**
 * Error Boundary Component
 * Catches and handles component errors gracefully with user-friendly fallbacks
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';
import { Button } from './Button';
import { Card } from './Card';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  showDetails?: boolean;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      errorId: Date.now().toString()
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Log error to external service or console
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);

    // Log to external error tracking service
    this.logErrorToService(error, errorInfo);
  }

  private logErrorToService(error: Error, errorInfo: ErrorInfo) {
    // This would integrate with services like Sentry, LogRocket, etc.
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      errorId: this.state.errorId
    };

    // Send to error tracking service
    fetch('/api/v1/errors/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorReport)
    }).catch(err => {
      console.error('Failed to report error to service:', err);
    });
  }

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    });
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  private copyErrorDetails = () => {
    const errorText = `
Error ID: ${this.state.errorId}
Message: ${this.state.error?.message}
Stack: ${this.state.error?.stack}
Component Stack: ${this.state.errorInfo?.componentStack}
URL: ${window.location.href}
Timestamp: ${new Date().toISOString()}
User Agent: ${navigator.userAgent}
    `.trim();

    navigator.clipboard.writeText(errorText).then(() => {
      alert('Error details copied to clipboard');
    }).catch(() => {
      console.error('Failed to copy error details');
    });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-gray-50">
          <Card className="max-w-2xl w-full">
            <div className="p-8 text-center">
              {/* Error Icon */}
              <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-6">
                <AlertTriangle className="w-8 h-8 text-red-600" />
              </div>

              {/* Error Message */}
              <h1 className="text-2xl font-bold text-gray-900 mb-2">
                Something went wrong
              </h1>
              <p className="text-gray-600 mb-8">
                We're sorry, but something unexpected happened. Our team has been notified and is working on a fix.
              </p>

              {/* Action Buttons */}
              <div className="space-y-3 sm:space-y-0 sm:space-x-3 sm:flex sm:justify-center">
                <Button
                  onClick={this.handleRetry}
                  className="w-full sm:w-auto"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Try Again
                </Button>
                
                <Button
                  onClick={this.handleReload}
                  variant="outline"
                  className="w-full sm:w-auto"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Reload Page
                </Button>
                
                <Button
                  onClick={this.handleGoHome}
                  variant="outline"
                  className="w-full sm:w-auto"
                >
                  <Home className="w-4 h-4 mr-2" />
                  Go Home
                </Button>
              </div>

              {/* Error Details (Development/Debug) */}
              {(this.props.showDetails || process.env.NODE_ENV === 'development') && (
                <details className="mt-8 text-left">
                  <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                    <Bug className="inline w-4 h-4 mr-2" />
                    Error Details (Click to expand)
                  </summary>
                  
                  <div className="mt-4 p-4 bg-gray-100 rounded-md text-xs font-mono overflow-auto max-h-64">
                    <div className="mb-4">
                      <strong>Error ID:</strong> {this.state.errorId}
                    </div>
                    
                    <div className="mb-4">
                      <strong>Message:</strong>
                      <pre className="mt-1 whitespace-pre-wrap">{this.state.error?.message}</pre>
                    </div>
                    
                    <div className="mb-4">
                      <strong>Stack Trace:</strong>
                      <pre className="mt-1 whitespace-pre-wrap">{this.state.error?.stack}</pre>
                    </div>
                    
                    <div className="mb-4">
                      <strong>Component Stack:</strong>
                      <pre className="mt-1 whitespace-pre-wrap">{this.state.errorInfo?.componentStack}</pre>
                    </div>

                    <Button
                      onClick={this.copyErrorDetails}
                      variant="outline"
                      size="sm"
                      className="mt-2"
                    >
                      Copy Error Details
                    </Button>
                  </div>
                </details>
              )}

              {/* Error ID for Support */}
              <div className="mt-6 p-3 bg-gray-100 rounded-md">
                <p className="text-sm text-gray-600">
                  <strong>Error ID:</strong> {this.state.errorId}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Please include this ID when contacting support
                </p>
              </div>
            </div>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

// Higher-order component for easier usage
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

// Hook for imperative error handling
export function useErrorHandler() {
  return (error: Error, errorInfo?: any) => {
    // Throw error to be caught by nearest ErrorBoundary
    throw error;
  };
}

// Specialized error boundaries for different parts of the app
export const KellyErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary
    showDetails={process.env.NODE_ENV === 'development'}
    onError={(error, errorInfo) => {
      // Kelly-specific error handling
      console.error('Kelly AI Error:', error, errorInfo);
      
      // Send to Kelly monitoring service
      fetch('/api/v1/kelly/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          error: error.message,
          stack: error.stack,
          component_stack: errorInfo.componentStack,
          timestamp: new Date().toISOString()
        })
      }).catch(console.error);
    }}
    fallback={
      <div className="p-8 text-center">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Kelly AI Interface Error
        </h2>
        <p className="text-gray-600 mb-4">
          There was an issue with the Kelly AI interface. Please try refreshing the page.
        </p>
        <Button onClick={() => window.location.reload()}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh Interface
        </Button>
      </div>
    }
  >
    {children}
  </ErrorBoundary>
);

export const ConversationErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary
    fallback={
      <div className="p-6 text-center border-2 border-dashed border-gray-300 rounded-lg">
        <AlertTriangle className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
        <h3 className="text-lg font-medium text-gray-900 mb-1">
          Conversation Load Error
        </h3>
        <p className="text-sm text-gray-600 mb-3">
          Unable to load this conversation. Please try selecting it again.
        </p>
        <Button size="sm" onClick={() => window.location.reload()}>
          Retry
        </Button>
      </div>
    }
  >
    {children}
  </ErrorBoundary>
);