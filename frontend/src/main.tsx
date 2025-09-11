/**
 * Main entry point for AI Consciousness Platform
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { BrowserRouter } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

import App from './App';
import './index.css';

// Configure React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        return failureCount < 3;
      },
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
      onError: (error: any) => {
        console.error('Mutation error:', error);
      },
    },
  },
});

// Error boundary for the entire app
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-surface-primary">
          <div className="text-center p-8">
            <h1 className="text-insight-title text-text-primary mb-4">
              Something went wrong
            </h1>
            <p className="text-body-text text-text-secondary mb-6">
              The AI consciousness platform encountered an unexpected error.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="bg-consciousness-primary text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition-colors"
            >
              Reload Application
            </button>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-sm text-text-tertiary">
                  Error Details
                </summary>
                <pre className="mt-2 text-xs bg-surface-secondary p-4 rounded overflow-auto">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Toast configuration with Apple-inspired styling
const toasterOptions = {
  duration: 4000,
  position: 'top-right' as const,
  style: {
    borderRadius: '12px',
    background: 'var(--surface-primary)',
    color: 'var(--text-primary)',
    border: '1px solid rgba(0, 122, 255, 0.2)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
    fontFamily: '-apple-system, BlinkMacSystemFont, SF Pro Display, SF Pro Text, system-ui, sans-serif',
  },
  success: {
    iconTheme: {
      primary: '#30D158',
      secondary: '#FFFFFF',
    },
  },
  error: {
    iconTheme: {
      primary: '#FF453A',
      secondary: '#FFFFFF',
    },
  },
};

// Root component with all providers
function Root() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
          <Toaster 
            toastOptions={toasterOptions}
            containerStyle={{
              top: 20,
              right: 20,
            }}
          />
          {process.env.NODE_ENV === 'development' && (
            <ReactQueryDevtools 
              initialIsOpen={false} 
              position="bottom-right"
            />
          )}
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

// Initialize React app
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(<Root />);

// Performance monitoring in development
if (process.env.NODE_ENV === 'development') {
  import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
    getCLS(console.log);
    getFID(console.log);
    getFCP(console.log);
    getLCP(console.log);
    getTTFB(console.log);
  });
}

// Service worker registration for PWA
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}