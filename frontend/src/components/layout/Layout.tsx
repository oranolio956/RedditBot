/**
 * Main Layout Component
 * Apple-inspired layout with sidebar navigation
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bars3Icon, 
  XMarkIcon,
  BellIcon,
  Cog6ToothIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';

import { useUIStore, useAuthStore } from '@/store';
import { useWebSocket, useNotifications } from '@/lib/websocket';
import { cn } from '@/lib/utils';

import Sidebar from './Sidebar';
import NotificationPanel from './NotificationPanel';
import UserMenu from './UserMenu';
import ConnectionStatus from './ConnectionStatus';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { sidebarOpen, setSidebarOpen, notifications } = useUIStore();
  const { user } = useAuthStore();
  const { connected } = useWebSocket();
  
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (e: KeyboardEvent) => {
      // Cmd/Ctrl + \ to toggle sidebar
      if ((e.metaKey || e.ctrlKey) && e.key === '\\') {
        e.preventDefault();
        setSidebarOpen(!sidebarOpen);
      }
      
      // Escape to close overlays
      if (e.key === 'Escape') {
        setShowNotifications(false);
        setShowUserMenu(false);
      }
    };

    window.addEventListener('keydown', handleKeyboard);
    return () => window.removeEventListener('keydown', handleKeyboard);
  }, [sidebarOpen, setSidebarOpen]);

  // Close overlays when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Element;
      
      if (!target.closest('[data-notification-panel]') && !target.closest('[data-notification-trigger]')) {
        setShowNotifications(false);
      }
      
      if (!target.closest('[data-user-menu]') && !target.closest('[data-user-trigger]')) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const unreadNotifications = notifications.filter(n => !n.read).length;

  return (
    <div className="min-h-screen bg-surface-primary flex">
      {/* Sidebar */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <motion.div
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed inset-y-0 left-0 z-50 w-70 lg:relative lg:translate-x-0"
          >
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top navigation bar */}
        <header className="bg-surface-primary/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-30">
          <div className="flex items-center justify-between px-4 sm:px-6 lg:px-8 h-16">
            {/* Left side - Menu toggle and breadcrumbs */}
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className={cn(
                  'p-2 rounded-lg transition-colors',
                  'hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-consciousness-primary',
                  'lg:hidden'
                )}
                aria-label="Toggle sidebar"
              >
                {sidebarOpen ? (
                  <XMarkIcon className="w-6 h-6" />
                ) : (
                  <Bars3Icon className="w-6 h-6" />
                )}
              </button>

              {/* Desktop menu toggle */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className={cn(
                  'hidden lg:flex p-2 rounded-lg transition-colors',
                  'hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-consciousness-primary'
                )}
                aria-label="Toggle sidebar"
              >
                <Bars3Icon className="w-5 h-5" />
              </button>

              {/* Connection status */}
              <ConnectionStatus connected={connected} />
            </div>

            {/* Right side - Notifications and user menu */}
            <div className="flex items-center space-x-3">
              {/* Notifications */}
              <div className="relative">
                <button
                  data-notification-trigger
                  onClick={() => setShowNotifications(!showNotifications)}
                  className={cn(
                    'relative p-2 rounded-lg transition-colors',
                    'hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-consciousness-primary',
                    showNotifications && 'bg-gray-100'
                  )}
                  aria-label="View notifications"
                >
                  <BellIcon className="w-6 h-6" />
                  {unreadNotifications > 0 && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-states-stress text-white text-xs rounded-full flex items-center justify-center">
                      {unreadNotifications > 9 ? '9+' : unreadNotifications}
                    </span>
                  )}
                </button>

                {/* Notification panel */}
                <AnimatePresence>
                  {showNotifications && (
                    <motion.div
                      data-notification-panel
                      initial={{ opacity: 0, scale: 0.95, y: -10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: -10 }}
                      transition={{ duration: 0.15 }}
                      className="absolute right-0 mt-2 w-80 origin-top-right"
                    >
                      <NotificationPanel 
                        notifications={notifications}
                        onClose={() => setShowNotifications(false)}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Settings */}
              <button
                onClick={() => window.location.href = '/settings'}
                className={cn(
                  'p-2 rounded-lg transition-colors',
                  'hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-consciousness-primary'
                )}
                aria-label="Settings"
              >
                <Cog6ToothIcon className="w-6 h-6" />
              </button>

              {/* User menu */}
              <div className="relative">
                <button
                  data-user-trigger
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className={cn(
                    'flex items-center space-x-2 p-2 rounded-lg transition-colors',
                    'hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-consciousness-primary',
                    showUserMenu && 'bg-gray-100'
                  )}
                  aria-label="User menu"
                >
                  {user?.avatar_url ? (
                    <img
                      src={user.avatar_url}
                      alt={user.full_name || user.username || 'User'}
                      className="w-8 h-8 rounded-full"
                    />
                  ) : (
                    <UserCircleIcon className="w-8 h-8" />
                  )}
                  <span className="hidden sm:block text-sm font-medium">
                    {user?.first_name || user?.username || 'User'}
                  </span>
                </button>

                {/* User dropdown menu */}
                <AnimatePresence>
                  {showUserMenu && (
                    <motion.div
                      data-user-menu
                      initial={{ opacity: 0, scale: 0.95, y: -10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: -10 }}
                      transition={{ duration: 0.15 }}
                      className="absolute right-0 mt-2 w-48 origin-top-right"
                    >
                      <UserMenu 
                        user={user}
                        onClose={() => setShowUserMenu(false)}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </header>

        {/* Main content area */}
        <main className="flex-1 overflow-auto">
          <div className="h-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}