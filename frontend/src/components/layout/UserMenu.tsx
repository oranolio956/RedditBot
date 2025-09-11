/**
 * User Menu Component
 * Dropdown menu for user actions and account management
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  UserIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
  ChartBarIcon,
  BellIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

import { useAuthStore } from '@/store';
import { Card } from '@/components/ui/Card';
import { User } from '@/types';

interface UserMenuProps {
  user: User | null;
  onClose: () => void;
}

export default function UserMenu({ user, onClose }: UserMenuProps) {
  const { logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    onClose();
  };

  const menuItems = [
    {
      icon: UserIcon,
      label: 'Profile',
      href: '/profile',
      description: 'View and edit your profile',
    },
    {
      icon: ChartBarIcon,
      label: 'Analytics',
      href: '/analytics',
      description: 'Consciousness insights and metrics',
    },
    {
      icon: Cog6ToothIcon,
      label: 'Settings',
      href: '/settings',
      description: 'Account and app preferences',
    },
    {
      icon: BellIcon,
      label: 'Notifications',
      href: '/notifications',
      description: 'Manage notification preferences',
    },
    {
      icon: ShieldCheckIcon,
      label: 'Privacy & Security',
      href: '/privacy',
      description: 'Privacy settings and security',
    },
  ];

  return (
    <Card className="w-64 shadow-dramatic overflow-hidden">
      {/* User info header */}
      <div className="p-4 bg-gradient-consciousness text-white">
        <div className="flex items-center space-x-3">
          {user?.avatar_url ? (
            <img
              src={user.avatar_url}
              alt={user.full_name || user.username || 'User'}
              className="w-12 h-12 rounded-full border-2 border-white/20"
            />
          ) : (
            <div className="w-12 h-12 rounded-full bg-white/20 flex items-center justify-center">
              <UserIcon className="w-6 h-6" />
            </div>
          )}
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold truncate">
              {user?.full_name || user?.username || 'User'}
            </h3>
            <p className="text-sm opacity-80 truncate">
              {user?.email || 'No email'}
            </p>
            <div className="flex items-center space-x-2 mt-1">
              <span className={`px-2 py-0.5 text-xs rounded-full ${
                user?.is_premium 
                  ? 'bg-consciousness-accent text-white' 
                  : 'bg-white/20 text-white'
              }`}>
                {user?.subscription_type || 'Free'}
              </span>
              {user?.is_premium && (
                <span className="text-xs opacity-75">✨</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Menu items */}
      <div className="py-2">
        {menuItems.map((item, index) => (
          <motion.div
            key={item.href}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <Link
              to={item.href}
              onClick={onClose}
              className="flex items-center px-4 py-3 text-sm hover:bg-gray-50 transition-colors group"
            >
              <item.icon className="w-5 h-5 text-text-tertiary group-hover:text-consciousness-primary mr-3" />
              <div className="flex-1">
                <div className="font-medium text-text-primary group-hover:text-consciousness-primary">
                  {item.label}
                </div>
                <div className="text-xs text-text-tertiary">
                  {item.description}
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Divider */}
      <div className="border-t border-gray-200" />

      {/* Logout */}
      <div className="p-2">
        <button
          onClick={handleLogout}
          className="flex items-center w-full px-4 py-3 text-sm text-states-stress hover:bg-red-50 rounded-lg transition-colors group"
        >
          <ArrowRightOnRectangleIcon className="w-5 h-5 mr-3" />
          <span>Sign Out</span>
        </button>
      </div>

      {/* Footer info */}
      <div className="px-4 py-3 bg-surface-secondary border-t border-gray-200">
        <div className="flex items-center justify-between text-xs text-text-tertiary">
          <span>AI Consciousness v1.0</span>
          <span>© 2024</span>
        </div>
      </div>
    </Card>
  );
}