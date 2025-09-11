/**
 * Sidebar Navigation Component
 * Apple-inspired navigation with consciousness features
 */

import { NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  HomeIcon,
  CpuChipIcon,
  BuildingOfficeIcon,
  HeartIcon,
  ChatBubbleLeftRightIcon,
  LinkIcon,
  SwatchIcon,
  CloudIcon,
  EyeIcon,
  SparklesIcon,
  Cog6ToothIcon,
  UserIcon,
} from '@heroicons/react/24/outline';

import { cn } from '@/lib/utils';
import { useConsciousnessStore, useTelegramStore, useUIStore } from '@/store';

interface SidebarProps {
  onClose?: () => void;
}

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
  description: string;
  badge?: string | number;
  subItems?: NavigationItem[];
}

export default function Sidebar({ onClose }: SidebarProps) {
  const location = useLocation();
  const { profile } = useConsciousnessStore();
  const { status } = useTelegramStore();
  const { setActiveFeature } = useUIStore();

  // Navigation structure
  const navigation: NavigationItem[] = [
    {
      name: 'Dashboard',
      href: '/',
      icon: HomeIcon,
      description: 'Overview of all consciousness features',
    },
    {
      name: 'Consciousness Mirror',
      href: '/consciousness',
      icon: CpuChipIcon,
      description: 'Digital twin and personality analysis',
      badge: profile?.confidence_level ? `${Math.round(profile.confidence_level * 100)}%` : undefined,
      subItems: [
        {
          name: 'Twin Chat',
          href: '/consciousness/twin-chat',
          icon: ChatBubbleLeftRightIcon,
          description: 'Conversation with your digital twin',
        },
        {
          name: 'Personality Evolution',
          href: '/consciousness/evolution',
          icon: SparklesIcon,
          description: 'Track personality changes over time',
        },
        {
          name: 'Future Self',
          href: '/consciousness/future-self',
          icon: EyeIcon,
          description: 'Simulate conversations with future you',
        },
      ],
    },
    {
      name: 'Memory Palace',
      href: '/memory',
      icon: BuildingOfficeIcon,
      description: 'Spatial memory organization system',
      subItems: [
        {
          name: 'Create Memory',
          href: '/memory/create',
          icon: SparklesIcon,
          description: 'Store new memories in your palace',
        },
      ],
    },
    {
      name: 'Emotional Intelligence',
      href: '/emotional',
      icon: HeartIcon,
      description: 'Emotional state tracking and insights',
      subItems: [
        {
          name: 'Mood Tracker',
          href: '/emotional/mood-tracker',
          icon: HeartIcon,
          description: 'Track daily emotional states',
        },
        {
          name: 'Empathy Trainer',
          href: '/emotional/empathy-trainer',
          icon: UserIcon,
          description: 'Improve emotional understanding',
        },
      ],
    },
    {
      name: 'Telegram Bot',
      href: '/telegram',
      icon: ChatBubbleLeftRightIcon,
      description: 'Bot management and monitoring',
      badge: status?.is_running ? 'Online' : 'Offline',
      subItems: [
        {
          name: 'Bot Monitoring',
          href: '/telegram/monitoring',
          icon: EyeIcon,
          description: 'Real-time bot performance metrics',
        },
        {
          name: 'Session Manager',
          href: '/telegram/sessions',
          icon: UserIcon,
          description: 'Manage active user sessions',
        },
      ],
    },
    {
      name: 'Quantum Consciousness',
      href: '/quantum',
      icon: LinkIcon,
      description: 'Quantum entanglement and telepathy',
      subItems: [
        {
          name: 'Network Visualization',
          href: '/quantum/network',
          icon: LinkIcon,
          description: 'Visualize quantum connections',
        },
        {
          name: 'Thought Teleporter',
          href: '/quantum/teleporter',
          icon: SparklesIcon,
          description: 'Send thoughts across quantum network',
        },
      ],
    },
    {
      name: 'Digital Synesthesia',
      href: '/synesthesia',
      icon: SwatchIcon,
      description: 'Cross-sensory experience engine',
      subItems: [
        {
          name: 'Modality Converter',
          href: '/synesthesia/converter',
          icon: SwatchIcon,
          description: 'Convert between sensory modalities',
        },
        {
          name: 'Experience Gallery',
          href: '/synesthesia/gallery',
          icon: EyeIcon,
          description: 'Browse shared synesthetic experiences',
        },
      ],
    },
    {
      name: 'Neural Dreams',
      href: '/dreams',
      icon: CloudIcon,
      description: 'Dream analysis and interpretation',
      subItems: [
        {
          name: 'Dream Interface',
          href: '/dreams/interface',
          icon: CloudIcon,
          description: 'Interactive dream exploration',
        },
        {
          name: 'Dream Library',
          href: '/dreams/library',
          icon: BuildingOfficeIcon,
          description: 'Personal dream collection',
        },
      ],
    },
    {
      name: 'Temporal Archaeology',
      href: '/archaeology',
      icon: EyeIcon,
      description: 'Conversation pattern analysis',
      subItems: [
        {
          name: 'Conversation Excavator',
          href: '/archaeology/excavator',
          icon: EyeIcon,
          description: 'Dig through conversation history',
        },
        {
          name: 'Timeline Visualization',
          href: '/archaeology/timeline',
          icon: SparklesIcon,
          description: 'Temporal communication patterns',
        },
      ],
    },
    {
      name: 'Meta Reality',
      href: '/meta-reality',
      icon: SparklesIcon,
      description: 'Reality layer manipulation',
      subItems: [
        {
          name: 'Layer Creator',
          href: '/meta-reality/layer-creator',
          icon: SparklesIcon,
          description: 'Create custom reality filters',
        },
      ],
    },
  ];

  const bottomNavigation: NavigationItem[] = [
    {
      name: 'Settings',
      href: '/settings',
      icon: Cog6ToothIcon,
      description: 'Application settings and preferences',
    },
    {
      name: 'Profile',
      href: '/profile',
      icon: UserIcon,
      description: 'User profile and account management',
    },
  ];

  const isActiveRoute = (href: string) => {
    if (href === '/') return location.pathname === '/';
    return location.pathname.startsWith(href);
  };

  const isActiveSubItem = (href: string) => {
    return location.pathname === href;
  };

  const handleItemClick = (item: NavigationItem) => {
    setActiveFeature(item.name);
    if (onClose) onClose();
  };

  const renderNavigationItem = (item: NavigationItem, isSubItem = false) => (
    <div key={item.href}>
      <NavLink
        to={item.href}
        onClick={() => handleItemClick(item)}
        className={({ isActive }) =>
          cn(
            'group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200',
            isSubItem ? 'ml-6 pl-8' : '',
            isActiveRoute(item.href) || isActive
              ? 'bg-consciousness-primary/10 text-consciousness-primary'
              : 'text-text-secondary hover:text-consciousness-primary hover:bg-consciousness-primary/5'
          )
        }
      >
        <item.icon
          className={cn(
            'mr-3 h-5 w-5 transition-colors',
            isActiveRoute(item.href)
              ? 'text-consciousness-primary'
              : 'text-text-tertiary group-hover:text-consciousness-primary'
          )}
          aria-hidden="true"
        />
        <span className="flex-1">{item.name}</span>
        
        {item.badge && (
          <span
            className={cn(
              'ml-2 px-2 py-1 text-xs rounded-full',
              item.badge === 'Online'
                ? 'bg-states-flow/20 text-states-flow'
                : item.badge === 'Offline'
                ? 'bg-states-stress/20 text-states-stress'
                : 'bg-consciousness-primary/20 text-consciousness-primary'
            )}
          >
            {item.badge}
          </span>
        )}
      </NavLink>

      {/* Sub-items */}
      {item.subItems && isActiveRoute(item.href) && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
          className="mt-1 space-y-1"
        >
          {item.subItems.map((subItem) => (
            <NavLink
              key={subItem.href}
              to={subItem.href}
              onClick={() => handleItemClick(subItem)}
              className={({ isActive }) =>
                cn(
                  'group flex items-center ml-6 pl-8 pr-3 py-2 text-sm rounded-lg transition-all duration-200',
                  isActive || isActiveSubItem(subItem.href)
                    ? 'bg-consciousness-secondary/10 text-consciousness-secondary'
                    : 'text-text-tertiary hover:text-consciousness-secondary hover:bg-consciousness-secondary/5'
                )
              }
            >
              <subItem.icon
                className={cn(
                  'mr-3 h-4 w-4 transition-colors',
                  isActiveSubItem(subItem.href)
                    ? 'text-consciousness-secondary'
                    : 'text-text-tertiary group-hover:text-consciousness-secondary'
                )}
                aria-hidden="true"
              />
              <span>{subItem.name}</span>
            </NavLink>
          ))}
        </motion.div>
      )}
    </div>
  );

  return (
    <div className="flex flex-col h-full bg-surface-secondary border-r border-gray-200">
      {/* Logo and brand */}
      <div className="flex items-center h-16 px-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-consciousness-gradient flex items-center justify-center">
            <CpuChipIcon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-text-primary">
              AI Consciousness
            </h1>
            <p className="text-xs text-text-tertiary">
              Digital Twin Platform
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 overflow-y-auto scrollbar-apple">
        <div className="space-y-1">
          {navigation.map((item) => renderNavigationItem(item))}
        </div>

        {/* Divider */}
        <div className="my-6 border-t border-gray-200" />

        {/* Bottom navigation */}
        <div className="space-y-1">
          {bottomNavigation.map((item) => renderNavigationItem(item))}
        </div>
      </nav>

      {/* Consciousness status indicator */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex items-center space-x-3 p-3 rounded-lg bg-consciousness-primary/5">
          <div className="w-3 h-3 rounded-full bg-states-flow animate-breathing" />
          <div className="flex-1">
            <p className="text-sm font-medium text-text-primary">
              Consciousness Active
            </p>
            <p className="text-xs text-text-tertiary">
              {profile?.confidence_level 
                ? `${Math.round(profile.confidence_level * 100)}% calibrated`
                : 'Calibrating...'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}