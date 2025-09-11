/**
 * Quick Actions Component
 * Common actions and shortcuts for the dashboard
 */

// import React from 'react';
import { motion } from 'framer-motion';
import {
  ChatBubbleLeftRightIcon,
  PlusIcon,
  SparklesIcon,
  CogIcon,
  ChartBarIcon,
  CloudArrowUpIcon,
} from '@heroicons/react/24/outline';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

const quickActions = [
  {
    title: 'Chat with Twin',
    description: 'Start a conversation with your digital consciousness',
    icon: ChatBubbleLeftRightIcon,
    action: 'chat',
    color: 'consciousness-primary',
    href: '/consciousness/twin-chat',
  },
  {
    title: 'Create Memory',
    description: 'Store a new memory in your palace',
    icon: PlusIcon,
    action: 'memory',
    color: 'purple-600',
    href: '/memory/create',
  },
  {
    title: 'Generate Insight',
    description: 'Get AI-powered consciousness insights',
    icon: SparklesIcon,
    action: 'insight',
    color: 'consciousness-accent',
    href: '/insights/generate',
  },
  {
    title: 'Calibrate Mirror',
    description: 'Fine-tune consciousness accuracy',
    icon: CogIcon,
    action: 'calibrate',
    color: 'consciousness-secondary',
    href: '/consciousness?calibrate=true',
  },
  {
    title: 'View Analytics',
    description: 'Explore your consciousness metrics',
    icon: ChartBarIcon,
    action: 'analytics',
    color: 'states-flow',
    href: '/analytics',
  },
  {
    title: 'Export Data',
    description: 'Download your consciousness data',
    icon: CloudArrowUpIcon,
    action: 'export',
    color: 'states-neutral',
    href: '/settings/export',
  },
];

export default function QuickActions() {
  const handleAction = (action: string) => {
    switch (action) {
      case 'insight':
        // Mock insight generation
        console.log('Generating insight...');
        break;
      case 'calibrate':
        console.log('Starting calibration...');
        break;
      case 'export':
        console.log('Exporting data...');
        break;
      default:
        // Navigate to href
        break;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <SparklesIcon className="w-5 h-5" />
          <span>Quick Actions</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {quickActions.map((action, index) => (
            <motion.div
              key={action.action}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <button
                onClick={() => handleAction(action.action)}
                className="w-full p-3 text-left rounded-lg border border-gray-200 hover:border-consciousness-primary/30 hover:bg-consciousness-primary/5 transition-all group"
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-10 h-10 rounded-lg bg-${action.color}/10 flex items-center justify-center group-hover:scale-110 transition-transform`}>
                    <action.icon className={`w-5 h-5 text-${action.color}`} />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-text-primary group-hover:text-consciousness-primary transition-colors">
                      {action.title}
                    </h3>
                    <p className="text-sm text-text-secondary">
                      {action.description}
                    </p>
                  </div>
                </div>
              </button>
            </motion.div>
          ))}
        </div>

        <div className="mt-6 pt-4 border-t border-gray-200">
          <Button variant="outline" className="w-full" size="sm">
            View All Features
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}