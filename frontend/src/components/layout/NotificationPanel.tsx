/**
 * Notification Panel Component
 * Displays real-time notifications and alerts
 */


import { motion } from 'framer-motion';
import { BellIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { formatRelativeTime } from '@/lib/utils';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read?: boolean;
  action?: {
    label: string;
    handler: () => void;
  };
}

interface NotificationPanelProps {
  notifications: Notification[];
  onClose: () => void;
}

export default function NotificationPanel({ notifications, onClose }: NotificationPanelProps) {
  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return '✅';
      case 'warning':
        return '⚠️';
      case 'error':
        return '❌';
      default:
        return 'ℹ️';
    }
  };

  const getNotificationColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'border-l-states-flow';
      case 'warning':
        return 'border-l-consciousness-accent';
      case 'error':
        return 'border-l-states-stress';
      default:
        return 'border-l-consciousness-primary';
    }
  };

  return (
    <Card className="w-80 max-h-96 overflow-hidden shadow-dramatic">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <BellIcon className="w-5 h-5 text-consciousness-primary" />
          <h3 className="font-semibold text-text-primary">Notifications</h3>
          {notifications.filter(n => !n.read).length > 0 && (
            <span className="px-2 py-1 text-xs bg-consciousness-primary text-white rounded-full">
              {notifications.filter(n => !n.read).length}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <XMarkIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Notifications list */}
      <div className="max-h-80 overflow-y-auto scrollbar-apple">
        {notifications.length === 0 ? (
          <div className="p-8 text-center text-text-tertiary">
            <BellIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p className="text-sm">No notifications</p>
            <p className="text-xs mt-1">You're all caught up!</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {notifications.map((notification, index) => (
              <motion.div
                key={notification.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`p-4 border-l-4 ${getNotificationColor(notification.type)} ${
                  !notification.read ? 'bg-consciousness-primary/5' : ''
                }`}
              >
                <div className="flex items-start space-x-3">
                  <span className="text-lg">{getNotificationIcon(notification.type)}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-medium text-text-primary truncate">
                        {notification.title}
                      </h4>
                      {!notification.read && (
                        <div className="w-2 h-2 bg-consciousness-primary rounded-full ml-2" />
                      )}
                    </div>
                    <p className="text-sm text-text-secondary mt-1">
                      {notification.message}
                    </p>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-text-tertiary">
                        {formatRelativeTime(notification.timestamp)}
                      </span>
                      {notification.action && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={notification.action.handler}
                          className="text-xs"
                        >
                          {notification.action.label}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {notifications.length > 0 && (
        <div className="p-3 border-t border-gray-200 bg-surface-secondary">
          <div className="flex justify-between items-center">
            <Button variant="ghost" size="sm" className="text-xs">
              Mark all as read
            </Button>
            <Button variant="ghost" size="sm" className="text-xs">
              View all
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}