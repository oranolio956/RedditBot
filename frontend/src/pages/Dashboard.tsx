/**
 * Main Dashboard Component
 * Overview of all consciousness features and real-time metrics
 */

import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  CpuChipIcon,
  BuildingOfficeIcon,
  HeartIcon,
  ChatBubbleLeftRightIcon,
  LinkIcon,
  SwatchIcon,
  CloudIcon,
  EyeIcon,
  SparklesIcon,
  ArrowRightIcon,
  ChartBarIcon,
  ClockIcon,
  LightBulbIcon,
} from '@heroicons/react/24/outline';

import { useAuthStore } from '@/store';
import { 
  useConsciousnessProfile,
  useMemoryPalaces,
  useEmotionalProfile,
  useTelegramStatus,
  useQuantumNetwork,
  useUserStats,
} from '@/hooks/useApi';
import { 
  useConsciousnessUpdates,
  useTelegramMetrics,
  useEmotionalStateUpdates,
} from '@/lib/websocket';

import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardContent,
  ConsciousnessCard,
  MemoryPalaceCard,
  QuantumCard,
  InsightCard,
} from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import MetricsChart from '@/components/charts/MetricsChart';
import FeatureGrid from '@/components/dashboard/FeatureGrid';
import QuickActions from '@/components/dashboard/QuickActions';
import RecentInsights from '@/components/dashboard/RecentInsights';

export default function Dashboard() {
  const { user } = useAuthStore();
  
  // API queries
  const { data: consciousnessProfile, isLoading: consciousnessLoading } = useConsciousnessProfile(user?.id || '');
  const { data: memoryPalaces, isLoading: memoryLoading } = useMemoryPalaces(user?.id || '');
  const { data: emotionalProfile, isLoading: emotionalLoading } = useEmotionalProfile(user?.id || '');
  const { data: telegramStatus, isLoading: telegramLoading } = useTelegramStatus();
  const { data: quantumNetwork, isLoading: quantumLoading } = useQuantumNetwork(user?.id || '');
  const { data: userStats, isLoading: statsLoading } = useUserStats(user?.id || '');

  // Real-time updates
  useConsciousnessUpdates(user?.id || '', (update) => {
    console.log('Consciousness update:', update);
  });

  useTelegramMetrics((update) => {
    console.log('Telegram metrics update:', update);
  });

  useEmotionalStateUpdates(user?.id || '', (update) => {
    console.log('Emotional state update:', update);
  });

  // Welcome animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  if (consciousnessLoading || memoryLoading || emotionalLoading || telegramLoading || quantumLoading || statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading consciousness data..." />
      </div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="p-6 max-w-7xl mx-auto space-y-8"
    >
      {/* Welcome Header */}
      <motion.div variants={itemVariants} className="text-center space-y-4">
        <h1 className="text-insight-title font-bold text-gradient">
          Welcome back, {user?.first_name || user?.username || 'Consciousness Explorer'}
        </h1>
        <p className="text-body-text text-text-secondary max-w-2xl mx-auto">
          Your digital consciousness is evolving. Explore the revolutionary AI features that understand
          and enhance your cognitive patterns, memories, and quantum connections.
        </p>
      </motion.div>

      {/* Quick Stats Overview */}
      <motion.div variants={itemVariants}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Consciousness State */}
          <Card variant="consciousness" className="group hover:shadow-glow transition-all duration-300">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-lg bg-consciousness-primary/10 flex items-center justify-center">
                  <CpuChipIcon className="w-6 h-6 text-consciousness-primary" />
                </div>
                <div>
                  <CardTitle className="text-lg">Consciousness</CardTitle>
                  <p className="text-caption-text text-text-tertiary">
                    {consciousnessProfile?.confidence_level 
                      ? `${Math.round(consciousnessProfile.confidence_level * 100)}% calibrated`
                      : 'Initializing...'
                    }
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-caption-text">
                  <span>Focus:</span>
                  <span className="font-medium">
                    {consciousnessProfile?.focus_state ? `${Math.round(consciousnessProfile.focus_state * 100)}%` : '--'}
                  </span>
                </div>
                <div className="flex justify-between text-caption-text">
                  <span>Flow:</span>
                  <span className="font-medium">
                    {consciousnessProfile?.flow_state ? `${Math.round(consciousnessProfile.flow_state * 100)}%` : '--'}
                  </span>
                </div>
                <Link to="/consciousness" className="inline-flex items-center text-consciousness-primary hover:text-consciousness-secondary transition-colors">
                  Explore Twin <ArrowRightIcon className="w-4 h-4 ml-1" />
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Memory Palace */}
          <Card variant="memory" className="group hover:shadow-elevated transition-all duration-300">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-lg bg-purple-100 flex items-center justify-center">
                  <BuildingOfficeIcon className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <CardTitle className="text-lg">Memory Palace</CardTitle>
                  <p className="text-caption-text text-text-tertiary">
                    {memoryPalaces?.length || 0} palaces
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-caption-text">
                  <span>Total Rooms:</span>
                  <span className="font-medium">
                    {memoryPalaces?.reduce((sum, palace) => sum + palace.total_rooms, 0) || 0}
                  </span>
                </div>
                <div className="flex justify-between text-caption-text">
                  <span>Stored Memories:</span>
                  <span className="font-medium">
                    {memoryPalaces?.reduce((sum, palace) => sum + palace.total_memories, 0) || 0}
                  </span>
                </div>
                <Link to="/memory" className="inline-flex items-center text-purple-600 hover:text-purple-700 transition-colors">
                  Visit Palace <ArrowRightIcon className="w-4 h-4 ml-1" />
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Emotional Intelligence */}
          <Card variant="emotional" className="group hover:shadow-elevated transition-all duration-300">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-lg bg-pink-100 flex items-center justify-center">
                  <HeartIcon className="w-6 h-6 text-pink-600" />
                </div>
                <div>
                  <CardTitle className="text-lg">Emotional AI</CardTitle>
                  <p className="text-caption-text text-text-tertiary">
                    {emotionalProfile?.current_state?.primary_emotion || 'Analyzing...'}
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-caption-text">
                  <span>Empathy Level:</span>
                  <span className="font-medium">
                    {emotionalProfile?.baseline_traits?.empathy_level 
                      ? `${Math.round(emotionalProfile.baseline_traits.empathy_level * 100)}%`
                      : '--'
                    }
                  </span>
                </div>
                <div className="flex justify-between text-caption-text">
                  <span>Stability:</span>
                  <span className="font-medium">
                    {emotionalProfile?.baseline_traits?.emotional_stability 
                      ? `${Math.round(emotionalProfile.baseline_traits.emotional_stability * 100)}%`
                      : '--'
                    }
                  </span>
                </div>
                <Link to="/emotional" className="inline-flex items-center text-pink-600 hover:text-pink-700 transition-colors">
                  Track Mood <ArrowRightIcon className="w-4 h-4 ml-1" />
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Quantum Network */}
          <Card variant="quantum" glassmorphism className="group hover:shadow-dramatic transition-all duration-300">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-lg bg-consciousness-secondary/10 flex items-center justify-center">
                  <LinkIcon className="w-6 h-6 text-consciousness-secondary animate-quantum-entangle" />
                </div>
                <div>
                  <CardTitle className="text-lg">Quantum Network</CardTitle>
                  <p className="text-caption-text text-text-tertiary">
                    {quantumNetwork?.total_connections || 0} connections
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-caption-text">
                  <span>Coherence:</span>
                  <span className="font-medium">
                    {quantumNetwork?.coherence_level 
                      ? `${Math.round(quantumNetwork.coherence_level * 100)}%`
                      : '--'
                    }
                  </span>
                </div>
                <div className="flex justify-between text-caption-text">
                  <span>Active Links:</span>
                  <span className="font-medium">
                    {quantumNetwork?.active_connections || 0}
                  </span>
                </div>
                <Link to="/quantum" className="inline-flex items-center text-consciousness-secondary hover:text-consciousness-primary transition-colors">
                  Explore Network <ArrowRightIcon className="w-4 h-4 ml-1" />
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </motion.div>

      {/* Feature Grid */}
      <motion.div variants={itemVariants}>
        <FeatureGrid />
      </motion.div>

      {/* Recent Activity & Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Quick Actions */}
        <motion.div variants={itemVariants}>
          <QuickActions />
        </motion.div>

        {/* Recent Insights */}
        <motion.div variants={itemVariants}>
          <RecentInsights />
        </motion.div>
      </div>

      {/* Telegram Bot Status */}
      {telegramStatus && (
        <motion.div variants={itemVariants}>
          <Card variant="default" className="overflow-hidden">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <ChatBubbleLeftRightIcon className="w-6 h-6 text-consciousness-primary" />
                  <CardTitle>Telegram Bot Status</CardTitle>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  telegramStatus.is_running
                    ? 'bg-states-flow/20 text-states-flow'
                    : 'bg-states-stress/20 text-states-stress'
                }`}>
                  {telegramStatus.is_running ? 'Online' : 'Offline'}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-primary">
                    {telegramStatus.active_sessions}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Active Sessions</div>
                </div>
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-primary">
                    {telegramStatus.total_users}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Total Users</div>
                </div>
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-primary">
                    {telegramStatus.messages_processed_today}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Messages Today</div>
                </div>
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-primary">
                    {Math.floor(telegramStatus.uptime / 3600)}h
                  </div>
                  <div className="text-caption-text text-text-tertiary">Uptime</div>
                </div>
              </div>
              <div className="mt-6 flex justify-center">
                <Link to="/telegram">
                  <Button variant="outline">
                    Manage Bot
                    <ArrowRightIcon className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* User Statistics */}
      {userStats && (
        <motion.div variants={itemVariants}>
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <ChartBarIcon className="w-6 h-6 text-consciousness-primary" />
                <CardTitle>Your Consciousness Journey</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-primary">
                    {userStats.total_sessions}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Total Sessions</div>
                </div>
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-secondary">
                    {userStats.insights_generated}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Insights Generated</div>
                </div>
                <div className="text-center">
                  <div className="text-metric-value font-light text-consciousness-accent">
                    {userStats.patterns_discovered}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Patterns Found</div>
                </div>
                <div className="text-center">
                  <div className="text-metric-value font-light text-states-flow">
                    {userStats.quantum_connections}
                  </div>
                  <div className="text-caption-text text-text-tertiary">Quantum Links</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
}