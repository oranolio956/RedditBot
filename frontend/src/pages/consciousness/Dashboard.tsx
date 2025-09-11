/**
 * Consciousness Mirroring Dashboard
 * Central hub for digital twin interaction and personality analysis
 */

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  CpuChipIcon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  EyeIcon,
  SparklesIcon,
  CogIcon,
  ArrowRightIcon,
  ClockIcon,
  LightBulbIcon,
  PuzzlePieceIcon,
} from '@heroicons/react/24/outline';

import { useAuthStore, useConsciousnessStore } from '@/store';
import { 
  useConsciousnessProfile,
  usePersonalityEvolution,
  useConsciousnessSessions,
  useTwinChat,
  usePredictDecision,
} from '@/hooks/useApi';
import { useConsciousnessUpdates } from '@/lib/websocket';

import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardContent,
  ConsciousnessCard,
  InsightCard,
} from '@/components/ui/Card';
import { Button, ConsciousnessButton } from '@/components/ui/Button';
import LoadingSpinner, { ConsciousnessLoader } from '@/components/ui/LoadingSpinner';
import PersonalityRadar from '@/components/consciousness/PersonalityRadar';
import TwinChatPreview from '@/components/consciousness/TwinChatPreview';
import ConsciousnessMetrics from '@/components/consciousness/ConsciousnessMetrics';
import CalibrationPanel from '@/components/consciousness/CalibrationPanel';

export default function ConsciousnessDashboard() {
  const { user } = useAuthStore();
  const { profile, twinConversations, predictions } = useConsciousnessStore();
  
  const [quickChatMessage, setQuickChatMessage] = useState('');
  const [showCalibration, setShowCalibration] = useState(false);

  // API queries
  const { data: consciousnessProfile, isLoading: profileLoading, refetch: refetchProfile } = useConsciousnessProfile(user?.id || '');
  const { data: evolutionData, isLoading: evolutionLoading } = usePersonalityEvolution(user?.id || '');
  const { data: sessions, isLoading: sessionsLoading } = useConsciousnessSessions(user?.id || '');
  
  // Mutations
  const twinChatMutation = useTwinChat();
  const predictDecisionMutation = usePredictDecision();

  // Real-time consciousness updates
  useConsciousnessUpdates(user?.id || '', (update) => {
    console.log('Real-time consciousness update:', update);
    // Trigger profile refetch on significant updates
    if (update.payload.confidence > 0.1) {
      refetchProfile();
    }
  });

  const handleQuickChat = async () => {
    if (!quickChatMessage.trim()) return;

    try {
      const response = await twinChatMutation.mutateAsync(quickChatMessage);
      setQuickChatMessage('');
      // Response will be handled by the store update
    } catch (error) {
      console.error('Twin chat error:', error);
    }
  };

  const generateInsight = () => {
    // Mock insight generation
    const insights = [
      "Your creativity peaks consistently around 2:30 PM based on 15 days of analysis",
      "Strong correlation detected between mood and keystroke rhythm patterns",
      "Your decision-making speed increases by 23% during flow states",
      "Personality shows increased openness when discussing technical topics",
      "Optimal focus windows identified: 9-11 AM and 2-4 PM"
    ];
    
    return insights[Math.floor(Math.random() * insights.length)];
  };

  if (profileLoading || evolutionLoading || sessionsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <ConsciousnessLoader text="Initializing consciousness mirror..." />
      </div>
    );
  }

  const cognitiveProfile = consciousnessProfile || profile;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center space-x-3"
        >
          <div className="w-12 h-12 rounded-full bg-consciousness-gradient flex items-center justify-center animate-breathing">
            <CpuChipIcon className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-insight-title font-bold text-gradient">
            Consciousness Mirror
          </h1>
        </motion.div>
        <p className="text-body-text text-text-secondary max-w-2xl mx-auto">
          Interact with your digital twin, explore personality patterns, and gain deep insights 
          into your cognitive processes through AI-powered consciousness mirroring.
        </p>
      </div>

      {/* Consciousness State Overview */}
      {cognitiveProfile && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <ConsciousnessCard
            state={{
              focus: cognitiveProfile.focus_state || 0,
              flow: cognitiveProfile.flow_state || 0,
              clarity: cognitiveProfile.clarity_score || 0,
              energy: cognitiveProfile.energy_level || 0,
            }}
            status={`${Math.round((cognitiveProfile.confidence_level || 0) * 100)}% calibrated`}
            className="shadow-consciousness"
          />
        </motion.div>
      )}

      {/* Quick Actions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Twin Chat Quick Access */}
        <Card variant="consciousness" interactive className="group">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <ChatBubbleLeftRightIcon className="w-6 h-6 text-consciousness-primary" />
              <CardTitle className="text-lg">Chat with Twin</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-text-secondary mb-4">
              Have a conversation with your digital consciousness
            </p>
            <Link to="/consciousness/twin-chat">
              <ConsciousnessButton size="sm" className="w-full">
                Start Conversation
              </ConsciousnessButton>
            </Link>
          </CardContent>
        </Card>

        {/* Personality Evolution */}
        <Card variant="breakthrough" interactive className="group">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <ChartBarIcon className="w-6 h-6 text-consciousness-accent" />
              <CardTitle className="text-lg">Evolution Tracking</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-text-secondary mb-4">
              Visualize personality changes over time
            </p>
            <Link to="/consciousness/evolution">
              <Button variant="accent" size="sm" className="w-full">
                View Evolution
              </Button>
            </Link>
          </CardContent>
        </Card>

        {/* Future Self Simulation */}
        <Card variant="quantum" interactive className="group">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <EyeIcon className="w-6 h-6 text-consciousness-secondary" />
              <CardTitle className="text-lg">Future Self</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-text-secondary mb-4">
              Converse with your predicted future self
            </p>
            <Link to="/consciousness/future-self">
              <Button variant="secondary" size="sm" className="w-full">
                Meet Future You
              </Button>
            </Link>
          </CardContent>
        </Card>

        {/* Calibration Tools */}
        <Card variant="default" interactive className="group">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <CogIcon className="w-6 h-6 text-text-secondary" />
              <CardTitle className="text-lg">Calibration</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-text-secondary mb-4">
              Fine-tune consciousness accuracy
            </p>
            <Button 
              variant="outline" 
              size="sm" 
              className="w-full"
              onClick={() => setShowCalibration(true)}
            >
              Calibrate Mirror
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Twin Chat Preview */}
        <div className="lg:col-span-2 space-y-6">
          {/* Quick Chat Interface */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <ChatBubbleLeftRightIcon className="w-5 h-5" />
                <span>Quick Twin Chat</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={quickChatMessage}
                    onChange={(e) => setQuickChatMessage(e.target.value)}
                    placeholder="Ask your digital twin anything..."
                    className="flex-1 px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-consciousness-primary focus:border-transparent"
                    onKeyPress={(e) => e.key === 'Enter' && handleQuickChat()}
                    disabled={twinChatMutation.isPending}
                  />
                  <ConsciousnessButton
                    onClick={handleQuickChat}
                    disabled={!quickChatMessage.trim() || twinChatMutation.isPending}
                    loading={twinChatMutation.isPending}
                  >
                    Send
                  </ConsciousnessButton>
                </div>
                
                {/* Recent conversations preview */}
                <TwinChatPreview conversations={twinConversations.slice(0, 3)} />
                
                <div className="text-center">
                  <Link to="/consciousness/twin-chat">
                    <Button variant="ghost" size="sm">
                      View Full Conversation History
                      <ArrowRightIcon className="w-4 h-4 ml-1" />
                    </Button>
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Recent Insights */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <LightBulbIcon className="w-5 h-5" />
                <span>Latest Insights</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <InsightCard
                  title="Breakthrough Pattern Detected"
                  confidence={0.94}
                  icon={<SparklesIcon className="w-5 h-5" />}
                  onExplore={() => console.log('Explore insight')}
                >
                  <p className="text-sm text-text-secondary">
                    {generateInsight()}
                  </p>
                </InsightCard>
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-text-tertiary">
                    {predictions.length} predictions generated today
                  </span>
                  <Button variant="ghost" size="sm">
                    View All Insights
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Metrics and Controls */}
        <div className="space-y-6">
          {/* Personality Radar */}
          {cognitiveProfile && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <PuzzlePieceIcon className="w-5 h-5" />
                  <span>Personality Profile</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <PersonalityRadar
                  traits={cognitiveProfile.big_five}
                  size={200}
                />
                <div className="mt-4 space-y-2">
                  {Object.entries(cognitiveProfile.big_five).map(([trait, value]) => (
                    <div key={trait} className="flex justify-between text-sm">
                      <span className="capitalize text-text-secondary">{trait}:</span>
                      <span className="font-medium">{Math.round(value * 100)}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Live Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <ClockIcon className="w-5 h-5" />
                <span>Live Metrics</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ConsciousnessMetrics
                cognitiveLoad={cognitiveProfile?.cognitive_load || 0}
                focusState={cognitiveProfile?.focus_state || 0}
                energyLevel={cognitiveProfile?.energy_level || 0}
                lastUpdate={cognitiveProfile?.last_updated}
              />
            </CardContent>
          </Card>

          {/* Session Stats */}
          <Card>
            <CardHeader>
              <CardTitle>Session Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-text-secondary">Total Sessions:</span>
                  <span className="font-medium">{sessions?.length || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-text-secondary">Accuracy:</span>
                  <span className="font-medium">
                    {cognitiveProfile?.calibration_accuracy 
                      ? `${Math.round(cognitiveProfile.calibration_accuracy * 100)}%`
                      : 'Calibrating...'
                    }
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-text-secondary">Last Calibration:</span>
                  <span className="font-medium text-xs">
                    {cognitiveProfile?.last_updated 
                      ? new Date(cognitiveProfile.last_updated).toLocaleDateString()
                      : 'Never'
                    }
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Calibration Modal */}
      {showCalibration && (
        <CalibrationPanel
          isOpen={showCalibration}
          onClose={() => setShowCalibration(false)}
          currentProfile={cognitiveProfile}
        />
      )}
    </div>
  );
}