/**
 * Main App component for AI Consciousness Platform
 * Handles routing, authentication, and global state
 */

import React, { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// Store and hooks
import { useAuthStore, useUIStore } from '@/store';
import { useWebSocket } from '@/lib/websocket';
import { applyTheme } from '@/lib/utils';

// Layout components
import Layout from '@/components/layout/Layout';
import AuthLayout from '@/components/layout/AuthLayout';

// Page components
import Dashboard from '@/pages/Dashboard';
import Login from '@/pages/auth/Login';
import Register from '@/pages/auth/Register';

// Kelly Phase 1 Pages
import ConversationManagerV2Page from '@/pages/kelly/ConversationManagerV2';

// Feature pages
import ConsciousnessDashboard from '@/pages/consciousness/Dashboard';
import TwinChat from '@/pages/consciousness/TwinChat';
import PersonalityEvolution from '@/pages/consciousness/PersonalityEvolution';
import FutureSelf from '@/pages/consciousness/FutureSelf';

import MemoryPalaceDashboard from '@/pages/memory/Dashboard';
import PalaceViewer from '@/pages/memory/PalaceViewer';
import MemoryCreator from '@/pages/memory/MemoryCreator';

import EmotionalDashboard from '@/pages/emotional/Dashboard';
import MoodTracker from '@/pages/emotional/MoodTracker';
import EmpathyTrainer from '@/pages/emotional/EmpathyTrainer';

import TelegramDashboard from '@/pages/telegram/Dashboard';
import BotMonitoring from '@/pages/telegram/BotMonitoring';
import SessionManager from '@/pages/telegram/SessionManager';

import QuantumDashboard from '@/pages/quantum/Dashboard';
import NetworkVisualization from '@/pages/quantum/NetworkVisualization';
import ThoughtTeleporter from '@/pages/quantum/ThoughtTeleporter';

import SynesthesiaDashboard from '@/pages/synesthesia/Dashboard';
import ModalityConverter from '@/pages/synesthesia/ModalityConverter';
import ExperienceGallery from '@/pages/synesthesia/ExperienceGallery';

import DreamsDashboard from '@/pages/dreams/Dashboard';
import DreamInterface from '@/pages/dreams/DreamInterface';
import DreamLibrary from '@/pages/dreams/DreamLibrary';

import ArchaeologyDashboard from '@/pages/archaeology/Dashboard';
import ConversationExcavator from '@/pages/archaeology/ConversationExcavator';
import TemporalTimeline from '@/pages/archaeology/TemporalTimeline';

import MetaRealityDashboard from '@/pages/meta-reality/Dashboard';
import RealityLayerCreator from '@/pages/meta-reality/RealityLayerCreator';

import Settings from '@/pages/Settings';
import Profile from '@/pages/Profile';
import KellySettings from '@/pages/KellySettings';
import NotFound from '@/pages/NotFound';

// Protected route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthStore();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface-primary">
        <div className="consciousness-loader"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// Route animation variants
const pageVariants = {
  initial: { opacity: 0, y: 20 },
  in: { opacity: 1, y: 0 },
  out: { opacity: 0, y: -20 }
};

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.4
};

// Animated route wrapper
function AnimatedRoute({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial="initial"
        animate="in"
        exit="out"
        variants={pageVariants}
        transition={pageTransition}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

function App() {
  const { isAuthenticated, user, token } = useAuthStore();
  const { theme } = useUIStore();
  const ws = useWebSocket();

  // Apply theme on mount and changes
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // Initialize WebSocket connection when authenticated
  useEffect(() => {
    if (isAuthenticated && user && token) {
      ws.connect(user.id).catch(console.error);
      
      // Cleanup on unmount
      return () => {
        ws.disconnect();
      };
    }
  }, [isAuthenticated, user, token]);

  // Auto-logout on token expiration
  useEffect(() => {
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        const exp = payload.exp * 1000;
        const now = Date.now();
        
        if (exp < now) {
          useAuthStore.getState().logout();
        } else {
          // Set timeout to logout when token expires
          const timeout = setTimeout(() => {
            useAuthStore.getState().logout();
          }, exp - now);
          
          return () => clearTimeout(timeout);
        }
      } catch (error) {
        console.warn('Invalid token format:', error);
        useAuthStore.getState().logout();
      }
    }
  }, [token]);

  return (
    <div className="App min-h-screen bg-surface-primary text-text-primary">
      <Routes>
        {/* Public authentication routes */}
        <Route path="/login" element={
          <AuthLayout>
            <AnimatedRoute>
              <Login />
            </AnimatedRoute>
          </AuthLayout>
        } />
        
        <Route path="/register" element={
          <AuthLayout>
            <AnimatedRoute>
              <Register />
            </AnimatedRoute>
          </AuthLayout>
        } />

        {/* Protected application routes */}
        <Route path="/*" element={
          <ProtectedRoute>
            <Layout>
              <AnimatedRoute>
                <Routes>
                  {/* Main dashboard */}
                  <Route index element={<Dashboard />} />
                  
                  {/* Consciousness Mirroring */}
                  <Route path="consciousness" element={<ConsciousnessDashboard />} />
                  <Route path="consciousness/twin-chat" element={<TwinChat />} />
                  <Route path="consciousness/evolution" element={<PersonalityEvolution />} />
                  <Route path="consciousness/future-self" element={<FutureSelf />} />
                  
                  {/* Memory Palace */}
                  <Route path="memory" element={<MemoryPalaceDashboard />} />
                  <Route path="memory/palace/:id" element={<PalaceViewer />} />
                  <Route path="memory/create" element={<MemoryCreator />} />
                  
                  {/* Emotional Intelligence */}
                  <Route path="emotional" element={<EmotionalDashboard />} />
                  <Route path="emotional/mood-tracker" element={<MoodTracker />} />
                  <Route path="emotional/empathy-trainer" element={<EmpathyTrainer />} />
                  
                  {/* Telegram Bot Management */}
                  <Route path="telegram" element={<TelegramDashboard />} />
                  <Route path="telegram/monitoring" element={<BotMonitoring />} />
                  <Route path="telegram/sessions" element={<SessionManager />} />
                  
                  {/* Quantum Consciousness */}
                  <Route path="quantum" element={<QuantumDashboard />} />
                  <Route path="quantum/network" element={<NetworkVisualization />} />
                  <Route path="quantum/teleporter" element={<ThoughtTeleporter />} />
                  
                  {/* Digital Synesthesia */}
                  <Route path="synesthesia" element={<SynesthesiaDashboard />} />
                  <Route path="synesthesia/converter" element={<ModalityConverter />} />
                  <Route path="synesthesia/gallery" element={<ExperienceGallery />} />
                  
                  {/* Neural Dreams */}
                  <Route path="dreams" element={<DreamsDashboard />} />
                  <Route path="dreams/interface" element={<DreamInterface />} />
                  <Route path="dreams/library" element={<DreamLibrary />} />
                  
                  {/* Temporal Archaeology */}
                  <Route path="archaeology" element={<ArchaeologyDashboard />} />
                  <Route path="archaeology/excavator" element={<ConversationExcavator />} />
                  <Route path="archaeology/timeline" element={<TemporalTimeline />} />
                  
                  {/* Meta Reality */}
                  <Route path="meta-reality" element={<MetaRealityDashboard />} />
                  <Route path="meta-reality/layer-creator" element={<RealityLayerCreator />} />
                  
                  {/* Kelly Brain System */}
                  <Route path="kelly" element={<KellySettings />} />
                  <Route path="kelly/conversations" element={<ConversationManagerV2Page />} />
                  
                  {/* User management */}
                  <Route path="profile" element={<Profile />} />
                  <Route path="settings" element={<Settings />} />
                  
                  {/* 404 page */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </AnimatedRoute>
            </Layout>
          </ProtectedRoute>
        } />
      </Routes>
    </div>
  );
}

export default App;