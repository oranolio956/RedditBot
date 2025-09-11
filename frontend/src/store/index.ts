/**
 * Zustand Store for AI Consciousness Platform
 * Centralized state management for all features
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  User,
  CognitiveProfile,
  MemoryPalace,
  EmotionalProfile,
  SynestheticProfile,
  QuantumNetwork,
  TelegramBotStatus,
  ThemeMode,
  NotificationProps,
} from '@/types';
import {
  KellyAccount,
  KellyConversation,
  AIFeatureConfig,
  KellyDashboardOverview,
  SafetyStatus,
  ConversationAnalytics,
  PaymentSettings,
} from '@/types/kelly';

// Auth Store
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
  updateUser: (updates: Partial<User>) => void;
  setLoading: (loading: boolean) => void;
}

const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        
        login: (token, user) => {
          set({
            token,
            user,
            isAuthenticated: true,
            isLoading: false,
          });
          localStorage.setItem('auth_token', token);
        },
        
        logout: () => {
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
          localStorage.removeItem('auth_token');
        },
        
        updateUser: (updates) => set(state => {
          if (state.user) {
            return {
              user: { ...state.user, ...updates },
            };
          }
          return state;
        }),
        
        setLoading: (loading) => set({ isLoading: loading }),
      }),
      {
        name: 'auth-storage',
        partialize: (state) => ({ 
          token: state.token, 
          user: state.user, 
          isAuthenticated: state.isAuthenticated 
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

// Consciousness Store
interface ConsciousnessState {
  profile: CognitiveProfile | null;
  isLoading: boolean;
  lastUpdate: string | null;
  predictions: any[];
  evolutionData: any | null;
  twinConversations: any[];
  calibrationData: any | null;
  
  setProfile: (profile: CognitiveProfile) => void;
  updateProfile: (updates: Partial<CognitiveProfile>) => void;
  setLoading: (loading: boolean) => void;
  addPrediction: (prediction: any) => void;
  setEvolutionData: (data: any) => void;
  addTwinConversation: (conversation: any) => void;
  setCalibrationData: (data: any) => void;
  reset: () => void;
}

const useConsciousnessStore = create<ConsciousnessState>()(
  devtools(
    (set) => ({
      profile: null,
      isLoading: false,
      lastUpdate: null,
      predictions: [],
      evolutionData: null,
      twinConversations: [],
      calibrationData: null,
      
      setProfile: (profile) => set({ 
        profile, 
        lastUpdate: new Date().toISOString() 
      }),
      
      updateProfile: (updates) => set(state => {
        if (state.profile) {
          return { 
            profile: { ...state.profile, ...updates },
            lastUpdate: new Date().toISOString()
          };
        }
        return state;
      }),
      
      setLoading: (loading) => set({ isLoading: loading }),
      
      addPrediction: (prediction) => set(state => ({ 
        predictions: [prediction, ...state.predictions].slice(0, 50) // Keep last 50
      })),
      
      setEvolutionData: (data) => set({ evolutionData: data }),
      
      addTwinConversation: (conversation) => set(state => ({ 
        twinConversations: [conversation, ...state.twinConversations].slice(0, 100)
      })),
      
      setCalibrationData: (data) => set({ calibrationData: data }),
      
      reset: () => set({
        profile: null,
        isLoading: false,
        lastUpdate: null,
        predictions: [],
        evolutionData: null,
        twinConversations: [],
        calibrationData: null,
      }),
    }),
    { name: 'ConsciousnessStore' }
  )
);

// Memory Palace Store
interface MemoryPalaceState {
  palaces: MemoryPalace[];
  activePalace: MemoryPalace | null;
  rooms: any[];
  memories: any[];
  isLoading: boolean;
  searchResults: any[];
  
  setPalaces: (palaces: MemoryPalace[]) => void;
  addPalace: (palace: MemoryPalace) => void;
  updatePalace: (id: string, updates: Partial<MemoryPalace>) => void;
  setActivePalace: (palace: MemoryPalace | null) => void;
  setRooms: (rooms: any[]) => void;
  addRoom: (room: any) => void;
  setMemories: (memories: any[]) => void;
  addMemory: (memory: any) => void;
  setSearchResults: (results: any[]) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const useMemoryPalaceStore = create<MemoryPalaceState>()(
  devtools(
    (set) => ({
      palaces: [],
      activePalace: null,
      rooms: [],
      memories: [],
      isLoading: false,
      searchResults: [],
      
      setPalaces: (palaces) => set({ palaces }),
      
      addPalace: (palace) => set(state => ({ 
        palaces: [palace, ...state.palaces]
      })),
      
      updatePalace: (id, updates) => set(state => ({
        palaces: state.palaces.map(palace => 
          palace.id === id ? { ...palace, ...updates } : palace
        ),
        activePalace: state.activePalace?.id === id 
          ? { ...state.activePalace, ...updates } 
          : state.activePalace
      })),
      
      setActivePalace: (palace) => set({ activePalace: palace }),
      setRooms: (rooms) => set({ rooms }),
      addRoom: (room) => set(state => ({ rooms: [room, ...state.rooms] })),
      setMemories: (memories) => set({ memories }),
      addMemory: (memory) => set(state => ({ memories: [memory, ...state.memories] })),
      setSearchResults: (results) => set({ searchResults: results }),
      setLoading: (loading) => set({ isLoading: loading }),
      
      reset: () => set({
        palaces: [],
        activePalace: null,
        rooms: [],
        memories: [],
        isLoading: false,
        searchResults: [],
      }),
    }),
    { name: 'MemoryPalaceStore' }
  )
);

// Emotional Intelligence Store
interface EmotionalState {
  profile: EmotionalProfile | null;
  currentMood: any | null;
  moodHistory: any[];
  insights: any[];
  compatibilityAnalyses: any[];
  isLoading: boolean;
  
  setProfile: (profile: EmotionalProfile) => void;
  setCurrentMood: (mood: any) => void;
  addMoodEntry: (mood: any) => void;
  setMoodHistory: (history: any[]) => void;
  setInsights: (insights: any[]) => void;
  addInsight: (insight: any) => void;
  setCompatibilityAnalyses: (analyses: any[]) => void;
  addCompatibilityAnalysis: (analysis: any) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const useEmotionalStore = create<EmotionalState>()(
  devtools(
    (set) => ({
      profile: null,
      currentMood: null,
      moodHistory: [],
      insights: [],
      compatibilityAnalyses: [],
      isLoading: false,
      
      setProfile: (profile) => set({ profile }),
      setCurrentMood: (mood) => set({ currentMood: mood }),
      
      addMoodEntry: (mood) => set(state => ({
        moodHistory: [mood, ...state.moodHistory],
        currentMood: mood
      })),
      
      setMoodHistory: (history) => set({ moodHistory: history }),
      setInsights: (insights) => set({ insights }),
      addInsight: (insight) => set(state => ({ 
        insights: [insight, ...state.insights].slice(0, 100)
      })),
      
      setCompatibilityAnalyses: (analyses) => set({ compatibilityAnalyses: analyses }),
      addCompatibilityAnalysis: (analysis) => set(state => ({ 
        compatibilityAnalyses: [analysis, ...state.compatibilityAnalyses]
      })),
      
      setLoading: (loading) => set({ isLoading: loading }),
      
      reset: () => set({
        profile: null,
        currentMood: null,
        moodHistory: [],
        insights: [],
        compatibilityAnalyses: [],
        isLoading: false,
      }),
    }),
    { name: 'EmotionalStore' }
  )
);

// Quantum Consciousness Store
interface QuantumState {
  network: QuantumNetwork | null;
  entanglements: any[];
  thoughtTransmissions: any[];
  coherenceLevel: number;
  quantumEvents: any[];
  isLoading: boolean;
  
  setNetwork: (network: QuantumNetwork) => void;
  setEntanglements: (entanglements: any[]) => void;
  addEntanglement: (entanglement: any) => void;
  setThoughtTransmissions: (transmissions: any[]) => void;
  addThoughtTransmission: (transmission: any) => void;
  setCoherenceLevel: (level: number) => void;
  addQuantumEvent: (event: any) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const useQuantumStore = create<QuantumState>()(
  devtools(
    (set) => ({
      network: null,
      entanglements: [],
      thoughtTransmissions: [],
      coherenceLevel: 0,
      quantumEvents: [],
      isLoading: false,
      
      setNetwork: (network) => set({ network }),
      setEntanglements: (entanglements) => set({ entanglements }),
      addEntanglement: (entanglement) => set(state => ({ 
        entanglements: [entanglement, ...state.entanglements]
      })),
      
      setThoughtTransmissions: (transmissions) => set({ thoughtTransmissions: transmissions }),
      addThoughtTransmission: (transmission) => set(state => ({ 
        thoughtTransmissions: [transmission, ...state.thoughtTransmissions].slice(0, 50)
      })),
      
      setCoherenceLevel: (level) => set({ coherenceLevel: level }),
      
      addQuantumEvent: (event) => set(state => ({ 
        quantumEvents: [event, ...state.quantumEvents].slice(0, 100)
      })),
      
      setLoading: (loading) => set({ isLoading: loading }),
      
      reset: () => set({
        network: null,
        entanglements: [],
        thoughtTransmissions: [],
        coherenceLevel: 0,
        quantumEvents: [],
        isLoading: false,
      }),
    }),
    { name: 'QuantumStore' }
  )
);

// Telegram Store
interface TelegramState {
  status: TelegramBotStatus | null;
  sessions: any[];
  metrics: any | null;
  historicalMetrics: any[];
  isLoading: boolean;
  
  setStatus: (status: TelegramBotStatus) => void;
  setSessions: (sessions: any[]) => void;
  setMetrics: (metrics: any) => void;
  addMetricsPoint: (metrics: any) => void;
  setHistoricalMetrics: (metrics: any[]) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const useTelegramStore = create<TelegramState>()(
  devtools(
    (set) => ({
      status: null,
      sessions: [],
      metrics: null,
      historicalMetrics: [],
      isLoading: false,
      
      setStatus: (status) => set({ status }),
      setSessions: (sessions) => set({ sessions }),
      setMetrics: (metrics) => set({ metrics }),
      
      addMetricsPoint: (metrics) => set(state => ({
        historicalMetrics: [
          ...state.historicalMetrics,
          { ...metrics, timestamp: new Date().toISOString() }
        ].slice(-1000) // Keep last 1000 points
      })),
      
      setHistoricalMetrics: (metrics) => set({ historicalMetrics: metrics }),
      setLoading: (loading) => set({ isLoading: loading }),
      
      reset: () => set({
        status: null,
        sessions: [],
        metrics: null,
        historicalMetrics: [],
        isLoading: false,
      }),
    }),
    { name: 'TelegramStore' }
  )
);

// UI Store
interface UIState {
  theme: ThemeMode;
  sidebarOpen: boolean;
  notifications: NotificationProps[];
  activeFeature: string | null;
  loadingStates: Record<string, boolean>;
  
  setTheme: (theme: ThemeMode) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  addNotification: (notification: Omit<NotificationProps, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  setActiveFeature: (feature: string | null) => void;
  setLoading: (key: string, loading: boolean) => void;
  getLoading: (key: string) => boolean;
}

const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set, get) => ({
        theme: 'system',
        sidebarOpen: true,
        notifications: [],
        activeFeature: null,
        loadingStates: {},
        
        setTheme: (theme) => set({ theme }),
        
        toggleSidebar: () => set(state => ({ 
          sidebarOpen: !state.sidebarOpen 
        })),
        
        setSidebarOpen: (open) => set({ sidebarOpen: open }),
        
        addNotification: (notification) => set(state => ({
          notifications: [
            { ...notification, id: Date.now().toString() },
            ...state.notifications
          ].slice(0, 10) // Keep max 10 notifications
        })),
        
        removeNotification: (id) => set(state => ({
          notifications: state.notifications.filter(n => n.id !== id)
        })),
        
        clearNotifications: () => set({ notifications: [] }),
        
        setActiveFeature: (feature) => set({ activeFeature: feature }),
        
        setLoading: (key, loading) => set(state => ({
          loadingStates: { ...state.loadingStates, [key]: loading }
        })),
        
        getLoading: (key) => get().loadingStates[key] || false,
      }),
      {
        name: 'ui-storage',
        partialize: (state) => ({ theme: state.theme, sidebarOpen: state.sidebarOpen }),
      }
    ),
    { name: 'UIStore' }
  )
);

// Synesthesia Store
interface SynesthesiaState {
  profile: SynestheticProfile | null;
  experiences: any[];
  mappings: any[];
  gallery: any[];
  isLoading: boolean;
  
  setProfile: (profile: SynestheticProfile) => void;
  setExperiences: (experiences: any[]) => void;
  addExperience: (experience: any) => void;
  setMappings: (mappings: any[]) => void;
  addMapping: (mapping: any) => void;
  setGallery: (gallery: any[]) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const useSynesthesiaStore = create<SynesthesiaState>()(
  devtools(
    (set) => ({
      profile: null,
      experiences: [],
      mappings: [],
      gallery: [],
      isLoading: false,
      
      setProfile: (profile) => set({ profile }),
      setExperiences: (experiences) => set({ experiences }),
      addExperience: (experience) => set(state => ({ 
        experiences: [experience, ...state.experiences]
      })),
      
      setMappings: (mappings) => set({ mappings }),
      addMapping: (mapping) => set(state => ({ 
        mappings: [mapping, ...state.mappings]
      })),
      
      setGallery: (gallery) => set({ gallery }),
      setLoading: (loading) => set({ isLoading: loading }),
      
      reset: () => set({
        profile: null,
        experiences: [],
        mappings: [],
        gallery: [],
        isLoading: false,
      }),
    }),
    { name: 'SynesthesiaStore' }
  )
);

// Dreams Store
interface DreamsState {
  sessions: any[];
  activeDream: any | null;
  library: any[];
  patterns: any[];
  interpretations: any[];
  gallery: any[];
  isLoading: boolean;
  
  setSessions: (sessions: any[]) => void;
  addSession: (session: any) => void;
  setActiveDream: (dream: any | null) => void;
  setLibrary: (library: any[]) => void;
  setPatterns: (patterns: any[]) => void;
  setInterpretations: (interpretations: any[]) => void;
  addInterpretation: (interpretation: any) => void;
  setGallery: (gallery: any[]) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const useDreamsStore = create<DreamsState>()(
  devtools(
    (set) => ({
      sessions: [],
      activeDream: null,
      library: [],
      patterns: [],
      interpretations: [],
      gallery: [],
      isLoading: false,
      
      setSessions: (sessions) => set({ sessions }),
      addSession: (session) => set(state => ({ 
        sessions: [session, ...state.sessions]
      })),
      
      setActiveDream: (dream) => set({ activeDream: dream }),
      setLibrary: (library) => set({ library }),
      setPatterns: (patterns) => set({ patterns }),
      setInterpretations: (interpretations) => set({ interpretations }),
      addInterpretation: (interpretation) => set(state => ({ 
        interpretations: [interpretation, ...state.interpretations]
      })),
      
      setGallery: (gallery) => set({ gallery }),
      setLoading: (loading) => set({ isLoading: loading }),
      
      reset: () => set({
        sessions: [],
        activeDream: null,
        library: [],
        patterns: [],
        interpretations: [],
        gallery: [],
        isLoading: false,
      }),
    }),
    { name: 'DreamsStore' }
  )
);

// Kelly Store
interface KellyState {
  // Account management
  accounts: KellyAccount[];
  selectedAccount: KellyAccount | null;
  
  // Conversations
  activeConversations: KellyConversation[];
  selectedConversation: KellyConversation | null;
  conversationAnalytics: Record<string, ConversationAnalytics>;
  
  // AI Features
  aiFeatures: AIFeatureConfig | null;
  
  // Dashboard data
  overview: KellyDashboardOverview | null;
  
  // Safety and monitoring
  safetyStatus: SafetyStatus | null;
  
  // Payment settings
  paymentSettings: PaymentSettings | null;
  
  // UI state
  isLoading: boolean;
  error: string | null;
  activeTab: string;
  
  // Actions
  setAccounts: (accounts: KellyAccount[]) => void;
  addAccount: (account: KellyAccount) => void;
  updateAccount: (id: string, updates: Partial<KellyAccount>) => void;
  removeAccount: (id: string) => void;
  setSelectedAccount: (account: KellyAccount | null) => void;
  
  setActiveConversations: (conversations: KellyConversation[]) => void;
  addConversation: (conversation: KellyConversation) => void;
  updateConversation: (id: string, updates: Partial<KellyConversation>) => void;
  setSelectedConversation: (conversation: KellyConversation | null) => void;
  
  setConversationAnalytics: (conversationId: string, analytics: ConversationAnalytics) => void;
  
  setAIFeatures: (features: AIFeatureConfig) => void;
  updateAIFeature: (feature: keyof AIFeatureConfig, config: any) => void;
  
  setOverview: (overview: KellyDashboardOverview) => void;
  setSafetyStatus: (status: SafetyStatus) => void;
  setPaymentSettings: (settings: PaymentSettings) => void;
  
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveTab: (tab: string) => void;
  
  reset: () => void;
}

const useKellyStore = create<KellyState>()(
  devtools(
    (set) => ({
      // Initial state
      accounts: [],
      selectedAccount: null,
      activeConversations: [],
      selectedConversation: null,
      conversationAnalytics: {},
      aiFeatures: null,
      overview: null,
      safetyStatus: null,
      paymentSettings: null,
      isLoading: false,
      error: null,
      activeTab: 'dashboard',
      
      // Account actions
      setAccounts: (accounts) => set({ accounts }),
      
      addAccount: (account) => set(state => ({
        accounts: [account, ...state.accounts]
      })),
      
      updateAccount: (id, updates) => set(state => ({
        accounts: state.accounts.map(account => 
          account.id === id ? { ...account, ...updates } : account
        ),
        selectedAccount: state.selectedAccount?.id === id 
          ? { ...state.selectedAccount, ...updates }
          : state.selectedAccount
      })),
      
      removeAccount: (id) => set(state => ({
        accounts: state.accounts.filter(account => account.id !== id),
        selectedAccount: state.selectedAccount?.id === id ? null : state.selectedAccount
      })),
      
      setSelectedAccount: (account) => set({ selectedAccount: account }),
      
      // Conversation actions
      setActiveConversations: (conversations) => set({ activeConversations: conversations }),
      
      addConversation: (conversation) => set(state => ({
        activeConversations: [conversation, ...state.activeConversations]
      })),
      
      updateConversation: (id, updates) => set(state => ({
        activeConversations: state.activeConversations.map(conversation => 
          conversation.id === id ? { ...conversation, ...updates } : conversation
        ),
        selectedConversation: state.selectedConversation?.id === id
          ? { ...state.selectedConversation, ...updates }
          : state.selectedConversation
      })),
      
      setSelectedConversation: (conversation) => set({ selectedConversation: conversation }),
      
      setConversationAnalytics: (conversationId, analytics) => set(state => ({
        conversationAnalytics: {
          ...state.conversationAnalytics,
          [conversationId]: analytics
        }
      })),
      
      // AI Features actions
      setAIFeatures: (features) => set({ aiFeatures: features }),
      
      updateAIFeature: (feature, config) => set(state => ({
        aiFeatures: state.aiFeatures ? {
          ...state.aiFeatures,
          [feature]: { ...state.aiFeatures[feature], ...config }
        } : null
      })),
      
      // Dashboard actions
      setOverview: (overview) => set({ overview }),
      setSafetyStatus: (status) => set({ safetyStatus: status }),
      setPaymentSettings: (settings) => set({ paymentSettings: settings }),
      
      // UI actions
      setLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),
      setActiveTab: (tab) => set({ activeTab: tab }),
      
      reset: () => set({
        accounts: [],
        selectedAccount: null,
        activeConversations: [],
        selectedConversation: null,
        conversationAnalytics: {},
        aiFeatures: null,
        overview: null,
        safetyStatus: null,
        paymentSettings: null,
        isLoading: false,
        error: null,
        activeTab: 'dashboard',
      }),
    }),
    { name: 'KellyStore' }
  )
);

// Export all stores
export {
  useAuthStore,
  useConsciousnessStore,
  useMemoryPalaceStore,
  useEmotionalStore,
  useQuantumStore,
  useTelegramStore,
  useUIStore,
  useSynesthesiaStore,
  useDreamsStore,
  useKellyStore,
};