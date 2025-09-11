/**
 * API Client for AI Consciousness Platform
 * Comprehensive client for all 220+ backend endpoints
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import {
  User,
  CognitiveProfile,
  ConsciousnessUpdate,
  TwinResponse,
  PersonalityEvolution,
  MemoryPalace,
  MemoryRoom,
  StoredMemory,
  ConversationFragment,
  TemporalPattern,
  GhostConversation,
  EmotionalProfile,
  MoodEntry,
  CompatibilityAnalysis,
  SynestheticProfile,
  SynestheticExperience,
  DreamSession,
  QuantumNetwork,
  EntanglementPair,
  ThoughtTeleportation,
  RealityLayer,
  ConsensusReality,
  TelegramBotStatus,
  TelegramSession,
  ApiResponse,
  PaginatedResponse,
  ErrorResponse,
} from '@/types';
import {
  KellyAccount,
  KellyConversation,
  KellyDashboardOverview,
  AIFeatureConfig,
  ClaudeIntegrationConfig,
  ClaudeUsageMetrics,
  GeneratedResponse,
  SafetyStatus,
  ConversationAnalytics,
  PaymentSettings,
} from '@/types/kelly';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_VERSION = 'v1';
const API_TIMEOUT = 30000; // 30 seconds

class ApiClient {
  private client: AxiosInstance;
  private authToken: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api/${API_VERSION}`,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor for auth
    this.client.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ErrorResponse>) => {
        if (error.response?.status === 401) {
          this.clearAuth();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication methods
  setAuthToken(token: string) {
    this.authToken = token;
    localStorage.setItem('auth_token', token);
  }

  clearAuth() {
    this.authToken = null;
    localStorage.removeItem('auth_token');
  }

  restoreAuth() {
    const token = localStorage.getItem('auth_token');
    if (token) {
      this.authToken = token;
    }
  }

  // Generic API methods
  private async get<T>(endpoint: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(endpoint, config);
    return response.data.data;
  }

  private async post<T>(endpoint: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(endpoint, data, config);
    return response.data.data;
  }

  private async put<T>(endpoint: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(endpoint, data, config);
    return response.data.data;
  }

  private async delete<T>(endpoint: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(endpoint, config);
    return response.data.data;
  }

  private async getPaginated<T>(endpoint: string, params?: Record<string, any>): Promise<PaginatedResponse<T>> {
    const response = await this.client.get<PaginatedResponse<T>>(endpoint, { params });
    return response.data;
  }

  // User Management APIs
  async getUsers(page = 1, perPage = 20, search?: string): Promise<PaginatedResponse<User>> {
    return this.getPaginated('/users', { page, per_page: perPage, search });
  }

  async getUserById(userId: string): Promise<User> {
    return this.get(`/users/${userId}`);
  }

  async getUserByTelegramId(telegramId: number): Promise<User> {
    return this.get(`/users/telegram/${telegramId}`);
  }

  async createUser(userData: Partial<User>): Promise<User> {
    return this.post('/users', userData);
  }

  async updateUser(userId: string, userData: Partial<User>): Promise<User> {
    return this.put(`/users/${userId}`, userData);
  }

  async deleteUser(userId: string, hard = false): Promise<void> {
    return this.delete(`/users/${userId}`, { params: { hard } });
  }

  async restoreUser(userId: string): Promise<User> {
    return this.post(`/users/${userId}/restore`);
  }

  async getUserStats(userId: string): Promise<any> {
    return this.get(`/users/${userId}/stats`);
  }

  async updateUserPreferences(userId: string, preferences: any): Promise<User> {
    return this.put(`/users/${userId}/preferences`, preferences);
  }

  async getGlobalUserStats(): Promise<any> {
    return this.get('/users/stats/overview');
  }

  // Consciousness Mirroring APIs
  async getConsciousnessProfile(userId: string): Promise<CognitiveProfile> {
    return this.get(`/consciousness/profile/${userId}`);
  }

  async updateConsciousness(userId: string, update: ConsciousnessUpdate): Promise<CognitiveProfile> {
    return this.post(`/consciousness/update/${userId}`, update);
  }

  async predictResponse(userId: string, context: string): Promise<TwinResponse> {
    return this.post(`/consciousness/predict/${userId}`, { context });
  }

  async simulateFutureSelf(userId: string, timeHorizon: string, scenario: string): Promise<TwinResponse> {
    return this.post(`/consciousness/future-self/${userId}`, { time_horizon: timeHorizon, scenario });
  }

  async chatWithTwin(userId: string, message: string): Promise<TwinResponse> {
    return this.post(`/consciousness/twin-chat/${userId}`, { message });
  }

  async predictDecision(userId: string, decision: string, options: string[]): Promise<any> {
    return this.post(`/consciousness/predict-decision/${userId}`, { decision, options });
  }

  async calibrateMirror(userId: string, calibrationData: any): Promise<any> {
    return this.post(`/consciousness/calibrate/${userId}`, calibrationData);
  }

  async getPersonalityEvolution(userId: string): Promise<PersonalityEvolution> {
    return this.get(`/consciousness/evolution/${userId}`);
  }

  async getConsciousnessSessions(userId: string): Promise<any[]> {
    return this.get(`/consciousness/sessions/${userId}`);
  }

  // Telegram Bot Management APIs
  async getTelegramStatus(): Promise<TelegramBotStatus> {
    return this.get('/telegram/status');
  }

  async getTelegramMetrics(): Promise<any> {
    return this.get('/telegram/metrics');
  }

  async getTelegramHistoricalMetrics(period: string): Promise<any> {
    return this.get(`/telegram/metrics/historical?period=${period}`);
  }

  async getWebhookInfo(): Promise<any> {
    return this.get('/telegram/webhook/info');
  }

  async testWebhook(): Promise<any> {
    return this.post('/telegram/webhook/test');
  }

  async restartWebhook(): Promise<any> {
    return this.post('/telegram/webhook/restart');
  }

  async sendTelegramMessage(chatId: number, message: string, options?: any): Promise<any> {
    return this.post('/telegram/send-message', { chat_id: chatId, message, ...options });
  }

  async getTelegramSessions(): Promise<TelegramSession[]> {
    return this.get('/telegram/sessions');
  }

  async getUserTelegramSessions(userId: string): Promise<TelegramSession[]> {
    return this.get(`/telegram/sessions/${userId}`);
  }

  async expireTelegramSession(sessionId: string): Promise<void> {
    return this.delete(`/telegram/sessions/${sessionId}`);
  }

  async getRateLimits(userId: string): Promise<any> {
    return this.get(`/telegram/rate-limits/${userId}`);
  }

  async resetRateLimits(userId: string): Promise<any> {
    return this.post('/telegram/rate-limits/reset', { user_id: userId });
  }

  async getCircuitBreakers(): Promise<any> {
    return this.get('/telegram/circuit-breakers');
  }

  async resetCircuitBreaker(name: string): Promise<any> {
    return this.post(`/telegram/circuit-breakers/${name}/reset`);
  }

  async getAntiBanMetrics(): Promise<any> {
    return this.get('/telegram/anti-ban/metrics');
  }

  async performMaintenanceCleanup(): Promise<any> {
    return this.post('/telegram/maintenance/cleanup');
  }

  // Memory Palace APIs
  async createMemoryPalace(data: Partial<MemoryPalace>): Promise<MemoryPalace> {
    return this.post('/memory-palace/create', data);
  }

  async getMemoryPalace(palaceId: string): Promise<MemoryPalace> {
    return this.get(`/memory-palace/${palaceId}`);
  }

  async getUserMemoryPalaces(userId: string): Promise<MemoryPalace[]> {
    return this.get(`/memory-palace/user/${userId}`);
  }

  async addRoomToPalace(palaceId: string, roomData: Partial<MemoryRoom>): Promise<MemoryRoom> {
    return this.post(`/memory-palace/${palaceId}/rooms`, roomData);
  }

  async getPalaceRooms(palaceId: string): Promise<MemoryRoom[]> {
    return this.get(`/memory-palace/${palaceId}/rooms`);
  }

  async storeMemory(roomId: string, memoryData: Partial<StoredMemory>): Promise<StoredMemory> {
    return this.post(`/memory-palace/rooms/${roomId}/memories`, memoryData);
  }

  async getRoomMemories(roomId: string): Promise<StoredMemory[]> {
    return this.get(`/memory-palace/rooms/${roomId}/memories`);
  }

  async searchMemories(query: string, palaceId?: string): Promise<StoredMemory[]> {
    return this.post('/memory-palace/search', { query, palace_id: palaceId });
  }

  async getPalaceStats(palaceId: string): Promise<any> {
    return this.get(`/memory-palace/${palaceId}/stats`);
  }

  async backupPalace(palaceId: string): Promise<any> {
    return this.post(`/memory-palace/${palaceId}/backup`);
  }

  async restorePalace(palaceId: string, backupData: any): Promise<any> {
    return this.post(`/memory-palace/${palaceId}/restore`, backupData);
  }

  // Temporal Archaeology APIs
  async excavateConversations(userId: string, parameters: any): Promise<ConversationFragment[]> {
    return this.post('/archaeology/excavate', { user_id: userId, ...parameters });
  }

  async getConversationFragments(userId: string): Promise<ConversationFragment[]> {
    return this.get(`/archaeology/fragments/${userId}`);
  }

  async reconstructMessages(fragmentIds: string[]): Promise<any> {
    return this.post('/archaeology/reconstruct', { fragment_ids: fragmentIds });
  }

  async getTemporalPatterns(userId: string): Promise<TemporalPattern[]> {
    return this.get(`/archaeology/patterns/${userId}`);
  }

  async generateLinguisticFingerprint(userId: string, textSample: string): Promise<any> {
    return this.post('/archaeology/fingerprint', { user_id: userId, text_sample: textSample });
  }

  async getCommunicationTimeline(userId: string): Promise<any> {
    return this.get(`/archaeology/timeline/${userId}`);
  }

  async createGhostConversation(userId: string, parameters: any): Promise<GhostConversation> {
    return this.post('/archaeology/ghost-conversation', { user_id: userId, ...parameters });
  }

  async getArchaeologySessions(userId: string): Promise<any[]> {
    return this.get(`/archaeology/sessions/${userId}`);
  }

  // Emotional Intelligence APIs
  async getEmotionalProfile(userId: string): Promise<EmotionalProfile> {
    return this.get(`/emotional-intelligence/profile/${userId}`);
  }

  async analyzeEmotion(text: string, context?: any): Promise<any> {
    return this.post('/emotional-intelligence/analyze', { text, context });
  }

  async trackMood(userId: string, moodData: Partial<MoodEntry>): Promise<MoodEntry> {
    return this.post('/emotional-intelligence/track-mood', { user_id: userId, ...moodData });
  }

  async getMoodHistory(userId: string, period?: string): Promise<MoodEntry[]> {
    return this.get(`/emotional-intelligence/mood-history/${userId}`, { params: { period } });
  }

  async startEmpathyTraining(userId: string, trainingType: string): Promise<any> {
    return this.post('/emotional-intelligence/empathy-training', { user_id: userId, training_type: trainingType });
  }

  async getEmotionalInsights(userId: string): Promise<any> {
    return this.get(`/emotional-intelligence/insights/${userId}`);
  }

  async calibrateEmotionalSystem(userId: string, calibrationData: any): Promise<any> {
    return this.post('/emotional-intelligence/calibrate', { user_id: userId, ...calibrationData });
  }

  async analyzeCompatibility(user1Id: string, user2Id: string): Promise<CompatibilityAnalysis> {
    return this.get('/emotional-intelligence/compatibility', { 
      params: { user1_id: user1Id, user2_id: user2Id } 
    });
  }

  // Digital Synesthesia APIs
  async createSynestheticProfile(userId: string, profileData: Partial<SynestheticProfile>): Promise<SynestheticProfile> {
    return this.post('/synesthesia/create-profile', { user_id: userId, ...profileData });
  }

  async getSynestheticProfile(userId: string): Promise<SynestheticProfile> {
    return this.get(`/synesthesia/profile/${userId}`);
  }

  async convertModalities(userId: string, inputData: any, fromModality: string, toModality: string): Promise<any> {
    return this.post('/synesthesia/convert', {
      user_id: userId,
      input_data: inputData,
      from_modality: fromModality,
      to_modality: toModality,
    });
  }

  async getSensoryMappings(userId: string): Promise<any> {
    return this.get(`/synesthesia/mappings/${userId}`);
  }

  async trainAssociations(userId: string, trainingData: any): Promise<any> {
    return this.post('/synesthesia/train', { user_id: userId, ...trainingData });
  }

  async visualizeConversion(conversionId: string): Promise<any> {
    return this.get(`/synesthesia/visualize/${conversionId}`);
  }

  async shareSynestheticExperience(userId: string, experienceData: any): Promise<SynestheticExperience> {
    return this.post('/synesthesia/share', { user_id: userId, ...experienceData });
  }

  async getSynesthesiaGallery(): Promise<SynestheticExperience[]> {
    return this.get('/synesthesia/gallery');
  }

  // Neural Dreams APIs
  async initiateDreamSession(userId: string, dreamParameters: any): Promise<DreamSession> {
    return this.post('/neural-dreams/initiate', { user_id: userId, ...dreamParameters });
  }

  async getDreamSession(sessionId: string): Promise<DreamSession> {
    return this.get(`/neural-dreams/session/${sessionId}`);
  }

  async guideDream(sessionId: string, guidanceData: any): Promise<any> {
    return this.post('/neural-dreams/guide', { session_id: sessionId, ...guidanceData });
  }

  async getDreamLibrary(userId: string): Promise<DreamSession[]> {
    return this.get(`/neural-dreams/library/${userId}`);
  }

  async interpretDream(sessionId: string, interpretationMethod?: string): Promise<any> {
    return this.post('/neural-dreams/interpret', { session_id: sessionId, method: interpretationMethod });
  }

  async getDreamPatterns(userId: string): Promise<any> {
    return this.get(`/neural-dreams/patterns/${userId}`);
  }

  async shareDream(sessionId: string, shareOptions: any): Promise<any> {
    return this.post('/neural-dreams/share', { session_id: sessionId, ...shareOptions });
  }

  async getDreamGallery(): Promise<DreamSession[]> {
    return this.get('/neural-dreams/gallery');
  }

  // Quantum Consciousness APIs
  async createQuantumEntanglement(user1Id: string, user2Id: string): Promise<EntanglementPair> {
    return this.post('/quantum/entangle', { user1_id: user1Id, user2_id: user2Id });
  }

  async getQuantumNetwork(userId: string): Promise<QuantumNetwork> {
    return this.get(`/quantum/network/${userId}`);
  }

  async teleportThought(senderId: string, receiverId: string, thoughtData: any): Promise<ThoughtTeleportation> {
    return this.post('/quantum/teleport-thought', {
      sender_id: senderId,
      receiver_id: receiverId,
      thought_data: thoughtData,
    });
  }

  async measureCoherence(userId: string): Promise<any> {
    return this.get(`/quantum/coherence/${userId}`);
  }

  async createSuperposition(userId: string, stateData: any): Promise<any> {
    return this.post('/quantum/superposition', { user_id: userId, ...stateData });
  }

  async getQuantumObservations(userId: string): Promise<any> {
    return this.get(`/quantum/observations/${userId}`);
  }

  async forceDecoherence(userId: string): Promise<any> {
    return this.post('/quantum/decohere', { user_id: userId });
  }

  // Meta Reality APIs
  async createRealityLayer(userId: string, layerData: Partial<RealityLayer>): Promise<RealityLayer> {
    return this.post('/meta-reality/create-layer', { user_id: userId, ...layerData });
  }

  async getRealityLayers(userId: string): Promise<RealityLayer[]> {
    return this.get(`/meta-reality/layers/${userId}`);
  }

  async blendRealities(layerIds: string[], blendParameters: any): Promise<any> {
    return this.post('/meta-reality/blend', { layer_ids: layerIds, ...blendParameters });
  }

  async analyzePerception(userId: string, perceptionData: any): Promise<any> {
    return this.get(`/meta-reality/perception/${userId}`, { params: perceptionData });
  }

  async shiftReality(userId: string, shiftParameters: any): Promise<any> {
    return this.post('/meta-reality/shift', { user_id: userId, ...shiftParameters });
  }

  async getConsensusReality(): Promise<ConsensusReality> {
    return this.get('/meta-reality/consensus');
  }

  async anchorReality(userId: string, anchorData: any): Promise<any> {
    return this.post('/meta-reality/anchor', { user_id: userId, ...anchorData });
  }

  // Kelly Bot Management APIs
  async getKellyAccounts(): Promise<KellyAccount[]> {
    return this.get('/kelly/accounts');
  }

  async createKellyAccount(accountData: Partial<KellyAccount>): Promise<KellyAccount> {
    return this.post('/kelly/accounts', accountData);
  }

  async updateKellyAccount(accountId: string, updates: Partial<KellyAccount>): Promise<KellyAccount> {
    return this.put(`/kelly/accounts/${accountId}`, updates);
  }

  async deleteKellyAccount(accountId: string): Promise<void> {
    return this.delete(`/kelly/accounts/${accountId}`);
  }

  async getKellyDashboard(): Promise<KellyDashboardOverview> {
    return this.get('/kelly/dashboard');
  }

  async getKellyConversations(accountId?: string): Promise<KellyConversation[]> {
    const endpoint = accountId ? `/kelly/conversations?account_id=${accountId}` : '/kelly/conversations';
    return this.get(endpoint);
  }

  async getKellyConversation(conversationId: string): Promise<KellyConversation> {
    return this.get(`/kelly/conversations/${conversationId}`);
  }

  async updateKellyConversation(conversationId: string, updates: Partial<KellyConversation>): Promise<KellyConversation> {
    return this.put(`/kelly/conversations/${conversationId}`, updates);
  }

  async getConversationAnalytics(conversationId: string): Promise<ConversationAnalytics> {
    return this.get(`/kelly/conversations/${conversationId}/analytics`);
  }

  async endKellyConversation(conversationId: string, reason?: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/end`, { reason });
  }

  async escalateKellyConversation(conversationId: string, escalationReason: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/escalate`, { reason: escalationReason });
  }

  // Claude AI Integration APIs
  async getClaudeConfig(accountId: string): Promise<ClaudeIntegrationConfig> {
    return this.get(`/kelly/accounts/${accountId}/claude-config`);
  }

  async updateClaudeConfig(accountId: string, config: Partial<ClaudeIntegrationConfig>): Promise<ClaudeIntegrationConfig> {
    return this.put(`/kelly/accounts/${accountId}/claude-config`, config);
  }

  async generateClaudeResponse(conversationId: string, prompt: string, options?: {
    model?: 'opus' | 'sonnet' | 'haiku';
    temperature?: number;
    max_tokens?: number;
  }): Promise<GeneratedResponse> {
    return this.post(`/kelly/conversations/${conversationId}/claude-response`, {
      prompt,
      ...options
    });
  }

  async getClaudeResponseAlternatives(conversationId: string, messageId: string): Promise<GeneratedResponse[]> {
    return this.get(`/kelly/conversations/${conversationId}/messages/${messageId}/alternatives`);
  }

  async selectClaudeResponse(conversationId: string, responseId: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/responses/${responseId}/select`);
  }

  async getClaudeUsageMetrics(accountId?: string): Promise<ClaudeUsageMetrics> {
    const endpoint = accountId ? `/kelly/claude/metrics?account_id=${accountId}` : '/kelly/claude/metrics';
    return this.get(endpoint);
  }

  async getClaudeCostBreakdown(period: 'daily' | 'weekly' | 'monthly' = 'daily'): Promise<any> {
    return this.get(`/kelly/claude/costs?period=${period}`);
  }

  async updateClaudeBudget(accountId: string, dailyLimit: number): Promise<void> {
    return this.put(`/kelly/accounts/${accountId}/claude-budget`, { daily_limit: dailyLimit });
  }

  // AI Features Management
  async getAIFeatures(accountId: string): Promise<AIFeatureConfig> {
    return this.get(`/kelly/accounts/${accountId}/ai-features`);
  }

  async updateAIFeatures(accountId: string, features: Partial<AIFeatureConfig>): Promise<AIFeatureConfig> {
    return this.put(`/kelly/accounts/${accountId}/ai-features`, features);
  }

  async toggleAIFeature(accountId: string, feature: keyof AIFeatureConfig, enabled: boolean): Promise<void> {
    return this.post(`/kelly/accounts/${accountId}/ai-features/${feature}/toggle`, { enabled });
  }

  // Safety and Monitoring APIs
  async getKellySafetyStatus(accountId?: string): Promise<SafetyStatus> {
    const endpoint = accountId ? `/kelly/safety?account_id=${accountId}` : '/kelly/safety';
    return this.get(endpoint);
  }

  async reportSafetyViolation(conversationId: string, violationType: string, description: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/safety-violation`, {
      violation_type: violationType,
      description
    });
  }

  async reviewSafetyAlert(alertId: string, action: 'approve' | 'reject' | 'escalate', notes?: string): Promise<void> {
    return this.post(`/kelly/safety/alerts/${alertId}/review`, {
      action,
      notes
    });
  }

  async updateSafetySettings(accountId: string, settings: any): Promise<void> {
    return this.put(`/kelly/accounts/${accountId}/safety-settings`, settings);
  }

  // Payment Management APIs
  async getPaymentSettings(accountId: string): Promise<PaymentSettings> {
    return this.get(`/kelly/accounts/${accountId}/payment-settings`);
  }

  async updatePaymentSettings(accountId: string, settings: Partial<PaymentSettings>): Promise<PaymentSettings> {
    return this.put(`/kelly/accounts/${accountId}/payment-settings`, settings);
  }

  async initializePaymentDiscussion(conversationId: string, pricingTierId: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/payment-discussion`, {
      pricing_tier_id: pricingTierId
    });
  }

  // Real-time Response Generation
  async streamClaudeResponse(conversationId: string, prompt: string, onUpdate: (data: any) => void): Promise<void> {
    const response = await this.client.post(`/kelly/conversations/${conversationId}/claude-stream`, {
      prompt
    }, {
      responseType: 'stream'
    });

    const reader = response.data.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onUpdate(data);
            } catch (error) {
              console.warn('Failed to parse streaming data:', error);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Conversation Management
  async sendKellyMessage(conversationId: string, message: string, options?: {
    useClaudeGeneration?: boolean;
    bypassSafetyCheck?: boolean;
  }): Promise<any> {
    return this.post(`/kelly/conversations/${conversationId}/send`, {
      message,
      ...options
    });
  }

  async pauseKellyConversation(conversationId: string, reason?: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/pause`, { reason });
  }

  async resumeKellyConversation(conversationId: string): Promise<void> {
    return this.post(`/kelly/conversations/${conversationId}/resume`);
  }

  async getConversationSuggestions(conversationId: string): Promise<string[]> {
    return this.get(`/kelly/conversations/${conversationId}/suggestions`);
  }

  // Health check
  async healthCheck(): Promise<any> {
    return this.get('/health');
  }

  // File upload helper
  async uploadFile(file: File, endpoint: string): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Auto-restore authentication on app start
apiClient.restoreAuth();

export default apiClient;