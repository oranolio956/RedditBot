/**
 * React Query hooks for API integration
 * Type-safe hooks for all AI consciousness features
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';
import { useAuthStore } from '@/store';
import {
  User,
  MemoryPalace,
} from '@/types';

// Query keys for consistent caching
export const queryKeys = {
  // User queries
  users: ['users'] as const,
  user: (id: string) => ['users', id] as const,
  userStats: (id: string) => ['users', id, 'stats'] as const,
  userByTelegram: (telegramId: number) => ['users', 'telegram', telegramId] as const,

  // Consciousness queries
  consciousness: (userId: string) => ['consciousness', userId] as const,
  consciousnessProfile: (userId: string) => ['consciousness', userId, 'profile'] as const,
  consciousnessEvolution: (userId: string) => ['consciousness', userId, 'evolution'] as const,
  consciousnessSessions: (userId: string) => ['consciousness', userId, 'sessions'] as const,

  // Memory Palace queries
  memoryPalaces: (userId: string) => ['memory-palaces', userId] as const,
  memoryPalace: (id: string) => ['memory-palaces', id] as const,
  palaceRooms: (palaceId: string) => ['memory-palaces', palaceId, 'rooms'] as const,
  roomMemories: (roomId: string) => ['memory-rooms', roomId, 'memories'] as const,
  memorySearch: (query: string) => ['memory-search', query] as const,

  // Emotional Intelligence queries
  emotional: (userId: string) => ['emotional', userId] as const,
  emotionalProfile: (userId: string) => ['emotional', userId, 'profile'] as const,
  moodHistory: (userId: string) => ['emotional', userId, 'mood-history'] as const,
  emotionalInsights: (userId: string) => ['emotional', userId, 'insights'] as const,

  // Telegram queries
  telegram: ['telegram'] as const,
  telegramStatus: ['telegram', 'status'] as const,
  telegramMetrics: ['telegram', 'metrics'] as const,
  telegramSessions: ['telegram', 'sessions'] as const,

  // Quantum queries
  quantum: (userId: string) => ['quantum', userId] as const,
  quantumNetwork: (userId: string) => ['quantum', userId, 'network'] as const,
  quantumObservations: (userId: string) => ['quantum', userId, 'observations'] as const,

  // Synesthesia queries
  synesthesia: (userId: string) => ['synesthesia', userId] as const,
  synesthesiaProfile: (userId: string) => ['synesthesia', userId, 'profile'] as const,
  synesthesiaMappings: (userId: string) => ['synesthesia', userId, 'mappings'] as const,
  synesthesiaGallery: ['synesthesia', 'gallery'] as const,

  // Dreams queries
  dreams: (userId: string) => ['dreams', userId] as const,
  dreamLibrary: (userId: string) => ['dreams', userId, 'library'] as const,
  dreamPatterns: (userId: string) => ['dreams', userId, 'patterns'] as const,
  dreamGallery: ['dreams', 'gallery'] as const,
};

// User Management Hooks
export function useUsers(page = 1, perPage = 20, search?: string) {
  return useQuery({
    queryKey: [...queryKeys.users, page, perPage, search],
    queryFn: () => apiClient.getUsers(page, perPage, search),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useUser(userId: string) {
  return useQuery({
    queryKey: queryKeys.user(userId),
    queryFn: () => apiClient.getUserById(userId),
    enabled: !!userId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

export function useUserStats(userId: string) {
  return useQuery({
    queryKey: queryKeys.userStats(userId),
    queryFn: () => apiClient.getUserStats(userId),
    enabled: !!userId,
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
  });
}

export function useUpdateUser() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ userId, userData }: { userId: string; userData: Partial<User> }) =>
      apiClient.updateUser(userId, userData),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.user(variables.userId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
    },
  });
}

// Consciousness Hooks
export function useConsciousnessProfile(userId: string) {
  return useQuery({
    queryKey: queryKeys.consciousnessProfile(userId),
    queryFn: () => apiClient.getConsciousnessProfile(userId),
    enabled: !!userId,
    refetchInterval: 30 * 1000, // Refetch every 30 seconds for real-time feel
  });
}

export function usePersonalityEvolution(userId: string) {
  return useQuery({
    queryKey: queryKeys.consciousnessEvolution(userId),
    queryFn: () => apiClient.getPersonalityEvolution(userId),
    enabled: !!userId,
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
}

export function useConsciousnessSessions(userId: string) {
  return useQuery({
    queryKey: queryKeys.consciousnessSessions(userId),
    queryFn: () => apiClient.getConsciousnessSessions(userId),
    enabled: !!userId,
  });
}

export function useUpdateConsciousness() {
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: (update: any) => apiClient.updateConsciousness(user!.id, update),
    onSuccess: () => {
      if (user) {
        queryClient.invalidateQueries({ queryKey: queryKeys.consciousnessProfile(user.id) });
      }
    },
  });
}

export function useTwinChat() {
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: (message: string) => apiClient.chatWithTwin(user!.id, message),
  });
}

export function usePredictDecision() {
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: ({ decision, options }: { decision: string; options: string[] }) =>
      apiClient.predictDecision(user!.id, decision, options),
  });
}

// Memory Palace Hooks
export function useMemoryPalaces(userId: string) {
  return useQuery({
    queryKey: queryKeys.memoryPalaces(userId),
    queryFn: () => apiClient.getUserMemoryPalaces(userId),
    enabled: !!userId,
  });
}

export function useMemoryPalace(palaceId: string) {
  return useQuery({
    queryKey: queryKeys.memoryPalace(palaceId),
    queryFn: () => apiClient.getMemoryPalace(palaceId),
    enabled: !!palaceId,
  });
}

export function usePalaceRooms(palaceId: string) {
  return useQuery({
    queryKey: queryKeys.palaceRooms(palaceId),
    queryFn: () => apiClient.getPalaceRooms(palaceId),
    enabled: !!palaceId,
  });
}

export function useRoomMemories(roomId: string) {
  return useQuery({
    queryKey: queryKeys.roomMemories(roomId),
    queryFn: () => apiClient.getRoomMemories(roomId),
    enabled: !!roomId,
  });
}

export function useMemorySearch(query: string, palaceId?: string) {
  return useQuery({
    queryKey: [...queryKeys.memorySearch(query), palaceId],
    queryFn: () => apiClient.searchMemories(query, palaceId),
    enabled: query.length > 2,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}

export function useCreateMemoryPalace() {
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: (data: Partial<MemoryPalace>) => apiClient.createMemoryPalace(data),
    onSuccess: () => {
      if (user) {
        queryClient.invalidateQueries({ queryKey: queryKeys.memoryPalaces(user.id) });
      }
    },
  });
}

export function useStoreMemory() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ roomId, memoryData }: { roomId: string; memoryData: any }) =>
      apiClient.storeMemory(roomId, memoryData),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.roomMemories(variables.roomId) });
    },
  });
}

// Emotional Intelligence Hooks
export function useEmotionalProfile(userId: string) {
  return useQuery({
    queryKey: queryKeys.emotionalProfile(userId),
    queryFn: () => apiClient.getEmotionalProfile(userId),
    enabled: !!userId,
    refetchInterval: 60 * 1000, // Refetch every minute
  });
}

export function useMoodHistory(userId: string, period?: string) {
  return useQuery({
    queryKey: [...queryKeys.moodHistory(userId), period],
    queryFn: () => apiClient.getMoodHistory(userId, period),
    enabled: !!userId,
  });
}

export function useEmotionalInsights(userId: string) {
  return useQuery({
    queryKey: queryKeys.emotionalInsights(userId),
    queryFn: () => apiClient.getEmotionalInsights(userId),
    enabled: !!userId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

export function useTrackMood() {
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: (moodData: any) => apiClient.trackMood(user!.id, moodData),
    onSuccess: () => {
      if (user) {
        queryClient.invalidateQueries({ queryKey: queryKeys.moodHistory(user.id) });
        queryClient.invalidateQueries({ queryKey: queryKeys.emotionalProfile(user.id) });
      }
    },
  });
}

export function useAnalyzeEmotion() {
  return useMutation({
    mutationFn: ({ text, context }: { text: string; context?: any }) =>
      apiClient.analyzeEmotion(text, context),
  });
}

// Telegram Bot Hooks
export function useTelegramStatus() {
  return useQuery({
    queryKey: queryKeys.telegramStatus,
    queryFn: () => apiClient.getTelegramStatus(),
    refetchInterval: 10 * 1000, // Refetch every 10 seconds
  });
}

export function useTelegramMetrics() {
  return useQuery({
    queryKey: queryKeys.telegramMetrics,
    queryFn: () => apiClient.getTelegramMetrics(),
    refetchInterval: 15 * 1000, // Refetch every 15 seconds
  });
}

export function useTelegramSessions() {
  return useQuery({
    queryKey: queryKeys.telegramSessions,
    queryFn: () => apiClient.getTelegramSessions(),
    refetchInterval: 30 * 1000, // Refetch every 30 seconds
  });
}

export function useSendTelegramMessage() {
  return useMutation({
    mutationFn: ({ chatId, message, options }: { chatId: number; message: string; options?: any }) =>
      apiClient.sendTelegramMessage(chatId, message, options),
  });
}

// Quantum Consciousness Hooks
export function useQuantumNetwork(userId: string) {
  return useQuery({
    queryKey: queryKeys.quantumNetwork(userId),
    queryFn: () => apiClient.getQuantumNetwork(userId),
    enabled: !!userId,
    refetchInterval: 20 * 1000, // Refetch every 20 seconds
  });
}

export function useQuantumObservations(userId: string) {
  return useQuery({
    queryKey: queryKeys.quantumObservations(userId),
    queryFn: () => apiClient.getQuantumObservations(userId),
    enabled: !!userId,
  });
}

export function useCreateEntanglement() {
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: (targetUserId: string) => apiClient.createQuantumEntanglement(user!.id, targetUserId),
    onSuccess: () => {
      if (user) {
        queryClient.invalidateQueries({ queryKey: queryKeys.quantumNetwork(user.id) });
      }
    },
  });
}

export function useTeleportThought() {
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: ({ receiverId, thoughtData }: { receiverId: string; thoughtData: any }) =>
      apiClient.teleportThought(user!.id, receiverId, thoughtData),
  });
}

// Synesthesia Hooks
export function useSynestheticProfile(userId: string) {
  return useQuery({
    queryKey: queryKeys.synesthesiaProfile(userId),
    queryFn: () => apiClient.getSynestheticProfile(userId),
    enabled: !!userId,
  });
}

export function useSensoryMappings(userId: string) {
  return useQuery({
    queryKey: queryKeys.synesthesiaMappings(userId),
    queryFn: () => apiClient.getSensoryMappings(userId),
    enabled: !!userId,
  });
}

export function useSynesthesiaGallery() {
  return useQuery({
    queryKey: queryKeys.synesthesiaGallery,
    queryFn: () => apiClient.getSynesthesiaGallery(),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

export function useConvertModalities() {
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: ({ inputData, fromModality, toModality }: {
      inputData: any;
      fromModality: string;
      toModality: string;
    }) => apiClient.convertModalities(user!.id, inputData, fromModality, toModality),
  });
}

// Dreams Hooks
export function useDreamLibrary(userId: string) {
  return useQuery({
    queryKey: queryKeys.dreamLibrary(userId),
    queryFn: () => apiClient.getDreamLibrary(userId),
    enabled: !!userId,
  });
}

export function useDreamPatterns(userId: string) {
  return useQuery({
    queryKey: queryKeys.dreamPatterns(userId),
    queryFn: () => apiClient.getDreamPatterns(userId),
    enabled: !!userId,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

export function useDreamGallery() {
  return useQuery({
    queryKey: queryKeys.dreamGallery,
    queryFn: () => apiClient.getDreamGallery(),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

export function useInitiateDream() {
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  
  return useMutation({
    mutationFn: (dreamParameters: any) => apiClient.initiateDreamSession(user!.id, dreamParameters),
    onSuccess: () => {
      if (user) {
        queryClient.invalidateQueries({ queryKey: queryKeys.dreamLibrary(user.id) });
      }
    },
  });
}

export function useInterpretDream() {
  return useMutation({
    mutationFn: ({ sessionId, method }: { sessionId: string; method?: string }) =>
      apiClient.interpretDream(sessionId, method),
  });
}

// Generic hooks for common patterns
export function useRefreshData() {
  const queryClient = useQueryClient();
  
  return {
    refreshAll: () => queryClient.invalidateQueries(),
    refresh: (queryKey: any) => queryClient.invalidateQueries({ queryKey }),
  };
}

export function useOptimisticUpdate<T>(queryKey: any, updateFn: (old: T, variables: any) => T) {
  const queryClient = useQueryClient();
  
  return (variables: any) => {
    queryClient.setQueryData<T>(queryKey, (old) => {
      if (!old) return old;
      return updateFn(old, variables);
    });
  };
}