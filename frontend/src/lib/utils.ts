/**
 * Utility functions for the AI Consciousness Platform
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Tailwind CSS class name merger with conflict resolution
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format date/time for consciousness features
 */
export function formatDateTime(date: string | Date): string {
  const d = new Date(date);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(d);
}

export function formatRelativeTime(date: string | Date): string {
  const d = new Date(date);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return formatDateTime(date);
}

/**
 * Format consciousness metrics
 */
export function formatPercentage(value: number, decimals = 0): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatMetricValue(value: number, unit?: string): string {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M${unit ? ` ${unit}` : ''}`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}K${unit ? ` ${unit}` : ''}`;
  }
  return `${value.toFixed(0)}${unit ? ` ${unit}` : ''}`;
}

/**
 * Color utilities for consciousness states
 */
export function getConsciousnessColor(level: number): string {
  if (level >= 0.8) return 'text-states-flow';
  if (level >= 0.6) return 'text-consciousness-primary';
  if (level >= 0.4) return 'text-consciousness-accent';
  return 'text-states-stress';
}

export function getEmotionalColor(valence: number): string {
  if (valence > 0.6) return 'text-states-flow';
  if (valence > 0.2) return 'text-consciousness-primary';
  if (valence > -0.2) return 'text-states-neutral';
  if (valence > -0.6) return 'text-consciousness-accent';
  return 'text-states-stress';
}

/**
 * Data validation utilities
 */
export function isValidUserId(id: string): boolean {
  return /^[a-zA-Z0-9-_]+$/.test(id) && id.length > 0;
}

export function isValidTelegramId(id: number): boolean {
  return Number.isInteger(id) && id > 0;
}

export function sanitizeInput(input: string): string {
  return input.trim().replace(/[<>]/g, '');
}

/**
 * Local storage utilities with error handling
 */
export function getStorageItem(key: string, defaultValue?: any): any {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.warn(`Failed to parse localStorage item: ${key}`, error);
    return defaultValue;
  }
}

export function setStorageItem(key: string, value: any): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`Failed to set localStorage item: ${key}`, error);
  }
}

export function removeStorageItem(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn(`Failed to remove localStorage item: ${key}`, error);
  }
}

/**
 * Error handling utilities
 */
export function getErrorMessage(error: any): string {
  if (typeof error === 'string') return error;
  if (error?.message) return error.message;
  if (error?.data?.message) return error.data.message;
  if (error?.response?.data?.message) return error.response.data.message;
  return 'An unexpected error occurred';
}

export function isNetworkError(error: any): boolean {
  return error?.code === 'NETWORK_ERROR' || 
         error?.message?.includes('Network Error') ||
         error?.response?.status === 0;
}

/**
 * Array utilities
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

export function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((groups, item) => {
    const value = String(item[key]);
    groups[value] = groups[value] || [];
    groups[value].push(item);
    return groups;
  }, {} as Record<string, T[]>);
}

export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array));
}

export function uniqueBy<T>(array: T[], key: keyof T): T[] {
  const seen = new Set();
  return array.filter(item => {
    const value = item[key];
    if (seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

/**
 * Number utilities
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function lerp(start: number, end: number, factor: number): number {
  return start + (end - start) * factor;
}

export function roundTo(value: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

/**
 * String utilities
 */
export function truncate(text: string, length: number, suffix = '...'): string {
  if (text.length <= length) return text;
  return text.substring(0, length - suffix.length) + suffix;
}

export function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}

export function camelToTitle(text: string): string {
  return text
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase())
    .trim();
}

/**
 * URL utilities
 */
export function buildUrl(base: string, params: Record<string, any>): string {
  const url = new URL(base);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.append(key, String(value));
    }
  });
  return url.toString();
}

/**
 * Debounce utility
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Sleep utility for async operations
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Random utilities for consciousness simulation
 */
export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function randomChoice<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

export function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Animation utilities
 */
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function easeOutBounce(t: number): number {
  const n1 = 7.5625;
  const d1 = 2.75;

  if (t < 1 / d1) {
    return n1 * t * t;
  } else if (t < 2 / d1) {
    return n1 * (t -= 1.5 / d1) * t + 0.75;
  } else if (t < 2.5 / d1) {
    return n1 * (t -= 2.25 / d1) * t + 0.9375;
  } else {
    return n1 * (t -= 2.625 / d1) * t + 0.984375;
  }
}

/**
 * Theme utilities
 */
export function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function applyTheme(theme: 'light' | 'dark' | 'system'): void {
  const root = document.documentElement;
  root.classList.remove('light', 'dark');
  
  if (theme === 'system') {
    const systemTheme = getSystemTheme();
    root.classList.add(systemTheme);
  } else {
    root.classList.add(theme);
  }
}

/**
 * File utilities
 */
export function formatFileSize(bytes: number): string {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${roundTo(bytes / Math.pow(1024, i), 2)} ${sizes[i]}`;
}

export function getFileExtension(filename: string): string {
  return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
}

/**
 * Consciousness-specific utilities
 */
export function calculateCoherenceLevel(entanglements: any[]): number {
  if (entanglements.length === 0) return 0;
  const totalStrength = entanglements.reduce((sum, e) => sum + e.entanglement_strength, 0);
  return clamp(totalStrength / entanglements.length, 0, 1);
}

export function formatConfidenceLevel(confidence: number): { text: string; color: string } {
  if (confidence >= 0.9) return { text: 'Very High', color: 'text-states-flow' };
  if (confidence >= 0.7) return { text: 'High', color: 'text-consciousness-primary' };
  if (confidence >= 0.5) return { text: 'Medium', color: 'text-consciousness-accent' };
  if (confidence >= 0.3) return { text: 'Low', color: 'text-states-neutral' };
  return { text: 'Very Low', color: 'text-states-stress' };
}

export function getPersonalityTraitDescription(trait: string, value: number): string {
  const descriptions: Record<string, Record<string, string>> = {
    openness: {
      high: 'Creative, curious, and open to new experiences',
      medium: 'Balanced between routine and novelty',
      low: 'Prefers familiar patterns and traditional approaches'
    },
    conscientiousness: {
      high: 'Organized, disciplined, and goal-oriented',
      medium: 'Moderately organized with good self-control',
      low: 'Spontaneous and flexible in approach'
    },
    extraversion: {
      high: 'Outgoing, energetic, and socially engaged',
      medium: 'Balanced between social and solitary activities',
      low: 'Introspective and prefers quieter environments'
    },
    agreeableness: {
      high: 'Cooperative, trusting, and empathetic',
      medium: 'Balanced in social interactions',
      low: 'Independent and direct in communication'
    },
    neuroticism: {
      high: 'Sensitive to stress with intense emotional responses',
      medium: 'Moderate emotional stability',
      low: 'Calm, resilient, and emotionally stable'
    }
  };

  const level = value > 0.6 ? 'high' : value > 0.4 ? 'medium' : 'low';
  return descriptions[trait]?.[level] || 'Trait description unavailable';
}