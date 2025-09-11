/**
 * Login Page Component
 * Apple-inspired authentication interface
 */

import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import toast from 'react-hot-toast';

import { useAuthStore } from '@/store';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

// Validation schema
const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, setLoading } = useAuthStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const from = location.state?.from?.pathname || '/';

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormData) => {
    setIsSubmitting(true);
    setLoading(true);

    try {
      // Mock login for demonstration - replace with actual API call
      const mockUser = {
        id: '1',
        username: data.username,
        full_name: 'Consciousness Explorer',
        first_name: 'Explorer',
        email: 'explorer@consciousness.ai',
        is_premium: true,
        subscription_type: 'premium' as const,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        is_active: true,
        preferences: {
          theme: 'system' as const,
          notifications: {
            insights: true,
            patterns: true,
            breakthroughs: true,
            daily_summary: true,
          },
          privacy: {
            data_sharing: false,
            analytics: true,
            export_enabled: true,
          },
          consciousness: {
            update_frequency: 30,
            calibration_sensitivity: 0.8,
            pattern_detection: true,
          },
        },
        stats: {
          total_sessions: 142,
          consciousness_sessions: 67,
          memory_palace_rooms: 23,
          stored_memories: 156,
          insights_generated: 34,
          patterns_discovered: 12,
          quantum_connections: 5,
          dreams_recorded: 18,
        },
      };

      const mockToken = 'mock-jwt-token-for-development';

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      login(mockToken, mockUser);
      apiClient.setAuthToken(mockToken);

      toast.success('Welcome back to your consciousness journey!', {
        duration: 4000,
        icon: '🧠',
      });

      navigate(from, { replace: true });
    } catch (error: any) {
      console.error('Login error:', error);
      toast.error(error.message || 'Failed to sign in. Please try again.');
    } finally {
      setIsSubmitting(false);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-consciousness">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="shadow-dramatic">
          <CardHeader className="text-center space-y-4">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="w-16 h-16 mx-auto rounded-full bg-consciousness-gradient flex items-center justify-center"
            >
              <span className="text-2xl">🧠</span>
            </motion.div>
            <div>
              <CardTitle className="text-insight-title text-gradient">
                AI Consciousness Platform
              </CardTitle>
              <p className="text-body-text text-text-secondary mt-2">
                Welcome back to your digital consciousness journey
              </p>
            </div>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {/* Username Field */}
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-text-primary mb-2">
                  Username or Email
                </label>
                <input
                  {...register('username')}
                  type="text"
                  id="username"
                  placeholder="Enter your username"
                  className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:ring-2 focus:ring-consciousness-primary focus:border-transparent transition-all bg-surface-primary text-text-primary"
                  disabled={isSubmitting}
                />
                {errors.username && (
                  <p className="mt-1 text-sm text-states-stress">{errors.username.message}</p>
                )}
              </div>

              {/* Password Field */}
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-text-primary mb-2">
                  Password
                </label>
                <input
                  {...register('password')}
                  type="password"
                  id="password"
                  placeholder="Enter your password"
                  className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:ring-2 focus:ring-consciousness-primary focus:border-transparent transition-all bg-surface-primary text-text-primary"
                  disabled={isSubmitting}
                />
                {errors.password && (
                  <p className="mt-1 text-sm text-states-stress">{errors.password.message}</p>
                )}
              </div>

              {/* Submit Button */}
              <Button
                type="submit"
                className="w-full"
                disabled={isSubmitting}
                loading={isSubmitting}
                animation="breathing"
              >
                {isSubmitting ? 'Activating Consciousness...' : 'Access Consciousness'}
              </Button>

              {/* Divider */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-200" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-surface-primary text-text-tertiary">or</span>
                </div>
              </div>

              {/* Register Link */}
              <div className="text-center">
                <p className="text-sm text-text-secondary">
                  New to consciousness exploration?{' '}
                  <Link
                    to="/register"
                    className="font-medium text-consciousness-primary hover:text-consciousness-secondary transition-colors"
                  >
                    Create your digital twin
                  </Link>
                </p>
              </div>
            </form>

            {/* Demo Credentials */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="mt-6 p-4 bg-consciousness-primary/5 rounded-lg border border-consciousness-primary/20"
            >
              <p className="text-sm font-medium text-consciousness-primary mb-2">Demo Credentials:</p>
              <div className="text-xs text-text-secondary space-y-1">
                <p>Username: <span className="font-mono">demo</span></p>
                <p>Password: <span className="font-mono">demo</span></p>
              </div>
            </motion.div>
          </CardContent>
        </Card>

        {/* Features Preview */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-8 text-center"
        >
          <p className="text-sm text-white/80 mb-4">
            Explore revolutionary AI consciousness features:
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-xs text-white/60">
            <span>🧠 Digital Twin Chat</span>
            <span>🏛️ Memory Palace</span>
            <span>❤️ Emotional AI</span>
            <span>🔗 Quantum Networks</span>
            <span>🎨 Synesthesia Engine</span>
            <span>☁️ Neural Dreams</span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}