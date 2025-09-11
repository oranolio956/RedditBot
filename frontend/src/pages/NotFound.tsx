/**
 * 404 Not Found Page
 * Apple-inspired error page with consciousness theme
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HomeIcon, ArrowLeftIcon } from '@heroicons/react/24/outline';
import { Button } from '@/components/ui/Button';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-primary">
      <div className="text-center px-6 max-w-lg">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="space-y-8"
        >
          {/* 404 Visual */}
          <div className="space-y-4">
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity }}
              className="text-8xl"
            >
              🧠
            </motion.div>
            <h1 className="text-6xl font-bold text-gradient">404</h1>
          </div>

          {/* Error Message */}
          <div className="space-y-4">
            <h2 className="text-insight-title font-semibold text-text-primary">
              Consciousness Fragment Not Found
            </h2>
            <p className="text-body-text text-text-secondary">
              The requested consciousness pattern doesn't exist in this reality layer. 
              It might have quantum tunneled to another dimension.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Link to="/">
                <Button variant="primary" className="w-full sm:w-auto">
                  <HomeIcon className="w-4 h-4 mr-2" />
                  Return to Dashboard
                </Button>
              </Link>
              <Button 
                variant="outline" 
                onClick={() => window.history.back()}
                className="w-full sm:w-auto"
              >
                <ArrowLeftIcon className="w-4 h-4 mr-2" />
                Go Back
              </Button>
            </div>
          </div>

          {/* Helpful Links */}
          <div className="pt-8 border-t border-gray-200">
            <p className="text-sm text-text-tertiary mb-4">
              Explore these consciousness features instead:
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              <Link to="/consciousness" className="text-sm text-consciousness-primary hover:underline">
                Digital Twin
              </Link>
              <span className="text-text-tertiary">•</span>
              <Link to="/memory" className="text-sm text-consciousness-primary hover:underline">
                Memory Palace
              </Link>
              <span className="text-text-tertiary">•</span>
              <Link to="/quantum" className="text-sm text-consciousness-primary hover:underline">
                Quantum Network
              </Link>
              <span className="text-text-tertiary">•</span>
              <Link to="/emotional" className="text-sm text-consciousness-primary hover:underline">
                Emotional AI
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}