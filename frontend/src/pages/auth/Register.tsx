/**
 * Register Page Component
 * User registration with consciousness onboarding
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

export default function Register() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-consciousness">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="shadow-dramatic">
          <CardHeader className="text-center">
            <CardTitle className="text-insight-title text-gradient">
              Create Your Digital Twin
            </CardTitle>
            <p className="text-body-text text-text-secondary">
              Begin your consciousness exploration journey
            </p>
          </CardHeader>
          <CardContent>
            <div className="text-center space-y-4">
              <p className="text-sm text-text-secondary">
                Registration is currently in development.
              </p>
              <Link to="/login">
                <Button variant="primary" className="w-full">
                  Back to Login
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}