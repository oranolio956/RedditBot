/**
 * Telegram Account Connection Modal
 * Allows users to connect their personal Telegram accounts to Kelly system
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  XMarkIcon,
  DevicePhoneMobileIcon,
  ShieldCheckIcon,
  KeyIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { Button } from '../ui/Button';
import LoadingSpinner from '../ui/LoadingSpinner';

interface TelegramConnectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (account: any) => void;
}

type ConnectionStep = 'phone' | 'code' | 'password' | 'connecting' | 'success' | 'error';

export default function TelegramConnectModal({ isOpen, onClose, onSuccess }: TelegramConnectModalProps) {
  const [step, setStep] = useState<ConnectionStep>('phone');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [password, setPassword] = useState('');
  const [phoneCodeHash, setPhoneCodeHash] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePhoneSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/v1/telegram/send-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone_number: phoneNumber }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setPhoneCodeHash(data.phone_code_hash);
        setStep('code');
      } else {
        setError(data.detail || 'Failed to send verification code');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCodeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/v1/telegram/verify-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phone_number: phoneNumber,
          phone_code: verificationCode,
          phone_code_hash: phoneCodeHash,
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        if (data.requires_2fa) {
          setStep('password');
        } else {
          setStep('connecting');
          await connectAccount(data.session_string);
        }
      } else {
        setError(data.detail || 'Invalid verification code');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handle2FASubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/v1/telegram/verify-2fa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phone_number: phoneNumber,
          password: password,
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setStep('connecting');
        await connectAccount(data.session_string);
      } else {
        setError(data.detail || 'Invalid 2FA password');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const connectAccount = async (sessionString: string) => {
    try {
      const response = await fetch('/api/v1/telegram/connect-account', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phone_number: phoneNumber,
          session_string: sessionString,
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setStep('success');
        setTimeout(() => {
          onSuccess(data.account);
          resetModal();
        }, 2000);
      } else {
        setStep('error');
        setError(data.detail || 'Failed to connect account');
      }
    } catch (err) {
      setStep('error');
      setError('Failed to connect account. Please try again.');
    }
  };

  const resetModal = () => {
    setStep('phone');
    setPhoneNumber('');
    setVerificationCode('');
    setPassword('');
    setPhoneCodeHash('');
    setError('');
    onClose();
  };

  const modalVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-background-primary/80 backdrop-blur-sm"
              onClick={onClose}
            />

            {/* Modal */}
            <motion.div
              variants={modalVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              className="relative bg-card-background rounded-2xl shadow-2xl w-full max-w-md p-6"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold text-text-primary">
                  Connect Telegram Account
                </h2>
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-background-secondary transition-colors"
                >
                  <XMarkIcon className="w-5 h-5 text-text-secondary" />
                </button>
              </div>

              {/* Content based on step */}
              {step === 'phone' && (
                <form onSubmit={handlePhoneSubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-text-secondary mb-2">
                      Phone Number
                    </label>
                    <div className="relative">
                      <DevicePhoneMobileIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary" />
                      <input
                        type="tel"
                        placeholder="+1234567890"
                        value={phoneNumber}
                        onChange={(e) => setPhoneNumber(e.target.value)}
                        className="w-full pl-10 pr-4 py-3 bg-background-secondary rounded-lg border border-border-primary focus:border-consciousness-primary focus:ring-1 focus:ring-consciousness-primary transition-all"
                        required
                      />
                    </div>
                    <p className="mt-2 text-xs text-text-tertiary">
                      Enter your phone number with country code
                    </p>
                  </div>

                  {error && (
                    <div className="p-3 bg-states-stress/10 border border-states-stress/20 rounded-lg">
                      <p className="text-sm text-states-stress flex items-center">
                        <ExclamationTriangleIcon className="w-4 h-4 mr-2" />
                        {error}
                      </p>
                    </div>
                  )}

                  <Button
                    type="submit"
                    variant="primary"
                    className="w-full"
                    disabled={loading || !phoneNumber}
                  >
                    {loading ? <LoadingSpinner size="sm" /> : 'Send Verification Code'}
                  </Button>
                </form>
              )}

              {step === 'code' && (
                <form onSubmit={handleCodeSubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-text-secondary mb-2">
                      Verification Code
                    </label>
                    <div className="relative">
                      <ShieldCheckIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary" />
                      <input
                        type="text"
                        placeholder="12345"
                        value={verificationCode}
                        onChange={(e) => setVerificationCode(e.target.value)}
                        className="w-full pl-10 pr-4 py-3 bg-background-secondary rounded-lg border border-border-primary focus:border-consciousness-primary focus:ring-1 focus:ring-consciousness-primary transition-all"
                        autoFocus
                        required
                      />
                    </div>
                    <p className="mt-2 text-xs text-text-tertiary">
                      Enter the code sent to {phoneNumber}
                    </p>
                  </div>

                  {error && (
                    <div className="p-3 bg-states-stress/10 border border-states-stress/20 rounded-lg">
                      <p className="text-sm text-states-stress flex items-center">
                        <ExclamationTriangleIcon className="w-4 h-4 mr-2" />
                        {error}
                      </p>
                    </div>
                  )}

                  <div className="flex space-x-3">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setStep('phone')}
                      className="flex-1"
                    >
                      Back
                    </Button>
                    <Button
                      type="submit"
                      variant="primary"
                      className="flex-1"
                      disabled={loading || !verificationCode}
                    >
                      {loading ? <LoadingSpinner size="sm" /> : 'Verify'}
                    </Button>
                  </div>
                </form>
              )}

              {step === 'password' && (
                <form onSubmit={handle2FASubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-text-secondary mb-2">
                      2FA Password
                    </label>
                    <div className="relative">
                      <KeyIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary" />
                      <input
                        type="password"
                        placeholder="Enter your 2FA password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full pl-10 pr-4 py-3 bg-background-secondary rounded-lg border border-border-primary focus:border-consciousness-primary focus:ring-1 focus:ring-consciousness-primary transition-all"
                        autoFocus
                        required
                      />
                    </div>
                    <p className="mt-2 text-xs text-text-tertiary">
                      Your account has 2FA enabled. Enter your password.
                    </p>
                  </div>

                  {error && (
                    <div className="p-3 bg-states-stress/10 border border-states-stress/20 rounded-lg">
                      <p className="text-sm text-states-stress flex items-center">
                        <ExclamationTriangleIcon className="w-4 h-4 mr-2" />
                        {error}
                      </p>
                    </div>
                  )}

                  <div className="flex space-x-3">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setStep('code')}
                      className="flex-1"
                    >
                      Back
                    </Button>
                    <Button
                      type="submit"
                      variant="primary"
                      className="flex-1"
                      disabled={loading || !password}
                    >
                      {loading ? <LoadingSpinner size="sm" /> : 'Connect'}
                    </Button>
                  </div>
                </form>
              )}

              {step === 'connecting' && (
                <div className="text-center py-8">
                  <LoadingSpinner size="lg" />
                  <p className="mt-4 text-text-secondary">Connecting your account...</p>
                  <p className="mt-2 text-sm text-text-tertiary">
                    Setting up Kelly AI features for your account
                  </p>
                </div>
              )}

              {step === 'success' && (
                <div className="text-center py-8">
                  <CheckCircleIcon className="w-16 h-16 mx-auto text-states-flow mb-4" />
                  <h3 className="text-xl font-semibold text-text-primary mb-2">
                    Account Connected!
                  </h3>
                  <p className="text-text-secondary">
                    Kelly is now ready to manage your conversations
                  </p>
                </div>
              )}

              {step === 'error' && (
                <div className="text-center py-8">
                  <ExclamationTriangleIcon className="w-16 h-16 mx-auto text-states-stress mb-4" />
                  <h3 className="text-xl font-semibold text-text-primary mb-2">
                    Connection Failed
                  </h3>
                  <p className="text-text-secondary mb-4">{error}</p>
                  <Button
                    variant="primary"
                    onClick={() => setStep('phone')}
                  >
                    Try Again
                  </Button>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      )}
    </AnimatePresence>
  );
}