/**
 * Calibration Panel Component
 * Modal for consciousness mirror calibration
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { XMarkIcon, CogIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

interface CalibrationPanelProps {
  isOpen: boolean;
  onClose: () => void;
  currentProfile?: any;
}

export default function CalibrationPanel({ isOpen, onClose, currentProfile }: CalibrationPanelProps) {
  const [calibrationStep, setCalibrationStep] = useState(0);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationComplete, setCalibrationComplete] = useState(false);

  const calibrationSteps = [
    {
      title: 'Personality Assessment',
      description: 'Answer questions about your personality traits',
      duration: 2000,
    },
    {
      title: 'Behavioral Patterns',
      description: 'Analyze typing and interaction patterns',
      duration: 3000,
    },
    {
      title: 'Cognitive Mapping',
      description: 'Map cognitive preferences and decision-making style',
      duration: 2500,
    },
    {
      title: 'Validation Testing',
      description: 'Verify mirror accuracy against known responses',
      duration: 1500,
    },
  ];

  const startCalibration = async () => {
    setIsCalibrating(true);
    setCalibrationStep(0);

    for (let i = 0; i < calibrationSteps.length; i++) {
      setCalibrationStep(i);
      await new Promise(resolve => setTimeout(resolve, calibrationSteps[i].duration));
    }

    setCalibrationComplete(true);
    setIsCalibrating(false);
  };

  const resetCalibration = () => {
    setCalibrationStep(0);
    setIsCalibrating(false);
    setCalibrationComplete(false);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="w-full max-w-2xl mx-4"
          onClick={(e) => e.stopPropagation()}
        >
          <Card className="shadow-dramatic">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-lg bg-consciousness-primary/10 flex items-center justify-center">
                    <CogIcon className="w-5 h-5 text-consciousness-primary" />
                  </div>
                  <div>
                    <CardTitle>Consciousness Mirror Calibration</CardTitle>
                    <p className="text-sm text-text-secondary">
                      Fine-tune your digital twin for maximum accuracy
                    </p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <XMarkIcon className="w-5 h-5" />
                </button>
              </div>
            </CardHeader>

            <CardContent className="space-y-6">
              {/* Current Status */}
              {currentProfile && (
                <div className="grid grid-cols-3 gap-4 p-4 bg-surface-secondary rounded-lg">
                  <div className="text-center">
                    <div className="text-lg font-semibold text-consciousness-primary">
                      {Math.round((currentProfile.confidence_level || 0) * 100)}%
                    </div>
                    <div className="text-xs text-text-tertiary">Current Accuracy</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-semibold text-consciousness-primary">
                      {Math.round((currentProfile.calibration_accuracy || 0) * 100)}%
                    </div>
                    <div className="text-xs text-text-tertiary">Calibration Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-semibold text-consciousness-primary">
                      {currentProfile.last_updated ? 
                        Math.floor((Date.now() - new Date(currentProfile.last_updated).getTime()) / (1000 * 60 * 60 * 24))
                        : 'Never'
                      }
                    </div>
                    <div className="text-xs text-text-tertiary">Days Since Calibration</div>
                  </div>
                </div>
              )}

              {!isCalibrating && !calibrationComplete && (
                <div className="space-y-4">
                  <h3 className="font-semibold text-text-primary">Calibration Process</h3>
                  <div className="space-y-3">
                    {calibrationSteps.map((step, index) => (
                      <div key={index} className="flex items-center space-x-3 p-3 rounded-lg bg-surface-secondary">
                        <div className="w-8 h-8 rounded-full bg-consciousness-primary/10 flex items-center justify-center">
                          <span className="text-sm font-semibold text-consciousness-primary">
                            {index + 1}
                          </span>
                        </div>
                        <div>
                          <h4 className="font-medium text-text-primary">{step.title}</h4>
                          <p className="text-sm text-text-secondary">{step.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex justify-end space-x-3">
                    <Button variant="outline" onClick={onClose}>
                      Cancel
                    </Button>
                    <Button 
                      variant="primary" 
                      onClick={startCalibration}
                      animation="breathing"
                    >
                      Start Calibration
                    </Button>
                  </div>
                </div>
              )}

              {isCalibrating && (
                <div className="space-y-6">
                  <div className="text-center">
                    <motion.div
                      className="w-16 h-16 mx-auto mb-4 rounded-full bg-consciousness-gradient flex items-center justify-center"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                    >
                      <CogIcon className="w-8 h-8 text-white" />
                    </motion.div>
                    <h3 className="text-lg font-semibold text-text-primary mb-2">
                      {calibrationSteps[calibrationStep]?.title}
                    </h3>
                    <p className="text-text-secondary">
                      {calibrationSteps[calibrationStep]?.description}
                    </p>
                  </div>

                  {/* Progress bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Progress</span>
                      <span>{Math.round(((calibrationStep + 1) / calibrationSteps.length) * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="h-2 bg-consciousness-primary rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${((calibrationStep + 1) / calibrationSteps.length) * 100}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>

                  {/* Step indicators */}
                  <div className="flex justify-center space-x-2">
                    {calibrationSteps.map((_, index) => (
                      <div
                        key={index}
                        className={`w-3 h-3 rounded-full transition-colors ${
                          index <= calibrationStep 
                            ? 'bg-consciousness-primary' 
                            : 'bg-gray-300'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              )}

              {calibrationComplete && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center space-y-4"
                >
                  <div className="w-16 h-16 mx-auto rounded-full bg-states-flow/10 flex items-center justify-center">
                    <CheckCircleIcon className="w-8 h-8 text-states-flow" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-text-primary">
                      Calibration Complete!
                    </h3>
                    <p className="text-text-secondary">
                      Your consciousness mirror has been successfully calibrated.
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 p-4 bg-surface-secondary rounded-lg">
                    <div className="text-center">
                      <div className="text-xl font-bold text-states-flow">96%</div>
                      <div className="text-xs text-text-tertiary">New Accuracy</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xl font-bold text-states-flow">+12%</div>
                      <div className="text-xs text-text-tertiary">Improvement</div>
                    </div>
                  </div>

                  <div className="flex justify-center space-x-3">
                    <Button variant="outline" onClick={resetCalibration}>
                      Recalibrate
                    </Button>
                    <Button variant="success" onClick={onClose}>
                      Done
                    </Button>
                  </div>
                </motion.div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}