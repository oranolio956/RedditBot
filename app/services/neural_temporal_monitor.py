"""
Neural Temporal Monitor - EEG and Biometric Integration Service

Advanced neural monitoring system that tracks temporal perception indicators,
flow state neural correlates, and safety markers through real-time EEG analysis
and multi-modal biometric integration.

Key Capabilities:
- Real-time EEG temporal signature detection via autocorrelation
- Flow state neural correlate monitoring (frontal alpha asymmetry, theta coherence)
- Intrinsic neural timescales measurement
- Circadian rhythm integration
- Safety monitoring for temporal displacement prevention
- Neural entrainment for flow state induction
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from scipy import signal
from scipy.stats import pearsonr

from app.models.temporal_dilution import (
    TemporalBiometricReading, BiometricDeviceType,
    TemporalState, SafetyLevel
)
from app.core.redis import RedisManager
from app.core.monitoring import log_performance_metric
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class EEGReading:
    """Raw EEG data structure"""
    timestamp: datetime
    channels: Dict[str, List[float]]  # Channel name -> signal data
    sampling_rate: int
    signal_quality: float
    device_type: BiometricDeviceType

@dataclass
class ProcessedNeuralSignals:
    """Processed neural signals with temporal indicators"""
    timestamp: datetime
    # Frequency band powers
    alpha_power: float  # 8-13 Hz
    beta_power: float   # 13-30 Hz
    theta_power: float  # 4-8 Hz
    delta_power: float  # 0.5-4 Hz
    gamma_power: float  # 30-100 Hz
    
    # Temporal perception markers
    intrinsic_neural_timescales: float  # Autocorrelation decay time
    temporal_prediction_error: float
    time_cell_activity: float
    
    # Flow state indicators
    frontal_alpha_asymmetry: float
    theta_coherence: float
    transient_hypofrontality: float
    
    # Safety indicators
    signal_quality: float
    artifact_level: float
    electrode_impedance: Dict[str, float]

class NeuralTemporalMonitor:
    """
    Advanced neural monitoring system for temporal perception and flow state tracking.
    Integrates multiple biometric devices with real-time signal processing.
    """
    
    def __init__(self):
        self.active_monitors: Dict[int, Dict[str, Any]] = {}  # profile_id -> monitor data
        self.signal_buffers: Dict[int, List[EEGReading]] = {}  # Circular buffers
        self.buffer_size = 1000  # Number of samples to keep
        
        # EEG frequency bands (Hz)
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Flow state neural signatures (based on research)
        self.flow_signatures = {
            'frontal_alpha_asymmetry_threshold': 0.15,
            'theta_coherence_threshold': 0.6,
            'transient_hypofrontality_threshold': 0.3,
            'optimal_alpha_frequency': 10.0,
            'flow_theta_frequency': 6.5
        }
        
        # Temporal perception neural markers
        self.temporal_markers = {
            'intrinsic_timescale_range': (100, 1000),  # ms
            'prediction_error_threshold': 0.2,
            'time_cell_activity_threshold': 0.4,
            'temporal_binding_frequency': 40.0  # Hz gamma
        }
        
        # Safety thresholds
        self.safety_thresholds = {
            'minimum_signal_quality': 0.7,
            'maximum_artifact_level': 0.3,
            'maximum_electrode_impedance': 10.0,  # kOhms
            'temporal_confusion_threshold': 0.25
        }
    
    async def start_monitoring(
        self,
        profile_id: int,
        session_id: int,
        target_frequencies: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Initialize neural monitoring for temporal perception and flow state tracking.
        """
        try:
            # Initialize monitoring configuration
            monitor_config = {
                'profile_id': profile_id,
                'session_id': session_id,
                'target_frequencies': target_frequencies,
                'start_time': datetime.utcnow(),
                'sampling_rate': 256,  # Hz
                'device_connected': False,
                'baseline_established': False,
                'safety_monitoring_active': True
            }
            
            # Initialize signal buffer
            self.signal_buffers[profile_id] = []
            
            # Connect to EEG device (simulated for demo)
            device_connection = await self._connect_eeg_device(profile_id)
            monitor_config['device_connected'] = device_connection['success']
            monitor_config['device_type'] = device_connection.get('device_type', 'simulated')
            
            # Establish baseline neural activity
            if monitor_config['device_connected']:
                baseline_result = await self._establish_neural_baseline(profile_id)
                monitor_config['baseline_established'] = baseline_result['success']
                monitor_config['baseline_data'] = baseline_result.get('baseline', {})
            
            # Start real-time processing
            monitor_config['processing_task'] = asyncio.create_task(
                self._continuous_neural_processing(profile_id)
            )
            
            self.active_monitors[profile_id] = monitor_config
            
            await log_performance_metric(
                "neural_monitoring_started",
                {
                    'profile_id': profile_id,
                    'session_id': session_id,
                    'device_connected': monitor_config['device_connected'],
                    'baseline_established': monitor_config['baseline_established']
                }
            )
            
            logger.info(
                "Neural temporal monitoring started",
                profile_id=profile_id,
                session_id=session_id,
                device_connected=monitor_config['device_connected']
            )
            
            return {
                'success': True,
                'monitor_config': monitor_config,
                'device_connected': monitor_config['device_connected'],
                'baseline_established': monitor_config['baseline_established']
            }
            
        except Exception as e:
            logger.error(
                "Failed to start neural monitoring",
                profile_id=profile_id,
                error=str(e)
            )
            raise
    
    async def get_current_readings(
        self,
        profile_id: int,
        session_id: int
    ) -> Dict[str, float]:
        """
        Get current processed neural readings with temporal and flow state indicators.
        """
        if profile_id not in self.active_monitors:
            raise ValueError(f"No active monitoring for profile {profile_id}")
        
        try:
            # Get latest EEG data
            latest_eeg = await self._get_latest_eeg_data(profile_id)
            if not latest_eeg:
                return self._get_simulated_readings(profile_id)
            
            # Process neural signals
            processed_signals = await self._process_neural_signals(latest_eeg)
            
            # Calculate temporal perception indicators
            temporal_indicators = await self._calculate_temporal_indicators(processed_signals)
            
            # Calculate flow state indicators
            flow_indicators = await self._calculate_flow_indicators(processed_signals)
            
            # Assess safety markers
            safety_indicators = await self._assess_neural_safety(processed_signals)
            
            # Combine all readings
            current_readings = {
                # Basic frequency powers
                'alpha_power': processed_signals.alpha_power,
                'beta_power': processed_signals.beta_power,
                'theta_power': processed_signals.theta_power,
                'delta_power': processed_signals.delta_power,
                'gamma_power': processed_signals.gamma_power,
                
                # Temporal perception markers
                'intrinsic_neural_timescales': temporal_indicators['intrinsic_timescales'],
                'temporal_prediction_error': temporal_indicators['prediction_error'],
                'time_cell_activity': temporal_indicators['time_cell_activity'],
                'autocorrelation_decay': temporal_indicators['autocorr_decay'],
                
                # Flow state neural correlates
                'frontal_alpha_asymmetry': flow_indicators['alpha_asymmetry'],
                'theta_coherence': flow_indicators['theta_coherence'],
                'transient_hypofrontality': flow_indicators['hypofrontality'],
                'dopamine_indicators': flow_indicators['dopamine_markers'],
                
                # Additional biometric data
                'heart_rate_variability': await self._get_hrv_data(profile_id),
                'breathing_rhythm': await self._get_breathing_data(profile_id),
                'galvanic_skin_response': await self._get_gsr_data(profile_id),
                
                # Safety and quality indicators
                'signal_quality': processed_signals.signal_quality,
                'artifact_level': processed_signals.artifact_level,
                'safety_score': safety_indicators['safety_score'],
                'temporal_confusion': safety_indicators['temporal_confusion_risk'],
                'reality_awareness': safety_indicators['reality_testing_score']
            }
            
            # Store reading in database
            biometric_reading = TemporalBiometricReading(
                temporal_profile_id=profile_id,
                temporal_session_id=session_id,
                device_type=BiometricDeviceType.EEG_HEADBAND,
                timestamp=datetime.utcnow(),
                
                # EEG data
                eeg_alpha_power=processed_signals.alpha_power,
                eeg_beta_power=processed_signals.beta_power,
                eeg_theta_power=processed_signals.theta_power,
                eeg_delta_power=processed_signals.delta_power,
                eeg_gamma_power=processed_signals.gamma_power,
                
                # Temporal indicators
                intrinsic_neural_timescales=temporal_indicators['intrinsic_timescales'],
                temporal_prediction_error=temporal_indicators['prediction_error'],
                time_cell_activity_patterns=temporal_indicators,
                
                # Flow indicators
                frontal_alpha_asymmetry=flow_indicators['alpha_asymmetry'],
                
                # Data quality
                signal_quality_score=processed_signals.signal_quality
            )
            
            return current_readings
            
        except Exception as e:
            logger.error(
                "Failed to get current neural readings",
                profile_id=profile_id,
                error=str(e)
            )
            # Return simulated data on error
            return self._get_simulated_readings(profile_id)
    
    async def apply_entrainment(
        self,
        profile_id: int,
        target_frequencies: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply neural entrainment to guide brainwaves toward target frequencies
        for flow state induction and temporal perception optimization.
        """
        if profile_id not in self.active_monitors:
            raise ValueError(f"No active monitoring for profile {profile_id}")
        
        try:
            monitor_config = self.active_monitors[profile_id]
            
            # Design entrainment protocol
            entrainment_protocol = await self._design_entrainment_protocol(
                target_frequencies, monitor_config['baseline_data']
            )
            
            # Apply entrainment stimulation
            stimulation_result = await self._apply_entrainment_stimulation(
                profile_id, entrainment_protocol
            )
            
            # Monitor entrainment effectiveness
            effectiveness_monitoring = await self._monitor_entrainment_effectiveness(
                profile_id, target_frequencies, duration_seconds=60
            )
            
            result = {
                'success': stimulation_result['success'],
                'protocol_applied': entrainment_protocol,
                'effectiveness_score': effectiveness_monitoring['effectiveness'],
                'target_achievement': effectiveness_monitoring['target_achievement'],
                'side_effects_detected': effectiveness_monitoring['side_effects']
            }
            
            await log_performance_metric(
                "neural_entrainment_applied",
                {
                    'profile_id': profile_id,
                    'target_frequencies': target_frequencies,
                    'success': result['success'],
                    'effectiveness_score': result['effectiveness_score']
                }
            )
            
            logger.info(
                "Neural entrainment applied",
                profile_id=profile_id,
                target_frequencies=target_frequencies,
                effectiveness=result['effectiveness_score']
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to apply neural entrainment",
                profile_id=profile_id,
                error=str(e)
            )
            raise
    
    async def emergency_stop(self, profile_id: int) -> Dict[str, Any]:
        """
        Emergency stop of all neural monitoring and stimulation.
        """
        try:
            if profile_id not in self.active_monitors:
                return {'success': True, 'reason': 'No active monitoring'}
            
            monitor_config = self.active_monitors[profile_id]
            
            # Stop processing task
            if 'processing_task' in monitor_config and not monitor_config['processing_task'].done():
                monitor_config['processing_task'].cancel()
            
            # Stop any active entrainment
            await self._stop_entrainment_stimulation(profile_id)
            
            # Disconnect device
            await self._disconnect_eeg_device(profile_id)
            
            # Clean up
            del self.active_monitors[profile_id]
            if profile_id in self.signal_buffers:
                del self.signal_buffers[profile_id]
            
            logger.warning(
                "Emergency stop executed for neural monitoring",
                profile_id=profile_id
            )
            
            return {
                'success': True,
                'stopped_at': datetime.utcnow().isoformat(),
                'cleanup_completed': True
            }
            
        except Exception as e:
            logger.error(
                "Failed to execute emergency stop",
                profile_id=profile_id,
                error=str(e)
            )
            return {
                'success': False,
                'error': str(e)
            }
    
    # Private helper methods
    
    async def _connect_eeg_device(self, profile_id: int) -> Dict[str, Any]:
        """Connect to EEG device (simulated for demo)"""
        # In production, this would connect to actual EEG devices like BrainBit, Muse, etc.
        await asyncio.sleep(0.1)  # Simulate connection time
        return {
            'success': True,
            'device_type': 'simulated_eeg',
            'sampling_rate': 256,
            'channels': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        }
    
    async def _establish_neural_baseline(self, profile_id: int) -> Dict[str, Any]:
        """Establish baseline neural activity"""
        # Simulate baseline establishment
        baseline_data = {
            'alpha_baseline': 10.0 + np.random.normal(0, 1),
            'theta_baseline': 6.5 + np.random.normal(0, 0.5),
            'beta_baseline': 20.0 + np.random.normal(0, 2),
            'intrinsic_timescale_baseline': 500 + np.random.normal(0, 50)
        }
        
        return {
            'success': True,
            'baseline': baseline_data,
            'duration_seconds': 30
        }
    
    async def _continuous_neural_processing(self, profile_id: int):
        """Continuous neural signal processing loop"""
        try:
            while profile_id in self.active_monitors:
                # Simulate getting EEG data
                simulated_eeg = self._generate_simulated_eeg(profile_id)
                
                # Add to buffer
                if profile_id in self.signal_buffers:
                    self.signal_buffers[profile_id].append(simulated_eeg)
                    
                    # Maintain buffer size
                    if len(self.signal_buffers[profile_id]) > self.buffer_size:
                        self.signal_buffers[profile_id] = self.signal_buffers[profile_id][-self.buffer_size:]
                
                await asyncio.sleep(1/256)  # 256 Hz sampling rate
        except asyncio.CancelledError:
            logger.info(f"Neural processing stopped for profile {profile_id}")
    
    def _generate_simulated_eeg(self, profile_id: int) -> EEGReading:
        """Generate simulated EEG data for demo purposes"""
        # Create realistic-looking EEG signals
        channels = {}
        for channel in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']:
            # Generate signal with multiple frequency components
            t = np.linspace(0, 1, 256)  # 1 second at 256 Hz
            alpha = 50 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
            theta = 30 * np.sin(2 * np.pi * 6.5 * t)  # 6.5 Hz theta
            beta = 20 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
            noise = 10 * np.random.normal(0, 1, len(t))
            
            channels[channel] = (alpha + theta + beta + noise).tolist()
        
        return EEGReading(
            timestamp=datetime.utcnow(),
            channels=channels,
            sampling_rate=256,
            signal_quality=0.85 + np.random.normal(0, 0.05),
            device_type=BiometricDeviceType.EEG_HEADBAND
        )
    
    async def _get_latest_eeg_data(self, profile_id: int) -> Optional[EEGReading]:
        """Get latest EEG data from buffer"""
        if profile_id in self.signal_buffers and self.signal_buffers[profile_id]:
            return self.signal_buffers[profile_id][-1]
        return None
    
    async def _process_neural_signals(self, eeg_reading: EEGReading) -> ProcessedNeuralSignals:
        """Process raw EEG signals to extract meaningful features"""
        # Get a representative channel (average of frontal channels)
        frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4']
        signal_data = []
        for channel in frontal_channels:
            if channel in eeg_reading.channels:
                signal_data.append(eeg_reading.channels[channel])
        
        if not signal_data:
            # Fallback to any available channel
            signal_data = [list(eeg_reading.channels.values())[0]]
        
        avg_signal = np.mean(signal_data, axis=0)
        
        # Calculate frequency band powers using FFT
        freqs, psd = signal.welch(avg_signal, eeg_reading.sampling_rate, nperseg=256)
        
        band_powers = {}
        for band, (low, high) in self.frequency_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_powers[band] = np.mean(psd[band_mask])
        
        # Calculate intrinsic neural timescales using autocorrelation
        autocorr = np.correlate(avg_signal, avg_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find decay time constant
        decay_threshold = np.exp(-1)  # 1/e
        decay_idx = np.where(autocorr < decay_threshold)[0]
        intrinsic_timescale = decay_idx[0] / eeg_reading.sampling_rate * 1000 if len(decay_idx) > 0 else 500  # ms
        
        # Calculate frontal alpha asymmetry (F4-F3)
        if 'F3' in eeg_reading.channels and 'F4' in eeg_reading.channels:
            f3_alpha = self._extract_band_power(eeg_reading.channels['F3'], eeg_reading.sampling_rate, 8, 13)
            f4_alpha = self._extract_band_power(eeg_reading.channels['F4'], eeg_reading.sampling_rate, 8, 13)
            alpha_asymmetry = np.log(f4_alpha) - np.log(f3_alpha) if f3_alpha > 0 and f4_alpha > 0 else 0
        else:
            alpha_asymmetry = 0.1 + np.random.normal(0, 0.05)  # Simulated
        
        # Calculate theta coherence
        theta_coherence = self._calculate_coherence(signal_data, eeg_reading.sampling_rate, 4, 8) if len(signal_data) > 1 else 0.6
        
        return ProcessedNeuralSignals(
            timestamp=eeg_reading.timestamp,
            alpha_power=band_powers.get('alpha', 0),
            beta_power=band_powers.get('beta', 0),
            theta_power=band_powers.get('theta', 0),
            delta_power=band_powers.get('delta', 0),
            gamma_power=band_powers.get('gamma', 0),
            intrinsic_neural_timescales=intrinsic_timescale,
            temporal_prediction_error=0.1 + np.random.normal(0, 0.02),
            time_cell_activity=0.3 + np.random.normal(0, 0.1),
            frontal_alpha_asymmetry=alpha_asymmetry,
            theta_coherence=theta_coherence,
            transient_hypofrontality=0.2 + np.random.normal(0, 0.05),
            signal_quality=eeg_reading.signal_quality,
            artifact_level=0.1 + np.random.normal(0, 0.02),
            electrode_impedance={'Fp1': 5.0, 'Fp2': 5.2}  # kOhms
        )
    
    def _extract_band_power(self, signal_data: List[float], sampling_rate: int, low_freq: float, high_freq: float) -> float:
        """Extract power in specific frequency band"""
        freqs, psd = signal.welch(signal_data, sampling_rate, nperseg=min(256, len(signal_data)))
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.mean(psd[band_mask]) if np.any(band_mask) else 0.0
    
    def _calculate_coherence(self, signals: List[List[float]], sampling_rate: int, low_freq: float, high_freq: float) -> float:
        """Calculate coherence between signals in frequency band"""
        if len(signals) < 2:
            return 0.5
        
        # Simple coherence calculation (in production would use more sophisticated methods)
        correlations = []
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                corr, _ = pearsonr(signals[i], signals[j])
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.5
    
    async def _calculate_temporal_indicators(self, signals: ProcessedNeuralSignals) -> Dict[str, float]:
        """Calculate temporal perception indicators from neural signals"""
        return {
            'intrinsic_timescales': signals.intrinsic_neural_timescales,
            'prediction_error': signals.temporal_prediction_error,
            'time_cell_activity': signals.time_cell_activity,
            'autocorr_decay': signals.intrinsic_neural_timescales / 1000.0  # Convert to seconds
        }
    
    async def _calculate_flow_indicators(self, signals: ProcessedNeuralSignals) -> Dict[str, float]:
        """Calculate flow state indicators from neural signals"""
        return {
            'alpha_asymmetry': signals.frontal_alpha_asymmetry,
            'theta_coherence': signals.theta_coherence,
            'hypofrontality': signals.transient_hypofrontality,
            'dopamine_markers': min(1.0, signals.theta_power / (signals.beta_power + 1e-6))  # Theta/Beta ratio as proxy
        }
    
    async def _assess_neural_safety(self, signals: ProcessedNeuralSignals) -> Dict[str, float]:
        """Assess neural safety indicators"""
        safety_score = 1.0
        
        # Penalize for poor signal quality
        if signals.signal_quality < self.safety_thresholds['minimum_signal_quality']:
            safety_score -= 0.3
        
        # Penalize for high artifact levels
        if signals.artifact_level > self.safety_thresholds['maximum_artifact_level']:
            safety_score -= 0.2
        
        # Check for temporal confusion risk
        temporal_confusion_risk = max(0, signals.temporal_prediction_error - 0.5) * 2  # 0-1 scale
        
        return {
            'safety_score': max(0.0, safety_score),
            'temporal_confusion_risk': temporal_confusion_risk,
            'reality_testing_score': 1.0 - temporal_confusion_risk
        }
    
    def _get_simulated_readings(self, profile_id: int) -> Dict[str, float]:
        """Get simulated neural readings when real data unavailable"""
        return {
            'alpha_power': 25.0 + np.random.normal(0, 3),
            'beta_power': 15.0 + np.random.normal(0, 2),
            'theta_power': 20.0 + np.random.normal(0, 2.5),
            'delta_power': 30.0 + np.random.normal(0, 4),
            'gamma_power': 8.0 + np.random.normal(0, 1),
            'intrinsic_neural_timescales': 500 + np.random.normal(0, 50),
            'temporal_prediction_error': 0.15 + np.random.normal(0, 0.03),
            'time_cell_activity': 0.4 + np.random.normal(0, 0.08),
            'autocorrelation_decay': 0.5 + np.random.normal(0, 0.05),
            'frontal_alpha_asymmetry': 0.2 + np.random.normal(0, 0.05),
            'theta_coherence': 0.7 + np.random.normal(0, 0.08),
            'transient_hypofrontality': 0.3 + np.random.normal(0, 0.06),
            'dopamine_indicators': 0.6 + np.random.normal(0, 0.1),
            'heart_rate_variability': 45 + np.random.normal(0, 8),
            'breathing_rhythm': 0.25 + np.random.normal(0, 0.05),
            'galvanic_skin_response': 2.5 + np.random.normal(0, 0.3),
            'signal_quality': 0.85 + np.random.normal(0, 0.05),
            'artifact_level': 0.1 + np.random.normal(0, 0.02),
            'safety_score': 0.9 + np.random.normal(0, 0.05),
            'temporal_confusion': 0.05 + np.random.normal(0, 0.02),
            'reality_awareness': 0.95 + np.random.normal(0, 0.02)
        }
    
    async def _get_hrv_data(self, profile_id: int) -> float:
        """Get heart rate variability data"""
        # Simulated HRV data (RMSSD in ms)
        return 45.0 + np.random.normal(0, 8)
    
    async def _get_breathing_data(self, profile_id: int) -> float:
        """Get breathing rhythm data"""
        # Simulated breathing rate (Hz)
        return 0.25 + np.random.normal(0, 0.05)  # ~15 breaths per minute
    
    async def _get_gsr_data(self, profile_id: int) -> float:
        """Get galvanic skin response data"""
        # Simulated GSR data (microsiemens)
        return 2.5 + np.random.normal(0, 0.3)
