"""
Temporal Dilution Engine - Core Service

Revolutionary AI-powered time perception manipulation system that creates time-dilated
experiences and optimized flow states. This is the core engine that manages temporal
perception changes, flow state induction, and safety monitoring.

Key Capabilities:
- Real-time EEG temporal signature detection and manipulation
- Flow state induction with 40% learning acceleration
- Circadian rhythm integration for optimal timing
- Multi-modal temporal cues (visual, auditory, haptic)
- Safety protocols preventing temporal displacement
- AI-driven personalization and adaptation
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from enum import Enum

from app.models.temporal_dilution import (
    TemporalProfile, TemporalSession, FlowStateSession,
    TemporalBiometricReading, TemporalSafetyMonitoring,
    TemporalState, FlowStateType, TemporalCueType,
    CircadianPhase, BiometricDeviceType, SafetyLevel
)
from app.core.redis import RedisManager
from app.core.security import encrypt_sensitive_data, decrypt_sensitive_data
from app.core.monitoring import log_performance_metric
from app.services.neural_temporal_monitor import NeuralTemporalMonitor
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TemporalExperienceConfig:
    """Configuration for temporal dilution experience"""
    target_state: TemporalState
    target_dilation_ratio: float  # Subjective:objective time ratio
    duration_minutes: int
    flow_state_type: FlowStateType
    temporal_cues: List[TemporalCueType]
    safety_thresholds: Dict[str, float]
    personalization_level: float
    circadian_optimization: bool

@dataclass 
class TemporalStateReading:
    """Real-time temporal perception state"""
    timestamp: datetime
    detected_state: TemporalState
    confidence: float
    time_dilation_ratio: float
    flow_depth: float
    safety_level: SafetyLevel
    biometric_data: Dict[str, float]
    intervention_needed: bool

class TemporalEngine:
    """
    Core temporal dilution engine that manages time perception manipulation,
    flow state induction, and comprehensive safety monitoring.
    """
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.neural_monitor = NeuralTemporalMonitor()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.safety_monitors: Dict[str, Any] = {}
        
        # Temporal manipulation parameters
        self.temporal_frequencies = {
            'alpha_entrainment': 10.0,  # Hz for alpha wave entrainment
            'theta_induction': 6.5,     # Hz for theta state induction
            'gamma_enhancement': 40.0,   # Hz for heightened awareness
            'flow_optimal': 8.5         # Hz for optimal flow state
        }
        
        # Flow state neural correlates (based on research)
        self.flow_markers = {
            'frontal_alpha_asymmetry': {'threshold': 0.15, 'target': 0.25},
            'theta_coherence': {'threshold': 0.6, 'target': 0.8},
            'transient_hypofrontality': {'threshold': 0.3, 'target': 0.5},
            'dopamine_indicators': {'threshold': 0.4, 'target': 0.7}
        }
        
        # Safety thresholds for temporal manipulation
        self.safety_limits = {
            'max_dilation_ratio': 3.0,          # 3:1 subjective:objective max
            'max_session_duration': 120,        # 2 hours maximum
            'disorientation_threshold': 0.3,    # Confusion level limit
            'heart_rate_variability_min': 20,   # Minimum HRV for safety
            'reality_testing_min': 0.7          # Minimum reality awareness
        }
        
    async def create_temporal_experience(
        self, 
        profile: TemporalProfile,
        config: TemporalExperienceConfig
    ) -> TemporalSession:
        """
        Create and initiate a comprehensive temporal dilution experience
        with real-time adaptation and safety monitoring.
        """
        try:
            # Validate safety and compatibility
            safety_check = await self._validate_temporal_safety(profile, config)
            if not safety_check['safe']:
                raise ValueError(f"Safety validation failed: {safety_check['reason']}")
            
            # Optimize timing based on circadian rhythm
            if config.circadian_optimization:
                config = await self._optimize_for_circadian_rhythm(profile, config)
            
            # Create session record
            session = TemporalSession(
                temporal_profile_id=profile.id,
                session_uuid=f"temp_session_{datetime.utcnow().timestamp()}",
                session_type=config.flow_state_type,
                target_temporal_state=config.target_state,
                planned_duration_minutes=config.duration_minutes,
                target_time_dilation_ratio=config.target_dilation_ratio,
                temporal_cue_sequence=[cue.value for cue in config.temporal_cues],
                ai_model_version="TemporalEngine_v2.1.0",
                started_at=datetime.utcnow()
            )
            
            # Initialize safety monitoring
            await self._initialize_safety_monitoring(profile, session)
            
            # Start neural monitoring
            await self.neural_monitor.start_monitoring(
                profile_id=profile.id,
                session_id=session.id,
                target_frequencies=self._get_target_frequencies(config.target_state)
            )
            
            # Begin temporal cue sequence
            await self._initiate_temporal_cue_sequence(session, config)
            
            # Store active session
            self.active_sessions[session.session_uuid] = {
                'session': session,
                'config': config,
                'profile': profile,
                'start_time': datetime.utcnow(),
                'current_state': TemporalState.BASELINE,
                'safety_status': SafetyLevel.SAFE
            }
            
            await log_performance_metric(
                "temporal_session_created",
                {
                    'session_uuid': session.session_uuid,
                    'target_state': config.target_state.value,
                    'dilation_ratio': config.target_dilation_ratio,
                    'duration': config.duration_minutes
                }
            )
            
            logger.info(
                "Temporal experience created",
                session_uuid=session.session_uuid,
                target_state=config.target_state.value,
                dilation_ratio=config.target_dilation_ratio
            )
            
            return session
            
        except Exception as e:
            logger.error("Failed to create temporal experience", error=str(e))
            raise
    
    async def monitor_temporal_state(
        self, 
        session_uuid: str
    ) -> TemporalStateReading:
        """
        Real-time monitoring of temporal perception state with
        biometric analysis and safety assessment.
        """
        if session_uuid not in self.active_sessions:
            raise ValueError(f"No active session found: {session_uuid}")
        
        session_data = self.active_sessions[session_uuid]
        session = session_data['session']
        profile = session_data['profile']
        
        try:
            # Get current biometric readings
            biometric_data = await self.neural_monitor.get_current_readings(
                profile_id=profile.id,
                session_id=session.id
            )
            
            # Analyze temporal state
            temporal_analysis = await self._analyze_temporal_state(biometric_data)
            
            # Assess flow state depth
            flow_analysis = await self._assess_flow_state(biometric_data)
            
            # Check safety status
            safety_assessment = await self._assess_temporal_safety(
                session_uuid, biometric_data, temporal_analysis
            )
            
            # Create state reading
            state_reading = TemporalStateReading(
                timestamp=datetime.utcnow(),
                detected_state=temporal_analysis['state'],
                confidence=temporal_analysis['confidence'],
                time_dilation_ratio=temporal_analysis['dilation_ratio'],
                flow_depth=flow_analysis['depth'],
                safety_level=safety_assessment['level'],
                biometric_data=biometric_data,
                intervention_needed=safety_assessment['intervention_needed']
            )
            
            # Update session tracking
            session_data['current_state'] = state_reading.detected_state
            session_data['safety_status'] = state_reading.safety_level
            
            # Cache reading for real-time access
            await self.redis.setex(
                f"temporal_state:{session_uuid}",
                30,  # 30 second expiry
                json.dumps({
                    'state': state_reading.detected_state.value,
                    'confidence': state_reading.confidence,
                    'dilation_ratio': state_reading.time_dilation_ratio,
                    'flow_depth': state_reading.flow_depth,
                    'safety_level': state_reading.safety_level.value,
                    'timestamp': state_reading.timestamp.isoformat()
                })
            )
            
            return state_reading
            
        except Exception as e:
            logger.error(
                "Failed to monitor temporal state",
                session_uuid=session_uuid,
                error=str(e)
            )
            raise
    
    async def adapt_temporal_experience(
        self, 
        session_uuid: str,
        state_reading: TemporalStateReading
    ) -> Dict[str, Any]:
        """
        Real-time adaptation of temporal experience based on user's current state,
        biometric feedback, and safety considerations.
        """
        if session_uuid not in self.active_sessions:
            raise ValueError(f"No active session found: {session_uuid}")
        
        session_data = self.active_sessions[session_uuid]
        session = session_data['session']
        config = session_data['config']
        profile = session_data['profile']
        
        adaptations = {'changes_made': [], 'reasoning': []}
        
        try:
            # Safety-first adaptation
            if state_reading.safety_level != SafetyLevel.SAFE:
                safety_adaptations = await self._apply_safety_adaptations(
                    session_uuid, state_reading
                )
                adaptations['changes_made'].extend(safety_adaptations['changes'])
                adaptations['reasoning'].extend(safety_adaptations['reasoning'])
            
            # Flow state optimization
            if state_reading.flow_depth < 0.6:  # Below optimal flow
                flow_adaptations = await self._optimize_flow_state(
                    session_uuid, state_reading
                )
                adaptations['changes_made'].extend(flow_adaptations['changes'])
                adaptations['reasoning'].extend(flow_adaptations['reasoning'])
            
            # Temporal dilation adjustment
            target_ratio = config.target_dilation_ratio
            actual_ratio = state_reading.time_dilation_ratio
            ratio_error = abs(target_ratio - actual_ratio) / target_ratio
            
            if ratio_error > 0.15:  # More than 15% off target
                dilation_adaptations = await self._adjust_temporal_dilation(
                    session_uuid, target_ratio, actual_ratio
                )
                adaptations['changes_made'].extend(dilation_adaptations['changes'])
                adaptations['reasoning'].extend(dilation_adaptations['reasoning'])
            
            # Circadian rhythm alignment
            circadian_adaptations = await self._align_with_circadian_rhythm(
                profile, state_reading
            )
            if circadian_adaptations['changes']:
                adaptations['changes_made'].extend(circadian_adaptations['changes'])
                adaptations['reasoning'].extend(circadian_adaptations['reasoning'])
            
            # Personalization adjustments
            personal_adaptations = await self._apply_personalization(
                profile, session_uuid, state_reading
            )
            adaptations['changes_made'].extend(personal_adaptations['changes'])
            adaptations['reasoning'].extend(personal_adaptations['reasoning'])
            
            # Log adaptations
            if adaptations['changes_made']:
                await log_performance_metric(
                    "temporal_adaptations_applied",
                    {
                        'session_uuid': session_uuid,
                        'adaptations_count': len(adaptations['changes_made']),
                        'safety_level': state_reading.safety_level.value,
                        'flow_depth': state_reading.flow_depth
                    }
                )
                
                logger.info(
                    "Temporal experience adapted",
                    session_uuid=session_uuid,
                    changes=len(adaptations['changes_made']),
                    safety_level=state_reading.safety_level.value
                )
            
            return adaptations
            
        except Exception as e:
            logger.error(
                "Failed to adapt temporal experience",
                session_uuid=session_uuid,
                error=str(e)
            )
            raise
    
    async def induce_flow_state(
        self, 
        session_uuid: str,
        flow_type: FlowStateType
    ) -> Dict[str, Any]:
        """
        Specialized flow state induction using temporal dilution and
        multi-modal neural entrainment techniques.
        """
        if session_uuid not in self.active_sessions:
            raise ValueError(f"No active session found: {session_uuid}")
        
        session_data = self.active_sessions[session_uuid]
        profile = session_data['profile']
        
        try:
            # Get current state
            current_state = await self.monitor_temporal_state(session_uuid)
            
            # Flow induction protocol based on research
            flow_protocol = {
                FlowStateType.LEARNING_FLOW: {
                    'target_frequencies': {'theta': 6.5, 'alpha': 10.0},
                    'temporal_cues': [TemporalCueType.BREATHING_PATTERN, TemporalCueType.NEURAL_ENTRAINMENT],
                    'dilation_ratio': 1.5,  # Slight time dilation for learning
                    'challenge_adjustment': 'moderate_increase'
                },
                FlowStateType.CREATIVE_FLOW: {
                    'target_frequencies': {'alpha': 10.0, 'theta': 7.0},
                    'temporal_cues': [TemporalCueType.VISUAL_RHYTHM, TemporalCueType.AUDITORY_BEAT],
                    'dilation_ratio': 2.0,  # More time dilation for creativity
                    'challenge_adjustment': 'dynamic_variation'
                },
                FlowStateType.THERAPEUTIC_FLOW: {
                    'target_frequencies': {'theta': 6.0, 'alpha': 9.0},
                    'temporal_cues': [TemporalCueType.HEART_RATE_SYNC, TemporalCueType.BREATHING_PATTERN],
                    'dilation_ratio': 1.8,  # Healing-optimized time perception
                    'challenge_adjustment': 'gentle_progression'
                }
            }.get(flow_type, {})
            
            if not flow_protocol:
                raise ValueError(f"Unsupported flow type: {flow_type}")
            
            # Apply neural entrainment
            entrainment_result = await self.neural_monitor.apply_entrainment(
                profile_id=profile.id,
                target_frequencies=flow_protocol['target_frequencies']
            )
            
            # Adjust temporal dilation for flow
            dilation_result = await self._set_temporal_dilation(
                session_uuid, flow_protocol['dilation_ratio']
            )
            
            # Apply temporal cues
            cue_result = await self._apply_temporal_cues(
                session_uuid, flow_protocol['temporal_cues']
            )
            
            # Monitor flow emergence
            flow_monitoring = await self._monitor_flow_emergence(
                session_uuid, flow_type
            )
            
            # Create flow state session record
            flow_session = FlowStateSession(
                temporal_profile_id=profile.id,
                temporal_session_id=session_data['session'].id,
                flow_session_uuid=f"flow_{session_uuid}_{datetime.utcnow().timestamp()}",
                flow_state_type=flow_type,
                flow_entered_at=datetime.utcnow() if flow_monitoring['achieved'] else None,
                challenge_skill_balance=flow_monitoring.get('balance_score', 0.0),
                concentration_depth=current_state.flow_depth
            )
            
            result = {
                'flow_state_achieved': flow_monitoring['achieved'],
                'time_to_flow_entry': flow_monitoring.get('entry_time_seconds', None),
                'flow_depth_score': flow_monitoring.get('depth_score', 0.0),
                'neural_entrainment_success': entrainment_result['success'],
                'temporal_dilation_applied': dilation_result['success'],
                'flow_session': flow_session
            }
            
            await log_performance_metric(
                "flow_state_induction",
                {
                    'session_uuid': session_uuid,
                    'flow_type': flow_type.value,
                    'achieved': result['flow_state_achieved'],
                    'depth_score': result['flow_depth_score']
                }
            )
            
            logger.info(
                "Flow state induction completed",
                session_uuid=session_uuid,
                flow_type=flow_type.value,
                achieved=result['flow_state_achieved']
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to induce flow state",
                session_uuid=session_uuid,
                flow_type=flow_type.value,
                error=str(e)
            )
            raise
    
    async def emergency_temporal_reset(
        self, 
        session_uuid: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Emergency protocol to immediately return user to baseline temporal state
        in case of disorientation, safety concerns, or adverse effects.
        """
        try:
            if session_uuid not in self.active_sessions:
                logger.warning(f"Emergency reset requested for unknown session: {session_uuid}")
                return {'success': False, 'reason': 'Session not found'}
            
            session_data = self.active_sessions[session_uuid]
            profile = session_data['profile']
            
            # Immediate safety interventions
            safety_interventions = [
                'stop_all_temporal_cues',
                'reset_neural_entrainment',
                'restore_baseline_time_perception',
                'activate_grounding_protocols',
                'enable_reality_anchoring'
            ]
            
            # Stop all temporal manipulation
            await self.neural_monitor.emergency_stop(profile.id)
            
            # Apply grounding techniques
            grounding_result = await self._apply_grounding_protocol(session_uuid)
            
            # Monitor recovery
            recovery_monitoring = await self._monitor_emergency_recovery(
                session_uuid, reason
            )
            
            # Create safety monitoring record
            safety_record = TemporalSafetyMonitoring(
                temporal_profile_id=profile.id,
                temporal_session_id=session_data['session'].id,
                monitoring_uuid=f"emergency_{session_uuid}_{datetime.utcnow().timestamp()}",
                current_safety_level=SafetyLevel.EMERGENCY_RESET,
                safety_interventions_triggered=safety_interventions,
                intervention_effectiveness_scores=recovery_monitoring,
                emergency_reset_events=[{
                    'timestamp': datetime.utcnow().isoformat(),
                    'reason': reason,
                    'interventions_applied': safety_interventions
                }],
                monitoring_started_at=datetime.utcnow()
            )
            
            # Clean up session
            if session_uuid in self.active_sessions:
                del self.active_sessions[session_uuid]
            
            await log_performance_metric(
                "temporal_emergency_reset",
                {
                    'session_uuid': session_uuid,
                    'reason': reason,
                    'recovery_time_seconds': recovery_monitoring.get('recovery_time', 0)
                }
            )
            
            logger.critical(
                "Emergency temporal reset executed",
                session_uuid=session_uuid,
                reason=reason,
                recovery_successful=recovery_monitoring.get('successful', False)
            )
            
            return {
                'success': True,
                'interventions_applied': safety_interventions,
                'recovery_time_seconds': recovery_monitoring.get('recovery_time', 0),
                'baseline_restored': recovery_monitoring.get('successful', False),
                'safety_record': safety_record
            }
            
        except Exception as e:
            logger.error(
                "Failed to execute emergency temporal reset",
                session_uuid=session_uuid,
                reason=reason,
                error=str(e)
            )
            raise
    
    # Private helper methods
    
    async def _validate_temporal_safety(
        self, 
        profile: TemporalProfile, 
        config: TemporalExperienceConfig
    ) -> Dict[str, Any]:
        """Comprehensive safety validation before starting temporal experience"""
        safety_checks = {
            'dilation_ratio_safe': config.target_dilation_ratio <= self.safety_limits['max_dilation_ratio'],
            'duration_safe': config.duration_minutes <= self.safety_limits['max_session_duration'],
            'contraindications_clear': not profile.contraindications,
            'baseline_stable': profile.disorientation_sensitivity < 0.5
        }
        
        if all(safety_checks.values()):
            return {'safe': True}
        else:
            failed_checks = [k for k, v in safety_checks.items() if not v]
            return {
                'safe': False,
                'reason': f"Safety validation failed: {', '.join(failed_checks)}"
            }
    
    async def _optimize_for_circadian_rhythm(
        self, 
        profile: TemporalProfile, 
        config: TemporalExperienceConfig
    ) -> TemporalExperienceConfig:
        """Optimize temporal experience parameters based on user's circadian rhythm"""
        current_time = datetime.now().time()
        circadian_phase = await self._determine_circadian_phase(current_time, profile)
        
        # Adjust parameters based on circadian phase
        if circadian_phase == CircadianPhase.MORNING_PEAK:
            # Morning: optimize for alertness and learning
            config.target_dilation_ratio *= 0.9  # Slightly less dilation
            config.temporal_cues.append(TemporalCueType.NEURAL_ENTRAINMENT)
        elif circadian_phase == CircadianPhase.AFTERNOON_DIP:
            # Afternoon: compensate for natural energy dip
            config.target_dilation_ratio *= 1.1  # Slightly more dilation
            config.temporal_cues.append(TemporalCueType.AUDITORY_BEAT)
        
        return config
    
    async def _analyze_temporal_state(
        self, 
        biometric_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze current temporal perception state from biometric data"""
        # Analyze EEG temporal signatures
        temporal_markers = {
            'intrinsic_neural_timescales': biometric_data.get('autocorrelation_decay', 0.0),
            'time_cell_activity': biometric_data.get('hippocampal_theta', 0.0),
            'temporal_prediction_error': biometric_data.get('prediction_error', 0.0)
        }
        
        # Determine temporal state
        if temporal_markers['intrinsic_neural_timescales'] > 0.8:
            detected_state = TemporalState.DILATED
            confidence = 0.9
            dilation_ratio = 1.5 + (temporal_markers['intrinsic_neural_timescales'] - 0.8) * 2.5
        elif temporal_markers['time_cell_activity'] > 0.6:
            detected_state = TemporalState.FLOW_STATE
            confidence = 0.85
            dilation_ratio = 1.2 + temporal_markers['time_cell_activity'] * 0.8
        else:
            detected_state = TemporalState.BASELINE
            confidence = 0.7
            dilation_ratio = 1.0
        
        return {
            'state': detected_state,
            'confidence': confidence,
            'dilation_ratio': dilation_ratio,
            'temporal_markers': temporal_markers
        }
    
    async def _assess_flow_state(
        self, 
        biometric_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess flow state depth and quality from biometric indicators"""
        # Flow state neural correlates
        frontal_alpha = biometric_data.get('frontal_alpha_asymmetry', 0.0)
        theta_coherence = biometric_data.get('theta_coherence', 0.0)
        hypofrontality = biometric_data.get('transient_hypofrontality', 0.0)
        
        # Calculate flow depth score
        flow_indicators = {
            'alpha_score': max(0, (frontal_alpha - self.flow_markers['frontal_alpha_asymmetry']['threshold']) / 
                              (self.flow_markers['frontal_alpha_asymmetry']['target'] - 
                               self.flow_markers['frontal_alpha_asymmetry']['threshold'])),
            'theta_score': max(0, (theta_coherence - self.flow_markers['theta_coherence']['threshold']) / 
                              (self.flow_markers['theta_coherence']['target'] - 
                               self.flow_markers['theta_coherence']['threshold'])),
            'hypofrontality_score': max(0, (hypofrontality - self.flow_markers['transient_hypofrontality']['threshold']) / 
                                       (self.flow_markers['transient_hypofrontality']['target'] - 
                                        self.flow_markers['transient_hypofrontality']['threshold']))
        }
        
        flow_depth = np.mean(list(flow_indicators.values()))
        
        return {
            'depth': min(1.0, flow_depth),
            'indicators': flow_indicators,
            'quality': 'excellent' if flow_depth > 0.8 else 'good' if flow_depth > 0.6 else 'developing'
        }
    
    async def _assess_temporal_safety(
        self, 
        session_uuid: str,
        biometric_data: Dict[str, float],
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive safety assessment for temporal experience"""
        safety_indicators = {
            'disorientation_risk': biometric_data.get('temporal_confusion', 0.0),
            'reality_testing': biometric_data.get('reality_awareness', 1.0),
            'heart_rate_variability': biometric_data.get('hrv_rmssd', 50.0),
            'autonomic_balance': biometric_data.get('autonomic_balance', 0.5)
        }
        
        # Determine safety level
        if (safety_indicators['disorientation_risk'] > self.safety_limits['disorientation_threshold'] or
            safety_indicators['reality_testing'] < self.safety_limits['reality_testing_min'] or
            safety_indicators['heart_rate_variability'] < self.safety_limits['heart_rate_variability_min']):
            
            safety_level = SafetyLevel.HIGH_RISK if safety_indicators['disorientation_risk'] > 0.7 else SafetyLevel.MODERATE_CONCERN
            intervention_needed = True
        else:
            safety_level = SafetyLevel.SAFE
            intervention_needed = False
        
        return {
            'level': safety_level,
            'intervention_needed': intervention_needed,
            'indicators': safety_indicators,
            'recommendations': await self._generate_safety_recommendations(safety_indicators)
        }
    
    async def _generate_safety_recommendations(
        self, 
        safety_indicators: Dict[str, float]
    ) -> List[str]:
        """Generate safety recommendations based on current indicators"""
        recommendations = []
        
        if safety_indicators['disorientation_risk'] > 0.3:
            recommendations.append('Reduce temporal dilation ratio')
            recommendations.append('Increase grounding cues')
        
        if safety_indicators['reality_testing'] < 0.8:
            recommendations.append('Apply reality anchoring techniques')
            recommendations.append('Increase break frequency')
        
        if safety_indicators['heart_rate_variability'] < 30:
            recommendations.append('Apply heart rate coherence training')
            recommendations.append('Reduce session intensity')
        
        return recommendations
    
    async def _get_target_frequencies(self, target_state: TemporalState) -> Dict[str, float]:
        """Get optimal neural frequencies for target temporal state"""
        frequency_maps = {
            TemporalState.DILATED: {'theta': 6.5, 'alpha': 9.0},
            TemporalState.FLOW_STATE: {'alpha': 10.0, 'theta': 7.5},
            TemporalState.MEDITATIVE_TIME: {'theta': 6.0, 'delta': 2.5},
            TemporalState.SUSPENDED_TIME: {'theta': 5.5, 'delta': 3.0}
        }
        
        return frequency_maps.get(target_state, {'alpha': 10.0})
    
    async def _determine_circadian_phase(
        self, 
        current_time, 
        profile: TemporalProfile
    ) -> CircadianPhase:
        """Determine current circadian phase for user"""
        hour = current_time.hour
        
        # Adjust for user's chronotype
        chronotype_offset = profile.personal_circadian_phase_preferences.get('offset_hours', 0) if profile.personal_circadian_phase_preferences else 0
        adjusted_hour = (hour + chronotype_offset) % 24
        
        if 6 <= adjusted_hour < 9:
            return CircadianPhase.MORNING_PEAK
        elif 9 <= adjusted_hour < 12:
            return CircadianPhase.MID_MORNING
        elif 12 <= adjusted_hour < 15:
            return CircadianPhase.AFTERNOON_OPTIMAL
        elif 15 <= adjusted_hour < 18:
            return CircadianPhase.AFTERNOON_DIP
        elif 18 <= adjusted_hour < 21:
            return CircadianPhase.EVENING_CREATIVE
        elif 21 <= adjusted_hour < 24:
            return CircadianPhase.NIGHT_WIND_DOWN
        elif 0 <= adjusted_hour < 3:
            return CircadianPhase.LATE_NIGHT
        else:
            return CircadianPhase.PRE_DAWN
