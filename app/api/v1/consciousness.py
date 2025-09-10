"""
Consciousness Mirroring API Endpoints

Revolutionary API for interacting with cognitive twins, future selves,
and consciousness mirroring technology.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Body, Query, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog
import jwt

from app.database import get_db_session
from app.models.user import User
from app.core.security_utils import RateLimiter, ConsentManager
from app.config.settings import get_settings
from app.core.security import verify_jwt_token
from app.models.consciousness import (
    CognitiveProfile, ConsciousnessSession, 
    DecisionHistory, PersonalityEvolution,
    KeystrokePattern, MirrorCalibration
)
from app.services.consciousness_mirror import get_consciousness_mirror
from app.schemas.consciousness import (
    CognitiveProfileResponse,
    ConsciousnessUpdateRequest,
    PredictResponseRequest,
    FutureSimulationRequest,
    TwinConversationRequest,
    DecisionPredictionRequest,
    KeystrokeData,
    CalibrationFeedback
)

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()
rate_limiter = RateLimiter()
consent_manager = ConsentManager()


# JWT verification is now handled by the centralized security module


@router.get("/profile/{user_id}", response_model=CognitiveProfileResponse)
async def get_cognitive_profile(
    user_id: UUID,
    authenticated_user_id: UUID = Depends(verify_jwt_token),
    db: AsyncSession = Depends(get_db_session)
) -> CognitiveProfileResponse:
    """
    Get the complete cognitive profile of a user's digital twin.
    
    This includes personality metrics, mirror accuracy, and thinking patterns.
    """
    try:
        # Get or create cognitive profile
        query = select(CognitiveProfile).where(CognitiveProfile.user_id == user_id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()
        
        if not profile:
            # Create new profile
            profile = CognitiveProfile(user_id=user_id)
            db.add(profile)
            await db.commit()
            await db.refresh(profile)
        
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        return CognitiveProfileResponse(
            user_id=user_id,
            personality={
                'openness': profile.openness,
                'conscientiousness': profile.conscientiousness,
                'extraversion': profile.extraversion,
                'agreeableness': profile.agreeableness,
                'neuroticism': profile.neuroticism
            },
            mirror_accuracy=profile.mirror_accuracy,
            thought_velocity=profile.thought_velocity,
            creativity_index=profile.creativity_index,
            prediction_confidence=profile.prediction_confidence,
            linguistic_fingerprint=profile.linguistic_fingerprint,
            cognitive_biases=profile.cognitive_biases,
            emotional_baseline=profile.emotional_baseline,
            last_sync=profile.last_sync,
            sync_message_count=profile.sync_message_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get cognitive profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cognitive profile")


@router.post("/update/{user_id}")
async def update_consciousness(
    user_id: UUID,
    request: ConsciousnessUpdateRequest,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Update the consciousness mirror with new user data.
    
    This processes messages, keystroke patterns, and behavioral data
    to improve the accuracy of the cognitive twin.
    """
    try:
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        # Process the message with optional keystroke data
        keystroke_data = None
        if request.keystroke_data:
            keystroke_data = request.keystroke_data.dict()
            
            # Store keystroke pattern in database
            pattern = KeystrokePattern(
                user_id=user_id,
                session_id=request.session_id,
                avg_dwell_time=sum(keystroke_data.get('dwell_times', [])) / len(keystroke_data.get('dwell_times', [1])),
                avg_flight_time=sum(keystroke_data.get('flight_times', [])) / len(keystroke_data.get('flight_times', [1])),
                typing_speed=keystroke_data.get('typing_speed', 0),
                deletion_rate=keystroke_data.get('deletion_rate', 0),
                raw_patterns=keystroke_data
            )
            db.add(pattern)
        
        # Process message and update profile
        analysis = await mirror.process_message(request.message, keystroke_data)
        
        # Update database profile
        query = select(CognitiveProfile).where(CognitiveProfile.user_id == user_id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()
        
        if profile:
            # Update personality vector
            profile.set_personality_vector(mirror.cognitive_profile.personality_vector)
            profile.mirror_accuracy = mirror.cognitive_profile.mirror_accuracy
            profile.thought_velocity = mirror.cognitive_profile.thought_velocity
            profile.linguistic_fingerprint = {
                k: list(v) if isinstance(v, list) else v 
                for k, v in mirror.cognitive_profile.linguistic_fingerprint.items()
            }
            profile.last_sync = datetime.utcnow()
            profile.sync_message_count += 1
            
            # Add personality snapshot
            profile.add_personality_snapshot()
            
            await db.commit()
        
        return {
            'status': 'updated',
            'analysis': analysis,
            'mirror_accuracy': mirror.cognitive_profile.mirror_accuracy,
            'messages_processed': profile.sync_message_count if profile else 1
        }
        
    except Exception as e:
        logger.error(f"Failed to update consciousness: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update consciousness")


@router.post("/predict/{user_id}")
async def predict_response(
    user_id: UUID,
    request: PredictResponseRequest,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Predict what the user would say in response to a given context.
    
    The cognitive twin analyzes the context and generates a response
    that matches the user's personality and communication style.
    """
    try:
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        # Predict response
        predicted_response, confidence = await mirror.predict_response(request.context)
        
        # Log prediction for future calibration
        if request.log_prediction:
            session_query = select(ConsciousnessSession).where(
                and_(
                    ConsciousnessSession.user_id == user_id,
                    ConsciousnessSession.is_active == True,
                    ConsciousnessSession.session_type == 'prediction'
                )
            )
            result = await db.execute(session_query)
            session = result.scalar_one_or_none()
            
            if not session:
                session = ConsciousnessSession(
                    user_id=user_id,
                    session_type='prediction'
                )
                db.add(session)
            
            session.add_exchange(request.context, predicted_response, confidence)
            await db.commit()
        
        return {
            'predicted_response': predicted_response,
            'confidence': confidence,
            'personality_influence': {
                'openness': mirror.cognitive_profile.personality_vector[0],
                'conscientiousness': mirror.cognitive_profile.personality_vector[1],
                'extraversion': mirror.cognitive_profile.personality_vector[2],
                'agreeableness': mirror.cognitive_profile.personality_vector[3],
                'neuroticism': mirror.cognitive_profile.personality_vector[4]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to predict response: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict response")


@router.post("/future-self/{user_id}")
async def simulate_future_self(
    user_id: UUID,
    request: FutureSimulationRequest,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Create a simulation of the user's future self.
    
    Based on personality trajectory analysis, this creates a version
    of the user's consciousness from the specified number of years ahead.
    """
    try:
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        # Create future self simulation
        future_mirror = await mirror.simulate_future_self(request.years_ahead)
        
        # Create session for future self conversation
        session = ConsciousnessSession(
            user_id=user_id,
            session_type='future_self',
            twin_config={'years_ahead': request.years_ahead}
        )
        db.add(session)
        await db.commit()
        
        # Generate initial message from future self
        future_message = f"Hello from {request.years_ahead} years in the future. "
        
        # Personality-based future message
        future_personality = future_mirror.cognitive_profile.personality_vector
        current_personality = mirror.cognitive_profile.personality_vector
        
        if future_personality[1] > current_personality[1]:  # More conscientious
            future_message += "I've become more organized and disciplined over the years. "
        
        if future_personality[4] < current_personality[4]:  # Less neurotic
            future_message += "I'm much calmer now and worry less about things. "
        
        if future_personality[0] < current_personality[0]:  # Less open
            future_message += "I've settled into my ways a bit more. "
        
        future_message += "What would you like to know about your future?"
        
        return {
            'session_id': session.id,
            'future_personality': {
                'openness': future_personality[0],
                'conscientiousness': future_personality[1],
                'extraversion': future_personality[2],
                'agreeableness': future_personality[3],
                'neuroticism': future_personality[4]
            },
            'personality_changes': {
                'openness_change': future_personality[0] - current_personality[0],
                'conscientiousness_change': future_personality[1] - current_personality[1],
                'extraversion_change': future_personality[2] - current_personality[2],
                'agreeableness_change': future_personality[3] - current_personality[3],
                'neuroticism_change': future_personality[4] - current_personality[4]
            },
            'future_message': future_message,
            'thought_velocity': future_mirror.cognitive_profile.thought_velocity
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate future self: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to simulate future self")


@router.post("/twin-chat/{user_id}")
async def chat_with_twin(
    user_id: UUID,
    request: TwinConversationRequest,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Have a conversation with your cognitive twin.
    
    The twin processes messages as if it were you, allowing you
    to have a conversation with your digital consciousness.
    """
    try:
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        # Get or create twin chat session
        if request.session_id:
            session_query = select(ConsciousnessSession).where(
                ConsciousnessSession.id == request.session_id
            )
            result = await db.execute(session_query)
            session = result.scalar_one_or_none()
        else:
            session = ConsciousnessSession(
                user_id=user_id,
                session_type='twin_chat'
            )
            db.add(session)
        
        # Get twin's response
        twin_response = await mirror.converse_with_twin(request.message)
        
        # Log the exchange
        if session:
            session.add_exchange(request.message, twin_response, mirror.cognitive_profile.mirror_accuracy)
            await db.commit()
        
        return {
            'session_id': session.id if session else None,
            'twin_response': twin_response,
            'mirror_accuracy': mirror.cognitive_profile.mirror_accuracy,
            'personality_state': {
                'openness': mirror.cognitive_profile.personality_vector[0],
                'conscientiousness': mirror.cognitive_profile.personality_vector[1],
                'extraversion': mirror.cognitive_profile.personality_vector[2],
                'agreeableness': mirror.cognitive_profile.personality_vector[3],
                'neuroticism': mirror.cognitive_profile.personality_vector[4]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to chat with twin: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to chat with twin")


@router.post("/predict-decision/{user_id}")
async def predict_decision(
    user_id: UUID,
    request: DecisionPredictionRequest,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Predict what decision the user would make in a given context.
    
    Based on historical decision patterns and personality profile.
    """
    try:
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        # Predict decision
        predicted_choice, confidence = mirror.decision_tree.predict_decision(request.context)
        
        # Store prediction for later validation
        decision_record = DecisionHistory(
            user_id=user_id,
            context_hash=mirror.decision_tree._hash_context(request.context),
            context_data=request.context,
            predicted_choice=predicted_choice,
            prediction_confidence=confidence,
            alternatives=request.alternatives
        )
        db.add(decision_record)
        await db.commit()
        
        # Generate reasoning based on personality
        personality = mirror.cognitive_profile.personality_vector
        reasoning = []
        
        if personality[1] > 0.7:  # High conscientiousness
            reasoning.append("Considering long-term consequences carefully")
        
        if personality[3] > 0.7:  # High agreeableness
            reasoning.append("Thinking about impact on others")
        
        if personality[4] > 0.7:  # High neuroticism
            reasoning.append("Weighing potential risks and downsides")
        
        if personality[0] > 0.7:  # High openness
            reasoning.append("Open to unconventional options")
        
        return {
            'predicted_choice': predicted_choice,
            'confidence': confidence,
            'reasoning': reasoning,
            'decision_id': decision_record.id,
            'personality_influence': {
                'primary_trait': max(
                    enumerate(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']),
                    key=lambda x: personality[x[0]]
                )[1]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to predict decision: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to predict decision")


@router.post("/calibrate/{user_id}")
async def calibrate_mirror(
    user_id: UUID,
    feedback: CalibrationFeedback,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Calibrate the consciousness mirror based on user feedback.
    
    This improves the accuracy of the cognitive twin over time.
    """
    try:
        # Get consciousness mirror
        mirror = await get_consciousness_mirror(str(user_id))
        
        # Get current accuracy
        accuracy_before = mirror.cognitive_profile.mirror_accuracy
        
        # Apply calibration based on feedback type
        calibration_data = {}
        
        if feedback.feedback_type == 'personality':
            # Adjust personality vector based on feedback
            if feedback.corrections:
                for trait, value in feedback.corrections.items():
                    if trait == 'openness':
                        mirror.cognitive_profile.personality_vector[0] = value
                    elif trait == 'conscientiousness':
                        mirror.cognitive_profile.personality_vector[1] = value
                    elif trait == 'extraversion':
                        mirror.cognitive_profile.personality_vector[2] = value
                    elif trait == 'agreeableness':
                        mirror.cognitive_profile.personality_vector[3] = value
                    elif trait == 'neuroticism':
                        mirror.cognitive_profile.personality_vector[4] = value
                
                calibration_data['personality_adjustments'] = feedback.corrections
        
        elif feedback.feedback_type == 'response':
            # Add correct response to templates
            if feedback.correct_response:
                if len(mirror.cognitive_profile.response_templates) > 100:
                    mirror.cognitive_profile.response_templates.pop(0)
                mirror.cognitive_profile.response_templates.append(feedback.correct_response)
                calibration_data['new_template'] = feedback.correct_response
        
        elif feedback.feedback_type == 'decision':
            # Update decision history with actual outcome
            if feedback.decision_id:
                decision_query = select(DecisionHistory).where(
                    DecisionHistory.id == feedback.decision_id
                )
                result = await db.execute(decision_query)
                decision = result.scalar_one_or_none()
                
                if decision:
                    decision.choice = feedback.actual_choice
                    decision.outcome_score = feedback.outcome_score
                    decision.prediction_correct = (
                        decision.predicted_choice == feedback.actual_choice
                    )
                    
                    # Record decision in mirror's decision tree
                    mirror.decision_tree.record_decision(
                        decision.context_data,
                        feedback.actual_choice,
                        feedback.outcome_score
                    )
                    
                    calibration_data['decision_updated'] = True
        
        # Recalculate accuracy
        mirror._calculate_mirror_accuracy()
        accuracy_after = mirror.cognitive_profile.mirror_accuracy
        
        # Store calibration record
        calibration = MirrorCalibration(
            user_id=user_id,
            calibration_type=feedback.feedback_type,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            improvement=accuracy_after - accuracy_before,
            calibration_data=calibration_data,
            user_feedback=feedback.user_comment,
            user_rating=feedback.rating
        )
        db.add(calibration)
        
        # Update database profile
        profile_query = select(CognitiveProfile).where(CognitiveProfile.user_id == user_id)
        result = await db.execute(profile_query)
        profile = result.scalar_one_or_none()
        
        if profile:
            profile.set_personality_vector(mirror.cognitive_profile.personality_vector)
            profile.mirror_accuracy = accuracy_after
            profile.response_templates = mirror.cognitive_profile.response_templates
        
        await db.commit()
        
        return {
            'calibration_id': calibration.id,
            'accuracy_before': accuracy_before,
            'accuracy_after': accuracy_after,
            'improvement': accuracy_after - accuracy_before,
            'status': 'calibrated',
            'message': f"Mirror accuracy {'improved' if accuracy_after > accuracy_before else 'adjusted'} by {abs(accuracy_after - accuracy_before):.2%}"
        }
        
    except Exception as e:
        logger.error(f"Failed to calibrate mirror: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to calibrate mirror")


@router.get("/evolution/{user_id}")
async def get_personality_evolution(
    user_id: UUID,
    days: int = Query(default=30, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get the personality evolution over a specified time period.
    
    Shows how the user's personality has changed and predicts future changes.
    """
    try:
        # Get cognitive profile
        profile_query = select(CognitiveProfile).where(CognitiveProfile.user_id == user_id)
        result = await db.execute(profile_query)
        profile = result.scalar_one_or_none()
        
        if not profile or not profile.personality_history:
            raise HTTPException(status_code=404, detail="No personality history available")
        
        # Filter history by time period
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_history = [
            snapshot for snapshot in profile.personality_history
            if datetime.fromisoformat(snapshot['timestamp']) > cutoff_date
        ]
        
        if len(recent_history) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data for evolution analysis")
        
        # Calculate changes
        first_snapshot = recent_history[0]['vector']
        last_snapshot = recent_history[-1]['vector']
        
        changes = {
            'openness': last_snapshot[0] - first_snapshot[0],
            'conscientiousness': last_snapshot[1] - first_snapshot[1],
            'extraversion': last_snapshot[2] - first_snapshot[2],
            'agreeableness': last_snapshot[3] - first_snapshot[3],
            'neuroticism': last_snapshot[4] - first_snapshot[4]
        }
        
        # Calculate trends
        trends = {}
        for trait_idx, trait_name in enumerate(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']):
            values = [snapshot['vector'][trait_idx] for snapshot in recent_history]
            
            # Simple linear regression for trend
            x = list(range(len(values)))
            x_mean = sum(x) / len(x)
            y_mean = sum(values) / len(values)
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(len(values)))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(len(values)))
            
            slope = numerator / denominator if denominator != 0 else 0
            trends[trait_name] = 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable'
        
        # Predict future state (30 days ahead)
        future_personality = []
        for trait_idx in range(5):
            values = [snapshot['vector'][trait_idx] for snapshot in recent_history[-10:]]
            if len(values) > 1:
                # Simple extrapolation
                change_rate = (values[-1] - values[0]) / len(values)
                future_value = values[-1] + (change_rate * 30)
                future_value = max(0, min(1, future_value))  # Clamp to [0, 1]
            else:
                future_value = values[-1] if values else 0.5
            
            future_personality.append(future_value)
        
        return {
            'period_analyzed': f"{days} days",
            'snapshots_analyzed': len(recent_history),
            'current_personality': {
                'openness': last_snapshot[0],
                'conscientiousness': last_snapshot[1],
                'extraversion': last_snapshot[2],
                'agreeableness': last_snapshot[3],
                'neuroticism': last_snapshot[4]
            },
            'personality_changes': changes,
            'trends': trends,
            'predicted_future': {
                'openness': future_personality[0],
                'conscientiousness': future_personality[1],
                'extraversion': future_personality[2],
                'agreeableness': future_personality[3],
                'neuroticism': future_personality[4]
            },
            'evolution_chart': [
                {
                    'timestamp': snapshot['timestamp'],
                    'openness': snapshot['vector'][0],
                    'conscientiousness': snapshot['vector'][1],
                    'extraversion': snapshot['vector'][2],
                    'agreeableness': snapshot['vector'][3],
                    'neuroticism': snapshot['vector'][4],
                    'accuracy': snapshot.get('accuracy', 0)
                }
                for snapshot in recent_history
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get personality evolution: {e}")
        raise HTTPException(status_code=500, detail="Failed to get personality evolution")


@router.get("/sessions/{user_id}")
async def get_consciousness_sessions(
    user_id: UUID,
    session_type: Optional[str] = None,
    active_only: bool = False,
    db: AsyncSession = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """
    Get all consciousness sessions for a user.
    
    Includes twin chats, future self conversations, and decision helper sessions.
    """
    try:
        query = select(ConsciousnessSession).where(
            ConsciousnessSession.user_id == user_id
        )
        
        if session_type:
            query = query.where(ConsciousnessSession.session_type == session_type)
        
        if active_only:
            query = query.where(ConsciousnessSession.is_active == True)
        
        query = query.order_by(ConsciousnessSession.created_at.desc())
        
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        return [
            {
                'session_id': session.id,
                'session_type': session.session_type,
                'twin_config': session.twin_config,
                'message_count': session.message_count,
                'prediction_accuracy': session.prediction_accuracy,
                'user_satisfaction': session.user_satisfaction,
                'is_active': session.is_active,
                'created_at': session.created_at,
                'ended_at': session.ended_at,
                'last_message': session.conversation_log[-1] if session.conversation_log else None
            }
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get consciousness sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sessions")