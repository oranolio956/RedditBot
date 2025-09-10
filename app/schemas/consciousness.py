"""
Consciousness Mirroring API Schemas

Pydantic models for consciousness mirroring API requests and responses.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class KeystrokeData(BaseModel):
    """Keystroke dynamics data for personality analysis."""
    dwell_times: List[float] = Field(default_factory=list, description="Key hold durations in ms")
    flight_times: List[float] = Field(default_factory=list, description="Time between keys in ms")
    typing_speed: float = Field(default=0.0, description="Words per minute")
    deletion_rate: float = Field(default=0.0, description="Correction frequency")
    emotional_pressure: float = Field(default=0.0, description="Pressure/speed correlation")


class ConsciousnessUpdateRequest(BaseModel):
    """Request to update consciousness mirror with new user data."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(..., min_length=1)
    keystroke_data: Optional[KeystrokeData] = None


class PredictResponseRequest(BaseModel):
    """Request to predict user response to context."""
    context: str = Field(..., min_length=1)
    log_prediction: bool = Field(default=True, description="Log prediction for calibration")


class FutureSimulationRequest(BaseModel):
    """Request to simulate future self."""
    years_ahead: int = Field(..., ge=1, le=50, description="Years into future")


class TwinConversationRequest(BaseModel):
    """Request to chat with cognitive twin."""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[UUID] = None


class DecisionPredictionRequest(BaseModel):
    """Request to predict user decision."""
    context: Dict[str, Any] = Field(..., description="Decision context")
    alternatives: List[str] = Field(default_factory=list, description="Available options")


class CalibrationFeedback(BaseModel):
    """Feedback to calibrate consciousness mirror."""
    feedback_type: str = Field(..., regex="^(personality|response|decision)$")
    
    # For personality calibration
    corrections: Optional[Dict[str, float]] = Field(None, description="Personality trait corrections")
    
    # For response calibration
    correct_response: Optional[str] = Field(None, description="What user actually said")
    
    # For decision calibration
    decision_id: Optional[UUID] = None
    actual_choice: Optional[str] = None
    outcome_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # General feedback
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="User rating")
    user_comment: Optional[str] = None


class CognitiveProfileResponse(BaseModel):
    """Response containing cognitive profile data."""
    user_id: UUID
    personality: Dict[str, float]
    mirror_accuracy: float
    thought_velocity: float
    creativity_index: float
    prediction_confidence: float
    linguistic_fingerprint: Dict[str, Any]
    cognitive_biases: Dict[str, float]
    emotional_baseline: List[float]
    last_sync: datetime
    sync_message_count: int
    
    class Config:
        orm_mode = True