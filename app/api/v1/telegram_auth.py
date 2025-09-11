"""
Telegram Authentication API Endpoints

FastAPI endpoints for Kelly's Telegram account authentication workflow including
two-factor authentication, code verification, and account connection.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import structlog

from app.core.auth import get_current_user
from app.core.redis import redis_manager
from app.services.kelly_telegram_userbot import kelly_userbot

logger = structlog.get_logger()

router = APIRouter(prefix="/telegram", tags=["telegram-auth"])

# Request/Response Models
class SendCodeRequest(BaseModel):
    api_id: int = Field(..., description="Telegram API ID")
    api_hash: str = Field(..., description="Telegram API hash")
    phone_number: str = Field(..., description="Phone number with country code")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        """Validate phone number format"""
        import re
        if not re.match(r'^\+?[1-9]\d{1,14}$', v):
            raise ValueError('Invalid phone number format')
        return v

class VerifyCodeRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from send-code response")
    verification_code: str = Field(..., description="SMS/App verification code")

class Verify2FARequest(BaseModel):
    session_id: str = Field(..., description="Session ID from verify-code response")
    password: str = Field(..., description="Two-factor authentication password")

class ConnectAccountRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from authentication")
    account_name: str = Field(..., description="Display name for the account")
    dm_only_mode: bool = Field(default=True, description="Enable DM-only mode")
    kelly_config: Optional[Dict[str, Any]] = Field(default=None, description="Kelly personality configuration")

class SendCodeResponse(BaseModel):
    session_id: str
    phone_code_hash: str
    message: str
    expires_at: str

class VerifyCodeResponse(BaseModel):
    session_id: str
    requires_2fa: bool
    message: str
    user_info: Optional[Dict[str, Any]] = None

class Verify2FAResponse(BaseModel):
    session_id: str
    authenticated: bool
    message: str
    user_info: Dict[str, Any]

class ConnectAccountResponse(BaseModel):
    account_id: str
    session_name: str
    connected: bool
    message: str

@router.post("/send-code")
async def send_verification_code(
    request: SendCodeRequest,
    current_user = Depends(get_current_user)
) -> SendCodeResponse:
    """
    Send SMS/App verification code to begin Telegram account authentication.
    
    This endpoint initiates the authentication process by sending a verification
    code to the provided phone number through Telegram's authentication system.
    """
    try:
        # Generate session ID for this authentication attempt
        import uuid
        session_id = str(uuid.uuid4())
        
        # Store authentication session data
        session_data = {
            "api_id": request.api_id,
            "api_hash": request.api_hash,
            "phone_number": request.phone_number,
            "user_id": current_user.get("id"),
            "created_at": datetime.now().isoformat(),
            "status": "code_sent",
            "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
        # Send verification code through Kelly userbot service
        result = await kelly_userbot.send_verification_code(
            api_id=request.api_id,
            api_hash=request.api_hash,
            phone_number=request.phone_number
        )
        
        if result.get("success"):
            # Update session with phone code hash
            session_data.update({
                "phone_code_hash": result.get("phone_code_hash"),
                "session_string": result.get("session_string")
            })
            
            # Store session in Redis with 5-minute expiration
            await redis_manager.setex(
                f"telegram_auth_session:{session_id}",
                300,  # 5 minutes
                json.dumps(session_data)
            )
            
            logger.info(f"Verification code sent for session {session_id}")
            
            return SendCodeResponse(
                session_id=session_id,
                phone_code_hash=result.get("phone_code_hash"),
                message="Verification code sent successfully",
                expires_at=session_data["expires_at"]
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to send verification code: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error sending verification code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-code")
async def verify_authentication_code(
    request: VerifyCodeRequest,
    current_user = Depends(get_current_user)
) -> VerifyCodeResponse:
    """
    Verify SMS/App authentication code and check for 2FA requirement.
    
    This endpoint verifies the code received via SMS or Telegram app and
    determines if two-factor authentication is required for the account.
    """
    try:
        # Retrieve authentication session
        session_key = f"telegram_auth_session:{request.session_id}"
        session_data_str = await redis_manager.get(session_key)
        
        if not session_data_str:
            raise HTTPException(
                status_code=404,
                detail="Authentication session not found or expired"
            )
        
        session_data = json.loads(session_data_str)
        
        # Verify the session belongs to current user
        if session_data.get("user_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Session access denied")
        
        # Verify authentication code
        result = await kelly_userbot.verify_authentication_code(
            session_string=session_data.get("session_string"),
            phone_code_hash=session_data.get("phone_code_hash"),
            verification_code=request.verification_code
        )
        
        if result.get("success"):
            # Update session status
            session_data.update({
                "status": "code_verified",
                "requires_2fa": result.get("requires_2fa", False),
                "user_info": result.get("user_info", {}),
                "verified_at": datetime.now().isoformat()
            })
            
            # Extend session expiration if 2FA is required
            expiry = 600 if result.get("requires_2fa") else 300  # 10 min vs 5 min
            await redis_manager.setex(session_key, expiry, json.dumps(session_data))
            
            logger.info(f"Code verified for session {request.session_id}, 2FA required: {result.get('requires_2fa')}")
            
            return VerifyCodeResponse(
                session_id=request.session_id,
                requires_2fa=result.get("requires_2fa", False),
                message="Verification code accepted" if not result.get("requires_2fa") else "Two-factor authentication required",
                user_info=result.get("user_info") if not result.get("requires_2fa") else None
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid verification code: {result.get('error', 'Code verification failed')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-2fa")
async def verify_two_factor_auth(
    request: Verify2FARequest,
    current_user = Depends(get_current_user)
) -> Verify2FAResponse:
    """
    Verify two-factor authentication password.
    
    This endpoint handles the second step of authentication for accounts
    with two-factor authentication enabled.
    """
    try:
        # Retrieve authentication session
        session_key = f"telegram_auth_session:{request.session_id}"
        session_data_str = await redis_manager.get(session_key)
        
        if not session_data_str:
            raise HTTPException(
                status_code=404,
                detail="Authentication session not found or expired"
            )
        
        session_data = json.loads(session_data_str)
        
        # Verify session belongs to current user and requires 2FA
        if session_data.get("user_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Session access denied")
        
        if not session_data.get("requires_2fa"):
            raise HTTPException(status_code=400, detail="Two-factor authentication not required for this session")
        
        if session_data.get("status") != "code_verified":
            raise HTTPException(status_code=400, detail="Verification code must be verified first")
        
        # Verify 2FA password
        result = await kelly_userbot.verify_2fa_password(
            session_string=session_data.get("session_string"),
            password=request.password
        )
        
        if result.get("success"):
            # Update session status
            session_data.update({
                "status": "authenticated",
                "user_info": result.get("user_info", {}),
                "authenticated_at": datetime.now().isoformat()
            })
            
            # Extend session for account connection
            await redis_manager.setex(session_key, 600, json.dumps(session_data))  # 10 minutes
            
            logger.info(f"2FA verified for session {request.session_id}")
            
            return Verify2FAResponse(
                session_id=request.session_id,
                authenticated=True,
                message="Two-factor authentication successful",
                user_info=result.get("user_info", {})
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid 2FA password: {result.get('error', '2FA verification failed')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying 2FA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect-account")
async def connect_telegram_account(
    request: ConnectAccountRequest,
    current_user = Depends(get_current_user)
) -> ConnectAccountResponse:
    """
    Connect authenticated Telegram account to Kelly system.
    
    This endpoint finalizes the authentication process by adding the
    authenticated account to Kelly's userbot system for management.
    """
    try:
        # Retrieve authentication session
        session_key = f"telegram_auth_session:{request.session_id}"
        session_data_str = await redis_manager.get(session_key)
        
        if not session_data_str:
            raise HTTPException(
                status_code=404,
                detail="Authentication session not found or expired"
            )
        
        session_data = json.loads(session_data_str)
        
        # Verify session belongs to current user and is authenticated
        if session_data.get("user_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Session access denied")
        
        expected_status = "authenticated" if session_data.get("requires_2fa") else "code_verified"
        if session_data.get("status") != expected_status:
            raise HTTPException(status_code=400, detail="Authentication not completed")
        
        # Generate account ID
        import uuid
        account_id = str(uuid.uuid4())[:8]
        
        # Create account configuration
        from app.services.kelly_telegram_userbot import AccountConfig
        from app.services.kelly_personality_service import KellyPersonalityConfig
        
        kelly_config = None
        if request.kelly_config:
            kelly_config = KellyPersonalityConfig(**request.kelly_config)
        
        account_config = AccountConfig(
            api_id=session_data["api_id"],
            api_hash=session_data["api_hash"],
            phone_number=session_data["phone_number"],
            session_name=f"kelly_{account_id}",
            dm_only_mode=request.dm_only_mode,
            kelly_config=kelly_config,
            enabled=True
        )
        
        # Add account to Kelly userbot system
        success = await kelly_userbot.add_authenticated_account(
            account_id=account_id,
            account_config=account_config,
            session_string=session_data.get("session_string"),
            user_info=session_data.get("user_info", {})
        )
        
        if success:
            # Clean up authentication session
            await redis_manager.delete(session_key)
            
            # Store account ownership
            await redis_manager.setex(
                f"kelly:account_owner:{account_id}",
                86400 * 30,  # 30 days
                current_user.get("id")
            )
            
            logger.info(f"Telegram account {account_id} connected successfully for user {current_user.get('id')}")
            
            return ConnectAccountResponse(
                account_id=account_id,
                session_name=account_config.session_name,
                connected=True,
                message="Telegram account connected successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to connect account to Kelly system"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/auth-status/{session_id}")
async def get_authentication_status(
    session_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current authentication status for a session.
    
    This endpoint allows checking the progress of an ongoing authentication
    session without modifying its state.
    """
    try:
        # Retrieve authentication session
        session_key = f"telegram_auth_session:{session_id}"
        session_data_str = await redis_manager.get(session_key)
        
        if not session_data_str:
            raise HTTPException(
                status_code=404,
                detail="Authentication session not found or expired"
            )
        
        session_data = json.loads(session_data_str)
        
        # Verify session belongs to current user
        if session_data.get("user_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Session access denied")
        
        # Return sanitized session status
        return {
            "session_id": session_id,
            "status": session_data.get("status"),
            "phone_number": session_data.get("phone_number"),
            "requires_2fa": session_data.get("requires_2fa", False),
            "created_at": session_data.get("created_at"),
            "expires_at": session_data.get("expires_at"),
            "user_info": session_data.get("user_info", {}) if session_data.get("status") in ["code_verified", "authenticated"] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting auth status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/auth-session/{session_id}")
async def cancel_authentication_session(
    session_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Cancel an ongoing authentication session.
    
    This endpoint allows users to abort an authentication process and
    clean up the session data.
    """
    try:
        # Retrieve authentication session
        session_key = f"telegram_auth_session:{session_id}"
        session_data_str = await redis_manager.get(session_key)
        
        if not session_data_str:
            raise HTTPException(
                status_code=404,
                detail="Authentication session not found or expired"
            )
        
        session_data = json.loads(session_data_str)
        
        # Verify session belongs to current user
        if session_data.get("user_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Session access denied")
        
        # Clean up session
        await redis_manager.delete(session_key)
        
        logger.info(f"Authentication session {session_id} cancelled by user {current_user.get('id')}")
        
        return {"message": "Authentication session cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling auth session: {e}")
        raise HTTPException(status_code=500, detail=str(e))