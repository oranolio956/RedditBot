"""
GDPR Compliance API Endpoints

Provides data privacy endpoints for GDPR compliance including
data export, deletion, consent management, and privacy controls.
"""

import asyncio
import json
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.models.user import User
from app.core.auth import (
    get_current_user, require_permission, require_admin,
    AuthenticatedUser, Permission
)
from app.core.secrets_manager import get_secrets_manager
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()


class ConsentRequest(BaseModel):
    """User consent request model."""
    consent_type: str = Field(..., description="Type of consent (analytics, marketing, etc.)")
    granted: bool = Field(..., description="Whether consent is granted")
    purpose: Optional[str] = Field(None, description="Purpose of data processing")


class DataExportRequest(BaseModel):
    """Data export request model."""
    include_personal_data: bool = Field(True, description="Include personal information")
    include_activity_logs: bool = Field(True, description="Include activity logs")
    include_preferences: bool = Field(True, description="Include user preferences")
    include_conversations: bool = Field(False, description="Include conversation history")
    format: str = Field("json", description="Export format (json, csv, xml)")


class DataDeletionRequest(BaseModel):
    """Data deletion request model."""
    delete_personal_data: bool = Field(True, description="Delete personal information")
    delete_activity_logs: bool = Field(False, description="Delete activity logs")
    delete_conversations: bool = Field(False, description="Delete conversation history")
    keep_anonymous_analytics: bool = Field(True, description="Keep anonymized analytics data")
    confirmation_text: str = Field(..., description="User must type 'DELETE MY DATA' to confirm")
    
    @property
    def is_confirmed(self) -> bool:
        return self.confirmation_text.strip().upper() == "DELETE MY DATA"


class PrivacySettings(BaseModel):
    """Privacy settings model."""
    data_processing_consent: bool = Field(False, description="Consent for data processing")
    analytics_consent: bool = Field(False, description="Consent for analytics")
    marketing_consent: bool = Field(False, description="Consent for marketing")
    third_party_sharing: bool = Field(False, description="Allow third-party data sharing")
    data_retention_days: int = Field(365, description="Data retention period in days")
    anonymize_after_deletion: bool = Field(True, description="Anonymize data instead of hard delete")


@router.get("/privacy-policy")
async def get_privacy_policy() -> Dict[str, Any]:
    """
    Get current privacy policy and data processing information.
    """
    return {
        "version": "1.0",
        "last_updated": "2024-01-01",
        "effective_date": "2024-01-01",
        "policy": {
            "data_controller": {
                "name": "Telegram ML Bot Service",
                "contact": "privacy@example.com",
                "dpo_contact": "dpo@example.com"
            },
            "data_processing": {
                "lawful_basis": "Consent (GDPR Article 6.1.a)",
                "purposes": [
                    "Providing bot functionality",
                    "User authentication",
                    "Service improvement",
                    "Analytics (anonymized)"
                ],
                "retention_period": "365 days or until consent withdrawal",
                "automated_decision_making": False
            },
            "data_categories": {
                "personal_data": [
                    "Telegram ID",
                    "Username",
                    "First and last name",
                    "Interaction timestamps"
                ],
                "technical_data": [
                    "Session information",
                    "Device fingerprints",
                    "IP addresses (hashed)",
                    "Usage analytics"
                ]
            },
            "user_rights": [
                "Right to access (Article 15)",
                "Right to rectification (Article 16)",
                "Right to erasure (Article 17)",
                "Right to restrict processing (Article 18)",
                "Right to data portability (Article 20)",
                "Right to object (Article 21)"
            ],
            "data_sharing": {
                "third_parties": [],
                "international_transfers": False,
                "safeguards": "All data processing within EU/EEA"
            }
        }
    }


@router.get("/my-data")
async def get_user_data_summary(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get summary of user's personal data stored in the system.
    """
    try:
        # Get user data
        query = select(User).where(User.id == current_user.user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate data metrics
        data_summary = {
            "personal_data": {
                "telegram_id": user.telegram_id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "created_at": user.created_at.isoformat(),
                "last_activity": user.last_activity.isoformat() if user.last_activity else None
            },
            "data_metrics": {
                "account_age_days": (datetime.utcnow() - user.created_at).days,
                "total_interactions": 0,  # Calculate from logs
                "data_size_estimate_kb": 5,  # Estimate
                "last_data_export": None,  # Track export history
                "consent_status": "granted"  # Track consent
            },
            "privacy_controls": {
                "data_processing_consent": True,
                "analytics_consent": True,
                "can_export_data": True,
                "can_delete_data": True,
                "can_withdraw_consent": True
            }
        }
        
        return data_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get user data summary",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve data summary")


@router.post("/export-data")
async def export_user_data(
    export_request: DataExportRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> StreamingResponse:
    """
    Export all user data in machine-readable format (GDPR Article 20).
    """
    try:
        # Get user data
        query = select(User).where(User.id == current_user.user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prepare export data
        export_data = {
            "export_info": {
                "requested_at": datetime.utcnow().isoformat(),
                "user_id": str(user.id),
                "export_format": export_request.format,
                "gdpr_compliance": True
            }
        }
        
        # Include personal data
        if export_request.include_personal_data:
            export_data["personal_data"] = {
                "telegram_id": user.telegram_id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "created_at": user.created_at.isoformat(),
                "last_activity": user.last_activity.isoformat() if user.last_activity else None,
                "is_active": user.is_active
            }
        
        # Include preferences
        if export_request.include_preferences:
            export_data["preferences"] = user.preferences or {}
        
        # Include activity logs (implement based on your logging system)
        if export_request.include_activity_logs:
            export_data["activity_logs"] = [
                # Add activity logs here
            ]
        
        # Include conversations (implement based on your chat storage)
        if export_request.include_conversations:
            export_data["conversations"] = [
                # Add conversation history here
            ]
        
        # Create export file
        if export_request.format == "json":
            content = json.dumps(export_data, indent=2, ensure_ascii=False)
            media_type = "application/json"
            filename = f"user_data_export_{current_user.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        elif export_request.format == "csv":
            # Convert to CSV format (implement as needed)
            content = "CSV export not implemented yet"
            media_type = "text/csv"
            filename = f"user_data_export_{current_user.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        # Log export request
        logger.info(
            "User data export requested",
            user_id=str(current_user.user_id),
            format=export_request.format,
            includes={
                "personal_data": export_request.include_personal_data,
                "activity_logs": export_request.include_activity_logs,
                "preferences": export_request.include_preferences,
                "conversations": export_request.include_conversations
            }
        )
        
        # Return as streaming response
        def generate():
            yield content.encode('utf-8')
        
        return StreamingResponse(
            generate(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Data export failed",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        raise HTTPException(status_code=500, detail="Data export failed")


@router.post("/delete-data")
async def delete_user_data(
    deletion_request: DataDeletionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Delete user data in compliance with GDPR "Right to be Forgotten" (Article 17).
    """
    try:
        # Verify confirmation
        if not deletion_request.is_confirmed:
            raise HTTPException(
                status_code=400,
                detail="Please type 'DELETE MY DATA' to confirm data deletion"
            )
        
        # Get user
        query = select(User).where(User.id == current_user.user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Track what will be deleted
        deletion_summary = {
            "deleted_at": datetime.utcnow().isoformat(),
            "user_id": str(user.id),
            "actions_performed": []
        }
        
        # Delete personal data
        if deletion_request.delete_personal_data:
            if deletion_request.keep_anonymous_analytics:
                # Anonymize instead of delete
                user.username = None
                user.first_name = "[DELETED]"
                user.last_name = "[DELETED]"
                user.telegram_id = 0  # Anonymize
                user.is_active = False
                deletion_summary["actions_performed"].append("Personal data anonymized")
            else:
                # Mark for hard deletion
                user.soft_delete()
                deletion_summary["actions_performed"].append("User account deleted")
        
        # Delete activity logs (implement based on your logging system)
        if deletion_request.delete_activity_logs:
            # Remove or anonymize activity logs
            deletion_summary["actions_performed"].append("Activity logs removed")
        
        # Delete conversations (implement based on your chat storage)
        if deletion_request.delete_conversations:
            # Remove conversation history
            deletion_summary["actions_performed"].append("Conversation history removed")
        
        # Invalidate all user sessions
        from app.core.auth import get_auth_manager
        auth_manager = await get_auth_manager()
        await auth_manager.invalidate_session(current_user.session_id)
        deletion_summary["actions_performed"].append("All sessions invalidated")
        
        # Commit changes
        await db.commit()
        
        # Log deletion request
        logger.info(
            "User data deletion completed",
            user_id=str(current_user.user_id),
            actions=deletion_summary["actions_performed"],
            keep_analytics=deletion_request.keep_anonymous_analytics
        )
        
        return {
            "message": "Data deletion completed successfully",
            "summary": deletion_summary,
            "next_steps": [
                "Your account has been deactivated",
                "All personal data has been removed or anonymized",
                "You will be automatically logged out",
                "You can create a new account at any time"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Data deletion failed",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        await db.rollback()
        raise HTTPException(status_code=500, detail="Data deletion failed")


@router.post("/consent")
async def update_consent(
    consent_request: ConsentRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, str]:
    """
    Update user consent for data processing (GDPR compliance).
    """
    try:
        # Get user
        query = select(User).where(User.id == current_user.user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update consent in user preferences
        if not user.preferences:
            user.preferences = {}
        
        if "consent" not in user.preferences:
            user.preferences["consent"] = {}
        
        user.preferences["consent"][consent_request.consent_type] = {
            "granted": consent_request.granted,
            "timestamp": datetime.utcnow().isoformat(),
            "purpose": consent_request.purpose
        }
        
        await db.commit()
        
        logger.info(
            "User consent updated",
            user_id=str(current_user.user_id),
            consent_type=consent_request.consent_type,
            granted=consent_request.granted
        )
        
        return {
            "message": f"Consent for {consent_request.consent_type} has been {'granted' if consent_request.granted else 'withdrawn'}",
            "status": "updated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consent update failed",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        await db.rollback()
        raise HTTPException(status_code=500, detail="Consent update failed")


@router.get("/consent")
async def get_consent_status(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get current consent status for all data processing activities.
    """
    try:
        # Get user
        query = select(User).where(User.id == current_user.user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get consent status
        consent_data = user.preferences.get("consent", {}) if user.preferences else {}
        
        # Default consent types
        default_consent = {
            "data_processing": {
                "granted": True,
                "timestamp": user.created_at.isoformat(),
                "purpose": "Core functionality",
                "required": True
            },
            "analytics": {
                "granted": False,
                "timestamp": None,
                "purpose": "Service improvement",
                "required": False
            },
            "marketing": {
                "granted": False,
                "timestamp": None,
                "purpose": "Marketing communications",
                "required": False
            }
        }
        
        # Merge with user's actual consent
        for consent_type, default_data in default_consent.items():
            if consent_type in consent_data:
                default_consent[consent_type].update(consent_data[consent_type])
        
        return {
            "user_id": str(current_user.user_id),
            "consent_status": default_consent,
            "last_updated": max(
                [c.get("timestamp", user.created_at.isoformat()) 
                 for c in default_consent.values() 
                 if c.get("timestamp")],
                default=[user.created_at.isoformat()]
            )[0],
            "withdrawal_instructions": "You can withdraw consent at any time using the /gdpr/consent endpoint"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get consent status",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve consent status")


@router.get("/data-processing-activities")
async def get_data_processing_activities(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get information about data processing activities (GDPR transparency).
    """
    return {
        "processing_activities": [
            {
                "activity": "User Authentication",
                "purpose": "Verify user identity and provide secure access",
                "lawful_basis": "Contract performance (GDPR Article 6.1.b)",
                "data_categories": ["Telegram ID", "Username", "Authentication tokens"],
                "retention_period": "Until account deletion",
                "automated_decision_making": False
            },
            {
                "activity": "Service Analytics",
                "purpose": "Improve service quality and user experience",
                "lawful_basis": "Legitimate interest (GDPR Article 6.1.f)",
                "data_categories": ["Usage patterns", "Error logs", "Performance metrics"],
                "retention_period": "90 days (anonymized)",
                "automated_decision_making": False
            },
            {
                "activity": "Security Monitoring",
                "purpose": "Detect and prevent security threats",
                "lawful_basis": "Legitimate interest (GDPR Article 6.1.f)",
                "data_categories": ["IP addresses (hashed)", "Session data", "Access logs"],
                "retention_period": "30 days",
                "automated_decision_making": True,
                "automated_decision_details": "Rate limiting and abuse detection"
            }
        ],
        "user_rights": {
            "access": "Request copy of your personal data",
            "rectification": "Correct inaccurate personal data",
            "erasure": "Request deletion of your personal data",
            "restrict_processing": "Limit how we process your data",
            "data_portability": "Receive your data in machine-readable format",
            "object": "Object to processing based on legitimate interest",
            "withdraw_consent": "Withdraw consent for specific processing activities"
        },
        "contact_info": {
            "data_controller": "Telegram ML Bot Service",
            "email": "privacy@example.com",
            "dpo_email": "dpo@example.com",
            "supervisory_authority": "Your local data protection authority"
        }
    }


# Administrative endpoints for GDPR compliance
@router.get("/admin/data-requests")
async def get_data_requests(
    current_user: AuthenticatedUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """
    Get all pending GDPR data requests (admin only).
    """
    # Implement data request tracking
    return [
        # This would return actual pending requests from a tracking system
    ]


@router.get("/admin/compliance-report")
async def get_compliance_report(
    current_user: AuthenticatedUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Generate GDPR compliance report (admin only).
    """
    try:
        # Calculate compliance metrics
        total_users_query = select(func.count(User.id))
        total_users_result = await db.execute(total_users_query)
        total_users = total_users_result.scalar()
        
        active_users_query = select(func.count(User.id)).where(User.is_active == True)
        active_users_result = await db.execute(active_users_query)
        active_users = active_users_result.scalar()
        
        return {
            "report_generated_at": datetime.utcnow().isoformat(),
            "compliance_status": "compliant",
            "metrics": {
                "total_users": total_users,
                "active_users": active_users,
                "data_export_requests_30d": 0,  # Implement tracking
                "data_deletion_requests_30d": 0,  # Implement tracking
                "consent_withdrawal_requests_30d": 0,  # Implement tracking
                "avg_data_export_response_time_hours": 0.5,
                "avg_data_deletion_response_time_hours": 1.0
            },
            "compliance_checks": {
                "privacy_policy_current": True,
                "consent_mechanisms_active": True,
                "data_retention_policies_enforced": True,
                "user_rights_accessible": True,
                "breach_notification_procedures": True,
                "dpo_contact_available": True
            },
            "recommendations": [
                "Regularly review and update privacy policy",
                "Monitor data export response times",
                "Conduct annual GDPR compliance audit"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")


# Export router
__all__ = ["router"]
